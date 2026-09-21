"""RMVPE: a pitch tracker built for the singing voice, even with instruments around it.

Adapted (inference only) from infer/rmvpe.py in
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI, under its MIT license:

    Copyright (c) 2023 liujing04
    Copyright (c) 2023 源文雨
    Copyright (c) 2023 Ftps

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

Module and attribute names match the original so its published weights load unchanged.
"""

import functools

import numpy as np
import torch
from torch import nn

HF_REPO = "lj1995/VoiceConversionWebUI"
WEIGHTS = "rmvpe.pt"  # 181 MB
HOP_SECONDS = 0.01  # one pitch estimate every 160 samples at 16 kHz


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if hasattr(self, "shortcut") else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)] +
                                  [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x if self.kernel_size is None else (x, self.pool(x))


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels, out_channels = out_channels, out_channels * 2
        self.out_channel = out_channels

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)] +
                                    [ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
                                     for _ in range(n_inters - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=stride, padding=(1, 1),
                               output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)] +
                                   [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks, momentum))
            in_channels //= 2

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        x = self.cnn(self.unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2)
        return self.fc(x)


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@functools.cache
def _model() -> tuple[E2E, torch.Tensor]:
    import librosa
    from huggingface_hub import hf_hub_download

    model = E2E(4, 1, (2, 2))
    model.load_state_dict(torch.load(hf_hub_download(HF_REPO, WEIGHTS), map_location="cpu", weights_only=True))
    model.eval().to(_device())
    mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    return model, torch.from_numpy(mel_basis).float().to(_device())


def pitch(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(f0 in Hz, confidence 0-1) every 10 ms of 16 kHz mono audio."""
    model, mel_basis = _model()
    with torch.no_grad():
        x = torch.from_numpy(np.array(audio, dtype=np.float32)).to(_device())[None]
        spectrum = torch.stft(x, n_fft=1024, hop_length=160, win_length=1024,
                              window=torch.hann_window(1024, device=x.device), center=True, return_complex=True)
        mel = torch.log(torch.clamp(mel_basis @ spectrum.abs(), min=1e-5))
        frames = mel.shape[-1]
        mel = nn.functional.pad(mel, (0, 32 * ((frames - 1) // 32 + 1) - frames))  # the U-Net halves 5 times
        salience = model(mel)[0, :frames].cpu().numpy()

    # Pitch: the salience-weighted average of the 9 bins (20 cents each) around the peak.
    peak = np.argmax(salience, axis=1)
    confidence = salience[np.arange(frames), peak]
    bins = np.clip(peak[:, None] + np.arange(-4, 5), 0, 359)
    weights = np.take_along_axis(salience, bins, axis=1)
    cents = (weights * (20 * bins + 1997.3794084376191)).sum(1) / np.maximum(weights.sum(1), 1e-9)
    return 10 * 2 ** (cents / 1200), confidence
