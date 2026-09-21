"""Separating your voice from the guitar with Demucs, so Whisper hears the lyrics.

On synthetic takes with the guitar 6 dB louder than the voice, Whisper caught 48% of
the words from the raw mix and 98% from the isolated voice.
"""

from pathlib import Path

import numpy as np

from woodshed import audio

MODEL = "htdemucs"
HF_REPO = "adefossez/HTDemucs"  # where Demucs fetches MODEL from


def vocals(src: Path, start: float, seconds: float) -> np.ndarray:
    """The voice alone from a section of the take, as 16 kHz mono for Whisper."""
    import julius
    import torch
    from demucs.api import Separator

    # The Apple GPU is ~5x faster than the CPU here: ~10 s for 2½ minutes on an M4.
    separator = Separator(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    mix = audio.load(src, rate=separator.samplerate, channels=2, start=start, seconds=seconds)
    _, stems = separator.separate_tensor(torch.from_numpy(mix), separator.samplerate)
    voice = stems["vocals"].mean(0)
    return julius.resample_frac(voice, separator.samplerate, audio.SAMPLE_RATE).numpy()
