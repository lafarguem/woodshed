"""Separating your voice from the instrument (guitar, piano…) with Demucs.

Whisper hears the lyrics far better in the voice alone: on synthetic takes with the guitar
6 dB louder than the voice, it caught 48% of the words from the raw mix and 98% from the
isolated voice. The rating uses the instrument part for its tuning and tempo.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woodshed import audio

MODEL = "htdemucs"
HF_REPO = "adefossez/HTDemucs"  # where Demucs fetches MODEL from


@dataclass
class Stems:
    mix: np.ndarray  # the take as recorded, mono, at `rate`
    vocals: np.ndarray
    accompaniment: np.ndarray  # everything but the voice: your instrument
    rate: int

    def vocals_16k(self, seconds: float | None = None) -> np.ndarray:
        """The voice as 16 kHz mono, for Whisper."""
        return self._16k(self.vocals if seconds is None else self.vocals[: int(seconds * self.rate)])

    def mix_16k(self) -> np.ndarray:
        return self._16k(self.mix)

    def _16k(self, samples: np.ndarray) -> np.ndarray:
        import julius
        import torch

        return julius.resample_frac(torch.from_numpy(samples), self.rate, audio.SAMPLE_RATE).numpy()


@functools.cache
def _separator():
    import torch
    from demucs.api import Separator

    # The Apple GPU is ~5x faster than the CPU here: ~10 s for 2½ minutes on an M4.
    # No random time shifts, so the same take always separates (and rates) the same way.
    return Separator(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu", shifts=0)


def separate(src: Path, start: float = 0.0, seconds: float | None = None) -> Stems:
    """Split a section of the take (all of it by default) into voice and instrument."""
    import torch

    separator = _separator()
    mix = audio.load(src, rate=separator.samplerate, channels=2, start=start, seconds=seconds)
    _, stems = separator.separate_tensor(torch.from_numpy(mix), separator.samplerate)
    vocals = stems.pop("vocals").mean(0)
    accompaniment = sum(stems.values()).mean(0)
    return Stems(mix.mean(0), vocals.numpy(), accompaniment.numpy(), separator.samplerate)
