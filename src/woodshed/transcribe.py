"""Local lyrics transcription with Whisper on Apple Silicon (MLX)."""

import os
import re

import numpy as np

from woodshed.lyrics import words

MODEL = os.environ.get("WOODSHED_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# Whisper learned from subtitled videos, so over instrumental passages it tends to "hear" these.
_PHANTOM_LINES = {
    "you", "bye", "thank you", "thanks for watching", "thank you for watching",
    "thank you so much for watching", "please subscribe", "music", "applause",
    "see you next time", "well see you next time", "thanks for listening",
}


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|[♪♫]", "", text)  # [Music], (guitar strumming), ♪
    return " ".join(text.split())


def transcribe(samples: np.ndarray, language: str | None = None) -> list[str]:
    """Return the sung lines Whisper hears in 16 kHz mono samples."""
    import mlx_whisper  # slow to import, so only when a take needs it

    # No filtering on Whisper's compression ratio: a sung chorus repeats just like a
    # hallucination loop does, and repeated lines don't skew matching anyway.
    result = mlx_whisper.transcribe(
        samples, path_or_hf_repo=MODEL, language=language,
        condition_on_previous_text=False,  # stops one misheard line from snowballing
    )
    lines = []
    for segment in result["segments"]:
        text = _clean(segment["text"])
        if text and " ".join(words(text)) not in _PHANTOM_LINES:
            lines.append(text)
    return lines
