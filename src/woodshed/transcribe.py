"""Local lyrics transcription with Whisper on Apple Silicon (MLX)."""

import os
import re
from dataclasses import dataclass

import numpy as np

from woodshed.lyrics import words

MODEL = os.environ.get("WOODSHED_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# Whisper learned from subtitled videos, so over instrumental passages it tends to "hear" these.
_PHANTOM_LINES = {
    "you", "bye", "thank you", "thanks for watching", "thank you for watching",
    "thank you so much for watching", "please subscribe", "music", "applause",
    "see you next time", "well see you next time", "thanks for listening",
}
_ASIDES = r"\[[^\]]*\]|\([^)]*\)"  # [Music], (guitar strumming)


@dataclass
class Word:
    text: str  # as lyrics.words() writes it: lowercase, without punctuation ("dont")
    start: float  # seconds into the samples transcribed
    end: float


@dataclass
class Line:
    text: str  # as heard
    words: list[Word]


def _clean(text: str) -> str:
    text = re.sub(rf"{_ASIDES}|[♪♫]", "", text)
    return " ".join(text.split())


def transcribe(samples: np.ndarray, language: str | None = None) -> list[Line]:
    """The sung lines Whisper hears in 16 kHz mono samples, with when each word is sung."""
    import mlx.core as mx
    import mlx_whisper  # slow to import, so only when a take needs it

    # A line that decodes badly is decoded again, sampling at random: seeded, so that the same take is always
    # heard the same way (and rated the same). Unseeded, one take read 87¢ or 125¢ off the melody from one run
    # to the next.
    mx.random.seed(0)
    # No filtering on Whisper's compression ratio: a sung chorus repeats just like a
    # hallucination loop does, and repeated lines don't skew matching anyway.
    result = mlx_whisper.transcribe(
        samples, path_or_hf_repo=MODEL, language=language,
        condition_on_previous_text=False,  # stops one misheard line from snowballing
        word_timestamps=True,  # to tell which note is sung on which word (melody.py)
    )
    lines = []
    for segment in result["segments"]:
        text = _clean(segment["text"])
        if text and " ".join(words(text)) not in _PHANTOM_LINES:
            lines.append(Line(text, _timed_words(segment.get("words", []))))
    return lines


def _timed_words(heard: list[dict]) -> list[Word]:
    """Whisper's words, without what _clean() takes out of the text. An aside can span several of them, so
    it's blanked out of their text joined up, keeping every character where it was."""
    joined = "".join(w["word"] for w in heard)
    kept = re.sub(_ASIDES, lambda aside: " " * len(aside.group()), joined)
    out, at = [], 0
    for w in heard:
        text = " ".join(words(kept[at:at + len(w["word"])]))  # ♪ has no letters, so it goes too
        at += len(w["word"])
        if text:
            out.append(Word(text, float(w["start"]), float(w["end"])))
    return out
