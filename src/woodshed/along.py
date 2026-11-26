"""Singing along to a song's reference recording, your pitch drawn as you sing (`shed serve`, the page's Sing
tab): the recording plays in your headphones, its singer's melody scrolls by, and what you sing is drawn over it,
a fifth of a second behind you. You can play your instrument along. Nothing is recorded.

- The recording sets the time: the melody was read from it, so what's drawn and what plays go together, and still
  do once you turn the recording down to sing on your own.
- Your pitch is read by the tracker that rates takes (rmvpe.py), which follows a voice over an instrument. On 90 s
  of each of 8 takes, voice and guitar in one microphone, it showed 69% of the sung moments at the right note; a
  tuner's way of finding the pitch (YIN), 23%, and it took the guitar alone for a voice a third of the time.
- It needs a little of what comes after a moment to be sure of it: read as it's sung, 29% of the sung moments
  were right; 0.1 s later, 61%; 0.2 s later, 67%. So each moment is read LAG_SECONDS after it.
- Half a second of sound is enough for it: 2 s did no better.
- Nothing is assumed of what you sing: the note to expect isn't used to find yours, as you may be far from it.
  What it's sure of (SURE) is drawn plainly, and what it's less sure of (down to FAINT), faintly. Read as Live
  reads them, 63% of the sung moments were drawn plainly at the right note, 2% at a wrong one, and 12% of the
  guitar alone was drawn as a voice; with the faint ones, 80%, 8% and 30%. Only showing the less sure moments
  that carry on a sure note did no better than showing them all.
- The melody heard in the recording wanders: its singer slides into notes and bends them, and the tracker misreads
  some. With a tab of the song (tab.py), the notes it writes are what's drawn and what you're set against instead.
  On one song, the singer's own pitch through a written note was within half a semitone of it for 73% of the 450
  notes it was heard on, and within a semitone and a half for 92%.
- The louder your voice against your instrument, the more is shown: 97% of the sung moments with the voice 10 dB
  over it, 52% with the voice up to 10 dB under. So the microphone goes near your mouth.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTOUR_HOP = 0.02  # the reference's melody is kept every 20 ms
WINDOW_SECONDS = 0.5  # of the latest sound given to the pitch tracker each time
LAG_SECONDS = 0.12  # a moment is read once this much has been heard after it
SURE = 0.5  # how sure the tracker is of a voice, for it to be drawn plainly…
FAINT = 0.2  # …and faintly
_RATE = 16_000  # what the pitch tracker takes
_MIN_SECONDS = 0.2  # heard before the first reading


def contour(track) -> list[float | None]:
    """The voice's pitch through a recording (rating.Track), every CONTOUR_HOP: semitones from A440, None where
    nobody sings."""
    step = max(1, round(CONTOUR_HOP / track.hop_seconds))
    semitones, voiced = track.cents / 100, track.voiced
    return [round(float(np.median(semitones[i:i + step][voiced[i:i + step]])), 2) if voiced[i:i + step].any() else None
            for i in range(0, len(semitones), step)]


class Live:
    """Your pitch as you sing, from the microphone's sound as it comes: each hundredth of a second is read once,
    LAG_SECONDS after it."""

    def __init__(self, rate: int, device: str | None = None):
        self.rate, self.device = rate, device
        self._audio = np.zeros(0, np.float32)
        self._heard = 0  # samples so far
        self._next = 0  # the next hundredth of a second to read

    def feed(self, samples: np.ndarray) -> list[list]:
        """Takes in what was just heard; the moments now read: [seconds since the first sample, semitones from
        A440 (None: no voice), how sure (0-1)]."""
        from scipy.signal import resample_poly

        from woodshed import rmvpe
        from woodshed.drill import level

        self._heard += len(samples)
        self._audio = np.concatenate([self._audio, samples])[-int((WINDOW_SECONDS + 0.02) * self.rate):]
        last = int((self._heard / self.rate - LAG_SECONDS) * 100)
        begin = self._heard - len(self._audio)
        first = -(-begin * 100 // self.rate)  # the first hundredth that starts in what's kept
        window = self._audio[round(first * self.rate / 100) - begin:]
        if last < max(self._next, first) or len(window) < _MIN_SECONDS * self.rate:
            return []
        if np.abs(window).max() < 1e-5:  # the microphone is off
            f0, confidence = np.ones(last - first + 1), np.zeros(last - first + 1)
        else:
            f0, confidence = rmvpe.pitch(level(resample_poly(window, _RATE, self.rate)), self.device)
        read = []
        for moment in range(max(self._next, first), min(last, first + len(f0) - 1) + 1):
            sure = float(confidence[moment - first])
            read.append([moment / 100, round(float(12 * np.log2(f0[moment - first] / 440)), 2) if sure >= FAINT
                         else None, round(sure, 2)])
        self._next = last + 1
        return read


@dataclass
class Song:
    """A song ready to sing along to."""

    title: str
    lines: list[dict]  # {"text", "start", "end", "words": [[word, start, end]…]}, timed in the recording
    contour: list[float | None]  # see contour()
    octaves: int  # how many octaves above the recording's singer your voice usually sits (under: negative)
    band: Path  # the recording without its singer…
    voice: Path  # …and its singer alone, each an mp3
    notes: list[list[float]] | None = None  # the melody as a tab writes it: [start, end, semitones from A440]…


def lines(melody) -> list[dict]:
    """The lines of a reference's melody (melody.Melody) with the times they and their words are sung at."""
    found: dict[int, dict] = {}
    for word, start, end, line in melody.words:
        entry = found.setdefault(line, {"text": melody.lines[line], "start": start, "end": end, "words": []})
        entry["end"] = max(entry["end"], end)
        entry["words"].append([word, start, end])
    return [found[line] for line in sorted(found)]


def octaves(contour: list[float | None], register: float | None) -> int:
    """How many octaves to move the melody to bring it nearest where your voice sits (`register`, semitones from
    A440; None: where it is)."""
    sung = [p for p in contour if p is not None]
    return 0 if register is None or not sung else round((register - float(np.median(sung))) / 12)


class Session:
    """Singing along from the phone: the songs it can pick, what its page needs of one (song(), audio()), and your
    pitch read from its microphone as it comes (start(), hear()). One at a time, the phone's latest. The pitch
    tracker runs on the CPU here, as filing a take uses the GPU (rmvpe.pitch()): 45 ms a reading on an M4."""

    def __init__(self, prepare: Callable[[str], Song], titles: list[str], device: str | None = "cpu",
                 started: Callable[[str], None] | None = None):
        self.titles = titles
        self._prepare, self._device, self._started = prepare, device, started
        self._live: Live | None = None
        self._lock = threading.Lock()

    def song(self, title: str) -> dict:
        """What the page draws and shows. Raises ValueError if it isn't one of `titles`."""
        if title not in self.titles:
            raise ValueError(title)
        song = self._prepare(title)
        return {"title": song.title, "lines": song.lines, "octaves": song.octaves,
                "contour": {"hop": CONTOUR_HOP, "pitch": song.contour}, "notes": song.notes, "sure": SURE}

    def audio(self, title: str, part: str) -> bytes:
        """The recording without its singer ("band"), or its singer alone ("voice"), as an mp3."""
        if title not in self.titles or part not in ("band", "voice"):
            raise ValueError(title)
        song = self._prepare(title)
        return (song.band if part == "band" else song.voice).read_bytes()

    def start(self, title: str, rate: int) -> None:
        """Sound is about to come at `rate`, for this song. Returns once the pitch tracker is ready."""
        if title not in self.titles:
            raise ValueError(title)
        live = Live(rate, self._device)
        live.feed(np.full(rate, 1e-3, np.float32))  # loads the tracker
        with self._lock:
            self._live = Live(rate, self._device)
        if self._started:
            self._started(title)

    def hear(self, samples: np.ndarray) -> list[list] | None:
        """As Live.feed(); None if start() wasn't called."""
        with self._lock:
            return None if self._live is None else self._live.feed(samples)

    def stop(self) -> None:
        with self._lock:
            self._live = None
