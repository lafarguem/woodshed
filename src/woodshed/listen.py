"""Following you as you sing, for `shed drill`: how far from the melody you sit, and asking for another song.

Every few seconds, the latest 15 s from the microphone go through what files a take (your voice separated from
the instrument, Whisper, the pitch tracker: melody.extract()), in a process of its own so the tuner never waits
for it. A pass takes about 4 s on an M4, so what it says is 4-8 s behind you. Nothing is kept.

- Against the melody: the notes held on words matched to the reference's (melody.note_offsets()), in the key
  your instrument is taken to play in. Each of your latest 15 votes for how many semitones it sits from the
  melody (octaves ignored), and the most votes win (settled()). A median would be thrown by the notes a
  semitone or two either side, and by those either side of 6 semitones off. On 49 takes of 9 songs, the votes
  mostly held from one reading to the next, and moved when the singer found the melody or lost it. Only shown,
  never played: it doesn't stop you singing.
- Another song: say its name, on its own or with a few words around it ("switch to…", "let's do…"; asked_for()).
  A name said alone is often misheard ("space song" was heard as "Space on"); after a few words, it wasn't. It
  counts only while your instrument is quiet under the name, so singing a line with a song's name in it doesn't
  count, and only if the name is most of what's said, in a line that isn't one of the song's own (sung without
  the instrument). On 30 s of 4 songs' takes, 90% of the words sung over the instrument had it no more than 19 dB under
  the voice; with the voice alone (separated again, the instrument gone), 90% had it at least 32 dB under. Only
  the name's words are judged, so a chord still ringing as you start to speak doesn't matter.
"""

import multiprocessing
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from woodshed.lyrics import words as _words
from woodshed.melody import Melody, align, note_offsets

WINDOW_SECONDS = 15.0  # of the latest audio in each pass
MIN_SECONDS = 5.0  # heard before the first pass
SETTLED_SECONDS = 2.0  # notes are taken once this far from the end of a pass: Whisper places the last words worst
COMMAND_SECONDS = 1.0  # a song's name is taken once it ends this far from the end of a pass, so it's all heard
RECENT_NOTES = 15
MIN_NOTES = 8  # compared before saying where you sit
QUIET_INSTRUMENT_DB = -25.0  # the instrument this far under the voice (or more), it isn't played
SAME_TITLE = 0.8  # how alike (difflib's ratio) what's said must be to a song's name
_FRAME_SECONDS = 0.02
MAX_EXTRA_WORDS = 3  # said with a song's name, in the same line, for it to count


@dataclass
class Heard:
    """What one pass heard, timed from when listening started."""

    end: float  # when the audio it went over ends
    melody: Melody
    instrument_db: list[float]  # for each word, how loud the instrument was against the voice


def hear(start: float, samples: np.ndarray, rate: int, language: str | None = None) -> Heard:
    """One pass over samples heard from `start` (seconds since listening started)."""
    from woodshed import isolate, melody, rating, transcribe

    stems = isolate.separate_samples(samples, rate)
    lines = transcribe.transcribe(stems.vocals_16k(), language)
    heard = melody.extract(stems, lines, rating.pitch_track(stems))
    loudness = [_instrument_db(stems, a, b) for _, a, b, _ in heard.words]
    shifted = Melody(heard.lines, [(w, a + start, b + start, n) for w, a, b, n in heard.words],
                     [(a + start, b + start, p) for a, b, p in heard.notes], heard.chords)
    return Heard(start + len(samples) / rate, shifted, loudness)


def _instrument_db(stems, start: float, end: float) -> float:
    """How loud the instrument is against the voice while a word is sung: where the voice is within 10 dB of its
    loudest in it. Whisper can start a word said after the instrument stops half a second early, in the chord's
    last ringing."""
    frame = int(_FRAME_SECONDS * stems.rate)
    n = min(len(stems.vocals), len(stems.accompaniment))  # Whisper can end a word past the end of what was heard
    a = min(int(start * stems.rate), n)
    b = min(max(int(end * stems.rate), a + frame), n)
    voice, instrument = (x[a:b][:(b - a) // frame * frame].reshape(-1, frame) for x in (stems.vocals,
                                                                                      stems.accompaniment))
    voice, instrument = np.sqrt(np.mean(voice**2, axis=1)), np.sqrt(np.mean(instrument**2, axis=1))
    sounding = voice >= voice.max() / np.sqrt(10) if len(voice) else voice
    if not sounding.any():
        return 0.0
    return float(20 * np.log10(max(np.sqrt(np.mean(instrument[sounding] ** 2)), 1e-9)
                               / max(np.sqrt(np.mean(voice[sounding] ** 2)), 1e-9)))


def asked_for(heard: Heard, titles: list[str], after: float, before: float,
              lyrics: list[tuple] | None = None) -> str | None:
    """The song whose name is said on its own, or with a few words around it, ending between `after` and
    `before`, the instrument quiet under it; not in a line of `lyrics` (the song's words, as Melody.words)."""
    said = heard.melody.words
    lines: dict[int, list[int]] = {}
    for k, word in enumerate(said):
        lines.setdefault(word[3], []).append(k)
    for line in lines.values():
        best, alike, where = None, SAME_TITLE, []
        for title in titles:
            name = " ".join(_words(title))
            size = len(name.split())
            if len(line) > size + MAX_EXTRA_WORDS:
                continue
            for n in {max(size - 1, 1), size, size + 1}:  # Whisper may run two words together, or split one
                for i in range(len(line) - n + 1):
                    likeness = SequenceMatcher(None, " ".join(said[k][0] for k in line[i:i + n]), name).ratio()
                    if likeness >= alike:
                        best, alike, where = title, likeness, line[i:i + n]
        if (best and after <= said[where[-1]][2] < before
                and np.median([heard.instrument_db[k] for k in where]) <= QUIET_INSTRUMENT_DB
                and not _sung(said, line, where, lyrics)):
            return best
    return None


def _sung(said: list[tuple], line: list[int], name: list[int], lyrics: list[tuple] | None) -> bool:
    """Whether the words said around a song's name are the song's own (2 or more of them, in their place): a
    line of it, sung without the instrument. Another song's name in its lyrics doesn't count."""
    if not lyrics:
        return False
    matched = {line[i] for i, _ in align([said[k] for k in line], lyrics)}
    return len(matched - set(name)) >= 2


def settled(offsets: list[float]) -> tuple[int, int, int] | None:
    """Where your latest notes sit against the melody: (semitones, under it if negative; how many of them sit
    there; out of how many), or None before MIN_NOTES are compared."""
    recent = np.asarray(offsets[-RECENT_NOTES:])
    if len(recent) < MIN_NOTES:
        return None
    votes = np.bincount(np.round(recent).astype(int) % 12, minlength=12)
    best = int(np.argmax(votes))
    return (best + 6) % 12 - 6, int(votes[best]), len(recent)


@dataclass
class Follow:
    """What's been heard of one song since it was chosen."""

    reference: Melody
    shift: int  # semitones your instrument plays above the reference
    since: float = 0.0  # when the song was chosen: nothing before counts
    offsets: list[float] = field(default_factory=list)  # each note compared, in semitones from the melody
    line: int | None = None  # the reference's line last sung
    sung_at: float = -np.inf  # when a note was last compared
    _notes_until: float = 0.0
    _commands_until: float = 0.0

    def __post_init__(self):
        self._notes_until = self._commands_until = self.since

    def update(self, heard: Heard, titles: list[str]) -> str | None:
        """Takes in a pass; returns the song asked for, if another one was."""
        until = heard.end - SETTLED_SECONDS
        for start, line, off in note_offsets(heard.melody, self.reference, self.shift):
            if self._notes_until <= start < until:
                self.offsets.append(off)
                self.line, self.sung_at = line, start
        self._notes_until = max(self._notes_until, until)
        until = heard.end - COMMAND_SECONDS
        song = asked_for(heard, titles, self._commands_until, until, self.reference.words)
        self._commands_until = max(self._commands_until, until)
        return song


class Listener:
    """Passes over the latest audio, in a process of their own; results come back as they're ready. They talk
    through a pipe, which holds no lock: stopped mid-pass, the process leaves nothing behind (a queue's lock
    would be, and Python warns of it on the way out)."""

    def __init__(self, rate: int, language: str | None = None):
        context = multiprocessing.get_context("spawn")
        self.rate = rate
        self._pipe, theirs = context.Pipe()
        self._process = context.Process(target=_work, args=(theirs, language), daemon=True)
        self._process.start()
        theirs.close()
        self._audio = np.zeros(0, np.float32)
        self._heard = 0  # samples fed so far
        self._busy = True  # until the models are loaded
        self.ready = False

    @property
    def now(self) -> float:
        """Seconds of audio fed so far."""
        return self._heard / self.rate

    def feed(self, samples: np.ndarray) -> None:
        self._heard += len(samples)
        self._audio = np.concatenate([self._audio, samples])[-int(WINDOW_SECONDS * self.rate):]
        if not self._busy and len(self._audio) >= MIN_SECONDS * self.rate:
            self._pipe.send((self.now - len(self._audio) / self.rate, self._audio.copy(), self.rate))
            self._busy = True

    def results(self) -> list[Heard]:
        out = []
        while self._pipe.poll():
            try:
                result = self._pipe.recv()
            except EOFError:  # the process stopped
                break
            self._busy = False
            if result == "ready":
                self.ready = True
            else:
                out.append(result)
        return out

    def close(self) -> None:
        """Stops at once, even mid-pass: nothing it would say is needed any more."""
        self._process.kill()
        self._process.join()
        self._pipe.close()


def _work(pipe, language: str | None) -> None:
    import threading

    import tqdm

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Whisper's progress bar would make a lock shared between processes, left behind when `shed drill` stops this
    # one mid-pass (Python warns of it on the way out). Nothing here is shown, so a lock of its own does.
    tqdm.tqdm.set_lock(threading.RLock())
    hear(0.0, np.zeros(16_000, np.float32), 16_000, language)  # loads the models
    pipe.send("ready")
    while True:
        try:
            job = pipe.recv()
        except EOFError:  # `shed drill` stopped
            return
        pipe.send(hear(*job, language))
