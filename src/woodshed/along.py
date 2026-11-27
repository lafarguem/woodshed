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
- The words shown are the ones heard in the recording (Whisper's), with what it mishears. With the song's lyrics
  as written (Genius's, genius.lyrics()), those are shown instead, each word at the time the same word was heard
  (written()): the heard words say when, the written ones what. On 9 songs, 274 of Genius's 289 lines were timed
  so; the others had no word that was heard, nor any other word heard in their place (a "woah" Whisper leaves out,
  a backing voice's line). A tab from Songsterr writes the words under its notes: then a word is shown when its
  note comes, syllable by syllable, as that's where its block is. On 2 songs, a word's note started 0.15 and 0.3 s
  after the word was heard, typically, and within 0.25 s of that for 6 words in 10.
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
_LOUDNESS = 0.1  # what's heard is brought to this RMS level (-20 dBFS)…
_MAX_GAIN = 100.0  # …by at most 40 dB
_SAME_WORD = 0.7  # how alike a written word and a heard one must be (difflib's ratio) to be the same word
# Of the words heard in a recording, how many its written lyrics must have, to be its own: on 9 songs, Genius's had
# 56 to 100% of them (85% overall), and another song's 18% at most.
MIN_HEARD = 0.35
_WORD_SECONDS = 0.4  # a written word nobody heard is given this long, next to one that was…
_APART = 1.0  # a line's words on a tab's notes this far (s) from where they were heard: the tab's have slid, there
_AROUND = 4.0  # …the same word being looked for this long before and after its note
_SLOWEST = 1.5  # …and whole lines nobody heard are only kept where there's from _FASTEST to this long a word
_FASTEST = 0.15


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


def written(text: list[str], melody, sung: list[list] | None = None,
            _not_by_notes: frozenset[int] = frozenset()) -> list[dict] | None:
    """A song's lyrics as written (line by line), timed as lines() times the heard ones: each word at the time the
    same word was heard in the recording (melody.Melody), and those nobody heard in between the ones around them.
    With `sung`, the words a tab writes under its notes, timed in the recording (tab.Placed.words), a word is
    rather timed by the note it's written on, as that's where its block is drawn, and syllable by syllable: its
    entry then ends with [[letters of the word sung so far, since when]…]. The heard words stay for what the tab
    doesn't write, moved by how far apart the two usually are, and for a line where the tab is off (_APART).
    None if they aren't this recording's lyrics: fewer than MIN_HEARD of its heard words are in them, and of the
    tab's."""
    from woodshed.lyrics import words as split

    theirs = [(word, n) for n, line in enumerate(text) for word in split(line)]
    heard = melody.words
    tab = [("".join(split(word)), start, end, parts) for word, start, end, parts in sung or []]
    tab = [word for word in tab if word[0]]
    pairs = _same_words([word for word, _ in theirs], [w[0] for w in heard])
    on_notes = _same_words([word for word, _ in theirs], [w[0] for w in tab])
    if not (heard and len(pairs) >= MIN_HEARD * len(heard)) and not (tab and len(on_notes) >= MIN_HEARD * len(tab)):
        return None
    was_heard = dict(pairs)
    noted = {i: j for i, j in on_notes if theirs[i][1] not in _not_by_notes}
    # The same word is on a note and was heard: the heard one of that text nearest the note, a few seconds around.
    # (Not the one laid along the same written word: where a line is written more times than it's sung, the notes
    # and the words heard may be laid along different ones of them.)
    starts = np.array([w[1] for w in heard])
    near: list[float | None] = []  # for each of the tab's words, how long after the same word was heard
    for word, start, _, _ in tab:
        around = range(int(np.searchsorted(starts, start - _AROUND)), int(np.searchsorted(starts, start + _AROUND)))
        gaps = [start - heard[j][1] for j in around if heard[j][0] == word]
        near.append(min(gaps, key=abs) if gaps else None)
    both = sorted(gap for gap in near if gap is not None and abs(gap) < _APART)
    apart = both[len(both) // 2] if both else 0.0  # the notes are this much after the words heard, usually
    for n in {theirs[i][1] for i in noted}:  # a line whose notes are far from where its words were heard: a
        # syllable too many or too few before it, and the tab's words have slid along its notes
        off = sorted(abs(near[j] - apart) for i, j in noted.items() if theirs[i][1] == n and near[j] is not None)
        if off and off[(len(off) - 1) // 2] > _APART:  # even on one word's say, if no other says otherwise
            noted = {i: j for i, j in noted.items() if theirs[i][1] != n}
    times: list[list[float] | None] = [None] * len(theirs)
    syllables: dict[int, list[list[float]]] = {}
    for i, j in noted.items():
        times[i] = [tab[j][1], tab[j][2]]
        if tab[j][0] == theirs[i][0]:  # the very word, not one like it ("outta" for "out"): its syllables are its own
            lit = _lit(len(theirs[i][0]), [(len("".join(split(part))), when) for part, when in tab[j][3]])
            if lit:
                syllables[i] = lit
    following = [None] * len(theirs)  # the start of the next word timed by its note
    for i in range(len(theirs) - 1, 0, -1):
        following[i - 1] = times[i][0] if times[i] is not None else following[i]
    # The words heard time what the tab doesn't write: not a word heard where the tab has a note with a word, as
    # that's the word sung there, whichever written word each was laid along.
    trusted = np.sort(np.array([tab[j][1] for j in noted.values()]))
    latest = None
    for i in range(len(theirs)):
        if times[i] is None and i in was_heard:
            start, end = heard[was_heard[i]][1] + apart, heard[was_heard[i]][2] + apart
            at = int(np.searchsorted(trusted, start - _APART))
            free = at == len(trusted) or trusted[at] > start + _APART
            if free and (latest is None or start >= latest) and (following[i] is None or start <= following[i]):
                times[i] = [round(start, 2), round(end if following[i] is None else min(end, following[i]), 2)]
        if times[i] is not None:
            latest = times[i][0]
    # Sung, but as no written word: what a line misheard all through was heard as, or written as in the tab.
    unnoted = [tab[j] for j in sorted(set(range(len(tab))) - set(noted.values()))]
    unmatched = [(w[0], w[1] + apart, w[2] + apart)
                 for w in (heard[j] for j in sorted(set(range(len(heard))) - set(was_heard.values())))]
    at = 0
    while at < len(theirs):  # each stretch of words nobody heard, between two that were
        if times[at] is not None:
            at += 1
            continue
        end = at
        while end < len(theirs) and times[end] is None:
            end += 1
        # The end of the line before them comes right after its last heard word, the start of the line after them
        # right before its first. Whole lines in between are sung where other words were heard meanwhile (the
        # line, misheard all through); failing any, they share what time is left, if it's a time to sing them in.
        head = next((k for k in range(at, end) if not at or theirs[k][1] != theirs[at - 1][1]), end)
        tail = next((k + 1 for k in range(end - 1, head - 1, -1)
                     if end == len(theirs) or theirs[k][1] != theirs[end][1]), head)
        low = times[at - 1][1] if at else None
        high = times[end][0] if end < len(theirs) else None
        if low is not None and head > at:
            reach = low + _WORD_SECONDS * (head - at)
            if tail == head and end < len(theirs) and theirs[end][1] == theirs[at - 1][1]:  # inside one line
                reach = min(high, low + _SLOWEST * (head - at))
            reach = reach if high is None else min(reach, high)
            _spread(times, at, head, low, reach)
            low = reach
        if high is not None and end > tail:
            since = max(high - _WORD_SECONDS * (end - tail), 0.0 if low is None else low)
            _spread(times, tail, end, since, high)
            high = since
        if tail > head and low is not None and high is not None:
            others = ([w for w in unnoted if low <= w[1] and w[2] <= high]
                      or [w for w in unmatched if low <= w[1] and w[2] <= high])
            if others:
                low, high = others[0][1], max(word[2] for word in others)
            # With a tab's words, a line that's neither on its notes nor heard isn't sung there: one written
            # more times than its singer sings it (13 "Fall back into place" for 8), say.
            fits = tail - head <= 2 * len(others) + 2 if others else not tab and high - low <= _SLOWEST * (tail - head)
            if fits and _FASTEST * (tail - head) <= high - low:  # not 17 written "woah"s on the 2 heard
                _spread(times, head, tail, low, high)
        at = end
    found: dict[int, dict] = {}
    for i, ((word, n), time) in enumerate(zip(theirs, times)):
        if time is None:
            continue
        entry = found.setdefault(n, {"text": text[n], "start": time[0], "end": time[1], "words": []})
        entry["end"] = max(entry["end"], time[1])
        entry["words"].append([word, time[0], time[1]] + ([syllables[i]] if i in syllables else []))
    # A line is shown whole, so one with words that couldn't be timed (its start, before anything heard) is left out.
    counts = {n: sum(1 for _, line in theirs if line == n) for n in found}
    # A line gone by faster than it can be sung (its notes say one thing, the words heard around it another): timed
    # again by the words heard alone, and left out if that's no better.
    rushed = {n for n, line in found.items() if line["end"] - line["start"] < _FASTEST * len(line["words"]) > _FASTEST}
    by_notes = {theirs[i][1] for i in noted}
    if rushed & by_notes:
        return written(text, melody, sung, _not_by_notes | (rushed & by_notes))
    return [found[n] for n in sorted(found) if len(found[n]["words"]) == counts[n] and n not in rushed]


def _lit(letters: int, parts: list[tuple[int, float]]) -> list[list[float]] | None:
    """How many of a word's letters are sung once each of its syllables starts: [[letters so far, since when]…],
    from the syllables' own lengths, or, written another way than the word, in proportion. None for one syllable."""
    if len(parts) < 2 or letters < len(parts):
        return None
    total, so_far, lit = sum(length for length, _ in parts) or 1, 0, []
    for k, (length, when) in enumerate(parts):
        so_far += length
        upto = letters if k == len(parts) - 1 else max(k + 1, min(letters - (len(parts) - 1 - k),
                                                                   round(letters * so_far / total)))
        lit.append([upto, when])
    return lit


def _spread(times: list, first: int, last: int, low: float, high: float) -> None:
    """Gives the words from `first` to `last` equal shares of the time from `low` to `high`."""
    for k in range(first, last):
        times[k] = [round(low + (high - low) * (k - first) / (last - first), 2),
                    round(low + (high - low) * (k - first + 1) / (last - first), 2)]


def _same_words(theirs: list[str], heard: list[str]) -> list[tuple[int, int]]:
    """The (written word, heard word) pairs that are the same word, the two texts laid along each other from start
    to end (a global alignment: a word matching scores up to 2, a word left out on either side costs 1). A pair is
    only kept next to another one: alone among words left out, it's a chance meeting of two "the"s."""
    from difflib import SequenceMatcher

    if not theirs or not heard:
        return []
    vocabulary = sorted(set(heard))
    position = {word: i for i, word in enumerate(vocabulary)}
    columns = [position[word] for word in heard]
    likeness: dict[str, list[float]] = {}
    for word in set(theirs):
        alike = [SequenceMatcher(None, word, v).ratio() for v in vocabulary]
        likeness[word] = [alike[c] if alike[c] >= _SAME_WORD else 0.0 for c in columns]
    n, m = len(theirs), len(heard)
    score = [[-float(j) for j in range(m + 1)]] + [[-float(i)] + [0.0] * m for i in range(1, n + 1)]
    move = [bytearray(m + 1) for _ in range(n + 1)]  # 1: both words, 2: leave out the written one, 3: the heard one
    for i in range(1, n + 1):
        row, above, moves, like = score[i], score[i - 1], move[i], likeness[theirs[i - 1]]
        for j in range(1, m + 1):
            best, how = above[j - 1] + (2 * like[j - 1] if like[j - 1] else -1.0), 1
            if above[j] - 1.0 > best:
                best, how = above[j] - 1.0, 2
            if row[j - 1] - 1.0 > best:
                best, how = row[j - 1] - 1.0, 3
            row[j], moves[j] = best, how
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if move[i][j] == 1:
            if likeness[theirs[i - 1]][j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move[i][j] == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    near = lambda a, b: 0 < b[0] - a[0] <= 3 and 0 < b[1] - a[1] <= 3  # noqa: E731
    return [pair for k, pair in enumerate(pairs)
            if (k and near(pairs[k - 1], pair)) or (k + 1 < len(pairs) and near(pair, pairs[k + 1]))]


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


def level(samples: np.ndarray) -> np.ndarray:
    """The samples brought to the same loudness, however far you are from the microphone."""
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    return (samples * min(_LOUDNESS / max(rms, 1e-9), _MAX_GAIN)).astype(np.float32)


def register(melodies: list) -> float | None:
    """Where your voice usually sits in these takes (melody.Melody): the median of their notes, in semitones from
    A440."""
    notes = [note[2] for m in melodies for note in m.notes]
    return float(np.median(notes)) if notes else None
