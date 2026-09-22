"""Rating pitch against a song's melody, taken from a recording of it (`shed reference`).

Without a reference, pitch is rated against the song's scale (rating.py), so a wrong note sung in tune
passes, and so does a whole take sung a semitone under the guitar, as long as it's in tune with itself.
With one, each note you sing is compared with the one the reference sings on the same word:

1. The lyrics line the two up. Each line Whisper hears in your take is placed where it fits best among the
   reference's words, misheard, missing and extra words allowed (a local alignment), so verses sung in
   another order, or left out, don't throw the rest off. On an amateur's takes of a song, 84% of the words
   found their place.
2. Your instrument may play in another key than the reference (a capo moved, the song transposed to suit
   your voice). The chords under the words that matched tell which: of the reference's chords moved to
   each of the 12 keys, the one closest to yours, word by word. That was right on all 16 takes it was tried
   on, where comparing the chords over the whole song got 13. The melody is moved there too, so you're
   judged against your instrument.
3. The notes you hold on each matched word (100 ms or more) are compared in order with the reference's on
   the same word: cents from its note (the nearest semitone of its own tuning), moved to your key, measured
   from your instrument's tuning. Octaves are ignored (a man and a woman sing the same melody an octave
   apart), nothing else is: a note a semitone off counts as a semitone off.
4. A line's offset is the median of its notes', so the odd misread note counts for little, but a line sung
   flat shows. How far off the take is: the median of its lines' offsets, either way. Where it sits (under
   or over): the median of the signed ones.

Against the original, a professional acoustic cover measured 12¢, with 3 lines of 22 at 60¢ or more:
the rate at which lines get misread even when they're sung right.
"""

import json
from dataclasses import dataclass

import numpy as np

VERSION = 1  # bump when what's extracted changes, so that takes get re-analyzed

MIN_NOTE_SECONDS = 0.1
MIN_LINES = 3  # with fewer lines compared, the take's pitch is rated against the scale instead
FAR_CENTS = 60  # lines this far from the melody are pointed out
_WORD_SLACK = 0.15  # a note starting this close to a word (in seconds) is sung on it
_SAME_WORD = 0.7  # how alike two words must be (difflib's ratio) to count as one, misheard or not
_MIN_LINE_SCORE = 4  # lines matching less than that (about 3 words) aren't placed
_MIN_LINE_NOTES = 3
_CHORD_RATE, _CHORD_HOP = 22050, 2048  # the chords are read about every 0.1 s


@dataclass
class Melody:
    """What's kept of a take, or of a reference recording, to compare melodies."""

    lines: list[str]  # the lines as heard
    words: list[tuple[str, float, float, int]]  # (word as lyrics.words() writes it, start, end, line number)
    notes: list[tuple[float, float, float]]  # (start, end, semitones from A of the instrument's tuning)
    chords: list[list[float]] | None  # the instrument's 12 pitch classes under each word; None a cappella

    def to_json(self) -> str:
        return json.dumps({"version": VERSION, "lines": self.lines, "words": self.words, "notes": self.notes,
                           "chords": self.chords}, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Melody | None":
        """None if unreadable, or extracted by an older version."""
        try:
            data = json.loads(text)
            if data["version"] != VERSION:
                return None
            return cls(data["lines"], [tuple(w) for w in data["words"]], [tuple(n) for n in data["notes"]],
                       data["chords"])
        except (ValueError, KeyError, TypeError):
            return None


@dataclass
class Line:
    start: float  # when the line starts in the take, in seconds
    text: str  # as heard in the take
    cents: float  # how far its notes are from the melody (the median): under it if negative


@dataclass
class Comparison:
    """A take against the song's reference melody."""

    shift: int  # semitones the take is played above the reference (0-11)
    by_instrument: bool  # the shift was read from the chords; a cappella, it's the key the voice fits best
    lines: list[Line]  # the lines compared, in the order sung

    @property
    def typical(self) -> float:
        """How far the lines typically are from the melody, either way."""
        return float(np.median([abs(line.cents) for line in self.lines]))

    @property
    def where(self) -> float:
        """Where the lines typically sit: under the melody if negative."""
        return float(np.median([line.cents for line in self.lines]))


def extract(stems, lines, track) -> Melody:
    """The notes held and the chords under each word, from a take's (or a reference's) separated stems,
    the lines Whisper heard in its voice (transcribe.Line) and its pitch track (rating.Track)."""
    from woodshed import rating

    cents, hop = track.cents, track.hop_seconds
    stretches = rating.held_stretches(cents, track.voiced, hop, MIN_NOTE_SECONDS)
    pitches = np.array([np.median(cents[a:b]) for a, b in stretches])
    tuning = track.tuning
    if tuning is None:  # a cappella: the notes' own tuning
        tuning = rating.circular_mean(pitches) if len(pitches) else 0.0
    notes = [(round(a * hop, 2), round(b * hop, 2), round(float(pitch - tuning) / 100, 3))
             for (a, b), pitch in zip(stretches, pitches)]
    words = [(w.text, round(float(w.start), 2), round(float(w.end), 2), n)
             for n, line in enumerate(lines) for w in line.words]
    chords = None if track.tuning is None else _chords(stems.accompaniment, stems.rate, track.tuning, words)
    return Melody([line.text for line in lines], words, notes, chords)


def _chords(accompaniment: np.ndarray, rate: int, tuning: float, words: list[tuple]) -> list[list[float]]:
    import librosa

    y = librosa.resample(accompaniment, orig_sr=rate, target_sr=_CHORD_RATE)
    chroma = librosa.feature.chroma_cqt(y=y, sr=_CHORD_RATE, tuning=tuning / 100, hop_length=_CHORD_HOP)
    frames = chroma.shape[1]
    out = []
    for _, start, end, _ in words:
        a = min(int(start * _CHORD_RATE / _CHORD_HOP), frames - 1)
        b = max(int(end * _CHORD_RATE / _CHORD_HOP) + 1, a + 1)
        c = chroma[:, a:b].mean(axis=1)
        out.append(np.round(c / max(np.linalg.norm(c), 1e-9), 3).tolist())
    return out


def compare(take: Melody, reference: Melody) -> Comparison | None:
    """How the take's melody compares with the reference's, line by line; None if too few of its lines
    could be compared (not enough words matched, or notes held on them)."""
    pairs = align(take.words, reference.words)
    take_notes, reference_notes = _notes_on_words(take), _notes_on_words(reference)
    compared = []  # (line, take note, reference note)
    for k, l in pairs:
        mine, theirs = take_notes[k], reference_notes[l]
        if mine and theirs:
            for p, note in enumerate(mine):  # in order, spread over the reference's notes on the word
                compared.append((take.words[k][3], note, theirs[round(p * (len(theirs) - 1) / max(len(mine) - 1, 1))]))
    if not compared:
        return None
    line_of, sung, meant = (np.array(column) for column in zip(*compared))
    pitch = np.array([note[2] for note in take.notes])[sung]
    target = np.round(np.array([note[2] for note in reference.notes]))[meant]

    def offsets(shift: int) -> np.ndarray:
        return 100 * ((pitch - (target + shift) + 6) % 12 - 6)  # cents, octaves ignored

    by_instrument = take.chords is not None and reference.chords is not None
    if by_instrument:
        mine = np.array([take.chords[k] for k, _ in pairs])
        theirs = np.array([reference.chords[l] for _, l in pairs])
        shift = int(np.argmax([np.sum(mine * np.roll(theirs, t, axis=1)) for t in range(12)]))
    else:  # no chords to go by: the key the voice fits best
        shift = int(np.argmax([np.mean(np.abs(offsets(t)) < 50) for t in range(12)]))
    off = offsets(shift)
    first_word = {}
    for word in take.words:
        first_word.setdefault(word[3], word[1])
    lines = [Line(first_word[n], take.lines[n], float(np.median(off[line_of == n])))
             for n in np.unique(line_of) if np.sum(line_of == n) >= _MIN_LINE_NOTES]
    return Comparison(shift, by_instrument, lines) if len(lines) >= MIN_LINES else None


def align(take: list[tuple], reference: list[tuple]) -> list[tuple[int, int]]:
    """Each of the take's lines placed where it fits best among the reference's words: the (take word,
    reference word) pairs heard as the same word."""
    from difflib import SequenceMatcher

    vocabulary = sorted({w[0] for w in reference})
    position = {word: i for i, word in enumerate(vocabulary)}
    in_reference = np.array([position[w[0]] for w in reference], int)
    likeness: dict[str, list[float]] = {}  # a take word -> how alike it is to each of the reference's words
    lines: dict[int, list[int]] = {}
    for i, w in enumerate(take):
        lines.setdefault(w[3], []).append(i)
        if w[0] not in likeness:
            alike = np.array([SequenceMatcher(None, w[0], v).ratio() for v in vocabulary])
            likeness[w[0]] = alike[in_reference].tolist()
    pairs = []
    for rows in lines.values():
        pairs += [(rows[i], j) for i, j in _place([likeness[take[k][0]] for k in rows])]
    return pairs


def _place(alike: list[list[float]]) -> list[tuple[int, int]]:
    """Local alignment (Smith-Waterman) of a line's words with the reference's: a word matching one of theirs
    scores up to 2, one that doesn't -1, and so does leaving out a word on either side. Returns the matching
    (line word, reference word) pairs of the best-scoring stretch, if it scores enough."""
    if not alike:
        return []
    n, m = len(alike), len(alike[0])
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]  # 1: both words, 2: leave out the line's, 3: the reference's
    best, end = 0.0, (0, 0)
    for i in range(1, n + 1):
        row, above, moves, like = score[i], score[i - 1], move[i], alike[i - 1]
        for j in range(1, m + 1):
            h, how = 0.0, 0
            both = above[j - 1] + (2 * like[j - 1] if like[j - 1] >= _SAME_WORD else -1.0)
            if both > h:
                h, how = both, 1
            if above[j] - 1.0 > h:
                h, how = above[j] - 1.0, 2
            if row[j - 1] - 1.0 > h:
                h, how = row[j - 1] - 1.0, 3
            row[j], moves[j] = h, how
            if h > best:
                best, end = h, (i, j)
    if best < _MIN_LINE_SCORE:
        return []
    found, (i, j) = [], end
    while i > 0 and j > 0 and move[i][j]:
        if move[i][j] == 1:
            if alike[i - 1][j - 1] >= _SAME_WORD:
                found.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move[i][j] == 2:
            i -= 1
        else:
            j -= 1
    return found[::-1]


def _notes_on_words(melody: Melody) -> list[list[int]]:
    """For each word, the notes sung on it, in order: those starting while it's sung (or just before or
    after), each on the word it overlaps most."""
    on = [[] for _ in melody.words]
    if not melody.notes or not melody.words:
        return on
    onset, end = (np.array(column)[:, None] for column in list(zip(*melody.notes))[:2])
    start, stop = (np.array(column)[None] for column in list(zip(*melody.words))[1:3])
    near = (onset >= start - _WORD_SLACK) & (onset <= stop + _WORD_SLACK)
    overlap = np.where(near, np.minimum(end, stop) - np.maximum(onset, start), -np.inf)
    for note in np.flatnonzero(near.any(axis=1)):
        on[int(np.argmax(overlap[note]))].append(int(note))
    return on


def transposed(comparison: Comparison) -> str | None:
    """Your instrument's key against the reference's, when it isn't the same."""
    shift = (comparison.shift + 6) % 12 - 6
    if not comparison.by_instrument or shift == 0:
        return None
    return (f"Your instrument plays {_semitones(abs(shift))} {'under' if shift < 0 else 'over'} the reference, "
            "so the melody was moved to match.")


def side(comparison: Comparison) -> str | None:
    """"under" or "over" when most of your lines (70%, of 5 or more) sit on that side of the melody, and
    typically by 15¢ or more."""
    where, offsets = comparison.where, np.array([line.cents for line in comparison.lines])
    if len(offsets) < 5 or abs(where) < 15 or np.mean(np.sign(offsets) == np.sign(where)) < 0.7:
        return None
    return "under" if where < 0 else "over"


def sitting(comparison: Comparison) -> str | None:
    """Where your lines sit against the melody, when most of them are on the same side."""
    where, direction = comparison.where, side(comparison)
    if not direction:
        return None
    if abs(where) < FAR_CENTS:
        return f"Your lines sit a little {direction} the melody (about {abs(where):.0f}¢)."
    return f"Your lines sit about {_semitones(int(abs(where) / 100 + 0.5))} {direction} the melody."


def furthest(comparison: Comparison, most: int = 3) -> list[Line]:
    """The lines furthest from the melody, furthest first."""
    far = [line for line in comparison.lines if abs(line.cents) >= FAR_CENTS]
    return sorted(far, key=lambda line: -abs(line.cents))[:most]


def _semitones(n: int) -> str:
    return "a semitone" if n == 1 else f"{n} semitones"
