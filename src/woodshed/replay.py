"""Replaying a take over its song's melody (`shed replay`): the take plays, the melody of the song's reference
recording scrolls by with its words, as when singing along (along.py), and what you sang is drawn over it.

The page is `shed serve`'s (phone.html), whose Replay tab does the same for any take (Session); `shed replay` shows
it with nothing but the take asked for (Server).

Singing along, the recording sets the time. A take has its own: another tempo, an intro of its own, a solo left
out, a chorus sung once more. So the melody is laid along the take first, each moment of the take given the moment
of the reference it sings (follow()). As nothing is live, the whole take is there to go by:

- Its pitch is read with all of the take around each moment (rating.pitch_track()'s way), not half a second of it.
- The two melodies are laid along each other moment by moment (dynamic time warping): the take may go up to twice
  as fast as the reference or half as fast, a note of yours on the same note of the melody (in any octave) costing
  nothing, one _FAR semitones off or more the most, and so does singing where the reference doesn't, or the other
  way round.
- A melody alone doesn't tell the first verse from the second. The words do: where a few words in a row were heard
  in both (anchors()), the take is drawn there. Every place a run of words is sung in the reference counts, so a
  chorus goes to whichever of its times keeps the take going forward.
- The take may leave the reference where it is and pick it up elsewhere (a solo left out, a verse sung again), at
  a price (_JUMP): what it's worth over some seconds of singing. And the reference may wait while the take goes
  on (a longer intro, a break), for next to nothing while nothing is sung: nothing is drawn of it meanwhile.
- A take sung in another key than the one the melody is looked for in would cost the most everywhere, and be laid
  anywhere. So it's first laid in each of the 12 keys, _ROUGHLY, and the cheapest is its key; then laid again
  with the melody moved to where the voice sat between two keys, until that's less than _LEAN away
  (Followed.moved).

The melody is then drawn as the take goes (laid()), in the key of the take's instrument (melody.key(): a capo
moved, the song transposed) and at its tuning, in the octave nearest the voice. When the voice is nowhere near it
there (OTHER_KEY or more away, all through), it's drawn where the voice sings it (drawn_in()).

Tried on 25 takes of 9 songs, by one amateur:
- 12 were sung a semitone and a half or more from the melody as their instrument played it, all through: 8 of
  them a fourth or a fifth away, though the chords were clearly in the reference's key (melody.key()'s closeness:
  0.71 to 0.87 there, and nowhere more). Laid where the voice sings (which is why the 12 keys are tried), 38 to
  64% of the moments sung in both were within half a semitone of the reference's melody on 10 of them; laid in
  the instrument's key, 13 to 41%, and the words a second or two from their notes.
- Over the 25, 54% of the moments sung in both were within half a semitone (19 to 79%), and a take was followed in
  1 to 7 stretches (2.7 on average).
- Leaving the reference at 80 and a moment sung in one only at 0.6 followed the takes in fewer stretches (1.9), but
  laid a chorus sung twice after a solo left out over the solo. At 50, and 0.8 for the take's, it's followed.
- With the reference free to wait for 1 (_WAIT_START) and a moment sung meanwhile at 0.6, it waited at every
  breath: one take's 150 s were laid on the reference's first 60, in 36 stretches.
- On a take of Space Song (four choruses where the reference sings two, then six for eight), each of the 20 lines
  came up within a second of where the take's own words were heard.
"""

import json
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from woodshed.along import CONTOUR_HOP

_STEP = 0.04  # seconds: how finely the take is laid along the reference…
_MOST_CELLS = 60e6  # …or coarser, for the two to fit in this many pairs of moments (a 5 minute take and reference do)
_FAR = 2.0  # semitones off the melody's note (in any octave) from which a moment costs the most, 1
_TAKE_ALONE = 0.8  # the cost of a moment sung in the take and not in the reference…
_REFERENCE_ALONE = 0.5  # …and of one sung in the reference only (a backing voice, a moment the take's wasn't heard)
_UNEVEN = 0.05  # the cost of going twice as fast as the reference, or half as fast, for a moment
_JUMP = 50.0  # the cost of leaving the reference to pick it up elsewhere (for moments of _STEP)
_WAIT = 0.02  # the cost of a moment of the take while the reference waits, when nothing is sung (sung: 1)…
_WAIT_START = 15.0  # …and of making it wait
_HELD, _STAYED = 4, 8  # see _lay()
_ROUGHLY = 4  # how many times coarser the take is laid, to find its key
SITS = 0.15  # semitones the voice sits from the melody as drawn, all through, from which it's said
OTHER_KEY = 1.5  # semitones from the melody as its instrument plays it, all through, from which a take sings it elsewhere
_MIN_NOTE = 0.03  # seconds: what's left of a tab's note where the take leaves it isn't drawn if shorter
_WORD = 0.5  # what a moment on a word heard in both is worth
_WORD_SLACK = 0.2  # seconds: a word is sung up to this long before or after where it was heard
_RUN = 3  # words in a row heard in both, for them to count
_SAME_WORD = 0.7  # how alike two words must be (difflib's ratio) to be the same word
_LEAN = 0.25  # semitones the voice sits from the melody, from which the take is laid again with the melody moved


@dataclass
class Followed:
    """A take laid along its song's reference."""

    # The stretches of the take that follow the reference without leaving it, in the take's order: each a list of
    # (seconds into the take, seconds into the reference), both going up.
    stretches: list[list[tuple[float, float]]]
    moved: float  # how many semitones above the reference's melody the take sings it (0-12, octaves aside)


def anchors(take: list[tuple], reference: list[tuple]) -> list[tuple[float, float, float, float]]:
    """Where the same words were heard in a take and in the reference (melody.Melody.words): (start and end in the
    take, start and end in the reference) of each word that's one of _RUN in a row heard in both. A word sung
    several times in the reference is there for each of them."""
    from difflib import SequenceMatcher

    if not take or not reference:
        return []
    theirs = sorted({w[0] for w in reference})
    alike = {mine: np.array([mine == other or SequenceMatcher(None, mine, other).ratio() >= _SAME_WORD
                             for other in theirs]) for mine in {w[0] for w in take}}
    column = np.array([theirs.index(w[0]) for w in reference])
    same = np.array([alike[w[0]][column] for w in take])
    before = np.zeros(same.shape, int)  # how many words in a row are the same up to each pair, and from it
    after = np.zeros(same.shape, int)
    for k in range(len(take)):
        before[k] = same[k]
        if k:
            before[k, 1:] += same[k, 1:] * before[k - 1, :-1]
    for k in range(len(take) - 1, -1, -1):
        after[k] = same[k]
        if k + 1 < len(take):
            after[k, :-1] += same[k, :-1] * after[k + 1, 1:]
    return [(take[k][1], take[k][2], reference[l][1], reference[l][2])
            for k, l in zip(*np.nonzero(before + after - 1 >= _RUN))]


def follow(sung: list[float | None], melody: list[float | None], hop: float = CONTOUR_HOP,
           words: list[tuple[float, float, float, float]] = ()) -> Followed:
    """Lays a take along its song's reference (see the top): `sung` and `melody` are their singers' pitch every
    `hop` (along.contour()), `words` where the same words were heard in both (anchors())."""
    step = max(1, round(_STEP / hop))
    while (len(sung) / step) * (len(melody) / step) > _MOST_CELLS:
        step *= 2
    # Which of the 12 keys the take sings the melody in: the one it's laid along most cheaply, _ROUGHLY.
    rough = _Laying(sung, melody, hop, step * _ROUGHLY, words)
    moved = float(np.argmin([rough.lay(key)[2] for key in range(12)]))
    laying = _Laying(sung, melody, hop, step, words)
    for _ in range(3):  # …and where between two keys the voice sits: laid again with the melody moved there
        position, left, _ = laying.lay(moved)
        followed = ~np.isnan(position)
        on = np.round(position[followed]).astype(int)
        both = ~np.isnan(laying.mine[followed]) & ~np.isnan(laying.theirs[on])
        if not both.any():
            break
        sits = float(np.median((laying.mine[followed][both] - laying.theirs[on][both] - moved + 6) % 12 - 6))
        moved = (moved + sits) % 12
        if abs(sits) < _LEAN:
            break
    stretches: list[list[tuple[float, float]]] = []
    for row in np.flatnonzero(followed):
        if left[row] or not stretches or np.isnan(position[row - 1]):
            stretches.append([])
        stretches[-1].append((laying.at(row), laying.at(position[row])))
    return Followed(stretches, moved)


class _Laying:
    """A take and its reference, every `step` moments, to lay one along the other (lay())."""

    def __init__(self, sung: list[float | None], melody: list[float | None], hop: float, step: int,
                 words: list[tuple[float, float, float, float]]):
        self.mine, self.theirs = _coarser(sung, step), _coarser(melody, step)
        self.seconds = step * hop  # between two moments
        self.middle = (step - 1) / 2 * hop  # of a moment, from its start
        frames = lambda start, end, count: (max(0, int((start - _WORD_SLACK) / self.seconds)),  # noqa: E731
                                            min(count, int((end + _WORD_SLACK) / self.seconds) + 1))
        # For each moment of the take, the stretches of the reference where a word sung at that moment is sung.
        self.worded: list[list[tuple[int, int]]] = [[] for _ in self.mine]
        for start, end, their_start, their_end in words:
            columns = frames(their_start, their_end, len(self.theirs))
            for row in range(*frames(start, end, len(self.mine))):
                self.worded[row].append(columns)

    def at(self, moment: float) -> float:
        """When a moment is, in seconds."""
        return float(moment * self.seconds + self.middle)

    def lay(self, moved: float) -> tuple[np.ndarray, np.ndarray, float]:
        """With the melody `moved` semitones up: for each moment of the take, the moment of the reference it's
        laid on (between two, when the take goes half as fast; NaN while the reference waits), whether the take
        left the reference just before it, and what it all costs: the cheapest way through (see the top)."""
        often = _STEP / self.seconds  # what's paid once is paid as for moments of _STEP
        return _lay(self.mine, self.theirs, self.worded, moved, _JUMP * often, _WAIT_START * often)


def _coarser(pitch: list[float | None], step: int) -> np.ndarray:
    """The pitch every `step` moments: the first one sung of each `step`, NaN where none is."""
    whole = np.array([np.nan if p is None else p for p in pitch] + [np.nan] * (-len(pitch) % step), float)
    groups = whole.reshape(-1, step)
    kept = groups[:, 0].copy()
    for column in range(1, step):
        kept = np.where(np.isnan(kept), groups[:, column], kept)
    return kept


def _lay(mine: np.ndarray, theirs: np.ndarray, worded: list[list[tuple[int, int]]], shift: float, jump: float,
         wait_start: float) -> tuple[np.ndarray, np.ndarray, float]:
    """See _Laying.lay(): `jump` and `wait_start` are what leaving the reference and making it wait cost."""
    n, m = len(mine), len(theirs)
    sings = ~np.isnan(theirs)
    silent = np.where(sings, _REFERENCE_ALONE, 0.0).astype(np.float32)
    # How each pair of moments is best come to. Following (the low bits): 0, both a moment on; 1, the reference two;
    # 2, the take two; 3, from elsewhere in the reference. _HELD: it's cheaper to be waiting there than following.
    # _STAYED: waiting there, the reference was already waiting.
    how = np.zeros((n, m), np.uint8)
    came = np.zeros(n, np.int32)  # where in the reference the take was before leaving it
    never = np.float32(np.inf)
    on = waiting = best = best_earlier = cost_before = None
    for i in range(n):
        if np.isnan(mine[i]):
            cost, wait = silent.copy(), _WAIT
        else:
            off = np.abs((mine[i] - theirs - shift + 6) % 12 - 6)
            cost, wait = np.where(sings, np.minimum(off, _FAR) / _FAR, _TAKE_ALONE).astype(np.float32), 1.0
        if worded[i]:
            on_word = np.zeros(m, bool)
            for first, last in worded[i]:
                on_word[first:last] = True
            cost[on_word] -= _WORD
        if i:
            ways = np.full((4, m), never, np.float32)
            ways[0, 1:] = best[:-1]
            ways[1, 2:] = best[:-2] + _UNEVEN
            if i > 1:
                ways[2, 1:] = best_earlier[:-1] + cost_before[1:] + _UNEVEN
            came[i] = int(np.argmin(best))
            ways[3] = best[came[i]] + jump
            way = np.argmin(ways, axis=0)
            stayed = waiting <= on + wait_start
            on, waiting = cost + ways[way, np.arange(m)], np.where(stayed, waiting, on + wait_start) + wait
            how[i] = way.astype(np.uint8) | (stayed * np.uint8(_STAYED))
        else:
            on, waiting = cost, np.full(m, wait, np.float32)
        how[i] |= (waiting < on) * np.uint8(_HELD)
        best_earlier, best, cost_before = best, np.minimum(on, waiting), cost
    position, left = np.full(n, np.nan), np.zeros(n, bool)
    i, j = n - 1, int(np.argmin(best))
    cheapest = float(best[j])
    held = bool(how[i, j] & _HELD)
    while i >= 0:
        marks = int(how[i, j])
        if held:
            held = bool(marks & _STAYED) if i else False
            i -= 1
            continue
        position[i] = j
        way = marks & 3 if i else 0
        if way == 3:
            left[i] = True
            j = int(came[i])
        elif way == 2:
            position[i - 1] = j - 0.5
            i, j = i - 1, j - 1
        else:
            j -= 1 + way
        i -= 1
        held = i >= 0 and j >= 0 and bool(how[i, j] & _HELD)
        j = max(j, 0)
    return position, left, cheapest


def sung(path: Path, device: str | None = None) -> list[float | None]:
    """A take's pitch all through, as along.contour() gives a reference's: read from the take as recorded, where
    RMVPE follows the voice over the instrument (rating.pitch_track()). On the GPU unless `device` says otherwise
    (rmvpe.pitch())."""
    from woodshed import along, audio, rating, rmvpe

    f0, confidence = rmvpe.pitch(audio.load(path), device)
    return along.contour(rating.Track(f0, confidence, None, rmvpe.HOP_SECONDS))


def tuning(notes: list[tuple[float, float, float]], contour: list[float | None], hop: float = CONTOUR_HOP) -> float:
    """How far from A440 an instrument is tuned, in semitones: a melody's notes (melody.Melody.notes) are counted
    from its tuning, the pitch they were read from (`contour`) from A440."""
    pitch = np.array([np.nan if p is None else p for p in contour], float)
    apart = []
    for start, end, note in notes:
        held = pitch[round(start / hop):max(round(end / hop), round(start / hop) + 1)]
        if np.any(~np.isnan(held)):
            apart.append((np.nanmedian(held) - note + 0.5) % 1 - 0.5)
    return float(np.median(apart)) if apart else 0.0


def drawn_in(moved: float, key: int | None, tuned: float = 0.0) -> tuple[float, float, int]:
    """Where the melody is drawn for a take that sings it `moved` semitones above the reference (Followed.moved),
    its instrument playing `key` semitones above the reference's (melody.key(); None: no chords to go by) and tuned
    `tuned` semitones above it: (how many semitones above the reference, how far from there the voice sits, how
    many semitones that is above where the instrument plays the melody: 0 unless the voice is OTHER_KEY or more
    from it)."""
    if key is None:  # the key the voice sings in, as near as a semitone
        return round(moved) % 12, moved - round(moved), 0
    lean = (moved - key - tuned + 6) % 12 - 6
    apart = round(lean) if abs(lean) >= OTHER_KEY else 0
    up = key + tuned + apart
    return up % 12, (moved - up + 6) % 12 - 6, apart


def laid(followed: Followed, mine: list[float | None], contour: list[float | None], notes: list[list[float]] | None,
         lines: list[dict], up: float, hop: float = CONTOUR_HOP,
         ) -> tuple[list[float | None], list[list[float]] | None, list[dict]]:
    """A reference's melody as a take goes: its singer's pitch (`contour`, every `hop`), the notes a tab writes
    and its lines with their words (along.Song's), each at the time the take (whose pitch is `mine`) sings it, `up`
    semitones up and in the octave nearest the take's voice. Nothing where the take sings none of it. What the
    take sings twice is there twice."""
    theirs = np.array([np.nan if p is None else p for p in contour], float)
    voice = [p for p in mine if p is not None]
    if voice and not np.all(np.isnan(theirs)):
        up += 12 * round((float(np.median(voice)) - float(np.nanmedian(theirs)) - up) / 12)
    drawn = np.full(len(mine), np.nan)
    placed: list[list[float]] = []
    shown: list[dict] = []
    for stretch in followed.stretches:
        when, there = (np.array(column) for column in zip(*stretch))
        around = (when[1] - when[0]) / 2 if len(when) > 1 else hop / 2
        frames = np.arange(max(0, int(np.ceil((when[0] - around) / hop))),
                           min(len(mine), int((when[-1] + around) / hop) + 1))
        drawn[frames] = theirs[np.clip(np.round(np.interp(frames * hop, when, there) / hop).astype(int), 0,
                                       len(theirs) - 1)] + up
        sung_at = lambda seconds: round(float(np.interp(seconds, there, when)), 2)  # noqa: E731
        for start, end, pitch in notes or []:
            if end > there[0] and start < there[-1] and sung_at(end) - sung_at(start) >= _MIN_NOTE:
                placed.append([sung_at(start), sung_at(end), round(pitch + up, 2)])
        for line in lines:  # a line the take leaves halfway goes where most of it is sung
            inside = [there[0] <= word[1] <= there[-1] for word in line["words"]]
            if 2 * sum(inside) > len(inside) or (2 * sum(inside) == len(inside) and inside and inside[0]):
                words = [[word[0], sung_at(word[1]), sung_at(word[2])]
                         + ([[[letters, sung_at(since)] for letters, since in word[3]]] if len(word) > 3 else [])
                         for word in line["words"]]
                shown.append({"text": line["text"], "start": words[0][1], "end": max(word[2] for word in words),
                              "words": words})
    return ([None if np.isnan(p) else round(float(p), 2) for p in drawn],
            sorted(placed) if notes is not None else None, sorted(shown, key=lambda line: line["start"]))


def listenable(path: Path) -> bytes:
    """A take as an mp3, which every browser plays (an Apple Lossless one, only Safari)."""
    import tempfile

    from woodshed import audio

    if path.suffix.lower() == ".mp3":
        return path.read_bytes()
    with tempfile.TemporaryDirectory() as folder:
        audio.save_mp3(audio.load(path, rate=44_100), 44_100, Path(folder) / "take.mp3")
        return (Path(folder) / "take.mp3").read_bytes()


@dataclass
class Replay:
    """A take over its song's melody, as the page shows it."""

    title: str
    take: str  # which take it is, as the page says it
    sung: list[float | None]  # the take's pitch every CONTOUR_HOP (sung())
    contour: list[float | None]  # the melody as the take goes, its notes and its lines (laid())
    notes: list[list[float]] | None
    lines: list[dict]
    key: int  # how many semitones above the reference's the take's instrument plays (under it if negative)
    apart: int  # how many semitones above where the instrument plays it the melody is drawn: where the voice sings it
    lean: float  # how far from the melody as drawn the voice sits all through, in semitones (under it if negative)
    audio: bytes  # the take, as an mp3

    def shown(self) -> dict:
        """What the page draws."""
        return {"title": self.title, "take": self.take, "lines": self.lines, "notes": self.notes,
                "lean": round(self.lean, 2), "sits": abs(self.lean) >= SITS, "contour": {"hop": CONTOUR_HOP, "pitch": self.contour},
                "sung": {"hop": CONTOUR_HOP, "pitch": self.sung}}


def of(title: str, label: str, path: Path, take, reference, contour: list[float | None],
       notes: list[list[float]] | None, lines: list[dict], device: str | None = None) -> Replay:
    """A take (the recording at `path`, and what was kept of it: a melody.Melody, or None) over its song's
    reference: its melody.Melody, its singer's pitch all through (`contour`), and the tab's notes and the lines
    shown when singing along to it (along.Song's). `device`: see sung()."""
    from woodshed import melody
    from woodshed.lyrics import words as split

    mine = sung(path, device)
    theirs = [("".join(split(word[0])), word[1], word[2]) for line in lines for word in line["words"]]
    followed = follow(mine, contour, words=anchors(take.words if take else [], [word for word in theirs if word[0]]))
    key = melody.key(take, reference, melody.align(take.words, reference.words)) if take else None
    tuned = tuning(take.notes, mine) - tuning(reference.notes, contour) if take else 0.0
    up, lean, apart = drawn_in(followed.moved, key, tuned)
    return Replay(title, label, mine, *laid(followed, mine, contour, notes, lines, up), ((key or 0) + 6) % 12 - 6,
                  apart, lean, listenable(path))


class Session:
    """Replaying takes from `shed serve`'s page: the songs and takes it can pick (`songs()`: [{"title", "takes":
    [when each was recorded, oldest first]}…]), and one of them laid along its melody (`prepare(title, number)`,
    which raises ValueError for a take that isn't there) when the page asks for it. One at a time; the latest few
    are kept, as the page asks for what to draw, then for the take itself, and may come back to it."""

    KEPT = 3

    def __init__(self, songs: Callable[[], list[dict]], prepare: Callable[[str, int], Replay]):
        self.songs, self._prepare = songs, prepare
        self._ready: dict[tuple[str, int], Replay] = {}
        self._lock = threading.Lock()

    def take(self, title: str, number: int) -> Replay:
        with self._lock:
            if (title, number) not in self._ready:
                self._ready[title, number] = self._prepare(title, number)
                for oldest in list(self._ready)[:-self.KEPT]:
                    del self._ready[oldest]
            return self._ready[title, number]


def page(only: bool = False) -> bytes:
    """The page that replays takes: `shed serve`'s (phone.html), or with `only`, that page showing nothing but the
    take `shed replay` was asked for."""
    shown = resources.files("woodshed").joinpath("phone.html").read_bytes()
    return shown.replace(b"<body>", b'<body data-only="replay">', 1) if only else shown


class Server(ThreadingHTTPServer):
    """The page that replays one take, on this Mac only: http://127.0.0.1:<its port>/. It answers what
    `shed serve` does of a take to replay (phone.Server), for this one, whatever is asked for."""

    daemon_threads = True

    def __init__(self, replay: Replay, port: int = 0):
        super().__init__(("127.0.0.1", port), _Handler)
        self.replay = replay

    def server_bind(self) -> None:  # not HTTPServer's, which looks this Mac's name up (socket.getfqdn): slow, unneeded
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:
        pass  # a page closed while it was being answered


class _Handler(BaseHTTPRequestHandler):
    server: Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:
        pass

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Only for this page by its address: a site whose name was made to lead here isn't answered.
        port, path = self.server.server_port, urlsplit(self.path).path
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return self._reply(404, "text/plain", b"Not found")
        if path == "/":
            return self._reply(200, "text/html; charset=utf-8", page(only=True))
        if path == "/replay/take":
            return self._reply(200, "application/json", json.dumps(self.server.replay.shown()).encode())
        if path == "/replay/audio":
            return self._reply(200, "audio/mpeg", self.server.replay.audio)
        self._reply(404, "text/plain", b"Not found")
