"""A song's melody as written in a tab, to sing along to (along.py) in place of the melody heard in the reference
recording: a singer slides into notes, bends them and adds some, so what's heard wanders; the tab says which notes
the song is made of. The tab comes from Songsterr (songsterr.py), or from a Guitar Pro file, as Songsterr saves them.

- A Guitar Pro file is a zip holding the score as XML (the .gp of Guitar Pro 7 and later). The older .gp3, .gp4,
  .gp5 and .gpx are other formats, not read here, nor is MIDI: it would say no more (notes and when), and a tab site
  gives the .gp.
- A tab doesn't say which of its tracks is the voice (Songsterr names it after the singer, and gives it any
  instrument), nor when the recording starts, nor always its key (a song played with a capo is often written as
  fingered, voice and all) or its exact tempo. All are found by laying each track over the melody heard in the
  recording (along.contour()), slid in time, up to 5% faster or slower, in each of the 12 keys: the track and
  the place where the most of its notes fall on the same note as the singer's, whatever the octave (place()).
- Repeats are played out as written (a section between repeat signs, the times it says, and on each time the
  alternate ending for it, from Songsterr). A Guitar Pro file with alternate endings or jumps (D.S., coda…) isn't
  read: where it goes next isn't followed here.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

_QUARTERS = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
             "128th": 0.03125}
_BEAT_UNITS = {"1": 0.5, "2": 1.0, "3": 1.5, "4": 2.0, "5": 3.0}  # what a tempo counts, in quarter notes
_GRID = 0.02  # seconds: tracks are laid over the melody heard this finely
MAX_SHIFT_SECONDS = 30.0  # how far from the tab's start the recording's may be, either way
_STRETCHES = np.arange(0.95, 1.0501, 0.002)  # the recording against the tab's tempo (then every 0.0002 around the best)
MIN_NOTES = 20  # a track with fewer isn't a melody
MIN_SHARE = 0.2  # of a track's notes on the singer's, for it to be the voice (other tracks: 0.21 at most)
# How far the voice's best fit stands out from its others (every key and moment), in standard deviations: 10 to 16
# on 3 songs, where other tracks got 6.6 at most (a synth doubling the voice). A voice whose share is low, as where a
# tab writes the upper of two stacked voices and the lower is heard, still stands out.
MIN_PROMINENCE = 8.0
# Two tracks that fall on the singer's for nearly as long as each other (this share of the longer, or more) can't
# be told apart by the recording: a backing voice doubling the voice fell on it 31.4 s, the voice 32.4 s. Their
# names do it then: Songsterr names them "Lead Vocals" and "Backing Vocals".
CLOSE = 0.85
_LEAD = re.compile(r"\b(lead|main)\W*vocal|\bvocals?\W*\(?(lead|main)\b", re.I)
_BACKING = re.compile(r"\b(backing|back|harmony|choir|additional|doubled?)\W*vocal|"
                      r"\bvocals?\W*\(?(backing|harmony|doubled?)\b|\bbvs?\b", re.I)


class TabError(Exception):
    pass


@dataclass
class Track:
    name: str
    notes: list[tuple[float, float, int]]  # (start, end, in seconds from the tab's start; MIDI note), one at a time


@dataclass
class Placed:
    """A tab's track laid over a recording."""

    track: Track
    notes: list[tuple[float, float, float]]  # (start, end, in the recording; semitones from A440)
    share: float  # of the time its notes sound, how much is on the note the recording's singer sings then
    key: int  # semitones the tab is written under the recording (over: negative): a capo's worth, usually
    close: str | None = None  # the name of another track that fit nearly as well (CLOSE), if one did


@dataclass
class Score:
    """A tab as read, before its notes are timed: each bar's beats, and in which order and how fast they're played."""

    lengths: list[float]  # each bar's, in quarter notes
    tempos: list[tuple[int, float, float]]  # (bar, how far into it (0 to 1), quarter notes a minute)
    order: list[int]  # the bars, in the order they're played (played())
    parts: list[tuple[str, list[list[tuple[float, int | None, bool]]]]]  # each track's name, and its bars' beats:
    # (length in quarter notes, the MIDI note it sounds (of several, the highest; None: silent), tied to the one before)


def tracks(path: Path) -> list[Track]:
    """The tracks of a Guitar Pro file, each as the notes it plays one at a time (of several at once, the highest)."""
    try:
        with zipfile.ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("Content/score.gpif"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError):
        raise TabError("it isn't a Guitar Pro file Woodshed can read (a .gp, from Guitar Pro 7 or later)") from None
    try:
        score = _score(root)
    except (AttributeError, KeyError, ValueError, IndexError, TypeError):
        raise TabError("its score isn't written the way Guitar Pro 7 writes it") from None
    return timed(score)


def _score(root) -> Score:
    by_id = {kind: {e.get("id"): e for e in root.find(kind)} for kind in ("Bars", "Voices", "Beats", "Notes", "Rhythms")}
    masters = list(root.find("MasterBars"))
    if any(m.find("AlternateEndings") is not None or m.find("Directions") is not None for m in masters):
        raise TabError("its score has alternate endings or jumps, which aren't followed")
    repeats = [m.find("Repeat") for m in masters]
    order = played([(r is not None and r.get("start") == "true",
                     int(r.get("count") or 2) if r is not None and r.get("end") == "true" else 0, frozenset())
                    for r in repeats])
    lengths = []
    for master in masters:
        beats, unit = master.findtext("Time").split("/")
        lengths.append(int(beats) * 4 / int(unit))
    tempos = [(int(a.findtext("Bar")), float(a.findtext("Position") or 0), _bpm(a.findtext("Value")))
              for a in root.find("MasterTrack").iter("Automation") if a.findtext("Type") == "Tempo"]
    parts = []
    for number, track in enumerate(root.find("Tracks")):
        bars = []
        for master in masters:
            ids = master.findtext("Bars").split()
            voices = [v for v in by_id["Bars"][ids[number]].findtext("Voices").split() if v != "-1"]
            beats = []
            for beat_id in (by_id["Voices"][voices[0]].findtext("Beats").split() if voices else []):
                beat = by_id["Beats"][beat_id]
                if beat.find("GraceNotes") is not None:  # takes no time of its own
                    continue
                sounding = [by_id["Notes"][n] for n in (beat.findtext("Notes") or "").split()]
                pitched = [(int(p.findtext("Number")), n) for n in sounding for p in n.iter("Property")
                           if p.get("name") == "Midi"]
                pitch, tied = None, False
                if pitched:
                    pitch, note = max(pitched, key=lambda found: found[0])
                    tie = note.find("Tie")
                    tied = tie is not None and tie.get("destination") == "true"
                beats.append((_length(by_id["Rhythms"][beat.find("Rhythm").get("ref")]), pitch, tied))
            bars.append(beats)
        parts.append((track.findtext("Name") or f"Track {number + 1}", bars))
    return Score(lengths, tempos, order, parts)


def timed(score: Score) -> list[Track]:
    """Each track of a tab as the notes it plays, in seconds from its start."""
    tempos = sorted(score.tempos)
    if not tempos:
        raise TabError("its score has no tempo")

    def clock(bar: int, at: float) -> float:
        """Seconds from a bar's start to `at` quarter notes into it."""
        tempo = [t for t in tempos if (t[0], t[1]) <= (bar, 0.0)][-1:] or tempos[:1]
        changes = [(0.0, 60 / tempo[0][2])] + [(p * score.lengths[bar], 60 / bpm) for b, p, bpm in tempos
                                               if b == bar and p > 0]
        total = 0.0
        for k, (start, each) in enumerate(changes):
            end = changes[k + 1][0] if k + 1 < len(changes) else np.inf
            total += max(0.0, min(at, end) - start) * each
        return total

    found = []
    for name, bars in score.parts:
        notes, now, ringing = [], 0.0, None  # ringing: the index of the note a tie would carry on
        for bar in score.order:
            at = 0.0
            for length, pitch, tied in (bars[bar] if bar < len(bars) else []):
                start, end = now + clock(bar, at), now + clock(bar, at + length)
                if pitch is None:
                    ringing = None
                elif tied and ringing is not None and notes[ringing][2] == pitch:
                    notes[ringing] = (notes[ringing][0], end, pitch)
                else:
                    notes.append((start, end, pitch))
                    ringing = len(notes) - 1
                at += length
            now += clock(bar, score.lengths[bar])
        found.append(Track(name, notes))
    return found


def played(bars: list[tuple[bool, int, frozenset[int]]]) -> list[int]:
    """The bars in the order they're played, each given as (whether a repeated section starts at it, the times the
    section that ends at it is played (0: none ends there), the times through its section it's played on, as an
    alternate ending (empty: every time)). A section that ends without a start begins after the one before."""
    order, start, time, bar = [], 0, 1, 0
    while bar < len(bars):
        starts, plays, endings = bars[bar]
        if starts and bar != start:
            start, time = bar, 1
        if endings and time not in endings:
            bar += 1
            continue
        if len(order) > 50 * len(bars):
            raise TabError("its repeats never end")
        order.append(bar)
        if time < plays:
            time, bar = time + 1, start
            continue
        if plays or (endings and not (bar + 1 < len(bars) and bars[bar + 1][2])):  # the section is over
            start, time = bar + 1, 1
        bar += 1
    return order


def _bpm(value: str) -> float:
    """A tempo, in quarter notes a minute: it's written as beats a minute and what a beat is."""
    beats, _, unit = value.partition(" ")
    return float(beats) * _BEAT_UNITS.get(unit.strip() or "2", 1.0)


def _length(rhythm) -> float:
    """A rhythm's length in quarter notes."""
    length = _QUARTERS[rhythm.findtext("NoteValue")]
    dots = rhythm.find("AugmentationDot")
    if dots is not None:
        length *= 2 - 0.5 ** int(dots.get("count") or 1)
    tuplet = rhythm.find("PrimaryTuplet")
    if tuplet is not None:
        length *= int(tuplet.get("den")) / int(tuplet.get("num"))
    return length


def place(found: list[Track], contour: list[float | None], hop: float, named: str | None = None) -> Placed | None:
    """The track that is the voice, laid over the recording whose singer's melody is `contour` (every `hop`
    seconds; along.contour()): in time with it, in its key, and in the octave nearest the singer's. Of the tracks
    whose notes fall on the singer's (MIN_SHARE), clearly more in one place than any other (MIN_PROMINENCE), the
    one that does so for longest: a backing voice of a few notes may fall on them as often, by chance, but not as
    long; one doubling the voice, nearly as long, when their names say which is the lead (CLOSE). `named`: only the
    tracks with that in their name (raises TabError if none has). None if no track fits."""
    if named:
        every, found = found, [track for track in found if named.casefold() in track.name.casefold()]
        if not found:
            raise TabError(f"none of its tracks is named “{named}”; they are "
                           + ", ".join(f"“{track.name}”" for track in every))
    heard = _classes(np.array([np.nan if p is None else p for p in contour], float), hop)
    melodies = [track for track in found if len(track.notes) >= MIN_NOTES]
    if not melodies or not heard.any():
        return None
    longest = max(track.notes[-1][1] for track in melodies) * _STRETCHES[-1]
    size = 1 << int(np.ceil(np.log2(longest / _GRID + 1 + heard.shape[1])))
    heard = np.fft.rfft(heard, size)
    fits = []
    for track in melodies:
        share, shift, key, stretch = max(((*_fit(track.notes, stretch, heard, size), stretch)
                                          for stretch in _STRETCHES), key=lambda fit: fit[0])
        for finer in np.arange(stretch - _STRETCHES[1] + _STRETCHES[0], stretch + _STRETCHES[1] - _STRETCHES[0],
                               0.0002):
            fit = _fit(track.notes, finer, heard, size, [key])
            if fit[0] > share:
                share, shift, stretch = fit[0], fit[1], finer
        if share >= MIN_SHARE and _prominence(track.notes, stretch, heard, size) >= MIN_PROMINENCE:
            sounding = sum(b - a for a, b, _ in track.notes) * stretch
            fits.append((share * sounding, share, shift, key, stretch, track))
    if not fits:
        return None
    fits.sort(key=lambda fit: -fit[0])
    close = [fit for fit in fits if fit[0] >= CLOSE * fits[0][0]]
    _, share, shift, key, stretch, track = min(close, key=lambda fit: _role(fit[5].name))  # the longest, of a role
    others = [fit[5].name for fit in close if fit[5] is not track]
    sung = np.array([p for p in contour if p is not None])
    written = float(np.median([n[2] - 69 + key for n in track.notes]))
    octave = 12 * round((float(np.median(sung)) - written) / 12)
    return Placed(track, [(round(float(a * stretch + shift), 2), round(float(b * stretch + shift), 2),
                           float(p - 69 + key + octave)) for a, b, p in track.notes], float(share),
                  int((key + 6) % 12 - 6), others[0] if others else None)


def _role(name: str) -> int:
    """How likely a track's name says it's the voice: 0 a lead vocal, 1 nothing either way, 2 a backing vocal."""
    return 0 if _LEAD.search(name) else 2 if _BACKING.search(name) else 1


def _classes(pitch: np.ndarray, hop: float) -> np.ndarray:
    """For each of the 12 notes, whether it's the one sung at each moment of the grid."""
    at = np.minimum((np.arange(int(len(pitch) * hop / _GRID)) * _GRID / hop).astype(int), len(pitch) - 1)
    sampled = pitch[at]
    classes = np.zeros((12, len(sampled)), np.float32)
    sung = ~np.isnan(sampled)
    classes[np.round(sampled[sung]).astype(int) % 12, np.flatnonzero(sung)] = 1
    return classes


def _written(notes: list[tuple[float, float, int]], stretch: float, size: int) -> np.ndarray:
    """The Fourier transform of when each of the 12 notes is written, played `stretch` times as slowly (conjugated:
    multiplied by the heard ones', it gives their correlation)."""
    written = np.zeros((12, int(notes[-1][1] * stretch / _GRID) + 1), np.float32)
    for start, end, pitch in notes:
        written[(pitch - 69) % 12, int(start * stretch / _GRID):int(end * stretch / _GRID)] = 1
    return np.conj(np.fft.rfft(written, size))


def _shifts() -> np.ndarray:
    reach = int(MAX_SHIFT_SECONDS / _GRID)
    return np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])


def _prominence(notes: list[tuple[float, float, int]], stretch: float, heard: np.ndarray, size: int) -> float:
    """How far the notes' best fit on the singer's stands above their fits in every key and moved by every shift,
    in standard deviations."""
    spectrum, shifts = _written(notes, stretch, size), _shifts()
    together = np.concatenate([np.fft.irfft((np.roll(heard, -key, axis=0) * spectrum).sum(0), size)[shifts]
                               for key in range(12)])
    spread = float(together.std())
    return float((together.max() - together.mean()) / spread) if spread > 0 else 0.0


def _fit(notes: list[tuple[float, float, int]], stretch: float, heard: np.ndarray, size: int,
         keys=range(12)) -> tuple[float, float, int]:
    """Where in the recording the notes, played `stretch` times as slowly, best fall on the singer's, and moved
    up by how many semitones (of `keys`): (the share of the time they sound that's on the singer's note, the
    seconds to move them by, the semitones). `heard` is the Fourier transform of _classes(), over `size`."""
    spectrum, shifts = _written(notes, stretch, size), _shifts()
    best = (-1.0, 0.0, 0)
    for key in keys:
        # For each shift, how long the written notes and the heard ones are the same: a correlation, note by note.
        together = np.fft.irfft((np.roll(heard, -key, axis=0) * spectrum).sum(0), size)[shifts]
        top = float(together.max())
        if top > best[0]:
            # A singer is silent at the edges of a note (a consonant, a breath), so the notes fall on theirs just
            # as well a little earlier or later: the middle of where they fall best.
            around = np.sort(shifts[together >= top - 0.5])
            around = around[np.abs(around - shifts[int(np.argmax(together))]) <= 0.3 / _GRID]
            best = (top, float(np.median(around) * _GRID), key)
    sounding = sum(int(end * stretch / _GRID) - int(start * stretch / _GRID) for start, end, _ in notes)
    return best[0] / max(float(sounding), 1.0), best[1], best[2]
