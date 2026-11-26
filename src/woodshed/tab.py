"""A song's melody as written in a tab (a Guitar Pro file, as Songsterr saves them), to sing along to (along.py)
in place of the melody heard in the reference recording: a singer slides into notes, bends them and adds some, so
what's heard wanders; the tab says which notes the song is made of.

- The file is a zip holding the score as XML (the .gp of Guitar Pro 7 and later). The older .gp3, .gp4, .gp5 and
  .gpx are other formats, not read here, nor is MIDI: it would say no more (notes and when), and a tab site gives
  the .gp.
- A tab doesn't say which of its tracks is the voice (Songsterr names it after the singer, and gives it any
  instrument), nor when the recording starts, nor always its key (a song played with a capo is often written as
  fingered, voice and all) or its exact tempo. All are found by laying each track over the melody heard in the
  recording (along.contour()), slid in time, up to 5% faster or slower, in each of the 12 keys: the track and
  the place where the most of its notes fall on the same note as the singer's, whatever the octave (place()).
- Repeats are played out as written (a section between repeat signs, the times it says); a score with alternate
  endings or jumps (D.S., coda…) isn't read: where it goes next isn't followed here.
"""

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
MIN_SHARE = 0.35  # of a track's notes on the singer's, for it to be the voice (other tracks: under 0.2)


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


def tracks(path: Path) -> list[Track]:
    """The tracks of a Guitar Pro file, each as the notes it plays one at a time (of several at once, the highest)."""
    try:
        with zipfile.ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("Content/score.gpif"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError):
        raise TabError("it isn't a Guitar Pro file Woodshed can read (a .gp, from Guitar Pro 7 or later)") from None
    try:
        return _tracks(root)
    except (AttributeError, KeyError, ValueError, IndexError, TypeError):
        raise TabError("its score isn't written the way Guitar Pro 7 writes it") from None


def _tracks(root) -> list[Track]:
    by_id = {kind: {e.get("id"): e for e in root.find(kind)} for kind in ("Bars", "Voices", "Beats", "Notes", "Rhythms")}
    masters = list(root.find("MasterBars"))
    if any(m.find("AlternateEndings") is not None or m.find("Directions") is not None for m in masters):
        raise TabError("its score has alternate endings or jumps, which aren't followed")
    order = _played(masters)
    lengths = []  # each master bar's, in quarter notes
    for master in masters:
        beats, unit = master.findtext("Time").split("/")
        lengths.append(int(beats) * 4 / int(unit))
    tempos = sorted((int(a.findtext("Bar")), float(a.findtext("Position") or 0), _bpm(a.findtext("Value")))
                    for a in root.find("MasterTrack").iter("Automation") if a.findtext("Type") == "Tempo")
    if not tempos:
        raise TabError("its score has no tempo")

    def seconds_of(bar: int) -> list[tuple[float, float, float]]:
        """The stretches of a bar at one tempo: (from, in quarter notes into the bar; seconds a quarter note)."""
        tempo = [t for t in tempos if (t[0], t[1]) <= (bar, 0.0)][-1:] or tempos[:1]
        changes = [(0.0, 60 / tempo[0][2])] + [(p * lengths[bar], 60 / bpm) for b, p, bpm in tempos if b == bar and p > 0]
        return changes

    def clock(bar: int, at: float) -> float:
        """Seconds from a bar's start to `at` quarter notes into it."""
        changes, total = seconds_of(bar), 0.0
        for k, (start, each) in enumerate(changes):
            end = changes[k + 1][0] if k + 1 < len(changes) else np.inf
            total += max(0.0, min(at, end) - start) * each
        return total

    found = []
    for number, track in enumerate(root.find("Tracks")):
        notes, now, ringing = [], 0.0, None  # ringing: the index of the note a tie would carry on
        for bar in order:
            ids = masters[bar].findtext("Bars").split()
            voices = [v for v in by_id["Bars"][ids[number]].findtext("Voices").split() if v != "-1"]
            at = 0.0
            for beat_id in (by_id["Voices"][voices[0]].findtext("Beats").split() if voices else []):
                beat = by_id["Beats"][beat_id]
                if beat.find("GraceNotes") is not None:  # takes no time of its own
                    continue
                length = _length(by_id["Rhythms"][beat.find("Rhythm").get("ref")])
                start, end = now + clock(bar, at), now + clock(bar, at + length)
                sounding = [by_id["Notes"][n] for n in (beat.findtext("Notes") or "").split()]
                pitched = [(int(p.findtext("Number")), n) for n in sounding for p in n.iter("Property")
                           if p.get("name") == "Midi"]
                if pitched:
                    pitch, note = max(pitched, key=lambda found: found[0])
                    tie = note.find("Tie")
                    if (tie is not None and tie.get("destination") == "true" and ringing is not None
                            and notes[ringing][2] == pitch):
                        notes[ringing] = (notes[ringing][0], end, pitch)
                    else:
                        notes.append((start, end, pitch))
                        ringing = len(notes) - 1
                else:
                    ringing = None
                at += length
            now += clock(bar, lengths[bar])
        found.append(Track(track.findtext("Name") or f"Track {number + 1}", notes))
    return found


def _played(masters: list) -> list[int]:
    """The bars in the order they're played: a section between repeat signs, the times it says."""
    order, start, bar, done = [], 0, 0, {}
    while bar < len(masters):
        repeat = masters[bar].find("Repeat")
        if repeat is not None and repeat.get("start") == "true":
            start = bar
        order.append(bar)
        if repeat is not None and repeat.get("end") == "true" and done.get(bar, 1) < int(repeat.get("count") or 2):
            done[bar] = done.get(bar, 1) + 1
            bar = start
        else:
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


def place(found: list[Track], contour: list[float | None], hop: float) -> Placed | None:
    """The track that is the voice, laid over the recording whose singer's melody is `contour` (every `hop`
    seconds; along.contour()): in time with it, in its key, and in the octave nearest the singer's. None if no
    track's notes fall on the singer's (MIN_SHARE)."""
    heard = _classes(np.array([np.nan if p is None else p for p in contour], float), hop)
    melodies = [track for track in found if len(track.notes) >= MIN_NOTES]
    if not melodies or not heard.any():
        return None
    longest = max(track.notes[-1][1] for track in melodies) * _STRETCHES[-1]
    size = 1 << int(np.ceil(np.log2(longest / _GRID + 1 + heard.shape[1])))
    heard = np.fft.rfft(heard, size)
    share, shift, key, stretch, track = max(((*_fit(track.notes, stretch, heard, size), stretch, track)
                                             for track in melodies for stretch in _STRETCHES), key=lambda fit: fit[0])
    for finer in np.arange(stretch - _STRETCHES[1] + _STRETCHES[0], stretch + _STRETCHES[1] - _STRETCHES[0], 0.0002):
        fit = _fit(track.notes, finer, heard, size, [key])
        if fit[0] > share:
            share, shift, stretch = fit[0], fit[1], finer
    if share < MIN_SHARE:
        return None
    sung = np.array([p for p in contour if p is not None])
    written = float(np.median([n[2] - 69 + key for n in track.notes]))
    octave = 12 * round((float(np.median(sung)) - written) / 12)
    return Placed(track, [(round(float(a * stretch + shift), 2), round(float(b * stretch + shift), 2),
                           float(p - 69 + key + octave)) for a, b, p in track.notes], float(share),
                  int((key + 6) % 12 - 6))


def _classes(pitch: np.ndarray, hop: float) -> np.ndarray:
    """For each of the 12 notes, whether it's the one sung at each moment of the grid."""
    at = np.minimum((np.arange(int(len(pitch) * hop / _GRID)) * _GRID / hop).astype(int), len(pitch) - 1)
    sampled = pitch[at]
    classes = np.zeros((12, len(sampled)), np.float32)
    sung = ~np.isnan(sampled)
    classes[np.round(sampled[sung]).astype(int) % 12, np.flatnonzero(sung)] = 1
    return classes


def _fit(notes: list[tuple[float, float, int]], stretch: float, heard: np.ndarray, size: int,
         keys=range(12)) -> tuple[float, float, int]:
    """Where in the recording the notes, played `stretch` times as slowly, best fall on the singer's, and moved
    up by how many semitones (of `keys`): (the share of the time they sound that's on the singer's note, the
    seconds to move them by, the semitones). `heard` is the Fourier transform of _classes(), over `size`."""
    written = np.zeros((12, int(notes[-1][1] * stretch / _GRID) + 1), np.float32)
    for start, end, pitch in notes:
        written[(pitch - 69) % 12, int(start * stretch / _GRID):int(end * stretch / _GRID)] = 1
    spectrum = np.conj(np.fft.rfft(written, size))
    reach = int(MAX_SHIFT_SECONDS / _GRID)
    shifts = np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])
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
    return best[0] / max(float(written.sum()), 1.0), best[1], best[2]
