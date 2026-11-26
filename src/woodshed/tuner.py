"""Tuning a guitar (`shed tune`, and the Tune tab of `shed serve`'s page): pluck a string, and it says which string
of the tuning it takes it for, and how far from it you are. Nothing is recorded.

- The pitch is found as tuners find it (YIN: the shortest time after which the sound repeats itself), not by the
  tracker that follows your voice (rmvpe.py), which is made for a voice over an instrument, 20 ms at a time. A
  string alone repeats itself cleanly: a quarter of a second of one, made up with a faint fundamental and some
  noise, gave its pitch to within 2 cents, for every string of every tuning here.
- The latest sound is read three times, a quarter of a second each, 75 ms apart, and the middle of what was read
  is kept (heard()): one misreading, as a string is plucked, doesn't make the needle jump.
- The string is the tuning's nearest to what's heard, unless you pick one: a new string, or one being taken to
  another tuning, starts nearer its neighbour.
- Pitch is measured from A440.
"""

import queue
import re
import time
from dataclasses import dataclass

import numpy as np

LOWEST_HZ = 55.0  # A1: 3 semitones under the lowest string here (Drop C's), and over the mains' hum
HIGHEST_HZ = 700.0  # F5: over the top string's harmonic at the 12th fret
WINDOW_SECONDS = 0.25  # of sound for one reading
LISTEN_SECONDS = 0.4  # of the latest sound, for the readings heard() takes the middle of
_READINGS = 3
_CLEAR = 0.15  # how unlike itself the sound may be a period later (0: the same, 1: unrelated), for a pitch to be read…
_UNCLEAR = 0.3  # …or at its best, when nowhere that clear…
_AS_CLEAR = 0.05  # …at the first lag this near its best
_SILENCE = 1e-4  # quieter than this (RMS, -80 dBFS), nothing is plucked
IN_TUNE_CENTS = 5
NEAR_CENTS = 15
_KEEP_SECONDS = 1.0  # a reading is shown this long once the string has died away
_BAR_CENTS = 50  # the bar goes from this many cents flat to this many sharp…
_BAR_CELLS = 41  # …2.5 cents a cell, the string in the middle

_FROM_C = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARPS = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"]


class UnknownTuning(Exception):
    """No tuning, or more than one, goes by that name: `matches` are the ones it could be."""

    def __init__(self, name: str, matches: list[str]):
        super().__init__(name)
        self.name, self.matches = name, matches


def semitones(note: str) -> int:
    """A note written like "E2", "F♯3" or "B♭3", in semitones from A440."""
    letter, accidental, octave = re.fullmatch(r"([A-G])([♯♭#b]?)(-?\d+)", note).groups()
    return 12 * (int(octave) + 1) + _FROM_C[letter] + {"": 0, "♯": 1, "#": 1, "♭": -1, "b": -1}[accidental] - 69


@dataclass(frozen=True)
class Tuning:
    name: str
    strings: tuple[str, ...]  # the notes of the open strings, from the lowest (the 6th string) to the highest

    @property
    def notes(self) -> list[int]:
        """The strings' notes, in semitones from A440."""
        return [semitones(string) for string in self.strings]


TUNINGS = [Tuning(name, tuple(strings.split())) for name, strings in (
    ("Standard", "E2 A2 D3 G3 B3 E4"),
    ("Drop D", "D2 A2 D3 G3 B3 E4"),
    ("Double drop D", "D2 A2 D3 G3 B3 D4"),
    ("DADGAD", "D2 A2 D3 G3 A3 D4"),
    ("Half step down", "E♭2 A♭2 D♭3 G♭3 B♭3 E♭4"),
    ("Whole step down", "D2 G2 C3 F3 A3 D4"),
    ("Drop C", "C2 G2 C3 F3 A3 D4"),
    ("Open D", "D2 A2 D3 F♯3 A3 D4"),
    ("Open E", "E2 B2 E3 G♯3 B3 E4"),
    ("Open G", "D2 G2 D3 G3 B3 D4"),
    ("Open A", "E2 A2 E3 A3 C♯4 E4"),
    ("Open C", "C2 G2 C3 G3 C4 E4"),
)]


def find(name: str) -> Tuning:
    """The tuning that goes by this name, or by this part of one, however it's spaced or capitalized ("dropd",
    "half step"). Raises UnknownTuning if none does, or several do."""
    def plain(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    asked = plain(name)
    exact = [t for t in TUNINGS if plain(t.name) == asked]
    partial = [t for t in TUNINGS if asked and asked in plain(t.name)]
    if len(exact or partial) != 1:
        raise UnknownTuning(name, [t.name for t in partial])
    return (exact or partial)[0]


def pitch(samples: np.ndarray, rate: int) -> float | None:
    """The pitch of a string ringing through these samples, in Hz; None if they're silent, or no single pitch is
    clear in them (a chord, talk, noise)."""
    longest = int(rate / LOWEST_HZ) + 2
    samples = np.asarray(samples, np.float64)
    width = len(samples) - longest  # what's compared with itself, later
    if width < longest:
        return None
    samples = samples - samples.mean()
    if np.sqrt(np.mean(samples**2)) < _SILENCE:
        return None
    # How unlike the sound is itself, `lag` samples later: the sum of (x[j] - x[j + lag])² over the first `width`.
    size = 1 << (len(samples) + width - 1).bit_length()
    products = np.fft.irfft(np.fft.rfft(samples, size) * np.conj(np.fft.rfft(samples[:width], size)), size)[:longest + 1]
    energy = np.concatenate([[0.0], np.cumsum(samples**2)])
    loudness = energy[width] + energy[width:width + longest + 1] - energy[:longest + 1]  # of the two compared
    unlike = loudness - 2 * products
    even = unlike / np.maximum(loudness, 1e-12)  # …for its loudness: a string dying away is still like itself
    # …against what it has been on average up to that lag, so that 0 isn't the best lag of all.
    unlike[1:] *= np.arange(1, longest + 1) / np.maximum(np.cumsum(unlike[1:]), 1e-12)
    unlike[0] = 1.0
    shortest = max(2, int(rate / HIGHEST_HZ))
    # The first lag at which it's clearly like itself: at twice or three times the period it's as like itself again,
    # and with noise over a faint string, a little more so at one of those than at the period itself.
    lowest = float(unlike[shortest:longest].min())
    if lowest > _UNCLEAR:
        return None
    bar = max(_CLEAR, lowest + _AS_CLEAR)
    lag = end = shortest + int(np.flatnonzero(unlike[shortest:longest] < bar)[0])
    while end < longest and unlike[end] < bar:
        end += 1
    lag += int(np.argmin(unlike[lag:end]))  # the bottom of that dip
    if lag <= shortest or lag >= longest - 1:  # still going down at an end: its pitch is out of reach
        return None
    # The bottom lies between two samples: it's that of the curve through the samples around it, as many as noise
    # needs for a low string, whose dip is hundreds of samples across. Taken on the sound evened for its loudness:
    # against its average so far, the dip leans to the short side, and a string read 2¢ sharp.
    reach = max(1, lag // 16)
    around = np.arange(max(lag - reach, 1), min(lag + reach, longest) + 1)
    bend, slope, _ = np.polyfit(around - lag, even[around], 2)
    return rate / (lag + (float(np.clip(-slope / (2 * bend), -reach, reach)) if bend > 0 else 0.0))


def heard(samples: np.ndarray, rate: int) -> float | None:
    """The string ringing at the end of these samples (their last LISTEN_SECONDS), in semitones from A440: the
    middle of three readings; None unless two of them found a pitch."""
    window, last = int(WINDOW_SECONDS * rate), int(LISTEN_SECONDS * rate)
    samples = samples[-last:]
    step = (last - window) // (_READINGS - 1)
    ends = [len(samples) - i * step for i in range(_READINGS)]
    read = [hz for end in ends if end >= window and (hz := pitch(samples[end - window:end], rate))]
    if len(read) < 2:
        return None
    return float(12 * np.log2(np.median(read) / 440))


@dataclass
class Reading:
    string: int  # which of the tuning's strings, from 0: the lowest
    cents: float  # how far over its note you are (sharp); under it (flat) if negative


def read(tuning: Tuning, sounding: float, string: int | None = None) -> Reading:
    """What's heard (semitones from A440) against the tuning's nearest string, or the one given."""
    notes = tuning.notes
    if string is None:
        string = min(range(len(notes)), key=lambda i: abs(sounding - notes[i]))
    return Reading(string, 100 * (sounding - notes[string]))


def advice(cents: float) -> str:
    if abs(cents) <= IN_TUNE_CENTS:
        return "in tune ✓"
    way, side = ("tune up ↑", "flat") if cents < 0 else ("tune down ↓", "sharp")
    if abs(cents) < 100:
        return f"{way} ({abs(cents):.0f}¢ {side})"
    away = round(abs(cents) / 100)
    return f"{way} ({'a semitone' if away == 1 else f'{away} semitones'} {side})"


def note_name(sounding: float) -> str:
    """The name of the nearest note, with its octave: 0 is A4."""
    midi = round(sounding) + 69
    return f"{_SHARPS[(midi - 69) % 12]}{midi // 12 - 1}"


def state(tuning: Tuning, sounding: float | None, string: int | None = None) -> dict | None:
    """What the phone shows of what's heard: the string, how far from it, and which way to go."""
    if sounding is None:
        return None
    reading = read(tuning, sounding, string)
    return {"string": reading.string, "cents": round(reading.cents, 1), "advice": advice(reading.cents),
            "heard": note_name(sounding)}


def _color(cents: float) -> str:
    return "green" if abs(cents) <= IN_TUNE_CENTS else "yellow" if abs(cents) <= NEAR_CENTS else "red"


def rows(tuning: Tuning, reading: Reading | None, only: int | None = None, fresh: bool = True,
         microphone: str | None = None, others: bool = False) -> list:
    """What the terminal shows: the tuning's strings, the one you're on marked, and how far from it you are on a
    bar from 50¢ flat to 50¢ sharp. `only`: the string you picked; `fresh`: the string is still ringing;
    `microphone`: the one listened to, and whether there are `others` to go to."""
    from rich.text import Text

    shown = [Text.assemble(("Tuner  ", "bold"), (tuning.name, "bold cyan")), Text()]
    strings = Text("  ")
    for i, string in enumerate(tuning.strings):
        on = reading is not None and reading.string == i
        style = f"bold {_color(reading.cents)}" if on else "bold" if only == i else "dim"
        strings.append(f"{len(tuning.strings) - i} ", "dim")  # numbered as guitarists do: the 6th is the lowest
        strings.append(string, f"{style} underline" if only == i else style)
        strings.append("   ")
    shown += [strings, Text()]
    cells = ["─"] * _BAR_CELLS
    cells[_BAR_CELLS // 2] = "┼"
    if reading is None:
        shown += [Text.assemble("  ♭ ", ("".join(cells), "grey50"), " ♯"),
                  Text(f"  Pluck {'the ' + tuning.strings[only] + ' string' if only is not None else 'a string'}…", "dim")]
    else:
        color = _color(reading.cents) if fresh else "grey50"
        at = round(np.clip(reading.cents, -_BAR_CENTS, _BAR_CENTS) / _BAR_CENTS * (_BAR_CELLS // 2)) + _BAR_CELLS // 2
        cells[at] = "●"
        shown += [Text.assemble("  ♭ ", ("".join(cells), color), " ♯"),
                  Text.assemble("  ", (tuning.strings[reading.string], "bold"), "  ", (advice(reading.cents), color))]
    shown.append(Text())
    if microphone:
        shown.append(Text.assemble(("  Listening to ", "dim"), microphone, ("  (m: another microphone)" if others else "",
                                                                             "dim")))
    return shown + [Text(f"←/→ another tuning · 1-{len(tuning.strings)} only that string · 0 any string · q quits",
                         "dim")]


def run(console, tuning: Tuning, device: int | str | None = None) -> None:
    """Tune until q, Escape or Ctrl+C: ←/→ go to another tuning, a string's number keeps to that string (6: the
    lowest), 0 goes back to whichever is nearest, and m goes to the next microphone (one that can't be listened
    to is passed over)."""
    import sounddevice as sd
    from rich.console import Group
    from rich.live import Live

    from woodshed import terminal

    microphone = sd.query_devices(device, "input")
    microphones = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    latest = np.zeros(0, np.float32)
    at, only, reading, read_at = TUNINGS.index(tuning), None, None, 0.0

    def listen(to: dict):
        stream = sd.InputStream(device=to["index"], channels=1, samplerate=int(to["default_samplerate"]),
                                dtype="float32", callback=lambda indata, *_: blocks.put(indata[:, 0].copy()))
        stream.start()
        return stream

    def render(now: float) -> Group:
        return Group(*rows(TUNINGS[at], reading, only, now - read_at < 0.3, microphone["name"], len(microphones) > 1))

    stream = listen(microphone)
    try:
        with terminal.keys() as next_key, Live(render(0.0), console=console, transient=True,
                                               auto_refresh=False) as live:
            try:
                while True:
                    key = next_key(timeout=0.05)
                    if key in ("q", "escape"):
                        break
                    if key in ("left", "right"):
                        at, only, reading = (at + (1 if key == "right" else -1)) % len(TUNINGS), None, None
                    elif key and key.isdigit() and int(key) <= len(TUNINGS[at].strings):
                        only, reading = (len(TUNINGS[at].strings) - int(key) if key != "0" else None), None
                    elif key == "m" and len(microphones) > 1:
                        stream.close()
                        stream = None
                        while not blocks.empty():  # what it heard isn't the next one's
                            blocks.get_nowait()
                        latest, reading = latest[:0], None
                        here = next((i for i, d in enumerate(microphones) if d["index"] == microphone["index"]), -1)
                        for step in range(1, len(microphones) + 1):  # the next that works: this one again, at worst
                            microphone = microphones[(here + step) % len(microphones)]
                            try:
                                stream = listen(microphone)
                                break
                            except sd.PortAudioError:
                                if step == len(microphones):
                                    raise
                    rate = int(microphone["default_samplerate"])
                    while not blocks.empty():
                        latest = np.concatenate([latest, blocks.get_nowait()])[-int(LISTEN_SECONDS * rate):]
                    now = time.monotonic()
                    sounding = heard(latest, rate) if len(latest) >= int(LISTEN_SECONDS * rate) else None
                    if sounding is not None:
                        reading, read_at = read(TUNINGS[at], sounding, only), now
                    elif now - read_at > _KEEP_SECONDS:
                        reading = None
                    live.update(render(now), refresh=True)
            except KeyboardInterrupt:
                pass
    finally:
        if stream:
            stream.close()
