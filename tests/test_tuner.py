"""Tuning a guitar: `shed tune`. The microphone is left out: strings are made up (pluck()), and the command is
checked up to where it would start listening."""

import contextlib
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from woodshed import cli, terminal, tuner


def pluck(note: float, rate: int = 48_000, seconds: float = 0.6, fundamental: float = 1.0, noise: float = 0.0,
          seed: int = 0) -> np.ndarray:
    """A plucked string on this note (semitones from A440): harmonics dying away, the high ones faster, each a
    little sharp of a whole multiple, as a stiff string's are."""
    t, hz = np.arange(int(seconds * rate)) / rate, 440 * 2 ** (note / 12)
    wave = sum((fundamental if k == 1 else 1 / k) * np.exp(-t * (1 + 0.5 * k))
               * np.sin(2 * np.pi * k * hz * np.sqrt(1 + 5e-5 * k * k) * t + k) for k in range(1, 13) if k * hz < rate / 2)
    return (0.1 * wave + noise * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


def test_notes_are_counted_from_a440():
    assert [tuner.semitones(note) for note in ("A4", "E2", "C4", "F♯3", "B♭3", "Eb2")] == [0, -29, -9, -15, -11, -30]
    assert tuner.TUNINGS[0].name == "Standard" and tuner.TUNINGS[0].notes == [-29, -24, -19, -14, -10, -5]


def test_every_tuning_has_six_strings_going_up():
    assert len({t.name for t in tuner.TUNINGS}) == len(tuner.TUNINGS)
    for tuning in tuner.TUNINGS:
        assert len(tuning.notes) == 6 and tuning.notes == sorted(tuning.notes), tuning.name


@pytest.mark.parametrize("asked, name", [("standard", "Standard"), ("drop d", "Drop D"), ("DropD", "Drop D"),
                                         ("double", "Double drop D"), ("dadgad", "DADGAD"), ("half-step", "Half step down"),
                                         ("open g", "Open G")])
def test_a_tuning_is_found_by_part_of_its_name(asked, name):
    assert tuner.find(asked).name == name


def test_a_name_that_could_be_several_tunings_or_none_says_which():
    with pytest.raises(tuner.UnknownTuning) as unknown:
        tuner.find("drop")
    assert unknown.value.matches == ["Drop D", "Double drop D", "Drop C"]
    with pytest.raises(tuner.UnknownTuning) as unknown:
        tuner.find("banjo")
    assert unknown.value.matches == []


@pytest.mark.parametrize("rate", [44_100, 48_000])
@pytest.mark.parametrize("tuning", tuner.TUNINGS, ids=lambda t: t.name)
def test_every_string_is_read_to_a_few_cents(tuning, rate):
    for note in tuning.notes:
        for off in (-0.4, 0.03):
            assert tuner.heard(pluck(note + off, rate), rate) == pytest.approx(note + off, abs=0.03)


def test_a_string_is_read_through_a_small_microphone_and_a_rooms_noise():
    # A phone's or a laptop's microphone lets little through under 100 Hz: the low strings come with next to none
    # of their fundamental, and mustn't be read an octave up; nor, with noise over them, an octave down.
    for note in tuner.TUNINGS[0].notes + [-33]:  # and Drop C's lowest
        for seed in range(3):
            read = tuner.heard(pluck(note, fundamental=0.05, noise=0.01, seed=seed), 48_000)
            assert read == pytest.approx(note, abs=0.04)


def bright(note: float, odd: float = 1.0, rate: int = 48_000, seconds: float = 0.6) -> np.ndarray:
    """A brighter, stiffer string than pluck()'s, its odd partials (the fundamental too) `odd` times as loud as
    they'd be: faint, the string is nearly the same half a period later, as the string an octave up would be."""
    t, hz = np.arange(int(seconds * rate)) / rate, 440 * 2 ** (note / 12)
    wave = sum((odd if k % 2 else 1.0) / k * np.exp(-t * (0.8 + 0.25 * k))
               * np.sin(2 * np.pi * k * hz * np.sqrt(1 + 1e-4 * k * k) * t + k * k) for k in range(1, 30) if k * hz < 8000)
    return (0.1 * wave + 0.003 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)


def test_a_string_with_faint_odd_partials_isnt_read_an_octave_up():
    for note in tuner.TUNINGS[0].notes:
        assert tuner.heard(bright(note, odd=0.15), 48_000) == pytest.approx(note, abs=0.1)


def test_a_string_is_read_while_the_one_plucked_before_still_rings():
    notes = tuner.TUNINGS[0].notes
    for note, before in zip(notes[1:], notes):  # going up the strings, the one below a quarter as loud
        assert tuner.heard(bright(note) + 0.25 * bright(before), 48_000) == pytest.approx(note, abs=0.1)
    for note, before in zip(notes, notes[1:]):  # and going down them
        assert tuner.heard(bright(note) + 0.25 * bright(before), 48_000) == pytest.approx(note, abs=0.1)


def test_silence_noise_and_a_strummed_chord_arent_read_as_a_string():
    rate = 48_000
    assert tuner.heard(np.zeros(rate, np.float32), rate) is None
    assert tuner.heard(0.05 * np.random.default_rng(1).standard_normal(rate).astype(np.float32), rate) is None
    assert tuner.heard(sum(pluck(note) for note in tuner.TUNINGS[0].notes), rate) is None
    assert tuner.heard(pluck(-29)[:4_000], rate) is None  # too little heard yet


def test_one_misreading_doesnt_move_the_needle(monkeypatch):
    read = iter([110.0, 220.0, 110.0, None, None, 110.0])
    monkeypatch.setattr(tuner, "pitch", lambda samples, rate: next(read))
    assert tuner.heard(np.zeros(48_000, np.float32), 48_000) == pytest.approx(-24)  # A2, not the A3 read once
    assert tuner.heard(np.zeros(48_000, np.float32), 48_000) is None  # read once in three: not enough


def test_whats_heard_is_set_against_the_nearest_string_or_the_one_picked():
    standard, drop_d = tuner.find("standard"), tuner.find("drop d")
    reading = tuner.read(standard, -24.12)
    assert reading.string == 1 and reading.cents == pytest.approx(-12)
    assert tuner.read(standard, -30.9).string == 0  # a low E on its way down to D: still the 6th string
    assert tuner.read(drop_d, -30.9) == tuner.Reading(0, pytest.approx(10))
    picked = tuner.read(standard, -21.0, string=1)  # nearer the D string, but it's the A string being wound up
    assert picked.string == 1 and picked.cents == pytest.approx(300)


def test_the_tuner_says_which_way_to_turn():
    assert tuner.advice(3) == tuner.advice(-5) == "in tune ✓"
    assert tuner.advice(-12.4) == "tune up ↑ (12¢ flat)"
    assert tuner.advice(40) == "tune down ↓ (40¢ sharp)"
    assert tuner.advice(-110) == "tune up ↑ (a semitone flat)"
    assert tuner.advice(300) == "tune down ↓ (3 semitones sharp)"


def plain(rows) -> list[str]:
    return [row.plain for row in rows]


def test_the_terminal_shows_the_strings_and_where_you_are_on_a_bar():
    drop_d = tuner.find("drop d")
    shown = plain(tuner.rows(drop_d, tuner.Reading(1, -25)))
    assert shown[0] == "Tuner  Drop D"
    assert shown[2].split() == ["6", "D2", "5", "A2", "4", "D3", "3", "G3", "2", "B3", "1", "E4"]
    bar = shown[4].strip("♭♯ ")
    assert len(bar) == 41 and bar.index("●") == 10 and bar[20] == "┼"  # halfway to 50¢ flat
    assert shown[5].split(maxsplit=1) == ["A2", "tune up ↑ (25¢ flat)"]
    assert plain(tuner.rows(drop_d, tuner.Reading(0, 400)))[4].strip("♭♯ ").index("●") == 40  # off the bar's end

    waiting = tuner.rows(drop_d, None)
    assert "●" not in waiting[4].plain and waiting[5].plain.strip() == "Pluck a string…"
    assert plain(tuner.rows(drop_d, None, only=0))[5].strip() == "Pluck the D2 string…"


def test_the_terminal_says_which_microphone_it_listens_to():
    shown = plain(tuner.rows(tuner.TUNINGS[0], None, microphone="Scarlett Solo", others=True))
    assert shown[-2].strip() == "Listening to Scarlett Solo  (m: another microphone)"
    assert plain(tuner.rows(tuner.TUNINGS[0], None, microphone="Scarlett Solo"))[-2].strip() == "Listening to Scarlett Solo"


class Microphones:
    """Stands in for sounddevice: three microphones and a speaker; `broken` can't be listened to. Each hears an A
    string as soon as it's listened to."""

    class PortAudioError(Exception):
        pass

    def __init__(self, broken=()):
        self.devices = [{"index": i, "name": name, "max_input_channels": inputs, "default_samplerate": 48_000.0}
                        for i, (name, inputs) in enumerate([("Built-in", 1), ("Speakers", 0), ("Scarlett", 2), ("AirPods", 1)])]
        self.broken, self.listened, self.closed = broken, [], []

    def query_devices(self, device=None, kind=None):
        return self.devices if kind is None else self.devices[device or 0]

    def InputStream(self, device, callback, **_):
        if device in self.broken:
            raise self.PortAudioError("unplugged")
        self.listened.append(device)
        return SimpleNamespace(start=lambda: callback(pluck(-24)[:, None], 0, None, None),
                               close=lambda: self.closed.append(device))


def run_with(monkeypatch, microphones, keys, device=None):
    """Runs the tuner with these keys pressed, one each time round; what it showed each time."""
    shown, pressed = [], iter(keys)
    monkeypatch.setitem(sys.modules, "sounddevice", microphones)
    monkeypatch.setattr(terminal, "keys", lambda: contextlib.nullcontext(lambda timeout=None: next(pressed)))
    rows = tuner.rows
    monkeypatch.setattr(tuner, "rows", lambda *args: shown.append(plain(rows(*args))) or rows(*args))
    tuner.run(Console(file=io.StringIO()), tuner.TUNINGS[0], device)
    return shown


def test_m_goes_to_the_next_microphone_and_round(monkeypatch):
    microphones = Microphones()
    shown = run_with(monkeypatch, microphones, [None, "m", "m", "m", "q"])
    assert microphones.listened == [0, 2, 3, 0] and microphones.closed == [0, 2, 3, 0]  # never the speakers
    assert ["Listening to Built-in" in rows[-2] for rows in shown] == [True, True, False, False, True]
    assert "Listening to Scarlett  (m: another microphone)" in shown[2][-2]
    assert "A2  in tune ✓" in shown[1][5] and "A2  in tune ✓" in shown[2][5]  # each is heard once listened to


def test_a_microphone_that_cant_be_listened_to_is_passed_over(monkeypatch):
    microphones = Microphones(broken=(2,))
    shown = run_with(monkeypatch, microphones, ["m", "q"], device=0)
    assert microphones.listened == [0, 3] and "Listening to AirPods" in shown[-1][-2]


def test_the_string_you_are_on_is_colored_by_how_far_off_it_is():
    def color(cents):
        strings = tuner.rows(tuner.TUNINGS[0], tuner.Reading(2, cents))[2]
        return next(str(span.style) for span in strings.spans if strings.plain[span.start:span.end] == "D3")

    assert [color(cents) for cents in (-4, 12, -30)] == ["bold green", "bold yellow", "bold red"]


def test_the_phone_is_told_the_string_and_which_way_to_go():
    assert tuner.state(tuner.find("standard"), None) is None
    assert tuner.state(tuner.find("standard"), -24.12) == {"string": 1, "cents": -12.0, "heard": "A2",
                                                           "advice": "tune up ↑ (12¢ flat)"}
    assert tuner.state(tuner.find("standard"), -21.0, 1)["heard"] == "C3"


def tuned(monkeypatch, *args, interactive=True):
    """Runs shed tune up to where it would start listening: the tuning and the microphone it would use."""
    run = []
    monkeypatch.setattr(cli, "_interactive", lambda: interactive)
    monkeypatch.setattr(tuner, "run", lambda console, tuning, device: run.append((tuning.name, device)))
    return CliRunner().invoke(cli.app, ["tune", *args]), run


def test_shed_tune_starts_in_standard_or_the_tuning_named(monkeypatch):
    assert tuned(monkeypatch)[1] == [("Standard", None)]
    assert tuned(monkeypatch, "drop", "d", "--device", "3")[1] == [("Drop D", 3)]
    assert tuned(monkeypatch, "open g", "--device", "Scarlett")[1] == [("Open G", "Scarlett")]


def test_shed_tune_lists_the_tunings_a_name_could_be(monkeypatch):
    result, run = tuned(monkeypatch, "drop")
    assert result.exit_code == 1 and not run
    assert "“drop” could be these tunings:" in result.output
    assert "Drop D  D2 A2 D3 G3 B3 E4" in result.output and "Drop C" in result.output and "Open G" not in result.output
    result, run = tuned(monkeypatch, "banjo")
    assert result.exit_code == 1 and "is none of these tunings" in result.output and "Open G" in result.output


def test_shed_tune_needs_a_terminal(monkeypatch):
    result, run = tuned(monkeypatch, interactive=False)
    assert result.exit_code == 1 and not run and "run it in one" in result.output
