"""Finding the note each line starts on: `shed drill`. The microphone and speakers are left out: what's played
and heard is checked through drill.py's parts, and the command up to where it would start listening."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import add_take
from test_melody import LINES, TUNE, sing
from typer.testing import CliRunner

from woodshed import cli, drill, listen, rmvpe
from woodshed.library import Library
from woodshed.lyrics import words
from woodshed.rating import Metrics


def test_each_line_starts_on_its_first_note():
    starts = drill.starts(sing())
    assert [s.line for s in starts] == LINES
    assert [s.word for s in starts] == [words(line)[0] for line in LINES]
    assert [s.note for s in starts] == [line[0] for line in TUNE]


def test_the_note_moves_to_your_key_and_your_register():
    first = TUNE[0][0]
    assert drill.starts(sing(), shift=3)[0].note == first + 3
    # Sung an octave or two away, the note is played where your voice sits.
    assert drill.starts(sing(), register=first - 12)[0].note == first - 12
    assert drill.starts(sing(), shift=3, register=first + 3 + 25)[0].note == first + 3 + 24
    assert drill.register([sing(voice_shift=-12)]) == pytest.approx(np.median([n for line in TUNE for n in line]) - 12)
    assert drill.register([]) is None


def test_notes_are_named_with_their_octave():
    assert [drill.note_name(n) for n in (0, 3, -9, 12, -21.4)] == ["A4", "C5", "C4", "A5", "C3"]


def test_the_tuner_says_which_way_to_go_octaves_ignored():
    assert drill.offset(-12.2, 0) == pytest.approx(-0.2)  # an octave down, a little under
    assert drill.offset(2, 0) == 2 and drill.offset(-3, 0) == -3
    assert drill.advice(-0.2) == "on it"
    assert drill.advice(-0.7) == "a little higher ↑ (70¢ under)"
    assert drill.advice(3) == "lower ↓ (3 semitones over)"
    assert drill.advice(-1.1) == "higher ↑ (a semitone under)"


def test_the_note_is_found_once_held_for_a_second():
    hold = drill.Hold()
    assert not any(hold.update(0.1, 0.25) for _ in range(3))
    assert not hold.update(None, 0.5)  # a breath pauses the count
    assert hold.update(-0.2, 0.25)
    hold = drill.Hold()
    for _ in range(3):
        hold.update(0.1, 0.25)
    assert not hold.update(1.0, 0.25)  # left the note: start again
    assert hold.held == 0


def test_the_note_played_is_the_one_asked_for():
    rate = 44_100
    sound = drill.tone(3, rate)  # C5
    spectrum = np.abs(np.fft.rfft(sound))
    assert np.fft.rfftfreq(len(sound), 1 / rate)[np.argmax(spectrum)] == pytest.approx(523.25, abs=1)
    assert np.max(np.abs(sound)) <= 1


def test_what_you_sing_is_read_a_little_behind_you():
    hop = 0.01
    f0 = np.r_[np.full(50, 220.0), np.full(25, 261.63), np.full(20, 330.0)]  # A3, C4, then E4 for the last 0.2 s
    assert drill.reading(f0, np.ones(95), hop) == pytest.approx(-9, abs=0.01)  # C4
    assert drill.reading(f0, np.r_[np.ones(60), np.zeros(35)], hop) is None  # mostly breath


def test_what_you_sing_is_heard_however_quietly():
    loud = np.sin(np.linspace(0, 200, 16_000)).astype(np.float32)
    quiet = drill.level(loud * 0.002)
    assert np.sqrt(np.mean(quiet**2)) == pytest.approx(np.sqrt(np.mean(drill.level(loud) ** 2)), rel=1e-3)
    assert np.max(np.abs(drill.level(np.zeros(100, np.float32)))) == 0  # silence stays silent


@pytest.fixture
def library(tmp_path, tone):
    songs = Library(tmp_path / "lib")
    add_take(songs, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), LINES, metrics=Metrics(20, 0.02),
             melody=sing(shift=5, voice_shift=-7))  # a capo on the 5th fret, sung an octave under
    add_take(songs, tone, "Winter Town", 2, 7, datetime(2026, 6, 1, 20, 0), [])
    return songs


def drilled(library, monkeypatch, *args):
    """Runs shed drill up to where it would start listening: what it would drill, and could switch to."""
    run = []
    monkeypatch.setattr(cli.models, "missing", lambda: [])
    monkeypatch.setattr(drill, "run", lambda console, prepare, title, titles, device, language:
                        run.append((prepare(title), titles, prepare)))
    result = CliRunner().invoke(cli.app, ["drill", *args, "--library", str(library.root)])
    return result, run


def test_drills_in_the_key_and_register_of_your_latest_take(library, monkeypatch):
    library.set_reference("Harbor Lights", sing(), "original.mp3")
    result, run = drilled(library, monkeypatch, "harbor")
    assert result.exit_code == 0, result.output
    (song, titles, _), = run
    assert song.title == "Harbor Lights" and titles == ["Harbor Lights"]
    assert song.transposed == ("Your instrument plays 5 semitones over the reference, so the melody was moved to "
                               "match.")
    assert song.shift == 5 and song.starts[0].note == TUNE[0][0] + 5 - 12

    result, run = drilled(library, monkeypatch, "harbor", "--transpose", "-2")
    song = run[0][0]
    assert song.starts[0].note % 12 == (TUNE[0][0] - 2) % 12
    assert song.transposed == "Your instrument plays 2 semitones under the reference, so the melody was moved to match."


def test_a_song_asked_for_as_you_drill_is_in_the_key_you_last_played_it(library, monkeypatch):
    library.set_reference("Harbor Lights", sing(), "original.mp3")
    library.set_reference("Winter Town", sing(), "original.mp3")
    result, run = drilled(library, monkeypatch, "harbor", "--transpose", "2")
    (song, titles, prepare), = run
    assert titles == ["Harbor Lights", "Winter Town"] and song.shift == 2
    assert prepare("Winter Town").shift == 0  # no take of it compared with its melody: the original key


def test_a_song_needs_a_reference_to_drill(library, monkeypatch):
    result, run = drilled(library, monkeypatch, "winter")
    assert result.exit_code == 1 and not run
    said = " ".join(result.output.split())  # undo the terminal's line wrapping
    assert "no reference melody" in said and "shed reference 'Winter Town'" in said


class FakeListener:
    """Stands in for listen.Listener: counts what it's fed, and hears nothing (no passes)."""

    def __init__(self, rate, language=None):
        self.rate, self.fed, self.ready = rate, [], True

    @property
    def now(self):
        return sum(len(block) for block in self.fed) / self.rate

    def feed(self, block):
        self.fed.append(block)

    def results(self):
        return []

    def close(self):
        pass


@pytest.fixture
def voice(monkeypatch):
    """What the pitch tracker hears: the note set (semitones from A440), wherever there's sound; the device it
    was asked to run on."""
    heard = SimpleNamespace(note=None, devices=[])

    def pitch(samples, device=None):
        heard.devices.append(device)
        frames = len(samples) // 160 + 1
        sounding = heard.note is not None and np.abs(samples).max() > 0
        return np.full(frames, 440 * 2 ** ((heard.note or 0) / 12)), np.full(frames, 1.0 if sounding else 0.0)

    monkeypatch.setattr(listen, "Listener", FakeListener)
    monkeypatch.setattr(rmvpe, "pitch", pitch)
    return heard


def harbor(title="Harbor Lights"):
    return drill.Song(title, sing(), 0, drill.starts(sing()), None)


def test_the_tone_is_not_taken_for_your_voice_then_the_note_is_found(voice):
    rate = 16_000
    d = drill.Drill(harbor, "Harbor Lights", ["Harbor Lights"], rate)
    d.play(0.0)
    assert d.played == 1
    voice.note = d.start.note  # the tone, heard by the microphone
    now = 0.0
    while now < drill.TONE_SECONDS + drill._AFTER_TONE:
        d.feed(np.ones(rate // 10, np.float32), now)
        now += 0.1
        d.update(now)
    assert d.sung is None and not d.found
    assert all(block.max() == 0 for block in d.listener.fed)  # the listener doesn't hear it either
    while now < 4.0:  # then you sing it
        d.feed(np.ones(rate // 10, np.float32), now)
        now += 0.1
        d.update(now)
    assert d.found and d.sung == pytest.approx(d.start.note)

    d.go(1, now)
    assert d.index == 1 and d.played == 2 and not d.found and d.sung is None


def test_a_drill_can_run_somewhere_else(voice):
    remote = drill.Remote(harbor, ["Harbor Lights"])
    with pytest.raises(ValueError):
        remote.start("Winter Town", 48_000)
    state = remote.start("Harbor Lights", 48_000)
    first = drill.starts(sing())[0]
    assert state["song"] == "Harbor Lights" and state["line"] == 1 and state["lines"] == len(LINES)
    assert state["note"] == drill.note_name(first.note) and state["played"] == 1 and state["listening"]
    voice.note = first.note + 0.7
    for _ in range(12):  # past the tone
        state = remote.hear(np.full(9600, 0.1, np.float32))
    assert not state["listening"] and state["sung"]["advice"] == "a little lower ↓ (70¢ over)"
    assert set(voice.devices) == {"cpu"}  # filing a take has the GPU
    assert remote.go(-1)["line"] == len(LINES)
    remote.stop()
    assert remote.state() is None and remote.hear(np.zeros(10, np.float32)) is None
    remote.close()
