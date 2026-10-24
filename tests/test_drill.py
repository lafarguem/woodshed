"""Finding the notes of a song word by word: `shed drill`. The microphone and speakers are left out: what's played
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


def test_every_word_is_drilled_on_its_note():
    steps = drill.steps(sing())
    assert [(s.line, s.position, s.word) for s in steps] == [(line, i, w) for line in LINES
                                                             for i, w in enumerate(words(line))]
    assert [s.note for s in steps] == [note for line in TUNE for note in line]


def test_a_word_with_no_note_is_skipped_but_the_others_keep_their_place():
    take = sing()
    take.notes = [n for n in take.notes if not 1.5 <= n[0] < 2.0]  # the second word's note
    steps = drill.steps(take)
    assert [s.position for s in steps[:2]] == [0, 2] and steps[1].word == words(LINES[0])[2]


def test_up_and_down_go_to_the_start_of_another_line():
    steps = drill.steps(sing())
    first = [i for i, s in enumerate(steps) if s.position == 0]
    assert drill.next_line(steps, first[0] + 2, 1) == first[1]
    assert drill.next_line(steps, first[1] + 1, -1) == first[0]
    assert drill.next_line(steps, first[0], -1) == first[-1]  # round to the last line


def test_the_word_sung_is_shown_in_its_line():
    assert drill.highlighted("don't let go", 1) == [("don't ", ""), ("let", "bold underline"), (" go", "")]
    assert drill.highlighted("one", 3) == [("one", "")]


def test_the_note_moves_to_your_key_and_your_register():
    first = TUNE[0][0]
    assert drill.steps(sing(), shift=3)[0].note == first + 3
    # Sung an octave or two away, the note is played where your voice sits.
    assert drill.steps(sing(), register=first - 12)[0].note == first - 12
    assert drill.steps(sing(), shift=3, register=first + 3 + 25)[0].note == first + 3 + 24
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
    assert song.shift == 5 and song.steps[0].note == TUNE[0][0] + 5 - 12

    result, run = drilled(library, monkeypatch, "harbor", "--transpose", "-2")
    song = run[0][0]
    assert song.steps[0].note % 12 == (TUNE[0][0] - 2) % 12
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
    return drill.Song(title, sing(), 0, drill.steps(sing()), None)


def sung_for(d, rate, now, until):
    while now < until:
        d.feed(np.ones(rate // 10, np.float32), now)
        now += 0.1
        d.update(now)
    return now


def test_the_tone_is_not_taken_for_your_voice_then_the_note_is_found_and_the_next_word_played(voice):
    rate = 16_000
    d = drill.Drill(harbor, "Harbor Lights", ["Harbor Lights"], rate)
    d.play(0.0)
    assert d.played == 1
    first = d.step.note
    voice.note = first  # the tone, heard by the microphone
    now = sung_for(d, rate, 0.0, drill.TONE_SECONDS + drill._AFTER_TONE)
    assert d.sung is None and not d.found
    assert all(block.max() == 0 for block in d.listener.fed)  # the listener doesn't hear it either
    now = sung_for(d, rate, now, 4.0)  # then you sing it
    assert d.index == 1 and d.played == 2 and d.held == drill.note_name(first)
    assert not d.found and d.sung is None and not d.singing()

    d.go(1, now)
    assert d.index == 2 and d.played == 3 and d.held is None
    d.go(1, now, by_line=True)
    assert d.step.line_number == 1 and d.step.position == 0

    d.go(-1 - d.index, now)  # the last word: once it's found, you sing the song
    voice.note = d.step.note
    sung_for(d, rate, now, now + 4.0)
    assert d.index == len(d.song.steps) - 1 and d.found and d.singing()


def test_a_drill_can_run_somewhere_else(voice):
    remote = drill.Remote(harbor, ["Harbor Lights"])
    with pytest.raises(ValueError):
        remote.start("Winter Town", 48_000)
    state = remote.start("Harbor Lights", 48_000)
    first = drill.steps(sing())[0]
    assert state["song"] == "Harbor Lights" and state["step"] == 1 and state["line"] == 1
    assert state["steps"] == sum(len(words(line)) for line in LINES) and state["textParts"][1][1]
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
