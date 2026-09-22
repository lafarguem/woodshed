"""Rating a song against a reference melody: `shed reference`, and what filing, `progress` and `play` then
say. The models are faked (see `shed` in conftest.py), and the melodies made up note by note (test_melody.py)."""

import json
from datetime import datetime

import pytest
from conftest import SONGS, add_take, mishear, recording
from test_melody import LINES, sing

from woodshed import player
from woodshed.lyrics import words
from woodshed.rating import Metrics


def text(result):
    return " ".join(result.output.split())  # undo the terminal's line wrapping


@pytest.fixture
def harbor(shed, tone, tmp_path):
    """Harbor Lights with one take, filed before Woodshed kept melodies, and a recording of the original."""
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), LINES, metrics=Metrics(20, 0.02))
    shed.original = recording(tmp_path, "original.mp3")
    return shed


def set_reference(shed, melody=None):
    shed.melody = melody or sing()
    result = shed("reference", "harbor", str(shed.original))
    assert result.exit_code == 0, result.output
    return result


def test_a_reference_is_set_shown_and_removed(harbor):
    result = set_reference(harbor)
    assert "“Harbor Lights” is now rated against the melody of original.mp3 (6 lines heard)" in text(result)
    assert "1 of its takes was filed before Woodshed kept their melody" in text(result)
    assert "Reference" in text(harbor("songs")) and "✓" in text(harbor("songs"))
    shown = text(harbor("reference", "harbor"))
    assert "“Harbor Lights” is rated against the melody of" in shown and "original.mp3" in shown

    removed = harbor("reference", "harbor", "--remove")

    assert removed.exit_code == 0 and "“Harbor Lights” is rated against its scale again" in text(removed)
    assert "Reference" not in text(harbor("songs"))
    assert "“Harbor Lights” is rated against its scale." in text(harbor("reference", "harbor"))


def test_a_reference_is_only_for_a_song_you_have(harbor):
    result = harbor("reference", "night river", str(harbor.original))
    assert result.exit_code == 1 and "No song called “night river”" in text(result)


def test_a_recording_with_too_little_singing_isnt_kept(harbor):
    harbor.melody = sing(order=[0, 1])  # 14 words
    result = harbor("reference", "harbor", str(harbor.original))
    assert result.exit_code == 1 and "Heard only 14 words sung in original.mp3: too few to follow its melody" \
        in text(result)
    assert not harbor.library.has_reference("Harbor Lights")


def verse_of(song):
    lines = SONGS[song].split(" / ")
    return sing(song=lines, tune=[[0] * len(words(line)) for line in lines])


def test_the_song_is_recognized_from_the_recording(harbor):
    harbor.melody = sing()

    result = harbor("reference", str(harbor.original), input="\n")  # Enter: yes

    assert result.exit_code == 0, result.output
    assert "It sounds like “Harbor Lights”. Rate that song against it?" in text(result)
    assert harbor.library.has_reference("Harbor Lights")


def test_you_say_which_song_when_it_isnt_recognized(harbor):
    harbor.melody = verse_of("Winter Town")  # a song you haven't filed takes of

    cancelled = harbor("reference", str(harbor.original), input="\n")
    named = harbor("reference", str(harbor.original), input="nope\nharbor\n")

    assert "Which of your songs is it?" in text(cancelled) and "Nothing was changed" in text(cancelled)
    assert "You have no song called “nope”" in text(named)
    assert "“Harbor Lights” is now rated against the melody of original.mp3" in text(named)


@pytest.mark.parametrize("answer, kept", [("\n", False), ("y\n", True)])
def test_a_recording_of_another_song_is_questioned(harbor, tone, answer, kept):
    add_take(harbor.library, tone, "Gravel Road", 2, 7, datetime(2026, 6, 1, 20, 0), SONGS["Gravel Road"].split(" / "))
    harbor.melody = verse_of("Gravel Road")

    result = harbor("reference", "harbor", str(harbor.original), input=answer)

    assert result.exit_code == 0, result.output
    assert "Its lyrics sound like “Gravel Road”, not “Harbor Lights”. Use it for “Harbor Lights” anyway?" \
        in text(result)
    assert harbor.library.has_reference("Harbor Lights") == kept and not harbor.library.has_reference("Gravel Road")


def test_a_recording_that_isnt_there_is_reported(harbor):
    result = harbor("reference", "harbor", "~/Music/orignal.mp3")
    assert result.exit_code == 1 and "There's no recording at ~/Music/orignal.mp3" in text(result)


def test_a_take_is_rated_on_the_melody_and_told_which_lines_are_off(harbor, tmp_path, rng):
    set_reference(harbor)
    harbor.heard = mishear(SONGS["Harbor Lights"], 0.2, rng)
    harbor.melody = sing(off=-1.0)  # a semitone under, all the way through

    result = harbor("add", str(recording(tmp_path, "memo.m4a")))

    assert result.exit_code == 0, result.output
    assert "Recognized Harbor Lights from your earlier takes" in text(result)
    # 0 on pitch, where the scale would give it 10; timing 8.0
    assert "Rated 3.2/10 · pitch 0.0/10 (100¢ off the melody)" in text(result)
    assert "your best" not in text(result)  # the earlier take isn't rated on the melody yet
    assert "Your lines sit about a semitone under the melody." in text(result)
    assert f"Furthest from the melody: 0:01 “{LINES[0]}” 100¢ under 0:05 “{LINES[1]}” 100¢ under" in text(result)
    new = harbor.library.takes_of("Harbor Lights", melody=True)[-1]
    assert new.melody == harbor.melody


def test_a_take_that_hardly_matches_the_reference_is_rated_on_the_scale(harbor, tmp_path, rng):
    set_reference(harbor)
    harbor.heard = mishear(SONGS["Harbor Lights"], 0.2, rng)
    harbor.melody = sing(order=[0, 1])

    result = harbor("add", str(recording(tmp_path, "memo.m4a")))

    assert "pitch 10.0/10 (10¢ off)" in text(result)
    assert "Too few lines of this take matched the reference, so its pitch is rated against the scale" in text(result)


def test_progress_follows_the_melody_of_takes_filed_before(harbor):
    set_reference(harbor)
    harbor.transcribed.clear()
    harbor.melody = sing(flat_line=2)  # what the take filed before turns out to have sung

    first = harbor("progress", "harbor")

    assert first.exit_code == 0, first.output
    assert len(harbor.transcribed) == 1
    assert "10.0 0¢ off the melody" in text(first)
    assert f"Take 1 against the melody: Furthest from the melody: 0:10 “{LINES[2]}” 100¢ under" in text(first)
    assert "Pitch: how close your lines are to the reference melody" in text(first)
    harbor("progress", "harbor")
    assert len(harbor.transcribed) == 1  # kept with the take, not redone


def test_the_worst_and_best_takes_are_picked_by_the_melody(harbor, tone, monkeypatch):
    set_reference(harbor)
    # On the scale, the take a semitone under would be the best (5¢ off), and the one on the melody the worst.
    for day, metrics, sung in ((2, Metrics(5, 0.02), sing(off=-1.0)), (3, Metrics(35, 0.02), sing())):
        add_take(harbor.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, day, 20, 0), LINES, metrics=metrics,
                                melody=sung)
    harbor.melody = sing(off=-0.3)  # the take filed before, once its melody is followed
    played = []
    monkeypatch.setattr(player, "play", lambda path, console: played.append(path.name[8:10]) or True)

    result = harbor("play", "harbor", "--pitch")

    assert result.exit_code == 0, result.output
    assert played == ["02", "03"]
    assert "pitch 0.0/10 (100¢ off the melody)" in text(result) and "pitch 10.0/10 (0¢ off the melody)" in text(result)


def test_a_reference_analyzed_by_an_older_version_is_set_again(harbor):
    set_reference(harbor)
    file = harbor.library.root / "Harbor Lights" / ".reference.json"
    data = json.loads(file.read_text())
    data["melody"]["version"] = 0
    file.write_text(json.dumps(data))

    result = harbor("progress", "harbor")

    assert result.exit_code == 0, result.output
    assert "was analyzed by an older version of Woodshed, so it's rated against its scale" in text(result)
    assert "shed reference 'Harbor Lights'" in text(result) and "20¢ off" in text(result)
