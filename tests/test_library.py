import subprocess
from datetime import datetime

import numpy as np
import pytest
from conftest import SONGS, mishear

from woodshed import audio
from woodshed.library import Library, is_confident


@pytest.fixture
def tone(tmp_path):
    """Two seconds of silence, five of a guitar-ish tone, two of silence."""
    path = tmp_path / "take.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=196:sample_rate=44100:duration=5",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
                    "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(path)], check=True)
    return path


def test_playing_bounds_trim_silence_but_ignore_clicks(tone):
    samples = audio.load(tone).copy()
    samples[int(0.5 * audio.SAMPLE_RATE)] = 0.9  # a key click before playing starts
    start, end = audio.playing_bounds(samples, pad=0)
    assert start == pytest.approx(2, abs=0.3) and end == pytest.approx(7, abs=0.3)


def test_filed_takes_are_recognized_and_follow_their_folder(tmp_path, tone, rng):
    songs = Library(tmp_path / "lib")
    when = datetime(2026, 9, 21, 18, 30)
    for song, text in SONGS.items():
        songs.add_take(tone, song, 2, 7, when, mishear(text, 0.3, rng), genius_id=hash(song) % 1000)

    new_take = mishear(SONGS["Winter Town"], 0.3, rng, keep=0.6)
    ranked = songs.match(new_take)
    assert ranked[0][0] == "Winter Town" and is_confident(new_take, ranked)
    assert songs.song_for_genius_id(hash("Winter Town") % 1000) == "Winter Town"
    assert songs.find_song("winter  town") == "Winter Town"

    # Moving a take in Finder re-files it: its lyrics now count for the new folder.
    (tmp_path / "lib" / "Winter Town" / "2026-09-21_18-30.mp3").rename(tmp_path / "lib" / "Snow Song.mp3")
    (tmp_path / "lib" / "Snow Song").mkdir()
    (tmp_path / "lib" / "Snow Song.mp3").rename(tmp_path / "lib" / "Snow Song" / "2026-09-21_18-30.mp3")
    assert songs.match(new_take)[0][0] == "Snow Song"


def test_same_minute_takes_dont_overwrite(tmp_path, tone):
    songs = Library(tmp_path)
    when = datetime(2026, 9, 21, 18, 30)
    first = songs.add_take(tone, "Song", 0, 9, when, [])
    second = songs.add_take(tone, "Song", 0, 9, when, [])
    assert (first.name, second.name) == ("2026-09-21_18-30.mp3", "2026-09-21_18-30_2.mp3")


def test_unrecognizable_takes_go_to_unsorted(tmp_path, tone):
    songs = Library(tmp_path)
    dest = songs.add_take(tone, None, 0, 9, datetime(2026, 1, 1), ["mm"])
    assert dest.parent.name == "Unsorted" and songs.songs() == {} and songs.unsorted() == [dest]
    assert not is_confident(["mm"], songs.match(["mm"]))
