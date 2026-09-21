"""Filing takes with `shed add` and `shed rec`: recognizing the song, asking when it isn't clear, and
what gets saved. The models are faked: these tests are about what Woodshed does with what they hear."""

import shutil
import subprocess
import wave
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd
from conftest import SONGS, mishear
from typer.testing import CliRunner

from woodshed import cli, config, genius, isolate, rating, recorder, transcribe
from woodshed.library import Library
from woodshed.rating import Metrics


class Shed:
    """Woodshed, set up, with a Whisper that hears `heard` and ratings that come out as `metrics`."""

    def __init__(self, library: Library):
        self.library, self.heard, self.metrics = library, [], Metrics(10, 0.02)

    def __call__(self, *args: str, input: str = ""):
        return CliRunner().invoke(cli.app, list(args), input=input)


@pytest.fixture
def shed(tmp_path, monkeypatch):
    shed = Shed(Library(tmp_path / "lib"))
    config.save(config.Config(library=str(shed.library.root)))
    monkeypatch.setattr(cli.models, "missing", lambda: [])
    monkeypatch.setattr(isolate, "separate", lambda src, start=0.0, seconds=None: SimpleNamespace(
        vocals_16k=lambda seconds: np.zeros(1, np.float32)))
    monkeypatch.setattr(transcribe, "transcribe", lambda samples, language=None: shed.heard)
    monkeypatch.setattr(rating, "analyze", lambda stems: shed.metrics)
    return shed


@pytest.fixture
def mic(monkeypatch):
    """`shed rec` records the sound file `mic.take` instead of listening."""
    mic = SimpleNamespace(take=None)
    monkeypatch.setattr(recorder, "record", lambda dest, *args: shutil.copy(mic.take, dest))
    return mic


def genius_finds(monkeypatch, title, genius_id):
    monkeypatch.setenv("GENIUS_ACCESS_TOKEN", "token")
    monkeypatch.setattr(genius, "search", lambda query, token: [
        {"title": title, "primary_artist": {"name": "The Originals"}, "id": genius_id}])


def text(result):
    return " ".join(result.output.split())  # undo the terminal's line wrapping


def test_a_take_of_a_song_you_have_recorded_is_filed_with_it(shed, tone, rng):
    shed.library.add_take(tone, "Harbor Lights", 2, 7, datetime(2020, 6, 2, 20, 0),
                          mishear(SONGS["Harbor Lights"], 0.3, rng), metrics=Metrics(15, 0.02))
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)

    result = shed("add", str(tone))

    assert result.exit_code == 0, result.output
    assert "Recognized Harbor Lights from your earlier takes" in text(result) and "(take 2 of 2)" in text(result)
    # pitch 7.5 and timing 8.0 make 7.7, against 6.2 for the earlier take
    assert "Rated 7.7/10" in text(result) and "your best take yet!" in text(result)
    takes = shed.library.takes_of("Harbor Lights")
    assert len(takes) == 2 and shed.metrics in [t.metrics for t in takes]


def test_an_older_recording_is_numbered_by_when_it_was_recorded(shed, tone, rng, tmp_path):
    for month in (7, 8, 9):
        shed.library.add_take(tone, "Harbor Lights", 2, 7, datetime(2026, month, 1, 20, 0),
                              mishear(SONGS["Harbor Lights"], 0.3, rng))
    memo = tmp_path / "memo.m4a"  # a voice memo from before those takes, its date inside it
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=196:duration=5",
                    "-metadata", "creation_time=2026-06-01T18:00:00Z", str(memo)], check=True)
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)

    result = shed("add", str(memo))

    assert result.exit_code == 0, result.output
    assert "(take 1 of 4)" in text(result)  # as `shed play` and `shed progress` number it (it used to say #4)


def test_a_new_song_is_recognized_on_genius_brackets_and_all(shed, tone, rng, monkeypatch):
    genius_finds(monkeypatch, "Gravel Road [sped up]", 42)  # "[sped up]" used to vanish, read as a style
    shed.heard = mishear(SONGS["Gravel Road"], 0.0, rng)

    result = shed("add", str(tone))

    assert result.exit_code == 0, result.output
    assert "Recognized Gravel Road [sped up] by The Originals on Genius" in text(result)
    [take] = shed.library.takes_of("Gravel Road [sped up]")
    assert take.genius_id == 42
    assert "Gravel Road [sped up]" in shed("songs").output


@pytest.mark.parametrize("answer, folder", [
    ("1", "Night River"),  # Genius's guess
    ("kitchen floor", "Kitchen Floor"),  # one of your songs, however it's typed
    ("Winter Town", "Winter Town"),  # a new song
    ("", "Unsorted"),
])
def test_you_choose_the_song_when_it_isnt_clear(shed, tone, monkeypatch, answer, folder):
    shed.library.add_take(tone, "Kitchen Floor", 2, 7, datetime(2020, 6, 2, 20, 0), [SONGS["Kitchen Floor"]])
    genius_finds(monkeypatch, "Night River", 7)
    shed.heard = ["we drove along the river in the rain tonight"]  # one search only: too little to be sure

    result = shed("add", str(tone), input=f"{answer}\n")

    assert result.exit_code == 0, result.output
    assert "Which song is this? 1. Night River — The Originals (Genius)" in text(result)
    filed = shed.library.unsorted() if folder == "Unsorted" else shed.library.songs()[folder]
    assert len(filed) == (2 if folder == "Kitchen Floor" else 1)


def test_add_says_which_files_it_cant_read_and_files_the_rest(shed, tone, tmp_path):
    notes = tmp_path / "notes.m4a"
    notes.write_text("not audio")

    result = shed("add", str(notes), str(tone), input="\n")  # Enter: the take goes to Unsorted

    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)  # not a traceback
    assert "Can't read notes.m4a" in text(result)
    assert len(shed.library.unsorted()) == 1


def test_a_recorded_take_is_filed_and_its_raw_recording_removed(shed, mic, tone):
    mic.take = tone  # 5 s of playing

    result = shed("rec", input="y\n\n")  # keep it although it's short; Enter: Unsorted

    assert result.exit_code == 0, result.output
    assert len(shed.library.unsorted()) == 1
    assert list(shed.library.incoming.iterdir()) == []


def test_a_recording_that_couldnt_be_filed_waits_until_shed_add_files_it(shed, mic, tone, monkeypatch):
    def crash(samples, language=None):
        raise RuntimeError("Whisper crashed")

    mic.take = tone
    monkeypatch.setattr(transcribe, "transcribe", crash)
    failed = shed("rec", input="y\n")  # keep it although it's short
    assert "Your recording is safe in" in text(failed)
    [raw] = shed.library.waiting()
    assert "1 recording(s) couldn't be filed yet. File them with: shed add" in text(shed("songs"))

    monkeypatch.setattr(transcribe, "transcribe", lambda samples, language=None: [])
    result = shed("add", str(raw), str(tone), input="\n\n")  # Enter twice: both to Unsorted

    assert result.exit_code == 0, result.output
    assert len(shed.library.unsorted()) == 2
    assert not raw.exists() and tone.exists()  # Woodshed's raw copy goes once filed; your own file stays
    assert "couldn't be filed" not in text(shed("songs"))


def test_a_silent_recording_is_not_kept(shed, mic, tmp_path):
    mic.take = tmp_path / "silence.wav"  # what macOS records when the terminal isn't allowed the microphone
    with wave.open(str(mic.take), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(bytes(2 * 16_000 * 30))

    result = shed("rec")

    assert result.exit_code == 0, result.output
    assert "completely silent, so it wasn't kept" in text(result)
    assert shed.library.unsorted() == [] and shed.library.songs() == {}
    assert list(shed.library.incoming.iterdir()) == []


def test_a_microphone_that_cant_record_that_way_is_reported(shed, monkeypatch):
    def refuse(**settings):
        raise sd.PortAudioError("Error opening InputStream: Invalid number of channels [PaErrorCode -9998]")

    monkeypatch.setattr(recorder.sd, "query_devices", lambda device, kind: {"default_samplerate": 48_000})
    monkeypatch.setattr(recorder.sd, "InputStream", refuse)

    result = shed("rec", "--channels", "2")

    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "Can't record from “your default microphone”: Error opening InputStream: Invalid number of channels" \
        in text(result)
    assert list(shed.library.incoming.iterdir()) == []  # no empty take left behind
