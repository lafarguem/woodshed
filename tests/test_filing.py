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
from conftest import SONGS, add_take, mishear, recording
from test_genius import hit, linked

from woodshed import audio, config, genius, isolate, recorder, transcribe
from woodshed.rating import Metrics


@pytest.fixture
def mic(monkeypatch):
    """`shed rec` records the sound file `mic.take` instead of listening."""
    mic = SimpleNamespace(take=None)
    monkeypatch.setattr(recorder, "record", lambda dest, *args: shutil.copy(mic.take, dest))
    return mic


def genius_finds(monkeypatch, title, genius_id, artist="The Originals"):
    monkeypatch.setenv("GENIUS_ACCESS_TOKEN", "token")
    monkeypatch.setattr(genius, "search", lambda query, token: [hit(title, artist, genius_id)])


def text(result):
    return " ".join(result.output.split())  # undo the terminal's line wrapping


def test_a_take_of_a_song_you_have_recorded_is_filed_with_it(shed, tone, rng):
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2020, 6, 2, 20, 0),
                          mishear(SONGS["Harbor Lights"], 0.3, rng), metrics=Metrics(25, 0.02))
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)

    result = shed("add", str(tone))

    assert result.exit_code == 0, result.output
    assert "Recognized Harbor Lights from your earlier takes" in text(result) and "(take 2 of 2)" in text(result)
    # pitch 10 and timing 8.0 make 9.2, against 6.2 for the earlier take
    assert "Rated 9.2/10" in text(result) and "your best take yet!" in text(result)
    takes = shed.library.takes_of("Harbor Lights")
    assert len(takes) == 2 and shed.metrics in [t.metrics for t in takes]


def test_an_older_recording_is_numbered_by_when_it_was_recorded(shed, tone, rng, tmp_path):
    for month in (7, 8, 9):
        add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, month, 1, 20, 0),
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


def test_a_song_genius_finds_as_a_cover_is_filed_as_the_originals(shed, tone, rng, monkeypatch, genius_pages):
    genius_finds(monkeypatch, "Gravel Road", 42, artist="A Cover Singer")
    genius_pages[42] = linked("cover_of", hit("Gravel Road", "The Originals", 7))
    shed.heard = mishear(SONGS["Gravel Road"], 0.0, rng)

    result = shed("add", str(tone))

    assert result.exit_code == 0, result.output
    assert "Recognized Gravel Road by The Originals on Genius" in text(result)
    [take] = shed.library.takes_of("Gravel Road")
    assert (take.artist, take.genius_id) == ("The Originals", 7)


@pytest.mark.parametrize("answer, folder", [
    ("1", "Night River"),  # Genius's guess
    ("kitchen floor", "Kitchen Floor"),  # one of your songs, however it's typed
    ("Winter Town", "Winter Town"),  # a new song
    ("", "Unsorted"),
])
def test_you_choose_the_song_when_it_isnt_clear(shed, tone, monkeypatch, answer, folder):
    add_take(shed.library, tone, "Kitchen Floor", 2, 7, datetime(2020, 6, 2, 20, 0), [SONGS["Kitchen Floor"]])
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


def test_a_recording_that_couldnt_be_filed_waits_until_shed_add_files_it(shed, mic, tone, tmp_path, monkeypatch):
    def crash(samples, language=None):
        raise RuntimeError("Whisper crashed")

    mic.take = tone
    monkeypatch.setattr(transcribe, "transcribe", crash)
    failed = shed("rec", input="y\n")  # keep it although it's short
    assert "Your recording is safe in" in text(failed)
    [raw] = shed.library.waiting()
    assert list(shed.library.incoming.glob("filing-*")) == []  # the take encoded for filing is gone
    assert "1 take waiting to be filed. File it with: shed add" in text(shed("songs"))

    monkeypatch.setattr(transcribe, "transcribe", lambda samples, language=None: [])
    memo = recording(tmp_path, "memo.m4a")
    result = shed("add", str(raw), str(memo), input="\n\n")  # Enter twice: both to Unsorted

    assert result.exit_code == 0, result.output
    assert len(shed.library.unsorted()) == 2
    assert not raw.exists() and memo.exists()  # Woodshed's raw copy goes once filed; your own file stays
    assert "waiting to be filed" not in text(shed("songs"))


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


def test_a_folder_files_its_songs_and_passes_over_the_rest(shed, tone, rng, tmp_path):
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2020, 6, 2, 20, 0), mishear(SONGS["Harbor Lights"], 0.3, rng))
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a")  # sung
    recording(memos, "Standard recording 2.m4a", hz=660)  # talking
    recording(memos, "Standard recording 3.m4a", seconds=5)
    recording(memos, ".recently_deleted_(1789)Standard recording 1.m4a")  # the recorder app's bin
    (memos / "setlist.txt").write_text("Harbor Lights")
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)

    result = shed("add", str(memos))

    assert result.exit_code == 0, result.output
    assert len(shed.library.takes_of("Harbor Lights")) == 2
    assert len(shed.transcribed) == 1  # the talking was never transcribed, let alone looked up on Genius
    assert "Filed 1 take · skipped 2:" in text(result)
    assert "no singing (1)" in text(result) and "shorter than 20 s (1)" in text(result)


def test_adding_a_folder_again_only_looks_at_new_recordings(shed, rng, tmp_path):
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a", when="2026-06-01T18:00:00Z")
    recording(memos, "Standard recording 2.m4a", hz=660)
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng)
    shed("add", str(memos), input="Harbor Lights\n")
    recording(memos, "Standard recording 3.m4a", seconds=30)  # recorded since
    shed.checked.clear()

    again = shed("add", str(memos))

    assert again.exit_code == 0, again.output
    assert len(shed.checked) == 1  # only the new one
    assert "already filed (1)" in text(again) and "no singing (1)" in text(again)
    assert "Recognized Harbor Lights from your earlier takes" in text(again)
    assert len(shed.library.takes_of("Harbor Lights")) == 2


def test_songs_it_isnt_sure_of_are_asked_about_at_the_end(shed, rng, tmp_path):
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a", when="2026-06-01T18:00:00Z")
    recording(memos, "Standard recording 2.m4a", when="2026-06-02T18:00:00Z", seconds=30)
    shed.heard = mishear(SONGS["Gravel Road"], 0.3, rng)  # both takes of a song you haven't filed yet

    # Name the first: the second is then recognized from it, without asking.
    result = shed("add", str(memos), input="Gravel Road\n")

    assert result.exit_code == 0, result.output
    assert result.output.index("Which song is this?") > result.output.index("Standard recording 2.m4a")
    assert text(result).count("Which song is this?") == 1 and "or - to skip it" in text(result)
    assert "Recognized Gravel Road from your earlier takes" in text(result)
    assert len(shed.library.takes_of("Gravel Road")) == 2


def test_a_song_you_skip_isnt_asked_about_again(shed, tmp_path):
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a")
    shed.heard = ["la la la la"]

    skipped = shed("add", str(memos), input="-\n")
    again = shed("add", str(memos))

    assert skipped.exit_code == 0 and shed.library.unsorted() == [] and shed.library.songs() == {}
    assert list(shed.library.incoming.glob("filing-*")) == []  # the take encoded for filing is gone
    assert "Which song is this?" not in again.output and "you skipped it (1)" in text(again)


def test_a_dry_run_files_and_remembers_nothing(shed, tmp_path):
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a")
    recording(memos, "Standard recording 2.m4a", hz=660)

    result = shed("add", "--dry-run", str(memos))

    assert result.exit_code == 0, result.output
    assert "Would file 1 take · skipped 1: no singing (1)" in text(result)
    assert shed.transcribed == [] and shed.library.songs() == {} and shed.library.unsorted() == []
    assert shed.library.skipped() == {}


def test_a_recording_is_never_filed_twice(shed, tone):
    first = shed("add", str(tone), input="\n")
    again = shed("add", str(tone))

    assert first.exit_code == 0 and again.exit_code == 0
    assert "is already filed, as Unsorted/" in text(again)
    assert len(shed.library.unsorted()) == 1


def test_takes_recorded_for_later_are_filed_together(shed, mic, rng, tmp_path):
    mic.take = recording(tmp_path, "take.wav")
    shed.heard = mishear(SONGS["Gravel Road"], 0.3, rng)  # a song you haven't filed takes of yet

    first, second = shed("rec", "--later"), shed("rec", "--later")

    assert first.exit_code == 0 and second.exit_code == 0, second.output
    assert "Kept for later (2 takes waiting). File them with shed add." in text(second)
    assert shed.transcribed == [] and len(shed.library.waiting()) == 2

    result = shed("add", input="Gravel Road\n")

    assert result.exit_code == 0, result.output
    # Neither was clear, so both waited for the end: naming the first made the second clear.
    assert "Unsure 1 of 2: take recorded" in text(result)
    assert text(result).count("Which song is this?") == 1 and "or - to skip it" not in text(result)
    assert "Recognized Gravel Road from your earlier takes" in text(result) and "Filed 2 takes" in text(result)
    assert len(shed.library.takes_of("Gravel Road")) == 2 and shed.library.waiting() == []
    assert "No takes are waiting to be filed" in text(shed("add"))


def test_a_silent_take_isnt_kept_for_later(shed, mic, tmp_path):
    mic.take = tmp_path / "silence.wav"
    with wave.open(str(mic.take), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(bytes(2 * 16_000 * 30))

    result = shed("rec", "--later")

    assert "completely silent, so it wasn't kept" in text(result) and shed.library.waiting() == []


def test_a_take_isnt_waiting_to_be_filed_until_its_recorded(shed, tone, monkeypatch):
    # So that `shed add`, run while you record, can't file (and then remove) half a take.
    while_recording = []

    def record(dest, *args):
        shutil.copy(tone, dest)  # the audio so far
        while_recording.append(shed.library.waiting())

    monkeypatch.setattr(recorder, "record", record)
    result = shed("rec", "--later", input="y\n")  # keep it although it's short

    assert result.exit_code == 0, result.output
    assert while_recording == [[]] and len(shed.library.waiting()) == 1


def test_whats_analyzed_is_the_take_as_filed(shed, tone, monkeypatch):
    # Not the recording it was made from, which reads a little differently: analyzed again later (after an
    # update, say), a take then reads the same.
    analyzed = []

    def separate(src, start=0.0, seconds=None):
        analyzed.append(audio.load(src))
        return SimpleNamespace(vocals_16k=lambda seconds=None: np.zeros(1, np.float32))

    monkeypatch.setattr(isolate, "separate", separate)
    result = shed("add", str(tone), input="\n")  # Enter: Unsorted

    assert result.exit_code == 0, result.output
    [take] = shed.library.unsorted()
    assert np.array_equal(analyzed[0], audio.load(take))
    assert list(shed.library.incoming.glob("*")) == []


def test_takes_can_be_saved_in_apple_lossless(shed, tone, rng):
    from test_melody import REFERENCE

    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2020, 6, 2, 20, 0),
             mishear(SONGS["Harbor Lights"], 0.3, rng))  # an mp3, filed before
    config.save(config.Config(library=str(shed.library.root), take_format="m4a"))
    shed.heard, shed.melody = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6), REFERENCE

    result = shed("add", str(tone))

    assert result.exit_code == 0, result.output
    assert "Recognized Harbor Lights from your earlier takes" in text(result) and "(take 2 of 2)" in text(result)
    before, new = shed.library.takes_of("Harbor Lights", melody=True)
    assert (before.path.suffix, new.path.suffix) == (".mp3", ".m4a")
    codec = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                            str(new.path)], capture_output=True, text=True, check=True).stdout.strip()
    assert codec == "alac"
    assert new.transcript == "\n".join(shed.heard) and new.metrics == shed.metrics and new.melody == REFERENCE
    shed.library.save_metrics(new.path, Metrics(12, 0.03))
    assert shed.library.takes_of("Harbor Lights")[1].metrics == Metrics(12, 0.03)
    assert "?" not in shed("songs", "harbor").output  # its length is read too


def test_stopping_at_the_question_leaves_the_recording_waiting(shed, mic, tmp_path, monkeypatch):
    def stop(prompt):
        raise KeyboardInterrupt  # Ctrl+C at "Which song is this?"

    mic.take = recording(tmp_path, "take.wav")  # long enough not to be asked whether to keep it
    monkeypatch.setattr("builtins.input", stop)
    result = shed("rec")

    assert result.exit_code != 0 and "Your recording is safe in" in text(result)
    assert len(shed.library.waiting()) == 1 and list(shed.library.incoming.glob("filing-*")) == []


def test_a_take_can_be_saved_in_another_format_than_usual(shed, mic, tmp_path, rng):
    def rec(*args, input=""):
        n = len(list(tmp_path.glob("take*")))
        mic.take = recording(tmp_path, f"take{n}.wav", hz=150 + 10 * n)
        return shed("rec", *args, input=input)  # each take its own recording, not a copy: never filed twice

    shed.heard = mishear(SONGS["Gravel Road"], 0.3, rng)
    now = rec("--m4a", input="Gravel Road\n")
    rec("--later", "--m4a")  # remembered until it's filed
    rec("--later")
    later = shed("add")

    assert now.exit_code == 0 and later.exit_code == 0, later.output
    assert [take.suffix for take in shed.library.songs()["Gravel Road"]] == [".m4a", ".m4a", ".mp3"]
    config.save(config.Config(library=str(shed.library.root), take_format="m4a"))
    rec("--mp3")
    assert shed.library.songs()["Gravel Road"][-1].suffix == ".mp3"


@pytest.mark.parametrize("keys, typed, folder", [
    (["enter"], "", "Night River"),  # Genius's guess comes first
    (["down", "enter"], "kitchen floor\n", "Kitchen Floor"),  # another song: one of yours, however it's typed
    (["down", "enter"], "Winter Town\n", "Winter Town"),  # or a new one
    (["down", "enter", "down", "down", "enter"], "\n", "Unsorted"),  # typing nothing goes back to the list
])
def test_you_pick_the_song_with_the_arrow_keys(shed, tone, monkeypatch, arrow_keys, keys, typed, folder):
    add_take(shed.library, tone, "Kitchen Floor", 2, 7, datetime(2020, 6, 2, 20, 0), [SONGS["Kitchen Floor"]])
    genius_finds(monkeypatch, "Night River", 7)
    shed.heard = ["we drove along the river in the rain tonight"]
    arrow_keys(*keys)

    result = shed("add", str(tone), input=typed)

    assert result.exit_code == 0, result.output
    assert "Number" not in result.output and "Skip it" not in result.output  # a file named is always filed
    filed = shed.library.unsorted() if folder == "Unsorted" else shed.library.songs()[folder]
    assert len(filed) == (2 if folder == "Kitchen Floor" else 1)


def test_a_recording_from_a_folder_can_be_skipped_with_the_arrow_keys(shed, tmp_path, arrow_keys):
    memos = tmp_path / "memos"
    recording(memos, "Standard recording 1.m4a")
    shed.heard = ["la la la la"]
    arrow_keys("up", "enter")  # up from the first: the last, "Skip it"

    result = shed("add", str(memos))

    assert result.exit_code == 0, result.output
    assert "you skipped it (1)" in text(result) and shed.library.unsorted() == [] and shed.library.songs() == {}
