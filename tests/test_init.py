import contextlib
import stat

import pytest
import sounddevice as sd
from typer.testing import CliRunner

from woodshed import cli, config, genius, rating, widgets

DEVICES = [
    {"name": "Galaxy Buds2 Pro", "max_input_channels": 1},
    {"name": "External Speakers", "max_input_channels": 0},
    {"name": "MacBook Pro Microphone", "max_input_channels": 1},
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PATH", tmp_path / "config" / "config.toml")
    monkeypatch.setattr(sd, "query_devices", lambda device=None, kind=None: DEVICES if device is None else DEVICES[device])
    monkeypatch.setattr(sd.default, "device", [0, 1])  # earbuds are the system default
    monkeypatch.setattr(cli.microphones, "connections", lambda: {
        "Galaxy Buds2 Pro": "coreaudio_device_type_bluetooth", "MacBook Pro Microphone": "coreaudio_device_type_builtin"})
    monkeypatch.setattr(cli.models, "missing", lambda: [])

    def search(query, token):
        if token != "good-token":
            raise genius.InvalidToken("rejected")
        return []

    monkeypatch.setattr(genius, "search", search)
    answers = iter([])
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: next(answers))  # hidden token input

    def run(*lines, tokens=(), keys=None):
        """Typed answers (lines), hidden token answers, and arrow-key presses (which make it interactive)."""
        nonlocal answers
        answers = iter(tokens)
        if keys is not None:
            presses = iter(keys)
            monkeypatch.setattr(cli, "_interactive", lambda: True)
            monkeypatch.setattr(widgets.terminal, "keys", lambda: contextlib.nullcontext(lambda: next(presses)))
        return CliRunner().invoke(cli.app, ["init"], input="".join(f"{line}\n" for line in lines))

    return run


def test_init_saves_folder_microphone_and_a_working_token(setup, tmp_path):
    folder = tmp_path / "My Covers"
    result = setup(str(folder), "", "", "", tokens=["bad-token", "good-token"])  # Enter: suggested mic, default ratings

    assert result.exit_code == 0, result.output
    assert "Genius rejected that token" in result.output and "Token works" in result.output
    saved = config.load()
    assert saved.library == str(folder) and folder.is_dir()
    assert saved.device == "MacBook Pro Microphone"  # suggested over the Bluetooth default
    assert saved.genius_token == "good-token"
    assert stat.S_IMODE(config.PATH.stat().st_mode) == 0o600


def test_first_run_works_by_pressing_enter_at_every_question(setup, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # so the default ~/Music/Woodshed lands in the test folder
    result = setup("", "", "", "", tokens=[""])

    assert result.exit_code == 0, result.output
    assert config.load() == config.Config(device="MacBook Pro Microphone")  # every other answer is the default
    assert (tmp_path / "Music" / "Woodshed").is_dir()


def test_running_init_again_keeps_previous_answers(setup, tmp_path):
    setup(str(tmp_path / "lib"), "", "1", "y", "4", "30", "20", "90", "2", "8", "70", tokens=["good-token"])
    result = setup("", "", "", "", tokens=[""])  # Enter everywhere

    assert result.exit_code == 0, result.output
    assert config.load() == config.Config(str(tmp_path / "lib"), "Galaxy Buds2 Pro", "good-token", 4, 30, 2, 8,
                                          pitch_weight_percent=70, melody_best_cents=20, melody_worst_cents=90)


def test_token_is_optional(setup, tmp_path):
    result = setup(str(tmp_path), "", "2", "", tokens=[""])
    assert result.exit_code == 0 and config.load().genius_token is None


def test_rating_reference_points_can_be_changed(setup, tmp_path):
    # An inconsistent pitch range (10/10 above 0/10) is asked again.
    result = setup(str(tmp_path), "", "", "y", "20", "10", "3", "15", "", "", "", "", "120", "80", tokens=[""])
    assert result.exit_code == 0, result.output
    assert "must be below the 0/10 value, and at most 50" in result.output and "Between 0 and 100" in result.output
    references = config.load().references()
    assert references.pitch_cents == (3, 15) and references.tempo_spread == (0.01, 0.06)  # timing kept
    assert references.melody_cents == (15, 100)  # and the melody's
    assert references.pitch_weight == 0.8


def test_the_melodys_reference_points_can_be_changed(setup, tmp_path):
    # Up to two semitones (200¢) for 0/10.
    result = setup(str(tmp_path), "", "", "y", "", "", "10", "300", "10", "60", "", "", "", tokens=[""])
    assert result.exit_code == 0, result.output
    assert "Pitch against a reference melody" in text(result) and "at most 200" in text(result)
    references = config.load().references()
    assert references.melody_cents == (10, 60) and references.pitch_cents == (12, 38)


def test_a_hand_edited_inconsistent_range_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PATH", tmp_path / "config.toml")
    config.save(config.Config(pitch_best_cents=30, pitch_worst_cents=10))
    with pytest.raises(SystemExit, match="inconsistent"):
        config.load().references()


def test_config_round_trips_awkward_values(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PATH", tmp_path / "config.toml")
    settings = config.Config('~/Musique/Répétitions "live"', "Scarlett 2i2 USB", None)
    config.save(settings)
    assert config.load() == settings
    assert "genius_token" not in config.PATH.read_text()


def text(result):
    return " ".join(result.output.split())  # undo the terminal's line wrapping


@pytest.mark.parametrize("chosen", ["MacBook Pro Microphone", "2"])  # a name or a number, as with `rec --device`
def test_devices_marks_the_microphone_rec_would_use(setup, monkeypatch, chosen):
    monkeypatch.setenv("WOODSHED_DEVICE", chosen)
    result = CliRunner().invoke(cli.app, ["devices"])

    assert result.exit_code == 0, result.output
    marked = [line for line in result.output.splitlines() if line.startswith("*")]
    assert len(marked) == 1 and "MacBook Pro Microphone" in marked[0]


def test_emoji_in_names_dont_break_the_config():
    settings = config.Config("~/Music/🎸 Covers", "Studio Mic 🎙", None)
    config.save(settings)
    assert config.load() == settings


def test_init_starts_over_from_an_unreadable_config(setup, tmp_path):
    config.PATH.parent.mkdir(parents=True)
    config.PATH.write_text('library = "~/Music/\\ud83c\\udfb8 Covers"\n')  # as older versions wrote emoji
    result = setup(str(tmp_path / "lib"), "", "", "", tokens=[""])

    assert result.exit_code == 0, result.output
    assert "starting from the defaults" in text(result)  # after the config's path, so wrapped wherever it ends
    assert config.load().library == str(tmp_path / "lib")


def test_an_endless_timing_range_is_asked_again(setup, tmp_path):
    result = setup(str(tmp_path), "", "", "y", "", "", "", "", "1", "inf", "1", "8", "", tokens=[""])
    assert result.exit_code == 0, result.output
    assert "at most 15" in text(result)
    assert config.load().references().tempo_spread == (0.01, 0.08)


def test_arrow_keys_pick_the_microphone_and_slide_the_ratings(setup, tmp_path):
    keys = ["up", "enter",  # from the suggested MacBook mic up to the earbuds
            "down", "enter",  # "Adjust them"
            "left", "left", "enter",  # pitch 10/10 within 12¢ -> 10¢
            "enter",  # pitch 0/10 from 38¢, kept
            "right", "enter",  # against a reference melody, 10/10 within 15¢ -> 16¢
            "down", "enter",  # and 0/10 from 100¢ -> 75¢ (a big step)
            "right", "enter",  # timing 10/10 within ±1% -> ±1.5%
            "up", "enter",  # timing 0/10 from ±6% -> ±8.5% (a big step)
            "left", "left", "enter"]  # the rating: 60% pitch -> 50%
    result = setup(str(tmp_path), tokens=[""], keys=["enter", *keys])  # Enter: MP3

    assert result.exit_code == 0, result.output
    saved = config.load()
    assert saved.device == "Galaxy Buds2 Pro"
    assert (saved.pitch_best_cents, saved.pitch_worst_cents) == (10, 38)
    assert (saved.melody_best_cents, saved.melody_worst_cents) == (16, 75)
    assert (saved.timing_best_percent, saved.timing_worst_percent) == (1.5, 8.5)
    assert saved.pitch_weight_percent == 50


def test_a_config_from_an_earlier_version_gets_todays_pitch_default():
    config.PATH.parent.mkdir(parents=True)
    config.PATH.write_text("pitch_best_cents = 5.0\npitch_worst_cents = 25.0\n"  # as earlier versions saved them all
                           "timing_best_percent = 1.0\ntiming_worst_percent = 6.0\n")
    assert config.load().references() == rating.DEFAULTS
    config.PATH.write_text("pitch_best_cents = 5.0\npitch_worst_cents = 10.0\n")  # chosen, since today's would clash
    assert config.load().references().pitch_cents == (5.0, 10.0)


def test_settings_left_at_their_defaults_arent_saved():
    config.save(config.Config(device="MacBook Pro Microphone", timing_worst_percent=8))
    saved = config.PATH.read_text()
    assert "pitch_" not in saved and "timing_best_percent" not in saved and "timing_worst_percent = 8" in saved
    config.save(config.Config(pitch_weight_percent=75))
    assert config.PATH.read_text() == "library = \"~/Music/Woodshed\"\ntake_format = \"mp3\"\npitch_weight_percent = 75\n"


@pytest.mark.parametrize("keys", [None, ["down", "enter", "enter", "enter"]])  # typed, or with the arrow keys
def test_takes_can_be_saved_in_apple_lossless(setup, tmp_path, keys):
    typed = ("m4a",) if keys is None else ()
    result = setup(str(tmp_path), *typed, "", "", tokens=[""], keys=keys)

    assert result.exit_code == 0, result.output
    assert "Apple Lossless" in result.output and config.load().take_format == "m4a"


def test_a_take_format_typed_by_hand_is_refused(tmp_path, monkeypatch):
    config.save(config.Config(library=str(tmp_path), take_format="flac"))
    monkeypatch.setattr(cli.models, "missing", lambda: [])
    result = CliRunner().invoke(cli.app, ["add", str(tmp_path)])
    assert result.exit_code == 1 and "must be mp3 or m4a" in " ".join(result.output.split())
