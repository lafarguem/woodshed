import contextlib
import stat

import pytest
import sounddevice as sd
from typer.testing import CliRunner

from woodshed import cli, config, genius, widgets

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
    result = setup(str(folder), "", "", tokens=["bad-token", "good-token"])  # Enter: suggested mic, default ratings

    assert result.exit_code == 0, result.output
    assert "Genius rejected that token" in result.output and "Token works" in result.output
    saved = config.load()
    assert saved.library == str(folder) and folder.is_dir()
    assert saved.device == "MacBook Pro Microphone"  # suggested over the Bluetooth default
    assert saved.genius_token == "good-token"
    assert stat.S_IMODE(config.PATH.stat().st_mode) == 0o600


def test_first_run_works_by_pressing_enter_at_every_question(setup, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # so the default ~/Music/Woodshed lands in the test folder
    result = setup("", "", "", tokens=[""])

    assert result.exit_code == 0, result.output
    assert config.load() == config.Config(device="MacBook Pro Microphone")  # every other answer is the default
    assert (tmp_path / "Music" / "Woodshed").is_dir()


def test_running_init_again_keeps_previous_answers(setup, tmp_path):
    setup(str(tmp_path / "lib"), "1", "y", "4", "30", "2", "8", tokens=["good-token"])
    result = setup("", "", "", tokens=[""])  # Enter everywhere

    assert result.exit_code == 0, result.output
    assert config.load() == config.Config(str(tmp_path / "lib"), "Galaxy Buds2 Pro", "good-token", 4, 30, 2, 8)


def test_token_is_optional(setup, tmp_path):
    result = setup(str(tmp_path), "2", "", tokens=[""])
    assert result.exit_code == 0 and config.load().genius_token is None


def test_rating_reference_points_can_be_changed(setup, tmp_path):
    # An inconsistent pitch range (10/10 above 0/10) is asked again.
    result = setup(str(tmp_path), "", "y", "20", "10", "3", "15", "", "", tokens=[""])
    assert result.exit_code == 0, result.output
    assert "must be below the 0/10 value, and at most 50" in result.output
    references = config.load().references()
    assert references.pitch_cents == (3, 15) and references.tempo_spread == (0.01, 0.06)  # timing kept


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
    result = setup(str(tmp_path / "lib"), "", "", tokens=[""])

    assert result.exit_code == 0, result.output
    assert "starting from the defaults" in result.output
    assert config.load().library == str(tmp_path / "lib")


def test_an_endless_timing_range_is_asked_again(setup, tmp_path):
    result = setup(str(tmp_path), "", "y", "", "", "1", "inf", "1", "8", tokens=[""])
    assert result.exit_code == 0, result.output
    assert "at most 15" in result.output
    assert config.load().references().tempo_spread == (0.01, 0.08)


def test_arrow_keys_pick_the_microphone_and_slide_the_ratings(setup, tmp_path):
    keys = ["up", "enter",  # from the suggested MacBook mic up to the earbuds
            "down", "enter",  # "Adjust them"
            "left", "left", "enter",  # pitch 10/10 within 5¢ -> 3¢
            "enter",  # pitch 0/10 from 25¢, kept
            "right", "enter",  # timing 10/10 within ±1% -> ±1.5%
            "up", "enter"]  # timing 0/10 from ±6% -> ±8.5% (a big step)
    result = setup(str(tmp_path), tokens=[""], keys=keys)

    assert result.exit_code == 0, result.output
    saved = config.load()
    assert saved.device == "Galaxy Buds2 Pro"
    assert (saved.pitch_best_cents, saved.pitch_worst_cents) == (3, 25)
    assert (saved.timing_best_percent, saved.timing_worst_percent) == (1.5, 8.5)
