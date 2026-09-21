import stat

import pytest
import sounddevice as sd
from typer.testing import CliRunner

from woodshed import cli, config, genius

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
    monkeypatch.setattr(cli.models, "missing", lambda: [])

    def search(query, token):
        if token != "good-token":
            raise genius.InvalidToken("rejected")
        return []

    monkeypatch.setattr(genius, "search", search)
    answers = iter([])
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: next(answers))  # hidden token input

    def run(*lines, tokens=()):
        nonlocal answers
        answers = iter(tokens)
        return CliRunner().invoke(cli.app, ["init"], input="".join(f"{line}\n" for line in lines))

    return run


def test_init_saves_folder_microphone_and_a_working_token(setup, tmp_path):
    folder = tmp_path / "My Covers"
    result = setup(str(folder), "", tokens=["bad-token", "good-token"])  # Enter accepts the suggested mic

    assert result.exit_code == 0, result.output
    assert "Genius rejected that token" in result.output and "Token works" in result.output
    saved = config.load()
    assert saved.library == str(folder) and folder.is_dir()
    assert saved.device == "MacBook Pro Microphone"  # suggested over the Bluetooth default
    assert saved.genius_token == "good-token"
    assert stat.S_IMODE(config.PATH.stat().st_mode) == 0o600


def test_running_init_again_keeps_previous_answers(setup, tmp_path):
    setup(str(tmp_path / "lib"), "1", tokens=["good-token"])
    result = setup("", "", tokens=[""])  # Enter everywhere

    assert result.exit_code == 0, result.output
    assert config.load() == config.Config(str(tmp_path / "lib"), "Galaxy Buds2 Pro", "good-token")


def test_token_is_optional(setup, tmp_path):
    result = setup(str(tmp_path), "2", tokens=[""])
    assert result.exit_code == 0 and config.load().genius_token is None


def test_config_round_trips_awkward_values(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PATH", tmp_path / "config.toml")
    settings = config.Config('~/Musique/Répétitions "live"', "Scarlett 2i2 USB", None)
    config.save(settings)
    assert config.load() == settings
    assert "genius_token" not in config.PATH.read_text()
