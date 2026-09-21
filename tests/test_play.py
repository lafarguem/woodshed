from datetime import datetime

import pytest
from typer.testing import CliRunner

from woodshed import cli, player, rating
from woodshed.library import Library
from woodshed.rating import Metrics


class FakeAfplay:
    """Records what would have played, and finishes instantly."""

    played: list[str] = []

    def __init__(self, args):
        FakeAfplay.played.append(args[1])

    def poll(self):
        return 0

    def terminate(self):
        pass

    def wait(self):
        pass


@pytest.fixture
def songs(tmp_path, tone, monkeypatch):
    library = Library(tmp_path / "lib")
    # Filed out of order: playback must follow the recording dates.
    # Pitch is worst on the 14th and best on the 30th; timing is worst on the 30th.
    for day, metrics in ((14, Metrics(35, 0.01)), (2, Metrics(28, 0.02)), (30, Metrics(10, 0.05))):
        library.add_take(tone, "Harbor Lights", 2, 7, datetime(2026, 6, day, 20, 0), [], metrics=metrics)
    library.add_take(tone, "Harbor Song", 2, 7, datetime(2026, 6, 1, 20, 0), [])
    library.add_take(tone, "Winter Town", 2, 7, datetime(2026, 6, 1, 20, 0), [])
    FakeAfplay.played = []
    monkeypatch.setattr(player.subprocess, "Popen", FakeAfplay)  # after ffmpeg has made the takes
    return library


def play(library, *args):
    return CliRunner().invoke(cli.app, ["play", *args, "--library", str(library.root)])


def text(result):
    return " ".join(result.output.split())  # undo the terminal's line wrapping


def played():
    return [p.split("/")[-1][8:10] for p in FakeAfplay.played]  # the day of each take played


def test_plays_the_first_take_then_the_latest(songs):
    result = play(songs, "harbor lights")
    assert result.exit_code == 0, result.output
    assert [p.split("/")[-1] for p in FakeAfplay.played] == ["2026-06-02_20-00.mp3", "2026-06-30_20-00.mp3"]
    assert "first take (1 of 3)" in result.output and "latest take (3 of 3)" in result.output


def test_a_single_take_plays_once(songs):
    result = play(songs, "winter")
    assert result.exit_code == 0 and len(FakeAfplay.played) == 1
    assert "Only one take so far" in result.output


def test_ambiguous_or_unknown_names_play_nothing(songs):
    ambiguous = play(songs, "harbor")
    assert ambiguous.exit_code == 1 and "did you mean Harbor Lights or Harbor Song?" in ambiguous.output
    assert play(songs, "nope").exit_code == 1
    assert FakeAfplay.played == []


def test_rating_plays_the_worst_take_then_the_best(songs):
    # overall = 0.6 pitch + 0.4 timing: the 2nd rates 5.5, the 14th 4.7, the 30th 6.8
    result = play(songs, "harbor lights", "--rating")
    assert result.exit_code == 0, result.output
    assert played() == ["14", "30"]
    assert "worst take by rating (2 of 3)" in text(result) and "rated 6.8/10" in text(result)


def test_rating_by_one_metric(songs):
    assert play(songs, "harbor lights", "--rating", "timing").exit_code == 0
    assert played() == ["30", "14"]  # timing: 2.0 on the 30th, 10 on the 14th


def test_a_metric_alone_implies_rating_and_unknown_ones_are_refused(songs):
    assert play(songs, "harbor lights", "pitch").exit_code == 0 and played() == ["14", "30"]
    FakeAfplay.played = []
    refused = play(songs, "harbor", "lights")
    assert refused.exit_code == 1 and "isn't a rating" in refused.output and FakeAfplay.played == []


def test_progress_rates_older_takes_once(songs, monkeypatch):
    from woodshed import isolate

    calls = []
    monkeypatch.setattr(isolate, "separate", lambda path: calls.append(path) or None)
    monkeypatch.setattr(rating, "analyze", lambda stems: Metrics(18, 0.01))
    monkeypatch.setattr(cli.models, "missing", lambda: [])
    runner = CliRunner()

    first = runner.invoke(cli.app, ["progress", "winter", "--library", str(songs.root)])
    assert first.exit_code == 0, first.output
    assert "Winter Town" in first.output and "8.6" in first.output and len(calls) == 1
    runner.invoke(cli.app, ["progress", "winter", "--library", str(songs.root)])
    assert len(calls) == 1  # saved with the take, not redone


def test_progress_shows_every_take_and_the_trend(songs):
    result = CliRunner().invoke(cli.app, ["progress", "harbor lights", "--library", str(songs.root)])
    assert result.exit_code == 0, result.output
    assert "35¢ off" in text(result) and "±5.0%" in text(result)
    assert "5.5 → 6.8 since your first take; best 6.8 (take 3)" in text(result)


@pytest.mark.parametrize("command", [["songs"], ["play", "harbor"], ["progress", "harbor"]])
def test_a_mistyped_library_is_reported_not_created(tmp_path, command):
    typo = tmp_path / "Woodshd"
    result = CliRunner().invoke(cli.app, [*command, "--library", str(typo)])
    assert result.exit_code == 1 and "There's no library at" in text(result)
    assert not typo.exists()


def test_stricter_reference_points_lower_every_score(songs, tmp_path, monkeypatch):
    from woodshed import config

    monkeypatch.setattr(config, "PATH", tmp_path / "config.toml")
    config.save(config.Config(pitch_best_cents=2, pitch_worst_cents=12))  # stricter pitch, same timing
    result = CliRunner().invoke(cli.app, ["progress", "harbor lights", "--library", str(songs.root)])
    assert result.exit_code == 0, result.output
    # the 30th: pitch 10¢ now scores 2.0 (was 10), so overall 0.6 * 2.0 + 0.4 * 2.0 = 2.0 (was 6.8), and the
    # best is the 14th, on its timing alone: 0.4 * 10 = 4.0
    assert "2.0 10¢ off" in text(result) and "best 4.0 (take 2)" in text(result)
