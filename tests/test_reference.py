"""Rating a song against a reference melody: `shed reference`, and what filing, `progress` and `play` then
say. The models are faked (see `shed` in conftest.py), and the melodies made up note by note (test_melody.py)."""

import contextlib
import json
import shutil
from datetime import datetime

import pytest
from conftest import SONGS, add_take, mishear, recording
from test_melody import LINES, lines_off, sing

from woodshed import cli, melody, player, widgets, youtube
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
    assert "“Harbor Lights” has no reference recording, so it's rated against its scale" \
        in text(harbor("reference", "harbor"))


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
    assert "Rated 3.2/10 · pitch 0.0/10 · timing 8.0/10 (tempo ±2.0%)" in text(result)
    assert "your best" not in text(result)  # the earlier take isn't rated on the melody yet
    assert ("Pitch, against the melody (6 lines compared): Typically 100¢ off, either way (10/10 at 15¢, 0/10 at "
            "100¢) Typically 100¢ under: 6 lines sit under the melody, 0 over") in text(result)
    assert f"Furthest from the melody: 0:01 “{LINES[0]}” 100¢ under 0:05 “{LINES[1]}” 100¢ under" in text(result)
    new = harbor.library.takes_of("Harbor Lights", melody=True)[-1]
    assert new.melody == harbor.melody


def test_a_take_that_hardly_matches_the_reference_is_rated_on_the_scale(harbor, tmp_path, rng):
    set_reference(harbor)
    harbor.heard = mishear(SONGS["Harbor Lights"], 0.2, rng)
    harbor.melody = sing(order=[0, 1])

    result = harbor("add", str(recording(tmp_path, "memo.m4a")))

    assert "Rated 9.2/10 · pitch 10.0/10 · timing 8.0/10 (tempo ±2.0%)" in text(result)
    assert ("Pitch, against the song's scale (too few lines of this take matched the reference, so its pitch is "
            "rated against the scale): Typically 10¢ off, either way (10/10 at 12¢, 0/10 at 38¢)") in text(result)


def test_progress_follows_the_melody_of_takes_filed_before(harbor):
    set_reference(harbor)
    harbor.transcribed.clear()
    harbor.melody = sing(flat_line=2)  # what the take filed before turns out to have sung

    first = harbor("progress", "harbor")

    assert first.exit_code == 0, first.output
    assert len(harbor.transcribed) == 1
    assert "10.0 0¢ off · centered" in text(first)  # one line of six sung flat: the median line isn't
    assert f"Take 1 against the melody: Furthest from the melody: 0:10 “{LINES[2]}” 100¢ under" in text(first)
    assert "Pitch: “off” is how far your lines typically are from the reference melody" in text(first)
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
    assert ("pitch 0.0/10 (100¢ off the melody · 100¢ under)" in text(result)
            and "pitch 10.0/10 (0¢ off the melody · centered)" in text(result))


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


@pytest.fixture
def online(harbor, tone, monkeypatch):
    """YouTube, through a yt-dlp that finds two videos: `searched` and `downloaded` are what it was asked."""
    harbor.searched, harbor.downloaded = [], []
    videos = [youtube.Video("https://www.youtube.com/watch?v=a1", "Harbor Lights (Official Audio)", "The Originals", 201,
                            "From the album Salt and Rain ..."),
              youtube.Video("https://www.youtube.com/watch?v=b2", "Harbor Lights (Live in Lisbon)", "A Fan", 355)]

    def download(url, folder):
        harbor.downloaded.append((url, folder))
        shutil.copy(tone, folder / "original.wav")
        return folder / "original.wav", "Harbor Lights (Official Audio)"

    monkeypatch.setattr(youtube, "available", lambda: True)
    monkeypatch.setattr(youtube, "search", lambda query: harbor.searched.append(query) or videos)
    monkeypatch.setattr(youtube, "download", download)
    harbor.melody = sing()
    return harbor


def test_the_original_is_found_on_youtube_for_a_song_without_a_reference(online):
    result = online("reference", "harbor", input="\n1\n")  # Enter: yes, look; 1: the official audio

    assert result.exit_code == 0, result.output
    assert "has no reference recording, so it's rated against its scale. Look for the original on YouTube?" \
        in text(result)
    assert ("1. Harbor Lights (Official Audio) The Originals · 3:21 · From the album Salt and Rain ... "
            "2. Harbor Lights (Live in Lisbon) A Fan · 5:55") in text(result)
    assert online.searched == ["Harbor Lights official audio"]
    [(url, folder)] = online.downloaded
    assert url == "https://www.youtube.com/watch?v=a1" and not folder.exists()  # the recording isn't kept
    assert "“Harbor Lights” is now rated against the melody of Harbor Lights (Official Audio)" in text(result)
    assert ("rated against the melody of Harbor Lights (Official Audio) (https://www.youtube.com/watch?v=a1)"
            in text(online("reference", "harbor")))


def test_nothing_is_downloaded_until_you_pick_a_video(online):
    result = online("reference", "harbor", input="y\n7\n\n")  # 7 isn't one of them; Enter cancels

    assert "Type a number from 1 to 2" in text(result) and "Nothing was changed" in text(result)
    assert online.downloaded == [] and not online.library.has_reference("Harbor Lights")


def test_a_search_replaces_a_reference_and_goes_by_the_songs_artist(online, tone):
    set_reference(online)
    add_take(online.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 2, 20, 0), LINES, genius_id=7,
             artist="The Originals")  # recognized on Genius

    result = online("reference", "harbor", "--search", input="2\n")

    assert result.exit_code == 0, result.output
    assert online.searched == ["The Originals Harbor Lights official audio"]
    assert online.library.reference("Harbor Lights").file == "https://www.youtube.com/watch?v=b2"


@pytest.fixture
def covered(online, tone):
    """Harbor Lights, as Genius once recognized it: as a cover's."""
    add_take(online.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 2, 20, 0), LINES, genius_id=9,
             artist="Some Cover Singer")
    return online


def test_you_can_look_again_with_the_real_artist_who_is_then_kept(covered):
    result = covered("reference", "harbor", "--search", input="The Originals\n1\n")

    assert result.exit_code == 0, result.output
    assert "Number to download, an artist to search for instead, or Enter to cancel" in text(result)
    assert covered.searched == ["Some Cover Singer Harbor Lights official audio",
                                "The Originals Harbor Lights official audio"]
    assert "Saved The Originals as the artist of “Harbor Lights”" in text(result)
    assert [take.artist for take in covered.library.takes_of("Harbor Lights")] == ["The Originals"] * 2
    assert covered.library.reference("Harbor Lights").file == "https://www.youtube.com/watch?v=a1"


def test_an_artist_is_only_kept_once_you_pick_a_video_it_found(covered):
    result = covered("reference", "harbor", "--search", input="The Originals\n\n")  # Enter: none of these

    assert result.exit_code == 0, result.output
    assert len(covered.searched) == 2 and covered.downloaded == [] and "Nothing was changed" in text(result)
    assert covered.library.takes_of("Harbor Lights")[-1].artist == "Some Cover Singer"


def test_you_can_look_again_with_the_arrow_keys(covered, monkeypatch):
    presses = iter(["up", "up", "enter", "enter"])  # up twice from the first: "Search again with another artist…"
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(widgets.terminal, "keys", lambda: contextlib.nullcontext(lambda: next(presses)))

    result = covered("reference", "harbor", "--search", input="The Originals\n")

    assert result.exit_code == 0, result.output
    assert covered.searched[-1] == "The Originals Harbor Lights official audio"
    assert [url for url, _ in covered.downloaded] == ["https://www.youtube.com/watch?v=a1"]
    assert covered.library.takes_of("Harbor Lights")[-1].artist == "The Originals"


def test_when_youtube_finds_nothing_you_can_name_the_artist(online, monkeypatch):
    search = youtube.search
    monkeypatch.setattr(youtube, "search", lambda query: search(query) if "The Originals" in query else [])

    result = online("reference", "harbor", "--search", input="The Originals\n1\n")

    assert result.exit_code == 0, result.output
    assert "YouTube found nothing for “Harbor Lights”. Another artist to search for, or Enter to cancel" \
        in text(result)
    assert online.library.takes_of("Harbor Lights")[0].artist == "The Originals"  # it had none
    assert online.library.has_reference("Harbor Lights")


def test_a_link_is_downloaded(online):
    result = online("reference", "https://www.youtube.com/watch?v=zz", "harbor")

    assert result.exit_code == 0, result.output
    assert online.searched == [] and [url for url, _ in online.downloaded] == ["https://www.youtube.com/watch?v=zz"]


def test_a_failed_download_is_reported(online, monkeypatch):
    def fail(url, folder):
        raise youtube.YouTubeError("[youtube] zz: Video unavailable")

    monkeypatch.setattr(youtube, "download", fail)
    result = online("reference", "harbor", "https://www.youtube.com/watch?v=zz")
    assert result.exit_code == 1 and "Couldn't get it from YouTube: [youtube] zz: Video unavailable" in text(result)


def test_without_yt_dlp_youre_told_how_to_get_it(harbor):
    assert "Or install yt-dlp (brew install yt-dlp), to find the original on YouTube" in text(harbor("reference", "harbor"))
    for args in (["--search"], ["https://www.youtube.com/watch?v=zz"]):
        result = harbor("reference", "harbor", *args)
        assert result.exit_code == 1 and "needs yt-dlp: brew install yt-dlp" in text(result)


def test_a_new_songs_first_take_says_how_to_rate_it_on_the_melody(shed, tone, tmp_path, rng):
    shed.heard = mishear(SONGS["Gravel Road"], 0.3, rng)
    first = shed("add", str(tone), input="Gravel Road\n")
    second = shed("add", str(recording(tmp_path, "memo.m4a")))

    assert "To rate your pitch against the original's melody: shed reference 'Gravel Road'" in text(first)
    assert "shed reference" not in text(second)


@pytest.mark.parametrize("keys, downloaded", [(["down", "enter"], ["https://www.youtube.com/watch?v=b2"]),
                                              (["up", "enter"], [])])  # up from the first: "None of these"
def test_the_original_is_picked_with_the_arrow_keys(online, monkeypatch, keys, downloaded):
    presses = iter(keys)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(widgets.terminal, "keys", lambda: contextlib.nullcontext(lambda: next(presses)))

    result = online("reference", "harbor", "--search")

    assert result.exit_code == 0, result.output
    assert [url for url, _ in online.downloaded] == downloaded
    assert online.library.has_reference("Harbor Lights") == bool(downloaded)


def test_any_take_is_shown_against_the_melody(harbor, tone):
    set_reference(harbor)
    add_take(harbor.library, tone, "Harbor Lights", 2, 7, datetime(2026, 7, 1, 20, 0), LINES, metrics=Metrics(20, 0.02),
             melody=sing())  # the latest: on the melody
    harbor.melody = sing(flat_line=2)  # the first take, once its melody is followed

    latest = harbor("progress", "harbor")
    first = harbor("progress", "harbor", "--take", "1")

    assert "Take 2 against the melody: No line strays 60¢ or more from the melody" in text(latest)
    assert "One take in detail: shed progress 'Harbor Lights' --take N" in text(latest)
    assert first.exit_code == 0, first.output
    assert "Harbor Lights, take 1 of 2, recorded 2026-06-01 20:00" in text(first)
    assert ("Pitch, against the melody (6 lines compared): Typically 0¢ off, either way (10/10 at 15¢, "
            "0/10 at 100¢) Typically centered: 1 line sits under the melody, 0 over") in text(first)
    assert f"Furthest from the melody: 0:10 “{LINES[2]}” 100¢ under" in text(first)


@pytest.mark.parametrize("keys, typed, kept", [
    (["enter", "enter"], "nope\nharbor\n", True),  # "Another of your songs…", twice: no song called "nope"
    (["down", "enter"], "", False),  # "Cancel"
])
def test_you_pick_which_of_your_songs_it_is_with_the_arrow_keys(harbor, arrow_keys, keys, typed, kept):
    harbor.melody = verse_of("Winter Town")  # not a song you have: it isn't recognized
    arrow_keys(*keys)

    result = harbor("reference", str(harbor.original), input=typed)

    assert result.exit_code == 0, result.output
    assert "Number" not in result.output and harbor.library.has_reference("Harbor Lights") == kept
    assert ("You have no song called “nope”" in text(result)) == kept


def test_a_take_is_told_its_lines_closest_to_the_melody_too(harbor, tmp_path, rng):
    set_reference(harbor)
    harbor.heard = mishear(SONGS["Harbor Lights"], 0.2, rng)
    harbor.melody = lines_off({0: 30, 1: 12, 2: -5, 3: -100, 4: 40, 5: 18})
    at = {n: cli._clock(next(w[1] for w in harbor.melody.words if w[3] == n)) for n in range(6)}

    result = harbor("add", str(recording(tmp_path, "memo.m4a")))

    assert result.exit_code == 0, result.output
    assert (f"Furthest from the melody: {at[3]} “{LINES[3]}” 100¢ under Closest to the melody: {at[2]} “{LINES[2]}” "
            f"5¢ under {at[1]} “{LINES[1]}” 12¢ over {at[5]} “{LINES[5]}” 18¢ over") in text(result)


@pytest.fixture
def practiced(harbor, tone):
    """Harbor Lights with a reference, and four takes: the first (filed before Woodshed kept melodies) sings three
    lines on the melody; the other three sing the third line flat, and one of them the fifth line sharp."""
    set_reference(harbor)
    for day, sung in ((2, lines_off({2: -100, 4: 80})), (3, lines_off({2: -90})), (4, lines_off({2: -110}))):
        add_take(harbor.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, day, 20, 0), LINES,
                 metrics=Metrics(20, 0.02), melody=sung)
    harbor.melody = sing(order=[0, 1, 3])
    return harbor


def test_progress_tells_the_lines_off_take_after_take_and_those_on_the_melody(practiced):
    result = practiced("progress", "harbor")

    assert result.exit_code == 0, result.output
    assert (f"Line by line, your last 4 takes compared with the melody: Often off it: “{LINES[2]}” typically 100¢ "
            f"under · under in 3 of 3 takes Closest to it: “{LINES[0]}” typically 0¢ off · within 20¢ in 4 of 4 takes "
            f"“{LINES[1]}” typically 0¢ off · within 20¢ in 4 of 4 takes “{LINES[3]}” typically 0¢ off") in text(result)
    assert LINES[4] not in text(result).split("Line by line")[1]  # off once: no pattern
    assert "Every line: shed progress 'Harbor Lights' --lines" in text(result)


def test_lines_are_judged_on_the_latest_takes(practiced, monkeypatch):
    monkeypatch.setattr(melody, "RECENT_TAKES", 3)
    result = practiced("progress", "harbor")
    assert "Line by line, your last 3 takes compared with the melody" in text(result)
    assert f"“{LINES[0]}” typically 0¢ off · within 20¢ in 3 of 3 takes" in text(result)  # take 1 left out


def test_every_line_in_detail(practiced):
    result = practiced("progress", "harbor", "--lines")

    assert result.exit_code == 0, result.output
    assert "Harbor Lights, line by line: your last 4 takes compared with the melody" in text(result)
    cells = {line: [cell.strip() for cell in row.split("│")[2:5]]  # each line's row, as the song goes
             for row in result.output.splitlines() for line in LINES if row.startswith(f"│ “{line[:20]}")}
    assert list(cells) == LINES
    assert cells[LINES[2]] == ["100¢ off · 100¢ under", "3", "·▁▁▁"]  # take 1 didn't sing it
    assert cells[LINES[4]] == ["0¢ off · centered", "3", "·▂██"]
    assert cells[LINES[0]] == ["0¢ off · centered", "4", "████"]


def test_lines_need_a_reference(harbor):
    result = harbor("progress", "harbor", "--lines")
    assert result.exit_code == 1 and "“Harbor Lights” has no reference recording, so there's no melody to follow " \
                                      "its lines against. Give it one: shed reference 'Harbor Lights'" in text(result)
    both = harbor("progress", "harbor", "--lines", "--take", "1")
    assert both.exit_code == 1 and "Choose one of --take and --lines" in text(both)
