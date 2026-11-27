"""Singing along to a song's reference recording (along.py): its melody as it's kept, and your pitch read as the
microphone's sound comes. The pitch tracker is made up: it hears the note `voice` says, as sure as it says."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_melody import LINES, sing

from woodshed import along, rating, rmvpe


@pytest.fixture
def voice(monkeypatch):
    """What the pitch tracker hears: `note` (semitones from A440), `sure` of it, in windows whose lengths are
    kept in `heard`."""
    voice = SimpleNamespace(note=0.0, sure=0.9, heard=[])

    def pitch(samples, device=None):
        voice.heard.append(len(samples))
        frames = len(samples) // 160 + 1
        return np.full(frames, 440 * 2 ** (voice.note / 12)), np.full(frames, voice.sure)

    monkeypatch.setattr(rmvpe, "pitch", pitch)
    return voice


def hum(seconds, rate=48_000):
    return np.full(int(seconds * rate), 0.1, np.float32)


def test_the_melody_is_kept_where_its_sung():
    track = rating.Track(np.array([220.0] * 4 + [440.0] * 4 + [100.0] * 4), np.array([0.9] * 8 + [0.1] * 4), 0.0, 0.01)
    assert along.contour(track) == [-12.0, -12.0, 0.0, 0.0, None, None]


def test_each_moment_is_read_once_a_little_after_its_sung(voice):
    live = along.Live(48_000)
    assert live.feed(hum(0.1)) == []  # too little heard yet
    read = live.feed(hum(0.2))
    assert [moment for moment, _, _ in read] == pytest.approx(np.arange(0, 0.19, 0.01))  # up to 0.12 s ago
    assert all(pitch == 0.0 and sure == 0.9 for _, pitch, sure in read)
    voice.note = 3.0
    later = live.feed(hum(0.06))
    assert [moment for moment, _, _ in later] == pytest.approx(np.arange(0.19, 0.245, 0.01))
    assert later[0][1] == 3.0


def test_only_the_latest_half_second_is_listened_to(voice):
    live = along.Live(48_000)
    for _ in range(20):
        live.feed(hum(0.06))
    assert max(voice.heard) <= 0.52 * 16_000 + 1


def test_a_voice_its_unsure_of_is_told_apart_and_none_isnt_read(voice):
    live = along.Live(44_100)
    voice.sure = 0.3
    assert {(pitch, sure) for _, pitch, sure in live.feed(hum(0.3, 44_100))} == {(0.0, 0.3)}
    assert along.FAINT <= 0.3 < along.SURE
    voice.sure = 0.1
    assert {pitch for _, pitch, _ in live.feed(hum(0.3, 44_100))} == {None}


def test_a_silent_microphone_isnt_given_to_the_tracker(voice):
    live = along.Live(48_000)
    assert {pitch for _, pitch, _ in live.feed(np.zeros(24_000, np.float32))} == {None}
    assert voice.heard == []


def test_the_lines_come_with_when_theyre_sung():
    lines = along.lines(sing())
    assert [line["text"] for line in lines] == LINES
    first = lines[0]
    assert first["start"] == first["words"][0][1] and first["end"] == first["words"][-1][2]
    assert all(a["start"] < b["start"] for a, b in zip(lines, lines[1:]))


def test_the_melody_moves_to_the_octave_your_voice_sits_in():
    assert along.octaves([0.0, 2.0, None, 4.0], register=-11.0) == -1
    assert along.octaves([0.0, 2.0, None, 4.0], register=3.0) == 0
    assert along.octaves([0.0, 2.0], register=None) == 0
    assert along.octaves([None], register=5.0) == 0


def test_a_session_gives_the_page_a_song_and_reads_what_it_hears(voice, tmp_path):
    band, singer = tmp_path / "band.mp3", tmp_path / "voice.mp3"
    band.write_bytes(b"band")
    singer.write_bytes(b"voice")
    started = []
    session = along.Session(lambda title: along.Song(title, along.lines(sing()), [0.0, None], -1, band, singer),
                            ["Harbor Lights"], started=started.append)
    song = session.song("Harbor Lights")
    assert song["octaves"] == -1 and song["contour"] == {"hop": 0.02, "pitch": [0.0, None]} and len(song["lines"]) == 6
    assert session.audio("Harbor Lights", "band") == b"band" and session.audio("Harbor Lights", "voice") == b"voice"
    for wrong in (lambda: session.song("Winter Town"), lambda: session.audio("Harbor Lights", "drums"),
                  lambda: session.start("Winter Town", 48_000)):
        with pytest.raises(ValueError):
            wrong()
    assert session.hear(hum(0.3)) is None  # not started

    session.start("Harbor Lights", 48_000)
    assert started == ["Harbor Lights"]
    assert session.hear(hum(0.3))[0] == [0.0, 0.0, 0.9]  # from the start: getting ready isn't counted
    session.stop()
    assert session.hear(hum(0.3)) is None


def written_like(hear):
    """LINES as written, timed in a recording where Whisper heard each word as `hear` says."""
    return along.written(LINES, sing(hear=hear))


def test_written_lyrics_take_the_times_their_words_were_heard_at():
    misheard = {"harbor": "arbor", "lights": "light"}
    lines = written_like(lambda w: misheard.get(w, w))
    assert [line["text"] for line in lines] == LINES  # as written, not as heard
    heard = along.lines(sing())
    assert [[w[1:] for w in line["words"]] for line in lines] == [[w[1:] for w in line["words"]] for line in heard]
    assert [(line["start"], line["end"]) for line in lines] == [(line["start"], line["end"]) for line in heard]


def test_a_written_word_nobody_heard_is_put_next_to_those_heard():
    whole = along.lines(sing())
    first, last = whole[0]["words"][0][0], whole[1]["words"][-1][0]
    count = {}

    def hear(word):  # the first line's first word and the second line's last aren't heard
        count[word] = count.get(word, 0) + 1
        return None if (word == first and count[word] == 1) or (word == last and count[word] == 1) else word

    lines = written_like(hear)
    assert [line["text"] for line in lines] == LINES
    start, then = lines[0]["words"][:2]
    assert start[0] == first and start[2] == then[1] and 0 < then[1] - start[1] <= 0.41  # right before the next
    before, end = lines[1]["words"][-2:]
    assert end[0] == last and end[1] == before[2] and end[2] <= lines[2]["start"]  # right after the one before
    for a, b in zip(lines, lines[1:]):
        assert a["end"] <= b["start"]


def test_a_written_line_nobody_heard_a_word_of_is_kept_only_where_theres_time_to_sing_it():
    line_of = [n for n, line in enumerate(LINES) for _ in along.lines(sing())[n]["words"]]
    at = iter(line_of)
    lines = written_like(lambda w: None if next(at) == 2 else w)  # the third line, between two heard: 4 s for it
    assert [line["text"] for line in lines] == LINES
    assert lines[1]["end"] <= lines[2]["start"] < lines[2]["end"] <= lines[3]["start"]
    at = iter(line_of)
    lines = written_like(lambda w: None if next(at) == 0 else w)  # the first: nothing says when it starts
    assert [line["text"] for line in lines] == LINES[1:]


def test_lyrics_of_another_song_arent_timed():
    assert along.written(["Down the winter town we go", "Nobody knows the road we chose", "The harbor is far"] * 4,
                         sing()) is None
    assert along.written([], sing()) is None


def test_a_written_line_misheard_all_through_is_shown_when_the_words_heard_in_its_place_were_sung():
    line_of = [n for n, line in enumerate(LINES) for _ in along.lines(sing())[n]["words"]]
    at = iter(line_of)
    misheard = sing(hear=lambda w: "yerps" if next(at) == 2 else w)
    lines = along.written(LINES, misheard)
    assert [line["text"] for line in lines] == LINES
    heard = along.lines(misheard)[2]
    assert (lines[2]["start"], lines[2]["end"]) == (heard["start"], heard["end"])
    # …but not a long line on a word or two heard
    wordy = LINES[:2] + [" ".join(["woah"] * 3 * len(heard["words"]))] + LINES[3:]
    assert [line["text"] for line in along.written(wordy, misheard)] == wordy[:2] + wordy[3:]
