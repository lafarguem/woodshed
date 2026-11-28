"""Replaying a take over its song's melody (`shed replay`): laying the melody along takes made up note by note,
and the page that shows it."""

import http.client
import json
import random
import threading
import webbrowser
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import add_take
from test_melody import LINES, sing

from woodshed import replay
from woodshed.replay import Followed

HOP = 0.02


def tune(seed: int, notes: int = 16) -> list[int]:
    """A melody of its own: `notes` notes, no two in a row the same."""
    rng, found = random.Random(seed), [0]
    while len(found) < notes:
        found.append(rng.choice([n for n in (-7, -5, -3, -2, 0, 2, 4, 5, 7) if n != found[-1]]))
    return found


def sung(*parts, pace: float = 1.0, up: float = 0.0) -> list[float | None]:
    """A singer's pitch every HOP: each part is a tune (a list of notes, each held 0.4 s of every 0.5 s) or a rest
    (how many seconds), sung at `pace` times the tempo, `up` semitones up."""
    pitch: list[float | None] = []
    for part in parts:
        if isinstance(part, (int, float)):
            pitch += [None] * round(part / pace / HOP)
            continue
        for note in part:
            pitch += [note + up] * round(0.4 / pace / HOP) + [None] * round(0.1 / pace / HOP)
    return pitch


def laid_on(followed: Followed, start: float, end: float) -> np.ndarray:
    """(seconds into the take, seconds into the reference) of the moments of the take from `start` to `end`."""
    knots = np.array([knot for stretch in followed.stretches for knot in stretch])
    return knots[(knots[:, 0] >= start) & (knots[:, 0] <= end)]


VERSE, CHORUS = tune(1), tune(2)
REFERENCE = sung(4, VERSE, 2, CHORUS, 3)  # the verse from 4 s to 12, the chorus from 14 to 22


def test_a_take_is_followed_at_its_own_tempo_in_its_own_key_after_its_own_intro():
    take = sung(9, VERSE, 2, CHORUS, pace=0.8, up=-12 + 3.3)  # slower, an octave down and a little over 3 semitones up

    followed = replay.follow(take, REFERENCE)

    assert followed.moved == pytest.approx(3.3, abs=0.05)
    for start, there in ((9 / 0.8, 4), (19 / 0.8, 14)):  # the verse, then the chorus
        knots = laid_on(followed, start + 0.5, start + 9.5)
        assert len(knots) > 200 and np.abs(knots[:, 1] - (there + (knots[:, 0] - start) * 0.8)).max() < 0.15


def test_a_part_left_out_and_a_part_sung_again_are_followed():
    reference = sung(4, VERSE, 12, CHORUS, 3)  # a long solo before the chorus, from 24 s
    take = sung(1, VERSE, 1, CHORUS, 1, CHORUS)  # no solo, and the chorus twice: from 10 s and from 19

    followed = replay.follow(take, reference)

    assert np.abs(laid_on(followed, 1.5, 8.5)[:, 1] - laid_on(followed, 1.5, 8.5)[:, 0] - 3).max() < 0.15
    for start in (10, 19):
        knots = laid_on(followed, start + 0.5, start + 7.5)
        assert len(knots) > 150 and np.abs(knots[:, 1] - (24 + knots[:, 0] - start)).max() < 0.15
    assert len(followed.stretches) >= 3  # it left the reference twice


def test_the_reference_waits_through_a_break_of_the_takes_own():
    take = sung(2, VERSE, 20, CHORUS)  # the chorus from 30 s: 18 s later than the reference comes to it

    followed = replay.follow(take, REFERENCE)

    assert len(laid_on(followed, 14, 26)) == 0  # nothing of the melody is drawn meanwhile
    knots = laid_on(followed, 30.5, 37.5)
    assert np.abs(knots[:, 1] - (14 + knots[:, 0] - 30)).max() < 0.15


def words_of(names: list[str], start: float) -> list[tuple[str, float, float, int]]:
    """A word on each note of a tune, as sung(): 0.4 s of every 0.5 s."""
    return [(name, start + 0.5 * i, start + 0.5 * i + 0.4, 0) for i, name in enumerate(names)]


FIRST = "the lanterns swing above the harbor wall we counted ships until the evening fall and".split()
SECOND = "your coat was wet with salt and rain and every bell was ringing out your name so".split()


def test_the_words_tell_which_verse_a_take_sings():
    reference = sung(2, VERSE, 2, VERSE, 2)  # two verses to the same tune: from 2 s, and from 12
    heard = words_of(FIRST, 2) + words_of(SECOND, 12)
    take = sung(1, VERSE, 1)
    misheard = [("yeah" if i in (3, 9) else word, start, end, 0) for i, (word, start, end, _) in
                enumerate(words_of(SECOND, 1))]

    words = replay.anchors(misheard, heard)
    knots = laid_on(replay.follow(take, reference, words=words), 1.5, 8.5)

    assert np.abs(knots[:, 1] - (12 + knots[:, 0] - 1)).max() < 0.15
    assert np.abs(laid_on(replay.follow(take, reference), 1.5, 8.5)[:, 1] - 2).min() < 8  # the tune alone: either


def test_words_count_where_three_in_a_row_were_heard_in_both_each_time_they_are_sung():
    reference = words_of("fall back into place".split(), 10) + words_of("fall back into place".split(), 20)
    take = words_of("oh fall bak into the place".split(), 1)  # "bak" is "back", misheard

    found = replay.anchors(take, reference)

    assert sorted((round(mine, 1), round(theirs, 1)) for mine, _, theirs, _ in found) == [
        (1.5, 10.0), (1.5, 20.0), (2.0, 10.5), (2.0, 20.5), (2.5, 11.0), (2.5, 21.0)]  # not "place", after "the"
    assert replay.anchors(take, []) == [] and replay.anchors([], reference) == []


def test_a_take_with_no_singing_is_laid_nowhere_without_failing():
    followed = replay.follow([None] * 500, REFERENCE)
    assert isinstance(followed.moved, float)
    assert replay.follow(sung(1, VERSE), [None] * 500).stretches is not None


def test_the_melody_is_drawn_as_the_take_goes_in_the_octave_of_its_voice():
    # The take sings the reference's 10th to 12th second from its 1st, then again from its 5th, an octave down.
    followed = Followed([[(1.0 + 0.04 * i, 10.0 + 0.04 * i) for i in range(51)],
                         [(5.0 + 0.04 * i, 10.0 + 0.04 * i) for i in range(51)]], 2.0)
    contour = [None] * 500 + [7.0] * 50 + [9.0] * 50 + [None] * 100  # a G from 10 s, an A from 11
    notes = [[9.0, 10.5, 7.0], [11.0, 13.0, 9.0], [20.0, 21.0, 4.0]]
    lines = [{"text": "Fall back", "start": 10.2, "end": 11.8,
              "words": [["fall", 10.2, 10.6], ["back", 11.0, 11.8, [[2, 11.0], [4, 11.4]]]]},
             {"text": "into place", "start": 13.0, "end": 14.0, "words": [["into", 13.0, 13.4], ["place", 13.4, 14.0]]}]
    mine = [None] * 50 + [-3.0] * 100 + [None] * 250

    drawn, placed, shown = replay.laid(followed, mine, contour, notes, lines, 2.0)

    assert len(drawn) == len(mine)
    assert drawn[75] == drawn[275] == 7 + 2 - 12 and drawn[125] == drawn[325] == 9 + 2 - 12  # nearest the voice
    assert drawn[20] is None and drawn[200] is None  # nothing where the take sings none of it
    assert placed == [[1.0, 1.5, -3.0], [2.0, 3.0, -1.0], [5.0, 5.5, -3.0], [6.0, 7.0, -1.0]]  # cut where it's left
    assert [(line["text"], line["start"], line["end"]) for line in shown] == [("Fall back", 1.2, 2.8),
                                                                                ("Fall back", 5.2, 6.8)]
    assert shown[1]["words"] == [["fall", 5.2, 5.6], ["back", 6.0, 6.8, [[2, 6.0], [4, 6.4]]]]
    assert replay.laid(followed, mine, contour, None, [], 2.0)[1] is None  # no tab


@pytest.mark.parametrize("moved, key, tuned, drawn, lean, apart", [
    (0.1, 0, 0.0, 0, 0.1, 0),  # in the instrument's key, a little over the melody
    (1.7, 2, 0.2, 2.2, -0.5, 0),  # a capo on the second fret, tuned a little sharp: the voice half a semitone under
    (10.9, 0, 0.0, 0, -1.1, 0),  # a semitone under the guitar all through: still drawn where the guitar plays it
    (5.2, 0, 0.0, 5, 0.2, 5),  # a fourth over the melody as the guitar plays it: drawn where it's sung
    (7.1, 2, 0.0, 7, 0.1, 5),
    (7.4, None, 0.0, 7, 0.4, 0),  # no chords to go by: the key the voice sings in
])
def test_the_melody_is_drawn_in_the_instruments_key_unless_the_voice_is_nowhere_near(moved, key, tuned, drawn, lean,
                                                                                     apart):
    assert replay.drawn_in(moved, key, tuned) == (pytest.approx(drawn), pytest.approx(lean), apart)


def test_an_instruments_tuning_is_read_from_its_notes_and_the_pitch_they_were_heard_at():
    contour = [None] * 50 + [3.2] * 50 + [None] * 50 + [-4.8] * 50
    assert replay.tuning([(1.0, 2.0, 3.0), (3.0, 4.0, -5.0)], contour) == pytest.approx(0.2)
    assert replay.tuning([], contour) == 0 and replay.tuning([(0.0, 0.5, 3.0)], contour) == 0  # none heard


def painted(melody, up: float = 0.0, seconds: float = 30.0) -> list[float | None]:
    """The pitch all through a take made up by test_melody.sing(): each of its notes held 0.4 s."""
    pitch: list[float | None] = [None] * round(seconds / HOP)
    for start, _, note in melody.notes:
        pitch[round(start / HOP):round((start + 0.4) / HOP)] = [note + up] * round(0.4 / HOP)
    return pitch


@pytest.fixture
def replayed(shed, tone, monkeypatch):
    """Harbor Lights with a reference and a take, played 2 semitones up; `shed replay` shows `.shown` at `.opened`
    until Ctrl+C (at once)."""
    hum = np.zeros(8000, np.float32)
    path = add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), LINES)
    shed.library.save_melody(path, sing(shift=2))
    shed.library.set_reference("Harbor Lights", sing(), "original.mp3", contour=painted(sing()),
                               stems=SimpleNamespace(accompaniment=hum, vocals=hum, rate=8000))
    seen = SimpleNamespace(shown=None, opened=[], sung=painted(sing(shift=2), up=-12), path=path)

    class Server:
        server_port = 4321

        def __init__(self, shown, port=0):
            seen.shown, seen.port = shown, port

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen.closed = True

    monkeypatch.setattr(replay, "Server", Server)
    monkeypatch.setattr(replay, "sung", lambda path, device=None: seen.sung)
    monkeypatch.setattr(webbrowser, "open", seen.opened.append)
    return seen


def test_shed_replay_lays_the_melody_along_the_latest_take_and_opens_its_page(shed, replayed):
    result = shed("replay", "harbor")

    output = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "▶ Harbor Lights, take 1 of 1, recorded 2026-06-01 20:00, over its melody: http://127.0.0.1:4321/" in output
    assert "Your instrument plays 2 semitones over the reference, so the melody was moved to match." in output
    assert replayed.opened == ["http://127.0.0.1:4321/"] and replayed.closed
    shown = replayed.shown.shown()
    assert shown["title"] == "Harbor Lights" and shown["take"] == "take 1 of 1, recorded 2026-06-01 20:00"
    assert [line["text"] for line in shown["lines"]] == LINES and shown["notes"] is None
    assert shown["sung"] == {"hop": HOP, "pitch": replayed.sung} and not shown["sits"]
    both = [(mine, theirs) for mine, theirs in zip(replayed.sung, shown["contour"]["pitch"]) if mine is not None]
    assert sum(theirs == mine for mine, theirs in both) > 0.95 * len(both)  # drawn where it was sung
    assert replayed.shown.audio[:3] in (b"ID3", b"\xff\xfb\x90") or replayed.shown.audio[0] == 0xFF  # an mp3


def test_shed_replay_says_where_the_voice_sits_and_when_it_sings_in_another_key(shed, replayed):
    def sings(take):
        shed.library.save_melody(replayed.path, take)
        replayed.sung = painted(take)

    sings(sing(shift=2, off=-0.4))
    assert "Your voice sits about 40¢ under the melody all through: the page can move the melody there" in " ".join(
        shed("replay", "harbor").output.split())
    assert replayed.shown.shown()["sits"] and replayed.shown.lean == pytest.approx(-0.4, abs=0.02)

    sings(sing(shift=2, voice_shift=7))  # the chords say 2 semitones up, the voice sings 7 up
    output = " ".join(shed("replay", "harbor", "--take", "1", "--port", "9000").output.split())
    assert "Your instrument plays 2 semitones over the reference, so the melody was moved to match." in output
    assert ("All through, you sing the melody about 5 semitones over where your instrument plays it: it's drawn where "
            "you sing it.") in output
    assert replayed.port == 9000 and (replayed.shown.key, replayed.shown.apart) == (2, 5)


def test_shed_replay_needs_a_take_and_a_reference_that_keeps_the_melody_all_through(shed, replayed, tone):
    assert "“Harbor Lights” has 1 take: --take goes from 1 to 1." in shed("replay", "harbor", "-t", "3").output
    shed.library.set_reference("Harbor Lights", sing(), "original.mp3")  # as before v0.9
    result = shed("replay", "harbor")
    assert result.exit_code == 1 and "Set it again: shed reference 'Harbor Lights' --search" in " ".join(
        result.output.split())
    add_take(shed.library, tone, "Winter Town", 2, 7, datetime(2026, 6, 1, 20, 0), LINES)
    result = shed("replay", "winter")
    assert result.exit_code == 1 and "“Winter Town” has no reference melody to replay a take over." in result.output
    assert replayed.opened == []


@pytest.fixture
def page():
    """`shed replay`'s page, on a free port."""
    shown = replay.Replay("Harbor Lights", "take 2 of 3", [None, -3.0], [None, -3.5], [[0.0, 0.5, -3.0]],
                          [{"text": "Fall back", "start": 0.0, "end": 0.5, "words": [["fall", 0.0, 0.5]]}], 2, 0,
                          -0.5, b"the take")
    running = replay.Server(shown)
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield running.server_port
    running.shutdown()
    running.server_close()


def get(port: int, path: str, host: str | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("GET", path, headers={"Host": host} if host else {})
    response = connection.getresponse()
    return response.status, response.read()


def test_the_page_gets_the_take_and_what_to_draw_over_it(page):
    status, body = get(page, "/")
    assert status == 200 and b'<body data-only="replay">' in body and b'id="tab-replay"' in body  # shed serve's page
    status, body = get(page, "/replay/take")
    assert status == 200 and json.loads(body) == {
        "title": "Harbor Lights", "take": "take 2 of 3", "lean": -0.5, "sits": True, "notes": [[0.0, 0.5, -3.0]],
        "lines": [{"text": "Fall back", "start": 0.0, "end": 0.5, "words": [["fall", 0.0, 0.5]]}],
        "contour": {"hop": HOP, "pitch": [None, -3.5]}, "sung": {"hop": HOP, "pitch": [None, -3.0]}}
    assert get(page, "/replay/audio") == (200, b"the take")
    assert get(page, "/takes")[0] == 404  # nothing else of shed serve's
    assert get(page, "/replay/take", host="woodshed.example")[0] == 404  # a site whose name leads here


def test_takes_replayed_from_the_phone_are_laid_once_and_the_latest_few_kept():
    laid = []
    session = replay.Session(lambda: [{"title": "Harbor Lights", "takes": ["2026-06-01 20:00"]}],
                             lambda title, number: laid.append(number) or (title, number))

    assert session.songs() == [{"title": "Harbor Lights", "takes": ["2026-06-01 20:00"]}]
    assert session.take("Harbor Lights", 1) == session.take("Harbor Lights", 1) == ("Harbor Lights", 1)
    for number in (2, 3, 4, 1):
        session.take("Harbor Lights", number)
    assert laid == [1, 2, 3, 4, 1]  # the first one was let go, three takes on


def test_shed_serve_replays_any_take_of_a_song_whose_reference_keeps_its_melody(shed, replayed, monkeypatch, tone):
    from test_phone import FakeServer

    from woodshed import phone

    served, devices = [], []
    monkeypatch.setattr(phone, "Server", lambda *args, **kwargs: served.append(kwargs["replay"]) or FakeServer(*args))
    monkeypatch.setattr(replay, "sung", lambda path, device=None: devices.append(device) or replayed.sung)
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 2, 21, 0), LINES)  # filed without its words
    add_take(shed.library, tone, "Winter Town", 2, 7, datetime(2026, 6, 1, 20, 0), LINES)  # no reference

    result = shed("serve")

    assert "Replay, on the page, plays one of your takes of those songs" in " ".join(result.output.split())
    assert served[0].songs() == [{"title": "Harbor Lights", "takes": ["2026-06-01 20:00", "2026-06-02 21:00"]}]
    shown = served[0].take("Harbor Lights", 1)
    assert shown.take == "take 1 of 2, recorded 2026-06-01 20:00" and shown.key == 2 and devices == ["cpu"]
    assert served[0].take("Harbor Lights", 2).key == 0 and shed.transcribed == []  # laid by its melody alone
    for title, number in (("Harbor Lights", 3), ("Harbor Lights", 0), ("Winter Town", 1), ("Gravel Road", 1)):
        with pytest.raises(ValueError):
            served[0].take(title, number)


def test_a_take_kept_as_apple_lossless_is_played_as_an_mp3(tone, tmp_path):
    from woodshed import audio

    audio.export(tone, tmp_path / "take.m4a", 0, 3)
    audio.export(tone, tmp_path / "take.mp3", 0, 3)
    played = replay.listenable(tmp_path / "take.m4a")
    (tmp_path / "played.mp3").write_bytes(played)
    assert len(audio.load(tmp_path / "played.mp3")) == pytest.approx(3 * audio.SAMPLE_RATE, rel=0.05)
    assert replay.listenable(tmp_path / "take.mp3") == (tmp_path / "take.mp3").read_bytes()
