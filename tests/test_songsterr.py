"""A tab from Songsterr (songsterr.py): what the bookmark sends, downloading its tracks, reading them, and `shed
reference --songsterr` waiting for one."""

import gzip
import http.client
import io
import json
import threading
import urllib.error
from urllib.parse import unquote

import pytest
from test_reference import harbor, set_reference, text  # noqa: F401 (a fixture)
from test_tab import heard

from woodshed import songsterr, tab

GUITAR = [64, 59, 55, 50, 45, 40]  # the highest string first, as Songsterr lists them
SENT = {"songId": 27, "revisionId": 5184162, "image": "v5-2-1-GEwclkOmjsFXZOxQ", "title": "Harbor Lights",
        "artist": "The Tides", "url": "https://www.songsterr.com/a/wsa/the-tides-harbor-lights-tab-s27",
        "tracks": [{"partId": 0, "instrumentId": 66, "name": "Singer", "tuning": GUITAR},
                   {"partId": 1, "instrumentId": 1024, "name": "Drums", "isDrums": True}]}


def note(pitch, string=0, **extra):
    """A note on one of GUITAR's strings."""
    return {"string": string, "fret": pitch - GUITAR[string], **extra}


def beat(fraction, *notes, **extra):
    """A beat lasting `fraction` of a whole note; no notes is a rest."""
    return {"duration": list(fraction), "notes": list(notes) or [{"rest": True}], **({"rest": True} if not notes else {}),
            **extra}


def track(bars, name="Singer", tempo=120, **measures):
    """A track as Songsterr writes it: `bars` are lists of beats; `measures` gives fields of bars, by number."""
    written = [{"voices": [{"beats": beats}], **measures.get(f"m{k}", {})} for k, beats in enumerate(bars)]
    written[0].setdefault("signature", [4, 4])
    return {"name": name, "tuning": GUITAR, "measures": written,
            "automations": {"tempo": [{"measure": 0, "position": 0, "bpm": tempo, "type": 4}]}}


def read(*payloads):
    return tab.timed(songsterr.score([(songsterr.Track(k, "", GUITAR, False), p) for k, p in enumerate(payloads)]))


def test_beats_are_timed_and_notes_read_off_the_strings():
    bars = [[beat((1, 4), note(60, 1)), beat((1, 8)), beat((1, 8), note(62, 1)), beat((1, 8), note(64)),
             beat((3, 8), note(65), note(57, 3))],  # a dotted quarter; of two notes at once, the higher
            [beat((1, 1), note(67, dead=True))],  # a muted string has no note
            [beat((1, 12), note(67)), beat((1, 32), note(70), graceNote="beforeBeat"), beat((1, 12), note(67)),
             beat((1, 12), note(69)), beat((3, 4), note(69, tie=True))]]  # triplets; the last carries on the one before
    [voice] = read(track(bars))
    assert [(round(a, 3), round(b, 3), p) for a, b, p in voice.notes] == [
        (0.0, 0.5, 60), (0.75, 1.0, 62), (1.0, 1.25, 64), (1.25, 2.0, 65),
        (4.0, 4.167, 67), (4.167, 4.333, 67), (4.333, 6.0, 69)]


def test_repeats_play_their_alternate_endings_and_tempo_changes_in_a_bar():
    bars = [[beat((1, 1), note(n))] for n in (60, 62, 64, 65, 67)]
    payload = track(bars, m0={"repeatStart": True}, m1={"alternateEnding": [1, 2], "repeat": 3},
                    m2={"alternateEnding": [3]})
    [voice] = read(payload)
    # Three times through the section: bar 1 (with the repeat sign) on the first two, bar 2 on the third.
    assert [p for _, _, p in voice.notes] == [60, 62, 60, 62, 60, 64, 65, 67]

    payload = track([[beat((1, 2), note(60)), beat((1, 2), note(62))]] * 2, tempo=60)
    payload["automations"]["tempo"].append({"measure": 1, "position": 2, "bpm": 120, "type": 4})
    [voice] = read(payload)
    assert [(a, b) for a, b, _ in voice.notes] == [(0, 2), (2, 4), (4, 6), (6, 7)]  # twice as fast halfway in bar 1


def test_the_bars_are_laid_out_by_the_longest_track():
    short = track([[beat((1, 1), note(60))]], name="Intro")
    long = track([[beat((1, 1), note(62))], [beat((1, 2), note(62))], [beat((1, 2), note(64))]], name="Singer",
                 m1={"signature": [2, 4]})
    intro, singer = read(short, long)
    assert intro.notes == [(0, 2, 60)] and [(a, b) for a, b, _ in singer.notes] == [(0, 2), (2, 3), (3, 4)]


def test_what_the_bookmark_sends_is_checked():
    sent = songsterr.song(SENT)
    assert (sent.id, sent.revision, sent.title, sent.url) == (27, 5184162, "Harbor Lights", SENT["url"])
    assert [(t.part, t.drums, t.tuning) for t in sent.tracks] == [(0, False, GUITAR), (1, True, None)]
    assert songsterr.song({**SENT, "url": "https://example.com/"}).url is None
    for wrong in ({**SENT, "image": "../../evil"}, {**SENT, "image": "a/b"}, {**SENT, "songId": "27"},
                  {**SENT, "revisionId": True}, {**SENT, "songId": 0}, {**SENT, "tracks": []},
                  {**SENT, "tracks": [{"partId": -1}]}, [], None):
        with pytest.raises(tab.TabError, match="isn't a tab"):
            songsterr.song(wrong)


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(gzip.compress(json.dumps(payload).encode()))
        self.headers = {"Content-Encoding": "gzip"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_tracks_are_downloaded_but_the_drums_from_another_cdn_if_need_be(monkeypatch):
    asked = []

    def urlopen(request, timeout):
        asked.append(request.full_url)
        if request.full_url.startswith(songsterr.CDNS[0]):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
        return Response(track([[beat((1, 1), note(60))]]))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    [(got, payload)] = songsterr.download(songsterr.song(SENT))
    assert got.name == "Singer" and payload["name"] == "Singer"
    assert asked == [f"{cdn}/27/5184162/v5-2-1-GEwclkOmjsFXZOxQ/0.json" for cdn in songsterr.CDNS]

    def unreachable(request, timeout):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    with pytest.raises(tab.TabError, match="couldn't be downloaded from Songsterr .no network."):
        songsterr.download(songsterr.song(SENT))
    with pytest.raises(tab.TabError, match="only drums"):
        songsterr.download(songsterr.song({**SENT, "tracks": SENT["tracks"][1:]}))


def test_the_bookmark_opens_woodshed_with_the_tab():
    link = songsterr.bookmarklet(8766)
    assert link.startswith("javascript:") and "http://127.0.0.1:8766/" in unquote(link)
    page = songsterr.page("Harbor <Lights>", "The Tides Harbor Lights")
    assert "Harbor &lt;Lights&gt;" in page and "?pattern=The%20Tides%20Harbor%20Lights" in page


def test_the_page_follows_the_tabs_sent_until_one_fits():
    answers = iter([(False, "None of its tracks follows the melody"), (True, "✓ Done")])
    reading, go_on, received = threading.Event(), threading.Event(), []

    def receive(sent):
        received.append(sent)
        reading.set()
        go_on.wait(5)
        return next(answers)

    server = songsterr.Waiting(0, "<p>the page</p>", receive)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def ask(method, path, body=None, host=f"127.0.0.1:{port}", kind="application/json"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(method, path, body, {"Host": host, "Content-Type": kind})
        response = connection.getresponse()
        return response.status, response.read().decode()

    def status():
        return json.loads(ask("GET", "/status")[1])

    def send():
        answered.append(json.loads(ask("POST", "/tab", json.dumps(SENT))[1]))

    try:
        assert ask("GET", "/") == (200, "<p>the page</p>")
        assert ask("GET", "/", host="songsterr.evil.example")[0] == 404  # a name made to lead here
        assert ask("POST", "/tab", "{}", kind="text/plain")[0] == 404  # what another site's page could send
        assert received == [] and status() == {"state": "waiting", "message": ""}

        answered = []
        sending = threading.Thread(target=send)
        sending.start()
        assert reading.wait(5)
        assert status() == {"state": "reading", "message": "Reading “Harbor Lights”… (see the terminal)"}
        go_on.set()
        sending.join(5)
        assert answered == [{"ok": False, "message": "None of its tracks follows the melody"}]
        assert status()["state"] == "failed" and not server.done

        send()
        assert answered[-1] == {"ok": True, "message": "✓ Done"} and server.done
        assert thread.is_alive()  # until the page following it is told
        assert status() == {"state": "done", "message": "✓ Done"}
        thread.join(2)
        assert not thread.is_alive() and received == [SENT, SENT]
    finally:
        go_on.set()
        server.shutdown()
        server.server_close()


def test_without_a_page_following_it_it_stops_soon_after_a_tab_fits(monkeypatch):
    monkeypatch.setattr(songsterr.Waiting, "LINGER", 0.1)
    server = songsterr.Waiting(0, "", lambda sent: (True, "✓ Done"))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert server.receive(SENT) == (True, "✓ Done")
        thread.join(2)
        assert not thread.is_alive()
        assert server.receive(SENT) == (True, "✓ Done")  # a second click doesn't read it again
    finally:
        server.shutdown()
        server.server_close()


def test_shed_reference_gets_a_tab_from_songsterr(harbor, monkeypatch):
    set_reference(harbor)
    tune = [57, 60, 62, 64, 60, 57, 55, 57]
    singer = track([[beat((1, 4), note(tune[(4 * bar + k) % 8], 1)) for k in range(4)] for bar in range(12)])
    bass = track([[beat((1, 2), note(45, 4)), beat((1, 2), note(45, 4))]] * 12, name="Bass")
    [_, voice] = read(bass, singer)
    folder = harbor.library.root / "Harbor Lights"
    kept = json.loads((folder / ".reference.json").read_text())
    kept["contour"] = heard([(a + 2.0, b + 2.0, p - 69) for a, b, p in voice.notes], 28)
    (folder / ".reference.json").write_text(json.dumps(kept))

    said = []

    class Waiting:
        """The bookmark sends a tab of another song, then this one."""

        def __init__(self, port, shown, received):
            assert port == songsterr.PORT and "Harbor Lights" in shown
            self.received, self.done = received, False

        def serve_forever(self):
            said.append(self.received({**SENT, "title": "Other"}))
            said.append(self.received(SENT))
            self.done = said[-1][0]

        def server_close(self):
            pass

    sent = iter([[(songsterr.Track(0, "Bass", GUITAR, False), bass)],
                 [(songsterr.Track(0, "Bass", GUITAR, False), bass), (songsterr.Track(2, "Singer", GUITAR, False), singer)]])
    monkeypatch.setattr(songsterr, "Waiting", Waiting)
    monkeypatch.setattr(songsterr, "download", lambda tab: next(sent))
    monkeypatch.setattr("webbrowser.open", lambda url: True)

    result = harbor("reference", "harbor", "--songsterr")
    assert result.exit_code == 0, result.output
    assert said[0][0] is False and "None of the tracks of Songsterr's tab of “Other”" in said[0][1]
    assert said[1][0] is True and said[1][1].startswith("✓ Singing along to “Harbor Lights”")
    shown = text(result)
    assert "Songsterr couldn't be reached (offline). Pick the tab yourself:" in shown  # asked first: see below
    assert "Waiting for another tab" in shown
    assert ("Singing along to “Harbor Lights” now goes by the 48 notes of “Singer” in Songsterr's tab of “Harbor "
            "Lights” by The Tides.") in shown
    reference = harbor.library.reference("Harbor Lights")
    assert reference.tab["file"] == SENT["url"] and reference.tab["notes"][0] == pytest.approx([2.0, 2.5, -12.0], abs=0.08)

    assert harbor("reference", "harbor", "--songsterr", "--no-tab").exit_code == 1


LYRICS = [{"line": 1, "offset": 2, "text": "Har-bor lights are low+now\nand I'm go- ing home to-night ex-tra words"},
          {"line": 2, "offset": 1, "text": ""}]


def sung_track():
    """A voice with words under its notes, from its second bar: a rest and a tied note take none."""
    bars = [[beat((1, 1), note(60, 1))],
            [beat((1, 4), note(57, 1)), beat((1, 4), note(60, 1)), beat((1, 4)), beat((1, 4), note(62, 1))],
            [beat((1, 4), note(64, 1)), beat((1, 4), note(64, 1, tie=True)), beat((1, 4), note(60, 1)),
             beat((1, 4), note(57, 1))],
            [beat((1, 4), note(55, 1)), beat((1, 4), note(57, 1)), beat((1, 4), note(60, 1)), beat((1, 4), note(62, 1))],
            [beat((1, 2), note(64, 1)), beat((1, 2), note(62, 1))]]
    return {**track(bars), "newLyrics": LYRICS}


def test_the_words_under_a_voices_notes_are_read_syllable_by_syllable():
    [voice] = read(sung_track())
    assert [(word, a, b) for word, a, b, _ in voice.words] == [
        ("Harbor", 2.0, 3.0), ("lights", 3.5, 4.0), ("are", 4.0, 5.0), ("low now", 5.0, 5.5), ("and", 5.5, 6.0),
        ("I'm", 6.0, 6.5), ("going", 6.5, 7.5), ("home", 7.5, 8.0), ("tonight", 8.0, 10.0)]  # the rest has no note
    assert voice.words[0][3] == [("Har", 2.0), ("bor", 2.5)] and voice.words[1][3] == []
    assert voice.words[-1][3] == [("to", 8.0), ("night", 9.0)]
    [plain] = read(track([[beat((1, 1), note(60, 1))]]))
    assert plain.words is None


def test_songsterr_is_asked_for_a_songs_tab(monkeypatch):
    asked = []
    answers = {"songs": [{"songId": 9, "artist": "Someone", "title": "Harbor Lights"},
                         {"songId": 8, "artist": "The Tides", "title": "Harbor Lights (Fingerstyle)"},
                         {"songId": 27, "artist": "The Tides", "title": "Harbor lights"}, "junk"],
               "meta/27": {"songId": 27, "revisionId": 5184162, "image": "v5-2-1-GEwclkOmjsFXZOxQ",
                           "title": "Harbor Lights", "artist": "The Tides",
                           "tracks": [{"instrumentId": 66, "name": "Singer", "tuning": GUITAR},
                                      {"instrumentId": 1024, "name": "Drums"}]}}
    monkeypatch.setattr(songsterr, "_ask", lambda path, timeout=15: asked.append(path) or answers[path.split("?")[0]])
    assert songsterr.find("Harbor Lights", "The Tides") == [27, 9]  # this very title, the artist's first
    assert "pattern=The+Tides+Harbor+Lights" in asked[0]
    found = songsterr.fetch(27)
    assert (found.id, found.revision, found.image, found.url) == (27, 5184162, SENT["image"], SENT["url"])
    assert [(t.part, t.name, t.drums) for t in found.tracks] == [(0, "Singer", False), (1, "Drums", True)]
    assert songsterr.song_id(found.url) == 27 and songsterr.song_id("~/Downloads/harbor-s27.gp") is None
    answers["meta/27"] = {"error": "not found"}
    with pytest.raises(tab.TabError):
        songsterr.fetch(27)


def test_shed_reference_asks_songsterr_before_the_bookmark_and_times_the_lyrics_by_its_notes(harbor, monkeypatch, tone):
    from datetime import datetime

    from conftest import add_take
    from test_melody import LINES

    from woodshed import cli
    from woodshed.lyrics import words
    from woodshed.melody import Melody

    add_take(harbor.library, tone, "Harbor Lights", 8, 13, datetime(2026, 6, 2, 20, 0), LINES, genius_id=7)
    harbor.lyrics[7] = ("https://genius.com/harbor-lights-lyrics", LINES)
    sung = [w for line in LINES for w in words(line)]
    tune = [57, 60, 62, 64, 60, 57, 55, 59]
    under = " ".join(w[:2] + "-" + w[2:] if len(w) > 3 else w for w in sung)  # the longer words on two notes
    bars = [[beat((1, 4), note(tune[(4 * bar + k) % 8], 1)) for k in range(4)]
            for bar in range(-(-len(under.replace("-", "- ").split()) // 4))]
    singer = {**track(bars), "newLyrics": [{"line": 1, "offset": 1, "text": under}]}
    [voice] = read(singer)
    # In the recording, the song starts 2 s in, and each word was heard a fifth of a second before its note.
    line_of = [n for n, line in enumerate(LINES) for _ in words(line)]
    set_reference(harbor, Melody(LINES, [(w, a + 1.8, b + 1.8, n) for (w, a, b, _), n in zip(voice.words, line_of)],
                                 [(a + 2.0, b + 2.0, p - 69) for a, b, p in voice.notes], None))
    folder = harbor.library.root / "Harbor Lights"
    kept = json.loads((folder / ".reference.json").read_text())
    kept["contour"] = heard([(a + 2.0, b + 2.0, p - 69) for a, b, p in voice.notes], voice.notes[-1][1] + 4)
    (folder / ".reference.json").write_text(json.dumps(kept))
    monkeypatch.setattr(songsterr, "find", lambda title, artist=None: [27])
    monkeypatch.setattr(songsterr, "fetch", lambda song_id: songsterr.song(SENT))
    monkeypatch.setattr(songsterr, "download", lambda tab: [(songsterr.Track(0, "Singer", GUITAR, False), singer)])
    monkeypatch.setattr(songsterr, "Waiting", None)  # the bookmark isn't waited for

    result = harbor("reference", "harbor", "--songsterr")
    assert result.exit_code == 0, result.output
    assert "Singing along to “Harbor Lights” now goes by" in text(result) and "Pick the tab" not in text(result)
    reference = harbor.library.reference("Harbor Lights")
    assert len(reference.tab["words"]) == len(sung) and reference.tab["file"] == SENT["url"]
    shown = cli._to_sing_along(harbor.library, "Harbor Lights").lines
    assert [line["text"] for line in shown] == LINES
    timed = [w for line in shown for w in line["words"]]
    assert [w[1] for w in timed] == [w[1] for w in reference.tab["words"]]  # each when its note starts
    long = next(w for w in timed if len(w[0]) > 3)  # written on two notes: lit up in two parts
    assert [letters for letters, _ in long[3]] == [2, len(long[0])] and long[3][0][1] == long[1] < long[3][1][1]

    harbor("reference", "harbor", "--no-tab")  # without the tab, by the words heard again
    assert all(len(w) == 3 for line in cli._to_sing_along(harbor.library, "Harbor Lights").lines for w in line["words"])
