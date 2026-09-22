"""Recording from a phone (`shed serve`): the page, and the takes it sends, over a real HTTPS connection."""

import http.client
import json
import shutil
import ssl
import stat
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import SONGS, add_take, mishear, recording

from woodshed import audio, cli, config, isolate, phone
from woodshed.library import Library


@pytest.fixture
def server(tmp_path):
    """`shed serve`'s server on a free port, for a library in tmp_path."""
    songs = Library(tmp_path / "lib")
    cert, key, secret = phone.credentials(tmp_path / "phone")
    received = []
    running = phone.Server(songs, secret, cert, key, 0, lambda *take: received.append(take), max_bytes=5_000_000)
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=running.server_address[1], secret=secret, songs=songs, received=received,
                          session=running.session)
    running.shutdown()
    running.server_close()


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPSConnection("127.0.0.1", server.port, context=ssl._create_unverified_context(),
                                             timeout=30)  # the certificate is Woodshed's own, as the phone sees it
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    return response.status, response.read()


def until(condition, seconds=10):
    """Wait for another thread to get there: a test fails, rather than hangs, if it never does."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def upload(server, body, key=None, started=None, content_type="audio/webm"):
    headers = {"X-Woodshed-Key": server.secret if key is None else key, "Content-Type": content_type}
    if started:
        headers["X-Recorded-At"] = str(int(started.timestamp() * 1000))
    status, answer = request(server, "POST", "/takes", body, headers)
    return status, json.loads(answer or b"{}")


def recorded(tmp_path, source, seconds):
    """What a phone's browser records: Opus in WebM."""
    path = tmp_path / "take.webm"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}:duration={seconds}",
                    "-codec:a", "libopus", str(path)], check=True)
    return path.read_bytes()


def test_the_certificate_and_the_secret_are_made_once(tmp_path):
    cert, key, secret = phone.credentials(tmp_path / "phone")
    assert phone.credentials(tmp_path / "phone") == (cert, key, secret)  # a bookmarked link keeps working
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    assert len(secret) >= 20


def test_the_page_opens_only_with_the_links_key(server):
    status, page = request(server, "GET", f"/?key={server.secret}")
    assert status == 200 and b"getUserMedia" in page and b"echoCancellation: false" in page
    assert request(server, "GET", "/")[0] == 403
    assert request(server, "GET", "/?key=guess")[0] == 403


def test_a_take_from_the_phone_waits_to_be_filed(server, tmp_path):
    started = (datetime.now() - timedelta(minutes=5)).replace(microsecond=0)

    status, answer = upload(server, recorded(tmp_path, "sine=frequency=196", 25), started=started)

    assert status == 200, answer
    [raw] = server.songs.waiting()
    assert raw.name == f"{started:%Y-%m-%d_%H-%M-%S}.wav"  # when the phone started recording
    assert answer["seconds"] == pytest.approx(25, abs=0.2) and answer["waiting"] == 1
    assert len(audio.load(raw)) / audio.SAMPLE_RATE == pytest.approx(25, abs=0.2)
    assert server.received == [(raw, answer["seconds"], 1)]
    assert sorted(p.name for p in server.songs.incoming.iterdir()) == [raw.name]  # nothing else left behind


@pytest.mark.parametrize("sent, status, error", [
    ("silence", 422, "the recording is silent"),
    ("not audio", 422, "isn't audio Woodshed can read"),
    ("nothing", 411, "nothing was sent"),
    ("too much", 413, "too long"),  # the server in these tests takes 5 MB at most
])
def test_a_take_that_isnt_worth_keeping_is_refused(server, tmp_path, sent, status, error):
    if sent == "too much":  # refused as soon as its size is known: only the size is sent
        answer = request(server, "POST", "/takes", headers={"X-Woodshed-Key": server.secret, "Content-Length": "6000000"})
        answer = answer[0], json.loads(answer[1])
    else:
        body = {"silence": lambda: recorded(tmp_path, "anullsrc=r=48000:cl=mono", 5), "not audio": lambda: b"hello",
                "nothing": lambda: b""}[sent]()
        answer = upload(server, body)
    assert answer[0] == status and error in answer[1]["error"]
    assert list(server.songs.incoming.iterdir()) == [] and server.received == []


def test_starting_doesnt_look_up_the_macs_name(tmp_path, monkeypatch):
    monkeypatch.setattr(phone.socket, "getfqdn", lambda *args: pytest.fail("looked this Mac's name up"))
    cert, key, secret = phone.credentials(tmp_path / "phone")
    phone.Server(Library(tmp_path / "lib"), secret, cert, key, 0, lambda *take: None).server_close()


def test_nobody_else_can_send_takes(server, tmp_path):
    status, answer = upload(server, recorded(tmp_path, "sine=frequency=196", 5), key="guess")
    assert status == 403 and server.songs.waiting() == []


class FakeServer:
    """Stands in for phone.Server: stops as soon as it starts, as if you'd pressed Ctrl+C."""

    def __init__(self, songs, *args, **kwargs):
        self.songs = songs

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def test_shed_serve_shows_the_link_and_what_to_expect(shed, monkeypatch):
    monkeypatch.setattr(phone, "Server", FakeServer)
    monkeypatch.setattr(phone, "addresses", lambda: ["192.168.1.23", "10.0.0.5"])

    result, later = shed("serve"), shed("serve", "--later")

    assert result.exit_code == 0, result.output
    secret = (shed.library.root.parent / "settings" / "phone" / "secret").read_text()
    assert f"https://192.168.1.23:{phone.PORT}/?key={secret}" in result.output
    output = " ".join(result.output.split())
    assert f"another of this Mac's networks: https://10.0.0.5:{phone.PORT}/?key={secret}" in output
    assert "Go on anyway (Advanced, then Proceed)" in output and "▀" in result.output  # the QR code
    assert "filed here as they come, and the page shows how they rated" in output
    assert "wait here to be filed: file them with shed add" in " ".join(later.output.split())


def test_a_take_left_half_filed_waits_to_be_filed_again(shed, monkeypatch):
    monkeypatch.setattr(phone, "Server", FakeServer)
    stuck = shed.library.incoming / "2026-09-22_18-30-00.wav.filing"  # `shed serve` stopped for good while filing it
    stuck.write_bytes(b"")
    shed("serve")
    assert [p.name for p in shed.library.waiting()] == ["2026-09-22_18-30-00.wav"]


def test_a_port_in_use_is_reported(shed, monkeypatch):
    def taken(*args, **kwargs):
        raise OSError(48, "Address already in use")

    monkeypatch.setattr(phone, "Server", taken)
    result = shed("serve", "--port", "8765")
    assert result.exit_code == 1 and "Can't use port 8765: Address already in use" in " ".join(result.output.split())


IFCONFIG = """lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet6 fe80::1c2b:3d4e%en0 prefixlen 64 secured scopeid 0xe
\tinet 192.168.1.11 netmask 0xffffff00 broadcast 192.168.1.255
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.12.3 netmask 0xffff0000
en7: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.40.1 netmask 0xffff0000
\tinet 10.0.0.5 netmask 0xffffff00 broadcast 10.0.0.255
utun4: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 192.168.123.8 --> 192.168.123.8 netmask 0xffffff00
"""


@pytest.mark.parametrize("used, expected", [
    ("192.168.123.8", ["192.168.1.11", "10.0.0.5"]),  # a VPN is on: its address is no use to the phone
    ("10.0.0.5", ["10.0.0.5", "192.168.1.11"]),  # the network the Mac uses the most comes first
])
def test_the_link_uses_the_macs_address_on_its_local_networks(monkeypatch, used, expected):
    monkeypatch.setattr(phone.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=IFCONFIG))
    monkeypatch.setattr(phone, "_address_used", lambda: used)
    assert phone.addresses() == expected


def test_without_a_local_network_the_link_uses_what_the_mac_does(monkeypatch):
    monkeypatch.setattr(phone.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="lo0: flags=8049<UP>\n"))
    monkeypatch.setattr(phone, "_address_used", lambda: "127.0.0.1")
    assert phone.addresses() == ["127.0.0.1"]


def test_the_phone_answers_which_song_a_take_is():
    session = phone.Session()
    take = session.add(Path("2026-09-22_18-30-00.wav"), datetime(2026, 9, 22, 18, 30), 191.5, "waiting")
    asked = []
    thread = threading.Thread(target=lambda: asked.append(session.ask(take, ["Harbor Lights (your songs)"])), daemon=True)
    thread.start()
    until(lambda: session.takes()[0]["status"] == "question")

    assert session.takes()[0]["options"] == ["Harbor Lights (your songs)"]
    for wrong in ({"option": 1}, {"option": True}, {"name": "  "}, {"unsorted": False}, {"option": 0, "name": "x"}, []):
        assert not session.answer(take, wrong)
    assert session.answer(take, {"option": 0})
    thread.join(5)
    assert asked == [{"option": 0}] and session.takes()[0]["status"] == "filing" and "options" not in session.takes()[0]
    assert not session.answer(take, {"option": 0})  # it isn't asking any more


def test_a_question_left_unanswered_ends_with_shed_serve():
    session = phone.Session()
    take = session.add(Path("take.wav"), datetime.now(), 30, "waiting")
    asked = []
    thread = threading.Thread(target=lambda: asked.append(session.ask(take, [])), daemon=True)
    thread.start()
    session.stop()
    thread.join(5)
    assert asked == [None]


def test_the_page_follows_its_takes_and_answers_questions(server):
    take = server.songs.incoming / "2026-09-22_18-30-00.wav"
    server_session = server.session
    server_session.add(take, datetime(2026, 9, 22, 18, 30), 30, "waiting")
    threading.Thread(target=lambda: server_session.ask(take.name, ["Harbor Lights (your songs)"]), daemon=True).start()
    until(lambda: server_session.takes()[0]["status"] == "question")
    key = {"X-Woodshed-Key": server.secret}

    status, answer = request(server, "GET", "/takes", headers=key)
    assert status == 200 and json.loads(answer)["takes"][0]["options"] == ["Harbor Lights (your songs)"]
    assert request(server, "GET", "/takes")[0] == 403
    answered = f"/takes/{take.name}/answer"
    assert request(server, "POST", answered, b'{"option": 0}', {"Content-Type": "application/json"})[0] == 403
    assert request(server, "POST", answered, b'{"option": 5}', key)[0] == 409  # not one of its options
    assert request(server, "POST", answered, b'{"option": 0}', key)[0] == 200


@pytest.fixture
def phone_take(shed, tmp_path):
    """A take from the phone, waiting to be filed, and the session the page follows it by."""
    raw = shed.library.raw_path(datetime(2026, 9, 22, 18, 30))
    shutil.copy(recording(tmp_path, "sent.wav"), raw)
    session = phone.Session()
    session.add(raw, datetime(2026, 9, 22, 18, 30), 25, "waiting")
    return SimpleNamespace(raw=raw, session=session, file=lambda: cli._file_from_phone(
        raw, shed.library, config.load(), session, None))


def test_a_take_from_the_phone_is_filed_and_its_rating_shown_there(shed, phone_take, tone, rng):
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), mishear(SONGS["Harbor Lights"], 0.3, rng))
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)

    phone_take.file()

    [take] = phone_take.session.takes()
    assert take["status"] == "filed", take
    assert take["result"]["song"] == "Harbor Lights" and (take["result"]["take"], take["result"]["of"]) == (2, 2)
    assert take["result"]["rating"].startswith("Rated 9.2/10 · pitch 10.0/10 (10¢ off) · timing 8.0/10")
    assert shed.library.waiting() == [] and list(shed.library.incoming.iterdir()) == []


@pytest.mark.parametrize("answer, song", [({"name": "Gravel Road"}, "Gravel Road"), ({"unsorted": True}, None)])
def test_the_phone_is_asked_when_the_song_isnt_clear(shed, phone_take, rng, answer, song):
    shed.heard = mishear(SONGS["Gravel Road"], 0.3, rng)  # a song you haven't filed yet: nothing to go by
    filing = threading.Thread(target=phone_take.file, daemon=True)
    filing.start()
    until(lambda: phone_take.session.takes()[0]["status"] == "question")
    assert shed.library.waiting() == []  # set aside while it's being filed: `shed add` can't file it twice

    assert phone_take.session.answer(phone_take.raw.name, answer)
    filing.join(30)

    assert phone_take.session.takes()[0]["status"] == "filed"
    assert (len(shed.library.takes_of(song)) if song else len(shed.library.unsorted())) == 1


def test_stopping_before_the_phone_answers_leaves_the_take_waiting(shed, phone_take, rng):
    shed.heard = ["la la la"]
    filing = threading.Thread(target=phone_take.file, daemon=True)
    filing.start()
    until(lambda: phone_take.session.takes()[0]["status"] == "question")

    phone_take.session.stop()
    filing.join(30)

    assert phone_take.session.takes()[0]["status"] == "kept"
    assert shed.library.waiting() == [phone_take.raw] and shed.library.unsorted() == []
    assert [p.name for p in shed.library.incoming.iterdir()] == [phone_take.raw.name]  # nothing else left behind


def test_a_take_that_cant_be_filed_waits_for_shed_add(shed, phone_take, monkeypatch):
    def crash(src, start=0.0, seconds=None):
        raise RuntimeError("Demucs ran out of memory")

    monkeypatch.setattr(isolate, "separate", crash)
    phone_take.file()

    [take] = phone_take.session.takes()
    assert take["status"] == "failed" and take["error"].startswith("Demucs ran out of memory. It waits on your Mac")
    assert shed.library.waiting() == [phone_take.raw]


def test_a_take_sent_from_the_phone_comes_back_filed(shed, tmp_path, tone, rng):
    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), mishear(SONGS["Harbor Lights"], 0.3, rng))
    shed.heard = mishear(SONGS["Harbor Lights"], 0.3, rng, keep=0.6)
    cert, key, secret = phone.credentials(tmp_path / "phone")
    session = phone.Session()
    filer = cli._PhoneFiler(shed.library, config.load(), session, None)
    running = phone.Server(shed.library, secret, cert, key, 0, lambda raw, *_: filer.takes.put(raw), session, filing=True)
    filer.start()
    threading.Thread(target=running.serve_forever, daemon=True).start()
    server = SimpleNamespace(port=running.server_address[1], secret=secret)
    try:
        status, sent = upload(server, recorded(tmp_path, "sine=frequency=196", 25))
        assert status == 200 and sent["id"]
        for _ in range(300):
            takes = json.loads(request(server, "GET", "/takes", headers={"X-Woodshed-Key": secret})[1])["takes"]
            if takes[0]["status"] not in ("waiting", "filing"):
                break
            time.sleep(0.1)
        assert takes[0]["status"] == "filed" and takes[0]["result"]["song"] == "Harbor Lights"
    finally:
        filer.takes.put(None)
        running.shutdown()
        running.server_close()


def test_the_phone_is_shown_the_lines_furthest_from_the_melody(shed, phone_take, tone, rng):
    from test_melody import LINES, sing

    add_take(shed.library, tone, "Harbor Lights", 2, 7, datetime(2026, 6, 1, 20, 0), LINES)
    shed.library.set_reference("Harbor Lights", sing(), "original.mp3")
    shed.heard, shed.melody = mishear(SONGS["Harbor Lights"], 0.2, rng), sing(off=-1.0)  # a semitone under

    phone_take.file()

    result = phone_take.session.takes()[0]["result"]
    assert result["rating"].startswith("Rated 3.2/10 · pitch 0.0/10 (100¢ off the melody)")
    assert result["notes"] == ["Your lines sit about a semitone under the melody."]
    assert result["furthest"][0] == {"time": "0:01", "text": LINES[0], "off": "100¢ under"}
