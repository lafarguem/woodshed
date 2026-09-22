"""Recording from your phone (`shed serve`): a page on your Wi-Fi that records, and sends each take here.

Phone browsers only let a page use the microphone over HTTPS, so the page is served with a certificate of
Woodshed's own, made the first time: your phone warns that it doesn't know it, and you go on anyway. The link
carries a secret, so that nobody else on the network can send takes. The takes join those waiting to be filed,
as with `shed rec --later`, and the page follows them (a Session) as they're filed: asking which song one is,
when that isn't clear, then showing how it rated.
"""

import hmac
import ipaddress
import json
import re
import secrets
import socket
import ssl
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from woodshed import audio
from woodshed.library import Library

PORT = 8765
MAX_UPLOAD_BYTES = 1 << 30  # about 9 hours, at the bitrate the page records at
_KEY_HEADER = "X-Woodshed-Key"
_SUFFIXES = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/aac": ".aac",
             "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"}


class Rejected(Exception):
    """A take that isn't kept: why, for the phone to say."""


class Session:
    """The takes the phone has sent since `shed serve` started, as its page shows them. Each has a "status":
    "kept" (to be filed later, by `shed add`), "waiting" (to be filed here), "filing", "question" (which song it is,
    with "options"), "filed" (with a "result") or "failed" (with an "error"). Kept in memory only."""

    def __init__(self):
        self._takes: dict[str, dict] = {}
        self._answers: dict[str, dict] = {}
        self._changed = threading.Condition()
        self._stopped = False

    def add(self, raw: Path, started: datetime, seconds: float, status: str) -> str:
        with self._changed:
            self._takes[raw.name] = {"id": raw.name, "started": started.isoformat(timespec="seconds"),
                                     "seconds": round(seconds, 2), "status": status}
        return raw.name

    def update(self, take: str, **fields) -> None:
        """Set these fields of the take (None removes one)."""
        with self._changed:
            entry = self._takes[take]
            for name, value in fields.items():
                if value is None:
                    entry.pop(name, None)
                else:
                    entry[name] = value

    def ask(self, take: str, options: list[str]) -> dict | None:
        """Which song the take is, as answered on the phone: {"option": index}, {"name": typed} or {"unsorted":
        True}. Waits for the answer; None if `shed serve` stops first."""
        self.update(take, status="question", options=options)
        with self._changed:
            while take not in self._answers and not self._stopped:
                self._changed.wait()
            answer = self._answers.pop(take, None)
        self.update(take, status="filing", options=None)
        return answer

    def answer(self, take: str, answer: dict) -> bool:
        """Whether it was waiting for one."""
        with self._changed:
            entry = self._takes.get(take)
            if not entry or entry["status"] != "question" or not _valid(answer, len(entry["options"])):
                return False
            self._answers[take] = answer
            self._changed.notify_all()
            return True

    def stop(self) -> None:
        with self._changed:
            self._stopped = True
            self._changed.notify_all()

    def takes(self) -> list[dict]:
        with self._changed:
            return [json.loads(json.dumps(entry)) for entry in self._takes.values()]  # copies


def _valid(answer, options: int) -> bool:
    if not isinstance(answer, dict) or len(answer) != 1:
        return False
    (kind, value), = answer.items()
    return ((kind == "option" and isinstance(value, int) and not isinstance(value, bool) and 0 <= value < options)
            or (kind == "name" and isinstance(value, str) and 0 < len(value.strip()) <= 200)
            or (kind == "unsorted" and value is True))


def addresses() -> list[str]:
    """This Mac's addresses on its local networks (Wi-Fi, Ethernet), the one it uses the most first. A VPN's
    address is left out: though the Mac sends everything through it, the phone isn't on the VPN."""
    try:
        listing = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        listing = ""
    found, interface = [], ""
    for line in listing.splitlines():
        if line[:1].strip():  # "en0: flags=…" starts an interface; its addresses follow, indented
            interface = line.split(":", 1)[0]
        elif re.fullmatch(r"en\d+", interface) and (inet := re.match(r"\s+inet (\S+)", line)):
            ip = ipaddress.ip_address(inet[1])
            if ip.is_private and not ip.is_link_local:
                found.append(inet[1])
    used = _address_used()
    return sorted(found, key=lambda a: a != used) or [used]


def _address_used() -> str:
    """The address of the network the Mac sends its traffic through (a VPN's, if one is on)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))  # nothing is sent: it only picks the network the Mac would use
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def credentials(folder: Path) -> tuple[Path, Path, str]:
    """The certificate and private key the page is served with, and the secret its link carries. Made the first
    time and kept, so that a link bookmarked on the phone keeps working."""
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    cert, key, secret = folder / "cert.pem", folder / "key.pem", folder / "secret"
    if not (cert.exists() and key.exists()):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
                        "-subj", "/CN=Woodshed", "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
        key.chmod(0o600)
    if not secret.exists():
        secret.write_text(secrets.token_urlsafe(16))
        secret.chmod(0o600)
    return cert, key, secret.read_text().strip()


class Server(ThreadingHTTPServer):
    """The page, and the takes it sends: `received(raw, seconds, waiting)` is told of each one kept."""

    daemon_threads = True

    def __init__(self, songs: Library, secret: str, cert: Path, key: Path, port: int,
                 received: Callable[[Path, float, int], None], session: Session | None = None, filing: bool = False,
                 max_bytes: int = MAX_UPLOAD_BYTES):
        super().__init__(("0.0.0.0", port), _Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        # The handshake happens in each request's own thread: a phone that stops at the certificate warning
        # mustn't hold up the others.
        self.socket = context.wrap_socket(self.socket, server_side=True, do_handshake_on_connect=False)
        self.songs, self.secret, self.received, self.max_bytes = songs, secret, received, max_bytes
        self.session, self.filing = session or Session(), filing  # filing: they're filed here as they come
        self._naming = threading.Lock()

    def handle_error(self, request, client_address) -> None:
        pass  # a phone that went away, or refused the certificate: nothing to do about it here

    def receive(self, body, length: int, suffix: str, started: datetime) -> tuple[str, float, int]:
        """Keep an upload among the takes waiting to be filed, as a wav: its id in the session, its length (s), and
        how many are waiting. Raises Rejected if it isn't worth keeping."""
        upload = self.songs.incoming / f".upload-{secrets.token_hex(4)}{suffix}"
        try:
            with upload.open("wb") as f:
                left = length
                while left:
                    if not (chunk := body.read(min(left, 1 << 20))):
                        raise Rejected("the upload was cut off")
                    f.write(chunk)
                    left -= len(chunk)
            try:
                samples = audio.load(upload)
            except audio.AudioError:
                raise Rejected("it isn't audio Woodshed can read") from None
            if samples.size == 0 or abs(samples).max() < 1e-4:
                raise Rejected("the recording is silent: let the page use the microphone")
            with self._naming:
                raw = self.songs.raw_path(started)
                recording = raw.with_suffix(".recording")
                recording.touch()  # the name is taken
            try:
                audio.to_wav(upload, recording)
            except BaseException:
                recording.unlink(missing_ok=True)
                raise
            recording.rename(raw)
        finally:
            upload.unlink(missing_ok=True)
        seconds, waiting = len(samples) / audio.SAMPLE_RATE, len(self.songs.waiting())
        take = self.session.add(raw, started, seconds, "waiting" if self.filing else "kept")
        self.received(raw, seconds, waiting)
        return take, seconds, waiting


def _started(header: str | None) -> datetime:
    """When the phone says the recording started (ms since 1970), unless its clock is clearly off."""
    now = datetime.now()
    try:
        when = datetime.fromtimestamp(int(header) / 1000)
    except (TypeError, ValueError, OverflowError, OSError):
        return now
    return when if abs((now - when).total_seconds()) < 24 * 3600 else now


class _Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, format, *args) -> None:
        pass

    def _allowed(self, key: str | None) -> bool:
        return key is not None and hmac.compare_digest(key.encode(), self.server.secret.encode())

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, **answer) -> None:
        self._reply(status, "application/json", json.dumps(answer).encode())

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/takes":
            if not self._allowed(self.headers.get(_KEY_HEADER)):
                return self._json(403, error="this isn't the link `shed serve` shows")
            return self._json(200, takes=self.server.session.takes(), filing=self.server.filing)
        if url.path != "/":
            return self._reply(404, "text/plain", b"Not found")
        if not self._allowed(parse_qs(url.query).get("key", [None])[0]):
            return self._reply(403, "text/plain; charset=utf-8",
                               "Open the link `shed serve` shows (or scan its QR code).".encode())
        self._reply(200, "text/html; charset=utf-8", resources.files("woodshed").joinpath("phone.html").read_bytes())

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if not self._allowed(self.headers.get(_KEY_HEADER)):
            return self._json(403, error="this isn't the link `shed serve` shows")
        if (answered := re.fullmatch(r"/takes/([^/]+)/answer", path)):
            return self._answer(answered[1])
        if path != "/takes":
            return self._json(404, error="not found")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= self.server.max_bytes:
            return self._json(413 if length else 411, error="the recording is too long" if length else "nothing was sent")
        try:
            take, seconds, waiting = self.server.receive(self.rfile, length,
                                                         _SUFFIXES.get(self.headers.get_content_type(), ""),
                                                         _started(self.headers.get("X-Recorded-At")))
        except Rejected as e:
            return self._json(422, error=str(e))
        self._json(200, id=take, seconds=seconds, waiting=waiting)

    def _answer(self, take: str) -> None:
        """Which song a take is, from the phone."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            answer = json.loads(self.rfile.read(length)) if 0 < length <= 4096 else None
        except ValueError:
            answer = None
        if not self.server.session.answer(take, answer):
            return self._json(409, error="that take isn't waiting for an answer")
        self._json(200, ok=True)
