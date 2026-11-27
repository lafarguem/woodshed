"""A song's tab from Songsterr, to sing along to its notes (tab.py), without saving a Guitar Pro file first.

Songsterr turns away pages and API calls that don't come from a browser (a "Quick check"), so a bookmark does the
asking: clicked on the song's tab on songsterr.com, it reads where Songsterr keeps that tab (the song, its revision
and tracks, which the page embeds) and opens Woodshed's page on this Mac with it, in a small window. While `shed reference <song>
--songsterr` waits, that page is served here (Waiting), and hands it on: each track is then downloaded from
Songsterr's CDN, which serves anyone, and read (score()). The page `shed` opened follows it (/status): reading,
then found, when it closes itself, or why the tab doesn't fit.

- A track is a list of bars, each with its beats: how long (a fraction of a whole note, dots and tuplets counted in),
  and the notes, as a string and a fret on the track's tuning (string 0 the highest).
- Bars are laid out by the track with the most: time signatures, repeats (`repeatStart`, and `repeat`, the times the
  section is played, on its last bar), alternate endings (the times through it's played on), and tempo changes.
- Drum tracks aren't downloaded: they have no melody.
- A voice's track has the words sung on it (`newLyrics`): a text, its syllables set apart by spaces and hyphens
  ("bo-ring a-way"), from the bar it starts at (`offset`, counted from 1). It doesn't say which note each is sung
  on: they go one a note, in order, rests and notes tied to the one before passed over (_syllables()). On a tab
  of 299 such notes, the text had 298 syllables; on another, 166 for 137 notes: more of its last line written
  than is sung, which falls off the end. Only the first text is read (others would be other verses, on a repeat).
- Songsterr's API answers anyone where a song's tab is kept (/api/meta/<song>), and which songs match a search
  (/api/songs): `shed reference --songsterr` asks it (find(), fetch()), and falls back on the bookmark if it
  doesn't answer, as its pages don't to anything but a browser.

As in songsterr-downloader (github.com/Metaphysics0/songsterr-downloader, MIT), whose bookmarklet this one follows.
"""

import gzip
import json
import re
import socketserver
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlencode, urlsplit

from woodshed.tab import Score, TabError, played

PORT = 8766  # the bookmark opens the page there, so it can't change
CDNS = ("https://dqsljvtekg760.cloudfront.net", "https://d3d3l6a6rcgkaf.cloudfront.net")
API = "https://www.songsterr.com/api"
DRUMS = 1024  # Songsterr's instrument id for a drum kit
MAX_TRACKS = 100
_IMAGE = re.compile(r"(?=.{1,100}$)[\w-]+(\.[\w-]+)*")  # goes into the CDN's paths: no "/" nor ".."


@dataclass
class Track:
    part: int  # its number in the CDN's paths
    name: str
    tuning: list[int] | None  # MIDI notes of its strings, the highest first
    drums: bool


@dataclass
class Song:
    """A tab on Songsterr, as the bookmark sends it."""

    id: int
    revision: int
    image: str
    title: str
    artist: str
    url: str | None  # its page on songsterr.com
    tracks: list[Track]


def song(sent) -> Song:
    """What the bookmark sent, checked: anything can open the page with anything. Raises TabError."""
    def number(value, name, low=1, top=2**53) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= top:
            raise TabError(f"what was sent from Songsterr isn't a tab (its {name})")
        return value

    def text(value) -> str:
        return value.strip()[:300] if isinstance(value, str) else ""

    if not isinstance(sent, dict):
        raise TabError("what was sent from Songsterr isn't a tab")
    image = sent.get("image")
    if not (isinstance(image, str) and _IMAGE.fullmatch(image)):
        raise TabError("what was sent from Songsterr isn't a tab (its image)")
    listed = sent.get("tracks")
    if not isinstance(listed, list) or not 0 < len(listed) <= MAX_TRACKS or not all(isinstance(t, dict) for t in listed):
        raise TabError("what was sent from Songsterr isn't a tab (its tracks)")
    tracks = []
    for track in listed:
        tuning = track.get("tuning")
        if not (isinstance(tuning, list) and 0 < len(tuning) <= 12
                and all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= 127 for n in tuning)):
            tuning = None
        instrument = track.get("instrumentId")
        tracks.append(Track(number(track.get("partId"), "tracks", 0, 1000), text(track.get("name")), tuning,
                            track.get("isDrums") is True or instrument == DRUMS))
    url = text(sent.get("url"))
    return Song(number(sent.get("songId"), "song"), number(sent.get("revisionId"), "revision"), image,
                text(sent.get("title")) or "the song", text(sent.get("artist")),
                url if url.startswith("https://www.songsterr.com/") else None, tracks)


def _ask(path: str, timeout: float = 15):
    """What Songsterr's API answers. Raises TabError."""
    request = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "woodshed", "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(8 << 20)
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
        return json.loads(body)
    except urllib.error.HTTPError as e:
        raise TabError(f"Songsterr didn't answer (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise TabError(f"Songsterr couldn't be reached ({getattr(e, 'reason', e)})") from None


def find(title: str, artist: str | None = None, limit: int = 3) -> list[int]:
    """The songs Songsterr has tabs of under this very title, as it ranks them, this artist's first: the song
    itself, and other tabs of it (as written for another instrument, say, which have no voice). Raises TabError."""
    from woodshed.lyrics import words

    hits = _ask(f"songs?{urlencode({'pattern': f'{artist} {title}' if artist else title, 'size': 20})}")
    wanted, by = words(title), words(artist or "")
    same = [hit for hit in hits if isinstance(hit, dict) and isinstance(hit.get("songId"), int)
            and words(str(hit.get("title", ""))) == wanted] if isinstance(hits, list) else []
    same.sort(key=lambda hit: words(str(hit.get("artist", ""))) != by)  # a stable sort: Songsterr's order otherwise
    return [hit["songId"] for hit in same[:limit]]


def fetch(song_id: int) -> Song:
    """Where Songsterr keeps a song's tab, as the bookmark would send it. Raises TabError."""
    meta = _ask(f"meta/{int(song_id)}")
    if not isinstance(meta, dict) or not isinstance(meta.get("tracks"), list):
        raise TabError("what Songsterr answered isn't a tab")
    slug = "-".join(re.findall(r"[a-z0-9]+", f"{meta.get('artist', '')} {meta.get('title', '')}".lower()))
    return song({**meta, "url": f"https://www.songsterr.com/a/wsa/{slug}-tab-s{int(song_id)}",
                 "tracks": [{**track, "partId": track.get("partId", part)} if isinstance(track, dict) else track
                            for part, track in enumerate(meta["tracks"])]})


def song_id(url: str) -> int | None:
    """The song a tab's page on songsterr.com is of (as Song.url)."""
    found = re.search(r"^https://www\.songsterr\.com/.*-s(\d+)$", url)
    return int(found.group(1)) if found else None


def download(tab: Song, timeout: float = 15) -> list[tuple[Track, dict]]:
    """The tab's tracks (but its drums), each as Songsterr writes it. Raises TabError if none could be got."""
    wanted = sorted((t for t in tab.tracks if not t.drums), key=lambda t: t.part)
    if not wanted:
        raise TabError("it has only drums")

    def get(track: Track) -> dict | str:
        """The track, or why it couldn't be got."""
        why = ""
        for cdn in CDNS:
            request = urllib.request.Request(f"{cdn}/{tab.id}/{tab.revision}/{tab.image}/{track.part}.json",
                                             headers={"User-Agent": "woodshed", "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read(64 << 20)
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    got = json.loads(body)
                return got if isinstance(got, dict) else "it isn't a track"
            except urllib.error.HTTPError as e:
                why = f"HTTP {e.code}"
                if e.code not in (403, 404):  # another of the CDNs may have it, else it's not worth asking
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                return str(getattr(e, "reason", e))
        return why

    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(get, wanted))
    found = [(track, payload) for track, payload in zip(wanted, got) if isinstance(payload, dict)]
    if not found:
        raise TabError(f"its tracks couldn't be downloaded from Songsterr ({got[0]})")
    return found


def score(found: list[tuple[Track, dict]]) -> Score:
    """The tracks as tab.timed() reads them. Raises TabError."""
    try:
        return _score(found)
    except (AttributeError, KeyError, ValueError, IndexError, TypeError, ZeroDivisionError):
        raise TabError("its tracks aren't written the way Songsterr writes them") from None


def _score(found: list[tuple[Track, dict]]) -> Score:
    master = max((payload for _, payload in found), key=lambda p: len(p["measures"]))
    lengths, layout, signature = [], [], (4, 4)
    for measure in master["measures"]:
        if isinstance(measure.get("signature"), list) and len(measure["signature"]) == 2:
            signature = tuple(int(n) for n in measure["signature"])
        lengths.append(signature[0] * 4 / signature[1])
        ending = measure.get("alternateEnding")
        endings = ([ending] if isinstance(ending, int) else ending if isinstance(ending, list) else [])
        times = measure.get("repeat", measure.get("repeatCount", 0))
        layout.append((measure.get("repeatStart") is True, times if isinstance(times, int) and times > 1 else 0,
                       frozenset(n for n in endings if isinstance(n, int) and n > 0)))
    tempos = [(int(t["measure"]), min(1.0, max(0.0, t.get("position", 0) / (t.get("type") or 4))), float(t["bpm"]))
              for t in (master.get("automations") or {}).get("tempo") or [] if t.get("bpm", 0) > 0]
    parts, sung = [], []
    for track, payload in found:
        tuning = payload.get("tuning") if isinstance(payload.get("tuning"), list) else track.tuning
        bars = []
        for measure in payload["measures"]:
            voices = measure.get("voices") or []
            beats = []
            for beat in (voices[0].get("beats") or [] if voices else []):
                if beat.get("graceNote"):  # takes no time of its own
                    continue
                if "duration" in beat:
                    numerator, denominator = beat["duration"]
                    length = 4 * numerator / denominator
                else:
                    length = 4 / beat["type"] * (2 - 0.5 ** beat.get("dots", 0))
                sounding = [(tuning[n["string"]] + n["fret"], n.get("tie") is True) for n in beat.get("notes") or []
                            if tuning and not n.get("rest") and not n.get("dead") and "fret" in n
                            and 0 <= n.get("string", -1) < len(tuning)]
                pitch, tied = max(sounding) if sounding and not beat.get("rest") else (None, False)
                beats.append((length, pitch, tied))
            bars.append(beats)
        parts.append((payload.get("name") or track.name or f"Track {track.part + 1}", bars))
        sung.append(_syllables(payload.get("newLyrics"), bars))
    return Score(lengths, tempos, played(layout), parts, sung if any(sung) else None)


def _syllables(lyrics, bars: list[list[tuple]]) -> list[list[str | None]] | None:
    """The syllables of a track's words, each on its beat (as tab.Score.sung): one a note, in order, from the bar
    the text starts at, passing over rests and notes tied to the one before. None if no words are written."""
    text = next((line for line in lyrics if isinstance(line, dict) and isinstance(line.get("text"), str)
                 and line["text"].strip()), None) if isinstance(lyrics, list) else None
    if text is None:
        return None
    # "a-way" is two syllables, "a+lot" two words on one note; a hyphen alone holds the syllable before.
    syllables = iter(s.replace("+", " ") for s in re.findall(r"[^\s-]+-?", text["text"]))
    first = text["offset"] - 1 if isinstance(text.get("offset"), int) and text["offset"] >= 1 else 0
    return [[next(syllables, None) if number >= first and pitch is not None and not tied else None
             for _, pitch, tied in beats] for number, beats in enumerate(bars)]


def search(query: str) -> str:
    """Songsterr's page of the tabs it has for this."""
    return f"https://www.songsterr.com/?pattern={quote(query)}"


def bookmarklet(port: int = PORT) -> str:
    """The bookmark, clicked on a song's tab on songsterr.com: it opens a small window of Woodshed's page with where
    the tab is, which hands it on, and closes once the tab is found (a window a script opened may close itself)."""
    script = """(function () {
  var page = 'http://127.0.0.1:%d/';
  var popup = null;
  function fail(why) { if (popup) popup.close(); alert('Woodshed: ' + why); }
  var ids = location.pathname.match(/-s(\\d+)/g);
  if (!/(^|\\.)songsterr\\.com$/.test(location.hostname) || !ids) {
    return fail('open a song\\u2019s tab on songsterr.com, then click this bookmark.');
  }
  var id = Number(ids[ids.length - 1].slice(2));
  /* Opened now, on the click: a window opened later, once the page is read again, would be blocked. */
  popup = window.open('', 'woodshed', 'popup,width=480,height=260');
  function read(doc) {
    try {
      var current = JSON.parse(doc.getElementById('state').textContent).meta.current;
      return current && current.songId === id ? current : null;
    } catch (e) { return null; }
  }
  function send(c) {
    var tab = JSON.stringify({songId: c.songId, revisionId: c.revisionId, image: c.image, title: c.title,
      artist: c.artist, url: location.origin + location.pathname,
      tracks: (c.tracks || []).map(function (t) {
        return {partId: t.partId, instrumentId: t.instrumentId, name: t.title || t.name, tuning: t.tuning,
                isDrums: t.isDrums};
      })});
    var to = page + '#' + encodeURIComponent(tab);
    if (popup) popup.location.href = to; else location.href = to;
  }
  var current = read(document);
  if (current) return send(current);
  /* Songsterr changes song without reloading: the page then describes the first one. */
  var again = function () { fail('couldn\\u2019t read this tab: reload the page, and click again.'); };
  fetch(location.href, {credentials: 'include'}).then(function (r) { return r.text(); }).then(function (html) {
    var fresh = read(new DOMParser().parseFromString(html, 'text/html'));
    fresh ? send(fresh) : again();
  }).catch(again);
})();""" % port
    return "javascript:" + quote(re.sub(r"\s*\n\s*", " ", script), safe="")


_PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Woodshed: a tab from Songsterr</title>
<style>
:root { color-scheme: light dark; --fg: #1d1d1f; --mut: #6e6e73; --bg: #fafafa; --card: #fff; --bd: #ddd;
        --acc: #2456c7; --ok: #0a7a2f; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8e8ea; --mut: #9a9aa0; --bg: #141416; --card: #1e1e22;
        --bd: #33333a; --acc: #7ea2ff; --ok: #5fd27f; --bad: #ff7a70; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width: 40rem; margin: 0 auto; padding: 2rem 1rem; }
h1 { font-size: 1.4rem; margin: 0 0 1rem; }
ol { padding-left: 1.3rem; } li { margin: .6rem 0; } a { color: var(--acc); }
.bookmark { display: inline-block; padding: .3rem .8rem; border: 1px solid var(--bd); border-radius: 999px;
            background: var(--card); color: var(--fg); text-decoration: none; font-weight: 600; cursor: grab; }
#status { margin-top: 1.5rem; padding: 1rem; border: 1px solid var(--bd); border-radius: .6rem;
          background: var(--card); }
#status.ok { border-color: var(--ok); } #status.bad { border-color: var(--bad); }
.ok { color: var(--ok); } .bad { color: var(--bad); } .mut { color: var(--mut); font-size: .9rem; }
textarea { width: 100%; box-sizing: border-box; font: 12px ui-monospace, Menlo, monospace; }
body.sending h1, body.sending ol { display: none; }
body.sending main { padding-top: 1rem; } body.sending #status { margin-top: 0; }
</style>
<main>
<h1>A tab of “{song}” from Songsterr</h1>
<ol>
<li>The first time: drag this to your bookmarks bar ({bar}):
  <a class="bookmark" href="{bookmarklet}">To Woodshed</a>
  <details class="mut"><summary>Or add it by hand</summary>
  Make a bookmark of any page, then edit it: name it <b>To Woodshed</b>, and paste this as its address.
  <textarea rows="4" readonly onclick="this.select()">{bookmarklet}</textarea></details></li>
<li>Open the song’s tab on Songsterr: <a href="{search}" target="_blank" rel="noopener">find “{query}”</a>.
  One with the singer’s track (often named after them, or “Vocals”).</li>
<li>Click the <b>To Woodshed</b> bookmark there.</li>
</ol>
<div id="status"></div>
</main>
<script>
var box = document.getElementById('status');
function say(text, kind, note) {
  box.className = kind || '';
  box.textContent = text;
  if (note) {
    var small = document.createElement('div');
    small.className = 'mut';
    small.textContent = note;
    box.appendChild(small);
  }
}
var GONE = 'Woodshed isn’t waiting for a tab anymore: run shed reference <song> --songsterr again.';

/* Opened by the bookmark, with the tab after the #: hand it on, then close once it's found. */
function send() {
  var sent = location.hash.slice(1);
  if (!sent) return false;
  document.body.className = 'sending';
  history.replaceState(null, '', location.pathname);
  var tab;
  try { tab = JSON.parse(decodeURIComponent(sent)); } catch (e) { say('That isn’t a tab.', 'bad'); return true; }
  say('Reading “' + (tab.title || 'the tab') + '”…');
  fetch('/tab', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(tab)})
    .then(function (r) { return r.json(); })
    .then(function (answer) {
      say(answer.message, answer.ok ? 'ok' : 'bad', answer.ok ? '' : 'Click the bookmark on another tab of it.');
      if (answer.ok) setTimeout(function () { window.close(); }, 1500);
    })
    .catch(function () { say(GONE, 'bad'); });
  return true;
}

/* Opened by shed: follow what becomes of the tabs the bookmark sends, and close once one is found. */
function follow() {
  say('Waiting for the tab…', '', 'Woodshed stops waiting when you press Ctrl+C in the terminal.');
  var shown = '';
  function check() {
    fetch('/status').then(function (r) { return r.json(); }).then(function (now) {
      if (now.message !== shown) {
        shown = now.message;
        if (now.state === 'reading') say(now.message);
        else if (now.state === 'failed') say(now.message, 'bad', 'Still waiting: click the bookmark on another tab of it.');
        else if (now.state === 'done') say(now.message, 'ok', 'This page closes itself.');
      }
      if (now.state !== 'done') return setTimeout(check, 500);
      setTimeout(function () {
        window.close();
        say(now.message, 'ok', 'You can close this page.');  // when the browser doesn't let it close
      }, 2500);
    }).catch(function () { say(GONE, 'bad'); });
  }
  check();
}

window.addEventListener('hashchange', send);  // the bookmark's window, sent another tab
if (!send()) follow();
</script>
"""


def page(song_name: str, query: str, port: int = PORT) -> str:
    return (_PAGE.replace("{song}", escape(song_name)).replace("{query}", escape(query))
            .replace("{search}", escape(search(query))).replace("{bookmarklet}", escape(bookmarklet(port)))
            .replace("{bar}", "⌘⇧B shows it"))


class Waiting(ThreadingHTTPServer):
    """Woodshed's page while it waits for a tab: what the bookmark sends goes to `received(sent)`, which answers
    (whether it was laid over the recording, what to say); once one is, it stops, when the page that follows it
    has been told (or after LINGER seconds, if none is open)."""

    daemon_threads = True
    LINGER = 3.0

    def __init__(self, port: int, shown: str, received: Callable[[object], tuple[bool, str]]):
        super().__init__(("127.0.0.1", port), _Handler)
        self.shown, self.received, self.done = shown, received, False
        self.state = {"state": "waiting", "message": ""}  # what /status says: "reading", "failed" or "done" next
        self._one = threading.Lock()  # a tab at a time
        self._told = threading.Event()

    def server_bind(self) -> None:  # not HTTPServer's, which looks this Mac's name up (socket.getfqdn): slow, unneeded
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:
        pass

    def receive(self, sent) -> tuple[bool, str]:
        with self._one:
            if self.done:
                return True, self.state["message"]
            title = sent.get("title") if isinstance(sent, dict) and isinstance(sent.get("title"), str) else ""
            self.state = {"state": "reading", "message": f"Reading “{title[:300] or 'the tab'}”… (see the terminal)"}
            ok, message = self.received(sent)
            self.state = {"state": "done" if ok else "failed", "message": message}
            if ok:
                self.done = True
                threading.Thread(target=self._stop, daemon=True).start()
            return ok, message

    def status(self) -> dict:
        state = dict(self.state)
        if state["state"] == "done":
            self._told.set()
        return state

    def _stop(self) -> None:
        self._told.wait(self.LINGER)
        self.shutdown()


class _Handler(BaseHTTPRequestHandler):
    server: Waiting

    def log_message(self, format, *args) -> None:
        pass

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _local(self) -> bool:
        """Whether the request is for this page by its address (a site whose name was made to lead here isn't)."""
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if not self._local() or path not in ("/", "/status"):
            return self._reply(404, "text/plain", b"Not found")
        if path == "/status":
            return self._reply(200, "application/json", json.dumps(self.server.status()).encode())
        self._reply(200, "text/html; charset=utf-8", self.server.shown.encode())

    def do_POST(self) -> None:
        # Another site's page can't send JSON here: that needs a permission this page never gives.
        if not self._local() or self.path != "/tab" or self.headers.get_content_type() != "application/json":
            return self._reply(404, "text/plain", b"Not found")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            sent = json.loads(self.rfile.read(length)) if 0 < length <= 1 << 20 else None
        except ValueError:
            sent = None
        ok, message = self.server.receive(sent)
        self._reply(200, "application/json", json.dumps({"ok": ok, "message": message}).encode())
