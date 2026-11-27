"""Recognizing a song you haven't recorded before by searching Genius for its lyrics, and reading a song's lyrics
as Genius writes them (lyrics()), to show while singing along (along.py).

Genius's API finds songs but doesn't give their lyrics: those are read from the song's page on genius.com, where
they're the text of the elements marked `data-lyrics-container`, a line ending at each <br>, around a header and
ads marked `data-exclude-from-selection`.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from html.parser import HTMLParser

from woodshed.lyrics import words

API = "https://api.genius.com/search"
SONGS = "https://api.genius.com/songs"
PAGE = "https://genius.com/songs"  # /<id> leads to the song's page
MAX_QUERIES = 5
_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())  # tags that aren't closed

_VERSION_TAG = re.compile(
    r"\s*[(\[][^)\]]*\b(remaster\w*|live|version|edit|mix|acoustic|demo|mono|stereo)\b[^)\]]*[)\]]", re.I)
_ORIGINAL = ("cover_of", "live_version_of")  # how Genius links a song's page to the song it's another version of


class GeniusError(Exception):
    pass


class InvalidToken(GeniusError):
    pass


@dataclass
class Candidate:
    title: str
    artist: str
    genius_id: int
    first_places: int = 0  # searches where this song was the top hit
    points: int = 0


def search(query: str, token: str) -> list[dict]:
    """Songs Genius returns for a query, best match first."""
    body = _get(f"{API}?{urllib.parse.urlencode({'q': query})}", token)
    return [hit["result"] for hit in body["hits"] if hit["type"] == "song"]


def song(genius_id: int, token: str) -> dict:
    """A song's page on Genius, with how it's linked to other songs (the one it's a cover of, say)."""
    return _get(f"{SONGS}/{genius_id}", token)["song"]


def _get(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "woodshed"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)["response"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise InvalidToken("Genius rejected the access token; run `shed init` to change it") from None
        raise GeniusError(f"Genius search failed (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise GeniusError(f"Couldn't reach Genius ({getattr(e, 'reason', e)})") from None


def lyrics(genius_id: int) -> tuple[str, list[str]]:
    """A song's page on genius.com and its lyrics, line by line, without the headings ("[Chorus]"). Raises
    GeniusError if the page can't be got; the lines are empty if it has no lyrics (an instrumental, or a page
    written another way)."""
    request = urllib.request.Request(f"{PAGE}/{int(genius_id)}", headers={"User-Agent": "woodshed"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            url, html = response.url, response.read(8 << 20).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise GeniusError(f"Genius didn't give the lyrics (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise GeniusError(f"Couldn't reach Genius ({getattr(e, 'reason', e)})") from None
    return url, lyrics_in(html)


def lyrics_in(html: str) -> list[str]:
    """The lyrics in a song's page, as lyrics() gives them."""
    page = _Page()
    page.feed(html)
    page.close()
    lines = (" ".join(line.replace("\u200b", "").split()) for line in "".join(page.text).split("\n"))
    return [line for line in lines if line and not re.fullmatch(r"\[[^\]]*\]", line)]


class _Page(HTMLParser):
    """Gathers the text of a page's lyrics."""

    def __init__(self):
        super().__init__()
        self.text: list[str] = []
        self._open: list[str] = []  # each element we're in: "lyrics", "left out" (and all in it), or ""

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            if self._reading():
                self.text.append("\n")
        if tag in _VOID:
            return
        names = {name for name, _ in attrs}
        self._open.append("left out" if "data-exclude-from-selection" in names
                          else "lyrics" if "data-lyrics-container" in names else "")

    def handle_endtag(self, tag):
        if tag not in _VOID and self._open and self._open.pop() == "lyrics":
            self.text.append("\n")  # a line doesn't run on into the next container

    def handle_data(self, data):
        if self._reading():
            self.text.append(data)

    def _reading(self) -> bool:
        return "lyrics" in self._open and "left out" not in self._open


def snippets(lines: list[str], limit: int = MAX_QUERIES) -> list[str]:
    """Distinct chunks of 6-10 words, spread across the take, to search for."""
    chunks: dict[str, str] = {}
    chunk: list[str] = []
    for line in lines:
        for word in line.split():
            chunk.append(word)
            if len(chunk) == 10:
                chunks.setdefault(" ".join(words(" ".join(chunk))), " ".join(chunk))
                chunk = []
        if len(chunk) >= 6:
            chunks.setdefault(" ".join(words(" ".join(chunk))), " ".join(chunk))
            chunk = []
    found = list(chunks.values())
    if len(found) <= limit:
        return found
    return [found[round(i * (len(found) - 1) / (limit - 1))] for i in range(limit)]


def clean_title(title: str) -> str:
    title = title.replace("​", "").replace("\xa0", " ")
    return " ".join(_VERSION_TAG.sub("", title).split())


def _is_song(result: dict) -> bool:
    # Genius also hosts translations ("Genius English Translations") and playlists ("Spotify").
    artist = result["primary_artist"]["name"]
    return not artist.startswith("Genius") and artist != "Spotify"


def identify(lines: list[str], token: str) -> list[Candidate]:
    """Songs matching the transcribed lines, most likely first. The first is by its original artist (see original())."""
    candidates: dict[str, Candidate] = {}
    for query in snippets(lines):
        hits = [r for r in search(query, token) if _is_song(r)][:3]
        for rank, result in enumerate(hits):
            title = clean_title(result["title"])
            # Covers of a song share its title, so they vote together, under whichever came up first.
            c = candidates.setdefault(" ".join(words(title)),
                                      Candidate(title, result["primary_artist"]["name"].strip(), result["id"]))
            c.points += 3 - rank
            c.first_places += rank == 0
    ranked = sorted(candidates.values(), key=lambda c: (c.first_places, c.points), reverse=True)
    if ranked:
        try:
            ranked[0] = original(ranked[0], token)
        except GeniusError:
            pass  # it's still the song you sang, if maybe as someone's cover of it
    return ranked


def original(candidate: Candidate, token: str) -> Candidate:
    """The song `candidate` is a cover or a live version of, when Genius links it to one. Only one of the same
    title, though: a cover in another language is sung to other words, so to Woodshed it's a song of its own."""
    for link in song(candidate.genius_id, token).get("song_relationships") or []:
        if link.get("relationship_type") in _ORIGINAL and len(originals := link.get("songs") or []) == 1:
            if words(clean_title(originals[0]["title"])) == words(candidate.title):
                return replace(candidate, artist=originals[0]["primary_artist"]["name"].strip(),
                               genius_id=originals[0]["id"])
    return candidate


def is_confident(ranked: list[Candidate]) -> bool:
    """The top song came first for at least two lines, and more often than any other."""
    if not ranked or ranked[0].first_places < 2:
        return False
    return len(ranked) == 1 or ranked[1].first_places < ranked[0].first_places
