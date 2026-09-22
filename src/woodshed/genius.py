"""Recognizing a song you haven't recorded before by searching Genius for its lyrics."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace

from woodshed.lyrics import words

API = "https://api.genius.com/search"
SONGS = "https://api.genius.com/songs"
MAX_QUERIES = 5

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
