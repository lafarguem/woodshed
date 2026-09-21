"""The library on disk: one folder per song, one mp3 per take.

Folders are the source of truth. Each take carries its transcript and rating in its own ID3 tags,
so moving a file to another folder (e.g. in Finder) re-files it, and renaming a folder
renames the song.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mutagen import MutagenError
from mutagen.id3 import ID3, TALB, TDRC, TIT2, TXXX

from woodshed import audio
from woodshed.lyrics import bigrams, containment, words
from woodshed.rating import Metrics

UNSORTED = "Unsorted"
_INCOMING = ".incoming"
_STAMP = "%Y-%m-%d_%H-%M"
_TRANSCRIPT = "woodshed:transcript"
_GENIUS_ID = "woodshed:genius_id"
_ARTIST = "woodshed:original_artist"
_METRICS = "woodshed:metrics"

# Tuned on simulated transcripts (see tests/test_lyrics.py): the same song scores
# 0.2-0.6 even with a third of the words misheard; different songs stay under 0.12.
MIN_WORD_PAIRS = 8
MATCH_SCORE = 0.15
MATCH_MARGIN = 0.10


@dataclass
class Take:
    path: Path
    song: str
    transcript: str
    genius_id: int | None
    metrics: Metrics | None = None  # None until the take is rated


def folder_name(name: str) -> str:
    return " ".join(re.sub(r"[/:\\]", "-", name).split()).strip(" .")


def take_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem[:16], _STAMP)
    except ValueError:
        return None


def _read(path: Path, song: str) -> Take:
    try:
        tags = ID3(path)
    except MutagenError:
        return Take(path, song, "", None)

    def field(desc: str) -> str:
        frames = tags.getall(f"TXXX:{desc}")
        return frames[0].text[0] if frames else ""

    genius_id = field(_GENIUS_ID)
    metrics = Metrics.from_json(field(_METRICS)) if field(_METRICS) else None
    return Take(path, song, field(_TRANSCRIPT), int(genius_id) if genius_id.isdigit() else None, metrics)


def is_confident(lines: list[str], ranked: list[tuple[str, float]]) -> bool:
    """Enough lyrics were heard, and one song clearly matches them."""
    if len(bigrams(words(" ".join(lines)))) < MIN_WORD_PAIRS or not ranked:
        return False
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][1] >= MATCH_SCORE and ranked[0][1] - runner_up >= MATCH_MARGIN


class Library:
    def __init__(self, root: Path):
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def incoming(self) -> Path:
        """Where raw recordings wait until they're filed."""
        path = self.root / _INCOMING
        path.mkdir(exist_ok=True)
        return path

    def songs(self) -> dict[str, list[Path]]:
        """Song name -> its takes, oldest first."""
        folders = [p for p in self.root.iterdir()
                   if p.is_dir() and not p.name.startswith(".") and p.name != UNSORTED]
        return {f.name: sorted(f.glob("*.mp3")) for f in sorted(folders, key=lambda p: p.name.casefold())}

    def unsorted(self) -> list[Path]:
        return sorted((self.root / UNSORTED).glob("*.mp3"))

    def takes(self) -> list[Take]:
        return [_read(path, song) for song, paths in self.songs().items() for path in paths]

    def takes_of(self, song: str) -> list[Take]:
        """A song's takes, oldest first."""
        return [_read(path, song) for path in self.songs().get(song, [])]

    def save_metrics(self, path: Path, metrics: Metrics) -> None:
        tags = ID3(path)
        tags.setall(f"TXXX:{_METRICS}", [TXXX(encoding=3, desc=_METRICS, text=metrics.to_json())])
        tags.save(path)

    def find_song(self, name: str) -> str | None:
        wanted = folder_name(name).casefold()
        return next((song for song in self.songs() if song.casefold() == wanted), None)

    def song_for_genius_id(self, genius_id: int) -> str | None:
        return next((t.song for t in self.takes() if t.genius_id == genius_id), None)

    def match(self, lines: list[str]) -> list[tuple[str, float]]:
        """Your songs, ranked by how many of this take's lyrics they share."""
        take = bigrams(words(" ".join(lines)))
        references: dict[str, set] = {}
        for t in self.takes():
            references.setdefault(t.song, set()).update(bigrams(words(t.transcript)))
        return sorted(((song, containment(take, ref)) for song, ref in references.items()),
                      key=lambda s: s[1], reverse=True)

    def add_take(self, src: Path, song: str | None, start: float, end: float, recorded: datetime,
                 lines: list[str], genius_id: int | None = None, artist: str | None = None,
                 metrics: Metrics | None = None) -> Path:
        """Encode the playing part of `src` into the song's folder (or Unsorted) as <date>_<time>.mp3."""
        folder = self.root / (song or UNSORTED)
        folder.mkdir(exist_ok=True)
        dest = folder / f"{recorded:{_STAMP}}.mp3"
        n = 2
        while dest.exists():
            dest = folder / f"{recorded:{_STAMP}}_{n}.mp3"
            n += 1
        audio.export_mp3(src, dest, start, end)

        tags = ID3()
        tags.add(TIT2(encoding=3, text=f"{song or 'Unsorted take'} ({recorded:%Y-%m-%d})"))
        tags.add(TALB(encoding=3, text=song or UNSORTED))
        tags.add(TDRC(encoding=3, text=f"{recorded:%Y-%m-%dT%H:%M}"))
        tags.add(TXXX(encoding=3, desc=_TRANSCRIPT, text="\n".join(lines)))
        if genius_id:
            tags.add(TXXX(encoding=3, desc=_GENIUS_ID, text=str(genius_id)))
        if artist:
            tags.add(TXXX(encoding=3, desc=_ARTIST, text=artist))
        if metrics:
            tags.add(TXXX(encoding=3, desc=_METRICS, text=metrics.to_json()))
        tags.save(dest)
        return dest
