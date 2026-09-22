"""The library on disk: one folder per song, one mp3 (or Apple Lossless m4a) per take.

Folders are the source of truth. Each take carries its transcript and rating in its own tags,
so moving a file to another folder (e.g. in Finder) re-files it, and renaming a folder
renames the song. A song's reference melody (`shed reference`) is kept in its folder too.
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mutagen import MutagenError
from mutagen.id3 import ID3, TALB, TDRC, TIT2, TXXX, ID3NoHeaderError
from mutagen.mp4 import MP4, AtomDataType, MP4FreeForm

from woodshed import audio
from woodshed.lyrics import bigrams, containment, words
from woodshed.melody import Melody
from woodshed.rating import Metrics

UNSORTED = "Unsorted"
FORMATS = ("mp3", "m4a")  # how takes are saved: mp3, or Apple Lossless (bigger, but exactly as recorded)
_INCOMING = ".incoming"
RAW_STAMP = "%Y-%m-%d_%H-%M-%S"  # how raw recordings waiting in .incoming/ are named: when they started
_STAMP = "%Y-%m-%d_%H-%M"
_TRANSCRIPT = "woodshed:transcript"
_GENIUS_ID = "woodshed:genius_id"
_ARTIST = "woodshed:original_artist"
_METRICS = "woodshed:metrics"
_SOURCE = "woodshed:source"  # fingerprint of the file the take was made from, so it's never filed twice
_MELODY = "woodshed:melody"  # its notes, words and chords, to compare it with a reference melody
_REFERENCE = ".reference.json"  # in a song's folder: the melody its takes are rated against
_SKIPPED = ".skipped.json"  # recordings found in folders and passed over, so they aren't checked again

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
    source: str | None = None  # see fingerprint()
    melody: Melody | None = None  # None until it's needed (and read only when asked for: it's long)
    artist: str | None = None  # the original's, when Genius recognized the song


@dataclass
class Reference:
    """The recording a song's takes are rated against."""

    melody: Melody | None  # None if it was analyzed by an older version, so it needs setting again
    file: str  # the recording, as it was given: a file, or a link it was downloaded from
    added: datetime
    title: str | None = None  # a download's title

    def __str__(self) -> str:
        return f"{self.title} ({self.file})" if self.title else self.file


class _Tags:
    """A take's own tags: ID3 in an mp3, iTunes-style atoms in an m4a."""

    def __init__(self, path: Path):
        self.path, self.mp4 = path, path.suffix.lower() == ".m4a"
        if self.mp4:
            self.file = MP4(path)
            if self.file.tags is None:
                self.file.add_tags()
        else:
            try:
                self.file = ID3(path)
            except ID3NoHeaderError:  # a file that's never been tagged
                self.file = ID3()

    def __getitem__(self, desc: str) -> str:
        if self.mp4:
            values = self.file.tags.get(f"----:com.apple.iTunes:{desc}")
            return bytes(values[0]).decode() if values else ""
        frames = self.file.getall(f"TXXX:{desc}")
        return frames[0].text[0] if frames else ""

    def __setitem__(self, desc: str, text: str) -> None:
        if self.mp4:
            self.file.tags[f"----:com.apple.iTunes:{desc}"] = [MP4FreeForm(text.encode(), AtomDataType.UTF8)]
        else:
            self.file.setall(f"TXXX:{desc}", [TXXX(encoding=3, desc=desc, text=text)])

    def describe(self, title: str, album: str, date: str) -> None:
        """What music apps show: the title, album and date."""
        if self.mp4:
            self.file.tags["©nam"], self.file.tags["©alb"], self.file.tags["©day"] = [title], [album], [date]
        else:
            self.file.setall("TIT2", [TIT2(encoding=3, text=title)])
            self.file.setall("TALB", [TALB(encoding=3, text=album)])
            self.file.setall("TDRC", [TDRC(encoding=3, text=date)])

    def save(self) -> None:
        self.file.save() if self.mp4 else self.file.save(self.path)


def is_take(path: Path) -> bool:
    return path.suffix.lower() in (".mp3", ".m4a") and not path.name.startswith(".")


def folder_name(name: str) -> str:
    return " ".join(re.sub(r"[/:\\]", "-", name).split()).strip(" .")


def take_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem[:16], _STAMP)
    except ValueError:
        return None


def _read(path: Path, song: str, melody: bool = False) -> Take:
    try:
        tags = _Tags(path)
    except MutagenError:
        return Take(path, song, "", None)

    genius_id = tags[_GENIUS_ID]
    metrics = Metrics.from_json(tags[_METRICS]) if tags[_METRICS] else None
    return Take(path, song, tags[_TRANSCRIPT], int(genius_id) if genius_id.isdigit() else None, metrics,
                tags[_SOURCE] or None, Melody.from_json(tags[_MELODY]) if melody and tags[_MELODY] else None,
                tags[_ARTIST] or None)


def fingerprint(path: Path) -> str:
    """Tells a recording apart from any other, whatever it's called or wherever it's moved."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def is_confident(lines: list[str], ranked: list[tuple[str, float]]) -> bool:
    """Enough lyrics were heard, and one song clearly matches them."""
    if len(bigrams(words(" ".join(lines)))) < MIN_WORD_PAIRS or not ranked:
        return False
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][1] >= MATCH_SCORE and ranked[0][1] - runner_up >= MATCH_MARGIN


class Library:
    def __init__(self, root: Path, take_format: str = "mp3"):
        self.root = root.expanduser()  # created by the first take filed there, not by reading it
        self.suffix = f".{take_format}"  # how new takes are saved: see FORMATS

    @property
    def incoming(self) -> Path:
        """Where raw recordings wait until they're filed."""
        path = self.root / _INCOMING
        path.mkdir(parents=True, exist_ok=True)
        return path

    def songs(self) -> dict[str, list[Path]]:
        """Song name -> its takes, oldest first."""
        if not self.root.is_dir():
            return {}
        folders = [p for p in self.root.iterdir()
                   if p.is_dir() and not p.name.startswith(".") and p.name != UNSORTED]
        return {f.name: _takes_in(f) for f in sorted(folders, key=lambda p: p.name.casefold())}

    def unsorted(self) -> list[Path]:
        return _takes_in(self.root / UNSORTED)

    def raw_path(self, started: datetime, chosen: str = "") -> Path:
        """A name for a new raw recording, among those waiting to be filed. `chosen` is the format it's to be saved
        in (".m4a"), if not the usual one. Until it's finished, it goes by the same name with the suffix
        .recording, so that it isn't taken for one waiting to be filed."""
        raw, n = self.incoming / f"{started:{RAW_STAMP}}{chosen}.wav", 2
        while raw.exists() or raw.with_suffix(".recording").exists():
            raw, n = self.incoming / f"{started:{RAW_STAMP}}_{n}{chosen}.wav", n + 1
        return raw

    def waiting(self) -> list[Path]:
        """Raw recordings that couldn't be filed (e.g. filing crashed), oldest first."""
        return sorted((self.root / _INCOMING).glob("*.wav"))

    def sources(self) -> dict[str, Path]:
        """The fingerprint of the file each take was made from -> the take (for takes that kept one)."""
        paths = [path for paths in self.songs().values() for path in paths] + self.unsorted()
        return {take.source: take.path for take in (_read(path, "") for path in paths) if take.source}

    def skipped(self) -> dict[str, str]:
        """Fingerprints of recordings passed over in folders -> why (e.g. "no singing")."""
        try:
            return {source: entry["reason"] for source, entry in json.loads((self.root / _SKIPPED).read_text()).items()}
        except (FileNotFoundError, ValueError, KeyError, TypeError, AttributeError):
            return {}

    def remember_skipped(self, source: str, path: Path, reason: str) -> None:
        file = self.root / _SKIPPED
        try:
            entries = json.loads(file.read_text())
        except (FileNotFoundError, ValueError):
            entries = {}
        entries[source] = {"reason": reason, "file": str(path)}
        self.root.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(entries, ensure_ascii=False, indent=1))

    def takes(self) -> list[Take]:
        return [_read(path, song) for song, paths in self.songs().items() for path in paths]

    def takes_of(self, song: str, melody: bool = False) -> list[Take]:
        """A song's takes, oldest first; with their melodies if asked."""
        return [_read(path, song, melody) for path in self.songs().get(song, [])]

    def save_metrics(self, path: Path, metrics: Metrics) -> None:
        self._save_tag(path, _METRICS, metrics.to_json())

    def save_melody(self, path: Path, melody: Melody) -> None:
        self._save_tag(path, _MELODY, melody.to_json())

    def save_artist(self, song: str, artist: str) -> None:
        """Who sings the song's original, on each of its takes."""
        for path in self.songs().get(song, []):
            self._save_tag(path, _ARTIST, artist)

    def _save_tag(self, path: Path, desc: str, text: str) -> None:
        tags = _Tags(path)
        tags[desc] = text
        tags.save()

    def has_reference(self, song: str) -> bool:
        return (self.root / song / _REFERENCE).is_file()

    def reference(self, song: str) -> Reference | None:
        try:
            data = json.loads((self.root / song / _REFERENCE).read_text())
            return Reference(Melody.from_json(json.dumps(data["melody"])), data["file"],
                             datetime.fromisoformat(data["added"]), data.get("title"))
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return None

    def set_reference(self, song: str, melody: Melody, file: str, title: str | None = None) -> None:
        data = {"file": file, "added": f"{datetime.now():%Y-%m-%dT%H:%M}", "melody": json.loads(melody.to_json())}
        if title:
            data["title"] = title
        (self.root / song / _REFERENCE).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    def remove_reference(self, song: str) -> bool:
        """Whether there was one."""
        try:
            (self.root / song / _REFERENCE).unlink()
        except FileNotFoundError:
            return False
        return True

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

    def encode(self, src: Path, start: float, end: float, take_format: str | None = None) -> Path:
        """The playing part of `src` (from `start` to `end`, in seconds), encoded the way takes are saved here
        (or in `take_format`), into a new file among the recordings waiting (not one of them: those are wav).
        That file is what gets analyzed, then filed as is by add_take(), so analyzing the take again reads the
        same audio."""
        suffix = f".{take_format}" if take_format else self.suffix
        handle, name = tempfile.mkstemp(prefix="filing-", suffix=suffix, dir=self.incoming)
        os.close(handle)
        audio.export(src, Path(name), start, end)
        return Path(name)

    def add_take(self, take: Path, song: str | None, recorded: datetime, lines: list[str],
                 genius_id: int | None = None, artist: str | None = None, metrics: Metrics | None = None,
                 source: str | None = None, melody: Melody | None = None) -> Path:
        """Tag a take from encode() and move it into the song's folder (or Unsorted), as <date>_<time>.mp3
        (or .m4a)."""
        folder = self.root / (song or UNSORTED)
        folder.mkdir(parents=True, exist_ok=True)
        stem, n = f"{recorded:{_STAMP}}", 2
        while (folder / f"{stem}.mp3").exists() or (folder / f"{stem}.m4a").exists():
            stem, n = f"{recorded:{_STAMP}}_{n}", n + 1

        tags = _Tags(take)
        tags.describe(f"{song or 'Unsorted take'} ({recorded:%Y-%m-%d})", song or UNSORTED, f"{recorded:%Y-%m-%dT%H:%M}")
        tags[_TRANSCRIPT] = "\n".join(lines)
        for desc, value in ((_GENIUS_ID, genius_id and str(genius_id)), (_ARTIST, artist),
                            (_METRICS, metrics and metrics.to_json()), (_SOURCE, source),
                            (_MELODY, melody and melody.to_json())):
            if value:
                tags[desc] = value
        tags.save()
        return take.rename(folder / f"{stem}{take.suffix}")


def _takes_in(folder: Path) -> list[Path]:
    """Oldest first: they're named by date."""
    return sorted(p for p in folder.glob("*") if is_take(p)) if folder.is_dir() else []
