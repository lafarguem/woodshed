"""Finding a song's original recording on YouTube, through yt-dlp if you've installed it (brew install yt-dlp).

It's only used when you ask (`shed reference`), and only downloads the video you pick. The download is
deleted once its melody has been followed: what's kept, in the song's folder, is the melody and the recording as
two mp3s to sing along to, its singer and the rest (library.set_reference()). yt-dlp is left to
you, not bundled, because YouTube keeps changing and yt-dlp needs updating every few weeks to keep up.

YouTube sometimes turns away a video's audio (HTTP 403) that it serves a moment later: the addresses yt-dlp is
given for it can be refused, and asking again gets new ones. So a download refused that way, or cut off, is tried
again (_PAUSES); an older yt-dlp is refused far more often, and is told to be updated.
"""

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class YouTubeError(Exception):
    pass


@dataclass
class Video:
    url: str
    title: str
    channel: str
    seconds: float | None  # None when YouTube doesn't say (a live stream, say)
    description: str = ""  # its start, as YouTube's search results show it


def available() -> bool:
    return shutil.which("yt-dlp") is not None


def _run(args: list[str]) -> str:
    try:
        done = subprocess.run(["yt-dlp", "--no-warnings", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise YouTubeError("yt-dlp is not installed (brew install yt-dlp)") from None
    if done.returncode:
        lines = done.stderr.strip().splitlines()
        raise YouTubeError(lines[-1].removeprefix("ERROR: ") if lines else f"yt-dlp failed ({done.returncode})")
    return done.stdout


def search(query: str, results: int = 5) -> list[Video]:
    """YouTube's first videos for `query`."""
    videos = []
    for line in _run(["--flat-playlist", "--dump-json", f"ytsearch{results}:{query}"]).splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if url := entry.get("webpage_url") or entry.get("url"):
            videos.append(Video(url, entry.get("title") or url, entry.get("channel") or entry.get("uploader") or "",
                                entry.get("duration"), " ".join((entry.get("description") or "").split())))
    return videos


_PAUSES = (3.0, 10.0)  # seconds before each download tried again
_PASSING = re.compile(r"HTTP Error (403|429|5\d\d)|timed out|Connection (reset|aborted)|IncompleteRead|Read timed",
                      re.I)  # what going again can get past (not a video gone or private)


def download(url: str, folder: Path) -> tuple[Path, str]:
    """The video's audio, saved into `folder` (as it is on YouTube: no re-encoding), and the video's title."""
    for pause in (*_PAUSES, None):
        for left in folder.glob("original.*"):  # a part of the last try
            left.unlink()
        try:
            title = _run(["--no-playlist", "--no-simulate", "--print", "title", "--no-progress",
                          "-f", "bestaudio[ext=m4a]/bestaudio", "-o", str(folder / "original.%(ext)s"), url]).strip()
            break
        except YouTubeError as e:
            if not _PASSING.search(str(e)):
                raise
            if pause is None:
                hint = " (an older yt-dlp is often turned away: brew upgrade yt-dlp)" if "403" in str(e) else ""
                raise YouTubeError(f"{e}, {len(_PAUSES) + 1} times{hint}") from None
            time.sleep(pause)
    saved = [p for p in folder.glob("original.*") if p.suffix != ".part"]
    if not saved:
        raise YouTubeError("yt-dlp didn't save the audio")
    return saved[0], title or url
