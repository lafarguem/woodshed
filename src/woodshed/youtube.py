"""Finding a song's original recording on YouTube, through yt-dlp if you've installed it (brew install yt-dlp).

It's only used when you ask (`shed reference`), and only downloads the video you pick. The recording is
deleted once its melody has been followed: woodshed keeps the melody, not the recording. yt-dlp is left to
you, not bundled, because YouTube keeps changing and yt-dlp needs updating every few weeks to keep up.
"""

import json
import shutil
import subprocess
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


def download(url: str, folder: Path) -> tuple[Path, str]:
    """The video's audio, saved into `folder` (as it is on YouTube: no re-encoding), and the video's title."""
    title = _run(["--no-playlist", "--no-simulate", "--print", "title", "--no-progress", "-f", "bestaudio[ext=m4a]/bestaudio",
                  "-o", str(folder / "original.%(ext)s"), url]).strip()
    saved = [p for p in folder.glob("original.*") if p.suffix != ".part"]
    if not saved:
        raise YouTubeError("yt-dlp didn't save the audio")
    return saved[0], title or url
