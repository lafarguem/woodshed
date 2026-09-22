"""youtube.py against a stand-in for yt-dlp: what it asks yt-dlp, and how it reads the answers."""

import json
import os
import sys

import pytest

from woodshed import youtube
from woodshed.youtube import Video

FAKE_YT_DLP = """#!{python}
import json, sys
args = sys.argv[1:]
with open({log!r}, "a") as log:
    log.write(json.dumps(args) + "\\n")
if any(a.startswith("ytsearch") for a in args):
    print(json.dumps({{"url": "https://www.youtube.com/watch?v=a1", "title": "Harbor Lights (Official Audio)",
                      "channel": "The Originals", "duration": 201}}))
    print("[youtube:search] not a video")
    print(json.dumps({{"url": "https://www.youtube.com/watch?v=b2", "title": "Harbor Lights (Live)", "uploader": "A Fan",
                      "duration": None}}))
elif args[-1].endswith("gone"):
    print("ERROR: [youtube] gone: Video unavailable", file=sys.stderr)
    sys.exit(1)
else:
    with open(args[args.index("-o") + 1].replace("%(ext)s", "webm"), "wb") as audio:
        audio.write(b"audio")
    print("Harbor Lights (Official Audio)")
"""


@pytest.fixture
def yt_dlp(tmp_path, monkeypatch):
    """A yt-dlp on the PATH that answers like the real one; returns the arguments of each call."""
    folder, log = tmp_path / "bin", tmp_path / "calls.jsonl"
    folder.mkdir()
    program = folder / "yt-dlp"
    program.write_text(FAKE_YT_DLP.format(python=sys.executable, log=str(log)))
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    return lambda: [json.loads(line) for line in log.read_text().splitlines()]


def test_videos_are_found_through_yt_dlp(yt_dlp):
    assert youtube.available()
    assert youtube.search("harbor lights official audio") == [
        Video("https://www.youtube.com/watch?v=a1", "Harbor Lights (Official Audio)", "The Originals", 201),
        Video("https://www.youtube.com/watch?v=b2", "Harbor Lights (Live)", "A Fan", None)]
    assert yt_dlp()[-1][-1] == "ytsearch5:harbor lights official audio"


def test_a_videos_audio_is_downloaded_as_it_is(yt_dlp, tmp_path):
    audio, title = youtube.download("https://www.youtube.com/watch?v=a1", tmp_path)

    assert (audio.name, audio.read_bytes(), title) == ("original.webm", b"audio", "Harbor Lights (Official Audio)")
    args = yt_dlp()[-1]
    assert "--no-playlist" in args and args[args.index("-f") + 1] == "bestaudio[ext=m4a]/bestaudio"


def test_what_went_wrong_is_said(yt_dlp, tmp_path):
    with pytest.raises(youtube.YouTubeError, match="^\\[youtube\\] gone: Video unavailable$"):
        youtube.download("https://www.youtube.com/watch?v=gone", tmp_path)


def test_without_yt_dlp(monkeypatch):
    monkeypatch.setenv("PATH", "/nowhere")
    assert not youtube.available()
    with pytest.raises(youtube.YouTubeError, match="brew install yt-dlp"):
        youtube.search("harbor lights")
