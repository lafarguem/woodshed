"""Playing takes through afplay, which ships with macOS."""

import subprocess
import time
from pathlib import Path

from mutagen import MutagenError
from mutagen.mp3 import MP3
from rich.console import Console
from rich.live import Live
from rich.text import Text

from woodshed.terminal import keypress

_BAR_WIDTH = 30


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def _progress(elapsed: float, length: float) -> Text:
    done = int(min(elapsed / length, 1) * _BAR_WIDTH) if length else 0
    return Text.assemble(
        f"  {_clock(elapsed)} / {_clock(length)}  ",
        ("━" * done, "green"), ("─" * (_BAR_WIDTH - done), "dim"),
        ("  any key skips, Ctrl+C stops", "dim"),
    )


def play(path: Path, console: Console) -> bool:
    """Play one take until it ends or a key is pressed. False if Ctrl+C stopped playback."""
    try:
        length = MP3(path).info.length
    except MutagenError:
        length = 0.0
    process = subprocess.Popen(["afplay", str(path)])
    started = time.monotonic()
    try:
        with keypress() as pressed, Live(console=console, transient=True, refresh_per_second=4) as live:
            while process.poll() is None and not pressed():
                live.update(_progress(time.monotonic() - started, length))
                time.sleep(0.1)
    except KeyboardInterrupt:
        return False
    finally:
        process.terminate()
        process.wait()
    return True
