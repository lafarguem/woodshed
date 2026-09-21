"""Reacting to single key presses while something runs (recording, playback)."""

import contextlib
import os
import select
import sys
import termios
import tty
from collections.abc import Callable, Iterator


@contextlib.contextmanager
def keypress() -> Iterator[Callable[[], bool]]:
    """Yield a function telling whether a key was pressed, without echoing it."""
    if not sys.stdin.isatty():
        yield lambda: False
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # Ctrl+C still works in cbreak mode

    def pressed() -> bool:
        if select.select([fd], [], [], 0)[0]:
            os.read(fd, 1024)
            return True
        return False

    try:
        yield pressed
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


_KEYS = {
    b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
    b"\x1bOA": "up", b"\x1bOB": "down", b"\x1bOC": "right", b"\x1bOD": "left",  # some terminals' arrow codes
    b"\n": "enter", b"\r": "enter", b"\x1b": "escape",
}


@contextlib.contextmanager
def keys() -> Iterator[Callable[[], str]]:
    """Yield a function that waits for the next key press and names it:
    "up", "down", "left", "right", "enter", "escape", or the character typed."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def next_key() -> str:
        data = os.read(fd, 1)
        if data == b"\x1b" and select.select([fd], [], [], 0.05)[0]:  # an arrow key, not Escape alone
            data += os.read(fd, 2)
        return _KEYS.get(data, data.decode(errors="ignore"))

    try:
        yield next_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
