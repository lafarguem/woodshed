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
