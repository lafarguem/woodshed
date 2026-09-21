"""Microphone capture to a WAV file, with a live level meter."""

import queue
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.text import Text

from woodshed.terminal import keypress

_METER_WIDTH = 30
_QUIET_DB = -45.0  # below this counts as silence for --auto-stop


def _meter(elapsed: float, db: float, auto_stop: float | None) -> Text:
    filled = int(np.clip((db + 60) / 60, 0, 1) * _METER_WIDTH)
    color = "red" if db > -1 else "yellow" if db > -9 else "green"
    minutes, seconds = divmod(int(elapsed), 60)
    text = Text.assemble(
        ("● REC ", "bold red"), f"{minutes:02d}:{seconds:02d}  ",
        ("█" * filled, color), ("░" * (_METER_WIDTH - filled), "dim"),
        ("  press any key to stop", "dim"),
    )
    if auto_stop:
        text.append(f" (or stay quiet {auto_stop:g}s)", "dim")
    return text


def record(dest: Path, console: Console, device: int | str | None = None,
           channels: int = 1, auto_stop: float | None = None) -> None:
    """Record from the microphone into `dest` (16-bit WAV) until a key is pressed."""
    rate = int(sd.query_devices(device, "input")["default_samplerate"])
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        blocks.put(indata.copy())

    def write(wav: wave.Wave_write, block: np.ndarray) -> float:
        wav.writeframes((np.clip(block, -1, 1) * 32767).astype("<i2").tobytes())
        return 10 * np.log10(np.mean(block**2) + 1e-12)

    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        stream = sd.InputStream(device=device, channels=channels, samplerate=rate,
                                dtype="float32", callback=on_audio)
        started = last_sound = time.monotonic()
        heard_anything = False
        with stream, keypress() as pressed, Live(console=console, transient=True, refresh_per_second=15) as live:
            try:
                while not pressed():
                    try:
                        db = write(wav, blocks.get(timeout=0.1))
                    except queue.Empty:
                        continue
                    now = time.monotonic()
                    if db > _QUIET_DB:
                        last_sound, heard_anything = now, True
                    elif auto_stop and heard_anything and now - last_sound > auto_stop:
                        break
                    live.update(_meter(now - started, db, auto_stop))
            except KeyboardInterrupt:
                pass
        while not blocks.empty():
            write(wav, blocks.get_nowait())
