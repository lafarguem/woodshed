"""Decoding, encoding and silence trimming, all through ffmpeg."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what Whisper expects

_FRAME = SAMPLE_RATE // 20  # 50 ms


class AudioError(Exception):
    pass


def _run(cmd: list[str]) -> bytes:
    try:
        return subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise AudioError("ffmpeg is not installed (brew install ffmpeg)") from None
    except subprocess.CalledProcessError as e:
        raise AudioError(e.stderr.decode(errors="replace").strip()) from None


def load(path: Path, rate: int = SAMPLE_RATE, channels: int = 1,
         start: float = 0.0, seconds: float | None = None) -> np.ndarray:
    """Decode any audio file to float32 samples, shaped (channels, n) when channels > 1."""
    window = ["-ss", f"{start:.2f}"] + (["-t", f"{seconds:.2f}"] if seconds else [])
    out = _run(["ffmpeg", "-nostdin", "-v", "error", *window, "-i", str(path),
                "-f", "f32le", "-ac", str(channels), "-ar", str(rate), "-"])
    samples = np.frombuffer(out, np.float32)
    return samples if channels == 1 else samples.reshape(-1, channels).T.copy()


def playing_bounds(samples: np.ndarray, pad: float = 1.5) -> tuple[float, float]:
    """Start and end (seconds) of the part where you're actually playing.

    A frame counts as playing when it is within 30 dB of the take's loud parts and
    most of the surrounding half second is too, so key clicks don't count.
    """
    n = len(samples) // _FRAME
    if n == 0:
        return 0.0, 0.0
    frames = samples[: n * _FRAME].reshape(n, _FRAME)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    loud = db > max(-55.0, np.percentile(db, 95) - 30.0)
    sustained = np.convolve(loud, np.ones(10) / 10, mode="same") >= 0.6
    idx = np.flatnonzero(sustained)
    if idx.size == 0:
        return 0.0, 0.0
    total = len(samples) / SAMPLE_RATE
    start = max(0.0, idx[0] * _FRAME / SAMPLE_RATE - pad)
    end = min(total, (idx[-1] + 1) * _FRAME / SAMPLE_RATE + pad)
    return start, end


def export_mp3(src: Path, dst: Path, start: float, end: float) -> None:
    _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
          "-i", str(src), "-vn", "-map_metadata", "-1", "-codec:a", "libmp3lame", "-q:a", "2", str(dst)])


def recorded_at(path: Path) -> datetime:
    """When a file was recorded: its embedded creation time (e.g. Voice Memos), else its file date."""
    try:
        out = _run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "json", str(path)])
        stamp = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
        if stamp:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    except (AudioError, ValueError):
        pass
    st = path.stat()
    return datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))
