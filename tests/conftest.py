import random
import subprocess

import pytest

from woodshed import config
from woodshed.lyrics import words


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    """Keep every test off your own config.toml and WOODSHED_* variables, so how you set up
    Woodshed (e.g. stricter ratings) can't change what the tests see."""
    monkeypatch.setattr(config, "PATH", tmp_path / "settings" / "config.toml")
    for name in ("WOODSHED_DIR", "WOODSHED_DEVICE", "WOODSHED_LANGUAGE", "GENIUS_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)

# Invented lyrics, so tests don't depend on real songs.
SONGS = {
    "Harbor Lights": "the lanterns swing above the harbor wall / we counted ships until the evening fall / "
                     "your coat was wet with salt and rain / and every bell was ringing out your name / "
                     "so sail me home across the silver bay / i kept the light on every single day",
    "Kitchen Floor": "coffee on the stove at half past three / you hum a tune that sounds a lot like me / "
                     "the radio is crackling in the hall / and nobody is waiting for the call / "
                     "dance with me barefoot on the kitchen floor / nobody needs to know what we came for",
    "Gravel Road": "gravel on the highway and the dust in my eyes / a thousand yellow lines under purple skies / "
                   "i left my brother standing at the door / he said you wont be coming back no more / "
                   "drive until the engine starts to cry / drive until the morning says goodbye",
    "Winter Town": "snow is falling slowly on the frozen town / i wear the sweater that you handed down / "
                   "the chimney smoke is drawing letters in the air / i read them all and wish that you were there / "
                   "winter hold me winter let me go / i am tired of the cold and the snow",
}


def mishear(text: str, error: float, rng: random.Random, keep: float = 1.0) -> list[str]:
    """Simulate a Whisper transcript of a sung take: verse sung twice, some words misheard, dropped or added."""
    vocab = sorted({w for t in SONGS.values() for w in words(t)}) + ["yeah", "baby", "gonna"]
    sung = (words(text) * 2)[: int(len(words(text)) * 2 * keep)]
    out = []
    for w in sung:
        r = rng.random()
        if r < error / 2:
            out.append(rng.choice(vocab))
        elif r < error * 0.75:
            continue
        elif r < error:
            out += [w, rng.choice(vocab)]
        else:
            out.append(w)
    return [" ".join(out[i:i + 8]) for i in range(0, len(out), 8)]


@pytest.fixture
def rng():
    return random.Random(7)


@pytest.fixture
def tone(tmp_path):
    """Two seconds of silence, five of a guitar-ish tone, two of silence."""
    path = tmp_path / "take.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=196:sample_rate=44100:duration=5",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
                    "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(path)], check=True)
    return path


SR = 44100


def sung_notes(seconds: float, detune_cents: float, rng: random.Random) -> "np.ndarray":
    """A voice-like tone holding 0.6 s notes of a major scale, each off by ±detune_cents, with vibrato."""
    import numpy as np

    t = np.arange(int(seconds * SR)) / SR
    note = (t // 0.6).astype(int)
    midi = 57 + np.array([0, 2, 4, 5, 7, 9, 11, 12])[note % 8]
    off = np.array([rng.choice((-1, 1)) * detune_cents for _ in range(note.max() + 1)])[note]
    cents = (midi - 69) * 100 + off + 25 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(440 * 2 ** (cents / 1200)) / SR
    return sum(np.sin(k * phase) / k for k in range(1, 12)).astype(np.float32) * 0.2


def strums(seconds: float, bpm: float, speed_up: float = 0.0) -> "np.ndarray":
    """Plucked chords on every eighth note; speed_up is the tempo change from start to end."""
    import numpy as np

    out = np.zeros(int(seconds * SR) + SR, np.float32)
    decay = np.arange(SR) / SR
    pluck = sum(np.sin(2 * np.pi * f * decay) * np.exp(-6 * decay) for f in (110, 165, 220, 277)).astype(np.float32)
    t = 0.2
    while t < seconds:
        i = int(t * SR)
        out[i:i + SR] += pluck
        t += 30 / (bpm * (1 + speed_up * t / seconds))
    return out[: int(seconds * SR)] * 0.1


def ring_out(seconds: float) -> "np.ndarray":
    """The last chord left ringing into room noise: how a take ends (filing keeps 1.5 s past the playing)."""
    import numpy as np

    t = np.arange(int(seconds * SR)) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (110, 165, 220, 277)) * np.exp(-1.2 * t) * 0.1
    return (chord + np.random.default_rng(0).normal(0, 1e-3, t.size)).astype(np.float32)


def held_chords(seconds: float, bpm: float, speed_up: float = 0.0) -> "np.ndarray":
    """An organ-like pad: a new chord each bar, fading in with no attack; speed_up as in strums()."""
    import numpy as np

    chords = [[57, 61, 64], [62, 66, 69], [52, 56, 59], [57, 60, 64]]
    out, t, k = np.zeros(int(seconds * SR), np.float32), 0.0, 0
    while t < seconds:
        bar = 240 / (bpm * (1 + speed_up * t / seconds))
        i, j = int(t * SR), min(int((t + bar) * SR), len(out))
        tt = np.arange(j - i) / SR
        fade = np.minimum(1, tt / 0.4) * np.minimum(1, (bar - tt) / 0.1)
        out[i:j] += fade * sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * tt) for m in chords[k % 4])
        t, k = t + bar, k + 1
    return out * 0.05
