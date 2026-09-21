import threading
import time
import wave

import numpy as np
from rich.console import Console

from woodshed import recorder

RATE = 8000


class FakeStream:
    """Stands in for the microphone: one second of playing, then silence."""

    def __init__(self, callback, **kwargs):
        self.callback, self.running = callback, True

    def _feed(self):
        block, t = RATE // 100, 0
        while self.running:
            level = 0.3 if t < RATE else 0.0
            self.callback(np.full((block, 1), level, np.float32), block, None, None)
            t += block
            time.sleep(0.01)

    def __enter__(self):
        threading.Thread(target=self._feed, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.running = False


def test_auto_stop_after_silence_writes_the_whole_take(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder.sd, "InputStream", FakeStream)
    monkeypatch.setattr(recorder.sd, "query_devices", lambda device, kind: {"default_samplerate": RATE})
    dest = tmp_path / "take.wav"

    recorder.record(dest, Console(quiet=True), auto_stop=0.5)

    with wave.open(str(dest)) as wav:
        seconds = wav.getnframes() / wav.getframerate()
        first = np.frombuffer(wav.readframes(10), "<i2")
    assert 1.4 < seconds < 2.5
    assert first[0] == round(0.3 * 32767)
