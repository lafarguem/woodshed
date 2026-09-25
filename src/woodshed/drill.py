"""Finding the note a line starts on, before singing it (`shed drill`).

Nothing is recorded. For each line of the song's reference melody, the note its first held word is sung on is
played, then a tuner follows your voice until you've held that note for a second:

- The note is moved to your key: by default, the one your instrument played in your latest take compared
  with the melody (melody.Comparison.shift), since you tend to play a song the same way each time.
- And to your register: the octave nearest where your voice usually sits in your takes of the song (the
  reference may be sung an octave above or below you). Octaves are ignored when you sing it, as they are
  in the rating: the right note an octave off counts.
- The microphone isn't listened to while the note plays, so the tuner doesn't take it for your voice.
- The pitch tracker is sure of a voice only when it's loud enough: on a take turned down to a laptop
  microphone's level (-53 dBFS), it heard 5% of it as sung, against 14% as recorded. So what it's given is
  brought to the same loudness first (level()), and then it hears 16% at any level.
- It's also less sure at the end of what it's given, with nothing after it to go by: it caught 20 of 43 moments
  sung in the last quarter second of each second heard, and 33 of 46, all right, in the quarter second ending
  0.2 s before. So the tuner is 0.2 s behind you, and keeps showing your note through a short gap.
- Pitch is measured from A440: an instrument tuned off it by a few cents doesn't change which note is nearest.
"""

import queue
import time
from dataclasses import dataclass

import numpy as np

from woodshed.melody import Melody, notes_on_words

NOTE_NAMES = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"]
FOUND_CENTS = 50  # this close to the note, you're on it rather than on its neighbour
FOUND_SECONDS = 1.0  # held that long, it's found
TONE_SECONDS = 1.5
_READING_SECONDS = 0.25  # the tuner shows the median pitch of a quarter second of singing…
_LAG_SECONDS = 0.2  # …ending this long ago
_KEEP_SECONDS = 0.5  # a reading is shown this long when the next ones miss
_LISTEN_SECONDS = 1.0  # of audio given to the pitch tracker each time, for context
_AFTER_TONE = 0.3  # the tone's echo in the room, not listened to either
_RATE = 16_000  # what the pitch tracker takes
_LOUDNESS = 0.1  # what's heard is brought to this RMS level (-20 dBFS)…
_MAX_GAIN = 100.0  # …by at most 40 dB
_BAR_SEMITONES = 6
_BAR_CELLS = 2 * 2 * _BAR_SEMITONES + 1  # two cells a semitone, the note in the middle


@dataclass
class Start:
    """Where one of the reference's lines starts."""

    line: str  # as heard in the reference
    word: str  # the first word with a note held on it
    note: int  # the note it's sung on, in semitones from A440, moved to your key and register


def starts(reference: Melody, shift: int = 0, register: float | None = None) -> list[Start]:
    """For each of the reference's lines with a note held on one of its words, the first such note: moved
    `shift` semitones, then to the octave nearest `register` (semitones from A440; None keeps the reference's)."""
    on = notes_on_words(reference)
    found = {}
    for w, (word, _, _, line) in enumerate(reference.words):
        if on[w] and line not in found:
            note = round(reference.notes[on[w][0]][2]) + shift
            if register is not None:
                note += 12 * round((register - note) / 12)
            found[line] = Start(reference.lines[line], word, note)
    return [found[line] for line in sorted(found)]


def register(melodies: list[Melody]) -> float | None:
    """Where your voice usually sits in these takes: the median of their notes, in semitones from A440."""
    notes = [note[2] for m in melodies for note in m.notes]
    return float(np.median(notes)) if notes else None


def note_name(semitones: float) -> str:
    """The name of the nearest note, with its octave: 0 is A4."""
    midi = round(semitones) + 69
    return f"{NOTE_NAMES[(midi - 69) % 12]}{midi // 12 - 1}"


def offset(sung: float, note: float) -> float:
    """How far (in semitones) what you sing is from the note, octaves ignored: under it if negative."""
    return (sung - note + 6) % 12 - 6


def advice(off: float) -> str:
    if abs(off) * 100 < FOUND_CENTS:
        return "on it"
    way = "higher ↑" if off < 0 else "lower ↓"
    if abs(off) < 1:
        return f"a little {way} ({abs(off) * 100:.0f}¢ {'under' if off < 0 else 'over'})"
    semitones = round(abs(off))
    return f"{way} ({'a semitone' if semitones == 1 else f'{semitones} semitones'} {'under' if off < 0 else 'over'})"


class Hold:
    """How long the note has been held without leaving it; silence (a breath) pauses the count."""

    def __init__(self):
        self.held = 0.0

    def update(self, off: float | None, seconds: float) -> bool:
        """Given how far off the latest reading is (None: not singing), whether the note is now found."""
        if off is not None:
            self.held = self.held + seconds if abs(off) * 100 < FOUND_CENTS else 0.0
        return self.held >= FOUND_SECONDS


def tone(note: float, rate: int, seconds: float = TONE_SECONDS) -> np.ndarray:
    """The note, soft and clear: a few harmonics fading out, like a plucked string."""
    t = np.arange(int(seconds * rate)) / rate
    f = 440 * 2 ** (note / 12)
    wave = sum(a * np.sin(2 * np.pi * k * f * t) for k, a in ((1, 1.0), (2, 0.4), (3, 0.15)) if k * f < rate / 2)
    envelope = np.minimum(t / 0.01, 1) * np.exp(-2.0 * t / seconds)
    return (0.25 * wave * envelope).astype(np.float32)


def level(samples: np.ndarray) -> np.ndarray:
    """The samples brought to the same loudness, however far you are from the microphone."""
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    return (samples * min(_LOUDNESS / max(rms, 1e-9), _MAX_GAIN)).astype(np.float32)


def reading(f0: np.ndarray, confidence: np.ndarray, hop_seconds: float) -> float | None:
    """What you're singing, in semitones from A440: the median pitch of the quarter second ending _LAG_SECONDS
    ago; None unless you sang through most of it."""
    frames, lag = max(1, int(_READING_SECONDS / hop_seconds)), int(_LAG_SECONDS / hop_seconds)
    end = len(f0) - lag
    f0, voiced = f0[end - frames:end], confidence[end - frames:end] > 0.5
    if voiced.mean() < 0.6:
        return None
    return float(np.median(12 * np.log2(f0[voiced] / 440)))


def run(console, starts_: list[Start], title: str, device: int | str | None = None,
        transposed: str | None = None) -> None:
    """Drill the lines' starting notes until q, Escape or Ctrl+C."""
    import sounddevice as sd
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text
    from scipy.signal import resample_poly

    from woodshed import rmvpe, terminal

    rate = int(sd.query_devices(device, "input")["default_samplerate"])
    out_rate = int(sd.query_devices(kind="output")["default_samplerate"])
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    heard = np.zeros(0, np.float32)  # the latest second from the microphone
    index, hold, found, sung, deaf_until = 0, Hold(), False, None, 0.0
    sung_at = 0.0  # when `sung` was last read

    def play_note():
        nonlocal deaf_until
        sd.play(tone(starts_[index].note, out_rate), out_rate)
        deaf_until = time.monotonic() + TONE_SECONDS + _AFTER_TONE

    def render() -> Group:
        start = starts_[index]
        rows = [Text.assemble((title, "bold"), (f"  line {index + 1} of {len(starts_)}", "dim"))]
        if transposed:
            rows.append(Text(transposed, "dim"))
        rows += [Text(f"“{start.line}”"),
                 Text.assemble("Starts on “", (start.word, "bold"), "”: ", (note_name(start.note), "bold cyan")),
                 Text()]
        if time.monotonic() < deaf_until:
            rows.append(Text("♪ listen…", "cyan"))
        elif found:
            rows.append(Text(f"✓ Found it: you held {note_name(start.note)}. → for the next line.", "bold green"))
        elif sung is None:
            rows.append(Text("Sing the note…", "dim"))
        else:
            off = offset(sung, start.note)
            cells = ["─"] * _BAR_CELLS
            cells[_BAR_CELLS // 2] = "┼"
            cells[int(np.clip(round(off * 2) + _BAR_CELLS // 2, 0, _BAR_CELLS - 1))] = "●"
            color = "green" if abs(off) * 100 < FOUND_CENTS else "yellow" if abs(off) < 1 else "red"
            rows += [Text.assemble("  ", ("".join(cells), color)),
                     Text.assemble("  You: ", (note_name(sung), "bold"), "  ", (advice(off), color))]
        rows += [Text(), Text("←/→ another line · Space plays the note again · q quits", "dim")]
        return Group(*rows)

    stream = sd.InputStream(device=device, channels=1, samplerate=rate, dtype="float32",
                            callback=lambda indata, *_: blocks.put(indata[:, 0].copy()))
    with stream, terminal.keys() as next_key, Live(render(), console=console, transient=True,
                                                    auto_refresh=False) as live:
        play_note()
        last = time.monotonic()
        try:
            while True:
                key = next_key(timeout=0.1)
                if key in ("q", "escape"):
                    break
                if key in ("left", "right", "up", "down", " "):
                    if key != " ":
                        index = (index + (1 if key in ("right", "down") else -1)) % len(starts_)
                    hold, found, sung = Hold(), False, None
                    play_note()
                while not blocks.empty():
                    heard = np.concatenate([heard, blocks.get_nowait()])[-int(_LISTEN_SECONDS * rate):]
                now = time.monotonic()
                if now >= deaf_until and len(heard) >= rate * (_READING_SECONDS + _LAG_SECONDS) and not found:
                    f0, confidence = rmvpe.pitch(level(resample_poly(heard, _RATE, rate)))
                    now_sung = reading(f0, confidence, rmvpe.HOP_SECONDS)
                    found = hold.update(None if now_sung is None else offset(now_sung, starts_[index].note),
                                        now - last)
                    if now_sung is not None:
                        sung, sung_at = now_sung, now
                    elif now - sung_at > _KEEP_SECONDS:
                        sung = None
                elif now < deaf_until:
                    heard = heard[:0]  # the tone, not you
                last = now
                live.update(render(), refresh=True)
        except KeyboardInterrupt:
            pass
    sd.stop()
