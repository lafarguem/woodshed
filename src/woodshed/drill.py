"""Finding the note a line starts on, before singing it, then where you sit against the melody as you sing
(`shed drill`). Nothing is recorded.

For each line of the song's reference melody, the note its first held word is sung on is played, then a tuner
follows your voice until you've held that note for a second. Then you sing the song: the tuner goes (a key brings
it back), and a chart
of where your latest notes landed against the melody takes its place, a few seconds behind you (listen.py):
nothing to read as you sing. Saying another song's name, your instrument quiet, drills that one.

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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from woodshed.melody import Melody, notes_on_words

if TYPE_CHECKING:
    from rich.text import Text

NOTE_NAMES = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"]
FOUND_CENTS = 50  # this close to the note, you're on it rather than on its neighbour
FOUND_SECONDS = 1.0  # held that long, it's found
TONE_SECONDS = 1.5
_READING_SECONDS = 0.25  # the tuner shows the median pitch of a quarter second of singing…
_LAG_SECONDS = 0.2  # …ending this long ago
_KEEP_SECONDS = 0.5  # a reading is shown this long when the next ones miss
_CHART_SEMITONES = 6  # the chart goes from this many under the melody to this many over
_CHART_ROWS = 5
_NEWS_SECONDS = 5.0  # "Switched to…" shows this long
# A note of the song compared this recently (listen.py), you're singing it. They come a few seconds late, and
# can be far apart where few words are matched to the melody's.
_SINGING_SECONDS = 30.0
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


@dataclass
class Song:
    """A song ready to drill."""

    title: str
    reference: Melody
    shift: int  # semitones your instrument plays above the reference
    starts: list[Start]
    transposed: str | None  # said when shift isn't 0


def chart(offsets: list[float]) -> list["Text"]:
    """Where your latest notes landed against the melody, drawn: a column for each semitone from 6 under to 6
    over, as tall as the notes that landed there, the melody in the middle. The column most landed in is colored
    (green on the melody, yellow a semitone off, red further), and an arrow under it says which way to go."""
    from rich.text import Text

    from woodshed import listen

    recent = (np.round(np.asarray(offsets[-listen.RECENT_NOTES:])).astype(int) + 6) % 12 - 6  # as settled() has it
    votes = np.bincount(np.clip(recent + _CHART_SEMITONES, 0, 2 * _CHART_SEMITONES), minlength=2 * _CHART_SEMITONES + 1)
    where = listen.settled(offsets)
    color = "grey50" if where is None else "green" if where[0] == 0 else "yellow" if abs(where[0]) == 1 else "red"
    top = max(int(votes.max()), 1)
    rows = []
    for level in range(_CHART_ROWS, 0, -1):
        row = Text("  ")
        for semitones, n in enumerate(votes, -_CHART_SEMITONES):
            tall = n and n * _CHART_ROWS >= (level - 0.5) * top
            row.append("███ " if tall else "    ", color if where and semitones == where[0] else "grey35")
        rows.append(row)
    middle = 2 + 4 * _CHART_SEMITONES + 1  # the melody's column, in characters
    rows.append(Text("  " + "".join("─┼──" if s == 0 else "────" for s in range(-_CHART_SEMITONES,
                                                                           _CHART_SEMITONES + 1)), "grey50"))
    labels = [" "] * (middle + 4 * _CHART_SEMITONES + 2)
    for at, word in ((2, "under"), (middle - 3, "melody"), (len(labels) - 5, "over")):
        labels[at:at + len(word)] = word
    rows.append(Text("".join(labels), "grey50"))
    if where is not None:
        arrow = "✓" if where[0] == 0 else "▲ higher" if where[0] < 0 else "▼ lower"
        rows.append(Text(" " * (middle - len(arrow) // 2) + arrow, f"bold {color}"))
    return rows


def run(console, prepare: Callable[[str], Song], title: str, titles: list[str], device: int | str | None = None,
        language: str | None = None) -> None:
    """Drill a song until q, Escape or Ctrl+C: its lines' starting notes, and where you sit against its melody as
    you sing it. `prepare` gets a song ready, by its title, when another one in `titles` is asked for."""
    import sounddevice as sd
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text
    from scipy.signal import resample_poly

    from woodshed import listen, rmvpe, terminal

    rate = int(sd.query_devices(device, "input")["default_samplerate"])
    out_rate = int(sd.query_devices(kind="output")["default_samplerate"])
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    heard = np.zeros(0, np.float32)  # the latest second from the microphone
    song = prepare(title)
    listener = listen.Listener(rate, language)
    follow = listen.Follow(song.reference, song.shift)
    index, hold, found, sung, deaf_until, news, news_until = 0, Hold(), False, None, 0.0, "", 0.0
    sung_at = 0.0  # when `sung` was last read
    back_at = 0.0  # when (listener.now) you last went back to a line's note: what the listener heard before is past

    def play_note():
        nonlocal deaf_until
        sd.play(tone(song.starts[index].note, out_rate), out_rate)
        deaf_until = time.monotonic() + TONE_SECONDS + _AFTER_TONE

    def singing() -> bool:
        """Whether you're singing the song: you found the line's note, so you went on, or the listener heard you
        sing it (a few seconds behind). Until you go back to a line's note with a key."""
        return found or follow.sung_at > max(back_at, listener.now - _SINGING_SECONDS)

    def render() -> Group:
        rows = [Text(song.title, "bold")]
        if time.monotonic() < news_until:
            rows.append(Text(news, "bold cyan"))
        if singing():  # nothing to read: where you sit, drawn
            if follow.line is not None:
                rows.append(Text(f"“{song.reference.lines[follow.line]}”", "italic grey50"))
            return Group(*rows, Text(), *chart(follow.offsets), Text(), Text("q quits", "grey35"))
        start = song.starts[index]
        rows[0].append(f"  line {index + 1} of {len(song.starts)}", "dim")
        if song.transposed:
            rows.append(Text(song.transposed, "dim"))
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
        rows.append(Text())
        if not listener.ready:
            rows.append(Text("Getting ready to follow you as you sing…", "dim"))
        elif follow.offsets:  # between verses: where you sat
            rows += chart(follow.offsets)
        else:
            rows.append(Text("Then sing the song: where you sit against the melody shows here.", "dim"))
        rows += [Text(), Text("←/→ another line · Space plays the note again · say a song's name to switch to "
                              "it · q quits", "dim")]
        return Group(*rows)

    stream = sd.InputStream(device=device, channels=1, samplerate=rate, dtype="float32",
                            callback=lambda indata, *_: blocks.put(indata[:, 0].copy()))
    try:
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
                            index = (index + (1 if key in ("right", "down") else -1)) % len(song.starts)
                        hold, found, sung, back_at = Hold(), False, None, listener.now
                        play_note()
                    now = time.monotonic()
                    while not blocks.empty():
                        block = blocks.get_nowait()
                        if now < deaf_until:
                            block = np.zeros_like(block)  # the tone, not you
                        heard = np.concatenate([heard, block])[-int(_LISTEN_SECONDS * rate):]
                        listener.feed(block)
                    for result in listener.results():
                        asked = follow.update(result, [t for t in titles if t != song.title])
                        if asked:
                            song = prepare(asked)
                            follow = listen.Follow(song.reference, song.shift, since=result.end)
                            index, hold, found, sung = 0, Hold(), False, None
                            news, news_until = f"Switched to {asked}.", time.monotonic() + _NEWS_SECONDS
                            play_note()
                    if now >= deaf_until and len(heard) >= rate * (_READING_SECONDS + _LAG_SECONDS) and not singing():
                        f0, confidence = rmvpe.pitch(level(resample_poly(heard, _RATE, rate)))
                        now_sung = reading(f0, confidence, rmvpe.HOP_SECONDS)
                        found = hold.update(None if now_sung is None else offset(now_sung, song.starts[index].note),
                                            now - last)
                        if now_sung is not None:
                            sung, sung_at = now_sung, now
                        elif now - sung_at > _KEEP_SECONDS:
                            sung = None
                    elif now < deaf_until:
                        heard = heard[:0]
                    last = now
                    live.update(render(), refresh=True)
            except KeyboardInterrupt:
                pass
    finally:
        sd.stop()
        listener.close()
