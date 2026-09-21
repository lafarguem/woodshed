"""Rating a take on pitch and timing, from its voice and its accompaniment (guitar, piano…).

What's measured, and stored with each take:
- pitch: how far your held notes are from the nearest note, in cents (1/100 of a
  semitone), relative to your instrument's tuning. Notes picked at random would average 25¢.
  RMVPE tracks your voice in the recording as is: on synthetic takes it read within 1.4¢
  of the truth, against 3-5¢ too harsh for CREPE or RMVPE on the separated voice.
- timing: how much your tempo wanders, as the spread of the local tempo across the song. It
  follows the instrument's attacks (strums, piano chords); for sustained sounds with hardly any
  (organ, pads, bowed strings), it follows when the chords change instead. It catches rushing
  and dragging; not the timing of individual notes, nor stops.

Scores out of 10 are derived from those numbers with reference points (`shed init` can change
them), so they can be retuned without re-analyzing any take. They're judgment calls, not standards.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np

from woodshed.isolate import Stems

VERSION = 1  # bump when a measurement changes, so older takes get re-analyzed
METRICS = ("pitch", "timing")

WEIGHTS = {"pitch": 0.6, "timing": 0.4}

EDGE_SECONDS = 5.0  # count-ins and final ringing chords aren't judged for timing
_SUSTAINED = 0.01  # below this share of percussive energy, follow chord changes instead of attacks
_MIN_NOTES = 10
# Held notes per minute of voice: 37-98 in 40 real sung takes (with guitar), 0-14 for speech (alone
# or over a guitar); the threshold leaves both about the same margin.
SINGING_NOTES_PER_VOICE_MINUTE = 22
_MIN_VOICE_SECONDS = 5


@dataclass(frozen=True)
class References:
    """For each measurement: (value scoring 10, value scoring 0), linear in between."""

    pitch_cents: tuple[float, float] = (5.0, 25.0)
    # On test takes, a steady player measured 0.1% and rushing 12% over a song measured 2.5%.
    tempo_spread: tuple[float, float] = (0.01, 0.06)


DEFAULTS = References()


@dataclass
class Metrics:
    pitch_cents: float | None  # None: not enough singing to judge
    tempo_spread: float | None  # None: not enough strumming to judge

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "version": VERSION})

    @classmethod
    def from_json(cls, text: str) -> "Metrics | None":
        """None if missing, unreadable, or measured by an older version."""
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("version") != VERSION:
            return None
        return cls(data.get("pitch_cents"), data.get("tempo_spread"))


def _linear(value: float, best: float, worst: float) -> float:
    return float(np.clip(10 * (worst - value) / (worst - best), 0, 10))


def scores(m: Metrics, references: References = DEFAULTS) -> dict[str, float | None]:
    """Scores out of 10 for each metric, plus the weighted "overall"."""
    parts = {
        "pitch": None if m.pitch_cents is None else _linear(m.pitch_cents, *references.pitch_cents),
        "timing": None if m.tempo_spread is None else _linear(m.tempo_spread, *references.tempo_spread),
    }
    known = {k: v for k, v in parts.items() if v is not None}
    parts["overall"] = (sum(v * WEIGHTS[k] for k, v in known.items()) / sum(WEIGHTS[k] for k in known)
                        if known else None)
    return parts


def analyze(stems: Stems) -> Metrics:
    return Metrics(_pitch_cents(stems), _tempo_spread(stems.accompaniment, stems.rate))


def held_notes(cents: np.ndarray, voiced: np.ndarray, hop_seconds: float) -> np.ndarray:
    """The median pitch of each note held for 200 ms or more.

    A note is a stretch where the (lightly smoothed) pitch moves slower than 1500¢/s:
    vibrato stays under that (±30¢ at 5.5 Hz peaks near 1000¢/s), slides between notes don't.
    """
    from scipy.ndimage import median_filter

    notes, min_frames = [], int(0.2 / hop_seconds)
    size = max(3, int(round(0.11 / hop_seconds)) | 1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(int), [0]])))
    for run_start, run_end in zip(edges[::2], edges[1::2]):  # each stretch of continuous singing
        raw = cents[run_start:run_end]
        if len(raw) < min_frames:
            continue
        smooth = median_filter(raw, size=size, mode="nearest")
        steady = np.abs(np.gradient(smooth)) < 1500 * hop_seconds
        i = 0
        while i < len(raw):
            if not steady[i]:
                i += 1
                continue
            j = i
            while j < len(raw) and steady[j] and abs(smooth[j] - smooth[i]) < 50:
                j += 1
            if j - i >= min_frames:
                notes.append(float(np.median(raw[i:j])))
            i = j
    return np.array(notes)


def sings(samples: np.ndarray) -> bool:
    """Whether someone sings in these 16 kHz samples (the playing part of a recording), judged on its
    middle three minutes. Singing holds its notes, speech glides from pitch to pitch: so it counts the
    notes held 200 ms or more, per minute that a voice is heard (however long the instrument plays alone).
    RMVPE reads the voice straight from the recording, in about a second per minute."""
    from woodshed import rmvpe

    rate = 16_000
    middle = samples[max(0, len(samples) // 2 - 90 * rate): len(samples) // 2 + 90 * rate]
    f0, confidence = rmvpe.pitch(middle)
    voiced = confidence > 0.5
    voice_minutes = voiced.sum() * rmvpe.HOP_SECONDS / 60
    if voice_minutes < _MIN_VOICE_SECONDS / 60:  # an instrument alone, or hardly a word
        return False
    notes = held_notes(1200 * np.log2(np.maximum(f0, 1.0) / 440), voiced, rmvpe.HOP_SECONDS)
    return len(notes) / voice_minutes >= SINGING_NOTES_PER_VOICE_MINUTE


def _pitch_cents(stems: Stems) -> float | None:
    from woodshed import rmvpe

    f0, confidence = rmvpe.pitch(stems.mix_16k())
    notes = held_notes(1200 * np.log2(np.maximum(f0, 1.0) / 440), confidence > 0.5, rmvpe.HOP_SECONDS)
    if len(notes) < _MIN_NOTES:
        return None
    tuning = _tuning(stems.accompaniment, stems.rate)
    if tuning is None:  # no instrument to compare against: judge how consistent the notes are
        tuning = _circular_mean(notes)
    return float(np.mean(np.abs((notes - tuning + 50) % 100 - 50)))


def _circular_mean(cents: np.ndarray) -> float:
    angles = 2 * np.pi * cents / 100
    return float(np.angle(np.mean(np.exp(1j * angles))) * 100 / (2 * np.pi))


def _tuning(accompaniment: np.ndarray, rate: int) -> float | None:
    """How far the instrument is tuned from A440, in cents."""
    import librosa

    if np.sqrt(np.mean(accompaniment**2)) < 1e-3:
        return None
    y = librosa.resample(accompaniment, orig_sr=rate, target_sr=22050)
    return 100 * float(librosa.estimate_tuning(y=y, sr=22050, bins_per_octave=12))


def _tempo_spread(accompaniment: np.ndarray, rate: int) -> float | None:
    """How much the tempo wanders, relative to its average."""
    import librosa

    sr = 22050
    y = librosa.resample(accompaniment, orig_sr=rate, target_sr=sr)
    if len(y) < sr * (2 * EDGE_SECONDS + 20) or np.sqrt(np.mean(y**2)) < 1e-3:  # too short, or a cappella
        return None
    mid = len(y) // 2
    middle = y[max(0, mid - 30 * sr): mid + 30 * sr]  # a minute is plenty to tell the instrument
    harmonic, percussive = librosa.effects.hpss(middle)
    if np.sum(percussive**2) < _SUSTAINED * (np.sum(harmonic**2) + np.sum(percussive**2)):
        return _chord_change_spread(y, sr)
    return _attack_spread(y, sr)


def _attack_spread(y: np.ndarray, sr: int) -> float | None:
    """From the instrument's attacks: strums, piano chords, plucked notes."""
    import librosa

    hop = 256
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop, units="time")
    duration = len(y) / sr
    if np.sum((onsets > EDGE_SECONDS) & (onsets < duration - EDGE_SECONDS)) < 16:
        return None
    bpm = float(librosa.feature.tempo(onset_envelope=envelope, sr=sr, hop_length=hop)[0])
    return _local_tempo_spread(envelope, sr, hop, bpm, window=8.0, duration=duration)


def _chord_change_spread(y: np.ndarray, sr: int) -> float | None:
    """From when the harmony changes, for sustained sounds without clear attacks."""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    tonal = librosa.decompose.nn_filter(librosa.feature.tonnetz(chroma=chroma), aggregate=np.median, width=5)
    change = np.r_[0, np.linalg.norm(np.diff(tonal, axis=1), axis=0)]  # how far the harmony moves each frame
    bpm = float(librosa.feature.tempo(onset_envelope=change, sr=sr, hop_length=hop, start_bpm=30, max_tempo=120)[0])
    # Chords change about once a bar, so each window needs to be long enough to hold several.
    return _local_tempo_spread(change, sr, hop, bpm, window=16.0, duration=len(y) / sr)


def _local_tempo_spread(envelope: np.ndarray, sr: int, hop: int, bpm: float, window: float,
                        duration: float) -> float | None:
    """In each window, the strongest repetition period near the overall tempo; then their spread."""
    import librosa

    fps = sr / hop
    tempogram = librosa.feature.tempogram(onset_envelope=envelope, sr=sr, hop_length=hop, win_length=int(window * fps))
    shortest, longest = int(fps * 60 / (bpm * 1.33)), int(np.ceil(fps * 60 / (bpm * 0.75)))
    local = []
    for frame in range(tempogram.shape[1]):
        if not EDGE_SECONDS + window / 2 <= frame / fps <= duration - EDGE_SECONDS - window / 2:
            continue
        column = tempogram[shortest:longest + 1, frame]
        k = int(np.argmax(column))
        if 0 < k < len(column) - 1:  # refine between lags, for sub-frame precision
            a, b, c = column[k - 1:k + 2]
            k += 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0
        local.append(60 * fps / (shortest + k))
    return float(np.std(local) / np.mean(local)) if len(local) >= 10 else None
