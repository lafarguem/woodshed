"""Rating a take on pitch and timing, from its voice and its accompaniment (guitar, piano…).

What's measured, and stored with each take:
- pitch: how far your held notes typically are (the median) from the nearest note of the song's
  scale, in cents (1/100 of a semitone), relative to your instrument's tuning. The scale is the
  major (or relative minor) one your notes fit best. Against all 12 notes, anything over 50¢ off
  would read as closer to the next note, so random notes would average 25¢ and nothing could read
  worse; against the scale's 7, errors up to 100¢ show, and random notes land around 38¢. RMVPE tracks your voice in the recording as is: on synthetic takes it read within 1.4¢ of the
  truth, against 3-5¢ too harsh for CREPE or RMVPE on the separated voice. With vibrato, scoops
  into notes and falls off them, it still read within 2-3¢; on a real voice, it agreed with YIN
  (an unrelated pitch tracker) within 1¢ overall.
- timing: how much your tempo wanders, as the spread of the local tempo across the song. It
  follows the instrument's attacks (strums, piano chords); for sustained sounds with hardly any
  (organ, pads, bowed strings), it follows when the chords change instead. It catches rushing
  and dragging; not the timing of individual notes, nor stops.

Scores out of 10 are derived from those numbers with reference points (`shed init` can change
them), so they can be retuned without re-analyzing any take. They're judgment calls, not standards,
set so that professional recordings score 8-10.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np

from woodshed.isolate import Stems

VERSION = 2  # bump when a measurement changes, so older takes get re-analyzed
METRICS = ("pitch", "timing")

WEIGHTS = {"pitch": 0.6, "timing": 0.4}

EDGE_SECONDS = 5.0  # count-ins and final ringing chords aren't judged for timing
# The local tempo is looked for within ±13% of the song's tempo: searched wider, the beat gets misread as 3/4 or
# 4/3 of itself now and then (in 5% of the windows of a steady professional recording, enough to score it 0/10).
_TEMPO_RANGE = 1.13
_SUSTAINED = 0.01  # below this share of percussive energy, follow chord changes instead of attacks
_MIN_NOTES = 10
_MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])
# Held notes per minute of voice: 37-98 in 40 real sung takes (with guitar), 0-14 for speech (alone
# or over a guitar); the threshold leaves both about the same margin.
SINGING_NOTES_PER_VOICE_MINUTE = 22
_MIN_VOICE_SECONDS = 5


@dataclass(frozen=True)
class References:
    """For each measurement: (value scoring 10, value scoring 0), linear in between."""

    # Four professional recordings (a studio one, three acoustic covers) measured 9-17¢, and an amateur's
    # takes 6-40¢ (median 27¢). Hence 10/10 at 12¢, a little under professional level to leave room at the
    # top, and 0/10 at 38¢, where notes picked at random land, which makes 25¢ (still passable) a 5.
    pitch_cents: tuple[float, float] = (12.0, 38.0)
    # The same professional recordings measured ±0.1-2.1% (a loose, stripped-back one ±5%); on synthetic
    # takes, a steady player measured ±0% and rushing 12% over a song ±3.5%.
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
    return float(np.median(off_scale(notes, tuning)))  # the odd misread (or chromatic) note counts for little


def off_scale(notes: np.ndarray, tuning: float) -> np.ndarray:
    """How far (¢) each note is from the nearest note of the song's scale: of the 12 major scales (each with
    the notes of its relative minor), the one the notes fit best on average (by the median, a wrong scale
    fitting half the notes can tie with the right one). Fitted to your singing rather than read from the
    instrument, whose chords can suggest the wrong key."""
    semitones = (np.asarray(notes) - tuning) / 100
    fits = [100 * np.min([np.abs((semitones - note + 6) % 12 - 6) for note in (root + _MAJOR_SCALE) % 12], axis=0)
            for root in range(12)]
    return min(fits, key=np.mean)


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
    """In each window, the strongest repetition period near the overall tempo; then how much they spread.

    A window whose strongest period is at the edge of the range searched has no peak inside it: the beat
    wasn't found there, so it's left out. The spread is robust (1.48 × the median deviation, which is the
    standard deviation for normal noise), so the odd misread window counts for little.
    """
    import librosa

    fps = sr / hop
    tempogram = librosa.feature.tempogram(onset_envelope=envelope, sr=sr, hop_length=hop, win_length=int(window * fps))
    shortest, longest = int(fps * 60 / (bpm * _TEMPO_RANGE)), int(np.ceil(fps * 60 * _TEMPO_RANGE / bpm))
    local = []
    for frame in range(tempogram.shape[1]):
        if not EDGE_SECONDS + window / 2 <= frame / fps <= duration - EDGE_SECONDS - window / 2:
            continue
        column = tempogram[shortest:longest + 1, frame]
        k = int(np.argmax(column))
        if not 0 < k < len(column) - 1:
            continue
        a, b, c = column[k - 1:k + 2]  # refine between lags, for sub-frame precision
        k += 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0
        local.append(60 * fps / (shortest + k))
    if len(local) < 10:
        return None
    ratio = np.array(local) / np.median(local)
    return float(1.4826 * np.median(np.abs(ratio - 1)))
