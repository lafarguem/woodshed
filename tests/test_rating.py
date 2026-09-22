import random

import numpy as np
import pytest
from conftest import SR, held_chords, ring_out, spoken, strums, sung_notes

from woodshed import models, rating
from woodshed.isolate import Stems
from woodshed.rating import Metrics


def test_scores_map_measurements_onto_ten_and_weight_them():
    s = rating.scores(Metrics(pitch_cents=5.0, tempo_spread=0.06))
    assert (s["pitch"], s["timing"]) == (10.0, 0.0)
    assert s["overall"] == pytest.approx(10 * rating.DEFAULTS.pitch_weight)
    assert rating.scores(Metrics(40.0, 0.0))["pitch"] == 0.0  # clamped


def test_how_much_pitch_counts_can_be_chosen():
    take = Metrics(pitch_cents=5.0, tempo_spread=0.06)  # 10 on pitch, 0 on timing
    assert rating.scores(take, rating.References(pitch_weight=0.8))["overall"] == pytest.approx(8)
    assert rating.scores(take, rating.References(pitch_weight=0))["overall"] == 0
    assert rating.scores(Metrics(5.0, None), rating.References(pitch_weight=0))["overall"] is None  # no timing to go by


def test_overall_uses_what_could_be_measured():
    assert rating.scores(Metrics(None, 0.01))["overall"] == 10.0  # instrumental: timing only
    assert rating.scores(Metrics(None, None))["overall"] is None


def test_ratings_from_an_older_version_are_redone():
    stored = Metrics(12.3, 0.02, -4.5).to_json()
    assert Metrics.from_json(stored) == Metrics(12.3, 0.02, -4.5)
    assert Metrics.from_json('{"pitch_cents": 12.3, "tempo_spread": 0.02, "version": 2}') == Metrics(12.3, 0.02)
    assert Metrics.from_json(stored.replace(f'"version": {rating.VERSION}', '"version": 0')) is None
    assert Metrics.from_json("not json") is None


def test_held_notes_ignore_vibrato_and_slides():
    hop = 0.02
    t = np.arange(int(0.5 / hop)) * hop
    held = lambda cents: cents + 30 * np.sin(2 * np.pi * 5.5 * t)
    slide = np.linspace(0, 200, 4)
    contour = np.concatenate([held(0), slide, held(210), slide + 210, held(395)])
    notes = rating.held_notes(contour, np.ones(len(contour), bool), hop)
    assert notes == pytest.approx([0, 210, 395], abs=6)


def test_notes_are_judged_against_the_songs_scale():
    scale = np.array([0, 200, 400, 500, 700, 900, 1100]) - 900  # C major, in cents from A440
    assert rating.off_scale(scale + 10, 0.0) == pytest.approx([10] * 7)  # a wrong scale can't tie with it
    # 60¢ above C reads 60¢ off, where against all 12 notes it would read 40¢ from C♯.
    assert rating.off_scale(np.r_[scale, scale[0] + 60], 0.0)[-1] == pytest.approx(60)
    random_notes = np.random.default_rng(0).uniform(-2400, 0, 2000)
    assert np.median(rating.off_scale(random_notes, 0.0)) > 35  # and 25 against all 12


def sung(notes, tuning=0.0):
    """A pitch track holding each note (cents from A440) for 0.5 s, with a short silence after each."""
    cents = np.concatenate([np.r_[np.full(50, c, float), np.full(10, np.nan)] for c in notes])
    f0 = np.where(np.isnan(cents), 0.0, 440 * 2 ** (np.nan_to_num(cents) / 1200))
    return rating.Track(f0, np.where(np.isnan(cents), 0.0, 0.9), tuning=tuning, hop_seconds=0.01)


def test_a_takes_pitch_is_read_against_the_scale_it_fits():
    # Nine notes of C major in tune, then eleven 60¢ sharp of notes that have a whole tone above them.
    in_tune = [-900, -700, -500, -400, -200, 0, 200, 300, 500]  # C4 to D5, in cents from A440
    sharp = [note + 60 for note in [-900, -700, -400, -200, 0] * 2 + [-900]]
    # Against all 12 notes, the sharp ones would read 40¢ (from the note above them), and so would the take.
    assert rating._pitch(sung(in_tune + sharp)) == (pytest.approx(60, abs=1), pytest.approx(60, abs=1))  # over
    flat = [note - 60 for note in [-700, -500, -200, 0, 200] * 2 + [-700]]  # of notes with a whole tone below
    assert rating._pitch(sung(in_tune + flat)) == (pytest.approx(60, abs=1), pytest.approx(-60, abs=1))  # under
    assert rating._pitch(sung(in_tune + sharp, tuning=None))[1] is None  # a cappella: nothing to sit against


def test_timing_leaves_out_where_the_beat_isnt_found_and_the_odd_misreading(monkeypatch):
    import librosa

    sr, hop, bpm, window, seconds = 22050, 256, 100.0, 8.0, 120
    fps = sr / hop
    tempogram = np.zeros((int(window * fps), int(seconds * fps)))
    for frame in range(tempogram.shape[1]):
        kind = frame % 20
        if kind < 12:  # most of the song: no beat near the tempo, only a slope towards either end of the range
            tempogram[:, frame] = np.linspace(0, 1, len(tempogram))[:: 1 if kind % 2 else -1]
        else:  # the beat, a little unsteady, and one window in eight misread 10% fast
            tempo = bpm * (1.10 if kind == 19 else (0.99, 1.0, 1.01)[kind % 3])
            tempogram[int(round(60 * fps / tempo)), frame] = 1
    monkeypatch.setattr(librosa.feature, "tempogram", lambda **kwargs: tempogram)
    assert rating._local_tempo_spread(np.zeros(tempogram.shape[1]), sr, hop, bpm, window, seconds) < 0.02


needs_rmvpe = pytest.mark.skipif(models.RMVPE in models.missing(), reason="the pitch model isn't downloaded")


def take(voice, guitar):
    return Stems(mix=voice + guitar, vocals=voice, accompaniment=guitar, rate=SR)


@pytest.fixture(scope="module")
def measured():
    """Pitch and timing measured on synthetic takes (voice and guitar already apart)."""
    rng = random.Random(1)
    guitar = strums(40, bpm=100)
    in_tune = rating.analyze(take(sung_notes(40, 3, rng), guitar))
    off = rating.analyze(take(sung_notes(40, 20, rng), guitar))
    rushed = rating.analyze(take(sung_notes(40, 3, rng), strums(40, bpm=100, speed_up=0.15)))
    return in_tune, off, rushed


@needs_rmvpe
def test_singing_off_the_notes_measures_worse(measured):
    in_tune, off, _ = measured
    assert in_tune.pitch_cents < 8 and off.pitch_cents > 15


@needs_rmvpe
def test_rushing_measures_worse_than_a_steady_tempo(measured):
    in_tune, _, rushed = measured
    assert in_tune.tempo_spread < 0.005 < rushed.tempo_spread


@needs_rmvpe
def test_no_singing_means_no_pitch_rating():
    silence = np.zeros(40 * SR, np.float32)
    assert rating.analyze(take(silence, strums(40, bpm=100))).pitch_cents is None


@needs_rmvpe
def test_singing_is_told_from_talking_and_from_an_instrument_alone():
    import librosa

    rng, guitar = random.Random(3), strums(40, bpm=100)
    at_16k = lambda samples: librosa.resample(samples, orig_sr=SR, target_sr=16_000)
    assert rating.sings(at_16k(sung_notes(40, 3, rng) + guitar))
    assert not rating.sings(at_16k(spoken(40, rng) + guitar))
    assert not rating.sings(at_16k(spoken(40, rng)))
    assert not rating.sings(at_16k(guitar))


def test_sustained_chords_are_timed_by_their_changes():
    # No attacks to follow: steady pads used to read as a wildly wandering tempo.
    steady = rating._tempo_spread(held_chords(75, bpm=90), SR)
    rushed = rating._tempo_spread(held_chords(75, bpm=90, speed_up=0.12), SR)
    assert steady < 0.02 and rushed > 2 * steady


def test_takes_under_a_minute_are_timed_by_their_strumming(monkeypatch):
    # Under a minute, the instrument used to be told from the take's last seconds: its final chord ringing out,
    # which reads as sustained, so a 56 s strummed take was timed by chord changes (3 points harsher when rushing).
    used = []
    monkeypatch.setattr(rating, "_attack_spread", lambda y, sr: used.append("attacks"))
    monkeypatch.setattr(rating, "_chord_change_spread", lambda y, sr: used.append("chord changes"))
    rating._tempo_spread(np.concatenate([strums(53, bpm=100), ring_out(3)]), SR)
    assert used == ["attacks"]


def test_a_cappella_has_no_timing_rating():
    assert rating._tempo_spread(np.zeros(60 * SR, np.float32), SR) is None
