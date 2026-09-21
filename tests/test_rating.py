import random

import numpy as np
import pytest
from conftest import SR, held_chords, ring_out, strums, sung_notes

from woodshed import models, rating
from woodshed.isolate import Stems
from woodshed.rating import Metrics


def test_scores_map_measurements_onto_ten_and_weight_them():
    s = rating.scores(Metrics(pitch_cents=5.0, tempo_spread=0.06))
    assert (s["pitch"], s["timing"]) == (10.0, 0.0)
    assert s["overall"] == pytest.approx(10 * rating.WEIGHTS["pitch"])
    assert rating.scores(Metrics(40.0, 0.0))["pitch"] == 0.0  # clamped


def test_overall_uses_what_could_be_measured():
    assert rating.scores(Metrics(None, 0.01))["overall"] == 10.0  # instrumental: timing only
    assert rating.scores(Metrics(None, None))["overall"] is None


def test_ratings_from_an_older_version_are_redone():
    stored = Metrics(12.3, 0.02).to_json()
    assert Metrics.from_json(stored) == Metrics(12.3, 0.02)
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


needs_rmvpe = pytest.mark.skipif(bool(models.missing()), reason="models not downloaded (run `shed init`)")


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
