"""Comparing a take's melody with a reference's (melody.py), on takes made up note by note."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SONGS

from woodshed import melody, rating, transcribe
from woodshed.lyrics import words
from woodshed.melody import Melody
from woodshed.rating import Metrics

LINES = SONGS["Harbor Lights"].split(" / ")
_rng = random.Random(5)
TUNE = [[_rng.choice([-12, -10, -9, -7, -5, -4, -2, 0, 3]) for _ in words(line)] for line in LINES]  # a note a word
PROGRESSION = [(9, 0, 4), (5, 9, 0), (0, 4, 7), (7, 11, 2)]  # Am F C G, a chord every two words


def chord(pitch_classes, shift=0):
    v = np.zeros(12)
    v[[(p + shift) % 12 for p in pitch_classes]] = 1
    return (v / np.linalg.norm(v)).tolist()


def sing(order=range(len(LINES)), shift=0, voice_shift=None, off=0.0, flat_line=None, hear=lambda w: w,
         instrument=True, song=LINES, tune=TUNE, melisma=()):
    """A take of the song's lines (in `order`), each word a note of its tune, 0.3 s long, a note 0.5 s apart.
    The instrument plays `shift` semitones up, the voice `voice_shift` (the same by default), `off` semitones
    off, and a semitone flat on `flat_line`. `hear` is how Whisper hears each word (None: not at all). With
    `melisma`, each word's note is followed by those notes (semitones from it), sung on the same word."""
    voice_shift = shift if voice_shift is None else voice_shift
    lines, heard, notes, chords, t = [], [], [], [], 1.0
    for n, k in enumerate(order):
        lines.append(song[k])
        for i, w in enumerate(words(song[k])):
            if (w := hear(w)) is not None:
                heard.append((w, t, t + 0.3, n))
                chords.append(chord(PROGRESSION[i // 2 % 4], shift))
            pitch = tune[k][i] + voice_shift + off - (k == flat_line)
            for n_note, step in enumerate((0, *melisma)):
                notes.append((t + 0.05 + 0.1 * n_note, t + 0.15 + 0.1 * n_note, pitch + step))
            t += 0.5
        t += 1.0
    return Melody(lines, heard, notes, chords if instrument else None)


REFERENCE = sing()


def test_a_take_sung_like_the_reference_is_on_the_melody():
    c = melody.compare(sing(off=0.05), REFERENCE)  # 5¢ sharp throughout
    assert len(c.lines) == len(LINES) and c.shift == 0
    assert c.typical == pytest.approx(5) and c.where == pytest.approx(5)
    assert melody.sitting(c) is None and melody.furthest(c) == []
    assert rating.scores(Metrics(30, 0.02), melody_cents=c.typical)["pitch"] == 10  # the scale's 30¢ doesn't count


def test_the_notes_meant_count_an_octave_apart_or_sung_a_little_flat():
    assert melody.compare(sing(voice_shift=-12), REFERENCE).typical == 0  # a man singing a woman's melody
    assert melody.compare(sing(), sing(off=-0.2)).typical == 0  # the reference is 20¢ flat, you aren't


def test_a_take_a_semitone_under_is_told_so_with_its_lines():
    take = sing(off=-1.0)
    c = melody.compare(take, REFERENCE)
    assert c.where == pytest.approx(-100) and c.typical == pytest.approx(100)
    assert rating.scores(Metrics(10, 0.02), melody_cents=c.typical)["pitch"] == 0  # in tune with itself: 10 on the scale
    assert melody.sitting(c) == "Your lines sit about a semitone under the melody."
    far = melody.furthest(c)
    assert len(far) == 3 and all(line.cents == pytest.approx(-100) for line in far)
    assert far[0].text in LINES and far[0].start == next(w[1] for w in take.words if w[3] == LINES.index(far[0].text))


def test_the_line_sung_flat_stands_out_even_misheard():
    # A third of the longer words misheard by a letter, and one word in five not heard at all.
    count = iter(range(1000))
    hear = lambda w: None if next(count) % 5 == 4 else (w[:-1] + "x" if len(w) > 4 else w)
    take = sing(flat_line=2, off=0.2, hear=hear)
    c = melody.compare(take, REFERENCE)
    assert len(c.lines) == len(LINES)
    [far] = melody.furthest(c)
    assert far.text == LINES[2] and far.cents == pytest.approx(-80)
    assert far.start == next(w[1] for w in take.words if w[3] == 2)  # its first word heard
    assert melody.sitting(c) == "Your lines sit a little over the melody (about 20¢)."


def test_notes_sung_on_one_word_are_compared_in_order():
    reference = sing(melisma=(2, 4))
    assert melody.compare(sing(melisma=(2, 4)), reference).typical == 0
    # Two notes where the reference sings three: the first and the last.
    assert melody.compare(sing(melisma=(4,)), reference).typical == 0


def test_verses_sung_in_another_order_are_compared_line_by_line():
    c = melody.compare(sing(order=[3, 4, 5, 0, 1, 2, 0]), REFERENCE)
    assert len(c.lines) == 7 and c.typical == pytest.approx(0)


def test_a_line_that_isnt_in_the_reference_isnt_compared():
    stray = SONGS["Gravel Road"].split(" / ")[0]  # it shares a word or two with the song, by chance
    take = sing(order=range(7), song=LINES + [stray], tune=TUNE + [[5] * len(words(stray))], melisma=(0, 0))
    c = melody.compare(take, REFERENCE)
    assert [line.text for line in c.lines] == LINES


def test_the_melody_moves_with_your_instrument():
    capo = melody.compare(sing(shift=3), REFERENCE)
    assert (capo.shift, capo.typical) == (3, 0)
    assert melody.transposed(capo) == ("Your instrument plays 3 semitones over the reference, so the melody was "
                                       "moved to match.")
    # The voice didn't follow the instrument: it's judged against it.
    same_voice = melody.compare(sing(shift=3, voice_shift=0), REFERENCE)
    assert same_voice.where == pytest.approx(-300)
    assert melody.sitting(same_voice) == "Your lines sit about 3 semitones under the melody."


def test_a_cappella_the_melody_moves_to_the_key_you_sing_in():
    c = melody.compare(sing(shift=5, instrument=False), REFERENCE)
    assert (c.shift, c.by_instrument, c.typical) == (5, False, 0)
    assert melody.transposed(c) is None


def test_where_you_sit_is_only_said_when_most_lines_agree():
    def sits(*cents):
        return melody.sitting(melody.Comparison(0, True, [melody.Line(0.0, "", c) for c in cents]))

    assert sits(30, 30, 30, 30, 30, -30) == "Your lines sit a little over the melody (about 30¢)."
    assert sits(30, 30, 30, 30, -30, -30) is None  # 4 in 6: not enough to say
    assert sits(-30, -30, -30, -30) is None  # too few lines
    assert sits(10, 10, 10, 10, 10) is None  # too close to matter
    assert sits(-160, -150, -170, -140, -160) == "Your lines sit about 2 semitones under the melody."


def test_too_few_lines_compared_means_no_comparison():
    assert melody.compare(sing(order=[0, 1]), REFERENCE) is None
    assert melody.compare(sing(hear=lambda w: None), REFERENCE) is None  # nothing heard


def test_a_melody_is_kept_and_read_back():
    text = REFERENCE.to_json()
    assert Melody.from_json(text) == REFERENCE
    assert Melody.from_json(text.replace(f'"version":{melody.VERSION}', '"version":0')) is None
    assert Melody.from_json("not json") is None


def test_notes_and_chords_are_read_from_the_take():
    """From audio: the instrument's chords under each word, and the voice's notes from its pitch track."""
    sr, hop = 22050, 0.01

    def take(shift, cents):  # cents: how sharp the instrument is tuned, and the voice sings with it
        progression = PROGRESSION * 3  # a chord every 2 s
        accompaniment, f0 = [], np.zeros(int(2 * len(progression) / hop))
        t = np.arange(2 * sr) / sr
        for n, pitch_classes in enumerate(progression):
            midi = [48 + pitch_classes[0] + shift] + [60 + p + shift for p in pitch_classes]
            accompaniment.append(sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69 + cents / 100) / 12) * t) for m in midi))
            for second in (0, 1):  # a word each second, its note held from 0.1 to 0.5 s
                a = int((2 * n + second + 0.1) / hop)
                f0[a:a + int(0.4 / hop)] = 440 * 2 ** ((72 + pitch_classes[0] + shift - 69 + cents / 100) / 12)
        heard = [words(line)[:4] for line in LINES]
        lines = [transcribe.Line(" ".join(ws), [transcribe.Word(w, 4 * n + i, 4 * n + i + 0.6) for i, w in enumerate(ws)])
                 for n, ws in enumerate(heard)]
        track = rating.Track(f0, (f0 > 0) * 0.9, tuning=float(cents), hop_seconds=hop)
        stems = SimpleNamespace(accompaniment=0.1 * np.concatenate(accompaniment).astype(np.float32), rate=sr)
        return melody.extract(stems, lines, track)

    reference, capo = take(0, 0), take(3, 30)  # a capo on the 3rd fret, and tuned 30¢ sharp
    # Semitones from A, on the instrument's tuning: the chords' roots (A, A, F, F) an octave up, and 3 up.
    assert [note[2] for note in capo.notes[:4]] == pytest.approx([72 + root + 3 - 69 for root in (9, 9, 5, 5)])
    for n, heard in enumerate(reference.chords):  # each word's chord, not the song's
        assert np.argmax(heard) in PROGRESSION[n // 2 % 4]
    c = melody.compare(capo, reference)
    assert c.shift == 3 and c.typical == pytest.approx(0, abs=2)


def lines_off(cents_by_line, **kwargs):
    """A take on the melody, but for the lines given, sung that many cents off it."""
    return sing(tune=[[note + cents_by_line.get(k, 0) / 100 for note in line] for k, line in enumerate(TUNE)], **kwargs)


def test_the_lines_sung_are_placed_among_the_references():
    take = sing(order=[3, 4, 5, 0, 1, 2, 0], flat_line=4)
    c = melody.compare(take, REFERENCE)
    assert [(s.line, round(s.cents)) for s in c.sung] == [(3, 0), (4, -100), (5, 0), (0, 0), (1, 0), (2, 0), (0, 0)]
    starts = [next(w[1] for w in take.words if w[3] == n) for n in range(7)]
    ends = [[w[2] for w in take.words if w[3] == n][-1] for n in range(7)]
    assert [(s.start, s.end) for s in c.sung] == list(zip(starts, ends))


def test_a_line_heard_as_two_of_the_references_is_split_between_them():
    # Whisper hears the take's first two lines as one; the second of them is sung a semitone flat.
    merged = sing(order=range(5), song=[f"{LINES[0]} {LINES[1]}", *LINES[2:]],
                  tune=[TUNE[0] + [note - 1 for note in TUNE[1]], *TUNE[2:]])
    c = melody.compare(merged, REFERENCE)
    assert len(c.lines) == 5
    assert [(s.line, round(s.cents)) for s in c.sung[:3]] == [(0, 0), (1, -100), (2, 0)]
    assert c.sung[1].start == merged.words[len(words(LINES[0]))][1]  # the first word of the reference's 2nd line


def test_the_lines_closest_to_the_melody_are_pointed_out_too():
    c = melody.compare(lines_off({0: 30, 1: 12, 2: -5, 3: -100, 4: 40, 5: 18}), REFERENCE)
    assert [(line.text, round(line.cents)) for line in melody.closest(c)] == [(LINES[2], -5), (LINES[1], 12),
                                                                               (LINES[5], 18)]
    assert [line.text for line in melody.furthest(c)] == [LINES[3]]
    assert melody.closest(melody.compare(sing(off=-1.0), REFERENCE)) == []


def test_across_takes_the_lines_off_take_after_take_stand_out_not_one_misread():
    takes = [lines_off({2: -100, 4: 80}), lines_off({2: -90}), lines_off({2: -110, 0: 5}), lines_off({2: 30}),
             sing(order=[0, 1, 3])]  # this one leaves out lines 2, 4 and 5
    comparisons = [c for c in (melody.compare(take, REFERENCE) for take in takes) if c]
    assert len(comparisons) == 5
    histories = melody.history(comparisons, REFERENCE)

    assert [h.text for h in histories] == LINES and [h.lines for h in histories] == [[n] for n in range(6)]
    [flat] = melody.often_off(histories)
    assert flat.text == LINES[2] and flat.cents == pytest.approx([-100, -90, -110, 30, None])
    assert flat.typical == pytest.approx(-95) and (flat.same_side, flat.close) == (3, 0)
    # Off in 1 of 4 (line 4): once is no pattern. On it every time: lines 0, 1, 3 and 5, in the song's order.
    assert [h.text for h in melody.closest_across(histories)] == [LINES[0], LINES[1], LINES[3]]
    assert len(melody.closest_across(histories, most=10)) == 5  # line 4 too: typically on it


def test_a_take_sung_under_throughout_doesnt_put_all_its_lines_off():
    # A semitone under the melody all through, and the fourth line a semitone further under.
    comparisons = [melody.compare(lines_off({3: -100}, off=-1.0), REFERENCE) for _ in range(3)]
    assert melody.sitting(comparisons[0]) == "Your lines sit about a semitone under the melody."
    histories = melody.history(comparisons, REFERENCE)
    [flat] = melody.often_off(histories)
    assert flat.text == LINES[3] and flat.typical == pytest.approx(-100)
    assert all(h.typical == pytest.approx(0) for h in histories if h is not flat)


def test_a_line_off_either_way_isnt_pointed_out():
    def off(*cents):
        return melody.often_off(melody.history([melody.compare(lines_off({1: c}), REFERENCE) for c in cents],
                                               REFERENCE))

    assert off(-80, 90, -70, 100) == []  # typically on it
    assert off(-80, -90, 60) == []  # under in 2 of 3: not enough
    assert off(-80, -90, 60, -70) != []  # 3 of 4


def test_a_line_sung_in_fewer_than_3_takes_isnt_judged():
    comparisons = [melody.compare(lines_off({5: -100}), REFERENCE), melody.compare(lines_off({5: -100}), REFERENCE),
                   melody.compare(sing(order=range(5)), REFERENCE)]
    histories = melody.history(comparisons, REFERENCE)
    assert histories[5].cents == pytest.approx([-100, -100, None])
    assert melody.often_off(histories) == [] and histories[5] not in melody.closest_across(histories, most=10)


def test_a_chorus_sung_the_same_each_time_is_one_line():
    chorus = sing(order=[0, 1, 2, 0, 3])  # the reference sings its first line again, as a chorus would be
    comparisons = [melody.compare(sing(order=[0, 1, 2, 0, 3], flat_line=0), chorus) for _ in range(3)]
    histories = melody.history(comparisons, chorus)
    assert [h.lines for h in histories] == [[0, 3], [1], [2], [4]]
    assert histories[0].cents == pytest.approx([-100] * 3) and melody.often_off(histories) == [histories[0]]


def test_a_key_a_fifth_away_must_beat_the_original_clearly():
    def played(shift, blur=0.0):  # the chords played `shift` semitones up, blurred with the original key's
        take = sing(shift=shift, voice_shift=0)
        chords = [np.add(c, np.multiply(blur, o)) for c, o in zip(take.chords, REFERENCE.chords)]
        return Melody(take.lines, take.words, take.notes, [(c / np.linalg.norm(c)).tolist() for c in chords])

    near = played(7, blur=0.85)
    closeness = [np.mean(np.sum(np.array(near.chords) * np.roll(REFERENCE.chords, t, axis=1), axis=1))
                 for t in (0, 7)]
    assert 0 < closeness[1] - closeness[0] < melody.FIFTH_MARGIN  # the fifth is ahead, but not clearly
    assert melody.compare(near, REFERENCE).shift == 0
    assert melody.compare(played(7), REFERENCE).shift == 7  # a capo on the 7th fret
    assert melody.compare(played(5), REFERENCE).shift == 5
