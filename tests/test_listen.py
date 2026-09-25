"""Following you as you sing in `shed drill` (listen.py): where you sit against the melody, and asking for another
song. The passes over the microphone are made up here (listen.Heard), as the melodies are in test_melody.py."""

from test_melody import LINES, REFERENCE, sing

from woodshed import drill, listen, melody
from woodshed.melody import Melody

TITLES = ["Harbor Lights", "Winter Town", "Space Song"]


def said(text, instrument_db, at=1.0):
    """A pass in which `text` is said (or sung) from `at`, a word every 0.4 s, the instrument that loud."""
    ws = text.split()
    return listen.Heard(at + 0.4 * len(ws) + 10, Melody([text], [(w, at + 0.4 * i, at + 0.4 * i + 0.3, 0)
                                                                  for i, w in enumerate(ws)], [], None),
                        [instrument_db] * len(ws))


def test_where_you_sit_is_what_most_of_your_latest_notes_say():
    assert listen.settled([0.1] * 7) is None  # too few yet
    assert listen.settled([0.1, -0.2, 0.3, 0.0, 1.1, -0.9, 0.2, 0.1]) == (0, 6, 8)
    assert listen.settled([-3.1, -2.8, -3.2, -2.0, -3.0, 0.1, -3.3, -2.9, -3.4]) == (-3, 7, 9)
    # A fourth under or a fifth over are the same notes: said the shorter way.
    assert listen.settled([-5.0, 7.1, -4.9, 6.8, -5.2, 7.0, -5.1, 7.2])[0] == -5
    # Only the latest 15 count: you found the melody.
    assert listen.settled([-3.0] * 20 + [0.0] * 10)[0] == 0


def test_where_you_sit_is_drawn_not_written():
    def drawn(offsets):
        return [row.plain for row in drill.chart(offsets)]

    under = drawn([-3.1, -2.8, -3.2, -2.0, -3.0, 0.1, -3.3, -2.9, -3.4, -1.2])
    melody_at = under[-3].index("┼")  # the axis: the melody's column
    assert under[0].index("███") == melody_at - 1 - 4 * 3  # the tallest column: 3 semitones under
    assert under[-1].strip() == "▲ higher"
    top = drill.chart([-3.0] * 10)[0]
    assert {span.style for span in top.spans if "███" in top.plain[span.start:span.end]} == {"red"}  # 3 off
    top = drill.chart([-1.0] * 10)[0]
    assert {span.style for span in top.spans if "███" in top.plain[span.start:span.end]} == {"yellow"}
    assert drawn([0.1, -0.2, 0.3, 0.0, 1.1, -0.9, 0.2, 0.1])[-1].strip() == "✓"
    assert drawn([0.9, 1.2, 1.1, 0.8, 1.0, 1.3, 0.7, 1.0])[-1].strip() == "▼ lower"
    assert "higher" not in drawn([-3.0] * 5)[-1] and "▲" not in "".join(drawn([-3.0] * 5))  # too few notes yet


def test_another_song_is_asked_for_by_its_name_the_instrument_quiet():
    assert listen.asked_for(said("switch to winter town", -40), TITLES, 0, 100) == "Winter Town"
    assert listen.asked_for(said("okay lets do space song", -40), TITLES, 0, 100) == "Space Song"
    assert listen.asked_for(said("winter town", -40), TITLES, 0, 100) == "Winter Town"  # no need to say more
    assert listen.asked_for(said("switch to winter towns", -40), TITLES, 0, 100) == "Winter Town"  # misheard
    assert listen.asked_for(said("switch to spacesong", -40), TITLES, 0, 100) == "Space Song"  # run together


def test_a_chord_still_ringing_as_you_start_to_speak_does_not_matter():
    ringing = said("switch to winter town", -40)
    ringing.instrument_db[:2] = [-5, -10]
    assert listen.asked_for(ringing, TITLES, 0, 100) == "Winter Town"


def test_singing_a_song_s_name_does_not_switch_to_it():
    assert listen.asked_for(said("switch to winter town", -5), TITLES, 0, 100) is None  # the guitar is playing
    assert listen.asked_for(said("switch to something else", -40), TITLES, 0, 100) is None  # no such song
    assert listen.asked_for(said("switch to winter town", -40, at=50), TITLES, 0, 40) is None  # not in this pass
    # A long line that holds a song's name is sung, not asked for.
    assert listen.asked_for(said("and we walked all the way to winter town", -40), TITLES, 0, 100) is None
    # A short one is too, when it's the song's own line (sung without the instrument).
    lyrics = [(w, 0.0, 0.0, 0) for w in "we walked to winter town tonight".split()]
    assert listen.asked_for(said("we walked to winter town", -40), TITLES, 0, 100, lyrics) is None
    assert listen.asked_for(said("switch to winter town", -40), TITLES, 0, 100, lyrics) == "Winter Town"


def test_each_note_is_taken_in_once_from_passes_that_overlap():
    take = sing(off=-3.0)  # sung 3 semitones under throughout, a note every 0.5 s from 1 s
    end = take.notes[-1][1]
    follow = listen.Follow(REFERENCE, 0)
    # Two passes over the same take, the second a little later: nothing is counted twice.
    for later in (end - 5, end + 5):
        follow.update(listen.Heard(later, take, [0.0] * len(take.words)), TITLES)
    assert len(follow.offsets) == len(melody.note_offsets(take, REFERENCE, 0))
    assert listen.settled(follow.offsets)[0] == -3
    assert follow.line == len(LINES) - 1
    # Nothing heard before the song was chosen counts.
    late = listen.Follow(REFERENCE, 0, since=end + 5)
    late.update(listen.Heard(end + 7, take, [0.0] * len(take.words)), TITLES)
    assert late.offsets == []


def test_a_switch_is_heard_once():
    follow = listen.Follow(REFERENCE, 0)
    assert follow.update(said("switch to winter town", -40), TITLES) == "Winter Town"
    assert follow.update(said("switch to winter town", -40), TITLES) is None  # the same words, heard again


def test_notes_are_compared_in_the_key_given():
    assert {round(off, 6) for _, _, off in melody.note_offsets(sing(shift=3), REFERENCE, 3)} == {0}
    assert {round(off, 6) for _, _, off in melody.note_offsets(sing(off=0.2), REFERENCE, 0)} == {0.2}
