"""A song's melody from a tab (tab.py): reading a Guitar Pro file, made up here bar by bar, and laying its voice
over the melody heard in a recording."""

import json
import zipfile

import pytest
from test_melody import sing
from test_reference import harbor, set_reference, text  # noqa: F401 (a fixture)

from woodshed import along, tab

TUNE = [57, 60, 62, 64, 60, 57, 55, 57]  # a melody, a quarter note each


def gp(path, tracks, tempo="120 2", repeats=None, time="4/4", alternate=False):
    """A Guitar Pro file: `tracks` are {name: bars}, a bar a list of beats (MIDI note or None for a rest, note
    value, and optionally "dot", "triplet" or "tie"). `repeats` are {bar: (starts, ends, count)}."""
    values = ["Whole", "Half", "Quarter", "Eighth", "16th"]
    rhythms, beats, notes, voices, bars, masters = {}, [], [], [], [], []
    for rhythm, (value, extra) in enumerate((v, e) for v in values for e in ("", "dot", "triplet")):
        rhythms[value, extra] = rhythm
    names = list(tracks)
    for number in range(len(tracks[names[0]])):
        ids = []
        for name in names:
            beat_ids = []
            for pitch, value, *extra in tracks[name][number]:
                kind = next((e for e in extra if e in ("dot", "triplet")), "")
                note = ""
                if pitch is not None:
                    tie = '<Tie origin="false" destination="true"/>' if "tie" in extra else ""
                    notes.append(f'<Note id="{len(notes)}">{tie}<Properties><Property name="Midi"><Number>{pitch}'
                                 "</Number></Property></Properties></Note>")
                    note = f"<Notes>{len(notes) - 1}</Notes>"
                beats.append(f'<Beat id="{len(beats)}"><Rhythm ref="{rhythms[value, kind]}"/>{note}</Beat>')
                beat_ids.append(str(len(beats) - 1))
            voices.append(f'<Voice id="{len(voices)}"><Beats>{" ".join(beat_ids)}</Beats></Voice>')
            bars.append(f'<Bar id="{len(bars)}"><Voices>{len(voices) - 1} -1 -1 -1</Voices></Bar>')
            ids.append(str(len(bars) - 1))
        starts, ends, count = (repeats or {}).get(number, (False, False, 0))
        repeat = (f'<Repeat start="{str(starts).lower()}" end="{str(ends).lower()}" count="{count}"/>'
                  if starts or ends else "")
        ending = "<AlternateEndings>1</AlternateEndings>" if alternate and number == 0 else ""
        masters.append(f"<MasterBar><Time>{time}</Time>{repeat}{ending}<Bars>{' '.join(ids)}</Bars></MasterBar>")
    written = "".join(
        f'<Rhythm id="{i}"><NoteValue>{value}</NoteValue>'
        + ('<AugmentationDot count="1"/>' if extra == "dot" else "")
        + ('<PrimaryTuplet num="3" den="2"/>' if extra == "triplet" else "") + "</Rhythm>"
        for (value, extra), i in rhythms.items())
    score = ("<GPIF><MasterTrack><Automations><Automation><Type>Tempo</Type><Bar>0</Bar><Position>0</Position>"
             f"<Value>{tempo}</Value></Automation></Automations></MasterTrack><Tracks>"
             + "".join(f'<Track id="{i}"><Name>{name}</Name></Track>' for i, name in enumerate(names))
             + f"</Tracks><MasterBars>{''.join(masters)}</MasterBars><Bars>{''.join(bars)}</Bars>"
             f"<Voices>{''.join(voices)}</Voices><Beats>{''.join(beats)}</Beats><Notes>{''.join(notes)}</Notes>"
             f"<Rhythms>{written}</Rhythms></GPIF>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Content/score.gpif", score)
    return path


def song(bars=12, shift=0):
    """A voice singing TUNE over and over, a bass on one note, and drums."""
    voice = [[(TUNE[(4 * bar + beat) % len(TUNE)] + shift, "Quarter") for beat in range(4)] for bar in range(bars)]
    return {"Bass": [[(33, "Half"), (33, "Half")]] * bars, "Singer": voice, "Drums": [[(36, "Quarter")] * 4] * bars}


def heard(notes, seconds, wobble=0.3):
    """The melody a recording's singer would be heard to sing on these notes (start, end, semitones from A440): a
    little off each, and silent for the first and last twentieth of a second of each."""
    contour = [None] * int(seconds / along.CONTOUR_HOP)
    for k, (start, end, pitch) in enumerate(notes):
        for i in range(int((start + 0.05) / along.CONTOUR_HOP), min(int((end - 0.05) / along.CONTOUR_HOP), len(contour))):
            contour[i] = pitch + wobble * (-1) ** k
    return contour


def test_notes_come_with_their_times_at_the_tabs_tempo(tmp_path):
    bars = [[(60, "Quarter"), (None, "Eighth"), (62, "Eighth"), (64, "Eighth"), (65, "Quarter", "dot")],
            [(67, "Quarter", "triplet")] * 3 + [(69, "Half")]]
    [track] = tab.tracks(gp(tmp_path / "song.gp", {"Voice": bars}, tempo="120 2"))
    assert track.name == "Voice"
    assert [(round(a, 3), round(b, 3), p) for a, b, p in track.notes] == [
        (0.0, 0.5, 60), (0.75, 1.0, 62), (1.0, 1.25, 64), (1.25, 2.0, 65),  # a dotted quarter is 3 eighths
        (2.0, 2.333, 67), (2.333, 2.667, 67), (2.667, 3.0, 67), (3.0, 4.0, 69)]  # 3 triplet quarters make a half


def test_a_tied_note_carries_on_and_a_tempo_counts_its_own_beat(tmp_path):
    bars = [[(60, "Half"), (62, "Half")], [(62, "Half", "tie"), (62, "Half")]]
    [track] = tab.tracks(gp(tmp_path / "song.gp", {"Voice": bars}, tempo="60 3"))  # 60 dotted quarters a minute
    assert [(round(a, 2), round(b, 2), p) for a, b, p in track.notes] == [(0, 1.33, 60), (1.33, 4.0, 62), (4.0, 5.33, 62)]


def test_a_repeated_section_is_played_the_times_it_says(tmp_path):
    bars = [[(60, "Whole")], [(62, "Whole")], [(64, "Whole")]]
    [track] = tab.tracks(gp(tmp_path / "song.gp", {"Voice": bars}, repeats={0: (True, False, 0), 1: (False, True, 3)}))
    assert [p for _, _, p in track.notes] == [60, 62, 60, 62, 60, 62, 64]
    assert track.notes[-1][:2] == (12.0, 14.0)


def test_what_isnt_a_readable_tab_is_refused(tmp_path):
    (tmp_path / "old.gp5").write_bytes(b"FICHIER GUITAR PRO v5.00")
    with pytest.raises(tab.TabError, match="Guitar Pro 7 or later"):
        tab.tracks(tmp_path / "old.gp5")
    with pytest.raises(tab.TabError, match="alternate endings"):
        tab.tracks(gp(tmp_path / "song.gp", {"Voice": [[(60, "Whole")]]}, alternate=True))


def test_the_voice_is_found_and_laid_over_the_recording(tmp_path):
    tracks = tab.tracks(gp(tmp_path / "song.gp", song()))
    sung = [(a * 1.02 + 3.2, b * 1.02 + 3.2, p - 69 - 12) for a, b, p in tracks[1].notes]  # slower, later, lower
    placed = tab.place(tracks, heard(sung, 32), along.CONTOUR_HOP)
    assert placed.track.name == "Singer" and placed.key == 0 and placed.share > 0.6
    assert len(placed.notes) == len(sung)
    for (start, end, pitch), (a, b, p) in zip(placed.notes, sung):
        assert start == pytest.approx(a, abs=0.08) and end == pytest.approx(b, abs=0.08)
        assert pitch == p  # the written note, in the singer's octave: not what was heard, 30 cents off it


def test_a_tab_written_as_fingered_under_a_capo_is_moved_to_the_recordings_key(tmp_path):
    tracks = tab.tracks(gp(tmp_path / "song.gp", song(shift=-3)))
    sung = [(a + 1.0, b + 1.0, p - 69 + 3) for a, b, p in tracks[1].notes]
    placed = tab.place(tracks, heard(sung, 30), along.CONTOUR_HOP)
    assert placed.key == 3 and [p for _, _, p in placed.notes] == [p for _, _, p in sung]


def test_a_voice_is_found_where_the_tab_writes_the_upper_of_two_stacked_voices(tmp_path, rng):
    # Where the recording stacks two voices and the tab writes the upper one, the lower one may be what's heard:
    # here a third under it (major or minor) in 2 bars of 3. Few of the voice's notes fall on the singer's, but they
    # do there more than anywhere else. Over a song's length, that is: not over a few bars of a tune going round.
    tune = rng.choices([57, 59, 60, 62, 64, 65, 67, 69], k=160)
    tracks = tab.tracks(gp(tmp_path / "song.gp", {**song(40), "Singer": [
        [(tune[4 * bar + beat], "Quarter") for beat in range(4)] for bar in range(40)]}))
    sung = [(a + 1.5, b + 1.5, p - 69 - [0, 4, 3][(k // 4) % 3]) for k, (a, b, p) in enumerate(tracks[1].notes)]
    placed = tab.place(tracks, heard(sung, 84), along.CONTOUR_HOP)
    assert placed.track.name == "Singer" and placed.key == 0 and tab.MIN_SHARE < placed.share < 0.35
    assert placed.notes[0][:2] == pytest.approx((1.5, 2.0), abs=0.08)


def test_a_tab_of_another_song_isnt_laid(tmp_path, rng):
    tracks = tab.tracks(gp(tmp_path / "song.gp", song()))
    other = [(k * 0.4, k * 0.4 + 0.4, rng.choice([-12, -10, -7, -5, -3])) for k in range(70)]
    assert tab.place(tracks, heard(other, 30), along.CONTOUR_HOP) is None
    assert tab.place(tracks, [None] * 1000, along.CONTOUR_HOP) is None


def test_shed_reference_takes_a_tab_and_singing_along_goes_by_its_notes(harbor, tmp_path):
    from woodshed import cli

    set_reference(harbor)
    file = gp(tmp_path / "harbor.gp", song())
    tracks = tab.tracks(file)
    folder = harbor.library.root / "Harbor Lights"
    kept = json.loads((folder / ".reference.json").read_text())
    kept["contour"] = heard([(a + 2.0, b + 2.0, p - 69) for a, b, p in tracks[1].notes], 28)
    (folder / ".reference.json").write_text(json.dumps(kept))

    result = harbor("reference", "harbor", "--tab", str(file))
    assert result.exit_code == 0, result.output
    assert "Singing along to “Harbor Lights” now goes by the 48 notes of “Singer” in harbor.gp." in text(result)
    reference = harbor.library.reference("Harbor Lights")
    assert reference.tab["track"] == "Singer" and reference.tab["notes"][0] == pytest.approx([2.0, 2.5, -12.0], abs=0.08)
    assert cli._to_sing_along(harbor.library, "Harbor Lights").notes == reference.tab["notes"]

    assert "goes by the melody heard in its recording again" in text(harbor("reference", "harbor", "--no-tab"))
    assert harbor.library.reference("Harbor Lights").tab is None
    assert cli._to_sing_along(harbor.library, "Harbor Lights").notes is None
    assert "has no tab" in text(harbor("reference", "harbor", "--no-tab"))


def test_a_tab_needs_a_reference_kept_and_a_file_that_fits(harbor, tmp_path):
    file = gp(tmp_path / "harbor.gp", song())
    harbor.library.set_reference("Harbor Lights", sing(), "original.mp3")  # as before recordings were kept
    result = harbor("reference", "harbor", "--tab", str(file))
    assert result.exit_code == 1 and "“Harbor Lights” has none kept" in text(result)

    set_reference(harbor)  # its singer holds one note: no track follows that
    result = harbor("reference", "harbor", "--tab", str(file))
    assert result.exit_code == 1 and "None of the tracks of harbor.gp follows the melody" in text(result)
    result = harbor("reference", "harbor", "--tab", str(tmp_path / "missing.gp"))
    assert result.exit_code == 1 and "There's no tab at" in text(result)
    (tmp_path / "notes.gp").write_text("not a tab")
    result = harbor("reference", "harbor", "--tab", str(tmp_path / "notes.gp"))
    assert result.exit_code == 1 and "Can't read notes.gp" in text(result)
