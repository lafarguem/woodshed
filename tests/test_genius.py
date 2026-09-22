import pytest

from woodshed import genius


def hit(title, artist, id):
    return {"title": title, "primary_artist": {"name": artist}, "id": id}


def fake_search(results_by_query):
    return lambda query, token: results_by_query.get(query, [])


def test_snippets_are_distinct_6_to_10_word_chunks():
    lines = ["one two three four five six seven", "One two three, four five six seven!", "short line",
             "a b c d e f g h i j k l"]
    # Repeats (a chorus) are searched once; short lines join the next; leftovers under 6 words are dropped.
    assert genius.snippets(lines) == ["one two three four five six seven", "short line a b c d e f g h"]


def test_snippets_are_spread_across_the_take():
    lines = [f"line {i} has quite a few words in it" for i in range(20)]
    picked = genius.snippets(lines, limit=5)
    assert picked[0] == lines[0] and picked[-1] == lines[-1] and len(picked) == 5


def test_clean_title_drops_version_tags():
    assert genius.clean_title("Wonder\xa0Song (Remastered 2009)") == "Wonder Song"
    assert genius.clean_title("Wonder Song [Live at Home]") == "Wonder Song"
    assert genius.clean_title("Song (Part II)") == "Song (Part II)"


def test_covers_vote_together_and_translations_are_ignored(monkeypatch):
    lines = [f"lyric line number {i} of the song" for i in range(3)]
    monkeypatch.setattr(genius, "search", fake_search({
        lines[0]: [hit("Harbor Lights", "Genius English Translations", 9), hit("Harbor Lights", "The Originals", 1)],
        lines[1]: [hit("Harbor Lights (Acoustic)", "Some Cover Band", 2), hit("Other Song", "X", 3)],
        lines[2]: [hit("Other Song", "X", 3), hit("Harbor Lights", "The Originals", 1)],
    }))
    ranked = genius.identify(lines, "token")
    assert (ranked[0].title, ranked[0].artist, ranked[0].first_places) == ("Harbor Lights", "The Originals", 2)
    assert genius.is_confident(ranked)


@pytest.mark.parametrize("firsts, confident", [((2, 1), True), ((1, 0), False), ((2, 2), False), ((3,), True)])
def test_confidence_needs_two_clear_first_places(firsts, confident):
    ranked = [genius.Candidate(f"song {i}", "a", i, first_places=n) for i, n in enumerate(firsts)]
    assert genius.is_confident(ranked) is confident


def linked(kind, *songs):
    """A song's page on Genius, linked to `songs` as `kind` (its "cover_of"…)."""
    return {"song_relationships": [{"relationship_type": kind, "songs": list(songs)}]}


def found_first(monkeypatch, version):
    """Two lines of Harbor Lights, for which Genius finds `version` of it first and the original second."""
    lines = [f"lyric line number {i} of the song" for i in range(2)]
    monkeypatch.setattr(genius, "search", fake_search({
        line: [version, hit("Harbor Lights", "The Originals", 1)] for line in lines}))
    return lines


@pytest.mark.parametrize("version, kind", [(hit("Harbor Lights", "Some Cover Singer", 2), "cover_of"),
                                           (hit("Harbor Lights (Live in Lisbon)", "The Originals", 2), "live_version_of")])
def test_a_cover_or_live_version_found_first_is_taken_back_to_its_original(monkeypatch, genius_pages, version, kind):
    genius_pages[2] = linked(kind, hit("Harbor Lights", "The Originals", 1))
    looked_up = []
    monkeypatch.setattr(genius, "song", lambda genius_id, token: looked_up.append(genius_id) or genius_pages[genius_id])

    ranked = genius.identify(found_first(monkeypatch, version), "token")

    assert (ranked[0].title, ranked[0].artist, ranked[0].genius_id) == ("Harbor Lights", "The Originals", 1)
    assert ranked[0].first_places == 2 and genius.is_confident(ranked)
    assert looked_up == [2]  # only the song it's likely to be


@pytest.mark.parametrize("page", [
    linked("cover_of", hit("Luces del Puerto", "Los Originales", 1)),  # a translation: sung to other words
    linked("cover_of", hit("Harbor Lights", "The Originals", 1), hit("Winter Town", "The Snowmen", 4)),  # a medley
    linked("samples", hit("Harbor Lights", "The Originals", 1)),
    {},  # not linked to any song
])
def test_otherwise_the_song_found_first_stays(monkeypatch, genius_pages, page):
    genius_pages[2] = page
    ranked = genius.identify(found_first(monkeypatch, hit("Harbor Lights", "Some Cover Singer", 2)), "token")
    assert (ranked[0].title, ranked[0].artist, ranked[0].genius_id) == ("Harbor Lights", "Some Cover Singer", 2)


def test_the_song_found_first_stays_if_genius_cant_say_what_its_a_cover_of(monkeypatch):
    def unreachable(genius_id, token):
        raise genius.GeniusError("Couldn't reach Genius (timed out)")

    monkeypatch.setattr(genius, "song", unreachable)
    ranked = genius.identify(found_first(monkeypatch, hit("Harbor Lights", "Some Cover Singer", 2)), "token")
    assert (ranked[0].artist, ranked[0].genius_id) == ("Some Cover Singer", 2)
