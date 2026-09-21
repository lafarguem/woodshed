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
