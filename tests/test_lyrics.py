from conftest import SONGS, mishear

from woodshed.library import MATCH_MARGIN, MATCH_SCORE
from woodshed.lyrics import bigrams, containment, words


def test_words_ignores_case_punctuation_and_apostrophes():
    assert words("Don't STOP, me now!") == ["dont", "stop", "me", "now"]


def test_stopword_only_pairs_are_dropped():
    assert bigrams(["in", "the", "harbor"]) == {("the", "harbor")}


def test_takes_of_the_same_song_clear_the_thresholds_and_others_dont(rng):
    for _ in range(50):
        references = {s: bigrams(words(" ".join(mishear(t, 0.3, rng)))) for s, t in SONGS.items()}
        for song, text in SONGS.items():
            take = bigrams(words(" ".join(mishear(text, 0.3, rng, keep=rng.uniform(0.4, 1)))))
            same = containment(take, references[song])
            other = max(containment(take, ref) for s, ref in references.items() if s != song)
            assert same >= MATCH_SCORE and same - other >= MATCH_MARGIN
