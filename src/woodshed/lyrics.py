"""Text helpers for comparing transcribed lyrics."""

import re
import unicodedata

# Word pairs made only of these carry no information about which song it is.
_STOPWORDS = frozenset(
    "a an and are as at be but by do for from i im in is it its me my no not "
    "of oh on or so that the this to we what when you your".split()
)


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"['’]", "", text)
    return re.findall(r"\w+", text)


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    pairs = zip(tokens, tokens[1:])
    return {p for p in pairs if not (p[0] in _STOPWORDS and p[1] in _STOPWORDS)}


def containment(take: set, reference: set) -> float:
    """Share of a take's word pairs that also appear in a reference set."""
    if not take:
        return 0.0
    return len(take & reference) / len(take)
