import random

import pytest

from woodshed.lyrics import words

# Invented lyrics, so tests don't depend on real songs.
SONGS = {
    "Harbor Lights": "the lanterns swing above the harbor wall / we counted ships until the evening fall / "
                     "your coat was wet with salt and rain / and every bell was ringing out your name / "
                     "so sail me home across the silver bay / i kept the light on every single day",
    "Kitchen Floor": "coffee on the stove at half past three / you hum a tune that sounds a lot like me / "
                     "the radio is crackling in the hall / and nobody is waiting for the call / "
                     "dance with me barefoot on the kitchen floor / nobody needs to know what we came for",
    "Gravel Road": "gravel on the highway and the dust in my eyes / a thousand yellow lines under purple skies / "
                   "i left my brother standing at the door / he said you wont be coming back no more / "
                   "drive until the engine starts to cry / drive until the morning says goodbye",
    "Winter Town": "snow is falling slowly on the frozen town / i wear the sweater that you handed down / "
                   "the chimney smoke is drawing letters in the air / i read them all and wish that you were there / "
                   "winter hold me winter let me go / i am tired of the cold and the snow",
}


def mishear(text: str, error: float, rng: random.Random, keep: float = 1.0) -> list[str]:
    """Simulate a Whisper transcript of a sung take: verse sung twice, some words misheard, dropped or added."""
    vocab = sorted({w for t in SONGS.values() for w in words(t)}) + ["yeah", "baby", "gonna"]
    sung = (words(text) * 2)[: int(len(words(text)) * 2 * keep)]
    out = []
    for w in sung:
        r = rng.random()
        if r < error / 2:
            out.append(rng.choice(vocab))
        elif r < error * 0.75:
            continue
        elif r < error:
            out += [w, rng.choice(vocab)]
        else:
            out.append(w)
    return [" ".join(out[i:i + 8]) for i in range(0, len(out), 8)]


@pytest.fixture
def rng():
    return random.Random(7)
