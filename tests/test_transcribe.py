import sys
from types import SimpleNamespace

from woodshed import transcribe


def test_lines_come_with_when_each_word_is_sung(monkeypatch):
    def heard(*words):  # Whisper's words: each with its leading space, and when it's sung
        return {"text": "".join(words), "words": [{"word": w, "start": i, "end": i + 0.5} for i, w in enumerate(words)]}

    segments = [heard(" [Music]"), heard(" Thank", " you."),
                heard(" Don't", " let", " (guitar", " strumming)", " the", " light", " go", " ♪")]
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=lambda *a, **kw: {"segments": segments}))

    [line] = transcribe.transcribe(None)

    assert line.text == "Don't let the light go"
    assert [(w.text, w.start) for w in line.words] == [("dont", 0), ("let", 1), ("the", 4), ("light", 5), ("go", 6)]


def test_the_same_take_is_always_heard_the_same_way(monkeypatch):
    import mlx.core as mx

    def whisper(*args, **kwargs):  # decodes a line again at random, as Whisper does when it goes badly
        guess = int(mx.random.randint(0, 1_000_000, [1]).item())
        return {"segments": [{"text": f" line {guess}", "words": [{"word": f" line{guess}", "start": 0, "end": 1}]}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=whisper))
    mx.random.seed(123)
    first = transcribe.transcribe(None)
    mx.random.randint(0, 10, [5])  # whatever else ran in between
    assert transcribe.transcribe(None) == first
