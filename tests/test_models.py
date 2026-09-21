import huggingface_hub

from woodshed import isolate, models, rmvpe, transcribe


def cache(root, repo, files):
    """A repo in a Hugging Face cache laid out like the real one, holding `files` {name: content}."""
    folder = root / f"models--{repo.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text("abc123")
    snapshot = folder / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for name, content in files.items():
        (snapshot / name).write_text(content)
    return snapshot


def test_a_half_downloaded_model_counts_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    assert models.missing() == [models.WHISPER, models.DEMUCS, models.RMVPE]

    cache(tmp_path, transcribe.MODEL, {"config.json": "{}", "weights.safetensors": ""})
    cache(tmp_path, rmvpe.HF_REPO, {rmvpe.WEIGHTS: ""})
    demucs = cache(tmp_path, isolate.HF_REPO, {"htdemucs.yaml": "models: ['955717e8']\n", "955717e8.safetensors": ""})
    assert models.missing() == []

    # `shed init` stopped after Demucs's bag file, before its weights: re-running it must download them.
    (demucs / "955717e8.safetensors").unlink()
    assert models.missing() == [models.DEMUCS]
