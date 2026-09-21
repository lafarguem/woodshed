"""The models Woodshed runs on your Mac. `shed init` downloads them once; after that
recognition works offline (apart from the Genius lookup)."""

from woodshed import isolate, transcribe

WHISPER, DEMUCS, RMVPE = "Whisper (lyrics)", "Demucs (voice isolation)", "RMVPE (pitch)"


def missing() -> list[str]:
    """Models that aren't fully downloaded yet (e.g. after an interrupted `shed init`). Checked without going online."""
    from huggingface_hub import snapshot_download

    from woodshed import rmvpe  # imports torch, which takes half a second: only when a command needs the models

    absent = []
    try:  # Whisper comes as a whole repo, whose cached file list says whether the snapshot is complete
        snapshot_download(transcribe.MODEL, local_files_only=True)
    except Exception:
        absent.append(WHISPER)
    if not _demucs_cached():
        absent.append(DEMUCS)
    if not _cached(rmvpe.HF_REPO, rmvpe.WEIGHTS):  # one file from a larger repo
        absent.append(RMVPE)
    return absent


def _cached(repo: str, file: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, file), str)


def _demucs_cached() -> bool:
    """Demucs's bag file and every model it lists. They're fetched one by one, leaving no file list
    to check, and a missing one would make Demucs quietly download a copy from elsewhere."""
    import yaml  # comes with Demucs
    from huggingface_hub import try_to_load_from_cache

    bag = try_to_load_from_cache(isolate.HF_REPO, f"{isolate.MODEL}.yaml")
    if not isinstance(bag, str):
        return False
    with open(bag) as f:
        signatures = (yaml.safe_load(f) or {}).get("models") or []
    return bool(signatures) and all(_cached(isolate.HF_REPO, f"{sig}.safetensors") for sig in signatures)


def download() -> None:
    from demucs.hf import get_hf_model
    from huggingface_hub import hf_hub_download, snapshot_download

    from woodshed import rmvpe

    snapshot_download(transcribe.MODEL)
    get_hf_model(isolate.MODEL)  # from HF_REPO, and no fallback: a failed download should say so
    hf_hub_download(rmvpe.HF_REPO, rmvpe.WEIGHTS)
