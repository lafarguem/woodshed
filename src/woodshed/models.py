"""The models Woodshed runs on your Mac. `shed init` downloads them once; after that
recognition works offline (apart from the Genius lookup)."""

from woodshed import isolate, rmvpe, transcribe

REPOS = {"Whisper (lyrics)": transcribe.MODEL, "Demucs (voice isolation)": isolate.HF_REPO}
FILES = {"RMVPE (pitch)": (rmvpe.HF_REPO, rmvpe.WEIGHTS)}  # one file from a larger repo


def missing() -> list[str]:
    """Models that aren't downloaded yet. Checked without going online."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    absent = [name for name, (repo, file) in FILES.items() if not isinstance(try_to_load_from_cache(repo, file), str)]
    for name, repo in REPOS.items():
        try:
            snapshot_download(repo, local_files_only=True)
        except Exception:
            absent.append(name)
    return absent


def download() -> None:
    from demucs.pretrained import get_model
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(transcribe.MODEL)
    get_model(isolate.MODEL)  # fetches just the files Demucs needs from HF_REPO
    for repo, file in FILES.values():
        hf_hub_download(repo, file)
