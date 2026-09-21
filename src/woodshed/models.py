"""The models Woodshed runs on your Mac. `shed init` downloads them once; after that
recognition works offline (apart from the Genius lookup)."""

from woodshed import isolate, transcribe

REPOS = {"Whisper (lyrics)": transcribe.MODEL, "Demucs (voice isolation)": isolate.HF_REPO}


def missing() -> list[str]:
    """Models that aren't downloaded yet. Checked without going online."""
    from huggingface_hub import snapshot_download

    absent = []
    for name, repo in REPOS.items():
        try:
            snapshot_download(repo, local_files_only=True)
        except Exception:
            absent.append(name)
    return absent


def download() -> None:
    from demucs.pretrained import get_model
    from huggingface_hub import snapshot_download

    snapshot_download(transcribe.MODEL)
    get_model(isolate.MODEL)  # fetches just the files Demucs needs from HF_REPO
