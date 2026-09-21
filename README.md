# Woodshed

Record your covers from the terminal. Woodshed works out which song you played and files
each take as `<Song>/<date>_<time>.mp3`, so you can listen back to how every song
evolves.

```
$ shed rec
● REC 03:12  ██████████████░░░░░░░░░░░░░░░░  press any key to stop
Heard: “…”
Recognized Harbor Lights from your earlier takes.
✓ Saved Harbor Lights/2026-09-21_18-30.mp3 (take #7)
```

## How it recognizes a song

1. **Demucs separates your voice from the guitar.** Whisper is trained on speech and
   loses the words under a loud guitar. On test takes with the guitar 6 dB louder than
   the voice, it caught 48% of the words from the raw mix and 98% from the separated
   voice. This adds about 10 s per take on the GPU; skip it with `--no-isolate`.
2. **Whisper transcribes the lyrics** from the first 2½ minutes of the take.
3. **Your earlier takes:** if the lyrics clearly match a song you've already recorded,
   the take is filed there.
4. **Genius:** for a song you haven't recorded before, a few lyric lines are searched on
   Genius. If the same song comes up first more than once, it's filed under that title.
5. **You:** otherwise you're shown the best guesses and pick one or type a name (Tab
   completes your song names). Press Enter to leave the take in `Unsorted/`.

Everything runs on your Mac. Only step 4 goes online, and it sends only a few lyric
lines.

## Setup

Needs a Mac with Apple Silicon.

```sh
brew install lafarguem/tap/woodshed
shed init
```

`shed init` asks three things, then downloads the models (about 1.7 GB, once):

1. **Where to save recordings.** Default: `~/Music/Woodshed`.
2. **Which microphone.** It suggests your Mac's built-in mic. Bluetooth earbuds record at
   phone-call quality.
3. **A Genius access token** (optional) to recognize new songs. Create a free API client
   at https://genius.com/api-clients and click "Generate Access Token". The token is
   checked before it's saved.

Answers go to `~/.config/woodshed/config.toml`, which only you can read. Run
`shed init` again anytime to change them.

## Usage

```sh
shed rec                    # record until you press a key
shed rec --auto-stop 8      # …or until 8 s of silence
shed add ~/Downloads/*.m4a  # file recordings you already have (voice memos: the date is kept)
shed songs                  # every song, with take counts and dates
shed songs harbor           # the takes of one song
shed devices                # microphones you can record from (* = the one in use)
shed init                   # change the folder, microphone or token
```

Flags and environment variables override what `shed init` saved:

| Setting | Flag | Environment variable |
|---|---|---|
| Library folder | `--library` | `WOODSHED_DIR` |
| Input device | `--device` | `WOODSHED_DEVICE` |
| Singing language (detected if unset) | `--language` | `WOODSHED_LANGUAGE` |
| Genius token | | `GENIUS_ACCESS_TOKEN` |
| Whisper model (run `shed init` after changing it) | | `WOODSHED_WHISPER_MODEL` |

## The library is just folders

```
~/Music/Woodshed/
  Harbor Lights/
    2026-06-02_21-14.mp3
    2026-09-21_18-30.mp3
  Unsorted/
```

Each mp3 carries its own transcript in its tags, so you can reorganize in Finder. Move a
misfiled take into the right folder, or rename a folder to rename the song, and Woodshed
follows. Raw recordings wait in the hidden `.incoming/` folder until they're filed, so a
crash never loses a take.

## Development

Needs [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```sh
uv sync
uv run pytest
uv run shed --help
```

## Releasing

Pushing a version tag runs `.github/workflows/release.yml`. It creates a GitHub release
and publishes an updated formula (rendered from `packaging/woodshed.rb.in`) to
[`lafarguem/homebrew-tap`](https://github.com/lafarguem/homebrew-tap).

1. Bump `version` in `pyproject.toml` and refresh the lockfile: `uv lock`
2. Commit the bump: `git commit -am "Release v0.2.0"`
3. Tag and push: `git tag v0.2.0 && git push origin main --tags`
4. Watch it run: `gh run watch`

Users then get it with `brew upgrade woodshed`.

The workflow needs a Personal Access Token with write access to the tap, saved as the
`HOMEBREW_TAP_TOKEN` secret on this repo (the default `GITHUB_TOKEN` can't push to
another repository). Without the secret, the release is still created and the formula
publish is skipped with a warning.

## License

MIT, see [LICENSE](LICENSE).
