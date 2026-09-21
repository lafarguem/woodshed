# Woodshed

Record your covers from the terminal, singing along to a guitar, a piano or any other
instrument. Woodshed works out which song you played, files each take as
`<Song>/<date>_<time>.mp3` and rates it, so you can hear and see how every song evolves.

```
$ shed rec
● REC 03:12  ██████████████░░░░░░░░░░░░░░░░  press any key to stop
Heard: “…”
Recognized Harbor Lights from your earlier takes.
✓ Saved Harbor Lights/2026-09-21_18-30.mp3 (take #7)
Rated 7.4/10 · pitch 7.2/10 (11¢ off) · timing 7.8/10 (tempo ±2.1%) · your best: 7.9
```

## How it recognizes a song

1. **Demucs separates your voice from the instrument.** Whisper is trained on speech and
   loses the words under a loud accompaniment. On test takes with a guitar 6 dB louder
   than the voice, it caught 48% of the words from the raw mix and 98% from the separated
   voice. It takes about 4 s per minute of recording on the GPU. Skip it with
   `--no-isolate`; the take is then rated later, when `progress` or `play --rating` needs it.
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

`shed init` asks four things, then downloads the models (about 1.9 GB, once). Pick with the
arrow keys. Every question has a default, so pressing Enter all the way through works:

1. **Where to save recordings.** Default: `~/Music/Woodshed`.
2. **Which microphone,** from a list that shows how each is connected. It suggests an audio
   interface if you have one, else your Mac's built-in mic. Bluetooth earbuds record at
   phone-call quality.
3. **A Genius access token** (optional) to recognize new songs. Create a free API client
   at https://genius.com/api-clients and click "Generate Access Token". The token is
   checked before it's saved.
4. **How strict ratings should be** (optional): sliders for the reference points below, with a
   live preview of what example takes would score. Changing them re-scores every take
   instantly.

Answers go to `~/.config/woodshed/config.toml`, which only you can read. Run
`shed init` again anytime to change them.

## Usage

```sh
shed rec                    # record until you press a key
shed rec --auto-stop 8      # …or until 8 s of silence
shed add ~/Downloads/*.m4a  # file recordings you already have (voice memos: the date is kept)
shed songs                  # every song, with take counts and dates
shed songs harbor           # the takes of one song
shed play harbor            # its first take, then its latest (any key skips)
shed progress harbor        # every take's rating, first to latest
shed play harbor --rating   # your worst take, then your best (or add: pitch, timing)
shed devices                # microphones you can record from, and how they're connected
shed init                   # change the folder, microphone, token or rating strictness
```

Flags and environment variables override what `shed init` saved:

| Setting | Flag | Environment variable |
|---|---|---|
| Library folder | `--library` | `WOODSHED_DIR` |
| Input device | `--device` | `WOODSHED_DEVICE` |
| Singing language (detected if unset) | `--language` | `WOODSHED_LANGUAGE` |
| Genius token | | `GENIUS_ACCESS_TOKEN` |
| Whisper model (run `shed init` after changing it) | | `WOODSHED_WHISPER_MODEL` |

## How takes are rated

Each take gets a rating out of 10 when it's filed. It's 60% pitch and 40% timing, and it
runs locally in a few seconds. Takes filed before rating existed get rated the first time
`progress` or `play --rating` needs them.

| | What's measured | 10/10 at | 0/10 at |
|---|---|---|---|
| **Pitch** | How far your held notes are from the nearest true note, in cents (1/100 of a semitone), compared with your instrument's tuning. Singing along with an instrument tuned a bit sharp isn't penalized. | 5¢ | 25¢ (random notes) |
| **Timing** | How much your tempo wanders across the song, followed through the instrument's attacks (strums, piano chords). For sustained sounds with hardly any attacks (organ, pads, bowed strings), it follows when the chords change instead. | ±1% | ±6% |

**How it works:** [RMVPE](https://arxiv.org/abs/2306.15412), a pitch model built for singing
voices, reads your voice straight from the recording. Demucs separates the instrument to find
its tuning and tempo.

**How accurate it is:** these results come from synthetic takes with known flaws, with
guitar, piano and organ accompaniment.
- Pitch read within 1.5¢ of the truth, and a singer 20¢ flat against the instrument dropped
  to 6.0.
- Rushing 12% over a song cost 3 points of timing on guitar or piano, and 6 on organ.
- Sloppy strums (±40 ms) cost 6 points.
- Following chord changes is a little less precise than following attacks, so a perfectly
  steady organ scores about 9 on timing.

**What it can't tell:**
- stops and restarts
- the timing of individual notes or strums
- timing when you sing a cappella (pitch then measures how consistent your notes are with
  each other)
- whether you sang the right melody (a wrong note sung in tune counts as in tune)
- tone and expression

The reference points are judgment calls, tuned on synthetic takes. Change them with
`shed init` (or the `pitch_*` and `timing_*` keys in `config.toml`). Scores are recomputed
from each take's saved measurements, so nothing gets re-analyzed.

## The library is just folders

```
~/Music/Woodshed/
  Harbor Lights/
    2026-06-02_21-14.mp3
    2026-09-21_18-30.mp3
  Unsorted/
```

Each mp3 carries its own transcript and rating in its tags, so you can reorganize in Finder. Move a
misfiled take into the right folder, or rename a folder to rename the song, and Woodshed
follows. Raw recordings wait in the hidden `.incoming/` folder until they're filed, so a
crash never loses a take: `shed songs` tells you when one is waiting, and `shed add` files it
(then removes the raw copy).

## Development

Needs [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```sh
uv sync
uv run pytest
uv run shed --help
```

## Releasing

Pushing a version tag runs `.github/workflows/release.yml`. It runs the tests on an Apple
Silicon Mac with Homebrew's Python 3.14 (what the formula installs, and what `.python-version`
pins for development). If they pass, it creates a GitHub release and publishes an updated
formula (rendered from `packaging/woodshed.rb.in`) to
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
