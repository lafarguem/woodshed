# Woodshed

Record your covers from the terminal, singing along to a guitar, a piano or any other
instrument. Woodshed works out which song you played, files each take as
`<Song>/<date>_<time>.mp3` and rates it, so you can hear and see how every song evolves.

```
$ shed rec
● REC 03:12  ██████████████░░░░░░░░░░░░░░░░  press any key to stop
Heard: “…”
Recognized Harbor Lights from your earlier takes.
✓ Saved Harbor Lights/2026-09-21_18-30.mp3 (take 7 of 7)
Rated 7.5/10 · pitch 7.3/10 (21¢ off) · timing 7.8/10 (tempo ±2.1%) · your best: 7.9
```

## How it recognizes a song

1. **Demucs separates your voice from the instrument.** Whisper is trained on speech and
   loses the words under a loud accompaniment. On test takes with a guitar 6 dB louder
   than the voice, it caught 48% of the words from the raw mix and 98% from the separated
   voice. It takes about 4 s per minute of recording on the GPU. Skip it with
   `--no-isolate`; the take is then rated later, when `progress` or `play --rating` needs it.
2. **Whisper transcribes the lyrics** of the whole take, with when each word is sung (for rating
   against a reference melody, below).
3. **Your earlier takes:** if the lyrics clearly match a song you've already recorded,
   the take is filed there.
4. **Genius:** for a song you haven't recorded before, a few lyric lines are searched on
   Genius. If the same song comes up first more than once, it's filed under that title.
5. **You:** otherwise you're shown the best guesses and pick one or type a name (Tab
   completes your song names). Press Enter to leave the take in `Unsorted/`.

Everything runs on your Mac. Only step 4 goes online, and it sends only a few lyric
lines. The only other time Woodshed goes online is when you ask it to find a song's original on YouTube
(see [Rating against the original melody](#rating-against-the-original-melody)).

## Setup

Needs a Mac with Apple Silicon.

```sh
brew install lafarguem/tap/woodshed
shed init
```

`shed init` asks four things, then downloads the models (about 1.9 GB, once). Pick with the
arrow keys. Every question has a default, so pressing Enter all the way through works:

1. **Where to save recordings, and how.** Default: `~/Music/Woodshed`, as MP3 (about 1.4 MB a minute).
   Apple Lossless (`.m4a`) keeps takes exactly as recorded, in files about 4 times larger. Changing it
   only affects the takes filed from then on, and `shed rec --m4a` (or `--mp3`) decides for one take.
2. **Which microphone,** from a list that shows how each is connected. It suggests an audio
   interface if you have one, else your Mac's built-in mic. Bluetooth earbuds record at
   phone-call quality.
3. **A Genius access token** (optional) to recognize new songs. Create a free API client
   at https://genius.com/api-clients and click "Generate Access Token". The token is
   checked before it's saved.
4. **How takes are rated** (optional): sliders for the reference points below, and for how much
   pitch and timing each count in the rating, with a live preview of what example takes would score.
   Changing them re-scores every take instantly.

Answers go to `~/.config/woodshed/config.toml`, which only you can read. Run
`shed init` again anytime to change them.

## Usage

```sh
shed rec                    # record until you press a key
shed rec --auto-stop 8      # …or until 8 s of silence
shed rec --later            # only record, to record the next take right away…
shed rec --m4a              # save this take as Apple Lossless (or --mp3), whatever `shed init` chose
shed add                    # …then file every take recorded that way
shed add ~/Downloads/*.m4a  # file recordings you already have (voice memos: the date is kept)
shed add ~/Downloads/Memos  # …or a whole folder: it picks out the songs (see below)
shed songs                  # every song, with take counts and dates
shed songs harbor           # the takes of one song
shed play harbor            # its first take, then its latest (any key skips)
shed progress harbor        # every take's rating, first to latest
shed play harbor --rating   # your worst take, then your best (or --pitch, --timing: on that alone)
shed reference ~/Music/harbor-lights.mp3  # rate a song's pitch against the original's melody
shed reference harbor       # …or find the original on YouTube (needs yt-dlp, see below)
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

### Recording several takes in a row

Filing a take takes about a minute (separating the voice, transcribing the lyrics, rating it). To keep
playing instead, record with `shed rec --later`: the take is kept as soon as you stop, and you can
record the next one. `shed add` then files them all, oldest first, and asks about the songs it isn't sure
of at the end, once it has filed the rest (naming one often makes the others clear). A silent take is
still caught as soon as you stop, so a microphone that isn't allowed doesn't cost you a session.

### Adding a backlog

Give `shed add` a folder of old recordings, and it files the songs in it:

1. **A quick check for singing** comes first, about a second per minute of recording. Recordings
   without singing (talking, an instrument alone), and those shorter than 20 s, are skipped, before
   anything is transcribed or sent to Genius. Hidden files, like a recorder app's recently deleted
   recordings, are left out.
2. **The songs are recognized as usual,** oldest first, so each take filed helps recognize the later
   ones.
3. **The ones it can't name are kept for the end,** one question each. Enter files a take in
   `Unsorted/`, and `-` skips it.

Run `shed add --dry-run <folder>` first to see what it would file and skip. Recordings already filed
or skipped are remembered, so adding the folder again only looks at the new ones. A file named on
its own (`shed add memo.m4a`) is always filed.

## How takes are rated

Each take gets a rating out of 10 when it's filed. It's 60% pitch and 40% timing unless you
choose otherwise in `shed init`, and it runs locally in a few seconds. Takes filed before rating existed, or rated by an earlier version
of the measurements, get rated the first time `progress` or `play --rating` needs them.

| | What's measured | 10/10 at | 0/10 at |
|---|---|---|---|
| **Pitch** | How far your held notes typically are from the nearest note of the song's scale, in cents (1/100 of a semitone), compared with your instrument's tuning. Singing along with an instrument tuned a bit sharp isn't penalized. | 12¢ | 38¢ (random notes) |
| **Timing** | How much your tempo wanders across the song, followed through the instrument's attacks (strums, piano chords). For sustained sounds with hardly any attacks (organ, pads, bowed strings), it follows when the chords change instead. | ±1% | ±6% |

**How it works:** [RMVPE](https://arxiv.org/abs/2306.15412), a pitch model built for singing
voices, reads your voice straight from the recording. Demucs separates the instrument to find
its tuning and tempo.
- Notes are measured against the scale your singing fits best (major, or its relative minor),
  not against all 12 notes. Against all 12, a note more than 50¢ off would count as nearer the
  next one, so random notes would read 25¢ and nothing could read worse. Against the scale's 7,
  errors up to 100¢ show, and random notes land around 38¢.
- The typical (median) note is what counts, so the odd misheard note, or one outside the scale,
  counts for little.
- The tempo is followed within ±13% of the song's own: searched wider, a steady beat gets misread
  now and then as 3/4 or 4/3 of itself. Stretches where the beat isn't found are left out.

**How accurate it is:**
- Professional recordings (a studio recording and two acoustic covers) score 8.7 to 9.9. A
  loose, stripped-back performance scores 5.4, on its timing.
- On synthetic takes with known flaws, pitch reads within 2-3¢ of the truth, even with vibrato,
  scoops into notes and falls off them. A singer 20¢ off on every note scores about 6.8 on pitch.
- On a professional voice, RMVPE and an unrelated pitch tracker (YIN) agree within 1¢.
- On synthetic takes, rushing 12% over a song costs about 5 points of timing, on guitar or organ
  alike, and strums off by up to ±40 ms about 4.5. A perfectly steady player scores 10.

**What it can't tell:**
- stops and restarts
- the timing of individual notes or strums
- timing when you sing a cappella (pitch then measures how consistent your notes are with
  each other)
- whether you sang the right melody (a wrong note sung in tune counts as in tune), unless the song
  has a reference recording (see below)
- tone and expression

The reference points are judgment calls, set so that professional recordings score about 9. Change
them with `shed init` (or the `pitch_*` and `timing_*` keys in `config.toml`), along with pitch's
share of the rating (`pitch_weight_percent`). Scores are recomputed from each take's saved
measurements, so nothing gets re-analyzed.

### Rating against the original melody

The scale can't tell a wrong note sung in tune from the right one, nor a take sung a semitone under
your guitar from one in key. Give a song a reference recording, such as the original, and its pitch is
rated against that melody instead:

```
$ shed reference ~/Music/harbor-lights.mp3
It sounds like “Harbor Lights”. Rate that song against it? [y/n] (y):
✓ “Harbor Lights” is now rated against the melody of harbor-lights.mp3 (38 lines heard).
$ shed rec
…
✓ Saved Harbor Lights/2026-09-22_18-30.mp3 (take 8 of 8)
Rated 6.1/10 · pitch 5.0/10 (58¢ off the melody) · timing 7.8/10 (tempo ±2.1%) · your best: 6.4
Your lines sit a little under the melody (about 40¢).
Furthest from the melody:
  1:12  “and every bell was ringing out your name”  130¢ under
  0:31  “we counted ships until the evening fall”  85¢ over
```

**How it works:**
1. The reference is analyzed like a take: its voice separated, its lyrics transcribed with when each word
   is sung, and the notes held on each word and the chords under it. That's kept in the song's folder
   (`.reference.json`); the recording itself isn't copied.
2. Each line of your take is placed where it fits best among the reference's words, misheard, missing and
   extra words allowed, so verses sung in another order, or left out, don't matter.
3. Your instrument's key is read from its chords under the matched words, so a capo or a transposition is
   followed: the melody is moved to your key, and your voice is judged against your instrument. (Sung a
   cappella, the melody moves to the key you sing in.)
4. Each note you hold on a matched word is compared with the note the reference sings on it, octaves
   aside, and nothing is capped: a note a semitone off counts as a semitone off. A line's offset is the
   median of its notes', and the rating goes by how far your lines typically are from the melody: 10/10 at
   15¢, 0/10 at 100¢ (a semitone). Since a line's notes are pooled first, a line sung flat throughout
   costs more than one whose notes stray either way.
5. When most of your lines sit on the same side of the melody, you're told, and up to 3 lines 60¢ or more
   off are pointed out, with when they start in the take.

**How accurate it is:** so far, it's been tried on one song: 16 takes by an amateur, and two
professional recordings.
- A professional acoustic cover measured 12¢ from the original (10/10). Still, 3 of its 22 lines read
  60¢ or more off: about 1 line in 7 gets misread even when it's sung right, so a line pointed out is
  worth a listen, not proof.
- The instrument's key was read right on all 16 takes, including two played without the original's capo.
- A take a listener heard sitting a semitone under the guitar measured 123¢ under; one heard in key,
  11¢ over.
- Between 5 and 30 lines of a take get measured, depending on how many words Whisper catches and how many
  notes are held on them. With fewer than 3, the take is rated on the scale.
- A take is analyzed as it's filed (the mp3, not the recording it came from), so analyzing it again reads
  the same. A slightly different copy of the same performance still reads differently, since Whisper can
  catch other words: read from their mp3s rather than their original recordings, takes moved by 8¢
  typically and 38¢ at most (3 points). So a point or two between takes means little.

**Limits:**
- Harmonies, ad-libs and melodies changed on purpose count as off.
- Takes filed before Woodshed kept melodies (v0.4.0 and earlier) have to be analyzed again, voice
  separated and lyrics transcribed: the first `progress` or `play --rating` after setting a reference
  does it, once.
- The reference points for the melody (15¢ and 100¢) can't be changed yet; those in `shed init` are for
  the scale.

The song is recognized from the recording's lyrics, and you're asked to confirm it. You can also name it
(`shed reference harbor ~/Music/harbor-lights.mp3`): if the lyrics don't match your takes of it, you're
told which song they sound like before anything is saved. `shed reference harbor` says what a song is rated
against, and `shed reference harbor --remove` goes back to the scale.

**Finding the original on YouTube.** If you've installed [yt-dlp](https://github.com/yt-dlp/yt-dlp)
(`brew install yt-dlp`), `shed reference harbor` offers to look for the original when the song has none,
and `--search` looks even when it has one. It searches for the artist (when Genius recognized the song)
and the title, lists what it finds, and downloads only the video you pick: pick the studio version, since a
live or acoustic one has another melody. You can also give it a link (`shed reference harbor
https://www.youtube.com/watch?v=…`). The download is deleted as soon as its melody has been followed.
yt-dlp isn't bundled with Woodshed because YouTube keeps changing, so it needs updating every few weeks
(`brew upgrade yt-dlp`). Note that YouTube's terms don't allow downloading outside its own apps: whether to
use this is up to you.

## The library is just folders

```
~/Music/Woodshed/
  Harbor Lights/
    2026-06-02_21-14.mp3
    2026-09-21_18-30.mp3
    .reference.json   (if you gave it one: the melody it's rated against)
  Unsorted/
```

Each take carries its own transcript, rating and melody in its tags, so you can reorganize in Finder. Move
a misfiled take into the right folder, or rename a folder to rename the song (its reference goes with
it), and Woodshed follows. Raw recordings wait in the hidden `.incoming/` folder until they're filed, so a
crash never loses a take: `shed songs` tells you when some are waiting, and `shed add` files them
(then removes the raw copies).

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
