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
Rated 7.5/10 · pitch 7.3/10 · timing 7.8/10 (tempo ±2.1%) · your best: 7.9
Pitch, against the song's scale:
  Typically 21¢ off, either way (10/10 at 12¢, 0/10 at 38¢)
  Typically 8¢ under, against your instrument's tuning
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
   Genius. If the same song comes up first more than once, it's filed under that title. Covers of a song
   share its title, and Genius often lists one first: when Genius links what it found to an original of the
   same title (a cover, or a live version), the take is filed as the original artist's song.
5. **You:** otherwise you pick one of the best guesses with the arrow keys, or another song, whose
   name you type (Tab completes your song names), or leave the take in `Unsorted/`.

Everything runs on your Mac. Only step 4 goes online, and it sends only a few lyric
lines, then the Genius id of the song they found, to ask whether it's a cover. The only other times Woodshed
goes online are when you ask it to find a song's original on YouTube
(see [Rating against the original melody](#rating-against-the-original-melody)), or to get a tab from Songsterr.

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
   Picking another folder offers to move your recordings there, unless it already has a song of the same name.
2. **Which microphone,** from a list that shows how each is connected. It suggests an audio
   interface if you have one, else your Mac's built-in mic. Bluetooth earbuds record at
   phone-call quality.
3. **A Genius access token** (optional) to recognize new songs. Create a free API client
   at https://genius.com/api-clients and click "Generate Access Token". The token is
   checked before it's saved.
4. **How takes are rated** (optional): sliders for the reference points below (for pitch against the
   song's scale, and against a reference melody; for timing), and for how much pitch and timing each count
   in the rating, with a live preview of what example takes would score.
   Changing them re-scores every take instantly.

Answers go to `~/.config/woodshed/config.toml`, which only you can read. Run
`shed init` again anytime to change them.

## Usage

```sh
shed rec                    # record until you press a key
shed rec --auto-stop 8      # …or until 8 s of silence
shed rec --m4a              # save this take as Apple Lossless (or --mp3), whatever `shed init` chose
shed rec --later            # only record, to record the next take right away…
shed add                    # …then file every take recorded that way
shed serve                  # record from your phone: open the page it shows (see below)
shed add ~/Downloads/*.m4a  # file recordings you already have (voice memos: the date is kept)
shed add ~/Downloads/Memos  # …or a whole folder: it picks out the songs (see below)
shed songs                  # every song, with take counts and dates
shed songs harbor           # the takes of one song
shed play harbor            # its first take, then its latest (any key skips)
shed progress harbor        # every take's rating, first to latest
shed progress harbor -t 3   # take 3 in detail: its rating, and its best and worst lines against the melody
shed progress harbor --lines  # each line of the song across your latest takes, against its melody
shed play harbor --rating   # your worst take, then your best (or --pitch, --timing: on that alone)
shed play harbor -t 3       # take 3 (--take 3)
shed reference ~/Music/harbor-lights.mp3  # rate a song's pitch against the original's melody
shed reference harbor       # …or find the original on YouTube (needs yt-dlp, see below)
shed reference harbor --songsterr  # sing along to the notes of Songsterr's tab of it (see Singing along)
shed reference harbor --tab ~/Downloads/harbor.gp  # …or of a Guitar Pro file
shed serve                  # …and sing along to the original on the page's Sing tab, your pitch drawn over its melody
shed tune                   # tune your guitar: pluck a string, see which way to turn it
shed tune drop d            # …in another tuning (see Tuning your guitar)
shed devices                # microphones you can record from, and how they're connected
shed init                   # change the folder, format, microphone, token or how takes are rated
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

### Recording from your phone

Run `shed serve` on the Mac, and open the page it shows on your phone, on the same Wi-Fi: scan its QR code,
or type the link. Record there: each take is sent to the Mac as soon as you stop, and filed there. The page
then shows how it went: the song, its rating, and the lines furthest from the melody and closest to it if the
song has a reference. When the song isn't clear, the page asks you which it is. Takes are filed one at a time, in the
order they arrive, so naming the first take of a new song lets the next ones be recognized.

- `shed serve --later` only receives the takes: they wait to be filed by `shed add`, as `shed rec --later`
  takes do.
- The page's Sing tab plays a song's reference recording in your headphones, and draws your pitch over its
  melody as you sing along (see Singing along).
- The page's Tune tab tunes your guitar, as `shed tune` does (see Tuning your guitar), from the phone's microphone.
- Ctrl+C stops it. A take being filed is finished first; one whose song you haven't said yet waits for
  `shed add`, as does any take that couldn't be filed.
- The page records music as it sounds: without the noise suppression, echo cancellation and automatic volume
  that phones apply to calls.
- Phone browsers only let a page use the microphone over HTTPS, so the page has a certificate of Woodshed's
  own. Your phone warns that it doesn't know it: go on anyway (Advanced, then Proceed). If macOS asks
  whether to accept incoming connections, allow them.
- The link carries a secret, so nobody else on the network can send you takes. It stays the same, so you can
  bookmark it (unless your Mac's address on the network changes).
- The screen stays on while you record, since a phone that goes to sleep can stop the recording. A take that
  couldn't be sent stays on the page: send it again, or save it on the phone.

### Adding a backlog

Give `shed add` a folder of old recordings, and it files the songs in it:

1. **A quick check for singing** comes first, about a second per minute of recording. Recordings
   without singing (talking, an instrument alone), and those shorter than 20 s, are skipped, before
   anything is transcribed or sent to Genius. Hidden files, like a recorder app's recently deleted
   recordings, are left out.
2. **The songs are recognized as usual,** oldest first, so each take filed helps recognize the later
   ones.
3. **The ones it can't name are kept for the end,** one question each. There, you can also skip a
   recording: it isn't filed, nor asked about again.

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

`shed progress` shows two figures for pitch: how far off your notes typically are, either way, which is
what's rated, and where they typically sit, under or over. `shed progress --take` spells both out for one
take, and so do `shed rec` and `shed add` once a take is filed. A take whose notes stray both ways and one
sung flat throughout can be just as far off, but only the second leans. Against the scale, takes rated
before Woodshed measured the second only have the first; against a reference melody, both are always there.

The reference points are judgment calls, set so that professional recordings score about 9. Change
them with `shed init` (or the `pitch_*`, `melody_*` and `timing_*` keys in `config.toml`), along with pitch's
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
Rated 6.1/10 · pitch 5.0/10 · timing 7.8/10 (tempo ±2.1%) · your best: 6.4
Pitch, against the melody (22 lines compared):
  Typically 58¢ off, either way (10/10 at 15¢, 0/10 at 100¢)
  Typically 40¢ under: 17 lines sit under the melody, 5 over
Furthest from the melody:
  1:12  “and every bell was ringing out your name”  130¢ under
  0:31  “we counted ships until the evening fall”  85¢ over
Closest to the melody:
  0:12  “the lanterns swing above the harbor wall”  4¢ over
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
   15¢, 0/10 at 100¢ (a semitone), unless you choose otherwise in `shed init`. Since a line's notes are
   pooled first, a line sung flat throughout costs more than one whose notes stray either way.
5. When most of your lines sit on the same side of the melody, you're told. Up to 3 lines 60¢ or more off
   are pointed out, with when they start in the take, and up to 3 lines within 20¢ of it.

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

The song is recognized from the recording's lyrics, and you're asked to confirm it. You can also name it
(`shed reference harbor ~/Music/harbor-lights.mp3`): if the lyrics don't match your takes of it, you're
told which song they sound like before anything is saved. `shed reference harbor` says what a song is rated
against, and `shed reference harbor --remove` goes back to the scale.

**Finding the original on YouTube.** If you've installed [yt-dlp](https://github.com/yt-dlp/yt-dlp)
(`brew install yt-dlp`), `shed reference harbor` offers to look for the original when the song has none,
and `--search` looks even when it has one. It searches for the artist (when Genius recognized the song)
and the title, lists what it finds (each video's channel, length and the start of its description), and
downloads only the one you pick with the arrow keys: pick the studio version, since a live or acoustic one
has another melody. If the original isn't there, because the song has no artist or the wrong one, pick
"Search again with another artist…" and say who sings the original: it's kept for the song once you pick
one of the videos found. You can also give it a link (`shed reference harbor
https://www.youtube.com/watch?v=…`). The download is deleted as soon as its melody has been followed. What's kept is the melody, and the
recording as two mp3s in the song's folder, to sing along to (see Singing along).
yt-dlp isn't bundled with Woodshed because YouTube keeps changing, so it needs updating every few weeks
(`brew upgrade yt-dlp`). Note that YouTube's terms don't allow downloading outside its own apps: whether to
use this is up to you.

### Your lines, take after take

With a reference, `shed progress` also follows each line of the song through your latest takes (the last 8
compared with the melody): the lines you're often off, and those you sing closest to it.

```
$ shed progress harbor
…
Line by line, your last 8 takes compared with the melody:
Often off it:
  “and every bell was ringing out your name”  typically 116¢ under · under in 7 of 8 takes
  “we counted ships until the evening fall”  typically 94¢ over · over in 6 of 7 takes
Closest to it:
  “the lanterns swing above the harbor wall”  typically 24¢ off · within 20¢ in 2 of 7 takes
```

`shed progress harbor --lines` shows every line: how far from the melody it's typically sung, either way, and
where, and how each take sang it. A line is said to be often off when it's typically 50¢ or more from the
melody, and on that side in 7 of 10 of the takes that sang it; lines are only judged once 3 takes have sung them.

Each line is compared with the melody moved to where the rest of its take sits. Whole takes move a lot from one
to the next: a take sung a semitone under your guitar is told so as it's filed, and it doesn't put all its lines
off here. On 16 takes of a song by an amateur, which sat anywhere from 274¢ under the melody to 24¢ over:
- How far each line was from the melody itself didn't repeat between halves of the takes (a rank correlation of
  0.08). How far it was from the rest of its take did (0.80, and 0.60 either way).
- Split into halves of 8 (odd and even takes, or the first and last 8), the 3 lines pointed out in each half all
  sat on the same side of the melody in the other half, and 8 of those 12 were pointed out there too.
- A professional cover sang all its lines within 42¢ of the rest of it: none would be pointed out. It sang 3 of
  the lines the amateur was often off, each within 17¢ of the rest: they were the amateur's, not the reference's.
- The closest lines are less sure: 6 of the 12 were among the other half's closest, and 1 was even often off
  there. That amateur sang 18% of their lines within 20¢ of where the rest of the take sat; the professional
  cover, 68%.

### Tuning your guitar

`shed tune` is a guitar tuner. Pluck a string, one at a time: it shows which string of the tuning it takes it
for, and how far from its note you are, on a bar from 50¢ flat to 50¢ sharp. The string is green within 5¢,
yellow within 15¢, red further. Nothing is recorded.

```
$ shed tune drop d
Tuner  Drop D

  6 D2   5 A2   4 D3   3 G3   2 B3   1 E4

  ♭ ──────────●─────────┼──────────────────── ♯
  A2  tune up ↑ (25¢ flat)

  Listening to MacBook Pro Microphone  (m: another microphone)
←/→ another tuning · 1-6 only that string · 0 any string · q quits
```

| Tuning | Strings, lowest first |
|---|---|
| Standard | E2 A2 D3 G3 B3 E4 |
| Drop D | D2 A2 D3 G3 B3 E4 |
| Double drop D | D2 A2 D3 G3 B3 D4 |
| DADGAD | D2 A2 D3 G3 A3 D4 |
| Half step down | E♭2 A♭2 D♭3 G♭3 B♭3 E♭4 |
| Whole step down | D2 G2 C3 F3 A3 D4 |
| Drop C | C2 G2 C3 F3 A3 D4 |
| Open D | D2 A2 D3 F♯3 A3 D4 |
| Open E | E2 B2 E3 G♯3 B3 E4 |
| Open G | D2 G2 D3 G3 B3 D4 |
| Open A | E2 A2 E3 A3 C♯4 E4 |
| Open C | C2 G2 C3 G3 C4 E4 |

- Name the tuning to start in (part of its name is enough: `shed tune dadgad`, `shed tune half`), or go from one
  to the next with ←/→. Without a name, it starts in standard tuning.
- The string is the tuning's nearest to what it hears. A new string, or one you're taking to another tuning,
  starts nearer its neighbour: press the string's number (6 is the lowest) to keep to it, and 0 to go back.
- It listens to the microphone `shed init` chose (or `--device`), and says which. Press m to go to the next
  one, e.g. from an audio interface your guitar isn't plugged into to the Mac's own: it goes round them all,
  passing over any it can't listen to. The choice lasts until you quit.
- A reading stays a second after the string has died away, greyed.
- Strummed strings, talk and noise aren't read: it waits for one string ringing alone.
- Notes are counted from A at 440 Hz.
- Away from the Mac, tune from your phone: `shed serve`, then the page's Tune tab. Pick the tuning in its list,
  and tap a string to keep to it. When the phone has more than one microphone (a headset's, an audio
  interface's), a list under the tuner picks the one it listens to, and the page remembers it.

### Singing along

The Sing tab of `shed serve`'s page plays a song's reference recording, and draws what you sing over its singer's
melody as it scrolls by, as karaoke games do. Put headphones on, so the microphone hears you and your instrument,
not the recording. Nothing is recorded.

- The melody is drawn as a band in every octave: blue in the octave its singer sings it in, grey in the others.
  Your notes are drawn at the pitch you sang them, so you see which octave you're in. They're green on the
  melody's note, yellow a semitone off, red further, going by the octave of the melody nearest them: any octave of
  a note counts, and a fifth off is a fifth off in all of them. The picture, 20 semitones tall at most
  so that no more than two of the melody's octaves show, starts on the octave your voice usually
  sits in, in your takes of the song, and slides after your voice when it nears the top or the bottom; drag it up
  or down (or scroll over it) to look elsewhere. Under the picture, the line being sung, the next one, and how much of what you sang was on the
  melody's note.
- Each note is a row of the picture. What you sang stays behind you as blocks, each filling the row of the note you
  were on for as long as you were on it, with a thin line through them: your pitch itself. In the middle of its
  block, the note is in tune; towards its edge, up to half a semitone off. Under the picture, the note you're on
  and how many cents from it you are (from A at 440 Hz), on a small gauge: green within 15¢, yellow within 30¢,
  red further.
- The melody heard in a recording wanders: its singer slides into notes and bends them, and some are misheard.
  With a tab of the song that has its melody, the picture shows the notes the tab writes instead, as blocks (what
  the singer sings stays, as a thin faint line over them), and you're set against those:
  `shed reference harbor --songsterr` opens a page on your Mac that waits for a tab from Songsterr. Open the song's
  tab on songsterr.com and click the **To Woodshed** bookmark (the first time, drag it to your bookmarks bar from
  that page): Woodshed downloads the tab's tracks itself, from Songsterr's CDN. Songsterr turns away anything but a
  browser asking where a tab is, hence the bookmark, which reads it off Songsterr's own page. Or give a Guitar Pro
  file: `shed reference harbor --tab ~/Downloads/harbor-lights.gp` (a
  .gp, as Songsterr saves them; the older .gp5 and .gpx aren't read, nor is MIDI). Woodshed finds which of the tab's
  tracks is the voice, and where its notes fall in the recording, in which key and at what tempo, by laying each
  track over the melody it heard. On 4 songs, the voice's notes fell on the singer's 25 to 40% of the time, and
  they did so far more in one place than anywhere else, which no instrument did, though one followed the voice
  on a synth. (On Space Song, the tab writes the upper of the chorus's two stacked voices, and the lower is
  heard.) Of the tracks that fit, the one on the singer's notes for longest is the voice: a backing voice of a few
  notes may fit as often, but not as long. A backing voice that doubles the voice fits nearly as long (on This
  Fire, 31.4 s against 32.4 s): then the tracks' names say which leads ("Lead Vocals"), and Woodshed says which
  other track fit nearly as well. `--track backing` (part of a track's name) takes that one instead. On one of the
  songs, the singer's pitch through a written note was within half a semitone of it for 73% of the notes.
  `--no-tab` goes back to the melody heard. The tab goes with the reference: set another recording,
  and give the tab again. Repeats and alternate endings are followed; a Guitar Pro file with alternate endings or
  jumps (D.S., coda) isn't read.
- Two sliders set how loud the band and the singer are: turn the singer down once you know the melody, and both
  once you have the rhythm. The melody keeps scrolling, in time.
- Your notes are drawn about a fifth of a second after you sing them: the pitch tracker needs to hear a little of
  what follows a moment to be sure of it (read as it's sung, it got 29% of the sung moments right; 0.1 s later, 61%).
- It follows your voice over your instrument, which a tuner can't. On 90 s of each of 8 takes, voice and guitar in
  one microphone, it drew 63% of the sung moments plainly at the right note and 2% at a wrong one, and 12% of the
  guitar alone as a voice. A tuner's way of finding the pitch got 23% right, and took the guitar alone for a voice
  a third of the time.
- Where it's less sure the sound is your voice, the dots are faint: with them, 80% of the sung moments are at the
  right note, 8% at a wrong one, and 30% of the guitar alone is drawn. They don't count in the figures under the
  picture.
- The louder your voice against your instrument, the more of it is drawn: 97% of the sung moments with the voice
  10 dB over the instrument, 52% with it up to 10 dB under. Keep the phone near your mouth.
- Nothing is assumed of what you sing: the melody's note isn't used to find yours, so a note far off is drawn where
  it is.
- If your notes are drawn late or early against the melody (Bluetooth headphones are later than the browser
  knows, and turn the microphone to phone-call quality: wired ones are better), the arrows under the sliders move
  them, 20 ms at a time. The phone remembers it.
- It needs the recording, which `shed reference` keeps since v0.9, as two mp3s in the song's folder (its singer,
  and the rest: about 5 MB a song). Set the reference again for a song whose reference is older.
- The page works on the Mac too: open its link in a browser there.

## The library is just folders

```
~/Music/Woodshed/
  Harbor Lights/
    2026-06-02_21-14.mp3
    2026-09-21_18-30.mp3
    .reference.json   (if you gave it one: the melody it's rated against)
    .reference-band.mp3, .reference-voice.mp3   (its recording, to sing along to)
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
