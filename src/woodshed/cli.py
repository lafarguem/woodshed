"""shed: record a cover, and Woodshed files it under the song's name."""

import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from mutagen.mp3 import MP3
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from woodshed import audio, config, genius, library as lib, microphones, models, rating, widgets
from woodshed.library import Library, Take

# Recognition runs offline; `shed init` downloads the models it needs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = typer.Typer(help="Record your covers; Woodshed recognizes the song and files each take by date.",
                  no_args_is_help=True, add_completion=False)
console = Console()

LISTEN_SECONDS = 150  # how much of each take Whisper listens to
MIN_TAKE_SECONDS = 20
GENIUS_CLIENTS_URL = "https://genius.com/api-clients"

LibraryOpt = Annotated[Path | None, typer.Option("--library", "-l", envvar="WOODSHED_DIR", show_default=False,
                                                 help="Folder holding your songs. [default: set by `shed init`]")]
LanguageOpt = Annotated[str | None, typer.Option(envvar="WOODSHED_LANGUAGE",
                                                 help="Language you sing in (en, fr…). Detected if omitted.")]
IsolateOpt = Annotated[bool, typer.Option(help="Separate your voice from the instrument: needed to rate the take, "
                                               "and far better lyrics when the instrument is loud.")]


@dataclass
class Choice:
    song: str | None  # None means Unsorted
    genius_id: int | None = None
    artist: str | None = None


@app.command()
def init():
    """Choose where recordings go, your microphone, your Genius token, and how strict ratings are."""
    os.environ["HF_HUB_OFFLINE"] = "0"  # this is where the models get downloaded
    try:
        settings = config.load()
    except config.Unreadable:  # start over, rather than leave no way to fix it but a text editor
        console.print(f"[yellow]Can't read {escape(_tilde(config.PATH))}, so starting from the defaults.[/yellow]\n")
        settings = config.Config()
    console.print("[bold]Let's set up Woodshed.[/bold] Press Enter to keep the value in brackets.\n")

    console.print("[bold]1. Where should your recordings go?[/bold]")
    folder = Path(Prompt.ask("Folder", default=settings.library)).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    settings.library = _tilde(folder)

    console.print("\n[bold]2. Which microphone do you record with?[/bold]")
    settings.device = _pick_device(settings.device)

    console.print("\n[bold]3. Genius access token[/bold] (optional)")
    console.print(f"It lets Woodshed recognize songs you haven't recorded yet. Create a free API client at "
                  f"{GENIUS_CLIENTS_URL}, then click “Generate Access Token”.")
    settings.genius_token = _ask_token(settings.genius_token)

    console.print("\n[bold]4. How strict should ratings be?[/bold]")
    console.print(f"  Pitch:  10/10 at {settings.pitch_best_cents:g}¢ off or closer, 0/10 at {settings.pitch_worst_cents:g}¢ "
                  "(100¢ = a semitone; notes picked at random average 25¢)")
    console.print(f"  Timing: 10/10 at ±{settings.timing_best_percent:g}% tempo wobble or steadier, "
                  f"0/10 at ±{settings.timing_worst_percent:g}%")
    console.print("[dim]Changing these re-scores all your takes instantly.[/dim]")
    if _interactive():
        if widgets.pick(console, ["Keep them", "Adjust them"]) == 1:
            _slide_references(settings)
    elif Confirm.ask("Change them?", default=False):
        settings.pitch_best_cents, settings.pitch_worst_cents = _ask_range(
            "Pitch, in cents off the true note", settings.pitch_best_cents, settings.pitch_worst_cents, limit=50)
        settings.timing_best_percent, settings.timing_worst_percent = _ask_range(
            "Timing, in ± % tempo wobble", settings.timing_best_percent, settings.timing_worst_percent, limit=15)

    config.save(settings)
    console.print(f"\n[green]✓[/green] Saved to {escape(_tilde(config.PATH))}")

    if missing := models.missing():
        console.print(f"Downloading {', '.join(missing)} (about 1.9 GB, only once)…")
        models.download()
    console.print("[green]✓[/green] Ready. Run [bold]shed rec[/bold] and play.")


@app.command()
def rec(
    library: LibraryOpt = None,
    language: LanguageOpt = None,
    auto_stop: Annotated[float | None, typer.Option(help="Stop after this many seconds of silence.")] = None,
    device: Annotated[str | None, typer.Option(envvar="WOODSHED_DEVICE", show_default=False,
                                               help="Input device number or name (see `shed devices`).")] = None,
    channels: Annotated[int, typer.Option(help="Input channels to record (2 for a stereo interface).")] = 1,
    isolate: IsolateOpt = True,
):
    """Record a take, recognize the song, and file it."""
    import sounddevice as sd

    from woodshed.recorder import record

    settings = _settings()
    songs = Library(library or Path(settings.library))
    device = device or settings.device
    started = datetime.now()
    raw = songs.incoming / f"{started:%Y-%m-%d_%H-%M-%S}.wav"
    try:
        record(raw, console, int(device) if device and device.isdigit() else device, channels, auto_stop)
    except (ValueError, sd.PortAudioError) as e:  # no such device, or it can't record that way (e.g. --channels)
        console.print(f"[red]Can't record from “{escape(device or 'your default microphone')}”: {escape(str(e))}."
                      "[/red] Plug it in, or pick another microphone with `shed init` or --device.")
        raise typer.Exit(1)
    try:
        _file_take(raw, started, songs, language, isolate, settings.genius_token, from_mic=True)
    except BaseException:
        console.print(f"[yellow]Your recording is safe in {escape(str(raw))}; run `shed add` on it to try again."
                      "[/yellow]")
        raise
    raw.unlink()


@app.command()
def add(
    files: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="Audio files (any format).")],
    library: LibraryOpt = None,
    language: LanguageOpt = None,
    isolate: IsolateOpt = True,
):
    """File recordings you already have, e.g. voice memos. The originals are left untouched."""
    settings = _settings()
    songs = Library(library or Path(settings.library))
    waiting = {p.resolve() for p in songs.waiting()}  # Woodshed's own raw copies, from a `shed rec` that failed
    unreadable = 0
    for path in files:
        console.rule(escape(path.name))
        try:
            saved = _file_take(path, audio.recorded_at(path), songs, language, isolate, settings.genius_token,
                               from_mic=False)
        except audio.AudioError as e:  # not audio, or damaged: say so, and go on with the others
            reason = str(e).splitlines()[-1] if str(e) else "ffmpeg couldn't decode it"
            console.print(f"[red]Can't read {escape(path.name)}: {escape(reason)}[/red]")
            unreadable += 1
            continue
        if saved and path.resolve() in waiting:  # filed at last: the raw copy has done its job
            path.unlink()
    if unreadable:
        raise typer.Exit(1)


@app.command(name="songs")
def list_songs(
    name: Annotated[str | None, typer.Argument(help="Show every take of this song.")] = None,
    library: LibraryOpt = None,
):
    """List your songs, or the takes of one song."""
    songs = _existing_library(library, config.load())
    catalog = songs.songs()
    if name:
        song = _find_song(songs, name)
        table = Table("#", "Recorded", "Length", title=escape(song), title_justify="left")
        for i, path in enumerate(catalog[song], 1):
            table.add_row(str(i), _when(path), _length(path))
        console.print(table)
        console.print(f"[dim]{escape(str(songs.root / song))}[/dim]")
        return

    table = Table("Song", "Takes", "First", "Latest")
    for song, takes in catalog.items():
        table.add_row(escape(song), str(len(takes)), *((_when(takes[0]), _when(takes[-1])) if takes else ("", "")))
    console.print(table)
    if unsorted := songs.unsorted():
        console.print(f"{len(unsorted)} take(s) waiting in [bold]{escape(str(songs.root / lib.UNSORTED))}[/bold]; "
                      "move them into a song folder to file them.")
    if waiting := songs.waiting():
        folder = shlex.quote(str(waiting[0].parent))
        console.print(f"[yellow]{len(waiting)} recording(s) couldn't be filed yet. File them with:[/yellow] "
                      f"shed add {escape(folder)}/*.wav")


@app.command()
def play(
    name: Annotated[str, typer.Argument(help="The song; part of its name is enough.")],
    metric: Annotated[str | None, typer.Argument(help="With --rating: pitch or timing instead of the overall rating.",
                                                 show_default=False)] = None,
    by_rating: Annotated[bool, typer.Option("--rating", help="Play your worst take, then your best.")] = False,
    library: LibraryOpt = None,
):
    """Play a song's first take, then its latest, to hear how far you've come."""
    from woodshed import player

    if metric is not None and metric not in rating.METRICS:
        console.print(f"“{escape(metric)}” isn't a rating; use {' or '.join(rating.METRICS)}. "
                      'Quote song names with spaces: shed play "harbor lights"')
        raise typer.Exit(1)
    settings = config.load()
    songs = _existing_library(library, settings)
    song = _find_song(songs, name)
    takes = songs.takes_of(song)
    if not takes:
        console.print(f"“{escape(song)}” has no takes yet.")
        raise typer.Exit(1)

    if by_rating or metric:
        references = settings.references()
        key = metric or "overall"
        rated = [(rating.scores(t.metrics, references)[key], number, t) for number, t in enumerate(_rate(songs, takes), 1)
                 if t.metrics and rating.scores(t.metrics, references)[key] is not None]
        if not rated:
            console.print(f"None of the takes of “{escape(song)}” could be rated on {key}.")
            raise typer.Exit(1)
        worst, best = min(rated, key=lambda r: r[:2]), max(rated, key=lambda r: r[:2])
        picks = [("worst", worst), ("best", best)] if len(rated) > 1 else [("only rated", worst)]
        picks = [(f"{which} take by {'rating' if key == 'overall' else key}", number, take,
                  f", {_score_text(take.metrics, key, references)}") for which, (_, number, take) in picks]
    else:
        if len(takes) == 1:
            console.print("[dim]Only one take so far.[/dim]")
        picks = [("first take" if n == 1 else "latest take", n, takes[n - 1], "")
                 for n in dict.fromkeys([1, len(takes)])]

    for label, number, take, detail in picks:
        console.print(f"[green]▶[/green] [bold]{escape(song)}[/bold], {label} ({number} of {len(takes)}), "
                      f"recorded {_when(take.path)}{detail}")
        if not player.play(take.path, console):
            break


@app.command()
def progress(
    name: Annotated[str, typer.Argument(help="The song; part of its name is enough.")],
    library: LibraryOpt = None,
):
    """Rate every take of a song, from first to latest."""
    settings = config.load()
    references = settings.references()
    songs = _existing_library(library, settings)
    song = _find_song(songs, name)
    takes = _rate(songs, songs.takes_of(song))
    if not takes:
        console.print(f"“{escape(song)}” has no takes yet.")
        raise typer.Exit(1)

    table = Table("#", "Recorded", "Rating", "Pitch", "Timing", title=escape(song), title_justify="left")
    for number, take in enumerate(takes, 1):
        if take.metrics is None:
            table.add_row(str(number), _when(take.path), "–", "–", "–")
            continue
        s = rating.scores(take.metrics, references)
        pitch = "–" if s["pitch"] is None else f"{s['pitch']:.1f}  [dim]{take.metrics.pitch_cents:.0f}¢ off[/dim]"
        timing = "–" if s["timing"] is None else f"{s['timing']:.1f}  [dim]±{100 * take.metrics.tempo_spread:.1f}%[/dim]"
        overall = "–" if s["overall"] is None else f"[bold]{s['overall']:.1f}[/bold]"
        table.add_row(str(number), _when(take.path), overall, pitch, timing)
    console.print(table)

    overall = [(n, rating.scores(t.metrics, references)["overall"]) for n, t in enumerate(takes, 1) if t.metrics]
    overall = [(n, score) for n, score in overall if score is not None]
    if len(overall) > 1:
        best_number, best = max(overall, key=lambda o: (o[1], o[0]))
        console.print(f"Rating {_sparkline([score for _, score in overall])} {overall[0][1]:.1f} → "
                      f"{overall[-1][1]:.1f} since your first take; best {best:.1f} (take {best_number}).")
    console.print("[dim]Pitch: how close your held notes are to true notes. Timing: how steady your tempo is. "
                  "Scores are out of 10.[/dim]")


@app.command()
def devices():
    """List the microphones and audio interfaces you can record from."""
    import sounddevice as sd

    chosen = os.environ.get("WOODSHED_DEVICE") or config.load().device  # what `shed rec` records from
    default = sd.default.device[0]
    connections = microphones.connections()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            in_use = chosen in (str(i), dev["name"]) if chosen else i == default
            marker = "[green]*[/green]" if in_use else " "
            kind = microphones.describe(connections.get(dev["name"], ""))
            console.print(f"{marker} {i}: {escape(dev['name'])} ({dev['max_input_channels']} ch)" + (f"  [dim]{kind}[/dim]" if kind else ""))


def _existing_library(library: Path | None, settings: config.Config) -> Library:
    """The library, for commands that only read it: a mistyped --library mustn't create a folder."""
    songs = Library(library or Path(settings.library))
    if not songs.root.is_dir():
        console.print(f"There's no library at {escape(str(songs.root))}. Check --library or WOODSHED_DIR, "
                      "or run `shed init`.")
        raise typer.Exit(1)
    return songs


def _find_song(songs: Library, name: str) -> str:
    """The song called `name`, or the only one whose name contains it."""
    partial = [song for song in songs.songs() if name.casefold() in song.casefold()]
    song = songs.find_song(name) or (partial[0] if len(partial) == 1 else None)
    if song is None:
        hint = f"; did you mean {' or '.join(partial)}?" if partial else f" in {songs.root}"
        console.print(f"No song called “{escape(name)}”{escape(hint)}")
        raise typer.Exit(1)
    return song


def _rate(songs: Library, takes: list[Take]) -> list[Take]:
    """The takes, after rating any that haven't been rated yet (saved, so only once)."""
    from woodshed import isolate

    unrated = [t for t in takes if t.metrics is None]
    if unrated and (missing := models.missing()):
        console.print(f"[yellow]{_not_downloaded(missing)} Run `shed init` to download.[/yellow]")
        raise typer.Exit(1)
    for i, take in enumerate(unrated, 1):
        with console.status(f"Rating take {i} of {len(unrated)} (only needed once)…"):
            take.metrics = rating.analyze(isolate.separate(take.path))
        songs.save_metrics(take.path, take.metrics)
    return takes


def _score_text(metrics: rating.Metrics, key: str, references: rating.References) -> str:
    score = rating.scores(metrics, references)[key]
    if key == "pitch":
        return f"pitch {score:.1f}/10 ({metrics.pitch_cents:.0f}¢ off)"
    if key == "timing":
        return f"timing {score:.1f}/10 (tempo ±{100 * metrics.tempo_spread:.1f}%)"
    return f"rated {score:.1f}/10"


def _rating_line(metrics: rating.Metrics, earlier: list[Take], references: rating.References) -> str:
    s = rating.scores(metrics, references)
    if s["overall"] is None:
        return "[dim]Not enough singing or strumming to rate this take.[/dim]"
    parts = [f"Rated [bold]{s['overall']:.1f}/10[/bold]"]
    parts += [_score_text(metrics, key, references) for key in rating.METRICS if s[key] is not None]
    if s["pitch"] is None:
        parts.append("pitch: not enough singing to judge")
    line = " · ".join(parts)
    best = max((score for t in earlier if (score := rating.scores(t.metrics, references)["overall"]) is not None),
               default=None)
    if best is not None:
        line += " · [green]your best take yet![/green]" if s["overall"] > best else f" · your best: {best:.1f}"
    return line


def _sparkline(values: list[float]) -> str:
    return "".join("▁▂▃▄▅▆▇█"[min(7, int(v / 10 * 8))] for v in values)


def _not_downloaded(names: list[str]) -> str:
    return f"{', '.join(names)} {'is' if len(names) == 1 else 'are'}n't downloaded yet."


def _settings() -> config.Config:
    """Saved settings, after checking setup is done (so a take is never recorded for nothing)."""
    if not config.exists():
        console.print("[yellow]Run `shed init` first to choose your folder and microphone.[/yellow]")
        raise typer.Exit(1)
    if missing := models.missing():
        console.print(f"[yellow]{_not_downloaded(missing)} Run `shed init` to download.[/yellow]")
        raise typer.Exit(1)
    settings = config.load()
    settings.genius_token = os.environ.get("GENIUS_ACCESS_TOKEN") or settings.genius_token
    return settings


def _pick_device(current: str | None) -> str | None:
    import sounddevice as sd

    names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    if not names:
        console.print("[yellow]No microphone found; plug one in and run `shed init` again.[/yellow]")
        return current
    connections = microphones.connections()
    system_default = sd.default.device[0]
    suggested = microphones.suggest(names, connections, current,
                                    sd.query_devices(system_default)["name"] if system_default >= 0 else None)
    notes = [microphones.describe(connections.get(name, "")) for name in names]
    if _interactive():
        return names[widgets.pick(console, names, names.index(suggested), notes)]
    for i, (name, note) in enumerate(zip(names, notes), 1):
        console.print(f"  {i}. {escape(name)}" + (f"  [dim]{note}[/dim]" if note else ""))
    number = IntPrompt.ask("Number", choices=[str(i) for i in range(1, len(names) + 1)],
                           default=names.index(suggested) + 1, show_choices=False)
    return names[number - 1]


def _ask_token(current: str | None) -> str | None:
    while True:
        token = Prompt.ask("Token" + (" (Enter keeps the saved one)" if current else " (Enter to skip)"),
                           password=True, default=current or "", show_default=False).strip()
        if not token:
            return None
        try:
            with console.status("Checking the token with Genius…"):
                genius.search("hello", token)
        except genius.InvalidToken:
            console.print("[yellow]Genius rejected that token. Paste it again, or press Enter to skip.[/yellow]")
            current = None
            continue
        except genius.GeniusError as e:
            console.print(f"[yellow]{escape(str(e))}; saving the token anyway.[/yellow]")
            return token
        console.print("[green]✓[/green] Token works.")
        return token


def _interactive() -> bool:
    """A real terminal, where arrow-key prompts work (not piped input, nor tests)."""
    return sys.stdin.isatty() and console.is_terminal


def _slide_references(settings: config.Config) -> None:
    def examples(values: list[float], fmt: str, references: rating.References, key: str) -> str:
        measure = (lambda v: rating.Metrics(v, None)) if key == "pitch" else (lambda v: rating.Metrics(None, v / 100))
        return "  ·  ".join(f"{fmt.format(v)} → {rating.scores(measure(v), references)[key]:.1f}" for v in values)

    def pitch(best: float, worst: float) -> str:
        return "A take " + examples([8, 12, 16, 20], "{:g}¢ off", rating.References(pitch_cents=(best, worst)), "pitch")

    def timing(best: float, worst: float) -> str:
        return "A take " + examples([1.5, 2.5, 4, 5], "±{:g}%", rating.References(tempo_spread=(best / 100, worst / 100)),
                                    "timing")

    cents, percent = "{:g}¢".format, "±{:g}%".format
    s = settings
    s.pitch_best_cents = widgets.slide(console, "  Pitch scores 10/10 within", s.pitch_best_cents, 0,
                                       s.pitch_worst_cents - 1, 1, cents, lambda v: pitch(v, s.pitch_worst_cents))
    s.pitch_worst_cents = widgets.slide(console, "  Pitch scores 0/10 from", s.pitch_worst_cents, s.pitch_best_cents + 1,
                                        50, 1, cents, lambda v: pitch(s.pitch_best_cents, v))
    s.timing_best_percent = widgets.slide(console, "  Timing scores 10/10 within", s.timing_best_percent, 0,
                                          s.timing_worst_percent - 0.5, 0.5, percent,
                                          lambda v: timing(v, s.timing_worst_percent))
    s.timing_worst_percent = widgets.slide(console, "  Timing scores 0/10 from", s.timing_worst_percent,
                                           s.timing_best_percent + 0.5, 15, 0.5, percent,
                                           lambda v: timing(s.timing_best_percent, v))


def _ask_range(label: str, best: float, worst: float, limit: float | None = None) -> tuple[float, float]:
    """Ask for the value scoring 10 and the one scoring 0 until they make sense."""
    console.print(label)
    while True:
        new_best = FloatPrompt.ask(f"  Scores 10 at or below ({best:g})", default=float(best), show_default=False)
        new_worst = FloatPrompt.ask(f"  Scores 0 at or above ({worst:g})", default=float(worst), show_default=False)
        if 0 <= new_best < new_worst and (limit is None or new_worst <= limit):
            return new_best, new_worst
        console.print("[yellow]The 10/10 value must be below the 0/10 value" +
                      (f", and at most {limit:g}." if limit else ".") + "[/yellow]")


def _tilde(path: Path) -> str:
    path = path.expanduser().resolve()
    return f"~/{path.relative_to(Path.home())}" if path.is_relative_to(Path.home()) else str(path)


def _file_take(src: Path, recorded: datetime, songs: Library, language: str | None,
               isolate_voice: bool, genius_token: str | None, from_mic: bool) -> Path | None:
    """File one take. Returns where it was saved, or None if it wasn't kept (silent, or too short)."""
    from woodshed import isolate, transcribe

    with console.status("Listening to the take…"):
        samples = audio.load(src)
        start, end = audio.playing_bounds(samples)
    if from_mic and (samples.size == 0 or abs(samples).max() < 1e-4):
        console.print("[yellow]The recording is completely silent, so it wasn't kept. Allow your terminal app to "
                      "use the microphone in System Settings → Privacy & Security → Microphone.[/yellow]")
        return None
    if end <= start:
        start, end = 0.0, len(samples) / audio.SAMPLE_RATE
    if from_mic and end - start < MIN_TAKE_SECONDS:
        if not Confirm.ask(f"Only {end - start:.0f}s of playing. Keep this take?", default=False):
            console.print("Discarded.")
            return None

    seconds = min(LISTEN_SECONDS, end - start)
    metrics = None
    if isolate_voice:
        with console.status("Separating your voice from the instrument…"):
            stems = isolate.separate(src, start, end - start)
        voice = stems.vocals_16k(seconds)
    else:
        voice = samples[int(start * audio.SAMPLE_RATE): int((start + seconds) * audio.SAMPLE_RATE)]
    with console.status("Transcribing the lyrics…"):
        lines = transcribe.transcribe(voice, language)
    if isolate_voice:
        with console.status("Rating the take…"):
            metrics = rating.analyze(stems)

    choice = _identify(lines, songs, genius_token)
    earlier = [t for t in songs.takes_of(choice.song) if t.metrics] if choice.song else []
    dest = songs.add_take(src, choice.song, start, end, recorded, lines, choice.genius_id, choice.artist, metrics)
    number = ""
    if choice.song:  # counted by date, as `play` and `progress` do: an old voice memo can be take 1 of 4
        takes = sorted(dest.parent.glob("*.mp3"))
        number = f" (take {takes.index(dest) + 1} of {len(takes)})"
    console.print(f"[green]✓[/green] Saved [bold]{escape(str(dest.relative_to(songs.root)))}[/bold]{number}")
    if metrics:
        console.print(_rating_line(metrics, earlier, config.load().references()))
    return dest


def _identify(lines: list[str], songs: Library, genius_token: str | None) -> Choice:
    if lines:
        heard = " / ".join(lines)
        console.print(f"[dim]Heard: “{escape(heard[:90])}{'…' if len(heard) > 90 else ''}”[/dim]")
    else:
        console.print("[dim]Didn't hear any lyrics.[/dim]")

    ranked = songs.match(lines)
    if lib.is_confident(lines, ranked):
        console.print(f"Recognized [bold]{escape(ranked[0][0])}[/bold] from your earlier takes.")
        return Choice(ranked[0][0])

    candidates = _search_genius(lines, genius_token)
    if genius.is_confident(candidates):
        top = candidates[0]
        console.print(f"Recognized [bold]{escape(top.title)}[/bold] by {escape(top.artist)} on Genius.")
        return _choice_for(top, songs)
    return _ask(ranked, candidates[0] if candidates else None, songs)


def _search_genius(lines: list[str], token: str | None) -> list[genius.Candidate]:
    if not token:
        console.print("[dim]Add a Genius token with `shed init` to recognize songs you haven't recorded before.[/dim]")
        return []
    if not genius.snippets(lines):
        return []
    try:
        with console.status("Looking the lyrics up on Genius…"):
            return genius.identify(lines, token)
    except genius.GeniusError as e:
        console.print(f"[yellow]{escape(str(e))}[/yellow]")
        return []


def _choice_for(candidate: genius.Candidate, songs: Library) -> Choice:
    """File under your existing folder for this song if there is one, even if you named it differently."""
    song = (songs.song_for_genius_id(candidate.genius_id) or songs.find_song(candidate.title)
            or lib.folder_name(candidate.title))
    return Choice(song, candidate.genius_id, candidate.artist)


def _ask(ranked: list[tuple[str, float]], suggestion: genius.Candidate | None, songs: Library) -> Choice:
    options: list[tuple[Choice, str]] = []
    if suggestion:
        options.append((_choice_for(suggestion, songs),
                        f"{escape(suggestion.title)} — {escape(suggestion.artist)} [dim](Genius)[/dim]"))
    for song, score in ranked[:3]:
        if score > 0.03 and all(choice.song != song for choice, _ in options):
            options.append((Choice(song), f"{escape(song)} [dim](your songs)[/dim]"))

    console.print("[bold]Which song is this?[/bold]")
    for i, (_, label) in enumerate(options, 1):
        console.print(f"  {i}. {label}")
    answer = _input("Number, song name (Tab completes), or Enter for Unsorted: ", list(songs.songs()))
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1][0]
    name = songs.find_song(answer) or lib.folder_name(answer)
    return Choice(name or None)


def _input(prompt: str, completions: list[str]) -> str:
    """Read a line, with Tab completing your song names."""
    try:
        import readline
    except ImportError:
        readline = None
    if readline:
        def complete(text: str, state: int) -> str | None:
            matches = [c for c in completions if c.casefold().startswith(text.casefold())]
            return matches[state] if state < len(matches) else None

        readline.set_completer_delims("")
        readline.set_completer(complete)
        readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")
    try:
        return input(prompt).strip()
    except EOFError:
        return ""
    finally:
        if readline:
            readline.set_completer(None)


def _when(path: Path) -> str:
    stamp = lib.take_time(path)
    return f"{stamp:%Y-%m-%d %H:%M}" if stamp else escape(path.stem)  # a file you put there yourself


def _length(path: Path) -> str:
    try:
        minutes, seconds = divmod(round(MP3(path).info.length), 60)
    except Exception:
        return "?"
    return f"{minutes}:{seconds:02d}"
