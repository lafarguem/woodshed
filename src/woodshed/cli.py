"""shed: record a cover, and Woodshed files it under the song's name."""

import os
import queue
import re
import shlex
import sys
import tempfile
import threading
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import mutagen
import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from woodshed import audio, config, genius, library as lib, melody, microphones, models, phone, rating, widgets, youtube
from woodshed.library import Library, Take

# Recognition runs offline; `shed init` downloads the models it needs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = typer.Typer(help="Record your covers; Woodshed recognizes the song and files each take by date.",
                  no_args_is_help=True, add_completion=False)
console = Console()

MIN_TAKE_SECONDS = 20
MELODY_LIMIT = 200  # the most a reference melody's 0/10 point can be set to, in cents
MIN_REFERENCE_WORDS = 20  # fewer words heard in a reference recording, and there's no melody to follow
AUDIO_SUFFIXES = frozenset({".aac", ".aif", ".aifc", ".aiff", ".caf", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
GENIUS_CLIENTS_URL = "https://genius.com/api-clients"

LibraryOpt = Annotated[Path | None, typer.Option("--library", "-l", envvar="WOODSHED_DIR", show_default=False,
                                                 help="Folder holding your songs. [default: set by `shed init`]")]
LanguageOpt = Annotated[str | None, typer.Option(envvar="WOODSHED_LANGUAGE",
                                                 help="Language you sing in (en, fr…). Detected if omitted.")]
TakeOpt = Annotated[int | None, typer.Option("--take", "-t", min=1, show_default=False,
                                             help="Just this take, numbered as `shed progress` numbers them.")]
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
    settings.take_format = _pick_format(settings.take_format)

    console.print("\n[bold]2. Which microphone do you record with?[/bold]")
    settings.device = _pick_device(settings.device)

    console.print("\n[bold]3. Genius access token[/bold] (optional)")
    console.print(f"It lets Woodshed recognize songs you haven't recorded yet. Create a free API client at "
                  f"{GENIUS_CLIENTS_URL}, then click “Generate Access Token”.")
    settings.genius_token = _ask_token(settings.genius_token)

    console.print("\n[bold]4. How should takes be rated?[/bold]")
    console.print(f"  Pitch:  10/10 at {settings.pitch_best_cents:g}¢ off or closer, 0/10 at {settings.pitch_worst_cents:g}¢ "
                  "(100¢ = a semitone; notes picked at random land around 38¢)")
    console.print(f"          against a song's reference melody (shed reference): 10/10 at {settings.melody_best_cents:g}¢ "
                  f"off or closer, 0/10 at {settings.melody_worst_cents:g}¢")
    console.print(f"  Timing: 10/10 at ±{settings.timing_best_percent:g}% tempo wobble or steadier, "
                  f"0/10 at ±{settings.timing_worst_percent:g}%")
    console.print(f"  Rating: {_shares(settings.pitch_weight_percent)}")
    console.print("[dim]Changing these re-scores all your takes instantly.[/dim]")
    if _interactive():
        if widgets.pick(console, ["Keep them", "Adjust them"]) == 1:
            _slide_references(settings)
    elif Confirm.ask("Change them?", default=False):
        settings.pitch_best_cents, settings.pitch_worst_cents = _ask_range(
            "Pitch, in cents off the true note", settings.pitch_best_cents, settings.pitch_worst_cents, limit=50)
        settings.melody_best_cents, settings.melody_worst_cents = _ask_range(
            "Pitch against a reference melody, in cents off it", settings.melody_best_cents,
            settings.melody_worst_cents, limit=MELODY_LIMIT)
        settings.timing_best_percent, settings.timing_worst_percent = _ask_range(
            "Timing, in ± % tempo wobble", settings.timing_best_percent, settings.timing_worst_percent, limit=15)
        settings.pitch_weight_percent = _ask_share(settings.pitch_weight_percent)

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
    later: Annotated[bool, typer.Option("--later", help="Only record: the take waits to be filed by `shed add`, "
                                                        "so you can record the next one right away.")] = False,
    lossless: Annotated[bool | None, typer.Option("--m4a/--mp3", show_default=False,
                                                  help="Save this take as Apple Lossless (.m4a) or as MP3, instead "
                                                       "of as `shed init` chose.")] = None,
):
    """Record a take, recognize the song, and file it."""
    import sounddevice as sd

    from woodshed.recorder import record

    settings = _settings()
    take_format = settings.take_format if lossless is None else "m4a" if lossless else "mp3"
    songs = Library(library or Path(settings.library), take_format)
    device = device or settings.device
    started = datetime.now()
    raw = songs.raw_path(started, "" if lossless is None else f".{take_format}")  # the format, for whenever it's filed
    recording = raw.with_suffix(".recording")  # not waiting to be filed until it's finished
    try:
        record(recording, console, int(device) if device and device.isdigit() else device, channels, auto_stop)
    except (ValueError, sd.PortAudioError) as e:  # no such device, or it can't record that way (e.g. --channels)
        console.print(f"[red]Can't record from “{escape(device or 'your default microphone')}”: {escape(str(e))}."
                      "[/red] Plug it in, or pick another microphone with `shed init` or --device.")
        raise typer.Exit(1)
    finally:
        if recording.exists():
            recording.rename(raw)
    if later:
        samples = audio.load(raw)
        if not _worth_keeping(samples, *audio.playing_bounds(samples)):
            raw.unlink()
            return
        waiting = len(songs.waiting())
        console.print(f"[green]✓[/green] Kept for later ({waiting} take{'' if waiting == 1 else 's'} waiting). "
                      f"File {'it' if waiting == 1 else 'them'} with [bold]shed add[/bold].")
        return
    try:
        _file_take(raw, started, songs, language, isolate, settings.genius_token, from_mic=True)
    except BaseException:
        console.print(f"[yellow]Your recording is safe in {escape(str(raw))}; run `shed add` to file it.[/yellow]")
        raise
    raw.unlink()


@app.command()
def add(
    paths: Annotated[list[Path] | None, typer.Argument(
        exists=True, show_default=False,
        help="Audio files (any format), or folders to pick the songs out of. None: the takes waiting to be filed "
             "(from `shed rec --later`).")] = None,
    library: LibraryOpt = None,
    language: LanguageOpt = None,
    isolate: IsolateOpt = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only show which recordings would be filed.")] = False,
):
    """File recordings you already have, e.g. voice memos, or the takes you recorded to file later. Give it a
    folder and it picks out the songs.

    The originals are left untouched, and a recording already filed is never filed twice.
    """
    settings = _settings()
    songs = Library(library or Path(settings.library), settings.take_format)
    recordings = _recordings(paths, songs) if paths else _waiting(songs)
    if not recordings and not paths:
        console.print("No takes are waiting to be filed. To file recordings you have: shed add ~/Downloads/memo.m4a")
        return
    filed, passed_over = songs.sources(), songs.skipped()
    waiting = {p.resolve() for p in songs.waiting()}  # Woodshed's own raw copies: `shed rec --later`, or one that failed
    tally: Counter[str] = Counter()
    unsure: list[tuple[Heard, str, list[genius.Candidate], str]] = []  # from folders or waiting: asked at the end
    listened: list[Heard] = []  # encoded for filing: see _listen()

    def filed_as(heard: Heard, choice: Choice, source: str) -> None:
        filed[source] = _save(heard, choice, songs, source).path
        tally["filed"] += 1
        if heard.src in waiting:  # filed at last: the raw copy has done its job
            heard.src.unlink()

    try:
        for path, recorded, kind in recordings:
            source = lib.fingerprint(path)
            if source in filed:
                if kind == "named":
                    console.print(f"{escape(path.name)} is already filed, as "
                                  f"{escape(str(filed[source].relative_to(songs.root)))}.")
                if path in waiting and not dry_run:
                    path.unlink()
                tally["already filed"] += 1
                continue
            if kind == "folder" and source in passed_over:
                tally[passed_over[source]] += 1
                continue
            console.rule(escape(path.name) if kind != "waiting" else f"Take recorded {recorded:%Y-%m-%d %H:%M}")
            try:
                if kind == "folder" and (reason := _why_not_a_song(path)):
                    console.print(f"[dim]Skipped: {reason}.[/dim]")
                    if not dry_run:
                        songs.remember_skipped(source, path, reason)
                    tally[reason] += 1
                    continue
                if dry_run:
                    console.print("Has singing: would be filed." if kind == "folder" else "Would be filed.")
                    tally["filed"] += 1
                    continue
                heard = _listen(path, recorded, songs, language, isolate, from_mic=False,
                                take_format=_format_chosen(path) if path in waiting else None)
                if heard is None:
                    continue
                listened.append(heard)
                choice, ranked, candidates = _recognize(heard.lines, songs, settings.genius_token)
                if choice is None and kind != "named":
                    console.print("[dim]Not sure which song this is: I'll ask at the end.[/dim]")
                    unsure.append((heard, source, candidates, kind))
                    continue
                filed_as(heard, choice or _ask(ranked, candidates[0] if candidates else None, songs), source)
            except audio.AudioError as e:  # not audio, or damaged: say so, and go on with the others
                reason = str(e).splitlines()[-1] if str(e) else "ffmpeg couldn't decode it"
                console.print(f"[red]Can't read {escape(path.name)}: {escape(reason)}[/red]")
                tally["unreadable"] += 1

        for i, (heard, source, candidates, kind) in enumerate(unsure, 1):
            name = escape(heard.src.name) if kind != "waiting" else f"take recorded {heard.recorded:%Y-%m-%d %H:%M}"
            console.rule(f"Unsure {i} of {len(unsure)}: {name}")
            # Your songs may have grown since it was heard, so it may be clear now; Genius isn't asked again.
            choice, ranked, _ = _recognize(heard.lines, songs, settings.genius_token, candidates)
            # A take you recorded is kept (in Unsorted if need be); one found in a folder can be skipped.
            if choice is None and (choice := _ask(ranked, candidates[0] if candidates else None, songs,
                                                  can_skip=kind == "folder")) is None:
                songs.remember_skipped(source, heard.src, "you skipped it")
                tally["you skipped it"] += 1
                continue
            filed_as(heard, choice, source)
    finally:
        for heard in listened:  # those not filed: skipped, or you stopped
            heard.take.unlink(missing_ok=True)

    if len(recordings) > 1 or not paths or any(path.is_dir() for path in paths):  # a folder's skips are only told here
        _summarize(tally, dry_run)
    if tally["unreadable"]:
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
        if reference := songs.reference(song):
            console.print(f"Rated against the melody of {escape(str(reference))}.")
        console.print(f"[dim]{escape(str(songs.root / song))}[/dim]")
        return

    marked = {song for song in catalog if songs.has_reference(song)}
    table = Table("Song", "Takes", "First", "Latest", *(["Reference"] if marked else []))
    for song, takes in catalog.items():
        table.add_row(escape(song), str(len(takes)), *((_when(takes[0]), _when(takes[-1])) if takes else ("", "")),
                      *(["✓" if song in marked else ""] if marked else []))
    console.print(table)
    if unsorted := songs.unsorted():
        console.print(f"{len(unsorted)} take(s) waiting in [bold]{escape(str(songs.root / lib.UNSORTED))}[/bold]; "
                      "move them into a song folder to file them.")
    if waiting := songs.waiting():
        console.print(f"[yellow]{len(waiting)} take{'' if len(waiting) == 1 else 's'} waiting to be filed.[/yellow] "
                      f"File {'it' if len(waiting) == 1 else 'them'} with: shed add")


@app.command()
def play(
    name: Annotated[list[str], typer.Argument(help="The song; part of its name is enough.", show_default=False)],
    by_rating: Annotated[bool, typer.Option("--rating", help="Play your worst take, then your best.")] = False,
    by_pitch: Annotated[bool, typer.Option("--pitch", help="The same, by pitch alone.")] = False,
    by_timing: Annotated[bool, typer.Option("--timing", "--tempo", help="The same, by timing alone.")] = False,
    take: TakeOpt = None,
    library: LibraryOpt = None,
):
    """Play a song's first take, then its latest, to hear how far you've come."""
    from woodshed import player

    chosen = [key for key, on in (("overall", by_rating), ("pitch", by_pitch), ("timing", by_timing)) if on]
    if len(chosen) > 1 or (chosen and take):
        console.print("Choose one of --rating, --pitch, --timing and --take.")
        raise typer.Exit(1)
    key = chosen[0] if chosen else None
    settings = config.load()
    songs = _existing_library(library, settings)
    song = _find_song(songs, " ".join(name))  # quoted or not
    takes = songs.takes_of(song)
    if not takes:
        console.print(f"“{escape(song)}” has no takes yet.")
        raise typer.Exit(1)

    if key:
        references = settings.references()
        takes, ratings, _ = _rate_all(songs, song)
        rated = [(r.scores(references)[key], number, take, r) for number, (take, r) in enumerate(zip(takes, ratings), 1)
                 if r and r.scores(references)[key] is not None]
        if not rated:
            console.print(f"None of the takes of “{escape(song)}” could be rated on {key}.")
            raise typer.Exit(1)
        worst, best = min(rated, key=lambda r: r[:2]), max(rated, key=lambda r: r[:2])
        picks = [("worst", worst), ("best", best)] if len(rated) > 1 else [("only rated", worst)]
        picks = [(f"{which} take by {'rating' if key == 'overall' else key} ({number} of {len(takes)})", t,
                  f", {_score_text(r, key, references)}") for which, (_, number, t, r) in picks]
    elif take:
        _check_take(song, take, len(takes))
        picks = [(f"take {take} of {len(takes)}", takes[take - 1], "")]
    else:
        if len(takes) == 1:
            console.print("[dim]Only one take so far.[/dim]")
        picks = [(f"{'first' if n == 1 else 'latest'} take ({n} of {len(takes)})", takes[n - 1], "")
                 for n in dict.fromkeys([1, len(takes)])]

    for label, t, detail in picks:
        console.print(f"[green]▶[/green] [bold]{escape(song)}[/bold], {label}, recorded {_when(t.path)}{detail}")
        if not player.play(t.path, console):
            break


@app.command()
def progress(
    name: Annotated[list[str], typer.Argument(help="The song; part of its name is enough.", show_default=False)],
    take: TakeOpt = None,
    lines: Annotated[bool, typer.Option("--lines", help="How each line of the song went in your latest takes, "
                                                        "against its reference melody.")] = False,
    library: LibraryOpt = None,
):
    """Rate every take of a song, from first to latest; or show one take, or each line, in detail."""
    settings = config.load()
    references = settings.references()
    songs = _existing_library(library, settings)
    song = _find_song(songs, " ".join(name))  # quoted or not
    if not songs.songs()[song]:
        console.print(f"“{escape(song)}” has no takes yet.")
        raise typer.Exit(1)
    if take and lines:
        console.print("Choose one of --take and --lines.")
        raise typer.Exit(1)
    if take:
        _show_take(songs, song, take, references)
        return
    if lines:
        _show_lines(songs, song, references)
        return
    takes, ratings, reference = _rate_all(songs, song)

    table = Table("#", "Recorded", "Rating", "Pitch", "Timing", title=escape(song), title_justify="left")
    for number, (take, r) in enumerate(zip(takes, ratings), 1):
        if r is None:
            table.add_row(str(number), _when(take.path), "–", "–", "–")
            continue
        s = r.scores(references)
        if s["pitch"] is None:
            pitch = "–"
        elif r.comparison:
            pitch = f"{s['pitch']:.1f}  [dim]{r.comparison.typical:.0f}¢ off · {_lean(r.comparison.where)}[/dim]"
        else:
            lean = "" if r.metrics.pitch_lean_cents is None else f" · {_lean(r.metrics.pitch_lean_cents)}"
            pitch = (f"{s['pitch']:.1f}  [dim]{r.metrics.pitch_cents:.0f}¢ off{' the scale' if reference else ''}"
                     f"{lean}[/dim]")
        timing = "–" if s["timing"] is None else f"{s['timing']:.1f}  [dim]±{100 * r.metrics.tempo_spread:.1f}%[/dim]"
        overall = "–" if s["overall"] is None else f"[bold]{s['overall']:.1f}[/bold]"
        table.add_row(str(number), _when(take.path), overall, pitch, timing)
    console.print(table)

    overall = [(n, score) for n, r in enumerate(ratings, 1) if r and (score := r.scores(references)["overall"]) is not None]
    if len(overall) > 1:
        best_number, best = max(overall, key=lambda o: (o[1], o[0]))
        console.print(f"Rating {_sparkline([score for _, score in overall])} {overall[0][1]:.1f} → "
                      f"{overall[-1][1]:.1f} since your first take; best {best:.1f} (take {best_number}).")
    compared = _compared(takes, ratings)
    if compared:
        number, _, comparison = compared[-1]
        console.print(f"\n[bold]Take {number}[/bold] against the melody:")
        _melody_feedback(comparison)
        if len(compared) > 1:
            console.print()
            _lines_summary(song, [c for _, _, c in compared], reference)
        console.print()
    if reference:
        console.print("[dim]Pitch: “off” is how far your lines typically are from the reference melody, either way "
                      "(that's what's rated), and “under” or “over” where they typically sit, in cents (100¢ = a "
                      "semitone). Timing: how steady your tempo is. Scores are out of 10.[/dim]")
    else:
        console.print("[dim]Pitch: “off” is how far your held notes typically are from the song's scale, either way "
                      "(that's what's rated), and “under” or “over” where they typically sit, against your "
                      "instrument's tuning. Timing: how steady your tempo is. Scores are out of 10.[/dim]")
    console.print(f"[dim]One take in detail: shed progress {escape(shlex.quote(song))} --take N[/dim]")


def _check_take(song: str, number: int, count: int) -> None:
    if number > count:
        console.print(f"“{escape(song)}” has {count} take{'' if count == 1 else 's'}: "
                      f"--take goes from 1 to {count}.")
        raise typer.Exit(1)


def _show_take(songs: Library, song: str, number: int, references: rating.References) -> None:
    """One take's rating, and what stands out against the song's reference melody if it has one."""
    reference = _reference(songs, song)
    takes = songs.takes_of(song, melody=reference is not None)
    _check_take(song, number, len(takes))
    [take] = _rate(songs, [takes[number - 1]], with_melody=reference is not None)  # only this one, if need be
    console.print(f"[bold]{escape(song)}[/bold], take {number} of {len(takes)}, recorded {_when(take.path)} "
                  f"({_length(take.path)})")
    _take_report(_rated(take.metrics, take.melody, reference), [], references, reference)
    console.print(f"[dim]{escape(str(take.path))}[/dim]")


def _lean(cents: float) -> str:
    """Where notes (or lines) typically sit: "40¢ under", "12¢ over" or "centered"."""
    return "centered" if abs(cents) < 0.5 else f"{abs(cents):.0f}¢ {'under' if cents < 0 else 'over'}"


def _compared(takes: list[Take], ratings: list["Rated | None"]) -> list[tuple[int, Take, melody.Comparison]]:
    """The latest takes compared with the song's reference melody (melody.RECENT_TAKES at most), oldest first,
    with their numbers."""
    return [(n, t, r.comparison) for n, (t, r) in enumerate(zip(takes, ratings), 1)
            if r and r.comparison][-melody.RECENT_TAKES:]


def _takes_compared(count: int) -> str:
    return "your take compared" if count == 1 else f"your last {count} takes compared"


def _typically(history: melody.History) -> str:
    """Where a line is typically sung against the melody: "95¢ under", "12¢ over"."""
    return "0¢" if round(history.typical) == 0 else _lean(history.typical)


def _off_note(history: melody.History) -> str:
    """How a line often off the melody is sung: "typically 95¢ under · under in 6 of 8 takes"."""
    side = "under" if history.typical < 0 else "over"
    return f"typically {_typically(history)} · {side} in {history.same_side} of {len(history.compared)} takes"


def _close_note(history: melody.History) -> str:
    """How one of the lines closest to the melody is sung: "typically 12¢ off · within 20¢ in 3 of 4 takes"."""
    return (f"typically {history.distance:.0f}¢ off · within {melody.CLOSE_CENTS}¢ in {history.close} of "
            f"{len(history.compared)} takes")


LINES_AGAINST = ("Each line against the melody moved to where the rest of its take sits: a take sung under throughout "
                 "is told so, and doesn't put all its lines off.")


def _lines_summary(song: str, comparisons: list[melody.Comparison], reference: lib.Reference) -> None:
    """The song's lines often off the melody in the latest takes, and those closest to it."""
    histories = melody.history(comparisons, reference.melody)
    off, close = melody.often_off(histories), melody.closest_across(histories)
    console.print(f"[bold]Line by line[/bold], {_takes_compared(len(comparisons))} with the melody:")
    if off:
        console.print("Often off it:")
        for h in off:
            console.print(f"  “{escape(_shorten(h.text))}”  {_off_note(h)}")
    else:
        console.print("No line is often off it.")
    if close:
        console.print("Closest to it:")
        for h in close:
            console.print(f"  “{escape(_shorten(h.text))}”  {_close_note(h)}")
    quoted = escape(shlex.quote(song))
    console.print(f"[dim]{LINES_AGAINST} Every line: shed progress {quoted} --lines[/dim]")


def _line_trend(history: melody.History, references: rating.References) -> str:
    """How close to the melody each take sang a line, scored as pitch is (· where it wasn't compared)."""
    return "".join("·" if cents is None else _sparkline([_melody_score(cents, references)]) for cents in history.cents)


def _melody_score(cents: float, references: rating.References) -> float:
    return rating.scores(rating.Metrics(None, None), references, melody_cents=abs(cents))["pitch"]


def _song_compared(songs: Library, song: str) -> tuple[list[tuple[int, Take, melody.Comparison]], lib.Reference]:
    """The song's latest takes compared with its reference melody (see _compared()), and the reference. Says why
    and stops when there are none."""
    if not songs.has_reference(song):
        console.print(f"“{escape(song)}” has no reference recording, so there's no melody to follow its lines "
                      f"against. Give it one: shed reference {escape(shlex.quote(song))}")
        raise typer.Exit(1)
    takes, ratings, reference = _rate_all(songs, song)
    if reference is None:  # analyzed by an older version: _reference() said to set it again
        raise typer.Exit(1)
    compared = _compared(takes, ratings)
    if not compared:
        console.print(f"None of the takes of “{escape(song)}” could be compared with its reference melody: too few "
                      "of their lines matched it.")
        raise typer.Exit(1)
    return compared, reference


def _show_lines(songs: Library, song: str, references: rating.References) -> None:
    """How each of the song's lines went in the latest takes compared with its reference melody."""
    compared, reference = _song_compared(songs, song)
    histories = melody.history([c for _, _, c in compared], reference.melody)
    off = {id(h) for h in melody.often_off(histories, most=len(histories))}
    close = {id(h) for h in melody.closest_across(histories)}
    first, last = compared[0][0], compared[-1][0]
    table = Table("Line", "Pitch", "Takes", "Take by take" if first == last else f"Takes {first}–{last}",
                  title=f"{escape(song)}, line by line: {_takes_compared(len(compared))} with the melody",
                  title_justify="left")
    for h in histories:
        pitch = f"{h.distance:.0f}¢ off · {_lean(h.typical)}"
        if id(h) in off:
            pitch = f"[yellow]{pitch}[/yellow]"
        elif id(h) in close:
            pitch = f"[green]{pitch}[/green]"
        table.add_row(f"“{escape(_shorten(h.text, 44))}”", pitch, str(len(h.compared)), _line_trend(h, references))
    console.print(table)
    console.print(f"[dim]{LINES_AGAINST} Pitch: how far from it the takes that sang the line typically sang it, either "
                  "way, and where they typically sang it, in cents (100¢ = a semitone). Once 3 takes have sung a line: "
                  f"in yellow, the lines often off it ({melody.OFTEN_OFF_CENTS}¢ or more, on the same side in most "
                  "takes), and in green, the 3 closest to it. Take by take: how close each take sang it, scored as pitch "
                  "is, oldest first (· where it wasn't sung, or not heard).[/dim]")


@app.command()
def reference(
    targets: Annotated[list[str], typer.Argument(
        metavar="[RECORDING] [SONG]", show_default=False,
        help="A recording of one of your songs (e.g. the original), or a link to one (YouTube…), and the song (part "
             "of its name is enough). Give only the recording and the song is recognized; only the song, and it says "
             "what it's rated against (and offers to look for the original).")],
    search: Annotated[bool, typer.Option("--search", help="Look for the original on YouTube, even if the song has a "
                                                          "reference already. Needs yt-dlp: brew install yt-dlp")] = False,
    remove: Annotated[bool, typer.Option("--remove", help="Rate the song against its scale again.")] = False,
    library: LibraryOpt = None,
    language: LanguageOpt = None,
):
    """Rate a song's pitch against the melody of a recording of it, instead of against its scale.

    Your notes are then compared with the ones sung on the same words, in your
    instrument's key, and the lines furthest from the melody are pointed out.
    """
    links = [t for t in targets if re.match(r"https?://", t)]
    files = [Path(t).expanduser() for t in targets if t not in links and Path(t).expanduser().is_file()]
    words = [t for t in targets if t not in links and not Path(t).expanduser().is_file()]
    if missing := [t for t in words if "/" in t or Path(t).suffix.lower() in AUDIO_SUFFIXES]:  # song names have no "/"
        console.print(f"There's no recording at {escape(missing[0])}.")
        raise typer.Exit(1)
    if len(files) + len(links) > 1:
        console.print("Give one recording at a time.")
        raise typer.Exit(1)
    name = " ".join(words) or None  # a song name left unquoted
    recording, link = (files[0] if files else None), (links[0] if links else None)
    settings = _settings() if (recording or link or search) and not remove else config.load()
    songs = _existing_library(library, settings)
    song = _find_song(songs, name) if name else None
    if song is None and not (recording or link):
        console.print("Name the song, e.g. shed reference harbor --remove")
        raise typer.Exit(1)
    if remove:
        had = songs.remove_reference(song)
        console.print(f"“{escape(song)}” is rated against its scale again." if had
                      else f"“{escape(song)}” has no reference melody.")
        return
    if not (recording or link or search):
        if current := songs.reference(song):
            console.print(f"“{escape(song)}” is rated against the melody of {escape(str(current))} "
                          f"(set {current.added:%Y-%m-%d}).")
            _reference(songs, song)  # says so if it needs setting again
            return
        console.print(f"“{escape(song)}” has no reference recording, so it's rated against its scale.")
        if not youtube.available():
            console.print(f"To rate it against the melody of the original: shed reference {escape(shlex.quote(song))} "
                          "<recording>. Or install yt-dlp (brew install yt-dlp), to find the original on YouTube.")
            return
        if not Confirm.ask("Look for the original on YouTube?", default=True):
            return
        settings, search = _settings(), True
    if (link or search) and not youtube.available():
        console.print("Getting a recording from YouTube needs yt-dlp: brew install yt-dlp")
        raise typer.Exit(1)
    if not songs.songs():
        console.print("You have no songs yet: a reference is for a song you've filed takes of.")
        raise typer.Exit(1)

    with tempfile.TemporaryDirectory(prefix="woodshed-") as scratch:  # a download is deleted once it's analyzed
        title = None
        try:
            if search and (link := _pick_original(song, songs)) is None:
                console.print("Nothing was changed.")
                return
            if link:
                with console.status("Downloading the recording…"):
                    recording, title = youtube.download(link, Path(scratch))
        except youtube.YouTubeError as e:
            console.print(f"[red]Couldn't get it from YouTube: {escape(str(e))}[/red]")
            raise typer.Exit(1)
        sung = _follow_melody(recording, title or recording.name, language)
    song = _reference_for(sung.lines, song, songs)
    if song is None:
        console.print("Nothing was changed.")
        return
    songs.set_reference(song, sung, link or _tilde(recording), title)
    console.print(f"[green]✓[/green] “{escape(song)}” is now rated against the melody of "
                  f"{escape(title or recording.name)} ({len(sung.lines)} lines heard).")
    older = sum(t.melody is None for t in songs.takes_of(song, melody=True))
    if older:
        console.print(f"{older} of its takes {'was' if older == 1 else 'were'} filed before Woodshed kept their "
                      "melody: `shed progress` analyzes them the first time (it takes a while).")
    console.print(f"See how your takes compare: shed progress {escape(shlex.quote(song))}")


def _pick_original(song: str, songs: Library) -> str | None:
    """The link of the video you say is the song's original recording, among those YouTube finds for it. When it
    isn't there (the song's artist is wrong, or unknown), you can say who sings the original, and it looks again:
    that artist is then kept for the song, once you pick one of the videos found."""
    known = artist = next((t.artist for t in reversed(songs.takes_of(song)) if t.artist), None)
    while True:
        query = f"{artist} {song}" if artist else song
        with console.status(f"Looking for “{escape(query)}” on YouTube…"):
            videos = youtube.search(f"{query} official audio")
        if videos:
            console.print("[bold]Which one is the original?[/bold] (a live or acoustic version has another melody)")
        else:
            console.print(f"YouTube found nothing for “{escape(query)}”.")
        answer = _pick_video(videos)
        if isinstance(answer, youtube.Video):
            if artist != known:
                songs.save_artist(song, artist)
                console.print(f"Saved {escape(artist)} as the artist of “{escape(song)}”.")
            return answer.url
        if not answer:
            return None
        artist = answer


def _pick_video(videos: list[youtube.Video]) -> youtube.Video | str | None:
    """The video you pick, or else the artist you'd look for instead (None: neither)."""
    about = [" · ".join(part for part in (video.channel, video.seconds and _clock(video.seconds), video.description)
                        if part) for video in videos]
    if _interactive():
        chosen = widgets.pick(console, [video.title for video in videos] + ["Search again with another artist…",
                                                                            "None of these"], details=about + ["", ""])
        if chosen < len(videos):
            return videos[chosen]
        return _input("Who sings the original? ", []) or None if chosen == len(videos) else None
    for i, (video, line) in enumerate(zip(videos, about), 1):
        console.print(f"  {i}. {escape(video.title)}")
        console.print(Text(f"     {line}", "dim", no_wrap=True, overflow="ellipsis"))
    prompt = ("Number to download, an artist to search for instead, or Enter to cancel: " if videos
              else "Another artist to search for, or Enter to cancel: ")
    while answer := _input(prompt, []):
        if not (videos and answer.isdigit()):
            return answer
        if 1 <= int(answer) <= len(videos):
            return videos[int(answer) - 1]
        console.print(f"Type a number from 1 to {len(videos)}.")
    return None


def _follow_melody(recording: Path, name: str, language: str | None) -> melody.Melody:
    """A reference recording's melody (see melody.extract()); stops if there isn't enough singing to follow."""
    from woodshed import isolate, transcribe

    try:
        with console.status("Listening to the recording…"):
            samples = audio.load(recording)
            start, end = audio.playing_bounds(samples)
        if end <= start:
            start, end = 0.0, len(samples) / audio.SAMPLE_RATE
        with console.status("Separating the voice from the instruments…"):
            stems = isolate.separate(recording, start, end - start)
    except audio.AudioError as e:
        console.print(f"[red]Can't read {escape(name)}: {escape(str(e).splitlines()[-1] if str(e) else '')}[/red]")
        raise typer.Exit(1)
    with console.status("Transcribing the lyrics…"):
        lines = transcribe.transcribe(stems.vocals_16k(), language)
    with console.status("Following the melody…"):
        sung = melody.extract(stems, lines, rating.pitch_track(stems))
    if len(sung.words) < MIN_REFERENCE_WORDS:
        console.print(f"[red]Heard only {len(sung.words)} words sung in {escape(name)}: too few to follow its "
                      "melody.[/red]")
        raise typer.Exit(1)
    return sung


def _reference_for(lines: list[str], named: str | None, songs: Library) -> str | None:
    """Which of your songs a reference recording is of, going by its lyrics: the one named, unless its lyrics
    don't match your takes of it and you'd rather not; else the one they match, if you agree; else the one
    you say. None if you change your mind."""
    ranked = songs.match(lines)
    likeliest = ranked[0][0] if lib.is_confident(lines, ranked) else None
    if named:
        if dict(ranked).get(named, 0.0) >= lib.MATCH_SCORE or not any(t.transcript for t in songs.takes_of(named)):
            return named
        console.print(f"[yellow]Its lyrics sound like “{escape(likeliest)}”, not “{escape(named)}”.[/yellow]" if likeliest
                      else f"[yellow]Its lyrics hardly match your takes of “{escape(named)}”.[/yellow]")
        return named if Confirm.ask(f"Use it for “{escape(named)}” anyway?", default=False) else None
    if likeliest and Confirm.ask(f"It sounds like “{escape(likeliest)}”. Rate that song against it?", default=True):
        return likeliest
    options = [song for song, score in ranked[:3] if score > 0.03 and song != likeliest]
    console.print("[bold]Which of your songs is it?[/bold]")
    while _interactive():
        chosen = widgets.pick(console, options + ["Another of your songs…", "Cancel"], 0,
                              [""] * len(options) + ["type its name (Tab completes)", ""])
        if chosen != len(options):
            return options[chosen] if chosen < len(options) else None
        if (name := _input("Song name (Enter to go back): ", list(songs.songs()))) and (song := _song_named(songs, name)):
            return song
        if name:
            console.print(f"You have no song called “{escape(name)}”.")
    for i, option in enumerate(options, 1):
        console.print(f"  {i}. {escape(option)}")
    while answer := _input("Number, song name (Tab completes), or Enter to cancel: ", list(songs.songs())):
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if song := _song_named(songs, answer):
            return song
        console.print(f"You have no song called “{escape(answer)}”.")
    return None


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="The port the page is on.")] = phone.PORT,
    later: Annotated[bool, typer.Option("--later", help="Only receive the takes: they wait to be filed by `shed add`, "
                                                        "as with `shed rec --later`.")] = False,
    library: LibraryOpt = None,
    language: LanguageOpt = None,
):
    """Record from your phone: open the page this puts up on your Wi-Fi. Each take you record there is sent here and
    filed, and the page shows how it rated (and asks which song it is, when that isn't clear)."""
    settings = _settings()
    songs = Library(library or Path(settings.library), settings.take_format)
    for stuck in songs.incoming.glob("*.wav.filing"):  # a `shed serve` stopped while filing it: waiting again
        stuck.rename(stuck.with_suffix(""))
    cert, key, secret = phone.credentials(config.PATH.parent / "phone")
    links = [f"https://{address}:{port}/?key={secret}" for address in phone.addresses()]
    session = phone.Session()
    filer = None if later else _PhoneFiler(songs, settings, session, language)

    def received(raw: Path, seconds: float, waiting: int) -> None:
        if filer:
            console.print(f"[green]✓[/green] {datetime.now():%H:%M} A take from your phone ({_clock(seconds)})")
            filer.takes.put(raw)
        else:
            console.print(f"[green]✓[/green] {datetime.now():%H:%M} A take from your phone ({_clock(seconds)}): "
                          f"{waiting} waiting to be filed")

    try:
        server = phone.Server(songs, secret, cert, key, port, received, session, filing=filer is not None)
    except OSError as e:
        console.print(f"[red]Can't use port {port}: {escape(e.strerror or str(e))}.[/red] "
                      f"Try another: shed serve --port {port + 1}")
        raise typer.Exit(1)
    console.print("[bold]On your phone, open this page[/bold] (on the same Wi-Fi as this Mac):")
    console.print(_qr(links[0]))
    console.print(links[0], soft_wrap=True)
    if len(links) > 1:
        console.print("[dim]If your phone is on another of this Mac's networks: " + ", ".join(links[1:]) + "[/dim]",
                      soft_wrap=True)
    console.print("[dim]Your phone will warn that the connection isn't private: the page uses a certificate of "
                  "Woodshed's own, which it doesn't know. Go on anyway (Advanced, then Proceed). If macOS asks whether "
                  "to accept incoming connections, allow them.[/dim]")
    if filer:
        console.print("The takes you record there are filed here as they come, and the page shows how they rated. "
                      "Ctrl+C stops.")
        filer.start()
    else:
        console.print("The takes you record there wait here to be filed: file them with [bold]shed add[/bold]. "
                      "Ctrl+C stops.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.stop()  # a question left unanswered: the take waits to be filed
        if filer:
            filer.takes.put(None)
            if filer.busy:
                console.print("Finishing the take being filed… (Ctrl+C again stops now: it will wait to be filed)")
            try:
                filer.join()
            except KeyboardInterrupt:
                pass
    if waiting := len(songs.waiting()):
        console.print(f"{waiting} take{'' if waiting == 1 else 's'} waiting to be filed. "
                      f"File {'it' if waiting == 1 else 'them'} with: shed add")


class _Stopped(Exception):
    """`shed serve` stopped before the phone said which song a take is."""


class _PhoneFiler(threading.Thread):
    """Files the takes the phone sends, one at a time, in the order they came (each can help recognize the next)."""

    def __init__(self, songs: Library, settings: config.Config, session: phone.Session, language: str | None):
        super().__init__(daemon=True)  # a second Ctrl+C mustn't wait for it
        self.songs, self.settings, self.session, self.language = songs, settings, session, language
        self.takes: queue.Queue[Path | None] = queue.Queue()
        self.busy = False

    def run(self) -> None:
        while (raw := self.takes.get()) is not None:
            self.busy = True
            try:
                _file_from_phone(raw, self.songs, self.settings, self.session, self.language)
            finally:
                self.busy = False


def _file_from_phone(raw: Path, songs: Library, settings: config.Config, session: phone.Session,
                     language: str | None) -> None:
    """File a take the phone sent, asking the phone which song it is when that isn't clear. If it can't be filed,
    it's left waiting to be filed (by `shed add`, or the next `shed serve`)."""
    take, filing = raw.name, raw.with_name(raw.name + ".filing")  # meanwhile not waiting: `shed add` leaves it be
    try:
        raw.rename(filing)
    except FileNotFoundError:
        session.update(take, status="failed", error="it was filed on your Mac already, by shed add")
        return
    session.update(take, status="filing")
    heard = None
    try:
        recorded = _raw_time(raw)
        console.rule(f"Take from your phone, recorded {recorded:%Y-%m-%d %H:%M}")
        heard = _listen(filing, recorded, songs, language, True, from_mic=False, take_format=_format_chosen(raw))
        choice, ranked, candidates = _recognize(heard.lines, songs, settings.genius_token)
        if choice is None:
            options = _song_options(ranked, candidates[0] if candidates else None, songs)
            console.print("[dim]Not sure which song this is: asking your phone.[/dim]")
            if (answer := session.ask(take, [f"{name} ({where})" for _, name, where in options])) is None:
                raise _Stopped
            choice = (options[answer["option"]][0] if "option" in answer
                      else Choice(None) if "unsorted" in answer else _choice_named(answer["name"], songs))
        filed = _save(heard, choice, songs, lib.fingerprint(filing))
    except _Stopped:
        filing.rename(raw)
        session.update(take, status="kept")
        return
    except Exception as e:  # whatever it was, the take isn't lost
        filing.rename(raw)
        session.update(take, status="failed", error=f"{e}. It waits on your Mac, to be filed with shed add")
        console.print(f"[red]Couldn't file the take from your phone: {escape(str(e))}[/red] It waits to be filed.")
        return
    finally:
        if heard:
            heard.take.unlink(missing_ok=True)  # if it wasn't filed
    filing.unlink()
    session.update(take, status="filed", result=_report(filed, config.load().references()))


def _report(filed: Filed, references: rating.References) -> dict:
    """What the phone shows of a take once it's filed: what the terminal says, as plain text."""
    result = {"song": filed.song or lib.UNSORTED, "take": filed.number, "of": filed.count, "notes": [], "furthest": [],
              "closest": []}
    if filed.rated is None:
        return result
    overall, details, compared = _rating_parts(filed.rated, filed.earlier, references)
    result["rating"] = UNRATED if overall is None else " · ".join(
        [f"Rated {overall:.1f}/10", *details, *([compared] if compared else [])])
    if comparison := filed.rated.comparison:
        transposed, sitting, far, close = _melody_points(comparison)
        result["notes"] = [note for note in (transposed, sitting) if note]
        result["furthest"], result["closest"] = (
            [{"time": _clock(line.start), "text": line.text, "off": _off(line)} for line in lines] for lines in (far, close))
        if not sitting and not far:
            result["notes"].append(ALL_CLOSE)
    elif filed.reference:
        result["notes"] = [ON_THE_SCALE]
    return result


def _qr(text: str) -> Text:
    """A QR code of `text`, two rows of it per line of characters, dark on light whatever the terminal's colors."""
    import segno

    rows = [list(row) for row in segno.make(text, error="l").matrix_iter(border=2)]
    rows += [[0] * len(rows[0])] * (len(rows) % 2)
    code = Text()
    for top, bottom in zip(rows[::2], rows[1::2]):
        for dark_top, dark_bottom in zip(top, bottom):
            code.append("▀", style=f"{'#000000' if dark_top else '#ffffff'} on {'#000000' if dark_bottom else '#ffffff'}")
        code.append("\n")
    return code


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


def _song_named(songs: Library, name: str) -> str | None:
    """The song called `name`, or the only one whose name contains it."""
    partial = [song for song in songs.songs() if name.casefold() in song.casefold()]
    return songs.find_song(name) or (partial[0] if len(partial) == 1 else None)


def _find_song(songs: Library, name: str) -> str:
    """The song called `name`, or the only one whose name contains it; stops if there's none."""
    partial = [song for song in songs.songs() if name.casefold() in song.casefold()]
    song = _song_named(songs, name)
    if song is None:
        hint = f"; did you mean {' or '.join(partial)}?" if partial else f" in {songs.root}"
        console.print(f"No song called “{escape(name)}”{escape(hint)}")
        raise typer.Exit(1)
    return song


@dataclass
class Rated:
    """A take's measurements, and how it compares with its song's reference melody (when the song has one,
    and enough of the take could be compared with it)."""
    metrics: rating.Metrics
    comparison: melody.Comparison | None = None

    def scores(self, references: rating.References) -> dict[str, float | None]:
        return rating.scores(self.metrics, references, self.comparison.typical if self.comparison else None)


def _rated(metrics: rating.Metrics | None, sung: melody.Melody | None, reference: lib.Reference | None,
           ) -> Rated | None:
    """None for a take not rated yet."""
    if metrics is None:
        return None
    return Rated(metrics, melody.compare(sung, reference.melody) if reference and sung else None)


def _reference(songs: Library, song: str) -> lib.Reference | None:
    """The song's reference melody, if it has one that's up to date."""
    reference = songs.reference(song)
    if reference and reference.melody is None:
        console.print(f"[yellow]The reference melody of “{escape(song)}” was analyzed by an older version of "
                      f"Woodshed, so it's rated against its scale. Set it again: shed reference "
                      f"{escape(shlex.quote(song))} {escape(shlex.quote(reference.file))}[/yellow]")  # a file or a link
        return None
    return reference


def _rate(songs: Library, takes: list[Take], with_melody: bool = False) -> list[Take]:
    """The takes, after rating any that haven't been rated yet, and with `with_melody`, following the melody of
    those filed before Woodshed kept it (saved, so only once)."""
    from woodshed import isolate, transcribe

    todo = [t for t in takes if t.metrics is None or (with_melody and t.melody is None)]
    if todo and (missing := models.missing()):
        console.print(f"[yellow]{_not_downloaded(missing)} Run `shed init` to download.[/yellow]")
        raise typer.Exit(1)
    for i, take in enumerate(todo, 1):
        with console.status(f"Rating take {i} of {len(todo)} (only needed once)…"):
            stems = isolate.separate(take.path)
            needs_melody = with_melody and take.melody is None
            track = rating.pitch_track(stems) if needs_melody else None
            if take.metrics is None:
                take.metrics = rating.analyze(stems, track)
                songs.save_metrics(take.path, take.metrics)
            if needs_melody:  # the whole take transcribed again, for when each word is sung
                lines = transcribe.transcribe(stems.vocals_16k(), os.environ.get("WOODSHED_LANGUAGE"))
                take.melody = melody.extract(stems, lines, track)
                songs.save_melody(take.path, take.melody)
    return takes


def _rate_all(songs: Library, song: str) -> tuple[list[Take], list[Rated | None], lib.Reference | None]:
    """A song's takes, oldest first, each with its rating (None if it has no measurements), against the
    song's reference melody when it has one."""
    reference = _reference(songs, song)
    takes = _rate(songs, songs.takes_of(song, melody=reference is not None), with_melody=reference is not None)
    with console.status("Comparing your takes with the reference melody…") if reference else nullcontext():
        return takes, [_rated(t.metrics, t.melody, reference) for t in takes], reference


def _score_text(rated: Rated, key: str, references: rating.References, figures: bool = True) -> str:
    score = rated.scores(references)[key]
    if key == "pitch" and not figures:
        return f"pitch {score:.1f}/10"
    if key == "pitch":
        if rated.comparison:
            return (f"pitch {score:.1f}/10 ({rated.comparison.typical:.0f}¢ off the melody · "
                    f"{_lean(rated.comparison.where)})")
        lean = "" if rated.metrics.pitch_lean_cents is None else f" · {_lean(rated.metrics.pitch_lean_cents)}"
        return f"pitch {score:.1f}/10 ({rated.metrics.pitch_cents:.0f}¢ off{lean})"
    if key == "timing":
        return f"timing {score:.1f}/10 (tempo ±{100 * rated.metrics.tempo_spread:.1f}%)"
    return f"rated {score:.1f}/10"


BEST_YET = "your best take yet!"
UNRATED = "Not enough singing or strumming to rate this take."
ON_THE_SCALE = "Too few lines of this take matched the reference, so its pitch is rated against the scale."


def _rating_parts(rated: Rated, earlier: list[Rated], references: rating.References, pitch_figures: bool = True,
                  ) -> tuple[float | None, list[str], str | None]:
    """A take's overall rating (None if it couldn't be rated), what it's made of, and how it compares with your
    best among `earlier` takes."""
    s = rated.scores(references)
    details = [_score_text(rated, key, references, figures=pitch_figures or key != "pitch")
               for key in rating.METRICS if s[key] is not None]
    if s["pitch"] is None:
        details.append("pitch: not enough singing to judge")
    best = max((score for r in earlier if (score := r.scores(references)["overall"]) is not None), default=None)
    compared = None if best is None or s["overall"] is None else BEST_YET if s["overall"] > best else f"your best: {best:.1f}"
    return s["overall"], details, compared


def _take_report(rated: Rated, earlier: list[Rated], references: rating.References,
                 reference: lib.Reference | None) -> None:
    """A take's rating, then its pitch in detail and what stands out against the song's reference melody: what
    `shed rec` and `shed add` say once a take is filed, and what `shed progress --take` shows."""
    overall, details, compared = _rating_parts(rated, earlier, references, pitch_figures=False)
    if overall is None:
        console.print(f"[dim]{UNRATED}[/dim]")
        return
    compared = f"[green]{compared}[/green]" if compared == BEST_YET else compared
    console.print(" · ".join([f"Rated [bold]{overall:.1f}/10[/bold]", *details, *([compared] if compared else [])]))
    if comparison := rated.comparison:
        best, worst = references.melody_cents
        under, over = sum(line.cents < 0 for line in comparison.lines), sum(line.cents > 0 for line in comparison.lines)
        console.print(f"Pitch, against the melody ({len(comparison.lines)} lines compared):")
        console.print(f"  Typically {comparison.typical:.0f}¢ off, either way (10/10 at {best:g}¢, 0/10 at {worst:g}¢)")
        console.print(f"  Typically {_lean(comparison.where)}: {under} line{'' if under == 1 else 's'} sit"
                      f"{'s' if under == 1 else ''} under the melody, {over} over")
        _melody_feedback(comparison, sitting=False)
    elif rated.metrics.pitch_cents is not None:
        best, worst = references.pitch_cents
        on_the_scale = ON_THE_SCALE.lower().rstrip(".")
        console.print("Pitch, against the song's scale" + (f" [dim]({on_the_scale})[/dim]:" if reference else ":"))
        console.print(f"  Typically {rated.metrics.pitch_cents:.0f}¢ off, either way "
                      f"(10/10 at {best:g}¢, 0/10 at {worst:g}¢)")
        if (lean := rated.metrics.pitch_lean_cents) is not None:
            console.print(f"  Typically {_lean(lean)}, against your instrument's tuning")
        else:
            console.print("  [dim]Where the notes sit, under or over, wasn't measured for this take.[/dim]")


ALL_CLOSE = f"No line strays {melody.FAR_CENTS}¢ or more from the melody."


def _melody_points(comparison: melody.Comparison,
                   ) -> tuple[str | None, str | None, list[melody.Line], list[melody.Line]]:
    """What stands out in a take against the reference melody: your instrument's key, where your lines sit,
    and the lines furthest from it and closest to it."""
    return (melody.transposed(comparison), melody.sitting(comparison), melody.furthest(comparison),
            melody.closest(comparison))


def _off(line: melody.Line) -> str:
    if round(abs(line.cents)) == 0:
        return "0¢"
    return f"{abs(line.cents):.0f}¢ {'under' if line.cents < 0 else 'over'}"


def _melody_feedback(comparison: melody.Comparison, sitting: bool = True) -> None:
    """_melody_points(), with when the lines furthest from the melody and closest to it start in the take.
    Without `sitting`, where the lines sit is left out (it's been said)."""
    transposed, sits, far, close = _melody_points(comparison)
    sits = sits if sitting else None
    if transposed:
        console.print(f"[dim]{transposed}[/dim]")
    if sits:
        console.print(sits)
    if not far and not sits:
        console.print(ALL_CLOSE)
    for heading, lines in (("Furthest from the melody:", far), ("Closest to the melody:", close)):
        if lines:
            console.print(heading)
        for line in lines:
            console.print(f"  {_clock(line.start)}  “{escape(_shorten(line.text))}”  {_off(line)}")


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _shorten(text: str, width: int = 40) -> str:
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


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
    if settings.take_format not in lib.FORMATS:
        console.print(f"[yellow]take_format in {escape(_tilde(config.PATH))} must be {' or '.join(lib.FORMATS)}. "
                      "Run `shed init` to choose again.[/yellow]")
        raise typer.Exit(1)
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


def _pick_format(current: str) -> str:
    """How takes are saved. Takes already filed stay as they are."""
    labels = {"mp3": "MP3", "m4a": "Apple Lossless (.m4a)"}
    notes = {"mp3": "about 1.4 MB a minute", "m4a": "exactly as recorded, about 4 times larger"}
    current = current if current in lib.FORMATS else lib.FORMATS[0]
    console.print("Save each take as:")
    if _interactive():
        return lib.FORMATS[widgets.pick(console, [labels[f] for f in lib.FORMATS], lib.FORMATS.index(current),
                                        [notes[f] for f in lib.FORMATS])]
    for f in lib.FORMATS:
        console.print(f"  {f}: {labels[f]}, {notes[f]}")
    return Prompt.ask("Format", choices=list(lib.FORMATS), default=current)


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
        return "A take " + examples([15, 20, 25, 30], "{:g}¢ off", rating.References(pitch_cents=(best, worst)), "pitch")

    def on_melody(best: float, worst: float) -> str:
        references = rating.References(melody_cents=(best, worst))
        return "A take " + "  ·  ".join(f"{v:g}¢ off → {_melody_score(v, references):.1f}" for v in (20, 40, 60, 80))

    def timing(best: float, worst: float) -> str:
        return "A take " + examples([1.5, 2.5, 4, 5], "±{:g}%", rating.References(tempo_spread=(best / 100, worst / 100)),
                                    "timing")

    cents, percent = "{:g}¢".format, "±{:g}%".format
    s = settings
    s.pitch_best_cents = widgets.slide(console, "  Pitch scores 10/10 within", s.pitch_best_cents, 0,
                                       s.pitch_worst_cents - 1, 1, cents, lambda v: pitch(v, s.pitch_worst_cents))
    s.pitch_worst_cents = widgets.slide(console, "  Pitch scores 0/10 from", s.pitch_worst_cents, s.pitch_best_cents + 1,
                                        50, 1, cents, lambda v: pitch(s.pitch_best_cents, v))
    s.melody_best_cents = widgets.slide(console, "  Against a reference melody, pitch scores 10/10 within",
                                        s.melody_best_cents, 0, s.melody_worst_cents - 5, 1, cents,
                                        lambda v: on_melody(v, s.melody_worst_cents))
    s.melody_worst_cents = widgets.slide(console, "  Against a reference melody, pitch scores 0/10 from",
                                         s.melody_worst_cents, s.melody_best_cents + 5, MELODY_LIMIT, 5, cents,
                                         lambda v: on_melody(s.melody_best_cents, v))
    s.timing_best_percent = widgets.slide(console, "  Timing scores 10/10 within", s.timing_best_percent, 0,
                                          s.timing_worst_percent - 0.5, 0.5, percent,
                                          lambda v: timing(v, s.timing_worst_percent))
    s.timing_worst_percent = widgets.slide(console, "  Timing scores 0/10 from", s.timing_worst_percent,
                                           s.timing_best_percent + 0.5, 15, 0.5, percent,
                                           lambda v: timing(s.timing_best_percent, v))
    s.pitch_weight_percent = widgets.slide(console, "  The rating is", s.pitch_weight_percent, 0, 100, 5, _shares,
                                           lambda v: f"A take with pitch 8.0 and timing 5.0 → {(8 * v + 5 * (100 - v)) / 100:.1f}")


def _shares(pitch_percent: float) -> str:
    return f"{pitch_percent:g}% pitch, {100 - pitch_percent:g}% timing"


def _ask_share(current: float) -> float:
    while True:
        share = FloatPrompt.ask(f"Pitch's share of the rating, in % (timing gets the rest) ({current:g})",
                                default=float(current), show_default=False)
        if 0 <= share <= 100:
            return share
        console.print("[yellow]Between 0 and 100.[/yellow]")


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


def _recordings(paths: list[Path], songs: Library) -> list[tuple[Path, datetime, str]]:
    """What to add, oldest first so each take filed can help recognize the later ones: the files named
    ("named"), and the audio files in the folders named ("folder"). Hidden files, like a recorder app's
    recently deleted recordings, and your library itself are left out."""
    found: dict[Path, str] = {}
    library = songs.root.resolve()
    for path in paths:
        if not path.is_dir():
            found[path.resolve()] = "named"  # always filed, even if it's in a folder named too
            continue
        for file in sorted(path.rglob("*")):
            if (file.suffix.lower() in AUDIO_SUFFIXES and file.is_file()
                    and not any(part.startswith(".") for part in file.relative_to(path).parts)
                    and not file.resolve().is_relative_to(library)):
                found.setdefault(file.resolve(), "folder")
    return sorted(((path, audio.recorded_at(path), kind) for path, kind in found.items()),
                  key=lambda recording: recording[1])


def _waiting(songs: Library) -> list[tuple[Path, datetime, str]]:
    """The takes recorded but not filed yet, oldest first, as _recordings() gives them."""
    return sorted(((path.resolve(), _raw_time(path), "waiting") for path in songs.waiting()), key=lambda r: r[1])


def _raw_time(path: Path) -> datetime:
    """When a raw recording waiting to be filed was made: its name says (see Library.raw_path)."""
    try:
        return datetime.strptime(path.name[:19], lib.RAW_STAMP)  # before any _2, or .m4a
    except ValueError:
        return audio.recorded_at(path)


def _why_not_a_song(path: Path) -> str | None:
    """Why a recording found in a folder isn't worth filing, or None if it is. It takes a few seconds and
    comes first, so nothing that isn't a song gets transcribed, let alone looked up on Genius."""
    with console.status("Checking for singing…"):
        samples = audio.load(path)
        start, end = audio.playing_bounds(samples)
        if end <= start:
            return "silent"
        if end - start < MIN_TAKE_SECONDS:
            return f"shorter than {MIN_TAKE_SECONDS} s"
        if not rating.sings(samples[int(start * audio.SAMPLE_RATE): int(end * audio.SAMPLE_RATE)]):
            return "no singing"
    return None


def _summarize(tally: Counter[str], dry_run: bool) -> None:
    skipped = {reason: n for reason, n in tally.most_common() if reason not in ("filed", "unreadable")}
    parts = [f"{'Would file' if dry_run else 'Filed'} {tally['filed']} take{'' if tally['filed'] == 1 else 's'}"]
    if skipped:
        parts.append(f"skipped {sum(skipped.values())}: " + ", ".join(f"{reason} ({n})" for reason, n in skipped.items()))
    if tally["unreadable"]:
        parts.append(f"couldn't read {tally['unreadable']}")
    console.rule()
    console.print(" · ".join(parts))


@dataclass
class Heard:
    """A take that's been listened to (voice separated, lyrics transcribed, rated), not yet filed."""
    src: Path  # the recording
    take: Path  # its playing part, encoded as it will be filed: what was listened to
    recorded: datetime
    lines: list[str]
    metrics: rating.Metrics | None
    melody: melody.Melody | None


def _format_chosen(raw: Path) -> str | None:
    """The format a take waiting to be filed was recorded to be saved in (`shed rec --m4a`), if any."""
    chosen = raw.suffixes[-2][1:] if len(raw.suffixes) > 1 else None
    return chosen if chosen in lib.FORMATS else None


def _file_take(src: Path, recorded: datetime, songs: Library, language: str | None, isolate_voice: bool,
               genius_token: str | None, from_mic: bool) -> Path | None:
    """File one take, asking which song it is when that isn't clear. Returns where it was saved, or None
    if it wasn't kept (silent, or too short)."""
    heard = _listen(src, recorded, songs, language, isolate_voice, from_mic)
    if heard is None:
        return None
    try:
        choice, ranked, candidates = _recognize(heard.lines, songs, genius_token)
        return _save(heard, choice or _ask(ranked, candidates[0] if candidates else None, songs), songs).path
    finally:
        heard.take.unlink(missing_ok=True)  # if it wasn't filed


def _listen(src: Path, recorded: datetime, songs: Library, language: str | None, isolate_voice: bool,
            from_mic: bool, take_format: str | None = None) -> Heard | None:
    """Encode the take as it will be filed (as the library saves takes, or in `take_format`), then separate
    its voice, transcribe the lyrics and rate it. None if it isn't worth keeping."""
    from woodshed import isolate, transcribe

    with console.status("Listening to the take…"):
        samples = audio.load(src)
        start, end = audio.playing_bounds(samples)
    if from_mic and not _worth_keeping(samples, start, end):
        return None
    if end <= start:
        start, end = 0.0, len(samples) / audio.SAMPLE_RATE

    # What's analyzed is the take as filed, not the recording, so that analyzing it again (after an update,
    # say) reads the same: an mp3 isn't exactly what it was made from, and Whisper can hear the difference.
    take = songs.encode(src, start, end, take_format)
    try:
        metrics = sung = None
        if isolate_voice:
            with console.status("Separating your voice from the instrument…"):
                stems = isolate.separate(take)
            voice = stems.vocals_16k()
        else:
            voice = audio.load(take)
        with console.status("Transcribing the lyrics…"):
            lines = transcribe.transcribe(voice, language)
        if isolate_voice:
            with console.status("Rating the take…"):
                track = rating.pitch_track(stems)
                metrics = rating.analyze(stems, track)
                sung = melody.extract(stems, lines, track)  # kept, for when the song has a reference melody
    except BaseException:
        take.unlink()
        raise
    return Heard(src, take, recorded, [line.text for line in lines], metrics, sung)


def _worth_keeping(samples: np.ndarray, start: float, end: float) -> bool:
    """Whether a take just recorded is kept: not if it's silent, and if it's short, only if you say so."""
    if samples.size == 0 or abs(samples).max() < 1e-4:
        console.print("[yellow]The recording is completely silent, so it wasn't kept. Allow your terminal app to "
                      "use the microphone in System Settings → Privacy & Security → Microphone.[/yellow]")
        return False
    playing = end - start if end > start else len(samples) / audio.SAMPLE_RATE
    if playing < MIN_TAKE_SECONDS and not Confirm.ask(f"Only {playing:.0f}s of playing. Keep this take?", default=False):
        console.print("Discarded.")
        return False
    return True


@dataclass
class Filed:
    """A take just filed, and how it rated."""
    path: Path
    song: str | None  # None: Unsorted
    number: int | None  # among the song's takes, oldest first
    count: int | None
    rated: Rated | None  # None if it wasn't rated (filed without separating the voice)
    earlier: list[Rated]  # the song's other takes rated the same way
    reference: lib.Reference | None


def _save(heard: Heard, choice: Choice, songs: Library, source: str | None = None) -> Filed:
    reference = _reference(songs, choice.song) if choice.song else None
    earlier = songs.takes_of(choice.song, melody=reference is not None) if choice.song else []
    dest = songs.add_take(heard.take, choice.song, heard.recorded, heard.lines, choice.genius_id, choice.artist,
                          heard.metrics, source, heard.melody)
    number = count = None
    if choice.song:  # counted by date, as `play` and `progress` do: an old voice memo can be take 1 of 4
        takes = songs.songs()[choice.song]
        number, count = takes.index(dest) + 1, len(takes)
    console.print(f"[green]✓[/green] Saved [bold]{escape(str(dest.relative_to(songs.root)))}[/bold]"
                  + (f" (take {number} of {count})" if number else ""))
    if choice.song and len(takes) == 1 and not songs.has_reference(choice.song):
        how = "it can find the original on YouTube" if youtube.available() else "give it a recording of the original"
        console.print(f"[dim]To rate your pitch against the original's melody: shed reference "
                      f"{escape(shlex.quote(choice.song))} ({how}).[/dim]")
    rated, others = None, []
    if heard.metrics:
        rated = _rated(heard.metrics, heard.melody, reference)
        # Your best, among the takes rated the same way (on the melody, or on the scale).
        others = [r for t in earlier if (r := _rated(t.metrics, t.melody, reference))
                  and (r.comparison is None) == (rated.comparison is None)]
        _take_report(rated, others, config.load().references(), reference)
    return Filed(dest, choice.song, number, count, rated, others, reference)


def _recognize(lines: list[str], songs: Library, genius_token: str | None,
               candidates: list[genius.Candidate] | None = None,
               ) -> tuple[Choice | None, list[tuple[str, float]], list[genius.Candidate]]:
    """The song, when your earlier takes or Genius make it clear. Never asks: with None comes what asking
    needs, your songs ranked by shared lyrics and Genius's guesses. Genius is only searched when
    `candidates` (from an earlier search) aren't given."""
    if lines:
        heard = " / ".join(lines)
        console.print(f"[dim]Heard: “{escape(heard[:90])}{'…' if len(heard) > 90 else ''}”[/dim]")
    else:
        console.print("[dim]Didn't hear any lyrics.[/dim]")

    ranked = songs.match(lines)
    if lib.is_confident(lines, ranked):
        console.print(f"Recognized [bold]{escape(ranked[0][0])}[/bold] from your earlier takes.")
        return Choice(ranked[0][0]), ranked, []

    if candidates is None:
        candidates = _search_genius(lines, genius_token)
    if genius.is_confident(candidates):
        top = candidates[0]
        console.print(f"Recognized [bold]{escape(top.title)}[/bold] by {escape(top.artist)} on Genius.")
        return _choice_for(top, songs), ranked, candidates
    return None, ranked, candidates


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


def _ask(ranked: list[tuple[str, float]], suggestion: genius.Candidate | None, songs: Library,
         can_skip: bool = False) -> Choice | None:
    """Which song this is, as you answer. None when `can_skip` and you skip it."""
    options = _song_options(ranked, suggestion, songs)
    console.print("[bold]Which song is this?[/bold]")
    if _interactive():
        return _pick_song(options, songs, can_skip)
    for i, (_, name, where) in enumerate(options, 1):
        console.print(f"  {i}. {escape(name)} [dim]({where})[/dim]")
    answer = _input("Number, song name (Tab completes), " + ("Enter for Unsorted, or - to skip it: " if can_skip
                                                            else "or Enter for Unsorted: "), list(songs.songs()))
    if can_skip and answer == "-":
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1][0]
    return _choice_named(answer, songs)


def _pick_song(options: list[tuple[Choice, str, str]], songs: Library, can_skip: bool) -> Choice | None:
    """_ask()'s question, answered with the arrow keys."""
    labels = [name for _, name, _ in options] + ["Another song…", lib.UNSORTED] + (["Skip it"] if can_skip else [])
    notes = [where for _, _, where in options] + ["type its name (Tab completes)", "", "and don't ask again"]
    while True:
        chosen = widgets.pick(console, labels, 0, notes[:len(labels)])
        if chosen < len(options):
            return options[chosen][0]
        if chosen == len(options):  # another song
            if name := _input("Song name (Enter to go back): ", list(songs.songs())):
                return _choice_named(name, songs)
            continue
        return Choice(None) if chosen == len(options) + 1 else None  # Unsorted, or skipped


def _song_options(ranked: list[tuple[str, float]], suggestion: genius.Candidate | None, songs: Library,
                  ) -> list[tuple[Choice, str, str]]:
    """The songs a take may be, as (choice, name, where it's from): Genius's guess, then your songs sharing its
    lyrics."""
    options = []
    if suggestion:
        options.append((_choice_for(suggestion, songs), f"{suggestion.title} — {suggestion.artist}", "Genius"))
    for song, score in ranked[:3]:
        if score > 0.03 and all(choice.song != song for choice, _, _ in options):
            options.append((Choice(song), song, "your songs"))
    return options


def _choice_named(name: str, songs: Library) -> Choice:
    """The song you typed: one of yours however it's typed, a new one, or Unsorted if you typed nothing."""
    return Choice(songs.find_song(name) or lib.folder_name(name) or None)


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
        minutes, seconds = divmod(round(mutagen.File(path).info.length), 60)
    except Exception:
        return "?"
    return f"{minutes}:{seconds:02d}"
