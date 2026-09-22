"""Arrow-key prompts: a list to pick from, and a slider."""

import contextlib
from collections.abc import Callable, Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from woodshed import terminal

_SLIDER_WIDTH = 36


@contextlib.contextmanager
def _keys(scripted: Iterator[str] | None) -> Iterator[Callable[[], str]]:
    if scripted is None:
        with terminal.keys() as next_key:
            yield next_key
    else:  # tests feed key presses
        yield lambda: next(scripted)


def pick(console: Console, options: list[str], default: int = 0, notes: list[str] | None = None,
         keys: Iterator[str] | None = None, details: list[str] | None = None) -> int:
    """Let the user choose an option with ↑/↓ and Enter. Returns its index. `notes` go next to the options,
    `details` on a line of their own under each, cut to the terminal's width."""
    notes = notes or [""] * len(options)
    details = details or [""] * len(options)
    width = max(map(len, options))
    index = default

    def render() -> Group:
        rows = []
        for i, (option, note, detail) in enumerate(zip(options, notes, details)):
            rows.append(Text.assemble(("  ❯ ", "bold cyan") if i == index else "    ",
                                      (option.ljust(width), "bold" if i == index else ""), ("  " + note, "dim")))
            if detail:
                rows.append(Text("      " + detail, "dim", no_wrap=True, overflow="ellipsis"))
        return Group(*rows, Text("  ↑/↓ to move, Enter to choose", "dim"))

    with _keys(keys) as next_key, Live(render(), console=console, transient=True, auto_refresh=False) as live:
        while (key := next_key()) != "enter":
            if key in ("up", "k"):
                index = (index - 1) % len(options)
            elif key in ("down", "j"):
                index = (index + 1) % len(options)
            live.update(render(), refresh=True)
    console.print(Text.assemble("  ", (options[index], "bold cyan")))
    return index


def slide(console: Console, label: str, value: float, low: float, high: float, step: float,
          show: Callable[[float], str], preview: Callable[[float], str] | None = None,
          keys: Iterator[str] | None = None) -> float:
    """Let the user set a number with ←/→ (↑/↓ for bigger steps) and Enter."""
    value = min(max(value, low), high)

    def render() -> Group:
        filled = round((value - low) / (high - low) * _SLIDER_WIDTH) if high > low else 0
        bar = Text.assemble(("  ◀ ", "dim"), ("━" * filled, "cyan"), ("●", "bold cyan"),
                            ("─" * (_SLIDER_WIDTH - filled), "dim"), (" ▶  ", "dim"), (show(value), "bold"))
        lines = [Text(label), bar]
        if preview:
            lines.append(Text("  " + preview(value), "dim"))
        lines.append(Text("  ←/→ to adjust (↑/↓ for bigger steps), Enter to confirm", "dim"))
        return Group(*lines)

    with _keys(keys) as next_key, Live(render(), console=console, transient=True, auto_refresh=False) as live:
        while (key := next_key()) != "enter":
            change = {"left": -1, "h": -1, "right": 1, "l": 1, "down": -5, "up": 5}.get(key, 0)
            value = min(max(round((value + change * step) / step) * step, low), high)
            live.update(render(), refresh=True)
    console.print(Text.assemble(f"{label} ", (show(value), "bold cyan")))
    return float(value)
