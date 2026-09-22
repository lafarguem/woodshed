"""Settings chosen with `shed init`, kept in ~/.config/woodshed/config.toml.

Command-line flags and environment variables override them.
"""

import json
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from woodshed import rating

PATH = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "woodshed" / "config.toml"
DEFAULT_LIBRARY = "~/Music/Woodshed"


# Earlier versions saved every setting, rating reference points included, so a saved default doesn't mean
# it was chosen: a value that was the default then gives way to today's (as long as that stays consistent).
_FORMER_DEFAULTS = {"pitch_best_cents": 5.0, "pitch_worst_cents": 25.0}
_REFERENCES = ("pitch_best_cents", "pitch_worst_cents", "timing_best_percent", "timing_worst_percent",
               "pitch_weight_percent", "melody_best_cents", "melody_worst_cents")


class Unreadable(SystemExit):
    """config.toml isn't valid TOML. Every command stops with this message, except `shed init`, which starts over."""


@dataclass
class Config:
    library: str = DEFAULT_LIBRARY
    device: str | None = None  # input device name; None is the system default
    genius_token: str | None = None
    # Rating reference points: the value that scores 10, and the one that scores 0.
    pitch_best_cents: float = rating.DEFAULTS.pitch_cents[0]
    pitch_worst_cents: float = rating.DEFAULTS.pitch_cents[1]
    timing_best_percent: float = 100 * rating.DEFAULTS.tempo_spread[0]
    timing_worst_percent: float = 100 * rating.DEFAULTS.tempo_spread[1]
    take_format: str = "mp3"  # or "m4a": Apple Lossless, about 4 times larger
    pitch_weight_percent: float = 100 * rating.DEFAULTS.pitch_weight  # of the overall rating; timing is the rest
    # Pitch against a song's reference melody (`shed reference`), instead of its scale: the same, in cents off it.
    melody_best_cents: float = rating.DEFAULTS.melody_cents[0]
    melody_worst_cents: float = rating.DEFAULTS.melody_cents[1]

    def references(self) -> rating.References:
        if not self.consistent():
            raise SystemExit(f"The rating settings in {PATH} are inconsistent (each 10/10 value must be below its "
                             "0/10 value, and pitch's share between 0 and 100%). Run `shed init` to set them again.")
        return rating.References((float(self.pitch_best_cents), float(self.pitch_worst_cents)),
                                 (float(self.timing_best_percent) / 100, float(self.timing_worst_percent) / 100),
                                 (float(self.melody_best_cents), float(self.melody_worst_cents)),
                                 float(self.pitch_weight_percent) / 100)

    def consistent(self) -> bool:
        return (0 <= float(self.pitch_best_cents) < float(self.pitch_worst_cents)
                and 0 <= float(self.timing_best_percent) < float(self.timing_worst_percent)
                and 0 <= float(self.melody_best_cents) < float(self.melody_worst_cents)
                and 0 <= float(self.pitch_weight_percent) <= 100)


def exists() -> bool:
    return PATH.exists()


def load() -> Config:
    try:
        data = tomllib.loads(PATH.read_text())
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as e:
        raise Unreadable(f"Can't read {PATH}: {e}. Fix it or run `shed init` again.") from None
    known = {f.name for f in fields(Config)}
    saved = Config(**{key: value for key, value in data.items() if key in known})
    today = Config(**{key: value for key, value in data.items() if key in known and _FORMER_DEFAULTS.get(key) != value})
    return today if today.consistent() else saved


def save(config: Config) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.touch(mode=0o600, exist_ok=True)
    PATH.chmod(0o600)  # it holds your Genius token
    defaults = Config()
    # Reference points left at their defaults aren't saved, so that better defaults reach you.
    values = {f.name: getattr(config, f.name) for f in fields(Config)
              if not (f.name in _REFERENCES and getattr(config, f.name) == getattr(defaults, f.name))}
    # A JSON string is also a valid TOML string, as long as characters like 🎸 are written as they are:
    # JSON would escape them as surrogate pairs (\ud83c\udfb8), which TOML refuses.
    PATH.write_text("".join(f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
                            for key, value in values.items() if value is not None))
