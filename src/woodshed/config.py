"""Settings chosen with `shed init`, kept in ~/.config/woodshed/config.toml.

Command-line flags and environment variables override them.
"""

import json
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

PATH = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "woodshed" / "config.toml"
DEFAULT_LIBRARY = "~/Music/Woodshed"


@dataclass
class Config:
    library: str = DEFAULT_LIBRARY
    device: str | None = None  # input device name; None is the system default
    genius_token: str | None = None


def exists() -> bool:
    return PATH.exists()


def load() -> Config:
    try:
        data = tomllib.loads(PATH.read_text())
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Can't read {PATH}: {e}. Fix it or run `shed init` again.") from None
    known = {f.name for f in fields(Config)}
    return Config(**{key: value for key, value in data.items() if key in known})


def save(config: Config) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.touch(mode=0o600, exist_ok=True)
    PATH.chmod(0o600)  # it holds your Genius token
    values = {f.name: getattr(config, f.name) for f in fields(Config)}
    # A JSON string is also a valid TOML string.
    PATH.write_text("".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items() if value is not None))
