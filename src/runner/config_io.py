"""Read a configuration file into a validated `PointstreamConfig`.

Kept in the runner rather than in contracts because contracts imports nothing
it does not have to, and PyYAML is a dependency contracts does not need in
order to validate a mapping someone else parsed.

There is one loader and it validates. A "just parse it" variant would be a
second door into the same room, and the door without the lock is the one that
gets used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.contracts.config import PointstreamConfig, load
from src.contracts.errors import ConfigValueError

#: Where the shipped configs live, relative to the repository root.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def read_mapping(path: str | Path) -> dict[str, Any]:
    """The raw document, before any schema is applied.

    Separate from `load_config_file` so a tool can report *every* unknown key
    in a file (`contracts.parsing.require_known_keys`) without the build
    stopping at the first bad value.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ConfigValueError(
            str(source),
            f"unsupported config extension {suffix!r}; expected .yaml, .yml or .json",
        )
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigValueError(
            str(source),
            f"a config file must be a mapping at the top level; got {type(data).__name__}",
        )
    return dict(data)


def load_config_file(path: str | Path) -> PointstreamConfig:
    """Parse and validate one config file.

    Raises:
        ConfigError: Collecting every structural and contractual problem, so a
            file with five mistakes reports five.
    """
    return load(read_mapping(path))


def load_tier(name: str) -> PointstreamConfig:
    """One of the shipped tier configs, by short name (`fast`, `balanced`, ...).

    Accepts either `fast` or `tier_fast`, because both spellings turn up in
    briefs and neither is worth arguing about.
    """
    stem = name if name.startswith("tier_") else f"tier_{name}"
    return load_config_file(CONFIG_DIR / f"{stem}.yaml")


__all__ = ["CONFIG_DIR", "load_config_file", "load_tier", "read_mapping"]
