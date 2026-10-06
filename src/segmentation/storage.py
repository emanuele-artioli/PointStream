"""Where weights and datasets live: the ``Models`` and ``Datasets`` links.

The repository root holds two gitignored symlinks, ``Models`` ->
``/home/itec/emanuele/Models`` and ``Datasets`` -> ``/home/itec/emanuele/Datasets``.
Code reaches every weight and dataset through them. A fleet snapshot carries no
links, so a missing link falls back to the same canonical directory. An
environment variable overrides either root, for tests and other machines.

Nothing here downloads. A missing weight is an error naming the expected path.
"""

from __future__ import annotations

import os
from pathlib import Path

#: This file is ``<repo>/src/segmentation/storage.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_MODELS = Path("/home/itec/emanuele/Models")
CANONICAL_DATASETS = Path("/home/itec/emanuele/Datasets")


def _root(env: str, link: str, canonical: Path) -> Path:
    override = os.environ.get(env, "").strip()
    if override:
        return Path(override).expanduser()
    linked = REPO_ROOT / link
    return linked if linked.exists() else canonical


def models_root() -> Path:
    """``PS_MODELS_ROOT``, else the ``Models`` link, else the canonical directory."""
    return _root("PS_MODELS_ROOT", "Models", CANONICAL_MODELS)


def datasets_root() -> Path:
    """``PS_DATASETS_ROOT``, else the ``Datasets`` link, else the canonical directory."""
    return _root("PS_DATASETS_ROOT", "Datasets", CANONICAL_DATASETS)


def model_path(family: str, name: str) -> Path:
    """The existing weight ``Models/<family>/<name>``, or an error naming that path.

    A dangling symlink counts as missing: Ultralytics treats a path that does not
    exist as a request to download into the working directory.
    """
    path = models_root() / family / name
    if not path.is_file():
        raise FileNotFoundError(f"weight {name!r} is not at {path}")
    return path
