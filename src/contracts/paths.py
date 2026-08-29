"""Where the data lives, which is not necessarily where the code lives.

`assets/` and `outputs/` hold roughly 565,000 files against the ~700 the
repository actually tracks. Both are gitignored, which stops git tracking them
and does nothing about a tool that walks the filesystem — and this home
directory is an NFS mount serving on the order of ten milliseconds per file
open. An editor asked to index the project therefore walks half a million files
at that rate and never finishes; measured on this host, VS Code's Source Control
view sits at "scanning folder for git repositories" indefinitely, and so does
anything waiting on it.

The fix is to let the data live somewhere the code tree does not contain.
`PS_DATA_ROOT` names that place. It defaults to the repository root, so a
checkout with its data still in place behaves exactly as before and nothing has
to be moved for the code to keep working.

**Why not a symlink.** A symlink inside the project is what tools follow, and it
is how one dataset became twelve: each git worktree carried `assets` and
`outputs` symlinks back to the same directories, so repository auto-detection
could find the same half-million files once per worktree. Point the environment
variable at the data instead and leave nothing in the tree to follow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

#: Environment variable naming the directory that holds `assets/` and
#: `outputs/`. Unset means "the repository root", which is the historical layout.
ENV_DATA_ROOT: Final = "PS_DATA_ROOT"

#: This file is `<repo>/src/contracts/paths.py`, so the root is three up.
_REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """The checkout itself. Code, configs and plans — never data."""
    return _REPO_ROOT


def data_root() -> Path:
    """Where `assets/` and `outputs/` live.

    `PS_DATA_ROOT` if it is set and non-empty, otherwise the repository root.
    The path is returned whether or not it exists: a caller that needs a
    directory to be present should say so itself, with a message naming what it
    was looking for, rather than being handed a silent fallback here.
    """
    override = os.environ.get(ENV_DATA_ROOT, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT


def assets() -> Path:
    """The dataset tree: source video, extracted frames, probe sets, weights."""
    return data_root() / "assets"


def outputs() -> Path:
    """The experiment tree: every run's artifacts and result files."""
    return data_root() / "outputs"


def describe() -> dict[str, str]:
    """What the paths resolved to, for a run record to carry.

    A result that cites `outputs/bp24-ladder/...` is ambiguous once the data can
    live outside the checkout, so a run that records its paths should record
    what they resolved to as well.
    """
    return {
        "repo_root": str(repo_root()),
        "data_root": str(data_root()),
        "data_root_source": ENV_DATA_ROOT if os.environ.get(ENV_DATA_ROOT, "").strip() else "repo root (default)",
        "assets": str(assets()),
        "outputs": str(outputs()),
    }


__all__ = [
    "ENV_DATA_ROOT",
    "assets",
    "data_root",
    "describe",
    "outputs",
    "repo_root",
]
