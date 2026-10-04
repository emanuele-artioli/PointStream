"""Read-only Git invocations for provenance on shared checkouts.

Plain ``git status`` and ``git diff`` may refresh stale index stat data and
rewrite ``.git/index`` under ``index.lock``. A stale index is the normal case
when another host last touched the checkout. On the NFS-shared model checkouts
that rewrite is a network write. On gpu5 (4 October 2026) it made both
commands about 15x slower than the lock-free form (1.2 s against 0.08 s). It
is also the most plausible source of the 30-second provenance timeout that
blocked the background smoke, though that stall was not reproduced. Provenance
must also never mutate the tree it describes. ``--no-optional-locks`` plus
``GIT_OPTIONAL_LOCKS=0`` stops ``status`` from writing. Porcelain ``diff`` can
still refresh racily-clean entries, so use plumbing ``diff-index -p HEAD --``
for patches. It compares content without touching the index and prints the
same patch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def readonly_git_command(root: Path | str, *args: str) -> list[str]:
    """``git -C root <args>`` that never takes optional locks."""
    return ["git", "--no-optional-locks", "-C", str(root), *args]


def readonly_git_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """`base` (default: this process) with optional Git locks disabled."""
    env = dict(os.environ if base is None else base)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def readonly_git_diff_command(root: Path | str) -> list[str]:
    """Tracked changes against HEAD as a patch, without refreshing the index."""
    return readonly_git_command(root, "diff-index", "-p", "--no-color", "HEAD", "--")
