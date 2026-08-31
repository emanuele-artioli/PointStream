"""Retarget an existing probe-set view's symlinks, without rebuilding it.

**Why this exists rather than a rebuild.** `regenerate` walks
`assets/dataset/` to reselect clips, and that tree is on the order of half a
million files on an NFS home serving ~10 ms per open — so a rebuild reads as a
hang, which is exactly what happened when one was attempted. The view itself is
3,033 links over 3,132 inodes. Repairing it in place is seconds, and it changes
nothing about *which* clips were selected — which matters, because reselecting
would silently move the probe set out from under every result measured on it.

**What broke.** The links were written absolute (`<repo>/assets/dataset/...`)
by a `_symlink` that called `src.resolve()`. The 2026-08-29 move of `assets/`
and `outputs/` out of the checkout dangled all of them at once. `materialize.py`
now writes relative links; this repairs the views built before it did.

**What this does not do.** It does not invent a target. A link whose repaired
path still does not exist is reported, not silently rewritten — a link pointing
confidently at nothing is worse than one that is obviously broken.

Run: ``python -m experiments.probe_set repair-links``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.contracts import paths as ps_paths


@dataclass
class RepairReport:
    """What the pass found and what it changed."""

    scanned: int = 0
    already_valid: int = 0
    repaired: int = 0
    unresolved: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unresolved

    def summary(self) -> str:
        return (
            f"{self.scanned} links scanned, {self.already_valid} already valid, "
            f"{self.repaired} repaired, {len(self.unresolved)} unresolved"
        )


def _retarget(target: Path, data_root: Path) -> Path | None:
    """Map an old absolute target onto ``data_root``.

    The old layout put `assets/` inside the checkout. Any recorded path
    containing an `assets/` component can be rebased by keeping the part from
    `assets/` onward and re-rooting it — which is a rename of the *prefix*, not
    a guess about the file.
    """
    parts = target.parts
    if "assets" not in parts:
        return None
    tail = parts[len(parts) - 1 - list(reversed(parts)).index("assets") :]
    return data_root.joinpath(*tail)


def repair(root: Path | None = None, *, data_root: Path | None = None, apply: bool = True) -> RepairReport:
    """Point every dangling link in ``root`` back at the data, relatively.

    Args:
        root: The probe-set view. Defaults to ``<data>/assets/probe_set``.
        data_root: Where ``assets/`` now lives. Defaults to the resolved root.
        apply: When False, report what would change and touch nothing.
    """
    view = root if root is not None else ps_paths.assets() / "probe_set"
    base = data_root if data_root is not None else ps_paths.data_root()
    report = RepairReport()
    if not view.exists():
        raise FileNotFoundError(f"no probe-set view at {view}")

    for path in sorted(view.rglob("*")):
        if not path.is_symlink():
            continue
        report.scanned += 1
        if path.exists():
            report.already_valid += 1
            continue
        candidate = _retarget(Path(os.readlink(path)), base)
        if candidate is None or not candidate.exists():
            report.unresolved.append(f"{path.relative_to(view)} -> {os.readlink(path)}")
            continue
        if apply:
            relative = Path(os.path.relpath(candidate, path.parent))
            path.unlink()
            path.symlink_to(relative)
        report.repaired += 1
    return report
