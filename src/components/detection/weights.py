"""Resolve named checkpoints so a missing file cannot trigger a silent download.

Ultralytics will auto-download a weight into the process working directory when
the configured path is a dangling symlink. That is how seven stale links under
``assets/weights/`` once populated the repo root. Callers pass the resolved
absolute path into the loader, never the bare filename.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.contracts.config import PointstreamConfig
from src.contracts.errors import ConfigValueError

#: Config slots whose ``model`` field names a checkpoint file.
WEIGHT_SLOTS: tuple[str, ...] = ("detector", "pose", "segmenter")


class WeightResolutionError(FileNotFoundError):
    """A named checkpoint is missing, or the path is a dangling symlink."""


def repo_root() -> Path:
    """Repository root for this worktree (the directory that contains ``src/``)."""
    return Path(__file__).resolve().parents[3]


def weights_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "assets" / "weights"


def named_weights(config: PointstreamConfig) -> dict[str, str]:
    """Every checkpoint a config actually names, keyed by slot.

    Empty / ``none`` backends are skipped: they load nothing. A backend that is
    named but has no ``model`` is also skipped — some backends (heuristic
    selection, the tracker) have no weights.
    """
    found: dict[str, str] = {}
    for slot in WEIGHT_SLOTS:
        section = getattr(config, slot)
        backend = getattr(section, "backend", "none")
        model = getattr(section, "model", None)
        if backend in ("", "none") or not model:
            continue
        found[slot] = str(model)
    return found


def resolve_weight(name: str, *, root: Path | None = None) -> Path:
    """Return a real, existing path for `name`.

    `name` may be an absolute/relative path, or a filename under
    ``assets/weights/``. A dangling symlink is an error of its own — ``exists()``
    is false for those, and treating them as "missing" is how the auto-download
    trap used to fire.

    Raises:
        WeightResolutionError: If the file is absent or the symlink is dangling.
    """
    if not name or name.strip() in {"none", "-"}:
        raise WeightResolutionError(f"No weight name given ({name!r}).")

    candidate = Path(name)
    if not candidate.is_absolute():
        planted = weights_dir(root) / name
        if _is_present(planted) or planted.is_symlink():
            candidate = planted
        elif candidate.exists() or candidate.is_symlink():
            candidate = candidate
        else:
            candidate = planted

    if candidate.is_symlink() and not candidate.exists():
        target = candidate.resolve()
        raise WeightResolutionError(
            f"Weight {name!r} is a dangling symlink at {candidate} -> {target}. "
            f"A dangling link used to make ultralytics auto-download a replacement "
            f"into the process working directory. Restore the target or replace the link."
        )
    if not candidate.exists() or not candidate.is_file():
        raise WeightResolutionError(
            f"Weight {name!r} not found at {candidate}. "
            f"Place the file under {weights_dir(root)} (or pass an existing path). "
            f"Auto-download is disabled so a missing file cannot populate the repo root."
        )
    return candidate.resolve()


def assert_named_weights_resolve(
    config: PointstreamConfig, *, root: Path | None = None, path: str = "config"
) -> None:
    """Raise unless every checkpoint `config` names resolves.

    Unit-testable against a temp tree by passing `root`. Also the check to run
    over shipped config names (``PointstreamConfig()`` defaults).
    """
    problems: list[str] = []
    for slot, name in named_weights(config).items():
        try:
            resolve_weight(name, root=root)
        except WeightResolutionError as exc:
            problems.append(f"{slot}.model={name!r}: {exc}")
    if problems:
        joined = "; ".join(problems)
        raise ConfigValueError(path, f"named weight(s) do not resolve: {joined}")


def _is_present(path: Path) -> bool:
    return path.exists() and (path.is_file() or path.is_dir())


def describe_named_weights(config: PointstreamConfig) -> Mapping[str, str]:
    """Slot to filename, for listings and tests."""
    return dict(named_weights(config))
