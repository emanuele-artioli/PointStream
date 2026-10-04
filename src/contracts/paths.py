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
`PS_DATA_ROOT` names that place. A checkout marker takes precedence over an
existing ~/Datasets root; otherwise the historical repository layout remains
available. Model sources and checkpoints have a separate Models root.

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
#: `outputs/`. Unset means "consult the marker file", then "the repository root".
ENV_DATA_ROOT: Final = "PS_DATA_ROOT"
ENV_MODELS_ROOT: Final = "PS_MODELS_ROOT"
MODELS_ROOT_MARKER: Final = ".ps-models-root"

#: Per-checkout marker naming the data root, one line, no quoting.
#:
#: The environment variable alone is a footgun: it has to be exported in every
#: shell, every editor terminal, every cron entry and every agent session, and
#: the failure mode when it is missing is a confusing "file not found" rather
#: than a clear one. The marker file travels with the checkout instead, so a
#: process that inherits nothing still finds the data. It is gitignored, because
#: where the data sits is a property of the machine and not of the branch.
DATA_ROOT_MARKER: Final = ".ps-data-root"

#: This file is `<repo>/src/contracts/paths.py`, so the root is three up.
_REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """The checkout itself. Code, configs and plans — never data."""
    return _REPO_ROOT


def data_root() -> Path:
    """Where `assets/` and `outputs/` live.

    In order: `PS_DATA_ROOT` if set and non-empty; then a `.ps-data-root` marker
    file in the repository root; then an existing ~/Datasets; then the repository
    root itself for the historical development layout.

    The path is returned whether or not it exists. A caller that needs a
    directory to be present should say so itself, with a message naming what it
    was looking for, rather than being handed a silent fallback here.
    """
    override = os.environ.get(ENV_DATA_ROOT, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    marker = _REPO_ROOT / DATA_ROOT_MARKER
    try:
        declared = marker.read_text(encoding="utf-8").strip()
    except OSError:
        declared = ""
    if declared:
        return Path(declared).expanduser().resolve()

    canonical = Path.home() / "Datasets"
    return canonical.resolve() if canonical.is_dir() else _REPO_ROOT


def assets() -> Path:
    """The dataset tree: source video, extracted frames and probe sets."""
    return data_root() / "assets"


def outputs() -> Path:
    """The experiment tree: every run's artifacts and result files."""
    return data_root() / "outputs"


def models_root() -> Path:
    """Model source and checkpoints; explicit configuration never falls back.

    POINTSTREAM_MODELS remains an alias for the demo's existing configuration.
    The old weights tree is supported until the canonical storage cutover.
    """
    override = (
        os.environ.get(ENV_MODELS_ROOT, "").strip()
        or os.environ.get("POINTSTREAM_MODELS", "").strip()
    )
    if override:
        return Path(override).expanduser().resolve()
    try:
        declared = (_REPO_ROOT / MODELS_ROOT_MARKER).read_text().strip()
    except OSError:
        declared = ""
    if declared:
        return Path(declared).expanduser().resolve()
    data = data_root()
    if data.name == "Datasets":
        return data.parent / "Models"
    canonical = Path.home() / "Models"
    return canonical.resolve() if canonical.is_dir() else assets() / "weights"


def model_asset(name: str | Path, *, legacy: str | Path | None = None) -> Path:
    """Locate a model without downloading it or substituting an explicit root.

    Accept historical assets/weights names, direct model-family paths, and
    absolute caller-supplied paths. Canonical files take precedence; a retained
    legacy file is used only when no model root has been explicitly configured.
    """
    raw = Path(name)
    if raw.is_absolute():
        return raw
    text = raw.as_posix()
    for prefix in ("assets/weights/", "weights/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    relative = Path(text)
    if ".." in relative.parts:
        raise ValueError("model asset must stay inside its root")
    root = models_root()
    candidates = [root / relative]
    if len(relative.parts) == 1 and root != assets() / "weights":
        family = model_family(relative.name)
        if family:
            candidates.insert(0, root / family / relative.name)
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return candidate
    try:
        declared = (_REPO_ROOT / MODELS_ROOT_MARKER).read_text().strip()
    except OSError:
        declared = ""
    explicit = bool(
        os.environ.get(ENV_MODELS_ROOT, "").strip()
        or os.environ.get("POINTSTREAM_MODELS", "").strip()
        or declared
    )
    if not explicit:
        historical = [assets() / "weights" / relative, data_root() / "weights" / relative]
        if legacy is not None:
            old = Path(legacy)
            historical.insert(0, old if old.is_absolute() else data_root() / old)
        for candidate in historical:
            if candidate.exists() or candidate.is_symlink():
                return candidate
    return candidates[0]


def model_family(name: str) -> str | None:
    """Stable family names shared by loaders and the storage migration."""
    lower = name.lower()
    for prefix, family in (
        ("yolo", "YOLO"),
        ("mobileclip", "YOLO"),
        ("sam", "SAM"),
        ("fastsam", "SAM"),
        ("pix2pix", "pix2pix"),
        ("spade4tennis", "spade4tennis"),
        ("vgg", "vgg19"),
        ("i3d", "i3d"),
    ):
        if lower.startswith(prefix):
            return family
    return None


def _source() -> str:
    """Which of the three mechanisms decided, so a run record can say."""
    if os.environ.get(ENV_DATA_ROOT, "").strip():
        return ENV_DATA_ROOT
    marker = _REPO_ROOT / DATA_ROOT_MARKER
    try:
        if marker.read_text(encoding="utf-8").strip():
            return f"{DATA_ROOT_MARKER} marker file"
    except OSError:
        pass
    return "~/Datasets" if (Path.home() / "Datasets").is_dir() else "repo root (legacy default)"


def describe() -> dict[str, str]:
    """What the paths resolved to, for a run record to carry.

    A result that cites `outputs/bp24-ladder/...` is ambiguous once the data can
    live outside the checkout, so a run that records its paths should record
    what they resolved to as well.
    """
    return {
        "repo_root": str(repo_root()),
        "data_root": str(data_root()),
        "data_root_source": _source(),
        "assets": str(assets()),
        "outputs": str(outputs()),
        "models_root": str(models_root()),
    }


__all__ = [
    "DATA_ROOT_MARKER",
    "ENV_DATA_ROOT",
    "ENV_MODELS_ROOT",
    "MODELS_ROOT_MARKER",
    "assets",
    "data_root",
    "describe",
    "outputs",
    "models_root",
    "model_asset",
    "model_family",
    "repo_root",
]
