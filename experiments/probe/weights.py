"""Every weight a shipped config names must resolve to a real file or directory.

A dangling symlink is a failure of its own: that is how ultralytics once
auto-downloaded replacements into the repo root.

Resolution splits in two. Structural checks (well-formed names, no doubled
``assets/weights/`` prefix, known slots, YOLO names a rule can produce) do
not need files on disk. Existence checks do, and belong only where the
weights tree is present.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.components.detection.weights import (
    intended_weight_path,
    named_weights,
    repo_root,
    resolve_weight,
    weights_dir,
)
from src.contracts.config import PointstreamConfig, default

_WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
_SKIP_VALUES = {"", "none", "null", "auto", "-", "false", "true"}
_WEIGHTS_POSIX_PREFIX = "assets/weights/"

# Ultralytics-style names we actually ship: yolo26n.pt, yolo26x-seg.pt,
# yoloe-26n-seg.pt. ``yolo26x-eg.pt`` is a typo for ``yolo26x-seg.pt`` and
# matches no production rule.
_YOLO_NAME = re.compile(
    r"^yolo(?:e-)?\d+[nslmx](?:-(?:pose|seg|cls|obb))?\.pt$",
    re.IGNORECASE,
)

# Last path component of a slot that is allowed to name a weight. Unknown
# keys that happen to look like a checkpoint are a config fault, not a
# discovery.
_KNOWN_SLOT_TAILS = frozenset(
    {
        "detector",
        "pose",
        "pose-estimator",
        "segmenter",
        "ball-det-model",
        "controlnet-id",
        "postgen-segmenter-model",
        "animate-anyone-model-dir",
        "checkpoint",
    }
)


@dataclass(frozen=True)
class NamedWeight:
    """One checkpoint a config actually names."""

    source: str
    slot: str
    name: str
    path: Path | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.error is None


def shipped_yaml_paths(root: Path | None = None) -> list[Path]:
    """``config/**/*.yaml`` under the worktree, including ``config/benchmarks/``."""
    config_dir = (root or repo_root()) / "config"
    if not config_dir.is_dir():
        return []
    return sorted(path for path in config_dir.rglob("*.yaml") if path.is_file())


def weights_are_present(root: Path | None = None) -> bool:
    """True when the weights tree exists. An empty tree still counts as present."""
    return weights_dir(root).is_dir()


def _posix_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def already_under_weights(name: str) -> bool:
    """True when `name` already carries the ``assets/weights/`` prefix."""
    posix = _posix_name(name)
    return posix.startswith(_WEIGHTS_POSIX_PREFIX) or f"/{_WEIGHTS_POSIX_PREFIX}" in f"/{posix}"


def _slot_tail(slot: str) -> str:
    return slot.split(".")[-1].replace("_", "-")


def is_weight_slot(slot: str) -> bool:
    return _slot_tail(slot) in _KNOWN_SLOT_TAILS


def structural_error(name: str, slot: str = "") -> str | None:
    """Why this name is not resolvable in principle. Does not touch the disk.

    Catches the two config faults that used to hide behind a runner-only skip:
    a YOLO name no production rule produces (``yolo26x-eg.pt``), and a value
    that already contains ``assets/weights/`` so the resolver would plant a
    doubled prefix.
    """
    if already_under_weights(name):
        return (
            f"{name!r} already contains {_WEIGHTS_POSIX_PREFIX}; "
            "use a bare name so the resolver does not plant a doubled prefix"
        )
    if slot and not is_weight_slot(slot):
        return f"weight named under unknown config key {slot!r}"
    base = Path(_posix_name(name)).name
    if base.lower().startswith("yolo") and base.lower().endswith(".pt"):
        if _YOLO_NAME.match(base) is None:
            return f"{base!r} matches no YOLO weight naming rule"
    return None


def resolve_named_artifact(name: str, *, root: Path | None = None) -> Path:
    """A file (via ``resolve_weight``) or an existing directory under weights/."""
    base = root or repo_root()
    candidate = Path(name)
    if not candidate.is_absolute():
        planted = intended_weight_path(name, root=base)
        repo_relative = base / name
        if planted.exists() or planted.is_symlink():
            candidate = planted
        elif repo_relative.exists() or repo_relative.is_symlink():
            candidate = repo_relative
        elif candidate.exists() or candidate.is_symlink():
            candidate = candidate
        else:
            candidate = planted

    if candidate.is_symlink() and not candidate.exists():
        target = candidate.resolve()
        raise FileNotFoundError(
            f"Weight {name!r} is a dangling symlink at {candidate} -> {target}."
        )
    if candidate.is_dir():
        return candidate.resolve()
    return resolve_weight(name, root=base)


def _looks_like_weight(value: str) -> bool:
    stripped = value.strip()
    if stripped.lower() in _SKIP_VALUES:
        return False
    lower = stripped.lower()
    if lower.endswith(_WEIGHT_SUFFIXES):
        return True
    return _WEIGHTS_POSIX_PREFIX in lower.replace("\\", "/")


def _walk(node: Any, prefix: str) -> Iterator[tuple[str, str]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            slot = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(value, slot)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{prefix}[{index}]")
        return
    if not isinstance(node, str):
        return
    if node.strip().lower() in _SKIP_VALUES:
        return
    if _looks_like_weight(node) or is_weight_slot(prefix):
        yield prefix, node


def _raw_names_from_yaml(path: Path) -> tuple[tuple[str, str], ...]:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        return ()
    return tuple(_walk(data, ""))


def _structural_item(source: str, slot: str, name: str, *, root: Path | None = None) -> NamedWeight:
    err = structural_error(name, slot)
    if err:
        return NamedWeight(source=source, slot=slot, name=name, path=None, error=err)
    return NamedWeight(
        source=source,
        slot=slot,
        name=name,
        path=intended_weight_path(name, root=root),
        error=None,
    )


def names_from_yaml(path: Path) -> tuple[NamedWeight, ...]:
    found: list[NamedWeight] = []
    rel = str(path.as_posix())
    for slot, name in _raw_names_from_yaml(path):
        try:
            resolved = resolve_named_artifact(name)
            found.append(NamedWeight(source=rel, slot=slot, name=name, path=resolved))
        except (FileNotFoundError, OSError) as exc:
            found.append(
                NamedWeight(source=rel, slot=slot, name=name, path=None, error=str(exc))
            )
    return tuple(found)


def structural_names_from_yaml(path: Path, *, root: Path | None = None) -> tuple[NamedWeight, ...]:
    """Named weights in a YAML file, checked for form only — no disk reads."""
    rel = str(path.as_posix())
    return tuple(
        _structural_item(rel, slot, name, root=root) for slot, name in _raw_names_from_yaml(path)
    )


def names_from_default_config() -> tuple[NamedWeight, ...]:
    """The nested ``PointstreamConfig`` default, which YAML has not all adopted."""
    config: PointstreamConfig = default()
    found: list[NamedWeight] = []
    for slot, name in named_weights(config).items():
        try:
            resolved = resolve_named_artifact(name)
            found.append(
                NamedWeight(source="PointstreamConfig.default()", slot=slot, name=name, path=resolved)
            )
        except (FileNotFoundError, OSError) as exc:
            found.append(
                NamedWeight(
                    source="PointstreamConfig.default()",
                    slot=slot,
                    name=name,
                    path=None,
                    error=str(exc),
                )
            )
    checkpoint = config.generator.checkpoint
    if checkpoint:
        try:
            resolved = resolve_named_artifact(checkpoint)
            found.append(
                NamedWeight(
                    source="PointstreamConfig.default()",
                    slot="generator.checkpoint",
                    name=checkpoint,
                    path=resolved,
                )
            )
        except (FileNotFoundError, OSError) as exc:
            found.append(
                NamedWeight(
                    source="PointstreamConfig.default()",
                    slot="generator.checkpoint",
                    name=checkpoint,
                    path=None,
                    error=str(exc),
                )
            )
    return tuple(found)


def structural_names_from_default_config(*, root: Path | None = None) -> tuple[NamedWeight, ...]:
    config: PointstreamConfig = default()
    items = [
        _structural_item("PointstreamConfig.default()", slot, name, root=root)
        for slot, name in named_weights(config).items()
    ]
    checkpoint = config.generator.checkpoint
    if checkpoint:
        items.append(
            _structural_item(
                "PointstreamConfig.default()", "generator.checkpoint", checkpoint, root=root
            )
        )
    return tuple(items)


def collect_shipped_weights(root: Path | None = None) -> tuple[NamedWeight, ...]:
    """Default nested config plus every YAML under ``config/``."""
    items = list(names_from_default_config())
    for path in shipped_yaml_paths(root):
        items.extend(names_from_yaml(path))
    return tuple(items)


def collect_structural_weights(root: Path | None = None) -> tuple[NamedWeight, ...]:
    """Shipped names checked for form only. Safe to run without weights on disk."""
    items = list(structural_names_from_default_config(root=root))
    for path in shipped_yaml_paths(root):
        items.extend(structural_names_from_yaml(path, root=root))
    return tuple(items)


def unresolved(items: tuple[NamedWeight, ...] | None = None) -> tuple[NamedWeight, ...]:
    found = items if items is not None else collect_shipped_weights()
    return tuple(item for item in found if not item.ok)
