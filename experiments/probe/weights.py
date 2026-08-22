"""Every weight a shipped config names must resolve to a real file or directory.

A dangling symlink is a failure of its own: that is how ultralytics once
auto-downloaded replacements into the repo root.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.components.detection.weights import named_weights, repo_root, resolve_weight, weights_dir
from src.contracts.config import PointstreamConfig, default

_WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
_SKIP_VALUES = {"", "none", "null", "auto", "-", "false", "true"}


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


def resolve_named_artifact(name: str, *, root: Path | None = None) -> Path:
    """A file (via ``resolve_weight``) or an existing directory under weights/."""
    base = root or repo_root()
    candidate = Path(name)
    if not candidate.is_absolute():
        planted = weights_dir(base) / name
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
    return "assets/weights/" in lower.replace("\\", "/")


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
    if isinstance(node, str) and _looks_like_weight(node):
        yield prefix, node


def names_from_yaml(path: Path) -> tuple[NamedWeight, ...]:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        return ()
    found: list[NamedWeight] = []
    rel = str(path.as_posix())
    for slot, name in _walk(data, ""):
        try:
            resolved = resolve_named_artifact(name)
            found.append(NamedWeight(source=rel, slot=slot, name=name, path=resolved))
        except (FileNotFoundError, OSError) as exc:
            found.append(
                NamedWeight(source=rel, slot=slot, name=name, path=None, error=str(exc))
            )
    return tuple(found)


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


def collect_shipped_weights(root: Path | None = None) -> tuple[NamedWeight, ...]:
    """Default nested config plus every YAML under ``config/``."""
    items = list(names_from_default_config())
    for path in shipped_yaml_paths(root):
        items.extend(names_from_yaml(path))
    return tuple(items)


def unresolved(items: tuple[NamedWeight, ...] | None = None) -> tuple[NamedWeight, ...]:
    found = items if items is not None else collect_shipped_weights()
    return tuple(item for item in found if not item.ok)
