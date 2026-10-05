"""Foreground/background segmentation: PointStream's first stage.

A domain names its foreground classes (``domains.yaml``); a backend turns a clip
into lossless per-instance masks (`ClipMasks`); everything unlabelled is
background. SAM 3.1 run offline is the reference the faster backends are scored
against (`src.segmentation.evaluate`).

    from src.segmentation import build, load_domain
    masks = build("yoloe-26s").segment("clip.mp4", load_domain("tennis"))
    fg = masks.foreground(0)          # bool HxW

Backends load lazily: importing this package needs only numpy and PyYAML, and
`src.segmentation.sam31` additionally imports inside the separate SAM 3.1 env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.segmentation.masks import ClipMasks, Instance

DOMAINS_YAML = Path(__file__).with_name("domains.yaml")
YOLOE_SIZES = ("n", "s", "m", "l", "x")
BACKENDS = ("sam31", *(f"yoloe-26{size}" for size in YOLOE_SIZES))
REFERENCE_BACKEND = "sam31"


@dataclass(frozen=True)
class Domain:
    """Foreground classes of one domain, with per-backend prompts and options."""

    name: str
    classes: tuple[str, ...]
    prompts: dict[str, str]
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)
    clips: tuple[str, ...] = ()
    summary: str = ""

    def prompts_for(self, family: str) -> dict[str, str]:
        override = self.backends.get(family, {}).get("prompts") or {}
        return {name: str(override.get(name) or self.prompts[name]) for name in self.classes}

    def options_for(self, family: str) -> dict[str, Any]:
        return {k: v for k, v in self.backends.get(family, {}).items() if k != "prompts"}

    def clip_paths(self) -> list[Path]:
        """Default clips under the data root; a missing one is an error, never a skip."""
        from src.contracts.paths import data_root

        paths = [data_root() / clip for clip in self.clips]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"domain {self.name!r} clips are not on disk: {missing}")
        return paths


def domain_names(path: Path = DOMAINS_YAML) -> tuple[str, ...]:
    import yaml

    return tuple(yaml.safe_load(path.read_text()) or {})


def load_domain(name: str, path: Path = DOMAINS_YAML) -> Domain:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    if name not in data:
        raise KeyError(f"unknown segmentation domain {name!r}; known: {sorted(data)}")
    entry = data[name]
    prompts = {str(k): str(v) for k, v in (entry.get("classes") or {}).items()}
    if not prompts:
        raise ValueError(f"domain {name!r} lists no foreground classes")
    return Domain(
        name=name,
        classes=tuple(prompts),
        prompts=prompts,
        backends={str(k): dict(v or {}) for k, v in (entry.get("backends") or {}).items()},
        clips=tuple(str(clip) for clip in entry.get("clips") or ()),
        summary=str(entry.get("summary") or ""),
    )


class Segmenter(Protocol):
    name: str

    def segment(self, source: Path | str, domain: Domain, *, max_frames: int | None = None) -> ClipMasks:
        """Masks for every frame of a video file or a directory of images."""


def build(name: str, **options: Any) -> Segmenter:
    """Construct a backend by name: ``sam31`` or ``yoloe-26{n,s,m,l,x}``."""
    if name == "sam31":
        from src.segmentation.sam31 import Sam31Segmenter

        return Sam31Segmenter(**options)
    if name.startswith("yoloe-26") and name[len("yoloe-26"):] in YOLOE_SIZES:
        from src.segmentation.yoloe import YoloeSegmenter

        return YoloeSegmenter(size=name[len("yoloe-26"):], **options)
    raise KeyError(f"unknown segmentation backend {name!r}; known: {', '.join(BACKENDS)}")


__all__ = [
    "BACKENDS",
    "REFERENCE_BACKEND",
    "ClipMasks",
    "Domain",
    "Instance",
    "Segmenter",
    "build",
    "domain_names",
    "load_domain",
]
