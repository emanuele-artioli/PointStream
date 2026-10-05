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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Sequence

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

    def with_overrides(self, prompts: Sequence[str] = (), options: Sequence[str] = ()) -> Domain:
        """Apply ``[FAMILY:]CLASS=TEXT`` prompts and ``FAMILY:KEY=VALUE`` options.

        Prompts for classes this domain does not have are skipped, so one
        command line can carry overrides for several domains.
        """
        import copy

        import yaml

        base = dict(self.prompts)
        backends = copy.deepcopy(self.backends)
        for item in prompts:
            target, _, text = item.partition("=")
            family, _, name = target.rpartition(":")
            if not text:
                raise ValueError(f"prompt override {item!r} is not [FAMILY:]CLASS=TEXT")
            if name not in self.classes:
                continue
            if family:
                backends.setdefault(family, {}).setdefault("prompts", {})[name] = text
            else:
                base[name] = text
        for item in options:
            target, _, value = item.partition("=")
            family, _, key = target.partition(":")
            if not family or not key or not value:
                raise ValueError(f"option override {item!r} is not FAMILY:KEY=VALUE")
            backends.setdefault(family, {})[key] = yaml.safe_load(value)
        return replace(self, prompts=base, backends=backends)

    def clip_paths(self) -> list[Path]:
        """Default clips under the datasets root; a missing one is an error, never a skip."""
        paths = [datasets_root() / clip for clip in self.clips]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"domain {self.name!r} clips are not on disk: {missing}")
        return paths


def datasets_root() -> Path:
    """Raw source datasets: ``PS_DATASETS_ROOT``, else ``~/Datasets``.

    Not `paths.data_root()`, which names PointStream's own assets/outputs tree
    (``Datasets/pointstream-data`` on the fleet).
    """
    import os

    override = os.environ.get("PS_DATASETS_ROOT", "").strip()
    return Path(override).expanduser() if override else Path.home() / "Datasets"


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

    def segment(
        self, source: Path | str, domain: Domain, *, max_frames: int | None = None
    ) -> ClipMasks:
        """Masks for every frame of a video file or a directory of images."""


def segment_array(
    backend: Segmenter, frames: Any, domain: Domain, *, rgb: bool = True
) -> ClipMasks:
    """Segment an in-memory ``(T, H, W, 3)`` clip (the runner's frames are RGB)."""
    import numpy as np

    clip = np.asarray(frames)
    bgr = clip[..., ::-1] if rgb else clip
    if hasattr(backend, "stream"):
        masks = ClipMasks(domain.classes, int(clip.shape[1]), int(clip.shape[2]), 30.0)
        for index, tracked in enumerate(backend.stream(iter(bgr), domain)):
            masks.ensure_frames(index + 1)
            for class_name, track_id, mask, score in tracked:
                masks.add(index, class_name, track_id, mask, score)
        return masks
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory(prefix="ps-seg-") as tmp:
        for index, frame in enumerate(bgr):
            cv2.imwrite(str(Path(tmp) / f"{index:05d}.png"), np.ascontiguousarray(frame))
        return backend.segment(tmp, domain)


def build(name: str, **options: Any) -> Segmenter:
    """Construct a backend by name: ``sam31`` or ``yoloe-26{n,s,m,l,x}``."""
    if name == "sam31":
        from src.segmentation.sam31 import Sam31Segmenter

        return Sam31Segmenter(**options)
    if name.startswith("yoloe-26") and name[len("yoloe-26") :] in YOLOE_SIZES:
        from src.segmentation.yoloe import YoloeSegmenter

        return YoloeSegmenter(size=name[len("yoloe-26") :], **options)
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
    "segment_array",
]
