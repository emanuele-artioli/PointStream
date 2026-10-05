"""YOLOE-26 open-vocabulary instance segmentation: the fast candidates.

One model per size (n, s, m, l, x), prompted with the domain's class texts via
the local MobileCLIP2 text encoder. Frames are processed one at a time, so
`stream` is the runtime path and `segment` is the same loop over a clip.

YOLOE has no memory, so a light mask-IoU tracker (`TrackFilter`) gives each
instance a stable id and, when a domain asks for it, suppresses one-frame
flickers and holds a confirmed object through short misses. ByteTrack is not
used: on egocentric footage the hands move too far between frames for its box
association.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.segmentation.masks import ClipMasks

Detection = tuple[str, np.ndarray, float]  # class, bool HxW mask, confidence
Tracked = tuple[str, int, np.ndarray, float]  # class, track id, mask, confidence

DEFAULTS: dict[str, Any] = {
    "imgsz": None,  # None: the frame's long side, capped at 1920
    "conf": 0.25,  # passed to the model
    "new_track_conf": None,  # None: same as conf; weaker boxes may only extend a track
    "match_iou": 0.05,
    "min_hits": 1,
    "hold_frames": 0,
    "min_area_ratio": 0.0,
    "largest_component": True,
    "half": True,
}
MAX_IMGSZ = 1920


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected region of one detection's mask."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def detections_from_result(result: Any, classes: tuple[str, ...], *, keep_largest: bool = True) -> list[Detection]:
    """Full-frame boolean masks with class names and confidences."""
    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return []
    data = _numpy(result.masks.data)
    if data.size == 0:
        return []
    if data.ndim == 2:
        data = data[None]
    cls = _numpy(result.boxes.cls).astype(int).reshape(-1)
    conf = _numpy(getattr(result.boxes, "conf", None)).reshape(-1)
    height, width = (int(v) for v in result.orig_shape[:2])
    found: list[Detection] = []
    for index, raw in enumerate(data):
        if index >= cls.size or not 0 <= cls[index] < len(classes):
            continue
        binary = raw > 0.5
        if binary.shape != (height, width):
            import cv2

            binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        if keep_largest:
            binary = largest_component(binary)
        if binary.any():
            found.append((classes[cls[index]], binary, float(conf[index]) if index < conf.size else 1.0))
    return found


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return int(np.count_nonzero(left & right)) / union if union else 0.0


@dataclass
class _Track:
    track_id: int
    class_name: str
    mask: np.ndarray
    score: float
    hits: int = 1
    missing: int = 0


@dataclass
class TrackFilter:
    """Greedy mask-IoU association across frames, per class.

    A detection at or above ``new_track_conf`` may start a track; a weaker one
    can only extend a track it overlaps. A track is emitted once it has
    ``min_hits`` matches and is repeated for up to ``hold_frames`` missed
    frames. Within a class, emitted masks smaller than ``min_area_ratio`` of the
    largest are dropped.
    """

    new_track_conf: float = 0.25
    match_iou: float = 0.05
    min_hits: int = 1
    hold_frames: int = 0
    min_area_ratio: float = 0.0
    tracks: list[_Track] = field(default_factory=list)
    next_id: int = 1

    def update(self, detections: list[Detection]) -> list[Tracked]:
        pairs = sorted(
            (
                (_iou(mask, track.mask), d, t)
                for d, (name, mask, _score) in enumerate(detections)
                for t, track in enumerate(self.tracks)
                if name == track.class_name
            ),
            reverse=True,
        )
        used_d: set[int] = set()
        used_t: set[int] = set()
        for iou, d, t in pairs:
            if iou < self.match_iou or d in used_d or t in used_t:
                continue
            track = self.tracks[t]
            _name, track.mask, track.score = detections[d]
            track.hits += 1
            track.missing = 0
            used_d.add(d)
            used_t.add(t)
        for t, track in enumerate(self.tracks):
            if t not in used_t:
                track.missing += 1
        for d, (name, mask, score) in enumerate(detections):
            if d not in used_d and score >= self.new_track_conf:
                self.tracks.append(_Track(self.next_id, name, mask, score))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missing <= self.hold_frames]
        emitted = [t for t in self.tracks if t.hits >= self.min_hits]
        largest: dict[str, int] = {}
        for track in emitted:
            largest[track.class_name] = max(largest.get(track.class_name, 0), int(track.mask.sum()))
        return [
            (t.class_name, t.track_id, t.mask, t.score)
            for t in emitted
            if t.mask.sum() >= self.min_area_ratio * largest[t.class_name]
        ]


def bind_local_text_encoder(path: Path) -> None:
    """Make Ultralytics load MobileCLIP2 from Models instead of downloading it."""
    import ultralytics.nn.text_model as text_model

    original = text_model.MobileCLIPTS.__init__
    if getattr(original, "_pointstream_local", False):
        return

    def init(self: Any, device: Any, weight: str = "mobileclip_blt.ts") -> None:
        if Path(str(weight)).name in {path.name, "mobileclip2_b.ts", "mobileclip_blt.ts"}:
            weight = str(path)
        original(self, device, weight)

    init._pointstream_local = True  # type: ignore[attr-defined]
    text_model.MobileCLIPTS.__init__ = init  # type: ignore[method-assign]


class YoloeSegmenter:
    name: str

    def __init__(self, size: str = "s", *, weights: str | Path | None = None, model: Any = None,
                 device: Any = None, **options: Any) -> None:
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown YOLOE options: {sorted(unknown)}")
        self.size = size
        self.name = f"yoloe-26{size}"
        self.weights_name = str(weights or f"yoloe-26{size}-seg.pt")
        self.overrides = options
        self.model = model
        self.device = device
        self.weights_path: Path | None = None
        self.text_encoder_path: Path | None = None
        self._prompts: tuple[str, ...] | None = None

    def options(self, domain: Any) -> dict[str, Any]:
        merged = {**DEFAULTS, **domain.options_for("yoloe"), **self.overrides}
        if merged["new_track_conf"] is None:
            merged["new_track_conf"] = merged["conf"]
        return merged

    def _cuda(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def load(self, domain: Any) -> Any:
        prompts = tuple(domain.prompts_for("yoloe").values())
        if self.model is None:
            from ultralytics import YOLOE

            from src.components.detection.weights import resolve_weight

            self.weights_path = resolve_weight(self.weights_name)
            self.text_encoder_path = resolve_weight("mobileclip2_b.ts")
            bind_local_text_encoder(self.text_encoder_path)
            self.model = YOLOE(str(self.weights_path))
        if prompts != self._prompts:
            # Text embeddings are computed in fp32; half precision applies to predict only.
            self.model.set_classes(list(prompts))
            self._prompts = prompts
        if self.device is None:
            self.device = 0 if self._cuda() else "cpu"
        return self.model

    def stream(self, frames: Iterable[np.ndarray], domain: Any, *, timings: list[float] | None = None) -> Iterator[list[Tracked]]:
        """Per BGR frame, the tracked instances. Appends per-frame ms to ``timings``."""
        model = self.load(domain)
        opts = self.options(domain)
        tracker = TrackFilter(
            new_track_conf=opts["new_track_conf"],
            match_iou=opts["match_iou"],
            min_hits=opts["min_hits"],
            hold_frames=opts["hold_frames"],
            min_area_ratio=opts["min_area_ratio"],
        )
        cuda = self._cuda()
        for frame in frames:
            imgsz = opts["imgsz"] or min(MAX_IMGSZ, -(-max(frame.shape[:2]) // 32) * 32)
            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=opts["conf"],
                retina_masks=True,
                half=bool(opts["half"] and cuda),
                device=self.device,
                verbose=False,
            )
            tracked = tracker.update(
                detections_from_result(results[0] if results else None, domain.classes,
                                       keep_largest=opts["largest_component"])
            )
            if cuda:
                import torch

                torch.cuda.synchronize()
            if timings is not None:
                timings.append((time.perf_counter() - t0) * 1000.0)
            yield tracked

    def segment(self, source: Path | str, domain: Any, *, max_frames: int | None = None) -> ClipMasks:
        from src.segmentation.sources import iter_frames, timing_summary, video_fps

        load_t0 = time.perf_counter()
        self.load(domain)
        model_load_s = time.perf_counter() - load_t0
        frames = iter_frames(source, max_frames)
        first = next(frames)
        masks = ClipMasks(domain.classes, first.shape[0], first.shape[1], video_fps(source))
        timings: list[float] = []
        peak = None
        if self._cuda():
            import torch

            torch.cuda.reset_peak_memory_stats()

        def all_frames() -> Iterator[np.ndarray]:
            yield first
            yield from frames

        for index, tracked in enumerate(self.stream(all_frames(), domain, timings=timings)):
            masks.ensure_frames(index + 1)
            for class_name, track_id, mask, score in tracked:
                masks.add(index, class_name, track_id, mask, score)
        if self._cuda():
            import torch

            peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        # The first frame includes CUDA warm-up; throughput excludes it.
        steady = timings[1:] or timings
        masks.meta.update({
            "backend": self.name,
            "prompts": domain.prompts_for("yoloe"),
            "options": self.options(domain),
            "model": {
                "weights": str(self.weights_path) if self.weights_path else self.weights_name,
                "text_encoder": str(self.text_encoder_path) if self.text_encoder_path else None,
            },
            "timing": {
                **timing_summary(steady, model_load_s=model_load_s, total_s=sum(steady) / 1000.0, frames=len(steady)),
                "first_frame_ms": round(timings[0], 3) if timings else None,
                "peak_gpu_mib": peak,
            },
        })
        return masks


__all__ = ["TrackFilter", "YoloeSegmenter", "detections_from_result", "largest_component"]
