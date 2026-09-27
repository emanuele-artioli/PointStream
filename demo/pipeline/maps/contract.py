"""Sidecar contract for every maps-gallery stream.

payload_kbps uses payload_bytes only. preview_bytes is recorded and must
never enter the formula. Overlay MP4s are previews, not payloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MAP_NAMES = (
    "canny",
    "canny_180",
    "canny_360",
    "canny_540",
    "canny_720",
    "canny_1080",
    "depth",
    "yoloe_masks",
    "sam31_masks",
    "dino_feat",
    "dwpose",
    "dwpose_hands",
    "dwpose_face",
    "dwpose_body",
    "mediapipe_hands",
)

MaskEmptyPolicy = Literal["skip_frame"]
STREAM_KINDS = ("native", "overlay")


class OverlayPayloadError(ValueError):
    """Raised when an overlay/preview file is offered as the counted payload."""


@dataclass
class MapStream:
    map: str
    backend: str
    payload_path: str
    payload_bytes: int
    preview_path: str
    preview_bytes: int
    duration_s: float
    n_frames: int
    fps: float
    extract_ms_p50: float
    extract_ms_p95: float
    pack_ms_p50: float
    codec_ms_p50: float
    decode_ms_p50: float
    gpu: str
    kind: str = "native"
    mask_empty_policy: MaskEmptyPolicy = "skip_frame"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.map not in MAP_NAMES:
            raise ValueError(f"unknown map {self.map!r}; expected one of {MAP_NAMES}")
        if self.kind not in STREAM_KINDS:
            raise ValueError(f"kind must be native|overlay, got {self.kind!r}")
        if self.mask_empty_policy != "skip_frame":
            raise ValueError("mask_empty_policy must be skip_frame (never whole_frame_true)")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        payload = Path(self.payload_path)
        if self.kind == "overlay" or _looks_like_overlay(payload, self.kind):
            raise OverlayPayloadError(
                f"payload_path {payload} looks like a preview overlay; "
                "count native bytes (RLE/zstd/bitpack/gray-AV1), not the visualization MP4"
            )

    @property
    def payload_kbps(self) -> float:
        return payload_kbps(self.payload_bytes, self.duration_s)

    @property
    def teleop_ok(self) -> bool:
        serial = (
            self.extract_ms_p50
            + self.pack_ms_p50
            + self.codec_ms_p50
            + self.decode_ms_p50
        )
        return serial < 50.0

    def to_sidecar(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra") or {}
        data["payload_kbps"] = self.payload_kbps
        data["teleop_ok"] = self.teleop_ok
        data.update(extra)
        return data


def payload_kbps(payload_bytes: int, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return (payload_bytes * 8.0) / duration_s / 1000.0


def _looks_like_overlay(path: Path, kind: str) -> bool:
    name = path.name.lower()
    if kind == "overlay":
        return True
    if name.startswith("preview_"):
        return True
    if name.startswith("overlay"):
        return True
    return False


def write_sidecar(stream: MapStream, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stream.to_sidecar(), indent=2) + "\n")
    return path


def read_sidecar(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
