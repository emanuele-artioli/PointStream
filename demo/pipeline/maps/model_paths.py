"""Absolute checkpoint paths. Never pass a bare Hub id into Ultralytics.

Inventory taken 2026-09-21 on gpu1:/home/itec/emanuele/Models.
Override the root with POINTSTREAM_MODELS. Missing files stay None — callers
must skip, not auto-download (see src.components.detection.weights).
"""

from __future__ import annotations

import os
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("POINTSTREAM_MODELS", "/home/itec/emanuele/Models"))


def _file(*parts: str) -> Path | None:
    path = MODELS_ROOT.joinpath(*parts)
    return path if path.is_file() else None


def _first(*candidates: Path | None) -> Path | None:
    for path in candidates:
        if path is not None:
            return path
    return None


MODELS: dict[str, Path | None] = {
    # YOLOE-26: s-scale not on disk; n is the fast default, x is quality.
    "yoloe26_seg": _first(
        _file("YOLO", "yoloe-26s-seg.pt"),
        _file("YOLO", "yoloe-26n-seg.pt"),
        _file("YOLO", "yoloe-26x-seg.pt"),
    ),
    "yoloe26_seg_x": _file("YOLO", "yoloe-26x-seg.pt"),
    "mobileclip2": _file("YOLO", "mobileclip2_b.ts"),
    # YOLO26-depth weights were not in Models/ as of this inventory.
    "yolo26s_depth": _first(
        _file("YOLO", "yolo26s-depth.pt"),
        _file("YOLO", "yolo26n-depth.pt"),
        _file("YOLO", "yolo26x-depth.pt"),
    ),
    "yolo26n_pose": _file("YOLO", "yolo26n-pose.pt"),
    "yolo26n_seg": _file("YOLO", "yolo26n-seg.pt"),
    # SAM 3.1 multiplex not on disk; SAM 3 is the quality mask backend.
    "sam31": _file("SAM", "sam3.1_multiplex.pt"),
    "sam3": _file("SAM", "sam3.pt"),
    "fastsam": _file("SAM", "FastSAM-x.pt"),
    "dinov3_vits": _file("dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    "dwpose_pose": _file("DWPose", "dw-ll_ucoco_384.onnx"),
    "dwpose_det": _file("DWPose", "yolox_l.onnx"),
}

MISSING: tuple[str, ...] = tuple(sorted(k for k, v in MODELS.items() if v is None))


def require(key: str) -> Path:
    """Raise FileNotFoundError naming the inventory key, never a Hub id."""
    path = MODELS.get(key)
    if path is None:
        raise FileNotFoundError(
            f"MODELS[{key!r}] is missing under {MODELS_ROOT}. "
            f"Known missing keys: {list(MISSING)}. Do not auto-download."
        )
    return path
