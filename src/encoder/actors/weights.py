"""Weight resolution and bounding-box geometry.

Weights are resolved from assets/weights/ before any backend loads, so a
missing file fails there rather than triggering a silent download."""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


def _assets_weights_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "weights"
def _configure_ultralytics_weights_dir() -> None:
    try:
        from ultralytics.utils import SETTINGS

        weights_dir = _assets_weights_dir()
        weights_dir.mkdir(parents=True, exist_ok=True)
        SETTINGS["weights_dir"] = str(weights_dir)
    except Exception:
        # Keep runtime robust even if ultralytics internals change.
        return
def _resolve_local_weight_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate

    assets_candidate = _assets_weights_dir() / model_name
    if assets_candidate.exists():
        return assets_candidate

    return None
def _require_local_or_optin_weight(model_name: str, allow_download: bool = True) -> str:
    local_path = _resolve_local_weight_path(model_name)
    if local_path is not None:
        return str(local_path)

    if allow_download:
        return model_name

    raise FileNotFoundError(
        f"Required model weights not found for '{model_name}'. "
        "Place weights in assets/weights/ or set allow-auto-model-download: true in config."
    )
def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0.0, width - 1.0))
    y1 = float(np.clip(y1, 0.0, height - 1.0))
    x2 = float(np.clip(x2, x1 + 1.0, width))
    y2 = float(np.clip(y2, y1 + 1.0, height))
    return [x1, y1, x2, y2]
