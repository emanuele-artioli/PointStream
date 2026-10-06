"""Score a candidate segmentation against the SAM 3.1 reference.

Where a domain has no dataset labels, SAM 3.1 (offline, bidirectional) is the
reference and every number here is agreement with it, not accuracy. Where the
reference is a dataset's labels (``ClipMasks.labelled`` set), only its labelled
frames are scored.

Per frame, on the foreground (union of classes) and per class:

* ``iou`` (J): region similarity. Frames where both masks are empty count as 1.
* ``boundary_f`` (F): DAVIS-style boundary F-measure, tolerance 0.8% of the diagonal.
* ``recall`` / ``precision``: missed foreground is encoded as background, which
  costs a codec more than extra foreground does, so they are reported apart.
* ``flicker``: mean |m_t xor m_{t-1}| / mean |m|, for candidate and reference.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.segmentation.masks import ClipMasks

BOUNDARY_TOLERANCE = 0.008


def region_scores(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {ref.shape}")
    inter = int(np.count_nonzero(pred & ref))
    union = int(np.count_nonzero(pred | ref))
    pred_area = int(np.count_nonzero(pred))
    ref_area = int(np.count_nonzero(ref))
    return {
        "iou": inter / union if union else 1.0,
        "precision": inter / pred_area if pred_area else (1.0 if ref_area == 0 else 0.0),
        "recall": inter / ref_area if ref_area else 1.0,
        "pred_area": float(pred_area),
        "ref_area": float(ref_area),
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = np.pad(mask.astype(np.uint8), 1)  # the image edge counts as boundary
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))[1:-1, 1:-1]
    binary = binary[1:-1, 1:-1]
    return (binary - eroded) > 0


def boundary_f(pred: np.ndarray, ref: np.ndarray, tolerance: float = BOUNDARY_TOLERANCE) -> float:
    import cv2

    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    pb, rb = _boundary(pred), _boundary(ref)
    if not pb.any() and not rb.any():
        return 1.0
    if not pb.any() or not rb.any():
        return 0.0
    radius = max(1, math.ceil(tolerance * math.hypot(*pred.shape)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    rb_dil = cv2.dilate(rb.astype(np.uint8), disk) > 0
    pb_dil = cv2.dilate(pb.astype(np.uint8), disk) > 0
    precision = np.count_nonzero(pb & rb_dil) / np.count_nonzero(pb)
    recall = np.count_nonzero(rb & pb_dil) / np.count_nonzero(rb)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _flicker(masks: list[np.ndarray]) -> float | None:
    if len(masks) < 2:
        return None
    changes = [np.count_nonzero(a ^ b) for a, b in zip(masks[:-1], masks[1:])]
    mean_area = float(np.mean([np.count_nonzero(m) for m in masks]))
    return float(np.mean(changes)) / mean_area if mean_area else 0.0


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def compare(candidate: ClipMasks, reference: ClipMasks) -> dict[str, Any]:
    """Aggregate agreement of ``candidate`` with ``reference`` over the shared labelled frames."""
    if (candidate.height, candidate.width) != (reference.height, reference.width):
        raise ValueError("candidate and reference were made at different resolutions")
    indices = [index for index in reference.labelled_frames() if index < len(candidate)]
    frames = len(indices)
    if frames == 0:
        raise ValueError("no frames to compare")
    scopes = {
        "foreground": None,
        **{name: name for name in reference.classes if name in candidate.classes},
    }
    rows: dict[str, dict[str, list[float]]] = {
        s: {"iou": [], "boundary_f": [], "precision": [], "recall": []} for s in scopes
    }
    series: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {s: ([], []) for s in scopes}
    empty_agree = 0
    for index in indices:
        for scope, class_name in scopes.items():
            if class_name is None:
                pred, ref = candidate.foreground(index), reference.foreground(index)
            else:
                pred, ref = (
                    candidate.class_mask(index, class_name),
                    reference.class_mask(index, class_name),
                )
            scores = region_scores(pred, ref)
            row = rows[scope]
            row["iou"].append(scores["iou"])
            row["boundary_f"].append(boundary_f(pred, ref))
            row["precision"].append(scores["precision"])
            row["recall"].append(scores["recall"])
            series[scope][0].append(pred)
            series[scope][1].append(ref)
            if class_name is None and scores["pred_area"] == 0 and scores["ref_area"] == 0:
                empty_agree += 1
    summary: dict[str, Any] = {"frames": frames, "both_empty_frames": empty_agree, "scopes": {}}
    for scope, row in rows.items():
        j, f = _mean(row["iou"]), _mean(row["boundary_f"])
        pred_flicker = _flicker(series[scope][0])
        ref_flicker = _flicker(series[scope][1])
        summary["scopes"][scope] = {
            "J": j,
            "F": f,
            "J&F": round((j + f) / 2, 4) if j is not None and f is not None else None,
            "J_p10": round(float(np.percentile(row["iou"], 10)), 4),
            "precision": _mean(row["precision"]),
            "recall": _mean(row["recall"]),
            "flicker": round(pred_flicker, 4) if pred_flicker is not None else None,
            "reference_flicker": round(ref_flicker, 4) if ref_flicker is not None else None,
        }
    return summary


def report_row(
    clip: str, backend: str, candidate: ClipMasks, reference: ClipMasks
) -> dict[str, Any]:
    scores = compare(candidate, reference)
    timing = candidate.meta.get("timing") or {}
    return {
        "clip": clip,
        "backend": backend,
        **scores,
        "ms_per_frame": timing.get("ms_per_frame"),
        "fps": timing.get("fps"),
        "peak_gpu_mib": timing.get("peak_gpu_mib"),
        "gpu": (
            (candidate.meta.get("worker_runtime") or candidate.meta.get("runtime") or {}).get("gpu")
            or {}
        ).get("name"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean over clips per backend: the accuracy-vs-speed table."""
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_backend.setdefault(row["backend"], []).append(row)
    table = []
    for backend, items in sorted(by_backend.items()):
        fg = [item["scopes"]["foreground"] for item in items]

        def avg(values: list[Any]) -> float | None:
            kept = [v for v in values if v is not None]
            return round(float(np.mean(kept)), 4) if kept else None

        table.append(
            {
                "backend": backend,
                "clips": len(items),
                "frames": sum(item["frames"] for item in items),
                **{
                    key: avg([s[key] for s in fg])
                    for key in ("J", "F", "J&F", "precision", "recall", "flicker")
                },
                "ms_per_frame": avg([item["ms_per_frame"] for item in items]),
                "fps": avg([item["fps"] for item in items]),
                "peak_gpu_mib": avg([item["peak_gpu_mib"] for item in items]),
            }
        )
    return table


__all__ = ["boundary_f", "compare", "region_scores", "report_row", "summarize"]
