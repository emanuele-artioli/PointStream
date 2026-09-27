"""Task-boundary score for a luma Canny map.

The judge is structure we already have, not a reconstruction of RGB:

* YOLOE instance masks, reduced to their outer contour
* DW-Pose preview strokes

Recall is how much of that contour a Canny map covers within a 2px tolerance.
Waste is the fraction of Canny pixels that sit off the band. Flicker is the
fraction of edge pixels that appear or vanish between frames. The scalar is
recall minus those two penalties. Payload size is a separate axis: the winner
is the cheapest mask whose recall stays close to dense 50/150.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from demo.pipeline.maps.canny import frame_to_luma, resize_binary_mask, scale_hw


def _disk(radius: int) -> np.ndarray:
    size = int(radius) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def boundary_band(filled: np.ndarray, radius: int = 1) -> np.ndarray:
    """Outer contour of a filled mask. ``radius`` thickens that contour before scoring."""
    mask = (np.asarray(filled) > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return mask
    eroded = cv2.erode(mask, _disk(1))
    outline = ((mask > 0) & (eroded == 0)).astype(np.uint8)
    if int(radius) <= 1:
        return outline
    return cv2.dilate(outline, _disk(int(radius) - 1))


def stroke_band(strokes: np.ndarray, radius: int = 1) -> np.ndarray:
    """Pose strokes. ``radius`` > 1 dilates them; the score's hit radius is the tolerance."""
    mask = (np.asarray(strokes) > 0).astype(np.uint8)
    if int(mask.sum()) == 0 or int(radius) <= 1:
        return mask
    return cv2.dilate(mask, _disk(int(radius) - 1))


def resize_keep(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Downsample a binary mask without dropping thin strokes."""
    plane = (np.asarray(mask) > 0).astype(np.uint8)
    if plane.shape == (height, width):
        return plane
    scaled = cv2.resize(plane * 255, (width, height), interpolation=cv2.INTER_AREA)
    return (scaled > 0).astype(np.uint8)


def gate_to_task_boundary(
    edge: np.ndarray,
    filled: np.ndarray,
    strokes: np.ndarray,
    *,
    radius: int = 2,
) -> np.ndarray:
    """Keep Canny pixels that sit on the YOLOE contour or the pose strokes.

    ``radius`` is the dilation of that contour in the edge map's own pixels.
    Radius 2 matches the full map's task recall and drops the bench grain.
    """
    plane = (np.asarray(edge) > 0).astype(np.uint8)
    height, width = int(plane.shape[0]), int(plane.shape[1])
    band = union_bands(
        boundary_band(resize_keep(filled, height, width), radius=1),
        stroke_band(resize_keep(strokes, height, width), radius=1),
    )
    if int(radius) > 0:
        band = cv2.dilate(band, _disk(int(radius)))
    return ((plane > 0) & (band > 0)).astype(np.uint8)


def union_bands(*bands: np.ndarray) -> np.ndarray:
    acc = None
    for band in bands:
        plane = (np.asarray(band) > 0).astype(np.uint8)
        acc = plane if acc is None else np.maximum(acc, plane)
    if acc is None:
        raise ValueError("no bands")
    return acc


def extract_tuned_frame(
    frame: np.ndarray,
    *,
    lo: int,
    hi: int,
    target_height: int,
    blur_sigma: float = 0.0,
    fullres_then_down: bool = False,
) -> np.ndarray:
    """Canny on luma at ``target_height``.

    ``fullres_then_down`` matches the gallery's current ladder: Canny at the
    source resolution, then area-downsample the binary mask. Otherwise the
    luma is area-downsampled first, which is the resolution knob.
    """
    luma = frame_to_luma(frame)
    src_h, src_w = int(luma.shape[0]), int(luma.shape[1])
    height, width = scale_hw(src_h, src_w, int(target_height))
    if fullres_then_down:
        if float(blur_sigma) > 0:
            luma = cv2.GaussianBlur(luma, (0, 0), float(blur_sigma))
        edges = cv2.Canny(luma, int(lo), int(hi))
        return resize_binary_mask(edges, height, width)
    if (height, width) != (src_h, src_w):
        luma = cv2.resize(luma, (width, height), interpolation=cv2.INTER_AREA)
    if float(blur_sigma) > 0:
        luma = cv2.GaussianBlur(luma, (0, 0), float(blur_sigma))
    edges = cv2.Canny(luma, int(lo), int(hi))
    return (edges > 0).astype(np.uint8)


def score_edges(
    edges: np.ndarray,
    boundary: np.ndarray,
    prev: np.ndarray | None = None,
    *,
    hit_radius: int = 2,
    waste_radius: int = 2,
) -> dict[str, float]:
    """Recall / waste / flicker for one binary edge plane against a boundary band."""
    edge = (np.asarray(edges) > 0).astype(np.uint8)
    band = (np.asarray(boundary) > 0).astype(np.uint8)
    if edge.shape != band.shape:
        raise ValueError(f"edge shape {edge.shape} != boundary {band.shape}")
    cover = cv2.dilate(edge, _disk(int(hit_radius))) if int(hit_radius) > 0 else edge
    band_n = int(band.sum())
    recall = 1.0 if band_n == 0 else float(np.count_nonzero(cover & band)) / float(band_n)
    near = cv2.dilate(band, _disk(int(waste_radius))) if int(waste_radius) > 0 else band
    edge_n = int(edge.sum())
    waste = 0.0 if edge_n == 0 else float(np.count_nonzero((edge > 0) & (near == 0))) / float(edge_n)
    if prev is None:
        flicker = 0.0
    else:
        previous = (np.asarray(prev) > 0).astype(np.uint8)
        if previous.shape != edge.shape:
            raise ValueError(f"prev shape {previous.shape} != edge {edge.shape}")
        changed = int(np.count_nonzero(edge != previous))
        denom = edge_n + int(previous.sum())
        flicker = 0.0 if denom == 0 else changed / float(denom)
    quality = recall - 0.35 * waste - 0.25 * flicker
    return {
        "recall": float(recall),
        "waste": float(waste),
        "flicker": float(flicker),
        "quality": float(quality),
        "edge_px": float(edge_n),
        "boundary_px": float(band_n),
    }


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average per-frame scores. Recall ignores frames with an empty boundary."""
    if not rows:
        raise ValueError("no scores")
    recall_rows = [row for row in rows if row["boundary_px"] > 0]
    recall_src = recall_rows or list(rows)
    keys = ("recall", "waste", "flicker", "quality", "edge_px")
    out = {key: float(np.mean([row[key] for row in recall_src])) for key in keys if key != "edge_px"}
    out["edge_px"] = float(np.mean([row["edge_px"] for row in rows]))
    out["n_scored"] = float(len(recall_src))
    return out
