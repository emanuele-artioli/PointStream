"""CPU court registration. Regulation geometry in metres; no weights or downloads.

ITF Rule 1: 23.77 x 10.97 m, singles width 8.23 m, service lines 6.40 m
from the net. https://www.itftennis.com/en/about-us/organisation/tennis-glossary/
The net is NOT a planar painted line and is excluded from fitting and error.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

WIDTH, LENGTH = 10.97, 23.77
ALLEY = (WIDTH - 8.23) / 2
SERVICE = LENGTH / 2 - 6.40
SEGMENTS = np.array([
    [[0, 0], [WIDTH, 0]], [[0, LENGTH], [WIDTH, LENGTH]],
    [[0, 0], [0, LENGTH]], [[WIDTH, 0], [WIDTH, LENGTH]],
    [[ALLEY, 0], [ALLEY, LENGTH]], [[WIDTH - ALLEY, 0], [WIDTH - ALLEY, LENGTH]],
    [[ALLEY, SERVICE], [WIDTH - ALLEY, SERVICE]],
    [[ALLEY, LENGTH - SERVICE], [WIDTH - ALLEY, LENGTH - SERVICE]],
    [[WIDTH / 2, SERVICE], [WIDTH / 2, LENGTH - SERVICE]],
], np.float32)
OUTLINE = np.array([[0, 0], [WIDTH, 0], [WIDTH, LENGTH], [0, LENGTH]], np.float32)


def project(points: np.ndarray, hom: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), hom).reshape(-1, 2)


def white_lines(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Local contrast suppresses flat white graphics and large white clothing regions.
    value = hsv[:, :, 2]
    ridge = value.astype(np.int16) - cv2.GaussianBlur(value, (0, 0), 4).astype(np.int16)
    raw = ((hsv[:, :, 1] < 80) & (value > 170) & (ridge > 5)).astype(np.uint8)
    # Colour alone includes pale worn grass. Keep only pixels supported by a
    # long straight line, otherwise chamfer becomes distance to grass texture.
    scale = rgb.shape[0] / 1080
    lines = cv2.HoughLinesP(raw * 255, 1, np.pi / 720, max(20, round(50 * scale)),
                            minLineLength=max(20, round(80 * scale)), maxLineGap=max(4, round(12 * scale)))
    support = np.zeros_like(raw)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(support, (int(x1), int(y1)), (int(x2), int(y2)), (1,), max(3, round(7 * scale)))
    return raw & support


def samples(hom: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_points, groups = [], []
    for i, segment in enumerate(SEGMENTS):
        ends = project(segment, hom)
        n = max(2, int(np.ceil(np.linalg.norm(ends[1] - ends[0]) / step)) + 1)
        image_points.append(np.linspace(ends[0], ends[1], n))
        groups.extend([i] * n)
    points = np.concatenate(image_points).astype(np.float32)
    model = project(points, np.linalg.inv(hom))
    return model, points, np.array(groups)


def plausible(hom: np.ndarray, shape: tuple[int, ...]) -> bool:
    corners = project(OUTLINE, hom)
    h, w = shape[:2]
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(corners.reshape(-1, 1, 2))
    # Broad geometry guard; rejects tiny court-like graphics and collapsed solutions.
    return bool(0.07 * h * w < area < 0.9 * h * w
                and np.all(corners[:, 0] > -0.1 * w) and np.all(corners[:, 0] < 1.1 * w)
                and np.all(corners[:, 1] > 0) and np.all(corners[:, 1] < 1.1 * h))


def chamfer(hom: np.ndarray, mask: np.ndarray, step: float = 1.0) -> dict[str, Any]:
    """Directed chamfer: mean projected-line-point distance to detected line pixels.

    Sample every <=1 projected pixel, at native resolution. Out-of-frame painted
    lines have no image pixels to measure and are excluded; at least half the
    model samples must remain visible. Failed fits remain missing frame errors.
    Error is scaled to 1080p by image height. Each frame contributes one mean.
    """
    _, points, groups = samples(hom, step)
    h, w = mask.shape
    xy = np.rint(points).astype(int)
    visible = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if not mask.any() or visible.mean() < 0.5:
        return {"chamfer_px": None, "support": 0.0, "segments_supported": 0}
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    xy, groups = xy[visible], groups[visible]
    errors = dist[xy[:, 1], xy[:, 0]] * (1080 / h)
    supports = [float((errors[groups == i] <= 6).mean()) if np.any(groups == i) else 0.0
                for i in range(len(SEGMENTS))]
    return {"chamfer_px": float(errors.mean()), "support": float((errors <= 6).mean()),
            "segments_supported": sum(s >= 0.5 for s in supports), "visible_fraction": float(visible.mean())}


def refine(hom: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """RANSAC correspondences from projected painted lines to nearby detected pixels.

    Initial hypotheses come from independent Hough lines (or the prior frame).
    Require support on six distinct model lines to prevent four-corner tautology.
    """
    if not mask.any():
        return None
    _, nearest = distance_transform_edt(1 - mask, return_indices=True)
    result = hom.copy()
    for _ in range(2):
        model, pts, groups = samples(result, step=10)
        h, w = mask.shape
        xy = np.rint(pts).astype(int)
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy, model, pts, groups = xy[valid], model[valid], pts[valid], groups[valid]
        target = nearest[:, xy[:, 1], xy[:, 0]][::-1].T.astype(np.float32)
        close = np.linalg.norm(target - pts, axis=1) <= 10 * h / 1080
        if close.sum() < 16 or len(set(groups[close])) < 6:
            return None
        fitted, inliers = cv2.findHomography(model[close], target[close], cv2.RANSAC, 3 * h / 1080,
                                            maxIters=300, confidence=0.995)
        if fitted is None or inliers is None or inliers.sum() < 12 or not plausible(fitted, mask.shape):
            return None
        result = fitted
    return result


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.cross(a, b)
    return p[:2] / p[2] if abs(p[2]) > 1e-8 else np.array([np.nan, np.nan])


def seed(mask: np.ndarray) -> np.ndarray | None:
    """Hough baseline/sideline rectangles ranked by support on the COMPLETE model."""
    h, w = mask.shape
    small = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    lines = cv2.HoughLinesP(small * 255, 1, np.pi / 720, 40, minLineLength=60, maxLineGap=20)
    if lines is None:
        return None
    horizontal: list[np.ndarray] = []
    sides: list[np.ndarray] = []
    hs: list[float] = []
    ss: list[float] = []
    ordered = sorted(lines[:, 0], key=lambda line: -float(np.linalg.norm(line[2:] - line[:2])))
    for coords in ordered:
        x1, y1, x2, y2 = coords.astype(float) * 2
        line = np.cross([x1, y1, 1], [x2, y2, 1])
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > 3 * dy:
            y = -(line[0] * w / 2 + line[2]) / line[1]
            if 0.12 * h < y < 0.95 * h and all(abs(y - p) > 12 for p in hs) and len(hs) < 8:
                horizontal.append(line)
                hs.append(y)
        elif dy > 0.35 * dx and abs(line[0]) > 1e-8:
            x = -(line[1] * 0.6 * h + line[2]) / line[0]
            if 0 < x < w and all(abs(x - p) > 12 for p in ss) and len(ss) < 8:
                sides.append(line)
                ss.append(x)
    distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    best, best_cost = None, float("inf")
    for ai, bi in combinations(range(len(horizontal)), 2):
        top, bottom = sorted((ai, bi), key=lambda i: hs[i])
        if hs[bottom] - hs[top] < h * 0.18:
            continue
        for ci, di in combinations(range(len(sides)), 2):
            left, right = sorted((ci, di), key=lambda i: ss[i])
            corners = np.array([intersection(horizontal[top], sides[left]), intersection(horizontal[top], sides[right]),
                                intersection(horizontal[bottom], sides[right]), intersection(horizontal[bottom], sides[left])], np.float32)
            if not np.isfinite(corners).all():
                continue
            for x0, x1 in ((0.0, WIDTH), (ALLEY, WIDTH - ALLEY)):
                model = np.array([[x0, 0], [x1, 0], [x1, LENGTH], [x0, LENGTH]], np.float32)
                hom = cv2.getPerspectiveTransform(model, corners)
                if not plausible(hom, mask.shape):
                    continue
                _, pts, _ = samples(hom, step=12)
                xy = np.rint(pts).astype(int)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                if inside.mean() < 0.5:
                    continue
                xy = xy[inside]
                cost = float(np.minimum(distance[xy[:, 1], xy[:, 0]], 30).mean())
                if cost < best_cost:
                    best, best_cost = hom, cost
    return best if best_cost < 12 else None


def register(rgb: np.ndarray, previous: np.ndarray | None = None) -> dict[str, Any]:
    mask = white_lines(rgb)
    candidate = previous if previous is not None else seed(mask)
    hom = refine(candidate, mask) if candidate is not None else None
    if hom is None and previous is not None:
        candidate = seed(mask)
        hom = refine(candidate, mask) if candidate is not None else None
    if hom is None:
        return {"homography": None, "chamfer_px": None, "registered": False}
    error = chamfer(hom, mask)
    if previous is not None and (error["chamfer_px"] is None or error["chamfer_px"] > 6):
        independent = seed(mask)
        alternate = refine(independent, mask) if independent is not None else None
        if alternate is not None:
            alternative_error = chamfer(alternate, mask)
            if alternative_error["chamfer_px"] is not None and (error["chamfer_px"] is None or alternative_error["chamfer_px"] < error["chamfer_px"]):
                hom, error = alternate, alternative_error
    valid = error["chamfer_px"] is not None and error["chamfer_px"] <= 2.0 and error["segments_supported"] >= 6
    return {"homography": hom.tolist(), **error, "registered": bool(valid)}


# K8b (docs/experiments.md, 2026-10-20 rule). K8's detector above is kept unchanged for
# reproducing K8's evidence; K8b adds a shade-robust detector, an independent ruler and a
# spurious-fit guard, and chooses among hypotheses from both detectors.
FIT_PX = 6.0          # a frame fits at the decision rule's p95 bar
CLUTTER_MAX = 0.25    # dev: true court fits <= 0.085, spurious fits 0.52-0.66


def hough_segments(raw: np.ndarray) -> np.ndarray:
    """Long straight segments (x1, y1, x2, y2), with K8's Hough parameters scaled to the frame."""
    scale = raw.shape[0] / 1080
    lines = cv2.HoughLinesP(raw * 255, 1, np.pi / 720, max(20, round(50 * scale)),
                            minLineLength=max(20, round(80 * scale)), maxLineGap=max(4, round(12 * scale)))
    return np.zeros((0, 4), np.int32) if lines is None else lines[:, 0]


def relative_lines(rgb: np.ndarray, kernel: int = 31, rel: float = 0.10, absolute: float = 10.0,
                   saturation: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """(line mask, 1 px centerline ruler), relative to the local background in shade or sun.

    A line pixel is brighter than the morphological opening of the value channel (which
    removes thin bright lines) by max(absolute, rel * background), and its saturation is
    at most `saturation` times the closing of the saturation channel.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    background = cv2.morphologyEx(v, cv2.MORPH_OPEN, shape)
    surround = cv2.morphologyEx(s, cv2.MORPH_CLOSE, shape)
    raw = ((v - background >= np.maximum(absolute, rel * background)) & (s <= saturation * surround)).astype(np.uint8)
    width = max(3, round(7 * rgb.shape[0] / 1080))
    support, ruler = np.zeros_like(raw), np.zeros_like(raw)
    for x1, y1, x2, y2 in hough_segments(raw):
        cv2.line(support, (int(x1), int(y1)), (int(x2), int(y2)), (1,), width)
        cv2.line(ruler, (int(x1), int(y1)), (int(x2), int(y2)), (1,), 1)
    return raw & support, ruler


def clutter(hom: np.ndarray, mask: np.ndarray, margin: int = 6) -> float:
    """Share of the projected court's in-frame interior, beyond `margin` px of every model
    line, covered by detected pixels; a fit on texture (nets, crowd) is cluttered."""
    corners = project(OUTLINE, hom)
    if not np.isfinite(corners).all():
        return 1.0
    inside = np.zeros_like(mask)
    cv2.fillConvexPoly(inside, np.rint(corners).astype(np.int32), (1,))
    near = np.zeros_like(mask)
    for segment in SEGMENTS:
        cv2.polylines(near, [np.rint(project(segment, hom)).astype(np.int32)], False, (1,), 2 * margin + 1)
    region = (inside > 0) & (near == 0)
    return float(mask[region].mean()) if region.any() else 1.0


def register_v2(rgb: np.ndarray, previous: np.ndarray | None = None) -> dict[str, Any]:
    """K8b registration: warm start, then both detectors' seeds when it fails or exceeds 2 px.

    Each hypothesis is refined against both masks; the valid fit (>= 6 supported segments on
    the ruler, clutter <= CLUTTER_MAX) with the lowest ruler chamfer wins. `error_px` is the
    ruler chamfer at 1080p; `fits` is error <= FIT_PX. Missing fits stay missing.
    """
    old = white_lines(rgb)
    mask, ruler = relative_lines(rgb)
    best: tuple[np.ndarray, dict[str, Any], float] | None = None

    def consider(candidate: np.ndarray) -> None:
        nonlocal best
        for target in (old, mask):
            fitted = refine(candidate, target)
            if fitted is None:
                continue
            error = chamfer(fitted, ruler)
            if error["chamfer_px"] is None or error["segments_supported"] < 6:
                continue
            mess = clutter(fitted, mask)
            if mess <= CLUTTER_MAX and (best is None or error["chamfer_px"] < best[1]["chamfer_px"]):
                best = (fitted, error, mess)

    if previous is not None:
        consider(previous)
    if best is None or best[1]["chamfer_px"] > 2.0:
        for start in (seed(old), seed(mask)):
            if start is not None:
                consider(start)
    if best is None:
        return {"homography": None, "error_px": None, "fits": False, "registered": False}
    hom, error, mess = best
    return {"homography": hom.tolist(), "error_px": error["chamfer_px"], "segments_supported": error["segments_supported"],
            "visible_fraction": error["visible_fraction"], "clutter": mess,
            "fits": error["chamfer_px"] <= FIT_PX, "registered": error["chamfer_px"] <= 2.0}
