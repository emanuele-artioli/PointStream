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
    return ((hsv[:, :, 1] < 110) & (value > 155) & (ridge > 5)).astype(np.uint8)


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
    lines are missing support, not clipped away to make a partial court pass.
    Error is scaled to 1080p by image height. Each frame contributes one mean.
    """
    _, points, groups = samples(hom, step)
    h, w = mask.shape
    xy = np.rint(points).astype(int)
    visible = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if not mask.any() or not visible.all():
        return {"chamfer_px": None, "support": 0.0, "segments_supported": 0}
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    errors = dist[xy[:, 1], xy[:, 0]] * (1080 / h)
    supports = [float((errors[groups == i] <= 6).mean()) for i in range(len(SEGMENTS))]
    return {"chamfer_px": float(errors.mean()), "support": float((errors <= 6).mean()),
            "segments_supported": sum(s >= 0.5 for s in supports)}


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
                if np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h):
                    continue
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
    valid = error["chamfer_px"] is not None and error["chamfer_px"] <= 2.0 and error["segments_supported"] >= 6
    return {"homography": hom.tolist(), **error, "registered": bool(valid)}
