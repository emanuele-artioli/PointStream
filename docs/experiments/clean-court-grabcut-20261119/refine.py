"""Use source-pixel GrabCut masks for the registered clean-court refinement."""

import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parent.parent / "clean-court-rally-20261118/pilot.py"
spec = importlib.util.spec_from_file_location("broad_court_pilot", SOURCE)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
original_mask = pilot.mask_at
LIMBS = [
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def mask_at(tick, frame, tracks, coords):
    mask = original_mask(tick, frame, {}, coords)
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    for runs in tracks.values():
        run = runs[0]
        x, y, width, height = run["boxes"][tick]
        x, y, width, height = x / 2, y * 544 / 1080, width / 2, height * 544 / 1080
        left = max(0, int(np.floor(x - width * 0.25)))
        top = max(0, int(np.floor(y - height * 0.15)))
        right = min(960, int(np.ceil(x + width * 1.25)))
        bottom = min(544, int(np.ceil(y + height * 1.15 + 12)))
        crop = bgr[top:bottom, left:right]
        labels = np.full(crop.shape[:2], cv2.GC_PR_BGD, np.uint8)
        inner_left = max(1, int(x - left))
        inner_top = max(1, int(y - top))
        inner_right = min(crop.shape[1] - 1, int(np.ceil(x + width - left)))
        inner_bottom = min(crop.shape[0] - 1, int(np.ceil(y + height - top)))
        labels[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_PR_FGD
        labels[[0, -1], :] = cv2.GC_BGD
        labels[:, [0, -1]] = cv2.GC_BGD
        kp = run["kp"][tick]
        points = np.rint(kp[:, :2] * np.array([0.5, 544 / 1080]) - [left, top]).astype(int)
        for a, b in LIMBS:
            if kp[a, 2] and kp[b, 2]:
                cv2.line(labels, tuple(points[a]), tuple(points[b]), cv2.GC_FGD, 2)
        for index, point in enumerate(points):
            if (
                kp[index, 2]
                and 1 < point[0] < labels.shape[1] - 2
                and 1 < point[1] < labels.shape[0] - 2
            ):
                cv2.circle(labels, tuple(point), 1, cv2.GC_FGD, -1)
        cv2.grabCut(
            crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK
        )
        foreground = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
        foreground = cv2.dilate(foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        for index in (9, 10):
            if kp[index, 2]:
                cv2.circle(foreground, tuple(points[index]), max(5, int(height * 0.06)), 255, -1)
        for index in (15, 16):
            if kp[index, 2]:
                foot = points[index] + [0, 3]
                cv2.ellipse(foreground, tuple(foot), (9, 6), 0, 0, 360, 255, -1)
        mask[top:bottom, left:right] |= foreground
    return mask


if __name__ == "__main__":
    pilot.mask_at = mask_at
    # The reused runner records its own source digest. Preserve both actual files.
    sys.argv[0] = str(Path(__file__).resolve())
    pilot.main()
