"""Is the gap between two points' plates camera motion, or is it content?

`plans/BP24-findings.md` §17 measured that two scenes' first frames differ by
13.75 dB, and `plate_delta.py` measured that coding one as a delta against the
other is dominated -- 1.5-1.7x the bytes for 13 dB less quality. Both scenes in
each pair are labelled `cluster_point` in the dataset's own scene metadata, so
that is the proposal ("reuse the plate across points of a match") measured on
the content it was proposed for.

But *why* they differ decides what happens next. A broadcast camera pans and
zooms to follow play, so two points can show the same court from a different
position. If the gap is camera geometry, a homography removes it and the idea
becomes "register, then share" -- which is panorama stitching, expensive but
already half-implemented in `src/components/background/plate.py`. If the gap
survives registration, it is content -- crowd, shadows, scoreboard, players --
and no amount of warping will make one plate serve two points.

This estimates a homography between the two plates from feature matches, warps
one onto the other, and scores **only the overlapping region**, because the
warp leaves undefined borders that would otherwise dominate the number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.headroom.real import load_rgb_stack
from experiments.tier.clip import BP21_CLIPS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "bp24-ladder" / "plate-register.json"

#: Below this many good matches a homography is fitted to noise and its result
#: says nothing. Reported rather than silently accepted.
MIN_MATCHES = 20


def _psnr_masked(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)[mask]
    got = np.asarray(candidate, dtype=np.float64)[mask]
    if ref.size == 0:
        return float("nan")
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def _first_frame(video: str, scene: str) -> np.ndarray:
    pngs = sorted((BP21_CLIPS / video / scene / "window").glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"{video}/{scene}")
    return load_rgb_stack(pngs[:1])[0]


def register(video: str, scene_a: str, scene_b: str) -> dict[str, Any]:
    plate_a = _first_frame(video, scene_a)
    plate_b = _first_frame(video, scene_b)
    grey_a = cv2.cvtColor(plate_a, cv2.COLOR_RGB2GRAY)
    grey_b = cv2.cvtColor(plate_b, cv2.COLOR_RGB2GRAY)

    detector = cv2.SIFT_create(nfeatures=4000)
    kp_a, des_a = detector.detectAndCompute(grey_a, None)
    kp_b, des_b = detector.detectAndCompute(grey_b, None)
    if des_a is None or des_b is None:
        return {"pair": f"{video} {scene_a}->{scene_b}", "note": "no features"}

    matcher = cv2.BFMatcher()
    raw = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    whole = np.ones(plate_a.shape[:2], dtype=bool)
    before = _psnr_masked(plate_b, plate_a, whole)

    result: dict[str, Any] = {
        "pair": f"{video} {scene_a} -> {scene_b}",
        "keypoints": [len(kp_a), len(kp_b)],
        "good_matches": len(good),
        "psnr_before_dB": before,
    }
    if len(good) < MIN_MATCHES:
        result["verdict"] = (
            f"only {len(good)} good matches; a homography here would be fitted "
            "to noise. The two plates do not share enough structure to register."
        )
        return result

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None:
        result["verdict"] = "RANSAC found no consistent homography."
        return result

    height, width = plate_b.shape[:2]
    warped = cv2.warpPerspective(plate_a, homography, (width, height))
    # Only score where the warp actually put pixels; the rest is undefined
    # border and would otherwise decide the number.
    valid = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8), homography, (width, height)
    ).astype(bool)
    coverage = float(valid.mean())
    after = _psnr_masked(plate_b, warped, valid)

    result.update(
        {
            "inliers": int(inliers.sum()) if inliers is not None else None,
            "overlap_fraction": coverage,
            "psnr_after_registration_dB": after,
            "gain_dB": after - before,
            "verdict": (
                "camera geometry: registration recovers most of the gap, so "
                "'register then share' is worth pursuing"
                if after - before > 6.0
                else "content, not geometry: registration does not close the gap, "
                "so one plate cannot serve two points however it is warped"
            ),
        }
    )
    return result


def main() -> int:
    pairs = [
        ("alcaraz_highlights", "scene_000", "scene_010"),
        ("federer_djokovic", "scene_001", "scene_003"),
    ]
    results = []
    for video, a, b in pairs:
        try:
            results.append(register(video, a, b))
        except FileNotFoundError as exc:
            results.append({"pair": f"{video} {a}->{b}", "error": repr(exc)})

    payload = {
        "question": (
            "is the gap between two points' plates camera geometry (removable "
            "by a homography) or content (not removable)?"
        ),
        "both_scenes_are_points": (
            "All four scenes are labelled cluster_point in the dataset's own "
            "scene_metadata.json, with confidence 1.000, 1.000, 0.957 and 0.886."
        ),
        "method": (
            "SIFT features, ratio-test matching, RANSAC homography, warp A onto "
            "B, PSNR scored only inside the warped region's valid mask."
        ),
        "pairs": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for block in results:
        print(f"== {block['pair']}")
        print(f"   matches {block.get('good_matches')}  before {block.get('psnr_before_dB', float('nan')):.2f} dB"
              f"  after {block.get('psnr_after_registration_dB', float('nan')):.2f} dB"
              f"  overlap {block.get('overlap_fraction', float('nan')):.2f}")
        print(f"   -> {block.get('verdict')}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
