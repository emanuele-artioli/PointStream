"""Compare mp_live vs dwpose vs rtm_hand on an optional clip.

Centroid distance when both fire. No MPJPE: MediaPipe 21 and COCO-WholeBody 21
are not the same topology (index-aligned only if forced).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from demo.evaluation.pose_backends import (
    BACKENDS,
    FrameWholeBody,
    wholebody_to_frame_hands,
)
from demo.evaluation.profile_map import profile_map
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand

TOPOLOGY_NOTE = (
    "Centroid distance only. MediaPipe Hands-21 and COCO-WholeBody-21 are not "
    "the same skeleton; this bakeoff does not report MPJPE. Any joint-wise "
    "number would be index-aligned only, not anatomically matched."
)


def as_hand_poses(result: list[Any]) -> list[FrameHandPose]:
    if not result:
        return []
    first = result[0]
    if isinstance(first, FrameHandPose):
        return result
    if isinstance(first, FrameWholeBody):
        return [wholebody_to_frame_hands(frame) for frame in result]
    raise TypeError(f"unsupported pose result type {type(first)!r}")


def n_hands(poses: list[FrameHandPose]) -> int:
    return int(sum(len(p.hands) for p in poses))


def _centroid(hand: SingleHand) -> np.ndarray:
    pts = np.asarray(hand.landmarks_pixel, dtype=np.float64)
    return pts.mean(axis=0)


def centroid_distances(
    a: list[FrameHandPose],
    b: list[FrameHandPose],
    match_px: float = 350.0,
) -> dict[str, float | int]:
    """Greedy centroid matches when both backends fire on the same frame."""
    n = min(len(a), len(b))
    dists: list[float] = []
    n_pairs = 0
    for i in range(n):
        leftover = list(b[i].hands)
        for hand_a in a[i].hands:
            if not leftover:
                break
            ca = _centroid(hand_a)
            best_j = -1
            best_d = match_px
            for j, hand_b in enumerate(leftover):
                d = float(np.linalg.norm(ca - _centroid(hand_b)))
                if d <= best_d:
                    best_d = d
                    best_j = j
            if best_j < 0:
                continue
            leftover.pop(best_j)
            dists.append(best_d)
            n_pairs += 1
    if not dists:
        return {"n_pairs": 0, "mean": 0.0, "p50": 0.0}
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "n_pairs": n_pairs,
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
    }


def _profile_extractor(name: str, clip: Path, frame: np.ndarray) -> float | None:
    extractor = BACKENDS[name]

    def extract_once(img: np.ndarray) -> Any:
        _ = img
        return extractor(clip, 1)

    try:
        stats = profile_map(extract_once, frame, n_warmup=0, n_runs=1)
        return float(stats["extract_ms_p50"])
    except Exception:
        t0 = time.perf_counter()
        try:
            extractor(clip, 1)
        except Exception:
            return None
        return round((time.perf_counter() - t0) * 1000.0, 3)


def run_bakeoff(clip: Path, max_frames: int | None) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"skipped": True, "reason": f"could not read frames from {clip}"}

    names = ("mp_live", "dwpose", "rtm_hand")
    poses_by: dict[str, list[FrameHandPose]] = {}
    backends: dict[str, Any] = {}
    for name in names:
        entry: dict[str, Any] = {"n_hands": 0, "extract_ms_p50": None, "n_frames": 0}
        try:
            raw = BACKENDS[name](clip, max_frames)
            hands = as_hand_poses(raw)
            poses_by[name] = hands
            entry["n_hands"] = n_hands(hands)
            entry["n_frames"] = len(hands)
            entry["extract_ms_p50"] = _profile_extractor(name, clip, frame)
        except Exception as exc:
            entry["error"] = str(exc)
            poses_by[name] = []
        backends[name] = entry

    pairs = {
        "mp_live_vs_dwpose": centroid_distances(poses_by["mp_live"], poses_by["dwpose"]),
        "mp_live_vs_rtm_hand": centroid_distances(poses_by["mp_live"], poses_by["rtm_hand"]),
        "dwpose_vs_rtm_hand": centroid_distances(poses_by["dwpose"], poses_by["rtm_hand"]),
    }
    return {
        "clip": str(clip),
        "skipped": False,
        "note": TOPOLOGY_NOTE,
        "mpjpe": None,
        "mpjpe_note": "omitted; topologies are not anatomically aligned",
        "backends": backends,
        "centroid_px_when_both_fire": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mp_live vs dwpose vs rtm_hand bakeoff.")
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/maps"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    out_json = args.out / "bakeoff_pose.json" if args.out.suffix.lower() != ".json" else args.out
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.clip is None or not args.clip.is_file():
        payload = {
            "skipped": True,
            "reason": "no --clip file",
            "note": TOPOLOGY_NOTE,
            "mpjpe": None,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"skipped (no clip); wrote {out_json}")
        return 0

    result = run_bakeoff(args.clip, args.max_frames)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
