"""Fixed K4b candidates and label scoring, frozen before opening human selections.

Feature rows come from real full-clip SAM/YOLO outputs. No image content or labels
are consulted to choose constants. The five candidates are exactly (a), (b),
(c), (e), and the sole prescribed combination without (d), (a)+(c).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.k4b import iou, numbered
from experiments.visor.b1 import file_sha256
from src.segmentation.masks import ClipMasks

RULES: dict[str, Any] = {
    "version": 1, "minimum_presence": .5, "picks": 2,
    "a": "sum consecutive centroid Euclidean distances / median mask height (K4 amendment 1)",
    "b": "sum consecutive absolute horizontal centroid distances; zero weight in the last second",
    "c": "racket-mask contact on >50% of all clip frames; rank by contact fraction; abstain if fewer than two qualify",
    "e": "sum mean shared-confident-joint Euclidean displacements / median mask height; consecutive fitted frames only",
    "a+c": "rank lexicographically by racket contact on most frames, then (a); fill remaining picks by (a)",
    "contact_dilation_sam_pixels": 1, "keypoint_confidence": .5,
    "pose_model": "yolo11x-pose", "pose_image_size": 1280,
    "prompt_track_merge": "greedy one-to-one person/player pairs with mask IoU >=.8 on >50% of co-visible frames; at least half the shorter track co-visible; prefer person masks",
    "tie_order": ["a", "b", "c", "e", "a+c"],
    "within_candidate_tie": "canonical (prompt, native track id)",
    "threshold": .95, "match_iou": .5,
    "none_visible": "correct only when no picked track has a mask on that frame",
    "missing_instance": "visible unlabelled-mask player is an identification failure; never exclude the frame",
}


def features(rows: list[dict[str, Any]], frames: int, fps: float) -> dict[str, Any]:
    """A track's fixed scalar features, with no interpolation across missing frames."""
    if frames <= 0 or fps <= 0 or not rows:
        raise ValueError("A track needs rows, a positive frame count and fps")
    ordered = sorted(rows, key=lambda r: r["t"])
    if len({r["t"] for r in ordered}) != len(ordered) or any(not 0 <= r["t"] < frames for r in ordered):
        raise ValueError("Track frames must be unique and within the clip")
    height = float(np.median([r["height"] for r in ordered]))
    if height <= 0:
        raise ValueError("Mask heights must be positive")
    motion = horizontal = joints = 0.0
    for a, b in zip(ordered, ordered[1:]):
        if b["t"] != a["t"] + 1:
            continue
        motion += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        if b["t"] < frames - fps:
            horizontal += abs(b["x"] - a["x"])
        if a.get("keypoints") is not None and b.get("keypoints") is not None:
            ka, kb = np.asarray(a["keypoints"]), np.asarray(b["keypoints"])
            confident = (ka[:, 2] >= .5) & (kb[:, 2] >= .5)
            if confident.any():
                joints += float(np.linalg.norm(kb[confident, :2] - ka[confident, :2], axis=1).mean())
    contact = sum(bool(r["contact"]) for r in ordered) / frames
    return {"presence": len(ordered) / frames, "a": motion / height, "b": horizontal,
            "c": contact, "e": joints / height, "racket_most": contact > .5}


def choose(tracks: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    eligible = sorted(k for k, v in tracks.items() if v["presence"] >= .5)
    out = {}
    for candidate in RULES["tie_order"]:
        pool = [k for k in eligible if tracks[k]["racket_most"]] if candidate == "c" else eligible
        def key(track: str) -> tuple[Any, ...]:
            value = tracks[track]
            if candidate == "a+c":
                return (-int(value["racket_most"]), -value["a"], track)
            return (-value[candidate], track)
        out[candidate] = sorted(pool, key=key)[:2]
    return out


def merge_prompt_tracks(counts: dict[str, int], pairs: dict[tuple[str, str], dict[str, float]]) -> dict[str, str]:
    """Full-clip, one-to-one aliases for duplicate prompt tracks; no marks used."""
    eligible = [(a, b, p) for (a, b), p in pairs.items()
                if p["agree"] > .5 * p["co_visible"]
                and p["co_visible"] >= .5 * min(counts[a], counts[b])]
    aliases = {k: k for k in counts}
    used: set[str] = set()
    for a, b, _ in sorted(eligible, key=lambda r: (-r[2]["iou_sum"] / r[2]["co_visible"], r[0], r[1])):
        if a in used or b in used:
            continue
        if not a.startswith("person:") or not b.startswith("player:"):
            raise ValueError("Only person/player prompt tracks may be merged")
        aliases[b] = a
        used.update((a, b))
    return aliases


def correct(picks: list[np.ndarray], truth: list[np.ndarray], visible: int, missing: bool) -> bool:
    """Injective IoU matching: one prediction cannot explain both marked players."""
    if missing:
        return False
    if len(truth) != visible or visible not in (0, 1, 2):
        raise ValueError("Confirmed visibility must equal the number of marked masks")
    if visible == 0:
        return not picks
    if visible == 1:
        return any(iou(p, truth[0]) >= .5 for p in picks)
    return any(i != j and iou(a, truth[0]) >= .5 and iou(b, truth[1]) >= .5
               for i, a in enumerate(picks) for j, b in enumerate(picks))


def validate_marks(marks: dict[str, Any], review: dict[str, Any], review_path: Path) -> dict[str, dict[str, Any]]:
    if (marks.get("schema") != 1 or marks.get("selection_sha256") != review["selection_sha256"]
            or marks.get("result_sha256") != file_sha256(review_path)):
        raise ValueError("Marks do not identify this exact review output")
    frames = {r["id"]: r for r in review["frames"]}
    labels = marks.get("marks", {})
    if set(labels) != set(frames):
        raise ValueError("Marks must cover every review frame, without unknown IDs")
    for frame_id, m in labels.items():
        ids = {r["id"] for r in frames[frame_id]["instances"]}
        selected = m.get("selected")
        visible, missing = m.get("visible"), m.get("missing_instance")
        if (m.get("confirmed") is not True or not isinstance(selected, list)
                or any(type(n) is not int or n not in ids for n in selected)
                or len(set(selected)) != len(selected) or len(selected) > 2
                or type(visible) is not int or visible not in (0, 1, 2) or type(missing) is not bool
                or len(selected) > visible or (not missing and len(selected) != visible)
                or (missing and len(selected) >= visible)):
            raise ValueError(f"Invalid or incomplete annotation: {frame_id}")
    return labels


def proposal_oracle(labels: dict[str, dict[str, Any]], review: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Best selection of the review's proposals, independently per frame.

    This bounds rules restricted to these exact proposals. It does not bound
    re-segmentation of the full clips, which may supply missing masks.
    """
    rows = []
    for frame in review["frames"]:
        clip = ClipMasks.load(directory / frame["masks"])
        if file_sha256(directory / frame["masks"]) != frame["masks_sha256"]:
            raise ValueError("Review mask identity changed")
        by_id = {r["id"]: r["mask"] for r in numbered(clip)}
        m = labels[frame["id"]]
        truth = [by_id[n] for n in m["selected"]]
        rows.append({"id": frame["id"], "possible": correct(truth, truth, m["visible"], m["missing_instance"])})
    possible = sum(r["possible"] for r in rows)
    return {"scope": "the review's exact proposals only; does not gate new full-clip SAM proposals",
            "frames": len(rows), "possible": possible, "share": possible / len(rows), "rows": rows}
