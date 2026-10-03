"""Bounded foreground diagnostics; all model work is dispatched via fleet.

This entry point intentionally accepts only the small cuts specified in
demo/docs/foreground-smoke-plan.md. It does not call legacy unbounded CLIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

import cv2
import numpy as np

from demo.pipeline.foreground_codec_v2 import (
    CausalTrackAssociator,
    HandDetection,
    TrackedHand,
    decode_segment,
    encode_segment,
)

DATASET_ROOT = Path("/home/itec/emanuele/Datasets/pointstream-demo")
RECORDINGS = {
    "clip_01": ("clip_01_factory001_worker001_00001/f000000-f035129", 35130),
    "clip_03": ("clip_03_factory001_worker001_00000/f000000-f012629", 12630),
    "factory002": ("factory002_worker001_00000/f000000-f035129", 35130),
}
IMAGE_CANDIDATES_PER_RECORDING = 24
FIT_FRAMES_PER_ARM = 120
FIT_SECONDS_PER_ARM = 120
PROFILE_IMAGE_LIMIT = 16
COMPARE_FRAME_LIMIT = 8
SAM_FRAME_LIMIT = 30
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
)


def validate_limits(args: argparse.Namespace) -> None:
    """Reject broad runs at the CLI boundary, before loading any data/model."""
    if args.command == "audit":
        if args.max_candidates_per_recording < 1 or args.max_candidates_per_recording > IMAGE_CANDIDATES_PER_RECORDING:
            raise ValueError("audit is capped at 24 pixel-inspected candidates per recording")
    elif args.command == "packet":
        if not 1 <= args.max_frames <= COMPARE_FRAME_LIMIT:
            raise ValueError("packet smoke is capped at 8 frames")
    elif args.command == "fit":
        if not 1 <= args.max_steps <= FIT_FRAMES_PER_ARM or not 1 <= args.seconds_per_arm <= FIT_SECONDS_PER_ARM:
            raise ValueError("fit smoke is capped at 120 steps and 120 seconds per arm")
    elif args.command == "profile":
        if not 1 <= args.max_images <= PROFILE_IMAGE_LIMIT:
            raise ValueError("profile smoke is capped at 16 images")
        if args.sam_frames < 0 or args.sam_frames not in (0, SAM_FRAME_LIMIT):
            raise ValueError("optional SAM profiling is either disabled or exactly 30 frames")
    elif args.command == "compare":
        if not 1 <= args.max_frames <= COMPARE_FRAME_LIMIT:
            raise ValueError("comparison smoke is capped at 8 frames per hold-out")
        if args.resolutions != [64, 128, 256]:
            raise ValueError("diagnostic AV1 resolutions are fixed at 64, 128, and 256")


def _bounded_read(path: Path, timeout_s: float = 90.0) -> bytes:
    """Read one file with the plan's hard blocking-I/O timeout."""
    result: dict[str, object] = {}

    def read() -> None:
        try:
            result["value"] = path.read_bytes()
        except BaseException as exc:  # return the original filesystem error
            result["error"] = exc

    worker = threading.Thread(target=read, name="foreground-bounded-read", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"file read exceeded {timeout_s:.0f}s: {path}")
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["value"]  # type: ignore[return-value]


def _source_provenance() -> dict:
    source_root = Path(__file__).resolve().parents[2]
    dispatch_path = source_root / "dispatch.json"
    if not dispatch_path.is_file():
        return {"dispatch_manifest": "unavailable"}
    try:
        dispatch = json.loads(_bounded_read(dispatch_path).decode("utf-8"))
    except (OSError, ValueError, TimeoutError) as exc:
        return {"dispatch_manifest": "unreadable", "error": str(exc)}
    return {
        "git_head": dispatch.get("git_metadata", {}).get("git_head"),
        "tracked_worktree_patch_sha256": dispatch.get("git_metadata", {}).get("tracked_worktree_patch_sha256"),
        "included_untracked_sha256": dispatch.get("git_metadata", {}).get("included_untracked_sha256", {}),
        "snapshot_sha256": dispatch.get("git_metadata", {}).get("snapshot_sha256"),
    }


def _geometry_flags(hand: dict, width: int, height: int) -> tuple[list[str], dict]:
    flags: list[str] = []
    points = np.asarray(hand.get("landmarks_pixel", []), dtype=np.float64)
    bbox = np.asarray(hand.get("box", []), dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all() or bbox.shape != (4,) or not np.isfinite(bbox).all():
        return ["invalid_points_or_bbox"], {"distinct_positions": None, "bone_lengths_px": None}
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0 or x1 > width or y1 > height:
        flags.append("zero_area_or_outside_bbox")
    distinct = int(len(np.unique(np.round(points, 1), axis=0)))
    if distinct < 8:
        flags.append("fewer_than_8_distinct_positions")
    lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in CONNECTIONS]
    positive = [length for length in lengths if length > 0]
    if any(length <= 0.5 for length in lengths):
        flags.append("collapsed_bone")
    median = float(np.median(positive)) if positive else 0.0
    if positive and max(positive) > max(25.0, median * 3.0):
        flags.append("extreme_relative_bone_length")
    clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
    if clipped:
        flags.append("bbox_edge_clipping")
    return flags, {"distinct_positions": distinct, "bone_lengths_px": lengths}


def _sample_by_second(items: list[dict], limit: int, seed: int) -> list[dict]:
    if len(items) <= limit:
        return sorted(items, key=lambda item: (item["frame_idx"], item["candidate_id"]))
    rng = random.Random(seed)
    per_second: dict[int, list[dict]] = {}
    for item in items:
        per_second.setdefault(item["frame_idx"] // 30, []).append(item)
    for rows in per_second.values():
        rng.shuffle(rows)
    seconds = sorted(per_second)
    picked: list[dict] = []
    while len(picked) < limit:
        progress = False
        for second in seconds:
            if per_second[second] and len(picked) < limit:
                picked.append(per_second[second].pop())
                progress = True
        if not progress:
            break
    return sorted(picked, key=lambda item: (item["frame_idx"], item["candidate_id"]))


def _box_iou(first: list[float], second: list[float]) -> float:
    x1=max(float(first[0]),float(second[0])); y1=max(float(first[1]),float(second[1]))
    x2=min(float(first[2]),float(second[2])); y2=min(float(first[3]),float(second[3]))
    intersection=max(0.0,x2-x1)*max(0.0,y2-y1)
    area_a=max(0.0,float(first[2])-float(first[0]))*max(0.0,float(first[3])-float(first[1]))
    area_b=max(0.0,float(second[2])-float(second[0]))*max(0.0,float(second[3])-float(second[1]))
    union=area_a+area_b-intersection
    return intersection/union if union>0 else 0.0


def _closest_component(mask: np.ndarray, bbox: list[float]) -> tuple[np.ndarray | None, int | None]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 8).astype(np.uint8), 8)
    x1, y1, x2, y2 = bbox
    bx1 = max(0, min(mask.shape[1], int(math.floor(x1))))
    by1 = max(0, min(mask.shape[0], int(math.floor(y1))))
    bx2 = max(bx1 + 1, min(mask.shape[1], int(math.ceil(x2))))
    by2 = max(by1 + 1, min(mask.shape[0], int(math.ceil(y2))))
    options = []
    for label in range(1, count):
        component = labels[by1:by2, bx1:bx2] == label
        overlap = int(component.sum())
        if overlap:
            options.append((overlap, int(stats[label, cv2.CC_STAT_AREA]), label))
    if not options:
        return None, None
    _overlap, _area, chosen = max(options)
    return labels == chosen, chosen


def _inside_fraction(points: np.ndarray, component: np.ndarray | None) -> float | None:
    if component is None:
        return None
    h, w = component.shape
    hits = 0
    for x, y in points:
        xi, yi = int(math.floor(float(x) + 0.5)), int(math.floor(float(y) + 0.5))
        if 0 <= xi < w and 0 <= yi < h and component[yi, xi]:
            hits += 1
    return hits / max(1, len(points))


def _record_candidate_summary(item: dict, clip: str) -> dict:
    hand = item["hand"]
    return {key: value for key, value in item.items() if key != "hand"} | {
        "bbox": hand.get("box"),
        "landmarks_pixel": hand.get("landmarks_pixel"),
        "handedness": hand.get("side"),
    }


def _candidate_group(flags: list[str], inside: float | None, distinct: int | None) -> str:
    if flags or (inside is not None and inside < 0.5):
        if distinct is not None and distinct in (8, 9, 10) and len(flags) <= 1:
            return "ambiguous"
        return "flagged"
    if inside is None or inside < 0.7 or distinct is None or distinct <= 12:
        return "ambiguous"
    return "plausible"


def audit_dataset(dataset_root: Path, output_dir: Path, seed: int = 1234, max_per_recording: int = 24) -> Path:
    """Write a bounded geometry audit and pending-label contact sheets."""
    if not 1 <= max_per_recording <= IMAGE_CANDIDATES_PER_RECORDING:
        raise ValueError("audit is capped at 24 pixel-inspected candidates per recording")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    for clip, (relative, total_frames) in RECORDINGS.items():
        folder = dataset_root / relative
        poses_path = folder / "sam_poses" / "rtmw-l.json"
        pose_rows = json.loads(_bounded_read(poses_path).decode("utf-8"))
        candidates = []
        for row in pose_rows:
            frame = int(row["frame_idx"])
            for index, hand in enumerate(row.get("hands", [])):
                flags, measurements = _geometry_flags(hand, 1920, 1080)
                candidate_id = str(hand.get("candidate_id", f"{clip}:{frame}:{index}"))
                candidates.append({
                    "recording": clip, "frame_idx": frame, "source_second": frame // 30,
                    "candidate_id": candidate_id, "file": row.get("file", f"{frame:06d}.jpg"),
                    "hand": hand, "flags": flags, "measurements": measurements,
                    "raw_mean_score": hand.get("confidence"), "raw_joint_scores": hand.get("joint_scores"),
                    "legacy_row_flags": {key: row.get(key) for key in ("aisle", "look")},
                })

        # Causal temporal jump flag. It is a review aid only: tracking does not
        # make the candidate a correct hand or assign a semantic side.
        previous_center: dict[int, tuple[int, float, float]] = {}
        by_frame: dict[int, list[dict]] = {}
        for item in candidates:
            by_frame.setdefault(item["frame_idx"], []).append(item)
        for frame_items in by_frame.values():
            valid_boxes = [item for item in frame_items if not item["flags"]]
            for index, first in enumerate(valid_boxes):
                for second in valid_boxes[index + 1:]:
                    first_box = first["hand"].get("box", [])
                    second_box = second["hand"].get("box", [])
                    if len(first_box) != 4 or len(second_box) != 4:
                        continue
                    overlap = _box_iou(first_box, second_box)
                    if overlap >= 0.85:
                        first["flags"].append("duplicate_candidate_box")
                        second["flags"].append("duplicate_candidate_box")
                        first["measurements"]["duplicate_box_iou"] = overlap
                        second["measurements"]["duplicate_box_iou"] = overlap
        associator = CausalTrackAssociator(1920, 1080)
        for row in sorted(pose_rows, key=lambda item: int(item["frame_idx"])):
            frame = int(row["frame_idx"])
            frame_items = by_frame.get(frame, [])
            valid_items = [item for item in frame_items if not item["flags"]]
            detections = []
            for item in valid_items:
                hand = item["hand"]
                side = str(hand.get("side", "")).lower()
                detections.append(HandDetection(
                    tuple(map(float, hand["box"])),
                    tuple(tuple(map(float, point[:2])) for point in hand["landmarks_pixel"]),
                    side if side in ("left", "right") else None, None, item["candidate_id"],
                ))
            assignment = associator.assign(detections)
            for tracked in assignment.hands:
                cx = (tracked.bbox[0] + tracked.bbox[2]) / 2
                cy = (tracked.bbox[1] + tracked.bbox[3]) / 2
                old = previous_center.get(tracked.track_id)
                if old and 0 < frame - old[0] <= 30:
                    jump = math.hypot(cx - old[1], cy - old[2]) / math.hypot(1920, 1080)
                    if jump > 0.25:
                        item = next((candidate for candidate in valid_items
                                     if tuple(map(float, candidate["hand"]["box"])) == tracked.bbox
                                     and tuple(tuple(map(float, p[:2])) for p in candidate["hand"]["landmarks_pixel"]) == tracked.joints), None)
                        if item is not None:
                            item["flags"].append("abrupt_temporal_change")
                            item["measurements"]["normalized_center_jump"] = jump
                            item["measurements"]["source_frame_gap"] = frame - old[0]
                previous_center[tracked.track_id] = (frame, cx, cy)

        # Geometry-only strata select the bounded image sample. Mask evidence
        # may refine a stratum but is never considered a semantic hand label.
        strata = {name: [] for name in ("plausible", "flagged", "ambiguous")}
        for item in candidates:
            group = _candidate_group(item["flags"], None, item["measurements"]["distinct_positions"])
            item["sample_group"] = group
            strata[group].append(item)
        sample = []
        for offset, (group, rows) in enumerate(strata.items()):
            bucket_cap = min(8, math.ceil(max_per_recording / 3))
            sample.extend(_sample_by_second(rows, bucket_cap, seed + offset))
        # If a stratum is small, fill remaining slots from other strata without
        # duplicating candidate records.
        if len(sample) < max_per_recording:
            selected = {item["candidate_id"] for item in sample}
            remainder = [item for item in candidates if item["candidate_id"] not in selected]
            sample.extend(_sample_by_second(remainder, max_per_recording - len(sample), seed + 17))
        sample = sample[:max_per_recording]
        if clip == "clip_03":
            frame_zero = next((item for item in candidates if item["frame_idx"] // 30 == 0), None)
            if frame_zero is not None and all(item["candidate_id"] != frame_zero["candidate_id"] for item in sample):
                if len(sample) >= max_per_recording:
                    sample[-1] = frame_zero
                else:
                    sample.append(frame_zero)
                sample.sort(key=lambda item: (item["frame_idx"], item["candidate_id"]))

        batches_path = folder / "sam_poses" / "batches.json"
        batch_info = {"path": str(batches_path), "exists": batches_path.is_file()}
        if batches_path.is_file():
            try:
                batch_bytes = _bounded_read(batches_path)
                batch_info["sha256"] = hashlib.sha256(batch_bytes).hexdigest()
                batches_value = json.loads(batch_bytes.decode("utf-8"))
                batch_info["top_level_type"] = type(batches_value).__name__
                if isinstance(batches_value, dict):
                    batch_info["top_level_keys"] = sorted(map(str, batches_value.keys()))[:64]
                    batch_info["entry_count"] = len(batches_value)
                elif isinstance(batches_value, list):
                    batch_info["entry_count"] = len(batches_value)
            except (OSError, ValueError, TimeoutError) as exc:
                batch_info["read_error"] = str(exc)

        provisional_anchor: np.ndarray | None = None
        clip_panels = []
        displayed_by_group = {"plausible": 0, "flagged": 0, "ambiguous": 0}
        for item in sample:
            frame_index = int(item["frame_idx"])
            item["review_label"] = None
            item["review_status"] = "pending_human"
            item["split"] = "factory002_audit_only" if clip == "factory002" else "unassigned"
            final_holdout = frame_index >= total_frames - 300
            recipe_excluded = clip == "clip_03" and frame_index // 30 in {210, 240, 420}
            item["training_eligible"] = bool(clip != "factory002" and not final_holdout and not recipe_excluded)
            item["training_excluded_reason"] = (
                "factory002_audit_only" if clip == "factory002" else
                "final_last_10_seconds" if final_holdout else
                "clip3_excluded_source_second" if recipe_excluded else None
            )
            item["clip3_second_zero_look_override"] = bool(
                clip == "clip_03" and frame_index // 30 == 0 and item["legacy_row_flags"].get("look")
            )
            image_path = folder / "original" / item["file"]
            mask_path = folder / "masks" / f"{Path(item['file']).stem}.png"
            try:
                image_bytes = _bounded_read(image_path)
                mask_bytes = _bounded_read(mask_path)
            except (OSError, TimeoutError) as exc:
                item["pixel_review_error"] = str(exc)
                item["review_status"] = "pixel_review_unavailable"
                records.append(_record_candidate_summary(item, clip))
                continue
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                item["pixel_review_error"] = "image_or_mask_unreadable"
                item["review_status"] = "pixel_review_unavailable"
                records.append(_record_candidate_summary(item, clip))
                continue
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            raw_box = np.asarray(item["hand"].get("box", []), dtype=np.float64)
            if raw_box.shape != (4,) or not np.isfinite(raw_box).all() or raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]:
                item["pixel_review_error"] = "invalid_bbox_no_crop"
                item["review_status"] = "pixel_review_unavailable"
                item["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                item["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
                records.append(_record_candidate_summary(item, clip))
                continue
            component, label = _closest_component(mask, list(map(float, raw_box)))
            points = np.asarray(item["hand"].get("landmarks_pixel", []), dtype=np.float64)
            inside = _inside_fraction(points, component) if points.shape == (21, 2) and np.isfinite(points).all() else None
            item["mask_component_label"] = label
            item["inside_fraction_own_component"] = inside
            if inside is not None and inside < 0.5:
                item["flags"].append("inside_fraction_below_0.5")
            item["sample_group"] = _candidate_group(item["flags"], inside, item["measurements"]["distinct_positions"])
            if displayed_by_group[item["sample_group"]] >= 8:
                item["review_status"] = "not_on_contact_sheet"
            else:
                displayed_by_group[item["sample_group"]] += 1
            item["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
            item["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
            x1, y1, x2, y2 = [int(round(float(v))) for v in item["hand"]["box"]]
            x1=max(0,min(image.shape[1]-1,x1)); x2=max(x1+1,min(image.shape[1],x2))
            y1=max(0,min(image.shape[0]-1,y1)); y2=max(y1+1,min(image.shape[0],y2))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                item["flags"].append("zero_area_crop")
                item["review_status"] = "pixel_review_unavailable"
                records.append(_record_candidate_summary(item, clip))
                continue
            crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)
            mask_tile = cv2.resize((component[y1:y2, x1:x2].astype(np.uint8) * 255) if component is not None else np.zeros((y2-y1,x2-x1),np.uint8), (192,192), interpolation=cv2.INTER_NEAREST)
            mask_tile = cv2.cvtColor(mask_tile, cv2.COLOR_GRAY2BGR)
            overlay = crop.copy()
            for px, py in points if points.shape == (21,2) and np.isfinite(points).all() else []:
                cx = int((px-x1)/max(1,x2-x1)*192); cy = int((py-y1)/max(1,y2-y1)*192)
                if 0 <= cx < 192 and 0 <= cy < 192:
                    cv2.circle(overlay, (cx,cy), 2, (0,255,255), -1)
            if provisional_anchor is None and item["sample_group"] == "plausible":
                provisional_anchor = crop.copy()
            anchor = provisional_anchor if provisional_anchor is not None else np.full_like(crop, 128)
            tiles = [crop, mask_tile, overlay, anchor]
            anchor_title = "provisional anchor" if provisional_anchor is not None else "anchor pending"
            for tile, title in zip(tiles, ("crop", "mask", "pose", anchor_title)):
                cv2.putText(tile, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
            panel = np.concatenate(tiles, axis=1)
            cv2.putText(panel, f"{clip} f{item['frame_idx']} {item['sample_group']} {item['flags']}", (4, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)
            if item["review_status"] == "pending_human":
                clip_panels.append(panel)
            records.append(_record_candidate_summary(item, clip))
        if clip_panels:
            sheet = np.concatenate(clip_panels, axis=0)
            sheet_path = output_dir / f"audit-sheet-{clip}.jpg"
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"failed to write contact sheet: {sheet_path}")
        for row in records:
            if row.get("recording") == clip:
                row["batches_manifest"] = batch_info

    manifest = {
        "schema": "pointstream.foreground.audit.v1", "seed": seed,
        "dataset_root": str(dataset_root), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_provenance": _source_provenance(),
        "max_pixel_inspected_candidates_per_recording": max_per_recording,
        "notes": ["Heuristic flags and mask agreement are diagnostics, not labels.",
                  "review_label remains null until a human assigns visible_hand, non_hand, or uncertain.",
                  "No source data or legacy manifest was modified."],
        "candidates": records,
    }
    path = output_dir / "audit-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _read_json(path: Path) -> dict:
    value = json.loads(_bounded_read(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


def run_packet(input_path: Path, output_dir: Path, max_frames: int) -> dict:
    """Pack already selected/track-assigned frames with all three byte modes."""
    input_bytes = _bounded_read(input_path)
    value = json.loads(input_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    frames_in = value.get("frames")
    if not isinstance(frames_in, list) or len(frames_in) > max_frames:
        raise ValueError("packet input must contain no more than the capped frame count")
    width, height = int(value["width"]), int(value["height"])
    start_frame = int(value.get("start_frame", 0))
    frames: list[list[TrackedHand]] = []
    for frame in frames_in:
        hands = []
        for row in frame:
            hands.append(TrackedHand(
                int(row["track_id"]), row.get("handedness"),
                tuple(map(float, row["bbox"])), tuple(tuple(map(float, point[:2])) for point in row["joints"]),
            ))
        frames.append(hands)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "pointstream.foreground.packet-smoke.v1", "input_sha256": hashlib.sha256(input_bytes).hexdigest(), "methods": {}}
    for method in ("raw", "zlib", "delta_zlib"):
        packet = encode_segment(frames, width=width, height=height, start_frame=start_frame, method=method)
        decoded = decode_segment(packet)
        path = output_dir / f"segment-{method}.psfg"
        path.write_bytes(packet)
        report["methods"][method] = {"bytes": len(packet), "file_bytes": path.stat().st_size, "roundtrip_frames": len(decoded["frames"])}
    (output_dir / "packet-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--max-candidates-per-recording", type=int, default=24)
    audit.add_argument("--seed", type=int, default=1234)
    packet = commands.add_parser("packet")
    packet.add_argument("--input", type=Path, required=True)
    packet.add_argument("--output-dir", type=Path, required=True)
    packet.add_argument("--max-frames", type=int, default=8)
    fit = commands.add_parser("fit")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--max-steps", type=int, default=120)
    fit.add_argument("--seconds-per-arm", type=int, default=120)
    fit.add_argument("--freeze-only", action="store_true",
                     help="validate human labels, freeze split, and serialize per-crop packets without training")
    profile = commands.add_parser("profile")
    profile.add_argument("--images", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--max-images", type=int, default=16)
    profile.add_argument("--sam-frames", type=int, default=0)
    compare = commands.add_parser("compare")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--max-frames", type=int, default=8)
    compare.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256])
    return parser


def _require_job_output(path: Path) -> Path:
    raw_job_dir = os.environ.get("PS_JOB_DIR")
    if not raw_job_dir:
        raise ValueError("output must be under the dispatcher-assigned PS_JOB_DIR")
    output = path.resolve()
    job_dir = Path(raw_job_dir).resolve()
    if output == job_dir or job_dir not in output.parents:
        raise ValueError("choose a new output directory below PS_JOB_DIR")
    return output


def freeze_training_split(audit_manifest: dict, output_dir: Path) -> dict:
    """Freeze only human-reviewed crops and reject source-second leakage."""
    if audit_manifest.get("schema") != "pointstream.foreground.audit.v1":
        raise ValueError("unsupported audit manifest schema")
    eligible = []
    for row in audit_manifest.get("candidates", []):
        if row.get("recording") not in ("clip_01", "clip_03"):
            continue
        if row.get("review_status") != "reviewed" or row.get("review_label") != "visible_hand":
            continue
        if not row.get("training_eligible"):
            continue
        if row.get("image_sha256") is None or row.get("mask_sha256") is None:
            continue
        bbox = row.get("bbox")
        points = row.get("landmarks_pixel")
        if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(points, list) or len(points) != 21:
            continue
        eligible.append(dict(row))

    # A source second is an indivisible split unit. Within a second, retain at
    # most one reviewed crop per available handedness; handedness remains
    # metadata and never supplies the packet track ID.
    by_clip_second: dict[tuple[str, int], list[dict]] = {}
    for row in eligible:
        key = (row["recording"], int(row["source_second"]))
        by_clip_second.setdefault(key, []).append(row)
    grouped: dict[str, list[tuple[tuple[str, int], list[dict]]]] = {"clip_01": [], "clip_03": []}
    for key, rows in by_clip_second.items():
        by_side: dict[str, list[dict]] = {}
        for row in rows:
            side = str(row.get("handedness") or "unknown").lower()
            side = side if side in ("left", "right") else "unknown"
            by_side.setdefault(side, []).append(row)
        chosen = []
        for side in ("left", "right", "unknown"):
            if by_side.get(side):
                chosen.append(sorted(by_side[side], key=lambda item: item["candidate_id"])[0])
        grouped[key[0]].append((key, chosen))
    for clip in grouped:
        grouped[clip].sort(key=lambda pair: (pair[0][1], pair[1][0]["candidate_id"] if pair[1] else ""))

    def pick_groups(limit: int, queues: dict[str, list], used: set[tuple[str, int]]) -> tuple[list[dict], set[tuple[str, int]]]:
        result: list[dict] = []
        selected_seconds: set[tuple[str, int]] = set()
        positions = {clip: 0 for clip in queues}
        clips = sorted(queues)
        progress = True
        while len(result) < limit and progress:
            progress = False
            for clip in clips:
                while positions[clip] < len(queues[clip]) and queues[clip][positions[clip]][0] in used | selected_seconds:
                    positions[clip] += 1
                if positions[clip] >= len(queues[clip]) or len(result) >= limit:
                    continue
                key, rows = queues[clip][positions[clip]]
                positions[clip] += 1
                slots = limit - len(result)
                result.extend(rows[:slots])
                selected_seconds.add(key)
                progress = True
        return result, selected_seconds

    available_crop_count = sum(len(rows) for values in grouped.values() for _key, rows in values)
    fit_limit = min(16, max(8, available_crop_count - 4))
    fit, fit_seconds = pick_groups(fit_limit, grouped, set())
    validation, validation_seconds = pick_groups(8, grouped, fit_seconds)
    if len(fit) < 8 or len(validation) < 4:
        raise ValueError(
            f"F1 review gate failed: need >=8 fit and >=4 validation visible-hand crops; "
            f"found {len(fit)} and {len(validation)}"
        )
    if fit_seconds & validation_seconds:
        raise AssertionError("source seconds leaked across fit and validation")

    output_dir.mkdir(parents=True, exist_ok=False)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir()

    def canonicalize(rows: list[dict], split: str) -> list[dict]:
        result = []
        for index, row in enumerate(rows):
            side = str(row.get("handedness") or "").lower()
            side = side if side in ("left", "right") else None
            hand = TrackedHand(
                0,
                side,
                tuple(map(float, row["bbox"])),
                tuple(tuple(map(float, point[:2])) for point in row["landmarks_pixel"]),
            )
            packet = encode_segment([[hand]], width=1920, height=1080,
                                    start_frame=int(row["frame_idx"]), method="raw")
            decoded = decode_segment(packet)["frames"][0][0]
            packet_path = packet_dir / f"{split}-{index:02d}.psfg"
            packet_path.write_bytes(packet)
            result.append({
                **row,
                "split": split,
                "packet_path": str(packet_path),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "packet_bytes": len(packet),
                "decoded_track_id": decoded.track_id,
                "decoded_handedness": decoded.handedness,
                "decoded_bbox": list(decoded.bbox),
                "decoded_joints": [list(point) for point in decoded.joints],
            })
        return result

    fit_rows = canonicalize(fit, "fit")
    validation_rows = canonicalize(validation, "validation")
    anchor_candidates = {}
    for row in fit_rows:
        anchor_candidates.setdefault(row["recording"], row["candidate_id"])
    frozen = {
        "schema": "pointstream.foreground.fit-split.v1",
        "audit_source_provenance": audit_manifest.get("source_provenance"),
        "fit_source_seconds": sorted([list(key) for key in fit_seconds]),
        "validation_source_seconds": sorted([list(key) for key in validation_seconds]),
        "anchors_from_fit_only": anchor_candidates,
        "fit": fit_rows,
        "validation": validation_rows,
        "gate": {"fit_count": len(fit_rows), "validation_count": len(validation_rows),
                 "passed": len(fit_rows) >= 8 and len(validation_rows) >= 4},
        "notes": ["All selected labels were human-reviewed visible_hand.",
                  "Every crop is encoded then decoded before later conditioning.",
                  "No final hold-out or factory002 crop enters these sets."],
    }
    (output_dir / "fit-split-manifest.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_limits(args)
        if args.command == "audit":
            output = _require_job_output(args.output_dir)
            result = {"audit_manifest": str(audit_dataset(args.dataset_root, output, args.seed, args.max_candidates_per_recording))}
        elif args.command == "packet":
            result = run_packet(args.input, _require_job_output(args.output_dir), args.max_frames)
        elif args.command == "fit" and args.freeze_only:
            result = freeze_training_split(_read_json(args.manifest), _require_job_output(args.output_dir))
        else:
            raise RuntimeError(
                f"{args.command} is deliberately gated; this build only supports audit, packet, and fit --freeze-only. "
                "Do not bypass F1 crop review or use a legacy unbounded runner."
            )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
