"""Full-clip, label-blind SAM and YOLO features for frozen K4b candidates.

One clip per fleet job, same path on the prefix smoke and the complete clip.
The input manifest contains no human selections. The full-size converted YUV
is the sole source of JPEGs and pose frames, as in K4. Video is offloaded to
CPU; tracker state stays on GPU. Smoke compares masks to the native path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4, k4b_rules as rules, windows as tw
from experiments.tennis.k4b import iou
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.segmentation.masks import ClipMasks, compress, decompress

PROMPTS = {"person": "person", "player": "tennis player", "racket": "tennis racket"}


def key(class_name: str, track: int) -> str:
    return f"{class_name}:{track:012d}"


def extract_features(clip: ClipMasks, detections: list[dict[str, np.ndarray]], targets: list[int]) -> tuple[dict[str, Any], ClipMasks]:
    """Derive all fixed CPU features, preserving SAM's native identities."""
    from experiments.visor.h1 import match_detections

    if len(clip.frames) != len(detections):
        raise ValueError("Pose and SAM frame coverage disagree")
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    counts: Counter[str] = Counter()
    target_raw: dict[int, list[Any]] = {}
    for t, (instances, det) in enumerate(zip(clip.frames, detections)):
        people = [(key(i.class_name, i.track_id), i, i.mask()) for i in instances if i.class_name in ("person", "player")]
        if t in targets:
            target_raw[t] = [i for _, i, _ in people]
        racket = np.zeros((clip.height, clip.width), bool)
        for inst in instances:
            if inst.class_name == "racket":
                racket |= inst.mask()
        racket = k4.dilate(racket, 1)
        matches = {}
        for prompt in ("person", "player"):
            masks = {k: m for k, i, m in people if i.class_name == prompt}
            matches.update(match_detections(masks, det["boxes"] * .5) if len(det["boxes"]) else {})
        for track, inst, mask in people:
            ys, xs = np.nonzero(mask)
            kp = det["kps"][matches[track]].copy() if track in matches else None
            if kp is not None:
                kp[:, :2] *= .5
            raw[track].append({"t": t, "x": float(xs.mean()), "y": float(ys.mean()),
                               "height": int(ys.max() - ys.min() + 1), "contact": bool((mask & racket).any()),
                               "keypoints": kp.tolist() if kp is not None else None})
            counts[track] += 1
        for a, ai, am in people:
            if ai.class_name != "person":
                continue
            for b, bi, bm in people:
                if bi.class_name != "player":
                    continue
                p = pairs.setdefault((a, b), {"co_visible": 0., "agree": 0., "iou_sum": 0.})
                # Most pairs have disjoint boxes; avoid a full-frame IoU there.
                box_a, box_b = ai.bbox, bi.bbox
                overlaps = box_a[0] < box_b[2] and box_b[0] < box_a[2] and box_a[1] < box_b[3] and box_b[1] < box_a[3]
                overlap = iou(am, bm) if overlaps else 0.
                p["co_visible"] += 1
                p["agree"] += overlap >= .8
                p["iou_sum"] += overlap
    aliases = rules.merge_prompt_tracks(dict(counts), pairs)
    canonical: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for track in sorted(raw):  # person before player: prefer its mask/pose on duplicate frames
        for row in raw[track]:
            canonical[aliases[track]].setdefault(row["t"], row)
    features = {k: rules.features(list(rows.values()), len(clip.frames), clip.fps) for k, rows in canonical.items()}
    ids = {k: n for n, k in enumerate(sorted(canonical))}
    target_clip = ClipMasks(("person",), clip.height, clip.width, clip.fps)
    for t in targets:
        kept: dict[str, Any] = {}
        for inst in sorted(target_raw[t], key=lambda i: (i.class_name, i.track_id)):
            canonical_key = aliases[key(inst.class_name, inst.track_id)]
            kept.setdefault(canonical_key, replace(inst, class_name="person", track_id=ids[canonical_key]))
        target_clip.frames.append(list(kept.values()))
    return {"features": features, "picks": rules.choose(features), "aliases": aliases,
            "track_ids": ids, "target_frames": targets, "raw_track_counts": dict(counts),
            "frame_count": len(clip.frames), "rules": rules.RULES}, target_clip


def rgb_batch(raw: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.stack([tw.yuv_to_rgb(np.asarray(raw[t]), 1920, 1080) for t in range(start, stop)])


def union_masks(clip: ClipMasks, t: int, name: str) -> np.ndarray:
    out = np.zeros((clip.height, clip.width), bool)
    for inst in clip.frames[t]:
        if inst.class_name == name:
            out |= inst.mask()
    return out


def run(args: argparse.Namespace) -> int:
    import cv2
    import torch

    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    manifest = json.loads(Path(args.manifest).read_text())
    info = next(c for c in manifest["clips"] if c["id"] == args.clip)
    frames = min(int(args.frames) or info["frames"], info["frames"])
    if frames < 3:
        raise ValueError("Temporal confirmation requires >=3 source frames")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "16000"))
    torch.cuda.set_per_process_memory_fraction(limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    if file_sha256(Path(args.source)) != info["sha256"]:
        raise ValueError("Original clip identity mismatch")
    window = {"id": info["id"], "size": [1920, 1080], "fps": info["fps"], "source": {"kind": "video"}}
    converted = tw.convert(args.ffmpeg, window, Path(args.source), scratch / "source.yuv", frames)
    raw = np.memmap(scratch / "source.yuv", mode="r", dtype=np.uint8, shape=(frames, 1620, 1920))
    jpeg = scratch / "jpeg"
    jpeg.mkdir()
    poses = k4.Poses([("yolo11x-pose", args.pose)])
    detections = []
    for start in range(0, frames, 8):
        rgb = rgb_batch(raw, start, min(start + 8, frames))
        detections.extend(poses.run("yolo11x-pose", rgb, batch=8))
        for offset, frame in enumerate(rgb):
            small = cv2.resize(frame, k4.SAM_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(jpeg / f"{start + offset:05d}.jpg"), small[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        progress(start + len(rgb))
    pose_record = poses.record()
    del poses, raw, rgb
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    gc.collect()
    torch.cuda.empty_cache()
    parity: dict[str, Any] = {"checked": False}
    if os.environ.get("PS_STAGE") == "smoke":
        parity_jpeg = scratch / "parity-jpeg"
        parity_jpeg.mkdir()
        parity_frames = min(12, frames)
        for p in sorted(jpeg.glob("*.jpg"))[:parity_frames]:
            (parity_jpeg / p.name).symlink_to(p)
        native = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
        print("phase=native-parity frames=" + str(parity_frames), flush=True)
        native_batch = native._require_predictor().model.batched_grounding_batch_size
        if native_batch != 16:
            raise RuntimeError("Unexpected native SAM grounding batch; parity control is invalid")
        reference = native.segment_frames(parity_jpeg, PROMPTS, policy="offline_bidirectional", fps=info["fps"])
        native_config = native.provenance("offline_bidirectional").config_sha256
        del native
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
    sam = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint, offload_video_to_cpu=True, offload_state_to_cpu=False,
                                grounding_batch_size=8)
    if os.environ.get("PS_STAGE") == "smoke":
        print("phase=offloaded-parity grounding_batch=8", flush=True)
        offloaded_reference = sam.segment_frames(parity_jpeg, PROMPTS, policy="offline_bidirectional", fps=info["fps"])
        overlaps = []
        for t in range(parity_frames):
            for name in PROMPTS:
                a, b = union_masks(reference, t, name), union_masks(offloaded_reference, t, name)
                overlaps.append(iou(a, b) if (a | b).any() else 1.)
        parity = {"checked": True, "minimum_union_iou": min(overlaps), "native_config_sha256": native_config,
                  "frames": parity_frames, "prompts": PROMPTS, "native_grounding_batch": native_batch, "grounding_batch": 8,
                  "reference_instances": sum(len(f) for f in reference.frames)}
        reference.save(publish / "native-parity.rle")
        offloaded_reference.save(publish / "offloaded-parity.rle")
        write_json(publish / "parity.json", parity)
        if min(overlaps) < .999:
            raise ValueError("CPU offload/batch size changed SAM masks in the representative smoke")
        del reference, offloaded_reference
    completed_masks: set[tuple[str, int]] = set()
    def on_frame(role: str, t: int) -> None:
        if (role, t) not in completed_masks:
            completed_masks.add((role, t))
            progress(frames + len(completed_masks))
    print("phase=production-sam frames=" + str(frames) + " grounding_batch=8", flush=True)
    clip = sam.segment_frames(jpeg, PROMPTS, policy="offline_bidirectional", fps=info["fps"], on_frame=on_frame)
    provenance = sam.provenance("offline_bidirectional")
    sam_record = {"runtime": runtime(), "checkpoint_sha256": provenance.checkpoint_sha256,
                  "model_revision": provenance.model_revision, "config_sha256": provenance.config_sha256,
                  "sdpa_backend_policy": sam.sdpa_backend_policy, "offload_video_to_cpu": True,
                  "offload_state_to_cpu": False, "grounding_batch_size": sam._require_predictor().model.batched_grounding_batch_size,
                  "prompts": PROMPTS, "parity": parity}
    del sam
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    print("phase=cpu-features", flush=True)
    targets = sorted(t for t in info["review_frames"] if t < frames)
    feature_record, target_clip = extract_features(clip, detections, targets)
    clip.save(publish / "sam-masks.rle")
    target_clip.save(publish / "target-masks.rle")
    (publish / "pose.json.zst").write_bytes(compress(json.dumps([{k: v.tolist() for k, v in d.items()} for d in detections]).encode()))
    write_json(publish / "features.json", feature_record)
    from src.codecs import svtav1

    summary = {"clip": info, "frames": frames, "configured_frames": int(args.frames), "stage": os.environ.get("PS_STAGE"),
               "manifest_sha256": file_sha256(Path(args.manifest)), "conversion": converted, "sam": sam_record, "pose": pose_record,
               "rules": rules.RULES, "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "ffmpeg": svtav1.tool(args.ffmpeg), "files": {p.name: file_sha256(p) for p in sorted(publish.iterdir())}}
    write_json(stage_dir() / "k4b-features.json", summary)
    progress(frames)
    return 0


def validate(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    clip = ClipMasks.load(directory / "sam-masks.rle")
    targets = ClipMasks.load(directory / "target-masks.rle")
    dets = [{k: np.asarray(v, np.float32).reshape((-1, 4) if k == "boxes" else (-1, 17, 3)) for k, v in r.items()}
            for r in json.loads(decompress((directory / "pose.json.zst").read_bytes()))]
    saved = json.loads((directory / "features.json").read_text())
    recomputed, target_again = extract_features(clip, dets, saved["target_frames"])
    sam = summary["sam"]
    checks = {
        "files_identified": all(file_sha256(directory / n) == h for n, h in summary["files"].items()),
        "every_frame_masked_and_posed": len(clip.frames) == len(dets) == summary["frames"] == summary["conversion"]["frames"],
        "full_source_coverage": summary["configured_frames"] != 0 or summary["frames"] == summary["clip"]["frames"],
        "sam_size_and_classes": (clip.width, clip.height) == (960, 540) and set(clip.classes) == set(PROMPTS),
        "sam_nonempty": any(i.class_name in ("person", "player") for f in clip.frames for i in f),
        "sam_pinned": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "sam_native_cuda": sam["sdpa_backend_policy"] == "native_flash" and sam["runtime"]["capability"][0] >= 8,
        "effective_grounding_batch": sam["grounding_batch_size"] == 8,
        "cpu_offload_parity": summary["stage"] != "smoke" or (sam["parity"].get("minimum_union_iou", 0) >= .999
                                                                            and sam["parity"].get("reference_instances", 0) > 0),
        "pose_cuda": all(v.startswith("cuda") for v in summary["pose"]["on_cuda"].values()),
        "pose_finite": all(np.isfinite(v).all() for d in dets for v in d.values()),
        "frozen_candidates": saved["rules"] == summary["rules"] == rules.RULES,
        "features_recomputed": saved == recomputed,
        "targets_complete": saved["target_frames"] == sorted(t for t in summary["clip"]["review_frames"] if t < summary["frames"]),
        "targets_match_full_clip": [[(i.class_name, i.track_id, hashlib.sha256(i.mask().tobytes()).hexdigest()) for i in f] for f in targets.frames]
                                    == [[(i.class_name, i.track_id, hashlib.sha256(i.mask().tobytes()).hexdigest()) for i in f] for f in target_again.frames],
        "features_finite": all(all(np.isfinite(v) for v in f.values()) for f in saved["features"].values()),
    }
    return {"passed": all(checks.values()), "checks": checks, "frames": summary["frames"], "clip": summary["clip"]["id"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for arg in ("manifest", "clip", "source", "sam-checkpoint", "pose", "frames"):
        r.add_argument("--" + arg, required=True)
    r.add_argument("--ffmpeg", default="ffmpeg")
    v = sub.add_parser("validate")
    v.add_argument("--directory", type=Path)
    v.add_argument("--summary", type=Path)
    args = p.parse_args()
    if args.command == "run":
        return run(args)
    summary = json.loads((args.summary or stage_dir() / "k4b-features.json").read_text())
    if args.directory:
        report = validate(args.directory, summary)
    else:
        with tempfile.TemporaryDirectory(prefix="k4b-feature-validate-", dir="/tmp") as temp:
            with tarfile.open(stage_dir() / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            report = validate(Path(temp) / "publish", summary)
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
