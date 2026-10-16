"""Full-clip, label-blind SAM and YOLO features for frozen K4b candidates.

One clip per fleet job, same path on the prefix smoke and the complete clip.
The input manifest contains no human selections. The full-size converted YUV
is the sole source of JPEGs and pose frames, as in K4. Video and tracker state
use native CPU offload options; completed boolean cache masks are copied to
CPU. Smoke compares masks and track identities to the native path.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import importlib
import inspect
import json
import os
import tarfile
import tempfile
import textwrap
import time
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


def mask_parity(reference: ClipMasks, other: ClipMasks) -> dict[str, Any]:
    if (reference.width, reference.height, len(reference.frames), reference.classes) != (
            other.width, other.height, len(other.frames), other.classes):
        raise ValueError("Parity clips have different dimensions/classes/frame counts")
    unions: list[float] = []
    instances: list[float] = []
    same_ids = True
    for t, frame in enumerate(reference.frames):
        for name in PROMPTS:
            a, b = union_masks(reference, t, name), union_masks(other, t, name)
            unions.append(iou(a, b) if (a | b).any() else 1.)
        a_masks = {(i.class_name, i.track_id): i.mask() for i in frame}
        b_masks = {(i.class_name, i.track_id): i.mask() for i in other.frames[t]}
        same_ids &= a_masks.keys() == b_masks.keys()
        instances.extend(iou(mask, b_masks[key]) if key in b_masks else 0. for key, mask in a_masks.items())
    return {"minimum_union_iou": min(unions), "minimum_instance_iou": min(instances, default=0.),
            "track_ids_match": same_ids, "reference_instances": sum(len(f) for f in reference.frames)}


def offload_tracker_state(predictor: Any) -> dict[str, Any]:
    """Use the pinned tracker's native storage option, absent at top level."""
    tracker = predictor.model.tracker
    original = tracker.init_state
    if "offload_state_to_cpu" not in inspect.signature(original).parameters:
        raise RuntimeError("Pinned tracker lacks native CPU-state storage")
    original_cache = predictor.model._cache_frame_outputs
    record: dict[str, Any] = {"api": "tracker.init_state+cached_frame_outputs", "version": 2,
                              "initialized": 0, "cpu": 0, "cached_masks": 0, "cached_cpu_masks": 0}

    def initialize(*args: Any, **kwargs: Any) -> Any:
        kwargs["offload_state_to_cpu"] = True
        state = original(*args, **kwargs)
        if not state.get("offload_state_to_cpu") or str(state.get("storage_device")) != "cpu":
            raise RuntimeError("Tracker CPU-state request was not effective")
        record["initialized"] += 1
        record["cpu"] += 1
        return state

    tracker.init_state = initialize

    def cache(inference_state: Any, frame_idx: int, *args: Any, **kwargs: Any) -> None:
        original_cache(inference_state, frame_idx, *args, **kwargs)
        masks = inference_state["cached_frame_outputs"][frame_idx]
        for object_id, mask in masks.items():
            stored = mask.cpu()
            if str(stored.device) != "cpu":
                raise RuntimeError("Cached SAM mask did not move to CPU")
            masks[object_id] = stored
            record["cached_masks"] += 1
            record["cached_cpu_masks"] += 1

    predictor.model._cache_frame_outputs = cache
    return record


def cpu_state_compatibility(module: Any) -> dict[str, Any]:
    """Repair two pinned storage helpers' missing device transfers, in memory."""
    replacements = {
        "_merge": ("d2[k2].to(dtype=d1[k1].dtype)", "d2[k2].to(device=d1[k1].device, dtype=d1[k1].dtype)"),
        "_append": ("torch.cat([d1[k1], d2[k2]], dim=dim)",
                    "torch.cat([d1[k1], d2[k2].to(device=d1[k1].device)], dim=dim)"),
    }
    patches = []
    for name, (old, new) in replacements.items():
        original = getattr(module, name)
        source = textwrap.dedent(inspect.getsource(original))
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected pinned SAM {name}; CPU-state compatibility patch rejected")
        revised = source.replace(old, new)
        patch = "".join(difflib.unified_diff(source.splitlines(keepends=True), revised.splitlines(keepends=True),
                                           fromfile=name, tofile=name + ".cpu-state"))
        namespace = dict(original.__globals__)
        namespace.update(vars(module))
        exec(compile(revised, f"<pointstream-k4b-cpu-state-{name}>", "exec"), namespace)
        setattr(module, name, namespace[name])
        patches.append({"function": name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "revised_sha256": hashlib.sha256(revised.encode()).hexdigest(),
                        "patch": patch, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()})
    return {"version": 1, "patches": patches,
            "patch_sha256": hashlib.sha256(json.dumps(patches, sort_keys=True).encode()).hexdigest()}


def estimate_full_seconds(phases: dict[str, float], frames: int, source_frames: int) -> float:
    """Keep first-batch pose warmup fixed; scale measured steady work."""
    first_frames = phases["pose_first_frames"]
    if frames <= first_frames:
        raise ValueError("Scale confirmation needs pose work beyond the first batch")
    variable = ((phases["preparation"] + phases["sam_compute"] + phases["export"]) * source_frames / frames
                + phases["pose_rest"] * (source_frames - first_frames) / (frames - first_frames))
    return 60 + phases["imports"] + phases["sam_load"] + phases["pose_load"] + phases["pose_first"] + 1.25 * variable


def run(args: argparse.Namespace) -> int:
    began = time.perf_counter()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import cv2
    import torch

    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    import_seconds = time.perf_counter() - began
    preparation_start = time.perf_counter()
    if not 0 <= args.full_stage_seconds <= 2700:
        raise ValueError("Full-stage allowance must be between zero and 2700 seconds")
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
    pose_load_start = time.perf_counter()
    poses = k4.Poses([("yolo11x-pose", args.pose)])
    pose_load_seconds = time.perf_counter() - pose_load_start
    pose_first = pose_rest = 0.0
    pose_first_frames = min(8, frames)
    detections = []
    for start in range(0, frames, 8):
        rgb = rgb_batch(raw, start, min(start + 8, frames))
        pose_start = time.perf_counter()
        detections.extend(poses.run("yolo11x-pose", rgb, batch=8))
        if start == 0:
            pose_first = time.perf_counter() - pose_start
        else:
            pose_rest += time.perf_counter() - pose_start
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
    preparation_seconds = time.perf_counter() - preparation_start - pose_load_seconds - pose_first - pose_rest
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
    sam_load_start = time.perf_counter()
    sam = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint, offload_video_to_cpu=True, offload_state_to_cpu=False,
                                grounding_batch_size=16)
    predictor = sam._require_predictor()
    cpu_compatibility = cpu_state_compatibility(importlib.import_module("sam3.model.video_tracking_multiplex"))
    write_json(publish / "cpu-state-compatibility.json", cpu_compatibility)
    state_offload = offload_tracker_state(predictor)
    del predictor
    sam_load_seconds = time.perf_counter() - sam_load_start
    if os.environ.get("PS_STAGE") == "smoke":
        print("phase=offloaded-parity grounding_batch=16", flush=True)
        offloaded_reference = sam.segment_frames(parity_jpeg, PROMPTS, policy="offline_bidirectional", fps=info["fps"])
        parity = {"checked": True, **mask_parity(reference, offloaded_reference), "native_config_sha256": native_config,
                  "frames": parity_frames, "prompts": PROMPTS, "native_grounding_batch": native_batch, "grounding_batch": 16,
                  "reference_instances": sum(len(f) for f in reference.frames)}
        parity["tracker_state_offload"] = dict(state_offload)
        reference.save(publish / "native-parity.rle")
        offloaded_reference.save(publish / "offloaded-parity.rle")
        write_json(publish / "parity.json", parity)
        if not parity["track_ids_match"] or min(parity["minimum_union_iou"], parity["minimum_instance_iou"]) < .999:
            raise ValueError("CPU offload/batch size changed SAM masks in the representative smoke")
        del reference, offloaded_reference
    state_offload.update(initialized=0, cpu=0, cached_masks=0, cached_cpu_masks=0)
    completed_masks: set[tuple[str, int]] = set()
    def on_frame(role: str, t: int) -> None:
        if (role, t) not in completed_masks:
            completed_masks.add((role, t))
            progress(frames + len(completed_masks))
    print("phase=production-sam frames=" + str(frames) + " grounding_batch=16", flush=True)
    sam_compute_start = time.perf_counter()
    clip = sam.segment_frames(jpeg, PROMPTS, policy="offline_bidirectional", fps=info["fps"], on_frame=on_frame)
    sam_compute_seconds = time.perf_counter() - sam_compute_start
    provenance = sam.provenance("offline_bidirectional")
    effective_config = {"segmenter_config_sha256": provenance.config_sha256,
                        "tracker_state_offload": {"api": "tracker.init_state+cached_frame_outputs", "version": 2, "enabled": True},
                        "cpu_state_compatibility": cpu_compatibility}
    sam_record = {"runtime": runtime(), "checkpoint_sha256": provenance.checkpoint_sha256,
                  "model_revision": provenance.model_revision, "effective_config": effective_config,
                  "config_sha256": hashlib.sha256(json.dumps(effective_config, sort_keys=True).encode()).hexdigest(),
                  "sdpa_backend_policy": sam.sdpa_backend_policy, "offload_video_to_cpu": True,
                  "offload_state_to_cpu": True, "tracker_state_offload": state_offload,
                  "grounding_batch_size": sam._require_predictor().model.batched_grounding_batch_size,
                  "allocator_config": os.environ["PYTORCH_ALLOC_CONF"],
                  "expandable_segments": sum(bool(s.get("is_expandable")) for s in torch.cuda.memory_snapshot()),
                  "prompts": PROMPTS, "parity": parity}
    export_start = time.perf_counter()
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
    timing = {"imports": import_seconds, "preparation": preparation_seconds, "sam_load": sam_load_seconds,
              "sam_compute": sam_compute_seconds, "export": time.perf_counter() - export_start,
              "pose_load": pose_load_seconds, "pose_first": pose_first, "pose_first_frames": pose_first_frames, "pose_rest": pose_rest}
    estimate = estimate_full_seconds(timing, frames, info["frames"])
    scale_gate = {"required": os.environ.get("PS_STAGE") == "smoke" and args.full_stage_seconds > 0,
                  "full_allowance_seconds": args.full_stage_seconds, "estimated_full_seconds": estimate,
                  "margin": 1.25, "fixed_reserve_seconds": 60, "phases": timing}
    from src.codecs import svtav1

    summary = {"clip": info, "frames": frames, "configured_frames": int(args.frames), "stage": os.environ.get("PS_STAGE"),
               "manifest_sha256": file_sha256(Path(args.manifest)), "conversion": converted, "sam": sam_record, "pose": pose_record,
               "scale_gate": scale_gate,
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
    checked_parity = (mask_parity(ClipMasks.load(directory / "native-parity.rle"), ClipMasks.load(directory / "offloaded-parity.rle"))
                      if summary["stage"] == "smoke" else None)
    checks = {
        "files_identified": all(file_sha256(directory / n) == h for n, h in summary["files"].items()),
        "every_frame_masked_and_posed": len(clip.frames) == len(dets) == summary["frames"] == summary["conversion"]["frames"],
        "full_source_coverage": summary["configured_frames"] != 0 or summary["frames"] == summary["clip"]["frames"],
        "sam_size_and_classes": (clip.width, clip.height) == (960, 540) and set(clip.classes) == set(PROMPTS),
        "sam_nonempty": any(i.class_name in ("person", "player") for f in clip.frames for i in f),
        "sam_pinned": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "sam_native_cuda": sam["sdpa_backend_policy"] == "native_flash" and sam["runtime"]["capability"][0] >= 8,
        "effective_grounding_batch": sam["grounding_batch_size"] == 16,
        "effective_config_identified": sam["config_sha256"] == hashlib.sha256(
            json.dumps(sam["effective_config"], sort_keys=True).encode()).hexdigest(),
        "cpu_state_patch_identified": sam["effective_config"]["cpu_state_compatibility"]["patch_sha256"] == hashlib.sha256(
            json.dumps(sam["effective_config"]["cpu_state_compatibility"]["patches"], sort_keys=True).encode()).hexdigest(),
        "cpu_state_patch_receipt": json.loads((directory / "cpu-state-compatibility.json").read_text())
                                   == sam["effective_config"]["cpu_state_compatibility"],
        "tracker_cpu_state": sam["offload_state_to_cpu"] and sam["tracker_state_offload"]["initialized"] > 0
                             and sam["tracker_state_offload"]["initialized"] == sam["tracker_state_offload"]["cpu"],
        "cached_masks_cpu": sam["tracker_state_offload"]["cached_masks"] > 0
                            and sam["tracker_state_offload"]["cached_masks"] == sam["tracker_state_offload"]["cached_cpu_masks"],
        "expandable_allocator": sam["allocator_config"] == "expandable_segments:True" and sam["expandable_segments"] > 0,
        "cpu_offload_parity": checked_parity is None or (all(sam["parity"][k] == v for k, v in checked_parity.items())
                                  and checked_parity["track_ids_match"] and checked_parity["reference_instances"] > 0
                                  and min(checked_parity["minimum_union_iou"], checked_parity["minimum_instance_iou"]) >= .999),
        "pose_cuda": all(v.startswith("cuda") for v in summary["pose"]["on_cuda"].values()),
        "pose_finite": all(np.isfinite(v).all() for d in dets for v in d.values()),
        "frozen_candidates": saved["rules"] == summary["rules"] == rules.RULES,
        "features_recomputed": saved == recomputed,
        "targets_complete": saved["target_frames"] == sorted(t for t in summary["clip"]["review_frames"] if t < summary["frames"]),
        "targets_match_full_clip": [[(i.class_name, i.track_id, hashlib.sha256(i.mask().tobytes()).hexdigest()) for i in f] for f in targets.frames]
                                    == [[(i.class_name, i.track_id, hashlib.sha256(i.mask().tobytes()).hexdigest()) for i in f] for f in target_again.frames],
        "features_finite": all(all(np.isfinite(v) for v in f.values()) for f in saved["features"].values()),
    }
    gate = summary.get("scale_gate")
    if gate is not None:
        phases = gate["phases"]
        estimate = estimate_full_seconds(phases, summary["frames"], summary["clip"]["frames"])
        checks["scale_estimate_identified"] = (all(np.isfinite(v) and v >= 0 for v in phases.values())
                                               and gate["margin"] == 1.25 and gate["fixed_reserve_seconds"] == 60
                                               and estimate == gate["estimated_full_seconds"])
        checks["full_scale_fits"] = not gate["required"] or estimate <= gate["full_allowance_seconds"] <= 2700
    return {"passed": all(checks.values()), "checks": checks, "frames": summary["frames"], "clip": summary["clip"]["id"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for arg in ("manifest", "clip", "source", "sam-checkpoint", "pose", "frames"):
        r.add_argument("--" + arg, required=True)
    r.add_argument("--ffmpeg", default="ffmpeg")
    r.add_argument("--full-stage-seconds", type=float, default=0, help="Require the smoke's conservative scale estimate to fit this full-stage allowance")
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
