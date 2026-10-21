"""Label-blind SAM and YOLO features for the frozen K4b candidates, on bounded windows.

Amendment 1 of K4b.1 (docs/experiments.md, user 2026-10-20): one window of K4's
length (240 frames) per marked frame, placed by ``window_span`` from the marked
frame's index and the clip's length only. SAM 3.1 runs natively, as in K4 (no
offload, default grounding batch). One run handles a part of the windows: YOLO
pose on every window frame first, then SAM per window, each finished window
published at once. The input manifest contains no human selections.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
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
#: K4's window length (first 240 frames of each clip); amendment 2 applies the rule to those windows.
WINDOW = 240
SAM_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def key(class_name: str, track: int) -> str:
    return f"{class_name}:{track:012d}"


def window_span(frame: int, frames: int, length: int = WINDOW) -> tuple[int, int]:
    """(start, size): ``length`` frames centred on ``frame``, shifted inside the clip; a shorter clip whole."""
    if not 0 <= frame < frames or length <= 0:
        raise ValueError("The marked frame must lie inside the clip")
    size = min(length, frames)
    return min(max(frame - length // 2, 0), frames - size), size


def plan_windows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every marked frame's window, by id ``<match>/<clip>@<frame>``; reads no labels."""
    if manifest.get("contains_human_marks") is not False:
        raise ValueError("The feature manifest must not contain human selections")
    out = {}
    for clip in manifest["clips"]:
        for frame in clip["review_frames"]:
            start, size = window_span(frame, clip["frames"])
            out[f"{clip['id']}@{frame}"] = {"id": f"{clip['id']}@{frame}", "clip": clip["id"], "frame": frame,
                                            "start": start, "frames": size, "target": frame - start,
                                            "fps": clip["fps"], "sha256": clip["sha256"]}
    return out


def convert_command(ffmpeg: str, source: Path, out: Path, start: int, frames: int) -> list[str]:
    """K4's conversion (1920x1080, area filter, limited-range 4:2:0) of source frames [start, start + frames)."""
    selected = f"select=between(n\\,{start}\\,{start + frames - 1})"
    return [ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-frames:v", str(frames),
            "-vf", f"{selected},scale=1920:1080:flags=area:out_range=tv,format=yuv420p",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-y", str(out)]


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




def default_config_sha256() -> str:
    """The segmenter's configuration hash at its defaults (K4's): no offload, no batch override."""
    from src.segmentation.sam31 import Sam31SequenceSegmenter

    return Sam31SequenceSegmenter(predictor=object()).provenance("offline_bidirectional").config_sha256


def convert(ffmpeg: str, window: dict[str, Any], source: Path, out: Path) -> dict[str, Any]:
    began = time.perf_counter()
    command = convert_command(ffmpeg, source, out, window["start"], window["frames"])
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    got = out.stat().st_size // (1920 * 1080 * 3 // 2)
    if got != window["frames"]:
        raise RuntimeError(f"{window['id']}: converted {got} of {window['frames']} frames")
    return {"command": command, "frames": got, "sha256": file_sha256(out), "seconds": round(time.perf_counter() - began, 2)}


def prepare(args: argparse.Namespace, window: dict[str, Any], source: Path, poses: Any, work: Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]], Path]:
    """Convert the window once; YOLO pose and SAM's 960x540 JPEGs both read that file, as in K4."""
    import cv2

    work.mkdir(parents=True)
    yuv = work / "source.yuv"
    converted = convert(args.ffmpeg, window, source, yuv)
    raw = np.memmap(yuv, mode="r", dtype=np.uint8, shape=(window["frames"], 1620, 1920))
    jpeg = work / "jpeg"
    jpeg.mkdir()
    began = time.perf_counter()
    detections: list[dict[str, np.ndarray]] = []
    for start in range(0, window["frames"], 8):
        rgb = np.stack([tw.yuv_to_rgb(np.asarray(raw[t]), 1920, 1080) for t in range(start, min(start + 8, window["frames"]))])
        detections.extend(poses.run("yolo11x-pose", rgb, batch=8))
        for offset, frame in enumerate(rgb):
            small = cv2.resize(frame, k4.SAM_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(jpeg / f"{start + offset:05d}.jpg"), small[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    del raw
    yuv.unlink()
    converted["pose_and_jpeg_seconds"] = round(time.perf_counter() - began, 2)
    return converted, detections, jpeg


def run(args: argparse.Namespace) -> int:
    began = time.perf_counter()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from src.codecs import svtav1
    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    plan = plan_windows(manifest)
    if len(args.windows) != len(args.sources) or len(set(args.windows)) != len(args.windows):
        raise ValueError("Give one source per window, without repeated windows")
    unknown = [w for w in args.windows if w not in plan]
    if unknown:
        raise ValueError(f"Windows not in the manifest: {unknown}")
    chosen = [plan[w] for w in args.windows][:args.max_windows or None]
    sources = {w: Path(s) for w, s in zip(args.windows, args.sources)}
    digests: dict[Path, str] = {}
    for w in chosen:
        source = sources[w["id"]]
        if digests.setdefault(source, file_sha256(source)) != w["sha256"]:
            raise ValueError(f"Original clip identity mismatch: {w['id']}")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "32000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))

    poses = k4.Poses([("yolo11x-pose", args.pose)])
    prepared = {}
    for n, w in enumerate(chosen):
        prepared[w["id"]] = prepare(args, w, sources[w["id"]], poses, scratch / "work" / k4.safe(w["id"]))
        print(f"phase=pose window={w['id']} done={n + 1}/{len(chosen)}", flush=True)
    pose_record = poses.record()
    pose_peak = torch.cuda.max_memory_allocated() / 2**30
    del poses
    gc.collect()
    torch.cuda.empty_cache()

    sam = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
    rows = []
    for n, w in enumerate(chosen):
        converted, detections, jpeg = prepared.pop(w["id"])
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        clip = sam.segment_frames(jpeg, PROMPTS, policy="offline_bidirectional", fps=float(w["fps"]))
        sam_seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        record, targets = extract_features(clip, detections, [w["target"]])
        out = publish / k4.safe(w["id"])
        out.mkdir()
        clip.save(out / "sam-masks.rle")
        targets.save(out / "target-masks.rle")
        (out / "pose.json.zst").write_bytes(compress(json.dumps([{k: v.tolist() for k, v in d.items()} for d in detections]).encode()))
        write_json(out / "features.json", record)
        row = {**w, "conversion": converted, "sam_seconds": round(sam_seconds, 2), "gpu_peak_gib": round(peak, 3),
               "files": {p.name: file_sha256(p) for p in sorted(out.iterdir())}}
        write_json(out / "window.json", row)
        rows.append(row)
        print(f"phase=sam window={w['id']} frames={w['frames']} seconds={sam_seconds:.1f} peak_gib={peak:.2f}", flush=True)
        progress(n + 1)
    provenance = sam.provenance("offline_bidirectional")
    sam_record = {"runtime": runtime(), "checkpoint_sha256": provenance.checkpoint_sha256,
                  "model_revision": provenance.model_revision, "config_sha256": provenance.config_sha256,
                  "sdpa_backend_policy": sam.sdpa_backend_policy, "prompts": PROMPTS,
                  "grounding_batch_size": getattr(sam._require_predictor().model, "batched_grounding_batch_size", None),
                  "allocator_config": os.environ["PYTORCH_ALLOC_CONF"]}
    clips = {c["id"]: {k: c[k] for k in ("frames", "fps", "review_frames", "sha256")}
             for c in manifest["clips"] if any(w["clip"] == c["id"] for w in chosen)}
    summary = {"stage": os.environ.get("PS_STAGE"), "requested": args.windows, "configured_windows": int(args.max_windows),
               "windows": rows, "clips": clips, "manifest_sha256": file_sha256(manifest_path),
               "window_rule": {"length": WINDOW, "placement": "start = min(max(f - length // 2, 0), n - min(length, n))"},
               "sam": sam_record, "pose": pose_record, "rules": rules.RULES,
               "gpu_peak_gib": max([pose_peak] + [r["gpu_peak_gib"] for r in rows]), "gpu_memory_limit_mib": limit,
               "ffmpeg": svtav1.tool(args.ffmpeg), "seconds": round(time.perf_counter() - began, 2)}
    write_json(stage_dir() / "k4b-features.json", summary)
    return 0


def masks_digest(clip: ClipMasks) -> list[list[tuple[str, int, str]]]:
    return [[(i.class_name, i.track_id, hashlib.sha256(i.mask().tobytes()).hexdigest()) for i in f] for f in clip.frames]


def validate(directory: Path, summary: dict[str, Any], manifest: Path | None = None) -> dict[str, Any]:
    """Recompute every window's placement, features and target masks from the published masks and poses."""
    windows, clips = summary["windows"], summary["clips"]
    requested = summary["requested"][:summary["configured_windows"] or None]
    placed = files = covered = classes = people = pose_finite = frozen = recomputed = targets_ok = target_frame = finite = True
    for w in windows:
        c = clips[w["clip"]]
        start, size = window_span(w["frame"], c["frames"])
        placed &= (w["id"] == f"{w['clip']}@{w['frame']}" and w["frame"] in c["review_frames"]
                   and (w["start"], w["frames"], w["target"]) == (start, size, w["frame"] - start)
                   and (w["fps"], w["sha256"]) == (c["fps"], c["sha256"]))
        root = directory / k4.safe(w["id"])
        files &= all(file_sha256(root / n) == h for n, h in w["files"].items())
        clip = ClipMasks.load(root / "sam-masks.rle")
        dets = [{k: np.asarray(v, np.float32).reshape((-1, 4) if k == "boxes" else (-1, 17, 3)) for k, v in r.items()}
                for r in json.loads(decompress((root / "pose.json.zst").read_bytes()))]
        saved = json.loads((root / "features.json").read_text())
        again, target_again = extract_features(clip, dets, [w["target"]])
        covered &= len(clip.frames) == len(dets) == w["frames"] == w["conversion"]["frames"]
        classes &= (clip.width, clip.height) == k4.SAM_SIZE and set(clip.classes) == set(PROMPTS)
        people &= any(i.class_name in ("person", "player") for f in clip.frames for i in f)
        pose_finite &= all(np.isfinite(v).all() for d in dets for v in d.values())
        frozen &= saved["rules"] == rules.RULES
        recomputed &= saved == again
        targets_ok &= masks_digest(ClipMasks.load(root / "target-masks.rle")) == masks_digest(target_again)
        target_frame &= saved["target_frames"] == [w["target"]] and saved["frame_count"] == w["frames"]
        finite &= all(all(np.isfinite(v) for v in f.values()) for f in saved["features"].values())
    sam = summary["sam"]
    checks = {
        "windows_requested": bool(windows) and [w["id"] for w in windows] == requested,
        "windows_placed": placed,
        "files_identified": files,
        "every_window_frame_masked_and_posed": covered,
        "sam_size_and_classes": classes,
        "people_found_in_every_window": people,
        "sam_pinned": sam["checkpoint_sha256"] == SAM_CHECKPOINT_SHA256,
        "sam_native_cuda": sam["sdpa_backend_policy"] == "native_flash" and sam["runtime"]["capability"][0] >= 8,
        "sam_default_config": sam["config_sha256"] == default_config_sha256() and sam["grounding_batch_size"] == 16,
        "pose_cuda": all(v.startswith("cuda") for v in summary["pose"]["on_cuda"].values()),
        "pose_finite": pose_finite,
        "frozen_candidates": frozen and summary["rules"] == rules.RULES,
        "features_recomputed": recomputed,
        "target_masks_recomputed": targets_ok,
        "target_is_the_marked_frame": target_frame,
        "features_finite": finite,
        "peak_within_declaration": summary["gpu_peak_gib"] * 1024 <= summary["gpu_memory_limit_mib"],
    }
    if manifest is not None:
        entries = {c["id"]: {k: c[k] for k in ("frames", "fps", "review_frames", "sha256")}
                   for c in json.loads(manifest.read_text())["clips"]}
        checks["manifest_identified"] = (file_sha256(manifest) == summary["manifest_sha256"]
                                         and all(entries.get(k) == v for k, v in clips.items()))
    return {"passed": all(checks.values()), "checks": checks, "windows": [w["id"] for w in windows],
            "frames": sum(w["frames"] for w in windows)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for arg in ("manifest", "sam-checkpoint", "pose"):
        r.add_argument("--" + arg, required=True)
    r.add_argument("--windows", nargs="+", required=True, help="window ids <match>/<clip>@<frame>, in order")
    r.add_argument("--sources", nargs="+", required=True, help="each window's original clip, in the same order")
    r.add_argument("--max-windows", type=int, required=True, help="0 runs every listed window")
    r.add_argument("--ffmpeg", default="ffmpeg")
    v = sub.add_parser("validate")
    v.add_argument("--directory", type=Path)
    v.add_argument("--summary", type=Path)
    v.add_argument("--manifest", type=Path)
    args = p.parse_args()
    if args.command == "run":
        return run(args)
    summary = json.loads((args.summary or stage_dir() / "k4b-features.json").read_text())
    if args.directory:
        report = validate(args.directory, summary, args.manifest)
    else:
        with tempfile.TemporaryDirectory(prefix="k4b-feature-validate-", dir="/tmp") as temp:
            with tarfile.open(stage_dir() / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            report = validate(Path(temp) / "publish", summary, args.manifest)
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
