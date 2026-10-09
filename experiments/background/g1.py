"""PLAN step G1: does each dataset's camera only rotate and zoom, and how long a warm-up does a panorama need?

    python -m experiments.background.g1 run --clips CLIPS.JSON --select all|ID,ID --limit-seconds S \\
        --source NAME PATH ... [--visor-archive DIR] [--visor-fill DIR] [--hand-objects DIR] [--checkpoint PT]
    python -m experiments.background.g1 validate
    python -m experiments.background.g1 report --result g1.json ... --published DIR ... --out DIR

``clips.json`` comes from ``tools/datasets/g1_inputs.py``: the selection rule,
and per clip its source, frame range, analysis rate and mask tier. Per clip:

1. *Frames.* Decoded from the source and downscaled to 960x540 at the clip's
   analysis rate (every frame for VISOR windows, ``rate`` frames per second
   elsewhere). VISOR frames are decoded frame indices placed by B1's verified
   rules; a window's released sparse JPEGs are checked against the decoded
   frames as in B1 and B1b.
2. *Foreground.* VISOR windows: the dense masks with B1b's object fill
   (tiers ``interpolated`` and ``sam_from_label_prompt``). Everything else:
   SAM 3.1 text prompts (tier ``sam_text``); VISOR stretches also mask the
   boxes of the EPIC-KITCHENS hand-object detector (score >= 0.5), since a
   held object has no fixed name to prompt. Dilated by ``camera.DILATE``. On a
   stretch's frames inside its evaluation window, the rotation residual is
   also measured with the dense masks (the tier check).
3. *Camera* (`experiments.background.camera`): frame-to-frame and
   frame-to-reference homographies, the rotation and zoom model with the
   clip's lens (a window uses its stretch's lens), residuals and their
   attribution, the translation test at ``camera.TRANSLATION_SECONDS``,
   and the warm-up coverage curve.

Writes ``g1.json`` (clip summaries) to ``PS_STAGE_DIR`` and per clip
``publish/clips/<id>/`` (``result.json`` with per-frame records, overlays, the
first-seen mosaic, and SAM's masks as ``masks.rle``).

Resume: each finished clip is also saved to ``PS_CHECKPOINT_DIR/clips/<id>/``
(written under a dot-name and renamed, so a stop never archives half a clip).
A resumed attempt (the job declares ``contention.resume_attempts``) restores
those clips into ``publish/`` and runs only the rest.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import camera
from experiments.visor.b1 import MATCH, file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any run (docs/experiments.md, G1 entry).
DECISION: dict[str, Any] = {
    "explained_px": camera.EXPLAINED_PX,
    "clip_explained_share": 0.90,
    "clip_direct_share": 0.95,
    "clip_measured_share": 0.80,
    "dataset_holds_share": 0.75,
    "dataset_subset_share": 0.25,
    "translation_prefers_f_share": 0.50,
    "translation_parallax_px": 2.0,
    "coverage_targets": [0.90, 0.99],
    "g2_play_after_warmup_s": 10.0,
    "start_visor_min_clips": 10,
    "start_visor_max_warmup90_s": 30.0,
}
SAM_TIER = "sam_text"
DETECTOR_TIER = "detector_box"
BOX_SCORE = 0.5
CHUNK = 300
TRACK_STRIDE = 100_000
WARMUP_STEP = 0.5
OVERLAY_AT = (0.25, 0.5, 0.9)
OVERLAY_WARMUP_S = 10.0
SELF_TEST = {"yaw_deg": 3.0, "pitch_deg": 1.0, "logf": 0.02, "max_angle_error_deg": 0.05, "max_flow_p90_px": 0.5}


# ----------------------------------------------------------------- planning


def select(spec: dict[str, Any], which: str) -> list[dict[str, Any]]:
    clips = spec["clips"]
    if which == "all":
        return clips
    wanted = which.split(",")
    by_id = {c["id"]: c for c in clips}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"unknown clips: {missing}")
    return [by_id[w] for w in wanted]


def plan(clip: dict[str, Any], limit_seconds: float) -> dict[str, Any]:
    """Analysis frame indices (source numbering) and their times from the clip's first frame."""
    fps = float(clip["fps"])
    first, last = int(clip["first_index"]), int(clip["last_index"])
    if limit_seconds > 0:
        last = min(last, first + int(round(limit_seconds * fps)) - 1)
    if clip["analysis"]["mode"] == "native":
        indices = list(range(first, last + 1))
    else:
        rate = float(clip["analysis"]["fps"])
        indices, k = [], 0
        while (at := first + int(round(k * fps / rate))) <= last:
            indices.append(at)
            k += 1
    times = [(i - first) / fps for i in indices]
    return {**clip, "last_index": last, "indices": indices, "times": times, "limit_seconds": limit_seconds}


def safe(clip_id: str) -> str:
    return clip_id.replace("/", "__")


# ----------------------------------------------------------------- decoding (spawned processes)


def decode_clip(task: dict[str, Any]) -> dict[str, Any]:
    """Write the clip's analysis frames to ``frames.npy`` (and JPEG chunks for SAM 3.1)."""
    import cv2

    cv2.setNumThreads(max(1, task["threads"]))
    began = time.time()
    clip, out = task["plan"], Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    width, height = camera.ANALYSIS_SIZE
    indices = clip["indices"]
    position = {index: i for i, index in enumerate(indices)}
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(indices), height, width, 3))
    filled = np.zeros(len(indices), bool)
    gate_at = {g["index"]: g for g in task.get("jpeg_gate", [])}
    full_res: dict[int, np.ndarray] = {}
    source = clip["source"]

    def put(index: int, rgb: np.ndarray) -> None:
        i = position[index]
        frames[i] = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA) if rgb.shape[1] != width else rgb
        filled[i] = True
        if task.get("jpeg_dir"):
            chunk = Path(task["jpeg_dir"]) / f"chunk_{(i // CHUNK) * CHUNK:06d}"
            chunk.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(chunk / f"{i % CHUNK:05d}.jpg"), frames[i][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

    if source["kind"] in ("video", "video_in_tree"):
        from src.segmentation import visor

        path = task["path"]
        wanted = set(indices)
        near = {g + d for g in gate_at for d in range(-2, 3)}
        last_seen = -1
        for at, frame in visor.decoded_frames(path, range(clip["first_index"], clip["last_index"] + 1),
                                              threads=task["threads"]):
            if at <= last_seen:
                raise RuntimeError(f"{clip['id']}: decoded frame {at} after {last_seen}")
            last_seen = at
            if at in wanted or at in near:
                rgb = frame.to_ndarray(format="rgb24")
                if at in near:
                    full_res[at] = rgb
                if at in wanted:
                    put(at, rgb)
    elif source["kind"] == "jpeg_dir":
        names = sorted(p for p in (Path(task["path"]) / source["member_dir"]).iterdir() if p.suffix == ".jpg")
        if len(names) != source["frames"]:
            raise RuntimeError(f"{clip['id']}: {len(names)} JPEGs, expected {source['frames']}")
        for index in indices:
            bgr = cv2.imread(str(names[index]))
            put(index, bgr[:, :, ::-1])
    else:
        raise ValueError(f"unknown source kind {source['kind']}")
    frames.flush()
    if not filled.all():
        raise RuntimeError(f"{clip['id']}: decoded {int(filled.sum())} of {len(indices)} analysis frames")
    gate = []
    if gate_at:
        from PIL import Image

        for index, g in gate_at.items():
            released = np.asarray(Image.open(g["path"]).convert("RGB")).astype(np.int16)
            mae = {i: float(np.abs(full_res[i].astype(np.int16) - released).mean()) for i in full_res if abs(i - index) <= 2}
            best = min(mae, key=lambda i: mae[i])
            gate.append({"jpeg": g["jpeg"], "index": index, "best_index": best, "rule_mae": round(mae[index], 3),
                         "best_mae": round(mae[best], 3), "holds": mae[index] - mae[best] <= 0.5 and mae[best] < MATCH})
    return {"frames": len(indices), "seconds": round(time.time() - began, 2), "jpeg_gate": gate}


def jpeg_gate(clip: dict[str, Any], archive: Path) -> list[dict[str, Any]]:
    """The window's released sparse JPEGs and their decoded indices by the verified rule."""
    from src.segmentation import visor

    item = clip["item"]
    mapping = json.loads((archive / "frame_mapping.json").read_text())[item["video"]]
    out = []
    for member in item["sparse_jpegs"]:
        epic = visor.frame_number(mapping[Path(member).name])
        out.append({"jpeg": member, "path": str(archive / member),
                    "index": visor.epic_frame_to_video_index(epic, item["fps"])})
    return out


# ----------------------------------------------------------------- foreground


def union(instances: list[Any], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    for inst in instances:
        out |= inst.mask().astype(bool)
    return out


def to_analysis(mask: np.ndarray) -> np.ndarray:
    import cv2

    width, height = camera.ANALYSIS_SIZE
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_AREA) > 0


def dilate(mask: np.ndarray) -> np.ndarray:
    import cv2

    size = 2 * camera.DILATE + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def visor_fill_masks(clip: dict[str, Any], fill_root: Path) -> tuple[Any, dict[str, Any]]:
    from experiments.visor.b1b import mask_root
    from src.segmentation.masks import ClipMasks

    path = mask_root(fill_root, clip["item"]["id"])
    masks = ClipMasks.load(path)
    tiers: dict[str, int] = {}
    for frame in masks.frames:
        for inst in frame:
            tiers[str(inst.provenance)] = tiers.get(str(inst.provenance), 0) + 1
    return masks, {"masks_rle": str(path), "masks_rle_sha256": file_sha256(path), "instances_by_tier": tiers}


def detector_boxes(clip: dict[str, Any], hand_objects_root: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Hand and object boxes (score >= BOX_SCORE) of the EPIC-KITCHENS detector on each analysis frame."""
    import cv2

    from experiments.visor.b1b import video_to_epic_frame
    from src.segmentation import hand_objects

    video = clip["video"]
    relative = f"hand-objects/{video[:3]}/{video}.pkl"
    detections = hand_objects.load_detections(hand_objects_root / relative)
    width, height = camera.ANALYSIS_SIZE
    masks, boxes, unmapped = [], 0, 0
    for index in clip["indices"]:
        mask = np.zeros((height, width), np.uint8)
        epic = video_to_epic_frame(index, clip["fps"])
        if epic is None or epic > len(detections):
            unmapped += 1
        else:
            frame = detections[epic - 1]
            items: list[Any] = [*frame.hands, *frame.objects]
            for item in items:
                if item.score >= BOX_SCORE:
                    x0, y0, x1, y1 = item.box.pixels(width, height)
                    cv2.rectangle(mask, (int(x0), int(y0)), (int(math.ceil(x1)), int(math.ceil(y1))), 1, -1)
                    boxes += 1
        masks.append(mask > 0)
    return masks, {"path": relative, "sha256": file_sha256(hand_objects_root / relative), "boxes": boxes,
                   "frames_without_epic_frame": unmapped, "min_score": BOX_SCORE}


def sam_text_masks(segmenter: Any, clip: dict[str, Any], jpeg_dir: Path) -> Any:
    """SAM 3.1 text prompts on each chunk of analysis frames, one session per class and chunk."""
    from src.segmentation.masks import ClipMasks

    prompts = clip["masks"]["prompts"]
    width, height = camera.ANALYSIS_SIZE
    out = ClipMasks(tuple(prompts), height, width, float(clip["analysis"].get("fps", clip["fps"])))
    out.ensure_frames(len(clip["indices"]))
    for number, chunk in enumerate(sorted(jpeg_dir.iterdir())):
        start = int(chunk.name.split("_")[1])
        part = segmenter.segment_frames(chunk, dict(prompts), policy="offline_bidirectional", fps=out.fps)
        for i, instances in enumerate(part.frames):
            out.frames[start + i].extend(
                replace(inst, track_id=number * TRACK_STRIDE + inst.track_id, provenance=SAM_TIER) for inst in instances)
        for name in part.meta.get("empty_classes", []):
            out.meta.setdefault("empty_chunks", []).append({"start": start, "class": name})
    return out


def window_dense(clip: dict[str, Any], visor_fill: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """A stretch's analysis frames inside its evaluation window: the window's dense masks with B1b's fill
    (analysis size, not dilated), by analysis frame number."""
    window = clip["masks"]["window_item"]
    masks, info = visor_fill_masks({"item": window}, visor_fill)
    dense = {}
    for i, index in enumerate(clip["indices"]):
        at = index - window["first_video_index"]
        if 0 <= at < len(masks.frames):
            dense[i] = to_analysis(union(masks.frames[at], (masks.height, masks.width)))
    return dense, info


def foreground(clip: dict[str, Any], work: Path, base: list[np.ndarray] | None, visor_fill: str | None,
               hand_objects: str | None) -> dict[str, Any]:
    """Write foreground.npy (dilated) and, for VISOR stretches, dense.npz on window frames."""
    record: dict[str, Any] = {"dilate_px_1080": camera.DILATE * camera.TO_1080}
    n = len(clip["indices"])
    width, height = camera.ANALYSIS_SIZE
    fgm = np.lib.format.open_memmap(work / "foreground.npy", mode="w+", dtype=bool, shape=(n, height, width))
    raw = base if base is not None else [np.zeros((height, width), bool) for _ in range(n)]
    if clip["masks"]["kind"] == "visor_fill":
        masks, info = visor_fill_masks(clip, Path(str(visor_fill)))
        record.update(tier="visor_dense_sam_fill", **info)
        offset = clip["indices"][0] - clip["item"]["first_video_index"]
        for i in range(n):
            raw[i] = to_analysis(union(masks.frames[offset + i], (masks.height, masks.width)))
    if clip["masks"].get("boxes"):
        boxes, info = detector_boxes(clip, Path(str(hand_objects)))
        record["detector"] = info
        for i in range(n):
            raw[i] = raw[i] | boxes[i]
    for i in range(n):
        fgm[i] = dilate(raw[i])
    fgm.flush()
    window = clip["masks"].get("window_item")
    if window:
        raw_dense, info = window_dense(clip, Path(str(visor_fill)))
        dense, recalls = {}, []
        for i, d in raw_dense.items():
            dense[str(i)] = dilate(d)
            if d.any():
                recalls.append(float((d & np.asarray(fgm[i])).sum() / d.sum()))
        np.savez(work / "dense.npz", **dense)
        record["tier_check"] = {"window_item": window["id"], "frames": len(dense), **info}
        record["dense_recall"] = {"frames": len(recalls), "mean": round(float(np.mean(recalls)), 4) if recalls else None,
                                  "p10": round(float(np.percentile(recalls, 10)), 4) if recalls else None}
    return record


# ----------------------------------------------------------------- analysis (spawned processes)


def strip(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: v for k, v in record.items() if not k.startswith("_") and k != "F"}


def analyse_clip(task: dict[str, Any]) -> dict[str, Any]:
    """Every measurement of one clip; writes ``result.json`` and images under ``task['publish']``."""
    import cv2

    cv2.setNumThreads(1)
    began = time.time()
    clip = task["plan"]
    work, publish = Path(task["work"]), Path(task["publish"])
    publish.mkdir(parents=True, exist_ok=True)
    frames = np.load(work / "frames.npy", mmap_mode="r")
    fg = np.load(work / "foreground.npy", mmap_mode="r")
    dense_fg = dict(np.load(work / "dense.npz")) if (work / "dense.npz").is_file() else {}
    times = np.asarray(clip["times"], float)
    n = len(frames)
    gray = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY) for f in frames]
    bg = [~np.asarray(m) for m in fg]
    feats = [camera.features(g, b) for g, b in zip(gray, bg)]

    f2f_h: list[np.ndarray | None] = [None]
    f2f_matches: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    f2f_counts: list[dict[str, int] | None] = [None]
    for t in range(1, n):
        pa, pb = camera.match(feats[t - 1], feats[t])
        H, keep = camera.homography(pa, pb)
        f2f_h.append(H)
        f2f_matches.append((pa[keep], pb[keep]) if H is not None else None)
        f2f_counts.append({"matches": int(len(pa)), "inliers": int(keep.sum())})

    rate = (n - 1) / max(times[-1], 1e-9) if n > 1 else 1.0
    delta = max(1, int(round(camera.TRANSLATION_SECONDS * rate)))
    if task.get("lens"):
        lens = camera.Lens(task["lens"]["f"], task["lens"]["k1"])
        lens_report = {**task["lens_report"], "from": task["lens_from"]}
    else:
        pairs = []
        starts = np.unique(np.linspace(0, max(0, n - 1 - delta), min(40, max(1, n - delta))).astype(int))
        for t in starts:
            if t + delta >= n:
                continue
            pa, pb = camera.match(feats[t], feats[t + delta])
            if len(pa) < camera.MIN_INLIERS:
                continue
            try:
                H, keep = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
            except cv2.error:  # degenerate sample set (see camera.homography)
                continue
            if H is not None and keep is not None and keep.sum() >= camera.MIN_INLIERS:
                keep = keep.ravel().astype(bool)
                pairs.append((pa[keep], pb[keep]))
        lens, lens_report = camera.fit_lens(pairs)
        lens_report["from"] = "clip"
    reg = camera.register(lens, feats, f2f_matches)

    records = []
    keep_images: dict[int, dict[str, Any]] = {}
    overlay_frames = sorted({min(n - 1, max(1, int(round(q * (n - 1))))) for q in OVERLAY_AT}) if n > 1 else []
    for t in range(n):
        pose = reg.poses[t]
        rec: dict[str, Any] = {
            "i": t, "index": clip["indices"][t], "time": round(float(times[t]), 4), "status": pose.status,
            "segment": pose.segment, "reference": pose.reference, "inliers": pose.inliers,
            "rotation_p50_px": None if pose.rotation_p50_px is None else round(pose.rotation_p50_px, 3),
            "logf": round(pose.logf, 5), "fg_share": round(float(np.asarray(fg[t]).mean()), 4),
            "sharpness": round(camera.sharpness(gray[t], bg[t]), 2), "luma": round(float(gray[t][bg[t]].mean()), 2) if bg[t].any() else None,
            "f2f": f2f_counts[t],
        }
        if t > 0 and reg.poses[t - 1].status != "lost" and pose.status != "lost" and reg.poses[t - 1].segment == pose.segment:
            dt = max(times[t] - times[t - 1], 1e-9)
            rec["speed_deg_s"] = round(camera.angle_deg(pose.R @ reg.poses[t - 1].R.T) / dt, 3)
            rec["zoom_rate"] = round((pose.logf - reg.poses[t - 1].logf) / dt, 5)
        H_ff = f2f_h[t]
        if t > 0 and H_ff is not None:
            rec["f2f_h"] = strip(camera.residual(lens, gray[t], gray[t - 1],
                                                 camera.warp_homography(H_ff, *camera.ANALYSIS_SIZE), bg[t], bg[t - 1]))
        if pose.status == "direct" and pose.reference is not None:
            k = pose.reference
            ref = reg.poses[k]
            rec["angle_to_reference_deg"] = round(camera.angle_deg(pose.R @ ref.R.T), 3)
            rec["pair_reference"] = strip(pose.pair)
            assert pose.matches is not None
            pa, pb = pose.matches
            H, _ = camera.homography(pa, pb)
            if H is not None:
                rec["f2r_h"] = strip(camera.residual(lens, gray[t], gray[k], camera.warp_homography(H, *camera.ANALYSIS_SIZE),
                                                     bg[t], bg[k]))
            F = pose.pair["F"] if pose.pair and pose.pair["prefers_f"] else None
            warp = camera.warp_rotation(lens, pose, ref)
            res = camera.residual(lens, gray[t], gray[k], warp, bg[t], bg[k], F=F, attribute=True,
                                  keep=t in overlay_frames)
            if "_images" in res:
                keep_images[t] = {**res.pop("_images"), "reference": k}
            rec["f2r_rot"] = strip(res)
            if str(t) in dense_fg and str(k) in dense_fg:
                rec["f2r_rot_dense"] = strip(camera.residual(lens, gray[t], gray[k], warp, ~dense_fg[str(t)],
                                                             ~dense_fg[str(k)]))
        if t >= delta:
            pa, pb = camera.match(feats[t - delta], feats[t])
            rec["pair_half_s"] = strip(camera.translation_pair(lens, pa, pb))
        records.append(rec)

    warmups = np.arange(0.0, max(times[-1], 0.0) + 1e-9, WARMUP_STEP)
    map_warmup = min(OVERLAY_WARMUP_S, float(times[-1]) / 2)
    cov = camera.coverage(lens, reg.poses, times, lambda t: bg[t], warmups, colour=lambda t: np.asarray(frames[t]),
                          maps={t: map_warmup for t in overlay_frames})
    for t, value in cov["online"].items():
        records[t]["online_coverage"] = value

    images = write_images(clip, publish, frames, fg, keep_images, cov, map_warmup)
    summary = summarize(clip, records, reg, lens, lens_report, cov)
    tier_check = tier_summary(records, task.get("dense_recall"))
    result = {
        "id": clip["id"], "dataset": clip["dataset"], "group": clip["group"], "summary": summary,
        "tier_check": tier_check, "lens": {"f": lens.f, "k1": lens.k1, "report": lens_report},
        "coverage": {k: v for k, v in cov.items() if not k.startswith("_") and k != "online"},
        "keyframes": reg.keyframes, "images": images, "analysis_seconds": round(time.time() - began, 2),
        "masks": task.get("masks_record"), "decode": task.get("decode"), "frames": records,
        "clip": {k: v for k, v in clip.items() if k not in ("indices", "times")},
    }
    if task.get("self_test"):
        result["self_test"] = self_test(gray[n // 2], lens)
    write_json(publish / "result.json", result)
    return {k: v for k, v in result.items() if k != "frames"}


def summarize(clip: dict[str, Any], records: list[dict[str, Any]], reg: camera.Registration, lens: camera.Lens,
              lens_report: dict[str, Any], cov: dict[str, Any]) -> dict[str, Any]:
    n = len(records)
    later = records[1:]
    statuses = [r["status"] for r in later]

    def share(values: list[bool]) -> float | None:
        return round(float(np.mean(values)), 4) if values else None

    def med(values: list[float]) -> float | None:
        return round(float(np.median(values)), 4) if values else None

    def model(key: str) -> dict[str, Any]:
        rows = [r[key] for r in later if r.get(key)]
        measured = [r for r in rows if r.get("measured")]
        return {
            "frames": len(rows), "measured": len(measured),
            "explained_share": share([r["explained"] for r in measured]),
            "flow_p90_px_median": med([r["flow_p90_px"] for r in measured]),
            "flow_p50_px_median": med([r["flow_p50_px"] for r in measured]),
            **{f"within_{x:g}px_median": med([r[f"flow_within_{x:g}px"] for r in measured]) for x in camera.FLOW_THRESHOLDS},
            "psnr_median": med([r["psnr"] for r in measured]),
            "psnr_gain_median": med([r["psnr_gain"] for r in measured]),
            "psnr_flow_median": med([r["psnr_flow"] for r in measured]),
        }

    rot = [r["f2r_rot"] for r in later if r.get("f2r_rot") and r["f2r_rot"].get("measured")]
    energy = {k: round(float(np.mean([r["energy_share"][k] for r in rot])), 4) for k in
              ("exposure", "parallax", "independent", "remainder")} if rot else None
    radius = []
    for i in range(camera.RADIUS_BINS):
        values = [r["flow_p50_by_radius_px"][i] for r in rot if r["flow_p50_by_radius_px"][i] is not None]
        radius.append(med(values))
    half = [r["pair_half_s"] for r in records if r.get("pair_half_s")]
    speeds = [r["speed_deg_s"] for r in records if r.get("speed_deg_s") is not None]
    sharp = np.array([r["sharpness"] for r in records], float)
    blur_cut = 0.5 * float(np.nanmedian(sharp)) if np.isfinite(sharp).any() else math.nan
    blurred = [r for r in later if r.get("f2r_rot", {}).get("measured") and r["sharpness"] < blur_cut]
    crisp = [r for r in later if r.get("f2r_rot", {}).get("measured") and r["sharpness"] >= blur_cut]
    paired = [(r["speed_deg_s"], r["f2r_rot"]["flow_p90_px"]) for r in later
              if r.get("speed_deg_s") is not None and r.get("f2r_rot", {}).get("measured")]
    rho = None
    if len(paired) >= 10:
        from scipy.stats import spearmanr

        rho = round(float(spearmanr([p[0] for p in paired], [p[1] for p in paired]).statistic), 4)
    direct = share([s == "direct" for s in statuses])
    measured_share = round(len(rot) / max(1, len(later)), 4)
    explained = model("f2r_rot")["explained_share"]
    holds = (explained is not None and direct is not None and explained >= DECISION["clip_explained_share"]
             and direct >= DECISION["clip_direct_share"] and measured_share >= DECISION["clip_measured_share"]
             and cov["segments"] == 1)
    logfs = [r["logf"] for r in records if r["status"] != "lost"]
    duration = float(records[-1]["time"]) if records else 0.0
    w90 = cov["warmup_90_s"]
    return {
        "frames": n, "duration_s": round(duration, 3),
        "status": {s: statuses.count(s) for s in sorted(set(statuses))},
        "direct_share": direct, "measured_share": measured_share, "segments": cov["segments"],
        "keyframes": len(reg.keyframes),
        "f2f_h": model("f2f_h"), "f2r_h": model("f2r_h"), "f2r_rot": model("f2r_rot"),
        "holds": bool(holds),
        "energy_share_mean": energy,
        "moved_share_median": med([r["moved_share"] for r in rot]),
        "flow_p50_by_radius_px": radius,
        "translation_half_s": {
            "pairs": len(half),
            "prefers_f_share": share([p["prefers_f"] for p in half]),
            "parallax_p50_px_median": med([p["parallax_p50_px"] for p in half]),
            "parallax_p90_px_median": med([p["parallax_p90_px"] for p in half]),
            "displacement_p50_px_median": med([p["displacement_p50_px"] for p in half]),
        },
        "speed_deg_s": {"median": med(speeds), "p90": round(float(np.percentile(speeds, 90)), 3) if speeds else None},
        "zoom_range": round(max(logfs) - min(logfs), 5) if logfs else None,
        "residual_vs_speed_spearman": rho,
        "blur": {"cut": round(blur_cut, 2) if np.isfinite(blur_cut) else None, "blurred_frames": len(blurred),
                 "explained_blurred": share([r["f2r_rot"]["explained"] for r in blurred]),
                 "explained_sharp": share([r["f2r_rot"]["explained"] for r in crisp])},
        "exposure": {"gain_p10_p90": [round(float(np.percentile([r["gain"] for r in rot], q)), 4) for q in (10, 90)] if rot else None,
                     "luma_range": [min(r["luma"] for r in records if r["luma"] is not None),
                                    max(r["luma"] for r in records if r["luma"] is not None)]
                     if any(r["luma"] is not None for r in records) else None},
        "fg_share_mean": round(float(np.mean([r["fg_share"] for r in records])), 4),
        "lens": {"f_1080": round(lens.f * camera.TO_1080, 1), "k1": round(lens.k1, 4),
                 "hfov_deg": round(math.degrees(2 * math.atan(lens.width / 2 / lens.f)), 2),
                 "observable": lens_report.get("observable"), "from": lens_report.get("from")},
        "warmup_90_s": w90, "warmup_99_s": cov["warmup_99_s"],
        "g2_usable": bool(holds and w90 is not None and duration >= w90 + DECISION["g2_play_after_warmup_s"]),
    }


def tier_summary(records: list[dict[str, Any]], recall: dict[str, Any] | None) -> dict[str, Any] | None:
    pairs = [(r["f2r_rot"], r["f2r_rot_dense"]) for r in records
             if r.get("f2r_rot_dense") and r.get("f2r_rot") and r["f2r_rot"].get("measured") and r["f2r_rot_dense"].get("measured")]
    if not pairs and not recall:
        return None
    return {
        "frames": len(pairs),
        "explained_sam": float(np.mean([a["explained"] for a, _ in pairs])) if pairs else None,
        "explained_dense": float(np.mean([b["explained"] for _, b in pairs])) if pairs else None,
        "flow_p90_sam_median": float(np.median([a["flow_p90_px"] for a, _ in pairs])) if pairs else None,
        "flow_p90_dense_median": float(np.median([b["flow_p90_px"] for _, b in pairs])) if pairs else None,
        "dense_recall": recall,
    }


def write_images(clip: dict[str, Any], publish: Path, frames: Any, fg: Any, kept: dict[int, dict[str, Any]],
                 cov: dict[str, Any], warmup: float) -> list[str]:
    """2x2 panels per overlay frame: frame and foreground, checkerboard of frame and warped reference,
    residual flow, and coverage after the warm-up; plus the first-seen mosaic."""
    import cv2

    names = []
    w, h = camera.ANALYSIS_SIZE[0] // 2, camera.ANALYSIS_SIZE[1] // 2
    for t, images in sorted(kept.items()):
        rgb = np.asarray(frames[t]).copy()
        mask = np.asarray(fg[t])
        tinted = rgb.copy()
        tinted[mask] = (0.5 * tinted[mask] + 0.5 * np.array([220, 40, 200])).astype(np.uint8)
        grey = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        warped = cv2.cvtColor(images["warped"], cv2.COLOR_GRAY2RGB)
        yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        checker = np.where((((yy // 60) + (xx // 60)) % 2 == 0)[..., None], grey, warped)
        checker[~images["region"]] = (checker[~images["region"]] * 0.35).astype(np.uint8)
        mag = np.clip(images["flow_mag"] / 8.0, 0, 1)
        heat = cv2.applyColorMap((mag * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
        heat[~images["region"]] = (40, 40, 40)
        cmap = cov["_maps"].get(t)
        cover = (grey * 0.6).astype(np.uint8)
        if cmap is not None:
            big = cv2.resize(cmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            cover[big == 1] = (0.5 * cover[big == 1] + 0.5 * np.array([40, 200, 80])).astype(np.uint8)
            cover[big == 2] = (0.4 * cover[big == 2] + 0.6 * np.array([230, 50, 40])).astype(np.uint8)
        panels = []
        for image, label in ((tinted, f"t={clip['times'][t]:.1f}s  foreground"),
                             (checker, f"frame / reference {images['reference']} warped (rotation)"),
                             (heat, "residual flow 0-8 px @1080p"),
                             (cover, f"seen within {warmup:.0f}s warm-up (green) / new (red)")):
            small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            cv2.putText(small, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(small)
        sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        name = f"overlay_{t:05d}.jpg"
        cv2.imwrite(str(publish / name), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        names.append(name)
    for i, mosaic in enumerate(cov["_mosaics"]):
        name = f"mosaic_{i}.jpg"
        cv2.imwrite(str(publish / name), mosaic[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        names.append(name)
    return names


# ----------------------------------------------------------------- self-test


def self_test(gray: np.ndarray, lens: camera.Lens) -> dict[str, Any]:
    """A real frame re-rendered by a known rotation and zoom must come back exactly."""
    import cv2

    truth = camera.rotation(np.radians([SELF_TEST["pitch_deg"], SELF_TEST["yaw_deg"], 0.0]))
    a = camera.Pose(np.eye(3), 0.0, "first", 0)
    b = camera.Pose(truth, SELF_TEST["logf"], "first", 0)
    warp = camera.warp_rotation(lens, b, a)
    moved = cv2.remap(gray, warp.map_x, warp.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bg_a = np.ones_like(gray, bool)
    bg_b = cv2.erode(warp.valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    fa, fb = camera.features(gray, bg_a), camera.features(moved, bg_b)
    pa, pb = camera.match(fa, fb)
    H, keep = camera.homography(lens.undistort(pa).astype(np.float32), lens.undistort(pb).astype(np.float32))
    if H is None:
        return {"passed": False, "reason": "no homography"}
    fit = camera.fit_rotation(lens, pa[keep], pb[keep])
    error = camera.angle_deg(fit.R @ truth.T)
    pose = camera.Pose(fit.R, fit.s, "direct", 0)
    res = camera.residual(lens, moved, gray, camera.warp_rotation(lens, pose, a), bg_b, bg_a)
    passed = (error <= SELF_TEST["max_angle_error_deg"] and abs(fit.s - SELF_TEST["logf"]) < 1e-3
              and res.get("measured") and res["flow_p90_px"] <= SELF_TEST["max_flow_p90_px"])
    return {"passed": bool(passed), "angle_error_deg": round(error, 5), "zoom_error": round(fit.s - SELF_TEST["logf"], 6),
            "flow_p90_px": res.get("flow_p90_px"), "inliers": int(keep.sum()), "settings": SELF_TEST}


# ----------------------------------------------------------------- checkpoints

CHECKPOINT_RECORD = "checkpoint.json"


def save_clip(root: Path, source: Path, clip_id: str, result: dict[str, Any], sam: dict[str, Any] | None) -> Path:
    """Copy a finished clip's published files into the checkpoint, atomically."""
    clips = root / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    final = clips / safe(clip_id)
    temporary = clips / f".{safe(clip_id)}.{os.getpid()}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(source, temporary)
    write_json(temporary / CHECKPOINT_RECORD, {"id": clip_id, "result": result, "sam": sam})
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    return final


def restore_clips(root: Path, publish: Path) -> dict[str, dict[str, Any]]:
    """Finished clips of an earlier attempt, copied back into ``publish/clips``."""
    restored: dict[str, dict[str, Any]] = {}
    clips = root / "clips"
    if not clips.is_dir():
        return restored
    for directory in sorted(clips.iterdir()):
        record = directory / CHECKPOINT_RECORD
        if directory.name.startswith(".") or not record.is_file():
            continue
        saved = json.loads(record.read_text())
        target = publish / "clips" / directory.name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(directory, target, ignore=shutil.ignore_patterns(CHECKPOINT_RECORD))
        restored[saved["id"]] = saved
    return restored


# ----------------------------------------------------------------- run


def command_run(args: argparse.Namespace) -> int:
    # One BLAS/OpenMP thread per process: the analysis runs one process per core.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import cv2

    spec_path = Path(args.clips)
    spec = json.loads(spec_path.read_text())
    clips = [plan(c, args.limit_seconds) for c in select(spec, args.select)]
    sources = {name: path for name, path in args.source}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    spawn = multiprocessing.get_context("spawn")
    decoder = ProcessPoolExecutor(max_workers=2, mp_context=spawn)
    analyser = ProcessPoolExecutor(max_workers=max(1, allowance - 3), mp_context=spawn)
    archive = Path(args.visor_archive) if args.visor_archive else None

    def decode_task(clip: dict[str, Any], jpegs: bool) -> dict[str, Any]:
        work = scratch / "work" / safe(clip["id"])
        source = clip["source"]
        path = sources[source["name"]]
        if source["kind"] == "video_in_tree":
            path = str(Path(path) / source["member"])
        task = {"plan": clip, "out": str(work), "path": path, "threads": max(1, allowance // 4),
                "jpeg_dir": str(work / "jpeg") if jpegs else None}
        if clip["group"] == "visor-window":
            assert archive is not None
            task["jpeg_gate"] = jpeg_gate(clip, archive)
        return task

    def analysis_task(clip: dict[str, Any], decoded: dict[str, Any], masks_record: dict[str, Any],
                      lens: dict[str, Any] | None = None) -> dict[str, Any]:
        task = {"plan": clip, "work": str(scratch / "work" / safe(clip["id"])),
                "publish": str(publish / "clips" / safe(clip["id"])), "decode": decoded, "masks_record": masks_record,
                "dense_recall": masks_record.get("dense_recall"), "self_test": clip["id"] == clips[0]["id"]}
        if lens is not None:
            task.update(lens={"f": lens["f"], "k1": lens["k1"]}, lens_report=lens["report"], lens_from=clip["lens_from"])
        return task

    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore_clips(checkpoints, publish) if checkpoints else {}
    futures: dict[str, Future[Any]] = {}
    sam_record: dict[str, Any] = {"clips": {}}
    if checkpoints and (checkpoints / "sam.json").is_file():
        sam_record = json.loads((checkpoints / "sam.json").read_text())
    for clip_id, saved in restored.items():
        future: Future[Any] = Future()
        future.set_result({**saved["result"], "restored_from_checkpoint": True})
        futures[clip_id] = future
        if saved.get("sam"):
            sam_record.setdefault("clips", {})[clip_id] = saved["sam"]

    def keep(clip_id: str, future: Future[Any]) -> None:
        if checkpoints is None or future.exception() is not None:
            return
        save_clip(checkpoints, publish / "clips" / safe(clip_id), clip_id, future.result(),
                  sam_record.get("clips", {}).get(clip_id))

    def submit_analysis(clip_id: str, task: dict[str, Any]) -> None:
        futures[clip_id] = analyser.submit(analyse_clip, task)
        futures[clip_id].add_done_callback(functools.partial(keep, clip_id))

    sam_clips = [c for c in clips if c["masks"]["kind"] == "sam_text" and c["id"] not in restored]
    other = [c for c in clips if c["masks"]["kind"] != "sam_text" and c["id"] not in restored]
    segmenter = None
    pending = {i: decoder.submit(decode_clip, decode_task(c, True)) for i, c in enumerate(sam_clips[:2])}
    done = 0
    for number, clip in enumerate(sam_clips):
        decoded = pending.pop(number).result()
        if number + 2 < len(sam_clips):
            pending[number + 2] = decoder.submit(decode_clip, decode_task(sam_clips[number + 2], True))
        work = scratch / "work" / safe(clip["id"])
        if segmenter is None:
            from src.segmentation.sam31 import Sam31SequenceSegmenter
            from src.segmentation.sam31_tracker import runtime

            began = time.time()
            segmenter = Sam31SequenceSegmenter(checkpoint_path=args.checkpoint)
            provenance = segmenter.provenance("offline_bidirectional")
            sam_record = {"load_seconds": round(time.time() - began, 2), "runtime": runtime(),
                          "sdpa_backend_policy": segmenter.sdpa_backend_policy,
                          "model_revision": provenance.model_revision,
                          "checkpoint_sha256": provenance.checkpoint_sha256, "config_sha256": provenance.config_sha256,
                          "prob_threshold": segmenter.prob_threshold, "clips": sam_record.get("clips", {})}
            if checkpoints is not None:
                write_json(checkpoints / "sam.json", {**sam_record, "clips": {}})
        import torch

        torch.cuda.reset_peak_memory_stats()
        began = time.time()
        masks = sam_text_masks(segmenter, clip, work / "jpeg")
        seconds = time.time() - began
        target = publish / "clips" / safe(clip["id"])
        target.mkdir(parents=True, exist_ok=True)
        masks.save(target)
        sam_record["clips"][clip["id"]] = {
            "frames": len(masks), "seconds": round(seconds, 2), "frames_per_s": round(len(masks) / max(seconds, 1e-9), 3),
            "passes": len(set(clip["masks"]["prompts"].values())),
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "masks_rle_sha256": file_sha256(target / "masks.rle"),
            "frames_with_foreground": int(sum(1 for f in masks.frames if f)),
            "empty_chunks": masks.meta.get("empty_chunks", []),
            "gpu": sam_record["runtime"]["gpu"],
        }
        shutil.rmtree(work / "jpeg", ignore_errors=True)
        base = [to_analysis(union(f, (masks.height, masks.width))) for f in masks.frames]
        record = {"tier": SAM_TIER, "prompts": clip["masks"]["prompts"], **foreground(clip, work, base, args.visor_fill, args.hand_objects)}
        submit_analysis(clip["id"], analysis_task(clip, decoded, record))
        done += 1
        progress(done)
    for clip in other:  # VISOR windows: decode here, foreground from the dense masks, lens from the stretch
        decoded = decoder.submit(decode_clip, decode_task(clip, False)).result()
        work = scratch / "work" / safe(clip["id"])
        record = foreground(clip, work, None, args.visor_fill, args.hand_objects)
        lens = None
        if clip.get("lens_from"):
            if clip["lens_from"] not in futures:
                raise SystemExit(f"{clip['id']} needs {clip['lens_from']} in the same run")
            lens = futures[clip["lens_from"]].result()["lens"]
        submit_analysis(clip["id"], analysis_task(clip, decoded, record, lens))
    results = []
    for clip in clips:
        results.append(futures[clip["id"]].result())
        shutil.rmtree(scratch / "work" / safe(clip["id"]), ignore_errors=True)
        done += 1
        progress(done)
    decoder.shutdown()
    analyser.shutdown()
    test = results[0].get("self_test") if results else None
    write_json(stage_dir() / "g1.json", {
        "clips_file": {"path": str(spec_path), "sha256": file_sha256(spec_path), "name": spec.get("name")},
        "select": args.select, "limit_seconds": args.limit_seconds,
        "settings": {k: getattr(camera, k) for k in (
            "ANALYSIS_SIZE", "DILATE", "MAX_FEATURES", "RATIO", "RANSAC_PX", "FIT_SCALE", "MIN_INLIERS", "MIN_INLIER_SHARE",
            "KEYFRAME_OVERLAP", "KEYFRAME_INLIERS", "TEXTURE", "EXPLAINED_PX", "MOVED_PX", "EPIPOLAR_PX",
            "GRIC_SIGMA", "TRANSLATION_SECONDS", "COVER_CELL", "PRIOR_HFOV_DEG")},
        "decision": DECISION, "sam": sam_record, "self_test": test,
        "restored_clips": sorted(restored),
        "opencv": cv2.__version__, "results": results,
    })
    return 0


# ----------------------------------------------------------------- validate


def unit(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and 0.0 <= value <= 1.0)


def validate_stage(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "g1.json").read_text())
    rows = result["results"]
    clips = {c["id"]: c for c in json.loads(Path(result["clips_file"]["path"]).read_text())["clips"]} \
        if Path(result["clips_file"]["path"]).is_file() else {}
    sam = result["sam"]
    sam_clips = [r for r in rows if (r.get("masks") or {}).get("tier") == SAM_TIER]
    windows = [r for r in rows if r["group"] == "visor-window"]
    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    checks = {
        "clips_analysed": bool(rows),
        "every_clip_has_result_and_images": all(
            f"publish/clips/{safe(r['id'])}/result.json" in names
            and all(f"publish/clips/{safe(r['id'])}/{i}" in names for i in r["images"]) for r in rows),
        "decoded_every_analysis_frame": all(r["decode"]["frames"] == r["summary"]["frames"] for r in rows),
        "clip_file_matches": not clips or file_sha256(Path(result["clips_file"]["path"])) == result["clips_file"]["sha256"],
        "self_test_recovers_known_rotation": bool(result["self_test"] and result["self_test"]["passed"]),
        "visor_window_jpegs_match_decoded_frames": all(
            all(g["holds"] for g in r["decode"]["jpeg_gate"])
            and (bool(r["decode"]["jpeg_gate"]) or not r["clip"]["item"]["sparse_jpegs"]) for r in windows),
        "visor_windows_use_the_fill": all(r["masks"]["tier"] == "visor_dense_sam_fill" for r in windows),
        "frames_registered": all(r["summary"]["status"].get("direct", 0) + r["summary"]["status"].get("chained", 0) > 0
                                 for r in rows if r["summary"]["frames"] > 1),
        "shares_in_unit_interval": all(
            unit(r["summary"]["f2r_rot"]["explained_share"]) and unit(r["summary"]["direct_share"])
            and all(unit(c) for c in r["coverage"]["curve"]) for r in rows),
        "coverage_curves_computed": all(len(r["coverage"]["curve"]) >= 1 for r in rows),
        "translation_test_ran": any(r["summary"]["translation_half_s"]["pairs"] > 0 for r in rows),
    }
    if sam_clips:
        runtime = sam["runtime"]
        checks.update({
            "sam_on_ada_or_a6000": any(name in runtime["gpu"] for name in ("RTX 6000 Ada", "RTX A6000")),
            "sam_pinned_checkpoint": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
            "sam_pinned_code": sam["model_revision"] == "2345a4ad109ac29c569da749c91d84f10dc08c40",
            "sam_native_attention": sam["sdpa_backend_policy"] != "efficient_then_math_fallback",
            "sam_masks_published": all(f"publish/clips/{safe(r['id'])}/masks.rle" in names for r in sam_clips),
            "sam_finds_foreground": all(v["frames_with_foreground"] > 0 for v in sam["clips"].values()),
        })
    visor_long = [r for r in rows if r["group"] == "visor-long"]
    if visor_long:
        checks["visor_detector_boxes_used"] = all(r["masks"]["detector"]["boxes"] > 0 for r in visor_long)
        checks["visor_tier_check_measured"] = all((r["masks"].get("dense_recall") or {}).get("frames", 0) > 0
                                                  for r in visor_long if overlaps_window(r["clip"]))
    return checks


def overlaps_window(clip: dict[str, Any]) -> bool:
    """Whether a stretch's analysed range (capped in a smoke) reaches its evaluation window."""
    window = clip["masks"].get("window_item")
    if not window:
        return False
    first = window["first_video_index"]
    return clip["first_index"] <= first + window["frames"] - 1 and first <= clip["last_index"]


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_stage(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--clips", required=True)
    run.add_argument("--select", required=True, help="all, or comma-separated clip ids")
    run.add_argument("--limit-seconds", type=float, default=0.0, help="cap every clip at this length (0: no cap)")
    run.add_argument("--source", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    run.add_argument("--visor-archive", help="extracted B1 archive: frame_mapping.json, rgb_frames/")
    run.add_argument("--visor-fill", help="extracted B1b fill (publish/masks/<item>/masks.rle)")
    run.add_argument("--hand-objects", help="extracted hand-object detections: hand-objects/<P>/<video>.pkl")
    run.add_argument("--checkpoint", help="SAM 3.1 checkpoint")
    sub.add_parser("validate")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True, help="a run's g1.json")
    report.add_argument("--published", action="append", default=[], help="that run's extracted published.tar")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        from experiments.background.g1_report import command_report

        return command_report(args)
    return {"run": command_run, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
