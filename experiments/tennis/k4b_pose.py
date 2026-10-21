"""K4b.2: YOLO11x-pose on each hand-identified player's crop, enlarged ×1, ×2, ×4.

The players are the canonical K4b.1 tracks the user's marks select on each
window's marked frame (``hand_players``, run locally); the GPU job reads only
the frozen track keys. Per window: K4's conversion, then every player-frame's
crop at each magnification, K4's whole-frame detections (K4b.1's) beside them,
and the synthetic far player (near players reduced to the far player's height).
Everything the validator recomputes is published with the window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4, k4b_rules as rules, windows as tw
from experiments.tennis.k4b import iou, numbered
from experiments.tennis.k4b_features import convert, key
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.segmentation.masks import ClipMasks, compress, decompress

FRAME = (1920, 1080)
SCALES = (1, 2, 4)
SETTINGS = ("whole", *(f"crop{s}" for s in SCALES))
#: COCO keypoint sigmas (pycocotools), for OKS.
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
RULE: dict[str, Any] = {
    "version": 1, "model": "yolo11x-pose", "conf": k4.POSE_CONF, "keypoint_confidence": k4.KP_CONF,
    "whole_frame_imgsz": k4.POSE_IMGSZ, "scales": list(SCALES), "margin_of_height": .2, "long_side_cap": 1280,
    "stride": 32, "pad_value": 114, "upscale": "bicubic", "reduce": "area",
    "inside_dilation_px": k4.KP_DILATE_1080, "match": "H1: >= 0.5 of the detection's box inside the mask's box",
    "synthetic": {"every": 8, "near_min_height": .15, "oks": .5},
    "pass": {"fitted_all": .9, "fitted_far": .9, "inside_median": .9, "synthetic_oks_share": .9},
    "choose": "smallest passing scale",
    "height_bins": [0, .06, .10, .15, .25, 1.01],
}


def hand_players(marks: dict[str, Any], review: dict[str, Any], review_path: Path, review_dir: Path,
                 windows: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Window id -> the canonical tracks the user's marks select on its marked frame (IoU >= 0.5, one to one)."""
    labels = rules.validate_marks(marks, review, review_path)
    out = {}
    for frame in review["frames"]:
        wid = f"{frame['match']}/{frame['clip']}@{frame['frame']}"
        root = windows[wid]
        feature = json.loads((root / "features.json").read_text())
        names = {v: k for k, v in feature["track_ids"].items()}
        by_track = {names[i.track_id]: k4.upsample(i.mask(), FRAME) for i in ClipMasks.load(root / "target-masks.rle").frames[0]}
        if file_sha256(review_dir / frame["masks"]) != frame["masks_sha256"]:
            raise ValueError("Hand-marked review mask identity changed")
        by_id = {r["id"]: k4.upsample(r["mask"], FRAME) for r in numbered(ClipMasks.load(review_dir / frame["masks"]))}
        label = labels[frame["id"]]
        chosen: list[str] = []
        overlaps = []
        for n in label["selected"]:
            best, track = max(((iou(m, by_id[n]), k) for k, m in by_track.items() if k not in chosen), default=(0., ""))
            if best < .5:
                raise ValueError(f"{wid}: marked instance {n} has no track at IoU >= 0.5")
            chosen.append(track)
            overlaps.append(round(best, 4))
        out[wid] = {"players": chosen, "iou": overlaps, "visible": label["visible"],
                    "missing_instance": label["missing_instance"], "review_frame": frame["id"]}
    return out


def player_masks(frame: list[Any], aliases: dict[str, str], players: list[str]) -> dict[str, np.ndarray]:
    """Each player's SAM mask on one frame: its person mask where both prompts have one (K4b.1's order)."""
    out: dict[str, np.ndarray] = {}
    for inst in sorted(frame, key=lambda i: (i.class_name, i.track_id)):
        if inst.class_name in ("person", "player"):
            canonical = aliases[key(inst.class_name, inst.track_id)]
            if canonical in players and canonical not in out:
                out[canonical] = inst.mask()
    return out


def crop_box(box: tuple[float, float, float, float], margin: float = RULE["margin_of_height"]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad = margin * (y1 - y0)
    return (max(0, int(math.floor(x0 - pad))), max(0, int(math.floor(y0 - pad))),
            min(FRAME[0], int(math.ceil(x1 + pad))), min(FRAME[1], int(math.ceil(y1 + pad))))


def magnify(image: np.ndarray, scale: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Resize by ``scale`` (bicubic up, area down), long side capped; returns the image and its (x, y) factors."""
    import cv2

    h, w = image.shape[:2]
    factor = min(scale, RULE["long_side_cap"] / max(h, w))
    if factor == 1:
        return image, (1., 1.)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    out = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
    return out, (size[0] / w, size[1] / h)


def bucket(image: np.ndarray) -> int:
    return max(64, int(math.ceil(max(image.shape[:2]) / RULE["stride"]) * RULE["stride"]))


def pad_square(image: np.ndarray, side: int) -> np.ndarray:
    out = np.full((side, side, 3), RULE["pad_value"], np.uint8)
    out[:image.shape[0], :image.shape[1]] = image
    return out


def fit(mask: np.ndarray, boxes: np.ndarray, kps: np.ndarray) -> dict[str, Any]:
    """K4's fit and keypoints-inside measure for one player against detections in frame coordinates."""
    from experiments.visor.h1 import match_detections

    boxes, kps = np.round(boxes, 2), np.round(kps, 2)  # as published, so the validator recomputes exactly
    pairs = match_detections({"p": mask}, boxes) if len(boxes) else {}
    if "p" not in pairs:
        return {"fitted": False}
    kp = kps[pairs["p"]]
    confident = kp[:, 2] >= k4.KP_CONF
    inside = within(mask, kp) & confident
    return {"fitted": True, "confident": int(confident.sum()),
            "inside": float(inside.sum() / confident.sum()) if confident.any() else None,
            "box": boxes[pairs["p"]].tolist(), "kps": kp.tolist()}


def within(mask: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Which keypoints fall inside the mask dilated by K4's 8 px (dilated only around the mask: same result)."""
    ys, xs = np.nonzero(mask)
    grow = k4.KP_DILATE_1080
    x0, y0 = max(0, int(xs.min()) - grow), max(0, int(ys.min()) - grow)
    x1, y1 = min(FRAME[0], int(xs.max()) + grow + 1), min(FRAME[1], int(ys.max()) + grow + 1)
    grown = k4.dilate(mask[y0:y1, x0:x1], grow)
    px = np.clip(np.round(kp[:, 0]).astype(int), 0, FRAME[0] - 1)
    py = np.clip(np.round(kp[:, 1]).astype(int), 0, FRAME[1] - 1)
    hit = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
    out = np.zeros(len(kp), bool)
    out[hit] = grown[py[hit] - y0, px[hit] - x0]
    return out


def oks(pred: list[list[float]] | None, ref: list[list[float]], area: float) -> float | None:
    """COCO OKS of ``pred`` against ``ref``'s confident keypoints; a missing prediction scores 0."""
    r = np.asarray(ref)
    visible = r[:, 2] >= k4.KP_CONF
    if not visible.any():
        return None
    if pred is None:
        return 0.
    d2 = ((np.asarray(pred)[:, :2] - r[:, :2]) ** 2).sum(1)
    variances = (2 * SIGMAS) ** 2
    return float(np.exp(-d2 / variances / (area + np.spacing(1)) / 2)[visible].mean())


def frame_dets(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(raw["boxes"], np.float32).reshape(-1, 4), np.asarray(raw["kps"], np.float32).reshape(-1, 17, 3))


def synchronize() -> None:
    """Wait for queued GPU work, so a timer measures it (no-op without CUDA)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Crops:
    """Batches same-size padded crops through YOLO and maps detections back to frame coordinates."""

    def __init__(self, poses: Any, batch: int = 16):
        self.poses, self.batch = poses, batch
        self.queue: dict[int, list[tuple[np.ndarray, Any, tuple[float, float, int, int]]]] = defaultdict(list)
        self.seconds: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def add(self, image: np.ndarray, factors: tuple[float, float], origin: tuple[int, int], tag: Any) -> None:
        self.queue[bucket(image)].append((pad_square(image, bucket(image)), tag, (*factors, *origin)))

    def run(self, group: str) -> list[tuple[Any, np.ndarray, np.ndarray]]:
        model = self.poses.models["yolo11x-pose"]
        out = []
        for side, items in sorted(self.queue.items()):
            for s in range(0, len(items), self.batch):
                chunk = items[s:s + self.batch]
                synchronize()
                began = time.perf_counter()
                results = model.predict([np.ascontiguousarray(c[0][:, :, ::-1]) for c in chunk], imgsz=side, conf=k4.POSE_CONF,
                                        classes=[0], device=self.poses.device, verbose=False)
                synchronize()
                self.seconds[group] += time.perf_counter() - began
                self.count[group] += len(chunk)
                for (_, tag, (fx, fy, ox, oy)), r in zip(chunk, results):
                    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32) if r.boxes is not None else np.zeros((0, 4), np.float32)
                    kps = r.keypoints.data.cpu().numpy().astype(np.float32) if r.keypoints is not None else np.zeros((0, 17, 3), np.float32)
                    boxes = boxes / np.array([fx, fy, fx, fy], np.float32) + np.array([ox, oy, ox, oy], np.float32)
                    kps[..., 0] = kps[..., 0] / fx + ox
                    kps[..., 1] = kps[..., 1] / fy + oy
                    out.append((tag, boxes, kps))
        self.queue.clear()
        return out


def window_heights(clip: ClipMasks, aliases: dict[str, str], players: list[str]) -> dict[str, dict[int, tuple[float, ...]]]:
    """Each player's mask box at 1920x1080 on every frame it has a mask."""
    from experiments.visor.h1 import box_of

    boxes: dict[str, dict[int, tuple[float, ...]]] = {p: {} for p in players}
    for t, frame in enumerate(clip.frames):
        for track, m in player_masks(frame, aliases, players).items():
            box = box_of(k4.upsample(m, FRAME))
            if box is not None:
                boxes[track][t] = box
    return boxes


def roles(boxes: dict[str, dict[int, tuple[float, ...]]]) -> dict[str, str]:
    """Far and near by median mask height when two players are marked; otherwise unassigned."""
    if len(boxes) != 2 or not all(boxes.values()):
        return {p: "single" for p in boxes}
    median = {p: float(np.median([b[3] - b[1] for b in v.values()])) for p, v in boxes.items()}
    far = min(sorted(median), key=lambda p: median[p])
    return {p: "far" if p == far else "near" for p in boxes}


def synthetic_plan(boxes: dict[str, dict[int, tuple[float, ...]]], role: dict[str, str]) -> dict[str, Any] | None:
    """The near player's sampled frames and its reduction to the far player's median height, if eligible."""
    if sorted(role.values()) != ["far", "near"]:
        return None
    near = next(p for p, r in role.items() if r == "near")
    far = next(p for p, r in role.items() if r == "far")
    near_h = float(np.median([b[3] - b[1] for b in boxes[near].values()]))
    far_h = float(np.median([b[3] - b[1] for b in boxes[far].values()]))
    if near_h < RULE["synthetic"]["near_min_height"] * FRAME[1]:
        return None
    frames = [t for t in sorted(boxes[near]) if t % RULE["synthetic"]["every"] == 0]
    return {"player": near, "reduce": far_h / near_h, "near_height": near_h, "far_height": far_h, "frames": frames}


def analyse_window(poses: Any, rgb_of: Any, clip: ClipMasks, features: dict[str, Any], players: list[str],
                   dets: list[dict[str, Any]], review: list[int], out: Path) -> dict[str, Any]:
    """Every player-frame at every setting, the synthetic far player and the review images of one window."""
    import cv2

    aliases = features["aliases"]
    boxes = window_heights(clip, aliases, players)
    role = roles(boxes)
    synth = synthetic_plan(boxes, role)
    crops = Crops(poses)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    synth_rows: dict[int, dict[str, Any]] = {}
    reviews: dict[tuple[int, str], dict[str, tuple[np.ndarray, tuple[float, float]]]] = defaultdict(dict)
    chunk = 32
    for start in range(0, len(clip.frames), chunk):
        masks: dict[tuple[int, str], np.ndarray] = {}
        for t in range(start, min(start + chunk, len(clip.frames))):
            present = {p: b for p in players if (b := boxes[p].get(t)) is not None}
            if not present:
                continue
            rgb = rgb_of(t)
            for p, box in present.items():
                masks[(t, p)] = k4.upsample(player_masks(clip.frames[t], aliases, [p])[p], FRAME)
                x0, y0, x1, y1 = crop_box(box)  # type: ignore[arg-type]
                base = rgb[y0:y1, x0:x1]
                for s in SCALES:
                    image, factors = magnify(base, s)
                    crops.add(image, factors, (x0, y0), (t, p, f"crop{s}"))
                    if t in review:
                        reviews[(t, p)][f"crop{s}"] = (image, factors)
                if synth is not None and p == synth["player"] and t in synth["frames"]:
                    small, (rx, ry) = magnify(base, synth["reduce"])
                    for s in SCALES:
                        image, (fx, fy) = magnify(small, s)
                        crops.add(image, (fx * rx, fy * ry), (x0, y0), (t, p, f"synthetic{s}"))
        results = crops.run("crop")
        whole: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for (t, p), m in masks.items():
            if t not in whole:
                whole[t] = frame_dets(dets[t])
            rows[(t, p)] = {"t": t, "track": p, "role": role[p], "height": round((boxes[p][t][3] - boxes[p][t][1]) / FRAME[1], 5),
                            "area": int(m.sum()), "settings": {"whole": fit_whole(m, masks, t, p, *whole[t])}}
        for (t, p, setting), found_boxes, found_kps in results:
            target = rows[(t, p)]
            result = fit(masks[(t, p)], found_boxes, found_kps)
            if setting.startswith("crop"):
                target["settings"][setting] = result
            else:
                synth_rows.setdefault(t, {"t": t, "track": p, "area": target["area"]})[setting] = result.get("kps")
    for t, row in synth_rows.items():
        reference = rows[(t, row["track"])]["settings"]["crop1"]
        row["reference"] = reference.get("kps")
        row["oks"] = {f"synthetic{s}": (oks(row[f"synthetic{s}"], reference["kps"], row["area"]) if reference["fitted"] else None)
                      for s in SCALES}
    for (t, p), images in reviews.items():
        panels = []
        for s in SCALES:
            image, (fx, fy) = images[f"crop{s}"]
            image = image.copy()
            kp = rows[(t, p)]["settings"][f"crop{s}"].get("kps")
            x0, y0, _, _ = crop_box(boxes[p][t])  # type: ignore[arg-type]
            if kp is not None:
                pts = [((x - x0) * fx, (y - y0) * fy, c) for x, y, c in kp]
                for i, j in k4.SKELETON:
                    if pts[i][2] >= k4.KP_CONF and pts[j][2] >= k4.KP_CONF:
                        cv2.line(image, (int(pts[i][0]), int(pts[i][1])), (int(pts[j][0]), int(pts[j][1])), (0, 255, 0), 2)
            panels.append(cv2.resize(image, (max(1, round(image.shape[1] * 320 / image.shape[0])), 320)))
        canvas = np.concatenate(panels, 1)
        cv2.putText(canvas, f"t={t} {role[p]} x1 x2 x4", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"review-t{t:03d}-{role[p]}.png"), canvas[:, :, ::-1])
    return {"rows": sorted(rows.values(), key=lambda r: (r["t"], r["track"])), "roles": role,
            "synthetic": {"plan": synth, "rows": sorted(synth_rows.values(), key=lambda r: r["t"])},
            "seconds": dict(crops.seconds), "crops": dict(crops.count)}


def fit_whole(mask: np.ndarray, masks: dict[tuple[int, str], np.ndarray], t: int, p: str,
              boxes: np.ndarray, kps: np.ndarray) -> dict[str, Any]:
    """K4's whole-frame matching: one to one over the frame's players, then this player's fit."""
    from experiments.visor.h1 import match_detections

    frame = {q: m for (u, q), m in masks.items() if u == t}
    pairs = match_detections(frame, boxes) if len(boxes) else {}
    if p not in pairs:
        return {"fitted": False}
    single = fit(mask, boxes[pairs[p]:pairs[p] + 1], kps[pairs[p]:pairs[p] + 1])
    return single


def review_frames(window: dict[str, Any]) -> list[int]:
    """The marked frame and one content-blind frame (K4's seed scheme)."""
    digest = int(hashlib.sha256(f"pointstream-k4b2-review:{window['id']}".encode()).hexdigest(), 16)
    return sorted({window["target"], digest % window["frames"]})


def run(args: argparse.Namespace) -> int:
    began = time.perf_counter()
    import torch

    from src.codecs import svtav1

    inputs = Path(args.inputs)
    index = json.loads((inputs / "inputs.json").read_text())
    chosen = args.windows[:args.max_windows or None]
    if len(args.windows) != len(args.sources) or any(w not in index["windows"] for w in args.windows):
        raise ValueError("Give one source per window, every window in the inputs")
    sources = dict(zip(args.windows, args.sources))
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "12000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    poses = k4.Poses([("yolo11x-pose", args.pose)])
    rows = []
    for n, wid in enumerate(chosen):
        entry = index["windows"][wid]
        window = entry["window"]
        root = inputs / "windows" / k4.safe(wid)
        for name, digest in entry["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Changed input: {root / name}")
        if file_sha256(Path(sources[wid])) != window["sha256"]:
            raise ValueError(f"Original clip identity mismatch: {wid}")
        out = publish / k4.safe(wid)
        out.mkdir()
        work = scratch / "work" / k4.safe(wid)
        work.mkdir(parents=True)
        yuv = work / "source.yuv"
        converted = convert(args.ffmpeg, window, Path(sources[wid]), yuv)
        if converted["sha256"] != window["conversion"]["sha256"]:
            raise ValueError(f"{wid}: conversion differs from K4b.1's")
        raw = np.memmap(yuv, mode="r", dtype=np.uint8, shape=(window["frames"], 1620, 1920))
        clip = ClipMasks.load(root / "sam-masks.rle")
        features = json.loads((root / "features.json").read_text())
        dets = json.loads(decompress((root / "pose.json.zst").read_bytes()))
        frames = np.stack([tw.yuv_to_rgb(np.asarray(raw[t]), 1920, 1080) for t in range(min(16, window["frames"]))])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        poses.run("yolo11x-pose", frames, batch=8)
        torch.cuda.synchronize()
        whole_ms = (time.perf_counter() - t0) / len(frames) * 1000
        del frames
        torch.cuda.reset_peak_memory_stats()
        def rgb_of(t: int, raw: np.ndarray = raw) -> np.ndarray:
            return tw.yuv_to_rgb(np.asarray(raw[t]), 1920, 1080)

        result = analyse_window(poses, rgb_of, clip, features, entry["players"], dets, review_frames(window), out)
        peak = torch.cuda.max_memory_allocated() / 2**30
        del raw
        shutil.rmtree(work)
        (out / "pose-crops.json.zst").write_bytes(compress(json.dumps(result).encode()))
        for name in ("sam-masks.rle", "features.json"):
            shutil.copyfile(root / name, out / name)
        row = {"id": wid, "frames": window["frames"], "players": entry["players"], "conversion": converted,
               "whole_frame_ms": round(whole_ms, 3), "crop_seconds": result["seconds"], "crops": result["crops"],
               "gpu_peak_gib": round(peak, 3), "files": {p.name: file_sha256(p) for p in sorted(out.iterdir())}}
        write_json(out / "window.json", row)
        rows.append(row)
        print(f"window={wid} player_frames={len(result['rows'])} crops={result['crops']} peak_gib={peak:.2f}", flush=True)
        progress(n + 1)
    summary = {"stage": os.environ.get("PS_STAGE"), "requested": args.windows, "configured_windows": int(args.max_windows),
               "windows": rows, "inputs_index_sha256": file_sha256(inputs / "inputs.json"), "rule": RULE, "pose": poses.record(),
               "gpu_peak_gib": max([0.] + [r["gpu_peak_gib"] for r in rows]), "gpu_memory_limit_mib": limit,
               "ffmpeg": svtav1.tool(args.ffmpeg), "seconds": round(time.perf_counter() - began, 2)}
    write_json(stage_dir() / "k4b-pose.json", summary)
    return 0


# ----------------------------------------------------------------- validation and report


def recompute(root: Path, players: list[str]) -> list[str]:
    """Problems in one published window: each setting's fit and inside recomputed from the masks and keypoints."""
    problems = []
    result = json.loads(decompress((root / "pose-crops.json.zst").read_bytes()))
    clip = ClipMasks.load(root / "sam-masks.rle")
    aliases = json.loads((root / "features.json").read_text())["aliases"]
    boxes = window_heights(clip, aliases, players)
    expected = {(t, p) for p in players for t in boxes[p]}
    got = {(r["t"], r["track"]) for r in result["rows"]}
    if got != expected:
        problems.append("player-frames differ from the masks")
    if result["roles"] != roles(boxes):
        problems.append("roles differ")
    by_frame: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for r in result["rows"]:
        by_frame[r["t"]][r["track"]] = k4.upsample(player_masks(clip.frames[r["t"]], aliases, [r["track"]])[r["track"]], FRAME)
    for r in result["rows"]:
        if set(r["settings"]) != set(SETTINGS):
            problems.append(f"t={r['t']}: settings")
            continue
        mask = by_frame[r["t"]][r["track"]]
        if r["area"] != int(mask.sum()):
            problems.append(f"t={r['t']}: area")
        for name, s in r["settings"].items():
            if not s["fitted"]:
                continue
            kp = np.asarray(s["kps"], np.float32)
            if kp.shape != (17, 3) or not np.isfinite(kp).all():
                problems.append(f"t={r['t']} {name}: keypoints")
                continue
            again = fit(mask, np.asarray([s["box"]], np.float32), kp[None])
            if not again["fitted"] or again["confident"] != s["confident"] or again["inside"] != s["inside"]:
                problems.append(f"t={r['t']} {name}: fit not recomputed")
    plan = synthetic_plan(boxes, result["roles"])
    if plan != result["synthetic"]["plan"]:
        problems.append("synthetic plan")
    rows_by = {(r["t"], r["track"]): r for r in result["rows"]}
    for s in result["synthetic"]["rows"]:
        ref = rows_by[(s["t"], s["track"])]["settings"]["crop1"]
        for scale in SCALES:
            want = oks(s[f"synthetic{scale}"], ref["kps"], s["area"]) if ref["fitted"] else None
            saved = s["oks"][f"synthetic{scale}"]
            # exp and mean differ by an ulp between platforms' numpy builds (4e-16 seen Linux vs macOS)
            if (want is None) != (saved is None) or (want is not None and abs(want - saved) > 1e-9):
                problems.append(f"t={s['t']}: oks{scale}")
    return problems


def validate(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    windows = summary["windows"]
    requested = summary["requested"][:summary["configured_windows"] or None]
    files = recomputed = settings_ok = reviews = True
    problems: dict[str, list[str]] = {}
    for w in windows:
        root = directory / k4.safe(w["id"])
        files &= all(file_sha256(root / n) == h for n, h in w["files"].items())
        found = recompute(root, w["players"])
        if found:
            problems[w["id"]] = found[:5]
        recomputed &= not found
        result = json.loads(decompress((root / "pose-crops.json.zst").read_bytes()))
        settings_ok &= all(set(r["settings"]) == set(SETTINGS) for r in result["rows"])
        reviews &= any(p.name.startswith("review-") for p in root.iterdir()) or not w["players"]
    checks = {
        "windows_requested": bool(windows) and [w["id"] for w in windows] == requested,
        "files_identified": files,
        "every_setting_on_every_player_frame": settings_ok,
        "fit_inside_oks_recomputed": recomputed,
        "review_images": reviews,
        "frozen_rule": summary["rule"] == RULE,
        "pose_cuda": all(v.startswith("cuda") for v in summary["pose"]["on_cuda"].values()),
        "crops_timed": all(w["crops"].get("crop", 0) > 0 for w in windows if w["players"]),
        "peak_within_declaration": summary["gpu_peak_gib"] * 1024 <= summary["gpu_memory_limit_mib"],
    }
    return {"passed": all(checks.values()), "checks": checks, "problems": problems,
            "windows": [w["id"] for w in windows], "frames": sum(w["frames"] for w in windows)}


def summarize(results: list[dict[str, Any]], latency: dict[str, float | None]) -> dict[str, Any]:
    """The rule's measures per setting over every window's rows (hand-identified players)."""
    bins = RULE["height_bins"]
    table: dict[str, Any] = {}
    for setting in SETTINGS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for res in results:
            for r in res["rows"]:
                s = r["settings"][setting]
                groups["all"].append(s)
                groups[r["role"]].append(s)
                b = next(i for i in range(len(bins) - 1) if bins[i] <= r["height"] < bins[i + 1])
                groups[f"height {bins[b]:.2f}-{min(bins[b + 1], 1):.2f}"].append(s)
        entry: dict[str, Any] = {}
        for name, items in sorted(groups.items()):
            inside = [s["inside"] for s in items if s["fitted"] and s["inside"] is not None]
            entry[name] = {"player_frames": len(items), "fitted": sum(s["fitted"] for s in items),
                           "fitted_share": sum(s["fitted"] for s in items) / len(items),
                           "inside_median": float(np.median(inside)) if inside else None}
        table[setting] = entry
    synthetic = {}
    for s in SCALES:
        values = [r["oks"][f"synthetic{s}"] for res in results for r in res["synthetic"]["rows"]
                  if r["oks"][f"synthetic{s}"] is not None]
        synthetic[f"crop{s}"] = {"samples": len(values), "share_oks_ge": sum(v >= RULE["synthetic"]["oks"] for v in values) / len(values)
                                 if values else None, "median_oks": float(np.median(values)) if values else None}
    passing = []
    bar = RULE["pass"]
    for s in SCALES:
        t, syn = table[f"crop{s}"], synthetic[f"crop{s}"]
        ok = (t["all"]["fitted_share"] >= bar["fitted_all"] and t.get("far", {}).get("fitted_share", 0) >= bar["fitted_far"]
              and (t["all"]["inside_median"] or 0) >= bar["inside_median"] and (t.get("far", {}).get("inside_median") or 0) >= bar["inside_median"]
              and (syn["share_oks_ge"] or 0) >= bar["synthetic_oks_share"])
        if ok:
            passing.append(s)
    return {"table": table, "synthetic": synthetic, "latency_ms": latency, "passing_scales": passing,
            "chosen_scale": min(passing) if passing else None,
            "next": "re-read K4 (b) with the chosen crop; K4b.3 on its keypoints" if passing else "stage 2: DWPose on the same crops"}


def report(jobs: list[Path], partial: bool = False) -> dict[str, Any]:
    """The rule's table from locally validated full stages (each job directory: ready.json, the stage's
    k4b-pose.json, local-validation.json and its extracted publish/)."""
    results, sources, ids = [], [], []
    seconds: dict[str, float] = defaultdict(float)
    crops: dict[str, int] = defaultdict(int)
    whole = []
    for root in jobs:
        validation = json.loads((root / "local-validation.json").read_text())
        summary = json.loads((root / "k4b-pose.json").read_text())
        ready = json.loads((root / "ready.json").read_text())
        if (validation.get("passed") is not True or not all(validation["checks"].values()) or summary["stage"] != "full"
                or summary["configured_windows"] != 0 or validation["windows"] != [w["id"] for w in summary["windows"]]):
            raise ValueError(f"{root}: only locally validated full stages may be reported")
        for w in summary["windows"]:
            if w["id"] in ids:
                raise ValueError(f"Duplicate window: {w['id']}")
            ids.append(w["id"])
            directory = root / "publish" / k4.safe(w["id"])
            if file_sha256(directory / "pose-crops.json.zst") != w["files"]["pose-crops.json.zst"]:
                raise ValueError(f"Changed window artifact: {directory}")
            results.append(json.loads(decompress((directory / "pose-crops.json.zst").read_bytes())))
            for k, v in w["crop_seconds"].items():
                seconds[k] += v
            for k, n in w["crops"].items():
                crops[k] += n
            whole.append(w["whole_frame_ms"])
        sources.append({"job": root.name, "git_head": ready["git_head"], "gpu": summary["pose"]["device"],
                        "windows": len(summary["windows"]), "k4b_pose_sha256": file_sha256(root / "k4b-pose.json")})
    if not partial and len(ids) != 48:
        raise ValueError(f"{len(ids)} of 48 windows")
    latency = {"crop_ms_mean_all_scales": 1000 * seconds["crop"] / max(crops["crop"], 1),
               "whole_frame_ms_median": float(np.median(whole)) if whole else None}
    out = summarize(results, latency)
    out.update({"windows": sorted(ids), "jobs": sources, "complete": len(ids) == 48})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", required=True, help="the extracted inputs archive")
    r.add_argument("--pose", required=True)
    r.add_argument("--windows", nargs="+", required=True)
    r.add_argument("--sources", nargs="+", required=True)
    r.add_argument("--max-windows", type=int, required=True, help="0 runs every listed window")
    r.add_argument("--ffmpeg", default="ffmpeg")
    rp = sub.add_parser("report")
    rp.add_argument("jobs", nargs="+", type=Path)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--partial", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--directory", type=Path)
    v.add_argument("--summary", type=Path)
    args = p.parse_args()
    if args.command == "run":
        return run(args)
    if args.command == "report":
        result = report(args.jobs, args.partial)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("table", "windows")}, indent=1))
        return 0
    summary = json.loads((args.summary or stage_dir() / "k4b-pose.json").read_text())
    if args.directory:
        checked = validate(args.directory, summary)
    else:
        with tempfile.TemporaryDirectory(prefix="k4b-pose-validate-", dir="/tmp") as temp:
            with tarfile.open(stage_dir() / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            checked = validate(Path(temp) / "publish", summary)
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), checked)
    print(json.dumps({k: v for k, v in checked.items() if k != "problems"}))
    return 0 if checked["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
