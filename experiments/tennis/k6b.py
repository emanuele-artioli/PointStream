"""K6b (players) 2: zero-shot oracle of a learned player renderer (docs/experiments.md, 2026-10-22 K6b (players) 2).

    python -m experiments.tennis.k6b prepare --inputs DIR --k6-stage TAR ... --windows ID ... --sources MP4 ... \\
        --max-windows N
    python -m experiments.tennis.k6b_generate generate ...          (in pointstream-jcv; npz in, npz out)
    python -m experiments.tennis.k6b judge --inputs DIR --k6-stage TAR ... --prepared TAR --generated TAR ... \\
        --windows ID ... --sources MP4 ... --lpips-backbone PTH --dists-backbone PTH --max-windows N --max-judged N
    python -m experiments.tennis.k6b validate
    python -m experiments.tennis.k6b report --stage DIR ... --k6-report JSON --out JSON

On K6's scored windows, with K6's published players, poses and coded keypoints (K4b.3's code, every 4th frame):
``prepare`` (CPU) cuts, per player, one reference crop (the window's first fitted scored frame, granted and unbilled)
and, per run of coded frames, a crop that follows the coded keypoints, with the coded pose drawn as AnimateAnyone's
DWPose body map. ``k6b_generate`` renders each run with Moore-AnimateAnyone (both profiles). ``judge`` pastes the
rendered crops into each player's region R (the generator's matte, granted, as ``warp`` granted the outline; Telea fill
elsewhere in R, as K6) and scores them with K6's judges on K6's S and judged frames, beside K6's ``warp`` and ``ref``
recomputed as a check. ``report`` applies `DECISION` against K6's envelope at each window's keypoint rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4
from experiments.tennis.k6 import CHOSEN, FRAME, SIGMA_OF_HEIGHT, code_name
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

CODE = code_name(CHOSEN)  # q4-velocity-e4, K6's operating point
GEN_W, GEN_H = 512, 784  # AnimateAnyone's inference size (pose2vid.py's defaults)
CROP_MARGIN = 1.4  # crop height over the larger of the confident keypoints' height and their width at GEN's aspect
CENTRE_SMOOTH = 9  # frames in the moving average of the crop centre (decoder-side, coded keypoints only)
SIZE_PERCENTILE = 90  # a run's crop height: this percentile of its frames' needed heights
PROFILES = ("original", "finetuned_tennis")
GEN_VARIANTS = tuple(f"gen-{p}" for p in PROFILES)
CHECK_VARIANTS = (f"warp@{CODE}", f"ref@{CODE}")
VARIANTS = GEN_VARIANTS + CHECK_VARIANTS
REPRO_TOLERANCE = 0.002  # LPIPS over S, per window, against K6's recorded value
CRITERION = 0.75
#: Matches absent from the only surviving list of the fine-tune's tracks (pre-reset `pointstream_aa_meta.json`, 114
#: tracks over 7 videos, written 2026-07-11, after the Feb 2026 weights). The K6b audit records the fine-tune as having
#: seen all of K6's matches, so every window counts as seen; these are reported apart.
FINETUNE_UNLISTED = ("alcaraz_hurkacz", "medvedev_struff", "rublev_auger-aliassime")

DECISION: dict[str, Any] = {
    "version": 2,
    "items": "K6's scored windows (8, from 5 matches)",
    "quality": "K6's: LPIPS and DISTS over S (R plus every K6 oracle variant's drawn pixels), mean over K6's judged frames",
    "item_won": "gen's value is below K6's envelope (upper concave hull of the anchors on (kbps on R, -metric)) at the "
                "window's keypoint rate (K6's pooled rate of q4-velocity-e4), per metric",
    "score": "mean over {LPIPS, DISTS} of the share of windows won (the user's criterion, 2026-10-21)",
    "pass": "gen-original scores >= 0.75: the zero-shot route holds with the reference granted; next, the reference "
            "billed (K6c's interface)",
    "fail": "gen-original < 0.75: step 3 (overfit oracle) gets its rule and budget and waits for the user",
    "amendment_1": "(user, 2026-10-22, before any score: both profiles would need ~6.4 GPU-hours against a 4-hour cap) "
                   "gen-original alone decides, on all 8 windows; gen-finetuned_tennis is a diagnostic on the pilot "
                   "window, where it rendered the far player only (the near player drawn by gen-original in both)",
    "check": f"warp@{CODE} and ref@{CODE} reproduce K6's LPIPS over S within {REPRO_TOLERANCE} per window, and S's "
             "pixel counts equal K6's on every judged frame, or nothing is read",
    "beside": "K6's saving statistic (1 - keypoint kbps / envelope kbps, bootstrap CI by window), PSNR over S, the share "
              "of R the generator covers, the fine-tune's windows from matches absent from its surviving track list",
}


# ----------------------------------------------------------------- pose map (AnimateAnyone's DWPose body drawing)

#: OpenPose-18 slot <- COCO-17 index (DWPose's `wholebody.py`: neck is the shoulders' mean, slot 1).
OPENPOSE_FROM_COCO: tuple[int | None, ...] = (0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)
#: DWPose's `draw_bodypose` (Moore-AnimateAnyone `src/dwpose/util.py`, Apache-2.0): limbs (1-based) and colours (RGB).
LIMBS = ((2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14), (2, 1),
         (1, 15), (15, 17), (1, 16), (16, 18), (3, 17), (6, 18))
COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85))
STICK = 4


def openpose18(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO-17 (x, y, conf) to OpenPose-18 points and visibility (conf >= K4's 0.5; the neck needs both shoulders)."""
    pts = np.zeros((18, 2))
    vis = np.zeros(18, bool)
    for slot, j in enumerate(OPENPOSE_FROM_COCO):
        if j is None:
            pts[slot] = kp[[5, 6], :2].mean(0)
            vis[slot] = bool(kp[5, 2] >= k4.KP_CONF and kp[6, 2] >= k4.KP_CONF)
        else:
            pts[slot] = kp[j, :2]
            vis[slot] = bool(kp[j, 2] >= k4.KP_CONF)
    return pts, vis


def pose_map(kp: np.ndarray, width: int = GEN_W, height: int = GEN_H) -> np.ndarray:
    """The body map AnimateAnyone was trained on (DWPose's `draw_bodypose`: 17 limbs as ellipses, x0.6, then 18 joints
    as discs), for COCO keypoints already in canvas pixels. Hands and face are not drawn: the code carries the body."""
    import cv2

    pts, vis = openpose18(kp)
    canvas = np.zeros((height, width, 3), np.uint8)
    for i in range(17):
        a, b = LIMBS[i][0] - 1, LIMBS[i][1] - 1
        if not (vis[a] and vis[b]):
            continue
        (x0, y0), (x1, y1) = pts[a], pts[b]
        length = math.hypot(x0 - x1, y0 - y1)
        angle = math.degrees(math.atan2(y0 - y1, x0 - x1))
        poly = cv2.ellipse2Poly((int((x0 + x1) / 2), int((y0 + y1) / 2)), (int(length / 2), STICK), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, np.asarray(poly, np.int32), COLOURS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        if vis[i]:
            cv2.circle(canvas, (int(pts[i, 0]), int(pts[i, 1])), 4, COLOURS[i], thickness=-1)
    return canvas


# ----------------------------------------------------------------- crops (decoder-side geometry)


def needed_height(kp: np.ndarray) -> float | None:
    """Crop height that holds the confident keypoints with CROP_MARGIN, at GEN's aspect; None with < 2 confident."""
    conf = kp[:, 2] >= k4.KP_CONF
    if conf.sum() < 2:
        return None
    h = float(np.ptp(kp[conf, 1]))
    w = float(np.ptp(kp[conf, 0]))
    return CROP_MARGIN * max(h, w * GEN_H / GEN_W, 8.0)


def keypoint_centre(kp: np.ndarray) -> np.ndarray | None:
    conf = kp[:, 2] >= k4.KP_CONF
    if not conf.any():
        return None
    return (kp[conf, :2].min(0) + kp[conf, :2].max(0)) / 2


def run_boxes(kps: list[np.ndarray]) -> np.ndarray:
    """Per frame of a run, (x0, y0, w, h): one height per run (SIZE_PERCENTILE of the needed heights), centres from the
    confident keypoints' boxes (frames without one take the nearest frame's), smoothed by a CENTRE_SMOOTH-frame moving
    average with edge padding. Only the coded keypoints are used, so a decoder computes the same boxes."""
    heights = [h for h in (needed_height(k) for k in kps) if h is not None]
    height = float(np.percentile(heights, SIZE_PERCENTILE)) if heights else 64.0
    width = height * GEN_W / GEN_H
    raw = [keypoint_centre(k) for k in kps]
    known = [i for i, c in enumerate(raw) if c is not None]
    if not known:
        raise ValueError("a run without a confident keypoint")
    centres = np.stack([raw[i] if raw[i] is not None else raw[min(known, key=lambda j: abs(j - i))] for i in range(len(kps))])  # type: ignore[misc]
    half = CENTRE_SMOOTH // 2
    padded = np.pad(centres, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(CENTRE_SMOOTH) / CENTRE_SMOOTH
    smooth = np.stack([np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)], 1)
    return np.concatenate([smooth - [width / 2, height / 2], np.tile([width, height], (len(kps), 1))], 1)


def inside_frame(box: np.ndarray, frame: tuple[int, int] = FRAME) -> np.ndarray:
    """The box shifted to lie inside the frame where it fits (FoundHand copied a padded band into its output, K6b hands)."""
    x0, y0, w, h = (float(v) for v in box)
    if w <= frame[0]:
        x0 = min(max(x0, 0.0), frame[0] - w)
    if h <= frame[1]:
        y0 = min(max(y0, 0.0), frame[1] - h)
    return np.array([x0, y0, w, h])


def int_box(box: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0, w, h = (float(v) for v in box)
    return int(round(x0)), int(round(y0)), max(1, int(round(w))), max(1, int(round(h)))


def to_crop(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Keypoints in the crop's GEN_W x GEN_H canvas (confidence kept)."""
    x0, y0, w, h = int_box(box)
    out = kp.astype(np.float64).copy()
    out[:, 0] = (kp[:, 0] - x0) * GEN_W / w
    out[:, 1] = (kp[:, 1] - y0) * GEN_H / h
    return out


def crop_image(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """The box's pixels (zero outside the image), resized to GEN (area)."""
    import cv2

    x0, y0, w, h = int_box(box)
    out = np.zeros((h, w, 3), np.uint8)
    H, W = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (GEN_W, GEN_H), interpolation=cv2.INTER_AREA)


def paste(canvas: np.ndarray, rendered: np.ndarray, box: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Write ``rendered`` (GEN, resized back to the box, Lanczos) onto ``canvas`` on ``mask`` within the box; returns
    the pixels written."""
    import cv2

    x0, y0, w, h = int_box(box)
    H, W = canvas.shape[:2]
    big = cv2.resize(rendered, (w, h), interpolation=cv2.INTER_LANCZOS4)
    written = np.zeros(mask.shape, bool)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if sx1 <= sx0 or sy1 <= sy0:
        return written
    window = mask[sy0:sy1, sx0:sx1]
    canvas[sy0:sy1, sx0:sx1][window] = big[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][window]
    written[sy0:sy1, sx0:sx1] = window
    return written


def runs(times: list[int]) -> list[list[int]]:
    """Consecutive frames (a gap starts a run), as K6's code tracks."""
    out: list[list[int]] = []
    for t in sorted(times):
        if out and t == out[-1][-1] + 1:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def unit_name(window_id: str, player: str) -> str:
    """File-safe, and free of '@', which separates a unit from its profile in the generator's outputs."""
    return f"{k4.safe(window_id).replace('@', '_at')}__{player.replace(':', '_')}"


def unit_seed(unit: str) -> int:
    """A seed from the unit's name alone, so samples do not depend on how units are split into jobs."""
    return int(hashlib.sha256(unit.encode()).hexdigest()[:7], 16)


# ----------------------------------------------------------------- K6's published windows


def extract_stages(tars: list[str], scratch: Path) -> dict[str, Path]:
    """K6's published.tar files, extracted; window id -> its published directory (with window.json)."""
    out: dict[str, Path] = {}
    for n, tar in enumerate(tars):
        root = scratch / "k6" / f"s{n}"
        if not root.is_dir():
            root.mkdir(parents=True)
            with tarfile.open(tar) as archive:
                archive.extractall(root, filter="data")
        for row_path in sorted((root / "publish").glob("*/window.json")):
            row = json.loads(row_path.read_text())
            if any(file_sha256(row_path.parent / f) != h for f, h in row["files"].items()):
                raise ValueError(f"changed K6 artifact under {row_path.parent}")
            out[row["id"]] = row_path.parent
    return out


def k6_window(directory: Path) -> dict[str, Any]:
    from src.segmentation.masks import decompress

    row = json.loads((directory / "window.json").read_text())
    poses = json.loads(decompress((directory / "poses.json.zst").read_bytes()))
    coded = json.loads(decompress((directory / "keypoints.json.zst").read_bytes()))["decoded"][CODE]
    decoded: dict[tuple[str, int], np.ndarray] = {}
    for key, value in coded.items():
        player, t = key.rsplit("@", 1)
        decoded[(player, int(t))] = np.asarray(value, np.float64)
    return {"row": row, "poses": poses, "decoded": decoded, "dir": directory}


def window_truth(args: argparse.Namespace, wid: str, source: Path, frames: int, work: Path, device: str) -> tuple[Any, dict[str, Any]]:
    """The window's frames as K6 converted them (K4's conversion, identity-checked against K6's inputs)."""
    from experiments.tennis.k4b_features import convert
    from experiments.visor.h3 import rgb_memmap

    inputs = json.loads((Path(args.inputs) / "inputs.json").read_text())
    w = inputs["windows"][wid]["window"]
    if file_sha256(source) != w["sha256"]:
        raise ValueError(f"{wid}: original clip identity mismatch")
    work.mkdir(parents=True, exist_ok=True)
    yuv = work / "source.yuv"
    converted = convert(args.ffmpeg, {**w, "frames": frames}, source, yuv)
    truth = rgb_memmap(yuv, work / "truth.npy", frames, FRAME[0], FRAME[1], False, device)
    yuv.unlink()
    return truth, converted


# ----------------------------------------------------------------- prepare (CPU)


def prepare_window(args: argparse.Namespace, wid: str, k6dir: Path, source: Path, scratch: Path, publish: Path) -> dict[str, Any]:
    w = k6_window(k6dir)
    row = w["row"]
    frames = int(row["frames"])
    truth, converted = window_truth(args, wid, source, frames, scratch / "work" / k4.safe(wid), "cpu")
    judged = set(row["judged"])
    units = []
    for player in row["players"]:
        coded_t = sorted(t for (p, t) in w["decoded"] if p == player and t < frames)
        fitted = [t for t in row["scored"] if t < frames and (rec := w["poses"][player][t]) is not None and rec.get("fitted")]
        if not coded_t or not fitted:
            continue
        t0 = fitted[0]
        ref_kp = np.asarray(w["poses"][player][t0]["kps"], np.float64)
        ref_box = inside_frame(run_boxes([ref_kp])[0])
        ref = crop_image(np.asarray(truth[t0]), ref_box)
        ts, run_id, boxes, maps = [], [], [], []
        for r, seq in enumerate(runs(coded_t)):
            kps = [w["decoded"][(player, t)] for t in seq]
            for t, kp, box in zip(seq, kps, run_boxes(kps)):
                ts.append(t)
                run_id.append(r)
                boxes.append(box)
                maps.append(pose_map(to_crop(kp, box)))
        name = unit_name(wid, player)
        np.savez_compressed(publish / "units" / f"{name}.npz", reference=ref, t=np.asarray(ts, np.int64),
                            run=np.asarray(run_id, np.int64), box=np.asarray(boxes, np.float64), pose=np.stack(maps),
                            judged=np.asarray([t in judged for t in ts]), reference_box=ref_box)
        units.append({"unit": name, "window": wid, "player": player, "reference_t": t0, "frames": len(ts),
                      "runs": int(max(run_id) + 1), "judged": int(sum(t in judged for t in ts)),
                      "crop_height_px": float(np.median([b[3] for b in boxes])), "seed": unit_seed(name)})
    shutil.rmtree(scratch / "work" / k4.safe(wid), ignore_errors=True)
    return {"window": wid, "frames": frames, "convert": converted, "units": units}


def command_prepare(args: argparse.Namespace) -> int:
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    (publish / "units").mkdir(parents=True, exist_ok=True)
    stages = extract_stages(args.k6_stage, scratch)
    chosen = args.windows[:args.max_windows or None]
    rows = []
    for n, wid in enumerate(chosen):
        rows.append(prepare_window(args, wid, stages[wid], Path(args.sources[args.windows.index(wid)]), scratch, publish))
        progress(n + 1)
    manifest = {"command": "prepare", "windows": rows, "requested": chosen,
                "code": CODE, "gen_size": [GEN_W, GEN_H], "crop_margin": CROP_MARGIN, "centre_smooth": CENTRE_SMOOTH,
                "inputs": {"k6_stages": [file_sha256(Path(t)) for t in args.k6_stage],
                           "k6_inputs": file_sha256(Path(args.inputs) / "inputs.json")},
                "seconds": round(time.time() - began, 2)}
    write_json(publish / "prepare.json", manifest)
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "k6b.json", manifest)
    return 0


def prepare_fit_window(args: argparse.Namespace, wid: str, k6dir: Path, source: Path, scratch: Path, publish: Path) -> dict[str, Any]:
    """Step 3's training pairs (the overfit oracle sees the window's own frames): per player and fitted scored frame,
    the true crop and the DWPose body map of the *uncoded* keypoints, with step 2's crop rule on runs of fitted frames.
    Rendering at test still uses step 2's coded maps and granted reference."""
    w = k6_window(k6dir)
    row = w["row"]
    frames = int(row["frames"])
    truth, _ = window_truth(args, wid, source, frames, scratch / "work" / k4.safe(wid), "cpu")
    units = []
    for player in row["players"]:
        fitted = [t for t in row["scored"] if (rec := w["poses"][player][t]) is not None and rec.get("fitted")]
        if not fitted:
            continue
        ts, images, maps = [], [], []
        for seq in runs(fitted):
            kps = [np.asarray(w["poses"][player][t]["kps"], np.float64) for t in seq]
            for t, kp, box in zip(seq, kps, run_boxes(kps)):
                ts.append(t)
                images.append(crop_image(np.asarray(truth[t]), box))
                maps.append(pose_map(to_crop(kp, box)))
        name = unit_name(wid, player)
        np.savez_compressed(publish / "fit" / f"{name}.npz", t=np.asarray(ts, np.int64), image=np.stack(images), pose=np.stack(maps))
        units.append({"unit": name, "window": wid, "player": player, "frames": len(ts)})
    shutil.rmtree(scratch / "work" / k4.safe(wid), ignore_errors=True)
    return {"window": wid, "units": units}


def command_prepare_fit(args: argparse.Namespace) -> int:
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    (publish / "fit").mkdir(parents=True, exist_ok=True)
    stages = extract_stages(args.k6_stage, scratch)
    chosen = args.windows[:args.max_windows or None]
    rows = []
    for n, wid in enumerate(chosen):
        rows.append(prepare_fit_window(args, wid, stages[wid], Path(args.sources[args.windows.index(wid)]), scratch, publish))
        progress(n + 1)
    manifest = {"command": "prepare-fit", "windows": rows, "requested": chosen, "gen_size": [GEN_W, GEN_H],
                "inputs": {"k6_stages": [file_sha256(Path(t)) for t in args.k6_stage]}, "seconds": round(time.time() - began, 2)}
    write_json(publish / "prepare-fit.json", manifest)
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "k6b.json", manifest)
    return 0


def prepare_match_window(args: argparse.Namespace, wid: str, entry: dict[str, Any], source: Path, crops: Any, scratch: Path,
                         publish: Path, device: str) -> dict[str, Any]:
    """Step 4's training pairs from one of K4b.1's other windows (K6 never ran them): K6's adopted players, regions and
    uncoded poses (`window_players`, `frame_regions`, `pose_window`, unchanged), then step 3's crops and DWPose maps on
    the fitted scored (court-view, registered) frames."""
    from experiments.tennis.k6 import frame_regions, heights, pose_window, window_players
    from src.segmentation.masks import ClipMasks, decompress

    w = entry["window"]
    frames = int(w["frames"])
    root = Path(args.inputs) / "windows" / k4.safe(wid)
    for name, digest in entry["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Changed input: {root / name}")
    truth, _ = window_truth(args, wid, source, frames, scratch / "work" / k4.safe(wid), device)
    features = json.loads((root / "features.json").read_text())
    clip = ClipMasks.load(root / "sam-masks.rle")
    whole = json.loads(decompress((root / "pose.json.zst").read_bytes()))
    fits = {int(f): np.asarray(h, np.float64) for f, h in entry["fits"].items()}
    players = window_players(features, clip, int(w["start"]), fits)["players"]
    per_frame = [frame_regions(clip.frames[t], features["aliases"], players) for t in range(frames)]
    median_height = heights([f["bodies"] for f in per_frame], players)
    poses = pose_window(crops, truth, frames, [f["bodies"] for f in per_frame], whole, players, median_height)
    scored = [t for t in range(frames) if int(w["start"]) + t in fits and any(r.any() for r in per_frame[t]["regions"].values())]
    units = []
    for player in players:
        fitted = [t for t in scored if (rec := poses[player][t]) is not None and rec.get("fitted")]
        if not fitted:
            continue
        ts, images, maps = [], [], []
        for seq in runs(fitted):
            kps = [np.asarray(poses[player][t]["kps"], np.float64) for t in seq]  # type: ignore[index]
            for t, kp, box in zip(seq, kps, run_boxes(kps)):
                ts.append(t)
                images.append(crop_image(np.asarray(truth[t]), box))
                maps.append(pose_map(to_crop(kp, box)))
        name = unit_name(wid, player)
        np.savez_compressed(publish / "fit" / f"{name}.npz", t=np.asarray(ts, np.int64), image=np.stack(images), pose=np.stack(maps))
        units.append({"unit": name, "window": wid, "player": player, "frames": len(ts), "median_height": median_height[player]})
    shutil.rmtree(scratch / "work" / k4.safe(wid), ignore_errors=True)
    return {"window": wid, "units": units, "players": players, "scored_frames": len(scored)}


def command_prepare_match(args: argparse.Namespace) -> int:
    from experiments.tennis.k4b_pose import Crops

    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    (publish / "fit").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    inputs = json.loads((Path(args.inputs) / "inputs.json").read_text())
    crops = Crops(k4.Poses([("yolo11x-pose", args.pose)]))
    chosen = args.windows[:args.max_windows or None]
    rows = []
    for n, wid in enumerate(chosen):
        rows.append(prepare_match_window(args, wid, inputs["windows"][wid], Path(args.sources[args.windows.index(wid)]), crops,
                                         scratch, publish, "cuda"))
        progress(n + 1)
    manifest = {"command": "prepare-match", "windows": rows, "requested": chosen, "gen_size": [GEN_W, GEN_H],
                "inputs": {"k6_inputs": file_sha256(Path(args.inputs) / "inputs.json"), "pose_sha256": file_sha256(Path(args.pose))},
                "seconds": round(time.time() - began, 2)}
    write_json(publish / "prepare-fit.json", manifest)  # the name k6b_fit reads
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "k6b.json", manifest)
    return 0


def validate_prepare_fit(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    checks = {"windows_requested": [w["window"] for w in result["windows"]] == result["requested"]}
    with tempfile_dir() as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        for w in result["windows"]:
            for u in w["units"]:
                a = np.load(temp / "publish" / "fit" / f"{u['unit']}.npz")
                name = u["unit"]
                checks[f"{name}:shapes"] = a["image"].shape[1:] == (GEN_H, GEN_W, 3) == a["pose"].shape[1:] and len(a["t"]) == u["frames"] > 0
                checks[f"{name}:images_not_flat"] = float(a["image"][:: max(1, len(a["t"]) // 8)].std()) > 5.0
                drawn = a["pose"].max(-1) > 0
                checks[f"{name}:poses_drawn"] = bool(drawn.reshape(len(drawn), -1).any(1).mean() > 0.95)
    return checks


# ----------------------------------------------------------------- judge (GPU)


def k6_oracle(t: int, w: dict[str, Any], k6names: list[str], truth: Any, regions: list[dict[str, np.ndarray]],
              fill: np.ndarray, order: list[str], scored: list[int], region_height: dict[tuple[str, int], float],
              decoded_all: dict[str, dict[tuple[str, int], np.ndarray]], gap: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """K6's oracle composites and drawn masks at ``t`` (`k6.process_window`'s loop, unchanged)."""
    from experiments.tennis.k6 import choose_reference, draw

    poses = w["poses"]
    truth_t = np.asarray(truth[t])
    composites, drawn = {}, {}
    for v in k6names:
        kind, code_key = v.split("@")
        canvas = fill.copy()
        mask_v = np.zeros(fill.shape[:2], bool)
        for p in order:
            pose = poses[p][t]
            if p not in regions[t] or pose is None or not pose.get("fitted"):
                continue
            uncoded = np.asarray(pose["kps"], np.float64)
            target = uncoded if code_key == "uncoded" else decoded_all[code_key].get((p, t))
            if target is None:
                continue
            sigma = SIGMA_OF_HEIGHT * region_height[(p, t)]
            if kind == "warp":
                draw(canvas, mask_v, truth_t, regions[t][p], uncoded, target, sigma)
            else:
                r = choose_reference(t, p, target, poses, scored, region_height, gap)
                if r is None:
                    continue
                draw(canvas, mask_v, np.asarray(truth[r]), regions[r][p], np.asarray(poses[p][r]["kps"], np.float64), target, sigma)
        composites[v] = canvas
        drawn[v] = mask_v
    return composites, drawn


def load_generated(tars: list[str], scratch: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """(unit, profile) -> {t, crops} from k6b_generate's published.tar files."""
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for n, tar in enumerate(tars):
        root = scratch / "generated" / f"g{n}"
        if not root.is_dir():
            root.mkdir(parents=True)
            with tarfile.open(tar) as archive:
                archive.extractall(root, filter="data")
        for path in sorted((root / "publish").glob("*@*.npz")):
            unit, profile = path.stem.split("@")
            if (unit, profile) in out:
                raise ValueError(f"{unit}@{profile} appears twice")
            meta = json.loads(path.with_suffix(".json").read_text())
            if file_sha256(path) != meta["npz_sha256"]:
                raise ValueError(f"changed generator output {path}")
            a = np.load(path)
            out[(unit, profile)] = {"t": a["t"], "crops": a["crops"]}
    return out


def judge_window(args: argparse.Namespace, wid: str, k6dir: Path, source: Path, units: dict[str, dict[str, Any]],
                 generated: dict[tuple[str, str], dict[str, np.ndarray]], nets: dict[str, Any], scratch: Path,
                 publish: Path, device: str) -> dict[str, Any]:
    from fractions import Fraction

    from experiments.tennis.k6 import DISTS_PAD, REF_GAP_S, frame_regions, heights, save_review, telea_fill
    from experiments.tennis.k6 import variant_names as k6_variant_names
    from experiments.visor.h3b import dists_crops, lpips_reduce
    from src.segmentation.masks import ClipMasks

    began = time.time()
    w = k6_window(k6dir)
    row = w["row"]
    frames = int(row["frames"])  # always the whole window: K6's `ref` searches every scored frame
    judged = [t for t in row["judged"] if t >= args.from_t][:args.max_judged or None]  # step 3's pilot: held-out frames
    profiles = tuple(args.profiles.split(","))
    gen_variants = tuple(f"gen-{p}" for p in profiles)
    variants = gen_variants + CHECK_VARIANTS
    work = scratch / "work" / k4.safe(wid)
    truth, _ = window_truth(args, wid, source, frames, work, device)
    features = json.loads((k6dir / "features.json").read_text())
    clip = ClipMasks.load(k6dir / "sam-masks.rle")
    players = row["players"]
    per_frame = [frame_regions(clip.frames[t], features["aliases"], players) for t in range(frames)]
    regions = [f["regions"] for f in per_frame]
    median_height = heights([f["bodies"] for f in per_frame], players)
    order = sorted(players, key=lambda p: median_height[p])  # far first, as K6
    region_height = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.
                     for t, r in enumerate(regions) for p in players}
    from src.segmentation.masks import decompress

    decoded_all: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    for name, d in json.loads(decompress((k6dir / "keypoints.json.zst").read_bytes()))["decoded"].items():
        decoded_all[name] = {(k.rsplit("@", 1)[0], int(k.rsplit("@", 1)[1])): np.asarray(v, np.float64) for k, v in d.items()}
    k6names = k6_variant_names()[len(k6_variant_names()) - 9:]  # K6's 9 oracle variants (S is their union with R)
    gap = max(1, round(REF_GAP_S * float(Fraction(row["fps"]).limit_denominator(1001))))
    # Generated crops by (profile, player, t), with the unit's box at t.
    crops: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for player in players:
        name = unit_name(wid, player)
        if name not in units:
            continue
        boxes = dict(zip(units[name]["t"].tolist(), units[name]["box"]))
        for profile in profiles:
            g = generated.get((name, profile))
            if g is None:
                if profile != "finetuned_tennis":
                    raise ValueError(f"no generated crops for {name}@{profile}")
                continue  # amendment 1: the fine-tune falls back to gen-original's crops (below)
            for t, crop in zip(g["t"].tolist(), g["crops"]):
                crops[(profile, player, t)] = (crop, boxes[t])
    k6_scores = np.load(k6dir / "scores.npz")
    k6_index = {n: i for i, n in enumerate(row["variants"])}
    n_v = len(variants)
    lp = np.full((n_v, frames), np.nan)
    lp_drawn = np.full((n_v, frames), np.nan)
    dists = np.full((n_v, frames), np.nan)
    sse = np.zeros((n_v, frames))
    covered = np.zeros((n_v, frames), np.int64)
    pixels_s = np.zeros(frames, np.int64)
    pixels_r = np.zeros(frames, np.int64)
    review_at = judged[len(judged) // 2] if judged else None
    for n, t in enumerate(judged):
        truth_t = np.asarray(truth[t])
        r_t = np.logical_or.reduce([*regions[t].values(), np.zeros((FRAME[1], FRAME[0]), bool)])
        fill = telea_fill(truth_t, r_t)
        composites, drawn = k6_oracle(t, w, k6names, truth, regions, fill, order, row["scored"], region_height, decoded_all, gap)
        s_t = r_t.copy()
        for m in drawn.values():
            s_t |= m
        outs: dict[str, np.ndarray] = {}
        mine: dict[str, np.ndarray] = {}
        for profile, v in zip(profiles, gen_variants):
            canvas = fill.copy()
            written = np.zeros_like(r_t)
            for p in order:
                key = (profile, p, t) if (profile, p, t) in crops else ("original", p, t)
                if p in regions[t] and key in crops:
                    crop, box = crops[key]
                    written |= paste(canvas, crop, box, regions[t][p])
            outs[v], mine[v] = canvas, written
        for v in CHECK_VARIANTS:
            outs[v], mine[v] = composites[v], drawn[v]
        masks = [[s_t, mine[v]] for v in variants]
        reduced = lpips_reduce(nets["lpips"], [outs[v] for v in variants], truth_t, masks, device)
        ys, xs = np.nonzero(s_t)
        s_box = (slice(max(0, ys.min() - DISTS_PAD), min(FRAME[1], ys.max() + 1 + DISTS_PAD)),
                 slice(max(0, xs.min() - DISTS_PAD), min(FRAME[0], xs.max() + 1 + DISTS_PAD)))
        dists[:, t] = dists_crops(nets["dists"], [outs[v][s_box] for v in variants], truth_t[s_box], s_t[s_box], device)
        for i, v in enumerate(variants):
            lp[i, t], lp_drawn[i, t] = reduced[i]
            sse[i, t] = float(((outs[v][s_t].astype(np.int32) - truth_t[s_t].astype(np.int32)) ** 2).sum())
            covered[i, t] = int((mine[v] & r_t).sum())
        pixels_s[t], pixels_r[t] = int(s_t.sum()), int(r_t.sum())
        if t == review_at:
            save_review(publish / k4.safe(wid) / f"review-t{t:03d}.png", truth_t, {v: outs[v] for v in variants}, regions[t], f"{wid} t={t}")
        print(f"phase=judge window={wid} frame={n + 1}/{len(judged)}", flush=True)
    out_dir = publish / k4.safe(wid)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scores.npz", lpips=lp, lpips_drawn=lp_drawn, dists=dists, sse=sse, covered=covered,
                        pixels_s=pixels_s, pixels_r=pixels_r, judged=np.asarray(judged, np.int64))
    k6_lp = {v: float(np.mean(k6_scores["lpips"][k6_index[v], judged])) for v in CHECK_VARIANTS} if judged else {}
    result = {
        "id": wid, "match": wid.split("/")[0], "frames": frames, "judged": judged, "players": players, "variants": list(variants),
        "k6_job_files": row["files"], "k6_lpips": k6_lp,
        "s_pixels_match_k6": bool(np.array_equal(pixels_s[judged], k6_scores["pixels_s"][judged])),
        "seconds": round(time.time() - began, 2),
    }
    result["files"] = {p.name: file_sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    write_json(out_dir / "window.json", result)
    shutil.rmtree(work, ignore_errors=True)
    return result


def command_judge(args: argparse.Namespace) -> int:
    import torch
    import torchmetrics

    from experiments.audit.env_smoke import device_record
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record

    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "24000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    stages = extract_stages(args.k6_stage, scratch)
    prepared_root = scratch / "prepared"
    if not prepared_root.is_dir():
        prepared_root.mkdir(parents=True)
        with tarfile.open(args.prepared) as archive:
            archive.extractall(prepared_root, filter="data")
    units = {p.stem: dict(np.load(p)) for p in sorted((prepared_root / "publish" / "units").glob("*.npz"))}
    generated = load_generated(args.generated, scratch)
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    chosen = args.windows[:args.max_windows or None]
    rows: list[dict[str, Any]] = []

    def checkpoint() -> None:
        with tarfile.open(stage_dir() / "published.tar", "w") as tar:
            tar.add(publish, arcname="publish")
        write_json(stage_dir() / "k6b.json", {
            "command": "judge", "stage": os.environ.get("PS_STAGE"), "requested": chosen, "max_judged": int(args.max_judged), "from_t": int(args.from_t), "profiles": args.profiles.split(","),
            "windows": rows, "decision": DECISION, "device": device_record(args.gpu_class),
            "on_cuda": {k: next(v.parameters()).is_cuda for k, v in nets.items()},
            "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "gpu_memory_limit_mib": limit,
            "lpips": lpips_record(Path(args.lpips_backbone)),
            "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
            "inputs": {"k6_stages": [file_sha256(Path(t)) for t in args.k6_stage], "prepared": file_sha256(Path(args.prepared)),
                       "generated": [file_sha256(Path(t)) for t in args.generated]},
            "seconds": round(time.time() - began, 2)})

    for n, wid in enumerate(chosen):
        rows.append(judge_window(args, wid, stages[wid], Path(args.sources[args.windows.index(wid)]), units, generated, nets,
                                 scratch, publish, device))
        checkpoint()
        progress(n + 1)
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate_prepare(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    with tempfile_dir() as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        units = {p.stem: np.load(p) for p in (temp / "publish" / "units").glob("*.npz")}
        checks = {"windows_requested": [w["window"] for w in result["windows"]] == result["requested"]}
        listed = [u for w in result["windows"] for u in w["units"]]
        checks["units_published"] = sorted(u["unit"] for u in listed) == sorted(units) and bool(units)
        for u in listed:
            a = units[u["unit"]]
            name = u["unit"]
            checks[f"{name}:shapes"] = (a["reference"].shape == (GEN_H, GEN_W, 3) and a["pose"].shape[1:] == (GEN_H, GEN_W, 3)
                                        and len(a["t"]) == len(a["box"]) == len(a["pose"]) == u["frames"])
            checks[f"{name}:reference_not_flat"] = float(a["reference"].std()) > 5.0
            # every pose map draws something, and the drawn skeleton sits inside the crop's middle (keypoints centred)
            drawn = (a["pose"].max(-1) > 0)
            checks[f"{name}:poses_drawn"] = bool(drawn.reshape(len(drawn), -1).any(1).mean() > 0.95)
            ys, xs = np.nonzero(drawn.any(0))
            checks[f"{name}:poses_centred"] = bool(len(xs) and abs(xs.mean() - GEN_W / 2) < GEN_W / 4 and abs(ys.mean() - GEN_H / 2) < GEN_H / 4)
            checks[f"{name}:judged_frames"] = int(a["judged"].sum()) == u["judged"] > 0
    return checks


def validate_judge(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    checks = {"windows_requested": [w["id"] for w in result["windows"]] == result["requested"],
              "on_cuda": all(result["on_cuda"].values()), "device_class": bool(result["device"]["class_matches"]),
              "decision_frozen": result["decision"] == DECISION,
              "peak_within_declaration": result["gpu_peak_gib"] * 1024 <= result["gpu_memory_limit_mib"]}
    with tempfile_dir() as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        for row in result["windows"]:
            root = temp / "publish" / k4.safe(row["id"])
            wid = row["id"]
            checks[f"{wid}:files"] = all(file_sha256(root / n) == h for n, h in row["files"].items())
            a = np.load(root / "scores.npz")
            j = row["judged"]
            checks[f"{wid}:finite"] = bool(np.isfinite(a["lpips"][:, j]).all() and np.isfinite(a["dists"][:, j]).all())
            checks[f"{wid}:s_pixels_match_k6"] = bool(row["s_pixels_match_k6"])
            variants = row.get("variants", list(VARIANTS))
            for v in CHECK_VARIANTS:  # K6's mean over the same judged frames
                mine = float(np.mean(a["lpips"][variants.index(v), j]))
                checks[f"{wid}:{v}_reproduces_k6"] = bool(abs(mine - row["k6_lpips"][v]) <= REPRO_TOLERANCE)
            for v in [x for x in variants if x.startswith("gen-") and x != "gen-finetuned_tennis"]:  # the scored renderers
                i = variants.index(v)
                share = a["covered"][i, j].sum() / max(1, a["pixels_r"][j].sum())
                checks[f"{wid}:{v}:draws"] = bool(share > 0.5)
                checks[f"{wid}:{v}:differs_from_warp"] = bool(np.mean(a["lpips"][i, j]) != np.mean(a["lpips"][variants.index(CHECK_VARIANTS[0]), j]))
            checks[f"{wid}:review"] = len(list(root.glob("review-t*.png"))) == 1
    return checks


class tempfile_dir:
    """A temporary directory under /tmp (validators run beside the stage, never on the NFS home)."""

    def __enter__(self) -> Path:
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="k6b-validate-", dir="/tmp")
        return Path(self._dir.name)

    def __exit__(self, *exc: Any) -> None:
        self._dir.cleanup()


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k6b.json").read_text())
    checks = {"prepare": validate_prepare, "prepare-fit": validate_prepare_fit, "prepare-match": validate_prepare_fit,
              "judge": validate_judge}[result["command"]](stage, result)
    report = {"passed": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()}}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def window_means(a: dict[str, np.ndarray], judged: list[int], variants: tuple[str, ...] | list[str] = VARIANTS) -> dict[str, dict[str, float]]:
    out = {}
    for i, v in enumerate(variants):
        sse = float(a["sse"][i, judged].sum())
        pixels = float(a["pixels_s"][judged].sum()) * 3
        out[v] = {"lpips": float(np.mean(a["lpips"][i, judged])), "dists": float(np.mean(a["dists"][i, judged])),
                  "psnr": 10 * math.log10(255 ** 2 / (sse / pixels)) if sse > 0 else math.inf,
                  "covered_share": float(a["covered"][i, judged].sum() / max(1, a["pixels_r"][judged].sum()))}
    return out


def won(anchors: dict[str, dict[str, float]], metric: str, kbps: float, value: float) -> dict[str, Any]:
    """Is ``value`` below K6's envelope at ``kbps``? (`k1.at_rate` on `g5.envelope` of (kbps, -metric).)"""
    from experiments.background.g5 import envelope
    from experiments.visor.k1 import at_rate

    hull = envelope([(p["kbps"], -p[metric]) for p in anchors.values()])
    q = at_rate(hull, kbps)
    bar = -(q if q is not None else hull[0][1])
    return {"value": value, "envelope": bar, "won": bool(value < bar)}


def score(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        return {"windows": 0}
    lp = float(np.mean([r[variant]["lpips"]["won"] for r in rows]))
    di = float(np.mean([r[variant]["dists"]["won"] for r in rows]))
    return {"windows": len(rows), "lpips_share": lp, "dists_share": di, "score": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION}


def decide(groups: dict[str, dict[str, Any]], reproduced: bool, step: int = 2) -> str:
    if not reproduced:
        return "not read: warp/ref do not reproduce K6"
    if step == 3:  # experiments.md "K6b (players) 3": the overfit oracle
        if groups["all"]["gen-fit"]["passes"]:
            return "pass: step 4 (a fit on other shots of the same match, billed)"
        return "fail: the generative route closes for players at the keypoint rate (the matte granted)"
    if groups["all"]["gen-original"]["passes"]:
        return "pass: the zero-shot route holds with the reference granted; next, the reference billed"
    return "fail: step 3 (overfit oracle) gets its rule and budget"


def billed_kbps(nbytes: float, hours: float) -> float:
    """K6b (players) 4: a model's bytes spread over ``hours`` of the match, in kbps."""
    return nbytes * 8 / (hours * 3600) / 1000


def payback_hours(hull: list[tuple[float, float]], lpips: float, kbps: float, nbytes: float) -> float:
    """Hours of the match after which the model's bits are repaid by the rate it saves at equal LPIPS (G5b): its
    bits over (envelope kbps at that LPIPS - keypoint kbps); infinite when it saves nothing."""
    from experiments.tennis.k6_report import rate_on_envelope

    needed, _ = rate_on_envelope(hull, -lpips)
    if needed is None or needed <= kbps:
        return math.inf
    return nbytes * 8 / ((needed - kbps) * 1000) / 3600


def billed_score(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        return {"windows": 0}
    lp = float(np.mean([r[variant]["billed"]["lpips"]["won"] for r in rows]))
    di = float(np.mean([r[variant]["billed"]["dists"]["won"] for r in rows]))
    return {"windows": len(rows), "lpips_share": lp, "dists_share": di, "score": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION}


def decide_billed(billed: dict[str, Any], unbilled: dict[str, Any], reproduced: bool) -> str:
    """experiments.md "K6b (players) 4"."""
    if not reproduced:
        return "not read: warp/ref do not reproduce K6"
    if billed["passes"]:
        return "pass: per-match player models pay for themselves; next, the reference billed too (K6c's route)"
    if unbilled["passes"]:
        return "to the user: passes unbilled only (a smaller model, a longer amortization, or another H)"
    return "fail: a match fit does not reach the envelope even unbilled; step 3's pass was memorization"


def command_report(args: argparse.Namespace) -> int:
    from experiments.tennis.k6_report import bootstrap, saving

    k6 = json.loads(Path(args.k6_report).read_text())
    billing = json.loads(Path(args.billing).read_text()) if args.billing else {}  # step 4: {variant: {match: bytes}}
    rows: list[dict[str, Any]] = []
    sources = []
    reproduced = True
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k6b.json").read_text())
        checks = validate_judge(stage, result)
        if not all(checks.values()) or result["stage"] != "full" or result["max_judged"] != 0:
            raise SystemExit(f"{stage} fails validation or is not a full stage: {[k for k, v in checks.items() if not v]}")
        sources.append({"stage": str(stage), "sha256": file_sha256(stage / "published.tar")})
        with tempfile_dir() as temp:
            with tarfile.open(stage / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            for w in result["windows"]:
                a = dict(np.load(temp / "publish" / k4.safe(w["id"]) / "scores.npz"))
                means = window_means(a, w["judged"], w["variants"])
                kw = k6["windows"][w["id"]]
                kbps = kw["keypoint_kbps"][CODE]
                row: dict[str, Any] = {"window": w["id"], "match": w["match"], "kbps": kbps,
                                       "finetune_unlisted": w["match"] in FINETUNE_UNLISTED}
                for v in w["variants"]:
                    row[v] = {**means[v], **{m: won(kw["anchors"], m, kbps, means[v][m]) for m in ("lpips", "dists")}}
                    hull = [tuple(p) for p in kw["savings"]["lpips"]["hull"]["points"]]
                    row[v]["saving"] = saving(hull, -means[v]["lpips"], kbps)["saving"]
                    if v in billing:
                        nbytes = billing[v][w["match"]]
                        rate = kbps + billed_kbps(nbytes, args.hours)
                        row[v]["billed"] = {"kbps": rate, "bytes": nbytes, "hours": args.hours,
                                            **{m: won(kw["anchors"], m, rate, means[v][m]) for m in ("lpips", "dists")},
                                            "saving": saving(hull, -means[v]["lpips"], rate)["saving"],
                                            "payback_hours": payback_hours(hull, means[v]["lpips"], kbps, nbytes)}
                for v in CHECK_VARIANTS:
                    row[v]["k6_lpips"] = kw["oracle"][v]["lpips"]
                    row[v]["reproduces"] = abs(means[v]["lpips"] - kw["oracle"][v]["lpips"]) <= REPRO_TOLERANCE
                    reproduced &= row[v]["reproduces"]
                rows.append(row)
    groups = {"all": rows, "finetune_unlisted": [r for r in rows if r["finetune_unlisted"]]}
    main = {3: "gen-fit", 4: "gen-match"}.get(args.step, GEN_VARIANTS[0])  # step 2, amendment 1: the fine-tune is a diagnostic
    decided = (main, *CHECK_VARIANTS)
    scored = {g: {v: score(rs, v) for v in decided} for g, rs in groups.items()}
    if args.step == 4:
        for g, rs in groups.items():
            scored[g][f"{main} billed"] = billed_score(rs, main)
    finetune = [{"window": r["window"], "gen-original": {m: r["gen-original"][m]["value"] for m in ("lpips", "dists")},
                 "gen-finetuned_tennis": {m: r["gen-finetuned_tennis"][m]["value"] for m in ("lpips", "dists")}}
                for r in rows if "gen-finetuned_tennis" in r and r["gen-finetuned_tennis"]["lpips"]["value"] != r["gen-original"]["lpips"]["value"]]
    verdict = (decide_billed(scored["all"][f"{main} billed"], scored["all"][main], reproduced) if args.step == 4
               else decide(scored, reproduced, args.step))
    out = {"decision_rule": DECISION, "step": args.step, "decision": verdict, "reproduced": reproduced, "rows": rows,
           "billing": {"hours": args.hours, "bytes": billing} if billing else None,
           "groups": scored, "saving_ci": {v: bootstrap([r[v]["saving"] for r in rows]) for v in decided},
           "finetune_diagnostic": finetune,
           "inputs": {"stages": sources, "k6_report": file_sha256(Path(args.k6_report)), "code_sha256": file_sha256(Path(__file__))},
           "overlap": "gen-original: off-the-shelf weights (not trained on PointStream's footage); gen-finetuned_tennis: "
                      "counted as having seen every window's match (K6b audit); 'finetune_unlisted' are the windows whose "
                      "match is absent from the fine-tune's surviving track list"}
    write_json(Path(args.out), out)
    for r in rows:
        print(r["window"], f"{r['kbps']:.2f} kbps", " ".join(
            f"{v}: L {r[v]['lpips']['value']:.3f}/{r[v]['lpips']['envelope']:.3f} D {r[v]['dists']['value']:.3f}/{r[v]['dists']['envelope']:.3f}"
            for v in decided))
    print(json.dumps({"decision": out["decision"], "groups": scored}, indent=1))
    return 0


def command_fit_report(args: argparse.Namespace) -> int:
    """Step 3's learning curve: each fitted checkpoint (and step 2's zero-shot) on the held-out judged frames, against
    K6's envelope rebuilt from its anchors' per-frame scores averaged over the same frames (K6's anchor rates)."""
    from experiments.tennis.k6 import ANCHORS, anchor_name

    k6 = json.loads(Path(args.k6_report).read_text())
    billing = json.loads(Path(args.billing).read_text()) if args.billing else {}  # step 4's pilot: {variant: billed bytes}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or "/tmp/k6b-fit-report")
    stages = extract_stages(args.k6_stage, scratch)
    out: dict[str, Any] = {"windows": {}, "inputs": {"stages": [], "k6_report": file_sha256(Path(args.k6_report)),
                                                    "code_sha256": file_sha256(Path(__file__))}}
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k6b.json").read_text())
        checks = validate_judge(stage, result)
        if not all(checks.values()) or result["stage"] != "full":
            raise SystemExit(f"{stage} fails validation or is not a full stage: {[k for k, v in checks.items() if not v]}")
        out["inputs"]["stages"].append({"stage": str(stage), "sha256": file_sha256(stage / "published.tar")})
        with tempfile_dir() as temp:
            with tarfile.open(stage / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            for w in result["windows"]:
                a = dict(np.load(temp / "publish" / k4.safe(w["id"]) / "scores.npz"))
                j = w["judged"]
                means = window_means(a, j, w["variants"])
                k6row = json.loads((stages[w["id"]] / "window.json").read_text())
                k6s = np.load(stages[w["id"]] / "scores.npz")
                kw = k6["windows"][w["id"]]
                kbps = kw["keypoint_kbps"][CODE]
                anchors = {}
                for anchor in map(anchor_name, ANCHORS):
                    if anchor in kw["anchors"]:
                        i = k6row["variants"].index(anchor)
                        anchors[anchor] = {"kbps": kw["anchors"][anchor]["kbps"], "lpips": float(np.mean(k6s["lpips"][i, j])),
                                           "dists": float(np.mean(k6s["dists"][i, j]))}
                rate = {v: kbps + (billed_kbps(billing[v], args.hours) if v in billing else 0.0) for v in w["variants"]}
                curve = {v: {m: won(anchors, m, rate[v], means[v][m]) for m in ("lpips", "dists")} for v in w["variants"]}
                for v in w["variants"]:
                    curve[v]["kbps"] = rate[v]
                    if v in billing:
                        curve[v]["unbilled"] = {m: won(anchors, m, kbps, means[v][m]) for m in ("lpips", "dists")}
                reached = [v for v in w["variants"] if v.startswith("gen-") and v != "gen-original"
                           and all(curve[v][m]["won"] for m in ("lpips", "dists"))]
                out["windows"][w["id"]] = {"held_out_frames": j, "kbps": kbps, "anchors_held_out": anchors, "curve": curve,
                                           "reached": reached}
                for v in w["variants"]:
                    print(w["id"], v, " ".join(f"{m} {curve[v][m]['value']:.3f}/{curve[v][m]['envelope']:.3f}" for m in ("lpips", "dists")))
    write_json(Path(args.out), out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("prepare", "prepare-fit", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", required=True, help="K6's extracted inputs.tar")
        p.add_argument("--k6-stage", action="append", required=True, help="K6's published.tar (pilot and parts)")
        p.add_argument("--windows", nargs="+", required=True)
        p.add_argument("--sources", nargs="+", required=True)
        p.add_argument("--max-windows", type=int, required=True, help="0 runs every listed window")
        p.add_argument("--ffmpeg", default="ffmpeg")
        if name == "judge":
            p.add_argument("--prepared", required=True)
            p.add_argument("--generated", action="append", required=True)
            p.add_argument("--lpips-backbone", required=True)
            p.add_argument("--dists-backbone", required=True)
            p.add_argument("--max-judged", type=int, default=0, help="0 judges every K6 judged frame (the smoke: a few)")
            p.add_argument("--from-t", type=int, default=0, help="judge only frames from here (step 3's held-out frames)")
            p.add_argument("--profiles", default=",".join(PROFILES), help="the generator profiles to composite and score")
            p.add_argument("--gpu-class", default="RTX")
    pm = sub.add_parser("prepare-match")
    pm.add_argument("--inputs", required=True, help="K6's extracted inputs.tar (K4b.1's windows' features, masks, fits)")
    pm.add_argument("--pose", required=True, help="YOLO11x-pose weights (K6's)")
    pm.add_argument("--windows", nargs="+", required=True)
    pm.add_argument("--sources", nargs="+", required=True)
    pm.add_argument("--max-windows", type=int, required=True)
    pm.add_argument("--ffmpeg", default="ffmpeg")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    fit = sub.add_parser("fit-report")
    fit.add_argument("--stage", action="append", required=True, help="validated judge stages of the pilot")
    fit.add_argument("--k6-stage", action="append", required=True, help="K6's published.tar holding the window")
    fit.add_argument("--k6-report", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--billing", default="", help="step 4's pilot: JSON {variant: billed bytes}")
    fit.add_argument("--hours", type=float, default=2.0)
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--k6-report", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--step", type=int, default=2, help="2: gen-original decides; 3: gen-fit (overfit oracle); 4: gen-match, billed")
    rep.add_argument("--billing", default="", help="step 4: JSON {variant: {match: billed bytes}}")
    rep.add_argument("--hours", type=float, default=2.0, help="step 4: H, the hours of the match the bytes are spread over")
    args = ap.parse_args(argv)
    if args.command in ("prepare", "prepare-fit", "prepare-match", "judge") and len(args.windows) != len(args.sources):
        raise SystemExit("Give one source per window")
    return {"prepare": command_prepare, "prepare-fit": command_prepare_fit, "prepare-match": command_prepare_match,
            "judge": command_judge, "validate": command_validate,
            "report": command_report, "fit-report": command_fit_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
