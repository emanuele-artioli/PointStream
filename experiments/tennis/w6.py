"""W6: are PointStream's players stable over time and in the right pose, at the full frame rate?

Fleet stages (docs/experiments/w6-motion-pose.md; the rule is frozen there):

    masks   receiver masks (k6b_receiver_sam's every-frame fresh-prompt predictor, run as its isolated worker) on the
            regenerated crops of every frame, after comparing the regenerated crops with the saved ones
    judge   J's pose and time judges (``j_players.Judge``, ``visor.j``) on all 240 frames of the hybrid composite on the
            paid scale-4 base and of plain SVT-AV1 at the hybrid's rate, the racket measure, and the visual clips
    validate-generate | validate-masks | validate-judge   the stages' fleet validators
    report  (CPU, local) the decision from the judge's arrays

Generation itself is ``k6b_generate generate --keep-all``. The pure functions (interpolation, checks, bootstrap, racket
and reproducibility statistics) are at the top so that tests cover them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.w1_roi import (FRAMES, PACKET_DIR, SOURCE_SHA, SOURCE_START, SOURCE_YUV_SHA, TARGET_KBPS, WINDOW,
                                       digest)

#: the second judge's thresholds, J's (docs/experiments/j.md): pose floor on the coded keypoints and the energy factor.
POSE_FLOOR = 1.5
ENERGY_FACTOR = 1.25
#: the racket region: source-region pixels farther than this share of the player's height from the keypoint hull.
RACKET_MARGIN = 0.10
RACKET_MIN_PIXELS = 100
FPS = 25
BLOCK = 24  # bootstrap block length in frames (one generator context)
ANCHOR_POLICY = "s2_o0"  # W1's plain half-resolution SVT-AV1 (offset 0)
ANCHOR_CRFS = (65, 66)  # the bracketing pair of the hybrid's rate (W1's selected 65-67 at offset 0)
VARIANTS = ("source", "hybrid")


# ---------------------------------------------------------------- pure statistics


def log_weight(target: float, low: float, high: float) -> float:
    """Weight of the ``high``-rate point at ``target``, linear in log rate (never extrapolated)."""
    if not (min(low, high) <= target <= max(low, high)) or low == high:
        raise ValueError("target rate outside the bracket")
    return (math.log(target) - math.log(low)) / (math.log(high) - math.log(low))


def at_rate(target: float, low: tuple[float, Any], high: tuple[float, Any]) -> Any:
    """Per-frame arrays (or numbers) of two points, combined linearly in log rate to ``target``."""
    w = log_weight(target, low[0], high[0])
    return (1 - w) * np.asarray(low[1], np.float64) + w * np.asarray(high[1], np.float64)


def mean_of(values: Any) -> float:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def block_bootstrap(values: Any, block: int = BLOCK, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of a per-frame series (NaN dropped), resampling blocks of consecutive frames."""
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = max(1, math.ceil(a.size / block))
    starts = rng.integers(0, max(1, a.size - block + 1), size=(n, blocks))
    means = np.array([np.concatenate([a[s:s + block] for s in row])[:a.size].mean() for row in starts])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pose_check(method: float, anchor: float, coded: float) -> dict[str, Any]:
    """J's pose rule: the method's error <= max(the anchor's, 1.5 x the coded keypoints' own error)."""
    limit = max(anchor, POSE_FLOOR * coded)
    return {"method": method, "anchor": anchor, "coded_keypoints": coded, "limit": limit, "pass": bool(method <= limit)}


def time_check(flicker: float, anchor_flicker: float, energy: float, source_energy: float) -> dict[str, Any]:
    """J's time rule (the OR reading): flicker <= the anchor's, or residual energy <= 1.25 x the source's."""
    by_flicker = bool(flicker <= anchor_flicker)
    by_energy = bool(energy <= ENERGY_FACTOR * source_energy)
    return {"flicker": flicker, "anchor_flicker": anchor_flicker, "energy": energy, "source_energy": source_energy,
            "energy_limit": ENERGY_FACTOR * source_energy, "pass_flicker": by_flicker, "pass_energy": by_energy,
            "pass": by_flicker or by_energy}


def racket_region(region: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Source-region pixels outside the (dilated) keypoint hull: the racket and anything else beyond the body."""
    return region & ~hull


def racket_stats(racket: np.ndarray, predicted: np.ndarray, written: np.ndarray) -> tuple[int, float, float] | None:
    """(racket pixels, share covered by the predicted mask, share covered by the written pixels); None if too few."""
    n = int(racket.sum())
    if n < RACKET_MIN_PIXELS:
        return None
    return n, float((racket & predicted).sum() / n), float((racket & written).sum() / n)


def racket_summary(pixels: np.ndarray, covered: np.ndarray) -> dict[str, Any]:
    """Shares of the (player, frame) cells with a racket region in which the generated player overlaps it."""
    present = np.isfinite(covered)
    n = int(present.sum())
    c = covered[present]
    return {"cells_with_racket": n, "cells": int(covered.size), "overlap_any": float((c > 0).mean()) if n else float("nan"),
            "overlap_half": float((c >= 0.5).mean()) if n else float("nan"), "mean_coverage": float(c.mean()) if n else float("nan"),
            "median_racket_pixels": float(np.median(pixels[present])) if n else float("nan")}


def reproduction(old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    """Saved against regenerated crops at the saved ticks."""
    diff = np.abs(old.astype(np.int16) - new.astype(np.int16))
    return {"frames": int(len(old)), "max_abs_diff": int(diff.max()) if diff.size else 0, "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "share_pixels_differing": float((diff > 0).mean()) if diff.size else 0.0,
            "frames_identical": int((diff.reshape(len(old), -1).max(1) == 0).sum()) if len(old) else 0,
            "within_one_level": bool(diff.max() <= 1) if diff.size else True}


def decide(rows: dict[str, Any]) -> dict[str, Any]:
    """The frozen rule on the window-level means: each check separately."""
    pose = pose_check(rows["pose"]["method"], rows["pose"]["anchor"], rows["pose"]["coded"])
    time_ = time_check(rows["time"]["flicker"], rows["time"]["anchor_flicker"], rows["time"]["energy"], rows["time"]["source_energy"])
    return {"pose": pose, "time": time_, "pass": bool(pose["pass"] and time_["pass"])}


def tick_series(values: np.ndarray, ticks: list[int]) -> np.ndarray:
    return values[..., ticks]


# ---------------------------------------------------------------- shared helpers


def tool_version(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception as error:  # a record only
        return f"unavailable: {error}"


def extract_member(archive: Path, member: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        data = tar.extractfile(member)
        if data is None:
            raise ValueError(f"{member} missing in {archive.name}")
        with out.open("wb") as handle:
            shutil.copyfileobj(data, handle)
    return out


def extract_all(archive: Path, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return root


def unit_files(root: Path) -> dict[str, Path]:
    """unit -> its ``<unit>@match5.npz`` in an extracted k6b_generate archive."""
    return {p.stem.split("@")[0]: p for p in sorted((root / "publish").glob("*@*.npz"))}


# ---------------------------------------------------------------- masks stage


def command_masks(a: argparse.Namespace) -> int:
    from experiments.jobs.monitor import publish_progress

    began = time.monotonic()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    if digest(a.transport) != "6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7":
        raise ValueError("transport archive changed")
    kps_dir = scratch / "kps"
    extract_member(a.transport, PACKET_DIR + "keypoints.kps", kps_dir / "keypoints.kps")
    old_root = extract_all(a.old_generated, scratch / "old-generated")
    old_masks = extract_all(a.old_masks, scratch / "old-masks")
    old = {u: np.load(p) for u, p in unit_files(old_root).items()}
    records: list[dict[str, Any]] = []
    for n, archive in enumerate(a.generated):
        root = extract_all(archive, scratch / f"new-generated-{n}")
        for unit, path in unit_files(root).items():
            new = np.load(path)
            t_new = new["t"].tolist()
            if t_new != list(range(len(t_new))) or (os.environ.get("PS_STAGE") == "full" and len(t_new) != FRAMES):
                raise ValueError(f"{unit}: regenerated frames are not contiguous from 0 (all {FRAMES} in the full stage)")
            saved = old[unit]
            index = {t: i for i, t in enumerate(t_new)}
            kept = [k for k, t in enumerate(saved["t"].tolist()) if t in index]
            repro = reproduction(saved["crops"][kept], new["crops"][[index[int(saved["t"][k])] for k in kept]])
            # SAM on the regenerated crops, one fresh isolated worker per unit (k6b_receiver_sam, every frame fresh)
            out = scratch / "isolated" / unit
            if (publish / f"{unit}-masks.npz").exists():
                continue
            # The worker creates its own directory and refuses an existing one; clear an unfinished attempt.
            shutil.rmtree(out, ignore_errors=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "HOME": str(out), "TMPDIR": str(out / "tmp"), "YOLO_CONFIG_DIR": str(out / "config"),
                   "PYTHONNOUSERSITE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
            command = [sys.executable, "-m", "experiments.tennis.k6b_receiver_sam", "--worker", "--independent-frames", "--out", str(out),
                       "--kps", str(kps_dir), "--weights", str(a.weights), "--text", str(a.weights), "--max-judged", str(a.judged),
                       "--generated", str(archive)]
            for forbidden in (a.source, a.transport):
                command += ["--forbidden", str(forbidden)]
            log = scratch / f"worker-{unit}.log"
            started = time.monotonic()
            with log.open("w") as handle:
                done = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if done.returncode:
                print(log.read_text()[-6000:], flush=True)
                raise RuntimeError(f"SAM worker exited {done.returncode} for {unit}")
            prediction = json.loads((out / "prediction.json").read_text())
            row = next(r for r in prediction["rows"] if r["name"] == unit)
            shutil.copyfile(out / row["file"], publish / f"{unit}-masks.npz")
            predicted = np.load(publish / f"{unit}-masks.npz")["masks"]
            same: dict[str, Any] = {"unit": unit}
            with np.load(old_masks / "publish" / f"{unit}-masks.npz") as z:
                ticks = z["t"].tolist()
                keep = [k for k, t in enumerate(ticks) if t in index]
                got = predicted[[index[ticks[k]] for k in keep]]
                prior = z["masks"][keep]
                same.update({"saved_ticks": len(keep), "identical_ticks": int((got == prior).reshape(len(keep), -1).all(1).sum()),
                             "differing_pixels": int((got != prior).sum())})
            records.append({"unit": unit, "reproduction": repro, "mask_reproduction": same, "inference_seconds": row["inference_seconds"],
                            "area": row["area"], "worker_seconds": time.monotonic() - started, "prompts": row["prompts"][:2],
                            "flash_events": prediction["flash_events"], "device": prediction["device"], "on_cuda": prediction["on_cuda"],
                            "weights_sha256": prediction["weights_sha256"], "sam_runtime": prediction["sam_runtime"]})
            publish_progress(os.environ["PS_STAGE"], len(records))
    summary = {"units": records, "seconds": time.monotonic() - began, "ticks": FRAMES,
               "inputs": {k: digest(getattr(a, k)) for k in ("transport", "old_generated", "old_masks", "weights")}}
    (publish / "masks.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    with tarfile.open(Path(os.environ["PS_STAGE_DIR"]) / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    (Path(os.environ["PS_STAGE_DIR"]) / "w6-masks.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return 0


def validate_masks() -> dict[str, bool]:
    stage = Path(os.environ["PS_STAGE_DIR"])
    summary = json.loads((stage / "w6-masks.json").read_text())
    checks = {"two_units": len(summary["units"]) == 2}
    with tarfile.open(stage / "published.tar") as tar:
        for row in summary["units"]:
            name = f"publish/{row['unit']}-masks.npz"
            member = tar.extractfile(name)
            checks[f"{row['unit']}:file"] = member is not None
            if member is None:
                continue
            import io
            with np.load(io.BytesIO(member.read())) as z:
                masks, t = z["masks"], z["t"]
            checks[f"{row['unit']}:frames"] = len(masks) == len(t) > 0 and (os.environ.get("PS_STAGE") != "full" or len(t) == FRAMES)
            checks[f"{row['unit']}:nonempty"] = bool(masks.reshape(len(masks), -1).any(1).mean() > 0.9)
            checks[f"{row['unit']}:cuda_flash"] = bool(row["on_cuda"]) and bool(row["flash_events"])
    return checks


# ---------------------------------------------------------------- judge stage


def load_wire(a: argparse.Namespace, scratch: Path) -> dict[str, Any]:
    import cv2

    from experiments.tennis import k6b
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks

    stages = k6b.extract_stages([str(a.anchor)], scratch / "anchor")
    row = k6b.k6_window(stages[WINDOW])["row"]
    members = scratch / "wire"
    matte = extract_member(a.transport, PACKET_DIR + "exact-rle.matte", members / "exact-rle.matte")
    kps = extract_member(a.transport, PACKET_DIR + "keypoints.kps", members / "keypoints.kps")
    _, source_masks = decode_masks(matte.read_bytes())
    meta, units = decode_keypoints(kps.read_bytes())
    boxes = box_maps(units)
    players = row["players"]
    region = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
               for i, p in enumerate(players) if source_masks[t, i].any()} for t in range(FRAMES)]
    coded = [{p: np.asarray(units[i]["kp"][list(units[i]["t"]).index(t)], np.float64) for i, p in enumerate(players)
              if t in set(units[i]["t"].tolist())} for t in range(FRAMES)]
    return {"row": row, "players": players, "order": meta["compositing_order"], "units": units, "boxes": boxes, "region": region, "coded": coded}


def plane_rgb(raw: Any, t: int, width: int, height: int, torch: Any) -> np.ndarray:
    from src.codecs.quality import yuv420_to_rgb

    plane = torch.from_numpy(np.array(raw[t], copy=True)[None]).cuda()
    return yuv420_to_rgb(plane, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()


class Clip:
    """x264 crf 12 at 25 fps from raw RGB frames piped to ffmpeg."""

    def __init__(self, path: Path, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
                                      str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait()


def command_judge(a: argparse.Namespace) -> int:
    import cv2
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis import k4, k6, w1_score
    from experiments.tennis.j_players import Judge
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k4b_pose import Crops
    from experiments.tennis.k6b import load_generated
    from experiments.tennis.k6b_real_base_compose import compose
    from experiments.tennis.k6b_receiver_sam import project
    from experiments.visor.j import pose_error, residual_energies
    from src.codecs import svtav1

    began = time.monotonic()
    torch.set_num_threads(4)
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "12000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    if digest(a.al) != SOURCE_SHA:
        raise ValueError("source clip changed")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    frames = a.frames or FRAMES
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    wire = load_wire(a, scratch)
    players, order, boxes, region, coded = wire["players"], wire["order"], wire["boxes"], wire["region"], wire["coded"]

    # source frames, converted as the paid pipeline did
    source_yuv = scratch / "source.yuv"
    subprocess.run(convert_command("ffmpeg", a.al, source_yuv, SOURCE_START, FRAMES), check=True, capture_output=True, timeout=900)
    if digest(source_yuv) != SOURCE_YUV_SHA:
        raise ValueError("source conversion changed")
    raw_source = np.memmap(source_yuv, np.uint8, "r", shape=(FRAMES, 1620, 1920))

    # the paid scale-4 base: dav1d decode, GPU RGB, the paid cubic upscale; its saved ticks must reproduce
    base_ivf = scratch / "base" / "al-s4.ivf"
    base_ivf.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.base) as outer:
        bundle = outer.extractfile("publish/bundle.tar")
        assert bundle is not None
        with tarfile.open(fileobj=bundle, mode="r:") as inner:
            manifest = json.load(inner.extractfile("base/manifest.json"))  # type: ignore[arg-type]
            ivf_row = next(r for r in manifest["rows"] if r["window"] == "al" and r["scale"] == 4)
            with inner.extractfile("base/al-s4.ivf") as src, base_ivf.open("wb") as dst:  # type: ignore[union-attr]
                shutil.copyfileobj(src, dst)
    if digest(base_ivf) != ivf_row["ivf"]["sha256"]:
        raise ValueError("paid base stream changed")
    base_yuv = scratch / "base" / "al-s4.yuv"
    subprocess.run(svtav1.decode_command("dav1d", base_ivf, base_yuv, threads=2), check=True, capture_output=True, timeout=900)
    raw_base = np.memmap(base_yuv, np.uint8, "r", shape=(FRAMES, 272 * 3 // 2, 480))
    saved_base = np.load(extract_member(a.gpu_base, "publish/al-s4-gpu.npy", scratch / "base" / "al-s4-gpu.npy"), mmap_mode="r")

    def base_frame(t: int) -> np.ndarray:
        return cv2.resize(plane_rgb(raw_base, t, 480, 272, torch), (1920, 1080), interpolation=cv2.INTER_CUBIC)

    base_diff = max(int(np.abs(base_frame(t).astype(np.int16) - np.asarray(saved_base[t // 3]).astype(np.int16)).max()) for t in range(0, FRAMES, 3))

    # plain SVT-AV1 at the hybrid's rate: W1's half-resolution streams at the bracketing CRFs
    half = extract_all(a.encode_half, scratch / "half") / "publish"
    points = [p for p in w1_score.selected_points(half) if p["kind"] == "roi" and p["policy"] == ANCHOR_POLICY and p["crf"] in ANCHOR_CRFS]
    points.sort(key=lambda p: p["crf"])
    if [p["crf"] for p in points] != list(ANCHOR_CRFS):
        raise ValueError("anchor streams missing")
    anchors: dict[str, Any] = {}
    for point in points:
        name = f"svt_crf{point['crf']}"
        yuv = scratch / f"{name}.yuv"
        subprocess.run(svtav1.decode_command("dav1d", point["path"], yuv, threads=2), check=True, capture_output=True, timeout=900)
        if digest(yuv) != point["decoded_sha256"]:
            raise ValueError("decoded anchor differs from the encoder job: " + name)
        width, height = point["size"]
        anchors[name] = {"raw": np.memmap(yuv, np.uint8, "r", shape=(FRAMES, height * 3 // 2, width)), "size": (width, height),
                         "rate_kbps": os.path.getsize(point["path"]) * 8 / (FRAMES / FPS) / 1000, "crf": point["crf"]}
    names = list(anchors)
    variants = [*VARIANTS, *names]
    low, high = anchors[names[0]]["rate_kbps"], anchors[names[1]]["rate_kbps"]
    log_weight(TARGET_KBPS, low, high)  # the bracket must hold

    # the regenerated crops and receiver masks
    generated = load_generated([str(p) for p in a.generated], scratch / "generated")
    crops = {u: g for (u, _), g in generated.items()}
    unit_names = [u["name"] for u in wire["units"]]
    masks_root = a.masks_dir if a.masks_dir is not None else extract_all(a.masks, scratch / "masks") / "publish"
    predicted_masks = {}
    for i, name in enumerate(unit_names):
        with np.load(masks_root / f"{name}-masks.npz") as z:
            predicted_masks[i] = {int(t): m for t, m in zip(z["t"], z["masks"])}

    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0 for t, r in enumerate(region) for p in players}
    median_height = {p: float(np.median([heights[(p, t)] for t in range(FRAMES) if p in region[t]] or [1.0])) for p in players}
    setting = {p: "whole" if median_height[p] >= k6.WHOLE_FRAME_ABOVE * k6.FRAME[1] else "crop1" for p in players}
    union = [np.logical_or.reduce([*r.values(), np.zeros((1080, 1920), bool)]) for r in region]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(scratch / "ultralytics"))
    poses = k4.Poses([("yolo11x-pose", a.pose)])
    judge = Judge(poses, Crops(poses))

    n_pairs = frames
    pose_err = np.full((len(variants), frames, len(players)), np.nan)
    kp_err = np.full((frames, len(players)), np.nan)
    flick = np.full((len(variants), n_pairs), np.nan)
    energy = np.full((len(variants), n_pairs), np.nan)
    energy_src = np.full(n_pairs, np.nan)
    racket_px = np.full((frames, len(players)), np.nan)
    racket_cov = np.full((frames, len(players)), np.nan)
    racket_written = np.full((frames, len(players)), np.nan)
    missed = np.zeros((len(variants), frames, len(players)), bool)
    clip_dir = Path(os.environ["PS_STAGE_DIR"]) / "clips"
    side = Clip(clip_dir / "w6-side-by-side-source-svt-hybrid.mp4", 2880, 540) if a.clips else None
    zoom = Clip(clip_dir / "w6-zoom2x-near-player-source-svt-hybrid.mp4", 2880, 540) if a.clips else None
    near = max(players, key=lambda p: median_height[p])
    centre = {p: np.array([960.0, 540.0]) for p in players}
    clip_anchor = names[-1] if a.clip_anchor not in names else a.clip_anchor

    def anchor_frame(name: str, t: int) -> np.ndarray:
        info = anchors[name]
        image = plane_rgb(info["raw"], t, info["size"][0], info["size"][1], torch)
        return cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC) if info["size"] != (1920, 1080) else image

    def hybrid_frame(t: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        player_crops, player_boxes, masks = {}, {}, {}
        for i, p in enumerate(players):
            g = crops[unit_names[i]]
            idx = np.flatnonzero(g["t"] == t)
            if len(idx) == 1:
                player_crops[p] = g["crops"][int(idx[0])]
            if t in boxes[i]:
                player_boxes[p] = boxes[i][t]
            masks[p] = project(predicted_masks[i][t], boxes[i][t])
        image, written = compose(base_frame(t), player_crops, player_boxes, masks, order)
        return image, masks, written

    chunk_size = 8
    for start in range(0, frames, chunk_size):
        chunk = list(range(start, min(frames, start + chunk_size)))
        times = sorted(set(chunk) | {t - 1 for t in chunk if t > 0})
        images: dict[tuple[str, int], np.ndarray] = {}
        extras: dict[int, Any] = {}
        for t in times:
            images[("source", t)] = plane_rgb(raw_source, t, 1920, 1080, torch)
            image, masks, written = hybrid_frame(t)
            images[("hybrid", t)] = image
            extras[t] = (masks, written)
            for name in names:
                images[(name, t)] = anchor_frame(name, t)
        bodies = [region[t] if t in times else {} for t in range(FRAMES)]
        found = judge.run({k: v for k, v in images.items() if k[1] in chunk}, bodies, setting, players)
        for t in chunk:
            for j, p in enumerate(players):
                src = found.get(("source", t, p))
                if src is None:
                    continue
                for v_i, v in enumerate(variants):
                    got = found.get((v, t, p))
                    err = pose_error(got, src, heights[(p, t)])
                    pose_err[v_i, t, j] = np.nan if err is None else err
                    missed[v_i, t, j] = got is None
                if p in coded[t]:
                    err = pose_error(coded[t][p], src, heights[(p, t)])
                    kp_err[t, j] = np.nan if err is None else err
                # racket: the source region beyond the keypoint hull, against the generated player
                if p in region[t] and p in coded[t]:
                    k = coded[t][p]
                    pts = k[k[:, 2] >= 0.5, :2].astype(np.int32)
                    if len(pts) >= 3:
                        hull = np.zeros((1080, 1920), np.uint8)
                        cv2.fillConvexPoly(hull, np.asarray(cv2.convexHull(pts), np.int32), (1,))
                        size = max(3, int(2 * RACKET_MARGIN * heights[(p, t)]) | 1)
                        body = cv2.dilate(hull, np.ones((size, size), np.uint8)) > 0
                        beyond = racket_stats(racket_region(region[t][p], body), extras[t][0][p], extras[t][1])
                        if beyond is not None:
                            racket_px[t, j], racket_cov[t, j], racket_written[t, j] = beyond[0], beyond[1], beyond[2]
            if t > 0:
                for v_i, v in enumerate(variants):
                    residual = residual_energies(images[(v, t)], images[(v, t - 1)], images[("source", t)], images[("source", t - 1)], union[t], union[t - 1])
                    if residual is not None:
                        flick[v_i, t], energy[v_i, t], energy_src[t] = residual
            if side is not None and zoom is not None:
                trio = [images[("source", t)], images[(clip_anchor, t)], images[("hybrid", t)]]
                side.write(np.concatenate([cv2.resize(x, (960, 540), interpolation=cv2.INTER_AREA) for x in trio], 1))
                if near in region[t]:
                    ys, xs = np.nonzero(region[t][near])
                    centre[near] = 0.8 * centre[near] + 0.2 * np.array([xs.mean(), ys.mean()])
                cx = int(np.clip(centre[near][0], 240, 1920 - 240))
                cy = int(np.clip(centre[near][1], 135, 1080 - 135))
                zoom.write(np.concatenate([cv2.resize(x[cy - 135:cy + 135, cx - 240:cx + 240], (960, 540), interpolation=cv2.INTER_CUBIC) for x in trio], 1))
        np.savez_compressed(publish / "judges.npz", variants=np.asarray(variants), players=np.asarray(players), pose_err=pose_err, kp_err=kp_err,
                            flicker=flick, energy=energy, energy_src=energy_src, racket_px=racket_px, racket_cov=racket_cov,
                            racket_written=racket_written, missed=missed, done=np.asarray(chunk[-1] + 1))
        print(f"phase=judge frames={chunk[-1] + 1}/{frames}", flush=True)
        publish_progress(os.environ["PS_STAGE"], chunk[-1] + 1)
    codes = [c.close() for c in (side, zoom) if c is not None]
    if any(codes):
        raise RuntimeError("ffmpeg clip encoding failed")
    last = max(times)
    strip = np.concatenate([cv2.resize(images[(v, last)], (960, 540)) for v in ("source", clip_anchor, "hybrid")], 1)
    cv2.imwrite(str(publish / f"look-t{last:03d}.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary = {"variants": variants, "frames": frames, "players": players, "setting": setting, "median_height": median_height,
               "anchor_rates_kbps": {n: anchors[n]["rate_kbps"] for n in names}, "anchor_crfs": {n: anchors[n]["crf"] for n in names},
               "target_kbps": TARGET_KBPS, "base_tick_max_abs_diff_vs_paid_rgb": base_diff,
               "tools": {"dav1d": tool_version(["dav1d", "--version"]), "ffmpeg": tool_version(["ffmpeg", "-version"])},
               "pose": poses.record(), "gpu": torch.cuda.get_device_name(0), "device": device_record(a.gpu_class),
               "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "seconds": round(time.monotonic() - began, 1),
               "inputs": {k: digest(getattr(a, k)) for k in ("al", "transport", "anchor", "base", "gpu_base", "encode_half", "pose", "masks") if getattr(a, k) is not None},
               "judges_sha256": digest(publish / "judges.npz"), "clips": sorted(p.name for p in clip_dir.glob("*.mp4")) if a.clips else []}
    (publish / "w6-judge.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    (stage / "w6-judge.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return 0


def command_post(a: argparse.Namespace) -> int:
    """Receiver masks on the regenerated crops, then the judges, in one job; the smoke's frame count limits both."""
    a.judged = a.frames
    a.source = a.al
    a.masks = None
    code = command_masks(a)
    if code:
        return code
    a.masks_dir = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    return command_judge(a)


def validate_judge() -> dict[str, bool]:
    stage = Path(os.environ["PS_STAGE_DIR"])
    summary = json.loads((stage / "w6-judge.json").read_text())
    with tarfile.open(stage / "published.tar") as tar:
        import io
        member = tar.extractfile("publish/judges.npz")
        assert member is not None
        z = np.load(io.BytesIO(member.read()))
    frames = summary["frames"]
    checks = {"frames_done": int(z["done"]) == frames,
              "base_reproduces_paid": summary["base_tick_max_abs_diff_vs_paid_rgb"] <= 2,
              "anchor_bracket": min(summary["anchor_rates_kbps"].values()) <= summary["target_kbps"] <= max(summary["anchor_rates_kbps"].values()),
              "pose_finite": bool(np.isfinite(z["pose_err"][:, :frames]).mean() > 0.5),
              "flicker_finite": bool(np.isfinite(z["flicker"][:, 1:frames]).mean() > 0.5),
              "source_energy_finite": bool(np.isfinite(z["energy_src"][1:frames]).mean() > 0.5),
              "kp_error_finite": bool(np.isfinite(z["kp_err"]).mean() > 0.5),
              "source_is_zero_flicker": bool(abs(np.nanmean(z["flicker"][0, 1:frames])) < 1e-6)}
    if summary["clips"]:
        checks["clips_present"] = all((stage / "clips" / n).stat().st_size > 10_000 for n in summary["clips"])
    return checks


def validate_generate() -> dict[str, bool]:
    import hashlib
    import io

    stage = Path(os.environ["PS_STAGE_DIR"])
    result = json.loads((stage / "k6b_generate.json").read_text())
    expected = result["max_frames"] or FRAMES
    checks = {"device_class": bool(result["device"]["class_matches"]), "keep_all": bool(result.get("keep_all")),
              "peak_within_declaration": result["gpu_peak_gib"] * 1024 <= result["gpu_memory_limit_mib"]}
    for profile, rec in result["models"].items():
        checks[f"{profile}:on_cuda"] = all(rec["on_cuda"].values())
        checks[f"{profile}:no_missing_keys"] = all(rec[k]["missing"] == 0 for k in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider"))
        checks[f"{profile}:kernels_on_gpu"] = rec["kernels"]["kernel_launches"] > 0
    with tarfile.open(stage / "published.tar") as tar:
        for name in result["units"]:
            for profile in result["profiles"]:
                blob = tar.extractfile(f"publish/{name}@{profile}.npz")
                meta_file = tar.extractfile(f"publish/{name}@{profile}.json")
                assert blob is not None and meta_file is not None
                data, meta = blob.read(), json.loads(meta_file.read())
                with np.load(io.BytesIO(data)) as z:
                    checks[f"{name}:frames"] = z["t"].tolist() == list(range(expected))
                    checks[f"{name}:shape"] = z["crops"].shape[1:] == (784, 512, 3) and len(z["crops"]) == expected
                checks[f"{name}:hash"] = hashlib.sha256(data).hexdigest() == meta["npz_sha256"]
                checks[f"{name}:not_flat"] = meta["crop_std_min"] > 5.0
                checks[f"{name}:generated_all"] = meta["frames_generated"] == meta["kept"] == expected
    return checks


# ---------------------------------------------------------------- report (local CPU)


def report(judges: Path, summary: Path) -> dict[str, Any]:
    z = np.load(judges)
    meta = json.loads(summary.read_text())
    variants = [str(v) for v in z["variants"]]
    frames = meta["frames"]
    target = meta["target_kbps"]
    names = [v for v in variants if v.startswith("svt_")]
    low, high = names[0], names[1]
    rate = meta["anchor_rates_kbps"]
    idx = {v: variants.index(v) for v in variants}

    def per_frame(array: np.ndarray, axis_player: bool) -> dict[str, np.ndarray]:
        out = {}
        for v in variants:
            a = array[idx[v]][:frames]
            out[v] = np.nanmean(a, axis=1) if axis_player else a
        return out

    def combined(series: dict[str, np.ndarray]) -> np.ndarray:
        return at_rate(target, (rate[low], series[low]), (rate[high], series[high]))

    pose = per_frame(z["pose_err"], True)
    flicker = per_frame(z["flicker"], False)
    energy = per_frame(z["energy"], False)
    pose_anchor, flicker_anchor = combined(pose), combined(flicker)
    coded = np.nanmean(z["kp_err"][:frames], axis=1)
    valid_pose = np.isfinite(pose["hybrid"]) & np.isfinite(pose_anchor) & np.isfinite(coded)
    valid_time = np.isfinite(flicker["hybrid"]) & np.isfinite(flicker_anchor)
    rows = {"pose": {"method": mean_of(pose["hybrid"][valid_pose]), "anchor": mean_of(pose_anchor[valid_pose]), "coded": mean_of(coded[valid_pose])},
            "time": {"flicker": mean_of(flicker["hybrid"][valid_time]), "anchor_flicker": mean_of(flicker_anchor[valid_time]),
                     "energy": mean_of(energy["hybrid"][valid_time]), "source_energy": mean_of(z["energy_src"][:frames][valid_time])}}
    decision = decide(rows)
    intervals = {"pose_method_minus_anchor": block_bootstrap((pose["hybrid"] - pose_anchor)[valid_pose]),
                 "flicker_method_minus_anchor": block_bootstrap((flicker["hybrid"] - flicker_anchor)[valid_time]),
                 "energy_method_minus_limit": block_bootstrap((energy["hybrid"] - ENERGY_FACTOR * z["energy_src"][:frames])[valid_time])}
    missed = {v: float(z["missed"][idx[v]][:frames].mean()) for v in variants}
    return {"rows": rows, "decision": decision, "bootstrap95": intervals, "missed_share": missed, "frames": frames,
            "per_variant": {v: {"pose": mean_of(pose[v][valid_pose]), "flicker": mean_of(flicker[v][valid_time]), "energy": mean_of(energy[v][valid_time])} for v in variants},
            "racket": racket_summary(z["racket_px"][:frames], z["racket_cov"][:frames]),
            "racket_written": racket_summary(z["racket_px"][:frames], z["racket_written"][:frames]),
            "valid_pose_frames": int(valid_pose.sum()), "valid_time_frames": int(valid_time.sum())}


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("masks")
    for name in ("transport", "old_generated", "old_masks", "weights", "source"):
        m.add_argument("--" + name.replace("_", "-"), type=Path, required=True, dest=name)
    m.add_argument("--generated", type=Path, action="append", required=True)
    m.add_argument("--judged", type=int, default=0)
    j = sub.add_parser("judge")
    for name in ("al", "transport", "anchor", "base", "gpu_base", "encode_half", "pose"):
        j.add_argument("--" + name.replace("_", "-"), type=Path, required=True, dest=name)
    j.add_argument("--masks", type=Path, required=True)
    j.add_argument("--generated", type=Path, action="append", required=True)
    j.add_argument("--frames", type=int, default=0)
    j.add_argument("--clips", action="store_true")
    j.add_argument("--clip-anchor", default="svt_crf66")
    j.add_argument("--gpu-class", default="RTX")
    j.set_defaults(masks_dir=None)
    # post: receiver masks and then the judges, in one job (the inputs are staged once)
    q = sub.add_parser("post")
    for name in ("al", "transport", "anchor", "base", "gpu_base", "encode_half", "pose", "old_generated", "old_masks", "weights"):
        q.add_argument("--" + name.replace("_", "-"), type=Path, required=True, dest=name)
    q.add_argument("--generated", type=Path, action="append", required=True)
    q.add_argument("--frames", type=int, default=0)
    q.add_argument("--clips", action="store_true")
    q.add_argument("--clip-anchor", default="svt_crf66")
    q.add_argument("--gpu-class", default="RTX")
    for name in ("validate-generate", "validate-masks", "validate-judge", "validate-post"):
        sub.add_parser(name)
    r = sub.add_parser("report")
    r.add_argument("judges", type=Path)
    r.add_argument("summary", type=Path)
    a = parser.parse_args(argv)
    if a.command == "masks":
        return command_masks(a)
    if a.command == "judge":
        return command_judge(a)
    if a.command == "post":
        return command_post(a)
    if a.command == "report":
        print(json.dumps(report(a.judges, a.summary), indent=1))
        return 0
    checks = {"validate-generate": validate_generate, "validate-masks": validate_masks, "validate-judge": validate_judge,
              "validate-post": lambda: {**validate_masks(), **validate_judge()}}[a.command]()
    result = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        Path(target).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
