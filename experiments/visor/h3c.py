"""PLAN step H3: H3c, a perceptually tuned DCVC-UF for the hand pixels (docs/experiments.md, 2026-10-11).

    python -m experiments.visor.h3c train --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --video V.MP4 ... --items ID,ID --segments all|N --steps N --lpips-weight W \\
        --image-ckpt PTH --video-ckpt PTH --lpips-backbone PTH --frames N
    python -m experiments.visor.h3c code --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --video V.MP4 ... --items ID,ID --frames N --image-ckpt PTH --arm NAME PTH ... [--anchor \\
        --svt-streams DIR --svt-result JSON ... --inspect BIN] --lpips-backbone PTH --dists-backbone PTH
    python -m experiments.visor.h3c validate
    python -m experiments.visor.h3c report --code TAR ... --h1-report JSON --h3b TAR ... --out DIR

``train`` builds the training data from evaluation-set windows (hand-centred 512×512 segments at each scale, the
loss mask the hand region R) and fine-tunes DCVC-UF HT-L with `experiments.background.g5b_train`, unchanged.
``code`` codes each window with every arm at every scale and QP (B2's worker), splits each NAL's bytes onto R by
the model's own per-cell bit estimate (`experiments.visor.h3c_bits`), and scores the decode inside R as H3b scored
SVT-AV1; with ``--anchor`` it also scores SVT-AV1's points (B2's streams and H3b2's low points). Each window is
published as ``code/<item>@<label>.npz`` with a ``.json`` and checkpointed. ``report`` applies `DECISION`.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import FILL, HANDS, LPIPS_BATCH, POINTS, RESAMPLES, bootstrap_mean, hand_kbps, psnr_of, rgb_memmap, seed_of
from experiments.visor.h3b import (DISTS_PAD, LOW_POINTS, dists_crops, encode_low, load_dists, low_name, pareto_front, rate_on_front,
                                   scaled_size)

SCALES = (1, 2, 4)
QPS = (0, 21, 42)
SEGMENT = 48  # window frames per training segment (one crop box)
CROP = 512  # G5b's patch: each segment is cropped to it, so the trainer's random offset is 0
PILOT = ("P01_107", "P02_02", "P03_10", "P09_106")  # B2's pilot videos
EXCLUDED = ("P01", "P02", "P03", "P09")  # the pilot items' participants, never in the component's data
BD_LEVELS = 64
SEED = "pointstream-h3c"
CONTAINER_HEADER = 9  # `src.codecs.dcvc_uf_worker.HEADER`

#: The decision rule, fixed before any fleet run (docs/experiments.md, H3c entry).
DECISION: dict[str, Any] = {
    "quality": "lpips_r",  # decode inside R, source outside, full-frame LPIPS averaged over R; mean over hand frames
    "anchor": "SVT-AV1's per-item front on R over B2's CRF 41/48/55/62 and H3b2's seven low points",
    "bd": "mean log-rate difference over the LPIPS range both fronts cover, linear in log rate, 64 levels",
    "upper_bound_passes_mean_bd": -0.10,  # at some beta, with at least 3 of 4 items below 0
    "upper_bound_fails_mean_bd": 0.0,  # at both beta, mean at or above it (or SVT-AV1 dominates): H3c is not built
    "items_below_zero": 3,
    "component_adopted_mean_bd": 0.0,  # against SVT-AV1, with at least 3 of 4 items below 0
    "component_vs_reference_bd": -0.10,  # the tuning, not DCVC-UF itself, earns it
}


# ----------------------------------------------------------------- windows (shared by train and code)


def participant(item_id: str) -> str:
    return item_id.split("_")[0]


def downscale_planes(plane: np.ndarray, width: int, height: int, sw: int, sh: int) -> np.ndarray:
    """One 4:2:0 frame (h*3/2, w) to (sh*3/2, sw), per plane by OpenCV's area filter (H3b2's ``encode_low``)."""
    import cv2

    if (sw, sh) == (width, height):
        return plane
    y = cv2.resize(plane[:height], (sw, sh), interpolation=cv2.INTER_AREA)
    uv = plane[height:].reshape(2, height // 2, width // 2)
    chroma = [cv2.resize(np.ascontiguousarray(c), (sw // 2, sh // 2), interpolation=cv2.INTER_AREA) for c in uv]
    return np.concatenate([y, *(c.reshape(sh // 4, sw) for c in chroma)])


def downscale_mask(mask: np.ndarray, sw: int, sh: int) -> np.ndarray:
    """H3b2's region at the coded size: area fraction at least 0.5."""
    import cv2

    if mask.shape == (sh, sw):
        return mask
    return cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) >= 0.5


def segment_box(union: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left (x, y), both even, of a ``size`` square centred on ``union``'s pixels and kept inside the frame
    (at 0 when the frame is smaller than the square)."""
    h, w = union.shape
    ys, xs = np.nonzero(union)
    cy, cx = (float(ys.mean()), float(xs.mean())) if len(ys) else (h / 2, w / 2)

    def place(centre: float, extent: int) -> int:
        return 2 * (int(min(max(centre - size / 2, 0), max(extent - size, 0))) // 2)

    return place(cx, w), place(cy, h)


def crop_planes(plane: np.ndarray, width: int, height: int, x: int, y: int, size: int) -> np.ndarray:
    """A ``size`` square of a 4:2:0 frame at even (x, y), edges replicated where the frame is smaller."""
    luma = plane[:height]
    uv = plane[height:].reshape(2, height // 2, width // 2)

    def cut(a: np.ndarray, x0: int, y0: int, s: int) -> np.ndarray:
        part = a[y0:y0 + s, x0:x0 + s]
        return np.pad(part, ((0, s - part.shape[0]), (0, s - part.shape[1])), mode="edge")

    half = size // 2
    chroma = [cut(c, x // 2, y // 2, half).reshape(half // 2, size) for c in uv]
    return np.concatenate([cut(luma, x, y, size), *chroma])


def crop_mask(mask: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    part = mask[y:y + size, x:x + size]
    return np.pad(part, ((0, size - part.shape[0]), (0, size - part.shape[1])))  # padding is outside the loss


def choose_segments(hand_pixels: np.ndarray, count: int | None) -> list[int]:
    """Indices of the window's ``SEGMENT``-frame segments to train on: all, or the ``count`` with most hand pixels."""
    starts = list(range(0, len(hand_pixels) - SEGMENT + 1, SEGMENT))
    totals = [int(hand_pixels[s:s + SEGMENT].sum()) for s in starts]
    order = [k for k in range(len(starts)) if totals[k] > 0]
    if count is not None:
        order = sorted(sorted(order, key=lambda k: -totals[k])[:count])
    return [starts[k] for k in order]


def load_window(args: argparse.Namespace, item: dict[str, Any], frames: int, decoded: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """The window's 4:2:0 planes (memmap) and R per frame (both hand classes of the fill set)."""
    from experiments.visor.b2 import load_mask_sets

    clips, info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    width, height = decoded["width"], decoded["height"]
    planes = np.memmap(decoded["path"], np.uint8, "r", shape=(frames, height * 3 // 2, width))
    return planes, regions, info[FILL]


def decode_windows(args: argparse.Namespace, items: list[dict[str, Any]], frames: int, scratch: Path) -> Any:
    """B2's window decode in fresh processes (PyAV before torch), two ahead; yields (item, decoded)."""
    from experiments.visor.b2 import decode_window

    videos = {Path(v).stem: Path(v) for v in args.video}
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))

    def submit(item: dict[str, Any]) -> Any:
        target = scratch / "source" / f"{item['id']}.yuv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        return pool.submit(decode_window, videos[item["video"]], item, frames, Path(args.archive), target, max(1, allowance // 2))

    pending = {i: submit(items[i]) for i in range(min(2, len(items)))}  # started now, before torch loads

    def windows() -> Any:
        for number, item in enumerate(items):
            decoded = pending.pop(number).result()
            if number + 2 < len(items):
                pending[number + 2] = submit(items[number + 2])
            yield item, decoded
        pool.shutdown()

    return windows()


# ----------------------------------------------------------------- train


def build_training_data(args: argparse.Namespace, items: list[dict[str, Any]], frames: int, count: int | None, work: Path,
                        scratch: Path) -> dict[str, Any]:
    """``frames.npy`` (n, CROP*3/2, CROP) 4:2:0 and ``keep.npy`` (n, CROP, CROP) R, segment after segment with one
    blank separator frame between them (outside the fitted frames, so no sequence crosses two segments)."""
    blocks: list[np.ndarray] = []
    keeps: list[np.ndarray] = []
    fit: list[int] = []
    segments: list[dict[str, Any]] = []
    decodes = []
    n = 0
    for item, decoded in decode_windows(args, items, frames, scratch):
        planes, regions, mask_info = load_window(args, item, frames, decoded)
        width, height = decoded["width"], decoded["height"]
        for scale in SCALES:
            sw, sh = scaled_size(width, height, scale)
            small = [downscale_mask(r, sw, sh) for r in regions]
            for start in choose_segments(np.array([int(m.sum()) for m in small]), count):
                x, y = segment_box(np.any(small[start:start + SEGMENT], axis=0), CROP)
                for t in range(start, start + SEGMENT):
                    plane = downscale_planes(np.asarray(planes[t]), width, height, sw, sh)
                    blocks.append(crop_planes(plane, sw, sh, x, y, CROP))
                    keeps.append(crop_mask(small[t], x, y, CROP))
                    fit.append(n)
                    n += 1
                blocks.append(np.zeros((CROP * 3 // 2, CROP), np.uint8))
                keeps.append(np.zeros((CROP, CROP), bool))
                n += 1
                segments.append({"item": item["id"], "scale": scale, "size": [sw, sh], "start": start, "box": [x, y],
                                 "hand_share": float(np.mean([keeps[-1 - k].mean() for k in range(1, SEGMENT + 1)]))})
        decodes.append({"item": item["id"], **{k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info})
        os.remove(decoded["path"])
    np.save(work / "frames.npy", np.stack(blocks))
    np.save(work / "keep.npy", np.stack(keeps))
    return {"fit": fit, "segments": segments, "decodes": decodes, "frames": n}


def trainer_command(prefix: Path, plan: Path, report: Path) -> tuple[list[str], dict[str, str], Path]:
    """`experiments.background.g5b_train`, unchanged, in DCVC's vendored tree (as G5b runs it)."""
    from experiments.background.g5b import trainer_command as g5b_trainer

    return g5b_trainer(prefix, plan, report)


def command_train(args: argparse.Namespace) -> int:
    import torch

    from experiments.background.g5b import TRAIN, monitor_converged
    from experiments.visor.b2 import select_items

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    if args.fit == "component" and (bad := [it["id"] for it in items if participant(it["id"]) in EXCLUDED]):
        raise SystemExit(f"the component never sees the pilot participants: {bad}")
    began = time.time()
    frames = int(args.frames)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "train"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    data = build_training_data(args, items, frames, None if args.segments == "all" else int(args.segments), work, scratch)
    prepare_seconds = time.time() - began
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    steps = int(args.steps)
    plan = {**TRAIN, "patch": [CROP, CROP], "frames": str(work / "frames.npy"), "keep": str(work / "keep.npy"),
            "fit": data["fit"], "steps": steps, "monitor_every": max(1, steps // 10), "save_every": max(1, steps // 10),
            "lpips_weight": float(args.lpips_weight), "lpips_backbone": args.lpips_backbone,
            "image_ckpt": args.image_ckpt, "image_sha256": file_sha256(Path(args.image_ckpt)),
            "video_ckpt": args.video_ckpt, "video_sha256": file_sha256(Path(args.video_ckpt)),
            "out": str(work / "model"), "resume_dir": str(checkpoints) if checkpoints else None}
    write_json(work / "plan.json", plan)
    command, env, cwd = trainer_command(Path(sys.prefix), work / "plan.json", work / "trained.json")
    log_path = work / "train.log"
    with log_path.open("w") as log:
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    log_text = log_path.read_text()
    if done.returncode:
        raise RuntimeError(f"fine-tuning failed ({done.returncode}):\n{log_text[-4000:]}")
    trained = json.loads((work / "trained.json").read_text())
    publish = scratch / "publish" / "model"
    shutil.rmtree(publish, ignore_errors=True)
    publish.mkdir(parents=True)
    shutil.copyfile(trained["checkpoint"], publish / "video_ft.pth.tar")
    shutil.copyfile(log_path, publish / "train.log")
    row = {"command": "train", "fit": args.fit, "items": [it["id"] for it in items], "segments": args.segments,
           "lpips_weight": float(args.lpips_weight), "steps": steps, "train": {k: plan[k] for k in (*TRAIN, "patch", "monitor_every")},
           "data": {k: data[k] for k in ("segments", "decodes", "frames")}, "fit_frames": len(data["fit"]),
           "model": {k: trained[k] for k in ("checkpoint_sha256", "checkpoint_bytes", "parameters")},
           "base_checkpoints": {"image_sha256": plan["image_sha256"], "video_sha256": plan["video_sha256"]},
           "curve": trained["curve"], "converged": monitor_converged(trained["curve"]),
           "train_seconds": trained["train_seconds"], "seconds_per_step": trained["seconds_per_step"],
           "peak_memory_gib": trained["peak_memory_gib"], "skipped_nonfinite": trained["skipped_nonfinite"],
           "sequence_starts": trained["sequence_starts"], "resumed_at": trained["resumed_at"],
           "prepare_seconds": round(prepare_seconds, 1), "seconds": round(time.time() - began, 1),
           "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
           "trainer_gpu": trained["gpu"], "trainer_torch": trained["torch"], "decision_rule": DECISION}
    write_json(publish / "train.json", row)
    write_json(stage_dir() / "h3c.json", row)
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- code


def nal_region_bits(nals: list[dict[str, Any]], estimate: list[dict[str, Any]], container_bytes: int) -> dict[str, float]:
    """Each NAL's bytes (with its SPS) split onto R by the estimate's share in that NAL; the container header by the
    stream's overall share. Also the estimate's total against the actual bits."""
    if len(nals) != len(estimate):
        raise RuntimeError(f"{len(nals)} NALs coded, {len(estimate)} estimated")
    region = actual = est = est_region = 0.0
    for nal, e in zip(nals, estimate):
        bits = 8.0 * (nal["nal_bytes"] + nal["sps_bytes"])
        share = e["estimated_region_bits"] / e["estimated_bits"] if e["estimated_bits"] > 0 else 0.0
        region += bits * share
        actual += bits
        est += e["estimated_bits"]
        est_region += e["estimated_region_bits"]
    overall = est_region / est if est > 0 else 0.0
    header = 8.0 * (container_bytes - actual / 8.0)
    return {"region_bits": region + header * overall, "bits": 8.0 * container_bytes, "estimated_bits": est,
            "estimate_over_actual": est / actual if actual else math.nan, "region_share": (region + header * overall) / (8.0 * container_bytes)}


def run_tree_script(script: Path, plan: dict[str, Any], work: Path, name: str) -> dict[str, Any]:
    """A script in DCVC's vendored tree (cwd and PYTHONPATH there), as `src.codecs.dcvc_uf_worker` runs."""
    tree = Path(sys.prefix) / "opt" / "DCVC"
    write_json(work / f"{name}-plan.json", plan)
    command = [str(Path(sys.prefix) / "bin" / "python"), str(script), "--plan", str(work / f"{name}-plan.json"),
               "--report", str(work / f"{name}.json")]
    done = subprocess.run(command, env={**os.environ, "PYTHONPATH": str(tree), "PYTHONNOUSERSITE": "1"}, cwd=tree,
                          capture_output=True, text=True, timeout=3600)
    if done.returncode:
        raise RuntimeError(f"{script.name} failed ({done.returncode}):\n{done.stderr[-4000:]}")
    return json.loads((work / f"{name}.json").read_text())


def code_dcvc(args: argparse.Namespace, source: Path, region_file: Path, size: tuple[int, int], frames: int, qp: int,
              video_ckpt: str, video_sha: str, work: Path) -> dict[str, Any]:
    """One DCVC-UF HT-L encode and decode (B2's worker) and the per-NAL bit split onto R."""
    from src.codecs.dcvc_uf_worker import dcvc_command

    work.mkdir(parents=True, exist_ok=True)
    width, height = size
    plan = {"structure": "htl", "qp": qp, "frame_count": frames, "height": height, "width": width, "src_type": "yuv420",
            "frames_file": str(source), "container": str(work / "stream.psdc"), "out_file": str(work / "decoded.yuv"),
            "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256, "video_ckpt": video_ckpt,
            "video_sha256": video_sha, "profile": False}
    write_json(work / "plan.json", plan)
    reports = {}
    for action in ("encode", "decode"):
        command, env, cwd = dcvc_command(Path(sys.prefix), action, work / "plan.json", work / f"{action}.json")
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
        if done.returncode:
            raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads((work / f"{action}.json").read_text())
    estimate = run_tree_script(Path(__file__).resolve().parent / "h3c_bits.py",
                               {"frames_file": str(source), "region_file": str(region_file), "width": width, "height": height,
                                "frame_count": frames, "qp": qp, "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256,
                                "video_ckpt": video_ckpt, "video_sha256": video_sha}, work, "bits")
    data = (work / "stream.psdc").read_bytes()
    encode, decode = reports["encode"], reports["decode"]
    split = nal_region_bits(encode["nals"], estimate["nals"], len(data))
    return {"qp": qp, "size": [width, height], "bytes": len(data), "stream_sha256": file_sha256(work / "stream.psdc"),
            "decoded": str(work / "decoded.yuv"), "deterministic": decode["deterministic"],
            "decoder_matches_encoder_intra": decode["passes"][0]["frame_sha256"][0] == encode["i_recon_sha256"],
            "encode_seconds": encode["encode_seconds"], "decode_seconds": [p["decode_seconds"] for p in decode["passes"]],
            "estimate_seconds": estimate["seconds"], "nals": len(encode["nals"]), **split}


class Scorer:
    """H3b's scoring of SVT-AV1 on R: the decode inside R, the source outside; LPIPS of the full frame averaged over R,
    the squared error on R, masked DISTS on R's box plus `DISTS_PAD`."""

    def __init__(self, nets: dict[str, Any], truth: np.ndarray, regions: list[np.ndarray], device: str):
        self.nets, self.truth, self.regions, self.device = nets, truth, regions, device
        self.hand = [t for t, r in enumerate(regions) if r.any()]

    def __call__(self, decoded: Any, upscale: bool) -> dict[str, np.ndarray]:
        import cv2
        import torch

        frames = len(self.regions)
        height, width = self.regions[0].shape
        lp, dists = np.full(frames, np.nan), np.full(frames, np.nan)
        sse = np.zeros(frames, np.int64)
        for first in range(0, len(self.hand), LPIPS_BATCH):
            ts = self.hand[first:first + LPIPS_BATCH]
            outs, truths = [], []
            for t in ts:
                frame = np.asarray(decoded[t])
                if upscale:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                truth_t = np.asarray(self.truth[t])
                out = truth_t.copy()
                r = self.regions[t]
                out[r] = frame[r]
                outs.append(out)
                truths.append(truth_t)
                sse[t] = int(((out[r].astype(np.int32) - truth_t[r].astype(np.int32)) ** 2).sum())
                ys, xs = np.nonzero(r)
                box = (slice(max(0, ys.min() - DISTS_PAD), min(height, ys.max() + 1 + DISTS_PAD)),
                       slice(max(0, xs.min() - DISTS_PAD), min(width, xs.max() + 1 + DISTS_PAD)))
                dists[t] = dists_crops(self.nets["dists"], [out[box]], truth_t[box], r[box], self.device)[0]

            def tensor(xs: list[np.ndarray]) -> Any:
                return torch.from_numpy(np.stack(xs)).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0

            with torch.inference_mode():
                spatial = self.nets["lpips"](tensor(outs), tensor(truths))[:, 0]
            for i, t in enumerate(ts):
                m = torch.from_numpy(self.regions[t]).to(self.device)
                lp[t] = float((spatial[i] * m).sum()) / int(self.regions[t].sum())
        return {"lpips": lp, "sse": sse, "dists": dists}


def parse_arms(specs: list[list[str]]) -> list[tuple[str, str]]:
    out = [(name, path) for name, path in specs]
    if len({n for n, _ in out}) != len(out):
        raise SystemExit("arm names must differ")
    return out


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, label: str, first: bool) -> None:
    import cv2

    from experiments.visor.h1 import stream_paths
    from src.codecs import svtav1

    began = time.time()
    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    seconds = frames / float(fps)
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info = load_window(args, item, frames, decoded)
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    score = Scorer(nets, truth, regions, device)
    points: list[dict[str, Any]] = []
    scores: list[dict[str, np.ndarray]] = []
    look = score.hand[len(score.hand) // 2] if score.hand else 0
    ys, xs = np.nonzero(regions[look]) if score.hand else (np.array([0]), np.array([0]))
    look_box = (slice(max(0, ys.min() - DISTS_PAD), min(height, ys.max() + 1 + DISTS_PAD)),
                slice(max(0, xs.min() - DISTS_PAD), min(width, xs.max() + 1 + DISTS_PAD)))
    frames_dir = publish / "frames" / item["id"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / f"source-f{look:03d}.png"), cv2.cvtColor(np.asarray(truth[look])[look_box], cv2.COLOR_RGB2BGR))

    def keep_look(name: str, decoded_rgb: Any, upscale: bool) -> None:
        frame = np.asarray(decoded_rgb[look])
        if upscale:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        out = np.asarray(truth[look]).copy()
        out[regions[look]] = frame[regions[look]]
        cv2.imwrite(str(frames_dir / f"{name.replace('/', '_').replace('@', '-')}-f{look:03d}.png"),
                    cv2.cvtColor(out[look_box], cv2.COLOR_RGB2BGR))

    checks: dict[str, Any] = {}
    if args.anchor:
        streams, stream_info = stream_paths(args, item["id"])
        allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
        for p in POINTS:
            name = f"{p:g}"
            yuv = work / f"crf{name}.yuv"
            svtav1.run(svtav1.decode_command(args.decoder, streams[name], yuv, threads=allowance), 1800)
            rgb = rgb_memmap(yuv, work / f"crf{name}-rgb.npy", frames, width, height, full_range, device)
            yuv.unlink()
            scores.append(score(rgb, upscale=False))
            keep_look(f"svt@{name}", rgb, False)
            points.append({"name": f"svt@{name}", "codec": "svtav1", "arm": "svtav1", "crf": p, "scale": 1,
                           "stream": stream_info[name]})
            os.remove(work / f"crf{name}-rgb.npy")
        for crf, scale in LOW_POINTS:
            rgb, record = encode_low(args, Path(decoded["path"]), frames, (width, height, full_range), crf, scale, regions, fps, work, device)
            scores.append(score(rgb, upscale=scale > 1))
            keep_look(low_name(crf, scale), rgb, scale > 1)
            points.append({"name": low_name(crf, scale), "codec": "svtav1", "arm": "svtav1", "crf": crf, "scale": scale,
                           "kbps_r": record["hand_bits"] / seconds / 1000, "kbps": 8 * record["payload_bytes"] / seconds / 1000,
                           "stream": record})
            shutil.rmtree(work / f"low-{crf:g}-{scale}")
    for scale in args.scales:
        sw, sh = scaled_size(width, height, scale)
        source = Path(decoded["path"])
        if scale > 1:
            source = work / f"source-{scale}.yuv"
            with source.open("wb") as handle:
                for t in range(frames):
                    handle.write(downscale_planes(np.asarray(planes[t]), width, height, sw, sh).tobytes())
        region_file = work / f"region-{scale}.npy"
        np.save(region_file, np.stack([downscale_mask(r, sw, sh) for r in regions]))
        for arm, ckpt in args.arms:
            for qp in args.qps:
                rec = code_dcvc(args, source, region_file, (sw, sh), frames, qp, ckpt, args.arm_sha256[arm], work / f"{arm}-{scale}-{qp}")
                if not rec["deterministic"] or not rec["decoder_matches_encoder_intra"]:
                    raise RuntimeError(f"{item['id']} {arm} scale {scale} QP {qp}: decode is not the encoder's")
                count = Path(rec["decoded"]).stat().st_size // (sw * sh * 3 // 2)
                if count < frames:
                    raise RuntimeError(f"{item['id']} {arm}: decoded {count} of {frames} frames")
                # DCVC-UF codes the source's planes as they are (B2): its decode is viewed in the source's range.
                rgb = rgb_memmap(Path(rec["decoded"]), work / "dcvc-rgb.npy", frames, sw, sh, full_range, device)
                name = f"{arm}@{qp}/{scale}"
                scores.append(score(rgb, upscale=scale > 1))
                keep_look(name, rgb, scale > 1)
                stream_dir = publish / "streams" / item["id"]
                stream_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / f"{arm}-{scale}-{qp}" / "stream.psdc", stream_dir / f"{name.replace('/', '_').replace('@', '-')}.psdc")
                points.append({"name": name, "codec": "dcvc-uf-htl", "arm": arm, "qp": qp, "scale": scale,
                               "kbps_r": rec["region_bits"] / seconds / 1000, "kbps": rec["bits"] / seconds / 1000,
                               "video_sha256": args.arm_sha256[arm], **{k: v for k, v in rec.items() if k != "decoded"}})
                shutil.rmtree(work / f"{arm}-{scale}-{qp}")
                os.remove(work / "dcvc-rgb.npy")
        if scale > 1:
            source.unlink()
        region_file.unlink()
    if first:
        from src.codecs.quality import load_lpips

        t = score.hand[0]
        out = np.asarray(truth[t]).copy()
        out[regions[t]] = 0
        cpu = load_lpips(Path(args.lpips_backbone), "cpu")
        gpu_net = nets["lpips"]
        import torch

        def lp(net: Any, dev: str) -> float:
            a = torch.from_numpy(out).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1.0
            b = torch.from_numpy(np.asarray(truth[t])).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1.0
            with torch.inference_mode():
                return float((net(a, b)[0, 0] * torch.from_numpy(regions[t]).to(dev)).sum()) / int(regions[t].sum())

        checks["lpips_cpu_vs_device"] = abs(lp(cpu, "cpu") - lp(gpu_net, device))
        checks["dists_identity"] = dists_crops(nets["dists"], [np.asarray(truth[t])], np.asarray(truth[t]), regions[t], device)[0]
    os.remove(decoded["path"])
    shutil.rmtree(work)
    arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64),
              "lpips": np.stack([s["lpips"] for s in scores]), "sse": np.stack([s["sse"] for s in scores]),
              "dists": np.stack([s["dists"] for s in scores])}
    meta = {"unit": f"{item['id']}@{label}", "item": item["id"], "label": label, "video": item["video"], "fps": float(item["fps"]),
            "frames": frames, "seconds_coded": seconds, "hand_frames": len(score.hand), "points": points, "look_frame": look,
            "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "checks": checks,
            "seconds": round(time.time() - began, 2)}
    save_unit(publish, "code", f"{item['id']}@{label}", arrays, meta)


def command_code(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.visor.b2 import select_items
    from src.codecs.quality import load_lpips, lpips_record

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    args.arms = parse_arms(args.arm)
    args.arm_sha256 = {name: file_sha256(Path(path)) for name, path in args.arms}
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.qps = [int(q) for q in args.qps.split(",")]
    args.scales = [int(s) for s in args.scales.split(",")]
    label = "+".join((["anchor"] if args.anchor else []) + [n for n, _ in args.arms])
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "code")
    todo = [it for it in items if f"{it['id']}@{label}" not in restored]
    if args.anchor:
        if not args.inspect:
            raise SystemExit("--anchor needs --inspect (H3b2's low points)")
        copy = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (H1)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.inspect, copy)
        copy.chmod(0o755)
        args.inspect_source, args.inspect = args.inspect, str(copy)
    windows = decode_windows(args, todo, int(args.frames), scratch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("h3c code needs a GPU (DCVC-UF)")
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    done = len(restored)
    began = time.time()
    for number, (item, decoded) in enumerate(windows):
        process_item(args, item, decoded, nets, device, publish, scratch, label, first=number == 0)
        done += 1
        progress(done)
    import torchmetrics
    from torchmetrics.functional.image import dists as dists_module

    from experiments.visor.h3b import _tool

    write_json(stage_dir() / "h3c.json", {
        "command": "code", "label": label, "device": device_record(torch.cuda.get_device_name(0)),
        "on_cuda": {k: next(v.parameters()).is_cuda for k, v in nets.items()}, "seconds": round(time.time() - began, 2),
        "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__,
                  "weights_sha256": file_sha256(Path(dists_module.__file__).parent / "dists_models" / "weights.pt"),
                  "backbone": args.dists_backbone, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "arms": args.arm_sha256},
        "anchor": {"decoder": _tool(args.decoder), "encoder": _tool(args.encoder), "preset": args.preset,
                   "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}} if args.anchor else None,
        "settings": {"frames": int(args.frames), "items": args.items, "qps": args.qps, "scales": args.scales,
                     "decision": DECISION, "dists_pad": DISTS_PAD},
        "items": [it["id"] for it in items], "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation


def validate_code(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    units = load_units([str(stage / "published.tar")], "code")
    dcvc = [p for m, _ in units for p in m["points"] if p["codec"] == "dcvc-uf-htl"]
    first = [m for m, _ in units if "lpips_cpu_vs_device" in m["checks"]]
    hand = [(m, a, a["pixels_r"] > 0) for m, a in units]
    checks = {
        "every_item_present": bool(units) and len(units) == len(result["items"]) and all(m["npz_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "every_point_scored": all(a["lpips"].shape[0] == len(m["points"]) for m, a in units),
        "scores_finite_on_hand_frames": all(np.isfinite(a["lpips"][:, h]).all() and np.isfinite(a["dists"][:, h]).all() for _, a, h in hand),
        "lpips_cpu_matches_device_1e-3": bool(first) and all(m["checks"]["lpips_cpu_vs_device"] < 1e-3 for m in first),
        "dists_of_identical_is_zero": all(abs(m["checks"]["dists_identity"]) < 1e-4 for m in first),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        # The model's estimate is a split, not the rate; still, it should be near the coded bytes.
        "estimate_near_actual_bits": all(0.5 < p["estimate_over_actual"] < 2.0 for p in dcvc),
        "region_rate_inside_stream_rate": all(0.0 < p["kbps_r"] < p["kbps"] for p in dcvc),
        # At each arm and scale, a higher QP costs more bits.
        "rate_rises_with_qp": all(a["kbps"] < b["kbps"] for m, _ in units for a in m["points"] for b in m["points"]
                                  if a["codec"] == b["codec"] == "dcvc-uf-htl" and a["arm"] == b["arm"] and a["scale"] == b["scale"]
                                  and a["qp"] < b["qp"]),
    }
    if result["anchor"]:
        checks["streams_match_records"] = all(p["stream"]["sha256"] == p["stream"]["recorded_sha256"] for m, _ in units
                                              for p in m["points"] if p["codec"] == "svtav1" and p["scale"] == 1 and "kbps_r" not in p)
    return checks


def validate_train(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    import tarfile

    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    return {
        "checkpoint_published": "publish/model/video_ft.pth.tar" in names,
        "trained_every_step": result["curve"][-1]["step"] == result["steps"],
        "monitor_finite": all(math.isfinite(c["loss"]) for c in result["curve"]),
        "model_changed": result["model"]["checkpoint_sha256"] != result["base_checkpoints"]["video_sha256"],
        "intended_gpu": tuple(result["gpu_capability"]) >= (8, 6),
        "loss_mask_has_hands": all(s["hand_share"] > 0 for s in result["data"]["segments"]),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for d in result["data"]["decodes"] for g in d["jpeg_gate"]),
        "component_excludes_pilot_participants": result["fit"] != "component"
        or all(participant(i) not in EXCLUDED for i in result["items"]),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "h3c.json").read_text())
    checks = validate_train(stage, result) if result["command"] == "train" else validate_code(stage, result)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def bd_rate(test: list[dict[str, float]], anchor: list[dict[str, float]], metric: str, levels: int = BD_LEVELS) -> dict[str, Any]:
    """Mean log-rate difference of ``test``'s front against ``anchor``'s over the ``metric`` range both cover (lower is
    better), linear in log rate; as a ratio minus 1. None where the ranges do not overlap."""
    ft, fa = pareto_front(test, metric), pareto_front(anchor, metric)
    lo = max(min(p[metric] for p in ft), min(p[metric] for p in fa))
    hi = min(max(p[metric] for p in ft), max(p[metric] for p in fa))
    if not lo < hi:
        better = max(p[metric] for p in ft) < min(p[metric] for p in fa)
        return {"bd": None, "where": "test better than every anchor point" if better else "test worse than every anchor point"}
    diffs = []
    for k in range(levels):
        value = lo + (hi - lo) * (k + 0.5) / levels
        rt, ra = rate_on_front(ft, metric, value), rate_on_front(fa, metric, value)
        assert rt is not None and ra is not None
        diffs.append(math.log(rt) - math.log(ra))
    return {"bd": math.expm1(float(np.mean(diffs))), "where": "overlap", "range": [lo, hi]}


def point_summary(a: dict[str, np.ndarray], j: int, hand: np.ndarray) -> dict[str, float]:
    return {"lpips": float(np.nanmean(a["lpips"][j][hand])), "dists": float(np.nanmean(a["dists"][j][hand])),
            "psnr": float(np.mean([psnr_of(s, p) for s, p in zip(a["sse"][j][hand], a["pixels_r"][hand])]))}


def command_report(args: argparse.Namespace) -> int:
    units = load_units(args.code, "code")
    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    h3b: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
    for path in args.h3b:  # H3b's and H3b2's units share their names: one archive at a time
        for m, a in load_units([path], "oracle"):
            h3b.setdefault(m["unit"], []).append((m, a))
    items: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cross: dict[str, Any] = {}
    for meta, a in units:
        hand = a["pixels_r"] > 0
        by_arm = items.setdefault(meta["item"], {})
        rates = hand_kbps(h1[meta["item"]])
        for j, p in enumerate(meta["points"]):
            kbps_r = p.get("kbps_r", rates.get(f"{p.get('crf', 0):g}") if p["codec"] == "svtav1" else None)
            row = {"name": p["name"], "kbps": float(kbps_r), "kbps_frame": p.get("kbps"), **point_summary(a, j, hand)}
            row["neg_psnr"] = -row["psnr"]
            by_arm.setdefault(p["arm"], []).append(row)
            # The anchor is H3b's: the same LPIPS on R frame by frame, the same low streams.
            for hm, ha in h3b.get(meta["item"], []):
                if p["name"] in hm["variants"] and p["name"].startswith("svt@"):
                    k = [v for v in hm["variants"] if v.startswith("svt@")].index(p["name"])
                    gap = np.abs(ha["svt_lpips_r"][k][hand] - a["lpips"][j][hand])
                    cross.setdefault(meta["item"], {})[p["name"]] = float(np.nanmax(gap))
                for low in hm.get("low_streams", []):
                    if p["codec"] == "svtav1" and p.get("scale") == low["scale"] and p.get("crf") == low["crf"]:
                        cross.setdefault(meta["item"], {})[f"{p['name']}:sha_matches_h3b2"] = low["sha256"] == p["stream"]["sha256"]
    arms = sorted({arm for by_arm in items.values() for arm in by_arm} - {"svtav1"})
    report: dict[str, Any] = {"decision_rule": DECISION, "anchor_cross_check": cross, "arms": {},
                              "runs": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.code]}
    for arm in arms:
        per_item = {}
        for item, by_arm in sorted(items.items()):
            if arm not in by_arm or "svtav1" not in by_arm:
                continue
            test, anchor = by_arm[arm], by_arm["svtav1"]
            entry: dict[str, Any] = {m: bd_rate(test, anchor, m) for m in ("lpips", "dists", "neg_psnr")}
            entry["lpips_frame_rate"] = bd_rate([{**p, "kbps": p["kbps_frame"]} for p in test],
                                                [{**p, "kbps": p["kbps_frame"]} for p in anchor if p["kbps_frame"] is not None], "lpips")
            if "reference" in by_arm and arm != "reference":
                entry["lpips_vs_reference"] = bd_rate(test, by_arm["reference"], "lpips")
            entry["front"] = pareto_front(test, "lpips")
            entry["anchor_front"] = pareto_front(anchor, "lpips")
            per_item[item] = entry
        values = [e["lpips"]["bd"] for e in per_item.values()]
        known = [v for v in values if v is not None]
        summary: dict[str, Any] = {"items": per_item, "scored": len(per_item), "unclear_items": len(values) - len(known),
                                   "items_below_zero": sum(v < 0 for v in known)}
        if known:
            summary["mean_bd"] = float(np.mean(known))
            summary["mean_bd_ci"] = bootstrap_mean(np.array(known), np.random.default_rng(seed_of(f"{SEED}:{arm}")), RESAMPLES)
        ref = [e["lpips_vs_reference"]["bd"] for e in per_item.values() if e.get("lpips_vs_reference", {}).get("bd") is not None]
        summary["mean_bd_vs_reference"] = float(np.mean(ref)) if ref else None
        report["arms"][arm] = summary
    report["decision"] = decide(report["arms"])
    out = Path(args.out or stage_dir())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "h3c-report.json", report)
    print(json.dumps({"decision": report["decision"], "arms": {a: {k: v for k, v in s.items() if k != "items"}
                                                               for a, s in report["arms"].items()}}, indent=1, default=str))
    return 0


def arm_verdict(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary or summary["scored"] < len(PILOT) or summary["unclear_items"]:
        return {"verdict": "incomplete" if not summary or summary["scored"] < len(PILOT) else "unclear (an item without overlap)"}
    mean, below = summary["mean_bd"], summary["items_below_zero"]
    return {"mean_bd": mean, "items_below_zero": below,
            "beats_svtav1": mean < DECISION["component_adopted_mean_bd"] and below >= DECISION["items_below_zero"],
            "upper_passes": mean < DECISION["upper_bound_passes_mean_bd"] and below >= DECISION["items_below_zero"],
            "upper_fails": mean >= DECISION["upper_bound_fails_mean_bd"]}


def decide(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Stage 1: the upper bound (arms ``upper-mse`` and ``upper-lpips``); stage 2: the component (``component-*``)."""
    upper = {a: arm_verdict(arms.get(a)) for a in ("upper-mse", "upper-lpips")}
    out: dict[str, Any] = {"upper": upper, "reference": arm_verdict(arms.get("reference"))}
    if all("upper_passes" in v for v in upper.values()):
        if any(v["upper_passes"] for v in upper.values()):
            out["upper_bound"] = "passes"
        elif all(v["upper_fails"] for v in upper.values()):
            out["upper_bound"] = "fails"
        else:
            out["upper_bound"] = "unclear"
        out["better_beta"] = min(upper, key=lambda a: upper[a]["mean_bd"]).split("-")[1]
    components = {a: arm_verdict(s) for a, s in arms.items() if a.startswith("component")}
    for a, v in components.items():
        if "beats_svtav1" in v:
            ref = arms[a].get("mean_bd_vs_reference")
            v["vs_reference"] = ref
            v["adopted"] = bool(v["beats_svtav1"] and ref is not None and ref < DECISION["component_vs_reference_bd"])
    out["component"] = components
    return out


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3c")
    commands = parser.add_subparsers(dest="command", required=True)

    def windows(p: argparse.ArgumentParser) -> None:
        p.add_argument("--eval-set", required=True)
        p.add_argument("--archive", required=True)
        p.add_argument("--masks", required=True)
        p.add_argument("--mask-record", required=True)
        p.add_argument("--video", action="append", default=[])
        p.add_argument("--items", required=True)
        p.add_argument("--frames", default="240")
        p.add_argument("--image-ckpt", required=True)
        p.add_argument("--lpips-backbone", required=True)

    train = commands.add_parser("train")
    windows(train)
    train.add_argument("--fit", choices=("upper", "component"), required=True)
    train.add_argument("--segments", default="all", help="all, or the N segments with most hand pixels per item and scale")
    train.add_argument("--steps", required=True)
    train.add_argument("--lpips-weight", default="0")
    train.add_argument("--video-ckpt", required=True)
    train.add_argument("--timeout", type=float, default=3000)
    code = commands.add_parser("code")
    windows(code)
    code.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "CKPT"),
                      help="an arm and its HT-L checkpoint (reference: the published one)")
    code.add_argument("--dists-backbone", required=True)
    code.add_argument("--qps", default=",".join(map(str, QPS)))
    code.add_argument("--scales", default=",".join(map(str, SCALES)))
    code.add_argument("--anchor", action="store_true", help="also SVT-AV1: B2's streams and H3b2's low points")
    code.add_argument("--svt-streams", action="append", default=[])
    code.add_argument("--svt-result", action="append", default=[])
    code.add_argument("--inspect", default="")
    code.add_argument("--decoder", default="dav1d")
    code.add_argument("--encoder", default="SvtAv1EncApp")
    code.add_argument("--preset", default="4")
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--code", action="append", required=True)
    report.add_argument("--h1-report", required=True)
    report.add_argument("--h3b", action="append", default=[], help="H3b's and H3b2's published.tar (anchor cross-check)")
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    return {"train": command_train, "code": command_code, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
