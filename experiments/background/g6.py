"""PLAN step G6: the background in the first working PointStream, per window of evaluation set v2.

    python -m experiments.background.g6 footage --eval-set E.JSON --video P.MP4 --item ID --set scene|native \\
        --hand-objects DIR --sam-checkpoint P --propainter-tree DIR --propainter-ckpt P --raft-ckpt P \\
        --flow-completion-ckpt P --limit-frames N
    python -m experiments.background.g6 train --eval-set E.JSON --item ID --footage DIR --image-ckpt P \\
        --video-ckpt P --lpips-backbone P --steps N --limit-frames N
    python -m experiments.background.g6 format WINDOW... --methods scene-540,scene-1080 --arm DIR --publish
    python -m experiments.background.g6 enhance WINDOW... --footage DIR --crfs 32,40,48 --steps N --publish
    python -m experiments.background.g6 validate

WINDOW is ``--eval-set E.JSON --archive DIR --masks DIR --video P.MP4 --item ID --image-ckpt P --video-ckpt P
--lpips-backbone P --dists-backbone P --inception P --propainter-tree DIR --propainter-ckpt P --raft-ckpt P
--flow-completion-ckpt P --limit-frames N``.

The window is decoded as B2 does (`experiments.visor.b2.decode_window`, its JPEG gate checked) and its
foreground F is B1b's mask set (`visor_dense_sam_fill`, the union of its instances), as agreed with the
foreground session (PLAN G6): F is filled by ProPainter at 960×540, upsampled into F only (``--dilate-px 0``;
steps 2-3 dilated by 16 px as G1), and the whole filled frame is coded. Methods:

* ``svt-1080`` / ``svt-540``: SVT-AV1 (G5's preset and CRFs) at 1080p, or at 960×540 and upsampled (bicubic);
* ``dcvc-1080``: DCVC-UF HT-L off the shelf at 1080p;
* ``scene-1080`` / ``scene-540``: the window's scene model (``--arm``: a ``train`` archive, or a G5c finetune
  archive of its stretch) at 1080p as trained, or at 960×540 and upsampled (G6 step 2's choice);
* ``enh-1080`` (``enhance``): SVT-AV1 at 1080p, then G5d's enhancer fitted on the window's own video outside
  the window (``footage --set native``: pairs at the window's size and rate, MSE + LPIPS on V).

The scene model's footage (``footage --set scene``) is G5b's split of a stretch applied to the window: the
120 s of its own video around it, less the window and 5 s on each side, at 10 frames/s and 960×540, with
G1's foreground (SAM 3.1 "hand" and "arm" plus the EPIC detector's boxes, dilated) filled by ProPainter;
``train`` fits recipe L on it (β 0.005, learning rate 3e-5, 540 steps; G5c step 4).

Every point is scored at 1080p on the background (not F), with `g5c.full_score` (PSNR, LPIPS, DISTS, flicker
and KID on V), and with ``--publish`` published for the foreground session's compositor: the decoded 1080p
frames (FFV1, 4:2:0, BT.709 full range), the bitstream, and a manifest (`publish_window`). Each finished
point is checkpointed with its files. Writes ``g6.json`` in ``PS_STAGE_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5c
from experiments.background.camera import DILATE, TO_1080
from experiments.background.g5b import GUARD_S
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

METHODS = ("svt-1080", "svt-540", "dcvc-1080", "scene-1080", "scene-540")
FULL = (1920, 1080)
HALF = (g5.WIDTH, g5.HEIGHT)
MASK_SET = "visor_dense_sam_fill"
#: H5's rate grid (about 10 kbps to 2 Mbps, log-spaced; spans from G6 steps 2-3): DCVC-UF's QP raises the rate.
GRID: dict[str, tuple[int, ...]] = {"svt-1080": g5.CRFS, "svt-540": g5.CRFS, "dcvc-1080": g5.DCVC_QPS,
                                    "scene-540": (0, 13, 25, 38, 50, 63), "scene-1080": (50, 63)}
STRETCH_S = 120.0  # G1d's stretches: the window's scene footage lies in the 120 s around it
NATIVE_SEGMENTS, NATIVE_SEGMENT_S = 4, 2.0  # the enhancer's pairs, at the window's size and rate
RECIPE = {"lpips_weight": 0.005, "lr": 3e-5, "steps": 540}  # recipe L (G5c step 4)
# The footage's foreground. g1: G1's (SAM 3.1 hand/arm plus the detector's hand and object boxes, dilated 16 px
# at 1080p). exact: shaped like the window's F, which is filled undilated: no dilation and no hand boxes (a box
# covers the band beside the hand that exact F leaves in V); object boxes stay, standing in for VISOR's objects.
FOOTAGE_MASKS: dict[str, dict[str, Any]] = {"g1": {"dilate_px": 8, "box_kinds": ("hands", "objects")},
                                            "exact": {"dilate_px": 0, "box_kinds": ("objects",)}}
YUV = {"pix_fmt": "yuv420p", "matrix": "bt709", "range": "full", "chroma": "2x2 mean", "size": list(FULL)}
ROUTES = {"svt": "anchor", "dcvc": "dcvc-uf", "scene": "scene-model", "enh": "enhancer"}
JOB_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")


def parse_method(name: str) -> tuple[str, tuple[int, int]]:
    """(codec, coded size): ``svt``, ``dcvc`` or ``scene``, at 1080p or 960×540."""
    codec, _, size = name.partition("-")
    if name not in METHODS:
        raise ValueError(f"method {name!r}: one of {METHODS}")
    return codec, FULL if size == "1080" else HALF


def parse_values(text: str | None) -> dict[str, tuple[int, ...]]:
    """``METHOD=V,V;METHOD=V`` over `GRID`."""
    out = dict(GRID)
    for part in filter(None, (text or "").split(";")):
        name, _, values = part.partition("=")
        parse_method(name)
        out[name] = tuple(int(v) for v in values.split(","))
    return out


def resize(frames: np.ndarray, size: tuple[int, int], *, area: bool) -> np.ndarray:
    """(n, h, w, c) uint8 or (n, h, w) bool to ``size`` (w, h): area average down, bicubic up; masks by any."""
    import cv2

    if frames.dtype == bool:
        out = np.stack([cv2.resize(f.astype(np.float32), size, interpolation=cv2.INTER_AREA) for f in frames])
        return out > 0
    flag = cv2.INTER_AREA if area else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, size, interpolation=flag) for f in frames])


def dilate_1080(mask: np.ndarray, px: int | None = None) -> np.ndarray:
    """A dilation at 1080p by ``px`` (G1's ``DILATE`` analysis pixels, 16 at 1080p, by default)."""
    import cv2

    size = 2 * (round(DILATE * TO_1080) if px is None else px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return np.stack([cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in mask])


def to_rgb(planes: np.ndarray, full_range: bool) -> np.ndarray:
    """(n, h·3/2, w) 4:2:0 planes in the source's range to RGB, as B2 scores them."""
    import torch

    from src.codecs import quality

    height, width = planes.shape[1] * 2 // 3, planes.shape[2]
    return np.concatenate([quality.yuv420_to_rgb(torch.from_numpy(np.ascontiguousarray(planes[i:i + 16])), width,
                                                 height, full_range=full_range)
                           .permute(0, 2, 3, 1).to(torch.uint8).numpy() for i in range(0, len(planes), 16)])


def load_item(eval_set: str, item_id: str) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {i["id"]: i for i in json.loads(Path(eval_set).read_text())["items"]}
    return items[item_id]


# ----------------------------------------------------------------- the window's own footage


def scene_parts(item: dict[str, Any], total: int, fps: float) -> list[tuple[int, int]]:
    """The window's scene footage, as G5b split a stretch: the `STRETCH_S` around the window in its own video
    (shifted to stay `g5c.EDGE_S` from either end), less the window and `GUARD_S` on each side; (first, last)
    source indices of the parts at least 1 s long."""
    edge, length = round(g5c.EDGE_S * fps), round(STRETCH_S * fps)
    first = int(item["first_video_index"])
    last = first + int(item["frames"]) - 1
    lo = max(edge, (first + last) // 2 - length // 2)
    hi = min(total - 1 - edge, lo + length - 1)
    lo = max(edge, hi - length + 1)
    guard = round(GUARD_S * fps)
    return [(a, b) for a, b in ((lo, first - guard - 1), (last + guard + 1, hi)) if b - a + 1 >= round(fps)]


def native_segments(parts: list[tuple[int, int]], fps: float, count: int = NATIVE_SEGMENTS,
                    seconds: float = NATIVE_SEGMENT_S) -> list[tuple[int, int]]:
    """``count`` segments of ``seconds`` at the native rate inside ``parts``, their starts spread evenly over
    the parts' possible starts (at (k + 1/2) / count); fixed before any run and blind to content."""
    length = round(seconds * fps)
    rooms = [(a, b - a + 2 - length) for a, b in parts if b - a + 1 >= length]
    total = sum(r for _, r in rooms)
    out: list[tuple[int, int]] = []
    for k in range(count if total > 0 else 0):
        at = int((k + 0.5) / count * total)
        for a, room in rooms:
            if at < room:
                out.append((a + at, a + at + length - 1))
                break
            at -= room
    return out


def decode_planes(video: Path, indices: list[int], threads: int = 8) -> tuple[np.ndarray, bool]:
    """The source's 4:2:0 planes at ``indices`` (B2's decode) and whether they are full range."""
    from src.segmentation import visor

    wanted, formats, ranges = set(indices), set(), set()
    planes: list[np.ndarray] = []
    for index, frame in visor.decoded_frames(video, indices, threads=threads):
        if index not in wanted:
            continue
        if index != indices[len(planes)]:
            raise RuntimeError(f"{video}: decoded frame {index}, expected {indices[len(planes)]}")
        formats.add(frame.format.name)
        ranges.add(str(getattr(frame, "color_range", None)))
        planes.append(np.ascontiguousarray(frame.to_ndarray()))
    if len(planes) != len(indices) or not formats <= {"yuv420p", "yuvj420p"}:
        raise RuntimeError(f"{video}: decoded {len(planes)} of {len(indices)} frames as {sorted(formats)}")
    return np.stack(planes), formats == {"yuvj420p"} or ranges in ({"2"}, {"JPEG"})


def decode_planes_task(task: dict[str, Any]) -> bool:
    """`decode_planes` in a spawned process, its planes saved to ``task["out"]``: PyAV's threaded H.264
    decoder has deadlocked on close inside the process that holds SAM 3.1 (P26_02, twice)."""
    planes, full_range = decode_planes(Path(task["video"]), task["indices"])
    np.save(task["out"], planes)
    return full_range


def command_footage(args: argparse.Namespace) -> int:
    """The window's own footage outside it: ``scene``, at 10 frames/s and 960×540 (the scene model's
    training set); ``native``, short segments at the window's rate and 1080p (the enhancer's pairs)."""
    import gc
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    from experiments.background import g1
    from src.segmentation.sam31 import Sam31SequenceSegmenter

    item = load_item(args.eval_set, args.item)
    if Path(args.video).stem != item["video"]:
        raise SystemExit(f"{args.video}: not {item['video']}")
    fps = float(Fraction(item["fps"]).limit_denominator(1001))
    _, total = g5c.video_info(Path(args.video))
    parts = scene_parts(item, total, fps)
    native = args.set == "native"
    ranges = native_segments(parts, fps) if native else parts
    if not ranges:
        raise SystemExit(f"{item['id']}: no footage outside the window in {total} frames")
    if args.limit_frames > 0:  # smoke: the first two pieces, shortened
        step = 1.0 if native else fps / g5c.FOOTAGE_FPS
        ranges = [(a, a + int(round(args.limit_frames * step)) - 1) for a, _ in ranges[:2]]
    analysis: dict[str, Any] = {"mode": "native"} if native else {"mode": "rate", "fps": g5c.FOOTAGE_FPS}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "footage"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    segmenter = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
    sam = segmenter.provenance("offline_bidirectional")
    frames_all, keep_all, big_all, records = [], [], [], []
    full_ranges = set()
    ctx = multiprocessing.get_context("spawn")
    for n, (a, b) in enumerate(ranges):
        clip = g1.plan({"id": f"g6/{item['id']}/footage/{a}", "video": item["video"], "dataset": "visor",
                        "source": {"kind": "video"}, "fps": fps, "first_index": a, "last_index": b,
                        "analysis": analysis, "masks": {"kind": "sam_text", "prompts": {"hand": "hand", "arm": "arm"},
                                                        "boxes": "epic_hand_objects"}}, 0)
        seg_work = work / f"seg{n}"
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            decoded = pool.submit(g1.decode_clip, {"plan": clip, "out": str(seg_work), "threads": 8,
                                                   "path": args.video, "jpeg_dir": str(seg_work / "jpeg")}).result()
            if native:
                full_ranges.add(pool.submit(decode_planes_task, {"video": args.video, "indices": clip["indices"],
                                                                 "out": str(seg_work / "planes.npy")}).result(600))
                big_all.append(np.load(seg_work / "planes.npy"))
        t0 = time.time()
        masks = g1.sam_text_masks(segmenter, clip, seg_work / "jpeg")
        sam_seconds = time.time() - t0
        shutil.rmtree(seg_work / "jpeg", ignore_errors=True)
        base = [g1.to_analysis(g1.union(f, (masks.height, masks.width))) for f in masks.frames]
        fg_record = g1.foreground(clip, seg_work, base, None, args.hand_objects, **FOOTAGE_MASKS[args.mask])
        frames_all.append(np.load(seg_work / "frames.npy"))
        keep_all.append(~np.load(seg_work / "foreground.npy"))
        records.append({"first_index": a, "last_index": b, "frames": len(clip["indices"]), "decode": decoded,
                        "sam_seconds": round(sam_seconds, 1), "foreground": fg_record,
                        "fg_share": float(1 - keep_all[-1].mean())})
        progress(n + 1)
    del segmenter  # SAM 3.1 holds about 30 GiB that ProPainter, in its own process, needs
    gc.collect()
    torch.cuda.empty_cache()
    frames, keep = np.concatenate(frames_all), np.concatenate(keep_all)
    del frames_all
    offsets = np.cumsum([0] + [len(k) for k in keep_all])
    chunks = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        chunks += [[int(lo + a), int(hi + a), int(c + a), int(d + a)]
                   for lo, hi, c, d in g5c.fill_chunks(int(b - a), 0, 0) if c < d]
    filled, propainter = g5c.run_propainter(frames, ~keep, chunks, args, work / "propainter")
    del frames
    publish = scratch / "publish" / "footage"
    publish.mkdir(parents=True)
    if native:  # 1080p: ProPainter's fill upsampled into the (2× nearest) holes only, as the window's
        if len(full_ranges) != 1:
            raise RuntimeError(f"{args.video}: mixed ranges {full_ranges}")
        yuv = np.lib.format.open_memmap(publish / "filled_yuv420.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(keep), FULL[1] * 3 // 2, FULL[0]))
        big_keep = np.lib.format.open_memmap(publish / "keep.npy", mode="w+", dtype=bool,
                                             shape=(len(keep), FULL[1], FULL[0]))
        for planes, a in zip(big_all, offsets[:-1]):
            for i in range(0, len(planes), 16):
                at = slice(int(a) + i, int(a) + min(i + 16, len(planes)))
                holes = np.repeat(np.repeat(~keep[at], 2, axis=1), 2, axis=2)
                rgb = np.where(holes[..., None], resize(filled[at], FULL, area=False),
                               to_rgb(planes[i:i + 16], full_ranges == {True}))
                yuv[at] = g5.rgb_to_yuv420(rgb)
                big_keep[at] = ~holes
        yuv.flush()
        big_keep.flush()
        del yuv, big_keep
    else:
        np.save(publish / "filled_yuv420.npy", g5.rgb_to_yuv420(filled))
        np.save(publish / "keep.npy", keep)
    for rec, a, b in zip(records, offsets[:-1], offsets[1:]):
        rec.update(offset=int(a), end=int(b))
    meta = {"target": item["id"], "video": item["video"], "set": args.set, "size": list(FULL if native else HALF),
            "fps": fps if native else g5c.FOOTAGE_FPS, "source_fps": fps, "video_frames": total,
            "parts": [list(p) for p in parts],
            "window": [int(item["first_video_index"]), int(item["first_video_index"]) + int(item["frames"]) - 1],
            "guard_s": GUARD_S, "stretch_s": STRETCH_S, "recordings": [item["video"]],
            "source_full_range": sorted(full_ranges) if native else None,
            "segments": [{"video": item["video"], **r} for r in records], "frames": int(len(keep)),
            "mask": {"name": args.mask, **FOOTAGE_MASKS[args.mask]}, "fill": "propainter", "yuv420_sha256": file_sha256(publish / "filled_yuv420.npy"),
            "keep_sha256": file_sha256(publish / "keep.npy"),
            "sam": {"checkpoint_sha256": sam.checkpoint_sha256, "model_revision": sam.model_revision},
            "propainter": propainter}
    write_json(publish / "meta.json", meta)
    write_json(stage_dir() / "g6.json", {"kind": "footage", "limit_frames": args.limit_frames, **meta,
                                         "seconds": round(time.time() - began, 1), **g5c.gpu_record()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- the scene model (recipe L on the footage)


def command_train(args: argparse.Namespace) -> int:
    import torch

    from experiments.background import g5b

    item = load_item(args.eval_set, args.item)
    [(meta, yuv_path, keep_path)] = g5b.load_footage([args.footage])
    if meta["target"] != item["id"] or meta["set"] != "scene":
        raise SystemExit(f"{args.footage}: {meta['set']} footage of {meta['target']}, not scene footage of {item['id']}")
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "train"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    f, k = np.load(yuv_path), np.load(keep_path)
    yuv, mask, fit, at = [], [], [], 0  # a masked blank frame before each segment, as G5c's footage
    for seg in meta["segments"]:
        yuv += [np.zeros_like(f[:1]), f[seg["offset"]:seg["end"]]]
        mask += [np.zeros_like(k[:1]), k[seg["offset"]:seg["end"]]]
        fit += list(range(at + 1, at + 1 + seg["end"] - seg["offset"]))
        at += 1 + seg["end"] - seg["offset"]
    np.save(work / "frames.npy", np.concatenate(yuv))
    np.save(work / "keep.npy", np.concatenate(mask))
    del f, k, yuv, mask
    if args.limit_frames > 0:  # smoke: a few dozen fitted frames
        fit = fit[:max(args.limit_frames, 24)]
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    if args.resume and args.limit_frames <= 0 and checkpoints is not None and not (checkpoints / "state.pt").is_file():
        checkpoints.mkdir(parents=True, exist_ok=True)  # a timed-out run's saved step, full stage only
        shutil.copyfile(Path(args.resume) / "checkpoint" / "state.pt", checkpoints / "state.pt")
    plan = {**g5b.TRAIN, "frames": str(work / "frames.npy"), "keep": str(work / "keep.npy"), "fit": fit,
            "lr": args.lr, "frame_weights": None, "steps": args.steps, "monitor_every": max(1, args.steps // 10),
            "save_every": max(1, args.steps // 10), "lpips_weight": args.lpips_weight,
            "lpips_backbone": args.lpips_backbone, "dists_weight": 0.0, "dists_backbone": None,
            "temporal_weight": 0.0, "dists_every": 1, "image_ckpt": args.image_ckpt,
            "image_sha256": file_sha256(Path(args.image_ckpt)), "video_ckpt": args.video_ckpt,
            "video_sha256": file_sha256(Path(args.video_ckpt)), "out": str(work / "model"),
            "resume_dir": str(checkpoints) if checkpoints else None}
    if args.limit_frames > 0:  # smoke: short sequences, small patches
        plan.update(groups=1, patch=[256, 256], monitor_batch=1, log_every=1)
    write_json(work / "plan.json", plan)
    command, env, cwd = g5b.trainer_command(Path(sys.prefix), work / "plan.json", work / "trained.json")
    log_path = work / "train.log"
    with log_path.open("w") as log:
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              timeout=args.timeout)
    log_text = log_path.read_text()
    if done.returncode:
        raise RuntimeError(f"fine-tuning failed ({done.returncode}):\n{log_text[-4000:]}")
    trained = json.loads((work / "trained.json").read_text())
    publish = scratch / "publish" / "finetune"
    publish.mkdir(parents=True)
    shutil.copyfile(trained["checkpoint"], publish / "video_ft.pth.tar")
    shutil.copyfile(log_path, publish / "train.log")
    row = {"kind": "train", "arm": f"window-footage-b{args.lpips_weight:g}-lr{args.lr:g}-{meta['fill']}",
           "id": f"g6-window/{item['video']}", "item": item["id"], "fit": "own-video", "fill": meta["fill"],
           "fit_frames": len(fit), "footage": {k: meta[k] for k in ("parts", "window", "guard_s", "yuv420_sha256",
                                                                     "keep_sha256", "frames")},
           "steps": args.steps, "limit_frames": args.limit_frames, "resumed_archive": args.resume,
           "train": {**g5b.TRAIN, **{k: plan[k] for k in ("groups", "patch", "monitor_batch", "monitor_every", "lr")},
                     "lpips_weight": args.lpips_weight},
           "model": {k: trained[k] for k in ("checkpoint_sha256", "checkpoint_bytes", "parameters")},
           "base_checkpoints": {"image_sha256": plan["image_sha256"], "video_sha256": plan["video_sha256"]},
           "curve": trained["curve"], "converged": g5b.monitor_converged(trained["curve"]),
           "train_seconds": trained["train_seconds"], "seconds_per_step": trained["seconds_per_step"],
           "peak_memory_gib": trained["peak_memory_gib"], "skipped_nonfinite": trained["skipped_nonfinite"],
           "seconds": round(time.time() - began, 1), "gpu": torch.cuda.get_device_name(0),
           "gpu_capability": list(torch.cuda.get_device_capability(0)), "trainer_gpu": trained["gpu"]}
    if file_sha256(publish / "video_ft.pth.tar") != row["model"]["checkpoint_sha256"]:
        raise RuntimeError("the published checkpoint is not the trained one")
    write_json(publish / "result.json", row)
    write_json(stage_dir() / "g6.json", row)
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- the window, coded and published


def load_window(args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """The item, its RGB frames at 1080p (B2's decode, in the source's range) and B1b's foreground F."""
    from experiments.visor import b2

    item = load_item(args.eval_set, args.item)
    frames = args.limit_frames if args.limit_frames > 0 else int(item["frames"])
    record = b2.decode_window(Path(args.video), item, frames, Path(args.archive), work / "window.yuv", 8)
    if (record["width"], record["height"]) != FULL:
        raise SystemExit(f"{args.item}: {record['width']}x{record['height']}, not 1080p")
    planes = np.fromfile(work / "window.yuv", np.uint8).reshape(frames, FULL[1] * 3 // 2, FULL[0])
    rgb = to_rgb(planes, record["full_range"])
    (work / "window.yuv").unlink()
    clips, info = b2.load_mask_sets(item, frames, {MASK_SET: args.masks}, {}, Path(args.archive))
    fg = np.stack([clips[MASK_SET].foreground(i) for i in range(frames)])
    if fg.shape[1:] != (FULL[1], FULL[0]):
        raise SystemExit(f"{args.item}: masks {fg.shape[1:]}, not 1080p")
    return item, rgb, fg, {"window": {k: v for k, v in record.items() if k != "jpeg_gate"},
                           "jpeg_gate_holds": all(c["holds"] for c in record["jpeg_gate"]), "masks": info[MASK_SET]}


def fill(rgb: np.ndarray, holes: np.ndarray, args: argparse.Namespace, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """ProPainter's fill at 960×540, upsampled into the 1080p holes only."""
    small, small_holes = resize(rgb, HALF, area=True), resize(holes, HALF, area=True)
    filled, record = g5c.run_propainter(small, small_holes, [[0, len(rgb), 0, len(rgb)]], args, work / "propainter")
    out = np.where(holes[..., None], resize(filled, FULL, area=False), rgb)
    return out, record


def prepare_window(args: argparse.Namespace, work: Path) -> dict[str, Any]:
    """The window, F (dilated by ``--dilate-px``), the filled frames, the clip to score and the metrics."""
    item, rgb, fg, source_record = load_window(args, work)
    holes = dilate_1080(fg, args.dilate_px) if args.dilate_px else fg
    filled, propainter = fill(rgb, holes, args, work)
    n = len(rgb)
    fps = Fraction(item["fps"]).limit_denominator(1001)
    clip = g5.Clip(item["id"], "window", fps, 0, rgb, ~holes,
                   {"dataset": (list(range(n)), ~fg), "own": (list(range(n)), ~fg)},
                   {"id": item["id"], "video": item["video"], "fps": str(fps), "frames": n})
    foreground = {"mask_set": MASK_SET, "dilate_px_1080": args.dilate_px, "share": float(fg.mean()),
                  "hole_share": float(holes.mean()),
                  "f_packbits_sha256": hashlib.sha256(np.packbits(fg).tobytes()).hexdigest(),
                  **{k: source_record["masks"][k] for k in ("masks_rle_sha256", "mask_sha256")}}
    return {"item": item, "filled": filled, "clip": clip, "fps": fps, "nets": g5c.load_nets(args),
            "flows": g5c.source_flows(clip), "source": source_record, "propainter": propainter,
            "foreground": foreground}


def write_ffv1(yuv: np.ndarray, path: Path, fps: Fraction, threads: int = 8) -> list[str]:
    """4:2:0 frames, losslessly (FFV1, G1d's options); returns each frame's sha256 (of its planes)."""
    import av

    from experiments.background.g1d import FFV1_OPTIONS

    digests = []
    with av.open(str(path), "w", format="matroska") as out:
        stream = out.add_stream("ffv1", rate=fps)
        stream.width, stream.height = yuv.shape[2], yuv.shape[1] * 2 // 3
        stream.pix_fmt = "yuv420p"
        stream.options = dict(FFV1_OPTIONS)
        stream.thread_count = threads
        for planes in yuv:
            planes = np.ascontiguousarray(planes)
            digests.append(hashlib.sha256(planes.tobytes()).hexdigest())
            for packet in stream.encode(av.VideoFrame.from_ndarray(planes, format="yuv420p")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return digests


def ffv1_digests(path: Path, threads: int = 8) -> list[str]:
    """Each decoded frame's sha256 (of its 4:2:0 planes)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = threads
        return [hashlib.sha256(np.ascontiguousarray(f.to_ndarray(format="yuv420p")).tobytes()).hexdigest()
                for f in container.decode(stream)]


def provenance() -> dict[str, Any]:
    """This fleet job's id and the code revision its ``ready.json`` records, when they can be found."""
    found: dict[str, Any] = {"job_id": None, "git_head": None}
    for var in ("PS_JOB_DIR", "PS_STAGE_DIR", "PS_SCRATCH_DIR"):
        value = os.environ.get(var)
        if not value:
            continue
        path = Path(value)
        found["job_id"] = found["job_id"] or next((p for p in path.parts if JOB_ID.fullmatch(p)), None)
        for d in [path, *path.parents][:4]:
            if found["git_head"] is None and (d / "ready.json").is_file():
                found["git_head"] = json.loads((d / "ready.json").read_text()).get("git_head")
    return found


def code_point(codec: str, size: tuple[int, int], value: int, filled: np.ndarray, clip: g5.Clip,
               args: argparse.Namespace, work: Path, video_ckpt: str | None = None
               ) -> tuple[np.ndarray, dict[str, Any], Path]:
    """One rate point: the filled frames coded at ``size`` and decoded, as the client shows them (1080p
    4:2:0 in `g5.rgb_to_yuv420`'s convention); its record and its bitstream (in ``work``)."""
    from src.codecs import svtav1

    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.yuv"
    frames = filled if size == FULL else resize(filled, size, area=True)
    g5.rgb_to_yuv420(frames).tofile(source)
    n = len(frames)
    if codec == "svt":
        rec = svtav1.code(source, work, width=size[0], height=size[1], fps=clip.fps, frames=n, crf=value,
                          preset=g5.PRESET, threads=8, full_range=True, encoder="SvtAv1EncApp", decoder="dav1d")
        nbytes, stream = int(rec["payload_bytes"]), Path(rec["stream"])
        extra = {"stream_sha256": rec["stream_sha256"], "file_bytes": rec["file_bytes"],
                 "decode_ms_per_frame": 1000 * rec["decode_seconds"] / n, "decoder": "dav1d (CPU, 8 threads)"}
    else:
        coder = argparse.Namespace(image_ckpt=args.image_ckpt, video_ckpt=video_ckpt or args.video_ckpt)
        try:
            rec, retried = g5.code_dcvc(coder, source, n, value, work / "dcvc", width=size[0], height=size[1]), None
        except RuntimeError as error:  # the decoder has segfaulted after its own encode (G5c step 1): once more
            shutil.rmtree(work / "dcvc", ignore_errors=True)
            rec, retried = g5.code_dcvc(coder, source, n, value, work / "dcvc", width=size[0], height=size[1]), \
                str(error)[:300]
        nbytes, stream = int(rec["bytes"]), work / "dcvc" / "stream.psdc"
        extra = {**{k: rec[k] for k in ("deterministic", "decoder_matches_encoder_intra", "decode_seconds")},
                 "decode_ms_per_frame": 1000 * min(rec["decode_seconds"]) / n, "decoder": "DCVC-UF (GPU)",
                 "retried_after": retried}
    raw = np.fromfile(Path(rec["decoded"]), np.uint8).reshape(n, size[1] * 3 // 2, size[0])
    upsample = None
    if size != FULL:  # the client's bicubic upsampling, then 4:2:0 as published
        t0 = time.time()
        raw = g5.rgb_to_yuv420(resize(g5.yuv420_to_rgb(raw), FULL, area=False))
        upsample = 1000 * (time.time() - t0) / n
    return raw, {"bytes": nbytes, "kbps": nbytes * 8 / clip.duration / 1000, **extra,
                 "upsample_ms_per_frame": upsample}, stream


def keep_point(points: Path | None, key: str, yuv: np.ndarray, fps: Fraction, stream: Path) -> dict[str, Any]:
    """The point's published files under ``points/key`` (the decoded frames as FFV1, checked by decoding them
    again, and the bitstream); their record."""
    if points is None:
        return {}
    d = points / key
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    digests = write_ffv1(yuv, d / "decoded.mkv", fps)
    if ffv1_digests(d / "decoded.mkv") != digests:
        raise RuntimeError(f"{key}: the FFV1 file does not decode to the frames written")
    shutil.copyfile(stream, d / f"stream{stream.suffix}")
    return {"decoded": f"points/{key}/decoded.mkv", "decoded_ffv1_sha256": file_sha256(d / "decoded.mkv"),
            "decoded_frame_sha256": digests, "bitstream": f"points/{key}/stream{stream.suffix}",
            "bitstream_sha256": file_sha256(d / f"stream{stream.suffix}")}


def point_dirs(args: argparse.Namespace) -> tuple[Path | None, Path | None, dict[str, dict[str, Any]], list[Path]]:
    """The checkpoint (finished points and their files live there, so a salvaged run keeps them), where new
    points' files go, the points restored from the checkpoint or ``--restore`` (full stage only), and the
    directories that may hold their files."""
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g5c.restore(checkpoints)
    roots = [checkpoints] if checkpoints else []
    if args.restore and args.limit_frames <= 0:  # a failed run's salvaged checkpoint (extracted)
        restored = {**g5c.restore(Path(args.restore) / "checkpoint"), **restored}
        roots.append(Path(args.restore) / "checkpoint")
    points = None
    if args.publish:
        points = (checkpoints or Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")) / "points"
        points.mkdir(parents=True, exist_ok=True)
        roots.append(points.parent)
    return checkpoints, points, restored, roots


def publish_window(args: argparse.Namespace, w: dict[str, Any], rows: list[dict[str, Any]], roots: list[Path],
                   extra: dict[str, Any]) -> dict[str, Any]:
    """Every point's files copied into ``publish`` and the manifest written; returns the publish check."""
    if not args.publish:
        return {}
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch") / "publish"
    entries, complete = [], True
    for row in rows:
        for p in row["points"]:
            key = f"{row['method']}-{p['value']}"
            source = next((r / "points" / key for r in roots if (r / "points" / key / "decoded.mkv").is_file()), None)
            if source is None or "decoded" not in p:
                complete = False
                continue
            shutil.rmtree(publish / "points" / key, ignore_errors=True)
            shutil.copytree(source, publish / "points" / key)
            complete = complete and file_sha256(publish / p["decoded"]) == p["decoded_ffv1_sha256"] \
                and file_sha256(publish / p["bitstream"]) == p["bitstream_sha256"] \
                and len(p["decoded_frame_sha256"]) == w["clip"].n
            entries.append({"method": row["method"], "route": ROUTES[row["codec"]], "codec": row["codec"],
                            "coded_size": row["coded_size"], "value": p["value"],
                            **{k: p.get(k) for k in ("bytes", "kbps", "bitstream", "bitstream_sha256", "decoded",
                                                     "decoded_ffv1_sha256", "decoded_frame_sha256",
                                                     "decode_ms_per_frame", "decoder", "upsample_ms_per_frame",
                                                     "enhancer")}})
    item = w["item"]
    manifest = {"window": item["id"], "video": item["video"], "first_video_index": int(item["first_video_index"]),
                "frames": w["clip"].n, "fps": str(w["fps"]), "yuv": YUV, "foreground": w["foreground"],
                "fill": "ProPainter at 960x540 (e870e79), upsampled (bicubic) into F only; the whole frame is coded",
                "source": w["source"]["window"], **provenance(), **g5c.gpu_record(),
                "timings": "decode times on a shared GPU are not citable; the job's state records contention",
                **extra, "points": entries}
    write_json(publish / "manifest.json", manifest)
    return {"complete": complete, "points": len(entries), "manifest_sha256": file_sha256(publish / "manifest.json")}


def window_record(w: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = w["item"]
    return {"item": item["id"], "video": item["video"], "fps": str(w["fps"]), "frames": w["clip"].n,
            "limit_frames": args.limit_frames, "mask_set": MASK_SET, "dilate_px_1080": args.dilate_px,
            "foreground_share": w["foreground"]["share"], "hole_share": w["foreground"]["hole_share"],
            "foreground": w["foreground"], "source": w["source"], "propainter": w["propainter"],
            "metrics": w["nets"].record, "restored_archive": args.restore, "published": bool(args.publish)}


def command_format(args: argparse.Namespace) -> int:
    methods = [parse_method(m) for m in args.methods.split(",")]
    grid = parse_values(args.values)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "format"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    arm_record, video_ckpt = None, None
    if any(c == "scene" for c, _ in methods):
        arm_dir = Path(args.arm) / "publish" / "finetune"
        arm = json.loads((arm_dir / "result.json").read_text())
        video_ckpt = str(arm_dir / "video_ft.pth.tar")
        if file_sha256(Path(video_ckpt)) != arm["model"]["checkpoint_sha256"]:
            raise RuntimeError("the published checkpoint is not the one its result recorded")
        arm_record = {k: arm.get(k) for k in ("arm", "id", "item", "fill", "steps", "fit", "footage")} | {
            "model_sha256": arm["model"]["checkpoint_sha256"], "converged": arm.get("converged")}
    w = prepare_window(args, work)
    item, clip = w["item"], w["clip"]
    if arm_record and arm_record["id"].split("/")[-1] != item["video"]:
        raise SystemExit(f"{args.arm}: a scene model of {arm_record['id']}, not of {item['video']}")
    checkpoints, points_dir, restored, roots = point_dirs(args)
    rows = []
    done = 0
    for name, (codec, size) in zip(args.methods.split(","), methods):
        points = []
        for value in grid[name]:
            key = f"{name}-{value}"
            if key in restored:
                points.append({**restored[key], "restored_from_checkpoint": True})
            else:
                yuv, rec, stream = code_point(codec, size, value, w["filled"], clip, args, work / key,
                                              video_ckpt if codec == "scene" else None)
                point = {"value": value, **rec, **keep_point(points_dir, key, yuv, w["fps"], stream),
                         "score": g5c.full_score(clip, g5.yuv420_to_rgb(yuv), w["nets"], w["flows"])}
                shutil.rmtree(work / key, ignore_errors=True)
                g5c.save(checkpoints, key, point)
                points.append(point)
            done += 1
            progress(done)
        rows.append({"method": name, "codec": codec, "coded_size": list(size), "points": points})
    check = publish_window(args, w, rows, roots, {"scene_model": arm_record})
    write_json(stage_dir() / "g6.json", {
        "kind": "format", **window_record(w, args), "arm": arm_record,
        "grid": {m: grid[m] for m in args.methods.split(",")}, "restored": sorted(restored), "publish_check": check,
        "seconds": round(time.time() - began, 1), **g5c.gpu_record(), "methods": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def enhance_method(inference: str, channels: int = 64) -> str:
    """Keep the fast correction's publication and checkpoint keys distinct."""
    base = "enh-correction-540" if inference == "fp16-channels-last-540-residual" else "enh-1080"
    return base if channels == 64 else f"{base}-b8-c{channels}"


def command_enhance(args: argparse.Namespace) -> int:
    """SVT-AV1 at 1080p (the anchor, published as is) and G5d's enhancer after it (the second route), fitted
    at each CRF on the window's own native footage coded by SVT-AV1 at that CRF."""
    import torch

    from experiments.background import g5b, g5d
    from src.codecs import svtav1

    crfs = [int(c) for c in args.crfs.split(",")]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "enhance"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    [(meta, yuv_path, keep_path)] = g5b.load_footage([args.footage])
    if meta["target"] != args.item or meta["set"] != "native":
        raise SystemExit(f"{args.footage}: {meta['set']} footage of {meta['target']}, not native footage of {args.item}")
    w = prepare_window(args, work)
    clip = w["clip"]
    ids: list[int] = []
    for seg in meta["segments"]:
        ids += list(range(seg["offset"], seg["end"]))
    if args.limit_frames > 0:
        ids = ids[:48]
    f_yuv = np.ascontiguousarray(np.load(yuv_path, mmap_mode="r")[ids])
    f_keep = np.ascontiguousarray(np.load(keep_path, mmap_mode="r")[ids])
    f_source = work / "footage.yuv"
    f_yuv.tofile(f_source)
    f_target = g5.yuv420_to_rgb(f_yuv)  # the fill: on V the source, through 4:2:0
    del f_yuv
    checkpoints, points_dir, restored, roots = point_dirs(args)
    inference = getattr(args, "inference", "legacy")
    channels = getattr(args, "enhancer_channels", 64)
    method = enhance_method(inference, channels)
    weights_directory = "enhancers" if channels == 64 else f"enhancers/{method}"
    enhancers = (checkpoints or work) / weights_directory
    enhancers.mkdir(parents=True, exist_ok=True)
    plain_points, enh_points = [], []
    for done, crf in enumerate(crfs, 1):
        keys = (f"svt-1080-{crf}", f"{method}-{crf}")
        if all(k in restored for k in keys):
            plain_points.append({**restored[keys[0]], "restored_from_checkpoint": True})
            enh_points.append({**restored[keys[1]], "restored_from_checkpoint": True})
            progress(done)
            continue
        (work / f"footage-{crf}").mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as pool:  # both encoders run as subprocesses
            footage = pool.submit(svtav1.code, f_source, work / f"footage-{crf}", width=FULL[0], height=FULL[1],
                                  fps=clip.fps, frames=len(ids), crf=crf, preset=g5.PRESET, threads=8,
                                  full_range=True, encoder="SvtAv1EncApp", decoder="dav1d")
            yuv, rec, stream = code_point("svt", FULL, crf, w["filled"], clip, args, work / keys[0])
            f_rec = footage.result()
        f_decoded = g5.yuv420_to_rgb(np.fromfile(f_rec["decoded"], np.uint8)
                                     .reshape(len(ids), FULL[1] * 3 // 2, FULL[0]))
        shutil.rmtree(work / f"footage-{crf}", ignore_errors=True)
        plain_rgb = g5.yuv420_to_rgb(yuv)
        plain = {"value": crf, **rec, **keep_point(points_dir, keys[0], yuv, w["fps"], stream),
                 "score": g5c.full_score(clip, plain_rgb, w["nets"], w["flows"])}
        net, fit_record = g5d.fit(f_decoded, f_target, f_keep, "lpips", args.steps, w["nets"].lpips, "cuda",
                                blocks=8, channels=channels,
                                checkpoint=(checkpoints / "fits" / f"{method}-{crf}.pt")
                                if checkpoints is not None and channels != 64 else None)
        del f_decoded
        inference_record: dict[str, Any] = {"config": inference, "batch": 4}
        if inference == "legacy":
            torch.cuda.synchronize()
            t0 = time.time()
            enhanced = g5d.apply(net, plain_rgb, "cuda")
            apply_ms = 1000 * (time.time() - t0) / clip.n
        else:
            from src.codecs.enhancer import apply as apply_enhancer

            enhanced, inference_record = apply_enhancer(net, plain_rgb, inference, profile_kernels=channels != 64)
            apply_ms = inference_record["wall_ms_mean"]
        weights = enhancers / f"crf{crf}.pt"
        torch.save({k: v.half() for k, v in net.state_dict().items()}, weights)
        enh_yuv = g5.rgb_to_yuv420(enhanced)
        enhancer = {"weights": f"{weights_directory}/crf{crf}.pt", "weights_sha256": file_sha256(weights),
                    "blocks": 8, "channels": channels,
                    "weights_bytes": weights.stat().st_size, "parameters": fit_record["parameters"],
                    "apply_ms_per_frame": apply_ms, "inference": inference_record, "fit": fit_record, "fit_frames": len(ids),
                    "fit_kbps": 8 * f_rec["payload_bytes"] / (len(ids) / float(clip.fps)) / 1000,
                    "changed_share": float((enhanced != plain_rgb).any(axis=3).mean())}
        enh = {"value": crf, **rec, "decode_ms_per_frame": rec["decode_ms_per_frame"] + apply_ms,
               **keep_point(points_dir, keys[1], enh_yuv, w["fps"], stream), "enhancer": enhancer,
               "score": g5c.full_score(clip, g5.yuv420_to_rgb(enh_yuv), w["nets"], w["flows"])}
        del net, enhanced, plain_rgb
        shutil.rmtree(work / keys[0], ignore_errors=True)
        g5c.save(checkpoints, keys[0], plain)
        g5c.save(checkpoints, keys[1], enh)
        plain_points.append(plain)
        enh_points.append(enh)
        progress(done)
    rows = [{"method": "svt-1080", "codec": "svt", "coded_size": list(FULL), "points": plain_points},
            {"method": method, "codec": "enh", "coded_size": list(FULL), "points": enh_points}]
    extra = {"enhancer": {**g5d.TRAIN, "blocks": 8, "channels": channels,
                          "steps": args.steps, "loss": "lpips", "fit": "own video, native rate",
                          "footage": {k: meta[k] for k in ("parts", "window", "guard_s", "yuv420_sha256", "frames")}}}
    check = publish_window(args, w, rows, [checkpoints or work, *roots], extra)
    if args.publish:  # the enhancers ship with the window's points (restored ones from the salvaged checkpoint)
        target = scratch / "publish" / weights_directory
        target.mkdir(parents=True, exist_ok=True)
        for crf in crfs:
            found = next((r / weights_directory / f"crf{crf}.pt" for r in [checkpoints or work, *roots]
                          if (r / weights_directory / f"crf{crf}.pt").is_file()), None)
            if found is not None:
                shutil.copyfile(found, target / f"crf{crf}.pt")
        check["enhancers"] = all((scratch / "publish" / p["enhancer"]["weights"]).is_file()
                                 and file_sha256(scratch / "publish" / p["enhancer"]["weights"])
                                 == p["enhancer"]["weights_sha256"] for p in enh_points)
        check["complete"] = check["complete"] and check["enhancers"]
    write_json(stage_dir() / "g6.json", {
        "kind": "enhance", **window_record(w, args), "crfs": crfs, **extra, "restored": sorted(restored),
        "publish_check": check, "seconds": round(time.time() - began, 1), **g5c.gpu_record(), "methods": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- validation


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g6.json").read_text())
    full = not result["limit_frames"]
    gpu = tuple(result.get("gpu_capability") or (0, 0)) >= (8, 6)
    if result["kind"] == "footage":
        mask = result.get("mask") or FOOTAGE_MASKS["g1"]
        guard = round(result["guard_s"] * result["source_fps"])
        lo, hi = result["window"][0] - guard, result["window"][1] + guard
        return {
            "intended_gpu": gpu,
            "segments_present": bool(result["segments"]) and result["frames"] == result["segments"][-1]["end"],
            "foreground_plausible": all(0.0 < s["fg_share"] < 0.8 for s in result["segments"]),
            "outside_window_and_guard": all(s["last_index"] < lo or s["first_index"] > hi for s in result["segments"]),
            "enough_frames": result["frames"] >= (300 if result["set"] == "scene" else 200) if full else result["frames"] > 0,
            "propainter_on_gpu": result["propainter"]["gpu"] == result["gpu"],
            "mask_as_named": all(s["foreground"]["dilate_px_1080"] == 2 * mask["dilate_px"]
                                 and s["foreground"].get("detector", {}).get("kinds", ["hands", "objects"])
                                 == list(mask["box_kinds"]) for s in result["segments"]),
        }
    if result["kind"] == "train":
        return {
            "intended_gpu": gpu,
            "trained_every_step": bool(result["curve"]) and result["curve"][-1]["step"] == result["steps"],
            "finite": not result["skipped_nonfinite"] and all(np.isfinite(p["loss"]) for p in result["curve"]),
            "converged": result["converged"]["converged"] if full else True,
            "enough_frames": result["fit_frames"] >= (300 if full else 2),
        }
    rows = result["methods"]

    def scored(p: dict[str, Any]) -> bool:
        s, k = p["score"], p["score"].get("kid", {})
        return (g5.finite(s["dataset"]["lpips_v"]) and g5.finite(s["dataset"]["psnr_v"])
                and g5.finite(s["flicker"]["flicker_v"]) and g5.finite(k.get("kid_v"))
                and k.get("patches", 0) >= (100 if full else 2))

    checks = {
        "intended_gpu": gpu,
        "methods_present": bool(rows) and all(r["points"] for r in rows),
        "every_point_scored": all(scored(p) for r in rows for p in r["points"]),
        "dcvc_decodes": all(p["deterministic"] and p["decoder_matches_encoder_intra"]
                            for r in rows if r["codec"] in ("dcvc", "scene") for p in r["points"]),
        "rate_follows_value": all((a["kbps"] > b["kbps"]) == (r["codec"] in ("svt", "enh"))  # CRF down, QP up
                                  for r in rows for a, b in zip(r["points"], r["points"][1:])),
        "foreground_plausible": 0.0 < result["foreground_share"] < 0.8,
        "window_is_the_released_one": result["source"]["jpeg_gate_holds"],
        "propainter_on_gpu": result["propainter"]["gpu"] == result["gpu"],
    }
    if result.get("published"):
        checks["published"] = bool(result["publish_check"].get("complete"))
    if result["kind"] == "enhance":
        checks["enhancer_changes_frames"] = all(p["enhancer"]["changed_share"] > 0.01 for p in rows[1]["points"])
        checks["enhancer_architecture"] = all(
            p["enhancer"].get("blocks", 8) == result["enhancer"]["blocks"]
            and p["enhancer"].get("channels", 64) == result["enhancer"]["channels"]
            and p["enhancer"]["fit"].get("channels", 64) == result["enhancer"]["channels"]
            for p in rows[1]["points"])
        if result["enhancer"]["channels"] != 64:
            checks["small_enhancer_on_cuda"] = all(
                p["enhancer"]["inference"]["config"] == "legacy"
                or bool(p["enhancer"]["inference"].get("cuda_kernels"))
                for p in rows[1]["points"])
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage or os.environ.get("PS_STAGE_DIR") or ".")
    checks = validate_stage(stage)
    payload = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), payload)
    print(json.dumps(payload, indent=1))
    return 0 if payload["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    propainter = ("propainter-tree", "propainter-ckpt", "raft-ckpt", "flow-completion-ckpt")
    window = ("eval-set", "archive", "masks", "video", "item", "image-ckpt", "video-ckpt", "lpips-backbone",
              "dists-backbone", "inception", *propainter)
    fo = sub.add_parser("format")
    en = sub.add_parser("enhance")
    for p in (fo, en):
        for name in window:
            p.add_argument(f"--{name}", required=True)
        p.add_argument("--limit-frames", type=int, default=0)
        p.add_argument("--dilate-px", type=int, default=0, help="F's dilation at 1080p (steps 2-3: 16)")
        p.add_argument("--restore", default=None, help="an extracted checkpoint.tar of a failed run of this window")
        p.add_argument("--publish", action="store_true", help="publish decoded frames, bitstreams and a manifest")
    fo.add_argument("--arm", default=None, help="the window's scene model: a train (or G5c finetune) archive")
    fo.add_argument("--methods", default=",".join(METHODS))
    fo.add_argument("--values", default=None, help="METHOD=V,V;... over the default grid")
    fo.set_defaults(func=command_format)
    en.add_argument("--inference", choices=("legacy", "fp16-channels-last", "fp16-channels-last-540-residual"), default="legacy",
                    help="legacy preserves pilot replay; fp16-channels-last uses batch one; "
                         "fp16-channels-last-540-residual retains the fast correction candidate for later evaluation")
    en.add_argument("--footage", required=True, help="the window's extracted native footage archive")
    en.add_argument("--enhancer-channels", type=int, choices=(16, 24, 32, 64), default=64,
                    help="eight-block enhancer width; 16 is the bounded speed-pilot candidate")
    en.add_argument("--crfs", default=",".join(map(str, g5.CRFS)))
    en.add_argument("--steps", type=int, required=True)
    en.set_defaults(func=command_enhance)
    ft = sub.add_parser("footage")
    for name in ("eval-set", "video", "item", "hand-objects", "sam-checkpoint", *propainter):
        ft.add_argument(f"--{name}", required=True)
    ft.add_argument("--set", choices=("scene", "native"), required=True)
    ft.add_argument("--mask", choices=tuple(FOOTAGE_MASKS), default="g1",
                    help="g1: G1's foreground (dilated, hand and object boxes); exact: shaped like the window's F")
    ft.add_argument("--limit-frames", type=int, default=0)
    ft.set_defaults(func=command_footage)
    tr = sub.add_parser("train")
    for name in ("eval-set", "item", "footage", "image-ckpt", "video-ckpt", "lpips-backbone"):
        tr.add_argument(f"--{name}", required=True)
    tr.add_argument("--steps", type=int, required=True)
    tr.add_argument("--lr", type=float, default=RECIPE["lr"])
    tr.add_argument("--lpips-weight", type=float, default=RECIPE["lpips_weight"])
    tr.add_argument("--limit-frames", type=int, default=0)
    tr.add_argument("--timeout", type=float, default=3000)
    tr.add_argument("--resume", default=None, help="an extracted checkpoint.tar of a timed-out run")
    tr.set_defaults(func=command_train)
    va = sub.add_parser("validate")
    va.add_argument("--stage", default=None)
    va.set_defaults(func=command_validate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
