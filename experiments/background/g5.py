"""PLAN step G5, the oracle: the best a neural background model could do on VISOR's background.

    python -m experiments.background.g5 baselines --prepared DIR --clips CLIPS.JSON --visor-fill DIR \\
        --select all|pilot|ID,ID --limit-frames N [--crfs 32,40,...]
    python -m experiments.background.g5 nvrc --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID \\
        --lamb L --epochs S1,S2 --limit-frames N --lpips-backbone PATH
    python -m experiments.background.g5 cond --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID \\
        --lamb L --epochs E --limit-frames N --lpips-backbone PATH
    python -m experiments.background.g5 validate
    python -m experiments.background.g5 report --result g5.json ... --out DIR

Clips come from G1d's prepared archive (`load_clip`): the 34 windows whole, and
a 240-frame excerpt of each of the 10 stretches that keeps its evaluation
window, whose frames carry VISOR's dense masks. Every method codes the same
960×540 RGB frames and is scored on the visible background V of the
`DECISION`'s mask tier, with its rate over the clip's duration
(docs/experiments.md, G5 oracle entry).

``baselines`` codes each clip with SVT-AV1 on two inputs (``frame``, the source
as is, and ``filled``, the foreground inpainted). ``nvrc`` fits NVRC (arm A,
``opt/NVRC`` in the environment) to one clip at one λ, writes its bitstream and
scores the frames decoded from it. ``cond`` does the same for arm B
(`experiments.background.g5_cond`). Each writes ``g5.json`` in
``PS_STAGE_DIR``; ``report`` merges them and applies the decision rule.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import camera, g1, g1d
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any fleet run (docs/experiments.md, G5 oracle entry).
DECISION: dict[str, Any] = {
    "quality": "psnr_v",  # mean over the scored frames of PSNR on V, RGB
    "gate_tier": {"visor-window": "dataset", "visor-long": "dataset"},
    "baseline": "upper envelope of SVT-AV1 frame and filled",
    "min_overlap_db": 0.5,
    "ruled_out_bd": 1.0,  # BD-rate >= +100% (or dominated) on every pilot clip, from converged runs
    "gate_median_bd": 0.0,  # after the rest: the median clip per group below 0
    "converged": {"max_gain_db": 0.3, "max_rate_saving": 0.10, "interval_epochs": 30},
    "pilot": ["visor-window/P02_12", "visor-window/P03_120", "visor-long/P26_02"],
}
EXCERPT_FRAMES = 240
CRFS = (32, 40, 48, 54, 59, 63)
PRESET = 4
REFRESH_S = 0.1
MOTION_BYTES = 128  # `planes4`'s price per frame (G1e)
TELEA_RADIUS = 5
WIDTH, HEIGHT = camera.ANALYSIS_SIZE

#: NVRC (arm A): HiNeRV-v2 at the size of NVRC's smallest model, upsampling 5·3·2·2 for 960×540.
NVRC_MODEL: dict[str, Any] = {
    "type": "HiNeRV",
    "config": {
        "base_encoding": {"base_grid_type": "v2", "base_grid_size": [None, 9, 16, 6], "base_grid_level": 4,
                          "base_grid_level_scale": [2.0, 2.0, 2.0, 0.5], "base_kernel": [3, 3, 3]},
        "decoder": {"channels": [224, 112, 56, 28], "depths": [3, 3, 3, 1], "exps": [4.0, 4.0, 4.0, 4.0],
                    "kernels": [3, 3, 3, 3], "scales_t": [1, 1, 1, 1], "scales_hw": [5, 3, 2, 2],
                    "paddings": [-1, -1, -1], "local_grid_size": [-1, 4], "local_grid_level": 3},
    },
}
NVRC_FRAMES_PER_GRID_T = 3  # base grid T = frames / 3, as NVRC's 1080p configurations
NVRC_PATCH = (1, 180, 120)
NVRC_BATCH = 96
NVRC_STAGES: dict[str, dict[str, Any]] = {  # NVRC's s1-360e and s2-30e, with the epochs from --epochs
    "s1": {"eval_epochs": 30, "warmup_epochs": 30, "lr": 2e-3, "min_lr": 1e-4, "weight_decay": 1e-6,
           "compress": "nvrc_s1.yaml"},
    "s2": {"eval_epochs": 10, "warmup_epochs": 5, "lr": 1e-4, "min_lr": 1e-5, "weight_decay": 0.0,
           "compress": "nvrc_s2.yaml"},
}


# ----------------------------------------------------------------- clips


def select_clips(spec: dict[str, Any], which: str) -> list[dict[str, Any]]:
    """G1d's 44 clips; ``pilot`` and ``smoke`` are G1d's pilot (2 windows, 1 stretch)."""
    return g1d.select_clips(spec, which)


def frame_rate(times: list[float]) -> Fraction:
    """The analysis frame rate: 10, 50 or 60000/1001 frames per second."""
    rate = (len(times) - 1) / (times[-1] - times[0])
    for exact in (Fraction(10), Fraction(25), Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)):
        if abs(rate - float(exact)) < 0.01 * float(exact):
            return exact
    raise ValueError(f"unexpected analysis frame rate {rate:.4f}")


def excerpt_start(total: int, dense_frames: list[int], length: int) -> int:
    """A stretch's excerpt starts at its first dense-mask frame, moved earlier to fit."""
    first = min(dense_frames) if dense_frames else 0
    return max(0, min(first, total - length))


def refresh_reference(n: int, fps: Fraction) -> list[int | None]:
    """Per frame, the last refresh point before it (refresh every `REFRESH_S`); None for the first frame."""
    step = max(1, round(REFRESH_S * float(fps)))
    return [None if t == 0 else ((t - 1) // step) * step for t in range(n)]


@dataclass
class Clip:
    id: str
    group: str
    fps: Fraction
    start: int
    frames: np.ndarray  # (n, h, w, 3) uint8 RGB
    keep: np.ndarray  # (n, h, w) bool: the pixels a method is fitted to (V of the training masks)
    scored: dict[str, tuple[list[int], np.ndarray]] = field(default_factory=dict)  # tier -> (frames, V)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.n / float(self.fps)


def clip_dirs(prepared: Path) -> dict[str, Path]:
    dirs = {}
    for d in (prepared / "publish" / "clips").iterdir():
        if (d / "meta.json").is_file():
            dirs[json.loads((d / "meta.json").read_text())["id"]] = d
    return dirs


def load_clip(clip_dir: Path, spec_clip: dict[str, Any], visor_fill: Path | None, limit_frames: int) -> Clip:
    """The clip's frames and masks. Windows: every frame, V = not the dataset foreground (VISOR dense with
    B1b's fill, dilated). Stretches: the excerpt; V = not the dense masks on the window frames (``dataset``)
    and not G1's `sam_text` foreground elsewhere (``own``, all frames)."""
    meta = json.loads((clip_dir / "meta.json").read_text())
    total = int(meta["frames"])
    fps = frame_rate(meta["times"])
    fg_all = g1d.load_foreground(clip_dir / "foreground.npz")
    dense: dict[int, np.ndarray] = {}
    if meta["group"] == "visor-long":
        if visor_fill is None:
            raise SystemExit("stretches need --visor-fill for their dense masks")
        raw, _ = g1.window_dense(g1.plan(spec_clip, meta["limit_seconds"]), visor_fill)
        dense = {i: g1.dilate(m) for i, m in raw.items()}
        length = min(EXCERPT_FRAMES, total)
        start = excerpt_start(total, sorted(dense), length)
    else:
        length, start = total, 0
    n = min(length, limit_frames) if limit_frames > 0 else length
    frames = np.empty((n, HEIGHT, WIDTH, 3), np.uint8)
    got = 0
    for i, rgb in enumerate(g1d.decode_ffv1(clip_dir / "frames.mkv", 4)):
        if i >= start + n:
            break
        if i >= start:
            frames[i - start] = rgb
            got += 1
    if got != n:
        raise RuntimeError(f"{meta['id']}: decoded {got} of {n} frames")
    fg = np.asarray(fg_all[start:start + n])
    keep = ~fg
    scored: dict[str, tuple[list[int], np.ndarray]] = {}
    if meta["group"] == "visor-long":
        local = sorted(i - start for i in dense if start <= i < start + n)
        for i in local:
            keep[i] = ~dense[i + start]
        scored["dataset"] = (local, np.stack([keep[i] for i in local]) if local else np.zeros((0, HEIGHT, WIDTH), bool))
        scored["own"] = (list(range(n)), keep)
        tier = "visor_dense on window frames; sam_text elsewhere"
    else:
        scored["dataset"] = (list(range(n)), keep)
        tier = meta["masks"]["tier"]
    record = {"id": meta["id"], "group": meta["group"], "frames": n, "start": start, "fps": str(fps),
              "duration_s": n / float(fps), "tier": tier, "dense_frames": len(scored["dataset"][0]),
              "v_share": float(keep.mean()), "frames_sha256_archive": meta["frames_sha256"]}
    return Clip(meta["id"], meta["group"], fps, start, frames, keep, scored, record)


# ----------------------------------------------------------------- colour and scores


def rgb_to_yuv420(frames: np.ndarray) -> np.ndarray:
    """(n, h, w, 3) RGB -> (n, h*3/2, w) 4:2:0, BT.709 full range, chroma averaged over 2×2 (the inverse of
    `src.codecs.quality.yuv420_to_rgb` with ``full_range=True``)."""
    from src.codecs.quality import KB, KR

    rgb = frames.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + (1.0 - KR - KB) * g + KB * b
    cb = (b - y) / (2.0 * (1.0 - KB))
    cr = (r - y) / (2.0 * (1.0 - KR))
    n, h, w = y.shape

    def pool(c: np.ndarray) -> np.ndarray:
        return c.reshape(n, h // 2, 2, w // 2, 2).mean(axis=(2, 4))

    out = np.empty((n, h * 3 // 2, w), np.uint8)
    out[:, :h] = np.clip(np.round(y * 255.0), 0, 255)
    chroma = np.stack([pool(cb), pool(cr)], axis=1) * 255.0 + 128.0
    out[:, h:] = np.clip(np.round(chroma), 0, 255).astype(np.uint8).reshape(n, h // 2, w)
    return out


def yuv420_to_rgb(yuv: np.ndarray) -> np.ndarray:
    import torch

    from src.codecs import quality

    height, width = yuv.shape[1] * 2 // 3, yuv.shape[2]
    rgb = quality.yuv420_to_rgb(torch.from_numpy(np.ascontiguousarray(yuv)), width, height, full_range=True)
    return rgb.permute(0, 2, 3, 1).to(torch.uint8).numpy()


def psnr_from_sse(sse: float, count: int) -> float | None:
    from src.codecs.quality import psnr

    return psnr(sse / count) if count else None


def score(clip: Clip, decoded: np.ndarray, lpips_net: Any = None, device: str = "cpu") -> dict[str, Any]:
    """Per tier: mean over its frames of PSNR on V (exact sums, RGB), the pooled PSNR, and LPIPS on V with
    the foreground pasted back from the source (G2); also whole-frame PSNR."""
    if decoded.shape != clip.frames.shape or decoded.dtype != np.uint8:
        raise ValueError(f"decoded {decoded.shape} {decoded.dtype} != source {clip.frames.shape}")
    out: dict[str, Any] = {}
    sq = np.empty(clip.frames.shape[:3], np.int64)
    for i in range(clip.n):
        d = clip.frames[i].astype(np.int32) - decoded[i].astype(np.int32)
        sq[i] = (d * d).sum(axis=2)
    lp: dict[int, np.ndarray] = {}
    if lpips_net is not None:
        lp = lpips_maps(clip, decoded, sorted({i for f, _ in clip.scored.values() for i in f}), lpips_net, device)
    for tier, (frames, masks) in clip.scored.items():
        per_frame, lp_frame, sse_total, count_total = [], [], 0, 0
        for i, mask in zip(frames, masks):
            sse, count = int(sq[i][mask].sum()), int(mask.sum())
            sse_total, count_total = sse_total + sse, count_total + count
            value = psnr_from_sse(sse, 3 * count)
            if value is not None:
                per_frame.append(value)
            if i in lp and count:
                lp_frame.append(float(lp[i][mask].mean()))
        out[tier] = {"frames": len(frames), "psnr_v": float(np.mean(per_frame)) if per_frame else None,
                     "psnr_v_pooled": psnr_from_sse(sse_total, 3 * count_total),
                     "lpips_v": float(np.mean(lp_frame)) if lp_frame else None, "psnr_v_per_frame": per_frame}
    from src.codecs.quality import psnr

    out["frame"] = {"psnr": float(np.mean([psnr(int(s.sum()) / (3 * s.size)) for s in sq]))}
    return out


def lpips_maps(clip: Clip, decoded: np.ndarray, frames: list[int], net: Any, device: str) -> dict[int, np.ndarray]:
    import torch

    maps = {}
    for first in range(0, len(frames), 8):
        idx = frames[first:first + 8]
        keep = clip.keep[idx][..., None]
        pasted = np.where(keep, decoded[idx], clip.frames[idx])
        a = torch.from_numpy(pasted).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        b = torch.from_numpy(clip.frames[idx]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.inference_mode():
            m = net(a, b)[:, 0].float().cpu().numpy()
        for j, i in enumerate(idx):
            maps[i] = m[j]
    return maps


def load_lpips(backbone: str | None, device: str) -> Any:
    if not backbone:
        return None
    from src.codecs import quality

    return quality.load_lpips(Path(backbone), device)


def strip_frames(result: dict[str, Any]) -> dict[str, Any]:
    """A score without its per-frame lists (for summaries)."""
    return {k: ({kk: vv for kk, vv in v.items() if kk != "psnr_v_per_frame"} if isinstance(v, dict) else v)
            for k, v in result.items()}


# ----------------------------------------------------------------- baselines (SVT-AV1)


def inpaint(clip: Clip) -> np.ndarray:
    """The ``filled`` input: the foreground (not V of the training masks) inpainted per frame (Telea)."""
    import cv2

    out = np.empty_like(clip.frames)
    for i in range(clip.n):
        hole = (~clip.keep[i]).astype(np.uint8)
        out[i] = cv2.inpaint(clip.frames[i], hole, TELEA_RADIUS, cv2.INPAINT_TELEA) if hole.any() else clip.frames[i]
    return out


def code_svtav1(task: dict[str, Any]) -> dict[str, Any]:
    """One encode and decode (spawned); returns the stream record and writes the decoded frames."""
    from src.codecs import svtav1

    work = Path(task["work"])
    work.mkdir(parents=True, exist_ok=True)
    record = svtav1.code(Path(task["source"]), work, width=WIDTH, height=HEIGHT, fps=Fraction(task["fps"]),
                         frames=task["frames"], crf=task["crf"], preset=PRESET, threads=task["threads"],
                         full_range=True, encoder="SvtAv1EncApp", decoder="dav1d", cpus=task["cpus"])
    return record


def command_baselines(args: argparse.Namespace) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from src.codecs import svtav1

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs = clip_dirs(Path(args.prepared))
    chosen = [c["id"] for c in select_clips(spec, args.select)]
    crfs = [int(x) for x in args.crfs.split(",")]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    allowance = max(4, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 4))
    threads = 4
    slots = max(1, allowance // threads)
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(allowance))
    tools = {name: svtav1.tool(name) for name in ("SvtAv1EncApp", "dav1d")}
    pool = ProcessPoolExecutor(max_workers=slots, mp_context=multiprocessing.get_context("spawn"))
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish) if checkpoints else {}
    rows = []
    for done, clip_id in enumerate(chosen, 1):
        if clip_id in restored:
            rows.append({**restored[clip_id]["result"], "restored_from_checkpoint": True})
            progress(done)
            continue
        clip = load_clip(dirs[clip_id], by_id[clip_id], Path(args.visor_fill) if args.visor_fill else None,
                         args.limit_frames)
        work = scratch / "work" / g1.safe(clip_id)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        inputs = {"frame": clip.frames, "filled": inpaint(clip)}
        sources = {}
        for name, rgb in inputs.items():
            sources[name] = work / f"{name}.yuv"
            rgb_to_yuv420(rgb).tofile(sources[name])
        ceiling = score(clip, yuv420_to_rgb(rgb_to_yuv420(clip.frames)))
        futures = {}
        for k, (name, crf) in enumerate((n, c) for n in inputs for c in crfs):
            slot = k % slots
            task = {"work": str(work / f"{name}-crf{crf}"), "source": str(sources[name]), "fps": str(clip.fps),
                    "frames": clip.n, "crf": crf, "threads": threads,
                    "cpus": cores[slot * threads:(slot + 1) * threads] if len(cores) >= slots * threads else None}
            futures[(name, crf)] = pool.submit(code_svtav1, task)
        points = []
        for (name, crf), future in futures.items():
            rec = future.result()
            decoded = np.fromfile(rec["decoded"], np.uint8)
            if decoded.size != clip.n * WIDTH * HEIGHT * 3 // 2:
                raise RuntimeError(f"{clip_id} {name} crf {crf}: decoded {decoded.size} bytes")
            rgb = yuv420_to_rgb(decoded.reshape(clip.n, HEIGHT * 3 // 2, WIDTH))
            target = publish / "clips" / g1.safe(clip_id) / "streams"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rec["stream"], target / f"{name}-crf{crf}.ivf")
            points.append({
                "codec": "svtav1", "input": name, "crf": crf,
                "kbps": rec["payload_bytes"] * 8 / clip.duration / 1000, "payload_bytes": rec["payload_bytes"],
                "stream_sha256": rec["stream_sha256"], "temporal_units": rec["temporal_units"],
                "decoded_frames": rec["decoded_frames"],
                "decode_ms_per_frame": 1000 * rec["decode_seconds"] / clip.n,
                "encode_seconds": rec["encode_seconds"], "score": score(clip, rgb),
                "encode_command": rec["encode_command"],
            })
            Path(rec["decoded"]).unlink()
        row = {**clip.record, "kind": "baselines", "points": points, "ceiling_420": ceiling}
        write_json(publish / "clips" / g1.safe(clip_id) / "baselines.json", row)
        if checkpoints is not None:
            g1.save_clip(checkpoints, publish / "clips" / g1.safe(clip_id), clip_id, row, None)
        rows.append(row)
        shutil.rmtree(work, ignore_errors=True)
        progress(done)
    pool.shutdown()
    write_json(stage_dir() / "g5.json", {"kind": "baselines", "select": args.select, "crfs": crfs,
                                         "restored_clips": sorted(restored),
                                         "limit_frames": args.limit_frames, "preset": PRESET, "tools": tools,
                                         "decision_rule": DECISION, "clips": [summarize_row(r) for r in rows]})
    return 0


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in ("points", "ceiling_420")}
    out["points"] = [{**{k: v for k, v in p.items() if k not in ("score", "encode_command")},
                      "score": strip_frames(p["score"])} for p in row["points"]]
    if "ceiling_420" in row:
        out["ceiling_420"] = strip_frames(row["ceiling_420"])
    return out


DCVC_QPS = (9, 27, 45, 63)


def code_dcvc(args: argparse.Namespace, source: Path, frames: int, qp: int, work: Path) -> dict[str, Any]:
    """One DCVC-UF HT-L encode and decode of a 4:2:0 full-range file (B2's worker, full range as is)."""
    from src.codecs.dcvc_uf_worker import dcvc_command

    work.mkdir(parents=True, exist_ok=True)
    plan = {"structure": "htl", "qp": qp, "frame_count": frames, "height": HEIGHT, "width": WIDTH,
            "src_type": "yuv420", "frames_file": str(source), "container": str(work / "stream.psdc"),
            "out_file": str(work / "decoded.yuv"), "image_ckpt": args.image_ckpt,
            "image_sha256": file_sha256(Path(args.image_ckpt)), "video_ckpt": args.video_ckpt,
            "video_sha256": file_sha256(Path(args.video_ckpt)), "profile": False}
    plan_path = work / "plan.json"
    write_json(plan_path, plan)
    reports = {}
    for action in ("encode", "decode"):
        report = work / f"{action}.json"
        command, env, cwd = dcvc_command(Path(sys.prefix), action, plan_path, report)
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True,
                              timeout=3600)
        if done.returncode:
            raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads(report.read_text())
    data = (work / "stream.psdc").read_bytes()
    decode, encode = reports["decode"], reports["encode"]
    return {"bytes": len(data), "decoded": str(work / "decoded.yuv"), "deterministic": decode["deterministic"],
            "decoder_matches_encoder_intra": decode["passes"][0]["frame_sha256"][0] == encode["i_recon_sha256"],
            "decode_seconds": [p["decode_seconds"] for p in decode["passes"]],
            "model_decode_seconds": [p.get("model_decode_seconds") for p in decode["passes"]]}


def command_dcvc(args: argparse.Namespace) -> int:
    """DCVC-UF HT-L on the ``filled`` input of the selected clips (comparison, not the gate)."""
    import torch

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs = clip_dirs(Path(args.prepared))
    qps = [int(x) for x in args.qps.split(",")]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    net = load_lpips(args.lpips_backbone, "cuda")
    rows = []
    for done, clip_id in enumerate(c["id"] for c in select_clips(spec, args.select)):
        clip = load_clip(dirs[clip_id], by_id[clip_id], Path(args.visor_fill) if args.visor_fill else None,
                         args.limit_frames)
        work = scratch / "work" / g1.safe(clip_id)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        source = work / "filled.yuv"
        rgb_to_yuv420(inpaint(clip)).tofile(source)
        points = []
        for qp in qps:
            rec = code_dcvc(args, source, clip.n, qp, work / f"qp{qp}")
            decoded = np.fromfile(rec.pop("decoded"), np.uint8).reshape(clip.n, HEIGHT * 3 // 2, WIDTH)
            target = publish / "clips" / g1.safe(clip_id) / "streams"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / f"qp{qp}" / "stream.psdc", target / f"filled-qp{qp}.psdc")
            points.append({"codec": "dcvc-uf-htl", "input": "filled", "qp": qp,
                           "kbps": rec["bytes"] * 8 / clip.duration / 1000, "decoded_frames": len(decoded),
                           "decode_ms_per_frame": 1000 * min(rec["decode_seconds"]) / clip.n, **rec,
                           "score": score(clip, yuv420_to_rgb(decoded), net, "cuda")})
            shutil.rmtree(work / f"qp{qp}")
        row = {**clip.record, "kind": "dcvc", "points": points, "gpu": torch.cuda.get_device_name(0),
               "gpu_capability": list(torch.cuda.get_device_capability(0))}
        write_json(publish / "clips" / g1.safe(clip_id) / "dcvc.json", row)
        rows.append({**row, "points": [{**p, "score": strip_frames(p["score"])} for p in points]})
        shutil.rmtree(work, ignore_errors=True)
        progress(done + 1)
    write_json(stage_dir() / "g5.json", {"kind": "dcvc", "qps": qps, "decision_rule": DECISION, "clips": rows})
    return 0


# ----------------------------------------------------------------- arm A: NVRC


def nvrc_root() -> Path:
    root = Path(sys.prefix) / "opt" / "NVRC"
    if not (root / "main_nvrc.py").is_file():
        raise FileNotFoundError(f"NVRC is not vendored in this environment: {root} (env/nvrc.sh)")
    return root


def write_pngs(clip: Clip, rgba_dir: Path, rgb_dir: Path) -> None:
    from PIL import Image

    rgba_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for i in range(clip.n):
        alpha = clip.keep[i].astype(np.uint8)[..., None] * 255
        Image.fromarray(np.concatenate([clip.frames[i], alpha], axis=2), "RGBA").save(rgba_dir / f"{i:04d}.png",
                                                                                     compress_level=1)
        Image.fromarray(clip.frames[i], "RGB").save(rgb_dir / f"{i:04d}.png", compress_level=1)


def yaml_dump(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))


def nvrc_configs(root: Path, n: int, epochs: dict[str, int], lamb: float) -> dict[str, Path]:
    """Model, task and per-stage configurations in ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    model = json.loads(json.dumps(NVRC_MODEL))
    model["config"]["base_encoding"]["base_grid_size"][0] = max(1, n // NVRC_FRAMES_PER_GRID_T)
    paths = {"model": root / "model.yaml", "task": root / "task.yaml"}
    yaml_dump(paths["model"], model)
    yaml_dump(paths["task"], {"loss": [1.0, "mse"], "metric": ["psnr"], "color_space": "RGB", "lamb": lamb})
    for stage, cfg in NVRC_STAGES.items():
        paths[stage] = root / f"{stage}.yaml"
        yaml_dump(paths[stage], {
            "epochs": epochs[stage], "eval_epochs": min(cfg["eval_epochs"], epochs[stage]),
            "warmup_epochs": min(cfg["warmup_epochs"], max(1, epochs[stage] // 4)), "rate_steps": 8,
            "log_epochs": -2, "opt": "adam", "lr": cfg["lr"], "warmup_lr": 1e-5, "min_lr": cfg["min_lr"],
            "max_norm": 1.0, "weight_decay": cfg["weight_decay"], "weight_decay_scaling": True,
            "auto_lr_scaling": True,
        })
    return paths


def nvrc_command(stage: str, paths: dict[str, Path], data: Path, output: Path, lamb: float,
                 resume: Path | None, dynamo: str = "inductor") -> list[str]:
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", "--num_machines=1",
           "--mixed_precision=fp16", f"--dynamo_backend={dynamo}", "main_nvrc.py",
           "--exp-config", str(paths[stage]), "--output", str(output), "--exp-name", stage,
           "--train-task-config", str(paths["task"]), "--eval-task-config", str(paths["task"]),
           "--compress-model-config", f"scripts/configs/nvrc/compress_models/{NVRC_STAGES[stage]['compress']}",
           "--model-config", str(paths["model"]),
           "--train-dataset-dir", str(data), "--train-dataset", "rgba", "--train-fmt", "png",
           "--eval-dataset-dir", str(data), "--eval-dataset", "rgb", "--eval-fmt", "png",
           "--lamb", f"{lamb:g}", "--start-frame", "-1", "--num-frames", "-1", "--intra-period", "-1",
           "--train-video-size", "-1", "-1", "-1", "--eval-video-size", "-1", "-1", "-1",
           "--train-patch-size", *map(str, NVRC_PATCH), "--eval-patch-size", "1", "-1", "-1",
           "--grad-accum", "1", "--rate-steps", "8", "--train-batch-size", str(NVRC_BATCH),
           "--eval-batch-size", "1", "--train-enable-log", "false",
           "--eval-enable-log", "true" if stage == "s2" else "false", "--log-epochs", "-2",
           "--opt", "adam", "--sched", "cosine", "--lr", str(NVRC_STAGES[stage]["lr"]), "--warmup-lr", "1e-5",
           "--min-lr", str(NVRC_STAGES[stage]["min_lr"]), "--auto-lr-scaling", "true", "--max-norm", "1.0",
           "--workers", "4", "--prefetch-factor", "4"]
    if resume is not None:
        cmd += ["--resume", str(resume), "--resume-model-only", "true"]
    return cmd


TRAIN_LINE = re.compile(r"Train - Epoch (\d+) \[(\d+)/(\d+)\].*?bpp: ([\d.]+)\s+psnr: ([\d.]+)")
EVAL_LINE = re.compile(r"Eval - \[(\d+)/(\d+)\]\s+img/s: ([\d.]+)")


def train_curve(log: str) -> list[dict[str, float]]:
    """The last logged step of every training epoch: (epoch, bpp, masked-training PSNR). NVRC logs every few
    steps, so an epoch's final step is not always among them."""
    last: dict[int, tuple[int, dict[str, float]]] = {}
    for m in TRAIN_LINE.finditer(log):
        epoch, step = int(m.group(1)), int(m.group(2))
        if epoch not in last or step >= last[epoch][0]:
            last[epoch] = (step, {"epoch": epoch, "bpp": float(m.group(4)), "psnr": float(m.group(5))})
    return [last[e][1] for e in sorted(last)]


def converged(curve: list[dict[str, float]], interval: int) -> dict[str, Any]:
    """The decision rule's convergence check on the last ``interval`` epochs: still gaining more than
    `DECISION` allows at equal rate, or saving more rate at equal quality, is not converged. Training PSNR
    rises with V's PSNR (the masked pixels add no error), so it ranks the end of training correctly."""
    rule = DECISION["converged"]
    if len(curve) <= interval:
        return {"checked": False, "reason": f"{len(curve)} epochs logged"}
    a, b = curve[-1 - interval], curve[-1]
    gain = b["psnr"] - a["psnr"]
    saving = 1.0 - b["bpp"] / a["bpp"] if a["bpp"] > 0 else 0.0
    ok = not ((gain > rule["max_gain_db"] and b["bpp"] <= a["bpp"] * 1.0001) or
              (saving > rule["max_rate_saving"] and b["psnr"] >= a["psnr"] - 1e-4))
    return {"checked": True, "converged": ok, "from_epoch": a["epoch"], "to_epoch": b["epoch"],
            "psnr_gain_db": gain, "rate_saving": saving}


def decoded_frames_dir(outputs: Path) -> Path:
    """NVRC's frames decoded from the bitstream: ``outputs/0000/decoded/<dataset>/<name>`` (one coding group)."""
    found = sorted({p.parent for p in (outputs / "0000" / "decoded").rglob("*.png")})
    if len(found) != 1:
        raise RuntimeError(f"{outputs}: expected one directory of decoded frames, found {found}")
    return found[0]


def read_pngs(directory: Path, n: int) -> np.ndarray:
    from PIL import Image

    names = sorted(p for p in directory.glob("*.png"))
    if len(names) != n:
        raise RuntimeError(f"{directory}: {len(names)} decoded frames, expected {n}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in names])


def nvrc_env(work: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HOME=str(work / "hf"), TORCHINDUCTOR_CACHE_DIR=str(work / "inductor"),
               TRITON_CACHE_DIR=str(work / "triton"), PATH=f"{Path(sys.executable).parent}:{env.get('PATH', '')}")
    return env


def run_nvrc_stage(stage: str, paths: dict[str, Path], data: Path, output: Path, lamb: float, resume: Path | None,
                   dynamo: str, work: Path) -> tuple[str, float]:
    began = time.time()
    log_path = work / f"{stage}.log"
    with log_path.open("w") as log:
        done = subprocess.run(nvrc_command(stage, paths, data, output, lamb, resume, dynamo), cwd=nvrc_root(),
                              env=nvrc_env(work), stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    if done.returncode:
        raise RuntimeError(f"NVRC {stage} failed ({done.returncode}):\n{text[-4000:]}")
    return text, round(time.time() - began, 1)


def command_nvrc(args: argparse.Namespace) -> int:
    """``--part s1``: NVRC's stage 1, published as a checkpoint. ``--part s2``: stage 2 from ``--s1-from`` (part
    1's extracted publish), the bitstream, its decode and the scores. ``--part both`` runs the two in one."""
    import torch

    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("NVRC needs a GPU")
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = only_clip(spec, args.select)
    clip = load_clip(clip_dirs(Path(args.prepared))[clip_id], by_id[clip_id],
                     Path(args.visor_fill) if args.visor_fill else None, args.limit_frames)
    s1, s2 = (int(x) for x in args.epochs.split(","))
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "nvrc"
    shutil.rmtree(work, ignore_errors=True)
    data = work / "data"
    write_pngs(clip, data / "rgba", data / "rgb")
    paths = nvrc_configs(work / "configs", clip.n, {"s1": s1, "s2": s2}, args.lamb)
    output = work / "out"
    publish = scratch / "publish"
    shutil.rmtree(publish, ignore_errors=True)
    run = {**clip.record, "lamb": args.lamb, "epochs": {"s1": s1, "s2": s2}, "model": NVRC_MODEL,
           "model_yaml_sha256": file_sha256(paths["model"]), "patch": NVRC_PATCH, "batch": NVRC_BATCH,
           "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
           "nvrc_root": str(nvrc_root()), "provenance": nvrc_provenance()}
    if args.part in ("s1", "both"):
        log_s1, seconds_s1 = run_nvrc_stage("s1", paths, data, output, args.lamb, None, args.dynamo_s1, work)
        progress(1)
        part1 = {**run, "kind": "nvrc-s1", "dynamo": args.dynamo_s1, "seconds": {"s1": seconds_s1},
                 "curve_s1": train_curve(log_s1),
                 "converged": converged(train_curve(log_s1), DECISION["converged"]["interval_epochs"]),
                 "checkpoint": str(output / "s1" / "checkpoints")}
        target = publish / "nvrc-s1"
        shutil.copytree(output / "s1", target / "s1", ignore=shutil.ignore_patterns("outputs", "tmp", "rank_0"))
        shutil.copyfile(work / "s1.log", target / "s1.log")
        write_json(target / "part1.json", part1)
        if args.part == "s1":
            write_json(stage_dir() / "g5.json", {"kind": "nvrc-s1", "decision_rule": DECISION, "clips": [part1]})
            shutil.rmtree(work, ignore_errors=True)
            return 0
        s1_root = target
    else:
        s1_root = Path(args.s1_from) / "publish" / "nvrc-s1"
    part1 = json.loads((s1_root / "part1.json").read_text())
    for key in ("id", "frames", "start", "lamb", "model_yaml_sha256"):
        if part1[key] != run[key]:
            raise SystemExit(f"part 1 does not match this run: {key} {part1[key]} != {run[key]}")
    log_s2, seconds_s2 = run_nvrc_stage("s2", paths, data, output, args.lamb, s1_root / "s1", args.dynamo_s2, work)
    progress(2)
    bitstream = output / "s2" / "bitstreams" / "bitstream.bits"
    decoded = read_pngs(decoded_frames_dir(output / "s2" / "outputs"), clip.n)
    evals = EVAL_LINE.findall(log_s2.split("Start evaluating the decoded model.")[-1])
    render_fps = float(evals[-1][2]) if evals else None
    net = load_lpips(args.lpips_backbone, device)
    result = {
        **run, "kind": "nvrc", "dynamo": {"s1": part1.get("dynamo"), "s2": args.dynamo_s2},
        "bitstream_bytes": bitstream.stat().st_size, "bitstream_sha256": file_sha256(bitstream),
        "kbps": bitstream.stat().st_size * 8 / clip.duration / 1000,
        "render_ms_per_frame": 1000 / render_fps if render_fps else None, "render_basis": "NVRC decoded-model "
        "evaluation, batch 1, model forward from the bitstream's weights (img/s of its last log line)",
        "score": score(clip, decoded, net, device),
        "seconds": {**part1["seconds"], "s2": seconds_s2}, "part1_gpu": part1["gpu"],
        "curve_s1": part1["curve_s1"], "curve_s2": train_curve(log_s2), "converged": part1["converged"],
    }
    (publish / "nvrc").mkdir(parents=True)
    shutil.copyfile(bitstream, publish / "nvrc" / "bitstream.bits")
    shutil.copyfile(work / "s2.log", publish / "nvrc" / "s2.log")
    write_json(publish / "nvrc" / "result.json", result)
    write_json(stage_dir() / "g5.json", {"kind": "nvrc", "decision_rule": DECISION, "clips": [strip_result(result)]})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def nvrc_provenance() -> dict[str, Any]:
    path = Path(sys.prefix) / "opt" / "PROVENANCE.json"
    return json.loads(path.read_text()).get("NVRC", {}) if path.is_file() else {}


def strip_result(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "score": strip_frames(result["score"])}


def only_clip(spec: dict[str, Any], which: str) -> str:
    chosen = [c["id"] for c in select_clips(spec, which)]
    if len(chosen) != 1:
        raise SystemExit(f"one clip per run: {which} selects {len(chosen)}")
    return chosen[0]


# ----------------------------------------------------------------- arm B (experiments.background.g5_cond)


def command_cond(args: argparse.Namespace) -> int:
    """Arm B on one clip: refreshes every `REFRESH_S` coded by SVT-AV1 at ``--crf`` as their own stream, and the
    frames between them rendered from the warped refresh, (a) as is (``warp``) and (b) corrected by the
    per-clip model at ``--lamb`` (``cond``)."""
    import torch

    from experiments.background import g5_cond

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = only_clip(spec, args.select)
    clip = load_clip(clip_dirs(Path(args.prepared))[clip_id], by_id[clip_id],
                     Path(args.visor_fill) if args.visor_fill else None, args.limit_frames)
    step = max(1, round(REFRESH_S * float(clip.fps)))
    if step == 1:
        raise SystemExit(f"{clip_id}: a {REFRESH_S} s refresh is every frame here, so arm B is the baseline itself")
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "cond"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    refresh = list(range(0, clip.n, step))
    rgb_to_yuv420(clip.frames[refresh]).tofile(work / "refresh.yuv")
    rec = code_svtav1({"work": str(work / "refresh"), "source": str(work / "refresh.yuv"),
                       "fps": str(clip.fps / step), "frames": len(refresh), "crf": args.crf, "threads": 8, "cpus": None})
    raw = np.fromfile(rec["decoded"], np.uint8).reshape(len(refresh), HEIGHT * 3 // 2, WIDTH)
    fixed = dict(zip(refresh, yuv420_to_rgb(raw)))
    refs = refresh_reference(clip.n, clip.fps)
    between = [t for t in range(clip.n) if t not in fixed]
    flows = g5_cond.oracle_flows(clip.frames, [refs[t] if t in between else None for t in range(clip.n)])
    flow_seconds = time.time() - began
    net = load_lpips(args.lpips_backbone, device)
    refresh_bits = 8 * rec["payload_bytes"]
    motion_bits = 8 * MOTION_BYTES * len(between)
    common = {**clip.record, "kind": "cond", "crf": args.crf, "refresh_every": step, "refresh_frames": len(refresh),
              "between_frames": len(between), "refresh_stream_sha256": rec["stream_sha256"],
              "refresh_encode_command": rec["encode_command"],
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "gpu_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None}
    warped = g5_cond.warp_only(fixed, flows, refs, clip.n, device)
    rows = [{**common, "arm": "warp", "lamb": None, "bits": {"refresh": refresh_bits, "motion": motion_bits},
             "kbps": (refresh_bits + motion_bits) / clip.duration / 1000, "score": score(clip, warped, net, device),
             "render_ms_per_frame": None, "converged": {"checked": False, "reason": "nothing trained"}}]
    fit = g5_cond.fit(clip.frames, clip.keep, flows, refs, lamb=args.lamb, epochs=args.epochs, device=device,
                      on_epoch=progress, fixed=fixed)
    bits = {"refresh": refresh_bits, "motion": motion_bits, **fit["bits"]}
    rows.append({**common, "arm": "cond", "lamb": args.lamb, "epochs": args.epochs, "bits": bits,
                 "kbps": sum(bits.values()) / clip.duration / 1000, "score": score(clip, fit["decoded"], net, device),
                 "score_float_weights": strip_frames(score(clip, fit["decoded_float"])),
                 "render_ms_per_frame": fit["render_ms_per_frame"], "render_basis": "the frames between refreshes: "
                 "model forward from 8-bit weights and rounded latents plus the warp, batch 1, flows precomputed; "
                 "the refreshes decode with dav1d", "dav1d_ms_per_refresh": 1000 * rec["decode_seconds"] / len(refresh),
                 "curve": fit["curve"], "converged": fit["converged"], "model": fit["model"]})
    for row in rows:
        row["seconds"] = round(time.time() - began, 1)
        row["flow_seconds"] = round(flow_seconds, 1)
    publish = scratch / "publish"
    shutil.rmtree(publish, ignore_errors=True)
    (publish / "cond").mkdir(parents=True)
    shutil.copyfile(rec["stream"], publish / "cond" / "refresh.ivf")
    write_json(publish / "cond" / "result.json", rows)
    write_json(stage_dir() / "g5.json", {"kind": "cond", "decision_rule": DECISION,
                                         "clips": [strip_result(r) for r in rows]})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- decision


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """(kbps, quality) points on the upper envelope: each better than every cheaper one."""
    front: list[tuple[float, float]] = []
    for rate, quality in sorted(points):
        if not front or quality > front[-1][1]:
            front.append((rate, quality))
    return front


def bd_rate(test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> float | None:
    """Bjøntegaard rate difference of ``test`` against ``anchor`` (fraction; +1.0 is twice the rate), with
    piecewise-cubic (PCHIP) log-rate over the overlapping quality range; None with fewer than two points
    each or an overlap under `DECISION['min_overlap_db']`."""
    from scipy.interpolate import PchipInterpolator

    t, a = pareto(test), pareto(anchor)
    if len(t) < 2 or len(a) < 2:
        return None
    lo, hi = max(t[0][1], a[0][1]), min(t[-1][1], a[-1][1])
    if hi - lo < DECISION["min_overlap_db"]:
        return None

    def area(points: list[tuple[float, float]]) -> float:
        f = PchipInterpolator([q for _, q in points], [math.log(r) for r, _ in points])
        return float(f.integrate(lo, hi))

    return math.exp((area(t) - area(a)) / (hi - lo)) - 1.0


def anchor_quality(anchor: list[tuple[float, float]], rate: float) -> float | None:
    """The envelope's quality at ``rate`` (linear in log rate), its best quality above its range, None below."""
    front = pareto(anchor)
    if rate < front[0][0]:
        return None
    if rate >= front[-1][0]:
        return front[-1][1]
    for (r0, q0), (r1, q1) in zip(front, front[1:]):
        if r0 <= rate <= r1:
            w = (math.log(rate) - math.log(r0)) / (math.log(r1) - math.log(r0))
            return q0 + w * (q1 - q0)
    return None


def clip_verdict(test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    """BD-rate, or with too little overlap: dominated (+inf) if every test point lies below the envelope,
    dominating (-inf) if every one lies above it, unclear (None) otherwise."""
    bd = bd_rate(test, anchor)
    if bd is not None:
        return {"bd_rate": bd, "basis": "bd"}
    above = []
    for rate, quality in test:
        q = anchor_quality(anchor, rate)
        above.append(None if q is None else quality > q)
    if above and all(a is False for a in above):
        return {"bd_rate": math.inf, "basis": "dominated"}
    if above and all(a is True for a in above):
        return {"bd_rate": -math.inf, "basis": "dominates"}
    return {"bd_rate": None, "basis": "unclear"}


def pilot_decision(verdicts: dict[str, dict[str, Any]]) -> str:
    """Rule step 1 for one arm over the pilot clips: ``ruled out``, ``candidate`` or ``unclear``."""
    values = [v.get("bd_rate") for v in verdicts.values()]
    if any(v is not None and v < 0 for v in values) or any(v is not None and 0 <= v < DECISION["ruled_out_bd"]
                                                             for v in values):
        return "candidate"
    if values and all(v is not None and v >= DECISION["ruled_out_bd"] for v in values) and \
            all(v.get("converged", True) for v in verdicts.values()):
        return "ruled out"
    return "unclear"


def group_gate(verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rule step 2: the median clip's BD-rate below 0 (unclear clips excluded and counted)."""
    values = sorted(v["bd_rate"] for v in verdicts.values() if v.get("bd_rate") is not None)
    unclear = sum(v.get("bd_rate") is None for v in verdicts.values())
    median = float(np.median(values)) if values else None
    return {"clips": len(verdicts), "unclear": unclear, "median_bd_rate": median,
            "passes": median is not None and median < DECISION["gate_median_bd"]}


def command_report(args: argparse.Namespace) -> int:
    baselines: dict[str, dict[str, Any]] = {}
    arms: dict[str, dict[str, list[dict[str, Any]]]] = {"nvrc": {}, "cond": {}, "warp": {}, "dcvc": {}}
    runs = []
    for path in args.result:
        doc = json.loads(Path(path).read_text())
        if "clips" not in doc and doc.get("kind") == "baselines":  # one clip's baselines.json (a stopped job's)
            doc = {"kind": "baselines", "clips": [summarize_row(doc)]}
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "kind": doc["kind"]})
        for row in doc["clips"]:
            if doc["kind"] == "baselines":
                baselines[row["id"]] = row
            elif doc["kind"] == "dcvc":
                arms["dcvc"].setdefault(row["id"], []).extend(row["points"])
            elif doc["kind"] in ("nvrc", "cond"):
                arms[row.get("arm", doc["kind"])].setdefault(row["id"], []).append(row)
    report: dict[str, Any] = {"runs": runs, "decision_rule": DECISION, "arms": {}}
    for arm, by_clip in arms.items():
        verdicts = {}
        for clip_id, rows in sorted(by_clip.items()):
            if clip_id not in baselines:
                continue
            group = baselines[clip_id]["group"]
            tier = DECISION["gate_tier"][group]
            anchor = [(p["kbps"], p["score"][tier]["psnr_v"]) for p in baselines[clip_id]["points"]
                      if p["score"][tier]["psnr_v"] is not None]
            test = [(r["kbps"], r["score"][tier]["psnr_v"]) for r in rows if r["score"][tier]["psnr_v"] is not None]
            verdict = clip_verdict(test, anchor)
            verdict.update(group=group, tier=tier, points=sorted(test), anchor=pareto(anchor),
                           converged=all(r.get("converged", {}).get("converged", True) for r in rows))
            verdicts[clip_id] = verdict
        pilot = {k: v for k, v in verdicts.items() if k in DECISION["pilot"]}
        groups = {g: group_gate({k: v for k, v in verdicts.items() if v["group"] == g})
                  for g in ("visor-window", "visor-long")}
        report["arms"][arm] = {"clips": verdicts, "pilot": pilot_decision(pilot) if pilot else None,
                               "groups": groups}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "g5-report.json", report)
    print(json.dumps({a: {"pilot": r["pilot"], "groups": r["groups"]} for a, r in report["arms"].items()},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------- validation


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def published_has(stage: Path, member: str) -> bool:
    """Whether the stage's ``published.tar`` holds ``member`` (as ``./member`` or ``member``)."""
    import tarfile

    tar = stage / "published.tar"
    if not tar.is_file():
        return False
    with tarfile.open(tar) as archive:
        names = {n.removeprefix("./") for n in archive.getnames()}
    return member in names


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g5.json").read_text())
    rows = result["clips"]
    checks: dict[str, bool] = {"clips_present": bool(rows)}
    if result["kind"] != "baselines":
        # First run of NVRC and of arm B on a GPU class: the intended device (Ada or A6000, sm_86 and up).
        checks["intended_gpu"] = all(r.get("gpu_capability") and tuple(r["gpu_capability"]) >= (8, 6) for r in rows)
    if result["kind"] == "nvrc-s1":
        checks["every_epoch_logged"] = all(len(r["curve_s1"]) == r["epochs"]["s1"] for r in rows)
        checks["training_improves"] = all(r["curve_s1"][-1]["psnr"] > r["curve_s1"][0]["psnr"] for r in rows)
        checks["checkpoint_published"] = published_has(stage, "publish/nvrc-s1/s1/checkpoints/0000/pytorch_model.bin")
        checks["nvrc_vendored"] = all(r["provenance"].get("revision") for r in rows)
        return checks
    if result["kind"] == "dcvc":
        checks["every_point"] = all(len(r["points"]) == len(result["qps"]) for r in rows)
        checks["decoded_every_frame"] = all(p["decoded_frames"] == r["frames"] for r in rows for p in r["points"])
        checks["deterministic_decode"] = all(p["deterministic"] for r in rows for p in r["points"])
        checks["decoder_matches_encoder_intra"] = all(p["decoder_matches_encoder_intra"] for r in rows
                                                      for p in r["points"])
        checks["rate_rises_with_qp"] = all(all(a["kbps"] < b["kbps"] for a, b in zip(r["points"], r["points"][1:]))
                                           for r in rows)
        checks["scores_finite"] = all(finite(p["score"]["dataset"]["psnr_v"]) for r in rows for p in r["points"])
        return checks
    if result["kind"] == "baselines":
        crfs = result["crfs"]
        checks["every_point"] = all(len(r["points"]) == 2 * len(crfs) for r in rows)
        checks["decoded_every_frame"] = all(p["decoded_frames"] == r["frames"] for r in rows for p in r["points"])
        checks["rate_falls_with_crf"] = all(
            all(a["kbps"] > b["kbps"] for a, b in zip(pts, pts[1:]))
            for r in rows for name in ("frame", "filled")
            for pts in [sorted((p for p in r["points"] if p["input"] == name), key=lambda p: p["crf"])])
        checks["scores_finite"] = all(finite(p["score"]["dataset"]["psnr_v"]) for r in rows for p in r["points"]
                                      if r["dense_frames"])
        checks["ceiling_above_points"] = all(
            r["ceiling_420"]["dataset"]["psnr_v"] >= max(p["score"]["dataset"]["psnr_v"] for p in r["points"]) - 1e-6
            for r in rows if r["dense_frames"])
        checks["v_share_plausible"] = all(0.3 < r["v_share"] < 1.0 for r in rows)
    else:
        checks["rate_positive"] = all(finite(r["kbps"]) and r["kbps"] > 0 for r in rows)
        checks["scores_finite"] = all(finite(r["score"]["dataset"]["psnr_v"]) for r in rows if r["dense_frames"])
        checks["quality_above_floor"] = all(r["score"]["dataset"]["psnr_v"] > 10 for r in rows if r["dense_frames"])
        trained = [r for r in rows if r.get("arm") != "warp"]
        checks["render_timed"] = all(finite(r["render_ms_per_frame"]) for r in trained)
        checks["training_logged"] = all(r.get("curve_s1") or r.get("curve") for r in trained)
        if result["kind"] == "nvrc":
            checks["bitstream_written"] = all(r["bitstream_bytes"] > 0 for r in rows)
            checks["nvrc_vendored"] = all(r["provenance"].get("revision") for r in rows)
        else:
            model_rows = [r for r in rows if r.get("arm") == "cond"]
            checks["both_variants"] = {r.get("arm") for r in rows} == {"warp", "cond"}
            checks["bits_counted"] = all(all(finite(v) and v > 0 for v in r["bits"].values()) for r in rows)
            checks["model_adds_bits"] = all(r["kbps"] > w["kbps"] for r in model_rows for w in rows
                                            if w.get("arm") == "warp")
            checks["quantized_close_to_float"] = all(
                abs(r["score"]["dataset"]["psnr_v"] - r["score_float_weights"]["dataset"]["psnr_v"]) < 1.0
                for r in model_rows if r["dense_frames"])
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


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prepared", required=True, help="G1d's extracted prepared archive")
        p.add_argument("--clips", required=True, help="G1's clips.json")
        p.add_argument("--visor-fill", default=None, help="B1b's fill (the stretches' dense masks)")
        p.add_argument("--select", required=True)
        p.add_argument("--limit-frames", type=int, default=0)

    base = sub.add_parser("baselines")
    common(base)
    base.add_argument("--crfs", default=",".join(map(str, CRFS)))
    base.set_defaults(func=command_baselines)
    nv = sub.add_parser("nvrc")
    common(nv)
    nv.add_argument("--lamb", type=float, required=True)
    nv.add_argument("--epochs", default="360,30", help="stage 1 and stage 2 epochs")
    nv.add_argument("--lpips-backbone", default=None)
    nv.add_argument("--part", choices=("s1", "s2", "both"), default="both")
    nv.add_argument("--s1-from", default=None, help="part 1's extracted publish (with --part s2)")
    nv.add_argument("--dynamo-s1", default="inductor", help="stage 1's torch.compile backend (or no)")
    nv.add_argument("--dynamo-s2", default="no", help="stage 2's: 30 epochs do not repay a compile")
    nv.set_defaults(func=command_nvrc)
    cond = sub.add_parser("cond")
    common(cond)
    cond.add_argument("--lamb", type=float, required=True)
    cond.add_argument("--crf", type=int, required=True, help="SVT-AV1 CRF of the refresh stream")
    cond.add_argument("--epochs", type=int, default=300)
    cond.add_argument("--lpips-backbone", default=None)
    cond.set_defaults(func=command_cond)
    dc = sub.add_parser("dcvc")
    common(dc)
    dc.add_argument("--qps", default=",".join(map(str, DCVC_QPS)))
    dc.add_argument("--image-ckpt", required=True)
    dc.add_argument("--video-ckpt", required=True)
    dc.add_argument("--lpips-backbone", default=None)
    dc.set_defaults(func=command_dcvc)
    val = sub.add_parser("validate")
    val.add_argument("--stage", default=None)
    val.set_defaults(func=command_validate)
    rep = sub.add_parser("report")
    rep.add_argument("--result", nargs="+", required=True, help="g5.json files, or one clip's baselines.json")
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
