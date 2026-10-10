"""PLAN step G5b: a scene-adapted neural background, DCVC-UF fine-tuned on the scene.

    python -m experiments.background.g5b rescore --prepared DIR --clips CLIPS.JSON --visor-fill DIR \\
        --streams DIR,DIR,... --select all|ID,ID --limit-frames N --lpips-backbone PATH
    python -m experiments.background.g5b finetune --prepared DIR --clips CLIPS.JSON --visor-fill DIR \\
        --select ID [--fit-clip ID] --fit heldout|scene --steps N --lpips-weight W \\
        --image-ckpt PATH --video-ckpt PATH --lpips-backbone PATH --limit-frames N
    python -m experiments.background.g5b warp --prepared DIR --clips CLIPS.JSON --select ID \\
        --refresh 0.3,1.0 --crfs 40,50,59 --lpips-backbone PATH
    python -m experiments.background.g5b validate
    python -m experiments.background.g5b report --result g5b.json ... --out DIR

The scene model is delivered beforehand and not billed (PLAN G5b), so it is
fitted on frames other than the ones it is scored on. On each of G1d's 10
VISOR stretches (120 s at 10 frames/s) the held-out segment is G5's 240-frame
excerpt (24 s), which holds the stretch's VISOR dense-mask window; the scene
frames are the rest of the stretch less `GUARD_S` on each side. Every method
codes the held-out excerpt's ``filled`` input and is scored as in G5
(`experiments.background.g5.score`), with LPIPS on V the gate.

``rescore`` decodes G5's stored SVT-AV1 streams and adds LPIPS. ``finetune``
fine-tunes DCVC-UF HT-L (`experiments.background.g5b_train`, in DCVC's tree) on
the held-out excerpt itself (the upper bound) or on the scene frames (the
component), of the same stretch or of another (the cross-scene control), then
codes the excerpt with the fine-tuned model through B2's worker. ``warp`` is
G1e's refresh check at sparser intervals (arm B of G5 without a model). Each
writes ``g5b.json`` in ``PS_STAGE_DIR``; ``report`` applies `DECISION`.
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
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.background import g1, g1d, g5
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any fleet run (docs/experiments.md, G5b entry).
DECISION: dict[str, Any] = {
    "quality": "lpips_v",  # mean over the tier's frames of LPIPS on V, foreground pasted back (G2)
    "quality_scale": "q = -100 * LPIPS_V, so higher is better and G5's 0.5 overlap is 0.005 LPIPS",
    "gate_tier": "dataset",  # VISOR dense masks (with B1b's fill) on the excerpt's window frames
    "anchor": "upper envelope of SVT-AV1 frame and filled (G5's streams, rescored)",
    "reference": "DCVC-UF HT-L off the shelf on filled (G5's dcvc command)",
    "pilot": ["visor-long/P26_02", "visor-long/P06_03"],
    "beats_reference_bd": -0.10,  # adaptation must save at least 10% against the unadapted model
    "converged_max_drop": 0.03,  # monitor loss falls less than 3% over the last fifth of the steps
    "control_share": 2 / 3,  # cross-scene gain >= 2/3 of the scene gain: the gain is not the scene's
    "gate_median_bd": 0.0,
}
GUARD_S = 5.0
QPS = g5.DCVC_QPS
TRAIN = {"lr": 1e-5, "batch": 4, "patch": [512, 512], "groups": 4, "lambdas": [1.0, 768.0],
         "monitor_qps": [9, 27, 45, 63], "monitor_batch": 2, "log_every": 25, "seed": 0}


# ----------------------------------------------------------------- the scene (a whole stretch)


def load_scene(clip_dir: Path, spec_clip: dict[str, Any], visor_fill: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """All analysis frames of a stretch, their loss masks (V: not G1's ``sam_text`` foreground, and not the
    dense masks on the window frames) and the held-out excerpt's start (`g5.load_clip`'s)."""
    meta = json.loads((clip_dir / "meta.json").read_text())
    if meta["group"] != "visor-long":
        raise SystemExit(f"{meta['id']}: a scene is a stretch")
    total = int(meta["frames"])
    raw, _ = g1.window_dense(g1.plan(spec_clip, meta["limit_seconds"]), visor_fill)
    dense = {i: g1.dilate(m) for i, m in raw.items()}
    start = g5.excerpt_start(total, sorted(dense), min(g5.EXCERPT_FRAMES, total))
    frames = np.empty((total, g5.HEIGHT, g5.WIDTH, 3), np.uint8)
    got = 0
    for i, rgb in enumerate(g1d.decode_ffv1(clip_dir / "frames.mkv", 4)):
        frames[i] = rgb
        got += 1
    if got != total:
        raise RuntimeError(f"{meta['id']}: decoded {got} of {total} frames")
    keep = ~np.asarray(g1d.load_foreground(clip_dir / "foreground.npz"))
    for i, mask in dense.items():
        keep[i] = ~mask
    return frames, keep, start


def fit_indices(total: int, start: int, length: int, fps: Fraction, fit: str) -> list[int]:
    """``heldout``: the excerpt itself (the upper bound). ``scene``: every other frame, less `GUARD_S`
    on each side of the excerpt (the component)."""
    if fit == "heldout":
        return list(range(start, start + length))
    guard = round(GUARD_S * float(fps))
    return [i for i in range(total) if not start - guard <= i < start + length + guard]


def inpaint_one(task: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    import cv2

    cv2.setNumThreads(1)
    rgb, keep = task
    hole = (~keep).astype(np.uint8)
    return cv2.inpaint(rgb, hole, g5.TELEA_RADIUS, cv2.INPAINT_TELEA) if hole.any() else rgb


def inpaint_frames(frames: np.ndarray, keep: np.ndarray, indices: list[int], workers: int) -> np.ndarray:
    """The ``filled`` input (`g5.inpaint`) of ``indices``, in parallel; other frames stay as they are."""
    out = frames.copy()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, rgb in zip(indices, pool.map(inpaint_one, ((frames[i], keep[i]) for i in indices), chunksize=8)):
            out[i] = rgb
    return out


# ----------------------------------------------------------------- rescore (SVT-AV1 streams with LPIPS)


def find_stored(stream_roots: list[Path], clip_id: str) -> Path:
    for root in stream_roots:
        d = root / "publish" / "clips" / g1.safe(clip_id)
        if (d / "baselines.json").is_file():
            return d
    raise SystemExit(f"{clip_id}: no stored baselines among {[str(r) for r in stream_roots]}")


def command_rescore(args: argparse.Namespace) -> int:
    import torch

    from src.codecs import svtav1

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs = g5.clip_dirs(Path(args.prepared))
    roots = [Path(p) for p in args.streams.split(",")]
    net = g5.load_lpips(args.lpips_backbone, "cuda")
    dav1d = svtav1.tool("dav1d")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish) if checkpoints else {}
    rows = []
    for done, clip_id in enumerate((c["id"] for c in g5.select_clips(spec, args.select)), 1):
        if clip_id in restored:
            rows.append({**restored[clip_id]["result"], "restored_from_checkpoint": True})
            progress(done)
            continue
        stored = find_stored(roots, clip_id)
        before = json.loads((stored / "baselines.json").read_text())
        clip = g5.load_clip(dirs[clip_id], by_id[clip_id], Path(args.visor_fill), args.limit_frames)
        work = scratch / "work" / g1.safe(clip_id)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        points = []
        for p in before["points"]:
            stream = stored / "streams" / f"{p['input']}-crf{p['crf']}.ivf"
            if file_sha256(stream) != p["stream_sha256"]:
                raise RuntimeError(f"{stream}: bytes differ from the recorded stream")
            out = work / "decoded.yuv"
            svtav1.run(svtav1.decode_command("dav1d", stream, out, threads=8), 600)
            raw = np.fromfile(out, np.uint8)
            if raw.size != before["frames"] * g5.WIDTH * g5.HEIGHT * 3 // 2:
                raise RuntimeError(f"{stream}: decoded {raw.size} bytes")
            rgb = g5.yuv420_to_rgb(raw.reshape(before["frames"], g5.HEIGHT * 3 // 2, g5.WIDTH)[:clip.n])
            points.append({**{k: v for k, v in p.items() if k not in ("score", "encode_command")},
                           "psnr_v_g5": p["score"]["dataset"]["psnr_v"], "score": g5.score(clip, rgb, net, "cuda")})
            out.unlink()
        row = {**clip.record, "kind": "rescore", "points": points, "stored": str(stored),
               "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0))}
        target = publish / "clips" / g1.safe(clip_id)
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "rescore.json", row)
        summary = {**row, "points": [{**q, "score": g5.strip_frames(q["score"])} for q in points]}
        if checkpoints is not None:
            g1.save_clip(checkpoints, target, clip_id, summary, None)
        rows.append(summary)
        shutil.rmtree(work, ignore_errors=True)
        progress(done)
    write_json(stage_dir() / "g5b.json", {"kind": "rescore", "select": args.select, "limit_frames": args.limit_frames,
                                          "dav1d": dav1d, "restored_clips": sorted(restored),
                                          "decision_rule": DECISION, "clips": rows})
    return 0


# ----------------------------------------------------------------- finetune (the oracle and the component)


def trainer_command(prefix: Path, plan: Path, report: Path) -> tuple[list[str], dict[str, str], Path]:
    """`experiments.background.g5b_train` in DCVC's vendored tree (as `src.codecs.dcvc_uf_worker`)."""
    tree = prefix / "opt" / "DCVC"
    script = Path(__file__).resolve().parent / "g5b_train.py"
    env = {"PYTHONPATH": str(tree), "PYTHONNOUSERSITE": "1"}
    return [str(prefix / "bin" / "python"), str(script), "--plan", str(plan), "--report", str(report)], env, tree


def monitor_converged(curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Converged when the monitor loss fell less than `DECISION['converged_max_drop']` over the last fifth."""
    if len(curve) < 3:
        return {"converged": False, "reason": "fewer than 3 monitor points"}
    last = curve[-1]
    cut = last["step"] * 0.8
    earlier = min((p for p in curve if p["step"] <= cut), key=lambda p: abs(p["step"] - cut))
    drop = (earlier["loss"] - last["loss"]) / earlier["loss"]
    return {"converged": drop < DECISION["converged_max_drop"], "drop_last_fifth": drop,
            "from_step": earlier["step"], "to_step": last["step"]}


def command_finetune(args: argparse.Namespace) -> int:
    import torch

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs = g5.clip_dirs(Path(args.prepared))
    clip_id = g5.only_clip(spec, args.select)
    fit_id = g5.only_clip(spec, args.fit_clip) if args.fit_clip else clip_id
    if args.fit == "heldout" and fit_id != clip_id:
        raise SystemExit("the upper bound fits on the scored clip's own excerpt")
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "finetune"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    visor_fill = Path(args.visor_fill)
    clip = g5.load_clip(dirs[clip_id], by_id[clip_id], visor_fill, args.limit_frames)

    # Training frames: the fitted clip's whole stretch (its excerpt only for the upper bound), filled.
    frames, keep, start = load_scene(dirs[fit_id], by_id[fit_id], visor_fill)
    if fit_id == clip_id and start != clip.start:
        raise RuntimeError(f"{clip_id}: scene excerpt {start} != G5's {clip.start}")
    fit = fit_indices(len(frames), start, g5.EXCERPT_FRAMES if args.limit_frames <= 0 else clip.n, clip.fps, args.fit)
    if args.limit_frames > 0:  # smoke: a few dozen fitted frames
        fit = fit[:max(args.limit_frames, 24)]
    workers = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 4))
    filled = inpaint_frames(frames, keep, fit, workers)
    np.save(work / "frames.npy", g5.rgb_to_yuv420(filled))
    np.save(work / "keep.npy", keep)
    del frames, filled
    prepare_seconds = time.time() - began

    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    plan = {**TRAIN, "frames": str(work / "frames.npy"), "keep": str(work / "keep.npy"), "fit": fit,
            "steps": args.steps, "monitor_every": max(1, args.steps // 10), "save_every": max(1, args.steps // 10),
            "lpips_weight": args.lpips_weight, "lpips_backbone": args.lpips_backbone,
            "image_ckpt": args.image_ckpt, "image_sha256": file_sha256(Path(args.image_ckpt)),
            "video_ckpt": args.video_ckpt, "video_sha256": file_sha256(Path(args.video_ckpt)),
            "out": str(work / "model"), "resume_dir": str(checkpoints) if checkpoints else None}
    if args.limit_frames > 0:  # smoke: short sequences, small patches
        plan.update(groups=1, patch=[256, 256], monitor_batch=1, log_every=1)
    write_json(work / "plan.json", plan)
    command, env, cwd = trainer_command(Path(sys.prefix), work / "plan.json", work / "trained.json")
    log_path = work / "train.log"
    with log_path.open("w") as log:
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              timeout=args.timeout)
    log_text = log_path.read_text()
    if done.returncode:
        raise RuntimeError(f"fine-tuning failed ({done.returncode}):\n{log_text[-4000:]}")
    trained = json.loads((work / "trained.json").read_text())
    train_log = [json.loads(line) for line in log_text.splitlines() if line.startswith("{")]

    # The held-out excerpt, coded by the fine-tuned model (and nothing else changed).
    net = g5.load_lpips(args.lpips_backbone, "cuda")
    source = work / "filled.yuv"
    g5.rgb_to_yuv420(g5.inpaint(clip)).tofile(source)
    coder = SimpleNamespace(image_ckpt=args.image_ckpt, video_ckpt=trained["checkpoint"])
    publish = scratch / "publish" / "finetune"
    shutil.rmtree(publish, ignore_errors=True)
    (publish / "streams").mkdir(parents=True)
    points = []
    for qp in [int(q) for q in args.qps.split(",")]:
        rec = g5.code_dcvc(coder, source, clip.n, qp, work / f"qp{qp}")  # type: ignore[arg-type]
        decoded = np.fromfile(rec.pop("decoded"), np.uint8).reshape(clip.n, g5.HEIGHT * 3 // 2, g5.WIDTH)
        shutil.copyfile(work / f"qp{qp}" / "stream.psdc", publish / "streams" / f"filled-qp{qp}.psdc")
        points.append({"codec": "dcvc-uf-htl-ft", "input": "filled", "qp": qp,
                       "kbps": rec["bytes"] * 8 / clip.duration / 1000, "decoded_frames": len(decoded),
                       "decode_ms_per_frame": 1000 * min(rec["decode_seconds"]) / clip.n, **rec,
                       "score": g5.score(clip, g5.yuv420_to_rgb(decoded), net, "cuda")})
        shutil.rmtree(work / f"qp{qp}")
    shutil.copyfile(trained["checkpoint"], publish / "video_ft.pth.tar")
    shutil.copyfile(log_path, publish / "train.log")
    arm = f"{'upper' if args.fit == 'heldout' else ('scene' if fit_id == clip_id else 'other')}-" \
          f"{'lpips' if args.lpips_weight > 0 else 'mse'}"
    row = {**clip.record, "kind": "finetune", "arm": arm, "fit": args.fit, "fit_clip": fit_id,
           "fit_frames": len(fit), "guard_s": GUARD_S if args.fit == "scene" else None,
           "lpips_weight": args.lpips_weight, "steps": args.steps, "train": {**TRAIN, **{k: plan[k] for k in (
               "groups", "patch", "monitor_batch", "monitor_every")}},
           "model": {k: trained[k] for k in ("checkpoint_sha256", "checkpoint_bytes", "parameters")},
           "model_bytes_fp16": 2 * trained["parameters"], "billed": False,
           "curve": trained["curve"], "converged": monitor_converged(trained["curve"]),
           "train_seconds": trained["train_seconds"], "seconds_per_step": trained["seconds_per_step"],
           "peak_memory_gib": trained["peak_memory_gib"], "skipped_nonfinite": trained["skipped_nonfinite"],
           "train_log_tail": train_log[-5:], "prepare_seconds": round(prepare_seconds, 1),
           "seconds": round(time.time() - began, 1), "points": points,
           "base_checkpoints": {"image_sha256": plan["image_sha256"], "video_sha256": plan["video_sha256"]},
           "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
           "trainer_gpu": trained["gpu"], "trainer_torch": trained["torch"]}
    write_json(publish / "result.json", row)
    write_json(stage_dir() / "g5b.json", {"kind": "finetune", "qps": [p["qp"] for p in points],
                                          "decision_rule": DECISION,
                                          "clips": [{**row, "points": [{**p, "score": g5.strip_frames(p["score"])}
                                                                       for p in points]}]})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- warp (G1e's refresh, sparser)


def command_warp(args: argparse.Namespace) -> int:
    """G5's ``warp`` arm (refreshes coded by SVT-AV1 as their own stream, the frames between them the last
    refresh warped by the oracle flow at `g5.MOTION_BYTES` per frame) at each ``--refresh`` interval."""
    import torch

    from experiments.background import g5_cond

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = g5.only_clip(spec, args.select)
    clip = g5.load_clip(g5.clip_dirs(Path(args.prepared))[clip_id], by_id[clip_id],
                        Path(args.visor_fill) if args.visor_fill else None, args.limit_frames)
    net = g5.load_lpips(args.lpips_backbone, "cuda")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "warp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rows = []
    for every in [float(r) for r in args.refresh.split(",")]:
        step = max(1, round(every * float(clip.fps)))
        refresh = list(range(0, clip.n, step))
        refs = g5.refresh_reference(clip.n, clip.fps, every)
        between = [t for t in range(clip.n) if t not in set(refresh)]
        t0 = time.time()
        flows = g5_cond.oracle_flows(clip.frames, [refs[t] if t in set(between) else None for t in range(clip.n)])
        flow_seconds = time.time() - t0
        rgb_to = work / f"refresh-{step}.yuv"
        g5.rgb_to_yuv420(clip.frames[refresh]).tofile(rgb_to)
        for crf in [int(c) for c in args.crfs.split(",")]:
            rec = g5.code_svtav1({"work": str(work / f"r{step}-crf{crf}"), "source": str(rgb_to),
                                  "fps": str(clip.fps / step), "frames": len(refresh), "crf": crf, "threads": 8,
                                  "cpus": None})
            raw = np.fromfile(rec["decoded"], np.uint8).reshape(len(refresh), g5.HEIGHT * 3 // 2, g5.WIDTH)
            fixed = dict(zip(refresh, g5.yuv420_to_rgb(raw)))
            warped = g5_cond.warp_only(fixed, flows, refs, clip.n, "cuda")
            bits = {"refresh": 8 * rec["payload_bytes"], "motion": 8 * g5.MOTION_BYTES * len(between)}
            rows.append({**clip.record, "kind": "warp", "arm": f"warp-{every:g}s", "refresh_s": every,
                         "refresh_every": step, "refresh_frames": len(refresh), "crf": crf, "bits": bits,
                         "kbps": sum(bits.values()) / clip.duration / 1000, "stream_sha256": rec["stream_sha256"],
                         "flow_seconds": round(flow_seconds, 1), "score": g5.strip_frames(
                             g5.score(clip, warped, net, "cuda")),
                         "gpu": torch.cuda.get_device_name(0),
                         "gpu_capability": list(torch.cuda.get_device_capability(0))})
            shutil.rmtree(work / f"r{step}-crf{crf}")
    write_json(stage_dir() / "g5b.json", {"kind": "warp", "decision_rule": DECISION, "clips": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- decision


def curve_points(points: list[dict[str, Any]], tier: str, metric: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        value = p["score"].get(tier, {}).get(metric)
        if value is not None:
            out.append((float(p["kbps"]), -100.0 * value if metric == "lpips_v" else float(value)))
    return out


def anchor_rate(anchor: list[tuple[float, float]], quality: float) -> float | None:
    """The envelope's rate at ``quality`` (linear in log rate), None outside its range."""
    front = g5.pareto(anchor)
    for (r0, q0), (r1, q1) in zip(front, front[1:]):
        if q0 <= quality <= q1:
            w = (quality - q0) / (q1 - q0) if q1 > q0 else 0.0
            return math.exp(math.log(r0) + w * (math.log(r1) - math.log(r0)))
    return None


def payback_seconds(model_bytes: int, test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> list:
    """Per test point: the playback after which the model's bytes are repaid by its rate saving at equal
    quality (None where the anchor has no equal-quality rate or the test point saves nothing)."""
    out = []
    for rate, quality in test:
        a = anchor_rate(anchor, quality)
        saving = None if a is None else a - rate
        out.append({"kbps": rate, "anchor_kbps": a,
                    "seconds": 8 * model_bytes / (1000 * saving) if saving and saving > 0 else None})
    return out


def verdict(test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    v = g5.clip_verdict(test, anchor)
    return {**v, "points": sorted(test), "anchor": g5.pareto(anchor)}


def stage_decision(arm_verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pilot rule for the upper bound or the component: passes if on some pilot stretch it beats SVT-AV1
    (BD < 0) and the unadapted DCVC-UF (BD < `DECISION['beats_reference_bd']`), on LPIPS_V, converged;
    fails if on every pilot stretch it does neither."""
    rows = [arm_verdicts[c] for c in DECISION["pilot"] if c in arm_verdicts]
    if len(rows) < len(DECISION["pilot"]):
        return {"decision": "incomplete", "pilot_clips": len(rows)}

    def below(v: dict[str, Any], bound: float) -> bool | None:
        return None if v.get("bd_rate") is None else v["bd_rate"] < bound

    wins = [below(r["vs_svtav1"], 0.0) is True and below(r["vs_dcvc"], DECISION["beats_reference_bd"]) is True
            and r["converged"] for r in rows]
    losses = [below(r["vs_svtav1"], 0.0) is False and below(r["vs_dcvc"], DECISION["beats_reference_bd"]) is False
              for r in rows]
    decision = "passes" if any(wins) else ("fails" if all(losses) and all(r["converged"] for r in rows)
                                           else "unclear")
    return {"decision": decision, "pilot_clips": len(rows)}


def command_report(args: argparse.Namespace) -> int:
    anchors: dict[str, list[dict[str, Any]]] = {}
    reference: dict[str, list[dict[str, Any]]] = {}
    arms: dict[str, dict[str, list[dict[str, Any]]]] = {}
    runs = []
    for path in args.result:
        doc = json.loads(Path(path).read_text())
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "kind": doc["kind"]})
        for row in doc["clips"]:
            if doc["kind"] == "rescore":
                anchors[row["id"]] = row["points"]
            elif doc["kind"] == "dcvc":
                reference.setdefault(row["id"], []).extend(row["points"])
            elif doc["kind"] == "finetune":
                arms.setdefault(row["arm"], {}).setdefault(row["id"], []).append(row)
            elif doc["kind"] == "warp":
                arms.setdefault(row["arm"], {}).setdefault(row["id"], []).append({**row, "points": [row]})
    tier = DECISION["gate_tier"]
    report: dict[str, Any] = {"runs": runs, "decision_rule": DECISION, "arms": {}}
    for arm, by_clip in sorted({"dcvc": {k: [{"points": v}] for k, v in reference.items()}, **arms}.items()):
        clips = {}
        for clip_id, rows in sorted(by_clip.items()):
            if clip_id not in anchors:
                continue
            points = [p for r in rows for p in r["points"]]
            entry: dict[str, Any] = {}
            for metric in ("lpips_v", "psnr_v"):
                anchor = curve_points(anchors[clip_id], tier, metric)
                test = curve_points(points, tier, metric)
                entry[metric] = {"vs_svtav1": verdict(test, anchor)}
                if arm != "dcvc" and clip_id in reference:
                    entry[metric]["vs_dcvc"] = verdict(test, curve_points(reference[clip_id], tier, metric))
            entry["own_tier_lpips_vs_svtav1"] = verdict(curve_points(points, "own", "lpips_v"),
                                                        curve_points(anchors[clip_id], "own", "lpips_v"))
            entry["converged"] = all(r.get("converged", {}).get("converged", True) for r in rows)
            models = [r["model_bytes_fp16"] for r in rows if r.get("model_bytes_fp16")]
            if models:
                entry["payback_vs_svtav1"] = payback_seconds(models[0], curve_points(points, tier, "lpips_v"),
                                                             curve_points(anchors[clip_id], tier, "lpips_v"))
            clips[clip_id] = entry
        gate = {c: {"vs_svtav1": e["lpips_v"]["vs_svtav1"], "vs_dcvc": e["lpips_v"].get("vs_dcvc", {}),
                    "converged": e["converged"]} for c, e in clips.items()}
        values = sorted(e["lpips_v"]["vs_svtav1"]["bd_rate"] for e in clips.values()
                        if e["lpips_v"]["vs_svtav1"]["bd_rate"] is not None)
        report["arms"][arm] = {
            "clips": clips, "pilot": stage_decision(gate) if arm != "dcvc" else None,
            "median_bd_vs_svtav1": float(np.median(values)) if values else None, "clips_scored": len(clips),
            "unclear": sum(e["lpips_v"]["vs_svtav1"]["bd_rate"] is None for e in clips.values())}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "g5b-report.json", report)
    print(json.dumps({a: {k: r[k] for k in ("pilot", "median_bd_vs_svtav1", "clips_scored", "unclear")}
                      for a, r in report["arms"].items()}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------- validation


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g5b.json").read_text())
    rows = result["clips"]
    checks: dict[str, bool] = {"clips_present": bool(rows),
                               "intended_gpu": all(r.get("gpu_capability") and tuple(r["gpu_capability"]) >= (8, 6)
                                                   for r in rows)}
    if result["kind"] == "rescore":
        checks["every_point"] = all(len(r["points"]) == 2 * len(g5.CRFS) for r in rows)
        checks["lpips_scored"] = all(g5.finite(p["score"]["dataset"]["lpips_v"]) for r in rows for p in r["points"]
                                     if r["dense_frames"])
        checks["psnr_matches_g5"] = all(abs(p["score"]["dataset"]["psnr_v"] - p["psnr_v_g5"]) < 1e-6
                                        for r in rows for p in r["points"]
                                        if r["dense_frames"] and not result["limit_frames"])
        checks["lpips_falls_with_rate"] = all(
            all(a["score"]["dataset"]["lpips_v"] >= b["score"]["dataset"]["lpips_v"] - 0.01 for a, b in zip(pts, pts[1:]))
            for r in rows if r["dense_frames"] for name in ("frame", "filled")
            for pts in [sorted((p for p in r["points"] if p["input"] == name), key=lambda p: p["kbps"])])
    elif result["kind"] == "finetune":
        checks["every_point"] = all(len(r["points"]) == len(result["qps"]) for r in rows)
        checks["decoded_every_frame"] = all(p["decoded_frames"] == r["frames"] for r in rows for p in r["points"])
        checks["deterministic_decode"] = all(p["deterministic"] for r in rows for p in r["points"])
        checks["decoder_matches_encoder_intra"] = all(p["decoder_matches_encoder_intra"] for r in rows
                                                      for p in r["points"])
        checks["rate_rises_with_qp"] = all(all(a["kbps"] < b["kbps"] for a, b in zip(r["points"], r["points"][1:]))
                                           for r in rows)
        checks["lpips_scored"] = all(g5.finite(p["score"]["dataset"]["lpips_v"]) for r in rows for p in r["points"]
                                     if r["dense_frames"])
        checks["trained_every_step"] = all(r["curve"][-1]["step"] == r["steps"] for r in rows)
        checks["monitor_finite"] = all(g5.finite(c["loss"]) for r in rows for c in r["curve"])
        checks["model_changed"] = all(r["model"]["checkpoint_sha256"] != r["base_checkpoints"]["video_sha256"]
                                      for r in rows)
        checks["checkpoint_published"] = g5.published_has(stage, "publish/finetune/video_ft.pth.tar")
    elif result["kind"] == "warp":
        checks["rate_positive"] = all(g5.finite(r["kbps"]) and r["kbps"] > 0 for r in rows)
        checks["scores_finite"] = all(g5.finite(r["score"]["dataset"]["psnr_v"]) and
                                      g5.finite(r["score"]["dataset"]["lpips_v"]) for r in rows)
        checks["sparser_is_cheaper"] = all(
            a["kbps"] > b["kbps"] for a in rows for b in rows if a["crf"] == b["crf"] and a["refresh_s"] < b["refresh_s"])
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
        p.add_argument("--lpips-backbone", required=True)

    rs = sub.add_parser("rescore")
    common(rs)
    rs.add_argument("--streams", required=True, help="comma-separated extracted G5 baselines archives")
    rs.set_defaults(func=command_rescore)
    ft = sub.add_parser("finetune")
    common(ft)
    ft.add_argument("--fit", choices=("heldout", "scene"), required=True)
    ft.add_argument("--fit-clip", default=None, help="another stretch's scene frames (the cross-scene control)")
    ft.add_argument("--steps", type=int, required=True)
    ft.add_argument("--lpips-weight", type=float, default=0.0)
    ft.add_argument("--image-ckpt", required=True)
    ft.add_argument("--video-ckpt", required=True)
    ft.add_argument("--qps", default=",".join(map(str, QPS)))
    ft.add_argument("--timeout", type=float, default=3000)
    ft.set_defaults(func=command_finetune)
    wp = sub.add_parser("warp")
    common(wp)
    wp.add_argument("--refresh", default="0.3,1.0", help="refresh intervals in seconds")
    wp.add_argument("--crfs", default="40,50,59")
    wp.set_defaults(func=command_warp)
    va = sub.add_parser("validate")
    va.add_argument("--stage", default=None)
    va.set_defaults(func=command_validate)
    rp = sub.add_parser("report")
    rp.add_argument("--result", action="append", required=True)
    rp.add_argument("--out", required=True)
    rp.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
