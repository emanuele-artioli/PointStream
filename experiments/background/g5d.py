"""PLAN step G5d (oracle): a venue-trained enhancer of SVT-AV1's decoded background.

    python -m experiments.background.g5d enhance --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID \\
        --prefill DIR --lpips-backbone PATH --dists-backbone PATH --inception PATH --limit-frames N --steps N
    python -m experiments.background.g5d validate

The stream is plain SVT-AV1 (G5's CRFs and preset) of the excerpt's ProPainter
fill; the client holds a small convolutional network that maps each decoded
frame to a better one (NAS, Yeo et al., OSDI 2018: a content-aware network
sent once, the stream unchanged). The network is fitted on pairs of SVT-AV1's
decoded frames and the source, on V only, at each CRF:

* ``excerpt``: on the scored excerpt itself, the oracle (no deployed enhancer
  can do better on these frames);
* ``scene``: on the stretch's other frames (G5b's split, the same footage as
  the scene model's recipe C), each coded by SVT-AV1 at the same CRF.

Each fit is trained with the masked MSE alone (``mse``) and with LPIPS on V
added at equal weight at step 0 (``lpips``). The enhanced excerpt is scored
as every other method (`g5c.full_score`: PSNR, LPIPS, DISTS, flicker, KID on
V); its rate is SVT-AV1's, the network's size is reported for payback. Writes
``g5d.json`` in ``PS_STAGE_DIR``; each finished CRF is checkpointed, and a
timed-out run's salvaged checkpoint can be passed back (``--restore``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5c
from experiments.background.g5b import GUARD_S, fit_indices, load_footage, load_prefilled, load_scene
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: Fixed before any fleet run (docs/experiments.md, G5d entry).
DECISION: dict[str, Any] = {
    "clips": ["visor-long/P26_02", "visor-long/P06_03"],
    "beats_bd": -0.10,  # a metric counts when the BD-rate against SVT-AV1 is this or better
    "metrics_needed": 3,  # of g5c.SELECTION_METRICS, on every clip
    "oracle": "excerpt",  # if no excerpt-fitted arm passes, the enhancer is closed
    "component": "scene",  # if a scene-fitted arm passes, a venue-footage enhancer is built next
}
TRAIN: dict[str, Any] = {"blocks": 8, "channels": 64, "patch": 128, "batch": 16, "lr": 2e-4, "seed": 0,
                         "min_v": 0.25}
FITS = ("excerpt", "scene", "footage")
LOSSES = ("mse", "lpips")


# ----------------------------------------------------------------- the enhancer


def enhancer(blocks: int, channels: int, temporal: bool = False) -> Any:
    """A residual CNN (EDSR-style, no upsampling): the decoded frame plus a learned correction. ``temporal``:
    the previous decoded frame is a second input (6 channels), so the correction can follow it."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(channels, channels, 3, padding=1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.body(x)

    class Enhancer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(6 if temporal else 3, channels, 3, padding=1)
            self.body = nn.Sequential(*[Block() for _ in range(blocks)])
            self.tail = nn.Conv2d(channels, 3, 3, padding=1)
            nn.init.zeros_(self.tail.weight)  # starts as the identity: the decoded frame unchanged
            nn.init.zeros_(self.tail.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.head(x)
            return x[:, :3] + self.tail(h + self.body(h))

    return Enhancer()


def sample_patches(rng: np.random.Generator, keep: np.ndarray, count: int, size: int,
                   min_v: float) -> list[tuple[int, int, int]]:
    """(frame, y, x) of ``count`` patches with at least ``min_v`` of V (the best of 8 draws otherwise)."""
    n, h, w = keep.shape
    out: list[tuple[int, int, int]] = []
    for _ in range(count):
        best, best_share = (0, 0, 0), -1.0
        for _ in range(8):
            i, y, x = int(rng.integers(0, n)), int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
            share = float(keep[i, y:y + size, x:x + size].mean())
            if share > best_share:
                best, best_share = (i, y, x), share
            if share >= min_v:
                break
        out.append(best)
    return out


def masked_mse(out: Any, target: Any, mask: Any) -> Any:
    return ((out - target) ** 2 * mask).sum() / (3 * mask.sum()).clamp_min(1.0)


def masked_lpips(net: Any, out: Any, target: Any, mask: Any) -> Any:
    """LPIPS on V (foreground pasted back from the target), as G2 scores it; images in [0, 1]."""
    import torch

    pasted = mask * out + (1 - mask) * target
    spatial = net(pasted * 2 - 1, target * 2 - 1)
    if spatial.shape[-2:] != mask.shape[-2:]:
        spatial = torch.nn.functional.interpolate(spatial, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return (spatial * mask).sum() / mask.sum().clamp_min(1.0)


def previous(frames: list[int]) -> np.ndarray:
    """Each frame's predecessor in the stream: the frame before it when consecutive, else itself."""
    return np.array([j - 1 if j > 0 and frames[j - 1] == f - 1 else j for j, f in enumerate(frames)])


def fit(decoded: np.ndarray, target: np.ndarray, keep: np.ndarray, loss: str, steps: int, lpips_net: Any,
        device: str, prev: np.ndarray | None = None, *, blocks: int | None = None,
        channels: int | None = None, checkpoint: Path | None = None, stop_after: int | None = None) -> tuple[Any, dict[str, Any]]:
    """The enhancer fitted on (decoded, target) pairs on V (``keep``); with ``prev`` (each pair's predecessor)
    the temporal enhancer, fed the previous decoded frame too."""
    import torch

    torch.manual_seed(TRAIN["seed"])
    rng = np.random.default_rng(TRAIN["seed"])
    blocks = TRAIN["blocks"] if blocks is None else blocks
    channels = TRAIN["channels"] if channels is None else channels
    if blocks < 1 or channels < 1 or steps < 1:
        raise ValueError("positive blocks, channels and steps required")
    net = enhancer(blocks, channels, prev is not None).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAIN["lr"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, steps))
    size = TRAIN["patch"]

    def batch(patches: list[tuple[int, int, int]]) -> tuple[Any, Any, Any]:
        def stack(a: np.ndarray) -> Any:
            crops = np.stack([a[i, y:y + size, x:x + size] for i, y, x in patches])
            return torch.from_numpy(crops).to(device)

        x = stack(decoded).permute(0, 3, 1, 2).float() / 255.0
        if prev is not None:
            before = np.stack([decoded[prev[i], y:y + size, x:x + size] for i, y, x in patches])
            x = torch.cat([x, torch.from_numpy(before).to(device).permute(0, 3, 1, 2).float() / 255.0], 1)
        t = stack(target).permute(0, 3, 1, 2).float() / 255.0
        m = stack(keep)[:, None].float()
        return x, t, m

    fixed = batch(sample_patches(np.random.default_rng(TRAIN["seed"] + 1), keep, TRAIN["batch"], size, TRAIN["min_v"]))
    with torch.no_grad():
        mse0 = float(masked_mse(fixed[0][:, :3], fixed[1], fixed[2]))
        lp0 = float(masked_lpips(lpips_net, fixed[0][:, :3], fixed[1], fixed[2])) if loss == "lpips" else None
    weight = mse0 / lp0 if lp0 else 0.0  # equal to the MSE term at step 0
    signature = {"blocks": blocks, "channels": channels, "steps": steps, "loss": loss,
                 "temporal": prev is not None, "shape": list(decoded.shape), "train": TRAIN}
    curve: list[dict[str, Any]] = []
    resumed, elapsed = 0, 0.0
    if checkpoint is not None and checkpoint.is_file():
        # Keep Adam step counters on CPU; load_state_dict moves moment tensors to each parameter.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("training checkpoint does not match this fit")
        net.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        curve, resumed, elapsed = state["curve"], state["step"], state["elapsed"]
        mse0, lp0, weight = state["mse0"], state["lp0"], state["weight"]
    began = time.time()
    last_step = min(steps, stop_after) if stop_after is not None else steps
    for step in range(resumed + 1, last_step + 1):
        x, t, m = batch(sample_patches(rng, keep, TRAIN["batch"], size, TRAIN["min_v"]))
        out = net(x)
        value = masked_mse(out, t, m)
        if weight:
            value = value + weight * masked_lpips(lpips_net, out, t, m)
        if not torch.isfinite(value):
            raise ValueError(f"nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        schedule.step()
        if step % max(1, steps // 10) == 0 or step == last_step:
            with torch.no_grad():
                curve.append({"step": step, "mse_fixed": float(masked_mse(net(fixed[0]), fixed[1], fixed[2])),
                              "lpips_fixed": float(masked_lpips(lpips_net, net(fixed[0]), fixed[1], fixed[2]))
                              if loss == "lpips" else None})
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_name("." + checkpoint.name)
                with temporary.open("wb") as f:
                    torch.save({"signature": signature, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                                "schedule": schedule.state_dict(), "rng": rng.bit_generator.state,
                                "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                                "step": step, "curve": curve, "elapsed": elapsed + time.time() - began,
                                "mse0": mse0, "lp0": lp0, "weight": weight}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, checkpoint)
    return net.eval(), {"steps": steps, "blocks": blocks, "channels": channels, "resumed_from_step": resumed,
                        "lpips_fixed_start": lp0, "lpips_weight": weight, "mse_fixed_start": mse0, "curve": curve,
                        "seconds": round(elapsed + time.time() - began, 1),
                        "parameters": int(sum(p.numel() for p in net.parameters()))}


def apply(net: Any, decoded: np.ndarray, device: str, temporal: bool = False) -> np.ndarray:
    """The enhanced frames; the temporal enhancer sees each frame's predecessor (the first frame, itself)."""
    import torch

    out = np.empty_like(decoded)
    with torch.inference_mode():
        for first in range(0, len(decoded), 4):
            x = torch.from_numpy(decoded[first:first + 4]).to(device).permute(0, 3, 1, 2).float() / 255.0
            if temporal:
                before = decoded[[max(0, i - 1) for i in range(first, min(first + 4, len(decoded)))]]
                x = torch.cat([x, torch.from_numpy(before).to(device).permute(0, 3, 1, 2).float() / 255.0], 1)
            y = net(x).clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
            out[first:first + 4] = y.cpu().numpy()
    return out


# ----------------------------------------------------------------- the command


def encode(task: dict[str, Any]) -> dict[str, Any]:
    """SVT-AV1 at one CRF (spawned), the decoded frames left on disk."""
    return g5.code_svtav1(task)


def parse_arm(name: str) -> tuple[str, str, bool]:
    """``FIT-LOSS`` or ``FIT-LOSS-t`` (temporal): FIT in `FITS`, LOSS in `LOSSES`."""
    parts = name.split("-")
    if len(parts) not in (2, 3) or parts[0] not in FITS or parts[1] not in LOSSES or parts[2:] not in ([], ["t"]):
        raise ValueError(f"arm {name!r}: FIT-LOSS[-t] with FIT in {FITS} and LOSS in {LOSSES}")
    return parts[0], parts[1], parts[2:] == ["t"]


def footage_frames(dirs: list[str], clip_id: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Step 3's footage of the clip's kitchen (other recordings): the fill (4:2:0), V, and frame ids whose gaps
    mark the cuts between segments."""
    yuv, keep, ids, at = [], [], [], 0
    for meta, yuv_path, keep_path in load_footage(dirs):
        if meta["target"] != clip_id:
            raise SystemExit(f"footage for {meta['target']}, not {clip_id}")
        f, k = np.load(yuv_path, mmap_mode="r"), np.load(keep_path, mmap_mode="r")
        for seg in meta["segments"]:
            yuv.append(np.asarray(f[seg["offset"]:seg["end"]]))
            keep.append(np.asarray(k[seg["offset"]:seg["end"]]))
            ids += list(range(at, at + seg["end"] - seg["offset"]))
            at += seg["end"] - seg["offset"] + 1  # a gap: no predecessor across a cut
    return np.concatenate(yuv), np.concatenate(keep), ids


def command_enhance(args: argparse.Namespace) -> int:
    arms = {name: parse_arm(name) for name in args.arms.split(",")}
    crfs = [int(c) for c in args.crfs.split(",")]
    fits = {f for f, _, _ in arms.values()}
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = g5.only_clip(spec, args.select)
    dirs = g5.clip_dirs(Path(args.prepared))
    clip = g5.load_clip(dirs[clip_id], by_id[clip_id], Path(args.visor_fill), args.limit_frames)
    meta, path = load_prefilled([args.prefill])[clip_id]
    frames, keep, start = load_scene(dirs[clip_id], by_id[clip_id], Path(args.visor_fill))
    if start != clip.start or meta["first"] != 0 or meta["frames"] != len(frames):
        raise SystemExit(f"{clip_id}: the prefill or the scene does not match G5's excerpt")
    filled = np.load(path, mmap_mode="r")
    scene = fit_indices(len(frames), start, g5.EXCERPT_FRAMES, clip.fps, "scene")
    if args.limit_frames > 0:  # smoke: a few dozen scene frames
        scene = scene[:max(2 * args.limit_frames, 48)]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "enhance"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sources = {"excerpt": work / "excerpt.yuv"}
    np.ascontiguousarray(filled[clip.start:clip.start + clip.n]).tofile(sources["excerpt"])
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "excerpt": (clip.frames, clip.keep, previous(list(range(clip.n))))}
    counts = {"excerpt": clip.n}
    if "scene" in fits:
        sources["scene"] = work / "scene.yuv"
        np.ascontiguousarray(filled[scene]).tofile(sources["scene"])
        targets["scene"] = (frames[scene], keep[scene], previous(scene))
        counts["scene"] = len(scene)
    del frames, keep
    if "footage" in fits:  # the target on V is the fill (equal to the source there, through 4:2:0)
        f_yuv, f_keep, f_ids = footage_frames(args.footage, clip_id)
        if args.limit_frames > 0:
            f_yuv, f_keep, f_ids = f_yuv[:48], f_keep[:48], f_ids[:48]
        sources["footage"] = work / "footage.yuv"
        np.ascontiguousarray(f_yuv).tofile(sources["footage"])
        targets["footage"] = (g5.yuv420_to_rgb(f_yuv), f_keep, previous(f_ids))
        counts["footage"] = len(f_yuv)
        del f_yuv

    began = time.time()
    nets = g5c.load_nets(args)
    flows = g5c.source_flows(clip)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g5c.restore(checkpoints)
    if args.restore:  # a timed-out job's salvaged checkpoint.tar (extracted): its finished CRFs, as recorded
        restored = {**g5c.restore(Path(args.restore) / "checkpoint"), **restored}
    allowance = max(4, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 4))
    tasks: list[dict[str, Any]] = [
        {"work": str(work / f"{name}-crf{crf}"), "source": str(src), "fps": str(clip.fps), "frames": counts[name],
         "crf": crf, "threads": 4, "cpus": None}
        for crf in crfs if f"crf{crf}" not in restored for name, src in sources.items()]
    with ProcessPoolExecutor(max_workers=max(1, allowance // 4), mp_context=multiprocessing.get_context("spawn")) as pool:
        coded = dict(zip(((t["crf"], Path(t["source"]).stem) for t in tasks), pool.map(encode, tasks)))
    encode_seconds = time.time() - began
    rows = []
    for done, crf in enumerate(crfs, 1):
        key = f"crf{crf}"
        if key in restored:
            rows.append({**restored[key], "restored_from_checkpoint": True})
            progress(done)
            continue
        ex = coded[(crf, "excerpt")]
        decoded = g5c.read_decoded(Path(ex["decoded"]), clip.n, clip.n)
        pairs = {name: (decoded if name == "excerpt" else
                        g5c.read_decoded(Path(coded[(crf, name)]["decoded"]), counts[name], counts[name]),
                        *targets[name]) for name in fits}
        kbps = 8 * ex["payload_bytes"] / clip.duration / 1000
        row: dict[str, Any] = {"crf": crf, "kbps": kbps, "stream_sha256": ex["stream_sha256"],
                               "fit_kbps": {name: 8 * coded[(crf, name)]["payload_bytes"]
                                            / (counts[name] / float(clip.fps)) / 1000 for name in fits},
                               "plain": {"score": g5c.full_score(clip, decoded, nets, flows)}, "arms": {}}
        for name, (fit_name, loss, temporal) in arms.items():
            dec, target, mask, prev = pairs[fit_name]
            net, record = fit(dec, target, mask, loss, args.steps, nets.lpips, "cuda", prev if temporal else None)
            enhanced = apply(net, decoded, "cuda", temporal)
            row["arms"][name] = {"train": record, "changed_share": float((enhanced != decoded).any(axis=3).mean()),
                                 "score": g5c.full_score(clip, enhanced, nets, flows)}
            del net
        del pairs, decoded
        for name in sources:
            shutil.rmtree(work / f"{name}-crf{crf}", ignore_errors=True)
        g5c.save(checkpoints, key, row)
        rows.append(row)
        progress(done)
    first_arm = next(iter(arms))
    write_json(stage_dir() / "g5d.json", {
        "kind": "enhance", "clip": clip.record, "limit_frames": args.limit_frames, "decision_rule": DECISION,
        "arms_run": list(arms), "crfs": crfs, "train": {**TRAIN, "steps": args.steps}, "fit_frames": counts,
        "guard_s": GUARD_S, "footage": args.footage,
        "prefill": {k: meta[k] for k in ("id", "first", "frames", "yuv420_sha256", "fill")},
        "prefill_sha256": file_sha256(path), "metrics": nets.record, "encode_seconds": round(encode_seconds, 1),
        "seconds": round(time.time() - began, 1), "restored": sorted(restored),
        "restored_archive": args.restore, **g5c.gpu_record(),
        "model_bytes_fp16": 2 * rows[0]["arms"][first_arm]["train"]["parameters"] if rows else None,
        "points": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- validation


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g5d.json").read_text())
    full = not result["limit_frames"]
    rows = result["points"]
    arms = result.get("arms_run") or [f"{f}-{loss}" for f in ("excerpt", "scene") for loss in LOSSES]

    def scored(s: dict[str, Any]) -> bool:
        k = s.get("kid", {})
        return (g5.finite(s["dataset"]["lpips_v"]) and g5.finite(s["dataset"]["psnr_v"])
                and g5.finite(s["flicker"]["flicker_v"]) and g5.finite(k.get("kid_v"))
                and k.get("patches", 0) >= (100 if full else 2))

    checks = {"intended_gpu": tuple(result.get("gpu_capability") or (0, 0)) >= (8, 6),
              "every_crf": sorted(r["crf"] for r in rows) == sorted(result.get("crfs") or g5.CRFS),
              "every_arm": all(sorted(r["arms"]) == sorted(arms) for r in rows),
              "every_point_scored": all(scored(r["plain"]["score"]) and all(scored(a["score"]) for a in r["arms"].values())
                                        for r in rows),
              "enhancer_changes_frames": all(a["changed_share"] > 0.01 for r in rows for a in r["arms"].values()),
              "rates_fall_with_crf": all(a["kbps"] > b["kbps"] for a, b in zip(sorted(rows, key=lambda r: r["crf"]),
                                                                            sorted(rows, key=lambda r: r["crf"])[1:]))}
    if full:  # MSE training lowered the fixed batch's error below the decoded frame's (it starts as the identity)
        checks["training_fits"] = all(a["train"]["curve"][-1]["mse_fixed"] < a["train"]["mse_fixed_start"]
                                      for r in rows for name, a in r["arms"].items() if "-mse" in name)
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
    en = sub.add_parser("enhance")
    en.add_argument("--prepared", required=True, help="G1d's extracted prepared archive")
    en.add_argument("--clips", required=True, help="G1's clips.json")
    en.add_argument("--visor-fill", required=True, help="B1b's fill (the stretches' dense masks)")
    en.add_argument("--select", required=True)
    en.add_argument("--limit-frames", type=int, default=0)
    en.add_argument("--prefill", required=True, help="the stretch's extracted G5c prefill archive")
    en.add_argument("--lpips-backbone", required=True)
    en.add_argument("--dists-backbone", required=True)
    en.add_argument("--inception", required=True)
    en.add_argument("--steps", type=int, required=True)
    en.add_argument("--restore", default=None, help="an extracted checkpoint.tar of a timed-out run of this clip")
    en.add_argument("--arms", default="excerpt-mse,excerpt-lpips,scene-mse,scene-lpips",
                    help="FIT-LOSS[-t]: FIT excerpt, scene or footage; LOSS mse or lpips; -t the temporal enhancer")
    en.add_argument("--crfs", default=",".join(map(str, g5.CRFS)))
    en.add_argument("--footage", nargs="*", default=[], help="G5c step 3's extracted footage archives")
    en.set_defaults(func=command_enhance)
    va = sub.add_parser("validate")
    va.add_argument("--stage", default=None)
    va.set_defaults(func=command_validate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
