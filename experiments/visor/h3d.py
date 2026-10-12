"""PLAN step H3: H3d, a content-aware super-resolution of the hand pixels (NAS; docs/experiments.md, 2026-10-12).

    python -m experiments.visor.h3d oracle --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video V.MP4 ... \\
        --items ID --frames N --base dcvc|svt --steps N --image-ckpt PTH --video-ckpt PTH --inspect BIN \\
        --lpips-backbone PTH --dists-backbone PTH [--points NAME,NAME]
    python -m experiments.visor.h3d validate
    python -m experiments.visor.h3d report --oracle TAR ... --h3c TAR ... --h1-report JSON [--h3b TAR ...] --out DIR

``oracle`` codes the window at a low resolution (*dcvc*: DCVC-UF HT-L off the shelf at ¼ and ½, QP 0, 21, 42, with
H3c's code; *svt*: SVT-AV1 at CRF 55 and 63, ¼ and ⅛, with H3b2's), upscales each decode bicubically to the window's
resolution, and fits one G5d enhancer per point on the window's own (upscaled decode, source) pairs on R, with MSE
plus LPIPS on R: the network a NAS sender would ship, fitted on the scored frames and billed nothing (the oracle).
Each point is scored by H3c's `Scorer` before and after its fit and published as ``point/<item>@<name>`` (checkpointed).
``report`` applies `DECISION` against H3c's bar: the per-item upper concave hull of SVT-AV1 and DCVC-UF off the shelf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import RESAMPLES, bootstrap_mean, hand_kbps, rgb_memmap, seed_of
from experiments.visor.h3b import DISTS_PAD, low_name, scaled_size
from experiments.visor.h3c import PILOT, QPS, decode_windows, downscale_mask, downscale_planes, load_window, point_summary

SEED = "pointstream-h3d"
#: Base points per arm: (QP, scale) for DCVC-UF, (CRF, scale) for SVT-AV1.
BASES: dict[str, list[tuple[float, int]]] = {
    "dcvc": [(float(qp), scale) for scale in (4, 2) for qp in QPS],
    "svt": [(55.0, 4), (63.0, 4), (55.0, 8), (63.0, 8)],
}
#: G5d's enhancer and its settings (`experiments.background.g5d.TRAIN`), fitted on patches centred on R.
TRAIN: dict[str, Any] = {"blocks": 8, "channels": 64, "patch": 128, "batch": 16, "lr": 2e-4, "seed": 0}
APPLY_MARGIN = 48  # px around R's box when applying: the network sees 37 px on each side (18 3x3 convolutions)
METRICS = {"lpips": ("lpips", -1.0, 0.005), "dists": ("dists", -1.0, 0.005), "psnr": ("psnr", 1.0, 0.5)}  # key, sign, min overlap

#: The decision rule, fixed before any fleet run (docs/experiments.md, H3d oracle entry).
DECISION: dict[str, Any] = {
    "quality": "lpips_r (H3c's Scorer: the output inside R, the source outside; LPIPS averaged over R, mean over hand frames)",
    "bar": "per item, the upper concave hull (experiments.background.g5.envelope) on (kbps on R, -LPIPS_R) of H3c's SVT-AV1 "
           "points (B2's four CRFs, H3b2's seven low points) and DCVC-UF HT-L off the shelf (nine points)",
    "test": "each arm's Pareto front over its fitted points",
    "bd": "PCHIP of log rate over the LPIPS range both span (at least 0.005), as a ratio minus 1; otherwise better or worse "
          "at every rate",
    "passes_mean_bd": -0.10,  # with at least `items_below_zero` items below 0
    "items_below_zero": 3,
    "fails_mean_bd": 0.0,  # mean at or above it, or at most `fails_items_below_zero` items below 0
    "fails_items_below_zero": 1,
    "min_items_with_bd": 2,
    "oracle": "passes if either arm passes; fails if both fail; otherwise unclear (to the user)",
}


def point_name(base: str, value: float, scale: int) -> str:
    return f"dcvc@{value:g}/{scale}" if base == "dcvc" else low_name(value, scale)


def h3c_name(base: str, value: float, scale: int) -> str:
    """The same point's name in H3c's units."""
    return f"reference@{value:g}/{scale}" if base == "dcvc" else low_name(value, scale)


# ----------------------------------------------------------------- the network


def region_boxes(regions: list[np.ndarray]) -> list[tuple[int, int, int, int] | None]:
    """R's bounding box per frame (y0, y1, x0, x1), None without hand pixels."""
    out: list[tuple[int, int, int, int] | None] = []
    for r in regions:
        if not r.any():
            out.append(None)
            continue
        ys, xs = np.nonzero(r.any(1))[0], np.nonzero(r.any(0))[0]
        out.append((int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1))
    return out


def sample_on_region(rng: np.random.Generator, regions: list[np.ndarray], boxes: list[Any], hand: list[int], count: int,
                     size: int) -> list[tuple[int, int, int]]:
    """(frame, y, x) of ``count`` patches of ``size``, each centred on a pixel of R (drawn inside R's box until one is R)."""
    height, width = regions[0].shape
    out = []
    for _ in range(count):
        t = hand[int(rng.integers(0, len(hand)))]
        y0, y1, x0, x1 = boxes[t]
        for _ in range(64):
            y, x = int(rng.integers(y0, y1)), int(rng.integers(x0, x1))
            if regions[t][y, x]:
                break
        out.append((t, min(max(0, y - size // 2), height - size), min(max(0, x - size // 2), width - size)))
    return out


def fit(upscaled: np.ndarray, truth: Any, regions: list[np.ndarray], steps: int, lpips_net: Any, device: str,
        blocks: int = TRAIN["blocks"], channels: int = TRAIN["channels"]) -> tuple[Any, dict[str, Any]]:
    """G5d's enhancer (identity at step 0) fitted on (upscaled decode, source) patches centred on R, MSE plus LPIPS on R
    at equal weight at step 0 (G5d's passing loss)."""
    import torch

    from experiments.background.g5d import enhancer, masked_lpips, masked_mse

    torch.manual_seed(TRAIN["seed"])
    rng = np.random.default_rng(TRAIN["seed"])
    boxes = region_boxes(regions)
    hand = [t for t, b in enumerate(boxes) if b is not None]
    net = enhancer(blocks, channels).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAIN["lr"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, steps))
    size = TRAIN["patch"]

    def batch(patches: list[tuple[int, int, int]]) -> tuple[Any, Any, Any]:
        def stack(get: Any) -> Any:
            return torch.from_numpy(np.stack([np.asarray(get(t)[y:y + size, x:x + size]) for t, y, x in patches])).to(device)

        x = stack(lambda t: upscaled[t]).permute(0, 3, 1, 2).float() / 255.0
        target = stack(lambda t: truth[t]).permute(0, 3, 1, 2).float() / 255.0
        mask = stack(lambda t: regions[t])[:, None].float()
        return x, target, mask

    fixed = batch(sample_on_region(np.random.default_rng(TRAIN["seed"] + 1), regions, boxes, hand, TRAIN["batch"], size))
    with torch.no_grad():
        mse0 = float(masked_mse(fixed[0], fixed[1], fixed[2]))
        lp0 = float(masked_lpips(lpips_net, fixed[0], fixed[1], fixed[2]))
    weight = mse0 / lp0 if lp0 > 0 else 0.0
    began, curve = time.time(), []
    for step in range(1, steps + 1):
        x, target, mask = batch(sample_on_region(rng, regions, boxes, hand, TRAIN["batch"], size))
        out = net(x)
        value = masked_mse(out, target, mask) + weight * masked_lpips(lpips_net, out, target, mask)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        schedule.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                out = net(fixed[0])
                curve.append({"step": step, "mse_fixed": float(masked_mse(out, fixed[1], fixed[2])),
                              "lpips_fixed": float(masked_lpips(lpips_net, out, fixed[1], fixed[2]))})
    parameters = int(sum(p.numel() for p in net.parameters()))
    return net.eval(), {"steps": steps, "blocks": blocks, "channels": channels, "lpips_weight": weight, "mse_fixed_start": mse0, "lpips_fixed_start": lp0,
                        "curve": curve, "seconds": round(time.time() - began, 1), "parameters": parameters,
                        "bytes_fp16": 2 * parameters}


def apply(net: Any, upscaled: np.ndarray, regions: list[np.ndarray], device: str) -> np.ndarray:
    """The network on R's box plus `APPLY_MARGIN` of each frame; elsewhere the upscaled decode, unchanged."""
    import torch

    out = upscaled.copy()
    height, width = regions[0].shape
    with torch.inference_mode():
        for t, box in enumerate(region_boxes(regions)):
            if box is None:
                continue
            y0, y1 = max(0, box[0] - APPLY_MARGIN), min(height, box[1] + APPLY_MARGIN)
            x0, x1 = max(0, box[2] - APPLY_MARGIN), min(width, box[3] + APPLY_MARGIN)
            x = torch.from_numpy(np.ascontiguousarray(upscaled[t, y0:y1, x0:x1])).to(device).permute(2, 0, 1)[None].float() / 255.0
            y = net(x).clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0)
            out[t, y0:y1, x0:x1] = y.cpu().numpy()
    return out


def flicker_flow_free(frames: Any, truth: Any, regions: list[np.ndarray]) -> np.ndarray:
    """Per frame t: the mean absolute luma of the composite's change from t-1 minus the source's, on R at both frames
    (NaN where R is empty at either). No motion compensation: compared between outputs at matching rates only."""
    from experiments.visor.h3b import luma

    out = np.full(len(regions), np.nan)
    for t in range(1, len(regions)):
        both = regions[t] & regions[t - 1]
        if not both.any():
            continue
        cur, prev = np.asarray(frames[t]), np.asarray(frames[t - 1])
        src, src_prev = np.asarray(truth[t]), np.asarray(truth[t - 1])
        d = (luma(cur) - luma(prev)) - (luma(src) - luma(src_prev))
        out[t] = float(np.abs(d[both]).mean())
    return out


def composites(source: np.ndarray, truth: Any, regions: list[np.ndarray]) -> Any:
    """Lazily, the output inside R and the source outside, per frame."""

    class Frames:
        def __getitem__(self, t: int) -> np.ndarray:
            out = np.asarray(truth[t]).copy()
            out[regions[t]] = source[t][regions[t]]
            return out

    return Frames()


# ----------------------------------------------------------------- the oracle job


def upscale_all(rgb: Any, frames: int, width: int, height: int) -> np.ndarray:
    import cv2

    out = np.empty((frames, height, width, 3), np.uint8)
    for t in range(frames):
        out[t] = cv2.resize(np.asarray(rgb[t]), (width, height), interpolation=cv2.INTER_CUBIC)
    return out


def code_point(args: argparse.Namespace, base: str, value: float, scale: int, decoded: dict[str, Any], planes: Any,
               regions: list[np.ndarray], frames: int, fps: Fraction, work: Path, device: str) -> tuple[Any, dict[str, Any]]:
    """The base point's decode at its coded size (RGB) and its record (rate on R as H3c measures it)."""
    from experiments.visor.h3b import encode_low
    from experiments.visor.h3c import code_dcvc

    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    seconds = frames / float(fps)
    if base == "svt":
        rgb, record = encode_low(args, Path(decoded["path"]), frames, (width, height, full_range), value, scale, regions, fps, work, device)
        return rgb, {"codec": "svtav1", "crf": value, "scale": scale, "kbps_r": record["hand_bits"] / seconds / 1000,
                     "kbps": 8 * record["payload_bytes"] / seconds / 1000, "stream": record}
    sw, sh = scaled_size(width, height, scale)
    source = work / f"source-{scale}.yuv"
    if not source.exists():
        with source.open("wb") as handle:
            for t in range(frames):
                handle.write(downscale_planes(np.asarray(planes[t]), width, height, sw, sh).tobytes())
    region_file = work / f"region-{scale}.npy"
    if not region_file.exists():
        np.save(region_file, np.stack([downscale_mask(r, sw, sh) for r in regions]))
    rec = code_dcvc(args, source, region_file, (sw, sh), frames, int(value), args.video_ckpt, args.video_sha256,
                    work / f"dcvc-{scale}-{value:g}")
    if not rec["deterministic"] or not rec["decoder_matches_encoder_intra"]:
        raise RuntimeError(f"DCVC-UF QP {value:g} scale {scale}: decode is not the encoder's (two decodes agree: "
                           f"{rec['deterministic']}, first differing frames {rec['decode_passes_differ_at']}; intra matches: "
                           f"{rec['decoder_matches_encoder_intra']})")
    count = Path(rec["decoded"]).stat().st_size // (sw * sh * 3 // 2)
    if count < frames:
        raise RuntimeError(f"DCVC-UF QP {value:g} scale {scale}: decoded {count} of {frames} frames")
    rgb = rgb_memmap(Path(rec["decoded"]), work / "dcvc-rgb.npy", frames, sw, sh, full_range, device)
    return rgb, {"codec": "dcvc-uf-htl", "qp": int(value), "scale": scale, "kbps_r": rec["region_bits"] / seconds / 1000,
                 "kbps": rec["bits"] / seconds / 1000, "video_sha256": args.video_sha256,
                 **{k: v for k, v in rec.items() if k != "decoded"}}


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, done: set[str]) -> int:
    import cv2

    from experiments.visor.h3c import Scorer

    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info = load_window(args, item, frames, decoded)
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    score = Scorer(nets, truth, regions, device)
    look = score.hand[len(score.hand) // 2]
    ys, xs = np.nonzero(regions[look])
    look_box = (slice(max(0, ys.min() - DISTS_PAD), min(height, ys.max() + 1 + DISTS_PAD)),
                slice(max(0, xs.min() - DISTS_PAD), min(width, xs.max() + 1 + DISTS_PAD)))
    crops = publish / "frames" / item["id"]
    crops.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crops / f"source-f{look:03d}.png"), cv2.cvtColor(np.asarray(truth[look])[look_box], cv2.COLOR_RGB2BGR))
    finished = 0
    for value, scale in args.points:
        name = point_name(args.base, value, scale)
        unit = f"{item['id']}@{name.replace('/', '_')}"
        if unit in done:
            continue
        began = time.time()
        rgb, record = code_point(args, args.base, value, scale, decoded, planes, regions, frames, fps, work, device)
        upscaled = upscale_all(rgb, frames, width, height)
        del rgb
        for leftover in (work / "dcvc-rgb.npy",):
            leftover.unlink(missing_ok=True)
        shutil.rmtree(work / f"low-{value:g}-{scale}", ignore_errors=True)
        shutil.rmtree(work / f"dcvc-{scale}-{value:g}", ignore_errors=True)
        before = score(upscaled, upscale=False)
        flicker_before = flicker_flow_free(composites(upscaled, truth, regions), truth, regions)
        net, fit_info = fit(upscaled, truth, regions, int(args.steps), nets["lpips"], device)
        out = apply(net, upscaled, regions, device)
        after = score(out, upscale=False)
        flicker_after = flicker_flow_free(composites(out, truth, regions), truth, regions)
        for label, frames_rgb in (("base", upscaled), ("sr", out)):
            composite = np.asarray(truth[look]).copy()
            composite[regions[look]] = frames_rgb[look][regions[look]]
            cv2.imwrite(str(crops / f"{name.replace('/', '_').replace('@', '-')}-{label}-f{look:03d}.png"),
                        cv2.cvtColor(composite[look_box], cv2.COLOR_RGB2BGR))
        # The network at step 0 is the identity: on the job's first point it must reproduce the base point exactly.
        checks: dict[str, Any] = {}
        if not finished:
            import torch

            from experiments.background.g5d import enhancer

            torch.manual_seed(TRAIN["seed"])
            identity = enhancer(TRAIN["blocks"], TRAIN["channels"]).to(device).eval()
            same = apply(identity, upscaled, regions, device)
            checks["identity_reproduces_base"] = bool(np.array_equal(same, upscaled))
            del same
        hand = np.array([r.any() for r in regions])
        arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64),
                  "lpips": np.stack([before["lpips"], after["lpips"]]), "sse": np.stack([before["sse"], after["sse"]]),
                  "dists": np.stack([before["dists"], after["dists"]]), "flicker": np.stack([flicker_before, flicker_after])}
        meta = {"unit": unit, "item": item["id"], "base": args.base, "name": name, "h3c_name": h3c_name(args.base, value, scale),
                "video": item["video"], "fps": float(item["fps"]), "frames": frames, "seconds_coded": frames / float(fps),
                "hand_frames": int(hand.sum()), "rows": ["base", "sr"], "point": record, "fit": fit_info, "look_frame": look,
                "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "checks": checks,
                "train": TRAIN, "apply_margin": APPLY_MARGIN, "seconds": round(time.time() - began, 2)}
        save_unit(publish, "point", unit, arrays, meta)
        del upscaled, out
        finished += 1
    os.remove(decoded["path"])
    shutil.rmtree(work)
    return finished


def command_oracle(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.visor.b2 import select_items
    from experiments.visor.h3b import _tool, load_dists
    from src.codecs.quality import load_lpips, lpips_record

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    wanted = set(args.only.split(",")) if args.only else None
    args.points = [(v, s) for v, s in BASES[args.base] if wanted is None or point_name(args.base, v, s) in wanted]
    if not args.points:
        raise SystemExit(f"no {args.base} point among {args.only}")
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "point")
    if args.base == "svt":
        if not args.inspect:
            raise SystemExit("--base svt needs --inspect (H1's libaom accounting)")
        copy = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (H1)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.inspect, copy)
        copy.chmod(0o755)
        args.inspect_source, args.inspect = args.inspect, str(copy)
    windows = decode_windows(args, items, int(args.frames), scratch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("h3d oracle needs a GPU")
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    for p in nets["lpips"].parameters():  # LPIPS is a loss here: gradients flow through it, not into it
        p.requires_grad_(False)
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items()}
    began = time.time()
    count = len(restored)
    for item, decoded in windows:
        count += process_item(args, item, decoded, nets, device, publish, scratch, set(restored))
        progress(count)
    write_json(stage_dir() / "h3d.json", {
        "command": "oracle", "base": args.base, "device": device_record(torch.cuda.get_device_name(0)),
        "gpu_capability": list(torch.cuda.get_device_capability(0)), "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
        "lpips": lpips_record(Path(args.lpips_backbone)), "dists_backbone_sha256": file_sha256(Path(args.dists_backbone)),
        "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256},
        "svt": {"decoder": _tool(args.decoder), "encoder": _tool(args.encoder), "preset": args.preset,
                "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}} if args.base == "svt" else None,
        "settings": {"frames": int(args.frames), "items": args.items, "steps": int(args.steps),
                     "points": [point_name(args.base, v, s) for v, s in args.points], "train": TRAIN, "decision": DECISION},
        "items": [it["id"] for it in items], "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    units = load_units([str(stage / "published.tar")], "point")
    expected = len(result["items"]) * len(result["settings"]["points"])
    hand = [(m, a, a["pixels_r"] > 0) for m, a in units]
    dcvc = [m["point"] for m, _ in units if m["point"]["codec"] == "dcvc-uf-htl"]
    first = [m for m, _ in units if "identity_reproduces_base" in m["checks"]]
    return {
        "every_point_present": len(units) == expected and all(m["npz_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "intended_gpu": tuple(result["gpu_capability"]) >= (8, 6),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "scores_finite_on_hand_frames": all(np.isfinite(a["lpips"][:, h]).all() and np.isfinite(a["dists"][:, h]).all() for _, a, h in hand),
        "identity_reproduces_base": bool(first) and all(m["checks"]["identity_reproduces_base"] for m in first),
        "fit_ran_every_step": all(m["fit"]["curve"][-1]["step"] == m["fit"]["steps"] for m, _ in units if m["fit"]["steps"] > 0),
        "fit_losses_finite": all(math.isfinite(c["mse_fixed"]) and math.isfinite(c["lpips_fixed"]) for m, _ in units for c in m["fit"]["curve"]),
        # Fitted on these very frames with LPIPS in the loss, the network should not raise LPIPS_R.
        "sr_lpips_not_worse": all(np.nanmean(a["lpips"][1][h]) <= np.nanmean(a["lpips"][0][h]) + 1e-3 for _, a, h in hand),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        "region_rate_inside_stream_rate": all(0.0 < m["point"]["kbps_r"] < m["point"]["kbps"] for m, _ in units),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "h3d.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}  # numpy booleans do not serialize
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def quality(points: list[dict[str, Any]], metric: str) -> list[tuple[float, float]]:
    key, sign, _ = METRICS[metric]
    return [(float(p["kbps"]), sign * float(p[key])) for p in points]


def quality_at(envelope: list[tuple[float, float]], rate: float) -> float | None:
    """The envelope's quality at ``rate`` (linear in log rate), its best above its range, None below it."""
    if rate < envelope[0][0]:
        return None
    if rate >= envelope[-1][0]:
        return envelope[-1][1]
    return float(np.interp(math.log(rate), [math.log(r) for r, _ in envelope], [q for _, q in envelope]))


def compare(test: list[dict[str, Any]], anchor: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """BD-rate of ``test`` (its Pareto front) against ``anchor``'s upper concave hull on ``metric`` (PCHIP of log rate over
    the shared quality range); "better/worse at every rate" when that range is under the metric's minimum."""
    from scipy.interpolate import PchipInterpolator

    from experiments.background.g5 import envelope, pareto

    t, a = pareto(quality(test, metric)), envelope(quality(anchor, metric))
    lo, hi = max(t[0][1], a[0][1]), min(t[-1][1], a[-1][1])
    if len(t) >= 2 and len(a) >= 2 and hi - lo >= METRICS[metric][2]:
        def area(points: list[tuple[float, float]]) -> float:
            f = PchipInterpolator([q for _, q in points], [math.log(r) for r, _ in points])
            return float(f.integrate(lo, hi))

        return {"bd": math.exp((area(t) - area(a)) / (hi - lo)) - 1.0, "range": [lo, hi], "where": "overlap"}
    sides = set()
    for rate, q in t:
        at = quality_at(a, rate)
        sides.add("better" if at is None or q > at else "worse")
    where = f"{sides.pop()} at every rate" if len(sides) == 1 else "the curves cross without a shared range"
    return {"bd": None, "where": where}


def payback(test: list[dict[str, Any]], anchor: list[dict[str, Any]], weight_bytes: int) -> dict[str, Any]:
    """Seconds of video for the network's bits to equal the rate it saves at the test front's median LPIPS."""
    from experiments.background.g5 import envelope, pareto

    t, a = pareto(quality(test, "lpips")), envelope(quality(anchor, "lpips"))
    q = float(np.median([x for _, x in t]))
    if not (a[0][1] <= q <= a[-1][1] and t[0][1] <= q <= t[-1][1]):
        return {"lpips": -q, "seconds": None, "why": "outside a front"}

    def rate(points: list[tuple[float, float]]) -> float:
        return math.exp(float(np.interp(q, [x for _, x in points], [math.log(r) for r, _ in points])))

    saved = rate(a) - rate(t)
    return {"lpips": -q, "kbps_saved": saved, "seconds": 8 * weight_bytes / (saved * 1000) if saved > 0 else None}


def arm_verdict(per_item: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if len(per_item) < len(PILOT):
        return {"verdict": "incomplete", "items": len(per_item)}
    results = [e["lpips"] for e in per_item.values()]
    numbers = [r["bd"] for r in results if r["bd"] is not None]
    below = sum(r["bd"] < 0 for r in results if r["bd"] is not None) + sum(r["where"] == "better at every rate" for r in results)
    worse = sum(r["where"] == "worse at every rate" for r in results)
    out: dict[str, Any] = {"items_below_zero": below, "items_worse_at_every_rate": worse, "items_with_bd": len(numbers)}
    if numbers:
        out["mean_bd"] = float(np.mean(numbers))
        out["mean_bd_ci"] = bootstrap_mean(np.array(numbers), np.random.default_rng(seed_of(SEED)), RESAMPLES)
    if len(numbers) < DECISION["min_items_with_bd"]:
        out["verdict"] = "fails" if below <= DECISION["fails_items_below_zero"] else "unclear"
    elif out["mean_bd"] <= DECISION["passes_mean_bd"] and below >= DECISION["items_below_zero"] and not worse:
        out["verdict"] = "passes"
    elif out["mean_bd"] >= DECISION["fails_mean_bd"] or below <= DECISION["fails_items_below_zero"]:
        out["verdict"] = "fails"
    else:
        out["verdict"] = "unclear"
    return out


def decide(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdicts = {a: s["verdict"]["verdict"] for a, s in arms.items()}
    if len(verdicts) < len(BASES) or "incomplete" in verdicts.values():
        return {"oracle": "incomplete", "arms": verdicts}
    if "passes" in verdicts.values():
        oracle = "passes"
    elif all(v == "fails" for v in verdicts.values()):
        oracle = "fails"
    else:
        oracle = "unclear"
    return {"oracle": oracle, "arms": verdicts}


def component_rows(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], durations: dict[str, float]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Each component row's points per item (arm -> item -> points). V's rate is the rate on R plus its network's fp16
    bits over the video's duration; ``V-*/free`` is V without them (memorization against payback)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for meta, a in units:
        hand = a["pixels_r"] > 0
        rate = float(meta["point"]["kbps_r"])
        for j, row in enumerate(meta["rows"]):
            q = {"name": meta["name"], **point_summary(a, j, hand), "flicker": float(np.nanmean(a["flicker"][j])),
                 "flicker_base": float(np.nanmean(a["flicker"][0]))}
            if row.startswith("V-"):
                seconds = durations[meta["video"]]
                bits = 8 * meta["fits"][row]["bytes_fp16"] / seconds / 1000
                out.setdefault(row, {}).setdefault(meta["item"], []).append({**q, "kbps": rate + bits, "network_kbps": bits})
                out.setdefault(f"{row}/free", {}).setdefault(meta["item"], []).append({**q, "kbps": rate})
            else:
                out.setdefault(row, {}).setdefault(meta["item"], []).append({**q, "kbps": rate})
    return out


def component_decision(arms: dict[str, Any], rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """`h3d_component.COMPONENT`: which arm (V at its best size, K, G) stands, and which is H3d's coder."""
    from experiments.visor.h3d_component import COMPONENT

    def stands(name: str) -> bool:
        return arms.get(name, {}).get("verdict", {}).get("verdict") == "passes"

    def mean(name: str) -> float:
        return arms[name]["verdict"].get("mean_bd", math.inf)

    sizes = [r for r in arms if r.startswith("V-") and not r.endswith("/free")]
    best_v = min(sizes, key=mean) if sizes else None

    def beats(a: str, b: str) -> dict[str, Any]:
        values = [compare(rows[a][i], rows[b][i], "lpips")["bd"] for i in rows[a] if i in rows.get(b, {})]
        known = [v for v in values if v is not None]
        m = float(np.mean(known)) if known else None
        return {"mean_bd": m, "items": len(known), "beats": m is not None and m <= COMPONENT["beats_by"]}

    out: dict[str, Any] = {"stands": {r: stands(r) for r in ("G-L", "K-L", *sizes)}, "best_v": best_v}
    shared = [r for r in ("G-L", "K-L") if stands(r)]
    choice = None
    if stands("G-L"):
        choice = "G-L"
        if stands("K-L"):
            out["K_vs_G"] = beats("K-L", "G-L")
            choice = "K-L" if out["K_vs_G"]["beats"] else "G-L"
    elif stands("K-L"):
        choice = "K-L"
    if best_v and stands(best_v):
        out["V_vs_shared"] = {r: beats(best_v, r) for r in shared}
        if not shared or all(v["beats"] for v in out["V_vs_shared"].values()):
            choice = best_v
    out["coder"] = choice or "none: H3d is not built"
    return out


def command_report(args: argparse.Namespace) -> int:
    from experiments.visor.h3c import point_summary as h3c_summary

    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    bar: dict[str, list[dict[str, Any]]] = {}
    h3c_scores: dict[tuple[str, str], dict[str, Any]] = {}
    for meta, a in load_units(args.h3c, "code"):
        hand = a["pixels_r"] > 0
        rates = hand_kbps(h1[meta["item"]])
        for j, p in enumerate(meta["points"]):
            if p["arm"] not in ("svtav1", "reference"):
                continue
            kbps = p["kbps_r"] if "kbps_r" in p else rates[f"{p['crf']:g}"]
            row = {"name": p["name"], "kbps": float(kbps), **h3c_summary(a, j, hand), "lpips_frames": a["lpips"][j]}
            bar.setdefault(meta["item"], []).append(row)
            h3c_scores[(meta["item"], p["name"])] = {**row, "stream_sha256": p.get("stream", {}).get("sha256") or p.get("stream_sha256")}
    tests: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cross: dict[str, dict[str, Any]] = {}
    weight_bytes: dict[str, int] = {}
    for meta, a in load_units(args.oracle, "point"):
        hand = a["pixels_r"] > 0
        p = meta["point"]
        base_row = {"name": meta["name"], "kbps": float(p["kbps_r"]), **point_summary(a, 0, hand)}
        sr_row = {"name": meta["name"], "kbps": float(p["kbps_r"]), **point_summary(a, 1, hand),
                  "flicker": float(np.nanmean(a["flicker"][1])), "flicker_base": float(np.nanmean(a["flicker"][0]))}
        tests.setdefault(meta["base"], {}).setdefault(meta["item"], []).append(sr_row)
        weight_bytes[meta["base"]] = meta["fit"]["bytes_fp16"]
        ref = h3c_scores.get((meta["item"], meta["h3c_name"]))
        if ref is not None:  # the base point reproduces H3c's (SVT-AV1 byte for byte; DCVC-UF within its GPU-class spread)
            lp_gap = float(np.nanmax(np.abs(ref["lpips_frames"][hand] - a["lpips"][0][hand])))
            entry = {"lpips_max_frame_gap": lp_gap, "lpips_mean_gap": base_row["lpips"] - ref["lpips"],
                     "rate_ratio": base_row["kbps"] / ref["kbps"]}
            sha = p.get("stream", {}).get("sha256") or p.get("stream_sha256")
            entry["stream_matches"] = sha == ref["stream_sha256"]
            entry["holds"] = (entry["stream_matches"] and lp_gap <= 2e-4) if p["codec"] == "svtav1" else \
                (abs(entry["lpips_mean_gap"]) <= 0.005 and abs(entry["rate_ratio"] - 1) <= 0.05)
            cross.setdefault(meta["item"], {})[meta["name"]] = entry
    durations = {name: float(seconds) for name, _, seconds in (v.partition("=") for v in args.video_seconds)}
    arms: dict[str, Any] = {}
    for base, per in sorted(tests.items()):
        per_item = {}
        for item, test in sorted(per.items()):
            anchor = bar.get(item)
            if not anchor:
                continue
            entry = {m: compare(test, anchor, m) for m in METRICS}
            entry["front"] = test
            entry["payback"] = payback(test, anchor, weight_bytes[base])
            video = item.rsplit("_", 1)[0]
            entry["payback"]["video_seconds"] = durations.get(video)
            entry["flicker_ratio"] = float(np.nanmean([q["flicker"] / q["flicker_base"] for q in test]))
            per_item[item] = entry
        arms[base] = {"items": per_item, "verdict": arm_verdict(per_item)}
    if args.component:
        comp_rows = component_rows(load_units(args.component, "component"), durations)
        comp: dict[str, Any] = {}
        for arm_row, per_arm in sorted(comp_rows.items()):
            arm_items: dict[str, Any] = {}
            for item, test in sorted(per_arm.items()):
                if item in bar:
                    arm_items[item] = {**{m: compare(test, bar[item], m) for m in METRICS}, "front": test,
                                       "flicker_ratio": float(np.nanmean([q["flicker"] / q["flicker_base"] for q in test]))}
            comp[arm_row] = {"items": arm_items, "verdict": arm_verdict(arm_items)}
        report_component = {"arms": comp, "decision": component_decision(comp, comp_rows)}
    report: dict[str, Any] = {"decision_rule": DECISION, "decision": decide(arms), "arms": arms, "base_cross_check": cross,
              **({"component": report_component} if args.component else {}),
              "bar": {item: [{k: v for k, v in q.items() if k != "lpips_frames"} for q in rows] for item, rows in bar.items()},
              "runs": [{"path": p, "sha256": file_sha256(Path(p))} for p in [*args.oracle, *args.component, *args.h3c]]}
    out = Path(args.out or stage_dir())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "h3d-report.json", report)
    if args.component:
        print(json.dumps({"component": {r: s["verdict"] for r, s in report["component"]["arms"].items()},
                          "decision": report["component"]["decision"]}, indent=1, default=str))
    print(json.dumps({"decision": report["decision"], "verdicts": {b: s["verdict"] for b, s in arms.items()},
                      "cross_check_holds": {i: all(e["holds"] for e in c.values()) for i, c in cross.items()}}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3d")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--eval-set", required=True)
    oracle.add_argument("--archive", required=True)
    oracle.add_argument("--masks", required=True)
    oracle.add_argument("--mask-record", required=True)
    oracle.add_argument("--video", action="append", default=[])
    oracle.add_argument("--items", required=True)
    oracle.add_argument("--frames", default="240")
    oracle.add_argument("--base", choices=sorted(BASES), required=True)
    oracle.add_argument("--only", default="", help="a comma list of point names (default: every point of the base)")
    oracle.add_argument("--steps", required=True)
    oracle.add_argument("--image-ckpt", required=True)
    oracle.add_argument("--video-ckpt", required=True)
    oracle.add_argument("--lpips-backbone", required=True)
    oracle.add_argument("--dists-backbone", required=True)
    oracle.add_argument("--inspect", default="")
    oracle.add_argument("--decoder", default="dav1d")
    oracle.add_argument("--encoder", default="SvtAv1EncApp")
    oracle.add_argument("--preset", default="4")
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--oracle", action="append", default=[], help="the oracle jobs' published.tar")
    report.add_argument("--component", action="append", default=[], help="the component's score jobs' published.tar")
    report.add_argument("--h3c", action="append", required=True, help="H3c's anchor-and-reference jobs' published.tar")
    report.add_argument("--h1-report", required=True)
    report.add_argument("--video-seconds", action="append", default=[], metavar="VIDEO=SECONDS",
                        help="each pilot video's duration (ffprobe), for the payback")
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    return {"oracle": command_oracle, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
