"""G6 item 2: replay shipped enhancer weights and streams, with isolated timing.

No fitting or encoding: compare precision/layout on identical published inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5c, g5d, g6
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.codecs import svtav1
from src.codecs.enhancer import CONFIGS, SPATIAL_CONFIGS, apply

def pixel_error(frames: np.ndarray, reference: np.ndarray) -> tuple[float, int]:
    """Paired uint8 RGB error with one frame of temporary storage."""
    if frames.shape != reference.shape or frames.dtype != np.uint8 or reference.dtype != np.uint8 or not frames.size:
        raise ValueError("expected matching nonempty uint8 frames")
    error_sum, maximum = 0, 0
    for f, ref in zip(frames, reference):
        error = np.abs(f.astype(np.int16) - ref.astype(np.int16))
        error_sum += int(error.sum())
        maximum = max(maximum, int(error.max()))
    return error_sum / frames.size, maximum


def command_run(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "speed"
    work.mkdir(parents=True)
    root = Path(args.published) / "publish"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["window"] != args.item or manifest["foreground"]["dilate_px_1080"] != 0:
        raise ValueError("pilot window or exact foreground mismatch")
    item, rgb, fg, source = g6.load_window(args, work)
    if source["masks"]["mask_sha256"] != manifest["foreground"]["mask_sha256"]:
        raise ValueError("mask identity mismatch")
    n = len(rgb)
    clip = g5.Clip(item["id"], "window", Fraction(manifest["fps"]), 0, rgb, ~fg,
                   {"dataset": (list(range(n)), ~fg), "own": (list(range(n)), ~fg)},
                   {"id": item["id"], "video": item["video"], "fps": manifest["fps"], "frames": n})
    nets, flows = g5c.load_nets(args), g5c.source_flows(clip)
    configs_run = args.configs.split(",")
    if configs_run[0] != "fp32" or any(c not in (*CONFIGS, *SPATIAL_CONFIGS) for c in configs_run):
        raise ValueError("configs must start with the paired FP32 reference")
    rows = []
    for done, crf in enumerate(map(int, args.crfs.split(",")), 1):
        anchor = next(p for p in manifest["points"] if p["method"] == "svt-1080" and p["value"] == crf)
        enhanced = next(p for p in manifest["points"] if p["method"] == "enh-1080" and p["value"] == crf)
        stream, weights = root / anchor["bitstream"], root / enhanced["enhancer"]["weights"]
        if file_sha256(stream) != anchor["bitstream_sha256"] or file_sha256(weights) != enhanced["enhancer"]["weights_sha256"]:
            raise ValueError("published stream or weight checksum mismatch")
        command = svtav1.decode_command("dav1d", stream, work / "decoded.yuv", threads=8)
        timing = svtav1.run(command, 120)
        yuv = np.fromfile(work / "decoded.yuv", np.uint8).reshape(manifest["frames"], 1620, 1920)
        digests = [hashlib.sha256(f.tobytes()).hexdigest() for f in yuv]
        if digests != anchor["decoded_frame_sha256"]:
            raise ValueError("native decode does not match published frames")
        decoded = g6.to_rgb(yuv[:n], full_range=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        scores = []
        reference = None
        for config in configs_run:
            net = g5d.enhancer(manifest["enhancer"]["blocks"], manifest["enhancer"]["channels"])
            net.load_state_dict(state, strict=True)
            frames, latency = apply(net, decoded, config)
            if reference is None:
                reference = frames.copy()
            roundtrip = np.concatenate([g6.to_rgb(g5.rgb_to_yuv420(frames[i:i + 16]), full_range=True)
                                        for i in range(0, n, 16)])
            score = g5c.full_score(clip, roundtrip, nets, flows)
            mae, maximum = pixel_error(frames, reference)
            scores.append({"config": config, "timing": latency, "score": score,
                           "rgb_mae_vs_fp32": mae,
                           "rgb_max_vs_fp32": maximum})
            write_json(stage_dir() / f"crf{crf}-{config}.json", scores[-1])
            del net, frames, roundtrip
        row = {"crf": crf, "kbps": anchor["kbps"], "stream_sha256": anchor["bitstream_sha256"],
               "weights_sha256": enhanced["enhancer"]["weights_sha256"], "decoded_hashes_match": True,
               "decode_command": command, "dav1d_ms_per_frame": timing["wall"] * 1000 / manifest["frames"],
               "plain_score": g5c.full_score(clip, decoded, nets, flows), "configs": scores}
        rows.append(row)
        write_json(stage_dir() / f"crf{crf}.json", row)
        checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"])
        g5c.save(checkpoints, f"crf{crf}", row)
        progress(done)
    write_json(stage_dir() / "g6-speed.json", {"item": args.item, "frames": n, "fps": manifest["fps"],
        "crfs": [int(c) for c in args.crfs.split(",")], "configs_run": configs_run, "manifest_sha256": file_sha256(root / "manifest.json"), "source": source,
        "metrics": nets.record, "dav1d": svtav1.tool("dav1d"), "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": True,
        "baseline": "published FP16 weights loaded into FP32; original pilot used unsaved FP32 weights",
        **g6.provenance(), **g5c.gpu_record(), "points": rows})
    return 0


def validate(stage: Path) -> dict[str, bool]:
    r = json.loads((stage / "g6-speed.json").read_text())
    rows = r["points"]
    configs = [c for p in rows for c in p["configs"]]
    def finite_score(score: dict[str, Any]) -> bool:
        values = [score["dataset"][m] for m in ("lpips_v", "dists_v", "psnr_v")]
        values += [score["flicker"]["flicker_v"], score["kid"]["kid_v"]]
        return all(v is not None and np.isfinite(v) for v in values)

    return {"intended_gpu": tuple(r["gpu_capability"]) >= (8, 6),
            "every_crf": [p["crf"] for p in rows] == r["crfs"],
            "native_decode_hashes": bool(rows) and all(p["decoded_hashes_match"] for p in rows),
            "every_config": all([c["config"] for c in p["configs"]] == r.get("configs_run", list(CONFIGS)) for p in rows),
            "finite_scores": all(finite_score(c["score"]) for c in configs),
            "timed_every_frame": all(len(c["timing"]["wall_ms_per_frame"]) == r["frames"] for c in configs),
            "positive_timings": all(c["timing"]["cnn_ms_mean"] > 0 for c in configs),
            "precision_lpips": all(c["score"]["dataset"]["lpips_v"] <= 1.01 * p["configs"][0]["score"]["dataset"]["lpips_v"]
                                   for p in rows for c in p["configs"] if c["config"] in CONFIGS),
            "precision_close": all(c["rgb_mae_vs_fp32"] < 1 for c in configs if c["config"] in CONFIGS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("eval-set", "archive", "masks", "video", "item", "published", "lpips-backbone", "dists-backbone", "inception"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--limit-frames", type=int, default=0)
    run.add_argument("--crfs", default="32,40,48")
    run.add_argument("--configs", default=",".join(CONFIGS))
    val = sub.add_parser("validate")
    val.add_argument("--stage", default=None)
    args = parser.parse_args()
    if args.command == "run":
        return command_run(args)
    checks = validate(Path(args.stage) if args.stage else stage_dir())
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
