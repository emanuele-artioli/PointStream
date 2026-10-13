"""G6 item 2: replay shipped enhancer weights and streams, with isolated timing.

No fitting or encoding: compare precision/layout on identical published inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5c, g5d, g6
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.codecs import svtav1

CONFIGS = ("fp32", "fp16", "fp16-channels-last")
SPATIAL_CONFIGS = ("fp16-channels-last-540", "fp16-channels-last-540-residual")


def apply(net: Any, decoded: np.ndarray, config: str, *, warmup: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
    """Batch-one CPU RGB → GPU enhance → CPU RGB; CUDA events measure just the CNN.

    Model setup and warmup excluded; transfers, conversion and rounding included
    in wall latency. Returned frames use the same rounding as g5d.apply.
    """
    import torch

    if config not in (*CONFIGS, *SPATIAL_CONFIGS) or decoded.dtype != np.uint8 or decoded.ndim != 4 or decoded.shape[-1] != 3:
        raise ValueError("expected a supported config and NHWC uint8 RGB frames")
    dtype = torch.float32 if config == "fp32" else torch.float16
    layout = torch.channels_last if "channels-last" in config else torch.contiguous_format
    net = net.eval().to(device="cuda", dtype=dtype, memory_format=layout)

    def input_frame(frame: np.ndarray) -> Any:
        return (torch.from_numpy(frame[None]).to("cuda").permute(0, 3, 1, 2)
                .to(dtype=dtype, memory_format=layout) / 255.0)

    if next(net.parameters()).device.type != "cuda" or next(net.parameters()).dtype != dtype:
        raise ValueError("enhancer device or precision mismatch")
    spatial = config in SPATIAL_CONFIGS

    def network_input(x: Any) -> Any:
        return (torch.nn.functional.interpolate(x, size=(540, 960), mode="area")
                .contiguous(memory_format=layout) if spatial else x)

    def restore_output(y: Any, small: Any, original: Any) -> Any:
        if not spatial:
            return y
        correction = y - small if config.endswith("residual") else y
        expanded = torch.nn.functional.interpolate(correction.float(), size=original.shape[-2:],
                                                   mode="bilinear", align_corners=False)
        return original.float() + expanded if config.endswith("residual") else expanded

    out = np.empty_like(decoded)
    wall, gpu = [], []
    with torch.inference_mode():
        x = network_input(input_frame(decoded[0]))
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            net(x)
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA})
        if not kernels:
            raise ValueError("no CUDA kernels observed")
        torch.cuda.reset_peak_memory_stats()
        for i, frame in enumerate(decoded):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            original = input_frame(frame)
            x = network_input(original)
            start.record()
            y = net(x)
            end.record()
            if not torch.isfinite(y).all():
                raise ValueError("nonfinite enhancer output")
            y = restore_output(y, x, original)
            out[i] = y.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            torch.cuda.synchronize()
            wall.append(1000 * (time.perf_counter() - began))
            gpu.append(start.elapsed_time(end))
    return out, {"config": config, "batch": 1, "warmup": warmup, "cuda_kernels": kernels,
                 "dtype": str(dtype), "layout": str(layout),
                 "wall_ms_mean": float(np.mean(wall)), "wall_ms_p95": float(np.percentile(wall, 95)),
                 "cnn_ms_mean": float(np.mean(gpu)), "cnn_ms_p95": float(np.percentile(gpu, 95)),
                 "wall_ms_per_frame": wall, "cnn_ms_per_frame": gpu,
                 "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20}


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
        "configs_run": configs_run, "manifest_sha256": file_sha256(root / "manifest.json"), "source": source,
        "metrics": nets.record, "dav1d": svtav1.tool("dav1d"), "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": True,
        "baseline": "published FP16 weights loaded into FP32; original pilot used unsaved FP32 weights",
        **g6.provenance(), **g5c.gpu_record(), "points": rows})
    return 0


def validate(stage: Path) -> dict[str, bool]:
    r = json.loads((stage / "g6-speed.json").read_text())
    rows = r["points"]
    configs = [c for p in rows for c in p["configs"]]
    return {"intended_gpu": tuple(r["gpu_capability"]) >= (8, 6),
            "native_decode_hashes": bool(rows) and all(p["decoded_hashes_match"] for p in rows),
            "every_config": all([c["config"] for c in p["configs"]] == r.get("configs_run", list(CONFIGS)) for p in rows),
            "finite_scores": all(np.isfinite(c["score"]["dataset"]["lpips_v"]) for c in configs),
            "timed_every_frame": all(len(c["timing"]["wall_ms_per_frame"]) == r["frames"] for c in configs),
            "positive_timings": all(c["timing"]["cnn_ms_mean"] > 0 for c in configs),
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
