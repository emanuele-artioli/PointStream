"""Measured batch-one inference for the shipped G5d background enhancer."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

CONFIGS = ("fp32", "fp16", "fp16-channels-last")
SPATIAL_CONFIGS = ("fp16-channels-last-540", "fp16-channels-last-540-residual")


def apply(net: Any, decoded: np.ndarray, config: str, *, warmup: int = 5, profile_kernels: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
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
        kernels: list[str] = []
        if profile_kernels:
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


