"""Measure extract / pack / codec / decode p50/p95. Never hardcode 13.6 ms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

ExtractFn = Callable[[np.ndarray], Any]
TimedFn = Callable[[], None]


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def time_calls(fn: TimedFn, *, n_warmup: int = 2, n_runs: int = 8) -> tuple[float, float]:
    import time

    for _ in range(n_warmup):
        fn()
        _sync_cuda()
    samples: list[float] = []
    for _ in range(n_runs):
        _sync_cuda()
        t0 = time.perf_counter()
        fn()
        _sync_cuda()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _percentile(samples, 50), _percentile(samples, 95)


def profile_map(
    extract_fn: ExtractFn,
    frame: np.ndarray,
    pack_fn: Callable[[Any], bytes] | None = None,
    decode_fn: Callable[[bytes], Any] | None = None,
    codec_fn: TimedFn | None = None,
    *,
    n_warmup: int = 2,
    n_runs: int = 8,
) -> dict[str, float | str]:
    """Time stages separately. extract_fn is model-forward only."""

    holder: dict[str, Any] = {"out": None}

    def extract_once() -> None:
        holder["out"] = extract_fn(frame)

    extract_p50, extract_p95 = time_calls(extract_once, n_warmup=n_warmup, n_runs=n_runs)
    packed = b""
    pack_p50 = pack_p95 = 0.0
    if pack_fn is not None:
        extract_once()

        def pack_once() -> None:
            holder["packed"] = pack_fn(holder["out"])

        pack_p50, pack_p95 = time_calls(pack_once, n_warmup=max(1, n_warmup // 2), n_runs=n_runs)
        packed = holder.get("packed") or pack_fn(holder["out"])

    decode_p50 = 0.0
    if decode_fn is not None and packed:

        def decode_once() -> None:
            decode_fn(packed)

        decode_p50, _ = time_calls(decode_once, n_warmup=max(1, n_warmup // 2), n_runs=n_runs)

    codec_p50 = 0.0
    if codec_fn is not None:
        codec_p50, _ = time_calls(codec_fn, n_warmup=1, n_runs=max(3, n_runs // 2))

    return {
        "gpu": gpu_name(),
        "extract_ms_p50": round(extract_p50, 3),
        "extract_ms_p95": round(extract_p95, 3),
        "pack_ms_p50": round(pack_p50, 3),
        "pack_ms_p95": round(pack_p95, 3),
        "codec_ms_p50": round(codec_p50, 3),
        "decode_ms_p50": round(decode_p50, 3),
    }
