from __future__ import annotations

from functools import lru_cache
import os
import warnings

import torch


def _dtype_alias_map() -> dict[str, torch.dtype]:
    aliases: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if hasattr(torch, "float8_e4m3fn"):
        aliases["float8_e4m3fn"] = torch.float8_e4m3fn
        aliases["fp8_e4m3fn"] = torch.float8_e4m3fn
    if hasattr(torch, "float8_e5m2"):
        aliases["float8_e5m2"] = torch.float8_e5m2
        aliases["fp8_e5m2"] = torch.float8_e5m2

    return aliases


def _dtype_name(dtype: torch.dtype) -> str:
    text = str(dtype)
    if text.startswith("torch."):
        return text.split(".", maxsplit=1)[1]
    return text


def parse_gpu_dtype_env(env_var: str = "POINTSTREAM_GPU_DTYPE") -> torch.dtype | None:
    raw = os.environ.get(env_var)
    if raw is None:
        return None

    key = raw.strip().lower()
    if key == "":
        return None

    aliases = _dtype_alias_map()
    parsed = aliases.get(key)
    if parsed is not None:
        return parsed

    supported = ", ".join(sorted(set(aliases.keys())))
    warnings.warn(
        f"Unsupported {env_var}='{raw}'. Supported values: {supported}. Falling back to defaults.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


@lru_cache(maxsize=64)
def _is_cuda_dtype_supported(dtype_name: str, device_index: int) -> bool:
    if not torch.cuda.is_available():
        return False

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return False

    try:
        sample = torch.ones((16,), dtype=dtype, device=torch.device("cuda", device_index))
        _ = (sample + sample).sum().item()
        return True
    except Exception:
        return False


def resolve_torch_dtype_for_device(
    device: str | torch.device,
    *,
    default_cuda: torch.dtype = torch.float16,
    allowed_cuda: set[torch.dtype] | None = None,
    env_var: str = "POINTSTREAM_GPU_DTYPE",
) -> torch.dtype:
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return torch.float32

    requested = parse_gpu_dtype_env(env_var=env_var)
    candidate = requested or default_cuda

    if allowed_cuda is not None and candidate not in allowed_cuda:
        warnings.warn(
            f"Requested GPU dtype {_dtype_name(candidate)} is not allowed in this module. "
            f"Falling back to {_dtype_name(default_cuda)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        candidate = default_cuda

    device_index = resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()
    if _is_cuda_dtype_supported(_dtype_name(candidate), int(device_index)):
        return candidate

    fallback = default_cuda
    if allowed_cuda is not None and fallback not in allowed_cuda:
        for preferred in (torch.float16, torch.bfloat16, torch.float32):
            if preferred in allowed_cuda:
                fallback = preferred
                break

    if _is_cuda_dtype_supported(_dtype_name(fallback), int(device_index)):
        warnings.warn(
            f"GPU dtype {_dtype_name(candidate)} is unsupported on this device. "
            f"Falling back to {_dtype_name(fallback)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback

    warnings.warn(
        f"GPU dtype {_dtype_name(candidate)} is unsupported and no preferred CUDA dtype was available. "
        "Falling back to float32.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.float32


def is_cuda_device_usable(device: str | torch.device) -> bool:
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        return False

    if not torch.cuda.is_available():
        return False

    device_index = resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()
    return _is_cuda_dtype_supported("float32", int(device_index))
