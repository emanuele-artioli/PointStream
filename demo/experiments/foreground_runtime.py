"""Timing, provider, budget, and interpreter checks for the foreground smoke."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PLAN_GPU_SECONDS = 1800
JOB_BUDGET_CEILING = 480
REALTIME_MS = 1000.0 / 24.0
POINTSTREAM_PYTHON = "/home/itec/emanuele/.conda/envs/pointstream/bin/python"
SAM_PYTHON = "/home/itec/emanuele/.conda/envs/pointstream-sam31/bin/python"
ALLOWED_INTERPRETERS = {POINTSTREAM_PYTHON, SAM_PYTHON}


class PlanCapError(ValueError):
    pass


def stage_rates(stage_seconds: Mapping[str, float], *, n_crops: int, n_frames: int) -> dict:
    if n_crops < 0 or n_frames <= 0:
        raise ValueError("frame count must be positive")
    total = float(sum(stage_seconds.values()))
    return {
        "stages_s": {key: float(value) for key, value in stage_seconds.items()},
        "total_s": total,
        "crops_per_s": (n_crops / total) if total else 0.0,
        "frames_per_s": (n_frames / total) if total else 0.0,
        "ms_per_frame": (1000.0 * total / n_frames) if n_frames else 0.0,
        "meets_24fps": bool(total and (1000.0 * total / n_frames) <= REALTIME_MS + 1e-9),
    }


def stages_reproduce_total(stage_seconds: Mapping[str, float], total_s: float, *, overhead_s: float) -> bool:
    return abs(float(sum(stage_seconds.values())) - float(total_s)) <= float(overhead_s)


def coordinates_agree(single: np.ndarray, batched: np.ndarray, tolerance_px: float = 1.0) -> bool:
    return float(np.max(np.abs(np.asarray(single) - np.asarray(batched)))) <= tolerance_px


def classify_execution_provider(configured: Sequence[str], profiled_nodes: Sequence[str]) -> str:
    """A CUDA provider name is not GPU timing unless profiled nodes actually ran there."""
    configured_text = " ".join(configured)
    node_text = " ".join(profiled_nodes)
    cuda_configured = "CUDA" in configured_text
    cuda_nodes = "CUDA" in node_text
    cpu_nodes = "CPU" in node_text
    if cuda_nodes and not cpu_nodes:
        return "cuda"
    if cpu_nodes and not cuda_nodes:
        return "cpu_fallback" if cuda_configured or not configured else "cpu"
    if cpu_nodes and cuda_nodes:
        return "mixed"
    if cuda_configured:
        return "unprofiled"
    return "unprofiled"


def check_job_budget(spec: Mapping) -> None:
    budget = float(spec["budget_seconds"])
    if budget > JOB_BUDGET_CEILING:
        raise PlanCapError("a foreground smoke job budget cannot exceed 480 seconds")
    reserved = float(spec["smoke"]["seconds"]) + float(spec["full"]["seconds"]) + float(spec.get("validator_seconds", 60))
    if budget < reserved:
        raise PlanCapError("budget_seconds must reserve smoke, full, and validation")
    if float(spec.get("stall_seconds", budget)) > budget:
        raise PlanCapError("stall_seconds cannot exceed the job budget")


def check_campaign_budgets(specs: Sequence[Mapping]) -> None:
    for spec in specs:
        check_job_budget(spec)
    total = sum(float(spec["budget_seconds"]) for spec in specs)
    if total > PLAN_GPU_SECONDS:
        raise PlanCapError("combined job budgets exceed the 1800-second plan cap")


def combined_count(smoke: int, full: int, cap: int, label: str) -> None:
    if smoke < 1 or full < smoke or smoke + full > cap:
        raise PlanCapError(f"{label} smoke+full work {smoke}+{full} exceeds the cap of {cap} or is not a nested subset")


def propagate_worker_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    base = os.environ if source is None else source
    env = {key: value for key, value in base.items() if key.startswith("PS_") or key == "CUDA_VISIBLE_DEVICES"}
    if "PATH" in base:
        env["PATH"] = base["PATH"]
    return env


def run_bounded_python(
    interpreter: str,
    argv: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    allowed: set[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an installed interpreter. No shell and no inline code."""
    permitted = allowed if allowed is not None else set(ALLOWED_INTERPRETERS)
    if interpreter not in permitted:
        raise ValueError(f"interpreter is not an installed smoke environment: {interpreter}")
    if any(arg == "-c" or arg.startswith("-c") for arg in argv):
        raise ValueError("inline Python is not allowed")
    command = [interpreter, *argv]
    process = subprocess.Popen(
        command,
        env=dict(env) if env is not None else None,
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise TimeoutError(f"interpreter exceeded {timeout_s:.0f}s") from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def av1_square_args(size: int, gop: int) -> list[str]:
    """Shared CRF 63 / preset 7 recipe at a square size, with GOP equal to the segment."""
    if size < 64:
        raise ValueError("SVT-AV1 cannot encode a side below 64")
    if gop < 1:
        raise ValueError("GOP must be positive")
    from demo.pipeline.maps.av1_crf import av1_output_args

    return [*av1_output_args(f"{size}:{size}"), "-g", str(gop)]
