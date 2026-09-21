"""Standardized GPU preflight guard utility.

Provides robust status queries and resource preflight verification
across single- and multi-GPU nodes before launching compute-intensive jobs.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class GpuUnavailableError(RuntimeError):
    """Raised when no GPU meets the minimum preflight requirements."""


def _parse_nvidia_smi_query(stdout: str) -> list[dict[str, Any]]:
    """Parse comma-separated nvidia-smi query output into structured dicts."""
    gpus: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            total_mib = int(float(parts[2]))
            free_mib = int(float(parts[3]))
            used_mib = int(float(parts[4]))
            util_pct = float(parts[5])
        except (ValueError, TypeError):
            continue
        gpus.append({
            "index": idx,
            "name": name,
            "memory_free_mib": free_mib,
            "memory_used_mib": used_mib,
            "memory_total_mib": total_mib,
            "util_pct": util_pct,
            "occupied_pids": [],
        })
    return gpus


def _parse_pmon(stdout: str) -> dict[int, list[int]]:
    """Parse nvidia-smi pmon output to collect occupied PIDs per GPU index."""
    pids_by_gpu: dict[int, list[int]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                g_idx = int(parts[0])
                if parts[1].isdigit():
                    pid = int(parts[1])
                    if g_idx not in pids_by_gpu:
                        pids_by_gpu[g_idx] = []
                    if pid not in pids_by_gpu[g_idx]:
                        pids_by_gpu[g_idx].append(pid)
            except (ValueError, TypeError):
                continue
    return pids_by_gpu


def _query_occupied_pids(num_gpus: int) -> dict[int, list[int]]:
    """Query occupied process IDs per GPU using nvidia-smi pmon or process tables."""
    pids_by_gpu: dict[int, list[int]] = {i: [] for i in range(num_gpus)}

    # 1. Try pmon
    try:
        proc = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            parsed = _parse_pmon(proc.stdout)
            for g_idx, pids in parsed.items():
                if g_idx in pids_by_gpu:
                    pids_by_gpu[g_idx].extend(pids)
    except Exception as exc:
        logger.debug("nvidia-smi pmon query failed: %s", exc)

    # 2. Try compute-apps query fallback / supplement
    try:
        app_proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if app_proc.returncode == 0 and app_proc.stdout.strip():
            for app_line in app_proc.stdout.strip().splitlines():
                parts = [p.strip() for p in app_line.split(",")]
                if parts and parts[0].isdigit():
                    pid = int(parts[0])
                    if num_gpus == 1 and pid not in pids_by_gpu[0]:
                        pids_by_gpu[0].append(pid)
    except Exception as exc:
        logger.debug("nvidia-smi compute-apps query failed: %s", exc)

    return pids_by_gpu


def _query_nvidia_smi() -> list[dict[str, Any]]:
    """Query GPU status directly from nvidia-smi CLI."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    gpus = _parse_nvidia_smi_query(proc.stdout)
    if not gpus:
        return []

    pids_by_gpu = _query_occupied_pids(len(gpus))
    for gpu in gpus:
        idx = gpu["index"]
        if idx in pids_by_gpu:
            gpu["occupied_pids"] = sorted(set(pids_by_gpu[idx]))

    return gpus


def _query_torch_cuda() -> list[dict[str, Any]]:
    """Fallback query using torch.cuda properties."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        device_count = torch.cuda.device_count()
        gpus: list[dict[str, Any]] = []
        for idx in range(device_count):
            name = torch.cuda.get_device_name(idx)
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                free_mib = free_b // (1024 * 1024)
                total_mib = total_b // (1024 * 1024)
                used_mib = total_mib - free_mib
            except Exception:
                props = torch.cuda.get_device_properties(idx)
                total_mib = props.total_memory // (1024 * 1024)
                free_mib = total_mib
                used_mib = 0
            gpus.append({
                "index": idx,
                "name": name,
                "memory_free_mib": free_mib,
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
                "util_pct": 0.0,
                "occupied_pids": [],
            })
        return gpus
    except Exception as exc:
        logger.debug("torch.cuda fallback query failed: %s", exc)
        return []


def check_gpu_status() -> list[dict[str, Any]]:
    """Inspects GPUs via nvidia-smi with graceful fallback to PyTorch torch.cuda.

    Returns structured dicts:
        {"index": int, "name": str, "memory_free_mib": int, "memory_used_mib": int,
         "memory_total_mib": int, "util_pct": float, "occupied_pids": list[int]}
    """
    try:
        gpus = _query_nvidia_smi()
        if gpus:
            return gpus
    except Exception as exc:
        logger.debug("nvidia-smi query failed, falling back to torch.cuda: %s", exc)

    return _query_torch_cuda()


def ensure_free_gpu(
    min_free_mib: int = 20000,
    max_util_pct: float = 20.0,
    allow_cpu: bool = False,
) -> int:
    """Checks each detected GPU. Returns the first GPU index meeting requirements.

    Criteria:
      - memory_free_mib >= min_free_mib
      - util_pct <= max_util_pct

    If no GPU qualifies:
      - If allow_cpu is True, returns -1.
      - Otherwise, raises GpuUnavailableError detailing the status and utilization
        of all tested GPUs.
    """
    gpus = check_gpu_status()
    for gpu in gpus:
        if gpu["memory_free_mib"] >= min_free_mib and gpu["util_pct"] <= max_util_pct:
            return gpu["index"]

    if allow_cpu:
        return -1

    if not gpus:
        msg = (
            f"No GPUs detected or accessible (required: min_free_mib >= {min_free_mib}, "
            f"max_util_pct <= {max_util_pct}%). None available."
        )
    else:
        details = "\n".join(
            f"  - GPU {g['index']} ({g['name']}): "
            f"free={g['memory_free_mib']} MiB / {g['memory_total_mib']} MiB "
            f"(used={g['memory_used_mib']} MiB), "
            f"util={g['util_pct']}%, occupied PIDs={g.get('occupied_pids', [])}"
            for g in gpus
        )
        msg = (
            f"No GPU qualifies for preflight check (required: min_free_mib >= {min_free_mib}, "
            f"max_util_pct <= {max_util_pct}%).\n"
            f"Detected GPU statuses:\n{details}"
        )

    raise GpuUnavailableError(msg)
