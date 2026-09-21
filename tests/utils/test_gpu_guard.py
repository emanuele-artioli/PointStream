"""Tests for GPU preflight guard utility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.gpu_guard import (
    GpuUnavailableError,
    _parse_nvidia_smi_query,
    _parse_pmon,
    check_gpu_status,
    ensure_free_gpu,
)


def test_parse_nvidia_smi_query() -> None:
    """Verify robust parsing of comma-separated nvidia-smi output."""
    sample_csv = (
        "0, NVIDIA RTX A6000, 49140, 48000, 1140, 5.0\n"
        "1, NVIDIA RTX 3090, 24576, 2100, 22476, 85.2\n"
    )
    gpus = _parse_nvidia_smi_query(sample_csv)
    assert len(gpus) == 2

    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "NVIDIA RTX A6000"
    assert gpus[0]["memory_total_mib"] == 49140
    assert gpus[0]["memory_free_mib"] == 48000
    assert gpus[0]["memory_used_mib"] == 1140
    assert gpus[0]["util_pct"] == 5.0
    assert gpus[0]["occupied_pids"] == []

    assert gpus[1]["index"] == 1
    assert gpus[1]["name"] == "NVIDIA RTX 3090"
    assert gpus[1]["memory_total_mib"] == 24576
    assert gpus[1]["memory_free_mib"] == 2100
    assert gpus[1]["memory_used_mib"] == 22476
    assert gpus[1]["util_pct"] == 85.2


def test_parse_pmon() -> None:
    """Verify parsing of nvidia-smi pmon table lines."""
    sample_pmon = """
# gpu         pid  type    sm    mem    enc    dec    command
# Idx           #   C/G     %      %      %      %    name
    0     10101     C      5      1      -      -    python
    0     10202     C     12      4      -      -    python
    1         -     -      -      -      -      -    -
"""
    pids_by_gpu = _parse_pmon(sample_pmon)
    assert pids_by_gpu == {0: [10101, 10202]}


def test_free_gpu_detection_when_gpu_meets_criteria() -> None:
    """Verify selection of the first qualifying GPU index."""
    mock_gpus = [
        # GPU 0 fails: free memory below 20000
        {
            "index": 0,
            "name": "GPU-0",
            "memory_free_mib": 10000,
            "memory_used_mib": 14000,
            "memory_total_mib": 24000,
            "util_pct": 5.0,
            "occupied_pids": [],
        },
        # GPU 1 passes: free memory >= 20000 and util <= 20%
        {
            "index": 1,
            "name": "GPU-1",
            "memory_free_mib": 32000,
            "memory_used_mib": 16000,
            "memory_total_mib": 48000,
            "util_pct": 10.0,
            "occupied_pids": [],
        },
    ]

    with patch("src.utils.gpu_guard.check_gpu_status", return_value=mock_gpus):
        chosen = ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0)
        assert chosen == 1


def test_free_gpu_detection_with_mocked_subprocess() -> None:
    """Verify end-to-end ensure_free_gpu with mocked nvidia-smi subprocess."""
    smi_output = "0, NVIDIA RTX A6000, 49140, 45000, 4140, 2.0\n"

    def side_effect(cmd: list[str], **kwargs: object) -> MagicMock:
        if "--query-gpu" in cmd[1]:
            return MagicMock(returncode=0, stdout=smi_output)
        if "pmon" in cmd:
            return MagicMock(returncode=0, stdout="# comment\n0 1234 C 1 1 - - python\n")
        return MagicMock(returncode=0, stdout="")

    with patch("subprocess.run", side_effect=side_effect):
        chosen = ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0)
        assert chosen == 0


def test_gpu_unavailable_error_raised_when_all_gpus_fail() -> None:
    """Verify GpuUnavailableError is raised with descriptive details when no GPU qualifies."""
    mock_busy_gpus = [
        {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "memory_free_mib": 5000,
            "memory_used_mib": 44140,
            "memory_total_mib": 49140,
            "util_pct": 92.5,
            "occupied_pids": [9999],
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 3090",
            "memory_free_mib": 18000,
            "memory_used_mib": 6576,
            "memory_total_mib": 24576,
            "util_pct": 45.0,
            "occupied_pids": [8888],
        },
    ]

    with patch("src.utils.gpu_guard.check_gpu_status", return_value=mock_busy_gpus):
        with pytest.raises(GpuUnavailableError) as exc_info:
            ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0)

        err_msg = str(exc_info.value)
        assert "No GPU qualifies for preflight check" in err_msg
        assert "GPU 0 (NVIDIA RTX A6000)" in err_msg
        assert "free=5000 MiB" in err_msg
        assert "util=92.5%" in err_msg
        assert "occupied PIDs=[9999]" in err_msg
        assert "GPU 1 (NVIDIA RTX 3090)" in err_msg


def test_gpu_unavailable_error_when_no_gpus_detected() -> None:
    """Verify GpuUnavailableError when no GPUs exist on system."""
    with patch("src.utils.gpu_guard.check_gpu_status", return_value=[]):
        with pytest.raises(GpuUnavailableError) as exc_info:
            ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0, allow_cpu=False)

        assert "No GPUs detected or accessible" in str(exc_info.value)


def test_allow_cpu_returns_minus_one() -> None:
    """Verify allow_cpu=True returns -1 when GPUs fail criteria or none exist."""
    # Case 1: GPUs fail criteria
    mock_busy = [
        {
            "index": 0,
            "name": "GPU-0",
            "memory_free_mib": 5000,
            "memory_used_mib": 40000,
            "memory_total_mib": 45000,
            "util_pct": 90.0,
            "occupied_pids": [],
        }
    ]
    with patch("src.utils.gpu_guard.check_gpu_status", return_value=mock_busy):
        assert ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0, allow_cpu=True) == -1

    # Case 2: No GPUs detected
    with patch("src.utils.gpu_guard.check_gpu_status", return_value=[]):
        assert ensure_free_gpu(min_free_mib=20000, max_util_pct=20.0, allow_cpu=True) == -1


def test_torch_cuda_fallback_on_nvidia_smi_failure() -> None:
    """Verify fallback to torch.cuda when nvidia-smi is unavailable or errors out."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="Fallback GPU"), \
             patch("torch.cuda.mem_get_info", return_value=(30 * 1024**3, 40 * 1024**3)):
            status = check_gpu_status()
            assert len(status) == 1
            assert status[0]["index"] == 0
            assert status[0]["name"] == "Fallback GPU"
            assert status[0]["memory_free_mib"] == 30 * 1024
            assert status[0]["memory_total_mib"] == 40 * 1024
            assert status[0]["util_pct"] == 0.0


def test_live_execution_on_current_machine() -> None:
    """Live execution test verifying check_gpu_status returns valid details on gpu3."""
    status = check_gpu_status()
    assert isinstance(status, list)
    assert len(status) >= 1, "Expected at least 1 detected GPU on gpu3"

    gpu0 = status[0]
    assert gpu0["index"] == 0
    assert isinstance(gpu0["name"], str)
    assert len(gpu0["name"]) > 0
    assert gpu0["memory_total_mib"] > 0
    assert gpu0["memory_free_mib"] > 0
    assert gpu0["memory_used_mib"] >= 0
    assert 0.0 <= gpu0["util_pct"] <= 100.0
    assert isinstance(gpu0["occupied_pids"], list)

    # Real ensure_free_gpu check with reasonable criteria on this live machine
    live_gpu_idx = ensure_free_gpu(min_free_mib=1000, max_util_pct=95.0)
    assert live_gpu_idx == 0


def test_oracle_ceiling_gpu_guard_integration(tmp_path) -> None:
    """Verify oracle_ceiling CLI respects --ignore-gpu-check and --dry-run flags."""
    from experiments.modular.oracle_ceiling import main as oracle_main

    out_dir = tmp_path / "oracle_out"

    # Case 1: When not dry_run and not ignore_gpu_check, ensure_free_gpu is called
    with patch("experiments.modular.oracle_ceiling.ensure_free_gpu") as mock_guard, \
         patch("sys.argv", ["oracle_ceiling.py", "--output-dir", str(out_dir)]):
        oracle_main()
        mock_guard.assert_called_once()

    # Case 2: When --ignore-gpu-check is provided, ensure_free_gpu is skipped
    with patch("experiments.modular.oracle_ceiling.ensure_free_gpu") as mock_guard, \
         patch("sys.argv", ["oracle_ceiling.py", "--output-dir", str(out_dir), "--ignore-gpu-check"]):
        oracle_main()
        mock_guard.assert_not_called()

    # Case 3: When --dry-run is provided, ensure_free_gpu is skipped
    with patch("experiments.modular.oracle_ceiling.ensure_free_gpu") as mock_guard, \
         patch("sys.argv", ["oracle_ceiling.py", "--output-dir", str(out_dir), "--dry-run"]):
        oracle_main()
        mock_guard.assert_not_called()

