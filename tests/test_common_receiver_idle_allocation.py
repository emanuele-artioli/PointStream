import json
from pathlib import Path

import pytest

from experiments.tennis.common_receiver_job import bind_allocation


@pytest.mark.parametrize("fault", [None, "foreign", "uuid", "memory", "resume"])
def test_exclusive_receiver_allocation(tmp_path: Path, monkeypatch, fault) -> None:
    stage = tmp_path / "smoke"
    stage.mkdir()
    uuid = "GPU-7754164a-33cd-265c-3696-c20966d228fe"
    monkeypatch.setenv("PS_CLAIMED_GPU_UUID", uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if fault == "uuid" else uuid)
    monkeypatch.delenv("PS_GPU_MEMORY_LIMIT_MIB", raising=False)
    sharing = {"mode": "exclusive"}
    gpu = {
        "uuid": uuid,
        "name": "NVIDIA RTX 6000 Ada Generation",
        "pointstream_claimed": False,
        "pointstream_shared_claims": [],
        "compute_processes": [222] if fault == "foreign" else [],
        "memory_used_mib": 9,
        "memory_free_mib": 48640,
        "utilization_pct": 0,
    }
    (stage / "dispatch.json").write_text(
        json.dumps(
            {
                "gpu_sharing": sharing,
                "environment": {"gpu": gpu},
                "contention_policy": {"policy": "stop", "resume_attempts": int(fault == "resume")},
            }
        )
    )
    (tmp_path / "spec.json").write_text(
        json.dumps(
            {
                "gpu_sharing": sharing,
                "gpu_memory_mib": 18500 if fault == "memory" else 18000,
            }
        )
    )
    if fault:
        with pytest.raises(ValueError, match="exclusive idle"):
            bind_allocation(stage)
    else:
        import os

        bind_allocation(stage)
        assert os.environ["PS_GPU_MEMORY_LIMIT_MIB"] == "18000"
