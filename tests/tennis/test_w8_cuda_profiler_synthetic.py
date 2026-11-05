"""Synthetic artifact tests; no CUDA device, weights, or model is used."""

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest
from experiments.tennis import w8_cuda_profiler_synthetic as oracle


def test_gpu_diagnostic_validator_rejects_tampered_trace_and_cpu_fallback(tmp_path, monkeypatch):
    stage = tmp_path / "smoke"
    stage.mkdir()
    root = Path(oracle.__file__).resolve().parents[2]
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as tar:
        for name in oracle.SOURCE:
            tar.add(root / name, arcname=name)

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    input_digest = "5452a5d6bb865152556a0e26dc99798a0ee88dff1c1c86b9a4fd1f5c904413dd"
    synthetic = tmp_path / "synthetic"
    synthetic.write_bytes(b"synthetic")
    actual_sha = oracle.base.sha
    monkeypatch.setattr(
        oracle.base,
        "sha",
        lambda path: (
            input_digest
            if path == synthetic
            else ("tracer" if path == Path("/usr/bin/strace") else actual_sha(path))
        ),
    )
    monkeypatch.setattr(oracle.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(oracle.policy, "validate_leaf_receipt", lambda *args: None)
    uuid = "GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162"
    clock = {"started": 0.0, "phase_started": 5.0, "budget_end": 180.0}
    gpu = {"uuid": uuid}
    dispatch = {
        "code": {"snapshot_sha256": actual_sha(source)},
        "campaign_clock": clock,
        "environment": {"gpu": gpu},
        "gpu_sharing": {"mode": "shared"},
    }
    (stage / "dispatch.json").write_text(json.dumps(dispatch))
    (tmp_path / "ready.json").write_text(json.dumps({"snapshot_sha256": actual_sha(source)}))
    (tmp_path / "spec.json").write_text(
        json.dumps(
            {
                "gpu_memory_mib": 4096,
                "gpu_sharing": {"mode": "shared"},
                "environment": {
                    "sha256": "bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e"
                },
            }
        )
    )
    clock_identity = [
        dict(
            path=str(p),
            device=1,
            inode=i + 1,
            uid=0,
            mode=(stat.S_IFDIR | 0o755) if i == 0 else (stat.S_IFREG | 0o644),
            sha256=None if i == 0 else "a" * 64,
        )
        for i, p in enumerate((oracle.base.CLOCK, *oracle.base.LEAVES))
    ]
    denied = [f"/synthetic/{i}" for i in range(10)]
    content = {}
    rows = []
    for index, arm in enumerate(oracle.ARMS):
        config = {
            "gpu_uuid": uuid,
            "gpu_memory_limit_mib": 4096,
            "denied": denied,
            "loader": {},
            "parent_ns": {"user": "u0", "net": "n0"},
            "clock_identity": clock_identity,
        }
        cuda = {
            "success": True,
            "uuid": uuid,
            "device_count": 1,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "capability": [8, 9],
            "sum": 262144.0,
            "inputs_cuda": True,
            "output_cuda": True,
            "cuda_initialized": True,
            "gpu_memory_limit_mib": 4096,
            "total_memory_bytes": 48 * 2**30,
            "allocator_fraction": 1 / 12,
            "allocator_fraction_before_allocation": 1 / 12,
            "allocator_fraction_before_profile": 1 / 12,
            "allocator_fraction_after_profile": 1 / 12,
            "peak_torch_reserved_bytes": 1024,
        }
        profile = {
            "attempted": True,
            "success": arm != "strict",
            "cuda_kernels": [] if arm == "strict" else ["synthetic_cuda_kernel"],
        }
        if arm == "strict":
            profile["error"] = "RuntimeError: Permission denied clocksource"
        else:
            profile["cuda_events"] = ["synthetic_cuda_kernel"]
        receipt = {
            "arm": arm,
            "scope": oracle.SCOPE,
            "source_origins": {str(i): name for i, name in enumerate(oracle.SOURCE)},
            "visible_uuid": uuid,
            "cuda_initialized_before": False,
            "cuda_initialized_after": True,
            "provider": "torch CUDA",
            "torch_version": "2.7.0+cu128",
            "torch_cuda_version": "12.8",
            "cpu_bootstrap_equal": True,
            "namespaces": {"user": "u1", "net": "n1"},
            "parent_ns": config["parent_ns"],
            "cuda": cuda,
            "profiler": profile,
        }
        if arm != "unrestricted":
            receipt.update(
                {
                    "landlock_abi": 4,
                    "handled_mask": oracle.policy.handled_rights(4),
                    "approved_leaves": [],
                    "denials": dict.fromkeys(denied, 13),
                    "mutations": dict.fromkeys(
                        ("write", "truncate", "create", "remove", "mkdir", "rename", "link"), 13
                    ),
                }
            )
        if arm == "clock":
            receipt["experimental_clock_grants"] = [
                dict(item, rights="READ_DIR recursive" if i == 0 else "READ_FILE", child_uid=65534)
                for i, item in enumerate(clock_identity)
            ]
        data = {
            "trace": b"W8_CUDA_PROFILE_BEGIN\nW8_CUDA_PROFILE_END\n",
            "config": json.dumps(config).encode(),
            "receipt": json.dumps(receipt).encode(),
            "stdout": b"",
            "stderr": b"",
        }
        content.update({f"publish/{arm}.{kind}": value for kind, value in data.items()})
        rows.append(
            {
                "arm": arm,
                "started": 10.0 + 20 * index,
                "ended": 15.0 + 20 * index,
                "artifacts": {k: sha(v) for k, v in data.items()},
                "receipt": receipt,
            }
        )
    projection = {
        "controls_admitted": True,
        "measured_first_stage_seconds": 20.0,
        "remaining_seconds": 135.0,
        "required_seconds": 85.0,
        "reserve_seconds": oracle.RESERVE_SECONDS,
        "multiplier": 1.5,
        "evaluated_at": 20.0,
        "clock_gate": {
            "evaluated_at": 40.0,
            "remaining_seconds": 115.0,
            "required_seconds": 55.0,
            "admitted": True,
        },
    }
    result = {
        "scope": oracle.SCOPE,
        "arms": rows,
        "projection": projection,
        "gpu_uuid": uuid,
        "dispatch_gpu": gpu,
        "campaign_clock": clock,
        "strace_sha256": "tracer",
        "code_sha256": {name: actual_sha(root / name) for name in oracle.SOURCE},
    }
    pub = stage / "published.tar"

    def publish():
        content["publish/cuda-profiler-oracle.json"] = json.dumps(result).encode()
        with tarfile.open(pub, "w") as tar:
            for name, data in content.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        (stage / "staging.json").write_text(
            json.dumps(
                {
                    "inputs": [
                        {
                            "name": "synthetic_reference",
                            "path": str(synthetic),
                            "sha256": input_digest,
                        }
                    ],
                    "published": {"sha256": actual_sha(pub)},
                }
            )
        )

    out = stage / "validation.json"
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(out))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setenv("PS_CLAIMED_GPU_UUID", uuid)
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "4096")
    monkeypatch.setenv("PS_GPU_SHARING", "shared")
    publish()
    oracle.validate()
    assert json.loads(out.read_text())["checks"]["cuda_outcomes_and_kernels"]
    monkeypatch.delenv("PS_GPU_MEMORY_LIMIT_MIB")
    with pytest.raises(ValueError, match="shared allocator reservation"):
        oracle.validate()
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "4096")
    row = result["arms"][1]
    row["receipt"]["cuda"]["allocator_fraction_after_profile"] = 0.5
    content["publish/unrestricted.receipt"] = json.dumps(row["receipt"]).encode()
    row["artifacts"]["receipt"] = sha(content["publish/unrestricted.receipt"])
    publish()
    with pytest.raises(ValueError, match="shared allocator cap differs"):
        oracle.validate()
    row["receipt"]["cuda"]["allocator_fraction_after_profile"] = 1 / 12
    content["publish/unrestricted.receipt"] = json.dumps(row["receipt"]).encode()
    row["artifacts"]["receipt"] = sha(content["publish/unrestricted.receipt"])
    content["publish/strict.trace"] += b"tamper"
    publish()
    with pytest.raises(ValueError, match="raw artifact hash"):
        oracle.validate()
    content["publish/strict.trace"] = content["publish/strict.trace"][:-6]
    row = result["arms"][1]
    row["receipt"]["cuda"]["output_cuda"] = False
    content["publish/unrestricted.receipt"] = json.dumps(row["receipt"]).encode()
    row["artifacts"]["receipt"] = sha(content["publish/unrestricted.receipt"])
    publish()
    with pytest.raises(ValueError, match="physical CUDA proof"):
        oracle.validate()
    row["receipt"]["cuda"]["output_cuda"] = True
    row["receipt"]["profiler"]["cuda_kernels"] = []
    content["publish/unrestricted.receipt"] = json.dumps(row["receipt"]).encode()
    row["artifacts"]["receipt"] = sha(content["publish/unrestricted.receipt"])
    publish()
    with pytest.raises(ValueError, match="actual kernel events"):
        oracle.validate()
    row["receipt"]["profiler"]["cuda_kernels"] = ["synthetic_cuda_kernel"]
    content["publish/unrestricted.receipt"] = json.dumps(row["receipt"]).encode()
    row["artifacts"]["receipt"] = sha(content["publish/unrestricted.receipt"])
    result["projection"]["required_seconds"] = 1.0
    publish()
    with pytest.raises(ValueError, match="control projection differs"):
        oracle.validate()
    # A budget-limited strict arm is valid diagnostic evidence, never a control pass.
    result["arms"] = result["arms"][:1]
    result["projection"] = {
        "controls_admitted": False,
        "measured_first_stage_seconds": 60.0,
        "remaining_seconds": 95.0,
        "required_seconds": 205.0,
        "reserve_seconds": oracle.RESERVE_SECONDS,
        "multiplier": 1.5,
        "evaluated_at": 60.0,
    }
    for arm in ("unrestricted", "clock"):
        for kind in oracle.KINDS:
            content.pop(f"publish/{arm}.{kind}")
    publish()
    oracle.validate()
    assert json.loads(out.read_text())["arms"] == ["strict"]


def test_shared_allocator_is_required_before_any_tensor_allocation():
    events = []

    class CUDA:
        fraction = 0.0

        def set_per_process_memory_fraction(self, value, device):
            events.append("cap")
            self.fraction = value

        def get_per_process_memory_fraction(self, device):
            return self.fraction

        def synchronize(self):
            events.append("synchronize")

    class Tensor:
        is_cuda = True

        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 262144.0

    class Torch:
        cuda = CUDA()
        float32 = "float32"

        @staticmethod
        def ones(*args, **kwargs):
            assert events == ["cap"]
            events.append("allocation")
            return Tensor()

    for invalid_limit, total in ((0, 48 * 2**30), (8192, 48 * 2**30), (4096, 1)):
        with pytest.raises(ValueError):
            oracle.prepare_cuda_probe(Torch, invalid_limit, total)
        assert events == []
    _, _, value, fraction, observed = oracle.prepare_cuda_probe(Torch, 4096, 48 * 2**30)
    assert value == 262144.0 and fraction == observed == 1 / 12
    assert events == ["cap", "allocation", "synchronize"]
    Torch.cuda.fraction = 0.5
    with pytest.raises(ValueError, match="cap differs"):
        oracle.verify_allocator(Torch, fraction)
