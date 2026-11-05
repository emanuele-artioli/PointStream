"""Checks that the oracle validator binds its raw trace evidence."""

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest
from experiments.tennis import w8_profiler_clock_oracle as oracle


def test_validator_checks_full_trace_and_rejects_tampering(tmp_path, monkeypatch):
    stage = tmp_path / "smoke"
    stage.mkdir()
    source = tmp_path / "source.tar"
    root = Path(oracle.__file__).resolve().parents[2]
    with tarfile.open(source, "w") as tar:
        for name in oracle.SOURCE:
            tar.add(root / name, arcname=name)

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    identity = "5452a5d6bb865152556a0e26dc99798a0ee88dff1c1c86b9a4fd1f5c904413dd"
    synthetic = tmp_path / "synthetic"
    synthetic.write_bytes(b"synthetic")
    actual_sha = oracle.sha
    monkeypatch.setattr(
        oracle,
        "sha",
        lambda path: (
            identity
            if path == synthetic
            else ("tracer" if path == Path("/usr/bin/strace") else actual_sha(path))
        ),
    )
    monkeypatch.setattr(oracle.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(oracle.policy, "validate_leaf_receipt", lambda *args: None)
    clock = []
    for i, path in enumerate((oracle.CLOCK, *oracle.LEAVES)):
        clock.append(
            {
                "path": str(path),
                "device": 1,
                "inode": i + 1,
                "uid": 0,
                "mode": (stat.S_IFDIR | 0o755) if i == 0 else (stat.S_IFREG | 0o644),
                "sha256": None if i == 0 else "a" * 64,
            }
        )
    content = {}
    rows = []
    markers = b"".join(
        f'write(2, "W8_PROFILE_{label}_{phase}")\n'.encode()
        for label in ("cpu", "requested_cpu_cuda")
        for phase in ("BEGIN", "END")
    )
    denied = [f"/synthetic/{i}" for i in range(10)]
    for arm in ("unrestricted", "strict", "clock"):
        config = {
            "parent_ns": {"user": "u0", "net": "n0"},
            "denied": denied,
            "loader": {},
            "clock_identity": clock,
        }
        receipt = {
            "arm": arm,
            "source_origins": {str(i): x for i, x in enumerate(oracle.SOURCE)},
            "cuda_visible_devices": "",
            "cuda_before": False,
            "cuda_after": False,
            "bootstrap_equal": True,
            "namespaces": {"user": "u1", "net": "n1"},
            "parent_ns": config["parent_ns"],
            "results": {
                "cpu": {"success": True, "warnings": []},
                "requested_cpu_cuda": {"success": True, "warnings": []},
            },
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
                dict(
                    x,
                    rights="READ_DIR recursive" if i == 0 else "READ_FILE",
                    sha256_before=x["sha256"],
                )
                for i, x in enumerate(clock)
            ]
        data = {
            "trace": markers,
            "config": json.dumps(config).encode(),
            "receipt": json.dumps(receipt).encode(),
            "stdout": b"",
            "stderr": b"",
        }
        content.update({f"publish/{arm}.{k}": v for k, v in data.items()})
        rows.append(
            dict(
                receipt,
                artifacts={k: sha(v) for k, v in data.items()},
                clocksource_trace=[],
                trace_sha256=sha(markers),
            )
        )
    result = {
        "scope": "CPU only; no model or GPU qualification",
        "arms": rows,
        "strace_sha256": "tracer",
        "child_cpu_seconds": 3,
        "code_sha256": {name: actual_sha(root / name) for name in oracle.SOURCE},
    }
    content["publish/profiler-oracle.json"] = json.dumps(result).encode()
    pub = stage / "published.tar"

    def publish():
        with tarfile.open(pub, "w") as tar:
            for name, data in content.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        (stage / "staging.json").write_text(
            json.dumps(
                {
                    "inputs": [
                        {"name": "synthetic_reference", "path": str(synthetic), "sha256": identity}
                    ],
                    "published": {"sha256": actual_sha(pub)},
                }
            )
        )

    (stage / "dispatch.json").write_text(
        json.dumps({"code": {"snapshot_sha256": actual_sha(source)}})
    )
    out = stage / "validation.json"
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(out))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    publish()
    oracle.validate()
    assert json.loads(out.read_text())["checks"]["publication_identity_and_raw_traces"]
    content["publish/strict.trace"] += b"tampered"
    publish()
    with pytest.raises(ValueError, match="raw artifact digest"):
        oracle.validate()
