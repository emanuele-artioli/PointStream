from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from demo.experiments import background_smoke as smoke


def test_stage_epochs_are_reported_separately() -> None:
    assert smoke.stage_epochs({"epoch": 19}, {"epoch": 10}) == {
        "stage0": 19, "stage1": 10,
    }


def test_hash_receipt_is_complete_and_changed_bytes_change_identity(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"first")
    first = smoke.sha256_file(path)
    assert first["complete"] is True
    assert first["bytes"] == 5
    assert first["sha256"] == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"other")
    second = smoke.sha256_file(path)
    assert second["sha256"] == hashlib.sha256(b"other").hexdigest()
    assert second["sha256"] != first["sha256"]


def test_timed_out_hash_never_returns_a_complete_receipt(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "slow.bin"
    path.write_bytes(b"bytes")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("hash worker", 90, output='{"sha256":"partial"}')

    monkeypatch.setattr(smoke.subprocess, "run", timeout)
    with pytest.raises(TimeoutError):
        smoke.sha256_file(path)


def test_hash_timeout_cannot_exceed_project_limit(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        smoke.sha256_file(path, timeout=90.1)


def test_manifest_read_and_output_paths_are_bounded_to_job_directory(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"ok":true}')
    assert smoke.read_json_bounded(source) == {"ok": True}
    job = tmp_path / "job"
    monkeypatch.setenv("PS_JOB_DIR", str(job))
    assert smoke._require_job_path(job / "report.json") == (job / "report.json").resolve()
    with pytest.raises(ValueError, match="inside PS_JOB_DIR"):
        smoke._require_job_path(tmp_path / "outside.json")


def test_strict_checkpoint_shape_validation_allows_only_module_prefix() -> None:
    tensor = SimpleNamespace(shape=(2, 3))
    assert smoke.compare_state_dict_shapes({"weight": tensor}, {"module.weight": tensor}) == {
        "keys": 1, "strict_compatible": True,
    }
    with pytest.raises(ValueError, match="shape_mismatch"):
        smoke.compare_state_dict_shapes({"weight": tensor}, {"weight": SimpleNamespace(shape=(3, 2))})
    with pytest.raises(ValueError, match="collision"):
        smoke.canonical_state_dict({"weight": tensor, "module.weight": tensor})


def test_codec_runtime_configuration_and_stage_checkpoint_weights_must_match() -> None:
    config = {
        "structure": "ld", "normalization": "rgb_0_1", "lambdas": [1.0, 768.0],
        "qp_mapping": list(range(64)), "precision": "float32",
        "extension_path": "/src/extension.so", "extension_version": "abc123",
    }
    smoke.compare_runtime_config(config, dict(config))
    wrong = dict(config, qp_mapping=list(reversed(range(64))))
    with pytest.raises(ValueError, match="runtime_config_mismatch"):
        smoke.compare_runtime_config(config, wrong)
    assert smoke.state_dicts_equal({"module.weight": [1, 2]}, {"weight": [1, 2]})
    assert not smoke.state_dicts_equal({"weight": [1, 2]}, {"weight": [1, 3]})


def test_training_holdout_excluded_seconds_and_room_resets_are_checked() -> None:
    smoke.validate_disjoint_frames(["clip1/f0001", "clip3/390s/f0"], ["clip3/holdout/f120"])
    with pytest.raises(ValueError, match="train_holdout_overlap"):
        smoke.validate_disjoint_frames(["same-frame"], ["same-frame"])
    with pytest.raises(ValueError, match="excluded_in_training"):
        smoke.validate_disjoint_frames(["clip3/210s/f0"], [], excluded_ids=("clip3/210s/f0",))
    smoke.validate_same_room_sequence(["factory001"] * 8, ["sec390"] * 8)
    with pytest.raises(ValueError, match="crosses"):
        smoke.validate_same_room_sequence(["factory001"] * 7 + ["factory002"], ["sec390"] * 8)
    with pytest.raises(ValueError, match="crosses"):
        smoke.validate_same_room_sequence(["factory001"] * 8, ["sec390"] * 7 + ["sec420"])


def test_score_provenance_requires_source_patch_and_checkpoint_hashes() -> None:
    manifest = {
        "code_revision": "e" * 40, "source_tree_sha256": "a" * 64,
        "patch_sha256": "b" * 64, "checkpoint_sha256": "c" * 64,
        "environment": {"python": "/env/bin/python"}, "codec_revision": "d" * 40,
        "gpu_uuid": "GPU-uuid", "command": ["test_video.py", "--qp_i", "21"],
        "encoder": {"path": "/opt/encoder", "version": "encoder 1"},
        "decoder": {"path": "/opt/decoder", "version": "decoder 1"},
        "peak_memory_mib": 8192,
    }
    assert smoke._score_provenance(manifest)["checkpoint_sha256"] == "c" * 64
    del manifest["patch_sha256"]
    with pytest.raises(ValueError, match="patch_sha256"):
        smoke._score_provenance(manifest)
    manifest["patch_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="complete lowercase SHA-256"):
        smoke._score_provenance(manifest)


def test_only_eight_or_32_frame_cuts_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="only 8- or 32"):
        smoke.validate_cut([], start=120, length=16)
