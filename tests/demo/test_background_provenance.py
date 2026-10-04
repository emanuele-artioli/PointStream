from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from demo.experiments import background_smoke as smoke
from demo.experiments import background_smoke_core as core


def test_stage_epochs_stay_separate_and_are_never_summed() -> None:
    epochs = core.stage_epochs({"epoch": 19}, {"epoch": 10})
    assert epochs == {"stage0": 19, "stage1": 10}
    assert 29 not in epochs.values() and 46 not in epochs.values()
    with pytest.raises(ValueError, match="stage1"):
        core.stage_epochs({"epoch": 19}, {"epoch": True})


def test_complete_receipt_and_changed_bytes_invalidate_cache(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth.tar"
    path.write_bytes(b"weights-a")
    receipt = core.sha256_file(path, timeout=30)
    assert receipt["complete"] is True and receipt["sha256"] == core.sha256_bytes(b"weights-a")
    cache = {str(path.absolute()): receipt}
    assert core.cached_identity(path, cache) == receipt
    path.write_bytes(b"weights-b")
    assert core.cached_identity(path, cache) is None


def test_timeout_never_produces_a_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "slow.bin"
    path.write_bytes(b"x")

    def slow(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="hash", timeout=kwargs["timeout"])

    monkeypatch.setattr(core.subprocess, "run", slow)
    with pytest.raises(TimeoutError):
        core.sha256_file(path, timeout=1)
    with pytest.raises(ValueError, match="timeout"):
        core.sha256_file(path, timeout=91)


def test_verified_copy_rejects_wrong_identity_and_keeps_source(tmp_path: Path) -> None:
    source = tmp_path / "ckpt.pth.tar"
    source.write_bytes(b"checkpoint bytes")
    copy = tmp_path / "scratch" / "copy.pth.tar"
    with pytest.raises(ValueError, match="identity changed"):
        core.copy_verified(source, copy, "0" * 64, timeout=30)
    assert not copy.exists() and source.read_bytes() == b"checkpoint bytes"
    receipt = core.copy_verified(source, copy, core.sha256_bytes(b"checkpoint bytes"), timeout=30)
    assert copy.read_bytes() == source.read_bytes() and receipt["bytes"] == len(b"checkpoint bytes")
    with pytest.raises(FileExistsError):
        core.copy_verified(source, copy, receipt["sha256"], timeout=30)


class Tensor:
    def __init__(self, *shape: int, dtype: str = "float32") -> None:
        self.shape, self.dtype = shape, dtype


def test_strict_shape_check_rejects_missing_extra_and_mismatched_keys() -> None:
    expected = {"a.weight": Tensor(4, 3), "b.bias": Tensor(4)}
    assert core.compare_state_dict_shapes(expected, {"module.a.weight": Tensor(4, 3), "module.b.bias": Tensor(4)})["strict_compatible"]
    with pytest.raises(ValueError, match="missing"):
        core.compare_state_dict_shapes(expected, {"a.weight": Tensor(4, 3)})
    with pytest.raises(ValueError, match="shape_mismatch"):
        core.compare_state_dict_shapes(expected, {"a.weight": Tensor(4, 4), "b.bias": Tensor(4)})
    with pytest.raises(ValueError, match="extra"):
        core.compare_state_dict_shapes(expected, {**expected, "c.weight": Tensor(1)})


def test_only_a_leading_module_prefix_is_removed() -> None:
    state = core.canonical_state_dict({"module.encoder.module.x": 1, "decoder.y": 2})
    assert set(state) == {"encoder.module.x", "decoder.y"}
    with pytest.raises(ValueError, match="collision"):
        core.canonical_state_dict({"module.x": 1, "x": 2})


def test_runtime_configuration_mismatch_fails_before_inference() -> None:
    config = {"structure": "htl", "normalization": "ycbcr", "lambdas": [1, 768], "qp_mapping": "uf", "precision": "fp16",
              "extension_path": "/x.so", "extension_version": "1"}
    core.compare_runtime_config(config, dict(config))
    with pytest.raises(ValueError, match="structure"):
        core.compare_runtime_config(config, {**config, "structure": "ld"})
    with pytest.raises(ValueError, match="incomplete"):
        core.compare_runtime_config(config, {"structure": "htl"})


def _description(rows: list[str]) -> dict:
    return {
        "seqs": [{"path": name, "height": 1080, "width": 1920, "seq_length": 30} for name in rows],
        "frames": [f"{i:06d}.jpg" for i in range(30)],
    }


CLIP1 = "clip_01_factory001_worker001_00001"
CLIP3 = "clip_03_factory001_worker001_00000"


def test_training_membership_policy_passes_and_reports_clip3_second0() -> None:
    report = core.training_identity_report(_description([f"{CLIP3}_t000000", f"{CLIP1}_t000030"]), "factory001", {})
    assert report["clip3_second0_present"] is True
    assert report["excluded_seconds_absent"] == {CLIP3: [210, 240, 420]}
    assert report["training_frames"] == 60


@pytest.mark.parametrize("rows, message", [
    ([f"{CLIP3}_t000210"], "excluded_in_training"),
    (["factory002_worker001_00000_t000000"], "not from factory001"),
    ([f"{CLIP1}_t000010", f"{CLIP1}_t000010"], "duplicate sampled second"),
    ([f"{CLIP1}_000010"], "unrecognized"),
])
def test_training_policy_violations_are_detected(rows: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        core.training_identity_report(_description(rows), "factory001", {})


def test_holdout_identity_intersection_is_detected() -> None:
    with pytest.raises(ValueError, match="train_holdout_overlap"):
        core.training_identity_report(_description([f"{CLIP1}_t000010"]), "factory001", {CLIP1: range(300, 330)})


def test_temporal_sequence_may_not_cross_rooms_or_seconds() -> None:
    core.validate_same_room_sequence(["f1", "f1"], ["s0", "s0"])
    with pytest.raises(ValueError, match="crosses"):
        core.validate_same_room_sequence(["f1", "f2"], ["s0", "s0"])
    with pytest.raises(ValueError, match="crosses"):
        core.validate_same_room_sequence(["f1", "f1"], ["s0", "s1"])


def test_every_score_requires_source_diff_and_checkpoint_identity() -> None:
    record = {
        "provenance": {"code_revision": "a" * 40, "source_tree_sha256": "b" * 64, "gpu_uuid": "GPU-1"},
        "checkpoints": {"image": {"sha256": "c" * 64}},
        "source_frames": [{"sha256": "d" * 64}],
        "codec_source": {"tracked_diff_sha256": "e" * 64},
    }
    assert core.require_score_provenance(record)["gpu_uuid"] == "GPU-1"
    with pytest.raises(ValueError, match="checkpoint"):
        core.require_score_provenance({**record, "checkpoints": {"image": {"sha256": "short"}}})
    with pytest.raises(ValueError, match="diff"):
        core.require_score_provenance({**record, "codec_source": {}})
    with pytest.raises(ValueError, match="provenance"):
        core.require_score_provenance({**record, "provenance": {}})


def test_serialized_state_comparison_is_labelled_not_byte_exact() -> None:
    import zipfile

    storage = core.StorageRef("float32", "0", "cpu", 4)
    tensor = core.TensorRef(storage, 0, (4,), (1,))
    info = zipfile.ZipInfo("archive/data/0")
    info.file_size, info.CRC = 16, 123
    other = zipfile.ZipInfo("archive/data/0")
    other.file_size, other.CRC = 16, 456
    same = core.serialized_state_equality(({"module.w": tensor}, {"0": info}), ({"w": tensor}, {"0": info}))
    assert same["all_storage_crc32_equal"] and "not byte-exact" in same["method"]
    differ = core.serialized_state_equality(({"w": tensor}, {"0": info}), ({"w": tensor}, {"0": other}))
    assert differ["crc_mismatched_keys"] == ["w"]


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict | None = None) -> smoke.Stage:
    root = tmp_path / "inbox" / "job" / "smoke"
    root.mkdir(parents=True)
    monkeypatch.setenv("PS_STAGE_DIR", str(root))
    monkeypatch.setattr(smoke, "SCRATCH_ROOT", tmp_path / "scratch")
    paths = []
    if manifest is not None:
        path = tmp_path / "selected-inputs.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    return smoke.Stage("inventory", 60, paths)


def test_part_outcomes_and_second_identical_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _stage(tmp_path, monkeypatch)

    def blocked():
        raise smoke.PartBlocked("missing extension")

    def broken():
        raise smoke.PartFailed("strict load failed")

    assert stage.run_part("a", blocked)["status"] == "blocked"
    assert stage.run_part("b", broken)["status"] == "failed"
    assert stage.run_part("c", broken)["status"] == "failed"
    third = stage.run_part("d", lambda: {"ok": True})
    assert third["status"] == "blocked" and "second identical failure" in third["reason"]
    assert json.loads((stage.root / "parts" / "b.json").read_text())["reason"].endswith("strict load failed")
    ledger = json.loads((stage.root / "ledger.json").read_text())
    assert [row["status"] for row in ledger["operations"]] == ["blocked", "failed", "failed", "blocked"]


def test_unverified_checkpoint_blocks_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"schema": "pointstream.background-smoke.inputs.v1", "checkpoints": {"uf_image": {"path": "/x", "sha256": "bad"}}}
    stage = _stage(tmp_path, monkeypatch, manifest)
    with pytest.raises(smoke.PartBlocked, match="uf_image"):
        stage.checkpoint("uf_image")
    with pytest.raises(smoke.PartBlocked, match="uf_video"):
        stage.checkpoint("uf_video:htl:pretrained")


def test_conflicting_manifests_are_rejected() -> None:
    base = {"schema": "pointstream.background-smoke.inputs.v1", "checkpoints": {"k": {"sha256": "a" * 64}}}
    other = {"schema": "pointstream.background-smoke.inputs.v1", "checkpoints": {"k": {"sha256": "b" * 64}}}
    assert smoke.merge_manifests([base, base])["checkpoints"]["k"]["sha256"] == "a" * 64
    with pytest.raises(ValueError, match="disagree"):
        smoke.merge_manifests([base, other])
    with pytest.raises(ValueError, match="schema"):
        smoke.merge_manifests([{"schema": "other"}])


def test_outputs_must_stay_inside_the_stage_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _stage(tmp_path, monkeypatch)
    assert stage.path("parts", "x.json").parent.name == "parts"
    with pytest.raises(ValueError, match="PS_STAGE_DIR"):
        core.require_stage_path(tmp_path / "elsewhere.json")
    core.write_json_new(stage.root / "result.json", {"a": 1})
    with pytest.raises(FileExistsError):
        core.write_json_new(stage.root / "result.json", {"a": 2})


def test_hash_children_do_not_import_the_model_or_numpy(tmp_path, monkeypatch):
    original = core.subprocess.run
    commands = []
    def recording(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)
    monkeypatch.setattr(core.subprocess, "run", recording)
    path = tmp_path / "input.bin"
    path.write_bytes(b"source")
    assert core.sha256_file(path)["sha256"] == core.sha256_bytes(b"source")
    script = Path(commands[0][1])
    assert script.name == "background_smoke_io.py"
    assert "import numpy" not in script.read_text() and "import torch" not in script.read_text()


def test_hnerv_missing_dependencies_block_without_stub_modules(tmp_path, monkeypatch):
    import importlib.util
    from demo.experiments import hnerv_frozen
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="installed dependencies unavailable"):
        hnerv_frozen.enable_imports(tmp_path / "stubs")
    assert not (tmp_path / "stubs").exists()
