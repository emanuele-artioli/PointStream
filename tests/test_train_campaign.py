"""Fast unit tests for scripts/train_campaign.py's pure logic.

No subprocess/training/GPU here: command building, ranking/promotion math,
the non-overlap safety check, and state persistence. The real rung execution
is covered by the end-to-end smoke run (report 10's Phase 5.4 findings
entry), not by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_campaign import (
    Variant,
    build_train_command,
    checkpoint_path_for_eval,
    default_variants,
    init_state,
    load_state,
    promote_survivors,
    rank_variants,
    save_state,
    variants_from_state,
    verify_data_root_excludes_probe_set,
)


def test_default_variants_are_the_three_wrapped_scripts() -> None:
    variants = default_variants()
    kinds = {v.kind for v in variants}
    assert kinds == {"pix2pix", "spade4tennis", "controlnet"}
    names = [v.name for v in variants]
    assert len(names) == len(set(names))  # unique names


def test_checkpoint_path_for_eval_per_kind(tmp_path: Path) -> None:
    pix2pix = Variant(name="p", arch="pix2pix", kind="pix2pix")
    spade = Variant(name="s", arch="spade4tennis", kind="spade4tennis", model_size="lite")
    cnet = Variant(name="c", arch="controlnet", kind="controlnet", condition_type="pose")

    assert checkpoint_path_for_eval(pix2pix, tmp_path) == tmp_path / "pix2pix_generator.pt"
    assert checkpoint_path_for_eval(spade, tmp_path) == tmp_path / "spade4tennis_lite_generator.pt"
    assert checkpoint_path_for_eval(cnet, tmp_path) == tmp_path


def test_checkpoint_path_for_eval_rejects_unknown_kind(tmp_path: Path) -> None:
    bogus = Variant(name="x", arch="x", kind="not-a-kind")
    with pytest.raises(ValueError, match="unknown variant kind"):
        checkpoint_path_for_eval(bogus, tmp_path)


def test_build_train_command_pix2pix_uses_absolute_epochs_and_resume(tmp_path: Path) -> None:
    variant = Variant(name="pix2pix", arch="pix2pix", kind="pix2pix")
    cmd = build_train_command(variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=4, rung_epochs=2, resume=True, python_bin="python")
    assert "scripts/train_pix2pix.py" in cmd
    assert "--epochs" in cmd and cmd[cmd.index("--epochs") + 1] == "4"
    assert "--resume" in cmd


def test_build_train_command_pix2pix_rung0_has_no_resume(tmp_path: Path) -> None:
    variant = Variant(name="pix2pix", arch="pix2pix", kind="pix2pix")
    cmd = build_train_command(variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=2, rung_epochs=2, resume=False, python_bin="python")
    assert "--resume" not in cmd


def test_build_train_command_spade4tennis_includes_model_size(tmp_path: Path) -> None:
    variant = Variant(name="spade4tennis_lite", arch="spade4tennis", kind="spade4tennis", model_size="lite")
    cmd = build_train_command(variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=3, rung_epochs=3, resume=False, python_bin="python")
    assert "--model-size" in cmd and cmd[cmd.index("--model-size") + 1] == "lite"
    assert cmd[cmd.index("--out-weights") + 1].endswith("spade4tennis_lite_generator.pt")


def test_build_train_command_controlnet_uses_delta_epochs_and_from_scratch_on_rung0(tmp_path: Path) -> None:
    variant = Variant(name="controlnet_pose", arch="controlnet", kind="controlnet", condition_type="pose")
    cmd = build_train_command(variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=2, rung_epochs=2, resume=False, python_bin="python")
    assert "--epochs" in cmd and cmd[cmd.index("--epochs") + 1] == "2"  # delta, not cumulative
    assert "--from-scratch" in cmd
    assert "--controlnet-model-id" not in cmd


def test_build_train_command_controlnet_resumes_from_previous_dir(tmp_path: Path) -> None:
    variant = Variant(name="controlnet_pose", arch="controlnet", kind="controlnet", condition_type="pose")
    ckpt_dir = tmp_path / "ckpt"
    cmd = build_train_command(variant, tmp_path / "data", ckpt_dir, cumulative_epochs=4, rung_epochs=2, resume=True, python_bin="python")
    assert "--epochs" in cmd and cmd[cmd.index("--epochs") + 1] == "2"  # rung_epochs, delta
    assert "--controlnet-model-id" in cmd and cmd[cmd.index("--controlnet-model-id") + 1] == str(ckpt_dir)
    assert "--from-scratch" not in cmd


def test_build_train_command_rejects_unknown_kind(tmp_path: Path) -> None:
    bogus = Variant(name="x", arch="x", kind="not-a-kind")
    with pytest.raises(ValueError, match="unknown variant kind"):
        build_train_command(bogus, tmp_path, tmp_path, 1, 1, False)


def test_verify_data_root_excludes_probe_set_clean(tmp_path: Path) -> None:
    manifest = {"excluded_training_keys": ["video_a/scene_001/track_0001"]}
    violations = verify_data_root_excludes_probe_set(tmp_path, manifest)
    assert violations == []


def test_verify_data_root_excludes_probe_set_detects_leak(tmp_path: Path) -> None:
    leaked = tmp_path / "video_a" / "segmentations" / "scene_001" / "track_0001"
    leaked.mkdir(parents=True)
    manifest = {"excluded_training_keys": ["video_a/scene_001/track_0001", "video_b/scene_002/track_0002"]}
    violations = verify_data_root_excludes_probe_set(tmp_path, manifest)
    assert violations == ["video_a/scene_001/track_0001"]


def test_rank_variants_prefers_higher_psnr_lower_fvd() -> None:
    aggregate = {
        "good": {"psnr_mean": 32.0, "ssim_mean": 0.9, "vmaf_mean": 80.0, "fvd": 1.0, "lpips_vgg_uncalibrated": 0.1},
        "bad": {"psnr_mean": 20.0, "ssim_mean": 0.6, "vmaf_mean": 40.0, "fvd": 10.0, "lpips_vgg_uncalibrated": 0.5},
    }
    ranked, composite = rank_variants(aggregate)
    assert ranked[0] == "good"
    assert composite["good"] > composite["bad"]


def test_rank_variants_skips_metrics_reported_by_fewer_than_two() -> None:
    aggregate = {
        "a": {"psnr_mean": 30.0, "ssim_mean": None, "vmaf_mean": None, "fvd": None, "lpips_vgg_uncalibrated": None},
        "b": {"psnr_mean": 25.0, "ssim_mean": None, "vmaf_mean": None, "fvd": None, "lpips_vgg_uncalibrated": None},
    }
    ranked, composite = rank_variants(aggregate)
    assert ranked[0] == "a"  # only psnr_mean is usable, a > b
    assert composite["a"] == 1.0
    assert composite["b"] == 0.0


def test_rank_variants_handles_all_metrics_missing() -> None:
    aggregate = {"a": {}, "b": {}}
    ranked, composite = rank_variants(aggregate)
    assert set(ranked) == {"a", "b"}
    assert composite["a"] == 0.0
    assert composite["b"] == 0.0


def test_promote_survivors_keeps_ceil_half() -> None:
    assert promote_survivors(["a", "b", "c"]) == ["a", "b"]
    assert promote_survivors(["a", "b"]) == ["a"]
    assert promote_survivors(["a"]) == ["a"]
    assert promote_survivors([]) == []


def test_init_state_and_roundtrip(tmp_path: Path) -> None:
    variants = default_variants()
    state = init_state(variants)
    assert state["rung"] == 0
    assert set(state["alive"]) == {v.name for v in variants}
    assert all(v == 0 for v in state["cumulative_epochs"].values())

    path = tmp_path / "state.json"
    save_state(path, state)
    loaded = load_state(path)
    assert loaded == state

    restored_variants = variants_from_state(loaded)
    assert {v.name for v in restored_variants.values()} == {v.name for v in variants}


def test_state_json_is_serializable(tmp_path: Path) -> None:
    state = init_state(default_variants())
    path = tmp_path / "state.json"
    save_state(path, state)
    # Just verify plain json can parse it (no custom types leaked in).
    assert isinstance(json.loads(path.read_text()), dict)
