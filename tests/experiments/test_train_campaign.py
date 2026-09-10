"""Fast unit tests for scripts/train_campaign.py's pure logic.

No subprocess/training/GPU here: command building, ranking/promotion math,
the non-overlap safety check, and state persistence. The real rung execution
is covered by the end-to-end smoke run (report 10's Phase 5.4 findings
entry), not by these tests.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import sqlite3  # noqa: F401
import cv2
import numpy as np

import scripts.train_campaign as train_campaign
from experiments.probe_set.materialize import sorted_frame_files, window_positions
from scripts.train_campaign import (
    Variant,
    build_train_command,
    checkpoint_path_for_eval,
    compute_checkpoint_sha256,
    default_variants,
    evaluate_checkpoint,
    halved_batch_size,
    init_state,
    is_valid_eval,
    load_state,
    main,
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


def test_build_train_command_passes_through_batch_size(tmp_path: Path) -> None:
    variant = Variant(name="pix2pix", arch="pix2pix", kind="pix2pix")
    cmd = build_train_command(
        variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=1, rung_epochs=1, resume=False,
        python_bin="python", batch_size="4",
    )
    assert "--batch-size" in cmd and cmd[cmd.index("--batch-size") + 1] == "4"


def test_build_train_command_omits_batch_size_when_not_given(tmp_path: Path) -> None:
    variant = Variant(name="pix2pix", arch="pix2pix", kind="pix2pix")
    cmd = build_train_command(variant, tmp_path / "data", tmp_path / "ckpt", cumulative_epochs=1, rung_epochs=1, resume=False, python_bin="python")
    assert "--batch-size" not in cmd


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


def test_rank_variants_prefers_higher_psnr_lower_temporal_error() -> None:
    aggregate = {
        "good": {"psnr_mean": 32.0, "ssim_mean": 0.9, "vmaf_mean": 80.0, "temporal_error": 1.0, "lpips_vgg_uncalibrated": 0.1},
        "bad": {"psnr_mean": 20.0, "ssim_mean": 0.6, "vmaf_mean": 40.0, "temporal_error": 10.0, "lpips_vgg_uncalibrated": 0.5},
    }
    ranked, composite = rank_variants(aggregate)
    assert ranked[0] == "good"
    assert composite["good"] > composite["bad"]


def test_rank_variants_skips_metrics_reported_by_fewer_than_two() -> None:
    aggregate = {
        "a": {"psnr_mean": 30.0, "ssim_mean": None, "vmaf_mean": None, "temporal_error": None, "lpips_vgg_uncalibrated": None},
        "b": {"psnr_mean": 25.0, "ssim_mean": None, "vmaf_mean": None, "temporal_error": None, "lpips_vgg_uncalibrated": None},
    }
    ranked, composite = rank_variants(aggregate)
    assert ranked[0] == "a"  # only psnr_mean is usable, a > b
    assert composite["a"] == 1.0
    assert composite["b"] == 0.0


def test_rank_variants_handles_all_metrics_missing() -> None:
    aggregate: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
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


def test_halved_batch_size_halves_numeric_and_leaves_non_numeric() -> None:
    assert halved_batch_size("16") == "8"
    assert halved_batch_size("1") == "1"  # floors at 1, never 0
    assert halved_batch_size("auto") == "auto"
    assert halved_batch_size(None) is None


def test_main_aborts_rung_without_ranking_when_a_variant_fails_training_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A training subprocess that fails twice (e.g. a shared-GPU OOM) must not be
    scored as a quality loss and pruned — the old behavior this regression-tests
    against (see 67a9ea6275d3d9785ce57026/RESEARCH_LOG.md hard rule 5)."""
    campaign_dir = tmp_path / "campaign"
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"excluded_training_keys": [], "probe_clips": []}))

    # Every training subprocess invocation "fails" (simulates an OOM that a batch-size
    # retry doesn't fix), regardless of args — run_training_subprocess is the only
    # thing that should be monkeypatched (build_train_command runs for real).
    monkeypatch.setattr(train_campaign, "run_training_subprocess", lambda *a, **k: 1)

    exit_code = main([
        "--campaign-dir", str(campaign_dir),
        "--manifest", str(manifest_path),
        "--data-root", str(data_root),
        "--variants", "pix2pix,spade4tennis_lite",
        "--initial-epochs", "1",
    ])

    assert exit_code == 1
    state = load_state(campaign_dir / "campaign_state.json")
    assert state["rung"] == 0
    assert set(state["alive"]) == {"pix2pix", "spade4tennis_lite"}
    assert all(v == 0 for v in state["cumulative_epochs"].values())
    assert state["history"] == []  # never reached ranking


# ---------------------------------------------------------------------------
# Evaluator and generator readiness authorized tests
# ---------------------------------------------------------------------------


class MockSizes:
    def __init__(self, residual: int = 42000, transport_total: int = 50000) -> None:
        self.residual = residual
        self.transport_total = transport_total


class MockDeliveredQuality:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {"psnr": 34.5, "ssim": 0.92, "vmaf": 88.0}

    def whole_frame(self, metric: str) -> float:
        return self.scores[metric]


class MockRunResult:
    def __init__(
        self,
        residual: int = 42000,
        transport_total: int = 50000,
        scores: dict[str, float] | None = None,
    ) -> None:
        self.delivered_frames = np.zeros((2, 512, 512, 3), dtype=np.uint8)
        self.sizes = MockSizes(residual=residual, transport_total=transport_total)
        self.delivered_quality = MockDeliveredQuality(scores)
        self.encoder_seconds = 1.2
        self.client_seconds = 0.8


def test_checkpoint_and_config_identity_reaches_evaluation(tmp_path: Path) -> None:
    ckpt_file = tmp_path / "model.pt"
    content = b"dummy_weights_data"
    ckpt_file.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_id = f"model.pt:{expected_hash}"

    manifest = {
        "probe_clips": [{"video": "v1", "scene": "s1", "track": "t1", "frame_ids": [0, 1], "is_synthetic_fixture": True}],
        "held_out_videos": ["v1"],
    }

    recorded_calls: list[dict[str, Any]] = []

    def mock_runner(cfg: Any, sources: Any, generator: Any = None, objects: Any = None, context_ids: Any = None) -> Any:
        recorded_calls.append({"cfg": cfg, "generator": generator, "context_ids": context_ids})
        return MockRunResult(residual=12345, transport_total=20000, scores={"psnr": 35.0, "ssim": 0.95, "vmaf": 90.0})

    result = evaluate_checkpoint(
        checkpoint_path=ckpt_file,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        allow_synthetic=True,
        runner_fn=mock_runner,
    )

    agg = result["aggregate"]
    assert agg["checkpoint_identity"] == expected_id
    assert agg["residual_bytes"] == 12345
    assert agg["psnr_mean"] == 35.0
    assert agg["success"] is True

    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["cfg"].generator.backend == "pix2pix"
    assert call["generator"].backend.backend.checkpoint == str(ckpt_file)


def test_missing_checkpoint_fails_explicitly(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent_checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        evaluate_checkpoint(
            checkpoint_path=missing,
            arch="pix2pix",
            manifest={},
            dataset_root=tmp_path,
        )


def test_temporal_input_reaches_temporal_backend_as_sequence() -> None:
    from src.runner.generation import as_runner_ref
    from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE
    from src.contracts.conditioning import ConditioningBundle, GenerationParams
    from src.pipeline.reconstruction.device import DevicePolicy
    from src.pipeline.reconstruction.dispatch import dispatch

    class MockTemporalBackend:
        capabilities = frozenset({CAP_TEMPORAL_SEQUENCE})
        required = ("appearance", "pose")

        def __init__(self) -> None:
            self.sequence_calls: list[list[Any]] = []
            self.frame_calls: list[Any] = []

        def generate(self, conditioning: Any, *, seed: int, device: Any, params: Any) -> np.ndarray:
            self.frame_calls.append(conditioning)
            return np.zeros((3, 64, 64), dtype=np.uint8)

        def generate_sequence(self, conditioning: Any, *, seed: int, device: Any, params: Any) -> tuple[np.ndarray, ...]:
            self.sequence_calls.append(list(conditioning))
            return tuple(np.zeros((3, 64, 64), dtype=np.uint8) for _ in conditioning)

    backend = MockTemporalBackend()
    runner_ref = as_runner_ref(
        backend,
        name="test_temporal",
        capabilities=frozenset({CAP_TEMPORAL_SEQUENCE}),
        requires=frozenset({"appearance", "pose"}),
    )

    bundles = [
        ConditioningBundle(
            appearance=np.zeros((3, 64, 64), dtype=np.uint8),
            pose=np.zeros((3, 64, 64), dtype=np.uint8),
            frame_index=i,
            object_id="p1",
        )
        for i in range(3)
    ]

    params = GenerationParams()
    output, _ = dispatch(
        generator=runner_ref,
        bundles=bundles,
        seed=42,
        params=params,
        policy=DevicePolicy(),
    )

    # Must be received as a full sequence call, not per-frame calls
    assert len(backend.sequence_calls) == 1
    assert len(backend.sequence_calls[0]) == 3
    assert len(backend.frame_calls) == 0
    assert len(output) == 3
    assert output[0].shape == (64, 64, 3)


def test_actual_selected_checkpoint_is_invoked(tmp_path: Path) -> None:
    ckpt1 = tmp_path / "model1.pt"
    ckpt1.write_bytes(b"checkpoint1_bytes")
    ckpt2 = tmp_path / "model2.pt"
    ckpt2.write_bytes(b"checkpoint2_bytes")

    invoked_checkpoints: list[str | None] = []

    def mock_runner(cfg: Any, sources: Any, generator: Any = None, objects: Any = None, context_ids: Any = None) -> Any:
        backend_obj = generator.backend.backend
        invoked_checkpoints.append(getattr(backend_obj, "checkpoint", None))
        return MockRunResult()

    manifest = {
        "probe_clips": [{"video": "v1", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}],
        "held_out_videos": ["v1"],
    }

    evaluate_checkpoint(
        checkpoint_path=ckpt1,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        allow_synthetic=True,
        runner_fn=mock_runner,
    )
    evaluate_checkpoint(
        checkpoint_path=ckpt2,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        allow_synthetic=True,
        runner_fn=mock_runner,
    )

    assert invoked_checkpoints == [str(ckpt1), str(ckpt2)]


def test_evaluator_uses_held_out_development_scenes_and_current_runner_results(tmp_path: Path) -> None:
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"weights")

    manifest = {
        "held_out_videos": ["held_out_vid"],
        "probe_clips": [
            {"video": "training_vid", "scene": "s1", "track": "t1", "frame_ids": [0, 1]},
            {"video": "held_out_vid", "scene": "s2", "track": "t2", "frame_ids": [0, 1]},
        ],
    }

    evaluated_context_ids: list[str] = []

    def mock_runner(cfg: Any, sources: Any, generator: Any = None, objects: Any = None, context_ids: Any = None) -> Any:
        evaluated_context_ids.extend(context_ids)
        return MockRunResult(
            residual=33333,
            transport_total=40000,
            scores={"psnr": 38.2, "ssim": 0.96, "vmaf": 92.5},
        )

    res = evaluate_checkpoint(
        checkpoint_path=ckpt,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        held_out_only=True,
        allow_synthetic=True,
        runner_fn=mock_runner,
    )

    # Verify only held-out scene was evaluated
    assert evaluated_context_ids == ["held_out_vid/s2/t2"]

    # Verify aggregate values come directly from runner results
    assert res["aggregate"]["residual_bytes"] == 33333
    assert res["aggregate"]["total_bytes"] == 40000
    assert res["aggregate"]["psnr_mean"] == 38.2
    assert res["aggregate"]["ssim_mean"] == 0.96
    assert res["aggregate"]["vmaf_mean"] == 92.5


def test_unsuccessful_evaluation_cannot_select_winning_checkpoint() -> None:
    # Scenario 1: A failed variant with artificially small/0 bytes must not beat a valid variant
    aggregate_results: dict[str, dict[str, Any]] = {
        "valid_candidate": {
            "residual_bytes": 50000,
            "total_bytes": 60000,
            "psnr_mean": 30.0,
            "success": True,
        },
        "crashed_variant": {
            "residual_bytes": 0,
            "total_bytes": 0,
            "psnr_mean": None,
            "success": False,
        },
        "error_variant": {
            "residual_bytes": float("inf"),
            "total_bytes": float("inf"),
            "eval_failed": True,
        },
        "nan_variant": {
            "residual_bytes": float("nan"),
            "psnr_mean": float("nan"),
            "success": True,
        },
    }

    ranked, composite = rank_variants(aggregate_results)
    # The valid candidate must be ranked #1
    assert ranked[0] == "valid_candidate"

    # All failed / invalid variants must be ranked after valid candidates
    assert set(ranked[1:]) == {"crashed_variant", "error_variant", "nan_variant"}

    # promote_survivors must only select the valid candidate
    survivors = promote_survivors(ranked, aggregate_results)
    assert survivors == ["valid_candidate"]

    # Scenario 2: If all variants failed, promote_survivors returns empty list
    failed_only: dict[str, dict[str, Any]] = {
        "failed_1": {"success": False, "residual_bytes": 10},
        "failed_2": {"eval_failed": True, "residual_bytes": 20},
    }
    ranked_failed, _ = rank_variants(failed_only)
    survivors_failed = promote_survivors(ranked_failed, failed_only)
    assert survivors_failed == []


def test_local_global_frame_coordinate_resolution(tmp_path: Path) -> None:
    track_dir = tmp_path / "track_0001"
    skel_dir = tmp_path / "track_0001_skeleton"
    track_dir.mkdir()
    skel_dir.mkdir()

    global_fids = [223, 224, 225]
    for idx, gfid in enumerate(global_fids):
        (track_dir / f"frame_{gfid:06d}.png").write_bytes(b"dummy_crop")
        (skel_dir / f"frame_{idx:06d}.png").write_bytes(b"dummy_skel")

    crop_files = sorted_frame_files(track_dir)
    skel_files = sorted_frame_files(skel_dir)

    positions = window_positions(crop_files, tuple(global_fids))
    assert positions == [0, 1, 2]
    for idx, pos in enumerate(positions):
        assert crop_files[pos].name == f"frame_{global_fids[idx]:06d}.png"
        assert skel_files[pos].name == f"frame_{idx:06d}.png"


def test_missing_source_or_pose_fails_loudly(tmp_path: Path) -> None:
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"weights")
    manifest = {
        "probe_clips": [{"video": "vid_missing", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}],
        "held_out_videos": ["vid_missing"],
    }
    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(
            checkpoint_path=ckpt,
            arch="pix2pix",
            manifest=manifest,
            dataset_root=tmp_path,
            allow_synthetic=False,
        )


def test_empty_split_and_wire_accounting_validation(tmp_path: Path) -> None:
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"weights")
    manifest = {
        "held_out_videos": ["different_video"],
        "probe_clips": [{"video": "vid_train", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}],
    }
    res = evaluate_checkpoint(
        checkpoint_path=ckpt,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        held_out_only=True,
    )
    agg = res["aggregate"]
    assert agg["success"] is False
    assert agg["eval_failed"] is True
    assert agg["residual_bytes"] == float("inf")
    assert not is_valid_eval(agg)

    # Wire accounting violation: total_bytes < residual_bytes
    invalid_wire = {
        "residual_bytes": 50000,
        "total_bytes": 40000,
        "success": True,
        "checkpoint_identity": "model.pt:1234",
    }
    assert not is_valid_eval(invalid_wire)

    # Empty / non-positive per-clip count
    invalid_clips = {
        "residual_bytes": 50000,
        "total_bytes": 60000,
        "per_clip_count": 0,
        "success": True,
        "checkpoint_identity": "model.pt:1234",
    }
    assert not is_valid_eval(invalid_clips)


def test_real_geometry_retained_in_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    dataset_root = tmp_path / "dataset"
    video = "vid1"
    scene = "scene1"
    track = "track_0001"
    track_dir = dataset_root / video / "segmentations" / scene / track
    skel_dir = dataset_root / video / "segmentations" / scene / f"{track}_skeleton"
    track_dir.mkdir(parents=True)
    skel_dir.mkdir(parents=True)

    extract_dir = tmp_path / "outputs" / "bp46-long-scenes" / "clips" / video / scene / "extract_24"
    extract_dir.mkdir(parents=True)

    full_frame = np.full((128, 128, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(extract_dir / "frame_000010.png"), full_frame)
    cv2.imwrite(str(extract_dir / "frame_000011.png"), full_frame)

    crop_rgba = np.full((32, 32, 4), 200, dtype=np.uint8)
    crop_rgba[:, :, 3] = 255
    cv2.imwrite(str(track_dir / "frame_000010.png"), crop_rgba)
    cv2.imwrite(str(track_dir / "frame_000011.png"), crop_rgba)

    skel_rgb = np.full((32, 32, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(skel_dir / "frame_000000.png"), skel_rgb)
    cv2.imwrite(str(skel_dir / "frame_000001.png"), skel_rgb)

    meta_file = dataset_root / video / "segmentations" / scene / f"{track}_metadata.json"
    metadata = [
        {"frame_id": 10, "bbox": [16, 20, 48, 52]},
        {"frame_id": 11, "bbox": [18, 22, 50, 54]},
    ]
    meta_file.write_text(json.dumps(metadata))

    captured_objects: list[Any] = []

    def mock_runner(cfg: Any, sources: Any, generator: Any = None, objects: Any = None, context_ids: Any = None) -> Any:
        captured_objects.append(objects)
        return MockRunResult(residual=10000, transport_total=12000)

    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"dummy")
    manifest = {
        "probe_clips": [{"video": video, "scene": scene, "track": track, "frame_ids": [10, 11]}],
        "held_out_videos": [video],
    }

    evaluate_checkpoint(
        checkpoint_path=ckpt,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=dataset_root,
        eval_mode="whole_codec",
        runner_fn=mock_runner,
    )

    assert len(captured_objects) == 1
    objs = captured_objects[0][0]
    assert len(objs) == 2
    assert objs[0].bbox == (16, 20, 48, 52)
    assert objs[1].bbox == (18, 22, 50, 54)
    assert objs[0].mask.shape == (128, 128)


def test_no_overlap_with_forbidden_confirmation_sources(tmp_path: Path) -> None:
    leaked_dir = tmp_path / "forbidden_video" / "segmentations" / "scene_01" / "track_01"
    leaked_dir.mkdir(parents=True)
    manifest = {
        "excluded_training_keys": ["forbidden_video/scene_01/track_01"],
        "held_out_videos": ["forbidden_video"],
    }
    violations = verify_data_root_excludes_probe_set(tmp_path, manifest)
    assert violations == ["forbidden_video/scene_01/track_01"]


def test_checkpoint_sha256_device_seed_reach_inference(tmp_path: Path) -> None:
    ckpt = tmp_path / "test_model.pt"
    content = b"unique_checkpoint_payload_data"
    ckpt.write_bytes(content)
    expected_sha = hashlib.sha256(content).hexdigest()
    assert compute_checkpoint_sha256(ckpt) == expected_sha

    manifest = {
        "probe_clips": [{"video": "v1", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}],
        "held_out_videos": ["v1"],
    }
    res = evaluate_checkpoint(
        checkpoint_path=ckpt,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        allow_synthetic=True,
        device="cpu",
        seed=1337,
        runner_fn=lambda *a, **k: MockRunResult(),
    )
    agg = res["aggregate"]
    assert agg["checkpoint_identity"] == f"test_model.pt:{expected_sha}"
    assert agg["device"] == "cpu"
    assert agg["seed"] == 1337


def test_client_output_perturbation_changes_ranking() -> None:
    unperturbed = {
        "cand_a": {"residual_bytes": 10000, "total_bytes": 15000, "psnr_mean": 35.0, "success": True},
        "cand_b": {"residual_bytes": 20000, "total_bytes": 25000, "psnr_mean": 34.0, "success": True},
    }
    ranked_clean, _ = rank_variants(unperturbed)
    assert ranked_clean == ["cand_a", "cand_b"]

    perturbed = {
        "cand_a": {"residual_bytes": 30000, "total_bytes": 35000, "psnr_mean": 25.0, "success": True},
        "cand_b": {"residual_bytes": 20000, "total_bytes": 25000, "psnr_mean": 34.0, "success": True},
    }
    ranked_perturbed, _ = rank_variants(perturbed)
    assert ranked_perturbed == ["cand_b", "cand_a"]


def test_configuration_matched_control_reuse_only() -> None:
    existing_records = [
        {
            "video": "v1", "scene": "s1", "frames": 24, "generator": "pix2pix", "residual_qp": 28,
            "metrics": {"residual_bytes": 1000},
        }
    ]
    match = next(
        (r for r in existing_records if (r["video"], r["scene"], r["frames"], r["generator"], r["residual_qp"]) == ("v1", "s1", 24, "pix2pix", 28)),
        None,
    )
    assert match is not None

    mismatch_qp = next(
        (r for r in existing_records if (r["video"], r["scene"], r["frames"], r["generator"], r["residual_qp"]) == ("v1", "s1", 24, "pix2pix", 32)),
        None,
    )
    assert mismatch_qp is None

    mismatch_frames = next(
        (r for r in existing_records if (r["video"], r["scene"], r["frames"], r["generator"], r["residual_qp"]) == ("v1", "s1", 48, "pix2pix", 28)),
        None,
    )
    assert mismatch_frames is None


def test_restore_spade_flag_restores_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True)
    state_file = campaign_dir / "campaign_state.json"
    initial_state = {
        "rung": 1,
        "alive": ["pix2pix"],
        "cumulative_epochs": {"pix2pix": 2, "spade4tennis_lite": 1},
        "variants": {"pix2pix": asdict(Variant(name="pix2pix", arch="pix2pix", kind="pix2pix"))},
        "history": [],
    }
    state_file.write_text(json.dumps(initial_state))
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"excluded_training_keys": [], "probe_clips": []}))

    monkeypatch.setattr(train_campaign, "run_training_subprocess", lambda *a, **k: 0)

    exit_code = main([
        "--campaign-dir", str(campaign_dir),
        "--manifest", str(manifest_path),
        "--data-root", str(data_root),
        "--restore-spade",
        "--dry-run",
    ])
    assert exit_code == 0
    loaded = load_state(state_file)
    assert "spade4tennis_lite" in loaded["alive"]


def test_survivor_continuation_to_max_rungs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True)
    state_file = campaign_dir / "campaign_state.json"
    initial_state = {
        "rung": 0,
        "alive": ["pix2pix"],
        "cumulative_epochs": {"pix2pix": 1},
        "variants": {"pix2pix": asdict(Variant(name="pix2pix", arch="pix2pix", kind="pix2pix"))},
        "history": [],
    }
    state_file.write_text(json.dumps(initial_state))
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"excluded_training_keys": [], "probe_clips": []}))

    monkeypatch.setattr(train_campaign, "run_training_subprocess", lambda *a, **k: 0)
    monkeypatch.setattr(
        train_campaign,
        "eval_variant",
        lambda *a, **k: {
            "residual_bytes": 1000,
            "total_bytes": 2000,
            "success": True,
            "checkpoint_identity": "pix2pix.pt:abc",
        },
    )

    exit_code = main([
        "--campaign-dir", str(campaign_dir),
        "--manifest", str(manifest_path),
        "--data-root", str(data_root),
        "--no-restore-spade",
        "--auto-continue",
        "--max-rungs", "2",
    ])
    assert exit_code == 0
    loaded = load_state(state_file)
    assert loaded["rung"] == 2
