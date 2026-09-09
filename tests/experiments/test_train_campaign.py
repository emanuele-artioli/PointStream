"""Fast unit tests for scripts/train_campaign.py's pure logic.

No subprocess/training/GPU here: command building, ranking/promotion math,
the non-overlap safety check, and state persistence. The real rung execution
is covered by the end-to-end smoke run (report 10's Phase 5.4 findings
entry), not by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sqlite3
import numpy as np

import scripts.train_campaign as train_campaign
from scripts.train_campaign import (
    Variant,
    build_eval_generator_ref,
    build_train_command,
    checkpoint_path_for_eval,
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
    ckpt_file.write_bytes(b"dummy_weights_data")
    stat = ckpt_file.stat()
    expected_id = f"model.pt:{stat.st_size}:{int(stat.st_mtime)}"

    manifest = {
        "probe_clips": [{"video": "v1", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}],
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
    from src.components.generation.base import as_runner_ref
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

    manifest = {"probe_clips": [{"video": "v1", "scene": "s1", "track": "t1", "frame_ids": [0, 1]}]}

    evaluate_checkpoint(
        checkpoint_path=ckpt1,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        runner_fn=mock_runner,
    )
    evaluate_checkpoint(
        checkpoint_path=ckpt2,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
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
    aggregate_results = {
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
    failed_only = {
        "failed_1": {"success": False, "residual_bytes": 10},
        "failed_2": {"eval_failed": True, "residual_bytes": 20},
    }
    ranked_failed, _ = rank_variants(failed_only)
    survivors_failed = promote_survivors(ranked_failed, failed_only)
    assert survivors_failed == []
