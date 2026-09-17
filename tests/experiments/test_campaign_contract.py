"""Campaign evidence contract: source-count policy, claim eligibility, plot ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tier.campaign_result import (
    CONTRACT_REVISION,
    campaign_record_from_generation_adapter,
    ingest_for_claim,
    load_example_records,
    load_producer_example_records,
    validate_campaign_record,
)
from experiments.tier.operating_points import load_operating_points
from experiments.tier.protocol import capture_current_identity, evaluate_confirmation_protocol
from experiments.tier.source_count_policy import (
    confirmation_label,
    load_required_matches,
    load_source_count_policy,
    policy_identity,
)
from experiments.tier.gate_b_confirmation import confirmation_verdict
from experiments.tier.source_split import training_scene_allowed
from src.runner.generation_adapter import adapt_diagnostic_matrix_result


def test_source_count_policy_is_versioned_and_not_six() -> None:
    policy = load_source_count_policy()
    assert policy["preferred_matches"] == 6
    assert policy["status"] == "accepted_target_pending_acquisition"
    assert policy["confirmation"]["fallback_if_unacquired"]["uses_exposed_sources"] is False
    assert load_required_matches("confirmation") == 3
    assert load_required_matches("development_pilot") == 2
    assert confirmation_label(3) == "small_sample_confirmation"
    assert confirmation_label(2) == "restricted_confirmation"
    assert confirmation_label(1) == "case_study"
    assert confirmation_label(6) == "preferred_confirmation"


def test_source_count_policy_rejects_unaccepted_boolean_and_unknown_stage(
    tmp_path: Path,
) -> None:
    good = load_source_count_policy()
    pending = dict(good)
    pending["status"] = "proposed_pending_coordinator_acceptance"
    pending_path = tmp_path / "pending.json"
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    with pytest.raises(ValueError, match="pending/unaccepted"):
        load_required_matches("confirmation", pending_path)

    boolean_count = dict(good)
    boolean_count["confirmation"] = dict(good["confirmation"])
    boolean_count["confirmation"]["required_matches"] = True
    bool_path = tmp_path / "bool.json"
    bool_path.write_text(json.dumps(boolean_count), encoding="utf-8")
    with pytest.raises(ValueError, match="positive int"):
        load_source_count_policy(bool_path)

    with pytest.raises(ValueError, match="unknown source-count stage"):
        load_required_matches("gate_b")


def test_exposed_source_fallback_is_rejected(tmp_path: Path) -> None:
    good = load_source_count_policy()
    bad = dict(good)
    bad["confirmation"] = dict(good["confirmation"])
    bad["confirmation"]["fallback_if_unacquired"] = {
        "required_matches": 2,
        "uses_exposed_sources": True,
    }
    path = tmp_path / "exposed.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="uses_exposed_sources"):
        load_source_count_policy(path)


def test_confirmation_count_gate_uses_policy_not_literal_six() -> None:
    sources = [
        {
            "source_id": f"src_{i}",
            "match_name": f"match_{i}",
            "comparisons": {
                codec: {"continuous": {"bd_rate_percent": -10.0}}
                for codec in ("av1", "vvc")
            },
        }
        for i in range(3)
    ]
    verdict = confirmation_verdict(sources, [])
    assert verdict["required_matches"] == 3
    assert not any("6 independent" in b for b in verdict["confirmation_blockers"])
    assert verdict["n_unique_matches"] == 3
    assert verdict["gate_b_passed"] is False


def test_two_matches_fail_three_match_confirmation_policy() -> None:
    sources = [
        {
            "source_id": sid,
            "match_name": sid,
            "comparisons": {
                codec: {"continuous": {"bd_rate_percent": -10.0}}
                for codec in ("av1", "vvc")
            },
        }
        for sid in ("a", "b")
    ]
    verdict = evaluate_confirmation_protocol(sources, [], is_pilot=False)
    assert verdict["required_matches"] == 3
    assert any("only 2 unique independent match(es)" in b for b in verdict["confirmation_blockers"])


def test_example_records_keep_rd_without_granting_runtime() -> None:
    records = load_example_records()
    assert all(not validate_campaign_record(item, purpose="structure") for item in records)

    diagnostic = ingest_for_claim(records, "rd", purpose="diagnostic")
    kept_ids = {item["artifact_id"] for item in diagnostic["kept"]}
    excluded = {item["artifact_id"]: item["reason"] for item in diagnostic["excluded"]}
    assert "example_rd_with_timing_but_runtime_excluded" in kept_ids
    assert "example_rd_missing_timing" in kept_ids
    assert "example_invalid_generation_for_trajectory" in excluded
    assert "example_runtime_eligible_stratum" in excluded
    assert "example_validated_rd_without_timing" in kept_ids

    validated = ingest_for_claim(records, "rd", purpose="validated")
    validated_ids = {item["artifact_id"] for item in validated["kept"]}
    assert "example_validated_rd_without_timing" in validated_ids
    assert "example_rd_missing_timing" not in validated_ids
    assert "example_rd_with_timing_but_runtime_excluded" not in validated_ids

    runtime = ingest_for_claim(records, "runtime", purpose="validated")
    runtime_kept = {item["artifact_id"] for item in runtime["kept"]}
    runtime_excluded = {item["artifact_id"]: item["reason"] for item in runtime["excluded"]}
    assert "example_rd_missing_timing" not in runtime_kept
    assert "example_rd_missing_timing" in runtime_excluded
    assert "example_runtime_eligible_stratum" not in runtime_kept
    assert runtime["n_kept"] == 0


def test_cleared_identities_and_null_controls_are_rejected() -> None:
    record = dict(load_example_records()[-1])
    record["artifact_path"] = ""
    record["artifact_sha256"] = ""
    record["code_revision"] = ""
    record["source_ids"] = []
    record["frame_ids"] = {}
    record["controls"] = None
    assert validate_campaign_record(record, purpose="structure") == []
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("artifact_path" in item for item in blockers)
    assert any("artifact_sha256" in item for item in blockers)
    assert any("code_revision" in item for item in blockers)
    assert any("source_ids" in item for item in blockers)
    assert any("frame_ids" in item for item in blockers)
    assert any("controls" in item for item in blockers)
    ingested = ingest_for_claim([record], "rd", purpose="validated")
    assert ingested["n_kept"] == 0


def test_malformed_runtime_does_not_drop_valid_rd() -> None:
    rd_row = dict(load_example_records()[-1])
    runtime_row = dict(load_example_records()[3])
    ingested_rd = ingest_for_claim([rd_row, runtime_row], "rd", purpose="validated")
    assert ingested_rd["n_kept"] == 1
    assert ingested_rd["kept"][0]["artifact_id"] == "example_validated_rd_without_timing"
    ingested_rt = ingest_for_claim([rd_row, runtime_row], "runtime", purpose="validated")
    assert ingested_rt["n_kept"] == 0


def test_plot_ingest_emits_reason_when_eligibility_omits_exclusion() -> None:
    broken = load_example_records()[0]
    broken["claim_eligibility"]["runtime"] = False
    broken["claim_eligibility"]["exclusions"] = [
        item for item in broken["claim_eligibility"]["exclusions"] if item["claim"] != "runtime"
    ]
    blockers = validate_campaign_record(broken, purpose="structure")
    assert any("runtime" in item for item in blockers)
    ingested = ingest_for_claim([broken], "rd", purpose="validated")
    assert ingested["n_kept"] == 0
    assert ingested["excluded"][0]["reason"]


def test_operating_points_separate_crop_from_display() -> None:
    contract = load_operating_points()
    assert contract["learning_crop"]["width"] == 256
    assert contract["display_endpoints"]["low"]["short_edge_px"] == 360
    assert contract["latency_policy"]["application_ceiling_ms_after_startup"] == 250
    assert contract["sustained_live_test"]["min_duration_seconds"] == 30.0
    assert contract["sustained_live_test"]["smoke_check_seconds"] == 8.0
    assert contract["quality_policy"]["practical_quality_floor"]["role"] == "diagnostic_exclusion"
    assert contract["quality_policy"]["practical_quality_floor"]["vmaf"] == 20.0
    assert "interpretable_quality_criterion" in contract["quality_policy"]
    assert contract["common_timebase_scoring"]["never_restore_from_untransmitted_source"] is True


def test_training_selector_excludes_validation_and_confirmation() -> None:
    ok, reason = training_scene_allowed("federer_djokovic", "scene_007")
    assert ok
    blocked, why = training_scene_allowed("djokovic_federer", "scene_009")
    assert not blocked
    assert why == "validation_block"
    reserved, reserved_why = training_scene_allowed(
        "other",
        "scene_000",
        match_id="conf_cand_01_sinner_medvedev_ao2024",
    )
    assert not reserved
    assert reserved_why == "confirmation_or_exposed_holdout"


def test_e02_adapter_output_requires_campaign_mapping(tmp_path: Path) -> None:
    matrix_output = {
        "normal": {
            "elapsed_seconds": 1.5,
            "psnr_mean": 32.5,
            "ssim_mean": 0.91,
            "vmaf_mean": 82.0,
            "residual_bytes": 15000,
            "total_bytes": 25000,
            "frame_hashes": ["hash1", "hash2", "hash3", "hash4"],
            "per_frame_psnr": [32.0, 33.0, 32.5, 32.5],
        },
        "shuffled": {"frame_hashes": ["shuff1", "shuff2", "shuff3", "shuff4"]},
        "seed_repeat": {"frame_hashes": ["hash1", "hash2", "hash3", "hash4"]},
    }
    artifact = tmp_path / "test_run_01.json"
    artifact.write_text("{}", encoding="utf-8")
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="test_run_01",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        artifact_path=artifact,
    )
    adapted["source_ids"] = ["federer_djokovic_scene_007"]
    adapted["frame_ids"] = {"start": 0, "count": 4, "fps": 12.0}
    adapted["code_revision"] = "2f63ae1"
    assert validate_campaign_record(adapted, purpose="structure")
    mapped = campaign_record_from_generation_adapter(adapted)
    assert mapped["contract_revision"] == CONTRACT_REVISION
    assert mapped["claim_eligibility"]["generalization"] is False
    assert mapped["claim_eligibility"]["trajectory"] is False
    assert mapped["evidence"]["metrics"]["psnr_y"] == 32.5
    blockers = validate_campaign_record(mapped, purpose="structure")
    assert blockers == []
    validated = ingest_for_claim([mapped], "rd", purpose="validated")
    assert validated["n_kept"] == 0
    diagnostic = ingest_for_claim([mapped], "rd", purpose="diagnostic")
    assert diagnostic["n_kept"] == 0
    assert mapped["claim_eligibility"]["rd"] is False


def test_producer_examples_and_identity_pin_policy() -> None:
    records = load_producer_example_records()
    assert records
    assert all(not validate_campaign_record(item, purpose="structure") for item in records)
    identity = capture_current_identity(Path("manifests/evaluation_20260914_source_count_policy.json"))
    pinned = policy_identity()
    assert identity.source_count_policy["sha256"] == pinned["sha256"]
    assert identity.source_count_policy["confirmation_required_matches"] == 3


def test_e03_e04_and_split_manifests_load() -> None:
    repo = Path(__file__).resolve().parents[2]
    for name in (
        "evaluation_20260914_source_split.json",
        "evaluation_20260914_reuse_map.json",
        "evaluation_20260914_e03_e04_inputs.json",
        "evaluation_20260914_source_count_policy.json",
        "evaluation_20260914_operating_points.json",
        "evaluation_20260914_result_contract.example.json",
        "evaluation_20260914_e02_producer.example.json",
        "evaluation_20260914_e01r_acquisition.json",
        "evaluation_20260915_e03a_contract_pin.json",
        "evaluation_20260915_e03a_anchor_card.json",
        "evaluation_20260915_e03a_confirmation_timestamps.json",
        "evaluation_20260915_e03a_derived_diagnostic.json",
        "evaluation_20260916_e03b_source_recipe.json",
        "evaluation_20260916_e03b_confirmation_eligibility.json",
    ):
        payload = json.loads((repo / "manifests" / name).read_text(encoding="utf-8"))
        assert "schema" in payload


def test_finite_number_rejects_infinity() -> None:
    from experiments.tier.campaign_result import _finite_number

    assert _finite_number(1.0) is True
    assert _finite_number(float("inf")) is False
    assert _finite_number(float("nan")) is False


def test_validated_claim_rejects_failed_controls_and_invalid_values() -> None:
    record = dict(load_example_records()[-1])
    record["evidence"] = {
        "metrics": {"psnr_y": float("inf"), "ssim": 0.91, "vmaf": None},
        "bytes": {"total": -1},
    }
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"]["rd_arms"] = {
        "psnr_y": True,
        "ssim": True,
        "vmaf": False,
        "bytes": True,
    }
    record["controls"] = {
        "standalone_decode": {"status": "failed"},
        "metric_calibration": {"status": "failed"},
        "wire_ledger": {"status": "failed"},
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("byte" in item for item in blockers)
    assert any("quality" in item or "psnr_y" in item for item in blockers)
    assert any("ledger" in item for item in blockers)
    assert any("calibration" in item for item in blockers)
    assert any("decode" in item for item in blockers)
    ingested = ingest_for_claim([record], "rd", purpose="validated")
    assert ingested["n_kept"] == 0


def test_partial_and_not_applicable_controls_are_not_verified() -> None:
    record = dict(load_example_records()[-1])
    record["controls"] = {
        "standalone_decode": "partial",
        "metric_calibration": "not_applicable",
        "wire_ledger": "present",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("decode" in item for item in blockers)
    assert any("calibration" in item for item in blockers)
    assert any("ledger" in item for item in blockers)


def test_runtime_does_not_require_quality_controls() -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"]["rd"] = False
    record["claim_eligibility"]["runtime"] = True
    record["claim_eligibility"]["standalone_transport"] = False
    record["claim_eligibility"]["exclusions"] = [
        {"claim": "rd", "reason": "timing-only stratum"},
        {"claim": "standalone_transport", "reason": "not a transport audit"},
        {"claim": "trajectory", "reason": "not a generation trajectory"},
        {"claim": "generalization", "reason": "single development source"},
    ]
    record["timing_evidence"] = {
        "timing_evidence_id": "timing.gpu6.display_low.paste.n3",
        "host": "gpu6",
        "n_repeats": 3,
        "repeat_seconds": [1.20, 1.25, 1.30],
        "measured_client_seconds": 1.25,
        "stages": {"deserialize_s": {"n": 3, "mean": 0.04}},
    }
    record["controls"] = {
        "standalone_decode": "not_this_row",
        "metric_calibration": "not_this_row",
        "wire_ledger": "not_this_row",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert blockers == []
    runtime = ingest_for_claim([record], "runtime", purpose="validated")
    assert runtime["n_kept"] == 1
    rd = ingest_for_claim([record], "rd", purpose="validated")
    assert rd["n_kept"] == 0


@pytest.mark.parametrize("measured", [None, float("inf"), float("nan"), 0.0, -1.0])
def test_runtime_rejects_missing_nonfinite_or_nonpositive_measurement(measured: object) -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"].update(
        {"rd": False, "runtime": True, "standalone_transport": False}
    )
    record["claim_eligibility"]["exclusions"] = [
        {"claim": "rd", "reason": "timing-only stratum"},
        {"claim": "standalone_transport", "reason": "not a transport audit"},
        {"claim": "trajectory", "reason": "not a generation trajectory"},
        {"claim": "generalization", "reason": "single development source"},
    ]
    record["timing_evidence"] = {
        "timing_evidence_id": "timing.gpu5.display_low.n3",
        "host": "gpu5",
        "n_repeats": 3,
        "repeat_seconds": [1.2, 1.3, 1.4],
        "measured_client_seconds": measured,
        "stages": {"deserialize_s": {"n": 3, "mean": 0.04}},
    }
    record["controls"] = {
        "standalone_decode": "not_this_row",
        "metric_calibration": "not_this_row",
        "wire_ledger": "not_this_row",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("positive finite measured_client_seconds" in item for item in blockers)
    assert ingest_for_claim([record], "runtime", purpose="validated")["n_kept"] == 0


def test_runtime_rejects_declared_single_repeat() -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"].update(
        {"rd": False, "runtime": True, "standalone_transport": False}
    )
    record["claim_eligibility"]["exclusions"] = [
        {"claim": "rd", "reason": "timing-only stratum"},
        {"claim": "standalone_transport", "reason": "not a transport audit"},
        {"claim": "trajectory", "reason": "not a generation trajectory"},
        {"claim": "generalization", "reason": "single development source"},
    ]
    record["timing_evidence"] = {
        "timing_evidence_id": "timing.gpu5.display_low.n1",
        "host": "gpu5",
        "n_repeats": 1,
        "repeat_seconds": [1.25],
        "measured_client_seconds": 1.25,
        "stages": {"deserialize_s": {"n": 1, "mean": 0.04}},
    }
    record["controls"] = {
        "standalone_decode": "not_this_row",
        "metric_calibration": "not_this_row",
        "wire_ledger": "not_this_row",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("at least two timed" in item for item in blockers)
    assert ingest_for_claim([record], "runtime", purpose="validated")["n_kept"] == 0


@pytest.mark.parametrize("case", ["missing", "unknown", "inconsistent_bytes"])
def test_validated_rd_requires_explicit_consistent_known_arms(case: str) -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"]["rd_arms"] = dict(
        record["claim_eligibility"]["rd_arms"]
    )
    if case == "missing":
        record["claim_eligibility"].pop("rd_arms")
    elif case == "unknown":
        record["evidence"] = {"metrics": {"latency": 1.0}, "bytes": {"total": 100}}
        record["claim_eligibility"]["rd_arms"] = {"latency": True, "bytes": True}
    else:
        record["claim_eligibility"]["rd_arms"]["bytes"] = False
    blockers = validate_campaign_record(record, purpose="validated")
    assert blockers
    assert ingest_for_claim([record], "rd", purpose="validated")["n_kept"] == 0


def test_validated_rd_may_exclude_a_finite_uncalibrated_quality_arm() -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"]["rd_arms"] = {
        "psnr_y": True,
        "ssim": False,
        "vmaf": False,
        "bytes": True,
    }
    record["evidence"] = {
        "metrics": {"psnr_y": 32.1, "ssim": 0.91, "vmaf": 80.0},
        "bytes": {"total": 125000},
    }
    assert validate_campaign_record(record, purpose="validated") == []
    assert ingest_for_claim([record], "rd", purpose="validated")["n_kept"] == 1


def test_standalone_transport_requires_positive_bytes_and_reconciled_ledger() -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"].update({"rd": False, "standalone_transport": True})
    record["claim_eligibility"]["exclusions"] = [
        {"claim": "rd", "reason": "standalone-only evidence"},
        {"claim": "runtime", "reason": "no timing stratum"},
        {"claim": "trajectory", "reason": "not a generation trajectory"},
        {"claim": "generalization", "reason": "single development source"},
    ]
    record["evidence"] = {"metrics": {}, "bytes": {"total": -1}}
    record["controls"] = {
        "standalone_decode": "verified",
        "metric_calibration": "not_this_row",
        "wire_ledger": "failed",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("ledger" in item for item in blockers)
    assert any("positive finite transport bytes" in item for item in blockers)
    result = ingest_for_claim([record], "standalone_transport", purpose="validated")
    assert result["n_kept"] == 0


def test_trajectory_requires_multiframe_conditioning_control() -> None:
    record = dict(load_example_records()[-1])
    record["claim_eligibility"] = dict(record["claim_eligibility"])
    record["claim_eligibility"].update(
        {"rd": False, "standalone_transport": False, "trajectory": True}
    )
    record["claim_eligibility"]["exclusions"] = [
        {"claim": "rd", "reason": "trajectory-only evidence"},
        {"claim": "runtime", "reason": "no timing stratum"},
        {"claim": "standalone_transport", "reason": "not a transport audit"},
        {"claim": "generalization", "reason": "single development source"},
    ]
    record["trajectory_coverage"] = "full_visible_track"
    record["frame_ids"] = {"start": 0, "count": 1}
    record["controls"] = {
        "standalone_decode": "not_this_row",
        "metric_calibration": "not_this_row",
        "wire_ledger": "not_this_row",
        "conditioned_vs_shuffled": "failed",
    }
    blockers = validate_campaign_record(record, purpose="validated")
    assert any("at least two" in item for item in blockers)
    assert any("conditioned-vs-shuffled" in item for item in blockers)
    assert ingest_for_claim([record], "trajectory", purpose="validated")["n_kept"] == 0


def test_producer_scores_timing_parts_map_into_campaign_record(tmp_path: Path) -> None:
    payload = {
        "run_id": "producer_nested_01",
        "backend_name": "pix2pix",
        "artifact_path": str(tmp_path / "diag.json"),
        "code_revision": "deadbeef",
        "source_ids": ["alcaraz_highlights_scene_028"],
        "frame_ids": {"start": 0, "count": 16},
        "scores": {"psnr_y": 28.3, "ssim": 0.97, "vmaf": 88.0},
        "timing": {"client_seconds": 24.5, "encoder_seconds": 85.7},
        "parts": {"residual": 110913, "transport_total": 7023087},
        "coded_bytes": 7023087,
        "delivered_shape": [16, 2160, 3840, 3],
        "wire_reconciliation": {
            "matched": True,
            "verdict": "matched",
            "wire_bytes": 7023087,
            "transport_total": 7023087,
        },
        "checkpoint_identity": {
            "checkpoint_id": "pix2pix:aa",
            "checkpoint_sha256": "a" * 64,
            "config_identity": "b" * 64,
        },
        "claim_eligibility": {"rd_claim": True, "speed_claim": False, "standalone_decode": False},
        "controls": {"conditioned_vs_shuffled_tested": True},
    }
    (tmp_path / "diag.json").write_text("{}", encoding="utf-8")
    mapped = campaign_record_from_generation_adapter(payload)
    assert mapped["contract_revision"] == CONTRACT_REVISION
    assert mapped["evidence"]["metrics"]["psnr_y"] == 28.3
    assert mapped["evidence"]["bytes"]["total"] == 7023087
    assert mapped["delivered_shape"] == [16, 2160, 3840, 3]
    assert mapped["controls"]["wire_ledger"] == "reconciled"
    assert mapped["controls"]["metric_calibration"] == "unverified"
    assert mapped["controls"]["standalone_decode"] == "unverified"
    assert mapped["record_class"] == "recoverable_evidence"
    validated = ingest_for_claim([mapped], "rd", purpose="validated")
    assert validated["n_kept"] == 0
    diagnostic = ingest_for_claim([mapped], "rd", purpose="diagnostic")
    assert diagnostic["n_kept"] == 0
    assert mapped["claim_eligibility"]["rd"] is False


def test_artifact_sha256_uses_file_bytes_not_canonical_json(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"hello":"world"}\n', encoding="utf-8")
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    mapped = campaign_record_from_generation_adapter(
        {
            "run_id": "hash_probe",
            "artifact_path": str(path),
            "artifact_sha256": "c" * 64,
            "code_revision": "abc",
            "source_ids": ["src"],
            "scores": {"psnr_y": 30.0},
            "parts": {"transport_total": 100},
            "checkpoint_identity": {"checkpoint_sha256": "d" * 64},
        }
    )
    assert mapped["artifact_sha256"] == expected
    assert mapped["artifact_sha256"] != "c" * 64
    assert mapped["checkpoint_identity"]["checkpoint_sha256"] == "d" * 64
