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


def test_e02_adapter_output_requires_campaign_mapping() -> None:
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
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="test_run_01",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    )
    adapted["source_ids"] = ["federer_djokovic_scene_007"]
    adapted["frame_ids"] = {"start": 0, "count": 4, "fps": 12.0}
    adapted["artifact_path"] = "outputs/evaluation-20260914/e02/test_run_01.json"
    adapted["code_revision"] = "2f63ae1"
    assert validate_campaign_record(adapted, purpose="structure")
    mapped = campaign_record_from_generation_adapter(adapted)
    assert mapped["contract_revision"] == CONTRACT_REVISION
    assert mapped["claim_eligibility"]["generalization"] is False
    assert mapped["claim_eligibility"]["trajectory"] is False
    blockers = validate_campaign_record(mapped, purpose="structure")
    assert blockers == []
    validated = ingest_for_claim([mapped], "rd", purpose="validated")
    assert validated["n_kept"] == 0
    diagnostic = ingest_for_claim([mapped], "rd", purpose="diagnostic")
    assert diagnostic["n_kept"] == 1


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
    ):
        payload = json.loads((repo / "manifests" / name).read_text(encoding="utf-8"))
        assert "schema" in payload
