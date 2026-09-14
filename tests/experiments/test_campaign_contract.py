"""Campaign evidence contract: source-count policy, claim eligibility, plot ingest."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.tier.campaign_result import (
    ingest_for_claim,
    load_example_records,
    validate_campaign_record,
)
from experiments.tier.operating_points import load_operating_points
from experiments.tier.protocol import evaluate_confirmation_protocol
from experiments.tier.source_count_policy import (
    confirmation_label,
    load_required_matches,
    load_source_count_policy,
)
from experiments.tier.gate_b_confirmation import confirmation_verdict


def test_source_count_policy_is_versioned_and_not_six() -> None:
    policy = load_source_count_policy()
    assert policy["preferred_matches"] == 6
    assert load_required_matches("confirmation") == 3
    assert load_required_matches("development_pilot") == 2
    assert confirmation_label(3) == "small_sample_confirmation"
    assert confirmation_label(2) == "restricted_confirmation"
    assert confirmation_label(1) == "case_study"
    assert confirmation_label(6) == "preferred_confirmation"


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
    # Still fail closed: evidence missing.
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
    assert all(not validate_campaign_record(item) for item in records)

    rd = ingest_for_claim(records, "rd")
    kept_ids = {item["artifact_id"] for item in rd["kept"]}
    excluded = {item["artifact_id"]: item["reason"] for item in rd["excluded"]}
    assert "example_rd_with_timing_but_runtime_excluded" in kept_ids
    assert "example_rd_missing_timing" in kept_ids
    assert "example_invalid_generation_for_trajectory" in excluded
    assert "example_runtime_eligible_stratum" in excluded

    runtime = ingest_for_claim(records, "runtime")
    runtime_kept = {item["artifact_id"] for item in runtime["kept"]}
    runtime_excluded = {item["artifact_id"]: item["reason"] for item in runtime["excluded"]}
    assert "example_rd_missing_timing" not in runtime_kept
    assert "example_rd_missing_timing" in runtime_excluded
    assert "example_runtime_eligible_stratum" in runtime_kept
    assert runtime["n_excluded"] >= 1


def test_plot_ingest_emits_reason_when_eligibility_omits_exclusion() -> None:
    broken = load_example_records()[0]
    broken["claim_eligibility"]["runtime"] = False
    broken["claim_eligibility"]["exclusions"] = [
        item for item in broken["claim_eligibility"]["exclusions"] if item["claim"] != "runtime"
    ]
    blockers = validate_campaign_record(broken)
    assert any("runtime" in item for item in blockers)
    ingested = ingest_for_claim([broken], "rd")
    assert ingested["n_kept"] == 0
    assert ingested["excluded"][0]["reason"]


def test_operating_points_separate_crop_from_display() -> None:
    contract = load_operating_points()
    assert contract["learning_crop"]["width"] == 256
    assert contract["display_endpoints"]["low"]["short_edge_px"] == 360
    assert contract["latency_policy"]["application_ceiling_ms_after_startup"] == 250
    assert contract["sustained_live_test"]["min_duration_seconds"] == 8.0
    assert contract["quality_policy"]["practical_quality_floor"]["vmaf"] == 20.0


def test_e03_e04_and_split_manifests_load() -> None:
    repo = Path(__file__).resolve().parents[2]
    for name in (
        "evaluation_20260914_source_split.json",
        "evaluation_20260914_reuse_map.json",
        "evaluation_20260914_e03_e04_inputs.json",
        "evaluation_20260914_source_count_policy.json",
        "evaluation_20260914_operating_points.json",
        "evaluation_20260914_result_contract.example.json",
    ):
        payload = json.loads((repo / "manifests" / name).read_text(encoding="utf-8"))
        assert "schema" in payload
