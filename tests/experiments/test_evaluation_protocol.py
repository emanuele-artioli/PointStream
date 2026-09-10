"""Authorized tests for evaluation drivers, protocol enforcement, and resolution-adaptive anchors.

Tests:
1. missing protocol evidence fails closed
2. incomplete/duplicate sources cannot satisfy independent-source count
3. mismatched configuration/model/anchor identity fails
4. no-overlap curves are unscorable (prohibits extrapolation)
5. known synthetic curves give expected comparison sign
6. rescaled outputs have original dimensions and declared timing/byte treatment
7. pilot completion cannot imply confirmation
8. metric calibration ordering and null controls
9. development recovery manifest holdout protection
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from typing import Any
from unittest.mock import patch

import numpy as np

from experiments.tier.calibrate import (
    PROPOSED_SSIM_CALIBRATION_POLICY,
    SSIM_UNRELATED_LEGACY_CEILING,
    run_full_metric_calibration,
    spatial_null_frames,
    synthetic_court_like_clip,
    synthetic_foreign_court_clip,
    synthetic_unrelated_clip,
    temporal_null_frames,
)
from experiments.tier.development_eval import (
    DEFAULT_DEV_MANIFEST,
    load_development_manifest,
)
from experiments.tier.protocol import (
    ExperimentIdentity,
    ProtocolEvidence,
    calculate_source_uncertainty,
    evaluate_confirmation_protocol,
)
from experiments.tier.resolution_adaptive import (
    compare_curves_no_extrapolation,
    encode_resolution_arm,
    recompute_stored_bd_rate_arithmetic_check,
    rescale_frames,
    restore_to_display_grid,
)
from src.components.background.scale import HEADER_BYTES


def _make_pilot_source(source_id: str, match_name: str, delta: float | None = -15.0) -> dict:
    return {
        "source_id": source_id,
        "match_name": match_name,
        "comparisons": {
            codec: {
                "continuous": {
                    "bd_rate_percent": delta,
                    "reason": None if delta is not None else "insufficient overlap",
                }
            }
            for codec in ("av1", "vvc")
        },
    }


def _make_valid_evidence() -> ProtocolEvidence:
    return ProtocolEvidence(
        client_output_scored=True,
        client_scoring_details={"client_sha256": "abcdef123456"},
        measured_wire_cost=True,
        wire_cost_reconciled=True,
        wire_cost_details={"wire_bytes": 10000, "ledger_bytes": 10000},
        calibrated_metrics=True,
        metric_calibration_details={"valid": True, "ordering_held": True},
        calibrated_nulls=True,
        null_control_details={"temporal_null_valid": True},
        source_eligibility=True,
        source_eligibility_details={"held_out_verified": True},
        source_uncertainty={"evaluated": True, "sem": 1.2},
    )


# ---------------------------------------------------------------------------
# 1. Missing protocol evidence fails closed
# ---------------------------------------------------------------------------


def test_missing_protocol_evidence_fails_closed() -> None:
    sources = [_make_pilot_source(f"src_{i}", f"match_{i}", -12.0) for i in range(6)]

    # Case A: No evidence provided at all
    verdict_none = evaluate_confirmation_protocol(sources, [], is_pilot=False)
    assert verdict_none["gate_b_passed"] is False
    assert verdict_none["confirmation_status"] == "incomplete_protocol"
    assert any("protocol validation is not implemented" in b for b in verdict_none["confirmation_blockers"])

    # Case B: Evidence missing client-output scoring
    ev_no_client = _make_valid_evidence()
    ev_no_client.client_output_scored = False
    verdict_no_client = evaluate_confirmation_protocol(
        sources, [], evidence=ev_no_client, is_pilot=False
    )
    assert verdict_no_client["gate_b_passed"] is False
    assert any("client-output scoring" in b for b in verdict_no_client["confirmation_blockers"])

    # Case C: Evidence missing wire cost reconciliation
    ev_no_wire = _make_valid_evidence()
    ev_no_wire.wire_cost_reconciled = False
    verdict_no_wire = evaluate_confirmation_protocol(
        sources, [], evidence=ev_no_wire, is_pilot=False
    )
    assert verdict_no_wire["gate_b_passed"] is False
    assert any("wire cost not reconciled" in b for b in verdict_no_wire["confirmation_blockers"])

    # Case D: Evidence missing metric calibration
    ev_no_calib = _make_valid_evidence()
    ev_no_calib.calibrated_metrics = False
    verdict_no_calib = evaluate_confirmation_protocol(
        sources, [], evidence=ev_no_calib, is_pilot=False
    )
    assert verdict_no_calib["gate_b_passed"] is False
    assert any("metric calibration" in b for b in verdict_no_calib["confirmation_blockers"])

    # Case E: Evidence missing null controls
    ev_no_null = _make_valid_evidence()
    ev_no_null.calibrated_nulls = False
    verdict_no_null = evaluate_confirmation_protocol(
        sources, [], evidence=ev_no_null, is_pilot=False
    )
    assert verdict_no_null["gate_b_passed"] is False
    assert any("null controls" in b for b in verdict_no_null["confirmation_blockers"])

    # Case F: Evidence missing uncertainty quantification
    ev_no_unc = _make_valid_evidence()
    ev_no_unc.source_uncertainty = {}
    verdict_no_unc = evaluate_confirmation_protocol(
        sources, [], evidence=ev_no_unc, is_pilot=False
    )
    assert verdict_no_unc["gate_b_passed"] is False
    assert any("uncertainty" in b for b in verdict_no_unc["confirmation_blockers"])


# ---------------------------------------------------------------------------
# 2. Incomplete or duplicate sources cannot satisfy independent-source count
# ---------------------------------------------------------------------------


def test_duplicate_sources_cannot_satisfy_independent_count() -> None:
    # 6 sources with duplicate match_name
    duplicate_sources = [_make_pilot_source(f"src_{i}", "same_match_repeated", -15.0) for i in range(6)]
    evidence = _make_valid_evidence()

    verdict = evaluate_confirmation_protocol(
        duplicate_sources, [], evidence=evidence, is_pilot=False, required_matches=6
    )
    assert verdict["gate_b_passed"] is False
    assert verdict["n_unique_matches"] == 1
    assert any("only 1 unique independent match(es)" in b for b in verdict["confirmation_blockers"])


def test_scenes_from_same_match_count_once() -> None:
    # 6 scenes originating from only 2 distinct matches (3 scenes per match)
    sources = [
        {"source_id": "alcaraz_scene_000", "match_name": "Alcaraz vs Sinner", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
        {"source_id": "alcaraz_scene_001", "match_name": "Alcaraz vs Sinner", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
        {"source_id": "alcaraz_scene_002", "match_name": "Alcaraz vs Sinner", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
        {"source_id": "djokovic_scene_000", "match_name": "Djokovic vs Federer", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
        {"source_id": "djokovic_scene_001", "match_name": "Djokovic vs Federer", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
        {"source_id": "djokovic_scene_002", "match_name": "Djokovic vs Federer", "comparisons": _make_pilot_source("s", "m")["comparisons"]},
    ]
    evidence = _make_valid_evidence()
    verdict = evaluate_confirmation_protocol(
        sources, [], evidence=evidence, is_pilot=False, required_matches=6
    )
    assert verdict["gate_b_passed"] is False
    assert verdict["n_unique_matches"] == 2
    assert any("only 2 unique independent match(es)" in b for b in verdict["confirmation_blockers"])

    # Raw scene IDs without match context are rejected and cannot masquerade as independent matches
    bare_scene_sources = [
        {"source_id": f"scene_{i:03d}", "comparisons": _make_pilot_source("s", "m")["comparisons"]}
        for i in range(6)
    ]
    verdict_bare = evaluate_confirmation_protocol(
        bare_scene_sources, [], evidence=evidence, is_pilot=False, required_matches=6
    )
    assert verdict_bare["gate_b_passed"] is False
    assert any("scene IDs do not identify independent matches" in b for b in verdict_bare["confirmation_blockers"])


def test_incomplete_sources_fail_closed() -> None:
    # Missing comparisons dictionary in source
    broken_source = {"source_id": "broken_source", "match_name": "match_broken"}
    evidence = _make_valid_evidence()

    verdict = evaluate_confirmation_protocol(
        [broken_source], [], evidence=evidence, is_pilot=False
    )
    assert verdict["gate_b_passed"] is False
    assert any("missing anchor comparison" in b for b in verdict["confirmation_blockers"])


# ---------------------------------------------------------------------------
# 3. Mismatched configuration/model/anchor identity fails
# ---------------------------------------------------------------------------


def test_mismatched_identity_fails() -> None:
    expected_id = ExperimentIdentity(
        config_fingerprint="fp_expected_123",
        manifest_sha256="sha_manifest_abc",
        codec_tools={
            "av1": {"binary": "/opt/local/bin/SvtAv1EncApp", "version": "SVT-AV1 v1.8.0 (release)", "preset": "8"},
            "vvc": {"binary": "/opt/local/bin/vvencapp", "version": "vvencapp version 1.11.0", "preset": "medium"},
        },
        model_hashes={"yolo": "yolo_hash_111"},
    )

    sources = [_make_pilot_source(f"src_{i}", f"match_{i}", -12.0) for i in range(6)]
    evidence = _make_valid_evidence()

    # Mismatch in AV1 version
    wrong_version_id = ExperimentIdentity(
        config_fingerprint="fp_expected_123",
        manifest_sha256="sha_manifest_abc",
        codec_tools={
            "av1": {"binary": "/opt/local/bin/SvtAv1EncApp", "version": "SVT-AV1 v1.7.0 (legacy)", "preset": "8"},
            "vvc": {"binary": "/opt/local/bin/vvencapp", "version": "vvencapp version 1.11.0", "preset": "medium"},
        },
        model_hashes={"yolo": "yolo_hash_111"},
    )

    verdict_bad_codec = evaluate_confirmation_protocol(
        sources,
        [],
        identity=wrong_version_id,
        expected_identity=expected_id,
        evidence=evidence,
        is_pilot=False,
    )
    assert verdict_bad_codec["gate_b_passed"] is False
    assert verdict_bad_codec["identity_verified"] is False
    assert any("av1 tool version mismatch" in b for b in verdict_bad_codec["confirmation_blockers"])

    # Mismatch in model weights
    wrong_model_id = ExperimentIdentity(
        config_fingerprint="fp_expected_123",
        manifest_sha256="sha_manifest_abc",
        codec_tools=expected_id.codec_tools,
        model_hashes={"yolo": "corrupted_or_different_weights"},
    )
    verdict_bad_model = evaluate_confirmation_protocol(
        sources,
        [],
        identity=wrong_model_id,
        expected_identity=expected_id,
        evidence=evidence,
        is_pilot=False,
    )
    assert verdict_bad_model["gate_b_passed"] is False
    assert any("model yolo hash mismatch" in b for b in verdict_bad_model["confirmation_blockers"])


def test_required_identity_absent_fails() -> None:
    sources = [_make_pilot_source(f"src_{i}", f"match_{i}", -12.0) for i in range(6)]
    evidence = _make_valid_evidence()

    # Case A: Neither identity nor expected_identity provided
    verdict_no_id = evaluate_confirmation_protocol(
        sources, [], identity=None, expected_identity=None, evidence=evidence, is_pilot=False
    )
    assert verdict_no_id["gate_b_passed"] is False
    assert any("missing required experiment identity" in b for b in verdict_no_id["confirmation_blockers"])

    # Case B: expected_identity provided, but actual identity is missing
    expected_id = ExperimentIdentity(
        config_fingerprint="fp123",
        manifest_sha256="sha123",
    )
    verdict_missing_actual = evaluate_confirmation_protocol(
        sources, [], identity=None, expected_identity=expected_id, evidence=evidence, is_pilot=False
    )
    assert verdict_missing_actual["gate_b_passed"] is False
    assert any("missing required experiment identity" in b or "actual identity is missing" in b for b in verdict_missing_actual["confirmation_blockers"])


# ---------------------------------------------------------------------------
# 4. No-overlap curves are unscorable (extrapolation prohibited)
# ---------------------------------------------------------------------------


def test_no_overlap_curves_are_unscorable() -> None:
    # Candidate VMAF in [20.0, 35.0]
    candidate_rows = [
        {"bytes": 10000, "scores": {"vmaf": 20.0}, "usable": True},
        {"bytes": 20000, "scores": {"vmaf": 28.0}, "usable": True},
        {"bytes": 30000, "scores": {"vmaf": 35.0}, "usable": True},
    ]
    # Anchor VMAF in [50.0, 75.0] - completely disjoint
    anchor_rows = [
        {"bytes": 50000, "scores": {"vmaf": 50.0}, "usable": True},
        {"bytes": 80000, "scores": {"vmaf": 65.0}, "usable": True},
        {"bytes": 120000, "scores": {"vmaf": 75.0}, "usable": True},
    ]

    comp = compare_curves_no_extrapolation(candidate_rows, anchor_rows, metric_name="vmaf")
    assert comp["is_scorable"] is False
    assert comp["bd_rate_percent"] is None
    assert comp["extrapolation_prohibited"] is True
    assert "no common quality support" in comp["reason"]

    # Narrow overlap test: candidate in [30.0, 45.0], anchor in [43.0, 80.0]
    # Overlap is [43.0, 45.0] (width 2.0), candidate span is 15.0 -> fraction 13.3% < 50%
    cand_narrow = [
        {"bytes": 10000, "scores": {"vmaf": 30.0}, "usable": True},
        {"bytes": 20000, "scores": {"vmaf": 38.0}, "usable": True},
        {"bytes": 30000, "scores": {"vmaf": 45.0}, "usable": True},
    ]
    anc_narrow = [
        {"bytes": 50000, "scores": {"vmaf": 43.0}, "usable": True},
        {"bytes": 80000, "scores": {"vmaf": 65.0}, "usable": True},
        {"bytes": 120000, "scores": {"vmaf": 80.0}, "usable": True},
    ]
    comp_narrow = compare_curves_no_extrapolation(cand_narrow, anc_narrow, metric_name="vmaf")
    assert comp_narrow["is_scorable"] is False
    assert comp_narrow["bd_rate_percent"] is None
    assert "need at least 50%" in str(comp_narrow["reason"])

    # Passing unscorable comparisons into confirmation protocol fails closed
    unscorable_source = {
        "source_id": "disjoint_source",
        "match_name": "disjoint_match",
        "comparisons": {
            "av1": {"continuous": comp},
            "vvc": {"continuous": comp},
        },
    }
    verdict = evaluate_confirmation_protocol([unscorable_source], [], is_pilot=False)
    assert verdict["gate_b_passed"] is False
    assert any("non-overlapping or unscorable curves" in b for b in verdict["confirmation_blockers"])


# ---------------------------------------------------------------------------
# 5. Known synthetic curves give expected comparison sign
# ---------------------------------------------------------------------------


def test_known_synthetic_curves_give_expected_comparison_sign() -> None:
    anchor_rows: list[dict[str, Any]] = [
        {"bytes": 100000, "scores": {"vmaf": 50.0}, "usable": True},
        {"bytes": 200000, "scores": {"vmaf": 65.0}, "usable": True},
        {"bytes": 400000, "scores": {"vmaf": 80.0}, "usable": True},
    ]

    # Candidate with 20% lower rate at same qualities
    cand_cheaper: list[dict[str, Any]] = [
        {"bytes": int(float(r["bytes"]) * 0.80), "scores": dict(r.get("scores", {})), "usable": True}
        for r in anchor_rows
    ]
    comp_cheaper = compare_curves_no_extrapolation(cand_cheaper, anchor_rows, metric_name="vmaf")
    assert comp_cheaper["is_scorable"] is True
    assert comp_cheaper["bd_rate_percent"] is not None
    # Rate saving should be ~ -20%
    assert -22.0 <= comp_cheaper["bd_rate_percent"] <= -18.0

    # Candidate with 25% higher rate at same qualities
    cand_pricier: list[dict[str, Any]] = [
        {"bytes": int(float(r["bytes"]) * 1.25), "scores": dict(r.get("scores", {})), "usable": True}
        for r in anchor_rows
    ]
    comp_pricier = compare_curves_no_extrapolation(cand_pricier, anchor_rows, metric_name="vmaf")
    assert comp_pricier["is_scorable"] is True
    assert comp_pricier["bd_rate_percent"] is not None
    # Higher rate should be ~ +25%
    assert 22.0 <= comp_pricier["bd_rate_percent"] <= 28.0

    # Candidate with 10% higher rate at same qualities
    cand_10pct: list[dict[str, Any]] = [
        {"bytes": int(float(r["bytes"]) * 1.10), "scores": dict(r.get("scores", {})), "usable": True}
        for r in anchor_rows
    ]
    comp_10pct = compare_curves_no_extrapolation(cand_10pct, anchor_rows, metric_name="vmaf")
    assert comp_10pct["is_scorable"] is True
    assert comp_10pct["bd_rate_percent"] is not None
    assert 9.0 <= comp_10pct["bd_rate_percent"] <= 11.0

    # Arithmetic check of stored BD-rate
    check_ok = recompute_stored_bd_rate_arithmetic_check(
        cand_10pct, anchor_rows, comp_10pct["bd_rate_percent"], metric_name="vmaf"
    )
    assert check_ok["arithmetic_check_passed"] is True
    assert check_ok["status"] == "arithmetic_check_verified"
    assert check_ok["discrepancy"] is not None and check_ok["discrepancy"] < 0.001

    # Discrepant stored BD-rate fails arithmetic check
    check_bad = recompute_stored_bd_rate_arithmetic_check(
        cand_10pct, anchor_rows, -15.0, metric_name="vmaf"
    )
    assert check_bad["arithmetic_check_passed"] is False
    assert check_bad["status"] == "arithmetic_discrepancy_detected"

    # Identical curves: exactly 0% BD-rate
    comp_identical = compare_curves_no_extrapolation(anchor_rows, anchor_rows, metric_name="vmaf")
    assert comp_identical["is_scorable"] is True
    assert comp_identical["bd_rate_percent"] is not None
    assert abs(comp_identical["bd_rate_percent"]) < 0.01


# ---------------------------------------------------------------------------
# 6. Rescaled outputs have original dimensions and declared timing/byte treatment
# ---------------------------------------------------------------------------


def test_rescaled_outputs_have_original_dimensions_and_timing_byte_treatment() -> None:
    T, H, W = 2, 64, 64
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    frames[..., 0] = 120
    frames[..., 1] = 80
    frames[..., 2] = 200

    # Test rescale to 1/2 (32x32)
    scaled_half, down_s = rescale_frames(frames, 0.5)
    assert scaled_half.shape == (T, 32, 32, 3)
    assert down_s >= 0.0

    # Test restore to display grid (64x64)
    restored_half, up_s = restore_to_display_grid(scaled_half, (H, W))
    assert restored_half.shape == (T, H, W, 3)
    assert up_s >= 0.0

    # Test rescale to 1/4 (16x16)
    scaled_quarter, _ = rescale_frames(frames, 0.25)
    assert scaled_quarter.shape == (T, 16, 16, 3)
    restored_quarter, _ = restore_to_display_grid(scaled_quarter, (H, W))
    assert restored_quarter.shape == (T, H, W, 3)

    # Test mock roundtrip arm timing and bytes
    from src.components.codec.measure import TimedRoundtrip
    mock_trip = TimedRoundtrip(
        size_bytes=1500,
        frames=scaled_half,
        encode_seconds=0.12,
        decode_seconds=0.04,
        tool_path="/bin/test_codec",
        tool_version="v1.0",
        preset="fast",
        qp=40,
    )

    with patch("experiments.tier.resolution_adaptive.timed_roundtrip", return_value=mock_trip):
        pt = encode_resolution_arm(
            frames,
            codec="av1",
            qp=40,
            preset="8",
            scale=0.5,
            fps=24.0,
        )
        assert pt["coded_bytes"] == 1500
        assert pt["scaling_bytes"] == HEADER_BYTES
        assert pt["bytes"] == 1500 + HEADER_BYTES
        assert pt["bytes"] > mock_trip.size_bytes
        assert pt["scale"] == 0.5
        assert pt["scale_label"] == "res_50"
        assert pt["original_resolution"] == "64x64"
        assert pt["coded_resolution"] == "32x32"
        assert pt["restored_resolution"] == "64x64"
        # Client seconds includes decode_seconds + upscale_seconds
        assert pt["timing"]["client_seconds"] >= mock_trip.decode_seconds
        # Encoder seconds includes encode_seconds + downscale_seconds
        assert pt["timing"]["encoder_seconds"] >= mock_trip.encode_seconds
        assert pt["timing"]["rescaling_seconds"] > 0


# ---------------------------------------------------------------------------
# 7. Pilot completion cannot imply confirmation
# ---------------------------------------------------------------------------


def test_pilot_completion_cannot_imply_confirmation() -> None:
    # 6 sources with good savings and complete evidence, but run as a pilot
    sources = [_make_pilot_source(f"src_{i}", f"match_{i}", -15.0) for i in range(6)]
    evidence = _make_valid_evidence()

    verdict = evaluate_confirmation_protocol(
        sources,
        [],
        evidence=evidence,
        is_pilot=True,  # Explicit pilot
        required_matches=6,
    )
    assert verdict["execution_completed"] is True
    assert verdict["pilot_alarms_clear"] is True
    assert verdict["gate_b_passed"] is False
    assert verdict["confirmation_status"] == "development_pilot"
    assert any("pilot completion cannot imply confirmation" in b for b in verdict["confirmation_blockers"])


# ---------------------------------------------------------------------------
# 8. Metric calibration and null controls
# ---------------------------------------------------------------------------


def test_metric_calibration_ordering_and_null_controls() -> None:
    # 2 frames reference
    ref = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    ref[:, :32, :] = 200
    ref[:, 32:, :] = 50

    unrelated = synthetic_unrelated_clip(ref.shape)
    assert unrelated.shape == ref.shape

    t_null = temporal_null_frames(ref)
    assert t_null.shape == ref.shape

    s_null = spatial_null_frames(ref)
    assert s_null.shape == ref.shape

    calib = run_full_metric_calibration(["psnr", "ssim"], ref, unrelated=unrelated)
    assert calib["valid"] is True
    assert len(calib["alarms"]) == 0
    psnr_data = calib["metrics"]["psnr"]
    assert psnr_data["blur_ordering_held"] is True
    assert psnr_data["noise_ordering_held"] is True
    assert psnr_data["by_anchor"]["identical"] == "inf"

    ssim_data = calib["metrics"]["ssim"]
    assert ssim_data["blur_ordering_held"] is True
    assert ssim_data["noise_ordering_held"] is True
    assert ssim_data["distortion_ordering_held"] is True
    assert ssim_data["unrelated_ordering_held"] is True
    assert float(ssim_data["by_anchor"]["identical"]) >= 0.999
    assert "proposed_policy" in calib
    assert calib["proposed_policy"]["name"] == PROPOSED_SSIM_CALIBRATION_POLICY["name"]
    assert calib["reference_identity"]["frame_hashes"]
    assert calib["unrelated_controls"]


def test_ssim_partial_order_without_legacy_unrelated_ceiling() -> None:
    """identity > mild > severe and mild > unrelated; do not require unrelated < 0.60.

    Full-frame SSIM on structured tennis-like content can sit above the legacy
    0.60 unrelated ceiling (Wave 2 observed 0.6702 on natural broadcasts). That
    is a scale finding. Validity is the partial order. Gate B is unchanged.
    """
    ref = synthetic_court_like_clip((2, 96, 96, 3), seed=1)
    unrelated = synthetic_foreign_court_clip((2, 96, 96, 3), seed=99)
    extra = synthetic_foreign_court_clip((2, 96, 96, 3), seed=7)
    calib = run_full_metric_calibration(
        ["ssim"], ref, unrelated=unrelated, extra_unrelated=[extra]
    )
    ssim = calib["metrics"]["ssim"]
    identical = float(ssim["by_anchor"]["identical"])
    mild = float(ssim["by_anchor"]["mild-blur"])
    severe = float(ssim["by_anchor"]["severe-blur"])
    unr = float(ssim["by_anchor"]["unrelated-clip"])
    extra_val = float(ssim["by_anchor"]["unrelated-extra-0"])
    assert identical > mild > severe
    assert mild > unr
    assert identical > extra_val
    assert unr > SSIM_UNRELATED_LEGACY_CEILING
    assert ssim["distortion_ordering_held"] is True
    assert ssim["unrelated_ordering_held"] is True
    finding = next(item for item in calib["scale_findings"] if item["name"] == "legacy_unrelated_ceiling")
    assert finding["ceiling"] == SSIM_UNRELATED_LEGACY_CEILING
    assert finding["observed"] == unr
    assert finding["held"] is False
    assert not any("SSIM unrelated score" in alarm for alarm in calib["alarms"])
    assert calib["valid"] is True
    assert calib["proposed_policy"]["legacy_ssim_unrelated_ceiling"] == 0.60
    assert "gate" in calib["proposed_policy"]["gate_b"].lower()


def test_legacy_ssim_ceiling_finding_records_when_exceeded() -> None:
    """If we kept 0.60 as a hard alarm it would fire on court-like controls.

    The proposed policy records that breach as a scale finding instead of
    invalidating calibration, and does not change Gate B.
    """
    ref = synthetic_court_like_clip((2, 96, 96, 3), seed=0)
    unrelated = synthetic_foreign_court_clip((2, 96, 96, 3), seed=1)
    calib = run_full_metric_calibration(["ssim"], ref, unrelated=unrelated)
    unr = float(calib["metrics"]["ssim"]["by_anchor"]["unrelated-clip"])
    finding = next(item for item in calib["scale_findings"] if item["name"] == "legacy_unrelated_ceiling")
    assert unr > SSIM_UNRELATED_LEGACY_CEILING
    assert finding["held"] is False
    assert calib["valid"] is True
    assert calib["alarms"] == []


def test_temporal_null_high_ssim_is_not_an_alarm() -> None:
    """SSIM is framewise; shuffled temporal order of similar frames scores high."""
    ref = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    ref[0] = 180
    ref[1] = 182
    calib = run_full_metric_calibration(
        ["ssim"], ref, unrelated=synthetic_unrelated_clip(ref.shape)
    )
    null_ssim = float(calib["metrics"]["ssim"]["null_controls"]["temporal-null-shuffled"])
    assert null_ssim > 0.90
    assert not any("temporal" in alarm.lower() for alarm in calib["alarms"])
    finding = next(item for item in calib["scale_findings"] if item["name"] == "legacy_unrelated_ceiling")
    assert finding["temporal_null_is_alarm"] is False
    assert calib["valid"] is True


# ---------------------------------------------------------------------------
# 9. Development recovery manifest holdout protection
# ---------------------------------------------------------------------------


def test_development_recovery_manifest_holdout_protection() -> None:
    manifest = load_development_manifest(DEFAULT_DEV_MANIFEST)
    assert manifest["schema"] == "pointstream.development_recovery.v1"
    assert manifest["status"] == "active_development"

    # Verify holdout protection list
    forbidden = manifest["holdout_protection"]["confirmation_sources_forbidden"]
    assert "bp57_ao2024_sabalenka_zheng" in forbidden
    assert "bp57_usopen2023_gauff_sabalenka" in forbidden

    # Verify no scene in manifest touches forbidden confirmation sources
    scene_ids = [s["source_id"] for s in manifest["scenes"]]
    for fid in forbidden:
        assert fid not in scene_ids
        for sid in scene_ids:
            assert fid not in sid


# ---------------------------------------------------------------------------
# 10. Source-level uncertainty quantification across independent matches
# ---------------------------------------------------------------------------


def test_source_level_uncertainty_quantification() -> None:
    sources = [
        _make_pilot_source("s1", "m1", -20.0),
        _make_pilot_source("s2", "m2", -10.0),
        _make_pilot_source("s3", "m3", -15.0),
    ]
    unc = calculate_source_uncertainty(sources, codecs=("av1", "vvc"))
    assert unc["evaluated"] is True
    assert unc["n_sources"] == 3
    av1_stats = unc["codecs"]["av1"]
    assert av1_stats["bd_rate_mean_percent"] == -15.0
    assert av1_stats["bd_rate_min_percent"] == -20.0
    assert av1_stats["bd_rate_max_percent"] == -10.0
    assert av1_stats["bd_rate_sem_percent"] > 0
