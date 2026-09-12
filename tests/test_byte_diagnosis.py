"""Focused unit tests for byte diagnosis and component budget (EVAL-ACT-08)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.byte_diagnosis import (
    analyze_federer_rungs,
    compute_bitrate_kbps,
    compute_bitrate_mbps,
    compute_runtime_scope,
    evaluate_hypothesis_and_interventions,
    interpolate_anchor_rate,
    reconcile_bound_alarms,
    run_arithmetic_controls,
    verify_provenance,
)


def test_reconciliation_arithmetic_equality() -> None:
    """Validate that component sum B + F + M + R + H strictly equals total bytes.

    Also verifies that H = 0 represents zero unallocated arithmetic remainder in
    B + F + M + R + H = T, while envelope/container overhead resides in M (metadata).
    """
    mock_source: dict[str, Any] = {
        "source_id": "test_scene",
        "video": "test_video",
        "scene": "scene_001",
        "n_frames": 48,
        "pointstream_rungs": [
            {
                "name": "R_test_zero_overhead",
                "bytes": 1000,
                "parts": {
                    "panorama": 600,
                    "actor_reference": 50,
                    "metadata": 150,
                    "residual": 200,
                },
                "scores": {"vmaf": 80.0, "psnr_y": 35.0, "ssim": 0.95},
                "timing": {"encoder_seconds": 10.0, "client_seconds": 1.0},
            },
            {
                "name": "R_test_with_overhead",
                "bytes": 1050,
                "parts": {
                    "panorama": 600,
                    "actor_reference": 50,
                    "metadata": 150,
                    "residual": 200,
                },
                "scores": {"vmaf": 80.0, "psnr_y": 35.0, "ssim": 0.95},
                "timing": {"encoder_seconds": 10.0, "client_seconds": 1.0},
            },
        ],
        "anchors": {
            "av1": {"nondominated_envelope": []},
            "vvc": {"nondominated_envelope": []},
        },
    }

    analysis = analyze_federer_rungs(mock_source)
    rungs = analysis["rungs"]

    # First rung: H == 0
    r0 = rungs[0]
    recon0 = r0["reconciliation"]
    assert recon0["B_panorama"] == 600
    assert recon0["F_actor_reference"] == 50
    assert recon0["M_metadata"] == 150
    assert recon0["R_residual"] == 200
    assert recon0["H_overhead"] == 0
    assert recon0["H_unallocated_remainder"] == 0
    assert "ledger_semantics" in recon0
    assert "disjoint" in recon0["ledger_semantics"]
    assert recon0["sum_parts"] == 1000
    assert recon0["equality_verified"] is True
    assert r0["shares_percent"]["B_share"] == 60.0
    assert r0["shares_percent"]["R_share"] == 20.0

    # Second rung: H == 50
    r1 = rungs[1]
    recon1 = r1["reconciliation"]
    assert recon1["H_overhead"] == 50
    assert recon1["H_unallocated_remainder"] == 50
    assert recon1["sum_parts"] == 1000
    assert recon1["equality_verified"] is False  # h_bytes != 0


def test_anchor_interpolation_and_remaining_budget() -> None:
    """Validate anchor interpolation, remaining budget A(q)-B-M-H, and recoverable contributions."""
    # Synthetic anchor envelope spanning quality [60, 90] with exponential rate [100, 1000]
    anchor_envelope: list[dict[str, Any]] = [
        {"bytes": 100, "scores": {"vmaf": 60.0, "psnr_y": 25.0}, "usable": True},
        {"bytes": 316, "scores": {"vmaf": 75.0, "psnr_y": 30.0}, "usable": True},
        {"bytes": 1000, "scores": {"vmaf": 90.0, "psnr_y": 35.0}, "usable": True},
    ]

    # Interpolate at target VMAF = 75.0 -> anchor rate should be ~316
    interp = interpolate_anchor_rate(anchor_envelope, 75.0, "vmaf")
    assert interp["linear_bytes"] is not None
    assert interp["linear_bytes"] == pytest.approx(316.0, rel=1e-2)

    # PointStream candidate at VMAF 75.0:
    # Total = 800 (B=400, F=50, M=100, R=250)
    mock_source: dict[str, Any] = {
        "source_id": "test_scene",
        "pointstream_rungs": [
            {
                "name": "R75",
                "bytes": 800,
                "parts": {"panorama": 400, "actor_reference": 50, "metadata": 100, "residual": 250},
                "scores": {"vmaf": 75.0, "psnr_y": 30.0, "ssim": 0.95},
                "timing": {},
            }
        ],
        "anchors": {
            "av1": {"nondominated_envelope": anchor_envelope},
            "vvc": {"nondominated_envelope": []},
        },
    }

    analysis = analyze_federer_rungs(mock_source)
    comp = analysis["rungs"][0]["anchor_comparisons"]["av1_vmaf"]

    # A(q) ~ 316
    # Remaining budget A(q) - B - M - H = 316 - 400 - 100 - 0 = -184
    assert comp["remaining_budget_A_minus_B_M_H"] == pytest.approx(-184.0, abs=1.0)
    # Remaining budget A(q) - B - M - F - H = -184 - 50 = -234
    assert comp["remaining_budget_A_minus_B_M_F_H"] == pytest.approx(-234.0, abs=1.0)

    # Required saving: 800 - 316 = 484
    assert comp["required_saving_bytes"] == pytest.approx(484.0, abs=1.0)

    # If B=0: saves 400 B -> 400 / 484 = 82.6% of required saving.
    # New total is 800 - 400 = 400 > 316, so beats_anchor_alone is False!
    recov_b = comp["recoverable_contributions"]["if_B_zero"]
    assert recov_b["saved_bytes"] == 400
    assert recov_b["fraction_of_req_saving"] == pytest.approx(400 / 484.0, rel=1e-2)
    assert recov_b["beats_anchor_alone"] is False


def test_arithmetic_controls_same_anchor_and_doubled_byte() -> None:
    """Validate same-anchor zero control (0.0% BD-rate) and doubled-byte control (+100.0% BD-rate)."""
    envelope: list[dict[str, Any]] = [
        {"bytes": 1000, "scores": {"vmaf": 30.0, "psnr_y": 20.0}, "usable": True},
        {"bytes": 2000, "scores": {"vmaf": 50.0, "psnr_y": 25.0}, "usable": True},
        {"bytes": 4000, "scores": {"vmaf": 70.0, "psnr_y": 30.0}, "usable": True},
        {"bytes": 8000, "scores": {"vmaf": 90.0, "psnr_y": 35.0}, "usable": True},
    ]

    controls = run_arithmetic_controls(envelope)
    assert controls["vmaf"]["controls_passed"] is True
    assert controls["vmaf"]["same_anchor_zero_control"]["bd_rate_percent"] == pytest.approx(
        0.0, abs=1e-3
    )
    assert controls["vmaf"]["doubled_byte_control"]["bd_rate_percent"] == pytest.approx(
        100.0, abs=1e-3
    )

    assert controls["psnr"]["controls_passed"] is True
    assert controls["psnr"]["same_anchor_zero_control"]["bd_rate_percent"] == pytest.approx(
        0.0, abs=1e-3
    )
    assert controls["psnr"]["doubled_byte_control"]["bd_rate_percent"] == pytest.approx(
        100.0, abs=1e-3
    )


def test_bound_alarms_reconciliation() -> None:
    """Validate bound alarms reconciliation matches expected fail-closed protocol status."""
    bounds_data = {
        "bands": {
            "psnr_y_dB": [12.0, 50.0],
            "ssim": [0.5, 1.0],
            "vmaf": [15.0, 99.0],
            "ps_rung_bytes": [80000, 20000000],
            "encoder_seconds": [5.0, 2400.0],
            "client_seconds": [0.2, 1800.0],
        },
        "gates": {"gate_b_passed": False},
    }
    report_data = {
        "gate_b_passed": False,
        "pilot_alarms_clear": True,
        "identity_verified": False,
        "evidence_verified": False,
        "alarms": [],
        "sources": [
            {
                "pointstream_rungs": [
                    {
                        "name": "R63",
                        "bytes": 700102,
                        "scores": {"vmaf": 73.65, "psnr_y": 33.61, "ssim": 0.9666},
                        "timing": {"encoder_seconds": 627.5, "client_seconds": 36.3},
                    }
                ]
            }
        ],
    }

    recon = reconcile_bound_alarms(bounds_data, report_data)
    assert recon["gate_b_passed"]["status"] == "gate_b_passed=false"
    assert recon["pilot_alarms_clear"]["status"] == "pilot_alarms_clear=true"
    assert recon["identity_verified"]["status"] == "identity_verified=false"
    assert recon["evidence_verified"]["status"] == "evidence_verified=false"


def test_provenance_verification_real_inputs() -> None:
    """Verify hashes of the real recovery files match docs/areas/evaluation.md."""
    input_dir = Path(
        "/home/itec/emanuele/pointstream-data/outputs/development-recovery/wave2-overlap-20260910"
    )
    if not input_dir.exists():
        pytest.skip("Input directory not present on this host")

    prov = verify_provenance(input_dir)
    assert prov["all_matched"] is True
    for fname, finfo in prov["files"].items():
        assert finfo["exists"] is True
        assert finfo["match"] is True


def test_bitrate_calculation_with_framerate() -> None:
    """Verify bitrate calculation helper includes frame rate (Finding 3).

    11,028 bytes/frame at 24.0 fps is 2,117.376 kbps (~2.12 Mbps), NOT 88.2 kbps.
    """
    bytes_per_frame = 11028.0

    # Default fps = 24.0
    kbps_default = compute_bitrate_kbps(bytes_per_frame)
    mbps_default = compute_bitrate_mbps(bytes_per_frame)
    assert kbps_default == pytest.approx(2117.376, rel=1e-4)
    assert mbps_default == pytest.approx(2.117376, rel=1e-4)

    # Explicit fps = 24.0
    kbps_24 = compute_bitrate_kbps(bytes_per_frame, fps=24.0)
    mbps_24 = compute_bitrate_mbps(bytes_per_frame, fps=24.0)
    assert kbps_24 == pytest.approx(2117.376, rel=1e-4)
    assert mbps_24 == pytest.approx(2.117376, rel=1e-4)

    # Demonstrating the defect in Finding 3: omitting fps gave 88.224 kbps
    omitted_fps_kbps = bytes_per_frame * 8.0 / 1000.0
    assert omitted_fps_kbps == pytest.approx(88.224, rel=1e-4)
    assert kbps_24 == pytest.approx(omitted_fps_kbps * 24.0, rel=1e-4)

    # Custom fps = 30.0
    kbps_30 = compute_bitrate_kbps(bytes_per_frame, fps=30.0)
    assert kbps_30 == pytest.approx(2646.72, rel=1e-4)


def test_dynamic_hypothesis_evaluation_and_background_sufficiency() -> None:
    """Validate dynamic hypothesis evaluation and background sufficiency limits (Finding 3)."""
    # Case 1: Setting background B to zero leaves more bytes than anchor budget (e.g. Federer R63)
    mock_federer_analysis: dict[str, Any] = {
        "source_id": "federer_djokovic_scene_007",
        "n_frames": 48,
        "rungs": [
            {
                "rung": "R63",
                "total_bytes": 700102,
                "reconciliation": {
                    "B_panorama": 529361,
                    "F_actor_reference": 30196,
                    "M_metadata": 70609,
                    "R_residual": 69936,
                    "H_overhead": 0,
                    "H_unallocated_remainder": 0,
                    "sum_parts": 700102,
                    "equality_verified": True,
                },
                "shares_percent": {
                    "B_share": 75.61,
                    "F_share": 4.31,
                    "M_share": 10.09,
                    "R_share": 9.99,
                    "H_share": 0.0,
                },
                "anchor_comparisons": {
                    "av1_vmaf": {
                        "anchor_rate_linear_bytes": 110842.3,
                        "remaining_budget_A_minus_B_M_H": -489127.7,
                        "required_saving_bytes": 589259.7,
                    },
                    "vvc_vmaf": {
                        "anchor_rate_linear_bytes": 130906.2,
                        "remaining_budget_A_minus_B_M_H": -469063.8,
                        "required_saving_bytes": 569195.8,
                    },
                },
            },
            {
                "rung": "H3",
                "total_bytes": 3645780,
                "reconciliation": {
                    "B_panorama": 529361,
                    "F_actor_reference": 30196,
                    "M_metadata": 70611,
                    "R_residual": 3015612,
                    "H_overhead": 0,
                    "H_unallocated_remainder": 0,
                    "sum_parts": 3645780,
                    "equality_verified": True,
                },
                "shares_percent": {
                    "B_share": 14.52,
                    "F_share": 0.83,
                    "M_share": 1.94,
                    "R_share": 82.72,
                    "H_share": 0.0,
                },
                "anchor_comparisons": {
                    "av1_vmaf": {"anchor_rate_linear_bytes": 354280.1},
                    "vvc_vmaf": {"anchor_rate_linear_bytes": 377342.8},
                },
            },
        ],
    }

    eval_result = evaluate_hypothesis_and_interventions(mock_federer_analysis, fps=24.0)

    # Check that output is not hardcoded and contains no canned fixed 50 kB rule
    eval_str = str(eval_result)
    assert "below 50 kB" not in eval_str
    assert "below ~50 kB" not in eval_str

    # Background sufficiency limit check:
    # Setting B=0 leaves 700,102 - 529,361 = 170,741 B > AV1 (110,842 B) and > VVC (130,906 B)
    suff_limit = eval_result["background_sufficiency_limit"]
    assert suff_limit["lowest_rung"] == "R63"
    assert suff_limit["bytes_without_background"] == 170741
    assert suff_limit["beats_anchor_without_background"] is False
    assert "alone cannot be assumed sufficient" in suff_limit["sufficiency_note"]

    # Ranked interventions check:
    rank1 = eval_result["ranked_interventions"][0]
    assert rank1["target"] == "Background representation (B)"
    assert rank1["fps"] == 24.0
    assert rank1["bitrate_mbps"] == pytest.approx(2.12, abs=0.01)
    assert rank1["bitrate_kbps"] == pytest.approx(2117.4, abs=1.0)
    assert "2.12 Mbps" in rank1["current_cost_summary"]

    # Promotion verdict:
    assert eval_result["promotion_verdict"]["promote_background_coding_for_wave2"] is True

    # Case 2: Synthetic case where non-background bytes ARE below anchor rate
    mock_federer_analysis["rungs"][0]["anchor_comparisons"]["av1_vmaf"][
        "anchor_rate_linear_bytes"
    ] = 200000.0
    mock_federer_analysis["rungs"][0]["anchor_comparisons"]["vvc_vmaf"][
        "anchor_rate_linear_bytes"
    ] = 200000.0
    eval_result_sufficient = evaluate_hypothesis_and_interventions(mock_federer_analysis, fps=24.0)
    # 170,741 B <= 200,000 B
    assert (
        eval_result_sufficient["background_sufficiency_limit"]["beats_anchor_without_background"]
        is True
    )


def test_dynamic_runtime_scope() -> None:
    """Validate dynamic calculation of runtime ranges and speed ratios."""
    mock_source: dict[str, Any] = {
        "pointstream_rungs": [
            {"timing": {"encoder_seconds": 100.0, "client_seconds": 5.0}},
            {"timing": {"encoder_seconds": 500.0, "client_seconds": 25.0}},
        ],
        "anchors": {
            "av1": {
                "nondominated_envelope": [
                    {"usable": True, "timing": {"encode_seconds": 2.0, "client_seconds": 4.0}},
                    {"usable": True, "timing": {"encode_seconds": 10.0, "client_seconds": 8.0}},
                ]
            },
            "vvc": {
                "nondominated_envelope": [
                    {"usable": True, "timing": {"encode_seconds": 5.0, "client_seconds": 6.0}},
                    {"usable": True, "timing": {"encode_seconds": 50.0, "client_seconds": 12.0}},
                ]
            },
        },
    }

    scope = compute_runtime_scope(mock_source)

    assert scope["pointstream_encoder_seconds_range"] == [100.0, 500.0]
    assert scope["pointstream_client_seconds_range"] == [5.0, 25.0]
    assert scope["av1_encoder_seconds_range"] == [2.0, 10.0]
    assert scope["av1_client_seconds_range"] == [4.0, 8.0]
    assert scope["vvc_encoder_seconds_range"] == [5.0, 50.0]
    assert scope["vvc_client_seconds_range"] == [6.0, 12.0]

    # AV1 speed ratio range: min(100/10)=10.0x, max(500/2)=250.0x
    assert scope["av1_encoder_speed_ratio_range"] == [10.0, 250.0]
    # VVC speed ratio range: min(100/50)=2.0x, max(500/5)=100.0x
    assert scope["vvc_encoder_speed_ratio_range"] == [2.0, 100.0]
    assert "10.0x–250.0x slower than SVT-AV1" in scope["speed_ratio_vs_anchor"]
    assert "2.0x–100.0x slower than VVC" in scope["speed_ratio_vs_anchor"]
