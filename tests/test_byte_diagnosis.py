"""Focused unit tests for byte diagnosis and component budget (EVAL-ACT-08)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.byte_diagnosis import (
    analyze_federer_rungs,
    interpolate_anchor_rate,
    reconcile_bound_alarms,
    run_arithmetic_controls,
    verify_provenance,
)


def test_reconciliation_arithmetic_equality() -> None:
    """Validate that component sum B + F + M + R + H strictly equals total bytes."""
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
    assert recon0["sum_parts"] == 1000
    assert recon0["equality_verified"] is True
    assert r0["shares_percent"]["B_share"] == 60.0
    assert r0["shares_percent"]["R_share"] == 20.0

    # Second rung: H == 50
    r1 = rungs[1]
    recon1 = r1["reconciliation"]
    assert recon1["H_overhead"] == 50
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
    assert controls["vmaf"]["same_anchor_zero_control"]["bd_rate_percent"] == pytest.approx(0.0, abs=1e-3)
    assert controls["vmaf"]["doubled_byte_control"]["bd_rate_percent"] == pytest.approx(100.0, abs=1e-3)

    assert controls["psnr"]["controls_passed"] is True
    assert controls["psnr"]["same_anchor_zero_control"]["bd_rate_percent"] == pytest.approx(0.0, abs=1e-3)
    assert controls["psnr"]["doubled_byte_control"]["bd_rate_percent"] == pytest.approx(100.0, abs=1e-3)


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
    input_dir = Path("/home/itec/emanuele/pointstream-data/outputs/development-recovery/wave2-overlap-20260910")
    if not input_dir.exists():
        pytest.skip("Input directory not present on this host")

    prov = verify_provenance(input_dir)
    assert prov["all_matched"] is True
    for fname, finfo in prov["files"].items():
        assert finfo["exists"] is True
        assert finfo["match"] is True
