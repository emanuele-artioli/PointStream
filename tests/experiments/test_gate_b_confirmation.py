"""Tests for Gate B confirmation driver."""

from __future__ import annotations

import sqlite3  # noqa: F401
import json
from pathlib import Path


import pytest

from experiments.tier.gate_b_confirmation import (
    DEFAULT_MANIFEST,
    confirmation_verdict,
    _fill_missing_track_frames,
    load_confirmation_manifest,
    run_dry_run,
    verify_source_integrity,
)
from src.contracts import paths as ps_paths


def _has_confirmation_raw_assets() -> bool:
    manifest = load_confirmation_manifest(DEFAULT_MANIFEST)
    data_root = ps_paths.data_root()
    return all((data_root / s["raw_file_path"]).is_file() for s in manifest.get("sources", []))


def test_gate_b_manifest_loads_and_validates() -> None:
    manifest = load_confirmation_manifest(DEFAULT_MANIFEST)
    assert manifest["schema"] == "pointstream.gate_b_confirmation.v1"
    assert manifest["roadmap_gate"] == "Gate B (Held-Out Confirmation)"
    assert len(manifest["sources"]) == 2

    # Check sources
    sids = [s["source_id"] for s in manifest["sources"]]
    assert "bp57_ao2024_sabalenka_zheng" in sids
    assert "bp57_usopen2023_gauff_sabalenka" in sids

    # Check procedure freeze
    assert manifest["procedure_freeze"]["module"] == "src.contracts.frozen_procedure"
    assert len(manifest["procedure_freeze"]["rate_ladder"]) == 4


@pytest.mark.skipif(
    not _has_confirmation_raw_assets(),
    reason="requires external confirmation raw video assets",
)
def test_gate_b_source_integrity() -> None:
    manifest = load_confirmation_manifest(DEFAULT_MANIFEST)
    paths = verify_source_integrity(manifest)
    assert len(paths) == 2
    for sid, p in paths.items():
        assert p.is_file()


@pytest.mark.skipif(
    not _has_confirmation_raw_assets(),
    reason="requires external confirmation raw video assets",
)
def test_gate_b_dry_run(tmp_path: Path) -> None:
    summary = run_dry_run(tmp_path, n_frames=48)
    assert summary["status"] == "gate_b_dry_run_complete"
    assert summary["n_frames"] == 48
    assert len(summary["sources"]) == 2
    assert len(summary["ladder_plan"]) == 4

    summary_file = tmp_path / "dry-run-summary.json"
    assert summary_file.is_file()
    saved = json.loads(summary_file.read_text(encoding="utf-8"))
    assert saved["status"] == "gate_b_dry_run_complete"


def test_fill_missing_track_frames() -> None:
    test_dict = {3: (10, 20, 30, 40), 7: (20, 30, 40, 50)}
    filled = _fill_missing_track_frames(test_dict, T=10, width=100, height=100)
    assert len(filled) == 10
    # Before earliest
    assert filled[0] == (10, 20, 30, 40)
    assert filled[2] == (10, 20, 30, 40)
    # At exact
    assert filled[3] == (10, 20, 30, 40)
    assert filled[7] == (20, 30, 40, 50)
    # Interpolated
    assert filled[5] == (15, 25, 35, 45)
    # After latest
    assert filled[9] == (20, 30, 40, 50)



def _pilot_source(source_id: str, delta: float | None) -> dict:
    return {
        "source_id": source_id,
        "comparisons": {
            codec: {"continuous": {"bd_rate_percent": delta}}
            for codec in ("av1", "vvc")
        },
    }


def test_empty_confirmation_cannot_pass() -> None:
    verdict = confirmation_verdict([], [])
    assert verdict["gate_b_passed"] is False
    assert verdict["execution_completed"] is False
    assert verdict["pilot_alarms_clear"] is False
    assert any("only 0" in reason for reason in verdict["confirmation_blockers"])


def test_two_source_pilot_is_not_six_source_confirmation() -> None:
    sources = [_pilot_source("a", -10.0), _pilot_source("b", -10.0)]
    verdict = confirmation_verdict(sources, [])
    assert verdict["execution_completed"] is True
    assert verdict["pilot_alarms_clear"] is True
    assert verdict["gate_b_passed"] is False
    assert any("only 2" in reason for reason in verdict["confirmation_blockers"])


@pytest.mark.parametrize("delta", [10.0, 0.0, None, float("nan")])
def test_bad_or_missing_comparison_is_reported(delta: float | None) -> None:
    verdict = confirmation_verdict([_pilot_source("match", delta)], [])
    assert verdict["gate_b_passed"] is False
    reasons = verdict["confirmation_blockers"]
    assert any("match/av1:" in reason for reason in reasons)
    assert any("match/vvc:" in reason for reason in reasons)


def test_favorable_curves_do_not_replace_missing_protocol_validation() -> None:
    sources = [_pilot_source(str(i), -10.0) for i in range(6)]
    verdict = confirmation_verdict(sources, [])
    assert verdict["pilot_alarms_clear"] is True
    assert verdict["gate_b_passed"] is False
    assert verdict["confirmation_status"] == "incomplete_protocol"
    assert any("validation is not implemented" in x for x in verdict["confirmation_blockers"])


def test_measurement_alarm_is_not_hidden_by_completed_execution() -> None:
    verdict = confirmation_verdict([_pilot_source("match", -10.0)], ["ledger mismatch"])
    assert verdict["execution_completed"] is True
    assert verdict["pilot_alarms_clear"] is False
    assert verdict["gate_b_passed"] is False
    assert any("alarms remain" in x for x in verdict["confirmation_blockers"])


def test_empty_manifest_driver_writes_nonpassing_report(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace
    from experiments.tier import gate_b_confirmation as driver

    monkeypatch.setattr(driver, "load_confirmation_manifest", lambda path: {"sources": []})
    monkeypatch.setattr(driver, "verify_source_integrity", lambda manifest: {})
    monkeypatch.setattr(driver, "write_tool_identity", lambda destination: {})
    monkeypatch.setattr(driver, "resolve_ffmpeg", lambda: SimpleNamespace(path="unused-ffmpeg"))
    result = driver.run_confirmation(tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == result
    assert saved["status"] == "gate_b_pilot_complete"
    assert saved["gate_b_passed"] is False
    assert saved["execution_completed"] is False
    assert saved["confirmation_blockers"]
