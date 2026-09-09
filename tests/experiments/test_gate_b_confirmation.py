"""Tests for Gate B confirmation driver."""

from __future__ import annotations

import sqlite3  # noqa: F401
import json
from pathlib import Path

import pytest

from experiments.tier.gate_b_confirmation import (
    DEFAULT_MANIFEST,
    _fill_missing_track_frames,
    load_confirmation_manifest,
    run_dry_run,
    verify_source_integrity,
)
from src.contracts import paths as ps_paths


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


def test_gate_b_source_integrity() -> None:
    manifest = load_confirmation_manifest(DEFAULT_MANIFEST)
    paths = verify_source_integrity(manifest)
    assert len(paths) == 2
    for sid, p in paths.items():
        assert p.is_file()


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

