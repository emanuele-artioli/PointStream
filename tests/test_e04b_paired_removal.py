"""Unit tests for PointStream E04B Paired Removal Probe runner and gates."""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest

from scripts import e04b_paired_removal
from scripts.e04b_paired_removal import build_visual_evidence_plate


def test_e04b_refuses_existing_nonempty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E04B runner must refuse to write to an existing non-empty directory."""
    nonempty_dir = tmp_path / "existing_e04b_run"
    nonempty_dir.mkdir(parents=True, exist_ok=True)
    (nonempty_dir / "prior_evidence.txt").write_text("prior output")

    monkeypatch.setattr("sys.argv", ["runner", "--output-dir", str(nonempty_dir)])
    with pytest.raises(FileExistsError, match="Refusing to write to existing non-empty directory"):
        e04b_paired_removal.main()


def test_e04b_aborts_on_calibration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E04B runner must abort immediately with RuntimeError if scorer calibration fails."""
    run_dir = tmp_path / "fresh_e04b_run"

    dummy_frames = np.zeros((2, 360, 640, 3), dtype=np.uint8)
    dummy_masks = np.zeros((2, 360, 640), dtype=bool)
    dummy_meta = {"common_preparation_seconds": 0.1}

    @contextmanager
    def mock_claim(**kwargs: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(token="test_token_1234")

    monkeypatch.setattr(e04b_paired_removal, "claim_resources", mock_claim)
    monkeypatch.setattr(
        e04b_paired_removal,
        "load_360p_input_data",
        lambda: (dummy_frames, dummy_masks, dummy_masks, [], dummy_meta),
    )
    monkeypatch.setattr(
        e04b_paired_removal,
        "run_scorer_calibration",
        lambda f, m, b: {"valid": False, "alarms": ["Forced synthetic calibration failure"]},
    )
    monkeypatch.setattr("sys.argv", ["runner", "--output-dir", str(run_dir), "--cpu-threads", "1"])

    with pytest.raises(RuntimeError, match="Scorer calibration failed with 1 alarms"):
        e04b_paired_removal.main()


def test_visual_evidence_plate_layout() -> None:
    """Visual evidence plate should stack 4 panels into a (2H, 2W, 3) BGR array."""
    h, w = 60, 100
    ref = np.full((h, w, 3), 120, dtype=np.uint8)
    off = np.full((h, w, 3), 130, dtype=np.uint8)
    on = np.full((h, w, 3), 125, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 30:70] = True

    plate = build_visual_evidence_plate(
        ref_rgb=ref,
        off_rgb=off,
        on_rgb=on,
        mask=mask,
        frame_idx=0,
        qp=32,
    )
    assert plate.shape == (2 * h, 2 * w, 3)
    assert plate.dtype == np.uint8
