from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from demo.experiments import background_smoke as smoke


def _save(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def test_known_rgb_errors_report_mean_frame_and_pooled_psnr_separately(tmp_path: Path) -> None:
    reference, reconstruction = [], []
    for index, error in enumerate((1, 10)):
        ref = tmp_path / f"ref{index}.png"
        rec = tmp_path / f"rec{index}.png"
        _save(ref, np.zeros((3, 4, 3), dtype=np.uint8))
        _save(rec, np.full((3, 4, 3), error, dtype=np.uint8))
        reference.append(ref)
        reconstruction.append(rec)
    result = smoke.rgb_sequence_metrics(reference, reconstruction)
    expected_frame = [10 * math.log10(255**2), 10 * math.log10(255**2 / 100)]
    assert result["frame_count"] == 2
    assert result["frame_mse"] == [1.0, 100.0]
    assert result["mean_frame_psnr_db"] == pytest.approx(sum(expected_frame) / 2)
    assert result["pooled_mse_psnr_db"] == pytest.approx(10 * math.log10(255**2 / 50.5))
    assert result["mean_frame_psnr_db"] != pytest.approx(result["pooled_mse_psnr_db"])


def test_temporal_error_uses_rgb_difference_changes(tmp_path: Path) -> None:
    ref0, ref1, rec0, rec1 = [tmp_path / f"{name}.png" for name in ("r0", "r1", "d0", "d1")]
    zeros = np.zeros((2, 2, 3), dtype=np.uint8)
    shifted = np.full((2, 2, 3), 3, dtype=np.uint8)
    for path, image in ((ref0, zeros), (ref1, zeros), (rec0, zeros), (rec1, shifted)):
        _save(path, image)
    result = smoke.rgb_sequence_metrics([ref0, ref1], [rec0, rec1])
    assert result["temporal_reconstruction_error"] == pytest.approx(3 / 255)


def test_rate_uses_actual_bytes_and_common_frame_duration() -> None:
    assert smoke.rate_kbps(1200, frames=8, fps=30) == pytest.approx(36.0)
    with pytest.raises(ValueError):
        smoke.rate_kbps(1200, frames=0)


def test_stream_rate_input_is_summed_from_actual_files(tmp_path: Path) -> None:
    first, second = tmp_path / "part-a.bin", tmp_path / "part-b.bin"
    first.write_bytes(b"header")
    second.write_bytes(b"payload")
    assert smoke.stream_file_bytes([first, second]) == 13
    with pytest.raises(OSError):
        smoke.stream_file_bytes([tmp_path / "missing.bin"])


def test_order_drop_pad_and_changed_source_hash_are_rejected(tmp_path: Path) -> None:
    records = []
    for index in range(120, 128):
        path = tmp_path / f"{index}.png"
        _save(path, np.zeros((2, 2, 3), dtype=np.uint8))
        records.append({
            "index": index, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    smoke.validate_cut(records, start=120, length=8)
    with pytest.raises(ValueError, match="index mismatch"):
        smoke.validate_cut([records[1], records[0], *records[2:]], start=120, length=8)
    with pytest.raises(ValueError, match="has 7 records"):
        smoke.validate_cut(records[:-1], start=120, length=8)
    _save(Path(records[0]["path"]), np.ones((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="content changed"):
        smoke.validate_cut(records, start=120, length=8)


def test_drift_report_requires_exact_32_ordered_frames() -> None:
    rows = [{"index": i, "psnr_db": 30.0 - (i - 120) / 10} for i in range(120, 152)]
    result = smoke.summarize_drift(rows)
    assert result["frame_count"] == 32
    assert result["last8_minus_first8_db"] < 0
    with pytest.raises(ValueError, match="ordered 32 frames"):
        smoke.summarize_drift(rows[:-1])
    rows[5]["psnr_db"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        smoke.summarize_drift(rows)


def test_outputs_are_append_only(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    smoke._write_json_new(target, {"status": "passed"})
    with pytest.raises(FileExistsError, match="overwrite"):
        smoke._write_json_new(target, {"status": "failed"})
