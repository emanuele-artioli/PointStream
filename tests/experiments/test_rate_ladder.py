"""Tests for PointStream Modular Rate Ladder Runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from experiments.modular.measured_ladder import MeasuredInputError
from experiments.modular.rate_ladder import DEFAULT_MANIFEST, run_rate_ladder


def _write_clip(
    root: Path,
    name: str,
    *,
    fg_rgb: tuple[int, int, int],
    n_frames: int = 2,
    size: int = 32,
) -> tuple[Path, Path]:
    frames_dir = root / name
    frames_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size), dtype=bool)
    mask[8:24, 8:24] = True
    for index in range(n_frames):
        frame = np.full((size, size, 3), 40, dtype=np.uint8)
        frame[mask] = np.asarray(fg_rgb, dtype=np.uint8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(str(frames_dir / f"frame_{index:04d}.png"), bgr)
    mask_path = root / f"{name}_mask.npy"
    np.save(mask_path, mask)
    return frames_dir, mask_path


def _write_manifest(path: Path, sources: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"sources": sources}), encoding="utf-8")


def test_missing_frames_dir_raises_before_results_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        out_dir = root / "output"
        frames_dir, mask_path = _write_clip(root, "clip", fg_rgb=(220, 30, 30))
        manifest_path = root / "manifest.json"
        _write_manifest(
            manifest_path,
            [
                {
                    "horizon_id": "short",
                    "n_frames": 2,
                    "scene": "a",
                    "video": "v1",
                    "mask_path": str(mask_path),
                }
            ],
        )
        del frames_dir  # intentionally unused; frames_dir omitted from manifest

        with pytest.raises(MeasuredInputError):
            run_rate_ladder(
                manifest_path=manifest_path,
                output_dir=out_dir,
                dry_run=True,
                enforce_gpu=False,
            )

        assert not (out_dir / "results.json").exists()


def test_two_sources_encoded_report_no_constant_beats() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        out_dir = root / "output"
        red_dir, red_mask = _write_clip(root, "red", fg_rgb=(220, 20, 20))
        blue_dir, blue_mask = _write_clip(root, "blue", fg_rgb=(20, 20, 220))
        manifest_path = root / "manifest.json"
        _write_manifest(
            manifest_path,
            [
                {
                    "horizon_id": "short",
                    "n_frames": 2,
                    "scene": "a",
                    "video": "red",
                    "frames_dir": str(red_dir),
                    "mask_path": str(red_mask),
                },
                {
                    "horizon_id": "long",
                    "n_frames": 2,
                    "scene": "b",
                    "video": "blue",
                    "frames_dir": str(blue_dir),
                    "mask_path": str(blue_mask),
                },
            ],
        )

        report = run_rate_ladder(
            manifest_path=manifest_path,
            output_dir=out_dir,
            dry_run=True,
            enforce_gpu=False,
            measure_anchors=False,
        )

        assert report["measurement"] == "encoded"
        assert len(report["horizons"]) == 2

        c0_appearance: list[int] = []
        for horizon in report["horizons"]:
            assert horizon["anchor_vvc_bytes"] is None
            for rung in horizon["rungs"]:
                assert rung["beats_vvc_rate"] is False
                total = (
                    rung["bytes_background"]
                    + rung["bytes_appearance"]
                    + rung["bytes_metadata"]
                    + rung["bytes_residual"]
                    + rung["bytes_container"]
                )
                assert rung["total_bytes"] == total
                if rung["rung_id"] == "C0_compact_baseline":
                    c0_appearance.append(int(rung["bytes_appearance"]))

        assert len(c0_appearance) == 2
        assert c0_appearance[0] != c0_appearance[1]


def test_generate_visuals_writes_non_flat_strip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        out_dir = root / "output"
        visuals_dir = root / "visuals"
        frames_dir, mask_path = _write_clip(root, "clip", fg_rgb=(200, 40, 40))
        manifest_path = root / "manifest.json"
        _write_manifest(
            manifest_path,
            [
                {
                    "horizon_id": "short",
                    "n_frames": 2,
                    "scene": "s0",
                    "video": "demo",
                    "frames_dir": str(frames_dir),
                    "mask_path": str(mask_path),
                }
            ],
        )

        run_rate_ladder(
            manifest_path=manifest_path,
            output_dir=out_dir,
            visuals_dir=visuals_dir,
            dry_run=True,
            generate_visuals=True,
            enforce_gpu=False,
        )

        strip_path = visuals_dir / "comparison_demo_s0_short.png"
        assert strip_path.is_file()
        image = cv2.imread(str(strip_path), cv2.IMREAD_COLOR)
        assert image is not None
        assert float(np.std(image)) > 0.0


def test_n_frames_larger_than_pngs_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        frames_dir, mask_path = _write_clip(root, "clip", fg_rgb=(180, 50, 50), n_frames=2)
        manifest_path = root / "manifest.json"
        _write_manifest(
            manifest_path,
            [
                {
                    "horizon_id": "short",
                    "n_frames": 8,
                    "scene": "a",
                    "video": "v1",
                    "frames_dir": str(frames_dir),
                    "mask_path": str(mask_path),
                }
            ],
        )

        with pytest.raises(MeasuredInputError):
            run_rate_ladder(
                manifest_path=manifest_path,
                output_dir=root / "output",
                dry_run=True,
                enforce_gpu=False,
            )


def test_default_manifest_without_frame_paths_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(MeasuredInputError):
            run_rate_ladder(
                manifest_path=DEFAULT_MANIFEST,
                output_dir=Path(tmp_dir) / "output",
                dry_run=True,
                enforce_gpu=False,
            )


def test_manifest_scope_weights_and_horizon_length() -> None:
    from experiments.modular.rate_ladder import _n_frames, _saliency_weights

    weights = _saliency_weights(
        {
            "evaluation_scope": {
                "saliency_weights": {"foreground": 0.25, "background": 0.75}
            }
        }
    )
    assert weights == (0.25, 0.75)
    assert _n_frames(
        {"horizon": "long"},
        {"horizons": [{"id": "long", "n_frames": 192}]},
    ) == 192
