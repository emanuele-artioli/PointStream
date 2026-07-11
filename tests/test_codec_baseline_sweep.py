from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import codec_baseline_sweep as sweep


def test_parse_kv_list() -> None:
    assert sweep._parse_kv_list(["libsvtav1=20,30,40"]) == {"libsvtav1": "20,30,40"}
    assert sweep._parse_kv_list(["libsvtav1=a", "libx265=b"]) == {"libsvtav1": "a", "libx265": "b"}
    assert sweep._parse_kv_list(None) == {}


def test_parse_kv_list_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        sweep._parse_kv_list(["not-a-kv-pair"])


def test_parse_crf_overrides() -> None:
    overrides = sweep._parse_crf_overrides(["libsvtav1=20, 30 ,40", "libx265=18,28"])
    assert overrides == {"libsvtav1": [20, 30, 40], "libx265": [18, 28]}


def test_parse_preset_overrides() -> None:
    overrides = sweep._parse_preset_overrides(["libsvtav1=slow, veryslow", "libx265=medium"])
    assert overrides == {"libsvtav1": ["slow", "veryslow"], "libx265": ["medium"]}


def _fake_encode_fn(bytes_by_crf: dict[int, int]):
    def _fn(
        input_path: Path,
        output_path: Path,
        codec: str,
        crf: int,
        preset: str,
        max_frames: int | None,
    ) -> float:
        output_path.write_bytes(b"\x00" * bytes_by_crf[crf])
        return 1.5

    return _fn


def _fake_evaluate_fn(quality_by_crf: dict[int, float]):
    def _fn(input_path: Path, output_path: Path, max_frames: int | None) -> dict[str, float | None]:
        crf = int(output_path.stem.rsplit("crf", maxsplit=1)[1])
        return {"psnr_mean": quality_by_crf[crf], "ssim_mean": 0.9, "vmaf_mean": 80.0}

    return _fn


def test_run_sweep_wires_encode_and_evaluate(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    input_path.write_bytes(b"\x00" * 100_000)
    output_root = tmp_path / "sweep_out"

    # Higher CRF -> smaller file, lower PSNR: a real sweep should show this shape.
    points = sweep.run_sweep(
        input_path,
        output_root,
        codecs=["libsvtav1"],
        crf_overrides={"libsvtav1": [20, 40]},
        preset_overrides={},
        encode_fn=_fake_encode_fn({20: 50_000, 40: 10_000}),
        evaluate_fn=_fake_evaluate_fn({20: 45.0, 40: 30.0}),
    )

    assert [p.crf for p in points] == [20, 40]
    assert points[0].output_bytes == 50_000
    assert points[1].output_bytes == 10_000
    assert points[0].psnr_mean == 45.0
    assert points[1].psnr_mean == 30.0
    assert points[0].codec_label == "AV1"
    assert points[0].preset == sweep.DEFAULT_PRESET
    assert points[0].source_bytes == 100_000


def test_run_sweep_default_crf_ladder_used_when_no_override(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    input_path.write_bytes(b"\x00" * 10)
    output_root = tmp_path / "sweep_out"

    points = sweep.run_sweep(
        input_path,
        output_root,
        codecs=["libx265"],
        crf_overrides={},
        preset_overrides={"libx265": ["medium"]},
        encode_fn=_fake_encode_fn(dict.fromkeys(sweep.DEFAULT_CRFS["libx265"], 1)),
        evaluate_fn=_fake_evaluate_fn(dict.fromkeys(sweep.DEFAULT_CRFS["libx265"], 40.0)),
    )

    assert [p.crf for p in points] == list(sweep.DEFAULT_CRFS["libx265"])
    assert all(p.preset == "medium" for p in points)


def test_run_sweep_cross_products_multiple_presets_with_crf(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    input_path.write_bytes(b"\x00" * 10)
    output_root = tmp_path / "sweep_out"

    points = sweep.run_sweep(
        input_path,
        output_root,
        codecs=["libsvtav1"],
        crf_overrides={"libsvtav1": [20, 40]},
        preset_overrides={"libsvtav1": ["slow", "veryslow"]},
        encode_fn=_fake_encode_fn({20: 1, 40: 1}),
        evaluate_fn=_fake_evaluate_fn({20: 40.0, 40: 30.0}),
    )

    assert len(points) == 4
    assert {(p.preset, p.crf) for p in points} == {
        ("slow", 20),
        ("slow", 40),
        ("veryslow", 20),
        ("veryslow", 40),
    }
    # Every point gets a distinct output path (preset baked into the filename).
    assert len({p.output_path for p in points}) == 4


def test_load_pointstream_point_reads_run_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "sizes_bytes": {"transport_total": 5000, "source": 100_000},
                    "psnr_mean": 38.0,
                    "ssim_mean": 0.95,
                    "vmaf_mean": 90.0,
                }
            }
        ),
        encoding="utf-8",
    )

    point = sweep._load_pointstream_point(run_dir)
    assert point is not None
    assert point["transport_total_bytes"] == 5000
    assert point["source_bytes"] == 100_000
    assert point["psnr_mean"] == 38.0


def test_load_pointstream_point_missing_summary_returns_none(tmp_path: Path) -> None:
    assert sweep._load_pointstream_point(tmp_path / "nonexistent") is None


def test_build_report_includes_pointstream_comparison(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    points = [
        sweep.SweepPoint(
            codec="libsvtav1",
            codec_label="AV1",
            crf=30,
            preset="fast",
            elapsed_sec=12.0,
            output_bytes=20_000,
            source_bytes=100_000,
            psnr_mean=35.0,
            ssim_mean=0.9,
            vmaf_mean=75.0,
            output_path=str(tmp_path / "libsvtav1_crf30.mp4"),
        )
    ]
    pointstream_point = {
        "label": "pointstream (semantic decomposition)",
        "transport_total_bytes": 8000,
        "source_bytes": 100_000,
        "psnr_mean": 40.0,
        "ssim_mean": 0.95,
        "vmaf_mean": 88.0,
        "run_dir": "/outputs/some_run",
    }

    markdown, payload = sweep.build_report(points, input_path=input_path, pointstream_point=pointstream_point)

    assert "AV1" in markdown
    assert "PointStream anchor" in markdown
    assert payload["points"][0]["crf"] == 30
    assert payload["pointstream_point"]["transport_total_bytes"] == 8000


def test_build_report_without_pointstream_comparison(tmp_path: Path) -> None:
    input_path = tmp_path / "source.mp4"
    markdown, payload = sweep.build_report([], input_path=input_path, pointstream_point=None)
    assert "PointStream anchor" not in markdown
    assert payload["pointstream_point"] is None
