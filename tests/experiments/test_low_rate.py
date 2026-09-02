"""Low-rate search helpers: reject bad decodes, require bounds, stay staged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tier.low_rate_bounds import bounds_document, write_bounds
from experiments.tier.low_rate_plan import STAGES, all_points, ledger_moved, points_for, stage_names
from experiments.tier.low_rate_validate import (
    decode_rejections,
    monotonicity_alarms,
    probe_qps,
    slowest_preset,
)


def test_slowest_av1_preset_is_zero() -> None:
    assert slowest_preset("av1") == "0"
    assert slowest_preset("vvc") == "placebo"
    assert slowest_preset("vvc", available=("medium", "slower", "fast")) == "slower"


def test_unknown_codec_has_no_slowest_preset() -> None:
    with pytest.raises(ValueError, match="no slowest-preset"):
        slowest_preset("avc")


def test_probe_qps_include_both_legal_endpoints() -> None:
    qps = probe_qps("av1")
    assert qps[0] == 63
    assert qps[-1] == 1
    assert qps == tuple(dict.fromkeys(qps))


def test_empty_or_wrong_decode_is_rejected() -> None:
    source = (8, 2160, 3840, 3)
    assert "bitstream is empty" in decode_rejections(
        bitstream_bytes=0, source_shape=source, decoded_shape=source
    )
    assert "decode produced no frames" in decode_rejections(
        bitstream_bytes=100, source_shape=source, decoded_shape=None
    )
    assert any("frames" in reason for reason in decode_rejections(
        bitstream_bytes=100, source_shape=source, decoded_shape=(7, 2160, 3840, 3)
    ))
    assert any("decoded" in reason and "x" in reason for reason in decode_rejections(
        bitstream_bytes=100, source_shape=source, decoded_shape=(8, 1080, 1920, 3)
    ))
    assert decode_rejections(
        bitstream_bytes=100, source_shape=source, decoded_shape=source
    ) == []


def test_a_monotone_coarse_to_fine_curve_is_quiet() -> None:
    alarms = monotonicity_alarms(
        qps=(63, 45, 20),
        rates=(1_000.0, 4_000.0, 20_000.0),
        qualities=(15.0, 40.0, 70.0),
        higher_is_better=True,
    )
    assert alarms == []


def test_quality_rising_with_qp_is_an_alarm() -> None:
    alarms = monotonicity_alarms(
        qps=(20, 45, 63),
        rates=(20_000.0, 4_000.0, 1_000.0),
        qualities=(70.0, 80.0, 90.0),
        higher_is_better=True,
    )
    assert any("quality rose" in item for item in alarms)


def test_bounds_file_carries_the_declared_vmaf_bd_rate_band() -> None:
    document = bounds_document()
    band = document["bounds"]["bd_rate_vmaf_percent"]["av1"]
    assert band["low"] == -80.0
    assert band["high"] == 180.0
    assert document["bounds"]["bd_rate_vmaf_percent"]["vvc"] == band
    assert "percent" in document["bounds"]["bd_rate_vmaf_percent"]["basis"]
    vmaf = document["quality_axes"]["vmaf"]
    assert vmaf["direction"] == "higher-is-better"
    assert vmaf["min_curve_span"] == 10.0
    assert document["quality_axes"]["lpips"]["curve_quality_transform"] == "negate"


def test_write_bounds_refuses_a_silent_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "bounds-before-run.json"
    write_bounds(dest)
    with pytest.raises(SystemExit, match="already exists"):
        write_bounds(dest)
    write_bounds(dest, force=True)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["experiment"] == "bp45-low-rate"


def test_the_search_is_staged_not_cartesian() -> None:
    names = stage_names()
    assert names == ("background", "correction", "appearance", "motion", "controls")
    staged = all_points()
    assert len(staged) == sum(len(points_for(name)) for name in names)
    # Four independent families at the sizes in STAGES would explode if crossed.
    cartesian = 1
    for _name, points in STAGES:
        cartesian *= len(points)
    assert cartesian > len(staged)
    assert len({point.name for point in staged}) == len(staged)


def test_ledger_moved_requires_more_than_one_byte_count() -> None:
    frozen = [
        {"pointstream": {"parts": {"panorama": 100, "residual": 10}}},
        {"pointstream": {"parts": {"panorama": 100, "residual": 20}}},
    ]
    assert not ledger_moved(frozen, key="panorama")
    assert ledger_moved(frozen, key="residual")
