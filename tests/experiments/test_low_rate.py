"""Low-rate search helpers: reject bad decodes, require bounds, stay staged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tier.low_rate_bounds import bounds_document, write_bounds
from experiments.tier.low_rate_clips import load_e1_sequence
from experiments.tier.low_rate_measure import (
    TIMING_KEYS,
    last_minus_first,
    primary_preset,
    recorded_slowest_preset,
    reference_request,
    timing_record,
)
from experiments.tier.low_rate_plan import STAGES, all_points, ledger_moved, points_for, stage_names
from experiments.tier.low_rate_references import (
    compare_candidate_to_anchor,
    reference_qps,
    residual_qps_in_plan,
)
from experiments.tier.low_rate_validate import (
    decode_rejections,
    monotonicity_alarms,
    probe_qps,
    slowest_preset,
)
from src.components.codec.measure import PRESETS, TimedRoundtrip


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


def _probe_file(path: Path, *, av1: str = "0", vvc: str = "slower") -> Path:
    path.write_text(
        json.dumps(
            {
                "tools": {
                    "av1": {"selected_preset": av1},
                    "vvc": {"selected_preset": vvc},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_recorded_preset_is_the_probe_not_the_convenience_table(tmp_path: Path) -> None:
    probe = _probe_file(tmp_path / "codec-floor.json")
    assert recorded_slowest_preset("av1", probe) == "0"
    assert recorded_slowest_preset("vvc", probe) == "slower"
    assert recorded_slowest_preset("av1", probe) != PRESETS["av1"]
    assert recorded_slowest_preset("vvc", probe) != PRESETS["vvc"]
    assert primary_preset("av1", probe_path=probe) == "0"
    assert primary_preset("vvc", probe_path=probe) == "slower"


def test_primary_preset_refuses_a_convenience_table_hit(tmp_path: Path) -> None:
    probe = _probe_file(tmp_path / "codec-floor.json", av1=str(PRESETS["av1"]))
    with pytest.raises(ValueError, match="measure.PRESETS"):
        primary_preset("av1", probe_path=probe)


def test_missing_probe_file_is_a_named_exit(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.json"
    with pytest.raises(SystemExit, match="does not exist"):
        recorded_slowest_preset("av1", missing)


def test_reference_qps_are_not_the_residual_knob() -> None:
    residual = residual_qps_in_plan()
    qps = reference_qps("av1")
    assert qps == probe_qps("av1")
    assert qps[0] == 63
    assert qps[-1] == 1
    assert not set(qps) <= residual
    request = reference_request("av1", qp=63, preset="0")
    assert request.rate == 63
    assert request.preset == "0"
    assert 63 not in residual


def test_timing_record_keeps_encode_and_decode_apart() -> None:
    trip = TimedRoundtrip(
        size_bytes=100,
        frames=None,  # type: ignore[arg-type]
        encode_seconds=1.25,
        decode_seconds=0.5,
        tool_path="/opt/local/bin/SvtAv1EncApp",
        tool_version="SVT-AV1 v1.8.0",
        preset="0",
        qp=63,
    )
    record = timing_record(trip)
    assert tuple(record) == TIMING_KEYS
    assert record["encode_seconds"] == 1.25
    assert record["decode_seconds"] == 0.5
    assert record["encode_seconds"] != record["decode_seconds"]


def test_late_frame_delta_is_last_minus_first() -> None:
    assert last_minus_first((40.0, 38.0, 30.0)) == -10.0
    with pytest.raises(ValueError, match="at least two frames"):
        last_minus_first((40.0,))


def test_e1_refuses_the_eight_frame_bp21_windows() -> None:
    with pytest.raises(SystemExit, match="at least 48"):
        load_e1_sequence("alcaraz_highlights", ["scene_000"], n_frames=8)


def test_e1_sequence_calls_the_long_scene_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str, int]] = []

    class _Shape:
        def __init__(self) -> None:
            self.shape = (48, 2160, 3840, 3)

    class FakeClip:
        frames = _Shape()
        objects: tuple[object, ...] = ()

    def fake_load(video: str, scene: str, n_frames: int) -> FakeClip:
        seen.append((video, scene, n_frames))
        return FakeClip()

    monkeypatch.setattr("experiments.tier.low_rate_clips._load_bp46", fake_load)
    clips = load_e1_sequence(
        "alcaraz_highlights", ["scene_000", "scene_028"], n_frames=48
    )
    assert seen == [
        ("alcaraz_highlights", "scene_000", 48),
        ("alcaraz_highlights", "scene_028", 48),
    ]
    assert len(clips) == 2


def test_sweep_does_not_slave_anchors_to_residual_or_convenience_presets() -> None:
    root = Path(__file__).resolve().parents[2]
    sweep = (root / "experiments" / "tier" / "low_rate_sweep.py").read_text(encoding="utf-8")
    clips = (root / "experiments" / "tier" / "low_rate_clips.py").read_text(encoding="utf-8")
    refs = (root / "experiments" / "tier" / "low_rate_references.py").read_text(encoding="utf-8")
    assert "PRESETS" not in sweep
    assert "load_scene_sequence" not in sweep
    assert "anchor_over_sequence" not in sweep
    assert "residual_qp or 45" not in sweep
    assert "from experiments.tier.ladder_scenes" not in sweep
    assert "from experiments.tier.clip" not in clips
    assert "from src.components.codec.measure import PRESETS" not in refs


def test_non_overlapping_vmaf_uses_the_floor_not_a_fake_bd_rate() -> None:
    anchor = [
        {"bytes": 43_865, "usable": True, "scores": {"vmaf": 86.53}, "qp": 63},
        {"bytes": 200_000, "usable": True, "scores": {"vmaf": 97.0}, "qp": 45},
    ]
    cheaper_and_worse = [
        {"bytes": 2_000, "usable": True, "scores": {"vmaf": 10.0}},
        {"bytes": 8_000, "usable": True, "scores": {"vmaf": 20.0}},
    ]
    report = compare_candidate_to_anchor(cheaper_and_worse, anchor)
    assert report["bd_rate_percent"] is None
    assert report["meets_or_beats_floor"] is False
    assert report["anchor_floor"]["bytes"] == 43_865

    cheaper_and_better = [
        {"bytes": 2_000, "usable": True, "scores": {"vmaf": 90.0}},
        {"bytes": 8_000, "usable": True, "scores": {"vmaf": 97.0}},
    ]
    win = compare_candidate_to_anchor(cheaper_and_better, anchor)
    assert win["bd_rate_percent"] is None
    assert win["meets_or_beats_floor"] is True


def test_overlapping_vmaf_reports_bd_rate() -> None:
    anchor = [
        {"bytes": 2_000, "usable": True, "scores": {"vmaf": 20.0}},
        {"bytes": 8_000, "usable": True, "scores": {"vmaf": 40.0}},
    ]
    half = [
        {"bytes": 1_000, "usable": True, "scores": {"vmaf": 20.0}},
        {"bytes": 4_000, "usable": True, "scores": {"vmaf": 40.0}},
    ]
    report = compare_candidate_to_anchor(half, anchor)
    assert report["bd_rate_percent"] == pytest.approx(-50.0, rel=1e-6)
    assert report["reason"] is None
