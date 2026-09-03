"""Low-rate search helpers: reject bad decodes, require bounds, stay staged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tier.low_rate_bounds import bounds_document, write_bounds
from experiments.tier.low_rate_checkpoint import load_checkpoint, save_checkpoint, write_json
from experiments.tier.low_rate_clips import load_e1_sequence
from experiments.tier.low_rate_measure import (
    TIMING_KEYS,
    last_minus_first,
    primary_preset,
    recorded_slowest_preset,
    reference_request,
    timing_record,
)
from experiments.tier.low_rate_plan import (
    STAGES,
    all_points,
    ledger_moved,
    named_point,
    points_for,
    select_work,
    stage_names,
)
from experiments.tier.low_rate_references import (
    compare_candidate_to_anchor,
    guard_reference_checkpoints,
    reference_checkpoint_identity,
    reference_qps,
    residual_qps_in_plan,
    selected_reference_qps,
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


def test_named_point_is_the_unique_operating_point() -> None:
    point = named_point("bg-crf51")
    assert point.stage == "background"
    assert point.stream_crf == 51
    assert point.name == "bg-crf51"


def test_bp52_background_walk_has_only_the_requested_crfs() -> None:
    from experiments.tier.bp52_background_search import POINT_NAMES

    points = [named_point(name) for name in POINT_NAMES]
    assert [point.stream_crf for point in points] == [51, 63, 57]
    assert all(
        point.residual_qp is None
        and not point.residual_on
        and point.appearance_jpeg_quality == 40
        and point.appearance_downscale == 2
        and point.motion_max_points == 16
        and point.object_stream_on
        for point in points
    )


def test_unknown_point_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown point"):
        named_point("bg-crf99")


def test_select_work_one_point_is_a_single_row() -> None:
    work = select_work(point="bg-crf51")
    assert len(work) == 1
    stage, points = work[0]
    assert stage == "background"
    assert len(points) == 1
    assert points[0].name == "bg-crf51"


def test_select_work_rejects_a_point_from_the_wrong_stage() -> None:
    with pytest.raises(ValueError, match="belongs to stage"):
        select_work(stage="motion", point="bg-crf51")


def test_late_frame_is_per_scene_not_across_the_join() -> None:
    from types import SimpleNamespace

    from experiments.tier.low_rate_measure import late_frame_by_scene, y_psnr

    ref0 = np.full((2, 8, 8, 3), 100, dtype=np.uint8)
    ref1 = np.full((2, 8, 8, 3), 100, dtype=np.uint8)
    pred0 = np.stack(
        [np.full((8, 8, 3), 90, dtype=np.uint8), np.full((8, 8, 3), 80, dtype=np.uint8)]
    )
    pred1 = np.stack(
        [np.full((8, 8, 3), 70, dtype=np.uint8), np.full((8, 8, 3), 50, dtype=np.uint8)]
    )
    clips = [
        SimpleNamespace(frames=ref0, video="v", scene="scene_a", context_id="c"),
        SimpleNamespace(frames=ref1, video="v", scene="scene_b", context_id="c"),
    ]
    source = np.concatenate([ref0, ref1], axis=0)
    pred = np.concatenate([pred0, pred1], axis=0)
    reports = late_frame_by_scene(clips, source, pred)
    assert reports[0]["scene"] == "scene_a"
    assert reports[0]["psnr_y_last_minus_first"] == pytest.approx(
        y_psnr(ref0[1], pred0[1]) - y_psnr(ref0[0], pred0[0])
    )
    assert reports[1]["psnr_y_last_minus_first"] == pytest.approx(
        y_psnr(ref1[1], pred1[1]) - y_psnr(ref1[0], pred1[0])
    )
    joined = y_psnr(ref1[1], pred1[1]) - y_psnr(ref0[0], pred0[0])
    assert reports[0]["psnr_y_last_minus_first"] != pytest.approx(joined)
    assert reports[1]["psnr_y_last_minus_first"] != pytest.approx(joined)


def test_late_frame_bound_reads_the_bp45_band() -> None:
    from experiments.tier.low_rate_bounds import bounds_document
    from experiments.tier.low_rate_measure import late_frame_bound_alarms

    bounds = bounds_document()
    held = late_frame_bound_alarms(
        [{"video": "v", "scene": "s", "vmaf_last_minus_first": -12.0, "psnr_y_last_minus_first": -1.8}],
        bounds,
    )
    assert held == []
    alarms = late_frame_bound_alarms(
        [{"video": "v", "scene": "s", "vmaf_last_minus_first": -40.0, "psnr_y_last_minus_first": -1.8}],
        bounds,
    )
    assert any("VMAF" in item for item in alarms)


def test_background_stream_provenance_is_libaom_not_svt() -> None:
    from experiments.tier.low_rate_measure import stream_codec_provenance

    spec = stream_codec_provenance("av1")
    assert spec["encoder"] == "libaom-av1"
    assert "-cpu-used" in spec["low_delay"]
    assert "8" in spec["low_delay"]
    assert "realtime" in spec["low_delay"]


def test_chunk_checkpoint_refuses_a_gap(tmp_path: Path) -> None:
    from src.runner.chunk_checkpoint import completed_indices

    (tmp_path / "chunk_01").mkdir()
    (tmp_path / "chunk_01" / "done").write_text("1\n")
    with pytest.raises(SystemExit, match="gap"):
        completed_indices(tmp_path)


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


def test_selected_reference_qps_stay_on_the_independent_walk() -> None:
    walk = reference_qps("av1")
    assert selected_reference_qps("av1", None) == walk
    assert selected_reference_qps("av1", [walk[0]]) == (walk[0],)
    assert 62 not in walk
    assert selected_reference_qps("av1", [63, 62, 63]) == (63, 62)
    with pytest.raises(SystemExit, match="legal range"):
        selected_reference_qps("av1", [0])
    with pytest.raises(SystemExit, match="legal range"):
        selected_reference_qps("av1", [999])


def test_reference_checkpoints_reuse_when_only_qps_change(tmp_path: Path) -> None:
    identity = reference_checkpoint_identity(
        {"video": "alcaraz_highlights", "codec": "av1", "implementation": "v1"},
        "0",
    )
    save_checkpoint(tmp_path, "continuous-qp63", {"bytes": 109198, "qp": 63})
    write_json(
        tmp_path / "identity.json",
        {
            "fingerprint": "stale-because-qps-were-in-the-key",
            "identity": {**identity, "qps": [63]},
        },
    )
    guard_reference_checkpoints(tmp_path, identity)
    resumed = load_checkpoint(tmp_path, "continuous-qp63")
    assert resumed is not None and resumed["bytes"] == 109198
    guard_reference_checkpoints(tmp_path, identity)
    with pytest.raises(SystemExit, match="identity changed"):
        guard_reference_checkpoints(
            tmp_path,
            reference_checkpoint_identity(identity["input"], "10"),
        )


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
        context_id = "alcaraz_highlights_main_court"

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
    assert "context_ids" in sweep
    assert "canonical" in sweep
    assert "run_seconds" in sweep
    assert "encode_seconds is reserved" in (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "tier"
        / "low_rate_measure.py"
    ).read_text(encoding="utf-8")


def test_reference_path_includes_scenes_and_duration() -> None:
    from experiments.tier.low_rate_identity import (
        assert_same_input,
        identity_slug,
        input_identity,
        references_path,
    )

    identity = input_identity(
        video="alcaraz_highlights",
        scenes=("scene_000", "scene_028"),
        frames_per_scene=48,
        codec="av1",
    )
    path = references_path(identity, root=Path("/tmp"))
    assert "scene_000+scene_028" in path.name
    assert "n48" in path.name
    assert "av1" in path.name
    assert identity_slug(identity) in path.name
    other = dict(identity)
    other["frames_per_scene"] = 96
    with pytest.raises(SystemExit, match="frames_per_scene"):
        assert_same_input(identity, other)
    other = dict(identity)
    other["scenes"] = ["scene_000"]
    with pytest.raises(SystemExit, match="scenes"):
        assert_same_input(identity, other)


def test_load_reference_curve_refuses_a_duration_mismatch(tmp_path: Path) -> None:
    from experiments.tier.low_rate_identity import input_identity
    from experiments.tier.low_rate_references import load_reference_curve

    written = input_identity(
        video="alcaraz_highlights",
        scenes=("scene_000", "scene_028"),
        frames_per_scene=48,
        codec="av1",
    )
    expected = dict(written)
    expected["frames_per_scene"] = 96
    payload = {
        "input": written,
        "curve": {"access_patterns": {"continuous": [{"bytes": 1, "usable": True}]}},
    }
    dest = tmp_path / "references.json"
    dest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="frames_per_scene"):
        load_reference_curve(dest, access_pattern="continuous", expected=expected)


def test_checkpoint_round_trips_and_skips_missing(tmp_path: Path) -> None:
    from experiments.tier.low_rate_checkpoint import load_checkpoint, save_checkpoint

    save_checkpoint(tmp_path, "bg-crf51", {"name": "bg-crf51", "bytes": 12})
    loaded = load_checkpoint(tmp_path, "bg-crf51")
    assert loaded is not None
    assert loaded["bytes"] == 12
    assert load_checkpoint(tmp_path, "absent") is None


def test_pointstream_timing_does_not_call_the_wall_encode() -> None:
    from experiments.tier.low_rate_measure import pointstream_timing

    record = pointstream_timing(12.5)
    assert record["run_seconds"] == 12.5
    assert record["encode_seconds"] is None
    assert record["decode_seconds"] is None
    assert "run_seconds" in record["timing_note"]


def test_missing_canvas_field_is_a_named_exit() -> None:
    from dataclasses import dataclass

    from experiments.tier.low_rate_canvas import with_canonical_background

    @dataclass(frozen=True)
    class NoCanvas:
        method: str = "panorama-stream"
        stream_codec: str = "av1"
        stream_crf: int = 45
        keyframe_interval: int = 0
        reference_mode: str = "last"

    with pytest.raises(SystemExit, match="canvas"):
        with_canonical_background(
            NoCanvas(),
            method="panorama-stream",
            stream_codec="av1",
            stream_crf=45,
            context_id="court",
        )


def test_canonical_canvas_and_context_ids_are_set() -> None:
    from dataclasses import dataclass

    from experiments.tier.low_rate_canvas import (
        clip_context_ids,
        require_run_accepts_context_ids,
        with_canonical_background,
    )

    @dataclass(frozen=True)
    class CanvasBg:
        method: str = "panorama-stream"
        stream_codec: str = "av1"
        stream_crf: int = 38
        keyframe_interval: int = 0
        reference_mode: str = "last"
        canvas: str = "independent"
        context_id: str = ""

    tuned = with_canonical_background(
        CanvasBg(),
        method="panorama-stream",
        stream_codec="av1",
        stream_crf=45,
        context_id="alcaraz_highlights_main_court",
    )
    assert tuned.canvas == "canonical"
    assert tuned.context_id == "alcaraz_highlights_main_court"

    class Clip:
        video = "alcaraz_highlights"
        scene = "scene_000"
        context_id = "alcaraz_highlights_main_court"

    assert clip_context_ids([Clip(), Clip()]) == (
        "alcaraz_highlights_main_court",
        "alcaraz_highlights_main_court",
    )

    def stub_run(*, context_ids: tuple[str, ...] | None = None) -> None:
        del context_ids

    require_run_accepts_context_ids(stub_run)

    def old_run() -> None:
        return None

    with pytest.raises(SystemExit, match="context_ids"):
        require_run_accepts_context_ids(old_run)


def test_clip_without_context_id_is_refused() -> None:
    from experiments.tier.low_rate_canvas import clip_context_ids

    class Clip:
        video = "alcaraz_highlights"
        scene = "scene_000"
        context_id = ""

    with pytest.raises(SystemExit, match="context_id"):
        clip_context_ids([Clip()])


def test_aligned_fallback_matches_the_reference_request() -> None:
    from experiments.tier.low_rate_fallback import (
        aligned_fallback_request,
        evaluate_fallback_equivalence,
    )
    from src.contracts.config import FallbackConfig

    default = FallbackConfig()
    ref = reference_request("av1", 63, "0")
    assert default.encode_request().rate != ref.rate
    aligned = aligned_fallback_request(default, codec="av1", qp=63, preset="0")
    assert aligned.codec_name == ref.codec_name
    assert aligned.rate == ref.rate
    assert aligned.preset == ref.preset
    assert aligned.rate_control == ref.rate_control
    held = evaluate_fallback_equivalence(
        {"bytes": 1000, "scores": {"vmaf": 20.0}},
        {"bytes": 1000, "scores": {"vmaf": 20.0}},
    )
    assert held["held"] is True
    assert held["rate_rel"] == pytest.approx(1.0)
    miss = evaluate_fallback_equivalence(
        {"bytes": 2000, "scores": {"vmaf": 10.0}},
        {"bytes": 1000, "scores": {"vmaf": 20.0}},
    )
    assert miss["held"] is False
    assert miss["rate_ok"] is False


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
