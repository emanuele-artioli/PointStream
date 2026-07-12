from __future__ import annotations

from src.shared.profiling import PipelineProfiler, derive_fps_throughput


def test_pipeline_profiler_accumulates_repeated_stage_calls() -> None:
    profiler = PipelineProfiler()
    with profiler.stage("residual"):
        pass
    with profiler.stage("residual"):
        pass

    timings = profiler.get_timings()
    assert "residual" in timings
    assert timings["residual"] >= 0.0


def test_derive_fps_throughput_flat_stage() -> None:
    timings_sec = {"pipeline_total": 2.0, "transport_send": 0.5}
    fps = derive_fps_throughput(timings_sec, num_frames=60)

    assert fps["pipeline_total"] == 30.0
    assert fps["transport_send"] == 120.0


def test_derive_fps_throughput_recurses_into_nested_dicts() -> None:
    timings_sec = {
        "encode_chunk": {
            "total": 4.0,
            "residual": {
                "total": 3.0,
                "genai_baseline": 2.0,
            },
        },
        "decode": {"total": 1.0},
    }
    fps = derive_fps_throughput(timings_sec, num_frames=60)

    assert fps["encode_chunk"]["total"] == 15.0
    assert fps["encode_chunk"]["residual"]["total"] == 20.0
    assert fps["encode_chunk"]["residual"]["genai_baseline"] == 30.0
    assert fps["decode"]["total"] == 60.0


def test_derive_fps_throughput_skips_factor_and_ratio_keys() -> None:
    timings_sec = {
        "encoder_realtime_factor": 3.5,
        "bytes_to_source_ratio": 0.2,
        "total": 2.0,
    }
    fps = derive_fps_throughput(timings_sec, num_frames=60)

    assert "encoder_realtime_factor" not in fps
    assert "bytes_to_source_ratio" not in fps
    assert fps["total"] == 30.0


def test_derive_fps_throughput_handles_zero_or_missing_seconds() -> None:
    timings_sec = {"total": 0.0, "other": None, "quality_evaluation": "n/a"}
    fps = derive_fps_throughput(timings_sec, num_frames=60)

    # Zero-duration and non-numeric stages must not raise a ZeroDivisionError
    # or crash on a bad type -- they simply have no fps figure.
    assert fps["total"] is None
    assert "other" not in fps
    assert "quality_evaluation" not in fps


def test_derive_fps_throughput_drops_empty_nested_dicts() -> None:
    timings_sec = {"encode_chunk": {"encoder_realtime_factor": 1.0}}
    fps = derive_fps_throughput(timings_sec, num_frames=60)

    # The only key inside "encode_chunk" is a skipped ratio key, so the
    # nested dict ends up empty and is omitted entirely rather than leaving
    # a dangling empty dict in the output.
    assert "encode_chunk" not in fps
