from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import benchmark_mask_codecs as bench


def _metric(
    *,
    codec: str,
    run_index: int,
    elapsed: float,
    metadata: int,
    transport: int,
    actual_counts: dict[str, int],
) -> bench.RunMetrics:
    return bench.RunMetrics(
        requested_codec=codec,
        run_index=run_index,
        elapsed_seconds=elapsed,
        metadata_size_bytes=metadata,
        transport_total_size_bytes=transport,
        residual_size_bytes=11,
        panorama_size_bytes=22,
        source_size_bytes=33,
        mask_frame_count=sum(actual_counts.values()),
        actual_codec_counts=actual_counts,
        output_dir=f"/tmp/{codec}_{run_index}",
        chunk_id=f"chunk_{codec}_{run_index}",
    )


def test_helper_functions_basic() -> None:
    assert bench._slugify(" Segmenter Native ") == "segmenter_native"
    assert bench._slugify("***") == "codec"

    assert bench._parse_codecs("auto, rle-v1, , bitpack-z1") == ["auto", "rle-v1", "bitpack-z1"]
    assert bench._coerce_int("12") == 12
    assert bench._coerce_int(None) is None
    assert bench._coerce_int("not-an-int") is None

    assert bench._mean_int([10, None, 14]) == 12
    assert bench._mean_int([None, None]) is None

    assert bench._format_codec_counts({}) == "none"
    assert bench._format_codec_counts({"rle-v1": 4, "bitpack-z1": 10}) == "bitpack-z1:10,rle-v1:4"

    assert bench._human_bytes(None) == "n/a"
    assert bench._human_bytes(100) == "100 B"
    assert bench._human_bytes(2048) == "2.0 KB"
    assert bench._human_bytes(2 * 1024 * 1024) == "2.00 MB"

    assert bench._format_delta(None) == "n/a"
    assert bench._format_delta(1.234) == "+1.23%"


def test_find_default_input_uses_project_root(monkeypatch, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    clip = assets / "real_tennis.mp4"
    clip.write_bytes(b"x")

    monkeypatch.setattr(bench, "PROJECT_ROOT", tmp_path)
    assert bench._find_default_input() == str(clip)

    clip.unlink()
    assert bench._find_default_input() is None


def test_load_summary_and_command_builder(tmp_path: Path) -> None:
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(json.dumps({"metadata_size_bytes": 12}), encoding="utf-8")

    loaded = bench._load_summary(summary_path)
    assert loaded["metadata_size_bytes"] == 12

    with pytest.raises(FileNotFoundError):
        bench._load_summary(tmp_path / "missing.json")

    cmd = bench._build_main_command(
        output_dir=tmp_path / "out",
        chunk_id="abc",
        summary_path=summary_path,
        input_video="/tmp/input.mp4",
        num_frames=24,
        requested_codec="rle-v1",
    )
    assert "--metadata-mask-codec" in cmd
    assert "rle-v1" in cmd
    assert "--input" in cmd


def test_load_mask_codec_counts(monkeypatch, tmp_path: Path) -> None:
    class _Mask:
        def __init__(self, codec: str) -> None:
            self.mask_codec = codec

    class _Actor:
        def __init__(self, codecs: list[str]) -> None:
            self.mask_frames = [_Mask(codec) for codec in codecs]

    class _Payload:
        def __init__(self) -> None:
            self.actors = [_Actor(["poly-v1", "poly-v1"]), _Actor(["rle-v1"])]

    class _Transport:
        def __init__(self, root_dir: Path) -> None:
            self.root_dir = root_dir

        def receive(self, chunk_id: str) -> _Payload:
            assert chunk_id == "chunk_001"
            return _Payload()

    monkeypatch.setattr(bench, "DiskTransport", _Transport)

    num_frames, counts = bench._load_mask_codec_counts(output_dir=tmp_path, chunk_id="chunk_001")
    assert num_frames == 3
    assert counts == {"poly-v1": 2, "rle-v1": 1}


def test_run_single_ablation_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        bench,
        "_load_summary",
        lambda _path: {
            "metadata_size_bytes": 101,
            "transport_total_size_bytes": 202,
            "residual_size_bytes": 303,
            "panorama_size_bytes": 404,
            "source_size_bytes": 505,
        },
    )
    monkeypatch.setattr(bench, "_load_mask_codec_counts", lambda **_kwargs: (7, {"bitpack-z1": 7}))

    perf_values = iter([100.0, 101.75])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(perf_values))

    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr=""),
    )

    metric = bench._run_single_ablation(
        output_dir=tmp_path / "run",
        chunk_id="chunk_ok",
        input_video=None,
        num_frames=8,
        requested_codec="auto",
        run_index=1,
    )

    assert metric.requested_codec == "auto"
    assert metric.mask_frame_count == 7
    assert metric.actual_codec_counts == {"bitpack-z1": 7}
    assert metric.metadata_size_bytes == 101
    assert metric.elapsed_seconds == pytest.approx(1.75)


def test_run_single_ablation_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="stdout sample",
            stderr="stderr sample",
        ),
    )

    with pytest.raises(RuntimeError, match="Ablation run failed"):
        bench._run_single_ablation(
            output_dir=tmp_path / "run",
            chunk_id="chunk_fail",
            input_video=None,
            num_frames=4,
            requested_codec="png",
            run_index=2,
        )


def test_aggregate_and_csv_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        _metric(codec="rle-v1", run_index=1, elapsed=10.0, metadata=400, transport=1000, actual_counts={"rle-v1": 6}),
        _metric(codec="rle-v1", run_index=2, elapsed=14.0, metadata=420, transport=1020, actual_counts={"rle-v1": 6}),
        _metric(codec="auto", run_index=1, elapsed=9.0, metadata=200, transport=800, actual_counts={"bitpack-z1": 6}),
    ]

    aggregate = bench._aggregate_results(rows)
    assert aggregate[0]["requested_codec"] == "auto"
    assert aggregate[1]["requested_codec"] == "rle-v1"

    rle = next(entry for entry in aggregate if entry["requested_codec"] == "rle-v1")
    auto = next(entry for entry in aggregate if entry["requested_codec"] == "auto")
    assert rle["metadata_vs_rle_percent"] == pytest.approx(0.0)
    assert auto["metadata_vs_rle_percent"] is not None
    assert auto["actual_codec_counts"] == {"bitpack-z1": 6}

    per_run_csv = tmp_path / "runs.csv"
    agg_csv = tmp_path / "aggregate.csv"
    bench._write_per_run_csv(path=per_run_csv, rows=rows)
    bench._write_aggregate_csv(path=agg_csv, rows=aggregate)

    assert per_run_csv.exists()
    assert agg_csv.exists()
    assert "requested_codec" in per_run_csv.read_text(encoding="utf-8")
    assert "actual_codec_counts" in agg_csv.read_text(encoding="utf-8")

    bench._print_aggregate_table(aggregate)
    output = capsys.readouterr().out
    assert "Mask codec ablation summary" in output


def test_main_happy_path_with_mocked_runs(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def _fake_run_single_ablation(
        *,
        output_dir: Path,
        chunk_id: str,
        input_video: str | None,
        num_frames: int,
        requested_codec: str,
        run_index: int,
    ) -> bench.RunMetrics:
        _ = chunk_id
        _ = input_video
        _ = num_frames
        calls.append((requested_codec, run_index))
        metadata = 400 if requested_codec == "rle-v1" else 200
        actual = {"rle-v1": 4} if requested_codec == "rle-v1" else {"bitpack-z1": 4}
        return bench.RunMetrics(
            requested_codec=requested_codec,
            run_index=run_index,
            elapsed_seconds=8.0 if requested_codec == "rle-v1" else 7.5,
            metadata_size_bytes=metadata,
            transport_total_size_bytes=1000,
            residual_size_bytes=111,
            panorama_size_bytes=222,
            source_size_bytes=333,
            mask_frame_count=4,
            actual_codec_counts=actual,
            output_dir=str(output_dir),
            chunk_id=f"chunk_{requested_codec}_{run_index}",
        )

    monkeypatch.setattr(bench, "_run_single_ablation", _fake_run_single_ablation)
    monkeypatch.setattr(bench, "_find_default_input", lambda: None)

    output_root = tmp_path / "bench_out"
    exit_code = bench.main(
        [
            "--num-frames",
            "4",
            "--repeats",
            "1",
            "--codecs",
            "rle-v1,auto",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert calls == [("rle-v1", 1), ("auto", 1)]

    created_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(created_dirs) == 1
    run_dir = created_dirs[0]
    assert (run_dir / "mask_codec_ablation_runs.csv").exists()
    assert (run_dir / "mask_codec_ablation_summary.csv").exists()
    summary_json = run_dir / "mask_codec_ablation_summary.json"
    assert summary_json.exists()

    parsed = json.loads(summary_json.read_text(encoding="utf-8"))
    assert parsed["codecs"] == ["rle-v1", "auto"]
    assert len(parsed["runs"]) == 2


def test_main_validates_arguments(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "_find_default_input", lambda: None)

    with pytest.raises(ValueError, match="--num-frames must be a positive integer"):
        bench.main(["--num-frames", "0", "--output-root", str(tmp_path / "a")])

    with pytest.raises(ValueError, match="--repeats must be a positive integer"):
        bench.main(["--repeats", "0", "--output-root", str(tmp_path / "b")])

    with pytest.raises(ValueError, match="--codecs must include at least one codec"):
        bench.main(["--codecs", " , ", "--output-root", str(tmp_path / "c")])


def test_main_raises_for_missing_explicit_input(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "_find_default_input", lambda: None)
    missing_input = tmp_path / "missing_clip.mp4"

    with pytest.raises(FileNotFoundError, match="Input video does not exist"):
        bench.main(
            [
                "--input",
                str(missing_input),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )