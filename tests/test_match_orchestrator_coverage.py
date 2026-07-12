"""Fast, mocked coverage of src/encoder/match_orchestrator.py's orchestration
logic (scene routing dispatch, frame-cursor accumulation, byte aggregation).

The real end-to-end path (real detector/pose/segmenter, real ffmpeg, real
outcome-safe comparison) is covered separately by the slow, real
`tests/test_match_orchestrator_integration.py`. This file exists because
that real test is marked `integration`/`slow` and is excluded from the
default coverage-gate run (pytest.ini), so match_orchestrator.py's
orchestration branches need fast, hermetic coverage too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.encoder import match_orchestrator as mo
from src.encoder.video_io import encode_video_frames_ffmpeg
from src.shared.config import PointstreamConfig
from src.shared.schemas import SceneClass, SceneSpan


class _FakeEncoderPipeline:
    instances: list["_FakeEncoderPipeline"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.shutdown_called = False
        self.encoded_chunks: list[Any] = []
        self.scene_contexts: list[Any] = []
        _FakeEncoderPipeline.instances.append(self)

    def set_scene_context(self, scene_key: Any) -> None:
        self.scene_contexts.append(scene_key)

    def encode_chunk(self, chunk: Any) -> Any:
        self.encoded_chunks.append(chunk)
        return SimpleNamespace(
            chunk=SimpleNamespace(chunk_id=chunk.chunk_id),
            panorama=SimpleNamespace(panorama_uri=None),
            residual=SimpleNamespace(residual_video_uri=None),
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeDiskTransport:
    def __init__(self, root_dir: str | Path, config: Any = None) -> None:
        self.root_dir = Path(root_dir)
        self._payloads: dict[str, Any] = {}

    def send(self, payload: Any) -> None:
        chunk_dir = self.root_dir / f"chunk_{payload.chunk.chunk_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "metadata.msgpack").write_bytes(b"M" * 300)
        residual_path = chunk_dir / "residual.mp4"
        residual_path.write_bytes(b"R" * 400)
        payload.residual.residual_video_uri = str(residual_path)
        self._payloads[payload.chunk.chunk_id] = payload

    def receive(self, chunk_id: str) -> Any:
        return self._payloads[chunk_id]


class _FakeDecoderRenderer:
    def __init__(self, output_root: Any = None, config: Any = None) -> None:
        self.output_root = output_root

    def process(self, payload: Any) -> Any:
        return SimpleNamespace(output_uri=f"fake://{payload.chunk.chunk_id}")


def _alternating_fallback_encode(sizes: list[int]):
    """Deterministic fallback-encode fake: cycles through `sizes` so both
    outcome-safe routing branches (semantic wins / fallback wins) are
    exercised against the fixed 700-byte fake semantic payload (300 + 400,
    see `_FakeDiskTransport.send`)."""
    counter = {"i": 0}

    def _fake(clip_path: Path, output_path: Path, config: PointstreamConfig) -> tuple[int, float]:
        size = sizes[counter["i"] % len(sizes)]
        counter["i"] += 1
        # _cached_fallback_encode's wrapper stats the real file afterward
        # (to also support cache-hit copies), so this fake must actually
        # write it, not just return a byte count.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00" * size)
        return size, 0.001

    return _fake


@pytest.fixture(autouse=True)
def _patch_pipeline_components(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEncoderPipeline.instances.clear()
    monkeypatch.setattr(mo, "EncoderPipeline", _FakeEncoderPipeline)
    monkeypatch.setattr(mo, "DiskTransport", _FakeDiskTransport)
    monkeypatch.setattr(mo, "DecoderRenderer", _FakeDecoderRenderer)
    monkeypatch.setattr(mo, "build_execution_pool", lambda config: None)
    monkeypatch.setattr(mo, "build_actor_extractor", lambda config: None)
    monkeypatch.setattr(mo, "build_ball_extractor", lambda config: None)
    monkeypatch.setattr(mo, "build_reference_extractor", lambda config: None)
    monkeypatch.setattr(mo, "build_residual_calculator", lambda config: None)
    monkeypatch.setattr(mo, "extract_scene_scores", lambda video_path, cache_file: (video_path, cache_file))


@pytest.fixture()
def tiny_source_video(tmp_path: Path) -> Path:
    """A real (tiny, low-res) 7.0s @ 10fps synthetic clip, so probe/extract
    stay real while everything ML/transport-heavy is faked above."""
    path = tmp_path / "tiny_source.mp4"
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(70)]
    encode_video_frames_ffmpeg(
        path, frames, fps=10.0, width=64, height=64, codec="libx264", crf=18, preset="veryfast"
    )
    return path


def _fixed_three_scene_split(video_path: str, cache_file: str) -> list[SceneSpan]:
    return [
        SceneSpan(t_start=0.0, t_end=4.0, scene_class=SceneClass.POINT, confidence=0.9, avg_score=0.001, std_score=0.001, max_score=0.001),
        SceneSpan(t_start=4.0, t_end=6.0, scene_class=SceneClass.INTERLUDE, confidence=0.9, avg_score=0.5, std_score=0.1, max_score=0.6),
        SceneSpan(t_start=6.0, t_end=7.0, scene_class=SceneClass.POINT, confidence=0.9, avg_score=0.001, std_score=0.001, max_score=0.001),
    ]


class TestEncodeFullMatch:
    def test_routes_point_and_interlude_scenes_and_aggregates_totals(
        self, monkeypatch: pytest.MonkeyPatch, tiny_source_video: Path, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(mo, "classify_video_scenes", _fixed_three_scene_split)
        # 3 point sub-chunks total (scene0: [0,2),[2,4); scene2: [6,7)).
        # Alternate fallback size around the fixed 700-byte fake semantic
        # payload so both routing outcomes occur.
        monkeypatch.setattr(mo, "_fallback_encode", _alternating_fallback_encode([100, 2000, 100]))

        config = PointstreamConfig(
            source_uri=str(tiny_source_video),
            run_mode="full_match",
            scene_chunk_duration_sec=2.0,
            ffmpeg_codec="libsvtav1",
            codec_crf=35,
            codec_preset="fast",
        )

        summary = mo.encode_full_match(
            config=config,
            video_path=tiny_source_video,
            transport_root=tmp_path / "transport",
            scene_cache_root=tmp_path / "scene_cache",
            anchor_cache_root=tmp_path / "anchor_cache",
        )

        assert summary["totals"]["num_scenes"] == 3
        assert summary["totals"]["num_point_scenes"] == 2
        assert summary["totals"]["num_fallback_scenes"] == 1
        assert summary["totals"]["num_point_subchunks"] == 3
        # Point sub-chunks' fallback sizes cycle [100, 2000, 100] against the
        # fixed 700-byte fake semantic payload: 100 < 700 -> fallback (smaller)
        # wins; 2000 > 700 -> semantic (smaller) wins. So 2 of 3 route fallback.
        assert summary["totals"]["num_point_subchunks_routed_to_fallback"] == 2
        assert summary["totals"]["source_bytes"] > 0
        assert summary["totals"]["total_bytes"] > 0
        assert summary["totals"]["bytes_to_source_ratio"] is not None

        scenes = summary["scenes"]
        assert scenes[0]["scene_class"] == SceneClass.POINT.value
        assert scenes[0]["routing_summary"] == "mixed"  # 100(semantic-win), 2000(fallback-win)
        assert len(scenes[0]["sub_chunks"]) == 2

        assert scenes[1]["scene_class"] == SceneClass.INTERLUDE.value
        assert scenes[1]["routing_summary"] == "fallback"
        assert scenes[1]["sub_chunks"] is None
        assert scenes[1]["bytes"] == 100  # third cycled fallback size

        assert scenes[2]["scene_class"] == SceneClass.POINT.value
        assert len(scenes[2]["sub_chunks"]) == 1

        # Frame cursor must accumulate monotonically with no overlap.
        chunk_ids = [sc["chunk_id"] for s in scenes if s["sub_chunks"] for sc in s["sub_chunks"]]
        assert chunk_ids == ["s0000c0000", "s0000c0001", "s0002c0000"]

        assert _FakeEncoderPipeline.instances[0].shutdown_called is True

        # Report 10 Phase 3: realtime factor fields present and sane.
        timings = summary["timings_sec"]
        assert timings["encode_total"] > 0
        assert timings["decode_total"] > 0
        assert timings["encoder_realtime_factor"] == pytest.approx(
            timings["encode_total"] / summary["duration_sec"]
        )
        assert timings["decoder_realtime_factor"] == pytest.approx(
            timings["decode_total"] / summary["duration_sec"]
        )

        # Report 10 Phase 5.1(b): fps_throughput mirrors timings_sec (skipping
        # the realtime-factor ratio keys) and the full resolved config is
        # echoed so the run_summary alone is self-describing.
        fps_throughput = summary["fps_throughput"]
        assert fps_throughput["encode_total"] == pytest.approx(
            summary["num_frames_total"] / timings["encode_total"]
        )
        assert fps_throughput["decode_total"] == pytest.approx(
            summary["num_frames_total"] / timings["decode_total"]
        )
        assert "encoder_realtime_factor" not in fps_throughput
        assert "decoder_realtime_factor" not in fps_throughput

        assert summary["config"]["source_uri"] == str(tiny_source_video)
        assert summary["config"]["run_mode"] == "full_match"
        assert summary["config"]["scene_chunk_duration_sec"] == 2.0
        assert summary["config"]["ffmpeg_codec"] == "libsvtav1"

        # Report 10 Phase 5.1(e): set_scene_context() is called once per
        # POINT scene (scenes 0 and 2 here; the interlude scene at index 1
        # never touches EncoderPipeline at all) with that scene's own index,
        # so the panorama cache resets exactly at scene boundaries.
        assert _FakeEncoderPipeline.instances[0].scene_contexts == [0, 2]

        # First run on a fresh anchor cache: nothing was cached yet.
        assert scenes[1]["cache_hit"] is False
        for s in (scenes[0], scenes[2]):
            for sub_chunk in s["sub_chunks"]:
                assert sub_chunk["fallback_cache_hit"] is False

        # Regression (found via a real run against djokovic_federer.mp4,
        # report 10 Phase 4 G1): every constructed VideoChunk must use
        # start_frame_id=0, never the match-global frame_cursor. Each
        # extracted clip is a self-contained per-chunk file, but
        # ResidualCalculator._process_residuals seeks start_frame_id frames
        # into chunk.source_uri before reading -- a nonzero, growing
        # frame_cursor there runs past a small clip's own frame count and
        # raises "ResidualCalculator received zero valid frames" on any
        # scene beyond the first. global_start_frame in the sub-chunk
        # result dict carries the real match-position bookkeeping instead.
        encoded_chunks = _FakeEncoderPipeline.instances[0].encoded_chunks
        assert len(encoded_chunks) == 3
        assert all(c.start_frame_id == 0 for c in encoded_chunks)
        global_starts = [sc["global_start_frame"] for s in scenes if s["sub_chunks"] for sc in s["sub_chunks"]]
        assert global_starts == sorted(global_starts)  # monotonically non-decreasing

        # Background-layer ladder rung 2 (report 10 Phase 5.3): every
        # sub-chunk of the same scene must carry the same scene_id (so
        # EncoderPipeline's panorama-delta state groups them), and different
        # scenes must get different scene_ids. scene0 has 2 sub-chunks,
        # scene2 has 1.
        assert [c.scene_id for c in encoded_chunks] == ["scene0000", "scene0000", "scene0002"]

    def test_second_run_on_same_video_hits_the_anchor_cache(
        self, monkeypatch: pytest.MonkeyPatch, tiny_source_video: Path, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(mo, "classify_video_scenes", _fixed_three_scene_split)
        monkeypatch.setattr(mo, "_fallback_encode", _alternating_fallback_encode([100, 2000, 100]))

        config = PointstreamConfig(source_uri=str(tiny_source_video), run_mode="full_match")
        shared_anchor_cache = tmp_path / "anchor_cache"

        first = mo.encode_full_match(
            config=config,
            video_path=tiny_source_video,
            transport_root=tmp_path / "transport_run1",
            scene_cache_root=tmp_path / "scene_cache",
            anchor_cache_root=shared_anchor_cache,
        )
        second = mo.encode_full_match(
            config=config,
            video_path=tiny_source_video,
            transport_root=tmp_path / "transport_run2",
            scene_cache_root=tmp_path / "scene_cache",
            anchor_cache_root=shared_anchor_cache,
        )

        assert first["scenes"][1]["cache_hit"] is False
        assert second["scenes"][1]["cache_hit"] is True
        for sub_chunk in second["scenes"][0]["sub_chunks"]:
            assert sub_chunk["fallback_cache_hit"] is True
        # Bytes must be identical across runs since the same cached anchors
        # were reused for the fallback side of the comparison.
        assert first["totals"]["total_bytes"] == second["totals"]["total_bytes"]

    def test_raises_when_classification_yields_no_scenes(
        self, monkeypatch: pytest.MonkeyPatch, tiny_source_video: Path, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(mo, "classify_video_scenes", lambda video_path, cache_file: [])

        config = PointstreamConfig(source_uri=str(tiny_source_video), run_mode="full_match")
        with pytest.raises(ValueError, match="no scenes"):
            mo.encode_full_match(
                config=config,
                video_path=tiny_source_video,
                transport_root=tmp_path / "transport",
                scene_cache_root=tmp_path / "scene_cache",
            anchor_cache_root=tmp_path / "anchor_cache",
            )

    def test_raises_when_scenes_do_not_tile_video_duration(
        self, monkeypatch: pytest.MonkeyPatch, tiny_source_video: Path, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            mo,
            "classify_video_scenes",
            lambda video_path, cache_file: [
                SceneSpan(t_start=0.0, t_end=1.0, scene_class=SceneClass.POINT, confidence=0.9, avg_score=0.0, std_score=0.0, max_score=0.0)
            ],
        )

        config = PointstreamConfig(source_uri=str(tiny_source_video), run_mode="full_match")
        with pytest.raises(ValueError, match="video duration"):
            mo.encode_full_match(
                config=config,
                video_path=tiny_source_video,
                transport_root=tmp_path / "transport",
                scene_cache_root=tmp_path / "scene_cache",
            anchor_cache_root=tmp_path / "anchor_cache",
            )


class TestSceneCachePaths:
    def test_default_cache_root_uses_video_stem(self, tmp_path: Path) -> None:
        dataset_dir, cache_file = mo.scene_cache_paths(tmp_path, Path("/videos/my_match.mp4"))
        assert dataset_dir == tmp_path / "my_match"
        assert cache_file == tmp_path / "my_match" / "scene_scores.csv"


class TestTransportTotalBytes:
    def test_sums_existing_components_and_skips_missing(self, tmp_path: Path) -> None:
        chunk_dir = tmp_path / "chunk_x"
        chunk_dir.mkdir()
        (chunk_dir / "metadata.msgpack").write_bytes(b"a" * 10)
        residual_path = chunk_dir / "residual.mp4"
        residual_path.write_bytes(b"b" * 20)

        payload = SimpleNamespace(
            residual=SimpleNamespace(residual_video_uri=str(residual_path)),
            panorama=SimpleNamespace(panorama_uri=None),
        )
        assert mo._transport_total_bytes(chunk_dir, payload) == 30

    def test_includes_panorama_and_actor_references(self, tmp_path: Path) -> None:
        chunk_dir = tmp_path / "chunk_y"
        chunk_dir.mkdir()
        (chunk_dir / "metadata.msgpack").write_bytes(b"a" * 5)
        residual_path = chunk_dir / "residual.mp4"
        residual_path.write_bytes(b"b" * 5)
        panorama_path = tmp_path / "panorama.jpg"
        panorama_path.write_bytes(b"c" * 5)
        actor_refs_dir = chunk_dir / "actor_references"
        actor_refs_dir.mkdir()
        (actor_refs_dir / "ref_0.jpg").write_bytes(b"d" * 5)

        payload = SimpleNamespace(
            residual=SimpleNamespace(residual_video_uri=str(residual_path)),
            panorama=SimpleNamespace(panorama_uri=str(panorama_path)),
        )
        assert mo._transport_total_bytes(chunk_dir, payload) == 20


class TestSafeFileSize:
    def test_none_path_returns_none(self) -> None:
        assert mo._safe_file_size(None) is None

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        assert mo._safe_file_size(tmp_path / "missing.bin") is None

    def test_directory_path_returns_none(self, tmp_path: Path) -> None:
        assert mo._safe_file_size(tmp_path) is None

    def test_existing_file_returns_size(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"x" * 42)
        assert mo._safe_file_size(f) == 42


class TestProcessFallbackSceneZeroFrames:
    def test_zero_frame_clip_is_skipped_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(mo, "_extract_scene_clip", lambda *a, **k: None)

        class _ZeroFrameMetadata:
            fps = 10.0
            num_frames = 0
            width = 64
            height = 64

        monkeypatch.setattr(mo, "probe_video_metadata", lambda _path: _ZeroFrameMetadata())

        scene = SceneSpan(t_start=0.0, t_end=0.05, scene_class=SceneClass.BLANK, confidence=1.0, avg_score=0.0, std_score=0.0, max_score=0.0)
        result, frames_used = mo._process_fallback_scene(
            video_path=Path("/dummy.mp4"),
            scene_idx=0,
            scene=scene,
            clips_dir=tmp_path,
            config=PointstreamConfig(),
            anchor_cache_root=tmp_path / "anchor_cache",
        )

        assert frames_used == 0
        assert result["bytes"] == 0
        assert result["skipped_zero_frames"] is True
