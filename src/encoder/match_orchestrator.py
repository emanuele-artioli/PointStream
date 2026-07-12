"""Full-match orchestrator: scene routing + per-scene encode (report 10 Phase 2).

Splits a full raw_4k match into scenes via
`src.shared.scene_classification` (deterministic, cached, shared across all
model tiers per report 10's "Deterministic shared segmentation" design
rule), then routes each scene:

- ``SceneClass.POINT`` scenes are split into ``scene_chunk_duration_sec``
  sub-chunks and run through the existing chunked semantic pipeline
  (``EncoderPipeline`` / ``DiskTransport`` / ``DecoderRenderer``, built once
  and reused across sub-chunks via `src.encoder.pipeline_builders` so model
  weights are not reloaded per chunk).
- All other scenes (interlude/other/blank) go straight to the fallback
  codec (the pipeline's own `ffmpeg-codec`/`codec-crf`/`codec-preset`); no
  semantic pipeline is attempted, so no GPU work is wasted on them.

Routing is outcome-safe (report 10 methodology decision 5): even a point
sub-chunk's semantic payload is compared against a fallback-codec encode of
the exact same span, and whichever is smaller is what actually counts
toward the match's transmitted bytes — the Residual Guarantee extended to
routing, capping any misclassification's damage at zero.

Scope note: this module produces byte/timing accounting only (report 10's
"G1 plumbing" goal) — no PSNR/SSIM/VMAF/LPIPS/FVD here. Quality scoring is
G2 (Phase 4), once the held-out-video generative models exist.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from src.encoder import anchor_cache
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.pipeline_builders import (
    build_actor_extractor,
    build_ball_extractor,
    build_execution_pool,
    build_reference_extractor,
    build_residual_calculator,
)
from src.encoder.video_io import (
    encode_video_frames_ffmpeg,
    iter_video_frames_ffmpeg,
    probe_video_metadata,
)
from src.decoder.decoder_renderer import DecoderRenderer
from src.shared.config import PointstreamConfig
from src.shared.scene_classification import classify_video_scenes, extract_scene_scores
from src.shared.schemas import SceneClass, SceneSpan, VideoChunk
from src.transport.disk import DiskTransport


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scene_cache_paths(cache_root: Path, video_path: Path) -> tuple[Path, Path]:
    """The one shared, cached scene-score location for a video (report 10:
    "one cached scene segmentation per video shared by all tiers/runs").

    `cache_root` defaults to `assets/dataset` (the same convention
    `scripts/process_dataset.py` uses, so a video already curated for
    training reuses that exact cache) but callers — tests, in particular —
    may point it elsewhere to avoid writing into the real curated dataset.
    """
    vname = video_path.stem
    dataset_dir = cache_root / vname
    return dataset_dir, dataset_dir / "scene_scores.csv"


def split_scene_into_subchunks(
    t_start: float, t_end: float, chunk_duration: float
) -> list[tuple[float, float]]:
    """Partition [t_start, t_end) into consecutive sub-chunks of at most
    `chunk_duration` seconds; the final sub-chunk may be shorter. Pure,
    unit-testable independent of any I/O."""
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if t_end <= t_start:
        return []

    boundaries: list[tuple[float, float]] = []
    cursor = t_start
    while cursor < t_end:
        next_cursor = min(t_end, cursor + chunk_duration)
        boundaries.append((cursor, next_cursor))
        cursor = next_cursor
    return boundaries


def choose_routing(semantic_bytes: int | None, fallback_bytes: int) -> str:
    """Outcome-safe routing decision (report 10 methodology decision 5):
    transmit whichever encoding is smaller. `semantic_bytes=None` means the
    semantic path was never attempted (non-point scenes)."""
    if semantic_bytes is None:
        return "fallback"
    return "semantic" if semantic_bytes <= fallback_bytes else "fallback"


def assert_scenes_tile_video(
    scenes: list[SceneSpan], video_duration_sec: float, tolerance_sec: float = 2.0
) -> None:
    """Scenes must partition [0, video_duration_sec] with no gaps or
    overlaps beyond `tolerance_sec` (report 10's frame-count-invariant
    guard). Raises ValueError, not AssertionError, so this check survives
    even under `python -O`."""
    if not scenes:
        raise ValueError("No scenes to validate against source duration")

    ordered = sorted(scenes, key=lambda s: s.t_start)
    if ordered[0].t_start > tolerance_sec:
        raise ValueError(f"First scene starts at {ordered[0].t_start:.3f}s, expected ~0.0s")

    for prev, curr in zip(ordered, ordered[1:]):
        gap = curr.t_start - prev.t_end
        if abs(gap) > tolerance_sec:
            raise ValueError(
                f"Scene boundary gap/overlap of {gap:.3f}s between scenes ending at "
                f"{prev.t_end:.3f}s and starting at {curr.t_start:.3f}s"
            )

    last_gap = video_duration_sec - ordered[-1].t_end
    if abs(last_gap) > tolerance_sec:
        raise ValueError(
            f"Last scene ends at {ordered[-1].t_end:.3f}s but video duration is "
            f"{video_duration_sec:.3f}s"
        )


def _extract_scene_clip(video_path: Path, output_path: Path, t_start: float, t_end: float) -> None:
    """Frame-accurate, lossless sub-clip extraction.

    The Residual Guarantee requires the server to compute residuals against
    the *actual original* pixels, so this intermediate must not itself lose
    information — hence lossless x264 (`-crf 0`), not a lossy re-encode.
    Two-stage seeking (fast seek to ~2s before the target, then an accurate
    seek/decode for the remainder) keeps this cheap even deep into a long
    match, while still landing on the exact requested frame boundary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, t_end - t_start)
    fast_seek = max(0.0, t_start - 2.0)
    precise_seek = t_start - fast_seek

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{fast_seek:.6f}", "-i", str(video_path),
        "-ss", f"{precise_seek:.6f}", "-t", f"{duration:.6f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p",
        "-y", str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _fallback_encode(clip_path: Path, output_path: Path, config: PointstreamConfig) -> tuple[int, float]:
    """Encode `clip_path` with the pipeline's own configured fallback codec
    (the same FFmpeg wrapper the pipeline uses for its residual stream)."""
    metadata = probe_video_metadata(clip_path)
    frames = iter_video_frames_ffmpeg(clip_path, width=metadata.width, height=metadata.height)

    started = time.perf_counter()
    encode_video_frames_ffmpeg(
        output_path,
        frames,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        codec=config.ffmpeg_codec,
        crf=config.codec_crf,
        preset=config.codec_preset,
    )
    elapsed = time.perf_counter() - started
    return int(output_path.stat().st_size), elapsed


def _cached_fallback_encode(
    anchor_cache_root: Path,
    original_video_path: Path,
    t_start: float,
    t_end: float,
    clip_path: Path,
    output_path: Path,
    config: PointstreamConfig,
) -> tuple[int, float, bool]:
    """Fallback-encode through the anchor cache (report 10 Phase 3), keyed
    on the *original* video's scene span rather than the extracted temp
    clip — repeated `encode_full_match` runs on the same video reuse the
    same fallback encode, since scene cuts are deterministic (methodology
    decision 3) and never change across tiers/runs on a given video.
    """
    elapsed_box: dict[str, float] = {"elapsed": 0.0}

    def _do_encode(dest: Path) -> None:
        _size, elapsed = _fallback_encode(clip_path, dest, config)
        elapsed_box["elapsed"] = elapsed

    size, cache_hit = anchor_cache.get_or_encode(
        cache_root=anchor_cache_root,
        video_path=original_video_path,
        t_start=t_start,
        t_end=t_end,
        codec=config.ffmpeg_codec,
        crf=config.codec_crf,
        preset=config.codec_preset,
        output_path=output_path,
        encode_fn=_do_encode,
    )
    return size, elapsed_box["elapsed"], cache_hit


def _transport_total_bytes(chunk_dir: Path, received_payload: Any) -> int:
    metadata_size = _safe_file_size(chunk_dir / "metadata.msgpack")
    residual_size = _safe_file_size(received_payload.residual.residual_video_uri)
    panorama_uri = getattr(received_payload.panorama, "panorama_uri", None)
    panorama_size = _safe_file_size(panorama_uri)
    actor_references_dir = chunk_dir / "actor_references"
    actor_reference_size = 0
    if actor_references_dir.exists() and actor_references_dir.is_dir():
        actor_reference_size = sum(
            path.stat().st_size for path in actor_references_dir.glob("*") if path.is_file()
        )

    components = [size for size in (metadata_size, residual_size, panorama_size) if size is not None]
    return int(sum(components)) + int(actor_reference_size)


def _safe_file_size(path_like: str | Path | None) -> int | None:
    if path_like is None:
        return None
    candidate = Path(str(path_like))
    if not candidate.exists() or not candidate.is_file():
        return None
    return int(candidate.stat().st_size)


def _process_point_scene(
    video_path: Path,
    scene_idx: int,
    scene: SceneSpan,
    clips_dir: Path,
    transport_root: Path,
    encoder: EncoderPipeline,
    transport: DiskTransport,
    decoder: DecoderRenderer,
    config: PointstreamConfig,
    start_frame_id: int,
    anchor_cache_root: Path,
) -> tuple[dict[str, Any], int]:
    sub_chunk_results: list[dict[str, Any]] = []
    total_bytes = 0
    total_frames = 0
    frame_cursor = start_frame_id

    boundaries = split_scene_into_subchunks(
        scene.t_start, scene.t_end, config.scene_chunk_duration_sec
    )
    for chunk_idx, (cs, ce) in enumerate(boundaries):
        clip_path = clips_dir / f"scene{scene_idx:04d}_chunk{chunk_idx:04d}.mp4"
        _extract_scene_clip(video_path, clip_path, cs, ce)
        clip_metadata = probe_video_metadata(clip_path)

        if clip_metadata.num_frames <= 0:
            continue

        chunk_id = f"s{scene_idx:04d}c{chunk_idx:04d}"
        video_chunk = VideoChunk(
            chunk_id=chunk_id,
            source_uri=str(clip_path),
            # NOTE: start_frame_id=0, not the match-global frame_cursor.
            # Each extracted clip is a self-contained file whose own frame 0
            # is this sub-chunk's first frame -- the convention every DAG
            # node actually implements (ActorExtractor._load_frames reads
            # from the start of source_uri regardless of start_frame_id;
            # ball/segmentation extractors and synthesis_engine only ever
            # do `start_frame_id + local_idx` arithmetic for event
            # numbering). ResidualCalculator._process_residuals is the one
            # place that additionally *seeks* start_frame_id frames into
            # source_uri before reading -- if source_uri were a per-chunk
            # file (as it is here) and start_frame_id were the match-global
            # offset, this scene's frame_cursor would run past the clip's
            # own (small) frame count, raising "zero valid frames" (found
            # via a real run tonight; flagged separately, not fixed here --
            # this discrepancy is pre-existing in residual_calculator.py,
            # not introduced by this module). frame_cursor is preserved
            # below as this sub-chunk's own `global_start_frame` metadata
            # instead, for match-level bookkeeping.
            start_frame_id=0,
            fps=clip_metadata.fps,
            num_frames=clip_metadata.num_frames,
            width=clip_metadata.width,
            height=clip_metadata.height,
            # Background-layer ladder rung 2 (report 10 Phase 5.3): groups this
            # scene's sub-chunks so EncoderPipeline can send a full panorama for
            # chunk_idx=0 and deltas for chunk_idx>0 when
            # config.background_layer == "panorama-delta". A no-op (both rungs
            # send full panoramas) under the default panorama-static layer.
            scene_id=f"scene{scene_idx:04d}",
        )

        encode_started = time.perf_counter()
        payload = encoder.encode_chunk(video_chunk)
        transport.send(payload)
        received = transport.receive(chunk_id)
        encode_elapsed = time.perf_counter() - encode_started

        decode_started = time.perf_counter()
        decoder.process(received)
        decode_elapsed = time.perf_counter() - decode_started

        chunk_dir = transport_root / f"chunk_{chunk_id}"
        semantic_bytes = _transport_total_bytes(chunk_dir, received)

        fallback_path = clips_dir / f"scene{scene_idx:04d}_chunk{chunk_idx:04d}_fallback.mp4"
        fallback_bytes, fallback_elapsed, fallback_cache_hit = _cached_fallback_encode(
            anchor_cache_root=anchor_cache_root,
            original_video_path=video_path,
            t_start=cs,
            t_end=ce,
            clip_path=clip_path,
            output_path=fallback_path,
            config=config,
        )

        routing = choose_routing(semantic_bytes, fallback_bytes)
        chosen_bytes = semantic_bytes if routing == "semantic" else fallback_bytes

        total_bytes += chosen_bytes
        total_frames += clip_metadata.num_frames

        sub_chunk_results.append(
            {
                "chunk_id": chunk_id,
                "t_start": cs,
                "t_end": ce,
                "num_frames": clip_metadata.num_frames,
                # This sub-chunk's position in the whole match, for
                # bookkeeping only -- NOT fed into the VideoChunk itself
                # (see the start_frame_id=0 note above).
                "global_start_frame": frame_cursor,
                "semantic_bytes": semantic_bytes,
                "fallback_bytes": fallback_bytes,
                "fallback_cache_hit": fallback_cache_hit,
                "routing": routing,
                "elapsed_sec": {
                    "semantic_encode": encode_elapsed,
                    "semantic_decode": decode_elapsed,
                    "fallback": fallback_elapsed,
                },
            }
        )
        frame_cursor += clip_metadata.num_frames

    routings = {sc["routing"] for sc in sub_chunk_results}
    if not routings:
        routing_summary = "fallback"
    elif routings == {"semantic"}:
        routing_summary = "semantic"
    elif routings == {"fallback"}:
        routing_summary = "fallback"
    else:
        routing_summary = "mixed"

    scene_result = {
        "scene_index": scene_idx,
        "t_start": scene.t_start,
        "t_end": scene.t_end,
        "scene_class": scene.scene_class.value,
        "routing_summary": routing_summary,
        "bytes": total_bytes,
        "num_frames": total_frames,
        "sub_chunks": sub_chunk_results,
    }
    return scene_result, total_frames


def _process_fallback_scene(
    video_path: Path,
    scene_idx: int,
    scene: SceneSpan,
    clips_dir: Path,
    config: PointstreamConfig,
    anchor_cache_root: Path,
) -> tuple[dict[str, Any], int]:
    clip_path = clips_dir / f"scene{scene_idx:04d}_full.mp4"
    _extract_scene_clip(video_path, clip_path, scene.t_start, scene.t_end)
    clip_metadata = probe_video_metadata(clip_path)

    if clip_metadata.num_frames <= 0:
        scene_result = {
            "scene_index": scene_idx,
            "t_start": scene.t_start,
            "t_end": scene.t_end,
            "scene_class": scene.scene_class.value,
            "routing_summary": "fallback",
            "bytes": 0,
            "num_frames": 0,
            "sub_chunks": None,
            "skipped_zero_frames": True,
        }
        return scene_result, 0

    output_path = clips_dir / f"scene{scene_idx:04d}_full_fallback.mp4"
    fallback_bytes, elapsed, cache_hit = _cached_fallback_encode(
        anchor_cache_root=anchor_cache_root,
        original_video_path=video_path,
        t_start=scene.t_start,
        t_end=scene.t_end,
        clip_path=clip_path,
        output_path=output_path,
        config=config,
    )

    scene_result = {
        "scene_index": scene_idx,
        "t_start": scene.t_start,
        "t_end": scene.t_end,
        "scene_class": scene.scene_class.value,
        "routing_summary": "fallback",
        "bytes": fallback_bytes,
        "num_frames": clip_metadata.num_frames,
        "sub_chunks": None,
        "elapsed_sec": elapsed,
        "cache_hit": cache_hit,
    }
    return scene_result, clip_metadata.num_frames


def encode_full_match(
    config: PointstreamConfig,
    video_path: str | Path,
    transport_root: str | Path,
    scene_cache_root: str | Path | None = None,
    anchor_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Encode a full match end to end: classify scenes, route each one,
    and return a match-level summary dict (JSON-serializable, mirrors
    `run_summary.json`'s top-level style).

    `scene_cache_root` defaults to `assets/dataset` (production
    convention); pass an isolated directory in tests. `anchor_cache_root`
    defaults to `outputs/_anchor_cache` (a derived/regenerable cache, kept
    out of `assets/dataset` which is reserved for curated dataset content);
    also overridable for tests.
    """
    match_started = time.perf_counter()
    resolved_video_path = Path(video_path).expanduser().resolve()
    resolved_transport_root = Path(transport_root).expanduser()
    resolved_transport_root.mkdir(parents=True, exist_ok=True)

    resolved_cache_root = (
        Path(scene_cache_root).expanduser()
        if scene_cache_root is not None
        else _project_root() / "assets" / "dataset"
    )
    dataset_dir, cache_file = scene_cache_paths(resolved_cache_root, resolved_video_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    resolved_anchor_cache_root = (
        Path(anchor_cache_root).expanduser()
        if anchor_cache_root is not None
        else _project_root() / "outputs" / "_anchor_cache" / resolved_video_path.stem
    )
    resolved_anchor_cache_root.mkdir(parents=True, exist_ok=True)

    classify_started = time.perf_counter()
    extract_scene_scores(str(resolved_video_path), str(cache_file))
    scenes = classify_video_scenes(str(resolved_video_path), str(cache_file))
    classify_elapsed = time.perf_counter() - classify_started

    if not scenes:
        raise ValueError(f"Scene classification produced no scenes for {resolved_video_path}")

    source_metadata = probe_video_metadata(resolved_video_path)
    video_duration_sec = source_metadata.num_frames / source_metadata.fps
    assert_scenes_tile_video(scenes, video_duration_sec)

    clips_dir = resolved_transport_root / "scene_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    encoder = EncoderPipeline(
        config=config,
        execution_pool=build_execution_pool(config),
        actor_extractor=build_actor_extractor(config),
        ball_extractor=build_ball_extractor(config),
        reference_extractor=build_reference_extractor(config),
        residual_calculator=build_residual_calculator(config),
    )
    transport = DiskTransport(config=config, root_dir=resolved_transport_root)
    decoder = DecoderRenderer(config=config, output_root=resolved_transport_root / "decoded")

    scene_results: list[dict[str, Any]] = []
    frame_cursor = 0
    try:
        for scene_idx, scene in enumerate(scenes):
            if scene.scene_class == SceneClass.POINT:
                result, frames_used = _process_point_scene(
                    video_path=resolved_video_path,
                    scene_idx=scene_idx,
                    scene=scene,
                    clips_dir=clips_dir,
                    transport_root=resolved_transport_root,
                    encoder=encoder,
                    transport=transport,
                    decoder=decoder,
                    config=config,
                    start_frame_id=frame_cursor,
                    anchor_cache_root=resolved_anchor_cache_root,
                )
            else:
                result, frames_used = _process_fallback_scene(
                    video_path=resolved_video_path,
                    scene_idx=scene_idx,
                    scene=scene,
                    clips_dir=clips_dir,
                    config=config,
                    anchor_cache_root=resolved_anchor_cache_root,
                )
            frame_cursor += frames_used
            scene_results.append(result)
    finally:
        encoder.shutdown()

    total_bytes = sum(int(s["bytes"]) for s in scene_results)
    source_bytes = int(resolved_video_path.stat().st_size)

    # Report 10 Phase 3: realtime factor (wall-clock / source duration),
    # encoder and decoder tracked separately. "Encoder" here includes both
    # attempted paths for point sub-chunks (semantic + fallback) since the
    # outcome-safe comparison genuinely pays for both; "decoder" is only the
    # semantic DecoderRenderer.process() cost (fallback-routed scenes never
    # run our decoder).
    encode_total_sec = classify_elapsed
    decode_total_sec = 0.0
    for s in scene_results:
        if s["sub_chunks"]:
            for sub_chunk in s["sub_chunks"]:
                elapsed = sub_chunk["elapsed_sec"]
                encode_total_sec += elapsed["semantic_encode"] + elapsed["fallback"]
                decode_total_sec += elapsed["semantic_decode"]
        else:
            encode_total_sec += float(s.get("elapsed_sec") or 0.0)

    num_point_scenes = sum(1 for s in scene_results if s["scene_class"] == SceneClass.POINT.value)
    num_point_subchunks = sum(len(s["sub_chunks"] or []) for s in scene_results if s["sub_chunks"] is not None)
    num_point_subchunks_to_fallback = sum(
        1
        for s in scene_results
        if s["sub_chunks"]
        for sc in s["sub_chunks"]
        if sc["routing"] == "fallback"
    )

    match_summary: dict[str, Any] = {
        "video_path": str(resolved_video_path),
        "fps": source_metadata.fps,
        "num_frames_total": source_metadata.num_frames,
        "duration_sec": video_duration_sec,
        "scenes": scene_results,
        "totals": {
            "num_scenes": len(scene_results),
            "num_point_scenes": num_point_scenes,
            "num_fallback_scenes": len(scene_results) - num_point_scenes,
            "num_point_subchunks": num_point_subchunks,
            "num_point_subchunks_routed_to_fallback": num_point_subchunks_to_fallback,
            "total_bytes": total_bytes,
            "source_bytes": source_bytes,
            "bytes_to_source_ratio": (float(total_bytes) / source_bytes) if source_bytes else None,
        },
        "timings_sec": {
            "scene_classification": classify_elapsed,
            "encode_total": encode_total_sec,
            "decode_total": decode_total_sec,
            "encoder_realtime_factor": (encode_total_sec / video_duration_sec) if video_duration_sec > 0 else None,
            "decoder_realtime_factor": (decode_total_sec / video_duration_sec) if video_duration_sec > 0 else None,
            "total": float(time.perf_counter() - match_started),
        },
    }
    return match_summary
