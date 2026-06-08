from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.shared.config import PointstreamConfig, load_config
from src.encoder.ball_extractor import BallExtractor
from src.encoder.segmentation_ball_extractor import SegmentationBallExtractor
from src.encoder.execution_pool import BaseExecutionPool, TaggedMultiprocessPool, WorkerConfig
from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.actor_pipeline import ActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.residual_calculator import BaseImportanceMapper, BinaryActorImportanceMapper, ResidualCalculator, UniformImportanceMapper
from src.encoder.video_io import encode_video_frames_ffmpeg, probe_video_metadata, ensure_ffmpeg_encoder_available
from src.experiment_evaluation import evaluate_run_summary
from src.shared.schemas import VideoChunk, ResidualMode
from src.transport.disk import DiskTransport

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _create_timestamped_output_dir(base_root: str | Path | None = None) -> Path:
    root = Path(base_root).expanduser() if base_root is not None else (_project_root() / "outputs")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _safe_file_size(path_like: str | Path | None) -> int | None:
    if path_like is None:
        return None
    candidate = Path(str(path_like))
    if not candidate.exists() or not candidate.is_file():
        return None
    return int(candidate.stat().st_size)

def _build_execution_pool(config: PointstreamConfig) -> BaseExecutionPool | None:
    mode = config.execution_pool.strip().lower()
    if mode == "inline":
        return None
    if mode == "tagged":
        return TaggedMultiprocessPool(
            cpu_workers=config.cpu_workers or 1,
            gpu_workers=config.gpu_workers or 1,
            worker_config=WorkerConfig(gpu_dtype=config.gpu_dtype),
        )
    raise ValueError(f"Unknown execution pool mode: {mode}")

def _build_actor_extractor(config: PointstreamConfig) -> ActorExtractor | None:
    normalized_mask_mode = config.compositing_mask_mode.strip().lower()
    include_mask_metadata = normalized_mask_mode == "metadata-source-mask"
    
    return ActorExtractor(
        config=config,
        render_debug_keyframes=False, # Debug output handling moved to explicit checks if needed
        detector_backend=config.detector,
        detector_caption=config.target_class_caption,
        pose_backend=config.pose_estimator,
        segmenter_backend=config.segmenter,
        segmenter_caption=config.target_class_caption,
        pose_delta_threshold=config.payload_pose_delta_threshold,
        include_mask_metadata=include_mask_metadata,
        metadata_mask_codec=config.metadata_mask_codec,
    )

def _build_ball_extractor(config: PointstreamConfig) -> Any | None:
    mode = config.ball_extractor.strip().lower()
    if mode == "segmentation":
        return SegmentationBallExtractor(
            detection_max_side=config.ball_max_side,
            confidence_threshold=config.ball_det_conf or 0.25,
            model_name=config.ball_det_model or "yolo26n.pt",
            class_id=config.ball_class_id or 32,
        )
    return BallExtractor(
        difference_threshold=config.ball_difference_threshold,
        min_blob_area=config.ball_min_blob_area,
        detection_max_side=config.ball_max_side,
    )

def _build_reference_extractor(config: PointstreamConfig) -> ReferenceExtractor:
    return ReferenceExtractor(
        jpeg_quality=config.reference_jpeg_quality,
        bbox_padding_ratio=config.reference_padding_ratio,
    )

def _build_residual_calculator(config: PointstreamConfig) -> ResidualCalculator:
    mapper_mode = config.importance_mapper.strip().lower()
    mapper: BaseImportanceMapper
    if mapper_mode == "uniform":
        mapper = UniformImportanceMapper()
    else:
        mapper = BinaryActorImportanceMapper()
        
    return ResidualCalculator(
        config=config,
        seed=config.seed,
        importance_mapper=mapper,
    )

def run_pipeline(
    config: PointstreamConfig,
    transport_root: str | Path | None = None,
    chunk_id: str = "0001",
    runtime_output_root: str | Path | None = None,
) -> dict[str, object]:
    pipeline_started = perf_counter()
    resolved_transport_root = Path(transport_root).expanduser() if transport_root is not None else _create_timestamped_output_dir()
    resolved_transport_root.mkdir(parents=True, exist_ok=True)
    
    resolved_runtime_root = Path(runtime_output_root).expanduser() if runtime_output_root is not None else resolved_transport_root
    resolved_runtime_root.mkdir(parents=True, exist_ok=True)

    if config.source_uri is None:
        resolved_source_uri = _ensure_mock_source_video(config, runtime_output_root=resolved_runtime_root)
    else:
        resolved_source_uri = str(Path(config.source_uri).expanduser().resolve())
        
    source_metadata = probe_video_metadata(resolved_source_uri)
    chunk_frames = source_metadata.num_frames if config.num_frames is None else min(source_metadata.num_frames, config.num_frames)

    chunk = VideoChunk(
        chunk_id=chunk_id,
        source_uri=resolved_source_uri,
        start_frame_id=0,
        fps=source_metadata.fps,
        num_frames=chunk_frames,
        width=source_metadata.width,
        height=source_metadata.height,
    )
    
    execution_pool = _build_execution_pool(config)
    actor_extractor = _build_actor_extractor(config)
    ball_extractor = _build_ball_extractor(config)
    reference_extractor = _build_reference_extractor(config)
    residual_calculator = _build_residual_calculator(config)

    encoder = EncoderPipeline(
        config=config,
        execution_pool=execution_pool,
        actor_extractor=actor_extractor,
        ball_extractor=ball_extractor,
        reference_extractor=reference_extractor,
        residual_calculator=residual_calculator,
    )

    try:
        encode_started = perf_counter()
        payload = encoder.encode_chunk(chunk)
        encode_finished = perf_counter()
    finally:
        encoder.shutdown()

    normalized_transport = config.transport.strip().lower()
    if normalized_transport != "disk":
        raise ValueError(f"Unsupported transport backend: {normalized_transport}")

    transport = DiskTransport(config=config, root_dir=resolved_transport_root)
    transport_send_started = perf_counter()
    transport.send(payload)
    transport_send_finished = perf_counter()
    
    transport_receive_started = perf_counter()
    received_payload = transport.receive(chunk.chunk_id)
    transport_receive_finished = perf_counter()

    resolved_decoder_root = resolved_transport_root / "decoded"
    decoder = DecoderRenderer(config=config, output_root=resolved_decoder_root)
        
    decode_started = perf_counter()
    decoded = decoder.process(received_payload)
    decode_finished = perf_counter()

    chunk_dir = resolved_transport_root / f"chunk_{received_payload.chunk.chunk_id}"
    metadata_size_bytes = _safe_file_size(chunk_dir / "metadata.msgpack")
    residual_size_bytes = _safe_file_size(received_payload.residual.residual_video_uri)
    actor_references_dir = chunk_dir / "actor_references"
    actor_reference_size_bytes = None
    if actor_references_dir.exists() and actor_references_dir.is_dir():
        actor_reference_size_bytes = int(
            sum(path.stat().st_size for path in actor_references_dir.glob("*") if path.is_file())
        )
    panorama_packet = getattr(received_payload, "panorama", None)
    panorama_uri = getattr(panorama_packet, "panorama_uri", None) if panorama_packet is not None else None
    panorama_size_bytes = _safe_file_size(panorama_uri)
    source_size_bytes = _safe_file_size(resolved_source_uri)

    transported_components = [size for size in (metadata_size_bytes, residual_size_bytes, panorama_size_bytes) if size is not None]
    if actor_reference_size_bytes is not None:
        transported_components.append(actor_reference_size_bytes)
    transport_total_size_bytes = int(sum(transported_components))

    residual_mode_obj = getattr(received_payload.residual, "mode", ResidualMode.FULL_VIDEO)
    residual_mode_value = residual_mode_obj.value if isinstance(residual_mode_obj, ResidualMode) else str(residual_mode_obj)

    summary = {
        "chunk_id": received_payload.chunk.chunk_id,
        "run_output_root": str(resolved_transport_root),
        "source_uri": resolved_source_uri,
        "num_actor_packets": len(received_payload.actors),
        "num_rigid_object_packets": len(received_payload.rigid_objects),
        "ball_object_id": received_payload.ball.object_id,
        "residual_uri": received_payload.residual.residual_video_uri,
        "residual_mode": residual_mode_value,
        "decoded_uri": decoded.output_uri,
        "transport_backend": normalized_transport,
        "source_size_bytes": source_size_bytes,
        "metadata_size_bytes": metadata_size_bytes,
        "actor_reference_size_bytes": actor_reference_size_bytes,
        "residual_size_bytes": residual_size_bytes,
        "panorama_size_bytes": panorama_size_bytes,
        "transport_total_size_bytes": transport_total_size_bytes,
        "pipeline_total_sec": float(perf_counter() - pipeline_started),
        "encode_chunk_sec": float(encode_finished - encode_started),
        "transport_send_sec": float(transport_send_finished - transport_send_started),
        "transport_receive_sec": float(transport_receive_finished - transport_receive_started),
        "decode_sec": float(decode_finished - decode_started),
        "compositing_mask_mode": config.compositing_mask_mode,
        "postgen_segmenter_backend": config.postgen_segmenter_backend,
        "metadata_mask_codec": config.metadata_mask_codec,
        "genai_enabled": bool(config.genai_backend),
        "genai_backend": config.genai_backend,
    }
    if source_size_bytes is not None and source_size_bytes > 0:
        summary["transport_to_source_ratio"] = float(transport_total_size_bytes) / float(source_size_bytes)

    return summary


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PointStream pipeline.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file.")
    parser.add_argument("--input", dest="source_uri", type=str, default=None, help="Input video override.")
    
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cli_overrides = {}
    if args.source_uri is not None:
        cli_overrides["source_uri"] = args.source_uri
        
    config = load_config(args.config, cli_overrides)
    
    root_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        import transformers
        transformers.logging.set_verbosity(root_level)
    except ImportError:
        pass

    try:
        import diffusers
        diffusers.logging.set_verbosity(root_level)
    except ImportError:
        pass

    try:
        import huggingface_hub
        huggingface_hub.utils.logging.set_verbosity(root_level)
    except ImportError:
        pass
    
    ensure_ffmpeg_encoder_available(config.ffmpeg_codec)
    
    run_output_root = _create_timestamped_output_dir(base_root=_project_root() / "outputs")
    
    run_summary = run_pipeline(
        config=config,
        transport_root=run_output_root,
        chunk_id="0001",
        runtime_output_root=run_output_root,
    )
    
    if config.evaluation_mode:
        run_summary["evaluation"] = evaluate_run_summary(
            summary=run_summary,
            experiment_dir=run_output_root,
            max_frames=config.evaluation_max_frames,
            metrics=config.evaluation_mode,
        )

    summary_json = json.dumps(run_summary, indent=2)
    print(summary_json)

    if config.summary_file:
        summary_path = run_output_root / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{summary_json}\n", encoding="utf-8")

    return 0


def _ensure_mock_source_video(config: PointstreamConfig, runtime_output_root: str | Path | None = None) -> str:
    run_root = Path(runtime_output_root).expanduser() if runtime_output_root is not None else _create_timestamped_output_dir()
    source_dir = run_root / "runtime_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "tennis_chunk_0001.mp4"

    if source_path.exists() and source_path.is_file():
        return str(source_path)

    frames: list[np.ndarray] = []
    for frame_idx in range(60):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x0 = 100 + (frame_idx * 8) % 900
        y0 = 200 + (frame_idx * 3) % 250
        cv2.rectangle(frame, (x0, y0), (x0 + 90, y0 + 180), (0, 220, 60), thickness=-1)
        cv2.rectangle(frame, (1280 - x0 - 120, 720 - y0 - 220), (1280 - x0 - 40, 720 - y0 - 40), (30, 200, 230), thickness=-1)
        frames.append(frame)

    encode_video_frames_ffmpeg(
        output_path=source_path,
        frames_bgr=frames,
        fps=30.0,
        width=1280,
        height=720,
        codec=config.ffmpeg_codec,
        pix_fmt="yuv420p",
        crf=config.codec_crf or 18,
        preset=config.codec_preset or "veryfast",
    )
    return str(source_path)

if __name__ == "__main__":
    raise SystemExit(run_cli())
