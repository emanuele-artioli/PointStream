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

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised when optional dependency is missing.
    yaml = None

from src.encoder.ball_extractor import BallExtractor
from src.encoder.segmentation_ball_extractor import SegmentationBallExtractor
from src.encoder.execution_pool import BaseExecutionPool
from src.encoder.execution_pool import TaggedMultiprocessPool
from src.encoder.execution_pool import WorkerConfig
from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.mock_extractors import ActorExtractor
from src.encoder.mock_extractors import BallTracker
from src.encoder.mock_extractors import MockActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.residual_calculator import BaseImportanceMapper
from src.encoder.residual_calculator import BinaryActorImportanceMapper
from src.encoder.residual_calculator import ResidualCalculator
from src.encoder.residual_calculator import UniformImportanceMapper
from src.encoder.video_io import encode_video_frames_ffmpeg, probe_video_metadata
from src.encoder.video_io import ensure_ffmpeg_encoder_available
from src.experiment_evaluation import evaluate_run_summary
from src.shared.schemas import VideoChunk
from src.shared.schemas import ResidualMode
from src.shared.synthesis_engine import SynthesisEngine
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


class _MockBallExtractorAdapter:
    """Adapter so the lightweight BallTracker can plug into EncoderPipeline."""

    def __init__(self) -> None:
        self._tracker = BallTracker()

    def process(self, chunk: VideoChunk, panorama: Any, frame_states: list[Any]) -> Any:
        _ = panorama
        _ = frame_states
        return self._tracker.process(chunk)


def run_mock_pipeline(
    transport_root: str | Path | None = None,
    execution_pool: BaseExecutionPool | None = None,
    source_uri: str | None = None,
    num_frames: int | None = None,
    actor_extractor: ActorExtractor | MockActorExtractor | None = None,
    ball_extractor: Any | None = None,
    reference_extractor: ReferenceExtractor | None = None,
    residual_calculator: ResidualCalculator | None = None,
    transport_backend: str = "disk",
    decoder_output_root: str | Path | None = None,
    chunk_id: str = "0001",
    runtime_output_root: str | Path | None = None,
) -> dict[str, object]:
    pipeline_started = perf_counter()
    resolved_transport_root = (
        Path(transport_root).expanduser() if transport_root is not None else _create_timestamped_output_dir()
    )
    resolved_transport_root.mkdir(parents=True, exist_ok=True)

    resolved_runtime_root = (
        Path(runtime_output_root).expanduser()
        if runtime_output_root is not None
        else resolved_transport_root
    )
    resolved_runtime_root.mkdir(parents=True, exist_ok=True)

    if source_uri is None:
        resolved_source_uri = _ensure_mock_source_video(runtime_output_root=resolved_runtime_root)
    else:
        resolved_source_uri = str(Path(source_uri).expanduser().resolve())
    source_metadata = probe_video_metadata(resolved_source_uri)
    chunk_frames = source_metadata.num_frames if num_frames is None else min(source_metadata.num_frames, num_frames)

    chunk = VideoChunk(
        chunk_id=chunk_id,
        source_uri=resolved_source_uri,
        start_frame_id=0,
        fps=source_metadata.fps,
        num_frames=chunk_frames,
        width=source_metadata.width,
        height=source_metadata.height,
    )

    resolved_actor_extractor = actor_extractor or MockActorExtractor()
    encoder = EncoderPipeline(
        execution_pool=execution_pool,
        actor_extractor=resolved_actor_extractor,
        ball_extractor=ball_extractor,
        reference_extractor=reference_extractor,
        residual_calculator=residual_calculator,
    )

    previous_debug_artifact_dir = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
    previous_runtime_output_dir = os.environ.get("POINTSTREAM_RUNTIME_OUTPUT_DIR")
    debug_disabled = os.environ.get("POINTSTREAM_DISABLE_DEBUG_ARTIFACTS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    previous_cwd = Path.cwd()
    os.chdir(resolved_runtime_root)
    os.environ["POINTSTREAM_RUNTIME_OUTPUT_DIR"] = str(resolved_runtime_root)
    if debug_disabled:
        os.environ.pop("POINTSTREAM_DEBUG_ARTIFACT_DIR", None)
    else:
        os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = str(resolved_runtime_root / "debug")
    try:
        encode_started = perf_counter()
        payload = encoder.encode_chunk(chunk)
        encode_finished = perf_counter()
    finally:
        encoder.shutdown()
        if previous_debug_artifact_dir is None:
            os.environ.pop("POINTSTREAM_DEBUG_ARTIFACT_DIR", None)
        else:
            os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = previous_debug_artifact_dir
        if previous_runtime_output_dir is None:
            os.environ.pop("POINTSTREAM_RUNTIME_OUTPUT_DIR", None)
        else:
            os.environ["POINTSTREAM_RUNTIME_OUTPUT_DIR"] = previous_runtime_output_dir
        os.chdir(previous_cwd)

    normalized_transport = transport_backend.strip().lower()
    if normalized_transport != "disk":
        raise ValueError(f"Unsupported transport backend: {transport_backend}")

    transport = DiskTransport(root_dir=resolved_transport_root)
    transport_send_started = perf_counter()
    transport.send(payload)
    transport_send_finished = perf_counter()
    transport_receive_started = perf_counter()
    received_payload = transport.receive(chunk.chunk_id)
    transport_receive_finished = perf_counter()

    resolved_decoder_root = (
        Path(decoder_output_root).expanduser()
        if decoder_output_root is not None
        else resolved_transport_root / "decoded"
    )
    try:
        decoder = DecoderRenderer(output_root=resolved_decoder_root)
    except TypeError:
        # Keep tests/simple stubs working when DecoderRenderer is monkeypatched.
        decoder = DecoderRenderer()
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
            sum(
                path.stat().st_size
                for path in actor_references_dir.glob("*")
                if path.is_file()
            )
        )
    panorama_packet = getattr(received_payload, "panorama", None)
    panorama_uri = getattr(panorama_packet, "panorama_uri", None) if panorama_packet is not None else None
    panorama_size_bytes = _safe_file_size(panorama_uri)
    source_size_bytes = _safe_file_size(resolved_source_uri)

    transported_components = [
        size for size in (metadata_size_bytes, residual_size_bytes, panorama_size_bytes)
        if size is not None
    ]
    if actor_reference_size_bytes is not None:
        transported_components.append(actor_reference_size_bytes)
    transport_total_size_bytes = int(sum(transported_components))

    residual_mode_obj = getattr(received_payload.residual, "mode", ResidualMode.FULL_VIDEO)
    residual_mode_value = (
        residual_mode_obj.value if isinstance(residual_mode_obj, ResidualMode) else str(residual_mode_obj)
    )

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
        "compositing_mask_mode": os.environ.get("POINTSTREAM_COMPOSITING_MASK_MODE", "postgen-seg-client"),
        "postgen_segmenter_backend": os.environ.get("POINTSTREAM_POSTGEN_SEGMENTER_BACKEND", "yolo"),
        "metadata_mask_codec": os.environ.get("POINTSTREAM_METADATA_MASK_CODEC", "auto"),
        "genai_enabled": os.environ.get("POINTSTREAM_ENABLE_GENAI", "0") == "1",
        "genai_backend": os.environ.get("POINTSTREAM_GENAI_BACKEND", "controlnet"),
    }
    if source_size_bytes is not None and source_size_bytes > 0:
        ratio = float(transport_total_size_bytes) / float(source_size_bytes)
        summary["transport_to_source_ratio"] = ratio
        summary["transport_savings_percent"] = (1.0 - ratio) * 100.0
    # include DAG profiling produced during encoding, if available
    try:
        profiling = getattr(encoder, "_last_dag_profile", None)
        if profiling is not None:
            summary["profiling"] = profiling
    except Exception:
        # defensive: don't fail summary construction on profiling issues
        pass
    return summary


def _build_execution_pool(mode: str, cpu_workers: int, gpu_workers: int) -> BaseExecutionPool | None:
    if mode == "inline":
        return None
    return TaggedMultiprocessPool(
        config=WorkerConfig(
            cpu_workers=int(cpu_workers),
            gpu_workers=int(gpu_workers),
        )
    )


def _build_actor_extractor(
    mode: str,
    detector: str,
    detector_caption: str,
    pose_estimator: str,
    segmenter: str,
    segmenter_caption: str,
    debug_enabled: bool = False,
    pose_delta_threshold: float = 20.0,
    compositing_mask_mode: str = "alpha-heuristic",
    metadata_mask_codec: str = "auto",
    disable_debug_keyframes: bool | None = None,
) -> ActorExtractor | MockActorExtractor | None:
    _ = disable_debug_keyframes
    normalized_mode = mode.strip().lower()
    if normalized_mode == "mock":
        return MockActorExtractor()

    normalized_mask_mode = compositing_mask_mode.strip().lower()
    include_mask_metadata = normalized_mask_mode == "metadata-source-mask"

    normalized_pose = str(pose_estimator).strip().lower()
    normalized_segmenter = str(segmenter).strip().lower()
    uses_default = (
        detector == "yolo26"
        and detector_caption.strip().lower() == "tennis player"
        and normalized_pose in {"yolo", "yolo26", "yolo26n"}
        and normalized_segmenter in {"yolo", "yolo26", "yolo26n"}
        and segmenter_caption.strip().lower() == "tennis player"
        and float(pose_delta_threshold) == 20.0
        and not include_mask_metadata
    )
    if uses_default:
        return None

    return ActorExtractor(
        render_debug_keyframes=debug_enabled,
        detector_backend=detector,
        detector_caption=detector_caption,
        pose_backend=pose_estimator,
        segmenter_backend=segmenter,
        segmenter_caption=segmenter_caption,
        pose_delta_threshold=float(pose_delta_threshold),
        include_mask_metadata=include_mask_metadata,
        metadata_mask_codec=str(metadata_mask_codec),
    )


def _build_ball_extractor(
    mode: str,
    difference_threshold: float,
    min_blob_area: int,
    detection_max_side: int,
) -> Any | None:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "mock":
        return _MockBallExtractorAdapter()
    # Segmentation-only mode
    if normalized_mode in {"segmentation", "detector"}:
        conf = float(os.environ.get("POINTSTREAM_BALL_DET_CONF", "0.25"))
        model_name = os.environ.get("POINTSTREAM_BALL_DET_MODEL", "yolo26n.pt")
        return SegmentationBallExtractor(model_name=model_name, confidence=conf)

    # Cascade mode: try segmentation detector first, then fallback to residual-based
    if normalized_mode in {"cascade", "hybrid"}:
        conf = float(os.environ.get("POINTSTREAM_BALL_DET_CONF", "0.25"))
        model_name = os.environ.get("POINTSTREAM_BALL_DET_MODEL", "yolo26n.pt")
        seg = SegmentationBallExtractor(model_name=model_name, confidence=conf)
        resid = BallExtractor(
            difference_threshold=float(difference_threshold),
            min_blob_area=int(min_blob_area),
            detection_max_side=int(detection_max_side),
        )

        class _Cascade:
            def process(self, chunk, panorama, frame_states):
                try:
                    return seg.process(chunk=chunk, panorama=panorama, frame_states=frame_states)
                except Exception:
                    return resid.process(chunk=chunk, panorama=panorama, frame_states=frame_states)

        return _Cascade()

    uses_default = (
        float(difference_threshold) == 18.0
        and int(min_blob_area) == 6
        and int(detection_max_side) <= 0
    )
    if uses_default:
        return None

    return BallExtractor(
        difference_threshold=float(difference_threshold),
        min_blob_area=int(min_blob_area),
        detection_max_side=int(detection_max_side),
    )


def _build_reference_extractor(jpeg_quality: int, padding_ratio: float) -> ReferenceExtractor | None:
    uses_default = int(jpeg_quality) == 75 and float(padding_ratio) == 0.10
    if uses_default:
        return None

    return ReferenceExtractor(
        jpeg_quality=int(jpeg_quality),
        bbox_padding_ratio=float(padding_ratio),
    )


def _build_importance_mapper(name: str) -> BaseImportanceMapper:
    normalized = name.strip().lower()
    if normalized == "uniform":
        return UniformImportanceMapper()
    return BinaryActorImportanceMapper()


def _parse_optional_int_or_none(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    return int(value)


def _build_residual_calculator(
    seed: int,
    importance_mapper: str,
    background_block_downscale_factor: int | None = 2,
    residual_batch_size: int = 8,
    downscale_interpolation: str = "bilinear",
    residual_block_size: int = 8,
    block_information_threshold: float = 0.0,
) -> ResidualCalculator | None:
    return _build_residual_calculator_impl(
        seed=seed,
        importance_mapper=importance_mapper,
        background_block_downscale_factor=background_block_downscale_factor,
        residual_batch_size=residual_batch_size,
        downscale_interpolation=downscale_interpolation,
        residual_block_size=residual_block_size,
        block_information_threshold=block_information_threshold,
    )


def _build_residual_calculator_impl(
    seed: int,
    importance_mapper: str,
    background_block_downscale_factor: int | None = 2,
    residual_batch_size: int = 8,
    downscale_interpolation: str = "bilinear",
    residual_block_size: int = 8,
    block_information_threshold: float = 0.0,
) -> ResidualCalculator | None:
    uses_default = (
        int(seed) == 1337
        and importance_mapper.strip().lower() == "binary"
        and background_block_downscale_factor == 2
        and int(residual_batch_size) == 8
        and str(downscale_interpolation).strip().lower() == "bilinear"
        and int(residual_block_size) == 8
        and float(block_information_threshold) == 0.0
    )
    if uses_default:
        return None

    mapper = _build_importance_mapper(importance_mapper)
    return ResidualCalculator(
        synthesis_engine=SynthesisEngine(seed=int(seed)),
        importance_mapper=mapper,
        background_block_downscale_factor=background_block_downscale_factor,
        residual_batch_size=residual_batch_size,
        downscale_interpolation=downscale_interpolation,
        residual_block_size=residual_block_size,
        block_information_threshold=block_information_threshold,
    )


def _apply_runtime_env_overrides(args: argparse.Namespace) -> None:
    if args.ffmpeg_codec is not None:
        os.environ["POINTSTREAM_FFMPEG_CODEC"] = str(args.ffmpeg_codec)
    if args.codec_crf is not None:
        os.environ["POINTSTREAM_CODEC_CRF"] = str(int(args.codec_crf))
    if args.codec_preset is not None:
        os.environ["POINTSTREAM_CODEC_PRESET"] = str(args.codec_preset)
    enable_genai = bool(args.genai_backend)

    os.environ["POINTSTREAM_ENABLE_GENAI"] = "1" if enable_genai else "0"

    if args.genai_backend is not None:
        os.environ["POINTSTREAM_GENAI_BACKEND"] = str(args.genai_backend)
    if args.animate_anyone_repo_dir is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_REPO_DIR"] = str(Path(args.animate_anyone_repo_dir).expanduser())
    if args.animate_anyone_model_dir is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR"] = str(Path(args.animate_anyone_model_dir).expanduser())
    if args.animate_anyone_window is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_WINDOW"] = str(int(args.animate_anyone_window))
    if args.animate_anyone_steps is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_STEPS"] = str(int(args.animate_anyone_steps))
    if args.animate_anyone_cfg is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_CFG"] = str(float(args.animate_anyone_cfg))
    if args.animate_anyone_width is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_WIDTH"] = str(int(args.animate_anyone_width))
    if args.animate_anyone_height is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_HEIGHT"] = str(int(args.animate_anyone_height))
    if args.genai_preroll_frames is not None:
        os.environ["POINTSTREAM_GENAI_PREROLL_FRAMES"] = str(int(args.genai_preroll_frames))
    os.environ["POINTSTREAM_GENAI_KEYFRAME_ONLY"] = "1" if bool(args.genai_keyframe_only) else "0"
    if args.animate_anyone_transparent_threshold is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_TRANSPARENT_THRESHOLD"] = str(
            int(args.animate_anyone_transparent_threshold)
        )
    if args.gpu_dtype is not None:
        os.environ["POINTSTREAM_GPU_DTYPE"] = str(args.gpu_dtype)
    if args.ball_max_side is not None:
        os.environ["POINTSTREAM_BALL_MAX_SIDE"] = str(int(args.ball_max_side))
    if args.ball_det_conf is not None:
        os.environ["POINTSTREAM_BALL_DET_CONF"] = str(float(args.ball_det_conf))
    if args.ball_det_model is not None:
        os.environ["POINTSTREAM_BALL_DET_MODEL"] = str(args.ball_det_model)
    if args.ball_class_id is not None:
        os.environ["POINTSTREAM_BALL_CLASS_ID"] = str(int(args.ball_class_id))
    if args.genai_resize_mode is not None:
        os.environ["POINTSTREAM_GENAI_RESIZE_MODE"] = str(args.genai_resize_mode)
    if args.animate_anyone_adaptive_threshold is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_ADAPTIVE_THRESHOLD"] = (
            "1" if bool(args.animate_anyone_adaptive_threshold) else "0"
        )
    if args.animate_anyone_alpha_smoothing is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_ALPHA_SMOOTHING"] = str(
            float(args.animate_anyone_alpha_smoothing)
        )
    if args.compositing_mask_mode is not None:
        os.environ["POINTSTREAM_COMPOSITING_MASK_MODE"] = str(args.compositing_mask_mode)
    if args.postgen_segmenter_backend is not None:
        os.environ["POINTSTREAM_POSTGEN_SEGMENTER_BACKEND"] = str(args.postgen_segmenter_backend)
    if args.metadata_mask_codec is not None:
        os.environ["POINTSTREAM_METADATA_MASK_CODEC"] = str(args.metadata_mask_codec)
    if args.panorama_warp_batch_size is not None:
        os.environ["POINTSTREAM_PANORAMA_WARP_BATCH_SIZE"] = str(int(args.panorama_warp_batch_size))
    if args.panorama_codec is not None:
        os.environ["POINTSTREAM_PANORAMA_CODEC"] = str(args.panorama_codec)
    if args.panorama_jpeg_quality is not None:
        os.environ["POINTSTREAM_PANORAMA_JPEG_QUALITY"] = str(int(args.panorama_jpeg_quality))
    if args.panorama_png_compression is not None:
        os.environ["POINTSTREAM_PANORAMA_PNG_COMPRESSION"] = str(int(args.panorama_png_compression))
    if args.allow_auto_model_download:
        os.environ["POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD"] = "1"
    if args.postgen_segmenter_model is not None:
        os.environ["POINTSTREAM_POSTGEN_SEGMENTER_MODEL"] = str(args.postgen_segmenter_model)
    debug_enabled = bool(getattr(args, "debug", False))
    os.environ["POINTSTREAM_DISABLE_DEBUG_ARTIFACTS"] = "0" if debug_enabled else "1"
    os.environ["POINTSTREAM_LOG_LEVEL"] = str(getattr(args, "log_level", "info")).strip().lower()
    os.environ["POINTSTREAM_ENABLE_SHIFTED_BALL"] = "1" if bool(args.enable_shifted_ball) else "0"


def _load_config_overrides(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    suffix = path.suffix.strip().lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config requires PyYAML. Install dependency: pyyaml")
        payload = yaml.safe_load(raw_text)
    else:
        raise ValueError("--config supports only .json, .yaml, and .yml files")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping/object")

    normalized = {
        str(key).strip().replace("-", "_"): value
        for key, value in payload.items()
    }
    if "input" in normalized and "source_uri" not in normalized:
        normalized["source_uri"] = normalized.pop("input")
    return normalized


def _detect_auto_worker_counts() -> tuple[int, int]:
    cpu_workers = max(1, int(os.cpu_count() or 1))
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible_devices and visible_devices not in {"-1", "none"}:
        gpu_workers = max(1, len([tok for tok in visible_devices.split(",") if tok.strip()]))
    else:
        gpu_workers = 1
    return cpu_workers, gpu_workers


def _resolve_worker_counts(args: argparse.Namespace) -> tuple[int, int]:
    auto_cpu, auto_gpu = _detect_auto_worker_counts()

    cpu_workers_arg = getattr(args, "cpu_workers", None)
    gpu_workers_arg = getattr(args, "gpu_workers", None)
    auto_requested = cpu_workers_arg is None or gpu_workers_arg is None
    if auto_requested:
        cpu_workers = auto_cpu if cpu_workers_arg is None else int(cpu_workers_arg)
        gpu_workers = auto_gpu if gpu_workers_arg is None else int(gpu_workers_arg)
        return cpu_workers, gpu_workers

    if cpu_workers_arg is None or gpu_workers_arg is None:
        return auto_cpu, auto_gpu
    return int(cpu_workers_arg), int(gpu_workers_arg)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PointStream on one input video chunk. Runtime artifacts are always written under "
            "outputs/<timestamp>/ in the project root."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON or YAML file with CLI defaults.",
    )
    parser.add_argument(
        "--input",
        dest="source_uri",
        default=None,
        help="Path to input video. If omitted, a synthetic mock source clip is generated.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process from the input video.",
    )
    parser.add_argument(
        "--summary-file",
        action="store_true",
        default=True,
        help="Write a summary JSON file in outputs/<timestamp> (enabled by default).",
    )
    parser.add_argument(
        "--ffmpeg-codec",
        type=str,
        default="libsvtav1",
        help=(
            "FFmpeg video encoder library used for all video outputs (decoded video, residuals, "
            "debug/mock clips). Default: libsvtav1."
        ),
    )
    parser.add_argument(
        "--codec-crf",
        type=int,
        default=None,
        help=(
            "FFmpeg codec quality setting (CRF - Constant Rate Factor). "
            "Lower values = higher quality. Range depends on codec (e.g., 0-63 for libx264/libx265/libsvtav1). "
            "If omitted, uses codec-specific defaults."
        ),
    )
    parser.add_argument(
        "--codec-preset",
        type=str,
        default=None,
        help=(
            "FFmpeg codec preset/speed setting. "
            "Common values: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow. "
            "For libsvtav1, use numeric values 0-13. If omitted, uses codec-specific defaults."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
        help="Runtime log verbosity level.",
    )
    parser.add_argument(
        "--dry-run",
        "--validate-only",
        dest="dry_run",
        action="store_true",
        help="Validate arguments and dependencies only; skip pipeline execution.",
    )
    parser.add_argument(
        "--transport",
        choices=("disk",),
        default="disk",
        help="Transport backend. Current implementation supports disk transport.",
    )
    parser.add_argument(
        "--execution-pool",
        choices=("inline", "tagged"),
        default="inline",
        help="DAG execution mode. 'tagged' uses the CPU/GPU tagged pool scaffold.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="CPU worker count for tagged execution pool. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="GPU worker count for tagged execution pool. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--actor-extractor",
        choices=("real", "mock"),
        default="real",
        help="Actor extraction module: real pipeline or lightweight mock module.",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo26", "yoloe"),
        default="yolo26",
        help="Player detector backend for real actor extractor.",
    )
    parser.add_argument(
        "--detector-caption",
        type=str,
        default="tennis player",
        help="Caption prompt for open-vocabulary detector backends such as YOLOE.",
    )
    parser.add_argument(
        "--pose-estimator",
        choices=("yolo26", "dwpose"),
        default="yolo26",
        help="Pose estimator backend: yolo26 (YOLO26n pose model from assets/weights) or dwpose (OpenMMPose).",
    )
    parser.add_argument(
        "--segmenter",
        choices=("yolo26", "yoloe", "sam3", "sam", "none"),
        default="yolo26",
        help=(
            "Segmentation backend: yolo26 (YOLO26n from assets/weights), "
            "yoloe (YOLO-E for open-vocab), sam3/sam (segment-anything), none (no segmentation)."
        ),
    )
    parser.add_argument(
        "--segmenter-caption",
        type=str,
        default="tennis player",
        help="Caption prompt for open-vocabulary segmenters such as YOLOE.",
    )
    parser.add_argument(
        "--payload-pose-delta-threshold",
        type=float,
        default=20.0,
        help="Pose keyframe delta threshold used by payload encoder.",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable optional debug artifacts.",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable optional debug artifacts.",
    )
    parser.set_defaults(debug=False)
    parser.add_argument(
        "--enable-shifted-ball",
        action="store_true",
        help="Stream actor frame states so ball extraction can start one frame earlier.",
    )
    parser.add_argument(
        "--ball-extractor",
        choices=("difference", "mock", "segmentation", "cascade"),
        default="difference",
        help="Ball extraction module: 'difference' (residual), 'segmentation' (detector), 'cascade' (detector then residual), or 'mock'.",
    )
    parser.add_argument(
        "--ball-difference-threshold",
        type=float,
        default=18.0,
        help="Pixel-difference threshold for real ball extractor.",
    )
    parser.add_argument(
        "--ball-min-blob-area",
        type=int,
        default=6,
        help="Minimum connected-component area for ball detection.",
    )
    parser.add_argument(
        "--ball-max-side",
        type=int,
        default=0,
        help="Optional max frame side for ball extraction (0 keeps native resolution).",
    )
    parser.add_argument(
        "--ball-det-conf",
        type=float,
        default=None,
        help="Detection confidence threshold for segmentation-based ball detector.",
    )
    parser.add_argument(
        "--ball-det-model",
        type=str,
        default=None,
        help="Model filename or path for segmentation-based ball detector.",
    )
    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=None,
        help="Class id to use for segmentation-based ball detector.",
    )
    parser.add_argument(
        "--reference-jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for actor reference crops.",
    )
    parser.add_argument(
        "--reference-padding-ratio",
        type=float,
        default=0.10,
        help="Padding ratio for actor reference crop extraction.",
    )
    parser.add_argument(
        "--importance-mapper",
        choices=("binary", "uniform"),
        default="binary",
        help=(
            "Residual weighting strategy (determines PER-PIXEL WEIGHT/PRECISION within residual regions): "
            "'binary' (weight 1.0 on actors, 0.0 elsewhere - low bandwidth), "
            "'uniform' (weight 1.0 everywhere - ablation/ground truth baseline)."
        ),
    )
    parser.add_argument(
        "--residual-background-downscale",
        type=_parse_optional_int_or_none,
        default=2,
        help=(
            "Optional downscale factor for background residual regions in full-video mode. "
            "Use 'None' to disable adaptive downscaling. Default: 2."
        ),
    )
    parser.add_argument(
        "--residual-batch-size",
        type=int,
        default=8,
        help="Number of frames to process per GPU batch in the residual calculator.",
    )
    parser.add_argument(
        "--downscale-interpolation",
        choices=("nearest", "bilinear", "bicubic", "area"),
        default="bilinear",
        help="Interpolation mode used when downscaling background residuals.",
    )
    parser.add_argument(
        "--residual-block-size",
        type=int,
        default=8,
        help="Block size used for residual activity pooling and block dropping.",
    )
    parser.add_argument(
        "--residual-block-threshold",
        type=float,
        default=0.0,
        help="Average absolute residual threshold per block in pixel units.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed used by synthesis and GenAI components.",
    )
    parser.add_argument(
        "--gpu-dtype",
        choices=("fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e5m2"),
        default=None,
        help="Global GPU compute dtype preference (falls back automatically if unsupported).",
    )
    parser.add_argument(
        "--panorama-warp-batch-size",
        type=int,
        default=4,
        help=(
            "Batch size for panorama warping operations during synthesis (performance tuning). "
            "Higher values trade memory for speed. Default: 4."
        ),
    )
    parser.add_argument(
        "--panorama-codec",
        choices=("jpeg", "png"),
        default=None,
        help="Codec used for panorama transport (jpeg or png).",
    )
    parser.add_argument(
        "--panorama-jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality used when --panorama-codec=jpeg.",
    )
    parser.add_argument(
        "--panorama-png-compression",
        type=int,
        default=None,
        help="PNG compression level used when --panorama-codec=png.",
    )
    parser.add_argument(
        "--allow-auto-model-download",
        action="store_true",
        help="Allow automatic model downloads when enabled (used by some backends).",
    )
    parser.add_argument(
        "--postgen-segmenter-model",
        type=str,
        default=None,
        help="Explicit model file for postgen segmenter backend.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("none", "psnr"),
        default="none",
        help="Post-run evaluation preset. 'psnr' computes framewise PSNR metrics.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Force-disable post-run evaluation regardless of evaluation-mode.",
    )
    parser.add_argument(
        "--evaluation-max-frames",
        type=int,
        default=None,
        help="Optional cap on frames used for post-pipeline PSNR evaluation.",
    )

    parser.add_argument(
        "--genai-backend",
        choices=("controlnet", "animate-anyone"),
        default=None,
        help="GenAI backend strategy. If omitted, GenAI is disabled.",
    )
    parser.add_argument(
        "--animate-anyone-repo-dir",
        default=None,
        help="Path to Moore-AnimateAnyone repository.",
    )
    parser.add_argument(
        "--animate-anyone-model-dir",
        default=None,
        help="Explicit AnimateAnyone model directory path.",
    )
    parser.add_argument(
        "--animate-anyone-steps",
        type=int,
        default=None,
        help="Number of diffusion inference steps for AnimateAnyone.",
    )
    parser.add_argument(
        "--animate-anyone-cfg",
        type=float,
        default=None,
        help="Guidance (CFG) scale for AnimateAnyone diffusion.",
    )
    parser.add_argument(
        "--animate-anyone-width",
        type=int,
        default=None,
        help="Width used by AnimateAnyone model (pixels).",
    )
    parser.add_argument(
        "--animate-anyone-height",
        type=int,
        default=None,
        help="Height used by AnimateAnyone model (pixels).",
    )
    parser.add_argument(
        "--animate-anyone-window",
        type=int,
        default=None,
        help="Temporal conditioning window length for AnimateAnyone decode compositing.",
    )
    parser.add_argument(
        "--genai-preroll-frames",
        type=int,
        default=None,
        help="Frames to keep residual-only before temporal GenAI compositing starts.",
    )
    parser.add_argument(
        "--genai-keyframe-only",
        action="store_true",
        help="Run GenAI only on received keyframe skeletons and interpolate generated frames locally.",
    )
    parser.add_argument(
        "--animate-anyone-transparent-threshold",
        type=int,
        default=None,
        help="Black-background alpha threshold for AnimateAnyone compositing.",
    )
    parser.add_argument(
        "--genai-resize-mode",
        choices=("plain", "aspect-recovery"),
        default=None,
        help="Resize mode for GenAI actor placement in the decode ROI.",
    )
    parser.add_argument(
        "--animate-anyone-adaptive-threshold",
        dest="animate_anyone_adaptive_threshold",
        action="store_true",
        help="Enable adaptive border-threshold masking for AnimateAnyone black backgrounds.",
    )
    parser.set_defaults(animate_anyone_adaptive_threshold=False)
    parser.add_argument(
        "--animate-anyone-alpha-smoothing",
        type=float,
        default=None,
        help="Temporal smoothing factor for AnimateAnyone alpha masks in range [0, 1].",
    )
    parser.add_argument(
        "--compositing-mask-mode",
        choices=("alpha-heuristic", "metadata-source-mask", "postgen-seg-client"),
        default="postgen-seg-client",
        help=(
            "Actor compositing alpha strategy: heuristic alpha extraction, source mask metadata, "
            "or post-generation client segmentation."
        ),
    )
    parser.add_argument(
        "--postgen-segmenter-backend",
        choices=("yolo", "heuristic"),
        default=None,
        help="Segmentation backend when --compositing-mask-mode=postgen-seg-client.",
    )
    parser.add_argument(
        "--metadata-mask-codec",
        choices=("auto", "rle-v1", "bitpack-z1", "png", "segmenter-native", "yolo-native"),
        default=None,
        help=(
            "Compression codec for metadata-source-mask payloads. "
            "Use segmenter-native/yolo-native to transport contour polygons when available."
        ),
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv_list)

    config_overrides: dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config_overrides = _load_config_overrides(bootstrap_args.config)
    if config_overrides:
        parser.set_defaults(**config_overrides)

    args = parser.parse_args(argv_list)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cpu_workers, gpu_workers = _resolve_worker_counts(args)
    args.cpu_workers = int(cpu_workers)
    args.gpu_workers = int(gpu_workers)

    if args.num_frames is not None and args.num_frames <= 0:
        raise ValueError("--num-frames must be a positive integer")
    if args.cpu_workers <= 0:
        raise ValueError("--cpu-workers must be a positive integer")
    if args.gpu_workers <= 0:
        raise ValueError("--gpu-workers must be a positive integer")
    if args.ball_min_blob_area <= 0:
        raise ValueError("--ball-min-blob-area must be a positive integer")
    if args.ball_max_side < 0:
        raise ValueError("--ball-max-side must be a non-negative integer")
    if args.reference_jpeg_quality <= 0 or args.reference_jpeg_quality > 100:
        raise ValueError("--reference-jpeg-quality must be in range 1..100")
    if args.reference_padding_ratio < 0.0 or args.reference_padding_ratio >= 1.0:
        raise ValueError("--reference-padding-ratio must be in range [0.0, 1.0)")
    if args.genai_preroll_frames is not None and args.genai_preroll_frames < 0:
        raise ValueError("--genai-preroll-frames must be a non-negative integer")
    if args.residual_background_downscale is not None and args.residual_background_downscale < 2:
        raise ValueError("--residual-background-downscale must be >= 2, or None to disable")
    if args.animate_anyone_alpha_smoothing is not None and not (0.0 <= args.animate_anyone_alpha_smoothing <= 1.0):
        raise ValueError("--animate-anyone-alpha-smoothing must be in range [0.0, 1.0]")
    if not str(args.ffmpeg_codec).strip():
        raise ValueError("--ffmpeg-codec must not be empty")
    ensure_ffmpeg_encoder_available(str(args.ffmpeg_codec))

    source_uri: str | None = args.source_uri
    if source_uri is not None:
        source_path = Path(source_uri).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source_path}")
        source_uri = str(source_path)

    debug_enabled = bool(args.debug)
    _apply_runtime_env_overrides(args)

    if bool(args.dry_run):
        dry_run_summary: dict[str, Any] = {
            "status": "dry-run",
            "config": str(args.config) if args.config is not None else None,
            "execution_pool": str(args.execution_pool),
            "cpu_workers": int(args.cpu_workers),
            "gpu_workers": int(args.gpu_workers),
            "debug": bool(args.debug),
            "evaluation_mode": str(args.evaluation_mode),
            "source_uri": source_uri,
        }
        print(json.dumps(dry_run_summary, indent=2))
        return 0

    run_output_root = _create_timestamped_output_dir(base_root=_project_root() / "outputs")
    chunk_id = "0001"

    execution_pool = _build_execution_pool(
        mode=str(args.execution_pool),
        cpu_workers=int(args.cpu_workers),
        gpu_workers=int(args.gpu_workers),
    )
    actor_extractor = _build_actor_extractor(
        mode=str(args.actor_extractor),
        detector=str(args.detector),
        detector_caption=str(args.detector_caption),
        pose_estimator=str(args.pose_estimator),
        segmenter=str(args.segmenter),
        segmenter_caption=str(args.segmenter_caption),
        debug_enabled=debug_enabled,
        pose_delta_threshold=float(args.payload_pose_delta_threshold),
        compositing_mask_mode=str(args.compositing_mask_mode),
        metadata_mask_codec=str(args.metadata_mask_codec or "auto"),
    )
    ball_extractor = _build_ball_extractor(
        mode=str(args.ball_extractor),
        difference_threshold=float(args.ball_difference_threshold),
        min_blob_area=int(args.ball_min_blob_area),
        detection_max_side=int(args.ball_max_side),
    )
    reference_extractor = _build_reference_extractor(
        jpeg_quality=int(args.reference_jpeg_quality),
        padding_ratio=float(args.reference_padding_ratio),
    )
    residual_calculator = _build_residual_calculator(
        seed=int(args.seed),
        importance_mapper=str(args.importance_mapper),
        background_block_downscale_factor=args.residual_background_downscale,
        residual_batch_size=int(args.residual_batch_size),
        downscale_interpolation=str(args.downscale_interpolation),
        residual_block_size=int(args.residual_block_size),
        block_information_threshold=float(args.residual_block_threshold),
    )

    run_summary = run_mock_pipeline(
        transport_root=run_output_root,
        execution_pool=execution_pool,
        source_uri=source_uri,
        num_frames=args.num_frames,
        actor_extractor=actor_extractor,
        ball_extractor=ball_extractor,
        reference_extractor=reference_extractor,
        residual_calculator=residual_calculator,
        transport_backend=str(args.transport),
        chunk_id=chunk_id,
        runtime_output_root=run_output_root,
    )

    should_evaluate = str(args.evaluation_mode).strip().lower() != "none" and not bool(args.skip_eval)
    if should_evaluate:
        run_summary["evaluation"] = evaluate_run_summary(
            summary=run_summary,
            experiment_dir=run_output_root,
            max_frames=args.evaluation_max_frames,
        )

    summary_json = json.dumps(run_summary, indent=2)
    print(summary_json)

    if bool(args.summary_file):
        summary_path = run_output_root / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{summary_json}\n", encoding="utf-8")

    return 0


def _ensure_mock_source_video(runtime_output_root: str | Path | None = None) -> str:
    run_root = (
        Path(runtime_output_root).expanduser()
        if runtime_output_root is not None
        else _create_timestamped_output_dir(base_root=_project_root() / "outputs")
    )
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
        codec=os.environ.get("POINTSTREAM_FFMPEG_CODEC", "libsvtav1"),
        pix_fmt="yuv420p",
        crf=int(os.environ.get("POINTSTREAM_CODEC_CRF", "18")),
        preset=os.environ.get("POINTSTREAM_CODEC_PRESET", "veryfast"),
    )
    return str(source_path)


if __name__ == "__main__":
    raise SystemExit(run_cli())
