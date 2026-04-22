from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.encoder.ball_extractor import BallExtractor
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
from src.shared.schemas import VideoChunk
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

    encoder = EncoderPipeline(
        execution_pool=execution_pool,
        actor_extractor=actor_extractor,
        ball_extractor=ball_extractor,
        reference_extractor=reference_extractor,
        residual_calculator=residual_calculator,
    )

    previous_debug_artifact_dir = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
    previous_cwd = Path.cwd()
    os.chdir(resolved_runtime_root)
    os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = str(resolved_runtime_root / "debug")
    try:
        payload = encoder.encode_chunk(chunk)
    finally:
        encoder.shutdown()
        if previous_debug_artifact_dir is None:
            os.environ.pop("POINTSTREAM_DEBUG_ARTIFACT_DIR", None)
        else:
            os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = previous_debug_artifact_dir
        os.chdir(previous_cwd)

    normalized_transport = transport_backend.strip().lower()
    if normalized_transport != "disk":
        raise ValueError(f"Unsupported transport backend: {transport_backend}")

    transport = DiskTransport(root_dir=resolved_transport_root)
    transport.send(payload)
    received_payload = transport.receive(chunk.chunk_id)

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
    decoded = decoder.process(received_payload)

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

    summary = {
        "chunk_id": received_payload.chunk.chunk_id,
        "run_output_root": str(resolved_transport_root),
        "num_actor_packets": len(received_payload.actors),
        "num_rigid_object_packets": len(received_payload.rigid_objects),
        "ball_object_id": received_payload.ball.object_id,
        "residual_uri": received_payload.residual.residual_video_uri,
        "decoded_uri": decoded.output_uri,
        "transport_backend": normalized_transport,
        "source_size_bytes": source_size_bytes,
        "metadata_size_bytes": metadata_size_bytes,
        "actor_reference_size_bytes": actor_reference_size_bytes,
        "residual_size_bytes": residual_size_bytes,
        "panorama_size_bytes": panorama_size_bytes,
        "transport_total_size_bytes": transport_total_size_bytes,
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
    disable_debug_keyframes: bool,
    pose_delta_threshold: float,
    compositing_mask_mode: str,
    metadata_mask_codec: str,
) -> ActorExtractor | MockActorExtractor | None:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "mock":
        return MockActorExtractor()

    normalized_mask_mode = compositing_mask_mode.strip().lower()
    include_mask_metadata = normalized_mask_mode == "metadata-source-mask"

    uses_default = (
        detector == "yolo26"
        and detector_caption.strip().lower() == "tennis player"
        and pose_estimator == "yolo"
        and segmenter == "yolo"
        and segmenter_caption.strip().lower() == "tennis player"
        and not disable_debug_keyframes
        and float(pose_delta_threshold) == 20.0
        and not include_mask_metadata
    )
    if uses_default:
        return None

    return ActorExtractor(
        render_debug_keyframes=not disable_debug_keyframes,
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


def _build_residual_calculator(seed: int, importance_mapper: str) -> ResidualCalculator | None:
    uses_default = int(seed) == 1337 and importance_mapper.strip().lower() == "binary"
    if uses_default:
        return None

    mapper = _build_importance_mapper(importance_mapper)
    return ResidualCalculator(
        synthesis_engine=SynthesisEngine(seed=int(seed)),
        importance_mapper=mapper,
    )


def _apply_runtime_env_overrides(args: argparse.Namespace) -> None:
    if bool(args.enable_genai):
        os.environ["POINTSTREAM_ENABLE_GENAI"] = "1"
    if bool(args.disable_genai):
        os.environ["POINTSTREAM_ENABLE_GENAI"] = "0"

    if args.genai_backend is not None:
        os.environ["POINTSTREAM_GENAI_BACKEND"] = str(args.genai_backend)
    if args.animate_anyone_repo_dir is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_REPO_DIR"] = str(Path(args.animate_anyone_repo_dir).expanduser())
    if args.animate_anyone_model_variant is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_MODEL_VARIANT"] = str(args.animate_anyone_model_variant)
    if args.animate_anyone_model_dir is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR"] = str(Path(args.animate_anyone_model_dir).expanduser())
    if args.animate_anyone_window is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_WINDOW"] = str(int(args.animate_anyone_window))
    if args.genai_preroll_frames is not None:
        os.environ["POINTSTREAM_GENAI_PREROLL_FRAMES"] = str(int(args.genai_preroll_frames))
    if args.animate_anyone_transparent_threshold is not None:
        os.environ["POINTSTREAM_ANIMATE_ANYONE_TRANSPARENT_THRESHOLD"] = str(
            int(args.animate_anyone_transparent_threshold)
        )
    if args.gpu_dtype is not None:
        os.environ["POINTSTREAM_GPU_DTYPE"] = str(args.gpu_dtype)
    if args.ball_max_side is not None:
        os.environ["POINTSTREAM_BALL_MAX_SIDE"] = str(int(args.ball_max_side))
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PointStream on one input video chunk. Runtime artifacts are always written under "
            "outputs/<timestamp>/ in the project root."
        ),
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
        "--no-summary-file",
        action="store_true",
        help="Do not write a summary JSON file in outputs/<timestamp>; print summary to stdout only.",
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
        default=1,
        help="CPU worker count for tagged execution pool.",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help="GPU worker count for tagged execution pool.",
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
        choices=("yolo", "dwpose"),
        default="yolo",
        help="Pose estimator backend for real actor extractor.",
    )
    parser.add_argument(
        "--segmenter",
        choices=("yolo", "yoloe", "sam3", "sam", "none"),
        default="yolo",
        help="Segmentation backend for real actor extractor.",
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
    parser.add_argument(
        "--disable-debug-keyframes",
        action="store_true",
        help="Disable actor keyframe skeleton debug image generation.",
    )
    parser.add_argument(
        "--ball-extractor",
        choices=("difference", "mock"),
        default="difference",
        help="Ball extraction module: background-difference extractor or lightweight mock tracker.",
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
        help="Residual weighting strategy.",
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

    genai_group = parser.add_mutually_exclusive_group()
    genai_group.add_argument(
        "--enable-genai",
        action="store_true",
        help="Enable heavy GenAI compositor path for decoding.",
    )
    genai_group.add_argument(
        "--disable-genai",
        action="store_true",
        help="Force lightweight mock compositor path.",
    )
    parser.add_argument(
        "--genai-backend",
        choices=("controlnet", "animate-anyone"),
        default=None,
        help="GenAI backend strategy when GenAI is enabled.",
    )
    parser.add_argument(
        "--animate-anyone-repo-dir",
        default=None,
        help="Path to Moore-AnimateAnyone repository.",
    )
    parser.add_argument(
        "--animate-anyone-model-variant",
        default=None,
        help="AnimateAnyone model variant alias (for example: original, finetuned_tennis).",
    )
    parser.add_argument(
        "--animate-anyone-model-dir",
        default=None,
        help="Explicit AnimateAnyone model directory path.",
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
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument(
        "--animate-anyone-adaptive-threshold",
        dest="animate_anyone_adaptive_threshold",
        action="store_true",
        help="Enable adaptive border-threshold masking for AnimateAnyone black backgrounds.",
    )
    adaptive_group.add_argument(
        "--disable-animate-anyone-adaptive-threshold",
        dest="animate_anyone_adaptive_threshold",
        action="store_false",
        help="Disable adaptive border-threshold masking for AnimateAnyone black backgrounds.",
    )
    parser.set_defaults(animate_anyone_adaptive_threshold=None)
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
    args = parser.parse_args(argv)

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
    if args.animate_anyone_alpha_smoothing is not None and not (0.0 <= args.animate_anyone_alpha_smoothing <= 1.0):
        raise ValueError("--animate-anyone-alpha-smoothing must be in range [0.0, 1.0]")

    source_uri: str | None = args.source_uri
    if source_uri is not None:
        source_path = Path(source_uri).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source_path}")
        source_uri = str(source_path)

    _apply_runtime_env_overrides(args)

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
        disable_debug_keyframes=bool(args.disable_debug_keyframes),
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
    )

    summary = run_mock_pipeline(
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

    summary_json = json.dumps(summary, indent=2)
    print(summary_json)

    if not bool(args.no_summary_file):
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
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )
    return str(source_path)


if __name__ == "__main__":
    raise SystemExit(run_cli())
