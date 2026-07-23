from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Optional

try:
    import importlib
    yaml: Any = importlib.import_module("yaml")
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


@dataclass
class PointstreamConfig:
    # Core Settings
    source_uri: Optional[str] = None
    num_frames: Optional[int] = None
    summary_file: bool = True
    ffmpeg_codec: str = "libsvtav1"
    codec_crf: Optional[int] = 35
    codec_preset: Optional[str] = "slow"
    log_level: str = "debug"
    transport: str = "disk"

    # Full-Match Orchestration (report 10 Phase 2)
    # "chunk" (default): existing single-VideoChunk behavior, source-uri is
    # one pre-trimmed clip. "full_match": source-uri is a full raw_4k match;
    # src.encoder.match_orchestrator splits it into scenes and routes each
    # (point -> semantic pipeline in scene-chunk-duration-sec sub-chunks,
    # interlude/other/blank -> fallback codec directly).
    run_mode: str = "chunk"
    scene_chunk_duration_sec: float = 2.0

    # Execution Pool
    execution_pool: str = "inline"
    cpu_workers: Optional[int] = None
    gpu_workers: Optional[int] = None

    # Actor Extraction Pipeline
    detector: str = "yolo26n.pt"
    target_class_caption: str = "tennis player"
    pose_estimator: str = "yolo26n-pose.pt"
    segmenter: str = "yolo26n-seg.pt"
    payload_pose_delta_threshold: float = 20.0

    # Ball Extraction Configuration
    ball_extractor: str = "difference"
    ball_difference_threshold: float = 18.0
    ball_min_blob_area: int = 6
    ball_max_side: int = 0
    ball_det_conf: Optional[float] = None
    ball_det_model: Optional[str] = None
    ball_class_id: Optional[int] = None

    # Reference Crop Configuration
    reference_jpeg_quality: int = 95
    reference_padding_ratio: float = 0.10

    # Residual and Importance Mapping
    importance_mapper: str = "uniform"
    residual_background_downscale: Optional[int] = 2
    residual_batch_size: int = 8
    downscale_interpolation: str = "bilinear"
    residual_block_size: int = 8
    residual_block_threshold: float = 0.0
    residual_pix_fmt: str = "yuv444p"
    seed: int = 1337

    # GPU and Performance Tuning
    gpu_dtype: str = "fp16"
    panorama_warp_batch_size: Optional[int] = None
    panorama_codec: str = "jpeg"
    panorama_jpeg_quality: int = 50
    panorama_png_compression: Optional[int] = None

    # Background-layer ladder (report 10 Phase 5.3):
    # "panorama-static" (default, rung 1): today's behavior, one full
    #   panorama per chunk via `panorama_codec`.
    # "panorama-delta" (rung 2): the first sub-chunk of a scene
    #   (`VideoChunk.scene_id`) sends a full panorama; later sub-chunks of
    #   the same scene send a delta against the previous one. No-op (same
    #   bytes as rung 1) when scene_id is unset -- only
    #   `src.encoder.match_orchestrator`'s point-scene sub-chunk loop sets it.
    # "roi-video" (rung 3): still one image per chunk, but encoded via
    #   `panorama_roi_*` (libx264 + addroi) instead of `panorama_codec`.
    background_layer: str = "panorama-static"
    panorama_roi_crf: Optional[int] = 30
    panorama_roi_preset: Optional[str] = "veryfast"

    # Model and Dependencies
    allow_auto_model_download: bool = True
    postgen_segmenter_model: Optional[str] = None

    # Evaluation Settings
    evaluation_mode: list[str] = field(default_factory=lambda: ["psnr", "ssim", "vmaf"])
    evaluation_max_frames: Optional[int] = None

    # GenAI and AnimateAnyone Configuration
    genai_backend: Optional[str] = None
    animate_anyone_repo_dir: Optional[str] = None
    animate_anyone_model_dir: Optional[str] = None
    animate_anyone_steps: int = 10
    animate_anyone_cfg: float = 7.5
    animate_anyone_width: int = 512
    animate_anyone_height: int = 512
    animate_anyone_window: Optional[int] = None
    genai_preroll_frames: int = 0
    genai_keyframe_only: bool = False
    animate_anyone_transparent_threshold: int = 64
    genai_resize_mode: str = "aspect-recovery"
    animate_anyone_adaptive_threshold: bool = True
    animate_anyone_alpha_smoothing: float = 0.5
    # Overrides the strategy's primary weight path so evaluation can score an
    # arbitrary campaign checkpoint through the decoder's own strategy classes.
    # Unset in normal pipeline runs; set by scripts/eval_checkpoint.py.
    genai_checkpoint_override: Optional[str] = None

    # ControlNet Configuration
    controlnet_id: Optional[str] = None
    controlnet_width: int = 512
    controlnet_height: int = 512
    controlnet_steps: int = 20
    controlnet_strength: float = 0.65
    controlnet_strength_temporal: float = 0.35
    controlnet_temporal_keyframe_interval: int = 8
    controlnet_temporal_strength_min: float = 0.30
    controlnet_temporal_strength_max: float = 0.55
    controlnet_temporal_flow_scale: float = 0.02
    controlnet_cfg: float = 7.0
    ip_adapter_scale: float = 0.5

    # Compositing and Masking
    compositing_mask_mode: str = "postgen-seg-client"
    postgen_segmenter_backend: str = "yolo"
    metadata_mask_codec: str = "auto"
    
    # Optimizations and Debug
    enable_shifted_ball: bool = False
    debug_artifact_dir: Optional[str] = None
    disable_debug_artifacts: bool = False
    runtime_output_dir: Optional[str] = None

    def __post_init__(self):
        # Automate debug artifact behavior via log_level
        if self.log_level.strip().lower() == "debug":
            # Dataclasses are frozen by default if frozen=True, but PointstreamConfig is not frozen=True.
            self.disable_debug_artifacts = False
        elif not getattr(self, "_explicit_debug_flag", False):
            # If log-level is not debug, disable debug artifacts unless explicitly enabled (simplification: just set True if not debug)
            self.disable_debug_artifacts = True
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointstreamConfig":
        normalized = {k.replace("-", "_"): v for k, v in data.items()}
        
        # Legacy/Alias remapping
        if "input" in normalized and "source_uri" not in normalized:
            normalized["source_uri"] = normalized.pop("input")
            
        if "detector_caption" in normalized and "target_class_caption" not in normalized:
            normalized["target_class_caption"] = normalized.pop("detector_caption")
            
        if "segmenter_caption" in normalized and "target_class_caption" not in normalized:
            normalized["target_class_caption"] = normalized.pop("segmenter_caption")
            
        # Parse list types manually (dataclasses don't auto-cast)
        if "evaluation_mode" in normalized:
            val = normalized["evaluation_mode"]
            if isinstance(val, str):
                if val.lower() == "none" or not val:
                    normalized["evaluation_mode"] = []
                else:
                    normalized["evaluation_mode"] = [v.strip().lower() for v in val.split(",")]
                    
        # Typecasting based on field definitions
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for k, v in normalized.items():
            if k in field_types:
                kwargs[k] = v
                
        return cls(**kwargs)

def load_config(config_path: str | Path, cli_overrides: Optional[dict[str, Any]] = None) -> PointstreamConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    suffix = path.suffix.strip().lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        if not HAS_YAML:
            raise RuntimeError("YAML config requires PyYAML. Install dependency: pyyaml")
        payload = yaml.safe_load(raw_text)
    else:
        raise ValueError("Config supports only .json, .yaml, and .yml files")

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping/object")

    # Override YAML values with any passed CLI overrides (if they are not None)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                payload[k] = v

    return PointstreamConfig.from_dict(payload)
