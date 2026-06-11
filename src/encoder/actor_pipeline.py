from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import (
    ActorPacket,
    FrameState,
    VideoChunk,
)
from src.shared.tags import gpu_bound


@dataclass(frozen=True)
class ActorExtractionResult:
    frame_states: list[FrameState]
    actor_packets: list[ActorPacket]
    profile: dict[str, float] = field(default_factory=dict)


class ActorExtractor:
    def __init__(
        self,
        detector_model: Any | None = None,
        segmenter_model: Any | None = None,
        pose_model: Any | None = None,
        render_debug_keyframes: bool = True,
        detector_backend: str = "yolo26n.pt",
        detector_caption: str = "tennis player",
        pose_backend: str = "yolo26n-pose.pt",
        segmenter_backend: str = "yolo26n-seg.pt",
        segmenter_caption: str = "tennis player",
        pose_delta_threshold: float = 20.0,
        include_mask_metadata: bool = False,
        metadata_mask_codec: str = "auto",
        dwpose_device: str = "cuda",
        config: Any | None = None,
    ) -> None:
        from src.encoder.actor_components import (
            BaseDetector,
            BasePoseEstimator,
            BaseSegmenter,
            CannySegmenter,
            DwposeEstimator,
            PayloadEncoder,
            PipelineBuilder,
            SamSegmenter,
            StandardTennisHeuristic,
            YoloEDetector,
            Yolo26Detector,
            YoloPoseEstimator,
            YoloeSegmenter,
            YoloSegmenter,
        )

        self._render_debug_keyframes = render_debug_keyframes
        self._config = config
        
        genai_backend = getattr(config, "genai_backend", "") if config else ""
        if genai_backend is None:
            genai_backend = ""

        caption_list = [part.strip() for part in str(detector_caption).split(",") if part.strip()]
        if not caption_list:
            caption_list = ["tennis player"]

        detector: BaseDetector
        normalized_detector_backend = detector_backend.strip().lower()
        if "yoloe" in normalized_detector_backend:
            detector = YoloEDetector(
                model_name=detector_backend.strip(),
                model=detector_model,
                captions=caption_list,
            )
        elif "yolo" in normalized_detector_backend:
            detector = Yolo26Detector(model_name=detector_backend.strip(), model=detector_model)
        else:
            raise ValueError(f"Unsupported detector backend: {detector_backend}")

        pose_estimator: BasePoseEstimator | None
        normalized_pose_backend = pose_backend.strip().lower()
        if "seg-controlnet" in genai_backend or "canny-controlnet" in genai_backend:
            # Exclusivity: Seg and Canny don't use pose estimator
            pose_estimator = None
        elif normalized_pose_backend in {"none", ""}:
            pose_estimator = None
        elif "yolo" in normalized_pose_backend:
            pose_estimator = YoloPoseEstimator(model_name=pose_backend.strip(), model=pose_model)
        elif "dwpose" in normalized_pose_backend:
            pose_estimator = DwposeEstimator(torchscript_device=dwpose_device)
        else:
            raise ValueError(f"Unsupported pose backend: {pose_backend}")

        segmenter: BaseSegmenter | None
        normalized_segmenter_backend = segmenter_backend.strip().lower()
        if "canny-controlnet" in genai_backend:
            # Exclusivity: Canny uses CannySegmenter, disables YOLO segmenter
            low = getattr(config, "canny_lower_threshold", "auto") if config else "auto"
            high = getattr(config, "canny_upper_threshold", "auto") if config else "auto"
            segmenter = CannySegmenter(lower_threshold=low, upper_threshold=high)
            
            # Since Canny is replacing the mask metadata for ControlNet, make sure mask is transmitted
            include_mask_metadata = True
        elif normalized_segmenter_backend in {"none", ""}:
            segmenter = None
        elif "yoloe" in normalized_segmenter_backend:
            segmenter_caption_list = [part.strip() for part in str(segmenter_caption).split(",") if part.strip()]
            if not segmenter_caption_list:
                segmenter_caption_list = ["tennis player"]
            segmenter = YoloeSegmenter(
                model_name=segmenter_backend.strip(),
                model=segmenter_model,
                captions=segmenter_caption_list,
            )
        elif "yolo" in normalized_segmenter_backend:
            segmenter = YoloSegmenter(model_name=segmenter_backend.strip(), model=segmenter_model)
        elif "sam" in normalized_segmenter_backend:
            segmenter = SamSegmenter(model_name=segmenter_backend.strip(), model=segmenter_model)
        else:
            raise ValueError(f"Unsupported segmenter backend: {segmenter_backend}")

        if "seg-controlnet" in genai_backend:
            # Ensure mask metadata is transmitted for seg controlnet
            include_mask_metadata = True

        # Models are loaded once in component initialization and reused frame-by-frame.
        self._pipeline = PipelineBuilder(
            detector=detector,
            heuristic=StandardTennisHeuristic(),
            segmenter=segmenter,  # type: ignore[arg-type]
            pose_estimator=pose_estimator,  # type: ignore[arg-type]
            payload_encoder=PayloadEncoder(
                pose_delta_threshold=float(pose_delta_threshold),
                include_mask_metadata=bool(include_mask_metadata),
                metadata_mask_codec=str(metadata_mask_codec),
            ),
        )

    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        return self.process_with_states(chunk).actor_packets

    @gpu_bound
    def process_with_states(self, chunk: VideoChunk) -> ActorExtractionResult:
        frames_bgr = self._load_frames(chunk)
        frame_states, packets = self._pipeline.run(chunk=chunk, frames_bgr=frames_bgr)
        if self._render_debug_keyframes:
            self._pipeline.render_debug_keyframes(
                chunk=chunk,
                frames_bgr=frames_bgr,
                frame_states=frame_states,
                actor_packets=packets,
                out_dir=self._resolve_debug_keyframes_dir(),
            )
        return ActorExtractionResult(
            frame_states=frame_states,
            actor_packets=packets,
            profile=self._pipeline.get_timings(),
        )

    @gpu_bound
    def process_with_states_streaming(
        self,
        chunk: VideoChunk,
        on_frame_state: Any | None = None,
    ) -> ActorExtractionResult:
        frames_bgr = self._load_frames(chunk)
        filtered_states: list[FrameState] = []
        for state in self._pipeline.iter_filtered_states(chunk=chunk, frames_bgr=frames_bgr):
            filtered_states.append(state)
            if on_frame_state is not None:
                on_frame_state(state)

        with self._pipeline.profiler.stage("metadata_generation"):
            packets = self._pipeline.payload_encoder.encode(chunk=chunk, frame_states=filtered_states)
            
        if self._render_debug_keyframes:
            self._pipeline.render_debug_keyframes(
                chunk=chunk,
                frames_bgr=frames_bgr,
                frame_states=filtered_states,
                actor_packets=packets,
                out_dir=self._resolve_debug_keyframes_dir(),
            )
        return ActorExtractionResult(
            frame_states=filtered_states,
            actor_packets=packets,
            profile=self._pipeline.get_timings(),
        )

    def _resolve_debug_keyframes_dir(self) -> Path:
        override = getattr(self._config, "debug_artifact_dir", None)
        if override:
            return Path(override) / "debug_actors"

        project_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return project_root / "outputs" / timestamp / "debug" / "debug_actors"

    def _load_frames(self, chunk: VideoChunk) -> list[np.ndarray]:
        source = Path(chunk.source_uri)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"ActorExtractor source video not found: {source}")

        metadata = probe_video_metadata(source)
        frames: list[np.ndarray] = []
        for frame in iter_video_frames_ffmpeg(
            source,
            width=metadata.width,
            height=metadata.height,
        ):
            frames.append(frame)
            if len(frames) >= chunk.num_frames:
                break
        if not frames:
            raise ValueError(f"ActorExtractor decoded zero frames from source: {source}")

        return frames



