from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.decoder.genai_compositor import DiffusersCompositor
from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.mask_codec import decode_binary_mask
from src.shared.schemas import ActorPacket, EncodedChunkPayload, FrameState, ResidualPacket, SceneActor, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound
from src.shared.track_id import scene_track_id_to_int
from src.shared.torch_dtype import is_cuda_device_usable


@dataclass(frozen=True)
class _DecodedActorMaskFrame:
    bbox: tuple[int, int, int, int]
    mask_gray: np.ndarray


@dataclass(frozen=True)
class _ServerActorState:
    object_id: str
    reference_crop_tensor: torch.Tensor
    dense_pose_tensor: torch.Tensor
    metadata_masks_by_frame: dict[int, _DecodedActorMaskFrame]


class BaseImportanceMapper(ABC):
    """Strategy interface for per-frame saliency weighting."""

    @abstractmethod
    def build_importance_map(
        self,
        frame_state: FrameState,
        frame_height: int,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return shape [Height, Width] with values in [0.0, 1.0]."""
        raise NotImplementedError


class BinaryActorImportanceMapper(BaseImportanceMapper):
    """Baseline binary saliency map from player/racket segmentation masks."""

    def __init__(self, target_classes: set[str] | None = None) -> None:
        self._target_classes = target_classes or {"player", "racket"}

    def build_importance_map(
        self,
        frame_state: FrameState,
        frame_height: int,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        importance = torch.zeros((frame_height, frame_width), dtype=torch.float32, device=device)

        for actor in frame_state.actors:
            if actor.class_name not in self._target_classes:
                continue
            actor_mask = self._paste_actor_mask(actor=actor, frame_height=frame_height, frame_width=frame_width, device=device)
            if actor_mask is None:
                continue
            importance = torch.maximum(importance, actor_mask)

        return importance.clamp(0.0, 1.0)

    def _paste_actor_mask(
        self,
        actor: SceneActor,
        frame_height: int,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if actor.mask is None:
            return None

        x1, y1, x2, y2 = self._clip_bbox(actor.bbox, frame_width=frame_width, frame_height=frame_height)
        if x2 <= x1 or y2 <= y1:
            return None

        raw_mask = np.asarray(actor.mask, dtype=np.float32)
        if raw_mask.ndim != 2 or raw_mask.size == 0:
            return None

        mask_tensor = torch.from_numpy(raw_mask).to(device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        resized_mask = F.interpolate(
            mask_tensor,
            size=(y2 - y1, x2 - x1),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        resized_mask = resized_mask.clamp(0.0, 1.0)

        pasted = torch.zeros((frame_height, frame_width), dtype=torch.float32, device=device)
        pasted[y1:y2, x1:x2] = torch.maximum(pasted[y1:y2, x1:x2], resized_mask)
        return pasted

    def _clip_bbox(self, bbox: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(y2))))
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2


class UniformImportanceMapper(BaseImportanceMapper):
    """Ablation mapper that applies full residual weight everywhere."""

    def build_importance_map(
        self,
        frame_state: FrameState,
        frame_height: int,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        _ = frame_state
        return torch.ones((frame_height, frame_width), dtype=torch.float32, device=device)


class ResidualCalculator:
    """Server-side weighted residual calculator."""

    def __init__(
        self,
        synthesis_engine: SynthesisEngine | None = None,
        importance_mapper: BaseImportanceMapper | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self._synthesis_engine = synthesis_engine or SynthesisEngine()
        if device is None:
            self._device = torch.device(self._synthesis_engine.device)
        else:
            self._device = torch.device(device)
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")
        self._importance_mapper = importance_mapper or BinaryActorImportanceMapper()

    @gpu_bound
    def process(
        self,
        chunk: VideoChunk,
        payload: EncodedChunkPayload,
        frame_states: list[FrameState],
        debug_output_path: str | Path | None = None,
    ) -> ResidualPacket:
        predicted_frames = self._synthesis_engine.synthesize(payload, include_guidance_overlays=False).frames_bgr
        if self._is_genai_enabled():
            predicted_frames = self._render_server_genai_prediction(payload=payload, frame_tensor=predicted_frames)

        source_metadata = probe_video_metadata(chunk.source_uri)
        available_source_frames = max(0, int(source_metadata.num_frames) - int(chunk.start_frame_id))

        valid_frames = min(
            int(chunk.num_frames),
            int(predicted_frames.shape[0]),
            int(available_source_frames),
        )
        if valid_frames <= 0:
            raise ValueError("ResidualCalculator received zero valid frames")

        output_path = Path(debug_output_path) if debug_output_path is not None else self._default_residual_path(chunk)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        source_iter = iter_video_frames_ffmpeg(
            chunk.source_uri,
            width=int(chunk.width),
            height=int(chunk.height),
        )

        for _ in range(int(chunk.start_frame_id)):
            try:
                next(source_iter)
            except StopIteration:
                raise ValueError("ResidualCalculator could not seek to chunk start frame in source video")

        def _iter_encoded_frames() -> Iterator[np.ndarray]:
            for frame_idx in range(valid_frames):
                try:
                    original_np = next(source_iter)
                except StopIteration:
                    break

                frame_state = self._select_frame_state(frame_states=frame_states, frame_idx=frame_idx)
                importance_map = self._importance_mapper.build_importance_map(
                    frame_state=frame_state,
                    frame_height=int(chunk.height),
                    frame_width=int(chunk.width),
                    device=self._device,
                )

                original_tensor = (
                    torch.from_numpy(np.asarray(original_np, dtype=np.uint8))
                    .permute(2, 0, 1)
                    .to(self._device, dtype=torch.float32)
                )
                predicted_tensor = predicted_frames[frame_idx].to(self._device, dtype=torch.float32)

                # Shape: [Channels, Height, Width]
                raw_diff = original_tensor - predicted_tensor
                masked_diff = raw_diff * importance_map.unsqueeze(0)
                encoded_residual = torch.clamp(masked_diff + 128.0, 0.0, 255.0).to(torch.uint8)

                yield np.asarray(encoded_residual.permute(1, 2, 0).contiguous().cpu().numpy(), dtype=np.uint8)

        encode_video_frames_ffmpeg(
            output_path=output_path,
            frames_bgr=_iter_encoded_frames(),
            fps=float(chunk.fps),
            width=int(chunk.width),
            height=int(chunk.height),
            codec="libx265",
            pix_fmt="yuv420p",
            crf=28,
            preset="medium",
        )

        return ResidualPacket(
            chunk_id=chunk.chunk_id,
            codec="libx265",
            residual_video_uri=str(output_path),
        )

    def _is_genai_enabled(self) -> bool:
        return os.environ.get("POINTSTREAM_ENABLE_GENAI", "0").strip() == "1"

    def _render_server_genai_prediction(
        self,
        payload: EncodedChunkPayload,
        frame_tensor: torch.Tensor,
    ) -> torch.Tensor:
        actor_state = self._build_actor_state(payload)
        if not actor_state:
            return frame_tensor

        compositor = self._synthesis_engine.get_genai_compositor()
        if not isinstance(compositor, DiffusersCompositor):
            # Handle long-lived pipeline instances where env flags changed after init.
            self._synthesis_engine = SynthesisEngine(
                seed=int(getattr(self._synthesis_engine, "seed", 1337)),
                device=self._device,
            )
            compositor = self._synthesis_engine.get_genai_compositor()
        if not isinstance(compositor, DiffusersCompositor):
            raise RuntimeError("GenAI residual path requires DiffusersCompositor when POINTSTREAM_ENABLE_GENAI=1")

        use_temporal_window = compositor.uses_temporal_pose_sequence()
        temporal_window = max(1, int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_WINDOW", "16")))
        preroll_frames = max(0, int(os.environ.get("POINTSTREAM_GENAI_PREROLL_FRAMES", "1")))

        out_frames: list[torch.Tensor] = []
        frame_count = int(frame_tensor.shape[0])
        chunk_start = int(payload.chunk.start_frame_id)
        for frame_idx in range(frame_count):
            composited = frame_tensor[frame_idx]
            global_frame_id = chunk_start + frame_idx
            for state in actor_state.values():
                if frame_idx >= int(state.dense_pose_tensor.shape[0]):
                    continue

                if use_temporal_window:
                    if frame_idx < preroll_frames:
                        continue
                    pose_condition = self._build_temporal_pose_condition(
                        dense_pose_tensor=state.dense_pose_tensor,
                        frame_idx=frame_idx,
                        temporal_window=temporal_window,
                    )
                else:
                    pose_condition = state.dense_pose_tensor[frame_idx]

                metadata_entry = state.metadata_masks_by_frame.get(global_frame_id)
                metadata_mask = None if metadata_entry is None else metadata_entry.mask_gray
                metadata_bbox = None if metadata_entry is None else metadata_entry.bbox

                composited = compositor.process(
                    reference_crop_tensor=state.reference_crop_tensor,
                    dense_dwpose_tensor=pose_condition,
                    warped_background_frame=composited,
                    actor_identity=state.object_id,
                    metadata_mask=metadata_mask,
                    metadata_bbox=metadata_bbox,
                ).to(frame_tensor.device)

            out_frames.append(composited)

        return torch.stack(out_frames, dim=0)

    def _build_actor_state(self, payload: EncodedChunkPayload) -> dict[int, _ServerActorState]:
        decoded_references = self._decode_reference_crops(payload)
        dense_pose_stream = self._synthesis_engine._unroll_sparse_actor_poses(payload)

        actor_state: dict[int, _ServerActorState] = {}
        for actor_packet in payload.actors:
            track_id = scene_track_id_to_int(actor_packet.object_id)
            reference_crop = decoded_references.get(track_id)
            dense_pose = dense_pose_stream.get(actor_packet.object_id)
            if reference_crop is None or dense_pose is None:
                continue

            actor_state[track_id] = _ServerActorState(
                object_id=actor_packet.object_id,
                reference_crop_tensor=reference_crop,
                dense_pose_tensor=dense_pose,
                metadata_masks_by_frame=self._decode_actor_masks(actor_packet),
            )
        return actor_state

    def _decode_reference_crops(self, payload: EncodedChunkPayload) -> dict[int, torch.Tensor]:
        decoded: dict[int, torch.Tensor] = {}
        for reference in payload.actor_references:
            jpeg_bytes = reference.reference_crop_jpeg
            if not jpeg_bytes and reference.reference_crop_uri is not None:
                reference_path = Path(str(reference.reference_crop_uri))
                if reference_path.exists() and reference_path.is_file():
                    jpeg_bytes = reference_path.read_bytes()
            if not jpeg_bytes:
                continue

            encoded_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            crop_bgr = cv2.imdecode(encoded_np, cv2.IMREAD_COLOR)
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            decoded[int(reference.track_id)] = (
                torch.from_numpy(crop_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)
            )
        return decoded

    def _decode_actor_masks(self, actor_packet: ActorPacket) -> dict[int, _DecodedActorMaskFrame]:
        decoded: dict[int, _DecodedActorMaskFrame] = {}
        for frame_mask in actor_packet.mask_frames:
            payload = frame_mask.mask_payload
            codec = frame_mask.mask_codec
            if payload is None and frame_mask.mask_png is not None:
                payload = frame_mask.mask_png
                codec = "png"
            if payload is None:
                continue

            default_h = max(1, int(frame_mask.bbox[3]) - int(frame_mask.bbox[1]))
            default_w = max(1, int(frame_mask.bbox[2]) - int(frame_mask.bbox[0]))
            mask_h = int(frame_mask.mask_height) if frame_mask.mask_height is not None else default_h
            mask_w = int(frame_mask.mask_width) if frame_mask.mask_width is not None else default_w

            try:
                mask_gray = decode_binary_mask(
                    codec=str(codec),
                    payload=payload,
                    height=mask_h,
                    width=mask_w,
                )
            except Exception:
                if codec != "png":
                    continue
                encoded_np = np.frombuffer(payload, dtype=np.uint8)
                mask_gray = cv2.imdecode(encoded_np, cv2.IMREAD_GRAYSCALE)
                if mask_gray is None or mask_gray.size == 0:
                    continue

            bbox = (
                int(frame_mask.bbox[0]),
                int(frame_mask.bbox[1]),
                int(frame_mask.bbox[2]),
                int(frame_mask.bbox[3]),
            )
            decoded[int(frame_mask.frame_id)] = _DecodedActorMaskFrame(
                bbox=bbox,
                mask_gray=np.asarray(mask_gray, dtype=np.uint8),
            )
        return decoded

    def _build_temporal_pose_condition(
        self,
        dense_pose_tensor: torch.Tensor,
        frame_idx: int,
        temporal_window: int,
    ) -> torch.Tensor:
        start_idx = max(0, int(frame_idx) - int(temporal_window) + 1)
        sequence = dense_pose_tensor[start_idx : int(frame_idx) + 1]
        if int(sequence.shape[0]) >= int(temporal_window):
            return sequence

        if int(sequence.shape[0]) == 0:
            first_pose = dense_pose_tensor[0].unsqueeze(0)
            return first_pose.repeat(int(temporal_window), 1, 1)

        pad_count = int(temporal_window) - int(sequence.shape[0])
        first_pose = sequence[0].unsqueeze(0).repeat(pad_count, 1, 1)
        return torch.cat([first_pose, sequence], dim=0)

    def _default_residual_path(self, chunk: VideoChunk) -> Path:
        override = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
        if override:
            return Path(override) / f"residual_{chunk.chunk_id}.mp4"

        project_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return project_root / "outputs" / timestamp / "debug" / f"residual_{chunk.chunk_id}.mp4"

    def _load_original_frames(self, chunk: VideoChunk) -> torch.Tensor:
        decoded = probe_video_metadata(chunk.source_uri)
        if int(decoded.num_frames) <= 0:
            raise ValueError(f"Source video has no frames: {chunk.source_uri}")
        raise RuntimeError("_load_original_frames is no longer used; original frames are streamed in process()")

    def _select_frame_state(self, frame_states: list[FrameState], frame_idx: int) -> FrameState:
        if frame_idx < len(frame_states):
            return frame_states[frame_idx]
        return FrameState(frame_id=frame_idx, actors=[])
