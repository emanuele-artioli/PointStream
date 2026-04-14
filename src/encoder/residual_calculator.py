from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import EncodedChunkPayload, FrameState, ResidualPacket, SceneActor, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound


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
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._importance_mapper = importance_mapper or BinaryActorImportanceMapper()

    @gpu_bound
    def process(
        self,
        chunk: VideoChunk,
        payload: EncodedChunkPayload,
        frame_states: list[FrameState],
        debug_output_path: str | Path | None = None,
    ) -> ResidualPacket:
        predicted_frames = self._synthesis_engine.synthesize(payload).frames_bgr

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

    def _default_residual_path(self, chunk: VideoChunk) -> Path:
        override = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
        if override:
            return Path(override) / f"residual_{chunk.chunk_id}.mp4"
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "assets" / "test_chunks" / f"residual_{chunk.chunk_id}.mp4"

    def _load_original_frames(self, chunk: VideoChunk) -> torch.Tensor:
        decoded = probe_video_metadata(chunk.source_uri)
        if int(decoded.num_frames) <= 0:
            raise ValueError(f"Source video has no frames: {chunk.source_uri}")
        raise RuntimeError("_load_original_frames is no longer used; original frames are streamed in process()")

    def _select_frame_state(self, frame_states: list[FrameState], frame_idx: int) -> FrameState:
        if frame_idx < len(frame_states):
            return frame_states[frame_idx]
        return FrameState(frame_id=frame_idx, actors=[])
