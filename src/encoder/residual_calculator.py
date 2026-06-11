from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.decoder.genai_compositor import DiffusersCompositor
from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.mask_codec import decode_binary_mask
from src.shared.profiling import PipelineProfiler
from src.shared.schemas import ActorPacket, EncodedChunkPayload, FrameState, ResidualPacket, ResidualMode, SceneActor, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound
from src.shared.track_id import scene_track_id_to_int
from src.shared.torch_dtype import is_cuda_device_usable


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DecodedActorMaskFrame:
    bbox: tuple[int, int, int, int]
    mask_gray: np.ndarray


@dataclass(frozen=True)
class _ServerActorState:
    object_id: str
    reference_crop_tensor: torch.Tensor
    dense_pose_tensor: torch.Tensor
    keyframe_frame_ids: frozenset[int]
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
    """Binary saliency map: residuals only on detected actors/rackets.
    
    This mapper creates an importance map with value 1.0 on all pixels belonging to actors
    (players, rackets, etc.) and 0.0 elsewhere. This focuses residual transmission on regions
    with semantic importance, reducing bandwidth for background regions that GenAI can hallucinate.
    
    Best for: Low-bandwidth scenarios where background reconstruction is less critical.
    """

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
    """Uniform saliency map: full residuals everywhere.
    
    This mapper assigns equal importance (1.0) to every pixel in every frame.
    All pixels receive equal residual bandwidth regardless of semantic content.
    
    Use case: Ablation studies and ground-truth reconstruction. This provides a upper bound
    on achievable reconstruction quality (perfect reconstruction) by transmitting all residuals.
    Compare against BinaryActorImportanceMapper to measure how much GenAI handles background.
    
    Best for: Baseline comparisons and measuring the effectiveness of semantic masking.
    """

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
    """Server-side weighted residual calculator with support for multiple residual modes."""

    def __init__(
        self,
        config: Any,
        synthesis_engine: SynthesisEngine | None = None,
        seed: int = 1337,
        importance_mapper: BaseImportanceMapper | None = None,
        device: str | torch.device | None = None,
        residual_mode: ResidualMode = ResidualMode.FULL_VIDEO,
        background_block_downscale_factor: int | None = 2,
        residual_batch_size: int = 8,
        downscale_interpolation: str = "bilinear",
        residual_block_size: int = 8,
        block_information_threshold: float = 0.0,
    ) -> None:
        self.config = config
        self._synthesis_engine = synthesis_engine or SynthesisEngine(config=self.config)
        self._seed = seed
        if device is None:
            self._device = torch.device(self._synthesis_engine.device)
        else:
            self._device = torch.device(device)
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")
        if importance_mapper is not None:
            self._importance_mapper = importance_mapper
        else:
            strategy = getattr(self.config, "importance_mapper", "uniform").strip().lower()
            if strategy == "uniform":
                self._importance_mapper = UniformImportanceMapper()
            else:
                self._importance_mapper = BinaryActorImportanceMapper()
        self._residual_mode = residual_mode
        self._background_block_downscale_factor = background_block_downscale_factor
        self._residual_batch_size = residual_batch_size
        self._downscale_interpolation = downscale_interpolation
        self._residual_block_size = residual_block_size
        self._block_information_threshold = block_information_threshold
        self.profiler = PipelineProfiler()

    def get_detailed_profile(self) -> dict[str, float]:
        return self.profiler.get_timings()

    @gpu_bound
    def process(
        self,
        chunk: VideoChunk,
        payload: EncodedChunkPayload,
        frame_states: list[FrameState],
        debug_output_path: str | Path | None = None,
    ) -> ResidualPacket:
        if self._residual_mode == ResidualMode.NONE:
            return ResidualPacket(
                chunk_id=chunk.chunk_id,
                codec="none",
                residual_video_uri="",
                mode=ResidualMode.NONE,
            )
        return self._process_residuals(chunk, payload, frame_states, debug_output_path)

    def _process_residuals(
        self,
        chunk: VideoChunk,
        payload: EncodedChunkPayload,
        frame_states: list[FrameState],
        debug_output_path: str | Path | None,
    ) -> ResidualPacket:
        if self._is_genai_enabled():
            compositor = self._synthesis_engine.get_genai_compositor()
            if hasattr(compositor, "set_debug_stage"):
                compositor.set_debug_stage("encoder")
            if hasattr(compositor, "clear_history"):
                compositor.clear_history()

        with self.profiler.stage("synthesis"):
            base_frames = self._synthesis_engine.synthesize(payload, include_guidance_overlays=False).frames_bgr
            predicted_frames = base_frames

        if self._is_genai_enabled():
            with self.profiler.stage("genai_baseline"):
                predicted_frames = self._render_server_actor_prediction(
                    payload=payload,
                    frame_tensor=base_frames.clone(),
                    debug_output_path=debug_output_path,
                )
            if _LOGGER.isEnabledFor(logging.DEBUG):
                try:
                    diff = (predicted_frames.to(dtype=torch.float32) - base_frames.to(dtype=torch.float32)).abs()
                    mean_diff = float(diff.mean().item())
                    max_diff = float(diff.max().item())
                    pct = float((diff > 1.0).float().mean().item() * 100.0)
                    _LOGGER.debug(
                        "GenAI encoder diff stats: mean=%.2f max=%.2f pct>1=%.1f%%",
                        mean_diff,
                        max_diff,
                        pct,
                    )
                except Exception as exc:
                    _LOGGER.debug("GenAI encoder diff stats unavailable: %s", exc)

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
            frame_idx = 0
            while frame_idx < valid_frames:
                batch_end = min(valid_frames, frame_idx + self._residual_batch_size)
                batch_originals: list[torch.Tensor] = []
                batch_predicted: list[torch.Tensor] = []
                batch_actor_masks: list[torch.Tensor] = []

                for batch_frame_idx in range(frame_idx, batch_end):
                    try:
                        original_np = next(source_iter)
                    except StopIteration:
                        batch_end = batch_frame_idx
                        break

                    batch_originals.append(
                        torch.from_numpy(np.asarray(original_np, dtype=np.uint8))
                        .permute(2, 0, 1)
                        .to(self._device, dtype=torch.float32)
                    )
                    batch_predicted.append(predicted_frames[batch_frame_idx].to(self._device, dtype=torch.float32))
                    frame_state = self._select_frame_state(frame_states=frame_states, frame_idx=batch_frame_idx)
                    batch_actor_masks.append(
                        self._build_actor_mask(
                            frame_state=frame_state,
                            frame_height=int(chunk.height),
                            frame_width=int(chunk.width),
                            device=self._device,
                        )
                    )

                if not batch_originals:
                    break

                originals = torch.stack(batch_originals, dim=0)
                predicted = torch.stack(batch_predicted, dim=0)
                actor_masks = torch.stack(batch_actor_masks, dim=0)

                if self._is_genai_enabled():
                    base_frames_batch = base_frames[frame_idx:batch_end].to(self._device, dtype=torch.float32)
                    predicted_actor_mask = (predicted - base_frames_batch).abs().max(dim=1)[0] > 10.0
                    actor_masks = torch.logical_or(actor_masks > 0.5, predicted_actor_mask).to(torch.float32)

                residual_batch = originals - predicted

                if not self._importance_mapper_is_uniform():
                    residual_batch = torch.where(
                        actor_masks.unsqueeze(1) > 0.0,
                        residual_batch,
                        torch.zeros_like(residual_batch),
                    )

                residual_batch = self._apply_block_activity_gate(
                    residual=residual_batch,
                    block_size=self._residual_block_size,
                    threshold=self._block_information_threshold,
                )
                if self._importance_mapper_is_uniform():
                    residual_batch = self._apply_background_downscale(
                        residual=residual_batch,
                        actor_mask=actor_masks,
                        factor=self._background_block_downscale_factor,
                        interpolation=self._downscale_interpolation,
                    )

                encoded_batch = torch.clamp(residual_batch + 128.0, 0.0, 255.0).to(torch.uint8)
                for encoded_residual in encoded_batch:
                    yield np.asarray(encoded_residual.permute(1, 2, 0).contiguous().cpu().numpy(), dtype=np.uint8)

                frame_idx = batch_end

        with self.profiler.stage("computation"):
            residual_codec = self.config.ffmpeg_codec
            residual_crf = self.config.codec_crf or 28
            residual_preset = self.config.codec_preset or "medium"
            residual_pix_fmt = "yuv444p"
            encode_video_frames_ffmpeg(
                output_path=output_path,
                frames_bgr=_iter_encoded_frames(),
                fps=float(chunk.fps),
                width=int(chunk.width),
                height=int(chunk.height),
                codec=residual_codec,
                pix_fmt=str(residual_pix_fmt),
                crf=residual_crf,
                preset=residual_preset,
            )

        return ResidualPacket(
            chunk_id=chunk.chunk_id,
            codec=residual_codec,
            residual_video_uri=str(output_path),
            mode=self._residual_mode,
        )

    def _apply_block_activity_gate(
        self,
        residual: torch.Tensor,
        block_size: int,
        threshold: float,
    ) -> torch.Tensor:
        """Drop low-activity blocks using pooled mean absolute error.

        The threshold is the average absolute residual value per block in pixel units,
        so a threshold of 2.0 drops blocks whose mean error is below 2 gray levels.
        """
        if block_size <= 1 or threshold <= 0.0:
            return residual

        squeeze_batch = False
        if residual.dim() == 3:
            residual = residual.unsqueeze(0)
            squeeze_batch = True
        elif residual.dim() != 4:
            raise ValueError(f"Expected residual tensor with 3 or 4 dims, got {tuple(residual.shape)}")

        _, _, height, width = residual.shape
        pad_h = (block_size - (height % block_size)) % block_size
        pad_w = (block_size - (width % block_size)) % block_size
        padded = F.pad(residual, (0, pad_w, 0, pad_h), mode="replicate")

        activity = F.avg_pool2d(
            padded.abs().mean(dim=1, keepdim=True),
            kernel_size=block_size,
            stride=block_size,
        )
        keep_blocks = (activity >= float(threshold)).to(dtype=padded.dtype)
        keep_full = F.interpolate(keep_blocks, size=padded.shape[-2:], mode="nearest")

        gated = padded * keep_full
        gated = gated[:, :, :height, :width]
        return gated[0] if squeeze_batch else gated

    def _apply_background_downscale(
        self,
        residual: torch.Tensor,
        actor_mask: torch.Tensor,
        factor: int | None,
        interpolation: str,
    ) -> torch.Tensor:
        if factor is None:
            return residual

        squeeze_batch = False
        if residual.dim() == 3:
            residual = residual.unsqueeze(0)
            actor_mask = actor_mask.unsqueeze(0)
            squeeze_batch = True
        elif residual.dim() != 4:
            raise ValueError(f"Expected residual tensor with 3 or 4 dims, got {tuple(residual.shape)}")

        _, _, height, width = residual.shape
        down_h = max(1, int(np.ceil(height / float(factor))))
        down_w = max(1, int(np.ceil(width / float(factor))))

        actor_mask_bool = actor_mask.unsqueeze(1) > 0.0
        if bool(torch.all(actor_mask_bool)):
            return residual[0] if squeeze_batch else residual

        downsample_kwargs: dict[str, object] = {}
        if interpolation in {"bilinear", "bicubic"}:
            downsample_kwargs["align_corners"] = False

        background_full = F.interpolate(
            F.interpolate(
                residual,
                size=(down_h, down_w),
                mode=interpolation,
                **downsample_kwargs,  # type: ignore[arg-type]
            ),
            size=(height, width),
            mode="nearest",
        )

        blended = torch.where(actor_mask_bool, residual, background_full)
        return blended[0] if squeeze_batch else blended

    def _build_actor_mask(
        self,
        frame_state: FrameState,
        frame_height: int,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        actor_mask = torch.zeros((frame_height, frame_width), dtype=torch.float32, device=device)

        for actor in frame_state.actors:
            if actor.class_name not in {"player", "racket"}:
                continue
            if actor.mask is None:
                continue

            x1, y1, x2, y2 = self._clip_bbox(actor.bbox, frame_width=frame_width, frame_height=frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            raw_mask = np.asarray(actor.mask, dtype=np.float32)
            if raw_mask.ndim != 2 or raw_mask.size == 0:
                continue

            mask_tensor = torch.from_numpy(raw_mask).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            resized_mask = F.interpolate(
                mask_tensor,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0, 0].clamp(0.0, 1.0)
            actor_mask[y1:y2, x1:x2] = torch.maximum(actor_mask[y1:y2, x1:x2], resized_mask)

        return actor_mask.clamp(0.0, 1.0)

    def _clip_bbox(self, bbox: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(y2))))
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2

    def _importance_mapper_is_uniform(self) -> bool:
        return isinstance(self._importance_mapper, UniformImportanceMapper)

    def _is_genai_enabled(self) -> bool:
        return bool(self.config.genai_backend)

    def _resolve_temporal_window(self, default_window: int) -> int:
        raw = getattr(self.config, "animate_anyone_window", None)
        if raw is None:
            return default_window
        raw = raw.strip().lower()
        if raw in {"", "none", "null", "off", "0"}:
            return default_window
        try:
            value = int(raw)
        except ValueError:
            return default_window
        return max(1, value)

    def _render_server_actor_prediction(
        self,
        payload: EncodedChunkPayload,
        frame_tensor: torch.Tensor,
        debug_output_path: str | Path | None = None,
    ) -> torch.Tensor:
        actor_state = self._build_actor_state(payload)
        if not actor_state:
            return frame_tensor

        compositor = self._synthesis_engine.get_genai_compositor()
        if self._is_genai_enabled() and not isinstance(compositor, DiffusersCompositor):
            self._synthesis_engine = SynthesisEngine(
                seed=self._seed,
                device=self._device,
                config=self.config,
            )
            compositor = self._synthesis_engine.get_genai_compositor()
        if not hasattr(compositor, "process"):
            raise RuntimeError("Residual actor compositor is unavailable")

        if (
            self._is_genai_enabled()
            and self.config.genai_keyframe_only
        ):
            return self._render_server_genai_keyframe_only(
                payload=payload,
                frame_tensor=frame_tensor,
                actor_state=actor_state,
                compositor=compositor,
            )

        if (
            self._is_genai_enabled()
            and hasattr(compositor, "process_sequence")
            and hasattr(compositor, "uses_temporal_pose_sequence")
            and compositor.uses_temporal_pose_sequence()
        ):
            return self._render_server_genai_sequence(
                payload=payload,
                frame_tensor=frame_tensor,
                actor_state=actor_state,
                compositor=compositor,
                debug_dir=debug_output_path,
            )

        use_temporal_window = bool(hasattr(compositor, "uses_temporal_pose_sequence") and compositor.uses_temporal_pose_sequence())
        preroll_frames = max(0, self.config.genai_preroll_frames)

        blended_frames: list[torch.Tensor] = []
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
                    temporal_window = self._resolve_temporal_window(default_window=int(frame_idx) + 1)
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
                    debug_dir=debug_output_path,
                    frame_idx=frame_idx,
                ).to(frame_tensor.device)

            blended_frames.append(composited)

        return torch.stack(blended_frames, dim=0)
    def _render_server_genai_keyframe_only(
        self,
        payload: EncodedChunkPayload,
        frame_tensor: torch.Tensor,
        actor_state: dict[int, _ServerActorState],
        compositor: Any,
    ) -> torch.Tensor:
        frame_count = int(frame_tensor.shape[0])
        if frame_count <= 0 or not actor_state:
            return frame_tensor

        selected: set[int] = set()
        for state in actor_state.values():
            selected.update(idx for idx in state.keyframe_frame_ids if 0 <= idx < frame_count)
        if not selected:
            return frame_tensor

        selected_indices = sorted(selected)
        if selected_indices[0] != 0:
            selected_indices.insert(0, 0)
        if selected_indices[-1] != frame_count - 1:
            selected_indices.append(frame_count - 1)
        selected_indices = sorted(set(selected_indices))

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "GenAI keyframe-only encoder: %s keyframes selected across %s frames",
                len(selected_indices),
                frame_count,
            )
            for state in actor_state.values():
                pose_count = int(state.dense_pose_tensor.shape[0])
                available_masks = sum(
                    1 for idx in selected_indices if (payload.chunk.start_frame_id + idx) in state.metadata_masks_by_frame
                )
                _LOGGER.debug(
                    "Actor %s: poses=%s, keyframes=%s, metadata_masks=%s",
                    state.object_id,
                    pose_count,
                    len(selected_indices),
                    available_masks,
                )

        use_temporal_window = bool(hasattr(compositor, "uses_temporal_pose_sequence") and compositor.uses_temporal_pose_sequence())
        preroll_frames = max(0, self.config.genai_preroll_frames)
        chunk_start = int(payload.chunk.start_frame_id)

        generated_deltas: list[torch.Tensor] = []
        logged_delta = False
        if use_temporal_window and hasattr(compositor, "process_sequence"):
            selected_frames = torch.stack([frame_tensor[idx] for idx in selected_indices], dim=0)
            out_frames = selected_frames.clone()
            for state in actor_state.values():
                pose_count = int(state.dense_pose_tensor.shape[0])
                selected_positions = [pos for pos, idx in enumerate(selected_indices) if idx < pose_count]
                if not selected_positions:
                    continue

                metadata_masks: list[np.ndarray | None] = []
                metadata_bboxes: list[tuple[int, int, int, int] | None] = []
                pose_sequence: list[torch.Tensor] = []
                background_sequence: list[torch.Tensor] = []
                for pos in selected_positions:
                    frame_idx = selected_indices[pos]
                    global_frame_id = chunk_start + frame_idx
                    metadata_entry = state.metadata_masks_by_frame.get(global_frame_id)
                    metadata_masks.append(None if metadata_entry is None else metadata_entry.mask_gray)
                    metadata_bboxes.append(None if metadata_entry is None else metadata_entry.bbox)
                    pose_sequence.append(state.dense_pose_tensor[frame_idx])
                    background_sequence.append(out_frames[pos])

                processed = compositor.process_sequence(
                    reference_crop_tensor=state.reference_crop_tensor,
                    dense_dwpose_tensor=torch.stack(pose_sequence, dim=0),
                    warped_background_frames=torch.stack(background_sequence, dim=0),
                    actor_identity=state.object_id,
                    metadata_masks=metadata_masks,
                    metadata_bboxes=metadata_bboxes,
                ).to(frame_tensor.device)

                for offset, pos in enumerate(selected_positions):
                    out_frames[pos] = processed[offset]

            generated_deltas = []
            for idx in range(out_frames.shape[0]):
                delta = out_frames[idx].to(dtype=torch.float32) - selected_frames[idx].to(dtype=torch.float32)
                if _LOGGER.isEnabledFor(logging.DEBUG) and not logged_delta:
                    _LOGGER.debug(
                        "GenAI encoder keyframe delta stats: mean=%.2f max=%.2f",
                        float(delta.abs().mean().item()),
                        float(delta.abs().max().item()),
                    )
                    logged_delta = True
                generated_deltas.append(delta)
        else:
            for frame_idx in selected_indices:
                bg_frame = frame_tensor[frame_idx]
                composited = bg_frame
                global_frame_id = chunk_start + frame_idx
                for state in actor_state.values():
                    if frame_idx >= int(state.dense_pose_tensor.shape[0]):
                        continue

                    if use_temporal_window:
                        if frame_idx < preroll_frames:
                            continue
                        temporal_window = self._resolve_temporal_window(default_window=int(frame_idx) + 1)
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

                delta = composited.to(dtype=torch.float32) - bg_frame.to(dtype=torch.float32)
                if _LOGGER.isEnabledFor(logging.DEBUG) and not logged_delta:
                    _LOGGER.debug(
                        "GenAI encoder keyframe delta stats: mean=%.2f max=%.2f",
                        float(delta.abs().mean().item()),
                        float(delta.abs().max().item()),
                    )
                    logged_delta = True
                generated_deltas.append(delta)

        blended_frames: list[torch.Tensor] = []
        anchor_idx = 0
        for frame_idx in range(frame_count):
            if frame_idx <= selected_indices[0]:
                blended_frames.append((frame_tensor[frame_idx].to(dtype=torch.float32) + generated_deltas[0]).clamp(0.0, 255.0).to(torch.uint8))
                continue
            while anchor_idx + 1 < len(selected_indices) and frame_idx > selected_indices[anchor_idx + 1]:
                anchor_idx += 1
            if anchor_idx + 1 >= len(selected_indices):
                blended_frames.append((frame_tensor[frame_idx].to(dtype=torch.float32) + generated_deltas[-1]).clamp(0.0, 255.0).to(torch.uint8))
                continue

            left_idx = selected_indices[anchor_idx]
            right_idx = selected_indices[anchor_idx + 1]
            left_delta = generated_deltas[anchor_idx]
            right_delta = generated_deltas[anchor_idx + 1]
            if right_idx <= left_idx:
                blended_frames.append((frame_tensor[frame_idx].to(dtype=torch.float32) + left_delta).clamp(0.0, 255.0).to(torch.uint8))
                continue
            alpha = float(frame_idx - left_idx) / float(right_idx - left_idx)
            blended_delta = torch.lerp(left_delta, right_delta, alpha)
            blended_frames.append((frame_tensor[frame_idx].to(dtype=torch.float32) + blended_delta).clamp(0.0, 255.0).to(torch.uint8))

        return torch.stack(blended_frames, dim=0)

    def _render_server_genai_sequence(
        self,
        payload: EncodedChunkPayload,
        frame_tensor: torch.Tensor,
        actor_state: dict[int, _ServerActorState],
        compositor: Any,
        debug_dir: str | Path | None = None,
    ) -> torch.Tensor:
        frame_count = int(frame_tensor.shape[0])
        if frame_count <= 0 or not actor_state:
            return frame_tensor

        out_frames = frame_tensor
        chunk_start = int(payload.chunk.start_frame_id)
        for state in actor_state.values():
            pose_count = int(state.dense_pose_tensor.shape[0])
            valid_count = min(frame_count, pose_count)
            if valid_count <= 0:
                continue

            metadata_masks: list[np.ndarray | None] = []
            metadata_bboxes: list[tuple[int, int, int, int] | None] = []
            for frame_idx in range(valid_count):
                global_frame_id = chunk_start + frame_idx
                metadata_entry = state.metadata_masks_by_frame.get(global_frame_id)
                metadata_masks.append(None if metadata_entry is None else metadata_entry.mask_gray)
                metadata_bboxes.append(None if metadata_entry is None else metadata_entry.bbox)

            processed = compositor.process_sequence(
                reference_crop_tensor=state.reference_crop_tensor,
                dense_dwpose_tensor=state.dense_pose_tensor[:valid_count],
                warped_background_frames=out_frames[:valid_count],
                actor_identity=state.object_id,
                metadata_masks=metadata_masks,
                metadata_bboxes=metadata_bboxes,
                debug_dir=debug_dir,
                global_frame_ids=[chunk_start + i for i in range(valid_count)],
            ).to(frame_tensor.device)

            if valid_count < frame_count:
                out_frames = torch.cat([processed, out_frames[valid_count:]], dim=0)
            else:
                out_frames = processed

        return out_frames

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
                keyframe_frame_ids=frozenset(
                    int(event.frame_id)
                    for event in actor_packet.events
                    if getattr(event, "event_type", None) == "keyframe"
                ),
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
            decoded[int(reference.track_id)] = torch.from_numpy(crop_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)
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
            decoded[int(frame_mask.frame_id)] = _DecodedActorMaskFrame(bbox=bbox, mask_gray=mask_gray)
        return decoded

    def _build_temporal_pose_condition(
        self,
        dense_pose_tensor: torch.Tensor,
        frame_idx: int,
        temporal_window: int,
    ) -> torch.Tensor:
        """Build causal temporal pose window for temporal GenAI compositors.

        Residual generation must avoid future-pose leakage; using look-ahead windows causes
        actor synthesis to lead motion and increases residual payload with shadow corrections.
        """
        start_idx = max(0, int(frame_idx) - int(temporal_window) + 1)
        sequence = dense_pose_tensor[start_idx : int(frame_idx) + 1]

        if int(sequence.shape[0]) == 0:
            return dense_pose_tensor[int(frame_idx)].unsqueeze(0)

        # Keep temporal conditioning as [Frames, 18, 3]; concatenation would flatten to [Frames*18, 3].
        return sequence

    def _default_residual_path(self, chunk: VideoChunk) -> Path:
        runtime_output_dir = getattr(self.config, "runtime_output_dir", None)
        if runtime_output_dir:
            base_dir = Path(runtime_output_dir).expanduser()
        else:
            try:
                base_dir = Path.cwd()
            except FileNotFoundError:
                base_dir = Path(__file__).resolve().parents[2] / "outputs"

        return base_dir / f"chunk_{chunk.chunk_id}" / "residual.mp4"

    def _load_original_frames(self, chunk: VideoChunk) -> torch.Tensor:
        decoded = probe_video_metadata(chunk.source_uri)
        if int(decoded.num_frames) <= 0:
            raise ValueError(f"Source video has no frames: {chunk.source_uri}")
        raise RuntimeError("_load_original_frames is no longer used; original frames are streamed in process()")

    def _select_frame_state(self, frame_states: list[FrameState], frame_idx: int) -> FrameState:
        if frame_idx < len(frame_states):
            return frame_states[frame_idx]
        return FrameState(frame_id=frame_idx, actors=[])
