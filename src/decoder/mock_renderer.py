from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.decoder.compositor import ResidualCompositor
from src.encoder.video_io import encode_video_frames_ffmpeg
from src.shared.mask_codec import decode_binary_mask
from src.shared.schemas import ActorPacket, DecodedChunkResult, EncodedChunkPayload
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound
from src.shared.track_id import scene_track_id_to_int


@dataclass(frozen=True)
class _ClientActorState:
    track_id: int
    object_id: str
    reference_crop_tensor: torch.Tensor
    dense_pose_tensor: torch.Tensor
    metadata_masks_by_frame: dict[int, "_DecodedActorMaskFrame"] = field(default_factory=dict)


@dataclass(frozen=True)
class _DecodedActorMaskFrame:
    bbox: tuple[int, int, int, int]
    mask_gray: np.ndarray


class DecoderRenderer:
    def __init__(self, output_root: str | Path | None = None, deterministic_seed: int = 1337) -> None:
        project_root = Path(__file__).resolve().parents[2]
        if output_root is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            self._output_root = project_root / "outputs" / timestamp / "decoded"
        else:
            self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._synthesis_engine = SynthesisEngine(seed=deterministic_seed)
        self._compositor = ResidualCompositor(device=self._synthesis_engine.device)
        self._genai_compositor = self._synthesis_engine.get_genai_compositor()
        self._actor_state: dict[int, _ClientActorState] = {}
        self._chunk_start_frame_id: int = 0

    @gpu_bound
    def process(self, payload: EncodedChunkPayload, output_path: str | Path | None = None) -> DecodedChunkResult:
        chunk = payload.chunk
        self._chunk_start_frame_id = int(chunk.start_frame_id)
        self._actor_state = self._build_actor_state(payload)
        synthesis = self._synthesis_engine.synthesize(payload, include_guidance_overlays=False)

        genai_enabled = os.environ.get("POINTSTREAM_ENABLE_GENAI", "0").strip() == "1"
        predicted_frames = synthesis.frames_bgr
        if genai_enabled:
            predicted_frames = self._render_genai_baseline(predicted_frames)

        frame_tensor = self._compositor.composite(
            predicted_frames=predicted_frames,
            residual_video_uri=payload.residual.residual_video_uri,
            width=int(chunk.width),
            height=int(chunk.height),
        )
        if not genai_enabled:
            frame_tensor = self._render_genai_baseline(frame_tensor)
        frames_bgr = [
            np.asarray(frame.permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
            for frame in frame_tensor
        ]
        target_output = Path(output_path) if output_path is not None else self._output_root / f"{chunk.chunk_id}.mp4"
        target_output.parent.mkdir(parents=True, exist_ok=True)
        encode_video_frames_ffmpeg(
            output_path=target_output,
            frames_bgr=frames_bgr,
            fps=float(chunk.fps),
            width=int(chunk.width),
            height=int(chunk.height),
            codec="libx264",
            pix_fmt="yuv420p",
            crf=18,
            preset="veryfast",
        )

        return DecodedChunkResult(
            chunk_id=chunk.chunk_id,
            output_uri=str(target_output),
            num_frames=chunk.num_frames,
            width=chunk.width,
            height=chunk.height,
        )

    def _build_actor_state(self, payload: EncodedChunkPayload) -> dict[int, _ClientActorState]:
        decoded_references = self._decode_reference_crops(payload)
        dense_pose_stream = self._synthesis_engine._unroll_sparse_actor_poses(payload)

        actor_state: dict[int, _ClientActorState] = {}
        for actor_packet in payload.actors:
            track_id = scene_track_id_to_int(actor_packet.object_id)
            reference_crop = decoded_references.get(track_id)
            dense_pose = dense_pose_stream.get(actor_packet.object_id)
            if reference_crop is None or dense_pose is None:
                continue
            metadata_masks = self._decode_actor_masks(actor_packet)
            actor_state[track_id] = _ClientActorState(
                track_id=track_id,
                object_id=actor_packet.object_id,
                reference_crop_tensor=reference_crop,
                dense_pose_tensor=dense_pose,
                metadata_masks_by_frame=metadata_masks,
            )
        return actor_state

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
            crop_tensor = torch.from_numpy(crop_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)
            decoded[int(reference.track_id)] = crop_tensor
        return decoded

    def _render_genai_baseline(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        if not self._actor_state:
            return frame_tensor

        use_temporal_window = bool(
            hasattr(self._genai_compositor, "uses_temporal_pose_sequence")
            and self._genai_compositor.uses_temporal_pose_sequence()
        )
        temporal_window = max(1, int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_WINDOW", "16")))
        preroll_frames = max(0, int(os.environ.get("POINTSTREAM_GENAI_PREROLL_FRAMES", "1")))

        out_frames: list[torch.Tensor] = []
        frame_count = int(frame_tensor.shape[0])
        for frame_idx in range(frame_count):
            composited = frame_tensor[frame_idx]
            global_frame_id = int(self._chunk_start_frame_id) + int(frame_idx)
            for actor_state in self._actor_state.values():
                if frame_idx >= int(actor_state.dense_pose_tensor.shape[0]):
                    continue

                if use_temporal_window:
                    if frame_idx < preroll_frames:
                        continue
                    pose_condition = self._build_temporal_pose_condition(
                        dense_pose_tensor=actor_state.dense_pose_tensor,
                        frame_idx=frame_idx,
                        temporal_window=temporal_window,
                    )
                else:
                    pose_condition = actor_state.dense_pose_tensor[frame_idx]

                metadata_entry = actor_state.metadata_masks_by_frame.get(global_frame_id)
                metadata_mask = None if metadata_entry is None else metadata_entry.mask_gray
                metadata_bbox = None if metadata_entry is None else metadata_entry.bbox

                try:
                    composited = self._genai_compositor.process(
                        reference_crop_tensor=actor_state.reference_crop_tensor,
                        dense_dwpose_tensor=pose_condition,
                        warped_background_frame=composited,
                        actor_identity=actor_state.object_id,
                        metadata_mask=metadata_mask,
                        metadata_bbox=metadata_bbox,
                    ).to(frame_tensor.device)
                except TypeError:
                    try:
                        # Keep compatibility with existing test doubles that don't accept metadata_bbox.
                        composited = self._genai_compositor.process(
                            reference_crop_tensor=actor_state.reference_crop_tensor,
                            dense_dwpose_tensor=pose_condition,
                            warped_background_frame=composited,
                            actor_identity=actor_state.object_id,
                            metadata_mask=metadata_mask,
                        ).to(frame_tensor.device)
                    except TypeError:
                        # Keep compatibility with older test doubles that don't accept metadata args.
                        composited = self._genai_compositor.process(
                            reference_crop_tensor=actor_state.reference_crop_tensor,
                            dense_dwpose_tensor=pose_condition,
                            warped_background_frame=composited,
                            actor_identity=actor_state.object_id,
                        ).to(frame_tensor.device)
            out_frames.append(composited)

        return torch.stack(out_frames, dim=0)

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
