from __future__ import annotations

import cv2
import numpy as np
import torch

from src.encoder.video_io import decode_video_to_tensor
from src.shared.schemas import (
    BallPacket,
    BallState,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    PanoramaPacket,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


class BallExtractor:
    """Parametric ball extractor based on background subtraction and actor masking."""

    def __init__(
        self,
        difference_threshold: float = 18.0,
        min_blob_area: int = 6,
        device: str | torch.device | None = None,
    ) -> None:
        self._difference_threshold = float(difference_threshold)
        self._min_blob_area = int(min_blob_area)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    @gpu_bound
    def process(
        self,
        chunk: VideoChunk,
        panorama: PanoramaPacket,
        frame_states: list[FrameState],
    ) -> BallPacket:
        original_frames = self._decode_original_frames(chunk)
        background_frames = self._warp_panorama_to_frames(panorama=panorama, chunk=chunk, frame_count=int(original_frames.shape[0]))

        frame_count = min(int(chunk.num_frames), int(original_frames.shape[0]), int(background_frames.shape[0]))
        if frame_count <= 0:
            raise ValueError("BallExtractor received zero valid frames")

        ball_states: list[BallState] = []
        previous_visible = False
        previous_x = 0.0
        previous_y = 0.0

        for frame_idx in range(frame_count):
            frame_state = self._resolve_frame_state(frame_states=frame_states, frame_idx=frame_idx)

            raw_diff = torch.abs(original_frames[frame_idx] - background_frames[frame_idx])
            actor_mask = self._build_actor_mask(frame_state=frame_state, frame_height=int(chunk.height), frame_width=int(chunk.width))
            masked_diff = raw_diff * (1.0 - actor_mask.unsqueeze(0))

            grayscale = masked_diff.mean(dim=0)
            binary = (grayscale > self._difference_threshold).to(torch.uint8)
            ball_x, ball_y, is_visible = self._largest_blob_center(binary)

            if is_visible:
                if previous_visible:
                    velocity_x = ball_x - previous_x
                    velocity_y = ball_y - previous_y
                else:
                    velocity_x = 0.0
                    velocity_y = 0.0
                previous_x = ball_x
                previous_y = ball_y
            else:
                ball_x = previous_x
                ball_y = previous_y
                velocity_x = 0.0
                velocity_y = 0.0

            previous_visible = is_visible
            state = BallState(
                frame_id=chunk.start_frame_id + frame_idx,
                ball_x=float(ball_x),
                ball_y=float(ball_y),
                velocity_x=float(velocity_x),
                velocity_y=float(velocity_y),
                is_visible=bool(is_visible),
            )
            ball_states.append(state)
            if frame_idx < len(frame_states):
                frame_states[frame_idx] = frame_states[frame_idx].model_copy(update={"ball_state": state})

        trajectory = torch.zeros((1, frame_count, 4), dtype=torch.float32)
        events: list[SemanticEvent] = []
        for idx, state in enumerate(ball_states):
            trajectory[0, idx] = torch.tensor(
                [state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                dtype=torch.float32,
            )
            if state.is_visible:
                events.append(
                    KeyframeEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        coordinates=[state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                    )
                )
            else:
                events.append(
                    InterpolateCommandEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        target_frame_id=state.frame_id,
                        method="linear",
                    )
                )

        return BallPacket(
            chunk_id=chunk.chunk_id,
            object_id="ball_0",
            trajectory_spec=TensorSpec(
                name="ball_trajectory",
                shape=list(trajectory.shape),
                dtype=str(trajectory.dtype),
            ),
            events=events,
            states=ball_states,
        )

    def _decode_original_frames(self, chunk: VideoChunk) -> torch.Tensor:
        decoded = decode_video_to_tensor(chunk.source_uri)
        return decoded.tensor[: int(chunk.num_frames)].to(self._device, dtype=torch.float32).mul(255.0)

    def _warp_panorama_to_frames(self, panorama: PanoramaPacket, chunk: VideoChunk, frame_count: int) -> torch.Tensor:
        try:
            import kornia
        except ModuleNotFoundError as exc:
            raise RuntimeError("kornia is required for BallExtractor panorama warping") from exc

        panorama_np = np.asarray(panorama.panorama_image, dtype=np.uint8)
        panorama_tensor = (
            torch.from_numpy(panorama_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device, dtype=torch.float32)
            / 255.0
        )

        homographies = torch.tensor(panorama.homography_matrices, dtype=torch.float32, device=self._device)
        if homographies.shape[0] < frame_count:
            pad = frame_count - int(homographies.shape[0])
            identity = torch.eye(3, dtype=torch.float32, device=self._device).unsqueeze(0).repeat(pad, 1, 1)
            homographies = torch.cat([homographies, identity], dim=0)
        elif homographies.shape[0] > frame_count:
            homographies = homographies[:frame_count]

        inverse_h = torch.linalg.inv(homographies)
        batched_panorama = panorama_tensor.expand(frame_count, -1, -1, -1)
        warped = kornia.geometry.transform.warp_perspective(
            src=batched_panorama,
            M=inverse_h,
            dsize=(int(chunk.height), int(chunk.width)),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.mul(255.0)

    def _build_actor_mask(self, frame_state: FrameState, frame_height: int, frame_width: int) -> torch.Tensor:
        mask = torch.zeros((frame_height, frame_width), dtype=torch.float32, device=self._device)
        for actor in frame_state.actors:
            if actor.class_name not in {"player", "racket"}:
                continue
            if actor.mask is None:
                continue

            x1, y1, x2, y2 = self._clip_bbox(actor.bbox, frame_width=frame_width, frame_height=frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            actor_mask = torch.from_numpy(np.asarray(actor.mask, dtype=np.float32)).to(self._device)
            actor_mask = actor_mask.unsqueeze(0).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                actor_mask,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            mask[y1:y2, x1:x2] = torch.maximum(mask[y1:y2, x1:x2], resized.clamp(0.0, 1.0))

        return mask

    def _largest_blob_center(self, binary_mask: torch.Tensor) -> tuple[float, float, bool]:
        mask_np = (binary_mask.detach().cpu().numpy() > 0).astype(np.uint8)
        if int(mask_np.sum()) == 0:
            return 0.0, 0.0, False

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            return 0.0, 0.0, False

        areas = stats[1:, cv2.CC_STAT_AREA]
        best_local = int(np.argmax(areas))
        best_label = best_local + 1
        best_area = int(stats[best_label, cv2.CC_STAT_AREA])
        if best_area < self._min_blob_area:
            return 0.0, 0.0, False

        center_x, center_y = centroids[best_label]
        return float(center_x), float(center_y), True

    def _resolve_frame_state(self, frame_states: list[FrameState], frame_idx: int) -> FrameState:
        if frame_idx < len(frame_states):
            return frame_states[frame_idx]
        return FrameState(frame_id=frame_idx, actors=[])

    def _clip_bbox(self, bbox: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(y2))))
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2
