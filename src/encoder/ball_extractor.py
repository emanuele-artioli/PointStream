from __future__ import annotations

from collections.abc import Iterator
import cv2
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Any

from src.encoder.video_io import iter_video_frames_ffmpeg
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
from src.shared.torch_dtype import is_cuda_device_usable, resolve_torch_dtype_for_device


class BallExtractor:
    """Parametric ball extractor based on background subtraction and actor masking."""

    def __init__(
        self,
        difference_threshold: float = 18.0,
        min_blob_area: int = 6,
        device: str | torch.device | None = None,
        detection_max_side: int | None = None,
    ) -> None:
        self._difference_threshold = float(difference_threshold)
        self._min_blob_area = int(min_blob_area)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")
        if detection_max_side is None:
            env_max_side = int(os.environ.get("POINTSTREAM_BALL_MAX_SIDE", "0"))
            self._detection_max_side = max(0, env_max_side)
        else:
            self._detection_max_side = max(0, int(detection_max_side))
        self._compute_dtype = resolve_torch_dtype_for_device(self._device, default_cuda=torch.float16)
        self._frame_buffer: torch.Tensor | None = None
        self._background_buffer: torch.Tensor | None = None
        self._actor_mask_buffer: torch.Tensor | None = None

    @gpu_bound
    def process(
        self,
        chunk: VideoChunk,
        panorama: PanoramaPacket,
        frame_states: list[FrameState],
    ) -> BallPacket:
        frame_count = int(chunk.num_frames)
        if frame_count <= 0:
            raise ValueError("BallExtractor received zero configured frames")

        detection_height, detection_width, scale_x, scale_y = self._resolve_detection_geometry(
            frame_width=int(chunk.width),
            frame_height=int(chunk.height),
        )

        panorama_tensor, inverse_h, kornia_module = self._prepare_panorama_warp(
            panorama=panorama,
            frame_count=frame_count,
        )

        ball_states: list[BallState] = []
        previous_visible = False
        previous_x = 0.0
        previous_y = 0.0

        frame_iterator = self._iter_source_frames(chunk)
        for frame_idx, original_frame_np in enumerate(frame_iterator):
            if frame_idx >= frame_count:
                break
            frame_state = self._resolve_frame_state(frame_states=frame_states, frame_idx=frame_idx)

            original_frame = self._prepare_detection_frame(
                frame_np=original_frame_np,
                detection_height=detection_height,
                detection_width=detection_width,
            )
            background_frame = self._warp_panorama_frame(
                kornia_module=kornia_module,
                panorama_tensor=panorama_tensor,
                inverse_h=inverse_h[frame_idx],
                frame_height=detection_height,
                frame_width=detection_width,
                output_buffer=self._ensure_background_buffer(
                    frame_height=detection_height,
                    frame_width=detection_width,
                ),
            )

            raw_diff = torch.abs(original_frame - background_frame)
            actor_mask = self._build_actor_mask(
                frame_state=frame_state,
                frame_height=detection_height,
                frame_width=detection_width,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            masked_diff = raw_diff * (1.0 - actor_mask.unsqueeze(0))

            grayscale = masked_diff.mean(dim=0)
            binary = (grayscale > self._difference_threshold).to(torch.uint8)
            ball_x, ball_y, is_visible = self._largest_blob_center(binary)

            if is_visible:
                ball_x = float(ball_x) / float(scale_x)
                ball_y = float(ball_y) / float(scale_y)

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

        if not ball_states:
            raise ValueError("BallExtractor decoded zero source frames")

        trajectory = torch.zeros((1, len(ball_states), 4), dtype=torch.float32)
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

    def _iter_source_frames(self, chunk: VideoChunk) -> Iterator[np.ndarray]:
        frame_iter = iter_video_frames_ffmpeg(
            chunk.source_uri,
            width=int(chunk.width),
            height=int(chunk.height),
        )

        for _ in range(int(chunk.start_frame_id)):
            try:
                next(frame_iter)
            except StopIteration:
                return

        emitted = 0
        for frame in frame_iter:
            if emitted >= int(chunk.num_frames):
                break
            emitted += 1
            yield frame

    def _resolve_detection_geometry(self, frame_width: int, frame_height: int) -> tuple[int, int, float, float]:
        max_side = int(self._detection_max_side)
        if max_side <= 0:
            return int(frame_height), int(frame_width), 1.0, 1.0

        current_max_side = max(int(frame_width), int(frame_height))
        if current_max_side <= max_side:
            return int(frame_height), int(frame_width), 1.0, 1.0

        scale = float(max_side) / float(current_max_side)
        detection_width = max(1, int(round(float(frame_width) * scale)))
        detection_height = max(1, int(round(float(frame_height) * scale)))
        scale_x = float(detection_width) / float(frame_width)
        scale_y = float(detection_height) / float(frame_height)
        return detection_height, detection_width, scale_x, scale_y

    def _prepare_detection_frame(
        self,
        frame_np: np.ndarray,
        detection_height: int,
        detection_width: int,
    ) -> torch.Tensor:
        frame_tensor = (
            torch.from_numpy(np.asarray(frame_np, dtype=np.uint8))
            .permute(2, 0, 1)
            .to(self._device, dtype=self._compute_dtype)
        )

        if int(frame_tensor.shape[1]) != int(detection_height) or int(frame_tensor.shape[2]) != int(detection_width):
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0),
                size=(int(detection_height), int(detection_width)),
                mode="bilinear",
                align_corners=False,
            )[0]

        if (
            self._frame_buffer is None
            or tuple(self._frame_buffer.shape) != (3, int(detection_height), int(detection_width))
            or self._frame_buffer.dtype != self._compute_dtype
            or self._frame_buffer.device != self._device
        ):
            self._frame_buffer = torch.empty(
                (3, int(detection_height), int(detection_width)),
                dtype=self._compute_dtype,
                device=self._device,
            )

        self._frame_buffer.copy_(frame_tensor)
        return self._frame_buffer

    def _ensure_background_buffer(self, frame_height: int, frame_width: int) -> torch.Tensor:
        if (
            self._background_buffer is None
            or tuple(self._background_buffer.shape) != (3, int(frame_height), int(frame_width))
            or self._background_buffer.dtype != self._compute_dtype
            or self._background_buffer.device != self._device
        ):
            self._background_buffer = torch.empty(
                (3, int(frame_height), int(frame_width)),
                dtype=self._compute_dtype,
                device=self._device,
            )
        return self._background_buffer

    def _prepare_panorama_warp(
        self,
        panorama: PanoramaPacket,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        try:
            import kornia
        except ModuleNotFoundError as exc:
            raise RuntimeError("kornia is required for BallExtractor panorama warping") from exc

        try:
            panorama_np = self._resolve_panorama_image(panorama)
            panorama_tensor = (
                torch.from_numpy(panorama_np)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self._device, dtype=self._compute_dtype)
                / 255.0
            )

            homographies = torch.tensor(panorama.homography_matrices, dtype=torch.float32, device=self._device)
            if homographies.shape[0] < frame_count:
                pad = frame_count - int(homographies.shape[0])
                identity = torch.eye(3, dtype=torch.float32, device=self._device).unsqueeze(0).repeat(pad, 1, 1)
                homographies = torch.cat([homographies, identity], dim=0)
            elif homographies.shape[0] > frame_count:
                homographies = homographies[:frame_count]

            inverse_h = torch.linalg.inv(homographies).to(dtype=self._compute_dtype)
            return panorama_tensor, inverse_h, kornia
        except RuntimeError as exc:
            if self._is_cuda_oom(exc):
                self._fallback_to_cpu()
                return self._prepare_panorama_warp(panorama=panorama, frame_count=frame_count)
            raise

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        if self._device.type != "cuda":
            return False
        return "out of memory" in str(error).lower()

    def _fallback_to_cpu(self) -> None:
        self._device = torch.device("cpu")
        self._compute_dtype = torch.float32
        self._frame_buffer = None
        self._background_buffer = None
        self._actor_mask_buffer = None

    def _resolve_panorama_image(self, panorama: PanoramaPacket) -> np.ndarray:
        if panorama.panorama_image is not None:
            panorama_np = np.asarray(panorama.panorama_image, dtype=np.uint8)
            if panorama_np.ndim != 3 or panorama_np.shape[2] != 3:
                raise ValueError(
                    "Invalid panorama image shape in packet: "
                    f"expected [H, W, 3], got {tuple(panorama_np.shape)}"
                )
            return panorama_np

        panorama_path = Path(str(panorama.panorama_uri))
        if not panorama_path.exists() or not panorama_path.is_file():
            raise FileNotFoundError(
                f"Panorama image not found: {panorama_path}. "
                "BallExtractor requires panorama_image pixels or a valid panorama_uri file."
            )

        decoded_panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
        if decoded_panorama is None or decoded_panorama.size == 0:
            raise ValueError(f"Failed to decode panorama image from {panorama_path}")
        return np.asarray(decoded_panorama, dtype=np.uint8)

    def _warp_panorama_frame(
        self,
        *,
        kornia_module: Any,
        panorama_tensor: torch.Tensor,
        inverse_h: torch.Tensor,
        frame_height: int,
        frame_width: int,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        warped = kornia_module.geometry.transform.warp_perspective(
            src=panorama_tensor,
            M=inverse_h.unsqueeze(0),
            dsize=(int(frame_height), int(frame_width)),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        warped_frame = warped[0].mul(255.0)
        if output_buffer is None:
            return warped_frame

        output_buffer.copy_(warped_frame)
        return output_buffer

    def _build_actor_mask(
        self,
        frame_state: FrameState,
        frame_height: int,
        frame_width: int,
        scale_x: float,
        scale_y: float,
    ) -> torch.Tensor:
        if (
            self._actor_mask_buffer is None
            or tuple(self._actor_mask_buffer.shape) != (int(frame_height), int(frame_width))
            or self._actor_mask_buffer.dtype != self._compute_dtype
            or self._actor_mask_buffer.device != self._device
        ):
            self._actor_mask_buffer = torch.empty(
                (int(frame_height), int(frame_width)),
                dtype=self._compute_dtype,
                device=self._device,
            )

        mask = self._actor_mask_buffer
        mask.zero_()
        for actor in frame_state.actors:
            if actor.class_name not in {"player", "racket"}:
                continue
            if actor.mask is None:
                continue

            x1, y1, x2, y2 = self._clip_bbox(
                actor.bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            actor_mask = torch.from_numpy(np.asarray(actor.mask, dtype=np.float32)).to(self._device, dtype=self._compute_dtype)
            actor_mask = actor_mask.unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(
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

    def _clip_bbox(
        self,
        bbox: list[float],
        frame_width: int,
        frame_height: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        scaled_x1 = float(x1) * float(scale_x)
        scaled_y1 = float(y1) * float(scale_y)
        scaled_x2 = float(x2) * float(scale_x)
        scaled_y2 = float(y2) * float(scale_y)

        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(scaled_x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(scaled_y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(scaled_x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(scaled_y2))))
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2
