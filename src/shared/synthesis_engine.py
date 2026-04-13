from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
import torch

from src.decoder.genai_compositor import DiffusersCompositor, MockCompositor
from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.schemas import ActorPacket, EncodedChunkPayload
from src.shared.tags import gpu_bound


@dataclass(frozen=True)
class SynthesisResult:
    """Dense synthesized reconstruction output."""

    # Shape: [Frames, Channels, Height, Width] in BGR uint8.
    frames_bgr: torch.Tensor


@dataclass(frozen=True)
class _BallRenderState:
    x: float
    y: float
    vx: float
    vy: float
    is_visible: bool


class SynthesisEngine:
    """Shared deterministic synthesis engine used symmetrically by encoder and decoder."""

    def __init__(self, seed: int = 1337, device: str | torch.device | None = None) -> None:
        self._configure_cuda_determinism_env()
        self.seed = int(seed)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._set_global_seed(self.seed)
        self._genai_compositor = self._build_genai_compositor()

    def _build_genai_compositor(self) -> MockCompositor | DiffusersCompositor:
        enabled = os.environ.get("POINTSTREAM_ENABLE_GENAI", "0").strip() == "1"
        if enabled:
            return DiffusersCompositor(seed=self.seed, device=self.device)
        return MockCompositor()

    def get_genai_compositor(self) -> MockCompositor | DiffusersCompositor:
        return self._genai_compositor

    def _configure_cuda_determinism_env(self) -> None:
        # CuBLAS requires this variable for reproducible GEMM-based kernels on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @gpu_bound
    def synthesize(self, payload: EncodedChunkPayload) -> SynthesisResult:
        self._set_global_seed(self.seed)

        dense_pose_stream = self._unroll_sparse_actor_poses(payload)
        dense_ball_states = self._unroll_ball_states(payload=payload)
        background_frames = self._reconstruct_background_frames(payload)
        composited_frames = self._composite_mock_skeletons(
            background_frames,
            dense_pose_stream,
            dense_ball_states,
        )
        return SynthesisResult(frames_bgr=composited_frames)

    def _set_global_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _reconstruct_background_frames(self, payload: EncodedChunkPayload) -> torch.Tensor:
        try:
            import kornia
        except ModuleNotFoundError as exc:
            raise RuntimeError("kornia is required for GPU-native panorama re-warping") from exc

        chunk = payload.chunk
        frame_count = int(chunk.num_frames)
        output_height = int(chunk.height)
        output_width = int(chunk.width)

        panorama_np = np.asarray(payload.panorama.panorama_image, dtype=np.uint8)
        if panorama_np.ndim != 3 or panorama_np.shape[2] != 3:
            raise ValueError(
                "Invalid panorama image shape in payload: "
                f"expected [H, W, 3], got {tuple(panorama_np.shape)}"
            )

        panorama_tensor = (
            torch.from_numpy(panorama_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
            / 255.0
        )

        homographies = torch.tensor(payload.panorama.homography_matrices, dtype=torch.float32, device=self.device)
        if homographies.shape[0] == 0:
            raise ValueError("Payload contains no homography matrices")
        if homographies.shape[0] < frame_count:
            pad = frame_count - int(homographies.shape[0])
            identity = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(pad, 1, 1)
            homographies = torch.cat([homographies, identity], dim=0)
        elif homographies.shape[0] > frame_count:
            homographies = homographies[:frame_count]

        inverse_h = torch.linalg.inv(homographies)
        batch_size = max(1, int(os.environ.get("POINTSTREAM_PANORAMA_WARP_BATCH_SIZE", "4")))
        warped_uint8 = torch.empty(
            (frame_count, 3, output_height, output_width),
            dtype=torch.uint8,
            device=self.device,
        )

        for start in range(0, frame_count, batch_size):
            end = min(frame_count, start + batch_size)
            current_batch = end - start
            batched_panorama = panorama_tensor.expand(current_batch, -1, -1, -1)
            warped_batch = kornia.geometry.transform.warp_perspective(
                src=batched_panorama,
                M=inverse_h[start:end],
                dsize=(output_height, output_width),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_uint8[start:end] = warped_batch.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)

        return warped_uint8.contiguous()

    def _unroll_sparse_actor_poses(self, payload: EncodedChunkPayload) -> dict[str, torch.Tensor]:
        dense_stream: dict[str, torch.Tensor] = {}
        chunk = payload.chunk
        frame_count = int(chunk.num_frames)
        start_frame = int(chunk.start_frame_id)

        for actor_packet in payload.actors:
            dense = torch.zeros((frame_count, 18, 3), dtype=torch.float32)
            keyframes = self._collect_keyframes(actor_packet=actor_packet, start_frame=start_frame, frame_count=frame_count)
            interpolate_mask = self._collect_interpolate_mask(
                actor_packet=actor_packet,
                start_frame=start_frame,
                frame_count=frame_count,
            )

            if not keyframes:
                dense_stream[actor_packet.object_id] = dense
                continue

            sorted_frames = sorted(keyframes.keys())
            for local_frame, pose in keyframes.items():
                dense[local_frame] = pose

            first_idx = sorted_frames[0]
            for local_idx in range(0, first_idx):
                dense[local_idx] = keyframes[first_idx]

            for left, right in zip(sorted_frames[:-1], sorted_frames[1:]):
                left_pose = keyframes[left]
                right_pose = keyframes[right]
                span = right - left
                if span <= 1:
                    continue
                for local_idx in range(left + 1, right):
                    if interpolate_mask[local_idx]:
                        alpha = float(local_idx - left) / float(span)
                        dense[local_idx] = left_pose * (1.0 - alpha) + right_pose * alpha
                    else:
                        dense[local_idx] = left_pose

            last_idx = sorted_frames[-1]
            for local_idx in range(last_idx + 1, frame_count):
                dense[local_idx] = keyframes[last_idx]

            self._apply_static_holds(
                actor_packet=actor_packet,
                dense=dense,
                start_frame=start_frame,
                frame_count=frame_count,
            )
            dense_stream[actor_packet.object_id] = dense

        return dense_stream

    def _collect_keyframes(self, actor_packet: ActorPacket, start_frame: int, frame_count: int) -> dict[int, torch.Tensor]:
        keyframes: dict[int, torch.Tensor] = {}
        for event in actor_packet.events:
            if event.event_type != "keyframe":
                continue

            local_idx = int(event.frame_id) - start_frame
            if local_idx < 0 or local_idx >= frame_count:
                continue

            coords = np.asarray(event.coordinates, dtype=np.float32)
            if coords.size != 54:
                raise ValueError(
                    f"Invalid keyframe coordinate size for actor {actor_packet.object_id}: "
                    f"expected 54 values (18x3), got {coords.size}"
                )
            keyframes[local_idx] = torch.from_numpy(coords.reshape(18, 3).copy())

        return keyframes

    def _collect_interpolate_mask(self, actor_packet: ActorPacket, start_frame: int, frame_count: int) -> torch.Tensor:
        mask = torch.zeros((frame_count,), dtype=torch.bool)
        has_interpolate_events = False

        for event in actor_packet.events:
            if event.event_type != "interpolate":
                continue

            has_interpolate_events = True
            local_from = int(event.frame_id) - start_frame
            local_to = int(event.target_frame_id) - start_frame
            begin = max(0, min(local_from, local_to))
            end = min(frame_count - 1, max(local_from, local_to))
            if end < 0 or begin >= frame_count:
                continue
            mask[begin : end + 1] = True

        if not has_interpolate_events:
            mask[:] = True
        return mask

    def _apply_static_holds(
        self,
        actor_packet: ActorPacket,
        dense: torch.Tensor,
        start_frame: int,
        frame_count: int,
    ) -> None:
        for event in actor_packet.events:
            if event.event_type != "static":
                continue
            hold_from = max(0, int(event.frame_id) - start_frame)
            hold_to = min(frame_count - 1, int(event.hold_until_frame_id) - start_frame)
            if hold_from < 0 or hold_from >= frame_count or hold_to < hold_from:
                continue
            dense[hold_from : hold_to + 1] = dense[hold_from].unsqueeze(0)

    def _composite_mock_skeletons(
        self,
        background_frames: torch.Tensor,
        dense_pose_stream: dict[str, torch.Tensor],
        dense_ball_states: list[_BallRenderState],
    ) -> torch.Tensor:
        frame_count = int(background_frames.shape[0])
        out_frames: list[torch.Tensor] = []

        for frame_idx in range(frame_count):
            frame_gpu = background_frames[frame_idx]
            frame_np = frame_gpu.permute(1, 2, 0).contiguous().cpu().numpy()

            actor_poses: list[np.ndarray] = []
            for actor_dense in dense_pose_stream.values():
                pose = actor_dense[frame_idx].cpu().numpy()
                if pose.shape == (18, 3):
                    actor_poses.append(pose.astype(np.float32, copy=False))

            if actor_poses:
                try:
                    skeleton_canvas = draw_dwpose_canvas(
                        height=int(frame_np.shape[0]),
                        width=int(frame_np.shape[1]),
                        people_dw=np.stack(actor_poses, axis=0),
                        confidence_threshold=0.2,
                    )
                    overlay_mask = np.any(skeleton_canvas > 0, axis=2)
                    frame_np[overlay_mask] = skeleton_canvas[overlay_mask]
                except ModuleNotFoundError:
                    pass

            if frame_idx < len(dense_ball_states):
                self._draw_motion_blurred_ball(frame_np=frame_np, state=dense_ball_states[frame_idx])

            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)
            out_frames.append(frame_tensor)

        return torch.stack(out_frames, dim=0).to(self.device)

    def _unroll_ball_states(self, payload: EncodedChunkPayload) -> list[_BallRenderState]:
        frame_count = int(payload.chunk.num_frames)
        start_frame_id = int(payload.chunk.start_frame_id)
        dense_states: list[_BallRenderState] = [
            _BallRenderState(x=0.0, y=0.0, vx=0.0, vy=0.0, is_visible=False)
            for _ in range(frame_count)
        ]

        if payload.ball.states:
            for ball_state in payload.ball.states:
                local_idx = int(ball_state.frame_id) - start_frame_id
                if local_idx < 0 or local_idx >= frame_count:
                    continue
                dense_states[local_idx] = _BallRenderState(
                    x=float(ball_state.ball_x),
                    y=float(ball_state.ball_y),
                    vx=float(ball_state.velocity_x),
                    vy=float(ball_state.velocity_y),
                    is_visible=bool(ball_state.is_visible),
                )

            for idx in range(1, frame_count):
                current = dense_states[idx]
                if current.is_visible:
                    continue
                previous = dense_states[idx - 1]
                if previous.is_visible:
                    dense_states[idx] = _BallRenderState(
                        x=previous.x,
                        y=previous.y,
                        vx=0.0,
                        vy=0.0,
                        is_visible=False,
                    )
            return dense_states

        keyframes: dict[int, _BallRenderState] = {}
        for event in payload.ball.events:
            if event.event_type != "keyframe":
                continue
            local_idx = int(event.frame_id) - start_frame_id
            if local_idx < 0 or local_idx >= frame_count:
                continue
            coords = event.coordinates
            if len(coords) < 2:
                continue
            keyframes[local_idx] = _BallRenderState(
                x=float(coords[0]),
                y=float(coords[1]),
                vx=float(coords[2]) if len(coords) > 2 else 0.0,
                vy=float(coords[3]) if len(coords) > 3 else 0.0,
                is_visible=True,
            )

        if not keyframes:
            return dense_states

        sorted_indices = sorted(keyframes.keys())
        for local_idx, render_state in keyframes.items():
            dense_states[local_idx] = render_state

        first_idx = sorted_indices[0]
        for idx in range(0, first_idx):
            dense_states[idx] = dense_states[first_idx]

        for left, right in zip(sorted_indices[:-1], sorted_indices[1:]):
            left_state = dense_states[left]
            right_state = dense_states[right]
            span = right - left
            if span <= 1:
                continue
            for idx in range(left + 1, right):
                alpha = float(idx - left) / float(span)
                dense_states[idx] = _BallRenderState(
                    x=left_state.x * (1.0 - alpha) + right_state.x * alpha,
                    y=left_state.y * (1.0 - alpha) + right_state.y * alpha,
                    vx=left_state.vx * (1.0 - alpha) + right_state.vx * alpha,
                    vy=left_state.vy * (1.0 - alpha) + right_state.vy * alpha,
                    is_visible=True,
                )

        last_idx = sorted_indices[-1]
        for idx in range(last_idx + 1, frame_count):
            dense_states[idx] = dense_states[last_idx]

        return dense_states

    def _draw_motion_blurred_ball(self, frame_np: np.ndarray, state: _BallRenderState) -> None:
        if not state.is_visible:
            return

        frame_h, frame_w = frame_np.shape[:2]
        uses_normalized_coordinates = 0.0 <= state.x <= 1.2 and 0.0 <= state.y <= 1.2

        if uses_normalized_coordinates:
            center_x = state.x * float(frame_w - 1)
            center_y = state.y * float(frame_h - 1)
            velocity_x = state.vx * float(frame_w - 1)
            velocity_y = state.vy * float(frame_h - 1)
        else:
            center_x = state.x
            center_y = state.y
            velocity_x = state.vx
            velocity_y = state.vy

        speed = float(np.hypot(velocity_x, velocity_y))
        trail_scale = max(1.0, min(4.0, speed / 3.5))

        head_x = int(np.clip(round(center_x), 0, frame_w - 1))
        head_y = int(np.clip(round(center_y), 0, frame_h - 1))
        tail_x = int(np.clip(round(center_x - velocity_x * trail_scale), 0, frame_w - 1))
        tail_y = int(np.clip(round(center_y - velocity_y * trail_scale), 0, frame_h - 1))

        overlay = frame_np.copy()
        cv2.line(
            overlay,
            (tail_x, tail_y),
            (head_x, head_y),
            color=(70, 220, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        radius = max(2, min(5, int(round(2.0 + speed * 0.12))))
        cv2.circle(
            overlay,
            (head_x, head_y),
            radius=radius,
            color=(90, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.78, frame_np, 0.22, 0.0, dst=frame_np)
