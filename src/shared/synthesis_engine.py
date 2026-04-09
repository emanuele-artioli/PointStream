from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch

from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.schemas import ActorPacket, EncodedChunkPayload
from src.shared.tags import gpu_bound


@dataclass(frozen=True)
class SynthesisResult:
    """Dense synthesized reconstruction output."""

    # Shape: [Frames, Channels, Height, Width] in BGR uint8.
    frames_bgr: torch.Tensor


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

    def _configure_cuda_determinism_env(self) -> None:
        # CuBLAS requires this variable for reproducible GEMM-based kernels on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @gpu_bound
    def synthesize(self, payload: EncodedChunkPayload) -> SynthesisResult:
        self._set_global_seed(self.seed)

        dense_pose_stream = self._unroll_sparse_actor_poses(payload)
        background_frames = self._reconstruct_background_frames(payload)
        composited_frames = self._composite_mock_skeletons(background_frames, dense_pose_stream)
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
        batched_panorama = panorama_tensor.expand(frame_count, -1, -1, -1)

        warped = kornia.geometry.transform.warp_perspective(
            src=batched_panorama,
            M=inverse_h,
            dsize=(output_height, output_width),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return warped.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous()

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

            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)
            out_frames.append(frame_tensor)

        return torch.stack(out_frames, dim=0).to(self.device)
