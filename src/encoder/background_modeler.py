from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import CameraPose, PanoramaPacket, VideoChunk
from src.shared.tags import cpu_bound


class BackgroundModeler:
    """Real background modeler with KLT + dynamic keyframe panorama stitching."""

    @cpu_bound
    def process(
        self,
        chunk: VideoChunk,
        decoded_video_tensor: torch.Tensor | None = None,
        translation_threshold_px: float = 30.0,
    ) -> PanoramaPacket:
        frames, used_fallback = self._resolve_frames(chunk=chunk, decoded_video_tensor=decoded_video_tensor)
        num_frames, _height, _width, _ = frames.shape

        homographies = self._estimate_homographies(frames)
        selected_indices = self._select_keyframes(
            homographies=homographies,
            translation_threshold_px=translation_threshold_px,
        )
        panorama = self._median_panorama(frames=frames, homographies=homographies, selected_indices=selected_indices)

        assets_dir = Path(__file__).resolve().parents[2] / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        chunk_debug_path = assets_dir / f"debug_panorama_{chunk.chunk_id}.jpg"
        cv2.imwrite(str(chunk_debug_path), panorama)

        # Keep a canonical debug artifact only for non-fallback decoding paths
        # that produce an expanded stitched canvas.
        latest_debug_path = assets_dir / "debug_panorama.jpg"
        source_height = int(frames.shape[1])
        source_width = int(frames.shape[2])
        is_expanded_canvas = panorama.shape[0] > source_height or panorama.shape[1] > source_width
        if not used_fallback and is_expanded_canvas:
            cv2.imwrite(str(latest_debug_path), panorama)

        camera_poses = [
            CameraPose(
                frame_id=chunk.start_frame_id + frame_idx,
                tx=float(homography[0, 2]),
                ty=float(homography[1, 2]),
                tz=0.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
            )
            for frame_idx, homography in enumerate(homographies)
        ]

        return PanoramaPacket(
            chunk_id=chunk.chunk_id,
            panorama_uri=str(chunk_debug_path),
            frame_width=int(panorama.shape[1]),
            frame_height=int(panorama.shape[0]),
            camera_poses=camera_poses,
            panorama_image=panorama.tolist(),
            homography_matrices=[homography.tolist() for homography in homographies],
            selected_frame_indices=selected_indices,
        )

    def _resolve_frames(
        self,
        chunk: VideoChunk,
        decoded_video_tensor: torch.Tensor | None,
    ) -> tuple[np.ndarray, bool]:
        if decoded_video_tensor is not None:
            # Shape: [Frames, Channels, Height, Width] -> [Frames, Height, Width, Channels]
            frames = (
                decoded_video_tensor.clamp(0.0, 1.0)
                .mul(255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            return frames, False

        source_path = Path(chunk.source_uri)
        if source_path.exists() and source_path.is_file():
            metadata = probe_video_metadata(source_path)
            streamed_frames = list(
                iter_video_frames_ffmpeg(
                    source_path,
                    width=metadata.width,
                    height=metadata.height,
                )
            )
            if streamed_frames:
                return np.stack(streamed_frames, axis=0), False
            raise ValueError(f"FFmpeg yielded no decodable frames: {source_path}")

        return np.zeros((chunk.num_frames, chunk.height, chunk.width, 3), dtype=np.uint8), True

    def _estimate_homographies(self, frames: np.ndarray) -> list[np.ndarray]:
        num_frames = int(frames.shape[0])
        if num_frames == 0:
            return []

        frame0_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        points0 = cv2.goodFeaturesToTrack(
            frame0_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        identity = np.eye(3, dtype=np.float64)
        if points0 is None or len(points0) < 4:
            return [identity.copy() for _ in range(num_frames)]

        homographies: list[np.ndarray] = [identity.copy()]
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

        for frame_idx in range(1, num_frames):
            frame_gray = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY)
            next_points_guess = points0.copy()
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                frame0_gray,
                frame_gray,
                points0,
                next_points_guess,
                winSize=(21, 21),
                maxLevel=3,
                criteria=criteria,
            )

            if tracked is None or status is None:
                homographies.append(identity.copy())
                continue

            valid_mask = status.reshape(-1) == 1
            src_points = tracked.reshape(-1, 2)[valid_mask]
            dst_points = points0.reshape(-1, 2)[valid_mask]

            if src_points.shape[0] < 4:
                homographies.append(identity.copy())
                continue

            homography, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if homography is None:
                homographies.append(identity.copy())
            else:
                homographies.append(homography.astype(np.float64))

        return homographies

    def _select_keyframes(
        self,
        homographies: list[np.ndarray],
        translation_threshold_px: float,
    ) -> list[int]:
        if not homographies:
            return []

        selected = [0]
        last_selected_h = homographies[0]

        for frame_idx in range(1, len(homographies)):
            current_h = homographies[frame_idx]
            relative = current_h @ np.linalg.inv(last_selected_h)
            tx = float(relative[0, 2])
            ty = float(relative[1, 2])
            translation_distance = float(np.hypot(tx, ty))
            if translation_distance >= translation_threshold_px:
                selected.append(frame_idx)
                last_selected_h = current_h

        return selected

    def _median_panorama(
        self,
        frames: np.ndarray,
        homographies: list[np.ndarray],
        selected_indices: list[int],
    ) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        height, width = int(frames.shape[1]), int(frames.shape[2])
        if not selected_indices:
            selected_indices = [0]

        panorama_homographies, canvas_size = self._build_panorama_canvas(
            homographies=homographies,
            selected_indices=selected_indices,
            width=width,
            height=height,
        )
        canvas_width, canvas_height = canvas_size

        warped_frames: list[np.ndarray] = []
        for frame_idx in selected_indices:
            warped = cv2.warpPerspective(
                frames[frame_idx],
                panorama_homographies[frame_idx],
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_frames.append(warped)

        stacked = np.stack(warped_frames, axis=0)
        return np.median(stacked, axis=0).astype(np.uint8)

    def _build_panorama_canvas(
        self,
        homographies: list[np.ndarray],
        selected_indices: list[int],
        width: int,
        height: int,
    ) -> tuple[list[np.ndarray], tuple[int, int]]:
        base_corners = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float64,
        )
        corners_h = np.concatenate([base_corners, np.ones((4, 1), dtype=np.float64)], axis=1)

        transformed_corners: list[np.ndarray] = []
        for frame_idx in selected_indices:
            warped = (homographies[frame_idx] @ corners_h.T).T
            warped = warped[:, :2] / warped[:, 2:3]
            transformed_corners.append(warped)

        all_corners = np.concatenate(transformed_corners, axis=0)
        min_x = float(np.floor(np.min(all_corners[:, 0])))
        min_y = float(np.floor(np.min(all_corners[:, 1])))
        max_x = float(np.ceil(np.max(all_corners[:, 0])))
        max_y = float(np.ceil(np.max(all_corners[:, 1])))

        tx = -min(0.0, min_x)
        ty = -min(0.0, min_y)
        translation = np.array(
            [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        canvas_width = max(width, int(np.ceil(max_x + tx)) + 1)
        canvas_height = max(height, int(np.ceil(max_y + ty)) + 1)

        translated_homographies = [translation @ homography for homography in homographies]
        return translated_homographies, (canvas_width, canvas_height)
