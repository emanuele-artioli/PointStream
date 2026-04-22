from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import CameraPose, FrameState, PanoramaPacket, SceneActor, VideoChunk
from src.shared.tags import cpu_bound


class BackgroundModeler:
    """Real background modeler with KLT + dynamic keyframe panorama stitching."""

    @cpu_bound
    def process(
        self,
        chunk: VideoChunk,
        decoded_video_tensor: torch.Tensor | None = None,
        frame_states: list[FrameState] | None = None,
        translation_threshold_px: float = 30.0,
    ) -> PanoramaPacket:
        frames = self._resolve_frames(chunk=chunk, decoded_video_tensor=decoded_video_tensor)
        num_frames, _height, _width, _ = frames.shape

        homographies = self._estimate_homographies(frames)
        selected_indices = self._select_keyframes(
            homographies=homographies,
            translation_threshold_px=translation_threshold_px,
        )
        panorama = self._median_panorama(
            frames=frames,
            homographies=homographies,
            selected_indices=selected_indices,
            frame_states=frame_states,
        )

        debug_root = self._resolve_debug_root()
        debug_root.mkdir(parents=True, exist_ok=True)
        chunk_debug_path = debug_root / f"debug_panorama_{chunk.chunk_id}.jpg"
        cv2.imwrite(str(chunk_debug_path), panorama)

        # Keep a canonical debug artifact only for non-fallback decoding paths
        # that produce an expanded stitched canvas.
        latest_debug_path = debug_root / "debug_panorama.jpg"
        source_height = int(frames.shape[1])
        source_width = int(frames.shape[2])
        is_expanded_canvas = panorama.shape[0] > source_height or panorama.shape[1] > source_width
        if is_expanded_canvas:
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

    def _resolve_debug_root(self) -> Path:
        override = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
        if override:
            return Path(override)

        project_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return project_root / "outputs" / timestamp / "debug"

    def _resolve_frames(
        self,
        chunk: VideoChunk,
        decoded_video_tensor: torch.Tensor | None,
    ) -> np.ndarray:
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
            return frames

        source_path = Path(chunk.source_uri)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"BackgroundModeler source video not found: {source_path}")

        metadata = probe_video_metadata(source_path)
        streamed_frames: list[np.ndarray] = []
        for frame in iter_video_frames_ffmpeg(
            source_path,
            width=metadata.width,
            height=metadata.height,
        ):
            streamed_frames.append(frame)
            if len(streamed_frames) >= chunk.num_frames:
                break
        if streamed_frames:
            return np.stack(streamed_frames, axis=0)

        raise ValueError(f"FFmpeg yielded no decodable frames: {source_path}")

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
        frame_states: list[FrameState] | None = None,
    ) -> np.ndarray:
        if frames.shape[0] == 0:
            raise ValueError("BackgroundModeler received zero frames; cannot compute panorama")

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

        warped_frames_masked: list[np.ndarray] = []
        warped_frames_unmasked: list[np.ndarray] = []
        source_valid = np.full((height, width), 255, dtype=np.uint8)
        for frame_idx in selected_indices:
            warped = cv2.warpPerspective(
                np.asarray(frames[frame_idx], dtype=np.float32),
                panorama_homographies[frame_idx],
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_valid = cv2.warpPerspective(
                source_valid,
                panorama_homographies[frame_idx],
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )

            exclusion_mask_src = self._build_actor_exclusion_mask(
                frame_idx=frame_idx,
                frame_states=frame_states,
                frame_height=height,
                frame_width=width,
            )
            warped_exclusion = cv2.warpPerspective(
                exclusion_mask_src,
                panorama_homographies[frame_idx],
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )

            warped_unmasked = warped.copy()
            warped_unmasked[warped_valid < 127] = np.nan
            warped_frames_unmasked.append(warped_unmasked)

            warped_masked = warped.copy()
            combined_invalid = (warped_valid < 127) | (warped_exclusion > 0)
            warped_masked[combined_invalid] = np.nan
            warped_frames_masked.append(warped_masked)

        masked_median = self._safe_nanmedian(np.stack(warped_frames_masked, axis=0))
        unmasked_median = self._safe_nanmedian(np.stack(warped_frames_unmasked, axis=0))

        filled = np.where(np.isnan(masked_median), unmasked_median, masked_median)
        if bool(np.any(np.isnan(filled))):
            filled = self._inpaint_nan_holes(filled)

        filled = np.nan_to_num(filled, nan=0.0, posinf=255.0, neginf=0.0)
        return np.asarray(np.clip(filled, 0.0, 255.0), dtype=np.uint8)

    def _safe_nanmedian(self, stacked: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmedian(stacked, axis=0)

    def _inpaint_nan_holes(self, image_with_nan: np.ndarray) -> np.ndarray:
        holes = np.any(np.isnan(image_with_nan), axis=2)
        if not bool(np.any(holes)):
            return image_with_nan

        base = np.nan_to_num(image_with_nan, nan=0.0, posinf=255.0, neginf=0.0)
        base_u8 = np.asarray(np.clip(base, 0.0, 255.0), dtype=np.uint8)
        hole_mask = np.asarray(holes, dtype=np.uint8) * 255
        inpainted = cv2.inpaint(base_u8, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return np.asarray(inpainted, dtype=np.float32)

    def _build_actor_exclusion_mask(
        self,
        frame_idx: int,
        frame_states: list[FrameState] | None,
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        if frame_states is None or frame_idx < 0 or frame_idx >= len(frame_states):
            return np.zeros((frame_height, frame_width), dtype=np.uint8)

        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        state = frame_states[frame_idx]
        for actor in state.actors:
            if actor.class_name not in {"player", "racket"}:
                continue

            x1, y1, x2, y2 = self._clip_bbox(actor.bbox, frame_width=frame_width, frame_height=frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            if actor.mask is None:
                polygon_mask = self._build_polygon_mask(actor=actor, width=x2 - x1, height=y2 - y1)
                if polygon_mask is None:
                    # Do not exclude full bbox rectangles when segmentation is missing.
                    continue

                roi = mask[y1:y2, x1:x2]
                roi[polygon_mask > 0] = 255
                mask[y1:y2, x1:x2] = roi
                continue

            local = np.asarray(actor.mask, dtype=np.uint8)
            if local.ndim != 2 or local.size == 0:
                mask[y1:y2, x1:x2] = 255
                continue

            resized = cv2.resize(local, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            roi = mask[y1:y2, x1:x2]
            roi[resized > 0] = 255
            mask[y1:y2, x1:x2] = roi

        return mask

    def _build_polygon_mask(self, actor: SceneActor, width: int, height: int) -> np.ndarray | None:
        polygons = getattr(actor, "mask_polygons", None)
        if polygons is None or not isinstance(polygons, list) or len(polygons) == 0:
            return None

        local_mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            poly_np = np.asarray(polygon, dtype=np.float32)
            if poly_np.ndim != 2 or poly_np.shape[1] != 2 or poly_np.shape[0] < 3:
                continue

            poly_i = np.round(poly_np).astype(np.int32)
            poly_i[:, 0] = np.clip(poly_i[:, 0], 0, width - 1)
            poly_i[:, 1] = np.clip(poly_i[:, 1], 0, height - 1)
            cv2.fillPoly(local_mask, [poly_i.reshape(-1, 1, 2)], color=(255.0,))

        if int(np.count_nonzero(local_mask)) == 0:
            return None
        return local_mask

    def _clip_bbox(self, bbox: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(y2))))
        return clipped_x1, clipped_y1, clipped_x2, clipped_y2

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
