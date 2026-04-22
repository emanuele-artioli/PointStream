from __future__ import annotations

import cv2
import numpy as np

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import FrameState, SceneActor, SceneActorReference, VideoChunk
from src.shared.tags import cpu_bound
from src.shared.track_id import scene_track_id_to_int


class ReferenceExtractor:
    """Extracts a compact JPEG visual reference per tracked player."""

    def __init__(self, jpeg_quality: int = 75, bbox_padding_ratio: float = 0.10) -> None:
        self._jpeg_quality = int(jpeg_quality)
        self._bbox_padding_ratio = float(bbox_padding_ratio)

    @cpu_bound
    def process(self, chunk: VideoChunk, frame_states: list[FrameState]) -> list[SceneActorReference]:
        metadata = probe_video_metadata(chunk.source_uri)
        frames = list(
            iter_video_frames_ffmpeg(
                chunk.source_uri,
                width=metadata.width,
                height=metadata.height,
            )
        )
        if not frames:
            return []

        frame_limit = min(int(chunk.num_frames), len(frames), len(frame_states))
        if frame_limit <= 0:
            return []

        selected: dict[str, tuple[int, list[int]]] = {}

        for frame_idx in range(frame_limit):
            for actor in frame_states[frame_idx].actors:
                if actor.class_name != "player":
                    continue

                bbox = self._clip_bbox(actor.bbox, frame_width=int(chunk.width), frame_height=int(chunk.height))
                if not self._is_confident_detection(actor=actor, bbox=bbox):
                    continue

                # Keep the first confident observation for each track to preserve
                # temporal identity consistency and avoid late-frame swaps.
                if actor.track_id not in selected:
                    selected[actor.track_id] = (frame_idx, bbox)

        references: list[SceneActorReference] = []
        for track_id_str, candidate in selected.items():
            frame_idx, bbox = candidate
            if frame_idx < 0 or frame_idx >= frame_limit:
                continue

            frame_bgr = frames[frame_idx]
            padded = self._add_padding(
                bbox=bbox,
                frame_width=int(chunk.width),
                frame_height=int(chunk.height),
                ratio=self._bbox_padding_ratio,
            )
            crop = frame_bgr[padded[1] : padded[3], padded[0] : padded[2]]
            if crop.size == 0:
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if not ok:
                continue

            references.append(
                SceneActorReference(
                    track_id=scene_track_id_to_int(track_id_str),
                    reference_crop_jpeg=encoded.tobytes(),
                )
            )

        references.sort(key=lambda item: item.track_id)
        return references

    def _is_confident_detection(self, actor: SceneActor, bbox: list[int]) -> bool:
        width = max(0, int(bbox[2]) - int(bbox[0]))
        height = max(0, int(bbox[3]) - int(bbox[1]))
        area = float(width * height)
        if area <= 4.0:
            return False

        # Prefer detections backed by segmentation masks when available.
        if actor.mask is not None:
            mask = np.asarray(actor.mask, dtype=np.uint8)
            if mask.ndim == 2 and mask.size > 0 and int(np.count_nonzero(mask)) > 0:
                return True

        # Fall back to pose confidence if present.
        if actor.pose_dw is not None:
            pose = np.asarray(actor.pose_dw, dtype=np.float32)
            if pose.ndim == 2 and pose.shape[1] >= 3 and pose.shape[0] > 0:
                conf = pose[:, 2]
                visible = int(np.count_nonzero(conf >= 0.20))
                if visible >= 4:
                    return True

        # Last-resort fallback when no mask/pose metadata is available.
        return area >= 16.0

    def _clip_bbox(self, bbox: list[float], frame_width: int, frame_height: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0, min(frame_width - 1, int(np.floor(x1))))
        clipped_y1 = max(0, min(frame_height - 1, int(np.floor(y1))))
        clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(np.ceil(x2))))
        clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(np.ceil(y2))))
        return [clipped_x1, clipped_y1, clipped_x2, clipped_y2]

    def _add_padding(self, bbox: list[int], frame_width: int, frame_height: int, ratio: float) -> list[int]:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(round(float(bw) * ratio))
        pad_y = int(round(float(bh) * ratio))

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(frame_width, x2 + pad_x)
        py2 = min(frame_height, y2 + pad_y)

        if px2 <= px1:
            px2 = min(frame_width, px1 + 1)
        if py2 <= py1:
            py2 = min(frame_height, py1 + 1)
        return [px1, py1, px2, py2]
