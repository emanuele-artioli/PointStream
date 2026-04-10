from __future__ import annotations

import cv2
import numpy as np

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import FrameState, SceneActorReference, VideoChunk
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

        selected: dict[str, tuple[int, list[int], float, float]] = {}
        center_x = float(chunk.width) * 0.5
        center_y = float(chunk.height) * 0.5

        for frame_idx in range(frame_limit):
            for actor in frame_states[frame_idx].actors:
                if actor.class_name != "player":
                    continue

                bbox = self._clip_bbox(actor.bbox, frame_width=int(chunk.width), frame_height=int(chunk.height))
                area = float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))
                if area <= 1.0:
                    continue

                actor_center_x = 0.5 * (bbox[0] + bbox[2])
                actor_center_y = 0.5 * (bbox[1] + bbox[3])
                center_distance = float(np.hypot(actor_center_x - center_x, actor_center_y - center_y))
                previous = selected.get(actor.track_id)
                if previous is None:
                    selected[actor.track_id] = (frame_idx, bbox, area, center_distance)
                    continue

                _prev_idx, _prev_bbox, prev_area, prev_center_distance = previous
                if area > prev_area or (abs(area - prev_area) < 1e-4 and center_distance < prev_center_distance):
                    selected[actor.track_id] = (frame_idx, bbox, area, center_distance)

        references: list[SceneActorReference] = []
        for track_id_str, candidate in selected.items():
            frame_idx, bbox, _area, _distance = candidate
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
