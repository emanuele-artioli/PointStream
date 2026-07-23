"""Packing actors into the transmitted payload.

Every byte counted here is a byte the Residual Guarantee must earn back,
so the size accounting is part of the contract, not diagnostics."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np
import torch
from src.shared.schemas import (
    ActorMaskFrame,
    ActorPacket,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    SceneActor,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.mask_codec import encode_binary_mask, encode_polygon_mask, normalize_mask_codec
from src.encoder.actors.weights import _clip_bbox
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


@dataclass
class PayloadEncoder:
    pose_delta_threshold: float = 20.0
    include_mask_metadata: bool = False
    metadata_mask_codec: str = "auto"

    def __post_init__(self) -> None:
        self._last_transmitted_pose_coords: dict[str, np.ndarray] = {}
        codec_raw = str(self.metadata_mask_codec).strip()
        if codec_raw == "auto":
            codec_raw = "poly-v1"
        codec_normalized = codec_raw.strip().lower().replace("_", "-")
        if codec_normalized in {"segmenter-native", "yolo-native"}:
            self.metadata_mask_codec = "segmenter-native"
        else:
            self.metadata_mask_codec = normalize_mask_codec(codec_normalized)

    def encode(self, chunk: VideoChunk, frame_states: list[FrameState]) -> list[ActorPacket]:
        actor_events: dict[str, list[SemanticEvent]] = {}
        actor_masks: dict[str, list[ActorMaskFrame]] = {}

        for state in frame_states:
            frame_id = chunk.start_frame_id + state.frame_id
            for actor in state.actors:
                if actor.class_name != "player":
                    continue

                if self.include_mask_metadata:
                    maybe_mask = self._encode_mask_frame(chunk=chunk, actor=actor, frame_id=frame_id)
                    if maybe_mask is not None:
                        actor_masks.setdefault(actor.track_id, []).append(maybe_mask)

                if actor.pose_dw is None:
                    continue

                coords = np.asarray(actor.pose_dw, dtype=np.float32)
                if coords.shape != (18, 3):
                    continue

                key = actor.track_id
                events = actor_events.setdefault(key, [])
                flattened = coords.reshape(-1)
                last = self._last_transmitted_pose_coords.get(key)

                if last is None:
                    events.append(
                        KeyframeEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            coordinates=flattened.tolist(),
                        )
                    )
                    self._last_transmitted_pose_coords[key] = flattened
                    continue

                delta = float(np.linalg.norm(flattened - last, ord=2))
                if delta >= self.pose_delta_threshold:
                    events.append(
                        KeyframeEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            coordinates=flattened.tolist(),
                        )
                    )
                    self._last_transmitted_pose_coords[key] = flattened
                else:
                    events.append(
                        InterpolateCommandEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            target_frame_id=min(frame_id + 1, chunk.start_frame_id + chunk.num_frames - 1),
                            method="linear",
                        )
                    )

        actor_ids = sorted(set(actor_events.keys()) | set(actor_masks.keys()))

        if not actor_ids:
            return []

        packets: list[ActorPacket] = []
        for actor_id in actor_ids[:2]:
            packets.append(
                ActorPacket(
                    chunk_id=chunk.chunk_id,
                    object_id=actor_id,
                    appearance_embedding_spec=TensorSpec(
                        name="actor_appearance",
                        shape=[1, 2, 256],
                        dtype=str(torch.float32),
                    ),
                    pose_tensor_spec=TensorSpec(
                        name="actor_pose_dw",
                        shape=[1, chunk.num_frames, 18, 3],
                        dtype=str(torch.float32),
                    ),
                    events=actor_events.get(actor_id, []),
                    mask_frames=actor_masks.get(actor_id, []),
                )
            )

        return packets

    def _encode_mask_frame(self, chunk: VideoChunk, actor: SceneActor, frame_id: int) -> ActorMaskFrame | None:
        if actor.mask is None:
            return None

        raw_mask = np.asarray(actor.mask, dtype=np.float32)
        if raw_mask.ndim != 2 or raw_mask.size == 0:
            return None

        if self.metadata_mask_codec == "segmenter-native":
            native_mask_frame = self._encode_segmenter_native_mask(
                chunk=chunk,
                actor=actor,
                frame_id=frame_id,
                mask_shape=(int(raw_mask.shape[0]), int(raw_mask.shape[1])),
            )
            if native_mask_frame is not None:
                return native_mask_frame

        try:
            codec_name, encoded_payload, mask_height, mask_width = encode_binary_mask(
                raw_mask,
                codec="auto" if self.metadata_mask_codec == "segmenter-native" else self.metadata_mask_codec,
            )
        except Exception:
            return None

        clipped_bbox = _clip_bbox(actor.bbox, width=int(chunk.width), height=int(chunk.height))
        x1 = int(np.floor(clipped_bbox[0]))
        y1 = int(np.floor(clipped_bbox[1]))
        x2 = int(np.ceil(clipped_bbox[2]))
        y2 = int(np.ceil(clipped_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        return ActorMaskFrame(
            frame_id=int(frame_id),
            bbox=[x1, y1, x2, y2],
            mask_codec=codec_name,
            mask_payload=encoded_payload,
            mask_height=int(mask_height),
            mask_width=int(mask_width),
            source="source",
        )

    def _encode_segmenter_native_mask(
        self,
        chunk: VideoChunk,
        actor: SceneActor,
        frame_id: int,
        mask_shape: tuple[int, int],
    ) -> ActorMaskFrame | None:
        if not actor.mask_polygons:
            return None

        mask_height, mask_width = mask_shape
        if mask_height <= 0 or mask_width <= 0:
            return None

        try:
            codec_name, encoded_payload, encoded_h, encoded_w = encode_polygon_mask(
                actor.mask_polygons,
                height=int(mask_height),
                width=int(mask_width),
                codec="poly-v1",
            )
        except Exception:
            return None

        clipped_bbox = _clip_bbox(actor.bbox, width=int(chunk.width), height=int(chunk.height))
        x1 = int(np.floor(clipped_bbox[0]))
        y1 = int(np.floor(clipped_bbox[1]))
        x2 = int(np.ceil(clipped_bbox[2]))
        y2 = int(np.ceil(clipped_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        return ActorMaskFrame(
            frame_id=int(frame_id),
            bbox=[x1, y1, x2, y2],
            mask_codec=codec_name,
            mask_payload=encoded_payload,
            mask_height=int(encoded_h),
            mask_width=int(encoded_w),
            source="source",
        )
