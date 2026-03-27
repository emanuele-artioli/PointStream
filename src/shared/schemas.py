from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TensorSpec(BaseModel):
    """Describes tensor shape/dtype for metadata-level contracts."""

    name: str
    shape: list[int]
    dtype: str


class ObjectClass(str, Enum):
    PERSON = "person"
    RACKET = "racket"
    BALL = "ball"
    BACKGROUND = "background"


class VideoChunk(BaseModel):
    chunk_id: str
    source_uri: str
    start_frame_id: int = Field(ge=0)
    fps: float = Field(gt=0)
    num_frames: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SceneActor(BaseModel):
    track_id: str
    class_name: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    mask: list[list[int]] | None = None
    pose_dw: list[list[float]] | None = None


class FrameState(BaseModel):
    frame_id: int = Field(ge=0)
    actors: list[SceneActor] = Field(default_factory=list)


class ObjectState(BaseModel):
    frame_id: int = Field(ge=0)
    object_id: str
    object_class: ObjectClass


class KeyframeEvent(ObjectState):
    event_type: Literal["keyframe"] = "keyframe"
    coordinates: list[float] = Field(
        description="Sparse keypoint or parametric coordinates payload."
    )


class InterpolateCommandEvent(ObjectState):
    event_type: Literal["interpolate"] = "interpolate"
    target_frame_id: int = Field(ge=0)
    method: Literal["linear", "spline"] = "linear"


class StaticCommandEvent(ObjectState):
    event_type: Literal["static"] = "static"
    hold_until_frame_id: int = Field(ge=0)


SemanticEvent = KeyframeEvent | InterpolateCommandEvent | StaticCommandEvent


class CameraPose(BaseModel):
    frame_id: int = Field(ge=0)
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


class PanoramaPacket(BaseModel):
    chunk_id: str
    panorama_uri: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    camera_poses: list[CameraPose]
    panorama_image: list[list[list[int]]]
    homography_matrices: list[list[list[float]]]
    selected_frame_indices: list[int] = Field(default_factory=list)


class ActorPacket(BaseModel):
    chunk_id: str
    object_id: str
    appearance_embedding_spec: TensorSpec
    pose_tensor_spec: TensorSpec
    events: list[SemanticEvent]


class RigidObjectPacket(BaseModel):
    chunk_id: str
    object_id: str
    trajectory_spec: TensorSpec
    events: list[SemanticEvent]


class BallPacket(BaseModel):
    chunk_id: str
    object_id: str = "ball_0"
    trajectory_spec: TensorSpec
    events: list[SemanticEvent]


class ResidualPacket(BaseModel):
    chunk_id: str
    codec: str = "hevc-placeholder"
    residual_video_uri: str


class EncodedChunkPayload(BaseModel):
    chunk: VideoChunk
    panorama: PanoramaPacket
    actors: list[ActorPacket]
    rigid_objects: list[RigidObjectPacket]
    ball: BallPacket
    residual: ResidualPacket


class DecodedChunkResult(BaseModel):
    chunk_id: str
    output_uri: str
    num_frames: int
    width: int
    height: int
