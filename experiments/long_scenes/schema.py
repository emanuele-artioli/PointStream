"""Data schemas for BP46 long eligible tennis-scene manifest.

Defines the machine-readable contracts for scene eligibility features, source
metadata, per-interval validation (48/96/192/384 frames), and the top-level
manifest following plans/BP46-long-tennis-scenes.md and plans/SESSION-REPORT.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

SCHEMA_ID: Final[str] = "pointstream.long_scenes.v1"

#: Exact target frame counts corresponding to 2s, 4s, 8s, 16s at 24 fps.
TARGET_SPANS: Final[tuple[int, ...]] = (48, 96, 192, 384)

#: Thresholds from project contracts (experiments/headroom/real.py, plate.py)
PASTE_MAE_MAX: Final[float] = 2.0
MAX_CANVAS_GROWTH: Final[float] = 2.5
MAX_CONSECUTIVE_MAD: Final[float] = 10.0


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata of the underlying 4K source video file."""

    video_file: str
    width: int
    height: int
    source_fps: float
    working_fps: float
    pix_fmt: str
    color_space: str
    color_primaries: str
    color_transfer: str
    sha256: str


@dataclass(frozen=True)
class CameraMotionFeatures:
    """Background/camera motion statistics over the candidate window."""

    consecutive_mad: float
    """Mean absolute difference between consecutive frames in grey levels."""
    vs_first_frame_mad: float
    """Mean absolute difference against frame 0."""
    last_vs_first_mad: float
    """Drift between the final frame and initial frame."""


@dataclass(frozen=True)
class PanoramaFeatures:
    """Panorama canvas growth and homography registration metrics."""

    canvas_width: int
    canvas_height: int
    growth_factor: float
    """Canvas area divided by source frame area (w*h). 1.0 = static."""
    registration_ok: bool


@dataclass(frozen=True)
class ObjectFeatures:
    """Foreground object metrics (tennis players)."""

    num_objects: int
    object_class: str
    player_pixel_fraction: float
    min_separation_px: float
    has_occlusion: bool
    track_continuity: bool


@dataclass(frozen=True)
class PasteBackFeatures:
    """Paste-back alignment verification against source frames."""

    convention: str
    opaque_mae: float
    threshold: float
    passes_threshold: bool


@dataclass(frozen=True)
class EligibilityFeatures:
    """Consolidated eligibility record evaluated before downstream encoding."""

    duration_24fps_frames: int
    camera_motion: CameraMotionFeatures
    panorama: PanoramaFeatures
    objects: ObjectFeatures
    paste_back: PasteBackFeatures
    route: str
    """'pointstream' if fully eligible; 'conventional_fallback' otherwise."""
    ineligibility_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntervalValidation:
    """Validation for an exact frame count (48, 96, 192, or 384 frames)."""

    frame_count: int
    start_frame: int
    end_frame: int
    status: str
    """'eligible', 'ineligible', or 'insufficient_duration'."""
    frame_hashes: dict[str, str]
    paste_back_mae: float
    canvas_growth: float
    failure_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SceneRecord:
    """Machine-readable manifest entry for one candidate tennis scene."""

    video: str
    scene: str
    t_start: float
    t_end: float
    duration: float
    cluster: str
    context_id: str
    role: str
    """'diagnostic_near_static', 'diagnostic_smooth_pan', 'confirmation', 'control_ineligible', or 'candidate'."""
    source_metadata: SourceMetadata
    eligibility: EligibilityFeatures
    intervals: dict[str, IntervalValidation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestPayload:
    """Complete machine-readable manifest payload."""

    schema: str
    created_utc: str
    source_data_root: str
    target_spans: list[int]
    diagnostic_videos: list[str]
    confirmation_videos: list[str]
    ineligible_controls: list[str]
    summary: dict[str, Any]
    scenes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
