"""Gate B Frozen Codec Procedure and Metric Bounds.

Locks down:
- Rate ladder: Rungs C0, C1, C2, C3.
- Background stream: VVC low-delay background streaming with intra refresh (`-period 1`).
- Appearance crops: Compressed WebP actor crops.
- Lattice constraints: generation=False, residual=False, pose=False.
- Pre-registered metric thresholds, rot bounds, and monotonicity criteria.
- Anchor configurations: AV1 and VVC reference curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from src.contracts.config import PointstreamConfig


@dataclass(frozen=True)
class FrozenRungSpec:
    """Immutable specification for a rate ladder rung."""

    name: str
    bg_crf: int
    appearance_jpeg: int
    appearance_downscale: int
    motion_max_points: int
    summary: str
    stream_codec: str = "vvc"
    appearance_format: str = "webp"


FROZEN_RUNGS: tuple[FrozenRungSpec, ...] = (
    FrozenRungSpec(
        name="C0",
        bg_crf=63,
        appearance_jpeg=30,
        appearance_downscale=2,
        motion_max_points=8,
        summary="Coarsest operating point; shared VVC QP 63 background, WebP Q30 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    FrozenRungSpec(
        name="C1",
        bg_crf=55,
        appearance_jpeg=45,
        appearance_downscale=1,
        motion_max_points=16,
        summary="Intermediate low-rate point; VVC QP 55 background, WebP Q45 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    FrozenRungSpec(
        name="C2",
        bg_crf=48,
        appearance_jpeg=60,
        appearance_downscale=1,
        motion_max_points=24,
        summary="Target competitive point; VVC QP 48 background, WebP Q60 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    FrozenRungSpec(
        name="C3",
        bg_crf=42,
        appearance_jpeg=75,
        appearance_downscale=1,
        motion_max_points=32,
        summary="Highest rate point; VVC QP 42 background, WebP Q75 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
)


FROZEN_ANCHOR_QPS: tuple[int, ...] = (63, 55, 47, 39)

FROZEN_ANCHOR_PRESETS: dict[str, str] = {
    "av1": "8",
    "vvc": "medium",
}


def get_frozen_bounds(
    n_frames: int,
    height: int = 2160,
    width: int = 3840,
    n_scenes: int = 2,
) -> dict[str, Any]:
    """Return the frozen instrument-alarm bounds for confirmation."""
    raw_bytes = n_scenes * n_frames * height * width * 3
    return {
        "n_frames_per_scene": n_frames,
        "n_scenes": n_scenes,
        "decoded_shape_required": [n_scenes * n_frames, height, width, 3],
        "coded_bytes": {"low_exclusive": 0, "high_inclusive": raw_bytes + 1048576},
        "quality": {
            "vmaf": [0.0, 98.0],
            "psnr_y_db": [8.0, 55.0],
            "ssim": [0.0, 1.0],
        },
        "late_frame_last_minus_first": {
            "vmaf": [-25.0, 8.0],
            "psnr_y_db": [-8.0, 3.0],
        },
        "bd_rate_vmaf_percent": [-90.0, 300.0],
        "timing": {
            "required": ["encoder_seconds", "client_seconds", "evaluation_seconds", "attempt_wall"],
            "finite_nonnegative": True,
            "component_le_attempt_wall_tolerance_seconds": 1.0,
            "ranked_encode_decode_must_be_non_null": True,
        },
        "curve": {
            "max_adjacent_inversion_fraction_of_span": 0.05,
            "endpoint_rate_and_quality_inversion_is_alarm": True,
            "continuous_to_segmented_anchor_bytes_max_ratio": 1.05,
            "minimum_usable_points": 4,
            "minimum_vmaf_overlap": 5.0,
        },
        "checkpoint_gap_seconds_max": 3599.0,
        "nonresumable_operation_timeout_seconds": 3300.0,
    }


def configure_frozen_rung(
    base: PointstreamConfig,
    rung: FrozenRungSpec,
    *,
    context_id: str,
) -> PointstreamConfig:
    """Build immutable PointstreamConfig for a specific rate ladder rung."""
    bg = replace(
        base.background,
        method="panorama-stream",
        stream_codec=rung.stream_codec,
        stream_crf=rung.bg_crf,
        transport_scale=1.0,
    )
    if hasattr(bg, "context_id"):
        bg = replace(bg, context_id=context_id)

    app = replace(
        base.appearance,
        representation="compressed-image",
        jpeg_quality=rung.appearance_jpeg,
        downscale=rung.appearance_downscale,
        format=rung.appearance_format,
    )
    mot = replace(
        base.motion,
        max_points=rung.motion_max_points,
    )
    lattice = replace(
        base.lattice,
        generation=False,
        residual=False,
        pose=False,
    )
    return replace(
        base,
        background=bg,
        appearance=app,
        motion=mot,
        lattice=lattice,
    )


def validate_ledger(parts: dict[str, int], total_bytes: int) -> list[str]:
    """Verify that the payload parts balance exactly with transport total."""
    alarms: list[str] = []
    parts_sum = sum(parts.values())
    if parts_sum != total_bytes:
        alarms.append(f"Ledger does not balance: parts sum {parts_sum} != total {total_bytes}")
    return alarms


def check_adjacent_rungs(prev_row: dict[str, Any], curr_row: dict[str, Any]) -> list[str]:
    """Verify rate and ledger monotonicity between adjacent rungs."""
    alarms: list[str] = []
    prev_bytes = prev_row["bytes"]
    curr_bytes = curr_row["bytes"]
    name = curr_row["name"]
    prev_name = prev_row["name"]

    # Total bytes must not fall by > 5%
    if curr_bytes < prev_bytes * 0.95:
        alarms.append(
            f"Rung {name} total bytes {curr_bytes} fell by >5% from {prev_name} ({prev_bytes})"
        )

    # Background bytes must be nondecreasing when CRF changes
    prev_bg = prev_row.get("parts", {}).get("panorama", 0)
    curr_bg = curr_row.get("parts", {}).get("panorama", 0)
    if curr_bg < prev_bg:
        alarms.append(f"Rung {name} background bytes {curr_bg} < {prev_name} {prev_bg}")

    return alarms


__all__ = [
    "FROZEN_ANCHOR_PRESETS",
    "FROZEN_ANCHOR_QPS",
    "FROZEN_RUNGS",
    "FrozenRungSpec",
    "check_adjacent_rungs",
    "configure_frozen_rung",
    "get_frozen_bounds",
    "validate_ledger",
]
