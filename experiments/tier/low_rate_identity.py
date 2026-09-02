"""The frames a low-rate file was measured on. Filename and payload must agree.

A stem that only names the video and the codec can load a 48-frame curve against
a 96-frame sweep and look like a comparison. The identity lives in the path
*and* inside the JSON; loaders refuse a mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.tier.low_rate_validate import DECLARED_FPS, OUT_DIR


IDENTITY_KEYS: tuple[str, ...] = (
    "video",
    "scenes",
    "frames_per_scene",
    "fps",
    "codec",
    "source",
    "implementation",
    "preset",
)


def input_identity(
    *,
    video: str,
    scenes: Sequence[str],
    frames_per_scene: int,
    codec: str,
    fps: float = DECLARED_FPS,
) -> dict[str, Any]:
    return {
        "video": str(video),
        "scenes": [str(name) for name in scenes],
        "frames_per_scene": int(frames_per_scene),
        "fps": float(fps),
        "codec": str(codec),
    }


def identity_slug(identity: Mapping[str, Any]) -> str:
    scenes = "+".join(str(name) for name in identity["scenes"])
    return (
        f"{identity['video']}-{scenes}-n{int(identity['frames_per_scene'])}"
        f"-{identity['codec']}"
    )


def references_path(identity: Mapping[str, Any], *, root: Path | None = None) -> Path:
    return (root or OUT_DIR) / f"references-{identity_slug(identity)}.json"


def sweep_path(identity: Mapping[str, Any], *, root: Path | None = None) -> Path:
    return (root or OUT_DIR) / f"sweep-{identity_slug(identity)}.json"


def checkpoint_dir(report_path: Path) -> Path:
    return report_path.with_name(report_path.stem + ".points")


def assert_same_input(found: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse a curve that was measured on different scenes, duration, fps or codec."""
    for key in IDENTITY_KEYS:
        got = found.get(key)
        want = expected.get(key)
        if key == "scenes":
            got = [str(item) for item in (got or [])]
            want = [str(item) for item in (want or [])]
        elif key == "fps" and got is not None and want is not None:
            if abs(float(got) - float(want)) < 1e-6:
                continue
        elif key == "frames_per_scene" and got is not None and want is not None:
            got = int(got)
            want = int(want)
        if got != want:
            raise SystemExit(
                f"curve input {key}={got!r} does not match this run {want!r}. "
                "Refusing a silent comparison across different scenes or durations."
            )


__all__ = [
    "IDENTITY_KEYS",
    "assert_same_input",
    "checkpoint_dir",
    "identity_slug",
    "input_identity",
    "references_path",
    "sweep_path",
]
