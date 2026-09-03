"""Panorama / background resolution, independently testable.

The encoder transmits a plate plus per-frame homographies (frame → plate).
Reconstruction inverts each map and warps the plate back to frame size. A
delta plate is the signed difference against the previous *decoded* plate
for that scene — the same 128-offset arithmetic the background component
uses, restated here so this layer does not import components.

When the background stage is off, there is no plate: frames are zeros and
the residual carries the background. That is a larger residual, not a
smaller one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline.reconstruction.clips import Clip
from src.pipeline.reconstruction.device import DeviceDecision, DevicePolicy

MODE_FULL = "full"
MODE_DELTA = "delta"
MODE_NONE = "none"


@dataclass(frozen=True)
class BackgroundModelView:
    """What reconstruction needs from a background transmission.

    C2 unpacks its background artifact into this. The pipeline does not
    import ``BackgroundArtifact`` — that type lives in components.
    ``plate`` is already-decoded pixels: a full plate, or a delta image
    still in the 128-offset uint8 representation.
    """

    plate: np.ndarray | None
    homographies: tuple[tuple[float, ...], ...] = ()
    mode: str = MODE_FULL
    deferred_to_residual: bool = False
    scene_id: str | None = None
    width: int = 0
    height: int = 0
    # Length of the sidecar payload the client actually receives, when the
    # transmission ran. ``None`` means nobody coded this plate, so a caller
    # must not present its pixel size as a transmitted cost (BP24).
    payload_bytes: int | None = None
    # Length of the charged geometry header. Zero when the method does not
    # send original/coded size metadata. Counted as metadata, not panorama.
    geometry_header_bytes: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {MODE_FULL, MODE_DELTA, MODE_NONE}:
            raise ValueError(
                f"background mode must be {MODE_FULL!r}, {MODE_DELTA!r} or "
                f"{MODE_NONE!r}; got {self.mode!r}."
            )
        if self.deferred_to_residual or self.mode == MODE_NONE:
            return
        if self.plate is None:
            raise ValueError(
                "a background that is not deferred to the residual must carry plate pixels."
            )
        array = np.asarray(self.plate)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"background plate must be (H, W, 3); got {array.shape}.")


def apply_plate_delta(previous: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Inverse of the 128-offset signed plate difference."""
    prev = _as_plate(previous, "previous")
    delta = _as_plate(diff, "delta")
    if prev.shape != delta.shape:
        raise ValueError(
            f"panorama delta shape mismatch: previous={prev.shape}, delta={delta.shape}."
        )
    return np.clip(prev.astype(np.int16) + delta.astype(np.int16) - 128, 0, 255).astype(np.uint8)


def warp_plate(
    plate: np.ndarray,
    homographies: tuple[tuple[float, ...], ...],
    *,
    height: int,
    width: int,
    frame_count: int,
) -> Clip:
    """Warp a plate to ``frame_count`` frames of ``height`` × ``width``.

    Homographies map frame → plate. An identity map on a plate that already
    matches the frame size is a copy — that is the bit-identity path for a
    static camera, not an interpolation coincidence.
    """
    source = _as_plate(plate, "plate")
    maps = _homography_stack(homographies, frame_count)
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    for index, matrix in enumerate(maps):
        frames[index] = _warp_one(source, matrix, height=height, width=width)
    return frames


class BackgroundResolver:
    """Resolves plates across chunks (delta needs the previous decoded plate)."""

    def __init__(self) -> None:
        self._plates: dict[str, np.ndarray] = {}

    def resolve(self, view: BackgroundModelView) -> np.ndarray | None:
        """The plate the client holds after this transmission, or None if deferred."""
        if view.deferred_to_residual or view.mode == MODE_NONE:
            return None
        assert view.plate is not None
        decoded = np.asarray(view.plate, dtype=np.uint8)
        if view.mode == MODE_DELTA:
            scene = view.scene_id
            if scene is None or scene not in self._plates:
                raise ValueError(
                    "a delta panorama needs a previously decoded plate for the same "
                    f"scene (scene_id={scene!r}). The first chunk of a scene must be full."
                )
            decoded = apply_plate_delta(self._plates[scene], decoded)
        if view.scene_id is not None:
            self._plates[view.scene_id] = decoded
        return decoded

    def frames_for(
        self,
        view: BackgroundModelView | None,
        *,
        frame_count: int,
        height: int,
        width: int,
        policy: DevicePolicy | None = None,
    ) -> tuple[Clip, DeviceDecision]:
        """Per-frame background, or zeros when the stage is off / deferred."""
        _ = policy  # Warp is CPU/cv2; the policy is accepted so callers share one object.
        decision = DeviceDecision("cpu")
        if view is None:
            return _zeros(frame_count, height, width), decision
        plate = self.resolve(view)
        if plate is None:
            return _zeros(frame_count, height, width), decision
        warped = warp_plate(
            plate,
            view.homographies,
            height=height,
            width=width,
            frame_count=frame_count,
        )
        return warped, decision


def _zeros(frame_count: int, height: int, width: int) -> Clip:
    return np.zeros((frame_count, height, width, 3), dtype=np.uint8)


def _as_plate(image: np.ndarray, path: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path} must be (H, W, 3); got {array.shape}.")
    return array


def _homography_stack(
    homographies: tuple[tuple[float, ...], ...], frame_count: int
) -> list[np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    if not homographies:
        return [identity.copy() for _ in range(frame_count)]
    maps: list[np.ndarray] = []
    for item in homographies:
        values = np.asarray(item, dtype=np.float64).reshape(-1)
        if values.size != 9:
            raise ValueError(f"homography must be 9 values row-major; got {values.size}.")
        maps.append(values.reshape(3, 3))
    if len(maps) < frame_count:
        maps.extend(identity.copy() for _ in range(frame_count - len(maps)))
    return maps[:frame_count]


def _is_identity(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix, np.eye(3), atol=1e-7))


def _warp_one(plate: np.ndarray, frame_to_plate: np.ndarray, *, height: int, width: int) -> np.ndarray:
    plate_h, plate_w = plate.shape[:2]
    if _is_identity(frame_to_plate) and plate_h == height and plate_w == width:
        return np.asarray(plate, dtype=np.uint8).copy()
    try:
        inverse = np.linalg.inv(frame_to_plate)
    except np.linalg.LinAlgError as exc:
        raise ValueError("homography is singular; cannot warp the plate back to the frame.") from exc
    return cv2.warpPerspective(
        plate,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
