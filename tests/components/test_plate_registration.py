"""Required behaviour: the plate's registration recovers real camera motion.

`build_plate` is the panorama the runner calls from BP29 onward, and a plate is
only worth its extra pixels if the homographies that come with it are right.
The property tested here is the one a wrong fit breaks quietly: a *known*
translation must come back as that translation, to within a pixel.

This is the instrument check that caught the fit BP29 shipped with. A loose
RANSAC threshold produced a homography that halved a real pan and spent the
remainder on a spurious zoom — perfectly stable, monotone-looking, and wrong,
which is exactly the failure a synthetic ground truth catches and an eyeball
does not.

Deliberately not tested here: OpenCV's own LK and RANSAC implementations, and
the absolute quality of a stitch on real broadcast content (that is a
measurement, in `outputs/bp29-panorama/`, not an assertion).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.components.background.plate import (
    MAX_CANVAS_SCALE,
    build_plate,
    estimate_homographies,
)


def _texture(height: int, width: int, seed: int = 7) -> np.ndarray:
    """A blocky pattern with corners everywhere, so tracking is not the variable."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(height // 4, width // 4, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _pan(n_frames: int, height: int, width: int, step: int) -> np.ndarray:
    """`n_frames` windows onto one texture, each `step` px right of the last."""
    base = _texture(height, width + step * n_frames + 8)
    return np.stack([base[:, k * step : k * step + width] for k in range(n_frames)])


def _centre_shift(matrix: np.ndarray, height: int, width: int) -> tuple[float, float]:
    centre = np.array([[[width / 2.0, height / 2.0]]], dtype=np.float64)
    moved = cv2.perspectiveTransform(centre, np.asarray(matrix, dtype=np.float64)).reshape(2)
    return float(moved[0] - width / 2.0), float(moved[1] - height / 2.0)


def _plate_errors(
    plate: np.ndarray, background: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[float, float, float]:
    """Plate error inside a box, outside it, and against the occluding object.

    The "outside" number is the control. A plate is resampled by its own warp
    even when the estimated map is a fraction of a pixel from identity, so some
    error against the clean background is the floor for *every* pixel, masked or
    not. Comparing the masked region against that floor is what separates
    "the mask worked" from "the mask left a scar".
    """
    y1, y2, x1, x2 = box
    height, width = background.shape[:2]
    cropped = np.asarray(plate[:height, :width], dtype=np.int32)
    truth = background.astype(np.int32)
    elsewhere = np.ones((height, width), dtype=bool)
    elsewhere[y1:y2, x1:x2] = False
    inside = float(np.mean(np.abs(cropped[y1:y2, x1:x2] - truth[y1:y2, x1:x2])))
    outside = float(np.mean(np.abs(cropped[elsewhere] - truth[elsewhere])))
    to_object = float(np.mean(np.abs(cropped[y1:y2, x1:x2])))
    return inside, outside, to_object


def to_background_is_clean(inside: float, outside: float, to_object: float) -> bool:
    """The masked region is the background, at the plate's own accuracy.

    Two conditions, and the loose one is deliberate. A masked region's median
    is taken over fewer samples than the rest of the plate — that is what
    masking means — so it carries somewhat more resampling error even when it
    is perfectly clean, and the factor of two is headroom for that, not for a
    scar. The condition that actually catches a failure is the second: an
    object that survived would sit at ``to_object`` near zero, one to two
    orders of magnitude away, not at 1.2x the plate's floor.
    """
    return inside <= max(outside * 2.0, 1.0) and inside < to_object / 10.0


def test_a_known_pan_comes_back_as_that_pan() -> None:
    height, width, step, n_frames = 96, 128, 3, 6
    frames = _pan(n_frames, height, width, step)
    maps = estimate_homographies(frames)
    assert len(maps) == n_frames
    for index, matrix in enumerate(maps):
        dx, dy = _centre_shift(matrix, height, width)
        assert dx == pytest.approx(index * step, abs=1.0), (
            f"frame {index} panned {index * step} px and the fit says {dx:.2f}. "
            "A fit that recovers only part of a translation spends the rest on "
            "a zoom, which ghosts the plate."
        )
        assert dy == pytest.approx(0.0, abs=1.0)


def test_a_still_camera_gets_an_identity_map() -> None:
    """The control. Registration that moves a static clip is fitting noise."""
    frames = np.repeat(_texture(96, 128)[None, ...], 5, axis=0)
    for matrix in estimate_homographies(frames):
        dx, dy = _centre_shift(matrix, 96, 128)
        assert abs(dx) < 0.1
        assert abs(dy) < 0.1


def test_the_canvas_grows_with_the_pan_and_the_plate_covers_it() -> None:
    height, width, step, n_frames = 96, 128, 3, 6
    frames = _pan(n_frames, height, width, step)
    plate, packed = build_plate(frames)
    assert plate.shape[1] > width, "a panned clip must produce a plate wider than a frame"
    assert plate.shape[1] <= MAX_CANVAS_SCALE * width
    assert len(packed) == n_frames
    assert all(len(row) == 9 for row in packed)


def test_the_plate_matches_later_frames_better_than_the_first_frame_does() -> None:
    """The whole argument for a panorama, stated as an assertion.

    A plate is worth more pixels only if warping it back reproduces frames the
    first frame cannot. The last frame of a pan is the case that decides it.
    """
    height, width, step, n_frames = 96, 128, 3, 6
    frames = _pan(n_frames, height, width, step)
    plate, packed = build_plate(frames)
    last = n_frames - 1
    matrix = np.asarray(packed[last], dtype=np.float64).reshape(3, 3)
    warped = cv2.warpPerspective(
        plate, np.linalg.inv(matrix), (width, height), flags=cv2.INTER_LINEAR
    )
    from_plate = float(np.mean(np.abs(warped.astype(np.int32) - frames[last].astype(np.int32))))
    from_first = float(
        np.mean(np.abs(frames[0].astype(np.int32) - frames[last].astype(np.int32)))
    )
    assert from_plate < from_first / 2.0, (
        f"plate error {from_plate:.2f} against first-frame error {from_first:.2f}; "
        "a panorama that does not beat the frame it replaces is costing bytes for nothing"
    )


def test_a_masked_object_does_not_burn_into_the_plate() -> None:
    """A thing that sits still for part of the clip is not scenery."""
    height, width, n_frames = 96, 128, 6
    background = _texture(height, width)
    frames = np.repeat(background[None, ...], n_frames, axis=0).copy()
    masks = np.zeros((n_frames, height, width), dtype=np.uint8)
    frames[:3, 20:40, 30:50] = 0
    masks[:3, 20:40, 30:50] = 1
    plate, _ = build_plate(frames, masks=masks)
    inside, outside, to_object = _plate_errors(plate, background, (20, 40, 30, 50))
    assert to_background_is_clean(inside, outside, to_object), (
        f"masked region is {inside:.2f} from the background, the never-masked "
        f"rest of the plate is {outside:.2f}, and the object is {to_object:.2f} "
        "away. The object's pixels reached the plate, so a still player would be "
        "transmitted as background and then corrected by the residual twice."
    )


def test_a_one_frame_span_is_that_frame_unchanged() -> None:
    """What makes `span=1` a control rather than a second implementation."""
    frames = _pan(4, 96, 128, 3)
    plate, packed = build_plate(frames[:1])
    assert np.array_equal(plate, frames[0])
    assert len(packed) == 1


def test_zero_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="zero frames"):
        build_plate(np.zeros((0, 8, 8, 3), dtype=np.uint8))


def test_registration_can_be_switched_off_and_that_changes_the_plate() -> None:
    """The control knob must actually control something.

    `register=False` exists so a win can be attributed to camera-motion
    compensation rather than to the temporal median. A knob that quietly did
    nothing would make that attribution a fiction, which is the exact shape of
    bug this project keeps finding in config axes.
    """
    height, width, step, n_frames = 96, 128, 3, 6
    frames = _pan(n_frames, height, width, step)
    plate, packed = build_plate(frames, register=False)
    assert plate.shape[:2] == (height, width), (
        "unregistered frames are composited where they lie, so the canvas "
        "cannot grow"
    )
    assert all(
        np.allclose(np.asarray(row, dtype=np.float64).reshape(3, 3), np.eye(3))
        for row in packed
    )

    registered, registered_maps = build_plate(frames)
    last = n_frames - 1
    matrix = np.asarray(registered_maps[last], dtype=np.float64).reshape(3, 3)
    warped = cv2.warpPerspective(
        registered, np.linalg.inv(matrix), (width, height), flags=cv2.INTER_LINEAR
    )
    with_registration = float(
        np.mean(np.abs(warped.astype(np.int32) - frames[last].astype(np.int32)))
    )
    without = float(np.mean(np.abs(plate.astype(np.int32) - frames[last].astype(np.int32))))
    assert with_registration < without / 2.0, (
        f"registered plate is {with_registration:.2f} from the last frame and "
        f"the unregistered median is {without:.2f}. If these were close, the "
        "homographies would be decoration and any saving would belong to the "
        "median instead."
    )
