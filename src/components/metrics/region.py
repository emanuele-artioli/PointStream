"""A metric is computed over a region, not only over a frame.

A whole-frame score hides a broken object: a perfect background with a mangled
player still posts a respectable frame PSNR, because the player is a small
fraction of the pixels. Object generation is therefore scored on the object
mask or crop, background modelling on the complement, and the whole frame is
reported *as well*, never *instead*. Every score records which of those it is.

Masks and bounding boxes are not interchangeable. A box includes background
pixels around the subject; a mask does not. Scoring a generated player against
a box flatters it. Both are accepted, the record says which was used, and a
mask wins when both are supplied.

A tiny region makes PSNR jumpy. The pixel count travels with the score, and a
region smaller than ``MIN_REGION_PIXELS`` is refused rather than reported as a
result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

#: Spatial pixels per frame below this are a small-sample artefact, not a score.
#: An 8x8 patch sits on the floor; a 40x80 player crop (3200) is well above it.
MIN_REGION_PIXELS: int = 64

Box = tuple[int, int, int, int]


class RegionKind(str, Enum):
    """How the region was specified. Recorded on every score."""

    FRAME = "frame"
    MASK = "mask"
    BOX = "box"


class RegionRole(str, Enum):
    """What the region is *for*. Whole-frame is always present alongside the rest."""

    WHOLE_FRAME = "whole-frame"
    OBJECT = "object"
    BACKGROUND = "background"


@dataclass(frozen=True, eq=False)
class Region:
    """One scoring scope: a whole frame, an object, or the background.

    For ``object``, ``mask`` / ``box`` select the subject. For ``background``
    they name the subject to *exclude*. Passing both prefers the mask.
    """

    role: RegionRole
    kind: RegionKind
    mask: NDArray[np.bool_] | None = None
    box: Box | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role is RegionRole.WHOLE_FRAME:
            if self.kind is not RegionKind.FRAME or self.mask is not None or self.box is not None:
                raise ValueError("a whole-frame region has kind 'frame' and no mask or box")
            return
        if self.kind is RegionKind.FRAME:
            raise ValueError(f"{self.role.value} region must be a mask or a box, not a frame")
        if self.kind is RegionKind.MASK:
            if self.mask is None:
                raise ValueError("a mask region needs a mask array")
            object.__setattr__(self, "mask", _as_bool_mask(self.mask))
        elif self.kind is RegionKind.BOX:
            if self.box is None:
                raise ValueError("a box region needs a (x1, y1, x2, y2) box")
            object.__setattr__(self, "box", _as_box(self.box))

    @classmethod
    def whole_frame(cls) -> Region:
        """The entire clip. Always scored; never a substitute for a scoped score."""
        return cls(role=RegionRole.WHOLE_FRAME, kind=RegionKind.FRAME)

    @classmethod
    def object(
        cls,
        *,
        mask: np.ndarray | None = None,
        box: Sequence[int] | Box | None = None,
        name: str | None = None,
    ) -> Region:
        """Score the subject. A mask is preferred when both are given."""
        kind, stored_mask, stored_box = _choose_kind(mask, box)
        return cls(
            role=RegionRole.OBJECT,
            kind=kind,
            mask=stored_mask,
            box=stored_box,
            name=name,
        )

    @classmethod
    def background(
        cls,
        *,
        mask: np.ndarray | None = None,
        box: Sequence[int] | Box | None = None,
        name: str | None = None,
    ) -> Region:
        """Score everything except the subject named by ``mask`` or ``box``."""
        kind, stored_mask, stored_box = _choose_kind(mask, box)
        return cls(
            role=RegionRole.BACKGROUND,
            kind=kind,
            mask=stored_mask,
            box=stored_box,
            name=name,
        )

    def boolean_mask(self, frames: int, height: int, width: int) -> NDArray[np.bool_]:
        """``(T, H, W)`` True where the score should look, matching the clip."""
        if self.kind is RegionKind.FRAME:
            return np.ones((frames, height, width), dtype=bool)
        if self.kind is RegionKind.MASK:
            selected = _broadcast_mask(self._require_mask(), frames, height, width)
        else:
            selected = _box_mask(self._require_box(), frames, height, width)
        if self.role is RegionRole.BACKGROUND:
            return np.logical_not(selected)
        return selected

    def crop(self, reference: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slice both clips to this object's box. Masks and backgrounds are not crops."""
        if self.kind is not RegionKind.BOX or self.role is not RegionRole.OBJECT:
            raise ValueError(
                "crop() is only defined for an object bounding box; "
                f"got kind={self.kind.value} role={self.role.value}"
            )
        x1, y1, x2, y2 = self._require_box()
        _check_box_bounds(self._require_box(), height=reference.shape[1], width=reference.shape[2])
        return reference[:, y1:y2, x1:x2, :], predicted[:, y1:y2, x1:x2, :]

    def _require_mask(self) -> NDArray[np.bool_]:
        if self.mask is None:
            raise ValueError("mask region is missing its mask")
        return self.mask

    def _require_box(self) -> Box:
        if self.box is None:
            raise ValueError("box region is missing its box")
        return self.box


def reject_if_too_small(mask: np.ndarray, *, role: str, kind: str, name: str | None) -> int:
    """Refuse a region that would produce a jumpy score. Returns per-frame pixel count."""
    per_frame = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1).sum(axis=1)
    n_pixels = int(np.round(float(per_frame.mean())))
    smallest = int(per_frame.min())
    if smallest >= MIN_REGION_PIXELS:
        return n_pixels
    label = name or f"{role} {kind}"
    raise ValueError(
        f"region {label!r} has {smallest} pixels in at least one frame; "
        f"minimum is {MIN_REGION_PIXELS}. A score on this few pixels is a "
        "small-sample artefact, not a result."
    )


def _choose_kind(
    mask: np.ndarray | None,
    box: Sequence[int] | Box | None,
) -> tuple[RegionKind, NDArray[np.bool_] | None, Box | None]:
    if mask is None and box is None:
        raise ValueError("region needs a mask or a box")
    if mask is not None:
        return RegionKind.MASK, _as_bool_mask(mask), None
    return RegionKind.BOX, None, _as_box(box)


def _as_bool_mask(mask: np.ndarray) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim not in (2, 3):
        raise ValueError(f"mask must be (H, W) or (T, H, W), got shape {array.shape}")
    if array.size == 0:
        raise ValueError("mask is empty")
    boolean = np.array(array > 0, dtype=bool)
    boolean.flags.writeable = False
    return boolean


def _as_box(box: Sequence[int] | Box | None) -> Box:
    if box is None:
        raise ValueError("box is missing")
    values = tuple(int(v) for v in box)
    if len(values) != 4:
        raise ValueError(f"box must be (x1, y1, x2, y2), got {box!r}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have positive area, got {values}")
    return x1, y1, x2, y2


def _broadcast_mask(mask: np.ndarray, frames: int, height: int, width: int) -> NDArray[np.bool_]:
    if mask.ndim == 2:
        if mask.shape != (height, width):
            raise ValueError(
                f"mask shape {mask.shape} does not match frame {(height, width)}; "
                "resampling the mask to the frame is a bug upstream, not a metric's job"
            )
        return np.broadcast_to(mask, (frames, height, width))
    if mask.shape == (frames, height, width):
        return np.asarray(mask, dtype=bool)
    if mask.shape[0] == 1 and mask.shape[1:] == (height, width):
        return np.broadcast_to(mask[0], (frames, height, width))
    raise ValueError(
        f"mask shape {mask.shape} does not match clip {(frames, height, width)}; "
        "resampling the mask to the clip is a bug upstream, not a metric's job"
    )


def _box_mask(box: Box, frames: int, height: int, width: int) -> NDArray[np.bool_]:
    _check_box_bounds(box, height=height, width=width)
    x1, y1, x2, y2 = box
    selected = np.zeros((frames, height, width), dtype=bool)
    selected[:, y1:y2, x1:x2] = True
    return selected


def _check_box_bounds(box: Box, *, height: int, width: int) -> None:
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"box {box} lies outside the frame {(width, height)}; "
            "clipping a box to the frame hides an upstream coordinate bug"
        )
