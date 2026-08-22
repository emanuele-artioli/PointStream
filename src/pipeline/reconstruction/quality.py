"""Quality on every reconstruction path: bit-identity, closeness, regions.

Deterministic stages are checked for bit-identity. Generative stages are
measured for closeness — encoder-side and client-side generation are not
guaranteed to match, and asserting that they do would fail for reasons that
have nothing to do with correctness.

A whole-frame score hides a broken object. Object reconstruction is scored on
the object region, background on the background region, and the whole frame is
reported as well, never instead.

This module does not import ``src.components.metrics``. The always-on floor is
numpy PSNR; a richer evaluator is injected. Region types here are structural
so the pipeline never depends on the components-layer ``Region`` class — C3
adapts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.pipeline.reconstruction.clips import as_clip, require_same_shape

#: Spatial pixels per frame below this are a small-sample artefact, not a score.
#: Mirrors the components-layer floor; belongs in contracts long-term.
MIN_REGION_PIXELS = 64

_PEAK = 255.0

ROLE_FRAME = "whole-frame"
ROLE_OBJECT = "object"
ROLE_BACKGROUND = "background"


@dataclass(frozen=True)
class Closeness:
    """How two reconstructions relate, without asserting they match.

    ``bit_identical`` is the deterministic-stage check. The rest is what a
    generative comparison actually has: a distance, and whether it sits inside
    a stated grey-level tolerance.
    """

    bit_identical: bool
    mean_abs_diff: float
    max_abs_diff: float
    psnr: float
    within_atol: bool
    atol: float = 1.0


@dataclass(frozen=True)
class RegionScore:
    """One metric on one labelled region. A score whose role is unstated is unusable."""

    metric: str
    value: float
    role: str
    n_pixels: int
    name: str | None = None


@dataclass(frozen=True)
class QualityReport:
    """Every path's quality record: whole-frame plus any scoped scores, plus closeness."""

    closeness: Closeness
    scoped: tuple[RegionScore, ...]
    enforced: tuple[str, ...] = ("psnr",)

    @property
    def bit_identical(self) -> bool:
        return self.closeness.bit_identical

    def whole_frame(self, metric: str = "psnr") -> float:
        """The frame-level score. Always present; never a substitute for a scoped one."""
        for item in self.scoped:
            if item.role == ROLE_FRAME and item.metric == metric:
                return item.value
        raise KeyError(f"no whole-frame {metric!r} score in this report")

    def for_role(self, role: str, *, name: str | None = None) -> tuple[RegionScore, ...]:
        return tuple(
            item
            for item in self.scoped
            if item.role == role and (name is None or item.name == name)
        )


class QualityEvaluator(Protocol):
    """Injected scorer. C3 binds the components-layer evaluator to this."""

    def evaluate(
        self,
        reference: np.ndarray,
        predicted: np.ndarray,
        *,
        object_mask: np.ndarray | None = None,
        background_mask: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> QualityReport:
        """Score ``predicted`` against ``reference``. Whole-frame always; regions when given."""
        ...


def bit_identical(reference: np.ndarray, predicted: np.ndarray) -> bool:
    """True when every sample matches exactly."""
    ref = as_clip(reference, path="reference")
    pred = as_clip(predicted, path="predicted")
    require_same_shape(ref, pred, path="bit-identity")
    return bool(np.array_equal(ref, pred))


def closeness(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    atol: float = 1.0,
) -> Closeness:
    """Measure how close two clips are. Does not assert they match."""
    ref = as_clip(reference, path="reference").astype(np.float64)
    pred = as_clip(predicted, path="predicted").astype(np.float64)
    require_same_shape(ref, pred, path="closeness")
    delta = np.abs(ref - pred)
    mean_abs = float(delta.mean())
    max_abs = float(delta.max())
    identical = bool(max_abs == 0.0)
    return Closeness(
        bit_identical=identical,
        mean_abs_diff=mean_abs,
        max_abs_diff=max_abs,
        psnr=_psnr(ref, pred),
        within_atol=bool(max_abs <= atol),
        atol=atol,
    )


def measure_symmetry(
    encoder_side: np.ndarray,
    client_side: np.ndarray,
    *,
    atol: float = 1.0,
) -> Closeness:
    """Encoder vs client reconstruction. A design goal, verified by measurement.

    Generation is statistical: do not treat a mismatch as a failed invariant.
    """
    return closeness(encoder_side, client_side, atol=atol)


def score(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    object_mask: np.ndarray | None = None,
    background_mask: np.ndarray | None = None,
    object_name: str | None = None,
    atol: float = 1.0,
) -> QualityReport:
    """Always-on numpy PSNR, region-scoped. The floor every path reports.

    Bounds before reading: identical clips → PSNR infinite, mean abs 0.
    One grey level on one pixel of an 8×8×3 frame → PSNR roughly 48–70 dB,
    never below ~20 unless the clip is badly wrong.
    """
    ref = as_clip(reference, path="reference")
    pred = as_clip(predicted, path="predicted")
    require_same_shape(ref, pred, path="quality")
    frames, height, width, _ = ref.shape
    relate = closeness(ref, pred, atol=atol)

    scoped: list[RegionScore] = [
        RegionScore(
            metric="psnr",
            value=relate.psnr,
            role=ROLE_FRAME,
            n_pixels=height * width,
        )
    ]
    if object_mask is not None:
        scoped.append(
            _masked_psnr(
                ref,
                pred,
                object_mask,
                role=ROLE_OBJECT,
                name=object_name,
                frames=frames,
                height=height,
                width=width,
            )
        )
    if background_mask is not None:
        scoped.append(
            _masked_psnr(
                ref,
                pred,
                background_mask,
                role=ROLE_BACKGROUND,
                name=object_name,
                frames=frames,
                height=height,
                width=width,
            )
        )
    elif object_mask is not None:
        complement = np.logical_not(_align_mask(object_mask, frames, height, width))
        scoped.append(
            _masked_psnr(
                ref,
                pred,
                complement,
                role=ROLE_BACKGROUND,
                name=object_name,
                frames=frames,
                height=height,
                width=width,
            )
        )
    return QualityReport(closeness=relate, scoped=tuple(scoped))


class NumpyPsnrEvaluator:
    """Default injected evaluator: the quality floor, no components import."""

    def evaluate(
        self,
        reference: np.ndarray,
        predicted: np.ndarray,
        *,
        object_mask: np.ndarray | None = None,
        background_mask: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> QualityReport:
        return score(
            reference,
            predicted,
            object_mask=object_mask,
            background_mask=background_mask,
            object_name=object_name,
        )


def _psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    mse = float(np.mean((reference - predicted) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((_PEAK**2) / mse)


def _masked_psnr(
    reference: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    *,
    role: str,
    name: str | None,
    frames: int,
    height: int,
    width: int,
) -> RegionScore:
    selected = _align_mask(mask, frames, height, width)
    n_pixels = _reject_if_too_small(selected, role=role, name=name)
    values = []
    for index in range(frames):
        pixels_ref = reference[index][selected[index]]
        pixels_pred = predicted[index][selected[index]]
        mse = float(np.mean((pixels_ref.astype(np.float64) - pixels_pred.astype(np.float64)) ** 2))
        values.append(math.inf if mse == 0.0 else 10.0 * math.log10((_PEAK**2) / mse))
    finite = [value for value in values if math.isfinite(value)]
    value = math.inf if not finite else float(sum(finite) / len(finite))
    return RegionScore(metric="psnr", value=value, role=role, n_pixels=n_pixels, name=name)


def _align_mask(mask: np.ndarray, frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if array.shape != (height, width):
            raise ValueError(
                f"mask shape {array.shape} does not match frame {(height, width)}; "
                "resampling the mask here would hide an upstream coordinate bug."
            )
        return np.broadcast_to(array, (frames, height, width))
    if array.shape != (frames, height, width):
        raise ValueError(
            f"mask shape {array.shape} does not match clip {(frames, height, width)}."
        )
    return array


def _reject_if_too_small(mask: np.ndarray, *, role: str, name: str | None) -> int:
    per_frame = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1).sum(axis=1)
    n_pixels = int(np.round(float(per_frame.mean())))
    smallest = int(per_frame.min())
    if smallest >= MIN_REGION_PIXELS:
        return n_pixels
    label = name or role
    raise ValueError(
        f"region {label!r} has {smallest} pixels in at least one frame; "
        f"minimum is {MIN_REGION_PIXELS}. A score on this few pixels is a "
        "small-sample artefact, not a result."
    )


def union_object_mask(
    masks: Sequence[np.ndarray],
    *,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    """OR of object masks, or None when nothing was supplied."""
    if not masks:
        return None
    combined = np.zeros((frames, height, width), dtype=bool)
    for mask in masks:
        combined |= _align_mask(mask, frames, height, width)
    return combined
