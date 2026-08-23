"""Colour-palette similarity: the crude companion to the ReID embedding.

`BP18` asks for an identity instrument and gets one from a learned
re-identification embedding (`reid`). This sits beside it and is deliberately
stupid: a normalised colour histogram of the crop, compared by histogram
intersection, which is the fraction of colour mass the two images share.

**Why keep something this crude.** In a tennis broadcast, kit colour is the
dominant identity cue, and that means the ReID embedding is partly a
sophisticated colour detector. When the two agree, the ReID number is probably
reading colour. When they *disagree* — same palette, different embedding, or the
reverse — one of them is wrong and it is worth finding out which. A learned
metric with nothing to check it against is how this project shipped an LPIPS
that could not tell a match from an unrelated image.

It also fails in ways a human can see. A histogram has no training set, no
domain gap and no checkpoint, so "the palette says these are the same and the
embedding disagrees" is a sentence anyone can go and verify by looking.

Higher is better; 1.0 for identical colour distributions. Note what it cannot
do: two players in the same kit are identical to it, and so are a person and a
wall of the same colour. It is a companion, never a headline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.metrics.frames import to_clip

#: Bins per channel. 8 gives 512 buckets — coarse enough that a shading change
#: does not move mass between bins, fine enough to separate a pink shirt from a
#: maroon one.
BINS_PER_CHANNEL = 8


class PaletteMetric:
    """Histogram intersection over RGB. Higher is better; 1.0 if identical.

    Like `reid` and unlike every distortion metric here, the two sides need not
    share a spatial shape: a histogram is normalised by pixel count, so a crop
    and a rescaled version of it compare equal.
    """

    name = "palette"

    def __init__(self, *, bins: int = BINS_PER_CHANNEL, mask_black: bool = True) -> None:
        self._bins = int(bins)
        self._mask_black = mask_black

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref = to_clip(reference)
        pred = to_clip(predicted)
        if ref.shape[0] != pred.shape[0]:
            raise ValueError(
                "reference and predicted clips must have the same number of "
                f"frames; got {ref.shape[0]} vs {pred.shape[0]}. Spatial shapes "
                "may differ — a histogram is normalised by pixel count."
            )
        scores = [
            _intersection(
                _histogram(left, self._bins, self._mask_black),
                _histogram(right, self._bins, self._mask_black),
            )
            for left, right in zip(ref, pred)
        ]
        return float(np.mean(scores))


def _histogram(frame: np.ndarray, bins: int, mask_black: bool) -> np.ndarray:
    """Normalised RGB histogram of one ``(H, W, C)`` frame."""
    pixels = frame[..., :3].reshape(-1, 3).astype(np.int32)
    if mask_black:
        # Letterbox padding is exactly black and is not part of the subject.
        # Dropping it stops the pad area dominating a tall, narrow crop.
        keep = pixels.sum(axis=1) > 12
        if keep.any():
            pixels = pixels[keep]
    edges = (pixels * bins) // 256
    flat = edges[:, 0] * bins * bins + edges[:, 1] * bins + edges[:, 2]
    counts = np.bincount(flat, minlength=bins**3).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def _intersection(left: np.ndarray, right: np.ndarray) -> float:
    """Shared colour mass. 1.0 for identical distributions, 0.0 for disjoint."""
    return float(np.minimum(left, right).sum())


def build(**kwargs: Any) -> PaletteMetric:
    return PaletteMetric(**kwargs)
