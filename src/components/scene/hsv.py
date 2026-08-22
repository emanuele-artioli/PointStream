"""HSV-histogram scene classifier. Intentionally small.

Splits a clip on pairwise HSV-histogram disagreement and labels each span
point (stable background) or interlude (large appearance change). Routing is
delegated to :mod:`src.components.scene.routing`. This is not the GMM/ffmpeg
pipeline in ``src.shared.scene_classification``; that is prior art, and this
row does not need more investment than the routing question it answers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.components.scene.routing import (
    INTERLUDE,
    POINT,
    SceneSpan,
    route_for,
    span as make_span,
)

#: Correlation below this between adjacent frames is a cut.
CUT_CORRELATION = 0.85
#: Mean adjacent-frame correlation below this inside a span → interlude.
POINT_MEAN_CORRELATION = 0.97


class HsvHistogramClassifier:
    """Classify a sequence of BGR frames into point / interlude spans."""

    def __init__(
        self,
        cut_correlation: float = CUT_CORRELATION,
        point_mean_correlation: float = POINT_MEAN_CORRELATION,
    ) -> None:
        self.cut_correlation = cut_correlation
        self.point_mean_correlation = point_mean_correlation

    def classify(self, frames: Sequence[np.ndarray]) -> list[SceneSpan]:
        if not frames:
            return []
        if len(frames) == 1:
            return [make_span(0, 1, POINT)]

        correlations = [
            hsv_correlation(frames[index], frames[index + 1])
            for index in range(len(frames) - 1)
        ]
        cuts = [0]
        for index, correlation in enumerate(correlations):
            if correlation < self.cut_correlation:
                cuts.append(index + 1)
        cuts.append(len(frames))

        spans: list[SceneSpan] = []
        for start, end in zip(cuts, cuts[1:]):
            if start >= end:
                continue
            window = correlations[start : max(start, end - 1)]
            mean = float(np.mean(window)) if window else 1.0
            label = POINT if mean >= self.point_mean_correlation else INTERLUDE
            spans.append(
                SceneSpan(
                    start_frame=start,
                    end_frame=end,
                    scene_class=label,
                    route=route_for(label),
                )
            )
        return spans


def hsv_correlation(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """HSV 2-D histogram correlation (OpenCV ``HISTCMP_CORREL``, typically in [-1, 1])."""
    import cv2

    def histogram(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    return float(cv2.compareHist(histogram(frame_a), histogram(frame_b), cv2.HISTCMP_CORREL))
