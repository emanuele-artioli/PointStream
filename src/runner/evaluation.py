"""Bind the components-layer metric set to the pipeline's `QualityEvaluator`.

`src/pipeline` deliberately imports nothing from `src/components`: its always-on
floor is a numpy PSNR so that a reconstruction can be scored on a machine where
no metric backend's dependencies are installed. The runner is the layer allowed
to look a registry up, so this is where `evaluation.metrics` stops being a
field nobody reads and starts producing SSIM and VMAF numbers.

Two shapes have to meet:

* the pipeline speaks in mask arrays (`object_mask`, `background_mask`) and
  returns a `QualityReport`;
* the components evaluator speaks in `Region` objects and returns an
  `EvaluationRecord`.

The adapter converts in both directions and keeps the roles labelled the whole
way, because a score whose scope is unstated is not usable. It does not invent a
`Closeness` from the metric set — closeness is a pixel comparison and is
computed directly, so bit-identity still means bit-identity rather than
"PSNR came back infinite".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.contracts.config import PointstreamConfig
from src.pipeline.reconstruction.quality import (
    ROLE_BACKGROUND,
    ROLE_FRAME,
    ROLE_OBJECT,
    QualityEvaluator,
    QualityReport,
    RegionScore,
    closeness,
)
from src.pipeline.reconstruction.clips import as_clip, require_same_shape

#: Roles the components layer labels, mapped to the pipeline's spelling. They
#: agree today; the map exists so a drift on either side is a failed lookup
#: rather than a silently unlabelled score.
_ROLES = {
    "whole-frame": ROLE_FRAME,
    "object": ROLE_OBJECT,
    "background": ROLE_BACKGROUND,
}


class ComponentMetricEvaluator:
    """`QualityEvaluator` backed by `src.components.metrics.Evaluator`.

    Args:
        metrics: The requested metric names. PSNR is always added back by the
            components evaluator if it is missing, and the report says so.
    """

    def __init__(self, metrics: Sequence[str]) -> None:
        from src.components.metrics import Evaluator
        from src.components.metrics.evaluator import _RECTANGULAR

        requested = [str(name).strip().lower() for name in metrics if str(name).strip()]
        self._evaluator = Evaluator(requested)
        # VMAF, LPIPS, FVMD, ReID and palette need a rectangle: a frame with a
        # person-shaped hole in it is not something any of them has been trained
        # or specified on, and the components evaluator refuses it rather than
        # returning a number anyway. Scoring a mask region with a *box* instead
        # would flatter a generated player, which is the substitution
        # `region.py` exists to prevent. So the full set scores the whole frame
        # and only the pixel-wise metrics are carried into the masked regions.
        self._pixelwise = tuple(
            name for name in self._evaluator.selection.names() if name not in _RECTANGULAR
        )
        self._skipped_on_regions = tuple(
            name for name in self._evaluator.selection.names() if name in _RECTANGULAR
        )
        self._region_evaluator = (
            Evaluator(list(self._pixelwise)) if self._pixelwise else None
        )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._evaluator.selection)

    @property
    def region_metric_names(self) -> tuple[str, ...]:
        """The subset that a mask or background region actually receives."""
        return tuple(self._pixelwise)

    @property
    def skipped_on_regions(self) -> tuple[str, ...]:
        """Requested metrics that only ever score the whole frame, and why.

        Reported rather than dropped silently: a metric that is absent from a
        region score because it cannot be computed there is a different fact
        from one that was never asked for.
        """
        return tuple(self._skipped_on_regions)

    @property
    def enforced(self) -> tuple[str, ...]:
        """Metrics added back because measurement is mandatory."""
        return tuple(self._evaluator.selection.enforced)

    def evaluate(
        self,
        reference: np.ndarray,
        predicted: np.ndarray,
        *,
        object_mask: np.ndarray | None = None,
        background_mask: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> QualityReport:
        from src.components.metrics import Region

        ref = as_clip(reference, path="reference")
        pred = as_clip(predicted, path="predicted")
        require_same_shape(ref, pred, path="quality")

        # The two layers spell "background" differently and the difference is
        # a negation, so it has to be written down rather than assumed:
        # `Region.background(mask=m)` scores *everything except* `m`, while the
        # pipeline protocol's `background_mask` names the pixels to score. A
        # caller-supplied background mask is therefore inverted on the way in;
        # the derived complement is not, because the object mask is already the
        # thing to exclude.
        regions: list[Region] = []
        if object_mask is not None:
            regions.append(Region.object(mask=_frame_mask(object_mask), name=object_name))
        if background_mask is not None:
            regions.append(
                Region.background(
                    mask=np.logical_not(_frame_mask(background_mask)), name=object_name
                )
            )
        elif object_mask is not None:
            regions.append(
                Region.background(mask=_frame_mask(object_mask), name=object_name)
            )

        frame_record = self._evaluator.evaluate(ref, pred, regions=None)
        rows = list(frame_record.scoped)
        if regions and self._region_evaluator is not None:
            region_record = self._region_evaluator.evaluate(ref, pred, regions=regions)
            rows.extend(
                item
                for item in region_record.scoped
                if _ROLES[item.role] != ROLE_FRAME
            )
        scoped = tuple(
            RegionScore(
                metric=item.metric,
                value=float(item.value),
                role=_ROLES[item.role],
                n_pixels=int(item.n_pixels),
                name=item.name,
            )
            for item in rows
        )
        return QualityReport(
            closeness=closeness(ref, pred),
            scoped=scoped,
            enforced=tuple(frame_record.enforced),
        )


def evaluator_for(config: PointstreamConfig) -> QualityEvaluator:
    """The scorer a config asks for — the same one for every config.

    A tempting shortcut here is to hand a PSNR-only config the pipeline's numpy
    floor and skip the registry. Measured on one 4K clip, that shortcut makes a
    tier ladder incomparable with itself: the floor's whole-frame PSNR is the
    pooled-MSE convention while the components metric is the mean of per-frame
    PSNRs, and on the same delivered pixels the two read 47.63 dB and 48.28 dB.
    A ladder whose PSNR convention changes between its rungs measures the
    evaluator, not the tier.

    So every config gets the same evaluator. `closeness` on the returned report
    still carries the pooled-MSE number, labelled as its own thing, because
    bit-identity has to be a pixel comparison rather than an inference from a
    metric that came back infinite.
    """
    names = tuple(config.evaluation.metrics) or ("psnr",)
    return ComponentMetricEvaluator(names)


def _frame_mask(mask: np.ndarray) -> np.ndarray:
    """A per-frame boolean mask, as `Region` wants it.

    `Region.boolean_mask` broadcasts a 2-D mask itself, so a 3-D `(T, H, W)`
    mask is passed straight through and a 2-D one is left 2-D. Reshaping here
    would hide a coordinate bug rather than surface it.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim not in (2, 3):
        raise ValueError(f"mask must be (H, W) or (T, H, W); got {array.shape}")
    return array


__all__ = ["ComponentMetricEvaluator", "evaluator_for"]
