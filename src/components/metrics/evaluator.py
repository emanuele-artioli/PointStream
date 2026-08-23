"""Build a metric set and score a reconstructed clip against its source.

Quality measurement is mandatory. An empty set is refused. Omitting PSNR is
not refused — it is added back and the result record says so, so a summary
can report the enforcement rather than hide it.

Scoring is region-scoped. Every call reports a whole-frame score labelled as
such; object and background regions ride along when supplied. The cheap path
is PSNR only (``triage`` / ``Evaluator.triage``), so a development caller does
not reach for VMAF, SSIM, LPIPS or FVMD by accident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.components.metrics.psnr import masked_psnr
from src.components.metrics.region import (
    Region,
    RegionKind,
    RegionRole,
    reject_if_too_small,
)
from src.components.metrics.ssim import masked_ssim
from src.contracts.errors import ConfigValueError
from src.contracts.metrics import ALWAYS_ON, DEFAULT_METRICS, MetricSelection, resolve

#: Metrics that need a rectangle. A mask or a background (a frame with a hole)
#: is not a rectangle; those scopes use PSNR (and SSIM on a mask).
#: ``reid`` is here because it embeds a crop of a whole person — a masked frame
#: with a person-shaped hole in it is not something a ReID backbone has ever
#: seen, and it would return a number anyway.
_RECTANGULAR = frozenset({"vmaf", "lpips", "fvmd", "reid", "palette"})


class MetricBackend(Protocol):
    """One compute backend. Registry names match ``src.contracts.metrics``."""

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ScopedScore:
    """One metric on one region. A score whose scope is unstated is not usable."""

    metric: str
    value: float
    role: str
    kind: str
    n_pixels: int
    name: str | None = None


@dataclass(frozen=True)
class EvaluationRecord:
    """One clip comparison. ``scores`` is the whole-frame map, for existing callers.

    ``scoped`` is the full record: whole-frame plus every region, each labelled.
    ``enforced`` is why PSNR is present if it was not asked.
    """

    scores: Mapping[str, float]
    enforced: tuple[str, ...]
    selection: MetricSelection
    n_frames: int
    scoped: tuple[ScopedScore, ...] = ()

    def describe(self) -> str:
        return self.selection.describe()

    def for_role(self, role: str, *, name: str | None = None) -> tuple[ScopedScore, ...]:
        """Scores computed over one role, optionally one named subject."""
        return tuple(
            item
            for item in self.scoped
            if item.role == role and (name is None or item.name == name)
        )


class Evaluator:
    """Scores a reference/predicted clip pair on a resolved metric set."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        backends: Mapping[str, MetricBackend] | None = None,
        registry: Any = None,
    ) -> None:
        requested = [str(name).strip().lower() for name in names if str(name).strip()]
        if not requested:
            raise ConfigValueError(
                "evaluation.metrics",
                "an empty metric set measures nothing. Quality measurement is "
                "mandatory in every configuration.",
            )
        self._selection = resolve(requested)
        table = registry
        if table is None:
            from src.components.metrics import REGISTRY as table
        self._backends = self._bind(self._selection, backends, table)

    @classmethod
    def triage(
        cls,
        *,
        backends: Mapping[str, MetricBackend] | None = None,
        registry: Any = None,
    ) -> Evaluator:
        """PSNR only. The development default; expensive metrics stay off."""
        return cls(DEFAULT_METRICS, backends=backends, registry=registry)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        backends: Mapping[str, MetricBackend] | None = None,
        registry: Any = None,
    ) -> Evaluator:
        return cls(config.evaluation.metrics, backends=backends, registry=registry)

    @property
    def selection(self) -> MetricSelection:
        return self._selection

    def evaluate(
        self,
        reference: np.ndarray,
        predicted: np.ndarray,
        *,
        regions: Sequence[Region] | None = None,
    ) -> EvaluationRecord:
        from src.components.metrics.frames import paired

        ref, pred = paired(reference, predicted)
        scopes = _scopes(regions)
        scoped: list[ScopedScore] = []
        for region in scopes:
            for spec in self._selection:
                backend = self._backends[spec.name]
                value, n_pixels, kind = _score_over_region(
                    backend, spec.name, ref, pred, region
                )
                scoped.append(
                    ScopedScore(
                        metric=spec.name,
                        value=value,
                        role=region.role.value,
                        kind=kind,
                        n_pixels=n_pixels,
                        name=region.name,
                    )
                )
        whole = {
            item.metric: item.value
            for item in scoped
            if item.role == RegionRole.WHOLE_FRAME.value
        }
        return EvaluationRecord(
            scores=whole,
            enforced=self._selection.enforced,
            selection=self._selection,
            n_frames=int(ref.shape[0]),
            scoped=tuple(scoped),
        )

    def _bind(
        self,
        selection: MetricSelection,
        backends: Mapping[str, MetricBackend] | None,
        registry: Any,
    ) -> dict[str, MetricBackend]:
        bound: dict[str, MetricBackend] = {}
        supplied = dict(backends or {})
        for spec in selection:
            if spec.name in supplied:
                bound[spec.name] = supplied[spec.name]
            else:
                bound[spec.name] = registry.build(spec.name)
        missing = sorted(ALWAYS_ON - set(bound))
        if missing:
            raise RuntimeError(f"Evaluator omitted mandatory metric(s): {', '.join(missing)}")
        return bound


def triage(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    regions: Sequence[Region] | None = None,
) -> EvaluationRecord:
    """PSNR-only, region-scoped. The path to call during development."""
    return Evaluator.triage().evaluate(reference, predicted, regions=regions)


def _scopes(regions: Sequence[Region] | None) -> tuple[Region, ...]:
    extra = tuple(
        region for region in (regions or ()) if region.role is not RegionRole.WHOLE_FRAME
    )
    return (Region.whole_frame(), *extra)


def _score_over_region(
    backend: MetricBackend,
    name: str,
    reference: np.ndarray,
    predicted: np.ndarray,
    region: Region,
) -> tuple[float, int, str]:
    frames, height, width, _channels = reference.shape
    if region.role is RegionRole.WHOLE_FRAME:
        return float(backend.score(reference, predicted)), height * width, RegionKind.FRAME.value

    mask = region.boolean_mask(frames, height, width)
    n_pixels = reject_if_too_small(
        mask, role=region.role.value, kind=region.kind.value, name=region.name
    )
    if name == "psnr":
        return masked_psnr(reference, predicted, mask), n_pixels, region.kind.value
    if name == "ssim":
        if region.kind is RegionKind.BOX and region.role is RegionRole.OBJECT:
            cropped_ref, cropped_pred = region.crop(reference, predicted)
            return (
                float(backend.score(cropped_ref, cropped_pred)),
                n_pixels,
                region.kind.value,
            )
        return masked_ssim(reference, predicted, mask), n_pixels, region.kind.value
    if name in _RECTANGULAR:
        if region.kind is RegionKind.BOX and region.role is RegionRole.OBJECT:
            cropped_ref, cropped_pred = region.crop(reference, predicted)
            return (
                float(backend.score(cropped_ref, cropped_pred)),
                n_pixels,
                region.kind.value,
            )
        raise ValueError(
            f"{name} cannot score a {region.kind.value} {region.role.value} region. "
            "Mask and background scopes need a pixel-wise metric; use PSNR "
            "(Evaluator.triage / triage()) rather than VMAF, LPIPS or FVMD."
        )
    raise ValueError(f"no region-scoring path for metric {name!r}")
