"""The quality metrics a run can compute, tiered by what they cost.

**Quality measurement is mandatory in every configuration.** That is an
architectural requirement rather than an evaluation preference, and it follows
from two properties of this system that no amount of care removes: every
practical residual quantizes, so some coarseness is always carried; and
generative inference is statistical, so encoder-side and client-side generation
are not *guaranteed* to produce identical pixels even from identical inputs with
identical seeds. Symmetry between the two sides is therefore a design goal
verified by measurement, not a guarantee asserted by construction — and a run
that measured nothing has verified nothing.

So there is a floor: PSNR always runs. `resolve` enforces it, and rejects a
configuration that asks for no metrics at all. The arrangement being replaced
accepted `metrics: none` and returned an empty list
(`experiment_evaluation.py:463-482`), which is how a run could complete, be
reported, and carry no quality number whatsoever.

Tiers exist so the floor stays cheap. PSNR on every development run costs
nothing; VMAF and LPIPS belong on the runs that produce headline tables, and
FVMD only where a temporal-coherence claim is being made.

Every metric declares its **direction**, because ranking code that has to
remember LPIPS is lower-better is ranking code that will one day forget.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.contracts.errors import ConfigValueError, UnknownBackendError

#: Axis name used in error messages, so an unknown metric reads the same way as
#: an unknown detector or codec.
AXIS: Final = "metric"


class MetricTier(str, Enum):
    """How expensive a metric is, and therefore when it is worth running."""

    FAST = "fast"
    """Cheap enough to run unconditionally. The development default."""

    TRADITIONAL = "traditional"
    """Standard video-quality scoring, for headline tables."""

    PERCEPTUAL = "perceptual"
    """For judging generated content, where a pixel-difference metric rewards a
    blurry frame over a plausible one.

    Mostly learned similarity, but not by definition: `palette` sits here
    because it is only meaningful beside `reid`, and selecting one without the
    other loses the cross-check that is the reason both exist. The tier is
    about what a metric is *for*, not how it is implemented."""

    TEMPORAL = "temporal"
    """Operates on a sequence rather than on frames, for coherence claims."""


class Direction(str, Enum):
    """Which way is better. Declared so ranking never special-cases a name."""

    HIGHER_IS_BETTER = "higher-is-better"
    LOWER_IS_BETTER = "lower-is-better"


class MetricCost(int, Enum):
    """Rough cost, ordered. Enough to schedule with, not a benchmark."""

    TRIVIAL = 1
    """Milliseconds per frame; arithmetic on decoded pixels."""

    MODERATE = 2
    """Seconds per clip; an ffmpeg filter pass or a small network."""

    HEAVY = 3
    """Minutes per clip, and a model on the GPU."""


@dataclass(frozen=True)
class MetricSpec:
    """One metric, described well enough to schedule, run and rank by.

    Args:
        name: Config key. Matched exactly.
        tier: Which tier this belongs to.
        direction: Whether a higher or lower score is better.
        cost: Rough expense, for deciding what a run can afford.
        needs_reference: Whether it compares against the source. A no-reference
            metric can score a decode when the original is not to hand.
        is_temporal: Whether it consumes a sequence. A temporal metric handed
            single frames measures nothing meaningful, so callers have to know.
        unit: What the number is in, where that is not obvious.
        range: Reportable bounds as ``(low, high)``, or None where unbounded.
            Present so a result outside them is recognisable as an alarm rather
            than a finding.
        summary: One line, for listings.
    """

    name: str
    tier: MetricTier
    direction: Direction
    cost: MetricCost
    needs_reference: bool = True
    is_temporal: bool = False
    unit: str = ""
    range: tuple[float, float] | None = None
    summary: str = ""

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger score means a better reconstruction."""
        return self.direction is Direction.HIGHER_IS_BETTER

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """Whether `candidate` beats `incumbent` under this metric's direction.

        The reason `direction` is a field. Comparison code calls this and stays
        correct when a lower-is-better metric joins the table.
        """
        if self.higher_is_better:
            return candidate > incumbent
        return candidate < incumbent

    def best(self, values: Iterable[float]) -> float:
        """The best of these scores, by this metric's direction."""
        ordered = list(values)
        if not ordered:
            raise ValueError(f"No values to rank for metric {self.name!r}.")
        return max(ordered) if self.higher_is_better else min(ordered)

    def in_range(self, value: float) -> bool:
        """Whether a score falls within this metric's reportable bounds.

        An out-of-range score is an implementation or evaluation bug, not a
        result — a VMAF of 118 means the comparison was misconfigured, and it
        should stop a report rather than enter one.
        """
        if self.range is None:
            return True
        low, high = self.range
        return low <= value <= high


# --------------------------------------------------------------------------
# The registered metrics
# --------------------------------------------------------------------------

PSNR = MetricSpec(
    name="psnr",
    tier=MetricTier.FAST,
    direction=Direction.HIGHER_IS_BETTER,
    cost=MetricCost.TRIVIAL,
    unit="dB",
    range=(0.0, 100.0),
    summary="Always on. The floor that makes every run report something.",
)

SSIM = MetricSpec(
    name="ssim",
    tier=MetricTier.TRADITIONAL,
    direction=Direction.HIGHER_IS_BETTER,
    cost=MetricCost.MODERATE,
    range=(0.0, 1.0),
    summary="Structural similarity.",
)

VMAF = MetricSpec(
    name="vmaf",
    tier=MetricTier.TRADITIONAL,
    direction=Direction.HIGHER_IS_BETTER,
    cost=MetricCost.MODERATE,
    range=(0.0, 100.0),
    summary="The headline video-quality number, and what the codec ladder is compared in.",
)

LPIPS = MetricSpec(
    name="lpips",
    tier=MetricTier.PERCEPTUAL,
    direction=Direction.LOWER_IS_BETTER,
    cost=MetricCost.HEAVY,
    range=(0.0, 1.0),
    summary=(
        "Learned perceptual distance, for generated content where PSNR misleads. "
        "Present in the codebase already, but wired only into checkpoint "
        "evaluation — never into pipeline evaluation."
    ),
)

REID = MetricSpec(
    name="reid",
    tier=MetricTier.PERCEPTUAL,
    direction=Direction.HIGHER_IS_BETTER,
    cost=MetricCost.MODERATE,
    range=(-1.0, 1.0),
    summary=(
        "Person re-identification similarity: did the right body appear, rather "
        "than merely a different one. The only metric here that is not a "
        "distance to the target frame, and the reason it exists is that a "
        "pasted keyframe wins every distance-based test of appearance use. "
        "Read it BESIDE a distortion metric, never instead of one. Cosine "
        "similarity has no natural zero on person crops: measured on this "
        "dataset two different people in the same match score 0.510 and two "
        "unrelated clips 0.37-0.42 depending on which pairs are sampled, "
        "against 1.000 for an identical crop. Quote the floor with the number."
    ),
)

PALETTE = MetricSpec(
    name="palette",
    tier=MetricTier.PERCEPTUAL,
    direction=Direction.HIGHER_IS_BETTER,
    cost=MetricCost.TRIVIAL,
    range=(0.0, 1.0),
    summary=(
        "Colour-palette overlap of the subject crop, as a check on the learned "
        "identity metric rather than a result of its own. Kit colour is most of "
        "what tells two players apart here, so `reid` is partly a colour "
        "detector; when these two disagree one of them is wrong and it is worth "
        "finding out which. Cannot separate two people in the same kit."
    ),
)

FVMD = MetricSpec(
    name="fvmd",
    tier=MetricTier.TEMPORAL,
    direction=Direction.LOWER_IS_BETTER,
    cost=MetricCost.HEAVY,
    is_temporal=True,
    summary=(
        "Fréchet Video Motion Distance. Chosen over FVD because the reviewer "
        "question is about temporal coherence specifically, which is what this "
        "measures; the existing FVD wiring is prior art to read, not to keep."
    ),
)

#: Every metric, by config name.
METRICS: Final[Mapping[str, MetricSpec]] = {
    spec.name: spec for spec in (PSNR, SSIM, VMAF, LPIPS, REID, PALETTE, FVMD)
}

#: Metrics no configuration may switch off. One entry today, and the reason
#: `resolve` refuses an empty selection.
ALWAYS_ON: Final[frozenset[str]] = frozenset({PSNR.name})

#: The development default: the floor and nothing else.
DEFAULT_METRICS: Final[tuple[str, ...]] = (PSNR.name,)


def metric(name: str) -> MetricSpec:
    """Look up a metric by config name.

    Raises:
        UnknownBackendError: With every registered metric and a close-match
            suggestion, so `psnr_y` reads as a typo rather than a mystery.
    """
    try:
        return METRICS[name]
    except KeyError:
        raise UnknownBackendError(AXIS, name, sorted(METRICS)) from None


def by_tier(tier: MetricTier) -> tuple[MetricSpec, ...]:
    """Every registered metric in one tier, by name."""
    return tuple(spec for spec in METRICS.values() if spec.tier is tier)


# --------------------------------------------------------------------------
# Resolving a requested set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSelection:
    """The metrics one run will compute, after the always-on rule is applied.

    Args:
        metrics: The resolved specs, in a stable order — cheapest tier first,
            so a run that dies partway through still has its PSNR.
        enforced: Names added because policy requires them rather than because
            the config asked. Reported rather than applied silently: a run
            summary should be able to say why PSNR is there.
    """

    metrics: tuple[MetricSpec, ...]
    enforced: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = sorted(ALWAYS_ON - {spec.name for spec in self.metrics})
        if missing:
            raise ValueError(
                f"MetricSelection omits mandatory metric(s): {', '.join(missing)}. "
                f"Every configuration measures quality; build selections with resolve()."
            )

    def names(self) -> tuple[str, ...]:
        """Metric names, in run order."""
        return tuple(spec.name for spec in self.metrics)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names()

    def __iter__(self) -> Iterator[MetricSpec]:
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    @property
    def temporal(self) -> tuple[MetricSpec, ...]:
        """Metrics that need a sequence rather than paired frames."""
        return tuple(spec for spec in self.metrics if spec.is_temporal)

    @property
    def per_frame(self) -> tuple[MetricSpec, ...]:
        """Metrics that score frame against frame."""
        return tuple(spec for spec in self.metrics if not spec.is_temporal)

    @property
    def reference_free(self) -> tuple[MetricSpec, ...]:
        """Metrics that can score a decode without the source to hand."""
        return tuple(spec for spec in self.metrics if not spec.needs_reference)

    @property
    def max_cost(self) -> MetricCost:
        """The most expensive metric in the selection."""
        return max((spec.cost for spec in self.metrics), default=MetricCost.TRIVIAL)

    def describe(self) -> str:
        """One line, for the run summary."""
        rendered = ", ".join(
            f"{spec.name}*" if spec.name in self.enforced else spec.name
            for spec in self.metrics
        )
        note = "  (* required in every configuration)" if self.enforced else ""
        return f"metrics: {rendered}{note}"


def _requested_names(requested: Iterable[str] | str | None, path: str) -> list[str]:
    """Normalise whatever the config carried into a list of names."""
    if requested is None:
        return list(DEFAULT_METRICS)
    if isinstance(requested, str):
        items = [part.strip().lower() for part in requested.split(",")]
    else:
        items = [str(part).strip().lower() for part in requested]
    items = [item for item in items if item]

    if any(item == "none" for item in items):
        raise ConfigValueError(
            path,
            "'none' is not a legal metric set. Quality measurement is mandatory "
            "in every configuration, because the residual always quantizes and "
            f"generative inference is statistical. Minimum: {', '.join(sorted(ALWAYS_ON))}.",
        )
    return items


def resolve(
    requested: Iterable[str] | str | None = None,
    *,
    path: str = "evaluation.metrics",
) -> MetricSelection:
    """Turn a requested metric set into the set that will actually run.

    Args:
        requested: Names, a comma-separated string, or None for the development
            default. Order is not significant; the result is ordered by tier.
        path: Config path to name in error messages.

    Returns:
        The resolved selection, with the always-on metrics added and reported
        in `enforced`.

    Raises:
        ConfigValueError: If the request disables measurement entirely.
        UnknownBackendError: If a name is not registered, listing every metric
            that is.
    """
    items = _requested_names(requested, path)
    if not items:
        raise ConfigValueError(
            path,
            "an empty metric set measures nothing. Quality measurement is "
            "mandatory in every configuration; omit the key entirely for the "
            f"default ({', '.join(DEFAULT_METRICS)}).",
        )

    unknown = [item for item in items if item not in METRICS]
    if unknown:
        raise UnknownBackendError(AXIS, unknown[0], sorted(METRICS))

    chosen = set(items)
    enforced = tuple(sorted(ALWAYS_ON - chosen))
    chosen.update(ALWAYS_ON)

    tier_order = list(MetricTier)
    ordered = sorted(
        (METRICS[name] for name in chosen),
        key=lambda spec: (tier_order.index(spec.tier), spec.cost, spec.name),
    )
    return MetricSelection(metrics=tuple(ordered), enforced=enforced)


def resolve_tiers(
    tiers: Iterable[MetricTier | str],
    *,
    path: str = "evaluation.metric-tiers",
) -> MetricSelection:
    """Resolve a selection by tier rather than by naming each metric.

    The form a tier ladder config wants: `tiers: [fast, traditional]` is the
    headline-table run, `[fast]` is every development run.

    Raises:
        ConfigValueError: If a tier is not one of the four, or the set is empty.
    """
    wanted: list[MetricTier] = []
    for tier in tiers:
        try:
            wanted.append(MetricTier(tier))
        except ValueError:
            legal = ", ".join(item.value for item in MetricTier)
            raise ConfigValueError(path, f"unknown metric tier {tier!r}. Tiers: {legal}.") from None
    if not wanted:
        raise ConfigValueError(
            path, "no tiers selected; quality measurement is mandatory in every configuration."
        )
    names = [spec.name for tier in wanted for spec in by_tier(tier)]
    return resolve(names, path=path)


def describe_metrics(names: Sequence[str] | None = None) -> str:
    """A readable table of the registered metrics and their tiers."""
    chosen = [metric(name) for name in names] if names else list(METRICS.values())
    width = max(len(spec.name) for spec in chosen)
    lines = ["metrics:"]
    for spec in chosen:
        scope = "sequence" if spec.is_temporal else "frame"
        arrow = "higher better" if spec.higher_is_better else "lower better"
        floor = " [always on]" if spec.name in ALWAYS_ON else ""
        lines.append(
            f"  {spec.name.ljust(width)}  {spec.tier.value:<12} {spec.cost.name:<8} "
            f"{scope:<8} {arrow:<14}{floor}  {spec.summary}"
        )
    return "\n".join(lines)
