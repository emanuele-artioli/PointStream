"""One payload ledger for a run.

The pre-rewrite tree split size counts across evaluation and invariants.
This module is the only place the runner writes ``sizes_bytes``. Parts use
the names the existing invariant check already reads: ``metadata``,
``actor_reference``, ``residual``, ``panorama``, plus ``source`` and
``transport_total``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.contracts.objectstream import WireCost

#: Same tolerance the legacy invariant used. A second constant here would be
#: a second ledger in disguise.
SIZE_SUM_TOLERANCE = 0.02

PARTS = ("metadata", "actor_reference", "residual", "panorama")


@dataclass(frozen=True)
class SizesBytes:
    """What one run (or one chunk) sent, in bytes.

    ``byte_count`` comes from measured ``WireCost`` / residual payloads, never
    from a model of an encoder.

    **``panorama`` under ``panorama-stream`` is a marginal cost.** Every other
    background method sends a whole plate per chunk, so the per-chunk figure and
    the per-chunk cost are the same thing. A streamed plate codes scene *n*
    against scene *n-1*'s reconstruction, so chunk *n*'s figure is only what
    that scene added — and the first chunk's keyframe is the whole plate.

    Summing across chunks is therefore still correct, and is correct *because*
    chunk 0 is in the sum. Dropping it, or treating the mean per-chunk figure as
    the cost of a plate, would report an amortisation the run never achieved.
    `plans/done/BP30-findings.md` §22 is the measurement this accounting has to carry.
    """

    source: int
    residual: int = 0
    panorama: int = 0
    actor_reference: int = 0
    metadata: int = 0
    transport_total: int = 0
    # Components still counted as raw array size rather than a coded bitstream.
    # A total mixing coded and raw parts is not a rate, and dividing it by the
    # source produces a number that looks like a compression ratio and is not
    # (`plans/done/RESEARCH-HISTORY.md` §3, BP24). Empty means every part was really coded.
    raw_parts: tuple[str, ...] = ()

    @property
    def is_rate(self) -> bool:
        """Whether ``transport_total`` may be compared against a codec at all."""
        return not self.raw_parts

    def as_dict(self) -> dict[str, Any]:
        ratio: float | None = (
            float(self.transport_total) / float(self.source)
            if self.source > 0 and self.is_rate
            else None
        )
        out: dict[str, Any] = {
            "source": self.source,
            "residual": self.residual,
            "panorama": self.panorama,
            "actor_reference": self.actor_reference,
            "metadata": self.metadata,
            "transport_total": self.transport_total,
            "transport_to_source_ratio": ratio,
            "is_rate": self.is_rate,
        }
        if self.raw_parts:
            out["raw_parts"] = list(self.raw_parts)
            out["not_a_rate"] = (
                "these parts are raw array sizes, not coded bitstreams: "
                + ", ".join(self.raw_parts)
                + ". transport_to_source_ratio is withheld because a mixed "
                "total is not a compression ratio."
            )
        return out

    @property
    def parts_sum(self) -> int:
        return self.metadata + self.actor_reference + self.residual + self.panorama

    def parts_fit(self) -> bool:
        """Whether the named parts do not exceed the transported total."""
        if self.transport_total <= 0:
            return False
        return self.parts_sum <= self.transport_total * (1.0 + SIZE_SUM_TOLERANCE)

    def __add__(self, other: SizesBytes) -> SizesBytes:
        return SizesBytes(
            source=self.source + other.source,
            residual=self.residual + other.residual,
            panorama=self.panorama + other.panorama,
            actor_reference=self.actor_reference + other.actor_reference,
            metadata=self.metadata + other.metadata,
            transport_total=self.transport_total + other.transport_total,
            # A raw part anywhere makes the sum raw. Dropping it here would
            # launder an uncoded chunk into a total that claims to be a rate.
            raw_parts=tuple(dict.fromkeys(self.raw_parts + other.raw_parts)),
        )


def measured(cost: WireCost) -> int:
    """Bytes from a stated ``WireCost``, or 0 when the cost is not on the wire."""
    if cost.byte_count is None:
        return 0
    return int(cost.byte_count)


def sizes_bytes(
    *,
    source: int,
    residual: int = 0,
    panorama: int = 0,
    actor_reference: int = 0,
    metadata: int = 0,
    raw_parts: Sequence[str] = (),
) -> SizesBytes:
    """Build one ledger. ``transport_total`` is the sum of transmitted parts.

    All-off transmits the source itself, so when every semantic part is zero
    the transported total is the source size — that is the baseline corner,
    not a missing measurement.

    ``raw_parts`` names any component still counted as an array size rather
    than a coded bitstream. Anything listed there withholds the ratio, because
    a total mixing coded and raw parts is not a rate (BP24).
    """
    parts = metadata + actor_reference + residual + panorama
    transport_total = parts if parts > 0 else source
    unknown = sorted(set(raw_parts) - set(PARTS))
    if unknown:
        raise ValueError(f"raw_parts names unknown components: {unknown}; known: {list(PARTS)}")
    return SizesBytes(
        source=source,
        residual=residual,
        panorama=panorama,
        actor_reference=actor_reference,
        metadata=metadata,
        transport_total=transport_total,
        raw_parts=tuple(dict.fromkeys(raw_parts)),
    )
