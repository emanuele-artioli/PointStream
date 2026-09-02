"""Pure checks for the ultra-low-rate probe. No encoder binaries, no torch.

A point that decodes to the wrong number of frames, or a curve that walks the
wrong way as QP rises, can still produce a smooth-looking table. These helpers
are the rejections the probe applies before a number is allowed onto a curve.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.components.codec.encode import QP_BOUNDS
from src.contracts import paths as ps_paths
from src.contracts.metrics import HEADLINE_CURVE_METRICS, VMAF, metric as metric_spec

OUT_DIR = ps_paths.outputs() / "bp45-low-rate"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"
PROBE_PATH = OUT_DIR / "codec-floor.json"
CALIBRATION_PATH = OUT_DIR / "metric-calibration.json"

#: ROADMAP §3 names 24 fps. Recorded on every point so a later bitrate
#: conversion cannot invent a frame rate.
DECLARED_FPS = 24.0

#: Slowest first. The probe picks the first entry the binary actually accepts.
PRESET_ORDER: dict[str, tuple[str, ...]] = {
    "av1": tuple(str(index) for index in range(0, 14)),
    "vvc": ("placebo", "veryslow", "slower", "slow", "medium", "fast", "faster"),
}

PRIMARY_ANCHORS: tuple[str, ...] = ("av1", "vvc")
HEADLINE_METRICS: tuple[str, ...] = HEADLINE_CURVE_METRICS


def slowest_preset(codec: str, available: Sequence[str] | None = None) -> str:
    """The slowest preset this codec lists, optionally intersected with ``available``.

    ``available`` is what the binary advertised or accepted. Passing None means
    "use the documented slowest", which is the primary-comparison default.
    """
    try:
        order = PRESET_ORDER[codec]
    except KeyError:
        raise ValueError(
            f"no slowest-preset table for {codec!r}; known: {sorted(PRESET_ORDER)}"
        ) from None
    if available is None:
        return order[0]
    wanted = {str(item) for item in available}
    for preset in order:
        if preset in wanted:
            return preset
    raise ValueError(
        f"none of the documented presets for {codec!r} are in {sorted(wanted)}. "
        f"Documented slowest-first: {list(order)}"
    )


def legal_qps(codec: str) -> tuple[int, int]:
    """Inclusive QP bounds from the encode layer, not a second copy of the table."""
    try:
        return QP_BOUNDS[codec]
    except KeyError:
        raise ValueError(f"no QP bounds for {codec!r}; known: {sorted(QP_BOUNDS)}") from None


def probe_qps(codec: str) -> tuple[int, ...]:
    """A sparse walk of the legal range, coarsest first, including both endpoints.

    Every legal QP would make the slowest-preset 4K probe a multi-day job.
    Endpoints plus interior samples still reject an empty, undecodable or
    non-monotone range. A crossover later adds neighbour QPs, not a denser
    default walk.
    """
    low, high = legal_qps(codec)
    interior = (55, 51, 45, 40, 32, 24, 16, 8)
    chosen = [high, *[qp for qp in interior if low < qp < high], low]
    # Preserve coarsest-first and drop duplicates if the endpoint is already
    # in the interior list.
    seen: set[int] = set()
    ordered: list[int] = []
    for qp in chosen:
        if qp not in seen:
            seen.add(qp)
            ordered.append(int(qp))
    return tuple(ordered)


def decode_rejections(
    *,
    bitstream_bytes: int,
    source_shape: tuple[int, int, int, int],
    decoded_shape: tuple[int, ...] | None,
) -> list[str]:
    """Why a decode must not enter a curve. Empty list means the point is usable."""
    reasons: list[str] = []
    frames, height, width, channels = source_shape
    if bitstream_bytes <= 0:
        reasons.append("bitstream is empty")
    if decoded_shape is None:
        reasons.append("decode produced no frames")
        return reasons
    if len(decoded_shape) != 4:
        reasons.append(f"decoded shape {decoded_shape} is not (T,H,W,C)")
        return reasons
    got_t, got_h, got_w, got_c = decoded_shape
    if got_t == 0:
        reasons.append("decode produced zero frames")
    elif got_t != frames:
        reasons.append(f"decoded {got_t} frames, source has {frames}")
    if (got_h, got_w) != (height, width):
        reasons.append(f"decoded {got_w}x{got_h}, source is {width}x{height}")
    if got_c != channels:
        reasons.append(f"decoded {got_c} channels, source has {channels}")
    return reasons


def monotonicity_alarms(
    qps: Sequence[int],
    rates: Sequence[float],
    qualities: Sequence[float],
    *,
    higher_is_better: bool,
    rate_rel_tolerance: float = 0.05,
    quality_rel_tolerance: float = 0.05,
) -> list[str]:
    """QP up should not grow rate or improve quality, beyond a small inversion.

    Encoders are allowed a little non-monotone noise. An end-to-end inversion
    — coarsest QP cheaper *and* better than the finest — is a broken probe.
    """
    if len(qps) != len(rates) or len(qps) != len(qualities):
        raise ValueError("qps, rates and qualities must be the same length")
    if len(qps) < 2:
        return ["need at least two valid points to judge monotonicity"]

    order = sorted(range(len(qps)), key=lambda index: qps[index])
    qp = [int(qps[index]) for index in order]
    rate = [float(rates[index]) for index in order]
    quality = [float(qualities[index]) for index in order]

    alarms: list[str] = []
    rate_span = max(rate) - min(rate)
    quality_span = max(quality) - min(quality)
    rate_slack = max(abs(rate_rel_tolerance * rate_span), 1.0)
    quality_slack = max(abs(quality_rel_tolerance * quality_span), 1e-6)

    for left, right in zip(range(len(qp) - 1), range(1, len(qp))):
        if rate[right] > rate[left] + rate_slack:
            alarms.append(
                f"rate rose as QP rose ({qp[left]}→{qp[right]}: "
                f"{rate[left]:.0f}→{rate[right]:.0f} B)"
            )
        improved = quality[right] > quality[left] + quality_slack
        worsened = quality[right] < quality[left] - quality_slack
        if higher_is_better and improved:
            alarms.append(
                f"quality rose as QP rose ({qp[left]}→{qp[right]}: "
                f"{quality[left]:.4g}→{quality[right]:.4g})"
            )
        if (not higher_is_better) and worsened:
            alarms.append(
                f"quality fell as QP rose on a lower-is-better axis "
                f"({qp[left]}→{qp[right]}: {quality[left]:.4g}→{quality[right]:.4g})"
            )

    if rate[-1] > rate[0] + rate_slack:
        alarms.append(
            f"end-to-end rate rose from QP {qp[0]} to {qp[-1]} "
            f"({rate[0]:.0f}→{rate[-1]:.0f} B)"
        )
    end_improved = quality[-1] > quality[0] + quality_slack
    end_worsened = quality[-1] < quality[0] - quality_slack
    if higher_is_better and end_improved:
        alarms.append(
            f"end-to-end quality rose from QP {qp[0]} to {qp[-1]} "
            f"({quality[0]:.4g}→{quality[-1]:.4g})"
        )
    if (not higher_is_better) and end_worsened:
        alarms.append(
            f"end-to-end quality fell from QP {qp[0]} to {qp[-1]} on a "
            f"lower-is-better axis ({quality[0]:.4g}→{quality[-1]:.4g})"
        )
    return alarms


def primary_quality_name() -> str:
    return VMAF.name


def higher_is_better(name: str) -> bool:
    return metric_spec(name).higher_is_better


__all__ = [
    "BOUNDS_PATH",
    "CALIBRATION_PATH",
    "DECLARED_FPS",
    "HEADLINE_METRICS",
    "OUT_DIR",
    "PRESET_ORDER",
    "PRIMARY_ANCHORS",
    "PROBE_PATH",
    "decode_rejections",
    "higher_is_better",
    "legal_qps",
    "monotonicity_alarms",
    "primary_quality_name",
    "probe_qps",
    "slowest_preset",
]
