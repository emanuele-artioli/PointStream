"""Machine-checkable versions of the methodology rules in CLAUDE.md.

The rules that matter here are the ones a bad run does not announce. A pipeline
that silently fell back to a mock source, or whose payload accounting does not
add up, or whose PSNR came back null, still writes a perfectly well-formed
`run_summary.json` — and three weeks later that summary is indistinguishable
from a good one. Prose in CLAUDE.md cannot prevent it being cited; a verdict
stored in the run itself can.

The verdict is written into the summary under `invariant_failures`. **A run
with a non-empty `invariant_failures` is not citable** — `/results-report` and
`/update-paper` refuse it. A run with *no* verdict at all predates the checks
and must be backfilled, because a missing verdict reads as clean:

    python -m src.shared.invariants outputs/
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# The core thesis: the payload we send must be smaller than the source it
# replaces. A run where it is not has not disproved the approach — it has failed
# to demonstrate it, and must not be reported as a saving.
MAX_TRANSPORT_TO_SOURCE_RATIO = 1.0

# Payload components are expected to sum to the transported total. Rounding and
# container overhead make an exact match unrealistic, so this is a tolerance on
# the accounting being *approximately* honest, not an equality check.
SIZE_SUM_TOLERANCE = 0.02


def _as_number(value: Any) -> Optional[float]:
    """The value as a float, or None if it is absent or not a real measurement.

    `bool` is excluded deliberately: it passes `isinstance(x, int)`, so a
    summary carrying `psnr_mean: true` would otherwise sail through as 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def check_run(summary: Dict[str, Any]) -> List[str]:
    """Everything checkable from one run summary. Returns failure descriptions."""
    failures: List[str] = []
    failures += _check_ran_on_real_input(summary)
    failures += _check_quality_measured(summary)
    failures += _check_size_accounting(summary)
    failures += _check_residual_guarantee(summary)
    return failures


def _check_ran_on_real_input(summary: Dict[str, Any]) -> List[str]:
    """Omitting --input falls back to a mock source and silently tests nothing.

    The run completes and produces numbers, which is precisely the problem.
    """
    source = summary.get("source_uri")
    if not source:
        return ["source_uri is missing; cannot tell whether this ran on real input"]
    if "mock" in str(source).lower():
        return [
            f"ran against a mock source ({source}); a real experiment must pass "
            f"--input explicitly"
        ]

    frames = summary.get("num_frames")
    if isinstance(frames, int) and frames <= 0:
        return [f"num_frames is {frames}; nothing was processed"]
    return []


def _check_quality_measured(summary: Dict[str, Any]) -> List[str]:
    """A null PSNR is a failed evaluation, not a quality of zero."""
    evaluation = summary.get("evaluation") or {}
    if not evaluation:
        return ["evaluation block is missing; this run carries no quality numbers"]

    failures = []
    raw_psnr = evaluation.get("psnr_mean")
    psnr = _as_number(raw_psnr)
    if raw_psnr is None:
        failures.append("psnr_mean is null — quality evaluation did not complete")
    elif psnr is None or psnr <= 0:
        failures.append(f"psnr_mean is not a plausible measurement ({raw_psnr!r})")

    frames = evaluation.get("psnr_num_frames")
    if isinstance(frames, int) and frames <= 0:
        failures.append("psnr was computed over zero frames")
    return failures


def _check_size_accounting(summary: Dict[str, Any]) -> List[str]:
    """The payload parts must roughly sum to the transported total.

    If they do not, the size axis of every Residual-Guarantee claim is measuring
    something other than what was sent.
    """
    sizes = (summary.get("evaluation") or {}).get("sizes_bytes") or {}
    if not sizes:
        return ["sizes_bytes is missing; payload claims cannot be made from this run"]

    total = _as_number(sizes.get("transport_total"))
    if total is None or total <= 0:
        return [f"transport_total is not a positive number ({sizes.get('transport_total')!r})"]

    parts = ["metadata", "actor_reference", "residual", "panorama"]
    present = [n for p in parts if (n := _as_number(sizes.get(p))) is not None]
    if not present:
        return []

    summed = sum(present)
    if summed > total * (1 + SIZE_SUM_TOLERANCE):
        return [
            f"payload components sum to {summed} bytes, more than the reported "
            f"transport_total of {total}"
        ]
    return []


def _check_residual_guarantee(summary: Dict[str, Any]) -> List[str]:
    """The thesis: what we send must be smaller than the source it replaces."""
    sizes = (summary.get("evaluation") or {}).get("sizes_bytes") or {}
    ratio = _as_number(sizes.get("transport_to_source_ratio"))
    if ratio is None:
        return []

    if ratio > MAX_TRANSPORT_TO_SOURCE_RATIO:
        return [
            f"transported payload is {ratio:.2f}x the source, so this run shows no "
            f"saving and must not be reported as one"
        ]
    return []


def backfill(outputs_dir: str = "outputs", force: bool = False) -> Dict[str, List[str]]:
    """Write an `invariant_failures` verdict into every run_summary.json in place.

    Metadata only — no re-running, no re-evaluation — and re-entrant.
    Returns the offending run directories and their failures.
    """
    import json
    import os

    offenders: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(outputs_dir)):
        path = os.path.join(outputs_dir, entry, "run_summary.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as handle:
                summary = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if "invariant_failures" in summary and not force:
            if summary["invariant_failures"]:
                offenders[entry] = summary["invariant_failures"]
            continue
        failures = check_run(summary)
        summary["invariant_failures"] = failures
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(summary, handle, indent=2)
        os.replace(tmp, path)
        if failures:
            offenders[entry] = failures
    return offenders


def main() -> None:
    """`python -m src.shared.invariants [outputs_dir] [--force]`"""
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    outputs_dir = args[0] if args else "outputs"
    offenders = backfill(outputs_dir, force="--force" in sys.argv[1:])
    if not offenders:
        print(f"{outputs_dir}: every run satisfies its invariants")
        return
    print(f"{len(offenders)} run(s) are NOT citable:")
    for run_id, failures in offenders.items():
        print(f"  {run_id}")
        for failure in failures:
            print(f"    - {failure}")


if __name__ == "__main__":
    main()
