"""Paired comparison of the reference modes, from a finished sweep.

Every arm codes the *same* scenes, so comparing arms by their unpaired spreads
throws away the pairing and understates what the run can resolve. `AGENTS.md`
requires n and the uncertainty with any comparison, and a difference under about
two standard errors is not a finding -- this computes the paired difference per
scene so that test is applied to the right quantity.

Reads `outputs/bp30-background/stream-sweep.json` and writes
`outputs/bp30-background/mode-comparison.json`. Does no encoding, so it is cheap
to re-run and cannot change the measurement it is reading.

Run: ``python -m experiments.tier.background_stream_compare``
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from src.contracts import paths as ps_paths

SWEEP_PATH = ps_paths.outputs() / "bp30-background" / "stream-sweep.json"
OUT_PATH = ps_paths.outputs() / "bp30-background" / "mode-comparison.json"

#: The bar this project uses before calling a difference real.
SIGNIFICANCE_SIGMA = 2.0


def _ratios_by_scene(arm: dict[str, Any]) -> dict[int, float]:
    """Per-scene marginal ratio, keyframes excluded.

    A keyframe is a fresh plate by construction, so including it would compare
    the arms on a scene where neither made a prediction.
    """
    return {
        int(row["index"]): float(row["ratio"])
        for row in arm["per_scene"]
        if row["type"] != "I" and row["ratio"] is not None
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Paired difference ``left - right`` over the scenes both predicted."""
    a, b = _ratios_by_scene(left), _ratios_by_scene(right)
    shared = sorted(set(a) & set(b))
    differences = [a[i] - b[i] for i in shared]
    if len(differences) < 2:
        return {"left": left["arm"], "right": right["arm"], "n": len(differences),
                "verdict": "too few shared scenes to compare"}
    mean = statistics.fmean(differences)
    stderr = statistics.stdev(differences) / len(differences) ** 0.5
    sigmas = abs(mean) / stderr if stderr else float("inf")
    better = left["arm"] if mean < 0 else right["arm"]
    return {
        "left": left["arm"],
        "right": right["arm"],
        "n_paired_scenes": len(differences),
        "mean_difference": round(mean, 5),
        "stderr": round(stderr, 5),
        "sigmas": round(sigmas, 2),
        "cheaper_arm": better,
        "verdict": (
            f"{better} is cheaper by {abs(mean) * 100:.2f} percentage points of the "
            f"fresh-intra cost, {sigmas:.1f} standard errors"
            if sigmas >= SIGNIFICANCE_SIGMA
            else (
                f"no difference: {abs(mean) * 100:.2f} points at {sigmas:.1f} standard "
                f"errors, under this project's {SIGNIFICANCE_SIGMA:.0f}-sigma bar"
            )
        ),
        "is_a_finding": sigmas >= SIGNIFICANCE_SIGMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=str(SWEEP_PATH))
    args = parser.parse_args()

    sweep = json.loads(open(args.sweep, encoding="utf-8").read())
    arms = {arm["arm"]: arm for arm in sweep["arms"]}

    pairs = [
        ("mode=best-scored", "mode=first"),
        ("mode=best-scored", "mode=last"),
        ("mode=last", "mode=first"),
    ]
    rows = [compare(arms[a], arms[b]) for a, b in pairs if a in arms and b in arms]
    for row in rows:
        print(f"  {row['left']:<20} vs {row['right']:<16} {row['verdict']}", flush=True)

    # The keyframe axis is a cost, not a comparison: report what each k spends
    # against the pure P-chain so the paper's robustness paragraph has a number.
    baseline = arms.get("mode=last")
    ladder = []
    for arm in sweep["arms"]:
        if arm.get("mode") != "periodic-i" and arm["arm"] != "mode=last":
            continue
        interval = arm.get("keyframe_interval")
        overhead = (
            arm["total_bytes"] / baseline["total_bytes"] if baseline and baseline["total_bytes"] else None
        )
        ladder.append(
            {
                "keyframe_interval": "never" if arm["arm"] == "mode=last" else interval,
                "total_bytes": arm["total_bytes"],
                "vs_all_intra": arm["sequence_ratio_vs_all_intra"],
                "vs_pure_p_chain": round(overhead, 4) if overhead else None,
                "keyframes": arm["keyframes"],
            }
        )
    ladder.sort(key=lambda r: (r["keyframe_interval"] == "never", r["keyframe_interval"] if isinstance(r["keyframe_interval"], int) else 0))
    print("\n  keyframe interval ladder (cost of random access and loss resilience):", flush=True)
    for row in ladder:
        print(
            f"    k={str(row['keyframe_interval']):<6} total={row['total_bytes']:>10,} B  "
            f"vs all-intra={row['vs_all_intra']}  vs pure-P={row['vs_pure_p_chain']}  "
            f"keyframes={row['keyframes']}",
            flush=True,
        )

    payload = {
        "question": "do the reference modes actually differ, once the pairing is used?",
        "source": args.sweep,
        "n_scenes": sweep.get("n_scenes"),
        "n_videos": sweep.get("n_videos"),
        "n_caveat": sweep.get("n_caveat"),
        "significance_bar_sigma": SIGNIFICANCE_SIGMA,
        "paired_comparisons": rows,
        "keyframe_ladder": ladder,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
