"""W4: BD-rates of the receiver-mask composites against `dcvc-venue`, beside H5p 4a's F̂ composites on the same pairs.

``--inputs`` is H5p 4d's ``h5p-scoring-inputs.json`` (4a's scored rows, the anchors and the weighted-DISTS pairs);
each ``--score`` is a W4 scoring stage's ``h5.json`` (tags ``w4-gho`` and ``w4-venue``). The rule is H5p's: weighted
DISTS (fill) mean BD-rate of at most -10% with at least 3 of 4 items below zero, the seed billed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import h5

METRICS = ("wdists_fill", "wlpips_fill", "lpips_frame", "dists_frame", "wpsnr_fill")
RULE_MEAN = -0.10
RULE_ITEMS = 3
ANCHOR = "dcvc-venue"


THIN_PAIRS = 4


def thin(chosen: set[tuple[str, str]], rate: dict[tuple[str, str], float], count: int = THIN_PAIRS) -> set[tuple[str, str]]:
    """The fallback pair set, by rate position only: the item's hull pairs sorted by 4a's billed rate, the ones at
    ranks round(k (n - 1) / (count - 1)) for k = 0 .. count - 1 (all of them when there are at most ``count``)."""
    ranked = sorted(chosen, key=lambda key: (rate[key], key))
    if len(ranked) <= count:
        return set(ranked)
    return {ranked[round(k * (len(ranked) - 1) / (count - 1))] for k in range(count)}


def without_mask(row: dict[str, Any]) -> dict[str, Any]:
    """4a's row with the MASK segment's bytes taken off its rate (F̂ granted)."""
    return {**row, "kbps": row["kbps"] * (1.0 - row["breakdown"].get("mask", 0) / row["bytes"])}


def compare_variants(variants: dict[str, list[dict[str, Any]]], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: {m: h5.compare(h5.curve(rows, m), h5.curve(anchors, m), h5.METRICS[m][3]) for m in METRICS}
            for name, rows in variants.items()}


def summarize(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per variant and metric: the mean BD over the items that have one, and how many items are below zero."""
    out: dict[str, Any] = {}
    names = sorted({v for item in items.values() for v in item["bd"]})
    for name in names:
        for m in METRICS:
            per = [item["bd"][name][m] for item in items.values() if name in item["bd"]]
            bds = [c["bd"] for c in per if c["bd"] is not None]
            out[f"{name}|{m}"] = {"items": len(per), "items_with_bd": len(bds),
                                  "items_below_zero": sum(h5.below_zero(c) for c in per),
                                  "mean_bd": float(np.mean(bds)) if bds else None}
    return out


def verdict(summary: dict[str, Any], variant: str = "receiver", planned: int = 4) -> dict[str, Any]:
    """The rule on the items scored. It can pass only with all ``planned`` items; it can fail on fewer: a mean
    BD-rate above -10% over the scored items cannot be pulled down to -10% by the missing ones, each bounded
    below by -100%, when ``bound`` (the best mean the missing items could still give) stays above the limit."""
    s = summary[f"{variant}|wdists_fill"]
    mean = s["mean_bd"]
    missing = planned - s["items"]
    passes = mean is not None and missing == 0 and s["items_with_bd"] == s["items"] and mean <= RULE_MEAN \
        and s["items_below_zero"] >= RULE_ITEMS
    bound = None if mean is None else (mean * s["items_with_bd"] - missing) / planned
    fails_whatever_is_missing = bool(bound is not None and bound > RULE_MEAN)
    return {"rule": f"weighted DISTS mean BD <= {RULE_MEAN:+.0%} and >= {RULE_ITEMS} items below zero",
            "mean_bd": mean, "items_below_zero": s["items_below_zero"], "items": s["items"], "planned": planned,
            "best_possible_mean_with_missing_items": bound,
            "fails_whatever_is_missing": fails_whatever_is_missing, "passes": bool(passes)}


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--score", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--thin", action="store_true", help="score only the rate-position pairs (`thin`)")
    p.add_argument("--skip-items", default="", help="comma-separated videos with no receiver rows (not run)")
    p.add_argument("--thin-items", default="", help="comma-separated videos (e.g. P02_02) judged on the thinned pairs")
    args = p.parse_args()
    base = json.loads(args.inputs.read_text())
    stages = [json.loads(path.read_text()) for path in args.score]
    result: dict[str, Any] = {
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.inputs, *args.score]},
        "jobs": [{k: st.get(k) for k in ("job_id", "git_head", "gpu", "item", "frames", "seconds")} for st in stages],
        "items": {}}
    for item, selection in sorted(base["selection"].items()):
        chosen = {(pair, coder) for pair, coder, ms in selection if "wdists_fill" in ms}
        if args.thin or item.rsplit("_", 1)[0] in args.thin_items.split(","):
            rates = {(r["pair"], r["tag"].removeprefix("h5p4-")): r["kbps"] for st in base["scores"]
                     if st["item"] == item for r in st["rows"]}
            chosen = thin(chosen, rates)
        reference = [r for st in base["scores"] if st["item"] == item for r in st["rows"]
                     if (r["pair"], r["tag"].removeprefix("h5p4-")) in chosen]
        received = [r for st in stages if st["item"] == item for r in st["rows"]
                    if (r["pair"], r["tag"].removeprefix("w4-")) in chosen and r["tag"].startswith("w4-")]
        anchors = [r for st in base["anchors"] if st["item"] == item for r in st["rows"] if r["codec"] == ANCHOR]
        if not received and item.rsplit("_", 1)[0] in args.skip_items.split(","):
            continue  # not run (budget); the verdict counts it as missing
        if len(reference) != len(chosen) or len(received) != len(chosen) or not anchors:
            raise ValueError(f"{item}: {len(reference)} reference, {len(received)} receiver rows for "
                             f"{len(chosen)} pairs, {len(anchors)} anchors")
        masks = [st["mask"] for st in stages if st["item"] == item]
        record = {"pairs": len(chosen), "thinned": len(chosen) < sum(1 for _, _, ms in selection if "wdists_fill" in ms),
                  "receiver_mask": {
            k: masks[0][k] for k in ("iou", "iou_frame_mean", "iou_frame_min", "missed_share", "extra_share",
                                      "bytes", "seed_bytes", "kbps", "seed_is_first_union_mask")},
            "kbps": {"receiver": [min(r["kbps"] for r in received), max(r["kbps"] for r in received)]},
            "bd": compare_variants({"f_hat_billed": reference, "f_hat_free": [without_mask(r) for r in reference],
                                    "receiver": received}, anchors)}
        if any(m != masks[0] for m in masks):
            raise ValueError(f"{item}: its two stages used different receiver masks")
        result["items"][item] = record
    result["summary"] = summarize(result["items"])
    result["verdict"] = verdict(result["summary"])
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"verdict": result["verdict"], "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
