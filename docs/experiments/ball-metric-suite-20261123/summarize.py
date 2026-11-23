"""Pool each completed pair at the match level and preserve the registered denominator."""

import argparse
import hashlib
import json
from math import log10
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent / "cohort-expansion-20261122" / "cohort-results.json"
METRICS = ("dists", "lpips", "luma_ssim", "rgb_psnr", "whole_rgb_psnr")


def summarize(inputs):
    cohort = {row["match"]: row for row in json.loads(COHORT.read_text())["rows"]}
    rows = {}
    for path in inputs:
        record = json.loads(path.read_text())
        event = record["match"]
        assert event in cohort and event not in rows
        prior = cohort[event]
        assert record["source_ticks"] == 240 and len(record["selected_visible_ticks"]) == record["total_visible"]
        assert record["matched_rate_gap"] <= 0.005 and all(record["checks"].values())
        pair = []
        for tag in ("candidate", "ordinary"):
            arm = record["arms"][tag]
            assert [v["tick"] for v in arm["rows"]] == record["selected_visible_ticks"]
            assert abs(arm["kbps"] - prior[tag + "_kbps"]) < 1e-7
            assert abs(arm["means"]["dists"] - prior[tag + "_dists"]) < 2e-5
            assert abs(arm["whole_rgb_psnr"] - prior[tag + "_psnr"]) < 1e-8
            scores = arm["means"]
            pair.append(dict(
                kbps=arm["kbps"], bytes=arm["bytes"], packet_sha256=arm["packet_sha256"],
                dists=scores["dists"], lpips=scores["lpips"], luma_ssim=scores["luma_ssim"],
                rgb_psnr=-10 * log10(scores["rgb_mse"]), whole_rgb_psnr=arm["whole_rgb_psnr"],
            ))
        rows[event] = dict(match=event, event=prior["event"], total_visible=record["total_visible"],
                           source_sha256=record["source_sha256"],
                           receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           candidate=pair[0], ordinary=pair[1])
    n = len(rows)
    assert n
    completed = [rows[m] for m in cohort if m in rows]
    rng = np.random.default_rng(61123)
    boots = rng.integers(n, size=(10000, n))
    stats = {}
    for metric in METRICS:
        candidate = np.array([r["candidate"][metric] for r in completed])
        ordinary = np.array([r["ordinary"][metric] for r in completed])
        delta = candidate - ordinary
        interval = np.quantile(delta[boots].mean(1), [.025, .975]).tolist() if n > 1 else None
        better = delta < 0 if metric in ("dists", "lpips") else delta > 0
        stats[metric] = dict(candidate=mean(candidate), ordinary=mean(ordinary),
                             paired_difference=mean(delta), improvement_count=int(sum(better)),
                             match_bootstrap_95=interval)
    return dict(registered=list(cohort), complete=list(rows), matches=n,
                visible=sum(r["total_visible"] for r in completed), scores=stats, rows=completed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("inputs", type=Path, nargs="+")
    args = p.parse_args()
    with args.output.open("x") as f:
        f.write(json.dumps(summarize(args.inputs), indent=2, allow_nan=False) + "\n")
