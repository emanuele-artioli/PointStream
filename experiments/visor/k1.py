"""PLAN K1: do keypoints win on egocentric hands at very low rates, against DCVC-UF + G? (docs/experiments.md, 2026-10-16).

    python -m experiments.visor.k1 --h3b2 H3B2-REPORT.json --h3d H3D-COMPONENT-REPORT.json --out K1.json

From recorded reports only, no new coding. H3b's oracle (perfect appearance at the coded pose) is scored on S, R plus
every rendered silhouette; H3d's points on R. Both scored SVT-AV1's eleven points (B2's four CRFs, H3b2's seven low
points; the same streams, sha256-checked by H3c), so each item's SVT-AV1 points give a monotone map from LPIPS on S to
LPIPS on R, linear between points: the oracle's LPIPS on S becomes the LPIPS on R that SVT-AV1 has where it scores the
same on S. That value is compared, at the oracle's rate, with the achievable hull (`g5.envelope`) of DCVC-UF + G alone
and of DCVC-UF + G together with H3c's bar (SVT-AV1 and DCVC-UF off the shelf).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background.g5 import envelope
from experiments.visor.b1 import file_sha256, write_json

VARIANTS = ("wilor:arm", "wilor:arm_uncoded", "wilor:hand", "hamer:arm")


def at_rate(hull: list[tuple[float, float]], kbps: float) -> float | None:
    """Quality (−LPIPS) on the hull at ``kbps``, linear in rate between its points (switching between encodes); None
    outside its rates."""
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    return float(np.interp(kbps, xs, ys)) if xs[0] <= kbps <= xs[-1] else None


def rate_for(hull: list[tuple[float, float]], quality: float) -> float | None:
    """The hull's rate for ``quality`` (−LPIPS), linear between its points; None outside its qualities."""
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    return float(np.interp(quality, ys, xs)) if ys[0] <= quality <= ys[-1] else None


def s_to_r(pairs: list[tuple[float, float]], lpips_s: float) -> float | None:
    """LPIPS on R for an LPIPS on S, linear between SVT-AV1's (S, R) points; None outside them or if not monotone."""
    pairs = sorted(pairs)
    s, r = [p[0] for p in pairs], [p[1] for p in pairs]
    if not all(np.diff(r) > 0) or not s[0] <= lpips_s <= s[-1]:
        return None
    return float(np.interp(lpips_s, s, r))


def compare(item: dict[str, Any], bar: list[dict[str, Any]], g_points: list[dict[str, Any]]) -> dict[str, Any]:
    on_r = {p["name"]: p["lpips"] for p in bar}
    on_s = {k: v["lpips"] for k, v in {**item["variants"], **item["low"]}.items() if k.startswith("svt@")}
    pairs = [(on_s[k], on_r[k]) for k in on_s if k in on_r]
    g = envelope([(p["kbps"], -p["lpips"]) for p in g_points])
    joint = envelope([(p["kbps"], -p["lpips"]) for p in g_points + bar])
    out: dict[str, Any] = {"svt_pairs": len(pairs), "g_hull": g, "joint_hull": joint, "variants": {}}
    for v in VARIANTS:
        kbps, lpips_s = item["variants"][v]["kbps"], item["variants"][v]["lpips"]
        lpips_r = s_to_r(pairs, lpips_s)
        row: dict[str, Any] = {"kbps": kbps, "lpips_s": lpips_s, "lpips_r": lpips_r}
        for name, hull in (("g", g), ("joint", joint)):
            q = at_rate(hull, kbps)
            row[f"{name}_lpips_at_rate"] = None if q is None else -q
            need = None if lpips_r is None else rate_for(hull, -lpips_r)
            row[f"{name}_kbps_at_quality"] = need
            row[f"{name}_saving"] = None if need is None else 1.0 - kbps / need
            # above the curve: better LPIPS than the hull at the oracle's rate
            row[f"{name}_above"] = None if lpips_r is None or q is None else bool(lpips_r < -q)
        out["variants"][v] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h3b2", required=True, help="H3b2's report (h3b report --low), with the oracle's and SVT-AV1's units on S")
    ap.add_argument("--h3d", required=True, help="H3d's component report (h3d report --component), with the bar and G on R")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    h3b2, h3d = json.loads(Path(args.h3b2).read_text()), json.loads(Path(args.h3d).read_text())
    items = {it["unit"]: compare(it, h3d["bar"][it["unit"]], h3d["component"]["arms"]["G-L"]["items"][it["unit"]]["front"])
             for it in h3b2["per_item"]}
    summary = {v: {name: sum(bool(it["variants"][v][f"{name}_above"]) for it in items.values()) for name in ("g", "joint")}
               for v in VARIANTS}
    write_json(Path(args.out), {
        "inputs": {"h3b2": {"path": args.h3b2, "sha256": file_sha256(Path(args.h3b2))},
                   "h3d": {"path": args.h3d, "sha256": file_sha256(Path(args.h3d))},
                   "code_sha256": file_sha256(Path(__file__))},
        "items": items, "items_above": summary,
    })
    for unit, it in items.items():
        for v, row in it["variants"].items():
            print(unit, v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()})
    print("items above:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
