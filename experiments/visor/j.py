"""J: experience-track re-judging (docs/experiments.md, 2026-10-22 "J: experience-track re-judging (rule)").

    python -m experiments.visor.j h3e --report H3E_REPORT_JSON --out DIR

An item passes when three checks hold at the method's rate against the anchor hull at that rate:

* realism: the method's KID (against a fixed held-out reference set) <= the anchor's upper 95% bound;
* same content: the method's pose error <= max(the anchor's, 1.5 x the coded keypoints' own error);
* same over time (the user's reading, 2026-10-22): the method's ``h3b.flicker`` <= the anchor's, OR the method's
  own residual energy <= 1.25 x the source's.

Each anchor quantity is read on its own hull: the lower convex hull of the quantity over log rate, linear in log rate
at the method's rate, clamped to the anchor's range (flagged). A result passes when at least 75% of its scored items
pass.

**Anchor subsets.** The rule's hand anchors are SVT-AV1 and DCVC-UF + G. Adding anchor points can only lower each
quantity's hull inside the rate range the smaller set already spans. So a check read against a subset of the anchors
(SVT-AV1 alone, which is CPU-only) is conclusive where it *fails* and the method's rate lies inside the subset's range:
the full anchor would be at most as high. A subset pass, or a rate outside the subset's range, decides nothing
(``None``). `decide` then reads a result's decision from whatever is conclusive.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CRITERION = 0.75  # share of a result's scored items that must pass
CONTENT_FACTOR = 1.5  # x the coded keypoints' own error against the source
ENERGY_FACTOR = 1.25  # x the source's own residual energy
DECISION = {"criterion": CRITERION, "content_factor": CONTENT_FACTOR, "energy_factor": ENERGY_FACTOR,
            "time": "h3b.flicker <= anchor's OR residual energy <= 1.25 x source's (user, 2026-10-22)",
            "hull": "lower convex hull of each quantity over log rate; linear in log rate; clamped, flagged"}


# ----------------------------------------------------------------- the anchor hull of one quantity


def hull(points: Sequence[tuple[float, float | None]]) -> list[tuple[float, float]]:
    """Lower convex hull of (kbps, value) in (log kbps, value), by rising rate. Points with a missing or non-finite
    value or a non-positive rate are left out; at equal rates the lower value is kept."""
    pts = sorted({(float(r), float(v)) for r, v in points if v is not None and r > 0 and math.isfinite(v) and math.isfinite(r)})
    out: list[tuple[float, float]] = []
    for r, v in pts:
        if out and out[-1][0] == r:
            continue  # sorted by value within a rate: the first is the lowest
        while len(out) >= 2:
            (r1, v1), (r2, v2) = out[-2], out[-1]
            x1, x2, x3 = math.log(r1), math.log(r2), math.log(r)
            # out[-1] on or above the chord out[-2] -> (r, v): not on the lower hull
            if (v2 - v1) * (x3 - x1) >= (v - v1) * (x2 - x1):
                out.pop()
            else:
                break
        out.append((r, v))
    return out


def read(points: Sequence[tuple[float, float | None]], kbps: float) -> dict[str, Any]:
    """The hull's value at ``kbps`` (linear in log rate), clamped to its range; ``flag`` says where it was clamped."""
    h = hull(points)
    if not h:
        return {"value": None, "flag": "no anchor points"}
    if kbps <= h[0][0]:
        return {"value": h[0][1], "flag": None if kbps == h[0][0] else "below the anchor's range"}
    if kbps >= h[-1][0]:
        return {"value": h[-1][1], "flag": None if kbps == h[-1][0] else "above the anchor's range"}
    for (r1, v1), (r2, v2) in zip(h, h[1:]):
        if r1 <= kbps <= r2:
            w = (math.log(kbps) - math.log(r1)) / (math.log(r2) - math.log(r1))
            return {"value": v1 + w * (v2 - v1), "flag": None}
    raise AssertionError("unreachable")


# ----------------------------------------------------------------- the three checks


def _against(value: float | None, anchor: dict[str, Any], complete: bool, floor: float | None = None) -> bool | None:
    """``value <= max(anchor, floor)``. With a subset of the anchors only a failure inside its range is conclusive."""
    if value is None or not math.isfinite(value) or anchor["value"] is None:
        return None
    bar = anchor["value"] if floor is None else max(anchor["value"], floor)
    ok = value <= bar
    if complete:
        return ok
    if ok or anchor["flag"] is not None:
        return None
    return False


def content(pose: float | None, anchor: dict[str, Any], keypoints_error: float | None, complete: bool) -> bool | None:
    floor = CONTENT_FACTOR * keypoints_error if keypoints_error is not None else None
    if not complete and floor is not None and pose is not None and pose <= floor:
        return True  # the anchor-free branch passes whatever the anchor reads
    return _against(pose, anchor, complete, floor)


def realism(kid: float | None, anchor_kid_high: dict[str, Any], complete: bool) -> bool | None:
    return _against(kid, anchor_kid_high, complete)


def over_time(flicker: float | None, anchor_flicker: dict[str, Any], energy: float | None, source_energy: float | None,
              complete: bool) -> bool | None:
    """The OR reading: the energy branch needs no anchor, so it is conclusive alone when it passes."""
    energy_ok = (energy <= ENERGY_FACTOR * source_energy if energy is not None and source_energy is not None
                 and math.isfinite(energy) and math.isfinite(source_energy) else None)
    if energy_ok:
        return True
    flicker_ok = _against(flicker, anchor_flicker, complete)
    if flicker_ok is None:
        return None
    if flicker_ok:
        return True
    return False if energy_ok is False else None


def item_status(checks: dict[str, bool | None]) -> bool | None:
    """An item passes when every check passes; one failing check fails it; otherwise it is undecided."""
    values = list(checks.values())
    if any(v is False for v in values):
        return False
    if values and all(v is True for v in values):
        return True
    return None


def decide(statuses: dict[str, bool | None]) -> dict[str, Any]:
    """A result passes when at least `CRITERION` of its scored items pass; it is decided as soon as the undecided
    items cannot change that."""
    n = len(statuses)
    need = math.ceil(CRITERION * n)
    passed = sum(1 for s in statuses.values() if s is True)
    failed = sum(1 for s in statuses.values() if s is False)
    if n and passed >= need:
        decision = "pass"
    elif n and n - failed < need:
        decision = "fail"
    else:
        decision = "undecided"
    return {"items": n, "need": need, "passed": passed, "failed": failed, "undecided": n - passed - failed,
            "decision": decision}


def check_rates(rows: dict[str, dict[str, bool | None]]) -> dict[str, dict[str, int]]:
    """Per check, how many items pass, fail and are unscored or undecided."""
    names = sorted({c for r in rows.values() for c in r})
    return {c: {"pass": sum(1 for r in rows.values() if r.get(c) is True), "fail": sum(1 for r in rows.values() if r.get(c) is False),
                "none": sum(1 for r in rows.values() if r.get(c) is None)} for c in names}


# ----------------------------------------------------------------- judges shared by hands and players

LUMA = (0.299, 0.587, 0.114)
MISSED = 1.0  # content error of an object the output's estimator does not fit (one box)
KP_CONF = 0.5  # K4's keypoint confidence (players); hands' WiLoR joints carry 1


def pose_error(pred: Any | None, ref: Any, height: float) -> float | None:
    """Mean distance over ``ref``'s confident keypoints, over ``height``; `MISSED` when ``pred`` is None."""
    import numpy as np

    visible = ref[:, 2] >= KP_CONF
    if not visible.any():
        return None
    if pred is None:
        return MISSED
    return float(np.linalg.norm(pred[visible, :2] - ref[visible, :2], axis=1).mean() / max(height, 1.0))


def residual_energies(out_t: Any, out_p: Any, truth_t: Any, truth_p: Any, region: Any,
                      region_p: Any) -> tuple[float, float, float] | None:
    """(h3b.flicker, the output's residual energy, the source's), on ``region`` at t, flow from the source (as H3e)."""
    import cv2
    import numpy as np

    from experiments.visor.h3 import dis_flow, flow_box, warp
    from experiments.visor.h3b import flicker

    if not region.any() or not region_p.any():
        return None
    fb = flow_box(region, region_p)
    crop = (slice(fb[1], fb[3]), slice(fb[0], fb[2]))
    tt, tp = truth_t[crop], truth_p[crop]
    flow = dis_flow(cv2.cvtColor(tt, cv2.COLOR_RGB2GRAY), cv2.cvtColor(tp, cv2.COLOR_RGB2GRAY))
    ow = warp(np.ascontiguousarray(out_p[crop]), flow)
    tw = warp(np.ascontiguousarray(tp), flow)
    m = region[crop]
    f = flicker(out_t[crop], ow, tt, tw, m)
    e_out = float(np.abs((out_t[crop].astype(np.float32) - ow.astype(np.float32)) @ np.asarray(LUMA, np.float32))[m].mean())
    e_src = float(np.abs((tt.astype(np.float32) - tw.astype(np.float32)) @ np.asarray(LUMA, np.float32))[m].mean())
    return f, e_out, e_src


# ----------------------------------------------------------------- H3e V1 (from its saved report)


def h3e_item(item: dict[str, Any]) -> dict[str, Any]:
    """Same content for every V1 point against SVT-AV1's points alone (a subset of the rule's anchor), with the
    hull H3e used (SVT-AV1 and DCVC-UF off the shelf) beside. Realism and time are not read here."""
    svt = [(p["kbps"], p["pose"]) for p in item["pixel_points"] if p["name"].startswith("svt@")]
    pixel = [(p["kbps"], p["pose"]) for p in item["pixel_points"]]
    kp = float(item["keypoints_pose"])
    points = []
    for p in sorted(item["v1"], key=lambda q: q["kbps"]):
        a = read(svt, p["kbps"])
        b = read(pixel, p["kbps"])
        checks = {"content": content(p["pose"], a, kp, complete=False), "realism": None, "time": None}
        points.append({"name": p["name"], "kbps": p["kbps"], "pose": p["pose"], "svt_pose": a, "h3e_hull_pose": b,
                       "floor": CONTENT_FACTOR * kp, "checks": checks, "status": item_status(checks)})
    # the encoder may choose: the lowest-rate point that passes all three, else the lowest-rate point
    passing = [q for q in points if q["status"] is True]
    if passing:
        status: bool | None = True
        chosen = passing[0]
    elif all(q["status"] is False for q in points):
        status, chosen = False, points[0]
    else:
        status, chosen = None, points[0]
    return {"keypoints_pose": kp, "judged_hands": item["judged_hands"], "points": points, "status": status,
            "operating_point": chosen["name"], "content": False if status is False else None}


def command_h3e(args: argparse.Namespace) -> int:
    from experiments.visor.b1 import file_sha256, write_json

    report = Path(args.report)
    source = json.loads(report.read_text())
    items = {k: h3e_item(v) for k, v in sorted(source["items"].items())}
    statuses = {k: v["status"] for k, v in items.items()}
    out = {"result": "H3e V1", "decision": decide(statuses), "rule": DECISION,
           "checks": check_rates({k: {"content": v["content"]} for k, v in items.items()}),
           "anchor": "SVT-AV1's points alone (H3c2's judge): a subset of SVT-AV1 + DCVC-UF + G, conclusive where it fails",
           "not_scored": "realism and time: the decision is reached on content (protocol: stop when the decision is made)",
           "items": items, "input": {"path": str(report), "sha256": file_sha256(report)}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out) / "j-h3e.json", out)
    print(json.dumps({"decision": out["decision"], "items": {k: [(q["name"], round(q["kbps"], 2), round(100 * q["pose"], 2),
                                                                  None if q["svt_pose"]["value"] is None else round(100 * q["svt_pose"]["value"], 2),
                                                                  q["svt_pose"]["flag"], q["checks"]["content"]) for q in v["points"]]
                                                         for k, v in items.items()}}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.j", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    h3e = sub.add_parser("h3e")
    h3e.add_argument("--report", required=True)
    h3e.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"h3e": command_h3e}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
