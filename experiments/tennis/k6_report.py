"""K6's report (``python -m experiments.tennis.k6 report``): `k6.DECISION` over locally validated stages, CPU only.

Keypoint rates are pooled over every window in the report (each symbol costs -log2 of its pooled frequency, K4b.3's
optimistic adaptive bound, plus the absolute starts). Per window, each variant's quality is its mean over the scored
frames; the anchors' achievable envelope is `g5.envelope` on (kbps on R, -metric).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background.g5 import envelope
from experiments.tennis import k4
from experiments.tennis.k6 import ANCHORS, CHOSEN, CODES, DECISION, HEIGHT_BINS, anchor_name, code_name, code_tracks, coded_keypoints
from experiments.visor.b1 import file_sha256, write_json
from src.segmentation.masks import decompress

ANCHOR_NAMES = [anchor_name(a) for a in ANCHORS]


def load(stages: list[Path]) -> list[dict[str, Any]]:
    out = []
    for stage in stages:
        summary = json.loads((stage / "k6.json").read_text())
        validation = json.loads((stage / "local-validation.json").read_text())
        if summary["stage"] != "full" or validation.get("passed") is not True:
            raise ValueError(f"{stage}: only locally validated full stages")
        for row in summary["windows"]:
            root = stage / "x" / "publish" / k4.safe(row["id"])
            if any(file_sha256(root / n) != h for n, h in row["files"].items()):
                raise ValueError(f"Changed artifact under {root}")
            a = dict(np.load(root / "scores.npz"))
            kp = json.loads(decompress((root / "keypoints.json.zst").read_bytes()))
            poses = json.loads(decompress((root / "poses.json.zst").read_bytes()))
            outline = json.loads(decompress((root / "outline.json.zst").read_bytes()))
            out.append({"row": row, "scores": a, "symbols": kp["symbols"], "poses": poses, "outline": outline["rows"],
                        "stage": stage.name, "git_head": json.loads((stage / "ready.json").read_text())["git_head"]
                        if (stage / "ready.json").exists() else None})
    if len({w["row"]["id"] for w in out}) != len(out):
        raise ValueError("A window appears twice")
    return out


def pooled_bits(symbol_sets: list[dict[str, Any]]) -> list[float]:
    """Each set's bits under the residual and visibility frequencies pooled over all sets, plus its absolute starts."""
    res: Counter[int] = Counter()
    vis: Counter[int] = Counter()
    for s in symbol_sets:
        res.update(s["residuals"])
        vis.update(s["visibility"])
    nr, nv = sum(res.values()), sum(vis.values())
    cost_r = {k: -math.log2(c / nr) for k, c in res.items()}
    cost_v = {k: -math.log2(c / nv) for k, c in vis.items()}
    return [sum(cost_r[x] for x in s["residuals"]) + sum(cost_v[x] for x in s["visibility"]) + s["start_bits"] for s in symbol_sets]


def window_metrics(w: dict[str, Any]) -> dict[str, dict[str, float]]:
    a, row = w["scores"], w["row"]
    scored = row["scored"]
    out = {}
    for i, name in enumerate(row["variants"]):
        sse = float(a["sse"][i, scored].sum())
        pixels = float(a["pixels_s"][scored].sum()) * 3
        out[name] = {"lpips": float(np.mean(a["lpips"][i, scored])), "dists": float(np.mean(a["dists"][i, scored])),
                     "psnr": 10 * math.log10(255 ** 2 / (sse / pixels)) if sse > 0 else math.inf,
                     "lpips_drawn": float(np.nanmean(a["lpips_drawn"][i, scored])) if np.isfinite(a["lpips_drawn"][i, scored]).any() else math.nan}
    return out


def rate_on_envelope(hull: list[tuple[float, float]], quality: float) -> tuple[float | None, str]:
    """The rate the hull needs for ``quality`` (higher is better), linear between hull points."""
    if quality >= hull[-1][1]:
        return hull[-1][0], "beyond"
    if quality < hull[0][1]:
        return None, "below"
    for (r0, q0), (r1, q1) in zip(hull, hull[1:]):
        if q0 <= quality <= q1:
            return r0 + (r1 - r0) * ((quality - q0) / (q1 - q0) if q1 > q0 else 0.), "on"
    raise AssertionError("unreachable")


def saving(hull: list[tuple[float, float]], quality: float, rate: float) -> dict[str, Any]:
    needed, where = rate_on_envelope(hull, quality)
    value = min(0., 1 - rate / hull[0][0]) if needed is None else 1 - rate / needed
    return {"saving": value, "envelope_kbps": needed, "where": where}


def bootstrap(values: list[float]) -> dict[str, float]:
    spec = DECISION["bootstrap"]
    rng = np.random.default_rng(int(hashlib.sha256(spec["seed"].encode()).hexdigest()[:16], 16))
    v = np.asarray(values, np.float64)
    idx = rng.integers(0, len(v), (spec["resamples"], len(v)))
    means = v[idx].mean(1)
    tail = (1 - spec["level"]) / 2 * 100
    return {"mean": float(v.mean()), "low": float(np.percentile(means, tail)), "high": float(np.percentile(means, 100 - tail)),
            "windows": len(v), "positive": int((v > 0).sum())}


def gate(per_window: dict[str, dict[str, Any]], variant: str, bar: float, metric: str = "lpips") -> dict[str, Any]:
    values = [w["savings"][metric][variant]["saving"] for w in per_window.values()]
    ci = bootstrap(values)
    state = "pass" if ci["low"] > bar else "fail" if ci["high"] < bar else "undecided"
    return {"variant": variant, "metric": metric, "bar": bar, **ci, "state": state}


def outline_table(windows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(HEIGHT_BINS, HEIGHT_BINS[1:]):
        sel = [r for w in windows for r in w["outline"] if r["variant"] == variant and lo <= r["height"] / 1080 < hi]
        region, drawn = sum(r["region_px"] for r in sel), sum(r["drawn_px"] for r in sel)
        rows.append({"bin": [lo, hi], "player_frames": len(sel),
                     "missed_share": sum(r["missed_px"] for r in sel) / region if region else None,
                     "over_share": sum(r["over_px"] for r in sel) / drawn if drawn else None})
    return rows


def fast_limbs(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """`warp` at every 4th frame against every frame, on the scored frames in the top decile of wrist speed (the
    fastest wrist of either player, in region heights per frame) and on the others."""
    every4, every1 = f"warp@{code_name(CHOSEN)}", f"warp@{code_name((4, 'velocity', 1))}"
    diffs, speeds = [], []
    for w in windows:
        row, a = w["row"], w["scores"]
        i4, i1 = row["variants"].index(every4), row["variants"].index(every1)
        for t in row["scored"]:
            best = 0.
            for seq in w["poses"].values():
                cur, prev = seq[t], seq[t - 1] if t > 0 else None
                if cur and prev and cur.get("fitted") and prev.get("fitted"):
                    c, p = np.asarray(cur["kps"]), np.asarray(prev["kps"])
                    h = max(np.ptp(c[c[:, 2] >= k4.KP_CONF, 1]) if (c[:, 2] >= k4.KP_CONF).any() else 1., 1.)
                    for j in (9, 10):
                        if c[j, 2] >= k4.KP_CONF and p[j, 2] >= k4.KP_CONF:
                            best = max(best, float(np.linalg.norm(c[j, :2] - p[j, :2])) / h)
            speeds.append(best)
            diffs.append(float(a["lpips"][i4, t] - a["lpips"][i1, t]))
    if not speeds:
        return {}
    cut = float(np.percentile(speeds, 90))
    top = [d for d, s in zip(diffs, speeds) if s >= cut]
    rest = [d for d, s in zip(diffs, speeds) if s < cut]
    return {"speed_cut": cut, "top_decile_lpips_increase": float(np.mean(top)) if top else None,
            "other_lpips_increase": float(np.mean(rest)) if rest else None, "frames": len(speeds)}


def amendment2(windows: list[dict[str, Any]], k4_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """K4's numbers on K4's windows with (c)+(d)'s players (all 240 frames), K4b.3's code and q 1 previous."""
    k4w = [w for w in windows if w["row"].get("k4_window")]
    if not k4w:
        return {}
    sets: dict[tuple[int, str, int], list[dict[str, Any]]] = {c: [] for c in (CHOSEN, (1, "previous", 1))}
    for w in k4w:
        decoded, symbols = coded_keypoints(code_tracks(w["poses"], list(range(w["row"]["frames"]))))
        for c in sets:
            sets[c].append(symbols[code_name(c)])
    bits = {c: pooled_bits(s) for c, s in sets.items()}
    out = []
    for n, w in enumerate(k4w):
        row = w["row"]
        seconds = row["frames"] / float(row["fps"])
        ball = k4_rows.get(row["k4_window"], {}).get("bits", {})
        r: dict[str, Any] = {"window": row["k4_window"], "players": row["players"], "scored_frames": len(row["scored"])}
        for crf in (48, 62):
            parts = row["anchors"][f"svt-{crf}-s1"]["parts_bits"]
            player, racket = parts["player"] / seconds / 1000, parts["racket"] / seconds / 1000
            ball_kbps = ball.get(str(crf), {}).get("kbps_by_part", {}).get("ball", 0.)
            r[f"crf{crf}"] = {"player_kbps": player, "player_share_of_foreground": player / (player + racket + ball_kbps),
                              "whole_kbps": row["anchors"][f"svt-{crf}-s1"]["kbps"],
                              "k4_whole_kbps": ball.get(str(crf), {}).get("kbps")}
        frames = [x for seq in w["poses"].values() for x in seq if x is not None]
        r["fitted"] = sum(bool(x["fitted"]) for x in frames) / len(frames) if frames else None
        r["fitted_whole_frame"] = sum(bool(x.get("whole_fitted", x["fitted"])) for x in frames) / len(frames) if frames else None
        for c in sets:
            kbps = bits[c][n] / seconds / 1000
            r[code_name(c)] = {"kbps": kbps, "of_players_crf48": kbps / r["crf48"]["player_kbps"],
                               "of_players_crf62": kbps / r["crf62"]["player_kbps"]}
        out.append(r)

    def mean(get: Any) -> float:
        return float(np.mean([get(r) for r in out]))

    return {"windows": out, "means": {
        "player_share_crf48": mean(lambda r: r["crf48"]["player_share_of_foreground"]),
        "player_share_crf62": mean(lambda r: r["crf62"]["player_share_of_foreground"]),
        "fitted": mean(lambda r: r["fitted"]), "fitted_whole_frame": mean(lambda r: r["fitted_whole_frame"]),
        **{f"{code_name(c)}_of_players_crf{crf}": mean(lambda r, c=c, crf=crf: r[code_name(c)][f"of_players_crf{crf}"])
           for c in sets for crf in (48, 62)}}}


def decide(per_window: dict[str, dict[str, Any]], matches: set[str]) -> dict[str, Any]:
    enough = len(per_window) >= 8 and len(matches) >= 4
    bound_bar, charged_bar = DECISION["bound"]["ci_low_above"], DECISION["charged"]["ci_low_above"]
    bounds = {code_name(c): gate(per_window, f"warp@{code_name(c)}", bound_bar) for c in CODES[:3]}
    point = next((name for name, g in bounds.items() if g["state"] == "pass"), None)
    charged = gate(per_window, f"ref@{point}", charged_bar) if point else None
    dists = {v: gate(per_window, v, 0., "dists") for v in ([f"warp@{point}", f"ref@{point}"] if point else [f"warp@{code_name(CHOSEN)}"])}
    if all(g["state"] == "fail" for g in bounds.values()):
        verdict = "no player mode: players stay pixels"
    elif point is None:
        verdict = "undecided (bound)"
    elif charged and charged["state"] == "pass":
        verdict = f"build a player renderer (K6b) at {point}"
    elif charged and charged["state"] == "fail":
        verdict = f"to the user: the bound passes at {point}, the charged oracle fails"
    else:
        verdict = "undecided (charged)"
    lpips_sign = {v: math.copysign(1, gate(per_window, v, 0.)["mean"]) for v in dists}
    sign_clash = [v for v, g in dists.items() if math.copysign(1, g["mean"]) != lpips_sign[v]]
    decided = enough and "undecided" not in verdict
    return {"bound": bounds, "operating_point": point, "charged": charged, "dists_savings": dists,
            "dists_sign_disagrees": sign_clash, "verdict": verdict, "stopping_rule_met": decided,
            "windows": len(per_window), "matches": sorted(matches)}


def command_report(args: argparse.Namespace) -> int:
    windows = load(args.stages)
    k4_rows: dict[str, dict[str, Any]] = {}
    if args.k4_rows:
        for p in sorted(args.k4_rows.glob("broadcast__*.json")):
            r = json.loads(p.read_text())
            k4_rows[r["id"]] = r
    scored = [w for w in windows if w["row"]["scored"]]
    sets = {code_name(c): [w["symbols"][code_name(c)] for w in scored] for c in CODES}
    bits = {name: pooled_bits(s) for name, s in sets.items()}
    per_window: dict[str, dict[str, Any]] = {}
    for n, w in enumerate(scored):
        row = w["row"]
        metrics = window_metrics(w)
        kbps = {name: bits[name][n] / row["scored_seconds"] / 1000 for name in bits}
        anchors = {a: {"kbps": row["anchors"][a]["kbps_r"], **metrics[a]} for a in ANCHOR_NAMES}
        savings: dict[str, dict[str, Any]] = {}
        for metric in ("lpips", "dists"):
            hull = envelope([(p["kbps"], -p[metric]) for p in anchors.values()])
            savings[metric] = {}
            for v in row["variants"][len(ANCHORS):]:
                code_key = v.split("@")[1]
                rate = kbps[code_key] if code_key != "uncoded" else kbps[code_name(CHOSEN)]
                savings[metric][v] = {**saving(hull, -metrics[v][metric], rate), "kbps": rate, metric: metrics[v][metric]}
            savings[metric]["hull"] = {"points": hull}
        per_window[row["id"]] = {"match": row["match"], "k4_window": row.get("k4_window"), "scored_frames": len(row["scored"]),
                                 "players": row["players"], "median_height": row["median_height"], "keypoint_kbps": kbps,
                                 "anchors": anchors, "oracle": {v: metrics[v] for v in row["variants"][len(ANCHORS):]},
                                 "savings": savings, "job": w["stage"], "git_head": w["git_head"]}
    result = {
        "decision_rule": DECISION, "decision": decide(per_window, {v["match"] for v in per_window.values()}),
        "keypoint_kbps_mean": {name: float(np.mean([v["keypoint_kbps"][name] for v in per_window.values()])) for name in bits},
        "outline": {v: outline_table(scored, v) for v in (f"ref@{code_name(CHOSEN)}", "ref@uncoded", f"warp@{code_name(CHOSEN)}")},
        "racket_lpips": {v: float(np.nanmean(np.concatenate([w["scores"]["lpips_racket"][w["row"]["variants"].index(v), w["row"]["scored"]]
                                                             for w in scored])))
                         for v in (f"warp@{code_name(CHOSEN)}", f"ref@{code_name(CHOSEN)}", "svt-62-s1", "dcvc-0-s2")},
        "fast_limbs": fast_limbs(scored), "amendment2": amendment2(windows, k4_rows), "windows": per_window,
        "inputs": {str(s): file_sha256(s / "k6.json") for s in args.stages},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "k6-report.json", result)
    d = result["decision"]
    print(json.dumps({k: d[k] for k in ("verdict", "operating_point", "windows", "matches", "stopping_rule_met")}))
    for name, g in d["bound"].items():
        print(f"bound warp@{name}: mean {g['mean']:.3f} CI [{g['low']:.3f}, {g['high']:.3f}] {g['state']}")
    if d["charged"]:
        g = d["charged"]
        print(f"charged {g['variant']}: mean {g['mean']:.3f} CI [{g['low']:.3f}, {g['high']:.3f}] {g['state']}")
    return 0
