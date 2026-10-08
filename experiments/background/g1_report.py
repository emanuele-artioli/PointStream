"""PLAN step G1 report: the decision rule applied per dataset, with tables and figures.

    python -m experiments.background.g1 report --result g1.json ... --published DIR ... --out DIR

``--result`` are the runs' ``g1.json``; ``--published`` their extracted
``published.tar`` (per-frame records, for the figures). Writes ``g1-report.json``,
``g1-report.md`` and PNG figures to ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background.g1 import DECISION, safe
from experiments.visor.b1 import file_sha256, write_json

GROUPS = ("visor-long", "visor-window", "ott", "racketvision", "tracknet")
LABELS = {"visor-long": "VISOR stretches", "visor-window": "VISOR windows", "ott": "OpenTTGames",
          "racketvision": "RacketVision", "tracknet": "TrackNet"}


def med(values: list[Any]) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.median(values)), 4) if values else None


def span(values: list[Any]) -> list[float] | None:
    values = [v for v in values if v is not None]
    return [round(float(min(values)), 3), round(float(max(values)), 3)] if values else None


def translation_confirmed(summary: dict[str, Any]) -> bool | None:
    t = summary["translation_half_s"]
    if not t["pairs"]:
        return None
    return bool(t["prefers_f_share"] >= DECISION["translation_prefers_f_share"]
                and (t["parallax_p50_px_median"] or 0) >= DECISION["translation_parallax_px"])


def verdict(share: float) -> str:
    if share >= DECISION["dataset_holds_share"]:
        return "holds"
    if share >= DECISION["dataset_subset_share"]:
        return "holds for a subset"
    return "does not hold"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = [r["summary"] for r in rows]
    holding = [r for r in rows if r["summary"]["holds"]]
    share = len(holding) / len(rows) if rows else 0.0
    energy = [s["energy_share_mean"] for s in sums if s["energy_share_mean"]]
    confirmed = [translation_confirmed(s) for s in sums]
    tiers = [r["tier_check"] for r in rows if r.get("tier_check")]
    return {
        "clips": len(rows), "hold": len(holding), "hold_share": round(share, 4), "verdict": verdict(share),
        "explained_share_median": med([s["f2r_rot"]["explained_share"] for s in sums]),
        "explained_share_range": span([s["f2r_rot"]["explained_share"] for s in sums]),
        "flow_p90_px_median": med([s["f2r_rot"]["flow_p90_px_median"] for s in sums]),
        "f2f_h_explained_median": med([s["f2f_h"]["explained_share"] for s in sums]),
        "f2r_h_explained_median": med([s["f2r_h"]["explained_share"] for s in sums]),
        "within_1px_median": med([s["f2r_rot"]["within_1px_median"] for s in sums]),
        "within_4px_median": med([s["f2r_rot"]["within_4px_median"] for s in sums]),
        "psnr_gain_median": med([s["f2r_rot"]["psnr_gain_median"] for s in sums]),
        "direct_share_median": med([s["direct_share"] for s in sums]),
        "segments_gt_1": sum(1 for s in sums if s["segments"] > 1),
        "energy_share_mean": {k: round(float(np.mean([e[k] for e in energy])), 4) for k in energy[0]} if energy else None,
        "translation": {
            "confirmed": sum(1 for c in confirmed if c), "measured": sum(1 for c in confirmed if c is not None),
            "prefers_f_share_median": med([s["translation_half_s"]["prefers_f_share"] for s in sums]),
            "parallax_p50_px_median": med([s["translation_half_s"]["parallax_p50_px_median"] for s in sums]),
        },
        "speed_deg_s_median": med([s["speed_deg_s"]["median"] for s in sums]),
        "residual_vs_speed_spearman_median": med([s["residual_vs_speed_spearman"] for s in sums]),
        "lens": {"observable": sum(1 for s in sums if s["lens"]["observable"]),
                 "f_1080_median": med([s["lens"]["f_1080"] for s in sums if s["lens"]["observable"]]),
                 "k1_median": med([s["lens"]["k1"] for s in sums if s["lens"]["observable"]]),
                 "hfov_deg_median": med([s["lens"]["hfov_deg"] for s in sums if s["lens"]["observable"]])},
        "radius_profile_px": [med([s["flow_p50_by_radius_px"][i] for s in sums]) for i in range(5)],
        "blur": {"explained_blurred_median": med([s["blur"]["explained_blurred"] for s in sums]),
                 "explained_sharp_median": med([s["blur"]["explained_sharp"] for s in sums])},
        "fg_share_mean": med([s["fg_share_mean"] for s in sums]),
        "duration_s_median": med([s["duration_s"] for s in sums]),
        "warmup_90_s": {"holding_median": med([r["summary"]["warmup_90_s"] for r in holding]),
                        "holding_range": span([r["summary"]["warmup_90_s"] for r in holding]),
                        "all_median": med([s["warmup_90_s"] for s in sums]),
                        "not_reached": sum(1 for s in sums if s["warmup_90_s"] is None)},
        "warmup_99_s": {"holding_median": med([r["summary"]["warmup_99_s"] for r in holding]),
                        "holding_range": span([r["summary"]["warmup_99_s"] for r in holding]),
                        "all_median": med([s["warmup_99_s"] for s in sums]),
                        "not_reached": sum(1 for s in sums if s["warmup_99_s"] is None)},
        "tier_check": {
            "clips": len(tiers),
            "dense_recall_mean": med([(t.get("dense_recall") or {}).get("mean") for t in tiers]),
            "explained_sam_median": med([t["explained_sam"] for t in tiers]),
            "explained_dense_median": med([t["explained_dense"] for t in tiers]),
            "flow_p90_sam_median": med([t["flow_p90_sam_median"] for t in tiers]),
            "flow_p90_dense_median": med([t["flow_p90_dense_median"] for t in tiers]),
        } if tiers else None,
        "holding_clips": [r["id"] for r in holding],
        "g2_usable": [r["id"] for r in rows if r["summary"]["g2_usable"]],
    }


def start_decision(groups: dict[str, Any]) -> dict[str, Any]:
    visor = groups.get("visor-long")
    racket = {g: groups[g] for g in ("ott", "racketvision", "tracknet") if g in groups}
    if visor and (visor["verdict"] == "holds" or (
            visor["verdict"] == "holds for a subset" and visor["hold"] >= DECISION["start_visor_min_clips"]
            and (visor["warmup_90_s"]["holding_median"] or 1e9) <= DECISION["start_visor_max_warmup90_s"])):
        return {"start": "visor", "reason": f"VISOR stretches: {visor['verdict']} ({visor['hold']}/{visor['clips']})"}
    holding = [g for g, a in racket.items() if a["verdict"] == "holds"]
    if holding:
        return {"start": "racket sports", "datasets": holding,
                "reason": "VISOR stretches " + (visor["verdict"] if visor else "not measured")
                          + f"; racket datasets that hold: {', '.join(LABELS[g] for g in holding)}"}
    return {"start": "neither: G5 leads", "reason": "no dataset holds"}


def figures(rows: list[dict[str, Any]], frames: dict[str, list[dict[str, Any]]], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    colours = {"visor-long": "#c2410c", "visor-window": "#f59e0b", "ott": "#0e7490", "racketvision": "#4338ca",
               "tracknet": "#15803d"}
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
    for ax, group in zip(axes, GROUPS):
        sel = [r for r in rows if r["group"] == group]
        for r in sel:
            w = np.array(r["coverage"]["warmups_s"], float)
            c = np.array([np.nan if v is None else v for v in r["coverage"]["curve"]], float)
            ax.plot(w, c, color=colours[group], alpha=0.35, lw=1)
        ax.axhline(0.9, color="grey", lw=0.6, ls="--")
        ax.axhline(0.99, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{LABELS[group]} ({len(sel)})")
        ax.set_xlabel("warm-up w (s)")
        ax.set_xscale("symlog", linthresh=1)
    axes[0].set_ylabel("C(w): later background already seen")
    fig.tight_layout()
    fig.savefig(out / "g1-coverage.png", dpi=110)
    plt.close(fig)
    names.append("g1-coverage.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, group in enumerate(GROUPS):
        values = [f["f2r_rot"]["flow_p90_px"] for r in rows if r["group"] == group for f in frames.get(r["id"], [])
                  if f.get("f2r_rot", {}).get("measured")]
        if values:
            parts = ax.violinplot([np.log10(np.maximum(values, 0.05))], positions=[i], showmedians=True, widths=0.8)
            for body in parts["bodies"]:
                body.set_facecolor(colours[group])
    ax.axhline(np.log10(DECISION["explained_px"]), color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(GROUPS)), [LABELS[g] for g in GROUPS])
    ax.set_ylabel("log10 p90 residual flow (px @1080p)")
    ax.set_title("Rotation-and-zoom model, frame to reference: per-frame residual")
    fig.tight_layout()
    fig.savefig(out / "g1-residual.png", dpi=110)
    plt.close(fig)
    names.append("g1-residual.png")
    return names


def command_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for path in args.result:
        doc = json.loads(Path(path).read_text())
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "clips_file": doc["clips_file"],
                     "self_test": doc["self_test"], "sam": {k: v for k, v in doc["sam"].items() if k != "clips"},
                     "sam_clips": doc["sam"].get("clips", {})})
        rows.extend(doc["results"])
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:  # a clip run twice (pilot and full) counts once: the last run given wins
        seen[r["id"]] = r
    rows = list(seen.values())
    frames: dict[str, list[dict[str, Any]]] = {}
    for root in args.published:
        for r in rows:
            path = Path(root) / "publish" / "clips" / safe(r["id"]) / "result.json"
            if path.is_file():
                frames[r["id"]] = json.loads(path.read_text())["frames"]
    groups = {g: aggregate([r for r in rows if r["group"] == g]) for g in GROUPS if any(r["group"] == g for r in rows)}
    sports = {s: aggregate([r for r in rows if r["group"] == "racketvision" and r["clip"].get("sport") == s])
              for s in sorted({r["clip"].get("sport") for r in rows if r["group"] == "racketvision"})}
    report = {"decision_rule": DECISION, "runs": runs, "groups": groups, "racketvision_by_sport": sports,
              "start": start_decision(groups),
              "clips": [{"id": r["id"], "group": r["group"], **{k: r["summary"][k] for k in (
                  "holds", "g2_usable", "duration_s", "direct_share", "measured_share", "segments", "warmup_90_s",
                  "warmup_99_s", "fg_share_mean")},
                  "explained_share": r["summary"]["f2r_rot"]["explained_share"],
                  "flow_p90_px_median": r["summary"]["f2r_rot"]["flow_p90_px_median"],
                  "translation_confirmed": translation_confirmed(r["summary"]),
                  "parallax_p50_px": r["summary"]["translation_half_s"]["parallax_p50_px_median"],
                  "images": r["images"]} for r in rows]}
    report["figures"] = figures(rows, frames, out)
    write_json(out / "g1-report.json", report)
    lines = ["| Dataset | clips | hold | verdict | explained (median) | p90 flow px | translation confirmed | "
             "warm-up 90% s (holding) | warm-up 99% s (holding) | G2 clips |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g, a in groups.items():
        lines.append(f"| {LABELS[g]} | {a['clips']} | {a['hold']} | {a['verdict']} | {a['explained_share_median']} | "
                     f"{a['flow_p90_px_median']} | {a['translation']['confirmed']}/{a['translation']['measured']} | "
                     f"{a['warmup_90_s']['holding_median']} {a['warmup_90_s']['holding_range']} | "
                     f"{a['warmup_99_s']['holding_median']} {a['warmup_99_s']['holding_range']} | {len(a['g2_usable'])} |")
    lines.append("")
    lines.append(f"Start: {report['start']}")
    (out / "g1-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0
