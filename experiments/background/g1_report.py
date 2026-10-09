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


#: Paper style: one colour per verdict, the dataset order of the tables.
HOLD, FAIL = "#2f7d4f", "#b3362f"
COVER_TICKS = [0, 1, 2, 5, 10, 30, 120]
MIN_LATER = 10  # frames after the warm-up for a point of C(w)


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    return plt


def _save(fig: Any, out: Path, stem: str) -> list[str]:
    names = []
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
        names.append(f"{stem}.{ext}")
    return names


def figures(rows: list[dict[str, Any]], frames: dict[str, list[dict[str, Any]]], out: Path,
            published: list[str] | None = None) -> list[str]:
    """Paper figures from the recorded results: vector PDF and PNG, plus sample overlays.

    ``g1-coverage``: C(w) per clip, one panel per racket dataset (the rotation
    model never holds on VISOR, so its curves are not drawn), green where the
    clip holds, grey where it does not. ``g1-residual``: each clip's median per-frame p90 residual flow (log
    scale) against the 2 px rule. ``g1-residual-frames``: the per-frame
    distribution per dataset. ``figures/overlay_<dataset>.jpg``: one overlay
    sheet per dataset, picked by hash, not by look.
    """
    import hashlib
    import shutil

    plt = _style()
    names: list[str] = []
    groups = [g for g in GROUPS if any(r["group"] == g for r in rows)]
    scale = lambda w: np.log10(1 + np.asarray(w, float))  # noqa: E731
    # Coverage is drawn only where the rotation model can hold (VISOR never does), failing clips in grey,
    # and each curve stops where fewer than MIN_LATER frames remain after the warm-up.
    covered = [g for g in groups if not g.startswith("visor")] or groups
    fig, axes = plt.subplots(1, len(covered), figsize=(7.0 * len(covered) / 5 + 1.4, 1.7), sharey=True)
    for ax, group in zip(np.atleast_1d(axes), covered):
        for r in sorted((r for r in rows if r["group"] == group), key=lambda r: r["summary"]["holds"]):
            later = np.array(r["coverage"]["frames_after"], float)
            w = np.array(r["coverage"]["warmups_s"], float)[later >= MIN_LATER]
            c = np.array([np.nan if v is None else v for v in r["coverage"]["curve"]], float)[later >= MIN_LATER]
            holds = r["summary"]["holds"]
            ax.plot(scale(w), c, color=HOLD if holds else "0.7", alpha=0.7 if holds else 0.6, lw=0.8)
        for level in (0.9, 0.99):
            ax.axhline(level, color="0.5", lw=0.5, ls="--")
        ax.set_xticks(scale(COVER_TICKS), [str(t) for t in COVER_TICKS])
        ax.set_ylim(0.85, 1.002)
        ax.set_title(LABELS[group])
        ax.set_xlabel("warm-up (s)")
    np.atleast_1d(axes)[0].set_ylabel("background seen")
    names += _save(fig, out, "g1-coverage")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(groups) + 0.6))
    for i, group in enumerate(groups):
        for r in (r for r in rows if r["group"] == group):
            v = r["summary"]["f2r_rot"]["flow_p90_px_median"]
            if v is None:
                continue
            jitter = (int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000 / 1000 - 0.5) * 0.5
            ax.scatter(max(v, 0.1), i + jitter, s=10, color=HOLD if r["summary"]["holds"] else FAIL, lw=0)
    ax.axvline(DECISION["explained_px"], color="0.3", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(groups)), [LABELS[g] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("median per-frame p90 residual flow after the rotation-and-zoom warp (px at 1080p)")
    names += _save(fig, out, "g1-residual")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    for i, group in enumerate(groups):
        values = [f["f2r_rot"]["flow_p90_px"] for r in rows if r["group"] == group for f in frames.get(r["id"], [])
                  if f.get("f2r_rot", {}).get("measured")]
        if values:
            parts = ax.violinplot([np.log10(np.maximum(values, 0.05))], positions=[i], showmedians=True, widths=0.8)
            for body in parts["bodies"]:
                body.set_facecolor("0.6")
    ax.axhline(np.log10(DECISION["explained_px"]), color="0.3", lw=0.7, ls="--")
    ax.set_xticks(range(len(groups)), [LABELS[g] for g in groups])
    ax.set_ylabel("log10 p90 residual (px)")
    names += _save(fig, out, "g1-residual-frames")
    plt.close(fig)

    sheets = out / "figures"
    sheets.mkdir(exist_ok=True)
    for group in groups:
        ids = sorted((r["id"] for r in rows if r["group"] == group),
                     key=lambda i: hashlib.sha256(f"g1-paper:{i}".encode()).hexdigest())
        for clip_id in ids:
            images = [i for i in next(r for r in rows if r["id"] == clip_id)["images"] if i.startswith("overlay_")]
            source = next((Path(root) / "publish" / "clips" / safe(clip_id) / images[len(images) // 2]
                           for root in (published or []) if images
                           and (Path(root) / "publish" / "clips" / safe(clip_id) / images[len(images) // 2]).is_file()), None)
            if source is not None:
                shutil.copyfile(source, sheets / f"overlay_{group}.jpg")
                names.append(f"figures/overlay_{group}.jpg ({clip_id})")
                break
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
    report["figures"] = figures(rows, frames, out, args.published)
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
