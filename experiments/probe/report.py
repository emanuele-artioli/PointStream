"""Read a probe run directory and produce the table it is safe to quote from.

A ranked list with a standard error on each row still invites "A beats B" from
a reader, and adjacent rows are usually not separable. So this reports every
comparison **paired on the same clips**, with n and a standard error, and it
lets ``compare_paired`` refuse a direction the sample cannot support.

**The unit is the clip, not the frame.** Eight offsets inside one clip share a
keyframe, a player and a tracking box, so they are not eight independent
observations; treating them as n=96 would shrink every standard error by about
three. Each engine's per-clip mean over its offsets is one item, n=12.

Every number is printed beside the two anchors measured in the same run: the
static-copy floor and the unrelated-image null. "0.067" means nothing; "0.067,
where an unrelated image scores 0.645" means something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.probe.engines import BASELINES, STATIC_COPY, UNRELATED_IMAGE
from src.components.metrics.comparison import PairedComparison, compare_paired

RANKING_METRIC = "object_lpips"


@dataclass(frozen=True)
class EngineRows:
    engine: str
    drive_mode: str
    refused: bool
    refuse_reason: str | None
    checkpoint_epoch: Any
    per_clip_lpips: dict[str, float]
    per_clip_psnr: dict[str, float]
    mean_wall_s: float | None
    peak_vram_bytes: int | None

    @property
    def mean_lpips(self) -> float | None:
        return _mean(list(self.per_clip_lpips.values()))

    @property
    def mean_psnr(self) -> float | None:
        return _mean(list(self.per_clip_psnr.values()))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_run(out_dir: Path) -> dict[str, EngineRows]:
    """Every ``<engine>.json`` in ``out_dir``, reduced to per-clip means."""
    loaded: dict[str, EngineRows] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"summary.json"} or path.name.startswith("cross-appearance"):
            continue
        payload = json.loads(path.read_text())
        engine = payload.get("engine")
        if not engine:
            continue
        lpips_by_clip: dict[str, list[float]] = {}
        psnr_by_clip: dict[str, list[float]] = {}
        for row in payload.get("clips", []):
            if row.get("error"):
                continue
            key = str(row["clip_key"])
            if isinstance(row.get("object_lpips"), (int, float)):
                lpips_by_clip.setdefault(key, []).append(float(row["object_lpips"]))
            value = row.get("object_psnr_db")
            if isinstance(value, (int, float)):
                psnr_by_clip.setdefault(key, []).append(float(value))
        headline = payload.get("headline") or {}
        loaded[engine] = EngineRows(
            engine=engine,
            drive_mode=payload.get("drive_mode", "frame"),
            refused=bool(payload.get("refused")),
            refuse_reason=payload.get("refuse_reason"),
            checkpoint_epoch=payload.get("checkpoint_epoch"),
            per_clip_lpips={
                key: sum(values) / len(values) for key, values in lpips_by_clip.items()
            },
            per_clip_psnr={
                key: sum(values) / len(values) for key, values in psnr_by_clip.items()
            },
            mean_wall_s=headline.get("mean_wall_s"),
            peak_vram_bytes=headline.get("peak_vram_bytes"),
        )
    return loaded


def paired_on_shared_clips(
    a: EngineRows, b: EngineRows, *, metric: str = RANKING_METRIC
) -> PairedComparison | None:
    """Compare two arms on the clips both completed. None if fewer than two."""
    left = a.per_clip_lpips if metric == RANKING_METRIC else a.per_clip_psnr
    right = b.per_clip_lpips if metric == RANKING_METRIC else b.per_clip_psnr
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return None
    return compare_paired(
        a.engine,
        [left[key] for key in shared],
        b.engine,
        [right[key] for key in shared],
        higher_is_better=metric != RANKING_METRIC,
    )


@dataclass(frozen=True)
class CrossRows:
    engine: str
    donor_mode: str
    drive_mode: str
    status: str
    note: str
    per_clip_lpips_delta: dict[str, float]
    per_clip_psnr_delta: dict[str, float]


def load_cross(out_dir: Path) -> list[CrossRows]:
    """Every ``cross-appearance-*.json`` in ``out_dir``, keyed per clip."""
    rows: list[CrossRows] = []
    for path in sorted(out_dir.glob("cross-appearance-*.json")):
        payload = json.loads(path.read_text())
        verdict = payload.get("verdict") or {}
        lpips: dict[str, float] = {}
        psnr: dict[str, float] = {}
        for pair in payload.get("pairs", []):
            if pair.get("error"):
                continue
            key = str(pair["clip_key"])
            if isinstance(pair.get("delta_lpips"), (int, float)):
                lpips[key] = float(pair["delta_lpips"])
            if isinstance(pair.get("delta_psnr_db"), (int, float)):
                psnr[key] = float(pair["delta_psnr_db"])
        rows.append(
            CrossRows(
                engine=str(payload["engine"]),
                donor_mode=payload.get("donor_mode", "different-video"),
                drive_mode=payload.get("drive_mode", "frame"),
                status=str(verdict.get("status", "no verdict")),
                note=str(verdict.get("note", "")),
                per_clip_lpips_delta=lpips,
                per_clip_psnr_delta=psnr,
            )
        )
    return rows


def cross_report(out_dir: Path) -> dict[str, Any]:
    """Cross-appearance deltas, and every engine paired against every other.

    Pairing the *deltas* is the comparison that matters. "This engine's
    appearance pathway is stronger than that one's" is a claim about the
    difference of two differences, and the clips are shared, so it is a paired
    comparison and gets a standard error like any other.
    """
    rows = load_cross(out_dir)
    report: dict[str, Any] = {"arms": [], "between": []}
    for row in rows:
        values = list(row.per_clip_lpips_delta.values())
        psnr_values = list(row.per_clip_psnr_delta.values())
        report["arms"].append(
            {
                "engine": row.engine,
                "donor_mode": row.donor_mode,
                "drive_mode": row.drive_mode,
                "n_clips": len(values),
                "delta_lpips": _mean(values),
                "delta_psnr_db": _mean(psnr_values),
                "status": row.status,
                "note": row.note,
            }
        )
    for first, second in _ordered_pairs(rows):
        shared = sorted(set(first.per_clip_lpips_delta) & set(second.per_clip_lpips_delta))
        if len(shared) < 2:
            continue
        comparison = compare_paired(
            f"{first.engine}[{first.donor_mode}]",
            [first.per_clip_lpips_delta[key] for key in shared],
            f"{second.engine}[{second.donor_mode}]",
            [second.per_clip_lpips_delta[key] for key in shared],
            higher_is_better=True,
        )
        report["between"].append(
            {"describe": comparison.describe(), "verdict": comparison.verdict,
             "winner": comparison.winner}
        )
    return report


def _ordered_pairs(rows: list[CrossRows]) -> list[tuple[CrossRows, CrossRows]]:
    ordered = sorted(
        rows,
        key=lambda row: -(_mean(list(row.per_clip_lpips_delta.values())) or 0.0),
    )
    return [(ordered[i], ordered[j])
            for i in range(len(ordered))
            for j in range(i + 1, len(ordered))]


def build_report(out_dir: Path) -> dict[str, Any]:
    """The ranked table, its anchors, and every comparison with its uncertainty."""
    loaded = load_run(out_dir)
    floor = loaded.get(STATIC_COPY)
    null = loaded.get(UNRELATED_IMAGE)
    ranked = sorted(
        (
            rows
            for name, rows in loaded.items()
            if name not in BASELINES and not rows.refused and rows.mean_lpips is not None
        ),
        key=lambda rows: rows.mean_lpips,  # type: ignore[arg-type,return-value]
    )
    report: dict[str, Any] = {
        "out_dir": str(out_dir),
        "unit": "one clip = one item; per-clip mean over the run's offsets",
        "ranking_metric": RANKING_METRIC,
        "ranking_lower_is_better": True,
        "anchors": {
            "static_copy_lpips": floor.mean_lpips if floor else None,
            "unrelated_image_lpips": null.mean_lpips if null else None,
            "static_copy_psnr_db": floor.mean_psnr if floor else None,
            "unrelated_image_psnr_db": null.mean_psnr if null else None,
            "published_unrelated_lpips": 0.645,
            "published_heavy_blur_lpips": 0.430,
        },
        "rank": [rows.engine for rows in ranked],
        "rows": [],
        "vs_floor": [],
        "adjacent": [],
        "refused": {
            name: rows.refuse_reason
            for name, rows in sorted(loaded.items())
            if rows.refused
        },
        "cross_appearance": cross_report(out_dir),
    }
    for rows in ranked:
        report["rows"].append(
            {
                "engine": rows.engine,
                "drive_mode": rows.drive_mode,
                "n_clips": len(rows.per_clip_lpips),
                "object_lpips": rows.mean_lpips,
                "object_psnr_db": rows.mean_psnr,
                "mean_wall_s": rows.mean_wall_s,
                "peak_vram_bytes": rows.peak_vram_bytes,
                "checkpoint_epoch": rows.checkpoint_epoch,
            }
        )
        if floor is not None:
            comparison = paired_on_shared_clips(rows, floor)
            if comparison is not None:
                report["vs_floor"].append(
                    {"engine": rows.engine, "describe": comparison.describe(),
                     "delta": comparison.mean_difference,
                     "standard_error": comparison.standard_error,
                     "verdict": comparison.verdict,
                     "winner": comparison.winner}
                )
    for first, second in zip(ranked, ranked[1:]):
        comparison = paired_on_shared_clips(first, second)
        if comparison is not None:
            report["adjacent"].append(
                {"pair": f"{first.engine} vs {second.engine}",
                 "describe": comparison.describe(),
                 "verdict": comparison.verdict,
                 "winner": comparison.winner}
            )
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """A table a human can read, with both anchors on every line of context."""
    anchors = report["anchors"]
    floor = anchors["static_copy_lpips"]
    null = anchors["unrelated_image_lpips"]
    lines = [
        f"Probe run: {report['out_dir']}",
        f"Unit: {report['unit']}",
        "Scope: PSNR over the letterboxed player MASK; LPIPS over the BOUNDING BOX",
        "       of that mask, because LPIPS is a patch metric and cannot take a",
        "       mask. Every arm uses the same scope, so each column compares; the",
        "       two columns are not the same region as each other.",
        "",
        "Anchors measured in this run:",
        f"  static copy (right player, wrong pose) : LPIPS {_f(floor)}  PSNR {_f(anchors['static_copy_psnr_db'], 2)} dB",
        f"  unrelated image (wrong player)         : LPIPS {_f(null)}  PSNR {_f(anchors['unrelated_image_psnr_db'], 2)} dB",
        f"  published reference: heavy blur {anchors['published_heavy_blur_lpips']}, "
        f"unrelated {anchors['published_unrelated_lpips']}",
        "",
        f"{'engine':<24} {'mode':<6} {'n':>3} {'LPIPS':>8} {'PSNR dB':>8} {'s/frame':>8}",
        "-" * 62,
    ]
    for row in report["rows"]:
        lines.append(
            f"{row['engine']:<24} {row['drive_mode']:<6} {row['n_clips']:>3} "
            f"{_f(row['object_lpips']):>8} {_f(row['object_psnr_db'], 2):>8} "
            f"{_f(row['mean_wall_s'], 2):>8}"
        )
    lines += ["", "Against the static-copy floor (paired on clips):"]
    for item in report["vs_floor"]:
        lines.append(f"  {item['describe']}")
    lines += ["", "Adjacent ranks (paired on clips):"]
    for item in report["adjacent"] or []:
        lines.append(f"  {item['describe']}")
    if not report["adjacent"]:
        lines.append("  (fewer than two ranked engines)")
    cross = report.get("cross_appearance") or {}
    if cross.get("arms"):
        lines += [
            "",
            "Cross-appearance: what the WRONG keyframe costs (higher = more use of appearance)",
            f"  scale: a paste of the right player instead of the wrong one is worth "
            f"{_f(null - floor) if floor and null else 'n/a'} LPIPS",
            "",
            f"{'engine':<24} {'donors':<16} {'n':>3} {'dLPIPS':>8} {'dPSNR':>7}  verdict",
            "-" * 78,
        ]
        for arm in cross["arms"]:
            lines.append(
                f"{arm['engine']:<24} {arm['donor_mode']:<16} {arm['n_clips']:>3} "
                f"{_f(arm['delta_lpips'], 3):>8} {_f(arm['delta_psnr_db'], 2):>7}  {arm['status']}"
            )
        if cross.get("between"):
            lines += ["", "  Between arms (paired on clips, on the deltas themselves):"]
            for item in cross["between"]:
                lines.append(f"    {item['describe']}")
    if report["refused"]:
        lines += ["", "Refused:"]
        for engine, reason in report["refused"].items():
            lines.append(f"  {engine}: {(reason or '')[:110]}")
    return "\n".join(lines)


def _f(value: Any, places: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{places}f}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit the record, not the table.")
    args = parser.parse_args(argv)
    report = build_report(args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
