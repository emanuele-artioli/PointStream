"""Render the ladder's JSON as the table the report quotes.

Reads whatever `outputs/bp24-ladder/*.json` the runs produced and prints one
markdown block per axis. Kept separate from `ladder.py` so re-reading a finished
run costs nothing, and so the shape of the table cannot influence the shape of
the measurement.

It prints the alarms and the exclusions **above** the numbers on purpose. A
table whose caveats are underneath it is a table whose caveats get cropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp24-ladder"


def _rows(pair: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return sorted(pair.get(key, []), key=lambda row: row["coded_bytes"])


def render(path: Path) -> str:
    payload = json.loads(path.read_text())
    clip = payload.get("clip", {})
    lines: list[str] = []
    lines.append(f"### `{path.name}`")
    lines.append("")
    lines.append(
        f"{clip.get('video')}/{clip.get('scene')} · {clip.get('resolution')} · "
        f"{clip.get('n_frames')} frames · source {clip.get('source_bytes'):,} B · "
        f"inter-frame MAD {payload.get('clip_motion_mad', float('nan')):.2f} · "
        f"sweep `{payload.get('sweep')}` · tier `{payload.get('tier')}`"
    )
    lines.append("")

    for pair in payload.get("pairs", []):
        codec = pair["codec"]
        lines.append(
            f"**{codec}**, preset `{pair['preset']}`, "
            f"{pair['rate_control']}, {pair['pix_fmt']} — on both arms."
        )
        lines.append("")
        for alarm in pair.get("bound_alarms", []):
            lines.append(f"> **ALARM** {alarm}")
        for excluded in pair.get("rungs_excluded_not_a_rate", []):
            lines.append(
                f"> **excluded, not a rate**: "
                f"{excluded.get('rung') or excluded.get('coarseness') or excluded['rate_value']}"
                f" — raw parts {excluded.get('raw_parts')}"
            )
        for failure in pair.get("failures", []):
            lines.append(f"> **failed**: {failure['arm']} at {failure['rate_value']} — {failure['error']}")
        if pair.get("bound_alarms") or pair.get("rungs_excluded_not_a_rate") or pair.get("failures"):
            lines.append("")

        lines.append("| arm | rung | coded bytes | Y-PSNR dB | RGB-PSNR dB | s |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in _rows(pair, "anchor_rungs"):
            lines.append(
                f"| {codec} on source | QP {row['rate_value']} | "
                f"{row['coded_bytes']:,} | {row['psnr_dB']:.2f} | "
                f"{row.get('psnr_rgb_dB', float('nan')):.2f} | {row['seconds']:.0f} |"
            )
        for row in _rows(pair, "pointstream_rungs"):
            label = row.get("rung") or row.get("coarseness") or f"rank {row['rate_value']}"
            lines.append(
                f"| PointStream via {codec} | {label} | "
                f"{row['coded_bytes']:,} | {row['psnr_dB']:.2f} | "
                f"{row.get('psnr_rgb_dB', float('nan')):.2f} | {row['seconds']:.0f} |"
            )
        lines.append("")

        if pair.get("bd_rate") is None:
            lines.append(f"**No BD-rate.** {pair.get('blocked_by')}")
        else:
            lines.append(
                f"**BD-rate {pair['bd_rate_percent']:+.1f}%** over "
                f"{pair['overlap_dB'][0]:.2f}-{pair['overlap_dB'][1]:.2f} dB "
                f"(overlap {pair['overlap_fraction']:.0%}), "
                f"BD-quality {pair['bd_quality_dB']:+.2f} dB. "
                "Negative rate is PointStream winning."
            )
        lines.append("")

        # What the payload was made of at the cheapest and dearest rung. The
        # split is the finding, not a footnote: on static content the plate has
        # been most of it, which is why the rungs move the plate.
        stream = _rows(pair, "pointstream_rungs")
        if stream:
            for row, where in ((stream[0], "cheapest"), (stream[-1], "dearest")):
                parts = row.get("parts", {})
                total = max(1, sum(parts.values()))
                split = ", ".join(
                    f"{name} {value:,} B ({value / total:.0%})"
                    for name, value in parts.items()
                )
                lines.append(f"- payload at the {where} rung: {split}")
            lines.append("")

    lines.append(
        "*Gains are stated beside their preset and are **not** ranked against "
        "another codec's: the presets are not equal effort "
        "(`plans/BP24-findings.md` §1).*"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None)
    args = parser.parse_args(argv)

    paths = (
        [Path(item) for item in args.paths]
        if args.paths
        else sorted(
            path
            for path in OUT_DIR.glob("*.json")
            if path.name not in {"bounds-before-run.json", "motion-survey.json", "appearance-cost.json"}
        )
    )
    for path in paths:
        print(render(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
