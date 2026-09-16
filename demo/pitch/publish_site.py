"""Assemble the public PointStream demo from a single source of truth.

Tracked inputs
--------------
- ``demo/outputs/results/comparison_results.json`` — numbers (the only committed
  experiment artifact; the rest of ``results/`` is local scratch).
- ``demo/pitch/interactive_demo.html`` and ``interactive_report.html`` — markup.
- ``demo/outputs/pitch/*.mp4`` and ``keypoints_*.json`` — media that CI cannot
  regenerate.

Generated outputs (do not edit by hand; ``demo/.gitignore`` excludes them)
------------------------------------------------------------------------
- ``index.html``, ``index_static_report.html``
- ``rd_curves_*.png``, ``latency_profile.png``

The GitHub Pages workflow runs this script, then uploads the assembled folder
to https://emanueleartioli.com/pointstream/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.experiments.plot_rd_curves import generate_plots

PITCH_SRC = REPO_ROOT / "demo" / "pitch"
MEDIA_DIR = REPO_ROOT / "demo" / "outputs" / "pitch"
RESULTS_JSON = REPO_ROOT / "demo" / "outputs" / "results" / "comparison_results.json"


def _bust(html: str, token: str) -> str:
    for suffix in (".mp4", ".png", ".json"):
        html = html.replace(f'{suffix}"', f'{suffix}?v={token}"')
        html = html.replace(f"{suffix}'", f"{suffix}?v={token}'")
    return html


def publish(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GITHUB_SHA", "")[:12] or "local"

    demo_html = (PITCH_SRC / "interactive_demo.html").read_text(encoding="utf-8")
    report_html = (PITCH_SRC / "interactive_report.html").read_text(encoding="utf-8")
    (out_dir / "index.html").write_text(_bust(demo_html, token), encoding="utf-8")
    (out_dir / "index_static_report.html").write_text(_bust(report_html, token), encoding="utf-8")

    generate_plots(json_path=RESULTS_JSON, out_dir=out_dir)

    if out_dir.resolve() != MEDIA_DIR.resolve():
        for pattern in ("*.mp4", "keypoints_*.json"):
            for src in MEDIA_DIR.glob(pattern):
                shutil.copy2(src, out_dir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MEDIA_DIR)
    args = parser.parse_args()
    publish(args.out)


if __name__ == "__main__":
    main()
