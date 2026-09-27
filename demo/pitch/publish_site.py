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
import json
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
MAPS_SRC = REPO_ROOT / "demo" / "outputs" / "maps"
PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}


def _bust(html: str, token: str) -> str:
    for suffix in (".mp4", ".webm", ".png", ".json"):
        html = html.replace(f'{suffix}"', f'{suffix}?v={token}"')
        html = html.replace(f"{suffix}'", f"{suffix}?v={token}'")
    return html


def copy_maps_gallery(out_dir: Path, maps_src: Path | None = None) -> None:
    """Copy index.json and preview media only — never native payloads or weights."""
    src = maps_src if maps_src is not None else MAPS_SRC
    index_path = src / "index.json"
    if not index_path.is_file():
        return
    dest_root = out_dir / "maps"
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, dest_root / "index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    copied: set[Path] = set()
    for entry in index.get("entries") or []:
        overlay = entry.get("overlay_url")
        if overlay:
            preview = src / overlay
            if preview.is_file() and preview.suffix.lower() in PREVIEW_SUFFIXES:
                dest = dest_root / overlay
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(preview, dest)
                copied.add(dest)
            continue
        preview_dir = entry.get("preview_dir")
        if preview_dir:
            src_dir = src / preview_dir
            if src_dir.is_dir():
                dest_dir = dest_root / preview_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                for item in src_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in PREVIEW_SUFFIXES:
                        shutil.copy2(item, dest_dir / item.name)
                        copied.add(dest_dir / item.name)
        rel = entry.get("preview_url")
        if not rel:
            continue
        preview = src / rel
        if preview.is_file() and preview.suffix.lower() in PREVIEW_SUFFIXES:
            dest = dest_root / rel
            if dest in copied:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview, dest)
    for leftover in dest_root.rglob("*"):
        if leftover.is_file() and leftover.suffix.lower() in {".pt", ".onnx", ".pth", ".bin"}:
            leftover.unlink()


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

    copy_maps_gallery(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MEDIA_DIR)
    args = parser.parse_args()
    publish(args.out)


if __name__ == "__main__":
    main()
