"""Which cached clip is high motion? Measure it; do not call one 'dynamic'.

`plans/BP24-findings.md` §7 records both of BP24's headline ratios as the *easy*
case — a residual that was 2.5% non-zero against a static plate — and says to
re-measure on high motion. That instruction is only actionable if "high motion"
is a number attached to a named clip rather than an impression of a sport.

The number here is the mean absolute difference between consecutive frames, in
grey levels, over the same window the tier ladder would load. It is deliberately
the crudest possible measure: it is the quantity the residual has to carry when
the plate is a single still frame, which is what the runner transmits today
(findings §6). A fancier motion estimate would describe the scene better and
predict the residual worse.

Also reported, because it is the quantity that actually drives the plate's
usefulness: the mean absolute difference of every frame against the **first**
frame, which is the plate. A clip can have small frame-to-frame motion and still
drift a long way from frame one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.headroom.real import load_rgb_stack
from experiments.tier.clip import BP21_CLIPS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "bp24-ladder" / "motion-survey.json"


def survey_window(window: Path, n_frames: int) -> dict[str, Any] | None:
    pngs = sorted(window.glob("frame_*.png"))
    if len(pngs) < n_frames:
        return None
    frames = load_rgb_stack(list(pngs[:n_frames])).astype(np.int16)
    consecutive = float(np.abs(frames[1:] - frames[:-1]).mean())
    from_first = float(np.abs(frames[1:] - frames[:1]).mean())
    drift = float(np.abs(frames[-1] - frames[0]).mean())
    return {
        "n_frames": int(frames.shape[0]),
        "resolution": f"{frames.shape[2]}x{frames.shape[1]}",
        "consecutive_mad": consecutive,
        "vs_first_frame_mad": from_first,
        "last_vs_first_mad": drift,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rank cached clips by motion.")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for video_dir in sorted(BP21_CLIPS.iterdir()):
        if not video_dir.is_dir():
            continue
        for scene_dir in sorted(video_dir.iterdir()):
            window = scene_dir / "window"
            if not window.is_dir():
                continue
            measured = survey_window(window, args.frames)
            if measured is None:
                continue
            rows.append({"video": video_dir.name, "scene": scene_dir.name, **measured})
            print(
                f"{video_dir.name}/{scene_dir.name:>10}  "
                f"consecutive={measured['consecutive_mad']:6.2f}  "
                f"vs-first={measured['vs_first_frame_mad']:6.2f}",
                flush=True,
            )

    rows.sort(key=lambda row: row["consecutive_mad"])
    payload = {
        "measure": "mean absolute difference in grey levels, RGB, over the first N frames",
        "why": (
            "The plate the runner transmits is the first source frame, so "
            "vs_first_frame_mad is what the residual has to carry. "
            "consecutive_mad is the conventional motion figure and is reported "
            "beside it because the two can disagree."
        ),
        "n_frames": args.frames,
        "clips_low_to_high": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
