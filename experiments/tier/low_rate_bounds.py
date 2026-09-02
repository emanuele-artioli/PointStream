"""Write the BP45 bounds file. Must land before the first system encode.

The interval on BD-rate is deliberately broad: this experiment is searching
for a regime, not confirming a predicted one. A result outside it is still
an alarm — investigate the measurement before celebrating or despairing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.tier.low_rate_validate import BOUNDS_PATH, OUT_DIR
from src.contracts.metrics import HEADLINE_CURVE_METRICS, metric as metric_spec


def bounds_document() -> dict[str, Any]:
    """Two-sided bands, with the reasoning that makes a later revision auditable."""
    axes = {
        spec.name: {
            "direction": spec.direction.value,
            "unit": spec.unit or "unitless",
            "range": list(spec.range) if spec.range is not None else None,
            "curve_quality_transform": spec.curve_quality_transform,
            "curve_rate_transform": spec.curve_rate_transform,
            "min_curve_span": spec.min_curve_span,
            "describe": spec.describe_axis(),
        }
        for spec in (metric_spec(name) for name in HEADLINE_CURVE_METRICS)
    }
    lpips = metric_spec("lpips")
    axes[lpips.name] = {
        "direction": lpips.direction.value,
        "unit": lpips.unit or "unitless",
        "range": list(lpips.range) if lpips.range is not None else None,
        "curve_quality_transform": lpips.curve_quality_transform,
        "curve_rate_transform": lpips.curve_rate_transform,
        "min_curve_span": lpips.min_curve_span,
        "describe": lpips.describe_axis(),
        "status": "diagnostic until calibrated at the working resolution",
    }
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "bp45-low-rate",
        "criterion_frozen": False,
        "criterion_frozen_on": None,
        "how_to_read": (
            "A value outside a band is an alarm, not a finding. Revise a band "
            "only with a recorded reason. Non-overlapping curves do not get a "
            "BD-rate; test strict dominance at the lowest valid anchor point."
        ),
        "quality_axes": axes,
        "bounds": {
            "bd_rate_vmaf_percent": {
                "av1": {"low": -80.0, "high": 180.0},
                "vvc": {"low": -80.0, "high": 180.0},
                "basis": (
                    "Two-sided and wide because the experiment tests whether "
                    "PointStream can win at ultra-low rate, not whether it "
                    "repeats the +91% high-fidelity loss. −80% would be a "
                    "spectacular win; +180% is worse than the current "
                    "high-rate result and would mean the low-rate move failed. "
                    "Units are percent, not fraction."
                ),
            },
            "decode_must_match_source": {
                "low": 1.0,
                "high": 1.0,
                "basis": (
                    "No codec point may decode to zero, the wrong frame count, "
                    "or the wrong size. 1 = usable, 0 = rejected."
                ),
            },
            "identical_anchor_at_ceiling": {
                "vmaf": {"low": 95.0, "high": 99.0},
                "psnr": {"low": 50.0, "high": None},
                "ssim": {"low": 0.99, "high": 1.0},
                "lpips": {"low": 0.0, "high": 0.02},
                "basis": (
                    "BP23 identical VMAF on this host's 4K tennis frames is "
                    "97.54, not 100. A score near 100 is the crossed-input "
                    "libvmaf bug. PSNR of identical frames may be inf."
                ),
            },
            "late_frame_quality_change": {
                "vmaf": {"low": -25.0, "high": 8.0},
                "psnr_y_dB": {"low": -8.0, "high": 3.0},
                "basis": (
                    "Last-frame minus first-frame. A large drop means the "
                    "reconstruction rots over the scene; a large rise is also "
                    "surprising and is why the band is two-sided. Written "
                    "before any long-scene encode. Independent of the mean."
                ),
            },
            "fallback_reproduces_reference": {
                "rate_rel": {"low": 0.95, "high": 1.05},
                "vmaf_abs": {"low": -1.0, "high": 1.0},
                "basis": (
                    "The conventional-fallback control must reproduce the "
                    "reference codec on the same frames, within encoder "
                    "noise. A gap here means the control is not the anchor."
                ),
            },
            "object_stream_off_isolates_background": {
                "required": True,
                "basis": (
                    "An object-stream-off control must run in the same session "
                    "as any claimed win, so a background-reuse-only saving "
                    "cannot be credited to the object stream."
                ),
            },
        },
    }


def write_bounds(dest: Path | None = None, *, force: bool = False) -> Path:
    path = dest or BOUNDS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists. Bounds are written once before the first "
            "encode; revising them is a recorded act, not a silent overwrite."
        )
    path.write_text(json.dumps(bounds_document(), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(BOUNDS_PATH))
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing bounds file. Only for a recorded revision.",
    )
    args = parser.parse_args(argv)
    dest = write_bounds(Path(args.out), force=bool(args.force))
    print(dest)
    print(f"also under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
