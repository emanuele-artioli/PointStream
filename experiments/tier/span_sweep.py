"""Frames per scene: the amortisation axis nobody has swept.

Brief: `plans/BP33-span-amortisation.md`, written by a parallel session and
reaching this one before the first span encode — which is why its bounds are
adopted verbatim in `outputs/bp31-ladder/bounds-before-span-run.json` rather
than rewritten here. Bounds authored after hearing a prediction are not bounds.

**The claim being tested.** The plate is 88-91% of the payload and is paid once
per scene *whatever the scene's length*, so frames-per-scene is a direct divisor
on the dominant cost. Every ladder in this project has run at eight frames; the
BP21 cache holds forty-eight, and `plans/done/RESEARCH-HISTORY.md` §2.14's headroom was measured over
those forty-eight. So the headroom was measured over 48 frames and the system
scored over 8.

**Span is the only thing that moves.** Scene count stays at N=2, the value
`plans/done/BP31-findings.md` §9's BD-rate was taken at, so the numbers here are
directly comparable to it. BP33 §5: span and scene count are both amortisation
axes on the same fixed cost, they interact, and one sweep run over both will not
separate them. BP30 moved one lever on one video and had two conclusions invert
at five; half the lesson was "more videos" and the other half was "move one
thing".

**Quality is read against frame index, not as a clip mean.** The failure a long
span is most likely to have is not a worse average but a reconstruction that
rots — the homography drifts off the plate, late frames decay, early ones do
not. A mean scores that the same as a uniform reconstruction, and they are
different products.

**The anchor gets the same span**, encoded jointly, with the joint-versus-separate
control from `plans/done/BP31-findings.md` §5 carried at every point. A longer span
helps the anchor too — one intra keyframe amortising over more inter frames — and
the whole question is which arm it helps more.

Run::

    python -m experiments.tier.span_sweep --video alcaraz_highlights
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.tier.ladder_scenes import (
    STREAM_PAYLOAD_RUNGS,
    anchor_over_sequence,
    load_scene_sequence,
    pointstream_over_sequence,
)
from src.components.codec.measure import PRESETS
from src.contracts import domain as domains
from src.contracts.codecs import RateControl
from src.contracts.config import PointstreamConfig
from src.runner.config_io import load_tier
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp31-ladder"
BOUNDS_PATH = OUT_DIR / "bounds-before-span-run.json"

#: A curve, not two points: the question is whether the gain saturates and where.
DEFAULT_SPANS: tuple[int, ...] = (8, 16, 24, 32, 48)

#: The reference rung. Held fixed across spans so span is the only axis moving.
#: `STREAM_PAYLOAD_RUNGS[2]` is (stream_crf 38, residual qp 38) — BP30's
#: operating point and the middle of the payload ladder.
REFERENCE_RUNG_INDEX = 2

#: From the adopted bounds file.
ANCHOR_PER_FRAME_BAND: tuple[float, float] = (0.45, 0.90)
PLATE_GROWTH_BAND_STATIC: tuple[float, float] = (1.00, 1.60)
QUALITY_DRIFT_BAND: tuple[float, float] = (-4.0, 0.5)


def build_config(codec: str, tier: str, stream_crf: int, rate: int) -> PointstreamConfig:
    base = load_tier(tier)
    residual = replace(
        base.residual, codec=codec, preset=PRESETS[codec], rate_control=RateControl.QP, rate=int(rate)
    )
    background = replace(
        base.background,
        method=domains.BACKGROUND_PANORAMA_STREAM,
        stream_codec=codec if codec in {"av1", "hevc", "avc"} else "av1",
        stream_crf=int(stream_crf),
        keyframe_interval=0,
        reference_mode="last",
    )
    return base.with_(residual=residual, background=background)


def drift(per_frame: list[float]) -> float | None:
    """Last frame's Y-PSNR minus the first's. Negative means the clip rots."""
    if len(per_frame) < 2:
        return None
    return round(per_frame[-1] - per_frame[0], 3)


def check_bounds(points: list[dict[str, Any]]) -> list[str]:
    """The adopted BP33 bounds, evaluated in the run."""
    alarms: list[str] = []
    base = next((p for p in points if p.get("span") == 8 and p.get("ok")), None)

    for point in points:
        if not point.get("ok"):
            continue
        span = point["span"]
        anchor, stream = point["anchor"], point["pointstream"]

        ratio = anchor.get("joint_over_separate")
        if isinstance(ratio, (int, float)) and ratio >= 1.0:
            alarms.append(
                f"span {span}: anchor joint/separate is {ratio}, not below 1.0 — it did not "
                "predict across the scene join, so any PointStream gain here is the rig."
            )

        for label, arm in (("pointstream", stream), ("anchor", anchor)):
            value = drift(arm.get("psnr_y_by_frame") or [])
            low, high = QUALITY_DRIFT_BAND
            if value is not None and not low <= value <= high:
                alarms.append(
                    f"span {span} {label}: Y-PSNR drifts {value:+.2f} dB from first frame to "
                    f"last, outside [{low}, {high}]. A clip mean would hide this; a "
                    "reconstruction that decays along the span is a different product from a "
                    "uniform one."
                )

        if base is not None and span != 8:
            low, high = ANCHOR_PER_FRAME_BAND
            per_frame_now = anchor["joint_bytes"] / anchor["n_frames"]
            per_frame_base = base["anchor"]["joint_bytes"] / base["anchor"]["n_frames"]
            got = per_frame_now / per_frame_base if per_frame_base else None
            if span == 48 and got is not None and not low <= got <= high:
                alarms.append(
                    f"span 48: anchor costs {got:.3f}x its 8-frame per-frame rate, outside "
                    f"[{low}, {high}] — suspect it is being run per-chunk rather than over "
                    "the whole span."
                )
            plate_low, plate_high = PLATE_GROWTH_BAND_STATIC
            plate_ratio = stream["parts"]["panorama"] / base["pointstream"]["parts"]["panorama"]
            if span == 48 and not plate_low <= plate_ratio <= plate_high:
                alarms.append(
                    f"span 48: the plate is {plate_ratio:.3f}x its 8-frame size, outside "
                    f"[{plate_low}, {plate_high}] — the median over a long span may not be "
                    "converging; check foreground exclusion."
                )
    return alarms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_010"])
    parser.add_argument("--spans", nargs="+", type=int, default=list(DEFAULT_SPANS))
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds go to disk before the first encode.")
    print(f"bounds (adopted from BP33): {BOUNDS_PATH}", flush=True)

    stream_crf, rate = STREAM_PAYLOAD_RUNGS[REFERENCE_RUNG_INDEX]
    config = build_config(args.codec, args.tier, stream_crf, rate)
    print(
        f"reference rung held fixed: stream_crf={stream_crf} residual qp={rate}, "
        f"codec={args.codec} preset={PRESETS[args.codec]}, N={len(args.scenes)} scenes",
        flush=True,
    )

    points: list[dict[str, Any]] = []
    for span in args.spans:
        point: dict[str, Any] = {"span": span, "ok": False}
        try:
            clips = load_scene_sequence(args.video, list(args.scenes), n_frames=span)
            request = replace(config.residual, rate=int(rate)).encode_request()
            request.validate()
            anchor = anchor_over_sequence(clips, request)
            stream = pointstream_over_sequence(clips, config)
            point.update({"ok": True, "anchor": anchor, "pointstream": stream})
            total_frames = anchor["n_frames"]
            print(
                f"  span {span:>2}  anchor {anchor['joint_bytes']:>10} B "
                f"({anchor['joint_bytes']/total_frames:>9.0f} B/f, j/s {anchor['joint_over_separate']}) "
                f"{anchor['psnr_y_dB']:6.2f} dB  |  stream {stream['coded_bytes']:>10} B "
                f"({stream['coded_bytes']/total_frames:>9.0f} B/f) {stream['psnr_y_dB']:6.2f} dB  "
                f"plate {stream['parts']['panorama']:>9} B  bg {stream['background_share']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            point["error"] = repr(exc)
            print(f"  span {span:>2}  FAILED {exc!r}", flush=True)
        points.append(point)

    alarms = check_bounds(points)
    report = {
        "video": args.video,
        "scenes": list(args.scenes),
        "n_scenes": len(args.scenes),
        "codec": args.codec,
        "preset": PRESETS[args.codec],
        "reference_rung": {"stream_crf": stream_crf, "residual_qp": rate},
        "bounds_file": str(BOUNDS_PATH),
        "brief": "plans/BP33-span-amortisation.md",
        "points": points,
        "alarms": alarms,
        "reading_note": (
            "Span is the only axis moving; scene count is held at N=2, the value "
            "findings §9's BD-rate was taken at. Rate is read per frame, because "
            "a longer span costs more in total and less per frame and only the "
            "second is the amortisation. Quality is read against frame index, "
            "not as a clip mean. One rung, two scenes, one video: this decides "
            "an operating point for the extraction campaign, not a claim."
        ),
    }
    dest = Path(args.out) if args.out else OUT_DIR / f"span-sweep-{args.video}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print("\n=== per-frame rate against span ===", flush=True)
    for point in points:
        if not point.get("ok"):
            continue
        frames = point["anchor"]["n_frames"]
        a = point["anchor"]["joint_bytes"] / frames
        s = point["pointstream"]["coded_bytes"] / frames
        print(
            f"  span {point['span']:>2}  anchor {a:>9.0f} B/f   pointstream {s:>9.0f} B/f   "
            f"ratio {s/a:5.2f}x   drift(ps) {drift(point['pointstream']['psnr_y_by_frame'])}",
            flush=True,
        )

    if alarms:
        print("\n=== ALARMS ===", flush=True)
        for alarm in alarms:
            print(f"  ! {alarm}", flush=True)
    else:
        print("\nno alarms", flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
