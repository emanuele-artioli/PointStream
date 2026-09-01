"""The paired ladder over N scenes, with the anchor given the same footage.

`experiments/tier/ladder.py` runs one scene per arm. That is the wrong shape for
the question BP31 asks, because PointStream's background now **amortises across
scenes** (`PLAN.md` §2.22): a single-scene ladder charges the plate to one scene
and measures the system at its least favourable amortisation. This module runs
the sequence.

**The fairness condition is the whole design.** A codec encoding a multi-scene
sequence can predict across a scene join, exactly as PointStream's background
stream does. So the anchor here encodes the **concatenation** of the same N
scenes in one pass, not N separate encodes summed. Running it per-scene and
adding up would hand PointStream an amortisation the anchor was forbidden, which
is the rig `plans/BP30-background-stream.md` §5 exists to prevent.

That is easy to get wrong silently, so this module measures it rather than
promising it: every rung also encodes the same scenes **separately** and reports
that total beside the joint one. The joint encode must be cheaper. If the two
are equal, the anchor is not predicting across the join and the comparison is
rigged in PointStream's favour — that is a bound in
`outputs/bp31-ladder/bounds-before-run.json`, and it is checked here.

**One bound model is one stream** (`plans/BP30-findings.md`, and the guard in
`tests/runner/test_background_stream_stage.py`). The runner binds the background
model once per `run()` call and reuses it across chunks, which is what carries
the previous scene's reconstruction forward. So one rung is one `run()` over N
chunks — never N calls stitched together, which would hand every scene a fresh
empty stream and pay a full keyframe each time, with the amortisation configured,
reported in the ledger, and absent.

**What each arm's quality is measured on.** The anchor's, on what the decoder
returned. PointStream's, on `delivered_frames` — not `RunResult.frames`, which
carries the residual before `residual.codec` ran on it (`plans/BP24-findings.md`
§8). One pooled-PSNR convention for both, imported from `ladder.py` rather than
restated, because BP23 found two conventions inside one ladder disagreeing by
0.65 dB.

Scope this does not hide: the scenes available with player tracks are those with
a cached BP21 window, so N is small until more are materialised. A BD-rate from
few scenes on one video is a configuration measurement, not a claim; `presley`'s
bar is n>=6 videos.

Run::

    python -m experiments.tier.ladder_scenes --video alcaraz_highlights \
        --scenes scene_000 scene_010 --codec av1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import BP21_CLIPS, TierClip, load_tier_clip
from experiments.tier.ladder import PAYLOAD_RUNGS, pooled_psnr
from src.components.codec.measure import PRESETS
from src.contracts import domain as domains
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import PointstreamConfig
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.runner import run
from src.runner.config_io import load_tier
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp31-ladder"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"

#: Low-delay flags for the anchor, per codec, so it runs under the same
#: constraint PointStream's background stream does. Without this the anchor gets
#: lookahead and B-frames the streamed arm is denied, and the pair is comparing
#: constraints rather than systems. Reported in the record either way: a
#: constraint that is not in the output is not a constraint anyone can check.
ANCHOR_LOW_DELAY: dict[str, tuple[str, ...]] = {
    "av1": ("-svtav1-params", "lookahead=0:enable-overlays=0"),
    "avc": ("-bf", "0", "-x264-params", "bframes=0"),
    "hevc": (),
    "vvc": (),
}


def _no_generator() -> GeneratorRef:
    raise AssertionError("generation is off in every tier config used here")


def load_scene_sequence(
    video: str, scenes: list[str], *, n_frames: int
) -> list[TierClip]:
    """The N scenes, in order, each with its tracks re-verified by paste-back.

    Refuses rather than skips a missing scene: a ladder quietly run over four
    scenes when six were asked for reports an amortisation that belongs to a
    different N.
    """
    clips: list[TierClip] = []
    for scene in scenes:
        window = BP21_CLIPS / video / scene / "window"
        if not window.is_dir():
            raise SystemExit(
                f"no cached window at {window}. Scenes with player tracks are "
                "limited to those BP21 cached; materialise more with "
                "`experiments.headroom.real.load_scene_clip` before asking for them."
            )
        clips.append(load_tier_clip(video=video, scene=scene, n_frames=n_frames))
    return clips


def anchor_over_sequence(clips: list[TierClip], request: EncodeRequest) -> dict[str, Any]:
    """The anchor coding the N scenes as one sequence, and as N separate ones.

    The joint encode is the arm. The separate total is the **control that proves
    the fairness condition held**: a codec given the concatenation can predict
    across the join, so joint must come in under separate. Equality means the
    anchor was effectively run per-scene, and any PointStream gain measured
    against it is an artefact.
    """
    from src.components.codec.frames import even_size
    from src.components.codec.measure import coded_roundtrip

    joined = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
    started = time.time()
    joint_bytes, decoded = coded_roundtrip(joined, request=request)
    seconds = time.time() - started

    separate_bytes = 0
    for clip in clips:
        part_bytes, _ = coded_roundtrip(np.asarray(clip.frames), request=request)
        separate_bytes += int(part_bytes)

    reference = even_size(joined)
    return {
        "joint_bytes": int(joint_bytes),
        "separate_bytes": int(separate_bytes),
        # Below 1.0 means the anchor really did predict across the scene joins.
        "joint_over_separate": (
            round(float(joint_bytes) / separate_bytes, 4) if separate_bytes else None
        ),
        "psnr_y_dB": pooled_psnr(reference, decoded, luma=True),
        "psnr_rgb_dB": pooled_psnr(reference, decoded),
        "seconds": round(seconds, 1),
        "n_frames": int(joined.shape[0]),
        "extra_args": list(request.extra_args),
    }


def pointstream_over_sequence(
    clips: list[TierClip], config: PointstreamConfig
) -> dict[str, Any]:
    """PointStream over the N scenes as N chunks of one run.

    One `run()` call, so the background model is bound once and the stream
    carries each scene's reconstruction into the next. Two calls would be two
    streams.
    """
    started = time.time()
    result = run(
        config,
        [np.asarray(clip.frames) for clip in clips],
        bind_generator_fn=_no_generator,
        objects=tuple(clip.objects for clip in clips),
    )
    seconds = time.time() - started
    source = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
    delivered = result.delivered_frames
    sizes = result.sizes
    total = int(sizes.transport_total)
    panorama = int(sizes.panorama)
    return {
        "coded_bytes": total,
        "psnr_y_dB": pooled_psnr(source, delivered, luma=True),
        "psnr_rgb_dB": pooled_psnr(source, delivered),
        "is_rate": bool(sizes.is_rate),
        "raw_parts": list(sizes.raw_parts),
        "parts": {
            "residual": int(sizes.residual),
            "panorama": panorama,
            "actor_reference": int(sizes.actor_reference),
            "metadata": int(sizes.metadata),
        },
        # The number BP31 exists to move: it was 88-91% with a fresh plate per
        # scene. `SizesBytes.panorama` is a *marginal* cost under
        # `panorama-stream`, and this total is right because chunk 0's keyframe
        # is in the sum.
        "background_share": round(panorama / total, 4) if total else None,
        "n_chunks": len(result.chunks),
        "precodec_vs_delivered_dB": pooled_psnr(
            np.asarray(result.frames), delivered, luma=True
        ),
        "seconds": round(seconds, 1),
    }


def check_bounds(rows: list[dict[str, Any]]) -> list[str]:
    """The bounds file, evaluated in the run rather than left beside it."""
    alarms: list[str] = []
    for row in rows:
        label = row.get("rung")
        anchor = row.get("anchor") or {}
        stream = row.get("pointstream") or {}

        ratio = anchor.get("joint_over_separate")
        if isinstance(ratio, (int, float)) and ratio >= 1.0:
            alarms.append(
                f"{label}: the anchor's joint encode ({anchor.get('joint_bytes')} B) is not "
                f"cheaper than N separate encodes ({anchor.get('separate_bytes')} B). It is "
                "not predicting across the scene joins, so it was effectively run per-scene "
                "and any PointStream gain here is an artefact of the rig, not a result."
            )

        share = stream.get("background_share")
        if isinstance(share, (int, float)):
            if share >= 0.88:
                alarms.append(
                    f"{label}: background is {share:.1%} of the payload, still in the 88-91% "
                    "band it occupied with a fresh plate per scene. The stream is not "
                    "reaching the ledger — the amortisation would be configured, reported "
                    "and absent."
                )
            elif share < 0.55:
                alarms.append(
                    f"{label}: background is {share:.1%} of the payload, below the 55% floor "
                    "written before the run. Something other than the plate changed; check "
                    "the residual's rung and delivered_frames before believing it."
                )

        if not stream.get("is_rate", True):
            alarms.append(
                f"{label}: ledger withheld the ratio ({stream.get('raw_parts')}), so this "
                "total is not a rate and the rung is not on the curve."
            )
    return alarms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_010"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--stream-crf", type=int, default=38)
    parser.add_argument(
        "--anchor-low-delay",
        action="store_true",
        help="constrain the anchor the way the background stream is constrained",
    )
    parser.add_argument(
        "--max-rungs",
        type=int,
        default=None,
        help="run only the first N payload rungs - validates the path, not a curve",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds go to disk before the first encode.")

    clips = load_scene_sequence(args.video, list(args.scenes), n_frames=args.frames)
    print(
        f"{args.video}: {len(clips)} scenes x {args.frames} frames "
        f"({clips[0].describe()['resolution']})",
        flush=True,
    )

    base = load_tier(args.tier)
    residual = replace(
        base.residual,
        codec=args.codec,
        preset=PRESETS[args.codec],
        rate_control=RateControl.QP,
    )
    background = replace(
        base.background,
        method=domains.BACKGROUND_PANORAMA_STREAM,
        stream_codec=args.codec if args.codec in {"av1", "hevc", "avc"} else "av1",
        stream_crf=int(args.stream_crf),
        keyframe_interval=0,
        reference_mode="last",
    )
    paired = base.with_(residual=residual, background=background)

    extra = ANCHOR_LOW_DELAY.get(args.codec, ()) if args.anchor_low_delay else ()

    rungs = PAYLOAD_RUNGS[: args.max_rungs] if args.max_rungs else PAYLOAD_RUNGS
    if len(rungs) < len(PAYLOAD_RUNGS):
        print(
            f"NOTE: {len(rungs)} of {len(PAYLOAD_RUNGS)} rungs - this validates the "
            "path and does not produce a curve worth a BD-rate.",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for index, (jpeg_quality, rate_value) in enumerate(rungs):
        label = f"q{jpeg_quality}/qp{rate_value}"
        request = replace(residual, rate=int(rate_value)).encode_request()
        request = EncodeRequest(
            codec_name=request.codec_name,
            rate_control=request.rate_control,
            rate=request.rate,
            preset=request.preset,
            pix_fmt=request.pix_fmt,
            extra_args=tuple(extra),
        )
        request.validate()

        row: dict[str, Any] = {"rung": label, "rank": len(rungs) - 1 - index}
        try:
            row["anchor"] = anchor_over_sequence(clips, request)
            print(
                f"  anchor  {label:<12} joint {row['anchor']['joint_bytes']:>10} B  "
                f"sep {row['anchor']['separate_bytes']:>10} B  "
                f"j/s {row['anchor']['joint_over_separate']}  "
                f"{row['anchor']['psnr_y_dB']:6.2f} dB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            row["anchor_error"] = repr(exc)
            print(f"  anchor  {label:<12} FAILED {exc!r}", flush=True)

        tuned = paired.with_(
            background=replace(paired.background, jpeg_quality=int(jpeg_quality)),
            residual=replace(paired.residual, rate=int(rate_value)),
        )
        try:
            row["pointstream"] = pointstream_over_sequence(clips, tuned)
            print(
                f"  stream  {label:<12} {row['pointstream']['coded_bytes']:>10} B  "
                f"{row['pointstream']['psnr_y_dB']:6.2f} dB  "
                f"bg {row['pointstream']['background_share']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row["pointstream_error"] = repr(exc)
            print(f"  stream  {label:<12} FAILED {exc!r}", flush=True)
        rows.append(row)

    alarms = check_bounds(rows)
    report: dict[str, Any] = {
        "video": args.video,
        "scenes": list(args.scenes),
        "frames_per_scene": args.frames,
        "n_rungs": len(rungs),
        "codec": args.codec,
        "preset": PRESETS[args.codec],
        "anchor_low_delay": list(extra),
        "background": {
            "method": paired.background.method,
            "stream_codec": paired.background.stream_codec,
            "stream_crf": paired.background.stream_crf,
            "keyframe_interval": paired.background.keyframe_interval,
            "reference_mode": paired.background.reference_mode,
        },
        "bounds_file": str(BOUNDS_PATH),
        "rows": rows,
        "alarms": alarms,
        "reading_note": (
            "The anchor arm is ONE encode over the concatenated scenes; "
            "`separate_bytes` is the control proving it predicted across the "
            "joins. Quality is Y-PSNR on delivered_frames for PointStream and "
            "on the decoder's output for the anchor. Few scenes on one video is "
            "a configuration measurement, not a claim."
        ),
    }
    dest = Path(args.out) if args.out else OUT_DIR / f"ladder-scenes-{args.video}-{args.codec}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str) + "\n")

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
