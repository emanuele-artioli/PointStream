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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import BP21_CLIPS, TierClip, load_tier_clip
from experiments.tier.ladder import PAYLOAD_RUNGS, pooled_psnr
from src.components.background.stream import (
    context_reset_indices,
    scene_groups,
    segmented_reset_indices,
)
from src.components.codec.measure import PRESETS
from src.contracts import domain as domains
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import PointstreamConfig
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.runner import run
from src.runner.config_io import load_tier
from src.contracts import paths as ps_paths

#: The payload rungs for a **streamed** background, as
#: ``(stream_crf, residual rate)``. `PAYLOAD_RUNGS` pairs
#: `background.jpeg_quality` with the residual's rate, and `jpeg_quality`
#: reaches nothing under `panorama-stream` (`plans/BP31-findings.md` §1). Run
#: with that table, the streamed arm's plate came back **789,304 B at all five
#: rungs** — byte-identical — so the curve swept the residual against a frozen
#: plate. That is the exact degenerate shape `PAYLOAD_RUNGS`'s own docstring was
#: written about: "a rung has to move everything that trades rate for quality".
#:
#: `stream_crf` is that knob for this method. The CRFs bracket BP30's operating
#: point (38) and span the range the §3 sweep measured, paired with the same
#: residual rates so the two tables describe the same five operating points.
STREAM_PAYLOAD_RUNGS: tuple[tuple[int, int], ...] = (
    (51, 55),
    (45, 46),
    (38, 38),
    (30, 28),
    (22, 18),
)

OUT_DIR = ps_paths.outputs() / "bp31-ladder"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"

#: Low-delay flags for the anchor, per codec, so it runs under the same
#: constraint PointStream's background stream does. Without this the anchor gets
#: lookahead and B-frames the streamed arm is denied, and the pair is comparing
#: constraints rather than systems. Reported in the record either way: a
#: constraint that is not in the output is not a constraint anyone can check.
#:
#: **These are per-binary, not per-format.** `src/components/codec/tools.py`
#: sends `avc` and `vvc` to ffmpeg, `hevc` to kvazaar and `av1` to
#: `SvtAv1EncApp` — three different command-line vocabularies. The first version
#: of this table assumed ffmpeg for everything and handed SvtAv1EncApp an
#: ffmpeg-style `-svtav1-params`, which it rejected outright
#: ("single dash long tokens have been removed"). A codec whose flags have not
#: been checked against its own binary is absent here rather than guessed at,
#: because a flag the encoder silently ignores is worse than one it refuses:
#: the run would complete and the constraint would not be there.
#:
#: av1, verified against SvtAv1EncApp v1.8.0 `--help` on this host:
#:   `--pred-struct 1` low-delay frames, `--lookahead 0`, and a `--keyint`
#:   larger than any sequence here so the anchor gets one I-frame — matching
#:   `background.keyframe_interval: 0`, which is a pure P-chain.
ANCHOR_LOW_DELAY: dict[str, tuple[str, ...]] = {
    "av1": ("--pred-struct", "1", "--lookahead", "0", "--keyint", "1000"),
}


class LowDelayUnavailable(SystemExit):
    """Asked to constrain an anchor whose flags have not been verified here."""


def anchor_low_delay_args(codec: str) -> tuple[str, ...]:
    """The verified low-delay argv for ``codec``, or a refusal naming why.

    Refusing beats returning ``()``. An empty tuple would run the anchor
    unconstrained while the report said `--anchor-low-delay` was requested,
    which is the pairing quietly not holding.
    """
    try:
        return ANCHOR_LOW_DELAY[codec]
    except KeyError:
        raise LowDelayUnavailable(
            f"--anchor-low-delay has no verified flag set for {codec!r}. Verified: "
            f"{sorted(ANCHOR_LOW_DELAY)}. `avc`/`vvc` run through ffmpeg, `hevc` "
            "through kvazaar and `av1` through SvtAv1EncApp, so each needs its own "
            "vocabulary checked against its own binary before it is used here. "
            "Run without --anchor-low-delay to measure the unconstrained anchor, "
            "and say so in the report."
        ) from None


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


def _psnr_by_frame(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    """Y-PSNR per frame index, so monotone decay cannot hide inside a clip mean.

    A longer span's most likely failure is not a worse average but a
    reconstruction that *rots*: the homography drifts off the plate and late
    frames degrade while early ones stay sharp. Those are different products and
    a mean scores them the same, so the span sweep reads this rather than the
    mean (`plans/BP33-span-amortisation.md` §3.3).
    """
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    count = min(int(ref.shape[0]), int(got.shape[0]))
    return [round(pooled_psnr(ref[i], got[i], luma=True), 3) for i in range(count)]


def context_ids_for_clips(clips: list[TierClip]) -> tuple[str, ...]:
    """One background context per camera/venue, not per scene file.

    Point-class scenes from one video share a canvas and may be predicted
    across. A later replay cut is a different id, and that is a new keyframe.
    Scene names are not ids: ``scene_000`` and ``scene_010`` are the same court.
    """
    return tuple(f"{clip.video}-point" for clip in clips)


def _concat_clips(clips: list[TierClip]) -> np.ndarray:
    return np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)


def anchor_over_sequence(
    clips: list[TierClip],
    request: EncodeRequest,
    *,
    context_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The anchor coding the N scenes as one sequence, and as N separate ones.

    The joint encode is the arm that asks whether the codec can predict across
    a join at all. The **continuous** encode is the fair comparison: it resets
    at exactly ``context_reset_indices`` of the PointStream configuration.
    The **segmented** encode resets every scene. Same-id sequences make
    continuous equal joint; all-different ids make continuous equal segmented.
    Reporting only joint against a PointStream that resets at a replay would
    hand the anchor a prediction PointStream is forbidden.
    """
    from src.components.codec.frames import even_size
    from src.components.codec.measure import coded_roundtrip

    ids = tuple(context_ids) if context_ids is not None else context_ids_for_clips(clips)
    if len(ids) != len(clips):
        raise ValueError(
            f"context_ids has {len(ids)} entries for {len(clips)} clips. "
            "Pair by track position, one id per scene."
        )
    groups = scene_groups(ids)
    joined = _concat_clips(clips)
    started = time.time()
    joint_bytes, decoded = coded_roundtrip(joined, request=request)
    seconds = time.time() - started

    separate_bytes = 0
    separate_parts: list[np.ndarray] = []
    for clip in clips:
        part_bytes, part_decoded = coded_roundtrip(np.asarray(clip.frames), request=request)
        separate_bytes += int(part_bytes)
        separate_parts.append(part_decoded)
    separate_decoded = np.concatenate(separate_parts, axis=0)

    if groups == ((0, len(clips)),):
        continuous_bytes = int(joint_bytes)
        decoded_continuous = decoded
    elif len(groups) == len(clips):
        continuous_bytes = int(separate_bytes)
        decoded_continuous = separate_decoded
    else:
        continuous_bytes = 0
        parts: list[np.ndarray] = []
        for start, end in groups:
            part_bytes, part_decoded = coded_roundtrip(
                _concat_clips(clips[start:end]), request=request
            )
            continuous_bytes += int(part_bytes)
            parts.append(part_decoded)
        decoded_continuous = np.concatenate(parts, axis=0)

    segmented_bytes = int(separate_bytes)
    reference = even_size(joined)
    return {
        "psnr_y_by_frame": _psnr_by_frame(reference, decoded),
        "joint_bytes": int(joint_bytes),
        "separate_bytes": int(separate_bytes),
        "continuous_bytes": int(continuous_bytes),
        "segmented_bytes": segmented_bytes,
        "context_ids": list(ids),
        "continuous_resets": list(context_reset_indices(ids)),
        "segmented_resets": list(segmented_reset_indices(len(clips))),
        "continuous_groups": [list(group) for group in groups],
        # Below 1.0 means the anchor really did predict across the scene joins.
        "joint_over_separate": (
            round(float(joint_bytes) / separate_bytes, 4) if separate_bytes else None
        ),
        "continuous_over_segmented": (
            round(float(continuous_bytes) / segmented_bytes, 4) if segmented_bytes else None
        ),
        "psnr_y_dB": pooled_psnr(reference, decoded, luma=True),
        "psnr_rgb_dB": pooled_psnr(reference, decoded),
        "continuous_psnr_y_dB": pooled_psnr(reference, decoded_continuous, luma=True),
        "seconds": round(seconds, 1),
        "n_frames": int(joined.shape[0]),
        "extra_args": list(request.extra_args),
    }


def pointstream_over_sequence(
    clips: list[TierClip],
    config: PointstreamConfig,
    *,
    context_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """PointStream over the N scenes as N chunks of one run.

    One `run()` call, so the background model is bound once and the stream
    carries each scene's reconstruction into the next. Two calls would be two
    streams. ``context_ids`` are the same list the continuous AV1/VVC control
    resets on — a mismatch here is the fairness condition quietly not holding.
    """
    ids = tuple(context_ids) if context_ids is not None else context_ids_for_clips(clips)
    started = time.time()
    result = run(
        config,
        [np.asarray(clip.frames) for clip in clips],
        bind_generator_fn=_no_generator,
        objects=tuple(clip.objects for clip in clips),
        context_ids=ids,
    )
    seconds = time.time() - started
    source = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
    delivered = result.delivered_frames
    sizes = result.sizes
    total = int(sizes.transport_total)
    panorama = int(sizes.panorama)
    return {
        "psnr_y_by_frame": _psnr_by_frame(source, delivered),
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
        "context_ids": list(ids),
        "continuous_resets": list(context_reset_indices(ids)),
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
        stream = row.get("pointstream") or {}

        for key in ("anchor", "anchor_low_delay"):
            anchor = row.get(key) or {}
            ratio = anchor.get("joint_over_separate")
            if isinstance(ratio, (int, float)) and ratio >= 1.0:
                alarms.append(
                    f"{label}/{key}: the joint encode ({anchor.get('joint_bytes')} B) is not "
                    f"cheaper than N separate encodes ({anchor.get('separate_bytes')} B). It "
                    "is not predicting across the scene joins, so it was effectively run "
                    "per-scene and any PointStream gain against it is an artefact of the "
                    "rig, not a result."
                )
            ids = anchor.get("context_ids") or []
            stream_ids = stream.get("context_ids") or []
            if ids and stream_ids and list(ids) != list(stream_ids):
                alarms.append(
                    f"{label}/{key}: anchor context_ids {ids} do not match PointStream "
                    f"{stream_ids}. The continuous control is resetting on a different "
                    "split than the system under test."
                )
            anchor_resets = anchor.get("continuous_resets")
            stream_resets = stream.get("continuous_resets")
            if (
                anchor_resets is not None
                and stream_resets is not None
                and list(anchor_resets) != list(stream_resets)
            ):
                alarms.append(
                    f"{label}/{key}: continuous resets {anchor_resets} do not match "
                    f"PointStream {stream_resets}."
                )
            if (
                ids
                and len(set(ids)) == 1
                and anchor.get("continuous_bytes") != anchor.get("joint_bytes")
            ):
                alarms.append(
                    f"{label}/{key}: one context but continuous_bytes "
                    f"({anchor.get('continuous_bytes')}) != joint_bytes "
                    f"({anchor.get('joint_bytes')}). Same-id sequences are one concat."
                )
            if (
                ids
                and len(set(ids)) == len(ids)
                and anchor.get("continuous_bytes") != anchor.get("segmented_bytes")
            ):
                alarms.append(
                    f"{label}/{key}: every scene is its own context but continuous_bytes "
                    f"({anchor.get('continuous_bytes')}) != segmented_bytes "
                    f"({anchor.get('segmented_bytes')})."
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

    # The failure that produced a plausible-looking curve carrying no plate
    # information: `PAYLOAD_RUNGS` sweeps `background.jpeg_quality`, which reaches
    # nothing under `panorama-stream`, so all five rungs coded the SAME plate to
    # the byte and the ladder swept only the residual. A rung must move what
    # dominates the payload; if the plate never moves, this is not a payload
    # ladder whatever the report is titled.
    plates = [
        int((row.get("pointstream") or {}).get("parts", {}).get("panorama", 0))
        for row in rows
        if row.get("pointstream")
    ]
    if len(plates) > 1 and len(set(plates)) == 1:
        alarms.append(
            f"the plate is {plates[0]} B at every one of {len(plates)} rungs — identical to "
            "the byte. The rung is not moving the plate, so this sweeps the residual "
            "against a frozen background and is not a payload ladder. Check that the "
            "rung table's plate knob is the one this background method reads."
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
        "--background-method",
        default=domains.BACKGROUND_PANORAMA_STREAM,
        help="`panorama-full` is the control: a fresh plate per scene, which is "
             "what the cross-scene stream has to beat in the ledger",
    )
    parser.add_argument(
        "--skip-low-delay-anchor",
        action="store_true",
        help="run only the unconstrained anchor (halves anchor cost; loses the "
             "latency-matched arm)",
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
    ids = context_ids_for_clips(clips)
    print(
        f"{args.video}: {len(clips)} scenes x {args.frames} frames "
        f"({clips[0].describe()['resolution']}) context_ids={list(ids)} "
        f"continuous_resets={list(context_reset_indices(ids))}",
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
        method=args.background_method,
        stream_codec=args.codec if args.codec in {"av1", "hevc", "avc"} else "av1",
        stream_crf=int(args.stream_crf),
        keyframe_interval=0,
        reference_mode="last",
    )
    paired = base.with_(residual=residual, background=background)

    # BOTH anchors run at every rung. Measured on this host, SvtAv1EncApp v1.8.0:
    # the low-delay flags take a 640x360 test encode from 66,485 B to 91,805 B,
    # +38%. So constraining the anchor makes it dearer, and a ladder that
    # reported only the constrained arm would be handing PointStream a 38%
    # head start and calling it fairness. The unconstrained anchor is the
    # harder comparison and leads; the latency-matched one is reported beside
    # it, and which question each answers is stated in the record.
    extra = () if args.skip_low_delay_anchor else anchor_low_delay_args(args.codec)

    streamed = args.background_method == domains.BACKGROUND_PANORAMA_STREAM
    table = STREAM_PAYLOAD_RUNGS if streamed else PAYLOAD_RUNGS
    rungs = table[: args.max_rungs] if args.max_rungs else table
    if len(rungs) < len(PAYLOAD_RUNGS):
        print(
            f"NOTE: {len(rungs)} of {len(PAYLOAD_RUNGS)} rungs - this validates the "
            "path and does not produce a curve worth a BD-rate.",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for index, (plate_knob, rate_value) in enumerate(rungs):
        label = (
            f"crf{plate_knob}/qp{rate_value}" if streamed
            else f"q{plate_knob}/qp{rate_value}"
        )
        request = replace(residual, rate=int(rate_value)).encode_request()

        row: dict[str, Any] = {"rung": label, "rank": len(rungs) - 1 - index}

        arms: list[tuple[str, tuple[str, ...]]] = [("anchor", ())]
        if extra:
            arms.append(("anchor_low_delay", tuple(extra)))
        for key, arm_extra in arms:
            arm_request = EncodeRequest(
                codec_name=request.codec_name,
                rate_control=request.rate_control,
                rate=request.rate,
                preset=request.preset,
                pix_fmt=request.pix_fmt,
                extra_args=arm_extra,
            )
            arm_request.validate()
            try:
                row[key] = anchor_over_sequence(clips, arm_request, context_ids=ids)
                print(
                    f"  {key:<17} {label:<12} joint {row[key]['joint_bytes']:>10} B  "
                    f"sep {row[key]['separate_bytes']:>10} B  "
                    f"cont {row[key]['continuous_bytes']:>10} B  "
                    f"seg {row[key]['segmented_bytes']:>10} B  "
                    f"j/s {row[key]['joint_over_separate']}  "
                    f"{row[key]['psnr_y_dB']:6.2f} dB",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                row[f"{key}_error"] = repr(exc)
                print(f"  {key:<17} {label:<12} FAILED {exc!r}", flush=True)

        # The plate knob differs by method, and using the wrong one freezes
        # the plate while the report still looks like a payload ladder.
        # Written as two explicit calls rather than one `replace(**kwargs)`:
        # a kwargs dict makes the field names invisible to the type checker,
        # which is precisely the mistake that produced the frozen plate in §8.
        plate_tuned = (
            replace(paired.background, stream_crf=int(plate_knob))
            if streamed
            else replace(paired.background, jpeg_quality=int(plate_knob))
        )
        tuned = paired.with_(
            background=plate_tuned,
            residual=replace(paired.residual, rate=int(rate_value)),
        )
        try:
            row["pointstream"] = pointstream_over_sequence(clips, tuned, context_ids=ids)
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
        "anchor_low_delay_args": list(extra),
        "plate_knob": "stream_crf" if streamed else "background.jpeg_quality",
        "background": {
            "method": paired.background.method,
            "stream_codec": paired.background.stream_codec,
            "stream_crf": paired.background.stream_crf,
            "keyframe_interval": paired.background.keyframe_interval,
            "reference_mode": paired.background.reference_mode,
        },
        "context_ids": list(ids),
        "continuous_resets": list(context_reset_indices(ids)),
        "segmented_resets": list(segmented_reset_indices(len(clips))),
        "bounds_file": str(BOUNDS_PATH),
        "rows": rows,
        "alarms": alarms,
        "reading_note": (
            "Two anchor arms at every rung. `anchor` is unconstrained and is "
            "the harder, primary comparison. `anchor_low_delay` is constrained "
            "the way PointStream's background stream is, which on this host "
            "costs SvtAv1EncApp +38% — so it is the latency-matched question, "
            "not the fair-by-default one, and reporting it alone would hand "
            "PointStream that 38%. `joint_bytes` asks whether the codec can "
            "predict across a join at all; `continuous_bytes` is the fair "
            "comparison, resetting at the same context_ids as PointStream; "
            "`segmented_bytes` is a fresh intra every scene. Same-id sequences "
            "make continuous equal joint; all-different ids make continuous "
            "equal segmented. Quality is Y-PSNR on delivered_frames for "
            "PointStream and on the decoder's output for the anchor. Few "
            "scenes on one video is a configuration measurement, not a claim."
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
