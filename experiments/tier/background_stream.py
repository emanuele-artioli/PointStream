"""What does scene *n*'s background cost, given scenes 1..n-1?

`plans/BP24-findings.md` §18 and §19 answered that for a *pair* of plates: av1
saves 31-53% coding the second as a P-frame against the first, and the saving
is causal. This prices it over a **sequence**, which is the form the claim has
to take before the paper can say the background amortises: the marginal cost per
scene, the total across N scenes, and what a periodic keyframe costs -- swept as
an axis rather than imposed, because brief §3 wants the robustness paragraph to
cite a number instead of a hand-wave.

**The control is the first thing this runs and the first thing to read.** Two
consecutive frames of one scene must come back at a few percent. §19 records why
that is not decoration: a misconfigured x265 made a P-frame come back *larger*
than a fresh intra, and without the control that would have been written up as
"low delay costs x265 60% more" -- a plausible-sounding finding about nothing.

**Bounds go to disk before the first encode**, per `AGENTS.md`. A result outside
them is an alarm to be investigated, not a headline.

Run: ``python -m experiments.tier.background_stream``
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from experiments.tier.scene_plates import (
    extract_consecutive,
    extract_plates,
    list_scenes,
    load_plates,
)
from src.components.background.stream import (
    KEYFRAME_NEVER,
    REFERENCE_BEST_SCORED,
    REFERENCE_FIRST,
    REFERENCE_LAST,
    REFERENCE_PERIODIC_I,
    BackgroundStreamTransmitter,
    ScenePayload,
    encode_chain,
    ffmpeg_provenance,
    stream_linear,
)
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp30-background"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"
RESULT_PATH = OUT_DIR / "stream-sweep.json"

#: Written before the first encode. Each bound carries the reasoning that
#: produced it, so a bound that fires wrongly can be diagnosed rather than
#: quietly widened -- one of this project's bounds fired against a correct
#: result because it had been derived in the wrong units.
BOUNDS: dict[str, Any] = {
    "control_ratio": {
        "best": 0.005,
        "worst": 0.10,
        "basis": (
            "findings §18/§19 measured 1.2-3.3% for av1 on two consecutive frames "
            "of one scene. Outside this range the harness is not measuring inter "
            "prediction at all and every other row is unreadable."
        ),
        "on_breach": "stop; fix the encoder configuration before reading any arm",
    },
    "marginal_ratio_last": {
        "best": 0.20,
        "worst": 0.85,
        "basis": (
            "§18/§19 measured 0.470-0.671 across two cross-scene pairs from two "
            "matches. These scenes come from one match with one camera setup, so "
            "they should be no harder and plausibly easier; 0.20 is better than "
            "anything yet seen and 0.85 is close to no saving. A ratio above 1.0 "
            "would mean inter costs more than intra, which §19 saw only from a "
            "broken x265 configuration."
        ),
        "on_breach": "suspect the plates, the chain, or the intra baseline",
    },
    "sequence_ratio_vs_all_intra": {
        "best": 0.30,
        "worst": 0.90,
        "basis": (
            "with one keyframe and per-scene ratio r, the total is "
            "(1 + (N-1)r)/N of all-intra. For N~12 and r in 0.30-0.70 that is "
            "0.36-0.72; the bound is widened either side for the keyframe's share."
        ),
        "on_breach": "check the all-intra baseline is coding each plate alone",
    },
    "keyframe_interval_monotone": {
        "expectation": (
            "cost falls as k rises (k=2 worst, `never` best), because a forced "
            "keyframe is a fresh plate. Break-even against all-intra is around "
            "k=2, so any k>=4 should still pay."
        ),
        "on_breach": "the arithmetic is wrong somewhere; do not report the sweep",
    },
    "best_scored_vs_first": {
        "expectation": (
            "brief §3: `best-scored` must beat `first` by enough to pay for the "
            "search, and its Canny ranking must agree with trial encodes. If it "
            "wins by under a few percent, the honest recommendation is `first`, "
            "which is free and already worth 31-53%."
        ),
        "on_breach": "report `first` and say the search is complexity for nothing",
    },
    "n": {
        "expectation": (
            "this is one video's scene sequence. `presley` requires n>=6 videos "
            "before a significance claim, so nothing here is a significance claim "
            "-- the per-scene spread is reported so the reader can see it."
        ),
    },
}


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summarise(payloads: list[ScenePayload], intra: list[int]) -> dict[str, Any]:
    """Marginal cost per scene and the total, against coding each plate alone."""
    predicted = [(p, intra[p.index]) for p in payloads if not p.is_keyframe]
    ratios = [p.byte_count / fresh for p, fresh in predicted if fresh]
    total = sum(p.byte_count for p in payloads)
    baseline = sum(intra)
    spread: dict[str, Any]
    if len(ratios) > 1:
        spread = {
            "mean": round(statistics.fmean(ratios), 4),
            "stdev": round(statistics.stdev(ratios), 4),
            "stderr": round(statistics.stdev(ratios) / len(ratios) ** 0.5, 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
        }
    else:
        spread = {"mean": round(ratios[0], 4) if ratios else None, "stdev": None,
                  "stderr": None, "min": None, "max": None}
    return {
        "scenes": len(payloads),
        "keyframes": sum(1 for p in payloads if p.is_keyframe),
        "predicted_scenes": len(predicted),
        "total_bytes": total,
        "all_intra_bytes": baseline,
        "sequence_ratio_vs_all_intra": round(total / baseline, 4) if baseline else None,
        "marginal_ratio": spread,
        "per_scene": [
            {
                "index": p.index,
                "type": p.picture_type,
                "reference": p.reference,
                "bytes": p.byte_count,
                "fresh_intra_bytes": intra[p.index],
                "ratio": round(p.byte_count / intra[p.index], 4) if intra[p.index] else None,
            }
            for p in payloads
        ],
    }


def _all_intra(plates: list[np.ndarray], *, codec: str, crf: int) -> list[int]:
    """Each plate coded alone as an I-frame -- the thing amortisation beats.

    Coded one at a time on purpose. Coding them as a sequence with forced
    keyframes would let rate control carry state between them, which is not what
    "send a fresh plate per scene" means.
    """
    sizes: list[int] = []
    for plate in plates:
        encoded = encode_chain([plate], codec=codec, crf=crf)
        if encoded.picture_types[0] != "I":
            raise RuntimeError(
                f"a lone plate coded as {encoded.picture_types[0]}, not I; baseline is not intra"
            )
        sizes.append(encoded.marginal_bytes)
    return sizes


def _control(video: str, *, codec: str, crf: int, height: int) -> dict[str, Any]:
    """Two adjacent but *distinct* frames of one scene. Decides what is readable.

    **The source duplicates frames.** `alcaraz_highlights` is 60000/1001 fps
    carrying content shot slower, so frame *t* and frame *t+1* are frequently
    byte-identical -- measured: md5-equal for the first pair of scene 0. Coding
    an identical picture as a P-frame returns 0.02% of a fresh intra, which
    looks like a superb control and tests nothing: it measures that the encoder
    can skip an unchanged frame, not that it can predict a changed one.

    So the control walks forward to the first frame that actually differs from
    the first, and reports how far it had to walk. That distance is part of the
    result -- if the source ever stops duplicating, the number changes meaning
    and the reader should be able to see it.
    """
    scene = list_scenes(video)[0]
    frames = load_plates(extract_consecutive(video, scene, count=8, height=height))
    offset = next(
        (i for i in range(1, len(frames)) if not np.array_equal(frames[i], frames[0])),
        None,
    )
    if offset is None:
        raise RuntimeError(
            f"all {len(frames)} frames sampled from {video} scene {scene.index} are "
            "identical; there is no pair to build a control from"
        )
    pair = [frames[0], frames[offset]]
    payloads = stream_linear(pair, codec=codec, crf=crf, keyframe_interval=KEYFRAME_NEVER)
    intra = _all_intra(pair[1:], codec=codec, crf=crf)
    ratio = payloads[1].byte_count / intra[0] if intra[0] else None
    return {
        "arm": "CONTROL two adjacent distinct frames, one scene",
        "duplicate_frames_skipped": offset - 1,
        "source_duplicates_frames": offset > 1,
        "frame_types": "".join(p.picture_type for p in payloads),
        "marginal_bytes": payloads[1].byte_count,
        "fresh_intra_bytes": intra[0],
        "ratio": round(ratio, 4) if ratio is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--crf", type=int, default=38)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scenes", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Before the first encode. Not after, and not "roughly what we expected".
    BOUNDS_PATH.write_text(json.dumps(BOUNDS, indent=2) + "\n", encoding="utf-8")
    print(f"bounds written to {BOUNDS_PATH} before any encode", flush=True)

    scenes = list_scenes(args.video)[: args.scenes]
    print(f"{args.video}: {len(scenes)} point-class scenes, height {args.height}", flush=True)
    plates = load_plates(extract_plates(args.video, scenes, height=args.height))
    print(f"plates {plates[0].shape}, n={len(plates)}", flush=True)

    started = time.time()
    control = _control(args.video, codec=args.codec, crf=args.crf, height=args.height)
    print(
        f"CONTROL types={control['frame_types']} ratio={control['ratio']} "
        f"(bound {BOUNDS['control_ratio']['best']}-{BOUNDS['control_ratio']['worst']}) "
        f"duplicate frames skipped={control['duplicate_frames_skipped']}",
        flush=True,
    )

    intra = _all_intra(plates, codec=args.codec, crf=args.crf)
    print(f"all-intra baseline: {sum(intra):,} B over {len(intra)} scenes", flush=True)

    arms: list[dict[str, Any]] = []

    def record(label: str, payloads: list[ScenePayload], extra: dict[str, Any]) -> None:
        summary = _summarise(payloads, intra)
        summary.update({"arm": label, **extra})
        arms.append(summary)
        marginal = summary["marginal_ratio"]
        print(
            f"  {label:<28} total={summary['total_bytes']:>10,} B  "
            f"vs all-intra={summary['sequence_ratio_vs_all_intra']}  "
            f"marginal={marginal['mean']}±{marginal['stderr']}  "
            f"keyframes={summary['keyframes']}",
            flush=True,
        )

    # Reference-mode ablation, one keyframe each, so the modes differ only in
    # what they predict from.
    record(
        "mode=last",
        stream_linear(plates, codec=args.codec, crf=args.crf, keyframe_interval=KEYFRAME_NEVER),
        {"mode": REFERENCE_LAST, "keyframe_interval": "never"},
    )
    for mode in (REFERENCE_FIRST, REFERENCE_BEST_SCORED):
        transmitter = BackgroundStreamTransmitter(mode=mode, codec=args.codec, crf=args.crf)
        record(
            f"mode={mode}",
            [transmitter.push(p) for p in plates],
            {"mode": mode, "keyframe_interval": "never"},
        )

    # Keyframe interval as an axis, not a floor (brief §3).
    for k in (2, 4, 8):
        record(
            f"mode=periodic-i k={k}",
            stream_linear(
                plates, codec=args.codec, crf=args.crf,
                keyframe_interval=k, mode=REFERENCE_PERIODIC_I,
            ),
            {"mode": REFERENCE_PERIODIC_I, "keyframe_interval": k},
        )

    payload = {
        "question": (
            "over a sequence of scenes, what does each scene's background cost "
            "given the ones before it, and what does a periodic keyframe cost?"
        ),
        "video": args.video,
        "ffmpeg": ffmpeg_provenance(),
        "codec": args.codec,
        "crf": args.crf,
        "plate_height": args.height,
        "plate_shape": list(plates[0].shape),
        "n_scenes": len(plates),
        "n_videos": 1,
        "n_caveat": (
            "one video. `presley` requires n>=6 videos before a significance "
            "claim, so no arm here is a significance claim; the per-scene spread "
            "is reported instead."
        ),
        "scene_source": "point-class scenes, one mid-scene frame each, not player-masked plates",
        "control": control,
        "all_intra_bytes": sum(intra),
        "arms": arms,
        "bounds": BOUNDS,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH} in {payload['elapsed_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
