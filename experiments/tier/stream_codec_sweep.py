"""Which `stream_codec` x `stream_crf` should the ladder's PointStream arm use?

**Why this exists rather than the sweep the brief asked for.**
`plans/done/BP31-paired-ladder-across-scenes.md` §0 asks for a `background.codec`
sweep to decide which plate codec the ladder arm uses, with the cross-scene
stream on at the same time. `plans/done/BP31-findings.md` §1 measured that this
configuration does not exist: `PanoramaStream` never touches the still-image
sidecar, so `background.codec` selects nothing under it and the plate goes
through `BackgroundStreamTransmitter` at `stream_codec` / `stream_crf` instead.
Those two are therefore the knob that carries 88-91% of the payload on the arm
the ladder will actually run, and this is the sweep that decides them.

**The control runs first and is the first thing to read.** Two adjacent distinct
frames of one scene must come back at a few percent of a fresh intra. BP30's
findings §19 records why: a misconfigured x265 made a P-frame come back *larger*
than a fresh intra, and without the control that would have been written up as a
finding about low delay rather than about a broken flag. A codec whose control
fails has every ratio below it disqualified, and this module says so per codec
rather than averaging it away.

**Per codec, against itself, never ranked across codecs.** The low-delay flag
sets differ per encoder (`libaom-av1 -usage realtime` against `libx265
-preset veryfast`) and are not equal effort, so an ordering of the magnitudes
would be measuring the flags (`plans/done/BP24-findings.md` §1). Each codec's number
is its own chain against its own intra baseline.

**Scope, stated rather than implied.** These are point-class scene *frames*, not
stitched panoramas and not player-masked plates — the same conservative choice
BP30 made, so the ratio sits on the axis `plans/done/RESEARCH-HISTORY.md` §2.22 already reports. A real
panorama has less moving content to mispredict, so this is the pessimistic case.
One video is enough to choose an operating point and is **not** enough for a
claim: BP30's per-video spread (0.294-0.624) is larger than every effect
measured inside it.

Bounds: `outputs/bp31-ladder/bounds-before-run.json`, written before the first
encode here.

Run::

    python -m experiments.tier.stream_codec_sweep --video alcaraz_highlights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.background_stream import _all_intra, _control
from experiments.tier.scene_plates import extract_plates, list_scenes, load_plates
from src.components.background.stream import (
    CODECS,
    KEYFRAME_NEVER,
    ffmpeg_provenance,
    stream_linear,
)
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp31-ladder"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"

#: The three the component ships. Swept because BP30 fixed av1 by assumption:
#: its §19 contrast was a *pair*, not a sequence, and the ladder runs sequences.
DEFAULT_CODECS: tuple[str, ...] = ("av1", "hevc", "avc")

#: Around BP30's operating point (CRF 38) and wide enough that the ladder can
#: pick a rung rather than inherit one. A plate is most of the payload, so this
#: axis is also the payload ladder's coarse end.
DEFAULT_CRFS: tuple[int, ...] = (30, 38, 45, 51)

#: The amortisation ratio's alarm band, from the bounds file. Duplicated here on
#: purpose: a bound that lives only in a JSON file beside the result is a bound
#: that gets skipped exactly when the number is exciting.
RATIO_BAND: tuple[float, float] = (0.25, 0.75)

#: The control's band. Above the top of it, the codec is not predicting and its
#: ratios are disqualified rather than merely suspect.
CONTROL_BAND: tuple[float, float] = (0.005, 0.15)


def sweep_point(
    plates: list[np.ndarray], *, codec: str, crf: int
) -> dict[str, Any]:
    """One codec at one CRF over the whole scene sequence, against its own intra."""
    started = time.time()
    payloads = stream_linear(plates, codec=codec, crf=crf, keyframe_interval=KEYFRAME_NEVER)
    chain_bytes = int(sum(p.byte_count for p in payloads))
    intra = _all_intra(plates, codec=codec, crf=crf)
    intra_bytes = int(sum(intra))
    seconds = time.time() - started
    ratio = (chain_bytes / intra_bytes) if intra_bytes else None
    return {
        "codec": codec,
        "crf": crf,
        "n_scenes": len(plates),
        "chain_bytes": chain_bytes,
        "fresh_intra_bytes": intra_bytes,
        # The headline: what N scenes cost as a chain, over N fresh plates.
        "amortisation_ratio": round(ratio, 4) if ratio is not None else None,
        "frame_types": "".join(p.picture_type for p in payloads),
        "marginal_bytes": [int(p.byte_count) for p in payloads],
        "fresh_intra_per_scene": [int(b) for b in intra],
        "seconds": round(seconds, 1),
    }


def check_bounds(
    points: list[dict[str, Any]], controls: dict[str, dict[str, Any]]
) -> list[str]:
    """The bounds file, evaluated here rather than left to whoever reads the table."""
    alarms: list[str] = []
    low, high = RATIO_BAND
    control_low, control_high = CONTROL_BAND

    for codec, control in controls.items():
        ratio = control.get("ratio")
        if not isinstance(ratio, (int, float)):
            alarms.append(f"{codec}: control produced no ratio ({control!r}); nothing below it is readable.")
            continue
        if ratio > control_high:
            alarms.append(
                f"{codec}: CONTROL {ratio:.4f} is above {control_high}. Two adjacent "
                "frames of one scene should cost a few percent of a fresh intra. This "
                "codec is not doing inter prediction, so every amortisation ratio for "
                "it is disqualified — findings §19 is exactly this failure."
            )
        elif ratio < control_low:
            alarms.append(
                f"{codec}: CONTROL {ratio:.4f} is below {control_low}. Suspect the two "
                "frames are identical and the encoder is skipping rather than predicting."
            )

    for point in points:
        codec, crf = point["codec"], point["crf"]
        types = point.get("frame_types") or ""
        if "B" in types:
            alarms.append(
                f"{codec} crf{crf}: frame types {types} contain a B-frame, so the encode "
                "was not causal and the low-delay constraint did not reach the encoder."
            )
        if types and types[0] != "I":
            alarms.append(f"{codec} crf{crf}: chain starts with {types[0]}, not I.")
        ratio = point.get("amortisation_ratio")
        if not isinstance(ratio, (int, float)):
            continue
        if ratio < low:
            alarms.append(
                f"{codec} crf{crf}: ratio {ratio:.4f} is below {low}, beating BP30's best "
                "case on any codec. Check the intra baseline really coded intra and that "
                "no scene was dropped from the chain, before believing it."
            )
        elif ratio > high:
            alarms.append(
                f"{codec} crf{crf}: ratio {ratio:.4f} is above {high} — the chain is barely "
                "predicting. Read this codec's control before reading this number."
            )

    for codec in {point["codec"] for point in points}:
        ordered = sorted(
            (p for p in points if p["codec"] == codec), key=lambda p: int(p["crf"])
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current["chain_bytes"] >= previous["chain_bytes"]:
                alarms.append(
                    f"{codec}: crf {current['crf']} cost {current['chain_bytes']} B, not "
                    f"less than crf {previous['crf']}'s {previous['chain_bytes']} B. A "
                    "coarser rung must be cheaper; if it is not, the CRF is not reaching "
                    "this encoder."
                )
    return alarms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--codecs", nargs="+", default=list(DEFAULT_CODECS))
    parser.add_argument("--crfs", nargs="+", type=int, default=list(DEFAULT_CRFS))
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    unknown = [name for name in args.codecs if name not in CODECS]
    if unknown:
        raise SystemExit(f"unknown stream codec(s) {unknown}; known: {sorted(CODECS)}")

    if not BOUNDS_PATH.is_file():
        raise SystemExit(
            f"{BOUNDS_PATH} does not exist. Bounds go to disk before the first encode."
        )

    print(f"bounds: {BOUNDS_PATH}", flush=True)
    print(f"ffmpeg: {ffmpeg_provenance()}", flush=True)

    scenes = list_scenes(args.video)[: args.scenes]
    print(
        f"{args.video}: {len(scenes)} point-class scenes at height {args.height}",
        flush=True,
    )
    plates = load_plates(extract_plates(args.video, scenes, height=args.height))
    print(f"plates loaded: {len(plates)} x {plates[0].shape}", flush=True)

    controls: dict[str, dict[str, Any]] = {}
    for codec in args.codecs:
        # The control uses this codec's own mid CRF; it asks whether the encoder
        # predicts at all, which is not a question about the rung.
        control = _control(args.video, codec=codec, crf=38, height=args.height)
        controls[codec] = control
        print(f"  CONTROL {codec:<5} ratio={control.get('ratio')} types={control.get('frame_types')}", flush=True)

    points: list[dict[str, Any]] = []
    for codec in args.codecs:
        for crf in args.crfs:
            point = sweep_point(plates, codec=codec, crf=crf)
            points.append(point)
            print(
                f"  {codec:<5} crf{crf:<3} chain {point['chain_bytes']:>10} B  "
                f"intra {point['fresh_intra_bytes']:>10} B  "
                f"ratio {point['amortisation_ratio']}  {point['seconds']:6.1f}s",
                flush=True,
            )

    alarms = check_bounds(points, controls)
    report = {
        "video": args.video,
        "n_scenes": len(plates),
        "height": args.height,
        "plate_shape": [int(x) for x in plates[0].shape],
        "ffmpeg": ffmpeg_provenance(),
        "bounds_file": str(BOUNDS_PATH),
        "controls": controls,
        "points": points,
        "alarms": alarms,
        "reading_note": (
            "Per codec against its own intra baseline. Do NOT rank the codecs "
            "against each other: their low-delay flag sets are not equal effort "
            "(findings §1). One video, scene frames rather than panoramas — this "
            "chooses an operating point and does not support a claim."
        ),
    }
    dest = Path(args.out) if args.out else OUT_DIR / f"stream-codec-sweep-{args.video}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")

    if alarms:
        print("\n=== ALARMS ===", flush=True)
        for alarm in alarms:
            print(f"  ! {alarm}", flush=True)
    else:
        print("\nno alarms: every point inside the bounds written before the run", flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
