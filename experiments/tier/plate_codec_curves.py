"""Plate codecs as curves, on three levers: quality, size, and encoding time.

**Why this replaces the earlier probe.** BP31's first pass compared plate codecs
at single operating points and read ratios off them. That is not a comparison:
`vvc` came back both cheaper *and* lower quality than `av1`, which says only
that the two were asked for different things. A codec is comparable to another
only through curves, read at **matched fidelity**, and only with the third lever
in the table — because "smaller at the same quality" is not a win if it costs an
order of magnitude more encode time, and on a 4K intra plate it can.

So every codec here is swept over its own knob across a range chosen to overlap
the others in Y-PSNR, and every cross-codec number is an interpolation on each
codec's own curve. `bytes_at_fidelity` **never extrapolates** — a target outside
a codec's measured range comes back as `None` with the range, because a
matched-fidelity claim built on an extrapolation is a claim about a fit.

**Encode and decode time are separated.** The earlier probe timed them together,
which hides the asymmetry that matters: a plate is encoded once by the sender
and decoded once per client, so the two costs sit on different sides of the
system and must not be added into one number.

**This is not a codec ranking**, and the report says so in its own body.
Encoder presets are not equal effort across codecs (`plans/done/BP24-findings.md`
§1), so these curves say what *this project's plate* costs under each encoder as
configured — an engineering choice about our plate, not a statement that one
codec beats another. Reporting encode time beside every size is what keeps that
distinction visible rather than merely asserted.

**Scope.** This prices the `panorama-full` arm only. `plans/done/BP31-findings.md` §1
measured `background.codec` reaching nothing under `panorama-stream`, so nothing
here chooses anything for the streamed arm.

Bounds: `outputs/bp31-ladder/bounds-before-codec-curves.json`, written before the
first encode.

Run::

    python -m experiments.tier.plate_codec_curves --video alcaraz_highlights
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.tier.clip import load_tier_clip
from experiments.tier.plate_codec_sweep import container_kind, pooled_psnr
from src.components.background.plate import build_plate
from src.components.background.sidecar import build_sidecar
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp31-ladder"
BOUNDS_PATH = OUT_DIR / "bounds-before-codec-curves.json"

#: Knobs per codec, coarse to fine. Chosen so the three curves overlap in
#: Y-PSNR: without an overlapping band every matched-fidelity read would be an
#: extrapolation, which is a bound in the file above rather than a detail.
KNOBS: dict[str, tuple[int, ...]] = {
    "jpeg": (10, 20, 30, 50, 65, 75, 85, 90, 95),
    "av1": (59, 51, 45, 40, 35, 30, 25, 20),
    "vvc": (48, 44, 40, 36, 32, 28, 24, 20),
}

#: Where the ladder operates. BP24's reference rung put the plate near 43 dB Y
#: and its payload rungs span roughly 37-48, so these are the fidelities a plate
#: codec choice is actually made at.
FIDELITY_TARGETS: tuple[float, ...] = (38.0, 40.0, 42.0, 43.0, 45.0)

#: Encode-time alarm bands, per codec, in seconds for one 4K panorama plate.
#:
#: **The vvc floor started at 5.0 s and was wrong.** It fired on three points
#: measured at 2.9-4.6 s, on the stated basis that "VVC intra at 4K is the
#: slowest thing on this roster". That basis was the error, not the
#: measurement: `vvencapp` at `faster` encodes this plate in 2.9-8.0 s while
#: `SvtAv1EncApp` at preset 10 takes 9-19 s, so vvc here is the *quicker* of the
#: two. Four independent things say vvc ran — bytes and Y-PSNR both monotone in
#: its own knob across eight points, a recognised bitstream, and BP24's separate
#: plate probe measuring vvc intra at 68,477 B near 38 dB where this curve
#: brackets 41,330 B at 37.46 and 60,242 B at 39.32.
ENCODE_SECONDS_BAND: dict[str, tuple[float, float]] = {
    "jpeg": (0.005, 2.0),
    "av1": (2.0, 90.0),
    "vvc": (1.0, 300.0),
}

#: How many times each encode is repeated for the timing lever.
#:
#: One sample per point is not a time measurement on a shared host. The first
#: run produced av1 encode times of 19.05, 14.69, 10.43, 9.06, **34.62**, 12.41,
#: 11.77 and 11.55 s as the knob went monotonically finer — non-monotone, with a
#: 3x outlier in the middle, because a co-tenant's job moved rather than because
#: the quantiser did. Bytes and PSNR are deterministic and need no repeats; wall
#: time does.
TIME_REPEATS: int = 3


@dataclass(frozen=True)
class CurvePoint:
    """One codec at one knob: all three levers, measured separately."""

    codec: str
    knob: int
    codec_id: str
    payload_bytes: int
    psnr_y_db: float
    psnr_rgb_db: float
    encode_seconds: float
    """Median of `encode_samples`. A median, not a mean: one contention spike
    should not move it."""

    decode_seconds: float
    encode_samples: tuple[float, ...] = ()
    container: str = "unknown"
    detail: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "knob": self.knob,
            "codec_id": self.codec_id,
            "bytes": self.payload_bytes,
            "psnr_y_dB": round(self.psnr_y_db, 3),
            "psnr_rgb_dB": round(self.psnr_rgb_db, 3),
            "encode_seconds": round(self.encode_seconds, 3),
            "encode_samples": [round(x, 3) for x in self.encode_samples],
            "encode_seconds_min": round(min(self.encode_samples), 3) if self.encode_samples else None,
            "encode_seconds_max": round(max(self.encode_samples), 3) if self.encode_samples else None,
            "decode_seconds": round(self.decode_seconds, 3),
            "container": self.container,
            **self.detail,
        }


def _kwargs_for(codec: str, knob: int) -> dict[str, Any]:
    return {"jpeg_quality": knob} if codec == "jpeg" else {"intra_qp": knob}


def measure_point(
    plate: np.ndarray, codec: str, knob: int, *, repeats: int = TIME_REPEATS
) -> CurvePoint:
    """Encode and decode the plate, timing the two halves apart.

    The encode runs ``repeats`` times and the median is kept. Bytes and PSNR are
    deterministic — every repeat produces the identical payload, which is
    asserted rather than assumed — so the repeats buy a time measurement and
    nothing else.
    """
    import statistics

    sidecar = build_sidecar(codec, **_kwargs_for(codec, knob))
    samples: list[float] = []
    payload = b""
    for index in range(max(1, repeats)):
        started = time.time()
        got = sidecar.encode(plate)
        samples.append(time.time() - started)
        if index == 0:
            payload = got
        elif got != payload:
            raise RuntimeError(
                f"{codec} at {knob}: repeat {index} produced a different payload "
                f"({len(got)} B against {len(payload)} B). The encode is not "
                "deterministic, so its bytes cannot be reported as the cost of this rung."
            )
    encode_seconds = statistics.median(samples)

    started = time.time()
    decoded = sidecar.decode(payload)
    decode_seconds = time.time() - started

    reference = np.asarray(plate)
    returned = np.asarray(decoded)
    # A video sidecar may crop an odd dimension to even; score the region that
    # survives rather than failing on a shape both sides agree about otherwise.
    height = min(int(reference.shape[0]), int(returned.shape[0]))
    width = min(int(reference.shape[1]), int(returned.shape[1]))
    ref, out = reference[:height, :width], returned[:height, :width]
    return CurvePoint(
        codec=codec,
        knob=knob,
        codec_id=sidecar.codec_id,
        payload_bytes=len(payload),
        psnr_y_db=pooled_psnr(ref, out, luma=True),
        psnr_rgb_db=pooled_psnr(ref, out),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        encode_samples=tuple(samples),
        container=container_kind(payload),
        detail={"decoded_shape": [int(returned.shape[0]), int(returned.shape[1])]},
    )


def at_fidelity(
    points: Sequence[CurvePoint], target_db: float, *, value: str = "bytes"
) -> dict[str, Any]:
    """``value`` this codec needs to hit ``target_db``, by interpolation.

    Interpolates in (Y-PSNR, log10 value), the shape an RD curve has. Refuses to
    extrapolate: a target outside the measured range returns ``None`` with the
    range, because a matched-fidelity number outside the data is a property of
    the fit rather than of the codec.
    """
    pairs = [
        (
            point.psnr_y_db,
            float(point.payload_bytes if value == "bytes" else point.encode_seconds),
        )
        for point in points
        if np.isfinite(point.psnr_y_db)
    ]
    pairs = [(quality, amount) for quality, amount in pairs if amount > 0]
    pairs.sort()
    if len(pairs) < 2:
        return {value: None, "reason": "fewer than two usable points"}
    lowest, highest = pairs[0][0], pairs[-1][0]
    if not lowest <= target_db <= highest:
        return {
            value: None,
            "reason": f"target {target_db} dB outside measured {lowest:.2f}-{highest:.2f} dB",
            "measured_range_dB": [round(lowest, 2), round(highest, 2)],
        }
    qualities = np.array([item[0] for item in pairs], dtype=np.float64)
    logs = np.log10(np.array([item[1] for item in pairs], dtype=np.float64))
    got = float(10.0 ** float(np.interp(target_db, qualities, logs)))
    return {
        value: int(round(got)) if value == "bytes" else round(got, 3),
        "interpolated": True,
        "measured_range_dB": [round(lowest, 2), round(highest, 2)],
    }


def three_lever_table(by_codec: dict[str, list[CurvePoint]]) -> dict[str, Any]:
    """At each fidelity: bytes AND encode seconds, per codec, plus ratios.

    Both levers in one row on purpose. A size ratio without the time beside it
    is the number that made the earlier probe misleading.
    """
    table: dict[str, Any] = {"axis": "Y-PSNR", "targets": {}}
    for target in FIDELITY_TARGETS:
        row: dict[str, Any] = {}
        for codec, points in by_codec.items():
            size = at_fidelity(points, target, value="bytes")
            row[codec] = {
                "bytes": size.get("bytes"),
                "reason": size.get("reason"),
                "measured_range_dB": size.get("measured_range_dB"),
            }
        base = (row.get("jpeg") or {}).get("bytes")
        for entry in row.values():
            if base and entry.get("bytes"):
                entry["bytes_vs_jpeg"] = round(entry["bytes"] / base, 3)
        table["targets"][f"{target:.1f} dB"] = row
    return table


def encode_time_summary(by_codec: dict[str, list[CurvePoint]]) -> dict[str, Any]:
    """The time lever, reported as measured rather than interpolated.

    Encode time is **not** interpolated against quality, because it is not
    monotone in it here: on a shared host the spread between repeats of one
    point is comparable to the spread across the whole knob range, so a
    log-linear fit through it would be fitting contention. Each point's median
    over `TIME_REPEATS` is summarised across the curve instead, and the
    per-point samples stay in the record so the noise is visible rather than
    smoothed away.
    """
    import statistics

    summary: dict[str, Any] = {}
    for codec, points in by_codec.items():
        medians = [p.encode_seconds for p in points]
        spreads = [
            (max(p.encode_samples) - min(p.encode_samples)) for p in points if p.encode_samples
        ]
        summary[codec] = {
            "encode_seconds_median_over_curve": round(statistics.median(medians), 3),
            "encode_seconds_range_over_curve": [round(min(medians), 3), round(max(medians), 3)],
            "worst_within_point_spread_seconds": round(max(spreads), 3) if spreads else None,
            "decode_seconds_median": round(
                statistics.median([p.decode_seconds for p in points]), 3
            ),
            "n_repeats": TIME_REPEATS,
        }
    base = (summary.get("jpeg") or {}).get("encode_seconds_median_over_curve")
    for entry in summary.values():
        if base:
            entry["encode_vs_jpeg"] = round(entry["encode_seconds_median_over_curve"] / base, 1)
    return summary


def check_bounds(by_codec: dict[str, list[CurvePoint]]) -> list[str]:
    """The pre-written bounds, evaluated in the run."""
    alarms: list[str] = []

    for codec, points in by_codec.items():
        ordered = sorted(points, key=lambda p: p.payload_bytes)
        for previous, current in zip(ordered, ordered[1:]):
            if current.psnr_y_db < previous.psnr_y_db:
                alarms.append(
                    f"{codec}: {current.knob} costs more than {previous.knob} "
                    f"({current.payload_bytes} vs {previous.payload_bytes} B) and scores "
                    f"lower ({current.psnr_y_db:.2f} vs {previous.psnr_y_db:.2f} dB). The "
                    "knob is not reaching this encoder; the whole curve is disqualified."
                )
        low, high = ENCODE_SECONDS_BAND.get(codec, (0.0, float("inf")))
        for point in points:
            if point.encode_seconds < low:
                alarms.append(
                    f"{codec} at {point.knob}: encode took {point.encode_seconds:.3f}s, "
                    f"below the {low}s floor. Suspect nothing ran — check the payload's "
                    f"own first bytes (container read as {point.container!r})."
                )
            elif point.encode_seconds > high:
                alarms.append(
                    f"{codec} at {point.knob}: encode took {point.encode_seconds:.1f}s, "
                    f"above the {high}s ceiling. Suspect the preset did not reach the binary."
                )

    spans = {
        codec: (min(p.psnr_y_db for p in pts), max(p.psnr_y_db for p in pts))
        for codec, pts in by_codec.items()
        if pts
    }
    if len(spans) > 1:
        overlap_low = max(low for low, _ in spans.values())
        overlap_high = min(high for _, high in spans.values())
        if overlap_high - overlap_low < 3.0:
            alarms.append(
                f"the curves overlap over only {max(0.0, overlap_high - overlap_low):.2f} dB "
                f"({overlap_low:.2f}-{overlap_high:.2f}); spans are {spans}. Under 3 dB there "
                "is no band where a matched-fidelity read is an interpolation for every "
                "codec. Widen the knob ranges and re-run rather than reporting a ratio."
            )
    return alarms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--scene", default="scene_000")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--codecs", nargs="+", default=list(KNOBS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds go to disk before the first encode.")
    print(f"bounds: {BOUNDS_PATH}", flush=True)

    clip = load_tier_clip(video=args.video, scene=args.scene, n_frames=args.frames)
    plate, homographies = build_plate(clip.frames, clip.union_mask, register=True)
    print(
        f"{args.video}/{args.scene}: panorama {plate.shape} from {args.frames} frames "
        f"({len(homographies)} homographies)",
        flush=True,
    )

    by_codec: dict[str, list[CurvePoint]] = {}
    for codec in args.codecs:
        points: list[CurvePoint] = []
        for knob in KNOBS[codec]:
            point = measure_point(plate, codec, knob)
            points.append(point)
            print(
                f"  {codec:<5} {knob:>3}  {point.payload_bytes:>9} B  "
                f"Y {point.psnr_y_db:6.2f} dB  enc {point.encode_seconds:7.2f}s  "
                f"dec {point.decode_seconds:6.2f}s  [{point.container}]",
                flush=True,
            )
        by_codec[codec] = points

    alarms = check_bounds(by_codec)
    report = {
        "video": args.video,
        "scene": args.scene,
        "plate_shape": [int(x) for x in plate.shape],
        "bounds_file": str(BOUNDS_PATH),
        "curves": {codec: [p.record() for p in pts] for codec, pts in by_codec.items()},
        "matched_fidelity_bytes": three_lever_table(by_codec),
        "encode_time": encode_time_summary(by_codec),
        "alarms": alarms,
        "reading_note": (
            "Three levers, and all three belong in any sentence comparing two "
            "codecs here: bytes at matched Y-PSNR, and the encode time that "
            "bought them. Cross-codec numbers are interpolations on each "
            "codec's own curve and are never extrapolated. Presets are not "
            "equal effort across codecs, so this says what THIS plate costs "
            "under each encoder as configured — an operating-point choice for "
            "the `panorama-full` arm, not a general codec ranking. It says "
            "nothing about `panorama-stream`, which never consults this "
            "sidecar (findings §1)."
        ),
    }
    dest = Path(args.out) if args.out else OUT_DIR / f"plate-codec-curves-{args.video}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== bytes at matched fidelity (Y-PSNR), interpolated per codec ===", flush=True)
    for target, row in report["matched_fidelity_bytes"]["targets"].items():
        print(f"  {target}", flush=True)
        for codec in args.codecs:
            entry = row.get(codec, {})
            if entry.get("bytes") is None:
                print(f"    {codec:<5} — {entry.get('reason')}", flush=True)
            else:
                print(
                    f"    {codec:<5} {entry['bytes']:>9} B   x{entry.get('bytes_vs_jpeg', '?')} vs jpeg",
                    flush=True,
                )

    print("\n=== encode time, MEASURED (median of repeats, never interpolated) ===", flush=True)
    for codec in args.codecs:
        entry = report["encode_time"].get(codec, {})
        low, high = entry.get("encode_seconds_range_over_curve", [None, None])
        print(
            f"  {codec:<5} median {entry.get('encode_seconds_median_over_curve'):>7}s over the "
            f"curve, range {low}-{high}s, worst within-point spread "
            f"{entry.get('worst_within_point_spread_seconds')}s, "
            f"x{entry.get('encode_vs_jpeg', '?')} vs jpeg",
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
