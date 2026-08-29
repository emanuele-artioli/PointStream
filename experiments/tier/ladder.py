"""The paired ladder: PointStream against a codec, one codec on both arms.

`PLAN.md` §7 P0 items 2 and 3. BP24 built the rate axis; this runs it.

**Why the arms are paired.** `plans/BP24-findings.md` §1 measured that the
project's codec presets are not equal effort — `avc: veryfast` against
`hevc: ultrafast` gave HEVC over AVC at -4.2% BD-rate where the literature
expects 30-50%, and the bias always understates the newer codec. That kills any
cross-codec claim built on these presets. It does **not** touch the claim the
paper actually needs, which is *PointStream against a codec*:

    For codec X, measure (a) X coding the source, and (b) PointStream using X
    for its coded components — same encoder, same preset, same rate control,
    same pixel format, same rungs. BD-rate between those two curves is the
    PointStream gain, and the preset cancels.

Both arms here are built from **one** `EncodeRequest` per rung, so "same preset"
is not a promise in a docstring: the anchor is encoded with the request the
PointStream arm's residual config produces, and the report carries it.

**What this refuses to do.** It does not rank the per-codec gains against each
other. Comparing the magnitudes re-imports the preset unfairness through the
back door, so the report states each gain beside its preset and stops. There is
no "PointStream beats VVC but not AV1" line to be read out of this file, and
adding one would be wrong rather than merely unsupported.

**What each arm's quality is measured on.** The anchor's quality is scored on
what the decoder returned, never on the array that went in — `coded_roundtrip`
returns cost and frames together for exactly that reason (findings §4). The
PointStream arm's quality is the delivered clip the runner produced at that
rung, scored against the same source with the same PSNR function. One
convention, computed here, for both arms: BP23 found two conventions inside one
ladder disagreeing by 0.65 dB.

**What is not a rate.** If the runner's ledger withholds
`transport_to_source_ratio` — because some component is still an array size —
that rung is recorded and excluded, and the run says so. A curve built from a
mixed total is not an RD curve.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import TierClip, load_tier_clip
from experiments.tier.run import run_config
from src.components.codec.frames import rgb_to_luma
from src.components.codec.measure import PRESETS
from src.components.metrics.bd_rate import (
    InsufficientOverlapError,
    RDCurve,
    compare_rd_curves,
)
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import PointstreamConfig
from src.pipeline.residual.spectrum import (
    Coarseness,
    ResidualPoint,
    ResidualVariant,
    coarseness_ladder,
)
from src.runner.config_io import load_tier

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "bp24-ladder"

#: Rungs, as QP values. Wide on purpose: the absolute-span guard in `bd_rate`
#: refuses a comparison over less than 3 dB, and a narrow sweep is how a ladder
#: ends up resolving nothing. Capped at 46 because `avc` and `hevc` take QP
#: 0-51 while `av1` and `vvc` take 0-63 — a rung outside one codec's range would
#: silently become a different rung there.
DEFAULT_RUNGS = (15, 25, 35, 45, 55)

#: `(background.jpeg_quality, residual rate)`, coarsest first.
#:
#: Sweeping `residual.rate` alone does **not** produce a PointStream RD curve on
#: this content, and the first run of this ladder is what showed it: over QP
#: 30 to 46 the payload moved 526,079 -> 495,739 B, a span of 6%, because the
#: plate was 463,334 B of it — 93% — and the plate does not move with the
#: residual's knob. The two rungs landed 0.55 dB apart and the comparison had
#: nothing to integrate over.
#:
#: So a rung has to move everything that trades rate for quality. That is what
#: the shipped tiers already do (`fast` is jpeg 50 with a coarse residual,
#: `quality` is jpeg 95 with a fine one); this table is that pairing, extended
#: at both ends so the curve spans enough quality to compare.
PAYLOAD_RUNGS = ((30, 55), (50, 46), (75, 38), (90, 28), (98, 18))

#: QP, not CRF. `src/contracts/codecs.py` gives CRF to `avc` and `av1` only:
#: `hevc` (kvazaar) and `vvc` (vvenc) declare QP and BITRATE. QP is the one mode
#: every codec on the roster accepts, so it is the only rate control a ladder
#: spanning all four can use. A QP is still not a quality level — which is why
#: this file compares curves and never single-QP totals.
LADDER_RATE_CONTROL = RateControl.QP


def pooled_psnr(
    reference: np.ndarray, candidate: np.ndarray, *, luma: bool = False
) -> float:
    """One PSNR over the whole clip's MSE. The convention for both arms.

    Not the mean of per-frame PSNRs. Either is defensible; using two inside one
    ladder is not, and BP23 found exactly that (47.63 against 48.28 dB on
    identical pixels).

    ``luma=True`` scores BT.601 luma, which is **the axis the BD-rate is taken
    on**. RGB is recorded beside it and is not comparable between the arms —
    see `luma_is_the_axis` below.
    """
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError(f"psnr shape mismatch: {ref.shape} vs {got.shape}")
    if luma:
        ref = rgb_to_luma(ref).astype(np.float64)
        got = rgb_to_luma(got).astype(np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


#: Why the BD-rate axis is Y-PSNR and not RGB-PSNR.
#:
#: Measured on this ladder, 4K tennis, av1 at `yuv420p`: **QP 15 scored 40.72 dB
#: in RGB and QP 30 scored 40.16 dB.** Fifteen QP steps moved the rate by a
#: factor of six and the RGB quality by half a dB, because the arm is capped by
#: the 4:2:0 chroma round-trip rather than by the quantizer. An RGB curve on
#: that arm resolves nothing — exactly the degenerate shape
#: `plans/BP24-findings.md` §2 is about.
#:
#: Worse, the cap is **asymmetric between the arms**. PointStream delivers a
#: JPEG plate with source crops pasted over it, so most of its pixels never make
#: the 4:2:0 round-trip at all; it sits above a ceiling the anchor cannot reach,
#: and the two curves have no overlapping quality range to integrate over. A
#: BD-rate taken there would be measuring the colour format.
#:
#: Y-PSNR is also what the BD-rate literature reports, so this is the
#: conventional axis rather than a convenient one. RGB is recorded per rung so
#: the chroma cost stays visible instead of being quietly dropped.
LUMA_IS_THE_AXIS = (
    "BD-rate is taken on BT.601 Y-PSNR. RGB-PSNR is recorded per rung but is "
    "not the axis: at yuv420p the anchor's RGB score is capped by the 4:2:0 "
    "round-trip (40.72 dB at QP 15 against 40.16 at QP 30 on this clip), and "
    "PointStream avoids most of that round-trip, so an RGB comparison would be "
    "measuring the colour format rather than the coding."
)


def motion_level(frames: np.ndarray) -> float:
    """Mean absolute difference between consecutive frames, in grey levels.

    The clip axis this ladder is swept over. `plans/BP24-findings.md` §7 records
    both of BP24's headline ratios as the *easy* case — a 2.5%-non-zero residual
    against a static plate — so "high motion" has to be a measured property of
    the clip picked, not an adjective attached to it afterwards.
    """
    clip = np.asarray(frames, dtype=np.int16)
    if clip.shape[0] < 2:
        return 0.0
    return float(np.abs(clip[1:] - clip[:-1]).mean())


@dataclass(frozen=True)
class Rung:
    """One point on one arm."""

    rate_value: int
    coded_bytes: int
    quality_db: float
    seconds: float
    detail: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "rate_value": self.rate_value,
            "coded_bytes": self.coded_bytes,
            "psnr_dB": self.quality_db,
            "seconds": round(self.seconds, 2),
            **self.detail,
        }


def _announce(rung: Rung, arm: str) -> Rung:
    """Print a rung the moment it lands, and hand it straight back.

    A 4K rung takes minutes, and printing only once the whole codec's pair
    finished left a job that looked hung for half an hour. AGENTS.md asks for a
    progress line at least every ten minutes so a real hang is visible in
    minutes rather than hours; this is that line.
    """
    label = rung.detail.get("coarseness") or rung.detail.get("rung") or f"r={rung.rate_value}"
    print(
        f"  {arm:<7} {label:>9}  {rung.coded_bytes:>10} B  "
        f"{rung.quality_db:6.2f} dB  {rung.seconds:6.1f}s",
        flush=True,
    )
    return rung


def anchor_rung(clip: TierClip, request: EncodeRequest) -> Rung:
    """Codec X coding the source clip, at one rung.

    The anchor for the pair. Uses the *same* `EncodeRequest` the PointStream arm
    hands its residual, so preset, rate control and pixel format are identical
    by construction rather than by two config files agreeing.
    """
    from src.components.codec.frames import even_size
    from src.components.codec.measure import coded_roundtrip

    started = time.time()
    coded_bytes, decoded = coded_roundtrip(clip.frames, request=request)
    seconds = time.time() - started
    # `coded_roundtrip` pads an odd dimension to an even one before encoding, so
    # the reference has to take the same step or the two shapes disagree. 4K is
    # already even; doing it unconditionally means a smaller test clip does not
    # fail here for a reason that has nothing to do with the ladder.
    reference = even_size(np.asarray(clip.frames))
    return Rung(
        rate_value=int(request.rate or 0),
        coded_bytes=int(coded_bytes),
        # Scored on what the decoder returned. `coded_roundtrip` hands back both
        # halves together so this cannot accidentally score the input array.
        quality_db=pooled_psnr(reference, decoded, luma=True),
        seconds=seconds,
        detail={
            "arm": "source-through-codec",
            "psnr_rgb_dB": pooled_psnr(reference, decoded),
            "codec": request.codec_name,
            "preset": request.preset,
            "rate_control": request.rate_control.value,
            "pix_fmt": request.pix_fmt,
        },
    )


def pointstream_rung(clip: TierClip, config: PointstreamConfig, rate_value: int) -> Rung:
    """PointStream at one rung, with its residual coded by the paired codec.

    The rate is the ledger's `transport_total` — everything transmitted, not the
    residual alone. Including the plate and the appearance is the point: they
    are what PointStream sends *instead of* the frames the anchor sends, and a
    curve that quietly dropped them would be comparing half a payload against a
    whole one.

    Quality is scored on `delivered_frames`, **not** `RunResult.frames`. The
    latter carries the residual as the residual stage produced it, before
    `residual.codec` ran on it; pairing it with `transport_total` would put the
    rate and the quality at different operating points (findings §4). The two
    differ by exactly the residual's coding loss, which is the axis this ladder
    sweeps — so on this curve the mistake would not look like a mistake.
    """
    tuned = config.with_(residual=replace(config.residual, rate=int(rate_value)))
    started = time.time()
    outcome = run_config(f"pointstream@{rate_value}", tuned, clip)
    seconds = time.time() - started
    sizes = outcome.result.sizes
    delivered = outcome.result.delivered_frames
    return Rung(
        rate_value=int(rate_value),
        coded_bytes=int(sizes.transport_total),
        quality_db=pooled_psnr(clip.frames, delivered, luma=True),
        seconds=seconds,
        detail={
            "arm": "pointstream",
            "psnr_rgb_dB": pooled_psnr(clip.frames, delivered),
            "codec": tuned.residual.codec,
            "preset": tuned.residual.preset,
            "rate_control": tuned.residual.rate_control.value,
            "pix_fmt": tuned.residual.pix_fmt,
            "is_rate": bool(sizes.is_rate),
            "raw_parts": list(sizes.raw_parts),
            # The gap between the two reconstructions. Zero means the codec did
            # not run on the residual and this rung is not on the rate axis at
            # all; a non-zero value is the coding loss the rung is sweeping.
            "precodec_vs_delivered_dB": pooled_psnr(
                np.asarray(outcome.result.frames), delivered, luma=True
            ),
            "parts": {
                "residual": sizes.residual,
                "panorama": sizes.panorama,
                "actor_reference": sizes.actor_reference,
                "metadata": sizes.metadata,
            },
            "source_bytes": sizes.source,
        },
    )


def check_bounds(
    anchor_rungs: list[Rung], stream_rungs: list[Rung], source_bytes: int
) -> list[str]:
    """The alarms from `outputs/bp24-ladder/bounds-before-run.json`, evaluated here.

    Written into the run rather than left to whoever reads the table. A bound
    that only exists in a JSON file next to the result is a bound that gets
    skipped exactly when the number is exciting — which is the direction this
    project has already been caught failing in.
    """
    alarms: list[str] = []

    for label, rungs in (("anchor", anchor_rungs), ("pointstream", stream_rungs)):
        ordered = sorted(rungs, key=lambda item: item.rate_value)
        for previous, current in zip(ordered, ordered[1:]):
            if current.coded_bytes >= previous.coded_bytes:
                alarms.append(
                    f"{label}: rate value {current.rate_value} cost "
                    f"{current.coded_bytes} B, not less than {previous.rate_value}'s "
                    f"{previous.coded_bytes} B. A coarser rung must be cheaper; "
                    "if it is not, the rung is not reaching the encoder."
                )
            if current.quality_db >= previous.quality_db:
                alarms.append(
                    f"{label}: rate value {current.rate_value} scored "
                    f"{current.quality_db:.2f} dB, not below {previous.rate_value}'s "
                    f"{previous.quality_db:.2f} dB. A coarser rung must be worse."
                )

    for rung in anchor_rungs:
        if source_bytes and rung.coded_bytes > 0.20 * source_bytes:
            alarms.append(
                f"anchor at {rung.rate_value} coded {rung.coded_bytes} B, over 20% "
                f"of the {source_bytes} B source. A real encoder does not land "
                "there on 4K; suspect pixels passing through."
            )

    for rung in stream_rungs:
        # A rung that transmits no residual has nothing for a codec to run on,
        # so an infinite gap there is the correct answer rather than an alarm.
        # The coarseness sweep's `absent` rung is exactly that, and it belongs
        # on the curve: it is the unaided control.
        sends_residual = int(rung.detail.get("parts", {}).get("residual", 0)) > 0
        if sends_residual and not np.isfinite(
            rung.detail.get("precodec_vs_delivered_dB", float("inf"))
        ):
            alarms.append(
                f"pointstream at {rung.rate_value}: the pre-codec and delivered "
                "reconstructions are identical, so residual.codec did not run "
                "on this rung and its byte count is not on the rate axis."
            )
        if not rung.detail.get("is_rate"):
            alarms.append(
                f"pointstream at {rung.rate_value}: ledger withheld the ratio "
                f"({rung.detail.get('raw_parts')}), so this total is not a rate."
            )

    # The unaided reconstruction is the floor: a residual can only add
    # information. BP23 measured 34.88 dB on this clip with the residual absent.
    for rung in stream_rungs:
        if rung.quality_db < 30.0:
            alarms.append(
                f"pointstream at {rung.rate_value} delivered "
                f"{rung.quality_db:.2f} dB, below the unaided reconstruction's "
                "neighbourhood. The residual should not be able to make the "
                "reconstruction worse — check what base it is being applied to."
            )
    return alarms


def _curve(rungs: list[Rung], label: str) -> RDCurve:
    ordered = sorted(rungs, key=lambda item: item.coded_bytes)
    return RDCurve(
        rates=tuple(float(item.coded_bytes) for item in ordered),
        qualities=tuple(float(item.quality_db) for item in ordered),
        label=label,
    )


def coarseness_rung(
    clip: TierClip, config: PointstreamConfig, point: ResidualPoint
) -> Rung:
    """PointStream at one rung of the residual-coarseness ladder (P0 item 3).

    The coarseness rung bundles four knobs — the codec's rate, the block gate's
    size and threshold, and the background downscale — because that is how
    `coarseness_ladder()` defines the axis. Sweeping it is a different question
    from sweeping the codec's rate alone: this one asks what the *residual
    representation* costs, the other what the *codec* costs on a fixed one.
    """
    lattice = replace(config.lattice, residual=point.variant is not ResidualVariant.NONE)
    tuned = config.with_(
        lattice=lattice,
        residual=point.config if point.config is not None else config.residual,
    )
    started = time.time()
    outcome = run_config(f"pointstream@{point.coarseness.value}", tuned, clip)
    seconds = time.time() - started
    sizes = outcome.result.sizes
    delivered = outcome.result.delivered_frames
    return Rung(
        # Not a QP. `rate_value` is a rank on one convention shared with the QP
        # sweep: **higher means coarser**, so "a coarser rung must be cheaper
        # and worse" stays one check rather than two with opposite signs.
        # `coarseness_ladder()` runs absent-to-lossless, so the rank is that
        # index reversed. The rung's identity is the `coarseness` name below.
        rate_value=len(Coarseness) - 1 - list(Coarseness).index(point.coarseness),
        coded_bytes=int(sizes.transport_total),
        quality_db=pooled_psnr(clip.frames, delivered, luma=True),
        seconds=seconds,
        detail={
            "arm": "pointstream",
            "psnr_rgb_dB": pooled_psnr(clip.frames, delivered),
            "coarseness": point.coarseness.value,
            "describes": point.describe(),
            "is_rate": bool(sizes.is_rate),
            "raw_parts": list(sizes.raw_parts),
            "parts": {
                "residual": sizes.residual,
                "panorama": sizes.panorama,
                "actor_reference": sizes.actor_reference,
                "metadata": sizes.metadata,
            },
            "precodec_vs_delivered_dB": pooled_psnr(
                np.asarray(outcome.result.frames), delivered, luma=True
            ),
        },
    )


def payload_rung(
    clip: TierClip,
    config: PointstreamConfig,
    *,
    jpeg_quality: int,
    rate_value: int,
    rank: int,
) -> Rung:
    """PointStream at one rung of the whole transmitted payload's quality.

    Moves the plate's sidecar quality and the residual's rate together, because
    the plate is most of what PointStream sends and a curve that held it fixed
    would be a curve of the 7% that moved.

    `rank` follows the same convention as every other rung here: **higher means
    coarser**, so one monotonicity check covers all three sweeps.
    """
    tuned = config.with_(
        background=replace(config.background, jpeg_quality=int(jpeg_quality)),
        residual=replace(config.residual, rate=int(rate_value)),
    )
    started = time.time()
    outcome = run_config(f"pointstream@q{jpeg_quality}/r{rate_value}", tuned, clip)
    seconds = time.time() - started
    sizes = outcome.result.sizes
    delivered = outcome.result.delivered_frames
    return Rung(
        rate_value=rank,
        coded_bytes=int(sizes.transport_total),
        quality_db=pooled_psnr(clip.frames, delivered, luma=True),
        seconds=seconds,
        detail={
            "arm": "pointstream",
            "psnr_rgb_dB": pooled_psnr(clip.frames, delivered),
            "rung": f"jpeg{jpeg_quality}/qp{rate_value}",
            "background_jpeg_quality": int(jpeg_quality),
            "residual_rate": int(rate_value),
            "codec": tuned.residual.codec,
            "preset": tuned.residual.preset,
            "rate_control": tuned.residual.rate_control.value,
            "is_rate": bool(sizes.is_rate),
            "raw_parts": list(sizes.raw_parts),
            "parts": {
                "residual": sizes.residual,
                "panorama": sizes.panorama,
                "actor_reference": sizes.actor_reference,
                "metadata": sizes.metadata,
            },
            "source_bytes": sizes.source,
            "precodec_vs_delivered_dB": pooled_psnr(
                np.asarray(outcome.result.frames), delivered, luma=True
            ),
        },
    )


def pair_for_codec(
    clip: TierClip,
    config: PointstreamConfig,
    *,
    codec_name: str,
    rungs: tuple[int, ...],
    sweep: str = "qp",
) -> dict[str, Any]:
    """One codec, both arms, same preset — and the BD-rate between them.

    The preset comes from `src.components.codec.measure.PRESETS`, not from the
    tier config, for two reasons. A tier config names one codec's preset
    vocabulary (`av1: "8"`), which means nothing to kvazaar or vvenc. And
    `PRESETS` is what BP21's 4K ladder used, so a rate measured here sits beside
    that one rather than establishing a second convention. It is applied to
    **both arms**, which is the property that makes the pair fair; it is *not*
    equal effort across codecs, which is why the gains must not be ranked.
    """
    base = replace(
        config.residual,
        codec=codec_name,
        preset=PRESETS[codec_name],
        rate_control=LADDER_RATE_CONTROL,
    )
    paired = config.with_(residual=base)

    anchor_rungs: list[Rung] = []
    stream_rungs: list[Rung] = []
    failures: list[dict[str, Any]] = []

    for rate_value in rungs:
        request = replace(base, rate=int(rate_value)).encode_request()
        try:
            request.validate()
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            failures.append({"rate_value": rate_value, "arm": "both", "error": repr(exc)})
            continue
        try:
            anchor_rungs.append(_announce(anchor_rung(clip, request), "anchor"))
        except Exception as exc:  # noqa: BLE001
            failures.append({"rate_value": rate_value, "arm": "anchor", "error": repr(exc)})
            print(f"  anchor  r={rate_value:>3}  FAILED {exc!r}", flush=True)
            continue
        if sweep == "qp":
            try:
                stream_rungs.append(
                    _announce(pointstream_rung(clip, paired, rate_value), "stream")
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"rate_value": rate_value, "arm": "pointstream", "error": repr(exc)}
                )
                print(f"  stream  r={rate_value:>3}  FAILED {exc!r}", flush=True)

    if sweep == "payload":
        for index, (jpeg_quality, rate_value) in enumerate(PAYLOAD_RUNGS):
            try:
                stream_rungs.append(
                    _announce(
                        payload_rung(
                            clip,
                            paired,
                            jpeg_quality=jpeg_quality,
                            rate_value=rate_value,
                            rank=len(PAYLOAD_RUNGS) - 1 - index,
                        ),
                        "stream",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "rate_value": f"jpeg{jpeg_quality}/qp{rate_value}",
                        "arm": "pointstream",
                        "error": repr(exc),
                    }
                )
                print(
                    f"  stream  jpeg{jpeg_quality}/qp{rate_value}  FAILED {exc!r}",
                    flush=True,
                )

    if sweep == "coarseness":
        # The candidate arm sweeps the residual representation instead of the
        # codec's rate. The anchor still sweeps QP, which is fine: BD-rate
        # integrates over the overlapping quality range and does not need the
        # two arms to share rung values.
        for point in coarseness_ladder():
            # Absent has no encode to configure, and lossless is left exactly as
            # `coarseness_ladder()` defines it: that rung names AVC on purpose,
            # because it is the one build on this host that honours
            # `rate_control=lossless`, and `av1` does not declare LOSSLESS at
            # all. Overriding it would turn a stated ceiling calibration into an
            # invalid request. Every lossy rung takes the paired codec.
            if point.config is None or point.variant is ResidualVariant.LOSSLESS:
                tuned_point = point
            else:
                tuned_point = replace(
                    point,
                    config=replace(
                        point.config,
                        codec=codec_name,
                        preset=PRESETS[codec_name],
                        rate_control=LADDER_RATE_CONTROL,
                    ),
                )
            try:
                stream_rungs.append(
                    _announce(coarseness_rung(clip, config, tuned_point), "stream")
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "rate_value": point.coarseness.value,
                        "arm": "pointstream",
                        "error": repr(exc),
                    }
                )
                print(
                    f"  stream  {point.coarseness.value:>9}  FAILED {exc!r}", flush=True
                )

    # A rung whose total is not a rate is not a point on an RD curve. Drop it
    # here rather than letting a mixed total into the fit.
    not_a_rate = [item for item in stream_rungs if not item.detail.get("is_rate")]
    stream_rungs = [item for item in stream_rungs if item.detail.get("is_rate")]

    result: dict[str, Any] = {
        "codec": codec_name,
        "preset": base.preset,
        "rate_control": base.rate_control.value,
        "pix_fmt": base.pix_fmt,
        "preset_note": (
            "This preset is on BOTH arms, which is why the comparison is fair. "
            "It is NOT matched-effort against other codecs' presets, so this "
            "gain must not be ranked against another codec's "
            "(plans/BP24-findings.md §1)."
        ),
        "anchor_rungs": [item.record() for item in anchor_rungs],
        "pointstream_rungs": [item.record() for item in stream_rungs],
        "rungs_excluded_not_a_rate": [item.record() for item in not_a_rate],
        "failures": failures,
        "bound_alarms": check_bounds(
            anchor_rungs, stream_rungs + not_a_rate, int(clip.frames.nbytes)
        ),
    }

    if len(anchor_rungs) < 2 or len(stream_rungs) < 2:
        result["bd_rate"] = None
        result["blocked_by"] = (
            f"need two usable rungs per arm; got {len(anchor_rungs)} anchor and "
            f"{len(stream_rungs)} pointstream"
        )
        return result

    anchor = _curve(anchor_rungs, f"{codec_name} on source")
    candidate = _curve(stream_rungs, f"pointstream via {codec_name}")
    try:
        comparison = compare_rd_curves(anchor, candidate)
    except InsufficientOverlapError as exc:
        result["bd_rate"] = None
        result["blocked_by"] = str(exc)
        result["overlap"] = list(exc.overlap)
        return result

    result["bd_rate"] = comparison.bd_rate
    result["bd_rate_percent"] = comparison.bd_rate_percent
    result["bd_quality_dB"] = comparison.bd_quality
    result["overlap_dB"] = list(comparison.overlap)
    result["overlap_fraction"] = comparison.overlap_fraction
    result["reading"] = (
        f"PointStream using {codec_name} costs "
        f"{comparison.bd_rate_percent:+.1f}% the rate of {codec_name} alone at "
        f"equal PSNR over {comparison.overlap[0]:.2f}-{comparison.overlap[1]:.2f} dB. "
        "Negative is PointStream winning."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codecs", nargs="+", default=["av1"])
    parser.add_argument("--rungs", nargs="+", type=int, default=list(DEFAULT_RUNGS))
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--video", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--sweep",
        choices=("payload", "qp", "coarseness"),
        default="payload",
        help=(
            "payload: the candidate arm sweeps plate quality and residual rate "
            "together, which is the only one of the three that moves "
            "PointStream's rate much (P0 item 2). "
            "qp: it sweeps the residual rate alone, holding the plate fixed. "
            "coarseness: it sweeps the residual-coarseness ladder (P0 item 3)."
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"n_frames": args.frames}
    if args.video:
        kwargs["video"] = args.video
    if args.scene:
        kwargs["scene"] = args.scene
    clip = load_tier_clip(**kwargs)
    motion = motion_level(clip.frames)
    print(
        f"clip {clip.video}/{clip.scene} {clip.describe()['resolution']} "
        f"x{args.frames}  inter-frame MAD={motion:.2f}",
        flush=True,
    )

    config = load_tier(args.tier)
    pairs = []
    for codec_name in args.codecs:
        print(f"--- {codec_name} ---", flush=True)
        pair = pair_for_codec(
            clip,
            config,
            codec_name=codec_name,
            rungs=tuple(args.rungs),
            sweep=args.sweep,
        )
        for alarm in pair["bound_alarms"]:
            print(f"  ALARM {alarm}", flush=True)
        print(f"  => {pair.get('reading') or pair.get('blocked_by')}", flush=True)
        pairs.append(pair)

    payload = {
        "brief": "BP24 — the paired ladder (PLAN.md §7 P0 items 2 and 3)",
        "bounds_written_before_measurement": "outputs/bp24-ladder/bounds-before-run.json",
        "quality_axis": LUMA_IS_THE_AXIS,
        "pairing": (
            "Each codec appears on both arms at the same preset, rate control "
            "and pixel format, from one EncodeRequest per rung. The preset "
            "cancels within a pair."
        ),
        "off_limits": (
            "Do not rank these gains against each other. The presets are not "
            "equal effort across codecs (plans/BP24-findings.md §1), so an "
            "ordering of the magnitudes would be measuring the presets."
        ),
        "clip": clip.describe(),
        "clip_motion_mad": motion,
        "tier": args.tier,
        "sweep": args.sweep,
        "rungs_requested": list(args.rungs),
        "pairs": pairs,
    }
    destination = Path(args.out) if args.out else OUT_DIR / "ladder.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
