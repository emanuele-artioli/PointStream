"""PSNR bounds, written before any BP5 probe number was read.

Date: 2026-08-22. Basis: the brief, PLAN.md §6.4–6.5, and the aligned-pair
triage in PLAN.md §2.1 (pose 19.0, seg 20.1, ip-adapter 11.0, pix2pix 18.5,
spade 15.1, trajectory 19.0, Animate-Anyone 14.0 in-set). Those earlier
numbers are not this ranking run; they only set the plausible band.

A result outside the alarm range is not a finding until the measurement is
checked. When a bound is wrong, record why here and in the run summary.
"""

from __future__ import annotations

from dataclasses import dataclass

# Alarm: below this, suspect the inference path before the model (once already
# a ControlNet 0.11 VMAF that was a broken path). ip-adapter's known txt2img
# floor sits just above this; see IP_ADAPTER_KNOWN_FLOOR.
OBJECT_PSNR_ALARM_LOW_DB = 10.0

# Expected first-pass band: high teens to mid twenties. A pass, not a paper
# number. Lightly trained checkpoints and pretrained models never adapted to
# broadcast tennis live here.
OBJECT_PSNR_EXPECTED_LOW_DB = 14.0
OBJECT_PSNR_EXPECTED_HIGH_DB = 28.0

# Alarm: scoring the source against itself, or a region that is mostly
# background rather than the player.
OBJECT_PSNR_ALARM_HIGH_DB = 35.0

# ip-adapter-controlnet is a txt2img path. PLAN.md §2.1 already posted 11.0 dB
# on an aligned pair. That floor is not a new path bug unless the output is
# identical to the input.
IP_ADAPTER_KNOWN_FLOOR_DB = 10.0
IP_ADAPTER_ENGINE = "ip-adapter-controlnet"

# Whole-frame PSNR much higher than object-scoped is the expected shape on a
# composited broadcast frame (§6.4). This harness scores a generation canvas
# (letterboxed crop), so the gap is the letterbox pad plus any non-player
# pixels inside the crop. A *small* gap is the surprise and is recorded, not
# treated as an alarm by itself.
FRAME_MINUS_OBJECT_SMALL_GAP_DB = 0.5

# Revisions after outputs/bp5-probe (the constants above were not moved).
# 1. Prior band (pose ~19, ip-adapter ~11, AA ~14) was whole-canvas vs
#    letterboxed appearance. Object-scoped (crop alpha) sits ~3–4 dB below
#    that on every ControlNet arm. Applying the 10 dB path-bug floor to
#    *object* PSNR therefore alarms ip-adapter at 7.9 dB even though its
#    frame PSNR is 11.1 dB — the same known txt2img floor, scoped. Output
#    differed from input. Not a new path bug.
# 2. SPADE object 12.0 is below the 14–28 expected band; frame 15.2 matches
#    BP7's 15.1 canvas number. Same unit mismatch as (1).
# 3. Upscale-refine inverts the frame/object gap because it *stretches* the
#    crop onto 512² while scoring letterboxes. Object 14.5 dB is the fair
#    floor; the inverted frame score is a canvas convention, not a model.
# 4. Animate-Anyone object 10.4 is below expected. Gap to ControlNet is ~6 dB
#    (not 15). Single-frame 512 px vs BP4's 256 px 4-frame window. Not a
#    wiring stop; it loses the quality-flagship slot on this triage.


@dataclass(frozen=True)
class BoundVerdict:
    """One number checked against the bounds written above."""

    metric: str
    value: float
    status: str
    note: str


def judge_object_psnr(engine: str, value: float) -> BoundVerdict:
    """Classify an object-scoped PSNR. Does not look at any stored run."""
    if engine == IP_ADAPTER_ENGINE and IP_ADAPTER_KNOWN_FLOOR_DB <= value < OBJECT_PSNR_EXPECTED_LOW_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="known-floor",
            note=(
                "ip-adapter txt2img floor (PLAN.md §2.1 ~11 dB). Not a path bug "
                "unless output equals input."
            ),
        )
    if value < OBJECT_PSNR_ALARM_LOW_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-low",
            note="below ~10 dB: suspect the inference path before the model",
        )
    if value > OBJECT_PSNR_ALARM_HIGH_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-high",
            note="above ~35 dB object-scoped: suspect scoring the source against itself",
        )
    if OBJECT_PSNR_EXPECTED_LOW_DB <= value <= OBJECT_PSNR_EXPECTED_HIGH_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="expected",
            note="high teens to mid twenties on a first pass = pass",
        )
    return BoundVerdict(
        metric="object_psnr_db",
        value=value,
        status="outside-expected",
        note=(
            f"inside the alarm gates but outside {OBJECT_PSNR_EXPECTED_LOW_DB:g}–"
            f"{OBJECT_PSNR_EXPECTED_HIGH_DB:g} dB expected band"
        ),
    )


def judge_frame_gap(frame_psnr: float, object_psnr: float) -> BoundVerdict:
    """Whole-frame minus object-scoped. Expected: frame much better."""
    gap = frame_psnr - object_psnr
    if gap < 0.0:
        return BoundVerdict(
            metric="frame_minus_object_db",
            value=gap,
            status="surprise-inverted",
            note=(
                "whole-frame worse than object-scoped. On a crop canvas this can "
                "mean the generator filled the letterbox; on a composited frame "
                "it would contradict §6.4."
            ),
        )
    if gap < FRAME_MINUS_OBJECT_SMALL_GAP_DB:
        return BoundVerdict(
            metric="frame_minus_object_db",
            value=gap,
            status="surprise-small-gap",
            note="small frame/object gap is the surprising outcome (§6.4)",
        )
    return BoundVerdict(
        metric="frame_minus_object_db",
        value=gap,
        status="expected",
        note="whole-frame better than object-scoped, the §6.4 shape",
    )
