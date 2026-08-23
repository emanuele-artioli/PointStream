<<<<<<< HEAD
"""Coding-task PSNR bounds, written before this harness generated a number.

Date: 2026-08-22. The previous constants in this file were calibrated on
self-reconstruction (expected 14–28 dB, alarm 10 / 35). That is no longer
what the probe measures. Those gates would fire on a correct coding-task
result sitting near the static-copy floor, which is why they were replaced
rather than reused.

The coding task: appearance from a keyframe, conditioning from frame N,
score against frame N. A pasted keyframe (no model) is the floor any
generator has to beat.

Plausible band for that floor at offset 24 on the 12 probe clips, written
before this harness ran:

* worst ~8 dB — the player has moved a lot, or the crop is small
* best ~16 dB — near-static clips; a frozen pose would sit higher
* PLAN.md §2.3 measured 11.82 object / 8.90 frame on an ad-hoc re-run
  (seed 42). That number is not this harness's result. It only sets the
  band. A result far outside 8–16 dB at offset 24 is an alarm: check that
  scoring uses the later frame, not the keyframe (identity against the
  keyframe is inf, not ~12).

Engines are judged relative to the measured static-copy floor on the same
clip and offset, not against an absolute dB band.

If a bound later fires against a correct result, record why it was wrong
in the revisions list below.
=======
"""PSNR bounds, written before any BP5 probe number was read.

Date: 2026-08-22. Basis: the brief, PLAN.md §6.4–6.5, and the aligned-pair
triage in PLAN.md §2.1 (pose 19.0, seg 20.1, ip-adapter 11.0, pix2pix 18.5,
spade 15.1, trajectory 19.0, Animate-Anyone 14.0 in-set). Those earlier
numbers are not this ranking run; they only set the plausible band.

A result outside the alarm range is not a finding until the measurement is
checked. When a bound is wrong, record why here and in the run summary.
>>>>>>> phase-bp/bp5
"""

from __future__ import annotations

from dataclasses import dataclass

<<<<<<< HEAD
# Static-copy floor at offset 24. Absolute dB, because this *is* the floor.
STATIC_COPY_EXPECTED_LOW_DB = 8.0
STATIC_COPY_EXPECTED_HIGH_DB = 16.0
STATIC_COPY_ALARM_LOW_DB = 4.0
STATIC_COPY_ALARM_HIGH_DB = 28.0

# An engine at or below the measured floor is not using appearance.
# §2.3's ControlNet arms were 0.6–0.8 dB below the floor; that is the
# gate, not a path-bug alarm. Several dB below paste is noise, still
# reported with the same sentence.
NOT_USING_APPEARANCE = "not using appearance"
BEATS_FLOOR = "beats floor"
ENGINE_ALARM_BELOW_FLOOR_DB = 6.0
ENGINE_EXPECTED_BEAT_HIGH_DB = 12.0
ENGINE_ALARM_BEAT_DB = 20.0
ENGINE_ALARM_HIGH_ABS_DB = 35.0

# Revisions after outputs/bp9-static-copy (the constants above were not moved).
# 1. The 8–16 dB expected band is for the 12-clip *mean* at offset 24. Applying
#    it per clip, per offset, fired against a correct result:
#    sinner_alcaraz/scene_012/track_0058 at offset 8 scored 5.91 dB object
#    (outside-expected). The keyframe crop is 319x164 and the target is
#    263x230; each is letterboxed independently onto 512, so a pasted
#    keyframe lands in a different content box than the reference. Raw
#    same-size crop PSNR on that pair is ~12 dB. Alarm gates (4 / 28) did
#    not fire. Offset-24 mean was 11.47 dB, inside the band. Per-clip
#    judging therefore uses only the alarm gates; the expected band stays
#    on the headline mean.
=======
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
>>>>>>> phase-bp/bp5


@dataclass(frozen=True)
class BoundVerdict:
    """One number checked against the bounds written above."""

    metric: str
    value: float
    status: str
    note: str


<<<<<<< HEAD
def appearance_use_label(engine_psnr: float, floor_psnr: float) -> str:
    """The gate sentence. At or below the floor is not a low rank."""
    if engine_psnr <= floor_psnr:
        return NOT_USING_APPEARANCE
    return BEATS_FLOOR


def judge_static_copy_clip(value: float) -> BoundVerdict:
    """Per-clip floor. Alarm gates only; the 8–16 dB band is the offset-24 mean."""
    if value < STATIC_COPY_ALARM_LOW_DB:
=======
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
>>>>>>> phase-bp/bp5
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-low",
<<<<<<< HEAD
            note=(
                "static copy below ~4 dB: suspect scoring a black canvas or "
                "the wrong reference, not a moved player"
            ),
        )
    if value > STATIC_COPY_ALARM_HIGH_DB:
=======
            note="below ~10 dB: suspect the inference path before the model",
        )
    if value > OBJECT_PSNR_ALARM_HIGH_DB:
>>>>>>> phase-bp/bp5
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-high",
<<<<<<< HEAD
            note=(
                "static copy above ~28 dB: suspect scoring the keyframe "
                "against itself instead of the later frame"
            ),
        )
    return BoundVerdict(
        metric="object_psnr_db",
        value=value,
        status="ok",
        note="per-clip static copy; 8–16 dB expected band is the offset-24 mean",
    )


def judge_static_copy_object_psnr(value: float) -> BoundVerdict:
    """Classify the 12-clip mean floor. Absolute dB, offset-24 band."""
    if value < STATIC_COPY_ALARM_LOW_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-low",
            note=(
                "static copy below ~4 dB: suspect scoring a black canvas or "
                "the wrong reference, not a moved player"
            ),
        )
    if value > STATIC_COPY_ALARM_HIGH_DB:
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="alarm-high",
            note=(
                "static copy above ~28 dB: suspect scoring the keyframe "
                "against itself instead of the later frame"
            ),
        )
    if STATIC_COPY_EXPECTED_LOW_DB <= value <= STATIC_COPY_EXPECTED_HIGH_DB:
=======
            note="above ~35 dB object-scoped: suspect scoring the source against itself",
        )
    if OBJECT_PSNR_EXPECTED_LOW_DB <= value <= OBJECT_PSNR_EXPECTED_HIGH_DB:
>>>>>>> phase-bp/bp5
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="expected",
<<<<<<< HEAD
            note="pasted keyframe vs a moved player at offset 24, 8–16 dB",
=======
            note="high teens to mid twenties on a first pass = pass",
>>>>>>> phase-bp/bp5
        )
    return BoundVerdict(
        metric="object_psnr_db",
        value=value,
        status="outside-expected",
        note=(
<<<<<<< HEAD
            f"inside the alarm gates but outside {STATIC_COPY_EXPECTED_LOW_DB:g}–"
            f"{STATIC_COPY_EXPECTED_HIGH_DB:g} dB expected floor band"
=======
            f"inside the alarm gates but outside {OBJECT_PSNR_EXPECTED_LOW_DB:g}–"
            f"{OBJECT_PSNR_EXPECTED_HIGH_DB:g} dB expected band"
>>>>>>> phase-bp/bp5
        ),
    )


<<<<<<< HEAD
def judge_vs_floor(engine_psnr: float, floor_psnr: float) -> BoundVerdict:
    """Classify an engine against the measured static-copy floor.

    The appearance-use sentence lives on ``appearance_use_label``. This
    verdict is the bound check: at-or-below is the gate, far below or
    near-identity is an alarm on the measurement.
    """
    delta = engine_psnr - floor_psnr
    if delta <= 0.0:
        if delta <= -ENGINE_ALARM_BELOW_FLOOR_DB:
            return BoundVerdict(
                metric="vs_static_copy_db",
                value=delta,
                status="alarm-low",
                note=(
                    f"{NOT_USING_APPEARANCE} ({delta:.2f} dB vs floor; "
                    "this far below paste is worth checking the path)"
                ),
            )
        return BoundVerdict(
            metric="vs_static_copy_db",
            value=delta,
            status="not-using-appearance",
            note=NOT_USING_APPEARANCE,
        )
    if engine_psnr > ENGINE_ALARM_HIGH_ABS_DB or delta > ENGINE_ALARM_BEAT_DB:
        return BoundVerdict(
            metric="vs_static_copy_db",
            value=delta,
            status="alarm-high",
            note=(
                "this close to the target at a non-zero offset: suspect "
                "scoring against the keyframe or feeding the reference as "
                "appearance"
            ),
        )
    if delta <= ENGINE_EXPECTED_BEAT_HIGH_DB:
        return BoundVerdict(
            metric="vs_static_copy_db",
            value=delta,
            status="beats-floor",
            note=f"{BEATS_FLOOR} by {delta:.2f} dB",
        )
    return BoundVerdict(
        metric="vs_static_copy_db",
        value=delta,
        status="outside-expected",
        note=f"{BEATS_FLOOR} by {delta:.2f} dB, above the first-pass +12 dB band",
    )


def judge_frame_gap(frame_psnr: float, object_psnr: float) -> BoundVerdict:
    """Whole-frame minus object-scoped.

    A pasted keyframe on a letterboxed crop often inverts this gap: the
    player moved and the pad did not, so object PSNR can sit above frame
    PSNR. That is recorded, not treated as an alarm.
    """
=======
def judge_frame_gap(frame_psnr: float, object_psnr: float) -> BoundVerdict:
    """Whole-frame minus object-scoped. Expected: frame much better."""
>>>>>>> phase-bp/bp5
    gap = frame_psnr - object_psnr
    if gap < 0.0:
        return BoundVerdict(
            metric="frame_minus_object_db",
            value=gap,
<<<<<<< HEAD
            status="inverted",
            note=(
                "whole-frame worse than object-scoped. Expected for a pasted "
                "keyframe whose player moved; not an alarm by itself"
            ),
        )
    return BoundVerdict(
        metric="frame_minus_object_db",
        value=gap,
        status="frame-higher",
        note="whole-frame better than object-scoped",
=======
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
>>>>>>> phase-bp/bp5
    )
