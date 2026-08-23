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
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class BoundVerdict:
    """One number checked against the bounds written above."""

    metric: str
    value: float
    status: str
    note: str


def appearance_use_label(engine_psnr: float, floor_psnr: float) -> str:
    """The gate sentence. At or below the floor is not a low rank."""
    if engine_psnr <= floor_psnr:
        return NOT_USING_APPEARANCE
    return BEATS_FLOOR


def judge_static_copy_clip(value: float) -> BoundVerdict:
    """Per-clip floor. Alarm gates only; the 8–16 dB band is the offset-24 mean."""
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
        return BoundVerdict(
            metric="object_psnr_db",
            value=value,
            status="expected",
            note="pasted keyframe vs a moved player at offset 24, 8–16 dB",
        )
    return BoundVerdict(
        metric="object_psnr_db",
        value=value,
        status="outside-expected",
        note=(
            f"inside the alarm gates but outside {STATIC_COPY_EXPECTED_LOW_DB:g}–"
            f"{STATIC_COPY_EXPECTED_HIGH_DB:g} dB expected floor band"
        ),
    )


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
    gap = frame_psnr - object_psnr
    if gap < 0.0:
        return BoundVerdict(
            metric="frame_minus_object_db",
            value=gap,
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
    )


# ---------------------------------------------------------------------------
# LPIPS bounds for clip mode, written 2026-08-23 before this harness produced
# a single LPIPS number. BP12 makes LPIPS the ranking key; PSNR stays reported
# beside it. Lower is better and the scale is anchored, which is why an
# absolute band is meaningful here where it was not for PSNR.
#
# Published calibration anchors for the wrapped ``lpips`` package (PLAN.md
# §2.7, asserted in tests/invariants/test_metric_calibration.py):
#
#     identical 0.000 | mild noise 0.250 | heavy blur 0.430 | unrelated 0.645
#
# The bands below are for the **12-clip mean over offsets 1-8**, scored on the
# bounding box of the letterboxed player mask.
#
# static-copy floor. §2.7 measured 0.239 at offset 1 and 0.582 at offset 8 on
# 4 clips with a box this harness does not reproduce exactly, so the mean over
# 1-8 should land near 0.42 and the band is widened for the box change:
#   * worst ~0.60 — fast clips, the player has left the box by offset 8
#   * best  ~0.20 — near-static clips where a paste is nearly right
# An alarm below 0.05 means the score is being taken against the keyframe
# rather than the later frame; above 0.75 means the paste is landing somewhere
# other than the player.
STATIC_COPY_LPIPS_EXPECTED_LOW = 0.20
STATIC_COPY_LPIPS_EXPECTED_HIGH = 0.60
STATIC_COPY_LPIPS_ALARM_LOW = 0.05
STATIC_COPY_LPIPS_ALARM_HIGH = 0.75

# unrelated-image null control. The published unrelated-image anchor is 0.645
# and these crops are all tennis players on a court, so some structure is
# shared and the control should sit a little under a random-image pair:
#   * expected 0.55-0.80
# Below 0.40 the control is scoring as well as a real match, which means the
# metric or the wiring cannot tell the right player from the wrong one — the
# exact fault that voided every ranking before 2026-08-23.
UNRELATED_LPIPS_EXPECTED_LOW = 0.55
UNRELATED_LPIPS_EXPECTED_HIGH = 0.80
UNRELATED_LPIPS_ALARM_LOW = 0.40

# The separation the whole run rests on. If a paste of the *right* player is
# not clearly better than a paste of the *wrong* one, nothing downstream is
# readable, whatever the engines score.
NULL_SEPARATION_MIN = 0.10

# An engine at or below 0.05 is not a good result, it is a scoring fault:
# 20-step diffusion from a pose does not reproduce a photograph.
ENGINE_LPIPS_ALARM_LOW = 0.05


def judge_static_copy_lpips(value: float) -> BoundVerdict:
    """Classify the 12-clip mean floor on LPIPS. Absolute, offsets 1-8."""
    if value < STATIC_COPY_LPIPS_ALARM_LOW:
        return BoundVerdict(
            metric="object_lpips",
            value=value,
            status="alarm-low",
            note=(
                "static copy near zero LPIPS: suspect scoring the keyframe "
                "against itself instead of the later frame"
            ),
        )
    if value > STATIC_COPY_LPIPS_ALARM_HIGH:
        return BoundVerdict(
            metric="object_lpips",
            value=value,
            status="alarm-high",
            note=(
                "static copy worse than an unrelated image: suspect the paste "
                "or the crop box, not the player's motion"
            ),
        )
    if STATIC_COPY_LPIPS_EXPECTED_LOW <= value <= STATIC_COPY_LPIPS_EXPECTED_HIGH:
        return BoundVerdict(
            metric="object_lpips",
            value=value,
            status="expected",
            note="pasted keyframe vs a moved player, offsets 1-8, 0.20-0.60",
        )
    return BoundVerdict(
        metric="object_lpips",
        value=value,
        status="outside-expected",
        note=(
            f"inside the alarm gates but outside {STATIC_COPY_LPIPS_EXPECTED_LOW:g}-"
            f"{STATIC_COPY_LPIPS_EXPECTED_HIGH:g} expected floor band"
        ),
    )


def judge_unrelated_lpips(value: float) -> BoundVerdict:
    """Classify the null control. Below the alarm, the run is not readable."""
    if value < UNRELATED_LPIPS_ALARM_LOW:
        return BoundVerdict(
            metric="object_lpips",
            value=value,
            status="alarm-low",
            note=(
                "the wrong player scores like a match. The metric or the "
                "appearance wiring cannot separate them; no ranking in this "
                "run is readable until that is explained"
            ),
        )
    if UNRELATED_LPIPS_EXPECTED_LOW <= value <= UNRELATED_LPIPS_EXPECTED_HIGH:
        return BoundVerdict(
            metric="object_lpips",
            value=value,
            status="expected",
            note="unrelated keyframe, 0.55-0.80 against a 0.645 published anchor",
        )
    return BoundVerdict(
        metric="object_lpips",
        value=value,
        status="outside-expected",
        note=(
            f"outside {UNRELATED_LPIPS_EXPECTED_LOW:g}-"
            f"{UNRELATED_LPIPS_EXPECTED_HIGH:g}; report the value with the anchors"
        ),
    )


def judge_null_separation(floor_lpips: float, unrelated_lpips: float) -> BoundVerdict:
    """The right player minus the wrong one. This gates the whole run."""
    delta = unrelated_lpips - floor_lpips
    if delta < NULL_SEPARATION_MIN:
        return BoundVerdict(
            metric="null_separation_lpips",
            value=delta,
            status="alarm-low",
            note=(
                f"right player and wrong player differ by only {delta:.3f} LPIPS. "
                "The instrument is not resolving identity on this content; do "
                "not rank engines from this run"
            ),
        )
    return BoundVerdict(
        metric="null_separation_lpips",
        value=delta,
        status="ok",
        note=f"the wrong player costs {delta:.3f} LPIPS; the instrument separates identity",
    )


def judge_engine_lpips(
    engine_lpips: float, floor_lpips: float, unrelated_lpips: float
) -> BoundVerdict:
    """Place an engine on the floor-to-null scale, with both ends quoted.

    ``0.067`` means nothing on its own. This returns the number *with* the two
    anchors measured in the same run, which is the form a result may be
    reported in.
    """
    anchors = f"floor {floor_lpips:.3f}, unrelated {unrelated_lpips:.3f}"
    if engine_lpips < ENGINE_LPIPS_ALARM_LOW:
        return BoundVerdict(
            metric="object_lpips",
            value=engine_lpips,
            status="alarm-low",
            note=(
                f"{engine_lpips:.3f} is near-identity from a 20-step generation "
                f"({anchors}): suspect the prediction is the reference"
            ),
        )
    if engine_lpips <= floor_lpips:
        return BoundVerdict(
            metric="object_lpips",
            value=engine_lpips,
            status="beats-floor",
            note=f"{engine_lpips:.3f} beats the static-copy floor ({anchors})",
        )
    if engine_lpips >= unrelated_lpips:
        return BoundVerdict(
            metric="object_lpips",
            value=engine_lpips,
            status="at-or-worse-than-null",
            note=(
                f"{engine_lpips:.3f} is no better than showing the wrong player "
                f"({anchors})"
            ),
        )
    return BoundVerdict(
        metric="object_lpips",
        value=engine_lpips,
        status="between-floor-and-null",
        note=f"{engine_lpips:.3f} sits between the two anchors ({anchors})",
    )


# ---------------------------------------------------------------------------
# Cross-appearance bounds. Written 2026-08-23 before the control ran, and
# **rewritten the same day because the first version was wrong**. Both versions
# are described here, because how a bound failed is worth as much as the bound.
#
# The test: hold the model, the pose, the target and the metric fixed, and vary
# only which keyframe the engine is shown. The delta is what showing the wrong
# player costs.
#
# WHAT THE FIRST VERSION CLAIMED, AND WHY IT WAS WRONG
#
# It read a large delta as "this engine has a working appearance pathway", took
# BP10's PSNR bands (>= +3 dB works, ~ +0.9 leakage, ~ 0 wiring fault) onto the
# LPIPS scale, and called the 0.285 LPIPS a paste is worth "what perfect use of
# the appearance signal buys".
#
# Then the copying baselines were driven through the identical code path:
#
#     static-copy    (no model at all)   +0.285   100%
#     upscale-refine (non-generative)    +0.185    65%
#     pose-controlnet                    +0.166    58%
#     animate-anyone (clip mode)         +0.107    37%
#     ip-adapter-controlnet              +0.055    19%
#
# **The two arms with no generative network score highest.** The delta measures
# how much of the reference image survives into the output -- copying -- and a
# paste maximises it by definition. It does not measure whether a model renders
# the right person; it cannot, because the arm that renders nothing wins.
#
# So 0.285 is the top of a *copying* axis, not of an appearance-use axis, and
# "uses appearance" was never a reading this number could support. A generator
# scoring below a paste on it is the expected case, not a finding.
#
# WHAT THIS VERSION CLAIMS
#
# Only that the output does or does not depend on the reference image, and by
# how much relative to a pure copy measured in the same run. Whether that
# dependence is *useful* is a separate question, answered by the arm's own
# quality score against the static-copy floor -- where every engine currently
# loses (PLAN.md 2.10).
#
# A verdict therefore requires the copying anchor. Without it this returns
# "unanchored" rather than a number dressed as a conclusion.
CROSS_DEPENDENT_LPIPS = 0.10
CROSS_WEAK_LPIPS = 0.02

CROSS_REFERENCE_DEPENDENT = "reference-dependent"
CROSS_WEAKLY_DEPENDENT = "weakly reference-dependent"
CROSS_REFERENCE_INDEPENDENT = "reference-independent"


def judge_cross_appearance(
    delta_lpips: float,
    *,
    sigmas: float,
    standard_error: float,
    copy_delta: float | None = None,
    underpowered: bool = False,
) -> BoundVerdict:
    """Classify a cross-appearance LPIPS delta (wrong minus right; higher = more dependent).

    Says only how far the output moves when the reference changes, against what
    a pure paste scores in the same run. It does **not** say the engine uses
    appearance well: a paste tops this scale with no network at all.

    Four questions, in order, because they need different evidence.

    1. **Is there an anchor?** Without a copying baseline the number has no
       scale, and this project has published a bare "0.067" before.
    2. **Is the sample big enough to say anything?** ``underpowered`` carries
       ``compare_paired``'s small-sample refusal, which a large sigma on three
       clips does not overrule.
    3. **Can we say the effect is absent?** That is a claim about the *interval*,
       not the point estimate: ``0.001 +/- 0.05`` is equally consistent with a
       real dependence, so absence needs ``delta + 2se`` below the weak band.
    4. **Is a claimed effect real?** Two standard errors, because a 1.5-sigma
       +0.98 dB difference was reported here as a finding.
    """
    if copy_delta is None:
        return BoundVerdict(
            metric="cross_appearance_lpips",
            value=delta_lpips,
            status="unanchored",
            note=(
                f"{delta_lpips:+.3f} LPIPS with no copying baseline measured in "
                "this run. The scale is unknown, so this is a number, not a result"
            ),
        )
    share = f", {delta_lpips / copy_delta:.0%} of the {copy_delta:.3f} a pure paste scores"
    if underpowered:
        return BoundVerdict(
            metric="cross_appearance_lpips",
            value=delta_lpips,
            status="underpowered",
            note=(
                f"{delta_lpips:+.3f} LPIPS{share}: too few clips to call a "
                "direction, whatever the sigma says"
            ),
        )
    upper = delta_lpips + 2.0 * standard_error
    if upper < CROSS_WEAK_LPIPS:
        return BoundVerdict(
            metric="cross_appearance_lpips",
            value=delta_lpips,
            status=CROSS_REFERENCE_INDEPENDENT,
            note=(
                f"{delta_lpips:+.3f} LPIPS, at most {upper:+.3f} at two standard "
                f"errors{share}: the wrong reference costs this engine nothing. "
                "Check the wiring before the architecture"
            ),
        )
    if sigmas < 2.0:
        return BoundVerdict(
            metric="cross_appearance_lpips",
            value=delta_lpips,
            status="inside-noise",
            note=(
                f"{delta_lpips:+.3f} LPIPS at {sigmas:.1f}σ{share}: the sample "
                "does not support a direction, and the interval does not rule "
                "an effect out either"
            ),
        )
    if delta_lpips >= CROSS_DEPENDENT_LPIPS:
        return BoundVerdict(
            metric="cross_appearance_lpips",
            value=delta_lpips,
            status=CROSS_REFERENCE_DEPENDENT,
            note=(
                f"{delta_lpips:+.3f} LPIPS at {sigmas:.1f}σ{share}. Dependence, "
                "not quality: read it beside this arm's own score against the floor"
            ),
        )
    return BoundVerdict(
        metric="cross_appearance_lpips",
        value=delta_lpips,
        status=CROSS_WEAKLY_DEPENDENT,
        note=(
            f"{delta_lpips:+.3f} LPIPS at {sigmas:.1f}σ{share}: real but small, "
            "the size of an untrained img2img init path"
        ),
    )
