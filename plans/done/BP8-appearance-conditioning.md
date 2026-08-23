> **SUPERSEDED 2026-08-23 by `BP10-appearance-pathway.md`.** Its options were
> tried in Wave 3. Option A (retrain with a reference) is measured, flat and
> understood — do not repeat it. Options B and B2 were read against the
> static-copy floor, which `PLAN.md` §2.4 shows answers a different question.
> Kept for the bounds discipline and the record of what was tried.

# B′8 — Make the generator actually use appearance

**The critical path. Everything in `PLAN.md` §7 P0 except item 1 is behind it.**

**Read first:** `PLAN.md` §2.6 — the evidence that this is a training-design
problem, not a tuning problem.

## The finding

Driven 2026-08-22 on the real coding task (appearance from frame 0, pose from
frame 24, scored against frame 24, 12 clips):

| Arm | Object PSNR |
|---|---|
| **static copy — paste the keyframe, no model** | **11.82 dB** |
| seg-controlnet | 11.01 dB |
| pose-controlnet | 11.20 dB |

Both generators lose to doing nothing. `scripts/train_controlnet.py` trains on
`{image_path, cond_path, prompt}` — **appearance is never an input.** The
checkpoints synthesise *a* tennis player from a pose and a fixed text prompt;
they cannot reproduce *this* player. `ip-adapter-controlnet` is a segmentation
ControlNet mislabelled (line 82).

**No parameter search fixes this.** Tuning is ruled out on evidence, which
removes the cheapest rung from `PLAN.md` §6.6's cost order.

## The options, cheapest first

### Option A — retrain ControlNet with a reference image

The data path already exists: `src/shared/tennis_dataset.py` supports
`include_reference` and draws `ref_color_path = random.choice(track_to_colors)`
— a different frame from the same track. That is exactly the right sampling for
learning to animate rather than to copy.

`ControlNetDataset` does not use it. Extend training to condition on
(reference appearance, pose), and retrain the pose variant first.

**Why this is first:** it keeps the comparison backbone that `subsec:eval-object`
depends on — a fixed backbone with swappable control encoders — which no
off-the-shelf model gives us.

**Bound before believing:** a working appearance-conditioned model must beat
**11.82 dB** object-scoped. Beating it by less than ~1 dB is not a result. Below
it, the model still is not using appearance.

### Option B — re-examine Animate-Anyone first (do this before A)

**Start here, not with A.** Animate-Anyone is the one architecture on the roster
whose entire purpose is reference conditioning — appearance enters through
ReferenceNet by design. It scored 10.4 dB and was dropped as flagship *on the
self-reconstruction framing*, which is the framing that also made ControlNet look
acceptable. A low score from the one reference-conditioned model is more likely
an inference-path fault than a model verdict.

This is the same mistake pattern as ControlNet's 0.11 VMAF, which was read as a
model result and was a broken path. Check, in order: that the reference image
actually reaches ReferenceNet; the DDIM step count (3 steps melted, 20 is the
class default); the letterboxing agreement between reference and pose; and the
scheduler configuration.

It is hours of work and it may be the whole answer. A is days.

### Option B2 — other architectures built for reference conditioning

A properly-wired IP-Adapter is the cheapest of these: the existing
`ip-adapter-controlnet` checkpoint is a mislabelled segmentation ControlNet
(`PLAN.md` §2.6), but the *architecture* accepts an image as appearance. Wiring
a real IP-Adapter against the stock SD-1.5 backbone needs no retraining.

StableAnimator remains licence-blocked on SVD-XT (`PLAN.md` §2.4).

### Option C — change what the paper claims is transmitted

If appearance genuinely is not needed and pose plus a caption suffices, that is a
*more* radical codec, not a failure — but it cannot make identity claims, and
every quality number would be "a plausible player", not "this player". **Do not
take this option to escape a training problem.** It is only honest if A and B are
tried and appearance turns out not to pay for itself — which would itself be a
finding worth reporting.

## Traps

**Do not tune.** Guidance scale, steps and strength cannot add an input the model
never had. Measure, do not fiddle.

**Fix the probe's framing at the same time.** `experiments/probe/run.py` scores
against the conditioning appearance, which measures self-reconstruction and
contradicts its own `differs_from_input` check. Score against a *later* frame
than the appearance came from, and **keep the static-copy baseline as a
permanent arm** — it is the floor that exposed this, and it costs nothing.

**Re-examine every Wave-2 roster conclusion.** ControlNet "holds both quality
slots" was decided on the self-reconstruction framing. On the coding task it
loses to a static copy, so the roster is not settled.

**This does not invalidate the rest.** The probe set, region metrics, C1 and C2
all stand. The pipeline work is unaffected; only the roster claim is.

## Order of work

1. **Re-examine Animate-Anyone's inference path** (Option B). Hours. May be the
   whole answer.
2. **Wire a real IP-Adapter** (Option B2) if B does not settle it. No retraining.
3. **Retrain pose-ControlNet with the reference frame** (Option A) if neither
   does. Days, but it is the option that preserves the fixed comparison backbone
   `subsec:eval-object` depends on.
4. **Option C only if 1–3 all fail**, and then as a reported finding, never as an
   escape.

## Done when

- One engine beats the 11.82 dB static-copy floor on the coding task, or all
  options are tried and the negative result is written down with numbers.
- `PLAN.md` §6.2 roster is re-decided on coding-task numbers.
