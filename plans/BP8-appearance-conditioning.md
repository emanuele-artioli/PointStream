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

### Option B — use an architecture built for reference conditioning

Animate-Anyone (ReferenceNet), a properly-wired IP-Adapter, or StableAnimator
all take a reference image **by design**.

Animate-Anyone scored 10.4 dB and was dropped as flagship — **re-examine that
number before accepting it.** It is the one architecture here whose whole purpose
is reference conditioning, so a low score is more likely an inference-path fault
than a model verdict. That is the same mistake pattern as ControlNet's 0.11 VMAF.

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

## Done when

- One engine beats the 11.82 dB static-copy floor on the coding task, or all
  three options are tried and the negative result is written down with numbers.
- The probe scores against a later frame and keeps a static-copy baseline.
- `PLAN.md` §6.2 roster is re-decided on coding-task numbers.
