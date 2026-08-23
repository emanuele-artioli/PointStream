# B′14 — Stop a training run that cannot clear the bar

**Scheduled, not deferred.** Every training run from here uses it, so it lands
before the next one starts.

**Owns:** `scripts/train_controlnet.py`, `src/shared/training/` (new),
`tests/components/test_training_stop.py`.

## What happened

The pose-ref retrain burned **~14 GPU hours** on a series that was flat from the
first evaluation: epoch 1 = 11.33 dB, epoch 10 = 11.18 dB, never once above the
11.47 dB static-copy floor. **Epoch 1 already contained the answer.**

## Why nobody noticed: the loss was the wrong signal

`scripts/train_controlnet.py:465` is

```python
loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
```

— the standard **diffusion noise-prediction** objective. It measures how well the
model denoises at a randomly sampled timestep. It does **not** measure whether
the sample looks like the target, and it is dominated by easy high-noise
timesteps, so it falls smoothly almost regardless of sample quality. Diffusion
loss is a well-known poor proxy for sample quality.

Worse, in this specific run identity *could not* be learned at all: appearance
entered through the control image, which the control branch cannot use for
appearance (`PLAN.md` §2.3). The model was genuinely learning — just not the
thing we wanted. **A falling loss is not evidence that the thing you want is
being learned.**

**And there was no early stopping at all** — no patience, no validation, no
best-checkpoint tracking. Nothing was set too leniently; nothing was set.

## Is there something off the shelf?

Partly, and not the part that matters.

- **`accelerate`** (what this script uses) is a distributed-training wrapper. No
  early stopping.
- **`diffusers`** training scripts are reference examples. No early stopping.
- **`transformers.Trainer`** *does* ship `EarlyStoppingCallback` — but it stops on
  a **monitored metric the caller supplies**, and adopting `Trainer` for a
  diffusion ControlNet loop is a large refactor for a callback we can write in
  twenty lines.

**The mechanism is not the hard part; the criterion is.** No off-the-shelf
callback can stop on *"region-scoped PSNR on the probe set, against the
static-copy floor"* — that is ours. And stopping on the diffusion loss, which is
what a generic callback would do by default, **would not have stopped this run**,
because the loss was falling the whole time.

So: write the criterion, keep it small, and keep it task-facing.

## What to build

**A per-epoch validation hook that evaluates the real task**, not the loss:

- Generate on the probe set at the **coding task** (appearance from a keyframe,
  conditioning from a later frame) — never self-reconstruction.
- Score **region-scoped PSNR and calibrated LPIPS** (`PLAN.md` §2.7). Both are
  cheap: ~8 ms/frame against 4–6 s/frame of generation.
- Compare against the **static-copy floor at the same offsets**, which is the
  arm every generative run must beat.

**The stopping rule, and it must not kill a slow starter:**

- Write the success bar to `bounds.json` **before** training starts, as the
  pose-ref run already did correctly.
- **Never stop before epoch 3.** A model that starts slowly is a real thing; a
  model that is flat for three consecutive evaluations is not.
- Stop when **no evaluation has improved on the best-so-far for 3 consecutive
  epochs** (patience), *or* when the trend across the last 3 is flat-to-down and
  still below the floor.
- **Always keep the best checkpoint by the task metric**, not the last one.
- Evaluate more often than once an epoch early on — an epoch here is 8929 steps,
  which is far too coarse a grain for a first signal.

**Report the series**, so a stopped run still teaches something: every
evaluation, the floor, the bar, and the reason it stopped.

## Traps

**Do not stop on training loss.** It is the signal that failed here. It may be
logged; it may not gate.

**Do not let the validation eval creep.** It runs every epoch, so it must stay
small — a handful of clips, one offset, fixed seed. It is a stopping signal, not
a result, and nothing it produces is citable.

**A stopped run is a finding, delivered in hours.** The 14-hour version told us
nothing the 90-minute version would not have. Write that outcome down rather
than treating it as a failed job.

## Done when

- Training evaluates the real task on a schedule and stops on the rule above.
- The best checkpoint is kept by task metric, not by recency.
- A deliberately hopeless configuration is shown to stop by epoch 3–4 in a test.
- The series and the stop reason are written to `outputs/`.
