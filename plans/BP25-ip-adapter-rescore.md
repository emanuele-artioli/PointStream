# B'25 — Re-score IP-Adapter on an instrument that can rank models

**Closes `PLAN.md` §7 P0 item 5, either way.** An honest negative closes it as
well as a win. What is not acceptable is the current state: a number that looks
like a verdict and is not one.

**Owns:** `scripts/train_controlnet.py` (eval path only),
`src/shared/training/task_eval.py`, `outputs/bp25-ip-adapter/**`,
`plans/BP19-conditioning-architecture.md`.
**Read first:** `AGENTS.md` ("control the instrument, then the result"),
`PLAN.md` §2.10 and §2.12, `plans/BP19-conditioning-architecture.md` L206-211.

## What happened, and why it settles nothing

Training ran 2026-08-25 on GPU 1 and self-stopped at epoch 3, step 18000:
*"flat-to-down over the last 3 epoch evals and still below the static-copy
floor."* The **stop rule worked exactly as BP14 designed** — that part is a
genuine validation and is not in question.

| | LPIPS |
|---|---|
| static-copy floor | 0.5269 |
| unrelated null | 0.7497 |
| stock untrained IP-Adapter | 0.7606 |
| best checkpoint (epoch 1) | **0.8112** |
| final (epoch 3) | 0.8281 |

Every value sits **above the unrelated null**, which the pre-written bound calls
an alarm. Two defects in the measurement explain why it cannot be read as a
result:

1. **The stop-eval generates at 4 diffusion steps** (`STOP_EVAL_STEPS = 4`,
   straight into `ControlNetGenerator(steps=4)`). Vanilla SD1.5 needs 20-50. And
   the two anchors it is scored against — the static-copy floor and the unrelated
   null — are **real images that never pass through diffusion at all**. A
   barely-denoised generation is being compared with undegraded photographs. It
   loses regardless of how well the adapter trained.
2. **Most evals scored stale weights.** 11 evals, **5 distinct values**.
   `_run_task_eval` writes `checkpoint-epoch-N` only when that directory does not
   already exist, so every mid-epoch eval after the first in an epoch re-scores
   frozen weights.

Both `bounds.json` and `stop_series.json` already carry `"not_citable": true`.

## What to do

1. **Calibrate the instrument first, before re-scoring anything.** Generate a
   known-good sample at 4 steps and at 30 steps and score both. The question to
   answer in writing: *can the 4-step eval distinguish a good generation from a
   bad one at all?* If it cannot, that is itself a finding — the tripwire is
   confirmed fit only for stopping runs, never for ranking models.
2. **Score against a fair anchor.** If the candidate goes through diffusion, the
   floor must too, or the comparison is a step-count comparison wearing a model
   comparison's clothes. Say which anchoring you chose and why.
3. **Re-score the three saved checkpoints** (`checkpoint-epoch-1/2/3`, all on
   disk under `assets/weights/ip-adapter-trained/`) at 20-30 steps, with **n and
   standard error**. n=4 clips is too thin for a verdict — widen it.
4. **Fix the stale-checkpoint bug** so the series measures what it claims.
5. **Then, and only then, state whether IP-Adapter uses appearance.**

## Bounds — already written, do not rewrite

From `plans/BP19-conditioning-architecture.md` (2026-08-25): object-bbox LPIPS
expected **0.50-0.78** (pose 0.60, paste 0.45, unrelated 0.74); below 0.45 is an
alarm for paste-through, above 0.74 an alarm for worse-than-unrelated. `reid`
through `TENNIS_SCALE` expected **0.53-0.72**; a score at the same-person anchor
(0.8663) is an alarm.

The declared ceiling stands: **semantic appearance match — kit colour, roughly
right build — not identity**, because CLIP image embeddings lack fine spatial
detail. That is a real result to report, not a disappointment.

**The number to beat to have done anything at all is 0.7606**, the stock
untrained adapter.

## Traps

- **Do not retrain first.** Three checkpoints are on disk; the cheap experiment
  comes before the expensive one.
- **If the re-score comes out good, add a check rather than stopping.** Better
  than the static-copy floor would be the first such result in this project's
  history.
- **Do not repeat pose-ref.** Uni-ControlNet remains last.

## Done when

- The 4-step instrument is calibrated and its verdict written down.
- The three checkpoints are re-scored at a realistic step count with n and SE.
- The stale-checkpoint bug is fixed.
- `PLAN.md` §7 P0 item 5 is marked closed with whichever result is true.
