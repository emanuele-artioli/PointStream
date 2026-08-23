# PointStream — handoff, 2026-08-23

You are picking up a rewrite of an object-centric semantic video codec, targeting
**ACM TOMM, September 30**. Everything is merged to `main` and pushed in both
repos. The suite is green: 1028 passed, 3 xfailed, ruff and mypy clean.

## Read, in this order

1. `AGENTS.md` (project) and the host rules it imports.
2. `PLAN.md` — especially **§2.3 through §2.7**, which are the findings that
   invalidate most earlier conclusions.
3. `plans/README.md` — what is live, and what is void.
4. Your one brief from `plans/`. Do not read the whole tree.

## The situation in five sentences

The component platform is built and works: 16 axes, a rebuilt probe set, a
pipeline, a runner, region-scoped metrics. **No generative engine produces a
usable player.** The ControlNet checkpoints were trained with **no appearance
input at all**, so they synthesise *a* tennis player and can never reproduce
*this* one — and no amount of tuning adds an input the training never had.
Animate-Anyone, which is architecturally right for this, was being evaluated one
frame at a time despite being a temporal model; driven correctly it improves but
still only reaches 0.570 LPIPS on the player region, against 0.582 for a static
copy and 0.645 for an unrelated image. The paper's central rate-distortion claim
against the codec ladder **has never been run in any configuration**.

## What you must not trust

- **Every engine ranking taken before 2026-08-23.** Two of three metrics were
  broken: LPIPS was an uncalibrated VGG feature distance that scored an unrelated
  image at 0.083 and a good reconstruction at 0.085; VMAF had its ffmpeg inputs
  crossed and scored blur above a perfect match. Both are fixed and both now have
  calibration invariants.
- **Anything in `plans/done/` that ranked engines.** `BP5`'s roster verdict was
  measured on self-reconstruction and is void.
- **PSNR as a ranking key for generative arms.** The usable range on this task is
  ~11–21 dB with a ~2 dB per-clip sd. Use LPIPS; keep PSNR reported alongside.

## Working rules that were learned expensively

**Seven times here, something passed its tests while not doing its job** — ten
generators registered that could not load weights; a probe verifier green while
five clips had no pose data; a roster ranked on self-reconstruction; a temporal
model run at T=1; a metric that could not tell a match from noise; a VMAF wiring
where blur beat perfection; a training run with no stopping criterion at all.

So:

- **Use the `verify-measurement` skill before reporting any measurement.** It
  carries the calibration anchors, the null-control table, and the significance
  bands.
- **Run the control in the same session, before reporting** — not when asked.
- **Quote the instrument's range beside the number.** "0.067" is meaningless;
  "0.067, where unrelated scores 0.645" is not.
- **Use `src.components.metrics.comparison.compare_paired`** for any arm-vs-arm
  claim. It refuses to name a winner the sample cannot support.
- **Check the invocation before blaming the model.**
- **When the news is good, add a check rather than stopping.** Every wrong
  conclusion in this project was a pleasing result reported before its control.

## Environment

- `conda run -n pointstream --no-capture-output <cmd>`; imports absolute from the
  repo root.
- Packages go in `pyproject.toml` and then get installed — **not** ad-hoc, and
  never a version bump on a pinned forked model.
- Before merging: `ruff check`, `mypy --config-file pyproject.toml`, the tests
  for what you touched, `python -m src.contracts.layers`.
- Three known `xfail`s, each with a reason: `DEFERRED.md` D5 (the all-off corner
  is a branch, not an architecture) and D6 (test pollution).
- The paper is a **separate git repo** at `67a9ea6275d3d9785ce57026/`. Commit
  there when you change it.

## What to do next

**`plans/BP12-clip-mode-roster.md` is the critical path.** Re-rank every engine
in clip mode on the corrected metrics — no ranking survives the metric fixes — and
re-run the cross-appearance control, which is the test that says whether an
engine uses appearance at all.

Then, in order:

- **`BP14`** before any training run. The last one burned 14 GPU hours on a
  series that was flat from epoch 1, because it stopped on nothing and the
  diffusion loss fell throughout regardless of sample quality.
- **`BP13`** — measure the foreground *and* background headroom and rewrite the
  motivating example, which is currently prose that motivates nothing. This
  number bounds the whole paper: if the players cost the codec little, the thesis
  needs rethinking, and August is when we want to know.
- **`BP15`** — retire ~15k lines of pre-rewrite code and its 433 tests.

**The open architectural question**, if the roster re-run does not settle it:
appearance must enter through a *trained* pathway. ControlNet adds its condition
residually and relies on CLIP embeddings that lack fine spatial detail — a
documented limitation, which makes our negative result citable rather than
embarrassing. ReferenceNet-family models (Animate-Anyone, and the SD-1.5 siblings
Champ and MusePose, both licence-clean) inject reference features into the UNet's
spatial self-attention instead. That is the direction if a fix is needed.

## One thing to hold on to

The negative results here are real and were expensively earned. They belong in
the paper as scoped findings — *these checkpoints* cannot do this, for a reason
the literature already documents — not as evidence that the architecture fails.
The lattice, the residual and the background are independently verified. Do not
let the generator result contaminate them, and do not soften it into "more tuning
needed" either.
