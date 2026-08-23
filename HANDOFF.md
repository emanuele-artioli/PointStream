# PointStream — handoff, 2026-08-23 (evening)

You are picking up a rewrite of an object-centric semantic video codec, targeting
**ACM TOMM, September 30**. Everything is merged to `main`. The suite is green:
1050 passed, 3 xfailed, ruff and mypy clean, import direction clean.

## Read, in this order

1. `AGENTS.md` (project) and the host rules it imports.
2. `PLAN.md` — especially **§2.3 through §2.10**. §2.10 is the newest and
   supersedes the engine readings in the others.
3. `plans/README.md` — what is live, and what is void.
4. Your one brief from `plans/`. Do not read the whole tree.

## The situation in five sentences

The component platform is built and works: 16 axes, a rebuilt probe set, a
pipeline, a runner, region-scoped metrics, and now a probe harness that drives
temporal models as clips and refuses to rank when its own control fails. **No
generative engine produces a usable player, and this is now measured properly
rather than suspected**: all eight lose to *pasting the keyframe* at 2.5σ–10.6σ,
and the best of them is `upscale-refine`, which is not a generative model. The
test that was supposed to decide whether any engine "uses appearance" has been
withdrawn, because a pasted keyframe tops its scale with no network at all. The
paper's central rate-distortion claim against the codec ladder **has still never
been run in any configuration**.

## What you must not trust

- **Every engine ranking taken before 2026-08-23.** Two of three metrics were
  broken: LPIPS was an uncalibrated VGG feature distance that scored an unrelated
  image at 0.083 and a good reconstruction at 0.085; VMAF had its ffmpeg inputs
  crossed and scored blur above a perfect match. Both are fixed and both now have
  calibration invariants.
- **`plans/BP10-appearance-pathway.md` — void, and marked so in its own file.**
  Its gate "≥ +3 dB = ReferenceNet works" certifies a paste, which scores
  +4.45 dB. Any conclusion drawn from a cross-appearance delta alone is void with
  it.
- **Anything in `plans/done/` that ranked engines.** `BP5`'s roster verdict was
  measured on self-reconstruction.
- **PSNR as a ranking key for generative arms.** Usable range is ~11–21 dB with a
  ~2 dB per-clip sd. Rank on calibrated LPIPS; keep PSNR reported alongside. On
  this roster pix2pix is 2nd on PSNR and 7th on LPIPS — the orders genuinely
  differ.

## Working rules that were learned expensively

**Eight times now, something passed its tests while not doing its job** — ten
generators registered that could not load weights; a probe verifier green while
five clips had no pose data; a roster ranked on self-reconstruction; a temporal
model run at T=1; a metric that could not tell a match from noise; a VMAF wiring
where blur beat perfection; a training run with no stopping criterion; and a
*control* that ranked four engines before anyone asked what an arm with no model
scores on it.

So:

- **Use the `verify-measurement` skill before reporting any measurement.**
- **A control needs its own null.** When a control produces a ranking, run the
  degenerate arm through it — the paste, the passthrough, the empty model —
  *before* reading the ranking. That is the newest rule and it is the one that
  caught the latest error.
- **Write the bound down before the number, and the failure branch with it.**
  BP12's cross-appearance prediction named what "the ControlNets come in above
  AA" would mean. They did, and the pre-written branch is what stopped a wrong
  claim.
- **Quote the instrument's range beside the number.** "0.067" is meaningless.
- **Use `src.components.metrics.comparison.compare_paired`** for any arm-vs-arm
  claim, and `python -m experiments.probe.report <run-dir>` for a whole run.
- **Check the invocation before blaming the model.**
- **When the news is good, add a check rather than stopping.**

## Environment

- `conda run -n pointstream --no-capture-output <cmd>`; imports absolute from the
  repo root. Pass `python -u` for long detached runs — stdout block-buffering
  delayed BP12's progress lines by minutes.
- Packages go in `pyproject.toml` and then get installed — never ad-hoc, and
  never a version bump on a pinned forked model.
- Before merging: `ruff check`, `mypy --config-file pyproject.toml`, the tests
  for what you touched, `python -m src.contracts.layers`.
- Three known `xfail`s, each with a reason: `DEFERRED.md` D5 and D6.
- The paper is a **separate git repo** at `67a9ea6275d3d9785ce57026/`. Commit
  there when you change it. **It does not yet know about §2.10.**

## What to do next

**`plans/BP13-motivating-headroom.md` is the critical path.** Encode a clip
normally, encode it again with the player regions flattened, difference the
bitrates. That number bounds the entire paper: the players are 1.07% of a 4K
frame each (§2.6), and if a conventional codec spends 3% of its bits on them
there is no prize here regardless of how good a generator gets. Nothing in BP12
changes this and BP12 makes it more urgent — we now know the generator side is
not close, so the premise had better be worth the trouble.

Then, in order:

- **`BP14`** before any training run. The last one burned 14 GPU hours on a
  series flat from epoch 1.
- **`BP15`** — retire ~15k lines of pre-rewrite code and its 433 tests.
- **The paper.** §2.10 is not in it yet. The Evaluation skeleton's `GOAL`/`HOLE`
  markers now have a real negative result to absorb, and `subsec:eval-operating`
  can be filled: clip mode costs 6.2 GiB against 3.3 for a ControlNet, both at
  ~1 s/frame.

**The open architectural question is now differently shaped.** It is no longer
"does ReferenceNet work" — nothing here can answer that. It is **"what
measurement would tell us?"**, and the literature's answer is an identity metric
(CSIM/ArcFace), which this project does not have. Adding Champ or MusePose before
that exists buys two more arms that lose to a pasted keyframe. Build the
instrument first; that is the lesson of the last three weeks in one sentence.

## One thing to hold on to

The negative results here are real, expensively earned, and now properly
controlled. They belong in the paper as scoped findings — *these checkpoints, on
this task, measured this way* — not as evidence that the architecture fails. The
lattice, the residual and the background are independently verified. Do not let
the generator result contaminate them, and do not soften it into "more tuning
needed" either.
