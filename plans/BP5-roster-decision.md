# B′5 — Probe, then fix the roster

**Wave 2. Depends on B′1, B′2, B′3 and B′4 all reporting back.** It cannot start
earlier: it needs a trustworthy probe set (B′1), region-scoped scoring (B′2), and
engines that actually load (B′3, B′4).

**Owns exclusively:** `experiments/probe/**`, `tests/invariants/**`, and the
roster section of `PLAN.md` §6.2.
**Read first:** `PLAN.md` §6.2–6.6, and the reports from all four Wave-1 streams.

## What to build

### 1. The probe harness

Drive every wired engine over the rebuilt probe set. **Region-scoped PSNR only** —
no VMAF, no LPIPS, no FVMD (`PLAN.md` §6.5). Record per engine, per clip:

- object-scoped PSNR, and whole-frame PSNR alongside it;
- the seed, the checkpoint epoch, peak VRAM, wall-clock per clip;
- the assertion that the output differs from the input.

**This is triage, not results.** Nothing here is citable, nothing gets a `CLAIM`
line, nothing enters the paper. Its only job: which engines produce a plausible
frame, and how far apart are they?

### 2. Fix the roster in writing

Update `PLAN.md` §6.2 with the decision and, for each engine, the reason it holds
its slot. Two questions to settle on evidence:

- **Does Animate-Anyone keep the quality-flagship slot**, or does the modern
  candidate take it?
- **Do the two flagship roles collapse into one?** If the comparison backbone is
  also the best performer, say so and simplify the narrative.

### 3. The two invariants Phase B should have written

`PLAN.md` §8 names a required-behaviour suite that is currently a three-test stub
that skips. Two of its assertions are checkable now and belong here, because this
is where weights start mattering:

- every registered backend constructs, or fails with a stated reason;
- every weight a shipped config names resolves to a real file.

## Bound before believing — write these down before looking

**Expected: object-scoped PSNR in the high teens to mid twenties on a first
pass**, and that is a **pass**. It means the engine runs and the measurement is
honest. These checkpoints are lightly trained, one of them on a single video, and
several are pretrained models never adapted to broadcast tennis.

- **Below ~10 dB, suspect the inference path before the model.** That error has
  been made here once already, when ControlNet's 0.11 VMAF was read as a model
  result and was in fact a broken path.
- **Above ~35 dB object-scoped on a first pass, suspect the reference.** Check
  you are not scoring the source against itself, and that the region really is
  the object rather than a box full of background.
- **A whole-frame PSNR much better than the object-scoped one is the expected
  shape**, not an anomaly — it is the exact effect §6.4 exists to expose. A
  *small* gap between them is the surprising outcome and deserves a second look.

**When a bound turns out to be wrong, record why.** One of ours once fired
against a correct result because it had been derived in the wrong units, and that
is as worth knowing as the result itself.

## Traps specific to this stream

**Weak results are findings, not failures.** `subsec:eval-general` exists to
report the gap between a fine-tuned backbone on its own domain and a pretrained
one off it. Expect the fine-tuned ControlNet variants to beat pretrained on
tennis and lose on DAVIS; expect Animate-Anyone to look strong on its own match
and poor elsewhere. Both are what that subsection was written to carry.

**Do not start improving anything during the probe.** Parameter tuning,
fine-tuning on our dataset, and model swaps are all real options, in that order
of cost, and all are decisions for *after* the roster is fixed.

**A gap you cannot explain is a stop condition.** If two engines differ by 15 dB,
find out why before ranking them. The most likely cause is a wiring difference,
not a quality difference.

**When something has no marker to serve, ask rather than dropping it.** The paper
is a work in progress and does not yet contain every `GOAL` we will want; a
component with no home may be telling us the evaluation is missing something.

## Done when

- Every wired engine has a region-scoped probe number, with seed, checkpoint and
  cost, stored under `outputs/`.
- The roster is fixed in writing, each engine carrying its reason.
- The two invariants exist and pass.
- A short report says what changed about the narrative, if anything.
- **Only then** does the full dataset get prepared.
