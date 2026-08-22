# B′2 — Region-scoped evaluation

**Wave 1.** Independent of the other Wave-1 streams.

**Owns exclusively:** `src/components/metrics/**`, `tests/components/test_metrics*.py`.
**Read first:** `PLAN.md` §6.4 and §6.5.

## The defect

**A frame-level score hides a broken object.** A reconstruction whose background
is perfect and whose player is mangled still posts a respectable frame PSNR,
because the player is a small fraction of the pixels. Every generative failure
this project cares about is exactly that shape.

Today `src/components/metrics/` has **no concept of a region at all** — no mask,
no crop, no bounding box anywhere in the package. `Evaluator` scores whole
frames. Every number B′ produces would therefore be measuring mostly background
and reporting it as if it said something about generation.

## What to build

**Region-scoped scoring, as a first-class part of the metric contract.**

- A metric is computed over a **region**, given as a mask or a bounding box, not
  only over a frame.
- **Object generation is scored on the object crop or mask.** Background
  modelling is scored on the background region with objects excluded.
- **The whole-frame score is reported as well, never instead.** Both numbers
  matter and they answer different questions; dropping either is what got us
  here.
- The record says **which region each score was computed over**. A score whose
  scope is unstated is not usable in a paper.

Wire it through `Evaluator` so that scoping is the normal path rather than an
option a caller can forget. A caller that supplies no region gets a whole-frame
score *labelled as such*.

## Metric discipline, and why this stream enforces it

Per `PLAN.md` §6.5:

- **Triage and development: PSNR only.** Fast, always comparable, enough to
  answer "did this run and produce something plausible".
- **Paper results: the full set** — VMAF, SSIM, LPIPS, FVMD — once the roster is
  fixed and the runs are the ones we intend to cite.

VMAF in particular is slow and buys nothing during triage. Make the cheap path
the easy one to call, so nobody reaches for the expensive set by accident.

## Traps specific to this stream

**A tiny region makes PSNR jumpy.** A 40x80 player crop has few pixels and its
PSNR is noisier than a frame's. Report the region's pixel count with the score
so a wild number can be recognised as a small-sample artefact rather than a
result. Consider a minimum region size below which the score is refused.

**Masks and crops are not interchangeable.** A bounding-box crop includes
background pixels around the subject; a mask does not. Scoring a generated
player against a box flatters it, because the easy background pixels are inside
the box. Support both, label which was used, and prefer the mask where one
exists.

**Do not resample the reference to match the prediction.** If shapes disagree
that is a bug upstream, and silently resizing hides it.

**The existing whole-frame behaviour is verified working** (`plans/B6-metrics.md`
records the numbers). Do not regress it: PSNR `inf` on identical frames, VMAF
97.4/28.9, SSIM 1.0/0.9885, LPIPS 0.0/0.00108, FVMD refusing `T=1`.

## Done when

- A metric can be computed over a mask or a box, and the record says which.
- Object-scoped and background-scoped scores are both reachable, with whole-frame
  reported alongside rather than replaced.
- A test proves the motivating case: a reconstruction with a **perfect background
  and a destroyed object** posts a good frame PSNR and a bad object-scoped PSNR.
  That test is the point of this stream.
- Region pixel counts travel with the scores.
- `ruff`, `mypy`, tests pass.
