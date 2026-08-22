# B′4 — The quality flagship

**Wave 1.** Independent of B′1–B′3 at the code level.

**Owns exclusively:** `src/components/generation/animate_anyone.py`,
`src/components/generation/mofa.py`, any **new** engine wrapper it adds, and
`tests/components/test_generation_flagship.py`.
**Does not own** the ControlNet family — that is B′3's.
**Read first:** `PLAN.md` §6.2 and §6.3.

## What this stream is for

`subsec:eval-ladder` carries the paper's central claim, and its figure must show
**the best PointStream can do** — not the most convenient engine. This stream
owns that slot. It also serves `subsec:eval-metrics`, because a meaningful FVMD
claim needs an engine that actually models time.

## The problem with the incumbent

Animate-Anyone holds the slot today by default rather than on evidence, and there
are two reasons to doubt it:

1. **Its checkpoint was fine-tuned on a single tennis match.** Every number it
   posts is scoped to that match, and that caveat travels with each one. It is
   not a general model and cannot honestly carry `subsec:eval-general` alone.
2. **It is old.** Recent comparisons repeatedly place it behind newer work,
   citing face and body distortion where the newer models hold up.

## What to build

**First, make the incumbent evaluable.** Wire `animate_anyone.py` to its
checkpoint and get a number. We cannot judge a replacement without a baseline,
and `scripts/eval_checkpoint.py`'s `ARCH_CHOICES` still has no entry for it.
A full retrain is explicitly *not* in scope (`PLAN.md` §7 P2 item 17).

**Then, evaluate one modern candidate.** Surveyed 2026-08-22, in adoption-cost
order:

| Candidate | Why | Cost |
|---|---|---|
| **StableAnimator** | Apache-2.0, weights on HF, ~10 GB VRAM for the 16-frame UNet (VAE decoder wants 16 GB, can run on CPU). Best reported identity preservation and FVD in its class. **Start here.** | low |
| **MTVCrafter** | SOTA on TikTok, +65% FID-VID over second best. Tokenises raw 4D motion instead of 2D pose images — *directly relevant to our motion-representation axis*, so it may matter beyond this slot. | medium |
| DisPose, Animate-X, StableAnimator++ | Incremental over the above | defer |
| **Sparse2Dense** (DCC 2026) | Architecturally a *generator* in our terms — VVC-coded reference frame + 3D keypoints. Would slot straight into our contract. | **check first** |

**On Sparse2Dense:** no public code or weights were found as of 2026-08-22.
**Check once whether that has changed** — if released, it is a strong candidate
backend rather than only related work, because its interface is already ours.
If not released, record that and move on; do not attempt a reimplementation.

**Adopt exactly one to begin with.** Wrap it against the existing
`ConditioningBundle` contract like any other backend — if the contract fights the
model, report that, because it is a finding about the contract.

## Traps specific to this stream

**Licence-check before integrating, not after.** This is why MOFA-Video is
stranded (`plans/DEFERRED.md` D4). StableAnimator reports Apache-2.0 — *verify it
on the model card yourself* rather than trusting a summary, and record what you
found. Weights that cannot ship cannot be a flagship.

**A newer model is not automatically our better model.** These are benchmarked on
TikTok and TED-Talk: single centred subject, clean framing. Ours is broadcast
tennis — small figures, motion blur, occlusion, a moving camera. A model that
wins on TikTok may lose here, and **that is a publishable finding**, not a
failure. Report it either way.

**VRAM is a real constraint and a real result.** If the flagship needs 16 GB to
decode one clip, `subsec:eval-operating` has to say so. Record peak memory and
wall-clock per clip alongside quality, from the first run.

**Do not let the flagship quietly become the comparison backbone.** They serve
different questions (`PLAN.md` §6.2). If this engine also wins `eval-object`, the
roles collapse and the narrative gets simpler — but that is `BP5`'s call to make
on evidence, not an assumption to build on.

## Done when

- Animate-Anyone loads its checkpoint and posts a region-scoped number, scoped in
  writing to its single match.
- One modern candidate is wrapped, licence-verified, and posts a number on the
  same clips.
- Peak VRAM and wall-clock per clip are recorded for both.
- A short written comparison goes to `BP5`, including the case for keeping the
  incumbent.
- `ruff`, `mypy`, tests pass; import direction clean.
