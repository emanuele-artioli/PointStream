# B′3 — Generator weight loading: the comparison backbone

**Wave 1.** Independent of B′1 and B′2 at the code level; its *probe* needs both,
which is Wave 2 (`BP5`).

**Owns exclusively:** `src/components/generation/controlnet.py`, `pix2pix.py`,
`spade.py`, `upscale.py`, `_numpy.py`, and `tests/components/test_generation.py`.
**Does not own** `animate_anyone.py` or `mofa.py` — those are B′4's.
**Read first:** `src/contracts/conditioning.py`, `PLAN.md` §6.2.

## The defect

Every `_load_model` and `_load_pipeline` in this package is an unconditional
`raise RuntimeError(...has no pipeline loaded...)`. Ten generators are
registered; driving all ten, **only `upscale-refine` — the non-generative bicubic
baseline — produces pixels.** The tests pass because they inject a `_FakePipe`,
and the one `@pytest.mark.integration` test asserts merely that a checkpoint
*path was recorded*.

The contracts work around it is good and stays: `ConditioningBundle` typing,
declared temporal capability, pairing validation. Only the loaders are missing.

## What this stream is for

This is the **comparison backbone** (`PLAN.md` §6.2) — the family that serves
`subsec:eval-object`, where the paper promises the generative backbone is *held
fixed across arms* while the conditioning signal changes. ControlNet on SD-1.5 is
the only family that can honour that promise, which is why it gets its own
stream and why the trajectory arm lands here rather than with a separate model.

## What to build

Replace the `raise` with a real load, in this order:

| Order | Engine | Weights on disk |
|---|---|---|
| 1 | `pose-controlnet` | `assets/weights/pose-controlnet` (10 fine-tuned epochs) + `stable-diffusion-v1-5` |
| 2 | `seg-controlnet` | `assets/weights/seg-controlnet` (7 epochs) |
| 3 | `ip-adapter-controlnet` | `assets/weights/ip-adapter-controlnet` (10 epochs) |
| 4 | `pix2pix` | `assets/weights/pix2pix_generator.pt` |
| 5 | `spade4tennis` | `spade4tennis_lite_generator.pt` — wire if cheap, drop if not |

**Stop and report after 1.** If a fine-tuned pose-ControlNet will not produce a
plausible frame, the roster changes and the rest of this list is wasted work.

**Then: the trajectory arm.** Render sparse trajectories as a control image into
the *same* backbone the keypoint arm uses. This is what makes `eval-object`'s
"backbone fixed" literally rather than approximately true. Do not reach for
MOFA-Video — it is licence-blocked and routing around it is the better
experiment (`plans/DEFERRED.md` D4).

**Fix the duplicated pose-rescale block** that `plans/B3-generation.md` names:
~40 lines copy-pasted across four ControlNet classes and visibly wrong. It
changes generated pixels, which is in scope — pre-rewrite generative results are
superseded anyway.

## Traps specific to this stream

**A frame returned is not a frame generated.** Assert the output differs from the
input. An identity passthrough scores deceptively well and means the model never
ran. This is the single most important check in the stream.

**Below ~15 PSNR, suspect the inference path before the model.** That exact
error has been made here once already: ControlNet scored 0.11 VMAF and it was a
broken path, not a weak model. A near-zero score is a bug report, not a finding.

**Ten checkpoint epochs are on disk — say which one you loaded.** Silently
picking `checkpoint-epoch-10` because it sorts last is how an unreproducible
number is born. Record the epoch with every result.

**Fix the seed and record it.** Generation is statistical, and `PLAN.md` §3
requires encoder-side and client-side reconstruction to be *measured* for
closeness rather than assumed identical. That measurement is meaningless without
a recorded seed.

**Do not tune anything.** Parameter tuning, fine-tuning and model swaps are
decisions for after `BP5` fixes the roster. Tuning an engine that is about to be
dropped is the most expensive way to waste September.

## Done when

- Each engine above loads real weights and returns a frame demonstrably different
  from its input, or is dropped with a recorded reason.
- The trajectory-render arm exists and shares the keypoint arm's backbone.
- Seed and checkpoint epoch travel with every generated output.
- The `_FakePipe` tests still pass — they test contract shape and remain useful.
- `ruff`, `mypy`, tests pass; import direction clean.
