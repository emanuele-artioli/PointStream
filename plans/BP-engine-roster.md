# B′ — The engine roster

**The next workstream, and the critical path.** Everything in `PLAN.md` §7 P0
except item 1 is blocked behind it.

**Owns exclusively:** the loader bodies in `src/components/generation/**`,
`tests/components/test_generation*.py`, and a new probe harness under
`experiments/`.
**Read first:** `PLAN.md` §6 (the roster and why each engine is on it),
`src/contracts/conditioning.py`, and the paper's Evaluation section — the
`GOAL` markers there are what this stream exists to serve.

## The rule this stream is built on

**The narrative drives the experiments; the experiments drive the models.** An
engine is wired because a named `GOAL` in the paper cannot be answered without
it. An engine no `GOAL` needs is not wired, however easy it would be. When in
doubt about whether to build something, find the marker it serves — if there
isn't one, that is the answer.

We do not need every generator to work. We need a **flagship**, plus
alternatives that differ along an axis the paper measures.

## What to build

### B′.1 — wire the loaders

Replace the unconditional `raise` in each `_load_model` / `_load_pipeline` with
a real load, in this order. Stop and reassess after the first two; if ControlNet
will not produce a plausible frame, the roster changes and the rest is wasted.

| Order | Engine | Weights already on disk |
|---|---|---|
| 1 | `pose-controlnet` | `assets/weights/pose-controlnet` (10 fine-tuned epochs) + `stable-diffusion-v1-5` |
| 2 | `seg-controlnet`, `ip-adapter-controlnet` | 7 and 10 epochs respectively |
| 3 | `pix2pix` | `pix2pix_generator.pt` |
| 4 | `animate-anyone` | fine-tuned on **one match** — the caveat travels with every number |
| 5 | `spade4tennis` | `spade4tennis_lite_generator.pt`; wire if cheap, drop if not |

`upscale-refine` already works and needs nothing.

**Do not wire MOFA-Video.** Licence-blocked, and routing around it improves the
experiment — see `PLAN.md` §6.2.

### B′.2 — probe on the minimal set

Drive every wired engine over `assets/probe_set` — the existing minimal set, not
the full one — and record PSNR, SSIM, VMAF and LPIPS per engine per scene.

**This is triage, not results.** Nothing measured here is citable, nothing gets a
`CLAIM` line, and nothing goes in the paper. Its only job is to answer: which
engines produce a plausible frame, and how far apart are they?

### B′.3 — fix the roster in writing

Confirm or replace the flagship on the evidence, and record the decision and its
reason in `PLAN.md` §6.2. Only then does the full dataset get prepared.

### B′.4 — the two invariants B should have written

`PLAN.md` §8 names a required-behaviour suite that does not exist yet. Two of its
assertions are checkable now and belong to this stream, because this stream is
where weights start mattering:

- every registered backend constructs, or fails with a stated reason;
- every weight a shipped config names resolves to a real file.

## Traps specific to this stream

**Bound before believing.** These checkpoints are lightly trained, some on a
single video, some not trained by us at all. **VMAF 25–45 on a first pass is the
expected outcome and counts as success** — it means the engine runs and the
measurement is honest. Below ~15, suspect a broken inference path before
concluding the model is weak: that exact error has already been made once, when
ControlNet scored 0.11 VMAF and it was a broken path, not a bad model. Above ~70
on a first pass, suspect the reference rather than celebrating.

**A frame returned is not a frame generated.** Check the output differs from the
input — an identity passthrough scores deceptively well on PSNR and means the
model never ran. The probe harness asserts this explicitly.

**Do not start improving the numbers during B′.** Parameter tuning, fine-tuning
on our dataset, and swapping a model are all real options, in that order of cost,
and all of them are decisions for *after* B′.3 fixes the roster. Tuning a model
that is about to be dropped is the most expensive way to waste September.

**Weak results here are findings, not failures.** `subsec:eval-general` exists to
report the gap between a fine-tuned backbone on its own domain and a pretrained
one off it. Expect the fine-tuned ControlNet variants to beat pretrained on
tennis and lose on DAVIS, and expect Animate-Anyone to look strong on its own
match and poor elsewhere. Both outcomes are exactly what that subsection was
written to carry.

**Determinism.** Generation is statistical, and `PLAN.md` §3 requires
encoder-side and client-side reconstruction to be *measured* for closeness rather
than assumed identical. Fix the seed and record it with every probe number.

## Done when

- Every engine on the `PLAN.md` §6.2 roster loads real weights and returns a
  frame demonstrably different from its input.
- Each has a probe-set number, with its seed, under `outputs/`.
- The roster is fixed in writing, each engine carrying the reason it is on it.
- The two invariants in B′.4 exist and pass.
- `ruff`, `mypy`, tests pass; import direction clean.
