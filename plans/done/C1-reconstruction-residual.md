# C1 — Reconstruction and the residual

**Wave 2, parallel.** Independent of `BP5`: this is pipeline structure, and it
does not depend on which generator wins the roster.

**Owns exclusively:** `src/pipeline/reconstruction/**`,
`src/pipeline/residual/**`, `tests/pipeline/test_reconstruction*.py`,
`tests/pipeline/test_residual*.py`.
**Read first:** `src/contracts/` (authoritative), `PLAN.md` §3 and §6.

## What is being replaced

Two pre-rewrite modules that each do far too much:

- **`src/shared/synthesis_engine.py`** (512 lines) — panorama resolution, pose
  unrolling, ball rendering, generative dispatch, CUDA determinism and OOM
  fallback, all in one class.
- **`src/encoder/residual_calculator.py`** (967 lines) — the residual.

Neither is assumed correct. They are prior art to read, not a foundation to
trust.

## What to build

**Reconstruction**, decomposed so each concern is separately testable:
panorama/background resolution, object placement and compositing, generative
dispatch through the registry (never by class identity or name matching), and
the device/OOM fallback as an explicit policy rather than scattered `try`.

**The residual coarseness spectrum, including absent.** This is a rate axis in
its own right (`PLAN.md` §5, `subsec:eval-residual`): absent → progressively
coarser → fine → lossless. The lossless setting is a ceiling calibration, not an
operating point. The absent setting is the one that reports the unaided quality
of the reconstruction itself, and it is measurable **before** any generator
question is settled — which is why this stream can start now.

**Bit-identity tests for deterministic stages.** `PLAN.md` §3 requires
deterministic stages to be checked for bit-identity and generative stages to be
measured for closeness. Deliver the former here.

## Traps specific to this stream

**The residual absorbs what disabled stages would have handled.** That is the
whole basis of the ablation lattice: turn off detection and the residual grows to
carry the players; turn everything off and what remains is the source video. If
a disabled stage makes the residual *smaller*, something is wrong.

**Encoder-side and client-side reconstruction are not guaranteed identical.**
Generation is statistical. Symmetry is a design goal *verified by measurement*,
never asserted by construction. Build the closeness measurement in; do not add an
assertion that they match.

**The all-off corner must reduce to the source video.** It is a Phase-C gate and
worth testing from the first commit rather than discovering at the end.

**Quality is measured on every path.** There is no reconstruction path that skips
evaluation — `PLAN.md` §3. Wire it in from the start; retrofitting it is how the
last iteration ended up with paths that reported nothing.

**Use the region-scoped metrics from `BP2`.** A whole-frame score hides a broken
object (`PLAN.md` §6.4). Background reconstruction is scored on the background
region, object reconstruction on the object.

## Done when

- Reconstruction is decomposed, with each concern independently testable.
- The residual spectrum runs end to end including absent and lossless.
- Deterministic stages have bit-identity tests.
- The all-off corner reduces to the source video, proven by test.
- Generative dispatch goes through the registry, never through a class check.
- `ruff`, `mypy`, tests pass; import direction clean.
