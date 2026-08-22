# C3 — The runner

**Wave 3.** C1 and C2 are merged on this branch. The runner is the only layer
that may look up registries and bind named backends into that pipeline.

**Owns exclusively:** `src/runner/**`, `tests/runner/**`.
Not `src/pipeline/**` (C1/C2), not `experiments/**` (Phase D).
**Read first:** this worktree's `src/pipeline/{dag,encoder,reconstruction,residual}/`,
`src/contracts/lattice.py`, `src/contracts/config.py`, `PLAN.md` §3.

## What to build

**One run path.** A chunk loop over a match. A single-chunk clip is that loop
with one iteration — not a second function, not a flag that means "skip the
real runner". `Encoder.encode` is per-chunk; this stream iterates it.

**Routing.** Bind config names to the types C1 and C2 actually inject:

- named backends → `StageCallable` (`Mapping[str, Any] -> Any`) for
  `Encoder.build(..., backends=)`. C2 ignores extra keys for disabled stages
  and never invokes them; do not re-introduce that work here.
- chosen generator → `GeneratorRef` via C1's `from_spec`. Pass
  `GeneratorRef.requires` as `conditioning=` into `Encoder.build`.
- metrics → `QualityEvaluator` (`src.pipeline.reconstruction.quality`).
  `reconstruct()` already scores every return; there is no runner path that
  omits this.

**One accounting implementation.** One `sizes_bytes` record per run, built from
`WireCost` and `ResidualPayload.byte_count` / the delivered bitstream — not a
second ledger beside `src/shared/experiment_evaluation.py` and
`src/shared/invariants.py`. Payload parts must be the numbers an invariant
check would read.

**Quality on every path.** Object and background region scores when masks
exist; whole-frame alongside, never instead. All-off and residual-only must
run without constructing a generator. Generators currently lose to a static
copy (`PLAN.md` §2.3); that does not block this stream and is not a reason to
wait on BP8.

**Library.** `experiments/` will import this package (Phase D). No subprocess,
no stdout scrape.

## The C1 / C2 boundary (observed, this worktree)

C2 is type-blind. A `StageCallable` reads the artifact bag (source under
`SOURCE = "source"`) and returns `Any`; `build_dag` stores that value under
the stage name and every artifact in `produces`. Disabled stages are absent
from the graph. `build_dag` already calls `assert_coherent(conditioning=)`.

C1 is typed. `ReconstructionRequest` wants a `BackgroundModelView` (C1's
docstring says "C2 unpacks" — **C2 does not**; it never imports that type),
a `GeneratorRef` (protocol + declared capabilities; dispatch never reads the
name), and a `QualityEvaluator`. `QualityEvaluator` is not re-exported from
`reconstruction/__init__.py`. Generation off does not require a
`GeneratorRef`; background off is zeros and the residual absorbs it.

**How the runner resolves it, without editing C1 or C2:** the background
`StageCallable` *returns* a `BackgroundModelView`, so the bag value is the
view; the runner copies it onto `ReconstructionRequest`. The same
`GeneratorRef` is used for encoder-side generation (residual needs
`ART_GENERATED_FRAMES`) and client-side `reconstruct`. Encoder-side and
client-side clips are compared with `measure_symmetry`, never asserted equal.
Import `QualityEvaluator` from `quality.py`. A one-line C1 `__all__` export
is the only adapter that might be worth it; flag it at review rather than
editing C1 now.

`config.validate()` still does **not** call `assert_coherent()`. That belongs
in contracts. This stream does not grow a third copy: always go through
`Encoder.build` and pass the generator's `requires`.

## Traps specific to this stream

**Two accountings.** The pre-rewrite tree split size ledgers. One
implementation, or the Residual-Guarantee numbers are fiction.

**A preview / dry / "single-chunk" path that skips quality.** Metrics are a
required stage; `reconstruct()` always scores. A result without a
`QualityReport` is a failed run.

**Assuming a working generator.** All-off and residual-only do not need one.

**Subprocess boundaries.** This package is a library, same rule as `pipeline/`.

**`if baseline:`.** All-off is `StageLattice.all_off()` /
`SOURCE_PASSTHROUGH`. It is a corner, not a routing special case.

**Filename-based frame pairing.** Resolve a frame by its position in the
track (`PLAN.md` §2.2). Two naming conventions live in one track group.

**Re-running skipped stages.** C2 already measured that a disabled stage
costs nothing at DAG level. Wrapping every backend "just in case" makes
every ablation number meaningless.

**Asserting encoder/client identity.** Generation is statistical.

## Done when

- One importable `run` (name is bikeshed) takes a `PointstreamConfig` plus
  source chunks and returns reconstruction, a `QualityReport`, and one
  `sizes_bytes`. No subprocess.
- One-chunk and multi-chunk sources call the same function; one-chunk is not
  a branch that skips the loop.
- All-off: reconstructed clip bit-identical to source, quality present,
  residual absent, generator never constructed.
- Residual-only (generation off): runs; quality present; scoped object /
  background scores when masks exist; whole-frame reported as well.
- Disabled-stage callables injected into the roster are not invoked (same
  clock pattern C2 already uses, or an equivalent call-count).
- One sizes record; parts sum to `transport_total` within the existing
  invariant tolerance.
- Encoder vs client closeness is measured, never asserted equal.
- `ruff`, `mypy`, tests for what this stream owns, and
  `python -m src.contracts.layers` — runner may import pipeline and
  components, not `experiments`.
