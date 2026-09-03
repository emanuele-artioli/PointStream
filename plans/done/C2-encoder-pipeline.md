# C2 — The encoder pipeline DAG

**Wave 2, parallel.** Shares no files with `C1`; coordinate at the boundary
through the contracts, not by editing each other's modules.

**Owns exclusively:** `src/pipeline/dag/**`, `src/pipeline/encoder/**`,
`tests/pipeline/test_dag*.py`, `tests/pipeline/test_encoder*.py`.
**Read first:** `src/contracts/lattice.py` (699 lines — the stage vocabulary and
the lattice), `src/contracts/config.py`, `plans/done/RESEARCH-HISTORY.md` §3.

## What to build

**A stage DAG built from the enabled stage set**, where the pipeline never knows
which backend was chosen — it asks the registry. Dependencies point inward
always, enforced by `python -m src.contracts.layers`.

**Skips that make a reduced corner genuinely cheaper, not nominally.** This is
the load-bearing requirement. If detection is disabled, the detector must not run
and its cost must not appear; a corner that "disables" a stage but still pays for
it makes every ablation number meaningless, because the lattice is measured in
BD-rate against corners that are supposed to differ in cost.

**Every lattice corner produces a runnable pipeline.** No stage is structurally
required except codec, transport and metrics (`plans/done/RESEARCH-HISTORY.md` §3). Graceful degradation
to the baseline codec is a property of the architecture, not a routing special
case — there is no `if baseline:` branch.

## Traps specific to this stream

**The arrangement being replaced had experiment scripts shell out to the CLI and
scrape stdout.** That is what an unchecked boundary decays into. `pipeline/`
is consumed as a library, never through a subprocess.

**A disabled stage is a configuration, not a code path.** `none` is a registered
backend on most axes precisely so that turning something off does not require a
branch. Use it.

**Test the corners, not just the happy path.** The required-behaviour suite
(`plans/done/RESEARCH-HISTORY.md` §8) asserts *every lattice corner produces a runnable pipeline*. That
is this stream's assertion to deliver, and it is a real check with real
combinatorics behind it — enumerate from `contracts/lattice.py` rather than
hand-listing a few.

**Do not build the runner.** C3 owns the single run path — chunk loop, routing,
accounting, evaluation. Stop at the DAG and the encoder.

## Done when

- The DAG is built from the enabled stage set, with no backend knowledge.
- A disabled stage demonstrably costs nothing — proven by measurement, not by
  reading the code.
- Every lattice corner enumerated from the contracts produces a runnable
  pipeline.
- `ruff`, `mypy`, tests pass; import direction clean.
