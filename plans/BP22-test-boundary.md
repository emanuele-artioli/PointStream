# B′22 — Finish the cull, and decide what `src/shared/` *is*

**Supersedes the unfinished half of `BP15-test-cull.md`.** BP15 landed as PR #19
and deleted real weight, but its central premise — *"only three modules in the
pre-rewrite tree are still imported by new code"* — is no longer true, and that
is why it stopped. Read BP15 for the history; take the numbers below as current.

**Owns:** `src/decoder/**`, `src/shared/**`, the remaining 32 top-level
`tests/test_*.py`. **Read first:** `AGENTS.md`, `PLAN.md` §2, `plans/BP15-test-cull.md`.

## Where the cull actually got to

Measured on `main` with all three wave-3 PRs merged (2026-08-25):

| | BP15 start | now | target |
|---|---|---|---|
| pre-rewrite tests (top-level `tests/test_*.py`) | 433 in 69 files | **220 in 32 files** | 0 |
| `src/encoder` | 6378 lines | **gone** | gone |
| `src/main.py` | present | **gone** | gone |
| `src/decoder` | 4175 lines | **3115** | only what the new tree needs |
| `src/shared` | 4782 lines | **4449** | see the decision below |

So roughly half. The suite is 981 tests; **761 of those are the rewrite suite**
(components 396, contracts 196, pipeline 110, experiments 34, invariants 14,
runner 11) and are not the target. The 220 are.

## Why it stopped, and the decision this brief exists to force

**The boundary got wider, not narrower.** BP15 named three inbound edges. There
are now twelve-ish:

| Module | Callers |
|---|---|
| `src.shared.tennis_dataset` | 7 |
| `src.shared.experiment_evaluation` | 4 |
| `src.shared.video_io` | 3 |
| `src.shared.racket_heuristic` | 3 |
| `src.shared.player_extraction` | 2 |
| `src.shared.{lpips_metric, hnerv_arch, geometry, dwpose_draw, config}` | 1 each |
| `src.decoder.genai_compositor` | 1 |

The three modules BP15 said to **move** (`torch_dtype`, `spade4tennis_arch`,
`animate_anyone_runtime`) were **deleted instead of ported**. That is fine only
if nothing wanted them; say which is true, because BP15 step 1 asked for a port.

**And BP14 built new code inside the demolition zone.** `src/shared/training/stop.py`
and `src/shared/training/task_eval.py` are new, current, and load-bearing — they
are the BP14 stop rule. `src/shared/` is therefore no longer "the pre-rewrite
tree"; it is a mix. **Nothing else in this brief can proceed until that is
resolved**, because "delete `src/shared`" and "keep the stop rule" cannot both
be true.

Pick one and record it in `PLAN.md` §3:

- **(a) `src/shared/` becomes a real layer** with a contract and a place in
  `src.contracts.layers`, and the pre-rewrite files move *out* of it or die.
- **(b) `src/shared/` stays condemned**, and `training/` plus every module in the
  table above moves into `src/components/` or `src/pipeline/` first.

(b) matches the rewrite's stated architecture; (a) is less work. Either is
defensible — an undeclared mixture is not.

## What to do

1. **Make the decision above and write it down** before touching a file.
2. **Classify all twelve edges**: port into the new tree, or rewrite the caller.
   `tennis_dataset.py` is live for training (`scripts/train_controlnet.py`) —
   see the collision note in the wave plan; it does not move this wave.
3. **Delete the rest with their tests**, one commit per subtree.
4. **Port, don't drop.** If a pre-rewrite test covers behaviour the new tree has
   and does not test, port it and say which. A smaller suite is not the goal.
5. Re-run `python -m src.contracts.layers` and the suite after each commit.

## Done when

- `src/shared/`'s status is decided, recorded, and true of the tree.
- The 220 pre-rewrite tests are gone or ported, and the report says which.
- `python -m src.contracts.layers`, `ruff`, `mypy` and the suite are green.
- The report states the new test total and its split by directory.

## Traps

- **Do not delete a test to make the suite green.** `DEFERRED.md` D6's `xfail`s
  go when their module goes, not before.
- **Read before deleting** — this tree has been mined twice for real findings.
- **`scripts/train_controlnet.py` must keep working**: a GPU run is live on it
  this wave.
