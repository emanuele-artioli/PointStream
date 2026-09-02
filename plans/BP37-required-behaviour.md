# BP37 — The required-behaviour suite, audited against the list that defines it

**Owns:** `tests/invariants/**`, `PLAN.md` §8, `scripts/check_coverage_gate.py`.
**Does not own** `tests/runner/**` while PR #45 is open.

**Read first:** `AGENTS.md` (never add a test to raise a coverage number) ·
`PLAN.md` §8 in full · `plans/done/BP27-metric-invariants.md` ·
`plans/DEFERRED.md` D5 and D6.

**No result dependency.** Half a day.

---

## 0. §8 describes the gate and then describes itself wrongly

`PLAN.md` §8 names eleven required behaviours, then closes with:

> *"The suite does not exist yet. `tests/invariants/` is a three-test stub that
> skips for want of a run summary."*

As of 2026-09-02 that directory holds **five modules and 1,145 lines**, and they
pass — `test_backend_constructs.py`, `test_metric_calibration.py` (679 lines),
`test_named_weights_resolve.py`, `test_outputs_tree.py`, `test_run_summary.py`.
Two of the three things §8 says are missing were written by BP′ as §8 asked.

So the first job is not writing tests. It is **auditing the eleven against what
exists** and making §8 tell the truth, because a gate description that
understates the gate is as misleading as one that overstates it: a session
reading §8 today concludes it has no safety net and behaves accordingly.

## 1. The audit — one row per named behaviour

Produce this table, with a test path or a gap for every row. First-pass reading
from a grep, to be confirmed rather than trusted:

| §8 required behaviour | looks covered by | verdict |
|---|---|---|
| every metric calibrated against known anchors | `tests/invariants/test_metric_calibration.py` | confirm the absolute-scale half, not just ordering |
| bit-identity for deterministic stages | `test_background_intra_sidecar.py`, `test_panorama_encoder.py`, `test_background_stream.py` | **partial** — component-level, not stage-level; no invariant asserts it |
| every lattice corner produces a runnable pipeline | `tests/contracts/test_lattice.py`, `tests/pipeline/test_dag.py` | **check** whether every corner is enumerated or a sample |
| config rejects unknown keys | scattered `unknown` tests per component | **likely a gap** as a global invariant |
| codec constraint violations raise | `tests/components/test_codec.py` | covered |
| an undecodable appearance/motion pair is rejected | `tests/components/test_codec.py`, `test_background.py` | confirm |
| no layer imports outward | `python -m src.contracts.layers` | covered, but **not run by CI** — check |
| every registered backend constructs | `tests/invariants/test_backend_constructs.py` | covered |
| every domain profile round-trips | `tests/components/test_domain.py` | covered; `BP36` will add the driving half |
| every weight a shipped config names resolves | `tests/invariants/test_named_weights_resolve.py` | covered |
| every run emits at least one quality metric | `tests/invariants/test_run_summary.py` | covered |

**Write only the rows that come back as gaps.** `AGENTS.md` is explicit that a
test written to move a number makes the gate lie, and that applies to a
completeness table exactly as much as to a percentage.

## 2. Three specific things the audit should not miss

1. **The D5 guard has a hole.** `tests/pipeline/test_dag.py::test_pipeline_source_has_no_baseline_routing_branch`
   scans `src/pipeline` only, and is a strict `xfail` because
   `src/pipeline/reconstruction/reconstruct.py:96` has the forbidden branch. But
   `src/runner/stages.py:816` has **the same `is_source_passthrough` shortcut**
   and no test forbids it there. Widen the scan to `src/runner` — it will simply
   stay xfail until `BP39` fixes both, which is the honest state.
2. **D6 is stale as written.** It says two Animate-Anyone tests fail only in the
   full suite, and defers on the grounds that they are pre-rewrite tests against
   a module Phase C deletes. The module did not die — it moved to
   `src/components/generation/animate_anyone_runtime.py` and the tests moved to
   `tests/components/`. "They die with their modules" has stopped being true, so
   either the isolation gets fixed or the deferral gets a new reason. Re-check
   whether the pollution still reproduces before doing either.
3. **The coverage gate and the required-behaviour suite are two different gates
   and CI runs only the first.** `scripts/check_coverage_gate.py` enforces 77%
   in CI, which `PLAN.md` §8 explicitly says a required-behaviour suite
   *replaces*, "because a percentage gate is satisfiable by padding and this one
   is not". Decide, in writing, whether the percentage stays as a second signal
   or goes. If it stays, its docstring should say it is not the gate §8 means.

## 3. Housekeeping in the same pass

- **`python -m src.contracts.layers`** is in `AGENTS.md`'s pre-merge list and is
  not a CI step. Add it. A check every session is told to run by hand is a check
  that gets skipped.
- **Local tooling was broken and is now fixed**: `tomli` is missing from the
  `pointstream` env, which made `pytest` and `mypy` refuse to start at all
  because both read their config out of `pyproject.toml` and Python 3.10 has no
  `tomllib`. It is now in the `dev` extra. Confirm a fresh env install brings it.
- The env also carries two **invalid distributions** — `-` and `-umpy` in
  `site-packages`, the residue of an interrupted `pip install` that was part way
  through numpy. Harmless today, and exactly the kind of thing that produces an
  unreproducible failure later. Worth a clean check.

## Done when

- The eleven-row audit exists, every gap is either closed or recorded with a
  reason, and **`PLAN.md` §8's closing paragraph describes the suite that
  exists.**
- The D5 guard covers `src/runner`.
- `python -m src.contracts.layers` runs in CI.
