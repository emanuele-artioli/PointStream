# BP39 — The all-off corner must degrade by architecture, not by a branch

**Preferred name:** *conventional fallback control*. “All-off corner” remains
only where it names an existing configuration or quotation; see
`plans/TERMINOLOGY.md`.

**`plans/DEFERRED.md` D5, promoted.** It was deferred in August on the grounds
that `BP10` would decide whether there is a paper at all and this only decides
how clean it is. `BP10` is long settled and this is now load-bearing for the
abstract.

**Owns:** `src/pipeline/reconstruction/**`, `src/runner/stages.py` (the
passthrough shortcut only), `tests/pipeline/test_dag.py`,
`src/contracts/lattice.py` (`is_source_passthrough`).

**Scheduling:** PR #45 is merged. Dispatch only when `plans/ROADMAP.md` assigns
the component-ablation workstream and file ownership is clear.

**Read first:** `AGENTS.md` · `plans/DEFERRED.md` D5 in full · `PLAN.md` §3 ·
`sections/system_design.tex` `CLAIM(subsec:lattice)` · the abstract.

---

## 0. Why this stopped being cosmetic

The abstract says, in the sentence that carries the paper's central contribution:

> *"Every component of this pipeline, including the residual itself, can be
> disabled, so the system is specified as a lattice of ablations whose all-off
> corner is simply the source video encoded conventionally."*

`PLAN.md` §3 states the same thing as a design rule: *"graceful degradation to
the baseline codec is a property of the architecture, not a routing special
case."* `tests/pipeline/test_dag.py::test_pipeline_source_has_no_baseline_routing_branch`
encodes that rule and is a **strict xfail**, because two files contain exactly
the branch it forbids:

- `src/pipeline/reconstruction/reconstruct.py:96` — `if request.lattice.is_source_passthrough:`
- `src/runner/stages.py:816` — the same shortcut, and **no test forbids it there
  at all** (the guard scans `src/pipeline` only).

So the property the abstract advertises is delivered by a hardcoded early return
in two places. It is true that the all-off corner returns the source; it is not
true that it does so because the architecture degrades. A referee who reads the
lattice claim and then the code finds a special case.

This is the shape of gate `AGENTS.md` warns about: the assertion passes while the
property it stands for goes untested.

## 1. The fix

Make the generic path degrade correctly, so no branch is needed:

- **background off** yields the source background,
- **objects off** composites nothing,
- **residual off** corrects nothing,

and the composition of those three is the source, arrived at by the same code
every other corner uses. Then delete both shortcuts and watch the reconstruction
test that motivated them still pass.

**Do not delete the guard test to make the suite green** — it is the only record
that this was broken. It is `strict=True`, so when the generic path works it
XPASSes and fails, which is the signal to remove the marker. Widen its scan to
`src/runner` first (`BP37` may already have done this).

## 2. Bounds

- **The all-off corner is bit-identical to the source** after the fix, on at
  least one real clip at 4K, not a synthetic fixture. Anything short of
  bit-identical means a stage is not fully neutral and the lattice claim needs
  weakening in the paper rather than the code being called done.
- **No other corner changes.** Re-run `outputs/bp24-ladder/av1-payload-lowmotion.json`'s
  five rungs and reproduce the bytes and PSNRs. A refactor of the reconstruction
  path that moves a published number is a regression, not a cleanup.
- Cost, from D5: **half a day.** If it runs past two days, the generic path has a
  real structural problem and that is itself the finding — surface it rather than
  pushing through.

## Done when

- Neither `src/pipeline` nor `src/runner` contains a passthrough branch.
- The guard test XPASSes, its marker is deleted, and its scan covers both trees.
- The all-off corner reproduces the source bit-identically on a real 4K clip, and
  `sections/system_design.tex`'s lattice claim can cite it.
