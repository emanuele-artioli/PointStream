# B′16 — Un-red the CI, so the next real regression is visible

**Owns:** `.github/workflows/**`, `pytest.ini`, `config/*.yaml`,
`tests/invariants/test_named_weights_resolve.py`, and `src/contracts/weights.py`
(or wherever `resolve` lives — find it, do not assume).

**Does not own** `src/encoder/**`, `src/decoder/**`, `src/shared/**` or the 69
top-level `tests/test_*.py`. That is `BP15`, and it must not be folded in here —
see *Why this is not BP15* below.

**This is a short brief on purpose.** Budget half a day, not a week.

## The problem

`main`'s CI has failed on **every push for at least 12 consecutive runs**, all
day on 2026-08-23 and before. Every one is the same single test:

```
FAILED tests/invariants/test_named_weights_resolve.py::test_every_shipped_config_weight_resolves
1 failed, 1045 passed, 6 skipped, 47 deselected, 3 xfailed
```

`assets/weights/` is gitignored, so on a GitHub runner the weights do not exist
and **this test can never pass there**. It passes locally, where they do.

**Why this matters more than one red X.** Red is now the normal state of `main`,
so the next genuine regression is invisible: nobody will look twice at a failing
run they have learned to expect. The suite is otherwise green and worth
trusting — 1045 passing tests are being wasted as a signal.

## What the test is right about, and must keep catching

Do **not** delete it and do not weaken it into always-skip. It is currently
reporting two real config faults that a runner-only skip would hide forever:

1. **`config/tier_quality.yaml` names `yolo26x-eg.pt`** for `detector`,
   `segmenter` and `ball-det-model`. That is a typo for `yolo26x-seg.pt`, and it
   has been papered over: `assets/weights/yolo26x-eg.pt` is a **symlink to
   `~/Models/YOLO/yolo26x-seg.pt`**. Fix the config, then remove the symlink;
   the file is real, the name in the config is not.
2. **`controlnet-id: assets/weights/custom-controlnet`** in `default.yaml`,
   `tier_balanced.yaml` and `tier_quality.yaml` resolves to
   `assets/weights/assets/weights/custom-controlnet` — the resolver prepends the
   weights root to a value that already contains it. Decide which side is wrong
   (the config should probably carry a bare name, like the others) and fix that
   side. Whichever you choose, a test must pin it.

Fix both **before** changing how the test runs, so the fix is verified by the
test in its current strict form.

## What to do

1. **Fix the two config faults above.** Run the test locally; it must still pass.
2. **Split the invariant into two tests that both mean something:**
   - *Always runs, everywhere, including CI:* every named weight is
     **well-formed and resolvable in principle** — no doubled prefix, no name
     that no rule could ever produce, every config key known. This catches both
     faults above with no weights on disk.
   - *Runs only where the weights are:* every named weight **exists**. Mark it
     with the existing integration marker so CI deselects it alongside the other
     47, and make it `fail` rather than `skip` when `assets/weights/` is present
     but a file is missing — a developer with the weights must still see it.
   The point is that neither half is a skip-if-inconvenient: one checks a
   property that holds without files, the other checks files where files exist.
3. **Confirm CI goes green** — `gh run watch <id>`, then
   `gh run view <id> --log-failed` if it does not. Do not infer from job names.
4. **Add a guard so this cannot silently recur.** A CI job that has been red for
   twelve pushes is a process failure, not just a test failure. Cheapest useful
   version: a line in `README.md` or the workflow that states CI is expected
   green on `main` and that a red `main` blocks merging. If the repo gains
   branch protection later, that is the real fix.

## Why this is not BP15

`BP15` retires ~15k lines of pre-rewrite code and 433 tests. It is tempting to
bundle: both touch the test suite, and the deletion would make CI faster.

**Do not.** BP15 deletes at a scale where the only safety net is a working
regression signal, and right now there is none. Deleting 433 tests under a red
CI means no baseline to check the deletion against — the exact situation this
brief exists to end. The order is:

1. This brief. CI green on `main`, confirmed with `gh`.
2. Then `BP15`, one commit per subtree, each verified against that green baseline.

BP15 also carries a host rule this brief does not: **read before deleting.**
Those modules are prior art that has already yielded two findings. It deserves
its own session with its own attention.

## Traps

**`ruff`, `mypy --config-file pyproject.toml` and `python -m src.contracts.layers`
must stay clean.** They are clean today; do not trade one signal for another.

**Check what the coverage gate does** before moving tests between tiers. The
workflow step is named *"Run tests with coverage gate"*. If deselecting a test
drops coverage below the gate, **lower the gate to the honest number** and say
so in the commit — never add a test to raise it back
(`AGENTS.md`: a test that exists only to raise a coverage number is a defect).

**The three `xfail`s are deliberate** (`DEFERRED.md` D5, D6). Leave them.

## Done when

- Both config faults are fixed, each pinned by a test.
- The named-weight invariant is split so the structural half runs in CI and the
  file-existence half runs where the files are.
- `gh run list --branch main` shows a **green** run, verified not assumed.
- No new `ruff` / `mypy` / layer-check findings, and the coverage gate is either
  unchanged or lowered with a stated reason.
