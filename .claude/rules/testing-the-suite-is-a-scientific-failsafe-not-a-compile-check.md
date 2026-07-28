---
paths:
  - "tests/**"
  - "pytest.ini"
  - "scripts/check_coverage_gate.py"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Testing — the suite is a scientific failsafe, not a compile check' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Testing — the suite is a scientific failsafe, not a compile check

- `python scripts/check_coverage_gate.py` is the CI entry point (runs
  `coverage run -m pytest`; threshold 80% in CI, 85% locally, override with
  `POINTSTREAM_COVERAGE_THRESHOLD`).
- Plain `pytest` excludes `integration` and `slow` markers by default
  (`pytest.ini`). ~383 tests, ~2 min.
- Lint/type: `ruff check src tests scripts` and `mypy` (config in
  `pyproject.toml`); pre-commit runs both.
- Tests are necessary but not sufficient: after a pipeline change, verify
  with a real `--input assets/real_tennis.mp4` run and show the command +
  the `run_summary.json` numbers, not just "tests pass". Run `/code-review`
  after non-trivial `src/` changes.

Three tiers, each catching a different kind of wrong:

1. **Unit tier** — behavior and misuse of pure logic, mocks for anything
   heavy, CPU-only, runs on every push.
2. **Stage-contract tier** — every stage validates its own output as it
   produces it, via validators on the `src/shared/schemas.py` models
   (masks non-empty, panorama and residual dimensions consistent,
   `sizes_bytes` actually summing, a null `psnr_mean` failing the run).
   A broken stage fails *there*, not three stages downstream.
3. **Goal-invariant tier** (`-m invariants`) — checks the *paper's* claim on
   a real run: the Residual Guarantee itself, payload accounting, quality
   floors. Violations are written into that run's own `run_summary.json`
   under `invariant_failures`, and **a run with a non-empty
   `invariant_failures` is never citable** — re-check it before it reaches a
   report or the paper.

**Every diagnosed bug or newly imagined edge case gets a test in the same
session it is diagnosed** — the RESEARCH_LOG dead-end entry and the
regression test are written together. Deleting a test requires saying why
its failure mode is now impossible.

Research code, so keep tests honest and thin: cover envisioned behavior and
plausible misuse of code we own. No tests for unreachable branches, for
third-party library behavior, or for errors a caller cannot produce. **A test
that exists only to raise the coverage number is a defect** — if the gate
fails after deleting padding, lower the gate to the honest number and ratchet
it back up as real tests land.
