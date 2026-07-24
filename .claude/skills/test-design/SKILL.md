---
name: test-design
description: Propose and then write the tests for a component you added or changed in POINTSTREAM — behaviour cases, plausible-misuse cases, and an explicit list of what is deliberately not tested. Use after writing or modifying anything under src/, and as the test step of a refactor PR. Surfaces the proposed list for approval before writing any test code.
---

Read `/home/itec/emanuele/.agent-rules/skills/test-design/SKILL.md` and follow it.

POINTSTREAM specifics:

- Coverage gate: `python scripts/check_coverage_gate.py` (80% CI / 85% local,
  override `POINTSTREAM_COVERAGE_THRESHOLD`). Omit list lives in
  `pyproject.toml`'s `[tool.coverage.report]`.
- Test command: plain `pytest` excludes `integration`/`slow` markers by
  default (`pytest.ini`); ~383 tests, ~2 min.
- Tiers map onto this repo as: unit → `tests/`, no marker; integration →
  `@pytest.mark.integration`/`@pytest.mark.slow`; stage-contract →
  validators on the Pydantic models in `src/shared/schemas.py`; goal-invariant
  → `-m invariants`, verdict written into that run's own `run_summary.json`
  under `invariant_failures` (non-empty means not citable).
- The invariant to hunt silent-wrong-answer bugs against is the **Residual
  Guarantee**: payload accounting that doesn't sum, a null `psnr_mean`
  passing as a result, encoder/decoder disagreeing because
  `SynthesisEngine` behaviour forked between them.
- Living-test rule ties to `RESEARCH_LOG.md`'s dead-end registry
  (`67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`) — write the dead-end entry and
  the regression test in the same session.
