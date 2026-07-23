---
name: test-design
description: Propose and then write the tests for a component you added or changed in POINTSTREAM — behaviour cases, plausible-misuse cases, and an explicit list of what is deliberately not tested. Use after writing or modifying anything under src/, and as the test step of a refactor PR. Surfaces the proposed list for approval before writing any test code.
---

# Designing tests for a POINTSTREAM component

The goal is a small number of tests that would actually fail if the code broke.
This repo already has ~400 tests and an 80% gate, and a chunk of that number
came from tests written to move the number — those are a liability, because
they make the gate describe coverage it does not have. Do not add more.

## Workflow

1. **Read the change.** `git diff`, or read the module. Identify what the
   component promises: inputs, outputs, and the invariants a caller relies on.

2. **Draft the list, in three groups.** Most components deserve 3–8 tests.

   - **Behaviour** — the envisioned cases, with expected values you can state
     by hand. A payload encoder gets a known byte accounting; a geometry helper
     gets a known transform.
   - **Plausible misuse** — what a caller in *this* repo could realistically do
     wrong: an empty actor mask, a chunk whose frame count disagrees with its
     metadata, a config naming a backend that does not exist, a residual whose
     dimensions do not match its source. Prefer mistakes that produce a
     *plausible number* over ones that raise — only the first kind survives to
     be cited.
   - **Deliberately not testing** — say what you are leaving out and why:
     unreachable branches, third-party library behaviour, errors a caller
     cannot produce, and anything whose only effect is the coverage number.

3. **Show the list to the user before writing test code.** Number the items so
   they can say "drop 3, add one for X". Do not skip this — the user knows
   failure modes the code does not show.

4. **Write only the approved tests**, then run `pytest` and report the result
   and the coverage delta.

## What makes a test worth writing here

Ask: *what would have to break for this test to fail, and could that plausibly
happen?* If the answer is "nothing realistic", drop it.

The category to hunt for is the silent wrong answer, not the crash. In this
repo that means anything that could let a run look successful while the
Residual Guarantee does not actually hold: payload accounting that does not
sum, a `psnr_mean` of null passing as a result, an encoder and decoder that
could disagree because a code path forked `SynthesisEngine` behaviour between
them.

## Which tier the test belongs in

- **Unit** — pure logic, mocks for anything heavy, CPU only. `tests/`, no
  marker; runs on every push.
- **Integration** — needs real weights, the dataset, or a GPU. Mark
  `@pytest.mark.integration` or `@pytest.mark.slow`; excluded by default.
- **Stage contract** — "this stage's own output is well formed" is not a test,
  it is a validator. Put it on the Pydantic model in `src/shared/schemas.py` so
  it fires during every real run, and unit-test the validator.
- **Goal invariant** — "this run supports the claim the paper makes" belongs in
  the invariants module, so the verdict is recorded in the run's own
  `run_summary.json` under `invariant_failures`. A run with a non-empty verdict
  is not citable.

The last two are the point. A rule that lives only in CLAUDE.md prose cannot
stop a bad run being cited three weeks later; a rule that writes its verdict
into the run's own summary can.

## The living-test rule

Every diagnosed bug and every newly imagined edge case gets a test in the same
session it is diagnosed — the RESEARCH_LOG dead-end entry and the regression
test are written together. Deleting a test requires saying why its failure mode
is now impossible.

## Coverage

The gate exists to stop untested code arriving unnoticed, not as a target.
`pyproject.toml`'s `[tool.coverage.report]` omit list is a debt ledger: when a
split makes part of an omitted module testable, remove its entry in the same
PR. Never add an entry to raise a number, and never write a test whose only
purpose is to raise one — if deleting padding drops the gate, lower the gate to
the honest number and ratchet it back up as real tests land.
