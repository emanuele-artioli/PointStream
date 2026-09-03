# BP51/BP52 review integration

User authorized repairs, commits, pushes and merges after the two reports.

## Repairs

PR #57: explicit prior-use audit fields required; absent/malformed values are
not clean. Confirmation identities are compared against development/diagnostic
and known-used matches, not just against other confirmation filenames. Repeated
pairings in different matches are allowed. Match labels normalize case/whitespace;
unresolved event aliases and compilation contents still need human/source audit.
Current manifest: zero accepted confirmation matches, diagnostic intervals intact.

PR #58: save partial report and stop before another encode after point alarms
or a missing/changed CRF51 regression control. Byte ledger and usable-rate status
are checked. Control size/quality is exact; variable host wall time is not a
quality regression. The timing comparison now reads the top-level time field.

Approved regression tests cover both provenance failures, interrupted expansion,
control mismatch and clean completion. Targeted results: BP51 26 passed/1 skipped;
BP52 43 passed; both targeted lint/type checks passed. Both repair PRs passed CI
and were merged (#57/#58, combined code bcb3a63). Combined local checks also
passed: full ruff, full mypy (335 files), import direction and the three touched
experiment suites (69 passed, 1 skipped). No native experiment was rerun.

## Evidence kept

BP52's output JSONs remain unchanged. All three PointStream ledgers balance;
controls are valid, no recorded alarms, full source identities match both
reference files, and fresh CRF51 size/quality reproduces BP49 exactly. Stronger
background quantization reduces bytes and quality without establishing a win.
n=1 diagnostic pair, not independent confirmation or a speed result.

The original PointStream wrapper log was lost because tee ran before its output
directory existed; durable result JSONs remain. New launch instructions require
creating logs first and preserving both process and logger exit status.

Paper commit 10a4c35 is pushed to its independent main. Evidence notes record
this scope, contamination and the missing separate
encoder/client timing. Build remains 26 pages, body/references 1–21 and appendices
22–26, with no unresolved citations. No headline win or BD-rate was added.

Next: BP53 transport scaling (Cursor); BP54 source shortlist (Antigravity).
Codex retains BP53 geometry/accounting review, next-axis selection and BP55's
timing contract. A source shortlist is not acquisition authorization or accepted
confirmation data. See current PLAN.md rather than archived dispatches.
