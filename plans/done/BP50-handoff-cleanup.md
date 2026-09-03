# BP50 — clean handoff, 3 September 2026

Completed documentation and repository cleanup after PRs #52, #53 and #54.
The active entry points are root `PLAN.md` and `HANDOFF.md`; the original
numbered plan is preserved as `plans/done/RESEARCH-HISTORY.md`. Completed,
superseded and parked briefs are distinguished in the plan indexes. Removed
the obsolete launch instructions from the two wave-5 reports, preserving their
actual reports. Old instructions remain recoverable from Git history.

The incomplete extraction edit remains in the pushed tag
`archive/bp46-incomplete-extraction-2026-09-03`. It was not merged. No assets,
source footage or experiment outputs were removed. All auxiliary worktrees
were removed before this handoff; only the primary checkout remains.

## Verification

- Python edits only repair documentation references. Comparing parsed syntax
  trees after stripping docstrings confirms no executable-code changes.
- Ruff passes; mypy passes on 333 files; import-direction check passes.
- Existing suite: 1080 passed, 1 skipped, 84 deselected, 3 expected failures.
  No new tests were added. Local coverage wrapper exits 2 because coverage is
  80.43%, below its 81% local threshold; the CI policy is 77%. Rechecking the
  same coverage data at 77% passes. No threshold changed.
- Logs: `/tmp/pointstream-handoff-checks.log` and
  `/tmp/pointstream-handoff-tests.log` (host-local, not permanent evidence).
- Paper builds with `conda run -n tex --no-capture-output bash tools/build_pdf.sh`.
  PDF: nested paper repo `build/main.pdf`; no unresolved citation placeholders
  or duplicate live labels. Build log: `/tmp/pointstream-handoff-paper.log`.

## Paper boundary and structural work remaining

Removed unsupported live-operation, automatic never-worse routing, and source
footage redistribution promises. Corrected raw passthrough versus coded fallback.
Recorded offline shared canvases, fair reference lifetimes, separate background
and reference encoder presets, recovery limits and the outstanding comparison.
Historical negative results and retractions remain; no new quantitative research
claim or codec win was introduced.

Rendered size: 30 pages. Body/references occupy 1–21; Appendix A starts on page
21, so appendices span 21–30. Against the project's 23 + 5 page budget, move
roughly five appendix pages (about half the appendix span) to versioned companion
material. An exact word cut requires choosing the material and rebuilding;
page layout cannot be converted reliably into an exact word target.

Approximate body prose counts (excluding floats/comments): introduction 630,
related work 1499, problem 517, system 3732, evaluation 2141, future work 461.
System design is about 42% of body prose, above the 40% structural warning;
roughly 235 words would need moving out of it at unchanged total body length.
There is no conclusion yet. Body floats: five, only one figure (20%, below the
25% heuristic); long prose-only stretches in related work and evaluation also
need review. These are writing tasks, not a claim that the draft is submission-ready.

## Next dispatch

Cursor: `plans/BP49-native-reference-pilot.md`; Codex reviews the report before
broader BP45 curves. Antigravity: BP46 confirmation-footage preparation or
`plans/PAPER-NEXT.md`. No experiment was launched in this cleanup. Read current
Git/job state before starting. No new host/platform knowledge candidate was needed.

Keep `experiments/` (tracked reproducible experiment programs) separate from
`outputs/` (generated, ignored data under the data root). Merging them would
mix source and results and reintroduce editor/indexing problems. Root `AGENTS.md`
is the Codex instruction entry point; no project-specific `.codex/` settings are
needed for this handoff. `.github/` is CI/review configuration, not agent memory.
