# Repository maintenance decisions — 4 October 2026

The branch/worktree audit is accepted with preservation safeguards. Keep the
current Cursor checkout and its unfinished files untouched; use scoped
checkouts for maintenance. Keep local-only branches that already preserve merged
or superseded work local. Push new coherent maintenance work and repair existing
PR heads rather than opening duplicate PRs. Never delete branches as part of
this campaign.

PR #150 contains the fleet dispatcher; its checks are green, but fleet deployment
still requires six-host shared-inbox verification. PR #151 contains clip sampling;
its baseline lint failure was repaired in commit `155e345`, and all checks pass.
PR #152 contains experiment evidence; commit `948cc06` repairs formatting and
CPU replay typing without rewriting recorded evidence, and all checks pass.
Review evidence claims separately from software test success. The suggested
review/merge sequence remains #150, #151, then #152; no merges are authorized
in this campaign. The storage compatibility PR is independent and must pass
its own review and checks before any physical cutover.

PR #153 was created during this campaign by the background-smoke task. Its
current checks fail; leave its active task to finish its lint, typing, and test
repairs before reviewing it for merge. Avoid copying that task's uncommitted
changes into maintenance work.

Active, dirty, paused, and unmerged worktrees remain useful recovery state.
Merged clean worktrees are archive candidates only after confirming ownership,
no active editor/process, and a recoverable snapshot including needed ignored
files. Do not use the cleanup script's recursive deletion fallback. Do not
remove the manuscript checkout while an editor or manuscript task owns it.

The storage mapping, compatibility contract, attended migration commands, and
remaining deployment blockers are in [storage-layout.md](storage-layout.md).
The plan does not flatten named corpora, rewrite historical provenance, or
move live datasets underneath fleet workers. No branch deletion, PR merge,
worker cancellation, or live storage relocation occurred in this campaign.
