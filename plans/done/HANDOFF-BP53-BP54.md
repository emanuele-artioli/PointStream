# Next bounded sessions: BP53 and BP54

Archived 4 September 2026. Both tasks are merged; use root HANDOFF.md instead.

Trigger: the user requested executable Cursor/Antigravity dispatches after
reviewing and merging BP51/BP52. This is a planned task boundary, not an outage.

PointStream seeks a defensible size–quality win over AV1 and VVC in a named
tennis regime before the hard 30 September TOMM submission deadline. Time must
be reported, but speed is not the current search gate. Read PLAN.md and one brief.

## Verified state

- Code PRs #57/#58 are merged into origin/main (combined code bcb3a63).
  They reject unaudited/reused confirmation matches and stop CRF batches after
  alarms or control mismatches. Both repair branches passed CI.
- Paper is a separate repo at 67a9ea6275d3d9785ce57026, main/origin main
  10a4c35. Evidence notes are pushed; PDF is 21 body/reference + 5 appendix pages.
- BP52 remains one diagnostic pair, no established win. Saved outputs are
  immutable. The committed manifest accepts zero independent confirmation matches.
- Old BP51/BP52 worktrees remain; do not resume them. Create new worktrees from
  updated origin/main. Their removal requires checking for paused user sessions.
- The primary checkout had a pre-existing deletion of HANDOFF.md, left untouched.
  This area-specific handoff avoids overwriting that user change.

## Running work

No experiment was launched by this review. Integration lint/type/tests and CI
are checked before dispatch. Before starting, inspect live processes with
`ps -u "$USER" -o pid,etime,args` and GPU allocation with `nvidia-smi`.
Do not kill unknown jobs; other users and sessions share the host.

## Next steps

1. Cursor: execute plans/BP53-background-transport-scale.md. Verify geometry,
   wire-byte accounting and recovery before its bounded three-point diagnostic.
   Return one PR and plans/BP53-background-scale-report.md; stop before expansion.
2. Antigravity: execute plans/BP54-fresh-confirmation-sources.md independently.
   Return a source shortlist/manifest and proposed acquisition batch, not downloads.
3. Codex reviews the reports, chooses the next search axis and resolves protocol
   conflicts. Timing boundaries are recorded in plans/BP55-timing-boundaries.md;
   do not implement that overlapping change during BP53.

## Open decisions and landmarks

Does transport scaling preserve enough quality to justify curves, or should the
next bounded test target longer contexts/background encoder effort? BP53 decides.
Which fresh sources should be acquired? BP54 supplies evidence for user approval.
No speed claim is licensed while separate semantic encoder/client clocks are absent.

Use plans/SESSION-REPORT.md for reports and plans/done/BP51-BP52-integration.md
for the repair record. experiments/ contains code; outputs/ points to generated
data. Set PYTHONPATH to the actual new worktree. Preserve BP49/BP52 output files.
