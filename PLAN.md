# PointStream Area Index & Active Plan

Current coordination: **14 September 2026**, code baseline `3810a56` (#100–#102).
Submission **30 September**; provisional evidence freeze **20 September**.
The [evaluation campaign](docs/workflow/session/evaluation-campaign/plan.md) is the
authoritative plan incorporating the user's full evaluation design and resource
policy. The September 12 handoff/audit are historical context, not dispatches.

## Areas

| Area | Current state | Next campaign work |
|---|---|---|
| [Evaluation](docs/areas/evaluation.md) | #100/#102 repair accounting and validation paths; saved results still need claim-specific reuse checks | E01 evidence/protocol, E03 anchors, E06 system |
| [Codec](docs/areas/codec.md) | #101 repairs probe transport/precision; known three-mode artifact uses removal ON, not requested OFF | E04 background coverage and paired removal |
| [Generation](docs/areas/generation.md) | #102 boundary repairs merged; sparse diagnostic is not full-trajectory evidence | E02 readiness, E05 staged training; one baseline-clearing model required |
| [Data](docs/areas/data.md) | Preserve exposure history; six fresh matches preferred but lower prospective count permitted | E01 split/protocol, E07 confirmation and second domain after tennis win |
| [Infrastructure](docs/areas/infrastructure.md) | Existing detached monitoring; verify host availability at each launch | Per-host CPU <=90% available; any free GPU; no cleanup helper |
| [Paper](docs/areas/paper.md) | Separate repo; no new competitive evidence certified by repairs | E08 setup/structure early, final claims after accepted results |

## Initial assignments

| Task | Owner | State | Scope |
|---|---|---|---|
| [E01](docs/workflow/session/evaluation-campaign/tasks/01-evidence-protocol.md) | Cursor | Ready to dispatch | Inventory/reuse, split, source-count and shared result/timing contract |
| [E02](docs/workflow/session/evaluation-campaign/tasks/02-generator-readiness.md) | Antigravity | Ready to dispatch | Full trajectories, actual backend identity, training evaluator readiness |

E01 and E02 run in parallel in separate worktrees. Later tasks E03–E08 are
specified in the campaign with dependencies and ownership; they are not already
running. Workers return to the coordinating Codex task after each bounded stage
for review and the next assignment. Only coordinator edits this active index.
No experiment jobs launched by campaign planning.

Gate A remains open and confirmation incomplete. User-authorized changes to
source count, neural-foreground milestone and secondary-domain timing are in
[roadmap](docs/roadmap.md); update manifest/verifier policy before scoring.
