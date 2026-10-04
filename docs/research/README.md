# Research evidence and preparation

Current reconciliation: 5 October 2026 (Vienna; source audit collected4 October UTC). Start with the
[scientific handoff](../workflow/pointstream-handoff.md) and
[repository maintenance record](../workflow/repository-maintenance.md).
Software integration and passing tests do not create new experimental evidence.

## What is currently qualified

The [Gate A receipt](gate-a-local-confirmation.json) preserves the completed
96-frame source000 receiver/control records and the user-requested pause.
Alpha costs58,139 complete bytes at VMAF62.695164; null costs52,831 at62.707728.
Static foreground adds bytes without improving this metric on that window.
The recorded native96 QP50/QPA0 point is66,955 bytes at68.285094; neither
point dominates. The48-frame QP52 pilot cannot substitute for a96-frame
comparison. Repeated initial poses, clipping, task truth, unseen scenes and
native overlap remain limitations. Keep the pinned execution and evidence
revisions distinct; this integration does not resume paused jobs.

[Packet packing](packet-rate-quality.md) supports lossless transport savings
relative to each retained parent and source-matched diagnostics, not a general
codec win. The reviewed optional adapter is now installed at
`src.runner.packet_packing`; scale1 preserves mask pixels and decoded output.
Scales2/4/8 are explicit interventions and reject residual-bearing inputs.
Native arrays and geometry remain intact; charge the complete packed archive,
including headers. Default client/workload behavior is unchanged. The adapter
rejects envelopes/arrays above1 GiB and masks above256 Mi pixels before their
corresponding allocations. Existing larger workloads need a reviewed design,
not a silently increased limit.

The installed landmark scorer measures already-associated display-pixel tracks
using actual timestamps. Missed detections remain in recall/hit-rate denominators;
velocity never bridges missing frames. Finite inputs producing nonfinite scores
are rejected. This is tested metric machinery, not annotated tennis task truth.

The [demo source audit](demo-source-identity-audit.md) confirms conflicting clip
names and absent source-to-stream hashes. Historical demo scores remain
quarantined. Future exports use explicit hashed clip IDs and new external output
roots; they do not repair old measurements retroactively.

## Read historical records in context

- [Research recovery dossier](../research-recovery/README.md): the full source-backed
  recovery and published working-paper history. Its dated records are preserved;
  [current status](../research-recovery/CURRENT_STATUS.md) supersedes old branch,
  checkout, manuscript and execution guidance.
- [Preparation chapters](01-introduction.md), [evaluation protocol](06-evaluation.md),
  [experiment plan](07-experiment-plan.md), [evidence ledger](08-evidence-ledger.md),
  [baseline intake](09-baseline-intake.md), and [preparation audit](10-preparation-audit.md):
  September28 research framing and prerequisites. Current dispatch follows the
  [gated fleet](../workflow/long-jobs.md). Old exposure and readiness statements
  must be rechecked for the specific selected experiment.
- [Background study](background-study-results.md),
  [mask diagnostics](mask-diagnostics-results.md), [pose packing](pose-packing.md),
  and [GVC-RT intake](gvcrt-intake.md): named retained campaigns and their explicit
  limits. Worker paths belong to the pinned source revisions recorded below;
  experimental runners have not all been integrated into maintained software.

[Source map](../workflow/reconciliation-source-map.json) records original branch
heads and SHA-256 values for every selectively imported file, plus any review
changes. Historical raw reports, sources, weights and datasets remain external.
No new GPU/CPU campaign, inference, training, native comparator or manuscript
publication was performed during this reconciliation.
