# Overnight continuation: 26–27 October 2026

## User authority and deadline

The user authorized progress without further input until **27 October 2026,
07:00 Europe/Vienna (06:00 UTC)**. Parallel routine workers may use
`gpt-6.1-terra` at high reasoning or `gpt-6-luna` at xhigh. This supersedes
the earlier preference for sequential model work for this overnight window;
it does not waive isolation, compute budgets, evidence requirements or data
protection. Do not silently substitute providers or paid API access.

Record scientific decisions/results in the item's experiment entry and
implementation/verification details in its PR. Preserve source data, failed
attempts, streams, raw decoded pixels and provenance. No paper submission,
data deletion or shared-history rewriting is authorized.

## Execution gate: unattended isolation not verified

The project policy says:

> Unattended execution requires verified isolation, including enabled file
> and MCP tools. If that protection is unavailable, use attended execution.

At 22:38 UTC on 26 October the coordinator checked the local runtime:

- `agentctl doctor --host local --probe` reports that `local` is not enrolled.
- `agentctl doctor --catalog` identifies this machine as
  `Emanueles-MacBook-Pro.local`, permits unattended Codex only conditionally
  on a scoped sandbox and isolated subscription profile, reports
  `unattended_gpu: false`, and prohibits remote PointStream agent sessions.
- The installed `agentctl.sandbox.probe(scoped=True)` run outside the nested
  tool sandbox reports `filesystem_verified: true`,
  `process_lifetime_verified: false`, `verified: false`. Its detail is
  `daemonized-process lifetime isolation is unverified; use attended execution`.
- Isolation of this desktop chat's enabled MCP tools has not been verified.
  A successful shell filesystem probe does not establish that MCP isolation.

Therefore no new unattended model workers, overnight fleet computation,
recurring heartbeat or cache keepalive was launched. No cache TTL is assumed.
Do not reinterpret a process handle or desktop notification as a verified
completion-triggered model wakeup. No one-shot wakeup has been scheduled.

The existing study chat was instructed to preserve its continuation and stop
before new submissions; inspect its final report before assuming that no job
was submitted while that instruction was in transit. Never replay an uncertain
submission. Existing bounded work, if any, must be inspected by recorded ID;
only task-owned processes may be cancelled.

## Scientific position and active item

The intended paper remains a selective, fully accounted tennis codec including
players and ball. A ball-only contribution is a starting component, not the
paper endpoint. Current two-window results are development evidence, not
independent confirmation.

- [PR #247](https://github.com/emanuele-artioli/PointStream/pull/247):
  independent pose-prompted SAM at every saved raw generated tick passes
  4/4 player LPIPS/DISTS cells; the frozen +0.001 sensitivity retains 3/4.
  There is no transmitted dynamic matte. The target-derived Telea background
  remains granted. Only 80 of 240 generated RGB frames per player were saved.
  Full-frame wire accounting and independent source/replay provenance remain
  unresolved. Its GPU ledger is closed; unused budget is not transferable.
- [PR #248](https://github.com/emanuele-artioli/PointStream/pull/248):
  CPU inventory examined the actual old anchor publication. Job
  `20261024T201059Z-29586984` is complete; its TAR is 5,601,280 bytes,
  SHA256 `67444a2a8f56c8700fb073a1184ce207ec3ad38cc003dbafa6710bb33836dff9`.
  It contains same-window metadata/poses/masks/scores/review images, but no
  native stream, decoded video or pixel manifest. The checked output,
  checkpoint and original scratch paths supply no missing stream; absence
  of other copies elsewhere is not proven. No qualified ball packet for
  Al031@129 or Med014@194 was established. Rate floors are not full-frame
  quality results. Initial whole-step CPU usage was not instrumented; do
  not invent an aggregate charge.
- The prepared recovery card is in
  [PR #249](https://github.com/emanuele-artioli/PointStream/pull/249): exact
  source hashes/240-frame intervals, 25/50 fps, SVT-AV1 CRF63 preset4 at
  scales8/4/2, native tool identity, preserved IVF streams and decoded-pixel
  manifests. A 24-frame smoke precedes a bounded CPU encode/decode.

The study's final handover (`9223880c`, PR #249) confirms **no recovery job
or smoke submitted**, no runnable fleet spec or encode worker implemented,
and no new decoded pixels or scientific result. The blocker is that fleet `budget_seconds` enforces wall time,
while the recovery card also requested 1,800 aggregate CPU-seconds. Workload
child CPU snapshots do not by themselves account for staging, publishing,
supervision and the subsequent validator. A generic ffmpeg RGB resize also
does not reproduce the old path: bilinear chroma conversion precedes bicubic
RGB reconstruction. Preserve these distinctions in the implementation.

Study chat: `01a16e7a-e752-78f3-b372-c338d8cdbf46`. Read its report directly;
the user authorized direct coordination. Do not launch duplicate work.

## Ready queue after execution protection is established

1. **Recovery input and accounting contract.** Finish the existing CPU item.
   Bound and measure the workload's true process tree; report other overhead
   separately unless it is actually measured/enforced. If the original strict
   aggregate cap cannot be honored, revise the card explicitly before running
   rather than claiming enforcement from a wall-time limit. Retain actual
   bitstreams and independently reproducible decoded pixels. Do not spend time
   redesigning fleet infrastructure merely to obtain a small input encode.
2. **Routine evidence/documentation worker.** A Luna xhigh or Terra high
   worker can reconcile merged reports with their own permitted PLAN sections
   and create a concise claim/evidence/assumption table. Read-only results;
   no new scientific runs, paper claims or broad source-data traversal.
3. **Routine accounting/validator worker.** In an isolated checkout, a Terra
   high worker can review native base-stream framing, metadata and corruption
   checks against the frozen card. It must not concurrently edit the active
   study's branch or launch compute. Integrate only selected reviewed changes.
4. **Real-base feasibility gate.** Once recovered inputs pass validation,
   quote the smallest actual decoded-base + saved-player/SAM comparison.
   Account for every stream and common-model assumption, compare whole-frame
   and foreground quality against achievable SVT/DCVC envelopes, and retain
   fair adaptation/ROI or simple enhancement alternatives. The former granted
   background changes, so component headroom is not an end-to-end bound.
   No obsolete tentative GPU quote is automatically approved.
5. **Ball and fresh evaluation.** Establish qualified same-window ball inputs
   before claiming players+ball integration. Freeze the method before new
   evaluation; current two windows and old twelve have development exposure.
   Source PTS/replays, training/reference/judge overlap and familiar-match
   dependence must remain explicit. No full-240 generation, new footage,
   large confirmation run or submission is automatically authorized.

Parallelize only independent routine tasks, with actual file/MCP isolation
verified for each harness/profile. Use independent checkouts, bounded work,
shared-resource inspection and no duplicate dispatch. Stop overnight work at
the user deadline, preserve the exact continuation, and report decisions,
budget use, jobs and unresolved blockers without fabricating progress.
