# Continuation — November 12, restored fleet access

## Current state — Ada cross-class failure retained; fresh protocol in preparation

Root closed **`20261112T202503Z-37b3b79b`** FAILED and acknowledged both events.
It actually ran on exclusive gpu6 Ada, source21482/archive5fd8, and prepared all
six points. Saved source YUV and source scores match the A6000 reference, but
15/16 WASB row coordinates differ slightly. All three SVT points match native/
wire/decoded/rendered hashes; all three DCVC points differ. Final scoring never
ran. This fails exact cross-class reproduction; it is not a quality result.
[Root closeout](experiments/k5d-inputs/ada-37b3-root-closeout-20261112.json).
Observed worker/init/supervisor PIDs, group and claims absent. Charge143GPU/
493CPU conservatively; prior monitoring200CPU separately charged. Earlier461e
expired unassigned and is also closed/acknowledged. No GPU job or fleet wait is
active, and the one-submit controller has ended. Never restart/replay it.

Root registered a **separate** [fresh Ada protocol](cards/k5d-ada-fresh-reference-2026-11-12.md)
before further inference: complete six-point16 A with a conditional independently
initialized B on the same Ada/fleet job, exact same-class repeatability, unchanged
source pixels/labels, original thresholds, actual bytes, source-hidden receivers
and real metrics/judges. Existing accepted16/failed strict gate remain unchanged.
A complete-cost gate must fit B with1.5margin, setup/validator/reserves. No scale
release follows. Builder `ada_fresh_reference_builder` is implementing on isolated
branch `codex/ball-ada-fresh-reference-20261112` in
`/private/tmp/pointstream-ball-ada-fresh-reference-20261112`, based21482. Frozen scientific source is now bbe94eddde3e8596b98d93667b3ffbdeaed184a6:
sourceed89052108dbd4a8242d7fd7ae35ac37aa1f41332d204a4d11e9d7b799bf715d,
archive4f7f6546b7eb4552ea77b8d1607a0ccc68c7ae42e73965a70a2902f40f4e7402,
MAIN32e. Root reviewed all five changed source/test files; actual archive bridge
has1,340unchanged files/52changed paths/8pinned patches and is approved SHA
0c32187b4a16d4b047c95c16c6145e973abcbdc9e4d9a165edbb6efcbb59ab0f.
Root integrated code as3b0b8caf/808a3481; full local checks2,134passed/40skipped,
Ruff and mypy581 pass. New native CPU proof was released at250CPU/0GPU;
runnerpreflight passed gpu5/gpu2 doctor, CPU0claim empty, separateSSD1.90TB,
actualnative/tool/input/cache hashes. Actual CPU proof **`20261112T204701Z-6c0c0a86`** is pending/running;
canonical specb8868b5c, source_archive:true, deadline21:26:06.069711UTC,
max250CPU/0GPU. One canonicalwait2500 backend25026, clientPID/outputs
`/private/tmp/pointstream-ada-fresh-cpu-run-20261112/`. ActualID/source/spec are
in Balljournal and ledger before waiting. Do not submit or wait again.
Source proof/root GPU release remain pending. Raw preparation `/private/tmp/pointstream-ada-fresh-bbe94edd-20261112/`.
No new GPU run is currently released.

Passive resource checks remain active, every5minutes for<=45wallminutes:
PID22535, start20:36:36UTC, maximum21:21:36UTC; no autosubmit or model turns.
State/logs `/Users/manu/.cache/pointstream-resource-watch-20261112-passive/`.
Inner80countedCPU plus20outer allowance is conservatively charged100CPU;
unmetered SSHserver/login substrate means no whole-host CPU bound is claimed.
Source preparation50CPU also precharged. Programme now **19,505GPU/50,324CPU**
of86,400/100,800. Remaining oldBall7,920GPU/800CPU; new freshAda1,400GPU/1,950CPU
includes250CPUproof+1,600futureGPUstageCPU+100unusedmonitorreserve. W8 remains
4,550GPU/929CPU and held for its complete-cost gate; shared1CPU uncharged.
No paid services, training, manuscript/PRESLEY change or submission.

Durable closed37b3 evidence: `/Users/manu/.cache/pointstream-evidence/ada-crossclass-20261112-a5/`,41read-onlyfiles+SHAmanifest.
Current raw37b3 audit `/private/tmp/pointstream-ada-37b3-20261112/`, including
paired prepared/reference JSON and cleanup proofs; root recomputed comparisons.
Durable source worktrees for0bb/21482/f7 live under
`/Users/manu/.cache/pointstream-source-worktrees/`. Existing source/model/data
archives and all earlier evidence remain preserved. Before any fleet action,
copy the actual ID from Ball experiment journal and run canonical status.
Older running/pending instructions below are historical, not current commands.

## Historical alternate-GPU qualification — closed expired

The user requested regular resource checks and an alternative to the A6000.
A machine-only five-minute watcher found both gpu5 Ada devices idle on its first
18:43:50UTC check and stopped; no recurring AI automation was created. Later
readiness checks found the Ada devices occupied again. The native fleet now
checks once a minute for the approved queued request below, with no AI polling.

**`20261112T192044Z-461e16e8`** is the actual new Ada16 qualification request,
pending at publication. Frozen source21482f35/tree9e2376/archive5fd841,
source_archive:true, specb4c7259f/raw917299cd, hosts gpu5/gpu6, hard deadline
**2026-11-12T20:00:05.184948UTC**. Maximum1,000GPU/1,150CPU including350external,
800campaign/580namespace/600smoke/30validator/5no-op,32GiB local, packedMAIN32e.
This entire maximum is nested in the remaining parent, not additional budget.
The single canonicalwait2500 has localPID17045/backend session19953; the runner
is attending it. Handle/outputs:
`/private/tmp/pointstream-ada-461e-wait-20261112/`. No second wait or replay.
Root's single expected-finish check is20:02UTC if no completion wake arrives.
Copy the complete actual ID from the Ball journal before any fleet action.

Root accepted CPU proof **`20261112T190032Z-d24d7c7c`**, canonicalCOMPLETE:
26 publication member hashes, four real native modes,19 observed births and
campaign/supervisor/claim absence. Normal2.89028inner/3.036772outerCPU; TERM
required KILL escalation. Gate3da5120c is approved for exact21482; canonical
eventc0cefd09 acknowledged. This is CPU containment evidence, not CUDA parity.
Actual source bridge c87f91d3 compares the real archives:1,340unchanged files,
48changed paths,8pinned patches. Default A6000 behavior and every accepted16
native/wire/pixel/metric/judge equality are unchanged. Downstream Ada admission
remains held even if the initial16 passes. Full integrated checks passed:
2,129tests/40skips, Ruff, mypy577files. [Qualification card](cards/k5d-ada-qualification-2026-11-12.md).

Current charges **19,362GPU/49,131CPU** of86,400/100,800. Remaining parent Ball
**7,920GPU/1,150CPU**, W8 **4,550GPU/929CPU**; shared1CPU uncharged. The native
watch was charged40CPU conservatively (not a measured whole-SSH-system bound).
CPU proof was300inclusiveCPU (50prep +250closure),0GPU. No GPU outcome or charge
is established yet; preserve the full queued maximum if communication fails.

Durable pre-outcome snapshot: `/Users/manu/.cache/pointstream-evidence/ada-qualification-20261112-a4/`; the source is also cached by archive SHA. Live wait snapshots do not establish completion.

Frozen source checkout `/private/tmp/pointstream-ball-ada-source-20261112`;
source/spec preparation `/private/tmp/pointstream-ada-preparation-21482f35-20261112/`.
GPU spec/submit receipts `/private/tmp/pointstream-ada-gpu-run-20261112/`;
CPU audit `/private/tmp/pointstream-ada-d24d-audit-20261112/`. Selected job records
were copied to the default tracker; isolated client pools and global workers
remain distinct, and the global configuration was not changed. Approved gate
and bridge are immutable new files under fleet/checks/k5d-ada-2148-d24d-root-20261112.
Source archive reuses the completed CPU job; it was not uploaded a second time.
A large read-only prerequisite fetch timed out after6MB: PID2266035/birth718893878
remains unproven; partialtar is unused and no remote signal/replay was sent.
The verified local calibration cache and tiny canonical d201 export were used.

No further job, automatic retry, source/numeric relaxation,48/292/586/full30,
training, paid server/API, manuscript/PRESLEY edit or submission is authorized by
this qualification. Failed exact parity narrows only cross-class reproduction;
it cannot prove a separately registered fresh Ada study impossible. Preserve
that distinction and the negative before any new scope decision. Older
recovery outcomes below are history, not current monitoring instructions.

## Final authoritative outcome — after the single18:10UTC check

**No active GPU request, wait or timer remains.** The one resumed request
`20261112T172806Z-8709406d` expired unassigned. Root canonical status, raw
source/spec/archive, absent execution/claims and both eligible workers' admission
reasons agree. All A6000 occupants in a fresh ownership probe were other users;
no own stale GPU process was identified among those occupants or signaled.
Canonical terminalevent07724b3fb54d5e15b3d089e42eaceb7d is acknowledged.
Physicalevent195375fc is the same event with a different documented UUID recipe;
the wait's earlier pending status is a status/event read race, not execution.
The sole wait ended after1301seconds, exit10, and its agent collected session46210
once. Root's one expected-finish check is complete; no second wait or wakeup.

Final immutable raw snapshot: `/Users/manu/.cache/pointstream-evidence/fleet-8709-closeout-20261112-a3/` (21files25,432bytes plusmanifest). Earlier resumption snapshota2 retains24files69,756bytes; do not overwrite either.

Current programme charges **19,362GPU/48,791CPU** of86,400/100,800. Remaining
nested Ball **7,920GPU/1,450CPU**, W8 **4,550GPU/929CPU**; no active child maximum.
The resumed queue is charged400externalCPU/0GPU conservatively. No queue wall
is counted as CPU, and no scientific result is inferred. The
[root closeout](experiments/k5d-inputs/8709-recovery-root-closeout-20261112.json)
preserves exact identities, mapping, raw joins and limitations.

**Decision:** keep NFS as the archive and use existing local staging; no broad
cleanup/migration. Actual readiness proved separate local storage on all six
hosts. RequiredA6000capacity is the present blocker, not a demonstrated lack of
disk space. Retain the isolated participating-pool config as a reviewed option,
without changing global config/workers. After an external capacity change,
inspect the actual prior ID and fresh eligible hosts before considering a new
bounded request. No further queue or hardware broadening is released now.

W8 remains resource-held; no complete forecast fits its remaining CPU allowance
under the currently qualified evidence. No Ball-to-W8 transfer, training, paid
hosting/API, legacy study, manuscript/PRESLEY edit or submission is authorized.
The user requested research-first; that remains active. Coauthors/rights remain
open. Prior notes below are historical; latest ledger and final outcome govern.

Start here and read the [storage/recovery decision](fleet-recovery-2026-11-12.md),
[approved research card](cards/submission-research-2026-11-04.md),
[live ledger](experiments/submission-research-budget-20261104.json) and
[Ball](experiments/k5d-ball-benchmark.md) / [W8](experiments/w8-wire-audit.md) journals.
Earlier [November10](continuation-2026-11-10.md) and
[November11](continuation-2026-11-11.md) records preserve history, not current job states.

The user restored SSH and asked whether to resume or clean/move work for NFS
slowness. Root decides **retain NFS archives and resume existing local staging**,
without broad cleanup or relocation. Five measured local filesystems have
1,353–6,732 GiB free; NFS approximately20,164GiB. Root disks remain forbidden
for large temporaries. gpu3 metadata access is intermittent; no storage or
authentication root cause is proved. All six canonical fleet probes and worker
heartbeats were available; those observations are not reservations.

Both previously unknown GPU requests are now root-accepted **expired unassigned**:

- Ball `20261110T190907Z-4d0dd928`: exact0bb source, archive450d, canonicalspecf995;
  no execution tree/claims, ownergpu4 expiry worker. Full400externalCPU charged,
  0GPU, terminalevent589b684cf7ee5e51a1af7117b5bbedf7 acknowledged.
- W8 `20261110T183845Z-4c240783`: exactaecf source, archivef784, canonicalspec67e4;
  no execution tree/claims, ownergpu5 expiry worker. Full150externalCPU charged
  as true preparation,0GPU; cumulativequalificationpreparation1450CPU with the
  same22578404nonce. Event6dcc8c6f752f517cb7d2e651eabe8867 acknowledged.

The [root closeout](experiments/fleet-recovery-closeout-20261112.json) records
independent joins and limits. Historical read-only probes still lack reliable
teardown evidence and were not signaled or replayed. No folder was deleted.

Initial recovery charges, before the prepublication stop above: **19,362GPU /47,991CPU seconds** of **86,400 /100,800**.
Remaining nested parent balances: Ball **7,920GPU /2,250CPU**, W8 **4,550 /929**.
These include no new child expenditure. Shared has41CPU uncharged. Administrative
CPU is conservatively charged, not fully measured; no queue wall is billed as CPU.

Ball: root conditionally permits preparation of **one** new deadline-only queue
for the same original corrected16 retained-receiver parity/lifecycle cost test,
at1,000GPU/1,200futureCPU including400externalCPU, nested in its parent. The
[prospective release](experiments/k5d-inputs/post-outage-corrected16-root-release-20261112.json)
requires fresh passing doctor, source/archive, input/tool and local-storage gates
before submission. Scientificsource0bb and all prior gates are unchanged.
Restoredcleanworktree `/private/tmp/pointstream-ball-0bb5121-recovery-20261112`;
rebuiltarchive matches450d. Rootcurrentclient contains the operational fixes.
Specgenerator/baseline `/private/tmp/pointstream-ball-queue-proposal-20261112/`.
Do not submit the test-only2100 spec. Both A6000 hosts were busy at inspection;
the bounded queue may expire. The actual new ID and active wait authority are recorded in the latest update above.
No48/292/586/full30 or further queue follows automatically.

W8: **hold further full-witness compute**. Its complete forecast is not shown to
fit929CPU; another550CPU pilot would leave379. The historical sampling CPU slope
suggests~1,150CPU for two full sampling phases (~1,724 with1.5margin), before
other required work; this is not a measured lower bound. Preserve the scoped
mechanism/cost evidence and fresh-witness limitation. No reallocation or narrowed
new endpoint is approved. Protect the Ball forecast before considering a concrete
resource decision. This is a resource stop, not a scientific-quality failure.

Durable recovery snapshot:
`/Users/manu/.cache/pointstream-evidence/fleet-recovery-20261112-a1/`
(65files110,170bytes plusmanifest); new records must not overwrite it.
No training, paid server/API, legacy200GPUh study, manuscript/PRESLEY edits or
submission. Footage rights and coauthor approval remain open. The narrowed-paper
fallback is an option; the user has not replaced the research-first instruction.
