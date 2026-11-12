# Continuation — November 12, restored fleet access

Latest operational update: **new request `20261112T172806Z-8709406d` is
pending**, source0bb/archive450d/source21cd, source_archive:true, spec6a3c3570 /
raw4aec7ec9, deadline **2026-11-12T17:52:06.008295UTC**. This is the original
corrected16 parity/lifecycle-cost gate only,1000GPU/1200futureCPU including400
external, nested in Ball7,920GPU/1,850CPU. A runner owns **one canonical wait2500**;
its recorded localPID is7932, execsession46210, handle
`/private/tmp/pointstream-ball-8709-wait-20261112/handle.json`. No second wait.
Background-completion wakeup is unverified. Root will make one attended
expected-finish check at18:10UTC, within45minutes of publication/release. No further
queue/scale/replay. Copy actual ID from that journal before fleet actions.

The first submit stopped in repeated all-host doctor on gpu1 TCP timeout
**before ID allocation** and was conservatively charged400externalCPU/0GPU.
A new isolated local client pool `[gpu5,gpu2,gpu3]` passed the same participating-
host visibility/atomic gates and successfully published the request. Required
scientific hosts remain gpu2/3; globalconfig/workers are unchanged. The selected
new job metadata was copied into the normal client tracker. Management access
uses `PS_FLEET_MANAGEMENT_HOST=gpu5 PS_FLEET_MANAGEMENT_PYTHON=/usr/bin/python3`.

Current programme charges **19,362GPU/48,391CPU**; remaining nested Ball
**7,920GPU/1,850CPU**, W8 **4,550GPU/929CPU**. The new1200CPU/1000GPU maximum is
already inside those balances. Source/spec receipts are under
`/private/tmp/pointstream-ball-partial-resume-20261112/`; isolated client config
under `/private/tmp/pointstream-ball-partial-client-20261112/`. Preserve their
continuation if communication fails; never infer failure or replay publication.

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
