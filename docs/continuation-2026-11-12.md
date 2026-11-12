# Continuation — November 12, restored fleet access

## Current state — Ada packet replay works; two CPU denial gates failed and are preserved

User requests regular native resource polling and a non-A6000 route, while
finishing worthwhile research before paper edits. No paid server/API, training,
legacy200GPUh study, PRESLEY edit or submission. Current root branch
`codex/ada-denial-cpu-qualification-2026-11-12`; closeout6460f014/PR369 merged
with green CI; PR368 merged
the separate Ada path. All new receiver/scale code below
is isolated, not merged, not experimentally qualified.

**Accepted scoped evidence:** `20261112T210150Z-2c8b49a0` COMPLETE on exclusive
gpu6 RTX6000Ada8.9, frozenbbe94edd/sourceed890521/archive4f7f6546/MAIN32e.
A126.574wall/125.244namespaceCPU; B112.491wall/111.751CPU. All six points' actual
native/wire/zero-packet/receiver bytes and YUV/RGB hashes, full metric/judge arrays
and source rows match exactly across fresh executions on the same physical GPU.
Root recomputed153 rawJSON SHA joins to249 physically hashed archive members,
66 checkpoint files each, marker job/stage/UUID and cleanup. Charge354GPU/754CPU
(conservative whole353.869s request span plus400externalCPU), terminal eventACKed.
A reference archive SHA4bfff442081d988039abb308a3e0b8ed22632f5f598858d6eeaf4d0ef6824a1a.
This is calibration repeat/cost, NOT independent data or A6000 equivalence.
[Root closeout](experiments/k5d-inputs/ada-fresh-2c8b-root-closeout-20261112.json).

**Scope correction:** actual `k5d_worker`, receiver receipts and encoder/assembler
explicitly say filesystem isolation NOT established. The previous card called
the packet/resources-only API source-hidden too broadly. Keep that unmet
condition; no filesystem witness or benchmark promotion is credited. Old37b3
failed exact cross-class comparison remains preserved (WASB xy and DCVC hashes
differed; sourceYUV/scores and SVT hashes matched). CPU6c0c and d24d were accepted
for containment/accounting only. No GPU job or fleet wait is active; all known
jobs are terminal and acknowledged. No replay or unrecorded second submission.

**Current staged work:** [receiver/scale card](cards/k5d-ada-scale-and-denial-2026-11-12.md),
[API contract](cards/k5d-receiver-isolation-contract-2026-11-12.md). Internal Ball
stage6,000GPU/9,000CPU, sourceprep50+runtimeinventory100+passivewatch100 charged;
remaining6,000GPU/7,800CPU after1,200CPU cumulative precharges: source/input/inventory250, native proof250,
two actual-SVT oracles500, resourcewatch200. The earlier source/input/inventory
250 includes50ELFmetadata.
Exact-source native proof20261112T224626Z-3db415d5 COMPLETE and root accepted.
Both actual filesystem CPU oracles b9e0fd16/cda90430 FAILED/rootclosed and
ACKed. No scientific job or wait active. No GPU denial witness,16/48 or larger
benchmark released.
Programme charged **19,859GPU/52,528CPU** of86,400/100,800. Ball spent13,497GPU/
28,527CPU against64,800/72,000; old Ball reservation7,920GPU/800CPU preserved;
W8 4,550GPU/929CPU remains held; shared1CPU uncharged. Never transfer Ball toW8
without authority; no outer budget increase.

Independent code branches, both basedbbe94edd, to review/merge/freeze together:
- `ball_denial_builder`: `/private/tmp/pointstream-ball-denial-20261112`, branch
  `codex/ball-receiver-denial-20261112`, latest ebb45de3.
  Exact-file manifests, LandlockABI>=3, fresh exec/FD/thread guard, opt-in encoder/
  worker and SVT CPU-oracle; proposed fresh private PID/mount/IPC/proc capsule.
- `ada_fresh_reference_builder`: `/private/tmp/pointstream-ada-scale-adapter-20261112`,
  branch`codex/ball-ada-scale-adapter-20261112`, latest6bcc9efa.
  Separate16/48 producer/scorer, source/config/CPU/realwitness and
  root producer attestations; CPU oracle and first GPU witness fixed modes/specs.
  Default A6000 chain remains intact, no later archived DCVC decode.
- `alternate_gpu_audit` reviewed code only. MATERIALBLOCKERS sent builders:
  three sites fabricated desired[8,9] instead of actual producer device;
  `_private_proc` rejected two stacked proc mounts actually observed; observer
  used outer namespace PID against host-mounted proc; native decoder started
  before live observer acknowledgement, allowing a fast dav1d to escape birth
  observation. Source fixes are integrated and source-reviewed; actual CPU
  execution is still required. Composite scientific worktree
  `/private/tmp/pointstream-ada-denial-scale-source-20261112`, branch
  `codex/ball-ada-denial-scale-source-20261112`, HEADfbca94d6. Full Ruff and
  mypy518files pass. Full pytest exit0 after correcting the single stale
  negative-test fixture; focused tests also exercise actual verifier rejecting
  the old API-only receipt. Production admission was not relaxed.
  Config/source closure preparation and exact-source CPU-proof preparation
  continue locally. Exact-source native proof is now the sole released job;
  filesystem oracle and all GPU/scale remain unreleased.

**Actual read-only prerequisites, not proof:** runtime inventory on gpu6:
40,201regular files9.02GB,1,683symlinks allwithinprefix,106linked-library records,
12typedproc/system leaves. Local
`/private/tmp/pointstream-ball-runtime-inventory-20261112/manifest.jsonl.gz`,
SHA6584396bfcd5290e8f4847779c8edf32cbf1f470a384df90e16fe9a3c7b17dfc.
Remote immutable path in root-receipt-candidate.json. CUDA supplement SHA
1fc625c2f8d45dd7df75f4bb0ab73a8f2f1f82c70bd44920f6fb07cb4b836fb4:
cleanldd sm89 lacks4Torch libs (maynormallybepreloaded), and3resolvedlibs use
external historical buildprefix; driver535 libraries recorded. No runtimefailure
is established, but exact-policy grant/closure must account for these files.
Root combined inventory receipt0ee4c7b7; helper11.0CPU plus0.3supplement observed,
full100CPU conservatively charged; administrative aggregate unmeasured.
The old2c source directory was released; config source_root must use exact
`$FROZEN_SOURCE` token with code-file hashes/actual dispatch, not future job path.

Nested-proc CPU-only feasibility on6 passed (receipt5611e6af, helper2851647e):
existing outer user/PID plus inner user/PID/mount/IPC/privateproc, distinct all
four namespaces, innerPID1/child2, outerhostPIDabsent. Maps was STAT-CHECKED,
not read; NO Landlock/model/decoder proof. Root prospectively permits readonly
PRIVATEproc only after private namespace/mount/no-encoder visibility verified,
never broad host-global proc. Native children need their own self/proc leaves;
this is why the private-proc candidate exists. All private source/GT/encoder
file probes must still prove readable-outside/EACCES-orEPERM-inside, notENOENT.
Original CPU oracle on6 with authenticated real SVT packet must pass before GPU.
Actual SVT CPU-oracle input package is now prepared, no decode/model/oracle:
`/private/tmp/pointstream-svt-denial-inputs-20261112/`, input.tarSHA
8917737ba210831004491aacc6a85a2726e3a91e0d6d716e432d3ab47019f79c,
manifestSHA14669e3117f216fecd931674f0b7e5debad914f676f622e7b6741e4022662cc4.
Actual conventional point is svt-48 (first registered), not nonexistent svt-0.
Contains authenticated packet/map, original JPEG, actual GT and encoder point
state; sourceZIP/privateBallTrack weight paths recorded separately. Extra50CPU
inputprep prospectively charged; helper2.437CPU/2.439wall observed. Remote
immutable unique path in local manifest/receipt; no full pub WAN transfer.
Constructor-ready assets are now materialized in a separate immutable sibling:
local `materialized-assets.json` SHA0e7675539a31548d1d69dc5b64188ff02182ecbbd0f2512c9946b97808f64440;
protected.json SHAaad0d8f49b3e41f701d11e8342d7d3474b4382cbdcf8148f8a53a0be32ae284e.
All five real targets positively readable outside and actual sourceZIP/detector
hashes verified. Helper2.134CPU/4.16wall within the same50CPU precharge.
This is input preparation, not a decoder/denial result.
Repeated inventory hashing is ~12*10.95CPU=131.4CPU for six receivers before
encoding/decoding/IO; caps are provisional, complete1.5forecast/reserves required.

**Resource monitoring:** first45minute passive watch ended21:21UTC. New native
watchPID28708 started21:44:05UTC, every300s/all4hosts, max4wallhours to
2026-11-13T01:44:05UTC or inner80countedCPU exhaustion;100CPUprecharged (80+20).
Path `/Users/manu/.cache/pointstream-resource-watch-20261112-passive-a2/`.
No autosubmit, model timer, paid API or verified chat-wakeup. SSHserver/login CPU
is unmetered; no whole-host100CPU guarantee. Inspect launch/run/finished state;
never restart an unknown process. Actual GPU jobs must use one canonical fleet
wait after actualID/source/spec/budget recorded. Read actualID from Balljournal,
then status before acting, even if a summary names it.
Watcha2 ended23:09:09UTC after18ticks atCPUguard(remote72.624, SSHchildren1.429,
controller.376). Next same-code nativewatcha3 PID43098 started23:10:57UTC;
`/Users/manu/.cache/pointstream-resource-watch-20261112-passive-a3/`, every300s,
80countedCPU/4hmax to2026-11-13T03:10:57UTC orbudgetstop; second100CPUfrom
original200monitorlineprecharged. Noautosubmit/LLM/API/wakeupguarantee. Inspect
actual launch/run/finished state, never restartunknownwatch.


**Latest exact-source qualification:** frozen fbca94d6, sourceebd37c62cd6c27be66ddfdf58547ee66ba14ff36f31380b8316a392f06e9d4c2,
archiveaa06c0cff324e084cae01b3dccaf2eea8e987e3a32cd571677d148cadd285cc7,
MAIN32e. Actual archive bridge1389unchanged/7added/3modified/0deleted root-reviewed,
approvedSHAa6188de546eb85337efa02d476580d34c95d584085d96e155b770cc8cfc934d4.
This is source compatibility only. ZeroGPU/250inclusiveCPU proof
`20261112T224626Z-3db415d5` COMPLETE/rootaccepted, spec c46ad99f769b7e3afeca1755a6b862ead30146b1e8f34c1af9971b24087711cb,
deadline23:25:08UTC, campaign100/onecore plus150external, sourcearchive:true.
Files/clientstate `/private/tmp/pointstream-denial-fbca-freeze-20261112/`.
gpu5 doctor timedout, gpu6 doctor/readiness passed with1.58TB separate localSSD,
CPU0/emptyclaim and actual native/input hashes; unshare9f41 size31336. Newjob
uses scoped clientstate/control/upload on existing gpu6 without worker restart
or global fleet reconfiguration. Sole canonicalwait24174 endedCOMPLETE/72s. Root recomputed9exportedraw/26tar
memberSHAjoins,19hostbirths/group/supervisor/claimabsent. NormalCPU1.233580inner/
1.317957outer; TERMrequiredKILL. ApprovedCPUgateSHA84ac21dd0e899cadfa3db0e871f3a1b763f0039945597a00042a0696baa6f698.
Nativecontainment/accountingONLY, noFS/GPU. TerminaleventACKed. No native-proof job/wait active.
Unknown publication is never replayed.

CPU-only candidate config `/private/tmp/pointstream-ball-config-20261112/`:
30 archive-matched sourcefiles,1106 actual runtimefiles and exact OS libraries;
NumPy/cv2 dynamic linkage still unqualified. It is not the full GPU profile.
A GPU witness must use an actual CPU-denial approval bound to its SAME config;
changing to GPU resources later therefore requires another exact-config CPU
oracle, not reuse of a CPU-only policy. No filesystem result exists yet.

Next root actions: review the new scoped diagnostics/import-directory fix from
ada_fresh_reference_builder (basedfbca); no source change is yet qualified.
Both actual CPU oracles are rootclosed/ACKed, no job/wait active. Firstb9e0
(config0ba0) failed restricted PythonexecEACCES before native startup. Secondcda9
(config6117:onlyexactELFloaderEXECadded) produced nativeobserver but no preexec;
parent reads missing preexec before checking returncode/printingcapturedstderr,
so actual childfailure is unreported. Source correction must preserve stderr and
may add exact READ_DIR-only source/PYTHONPATH roots; importbootstrap gap is a
hypothesis, not established actualcause. No broad directory READ_FILE grants.
Any code change requires fresh full checks, archive/bridge/source-bound CPU proof
and exact-config real SVT oracle before GPU. Retain both failed scratch roots and
rawreceipts. Rootcloseouts under k5d-inputs; consumed250CPU each/0GPU.

Prospective GPU config is local-only and bound to OLDfbca source:
`/private/tmp/pointstream-ball-config-20261112/gpu-candidate/refined-v2/`,
SHAe45db9422431c8608f8e9582daec95f60f333578327b02d88c858f68cee02cdf,
33086files8.7904GB (7115excluded test/docs/example/image/detector assets).
All16knownsourceJPEG SHAintersections empty. Exact31sourcefiles, publicDCVC
weights/drivers/typedleaves from actualinventory; noGPUqualification. Needs
new execution_identity/sourcefilehashes for any future frozen code, plus exact
opt/DCVC runtime_import_path for its fresh payload bootstrap. Twelve full hash
passes roughly128CPU by byte-only heuristic; excludes rules/metadata/dlopen.
Must pass actual CPU oracle under this EXACTfinalGPUconfig before GPU witness.

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
