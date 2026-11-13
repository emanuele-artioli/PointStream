# Continuation — November 13, 2026 (UTC)

This snapshot supersedes the current-status summaries in the November 12 handoff.
The [budget ledger](experiments/submission-research-budget-20261104.json) is the
accounting authority; [Ball journal](experiments/k5d-ball-benchmark.md) retains
chronology and raw-result references. Times below are UTC.

## Current A2 failure and preserved continuation (UTC)

### A2 failure audited; public-runtime diagnostic released (UTC)

Root closed `20261113T033048Z-1461dd9d` as a failed GPU witness on gpu6 Ada.
All six native encode/standalone decode preparations completed, but the first
restricted DCVC receiver failed importing public `torch/testing/__init__.py`:
config `f826` omitted that runtime module. No accepted receiver receipt,
end-to-end point, quality comparison or scale gate follows. Both events are
acknowledged. The [root closeout](experiments/k5d-inputs/ada-1461-root-failure-closeout-20261113.json)
binds exact source `5ebc7831` / `abccdf05` / archive `7fd4e0e6` / environment `32e`,
physical config and raw failure evidence.

Observer 0 was selected from the actual admitted UUID before quality inspection.
Its independently replayed reservation upper bound is 150.362107921 seconds,
conditional on NFS visibility; observer 1 remains UNKNOWN. Both observer batches
85796 and old-af6 66293 ended, all wrappers/timeouts exited, and all 11 captured
A2 host PIDs / five groups were absent. The actual FQDN CPU claim file has zero
allocations; the selected GPU claim is absent. Earlier alias-path checks remain
unsupported; the corrected current-state check does not prove a historical interval.

Retain the full **900 GPU / 1,100 CPU-second A2 debit**. Programme charges are
**22,041 GPU / 58,230 CPU seconds**; Ball: 15,679 GPU / 34,229 CPU. The
[public-runtime repair card](cards/k5d-public-runtime-repair-2026-11-13.md)
releases only a **90 CPU-second / zero-GPU import-closure diagnostic**. Its
separate reservation has **900 GPU / 1,610 CPU seconds unspent**; config repair,
fresh CPU denial and another GPU witness remain unreleased. Existing Ada-stage
headroom is 3,218 GPU / 388 CPU; held SVT-only 600 GPU / 1,800 CPU is unchanged.
No transfer, retry, numerical change, training or scale release is implied.

Watch a7 found both gpu6 Ada cards idle at 03:23:52 UTC, enabling A2 admission,
and ended 03:33:49 with PID 73984 absent. Requested passive watch **a8** is now
active: PID 80489, start 03:51:59.829792 UTC, deadline 04:36:59 UTC, four hosts /
300-second cadence, same source `2b931`, 100 CPU precharged with an 80-CPU guard.
It never submits jobs or invokes models/APIs/chat wakeups. Raw handle:
`/private/tmp/pointstream-resource-watch-20261113-passive-a8/launch.json`.

The independent [SVT-only source build](cards/k5d-svt-only-confirmation-2026-11-13.md)
continues without remote scientific release. Paired 16/48 producer/scorer stages
require an accepted GPU witness and complete forecasts; Ada 292/586 still needs
a profile port. No manuscript/PRESLEY edit or submission is released.

### Closed A2 handles and unreleased next-stage conditions

Selected A2 evidence is also preserved read-only at
`.cache/pointstream-evidence/ada-1461-failure-20261113/` (21 files, 280,904 bytes).
The [tracked cache index](experiments/k5d-inputs/ada-1461-failure-evidence-cache-20261113.json)
binds manifest `7a4a4a476bfb6c1cfb98ae287d3271aa2558aa732a6d7cff34bbb6ba35953171`;
original remote/scratch evidence remains preserved. This is failure evidence,
not a successful receiver or quality result.


Raw A2 artifacts/handles remain under
`/private/tmp/pointstream-ada-window-20261113-a2/`; failure evidence is in `failure-audit/`.
Wrappers PID/birth: `295501/560333327`, `295502/560333328`; owned GNU timeout
PID/birth: `295503/560333375`, `295505/560333375`.
Batch 85796 ended 0; the original deadline `5605933.27365` is retained as provenance,
not an active wait. All listed wrapper/timeout identities are absent.
A7 observed 37.944 remote + 0.501 SSH + 0.149 controller CPU seconds; retain
its conservative charge. Old af6 batch 66293 ended 0 with both journals UNKNOWN
at deadline; all four old wrapper/timeout PIDs are absent. No new wait or replay.

Frozen 5eb already implements paired 16/48 via
`experiments.tennis.k5d_ada_scale_worker.spec(base, action='prepare', assets=...,
deadline=..., unshare_sha256=..., identity=...)`; no source edit is needed there.
Keep original base inputs/arguments, six points, exclusive gpu6 Ada and whole
clip context. Seven assets (each local/remote/sha256): bridge,
receiver_isolation_config, native_cpu_receipt, denial_witness, fidelity_gate,
accepted16 (`cf.D8_SMOKE_SHA`), ada_reference (`4bfff...`). Constructor gates:
smoke16 namespace280/wall300; full48 namespace580/wall600; campaign900,
validator30. Full48 follows only the authenticated smoke forecast.

The root GPU denial approval must bind approved/actual_filesystem_denial_passed,
host gpu6, codecs dcvc+svt, exact consumer execution identity/config, producing
job/stage and physical Ada UUID/name/capability [8,9], actual policy/profile/
runtime-inventory hashes and a complete receipt accepted by verify_receipt.
Root must independently audit all six producing receipts before approval.
CPU approval cannot substitute for that GPU gate.

Then use the same constructor with action='score', adding prepare16, prepare48,
producer16_attestation, producer48_attestation and actual producer16_job /
producer48_job. Attestations bind schema1/approved, identity, producing job/stage,
publication hash, GPU, receiver-denial flag and config/profile/runtime hashes.
Scoring consumes sealed pixels; co.score_all and validate_score retain all six
score proofs, assembly and exact16 reference checks. The current GPU witness
measures preparation/denial, not this complete scorer workload.

Cost evidence required: execution.json exit_code, seconds, timing_contaminated,
contention and paused_seconds; the sole K5D_NAMESPACE_CANDIDATE record's complete
reason/status, complete_descendant_wait, namespace_init PID1, children_cpu_seconds
and init_cpu_seconds; dispatch.campaign_clock started/phase_started/budget_end;
actual cold setup/staging/publication bytes and hashes, validator costs, claim
interval, teardown and external observer/audit/administrative accounting.
For each separate producer/scorer smoke wall W, smoke_forecast requires
estimate48 = 1.5 * 3 * W + 50 <=580 and remaining campaign time >=
max(600, estimate48+130). This is an admission rule, not measured48 evidence.
Use independently measured prepare AND score costs, 1.5 margin/failure reserve,
smokes<=600 and stages<=45 minutes. Root must reserve sufficient internal CPU
headroom prospectively; current Ada488CPU does not fund the card's provisional
paired1300CPU+1300CPU envelopes. Held SVT/old Ball funds are not silently borrowed.

Ada worker/adapter explicitly allow only smoke16/full48. Existing larger-study
k5d_control_study gates require A6000 capability [8,6], so spec-only GPU-name
changes cannot admit Ada292/586. Defer that port until valid16/48 evidence.
No next-stage scientific release is made by this audit.

### Historical af6 phase — balances, waits and watch superseded above

Root accepted exact-source native CPU proof `20261113T021221Z-f2db8617` and
real two-case SVT CPU receiver oracle `20261113T022437Z-50b4e24e` on frozen
`5ebc7831` / source `abccdf05` / archive `7fd4e0e6` / environment `32e`.
The native proof covers CPU containment/accounting. The real oracle passed
normal SVT filesystem denial and initial-launcher interruption with scoped
teardown: 16 distinct host processes reconcile as 14 main-audit + 2 outer-audit
checks (14 captured births plus supervisor and campaign). No active-native, unshare-only or fleet-watchdog claim follows.
The 50b event is acknowledged; root verified its canonical status.

The full real-CPU approval is outside git (SHA `75ae95a6984c`); its thin
[closeout](experiments/k5d-inputs/ada-5ebc-50b4-root-CPU-denial-closeout-20261113.json) records the exact source, policy, publication and audits.
Prior `c1bededa` remains a failed combined gate with acknowledged events; its
signal-enum diagnosis and the accepted caller fix are retained beside the pass.

Exclusive Ada witness `20261113T030323Z-af6c09da` **EXPIRED unassigned**.
Physical inspection establishes NO ADMISSION and no execution; the canonical
event is acknowledged. Its sole native wait (backend 70306) ended after 599
seconds, exit 10. The original observer batch wait (backend 66293) later ended 0; both journals
retained UNKNOWN and all four wrapper/timeout PIDs are absent. Do not replay the job.
No assigned UUID, GPU/decoder result, quality or scale claim follows.

The unused 900 GPU / 700 campaign CPU seconds were released. Retained cost is
400 CPU seconds: 50 preparation + 200 observers + 150 administration.
Programme charges: **21,141 GPU / 56,940 CPU seconds**; Ball: 14,779 GPU /
32,939 CPU. After a transfer of unspent funds, the Ada stage has 4,118 GPU /
1,588 CPU seconds remaining. A separate **600 GPU / 1,800 CPU reservation**
now funds conventional-only feasibility; it is not spending or new user authority.
Old held Ball 7,920 GPU / 800 CPU and W8 4,550 GPU / 929 CPU are unchanged;
the programme ceiling remains 24 GPU-hours / 28 workload CPU-hours.

**Next PLAN point: [SVT-only confirmation feasibility](cards/k5d-svt-only-confirmation-2026-11-13.md).** Source design/build
only is authorized; scientific compute is UNRELEASED. Preserve the same 30
registered clips and original three SVT settings (CRF48/55/62, preset4), with
restricted CPU decoding, ordinary models on shared Ada, full-frame and ROI
metrics, independent judges and existing guards. First qualify a separately
named 16-frame correctness/cost oracle; only then prospectively admit 48-frame/
absent controls and a complete scale forecast. The original dual-codec goal
remains open; this narrower route does not replace its missing neural comparison.

[CPU claim-path correction](experiments/k5d-inputs/ada-cpu-claim-path-correction-20261113.json): earlier alias `gpu6.json` absence checks
were unsupported. The actual `gpu6.itec.aau.at.json` file exists with zero
allocations; all six own allocations are absent now. This proves CURRENT
cleanup only, not immediate historical release intervals. Native process,
normal decoding and filesystem-denial proofs are unaffected. Preserve all
original gate, source and raw hashes; the addendum corrects interpretation.

At the af6 phase, watch a7 had not yet found capacity; it later found idle
Ada at 03:23:52 UTC and ended 03:33:49 with PID 73984 absent. Current watch a8
is recorded above. Earlier a6 ended 02:51:25 UTC after nine ticks with no idle
GPU. PR 378 later merged with green CI (Ruff, mypy 591 files, 2,197 tests /
40 skipped); the main integration advanced to `1a1f63c7`, while frozen
scientific source `5ebc7831` is unchanged.

Ada's same-GPU repeat `2c8b49a0` supports its own qualified reference; failed
cross-A6000 comparison `37b3b79b` is retained. Current validators require exclusive
GPU access. GPU/CUDA/DCVC denial, new clips and scale/full30 remain unqualified.
The expired witness provides no GPU qualification. The next route has separate
source, correctness, resource and budget gates. Paper/PRESLEY
remain paused; no training, paid hosting or submission is authorized.

Native gate: [root-accepted f2 proof](experiments/k5d-inputs/ada-5ebc-f2db-root-approved-CPU-gate-20261113.json).
Full real-CPU approval: `/private/tmp/pointstream-pidfd-signal-real-cpu-prep-20261113/root-approved-real-CPU-denial.json`,
SHA256 `75ae95a6984cd880cfadfa92f46589320cb4b165bb44eba5f5b1e33c1b471479`.
Frozen scientific source remains `5ebc7831291c91a6c45f72791e1959dbd76ac5f3`, tree
`abccdf05666392ae8ad6d2ac8d944a8a30c25ccf5d944d29eccdb3f84d595b2e`, archive
`7fd4e0e6ed2655b646d5601ba458fe66532f03040bba9b87c1607cca2e2fbfd0`.
Selected source archive, exact config, full CPU approval and raw audits are also
read-only under `.cache/pointstream-evidence/5ebc-CPU-gates-20261113/` in this
checkout (21 files,29,973,174 bytes).
[Tracked cache manifest](experiments/k5d-inputs/ada-5ebc-CPU-evidence-cache-20261113.json)
records every physical hash; CPU acceptance does not imply GPU qualification.

PR 377 is merged with green CI. That phase used documentation branch
`codex/pointstream-ada-witness-queue-2026-11-13`. Release, accepted metadata audit and orchestration
artifacts are at `/private/tmp/pointstream-real-claim-observer-prep-20261113/`;
manifest `b60a98a3`, orchestrator `b64e889f`, wrappers `3edabc0b`.
Actual wrapper PID/birth-ticks: `182609/560163568`, `182610/560163569`;
owned GNU timeout PID/birth-ticks: `182655/560163613`, `182645/560163605`.
Both native and observer batch waits later ended; preserve their original
handles and receipts at that path. All four wrapper/timeout identities are absent.
Never infer an assignment or replay an uncertain submission.

All earlier release/wait/watch statuses below are historical; this current section
supersedes them without changing their provenance or earlier charges.

## Historical recovery state before f2/50b acceptance

Native CPU upload requests `20261113T015222Z-c4503e69` and
`20261113T015753Z-3e136ddc` are reconciled UNPUBLISHED: no spec/ready/owner/
execution output or live transfer remains. First has events only; second
retains a partial1,291,699-byte archive. Cause among startup/network/NFS delay
is unresolved. Preserve both; no canonical terminal state or ACK is invented.

Root changed transport after inspection: a69,663-byte delta reconstructed
exact source archive7fd4 on gpu6, physically rehashed16,752,640bytes,0.226CPU/
1.116wall. New immutable cache:
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/checks/k5d-source-7fd4-delta-b18b0fd9aa3a4ebf9c5fcb9fd11a6b44/source.tar`.
ONE cache-based canonical native proof is released after required singlehost
doctor, now published actual `20261113T021221Z-f2db8617`; first canonical status
PENDING. Sole canonicalwait600 backend40409 is active; do not start another wait. Same250CPU allocation
(100campaign still unexecuted,150external includes prior70 +cacheprep20 +
remaining60); no new250charge/source/quality change. Same5ebc/abcc/7fd4/env32e/
spec1223945d/deadline02:27:01UTC. ActualID is recorded in Balljournal and job-id-cache-attempt.txt before
status/wait. Any uncertainty stops.
Freeze `/private/tmp/pointstream-pidfd-signal-freeze-20261113/` preserves all
releases, raw transport and delta receipts. No actualreceiver/GPU/scale release.

Resource watcha5 ended02:08:02UTC9ticks/noidle/PIDabsent; its100C is retained.
Newwatcha6PID70368 started02:11:03UTC, fourhosts/300s/max45min to02:56:03UTC
orCPUguard, same2b931script,100CPU precharged. Programme21,141GPU/56,075CPU,
Ball14,779GPU/32,074CPU; stage4,718GPU/4,253CPU remaining. Noautosubmit/model
timer/API/chatwake. Prior watcha5 section below is historical.

PR374(observer) and375(researchstate) mergedCIgreen. PR376(client-only
non-login upload hygiene) has auto-merge enabled; rootcause remains unproven.
Currentrootbranch `codex/pointstream-native-proof-recovery-2026-11-13`.

## Historical caller-fix and synthetic-canary decision

The failed CPU oracle is closed and both events acknowledged. Its normal SVT
partial output is retained; no combined gate pass. Minimal caller fix5ebc7831
is pushed with Ruff/mypy520/1,855 passed,37 skipped. New freeze/config/bridge
preparation is charged50CPU; no new scientific request is submitted.
Observer synthetic normal canary passed7.112wall/1.068933countedCPU, missing
final UNKNOWN, recorded watchdog group and leader absent. Root accepted its
same-host fixture scope and retained100CPU; no timeout-expiry/PDEATHSIG failure
test or production/GPU credit. Root acceptance is
`experiments/k5d-inputs/claim-observer-normal-canary-root-acceptance-20261113.json`.
Read-only durable selected source/audits:
`/Users/manu/.cache/pointstream-evidence/ada-d2db-and-observer-20261113/`.

## Historical failed c1 receiver oracle

Real receiver CPU oracle `20261113T012427Z-c1bededa` is **FAILED**. Its sole
canonical wait ended after 9 seconds (backend 17491, exit 10). Bounded read-only
failure audit is complete at
`/private/tmp/pointstream-pgid-c1bededa-failure-audit-20261113`.
Do not retry, edit the frozen source, release GPU work or promote filesystem,
quality or scale claims before root reviews the actual failure. Partial normal
receiver outputs cannot substitute for the complete registered witness.
Failure diagnosis: strict integer pidfd validation rejected the SIGKILL enum
before sending any signal. Normal SVT completed with matching pixels and all
five denials, but the combined gate failed. All14 host births/3groups/CPUclaim
were absent. Both failure events are acknowledged; root closeout is
`experiments/k5d-inputs/ada-c1bededa-root-failure-closeout-20261113.json`.
Minimal caller integer conversions plus behavioral regressions are pushed
on branch `codex/ball-pidfd-signal-type-20261113`, HEAD5ebc7831. Root reviewed
sourceabcc/archive7fd4 and released only one source-bound native CPU proof250C;
actualID must be recorded in the Ball journal before any wait. No real receiver
retry or GPU release. Native proof readiness/submission delegated.

The entire 365 CPU-second / zero-GPU pilot was precharged (265 campaign + 100
external). Canonical spec
`3717ea7dc63b423c1207d51df7a6d9837ccc456a11d3d0b108360fa2851817ec`,
raw spec `ee26f91c`, deadline `2026-11-13T01:59:26.284491Z`;
namespace 220, smoke 230, validator 30 and no-op 5 seconds. Preparation, release,
actual ID and wait handles are in
`/private/tmp/pointstream-pgid-d2db-real-cpu-prep-20261113`.

## Historical d2db source and accepted native proof

Scientific branch `codex/ball-campaign-pgid-oracle-20261113` is pushed and frozen:
source `d2db015d4d90094bcd9dde8286f1ef05b9bc164d`, tree
`b1a6b36cc14637db4a390db9a12c615d7ac042eb42b1415821e797b15af6e331`, archive
`177c96760f9429b2fbf7c5756d026b7b009ba292dc885ff727f18457924a5249`, environment
MAIN `32e`. Actual 2c8b bridge: 9 added / 4 modified / 0 deleted / 1,388 unchanged.
Root rehashed source/archive and reviewed paths; Ruff/mypy passed (520 files),
pytest 1,853 passed / 37 skipped. No default math, argv or reference relaxation.
Freeze: `/private/tmp/pointstream-pgid-source-freeze-d2db015d-20261113/`.
Prior PR373 merged with green CI. Current root documentation branch is
`codex/ada-cpu-requalification-2026-11-13`; it does not replace the frozen source.

Native CPU proof `20261113T010449Z-6b186df7` is COMPLETE, independently accepted,
and its event `077b5991437b589287c5ca9676ed2c1e` is acknowledged. No wait was
started: it was complete before first status. Root gate `7148f0d5` follows
8 raw metadata / 26 physical member hashes and 19 birth/cleanup checks; normal
CPU was 1.366698 inner / 1.453698 outer seconds, and TERM required KILL.
Audit: `/private/tmp/pointstream-pgid-d2db-native-audit-20261113`.
This is native lifecycle proof only. The combined receiver oracle was narrowed
before outcomes to real normal SVT plus initial-launcher pidfd SIGKILL, with
production scoped-supervisor assistance allowed, all captured births/groups
absent and unrelated control alive. No unshare-only, active-native-abort,
fleet-watchdog or external emergency-cleanup credit follows.

## Historical a5 watch and synthetic observer evidence

User requests bounded resource checks and an alternative GPU path. Passive
watch a5 PID 59213 started `2026-11-13T01:27:37.040296Z`; deadline
`2026-11-13T02:12:37.040296Z` or CPU guard. It checks gpu2/gpu3 A6000 and gpu5/gpu6
Ada every 300 seconds. Its 100 CPU seconds are precharged (80 inner / 20 outer);
SSH server/login aggregate CPU remains unmetered. Reviewed script SHA:
`2b931eb5a25f7a87cc3594103ee845ec2582cc017309b7d479a99f637f6ee10c`.
Path: `/Users/manu/.cache/pointstream-resource-watch-20261113-passive-a5`.
Prior a4 ended at 01:24:23 after 9 ticks, no idle GPU; its PID was absent.
Inspect existing run/finished records before any restart. No autosubmit, chat
wake, model timer, API bridge or recurring LLM polling is installed by this watch;
completion-triggered agent resumption remains unverified.

Standalone observer `589470b3` is code-reviewed with hard bounds; byte-only
canary `cd45a0b3` and collector `8e340a78` passed independent source review.
Its100CPU/GPU0 is precharged and ONE attended synthetic fixture call completed,
backend44215 ENDED return0; release/command/handles at `/private/tmp/pointstream-claim-canary-prep-20261113`.
GNUtimeout120s, strict process CPU limits, separate new SSD/NFS fixture roots;
no production claims. Runtime receipt independently accepted within synthetic normal scope; no retry on UNKNOWN. This is observational work,
not live fleet deployment or GPU qualification. No GPU release until independent
durable claim accounting and renewed exact-source receiver/denial gates pass.

## Budget snapshot — seconds, inclusive conservative charges

- Programme: 22,041 GPU / 58,230 CPU charged (6.1225 / 16.175 hours).
  User ceiling remains 86,400 GPU / 100,800 CPU (24 / 28 hours).
- Ball: 15,679 GPU / 34,229 CPU charged. Ada stage remaining:
  3,218 GPU / 388 CPU; held SVT-only reservation 600 GPU / 1,800 CPU is unspent.
- Public-runtime repair: first90 CPU diagnostic charged; 900 GPU / 1,610 CPU
  remains unspent, with all later stages unreleased.
- Old held Ball reservation: 7,920 GPU / 800 CPU. W8: 4,550 GPU / 929 CPU,
  held on complete-cost/prerequisite gates. These held balances are unchanged;
  the SVT-only transfer above uses the Ada stage, within existing authority.
- Charges are conservative allocations where stated, not complete measured use.
  The contended witness's 1,282 GPU / 1,632 CPU debit leaves per-job cap compliance
  unverified; it does not establish an actual overrun.

## Valid evidence, failures and remaining scientific gates

Ada `20261112T210150Z-2c8b49a0` reproduced six real 16-frame packet outputs,
metrics and judges in fresh same-GPU executions (126.574 / 112.491 seconds).
It is packet-repeat/cost calibration using Ada's own qualified reference.
Cross-A6000 comparison `20261112T202503Z-37b3b79b` failed and remains retained.
Earlier source's real SVT CPU denial oracle `20261112T233107Z-030de220` passed
exact original packet/native/YUV/RGB and five real file-denial checks. Its native
CPU proof was `20261112T232248Z-69c7f4d9`. These do not qualify d2db receiver
filesystem behavior, CUDA/DCVC denial, a new clip or a full 30-clip benchmark.

First GPU denial witness `20261112T235535Z-13aba806` stopped CONTENDED before
points/receivers; sole wait ended, events acknowledged, own cleanup observed,
failed scratch preserved, exact claim-release time unrecorded. No automatic retry.
Earlier CPU oracle failures `20261112T225652Z-b9e0fd16` and
`20261112T230721Z-cda90430` remain failures; old native CPU proofs
`20261112T190032Z-d24d7c7c`, `20261112T204701Z-6c0c0a86` and
`20261112T224626Z-3db415d5` apply only to their original sources.
Expired/unassigned, closed and acknowledged: `20261110T190907Z-4d0dd928`,
`20261110T183845Z-4c240783`, `20261112T172806Z-8709406d`,
`20261112T192044Z-461e16e8`. Never replay unknown submissions.

Ada can perform the same benchmark under its own qualified reference; current
validators require exclusive GPU access. Shared Ada needs separate numerical and
memory qualification. After failure diagnosis, require source review, source-bound
native CPU proof, real exact-policy CPU denial and complete forecasts before any
same-job six-setting 16-frame Ada CUDA/DCVC denial witness. Then separately admit
paired 16/48 prepare/score costs; 188/292/586 and full30 are not automatic.
292/586 require whole-clip context, complete prepare/score accounting, 1.5 margin
and failure reserve. Every wait stage ends within 45 minutes with saved results.
See [stage card](cards/k5d-ada-scale-and-denial-2026-11-12.md) for binding stops.
The earlier PLAN audit preceded the newly authorized SVT-only source-build
feasibility point; its scientific compute remains unreleased. No training, paid hosting, manuscript/PRESLEY edit or submission.
Preserve raw evidence, frozen sources, datasets, models and failed scratch.
