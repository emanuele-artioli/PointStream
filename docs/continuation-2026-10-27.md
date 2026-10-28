## LATEST — parallel development batch CLOSED, all requests terminal (2026-10-28)

Read [consolidated report](parallel-development-report-2026-10-28.md) and
[live record](experiments/parallel-development.md). Charged2400/7200GPU-s,
4800unused; CPUanalysis120/3600s conservative. Prior2980/3600study remains
separate and CLOSED. No task-owned process, GPUclaim or recurringtimer remains.
Five requests, each submitted once, all terminal:
static20261028T143341Z-c50f6395 COMPLETE; pixels20261028T143713Z-a3ae0af0
COMPLETEbothwindows; DCVC20261028T143856Z-ae2c81a7 FAILEDnativecorrectness;
SVT20261028T150051Z-445e051d FAILEDcostgate/no full; motion
20261028T152201Z-244ab9ae COMPLETE. Never replay these IDs or expiredspecs.
On any futurejobaction, re-read its ID in batchrecord and canonicalps-fleetstatus.

No validated baseline win: motion improves static but LPIPS BD-rate+230.28%
on measured qualityoverlap; higherqualitypoint and pixelcurves lack stronger
anchorcoverage. DISTSnoqualityoverlap is not a claimed equal-ratewin. DCVC's
two decodedpasses differ24/24frames andfromencoderI; no metrics accepted.
SVTsmoke correctness passes but1195sfullprojection>900allowance; nofullrun.
Fullacceptedraw/bytes/controls/geometry auditpasses. No newrenderer training,
generation, untouchedmanifest or completewire claim. Nextdecision: stronger
reproducible baselines, potentially CPUencoding/GPUjudging placement; unexecuted.

PR270implementation mergedCIpassing,adeab13c. Source140b56ea/0c1381c4/31421e6e,
exact reconstructedarchives211b4e69/3d4aa208/1d46ff03. All10ownPIDsabsent,
eventsacknowledged, foreignprocesses preserved. User's latestsubagentpolicy:
Luna6/max, precise simple launch/report tasks; rootownsdesign/review/science.
Read-onlyevidence preserved at .../tennis/parallel-development-20261028-244ab9ae/
evidence.tar SHA71900d0d,100720640bytes/105members, allmemberhashs verified;
5nativepublications also0444. Portableauditor replay byte-identical. Source
and result details in consolidatedreport. ResultsPR271 attached, auto-merge
requested after requiredLinuxCI; inspect its live state for CI/merge outcome.
Currentbranchcodex/parallel-feasibility-batch; no scientificwork pending.
All earlier RUNNING/PENDING headers below are superseded history.

## PRIOR — NEW parallel development batch started (2026-10-28)

Human approved a NEW7200GPU-s allocation: background2700,pixelcorrections2700,
DCVC/SVTROIbaselines1800, plus3600CPU-s. Old2980/3600study remains CLOSED,
620unspent is not borrowed. Read docs/experiments/parallel-development.md and
docs/cards/parallel-development-2026-10-28.md. Jobs submitted exactlyONCE:
background20261028T143341Z-c50f6395, pixels20261028T143713Z-a3ae0af0,
baselines20261028T143856Z-ae2c81a7. Re-read IDs in batchrecord and canonical
ps-fleetstatus BEFORE any jobaction. Never replay unknown/terminalrequests.
Source140b56ea, allsourcearchives211b4e69, pinned specs in parallel-specs/.
Deadline2026-10-28T16:00:00Z. Background/pixels running at firststatus; baseline
PENDINGreceipt, canonical status requested. No new quality accepted yet.
CPU gap analysis complete; confirmation manifest NOTverified. Timingsecondary,
correctness/memory/bytes/cost/deadlineshard. Root owns design/review/science;
subagents now Luna6/max with precise simple launch/report-onlytasks (user).
One bounded attended batch wait, no automatic modelcallback verified and no
recurringLLMtimer. PR270 implementation, docschemaCIheaderfailurecorrected,
focused docchecks pass; newCI pending. Preserve all priorread-onlyevidence.

## PRIOR — paid clean-background FULL COMPLETE; gate failed (2026-10-28)

Read [the audited full report](k6b-paid-clean-background-report-2026-10-28.md).
CPUbf9f97d2 and SAMEGPU20261028T100153Z-27f39d64 COMPLETE, all80Al ticks.
No replacement: first publication recovered after transfer/quota interruption;
only expired queue deadline changed before admission, cap800unchanged.
Original source/metric controls exact. Full command157.656945s; independent
pixel/rate/decision audit PASS. New base saves898bytes; total21.989629kbps.
ActualLPIPS/DISTS0.274039/0.168824; source-in-R0.268848/0.166890 versus
SVT0.240532/0.168336. OracleDISTSrobustwin, bothLPIPSrobustloss. Formal
MIXED/UNRESOLVED; NO one policy wins both>0.001. Fixed test CLOSED.
Not a both-metric NO-GO theorem or codec/TOMM win. No further renderer/recipe
or GPU request authority; one-request grant consumed, never replay old jobs.

Newcharge250GPU-s; cumulative2980/3600GPU-s,620unused. GPU3A6000, native
PIDs2778255/2778260gone, foreign2655895untouched. CPU/GPU waits ended, events
acknowledged, no recurring LLM timer. Code10a1ae76/sourcearchive7bc3be42,
nativetree82eee1d0/spec e1e7b877, fullrawpubeea71476; complete hashes inreport.
Read-only evidence archive .../tennis/k6b-paid-clean-background-20261028-27f39d64/
evidence.tar SHA3da5b858,8611840bytes/49members; raw prepared/GPU pixel payloads
retained separately/read-only by exact native job paths. Branch
codex/paid-clean-background; PR268 implementation merged with CI passing;
final result/evidence changes are being published. Baselines, wire, untouched
footage and experience-track gaps remain. All earlier running/pending headers
below are history superseded by this completed state.

## CURRENT — same request publication RECOVERED, native observation starts

ID20261028T100153Z-27f39d64 FIRST publication confirmed PENDING. Source identity
82eee1d0 matches all214files/archive7bc3be42; Git10a1ae76 unchanged. New spec
e1e7b877 changes only expired queue deadline to2026-10-28T13:01:15Z; cap800s,
all scientific/resource checks unchanged. No replacement or extraction replay.
Lastconfirmed2730/3600GPU-s before admission. Receipt
/private/tmp/clean-background-gpu-first-publication-receipt.json. Read latest
experiment entry/status before acting; never resubmit. CPUbf9f97d2complete,
independent auditPASS. Await native smoke/full outcome; no new quality accepted.

## CURRENT — same request FIRST publication in flight after usage reset

ID20261028T100153Z-27f39d64 only. No replacement. Original extraction finished;
214/214files exact, no extractor on actualhostgpu2, freshready/state/spec absent,
no admission. CPU-only FIRST publish RPC session36504 is now running, max300s.
Handle /private/tmp/clean-background-first-publish-handle.json; receipt
/private/tmp/clean-background-gpu-first-publication-receipt.json. If interrupted,
inspect same ID before any action; never replay publish or submit. Source code
10a1ae76/archive7bc3be42/nativeidentity82eee1d0 unchanged. Only expired absolute
queue deadline refreshed prospectively before any publication/admission; all
inputs/settings/800s allocation unchanged. Last confirmed spend2730/3600GPU-s.
Earlier auto-review failed from quota and DID NOT EXECUTE the recovery action;
user explicitly resumed. Read latest experiment entry for complete digests.

## URGENT current GPU request — publication uncertain, never resubmit

The ONE approved request is20261028T100153Z-27f39d64, allocated but source
extraction timed out90s before publish. SourceGit10a1ae76, archive7bc3be42,
spec58c71c85; exact digests in latest experiment entry/local request JSON.
No ready/admission/smoke/full confirmed. Last confirmed spend2730/3600s,
800s cap on this same request;870unspent before admission. Do NOT create a
new ID or replay extraction. Inspect exact214-source-file identities and
canonical ready state. If complete/unpublished, recover FIRST publish under
this same ID, using preserved spec/snapshot and bounded CPU-only RPC. This
pattern previously recovered f6a754eb without a replacement submission.
CPU source preparationbf9f97d2 COMPLETE and independently audited; no GPU cost.
Source/meta recovery is the current task, not a new research decision.

## 2026-10-28 paid clean-background CPU preparation COMPLETE/audited

Job20261028T091736Z-bf9f97d2 COMPLETE on gpu3, ZERO GPU time. All240source/
stream frames and80saved controls prepared; independent audit PASS240RGB hashes,
80overlay/Rreplays, rate/member checks. Raw outer prepared payload SHA56f54ccd
(full digest in experiment entry),660490240bytes. Native new base19659bytes,
16.3825kbps + unchanged side5.607128889 =21.989628889kbps, within both old
SVT hulls. No new quality metrics yet. CPU wait21654ended; event acknowledged.

One <=800GPU-s request remains accepted within870unspent of original3600cap;
none submitted yet. Score code and inputs ready, spec at
/private/tmp/clean-background-gpu-spec.json. Original2730s spent. A6000gpu3
fits8000+8192margin, gpu2doesnotatlastprobe. Full-history inventory times out
on both managers; own canonical jobs and qualified-host claims are checked,
not uncertain research submissions. Native admission still required. Do NOT
replay old jobs. Read card and latest experiment entry; preserve no-retry grant.
Branchcodex/paid-clean-background; PR268 tracks implementation; PR267 merged.

## 2026-10-28 CPU preparation request published

CPU-only job20261028T091736Z-bf9f97d2 PENDING at receipt, sourcee542dfed,
snapshot6a5cf437/specc6acbea3. Read its experiment entry and canonical fleet
status before acting; never replay. All6CPU hosts eligible and workers fresh.
GPU grant remains one <=800s request after successful CPU source/stream controls;
no GPU request yet,2730/3600s spent/870unused. Native bounded attended wait,
no recurring model monitor. Local spec/submission under
/private/tmp/clean-background-cpu-{spec,submission}.json. CPU/GPU code frozen,
27focused checks and ruff/mypy pass; full suite984pass/25skip plus the two
previous macOS limitations. New quality-validator tests also10/10pass.

## 2026-10-28 paid clean-background oracle ACCEPTED

Human accepted “Approve the paid-background oracle”: one new GPU request <=800
of870remaining, after exact CPU preparation and passing smoke, no retry. Read
[candidate card](cards/k6b-paid-clean-background.md). CPU preparation and four-
policy scoring code are on codex/paid-clean-background; ruff/mypy and27focused
checks pass. No job submitted yet; re-read latest experiment entry for IDs.
Frozen erasure is Telea3 independently on native YUV420 planes with original R
nearest, then original scale4/CRF63/preset4. Source YUV hash must match and every
judged source RGB/old actual control must reproduce exactly. No training,
generation or SAM. Existing ledger2730/3600GPU-s,870unused; prior negatives
stand. PR267 merged db3671cae36c349d80412e8b9796547ddef67b53 with LinuxCI PASS.

## 2026-10-28 CPU follow-up — side-rate feasibility COMPLETE

Read [the saved-score accounting report](k6b-side-feasibility-report-2026-10-28.md).
No new GPU work. Robust joint side ceiling is0.668734kbps for actual overlay
and1.230365kbps for ideal_R, compared with KPS3.53 and other side2.077129.
Each group alone exceeds ideal_R's ceiling. All three predeclared lossless
wrappers grow the original KPS packet; all roundtrips exact. Machine-readable
JSON and CPU implementation are committed, with frozen parent archive checks.
Next [paid clean-background proposal](cards/k6b-paid-clean-background.md) changes
the base to remove players before encoding, then tests actual/source-pixel
policies while retaining every side cost. Proposed <=800 of870 remaining GPU-s;
awaiting explicit acceptance, no new stream/worker/request. Do not treat proposal
as permission. Original finite NO-GO and TOMM evidence gaps remain unchanged.

## 2026-10-28 latest result — full ideal-player finite NO-GO

User directed full launch with timing secondary. Job20261028T080453Z-34e50ec2
COMPLETE: all80Al ticks, exact original metric controls, full command144.138s.
Independent CPU audit passes400input images/160oracle reconstructions, scores
and decision. Best ideal_R LPIPS0.270927>anchor0.234057 and
DISTS0.167818>0.164586 at total22.737962kbps, robust+/-0.001. Other support
also loses both. Finite NO-GO; no new appearance-only renderer without a new
dominating bound. Read [full report](k6b-ideal-player-full-report-2026-10-28.md).
Cumulative2730/3600GPU-s,870unused. All task-owned processes/waits ended,
events acknowledged, no timer. No new GPU request authorized by this result.
Re-read latest experiment ID/status before any action; never replay unknown work.
Original actual-player failure and TOMM baseline/wire/fresh-evaluation gaps remain.

## 2026-10-28 latest continuation — ideal-player attempt CLOSED

Read [the final oracle report](k6b-ideal-player-report-2026-10-28.md).
First request20261028T071835Z-2964e9ef failed before campaign import (60s).
User approved one repaired request20261028T073233Z-36f21aa5; it reproduced
smoke controls exactly but FAILED25% warm stability (ratios1.9233/1.9294).
Full80-tick run never launched; NO accepted quality result or appearance verdict.
CPU audit passed all400input images/160oracle reconstructions and smoke controls.
Repaired request charged70s; cumulative2480/3600GPU-s,1120s unused.
Both one-request grants consumed; NO further GPU authority, retry or live wait.
All task-owned processes ended, foreign workload untouched, events acknowledged.
Source inputs468250a7 and all evidence preserved read-only. Prior actual-player
scientific failure and TOMM evidence gaps remain. Re-read the latest experiments
entry and canonical status before acting on any ID; never replay an uncertain job.

# PointStream continuation after root-chat archive — 2026-10-27

Current accepted outcome: **SCIENTIFIC FAIL** of the fixed real-base player
recipe. Job20261027T211451Z-f6a754eb is canonically COMPLETE, valid early stop
after Al loses both full-frame metrics. Med actual-base scores remain UNSCORED.
Both historical control windows passed all80ticks. Independent score/rate
audit passed55checks; CPU replay matched240hybrids/480image hashes without
source video or target RGB. Read the
[final report](k6b-real-base-quality-report-2026-10-27.md) before any new work.

Conservative study **2350/3600GPU-s**, **1250s unspent**. Final request
charged1320s (1276.951465s primary,1318.620690s through terminal reconciliation).
No active wait, scheduled callback or recurring model monitor; all events
acknowledged. Background callbacks and model cache lifetime remain unknown.
Do not replay any original request or treat unused time as a new-method grant.

Source01888c76ce2dd2d7ffbb5ceb63f9a28e774a41f1, source identitybe7dfb19,
normalized spec33474777; actual GPU2 index1, A6000, UUID40e9a544-b3d6-e5f3-a736-7020a6401bfc.
Source archive0c153bba and full published archivee346844d are preserved.
The original transfer timeout was recovered by verifying210files then completing
the FIRST publish RPC under the SAME ID, never resubmitting. PR259 merged.
Read the experiment ledger for complete checksums and all immutable input paths.

The [ideal-player bound](cards/k6b-ideal-player-bound.md) is a NEW PROPOSAL
requiring explicit acceptance: no training/generation, finite target-appearance
oracle, proposed1100s within1250s left, fresh controls/cost gate and8,000MiB
reservation supported by measured3.522GiB peak plus existing8,192MiB margin.
CPU preflight confirms all80Al source RGB hashes in preserved failed-smoke
truth-al/source.npy; protect that data until any new input archive is published.
No GPU request exists for the proposal. The original failure is not relabeled.

The historical archived state below predates this completed study and is kept
for provenance; its580/3020figures and next steps are superseded by this header.

## Historical goal and original stop

Seek a selective semantic-coding win that can support a paper on players and
ball, followed by extension beyond these development windows. A ball-only
contribution is insufficient (user). No complete, fully delivered PointStream
win over both SVT-AV1 and the required neural baseline is established.

The original handover required CPU-only planning. That allocation was accepted
and the current task is the same real-background study, through fresh controls
and resource gates. Do not start a new generator, fitting, full240
rendering, ball integration or another baseline experiment as a substitute.

The governing card's goal is:

> Goal: test whether the frozen PR242 lossy-reference generation and PR247
> per-frame SAM players improve a REAL decoded whole-frame SVT base at equal
> TOTAL rate on the same Al031@129 and Med014@194 development windows.

Read `AGENTS.md`, `PLAN.md`, `docs/cards/k6b-real-base-quality-pilot.md`, and
only the relevant end sections of `docs/experiments/k6b-players.md` (check file
length first). The experiment's final section is titled "2026-10-27 — Real-base
quality smoke stopped at the cost gate". Earlier "running" entries record
history; this terminal section supersedes them. PR #252 merged its report and
worker; PR #253 merged storage safeguards. This continuation is maintained by
the final root handover PR linked in PLAN's ledger.

## What has been learned

- Egocentric split gains disappear once masks and fair adaptation are charged.
  Hand generators failed transfer at the keypoint rate. Tennis pixel splitting
  also failed its tested oracle/fully billed configuration.
- The tennis ball streak component improves ball detection at very low rate,
  with its documented whole-frame quality tradeoff; this does not establish
  the desired broad paper contribution.
- Earlier player fit variants failed their frozen six-window rules. A later
  12-window familiar-broadcast diagnostic reached 23/24 metric wins with
  references and exact mattes granted. Actual side-information accounting then
  removed the apparent complete win. These repeatedly used windows are now
  development data, not fresh confirmation.
- Video-compressed **unmasked full-RGB references** retained actual pilot
  generation quality (4/4 strict metric cells). Alpha/black references, dense
  carriers, and the tested paid coarse mattes failed. See PRs #242–243.
- Receiver silhouettes: YOLOE 1/4; SAM tracking 2/4; independent-frame
  pose-prompted SAM **4/4** strict, sensitivity +0.001 **3/4**, zero empty masks
  across 320 player-mask predictions. Independent receiver-read isolation and
  reset checks passed. This is the promising component (PR #247), with no
  transmitted matte bits, but its background was target-derived Telea and
  granted. It is not a delivered-background win.
- Native source RGB conversion is now reproduced exactly on A6000; CPU/GPU
  rounding differed by at most one 8-bit level. Do not relax source identity
  checks or reuse CPU arrays as if they were byte-identical GPU controls.

## Frozen scope and stop conditions

Two windows Al031@129 and Med014@194; **80 saved judged ticks/window out of240**.
Actual native whole-frame SVT base scales8/4/2; **one global scale4 central
operating point**, with scale8/2 completing the limited curve. Same exact
source, R/S evaluation masks, crop/reference order, conditioning and tick
membership. Original base has240 frames; missing160 generated ticks are not
silently interpolated or claimed delivered. No new generation, model, fitting,
reference, ball inference, DCVC reproduction or ROI comparison in this budget.

The card requires >=3/4 strict LPIPS/DISTS full-frame cells AND >=3/4 fixed
foreground cells at equal total rate, with frozen ±0.001 sensitivity. When
quality ranges overlap, additional per-metric BD-rate uses PCHIP and the
achievable anchor envelope, mean <=−10% and >=75% window wins. No extrapolated
BD-rate, no per-window quality-selected settings. The existing early-stop rule
and original controls remain unchanged; read the card and worker for detail.

Original source/granted-matte/predicted-mask controls precede new quality.
Stop on failed identities, controls, resource or decision gates, exhausted
budget, uncertain submission, or a required scope/budget decision. Every waited
stage is <=45 min. A failed projection is not permission to raise a reservation
and continue silently. The next plan must be prospective and separately
accepted; no new GPU job was authorized at this handover.

## Jobs, code, results and budget

Re-read IDs here and run canonical `scripts/ps-fleet status JOB_ID` before any
job action. Inspect an uncertain request; never replay it. The terminal quality
state was rechecked at closure. The two recovery states are also checked for
this handover and their result is recorded in the closure note below.

| Job | State/outcome | Code/budget |
|---|---|---|
| `20261027T072713Z-2888cefe` | COMPLETE, CPU native-base recovery, six240-frame streams and retained YUV | 184.54109/1800 workload CPU-s, no GPU; audit37.743262 CPU-s separate |
| `20261027T074823Z-135421ee` | COMPLETE, GPU conversion; 80/80 original source hashes per window; six80-frame base RGB sets independently repeated/verified | frozen diagnostic worker `146f06ee`; admitted282.821585s, conservatively370 GPU-s |
| `20261027T080644Z-9dba0803` | FAILED **cost projection only**; both window controls pass on the predeclared two smoke ticks/window; full never launched | worker `8bd01ff5703ebf5b1cf41feef4f564f948cf44b2`; admitted151.469499s, conservatively210 GPU-s |

**Historical original checkpoint total: 580/3600 GPU-s conservative (0.161111/1.0 GPU-hour),
434.291084 actually admitted; nominal3020s remain.** All setup, smoke, failure,
validation and publication costs stay inside the same cap. Previous study
budgets are closed. No accepted full quality score, win/loss, or BD-rate from
the latest real-base run exists.

The full worker projected **2227.523141s**, **2448.314646s including
publication**, exceeding its frozen1800s full-stage reservation. Other smoke
checks passed. This is a computational feasibility stop, not a quality failure.

Immutable evidence lives under
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/`:

- CPU recovery: `20261027T072713Z-2888cefe/full/published.tar`,
  SHA256 `00b13e89077a27a3e6ff23bf476e9795c2120c5b1f5252ddfdd6f533a8119a39`.
- GPU conversion: `20261027T074823Z-135421ee/full/published.tar`,
  SHA256 `5c4a0cfced5501e085f67d394f3a50ca7f92d2cb592f73fbd3ad661461cb691b`.
- Exact smoke validation is versioned at
  `docs/cards/k6b-real-base-quality-smoke-validation.json`,
  SHA256 `fd5509a633ae0a4dccdc944d66b4f66113f3c8a2900c463630e92ed3b7455085`.
  The experiment/card records input, spec, environment and patch identities.

## Next actions, in order

1. Reconcile current job/worker health with these terminal records. Preserve all
   existing evidence and inspect unknown jobs rather than duplicating them.
2. On CPU, decide whether a computational redesign or a revised bounded stage
   allocation can fit the unchanged full test, controls, new smoke, publication
   and overhead inside **3020 remaining seconds**. The prior2448s estimate alone
   is not a validated complete quote; shared-host timing is not a speed claim.
   Record the proposed revision and acceptance before execution. Do not tune
   cohort, central scale, metrics or quality thresholds to make it fit.
3. If no defensible plan fits, stop and explicitly decide whether more compute
   is worth buying with a new user-approved budget. If a revised plan is
   accepted and its smoke passes, finish the same real-base study and stop at
   its predeclared decision. No scientific quality conclusion is implied now.
4. Only if delivered-background feasibility survives: serialize and independently
   decode the missing adapter tensor/bootstrap directory, close all delivery
   accounting, add qualified DCVC and fair ROI/pixel-residual controls with
   separately approved budgets, then integrate the paid ball component on the
   same windows. Freeze the method before acquiring untouched evaluation data
   and auditing train/reference/judge lineage and original source/PTS/replays.

Rates in the current study are explicitly **optimistic floors**: native IVF,
KPS and actual reference/update payload bytes are known, but required adapter
serialization/bootstrap/framing is incomplete. A negative result can reject
the tested configuration even at that floor; a positive remains conditional.
Installed common tennis/SAM models must be declared and installation sensitivity
reported. Three SVT points do not establish superiority over DCVC or stronger
ROI/pixel alternatives. All12 windows remain development; the115px cutoff is
exploratory, and historical step5 failure is not erased.

## Storage, workflow and coordination

Use `/local/users/emanuele` for host-local runtime, environments and scratch;
never large `/tmp` or `/var/tmp` files. It avoids NFS small-file latency while
using a separate disk. Shared NFS remains durable source of truth, with packed
inputs/environments staged through the fleet.

Read `docs/storage-maintenance-2026-10-27.md` before storage or worker changes.
Inactive files on all six hosts were copied/checksum verified; originals and
old paths remain accessible through compatibility symlinks. Remaining real
root-disk trees have active process references. They must not be removed to
"finish cleanup". Relocation journals and hidden live originals are application
state, not all disposable cache. Fresh-shell defaults are installed/verified;
manager safeguard rollout remains blocked before any worker change by the
existing20s environment-probe gate on gpu2/gpu3 (bounded diagnostic: gpu2 valid
in38.84s; gpu3 >60s). Do not bypass the health gate or infer that reachable
hosts are unavailable. gpu5/gpu6 live public-directory switching was rejected
by automatic approval review; no retry bypass occurred. Finish migration only
when safely idle or with a specifically reviewed live-migration plan.

Existing study chat: `01a16e7a-e752-78f3-b372-c338d8cdbf46`, title
"Prepare tennis confirmation study", local checkout
`/Users/manu/.codex/worktrees/7e8e/PointStream`. It is idle, not archived by this
request. The user has authorized direct coordination with that study chat.
Do not start a turn there merely to read status. The user prefers paced active
sessions; no new chat was created as part of this closure.

The root chat being archived is `01a16cfc-1a88-7013-bd79-8489fe7c1eb2`.
No new research job, recurring LLM heartbeat, keepalive or pending overnight
execution was created. Prior overnight work did not run because unattended
file/process/MCP isolation was not verified. Attend future execution unless
that gate is actually satisfied. Cache lifetime is unknown; do not promise KV
retention. Preserve this record and existing stopped work when starting a new
root chat from current main.

## Closure verification

Canonical fleet status rechecked all three IDs at root-chat closure: CPU recovery
COMPLETE, GPU conversion COMPLETE, real-base quality FAILED at its smoke gate.
The full stage remains unlaunched. All scientific jobs listed here are terminal.
Storage PR #253 merged with Linux CI passing. The final documentation is in PR #254,
linked in PLAN.md; no future session needs the archived conversation to resume.
