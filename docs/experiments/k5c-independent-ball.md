# Experiments: K5c — independent image judge for the ball component

## Current decision, 2026-11-03: approved pilot complete

The original 48-frame development target completed under the explicit
[operational amendment](../cards/k5c-48-frame-continuation.md), using the
independently audited 16-frame controls. The first cost failure is preserved
below and in `k5c-inputs/first-attempt-20261103/`; it was not retroactively
passed. Continuation `20261103T141758Z-f5d52fac` completed on gpu6 RTX6000 Ada.
Its actual 48-frame component output passed independent CPU audit, not a larger
confirmation. The fleet full stage performed no additional experiment.

**Accepted cumulative envelopes:** **463/900 GPU-s and 1,413/1,800 workload CPU-s**.
The continuation charge is 262 GPU-s (261.330-second creation-to-terminal worker
settlement, corroborated by absent supervisor/child/process group and claims),
and 312 CPU-s: 20 preparation +262 single-core lifecycle +20 audit +10 monitoring. This
includes the first phase's 201 GPU-s and 1,101 CPU-s. These are conservative
allocation envelopes, not invented exact allocation or aggregate administrative
CPU telemetry. All task-owned GPU processes/claims and the single attended wait
are closed; terminal events acknowledged. No new computation, training, W8
retry, third attempt or larger study is admitted by the unused remainder.
Receipts: [`k5c-inputs/continuation-20261103/`](k5c-inputs/continuation-20261103/).

**Frozen decision rule result:** PASS for source qualification, complete native/
BALL records, actual independent-image-judge path, physical byte ledger and
feasible cost on the registered 48-frame prefix. Source WASB is 45/48 within 10 px
(43/48 within 5 px), above the 80% source gate. All 48 labels are visible positives;
zero negative or excluded frames, so negative-frame false positives remain
unmeasured. All six native streams declare and contain 48 frames; packet geometry,
hashes, label/row joins, 5/10 px counts/F1, byte rates, native decoder-hash equality
and per-setting RGB-PSNR guards pass. No guard-off ball was present.

| Codec setting | Native bytes | BALL container bytes | Total kbps | WASB 10 px native -> streak | WASB 5 px native -> streak |
|---|---:|---:|---:|---:|---:|
| DCVC-UF QP 0 | 6,689 | 237 | 34.630 | 0/48 -> 42/48 | 0/48 -> 40/48 |
| DCVC-UF QP 21 | 20,118 | 247 | 101.825 | 1/48 -> 44/48 | 1/48 -> 40/48 |
| DCVC-UF QP 42 | 62,374 | 245 | 313.095 | 6/48 -> 43/48 | 6/48 -> 40/48 |
| SVT-AV1 CRF 48 | 80,787 | 254 | 405.205 | 8/48 -> 44/48 | 7/48 -> 40/48 |
| SVT-AV1 CRF 55 | 55,833 | 248 | 280.405 | 5/48 -> 44/48 | 4/48 -> 41/48 |
| SVT-AV1 CRF 62 | 27,856 | 244 | 140.500 | 0/48 -> 44/48 | 0/48 -> 41/48 |

The 48 frames are tracknet/game8/Clip6 at 30 fps (1.6 s), historical development
footage. These physical native+BALL containers include native IVF framing and
are distinct from the historical four-window component rates. No common
integrated-system envelope or installed-model charge is asserted. The first 16
prefix is nested, not 16 additional independent observations. The WASB reported
training split excludes game 8; checkpoint optimizer inputs were not independently
audited. Source identity strings were bound, not all external original images
reread during CPU audit. Native decode replay ran inside the GPU worker; CPU
audit checked saved parity hashes and packets, not fresh native/model inference.

**Narrow finding:** a separate source-qualified image model recognizes rendered
ball marks more often in these observed settings, at the billed additional
bytes. This cannot distinguish realistic ball appearance from a recognizable
bright mark, establish negative-frame behavior, untouched-clip generalization,
matched-rate BD-rate or a complete codec/source-hidden receiver advantage.
Crop LPIPS/DISTS were not implemented here. The historical +1% PSNR BD-rate guard
failure and original 6/8 rule are unchanged; nondecreasing PSNR at the same codec
setting is a different contract. Next scientific test would freeze distinct
eligible clips with positive/negative labels and the independent judge, then
seek a separately reviewed, costed confirmation allocation. No confirmation is
scheduled or authorized by this pilot.

Independent CPU audit SHA256
`ae6a285b7213403328c9cd31eb77aa5bad35dfcbfdb72fad7254589d22013fe0`,
script SHA256 `dd879d32a6f7e3fc138851808fa231db227c5226feaffc4fda4c2d8d96af6fd2`;
record [`continuation-20261103/independent-audit.json`](k5c-inputs/continuation-20261103/independent-audit.json).
The process reported about 0.058 CPU-s; the 20 CPU-s audit envelope includes
extraction/launch. Actual model/codec work was not rerun by the audit.

## 2026-11-01: bounded pilot approved; pre-admission gates pending

The user asked to strengthen the paper beyond visual presentation and retained
the ball as a main-paper result. Restoring the original K5b evidence does not
supply an independent output judge: BallTrack provides both primary encoder
positions and the old image score. An additional image-based judge addresses
this competing explanation without repeating the stopped K5b stage.

**Hypothesis:** a separately trained, source-qualified WASB judge finds the
rendered ball against independent TrackNet annotations where native codecs
miss it, after charging actual extra side information. **Competing explanation:**
the old gain reflects BallTrack's preference for marks produced from its own
positions or a bright synthetic mark, rather than an independently detectable
ball. Neither outcome settles realism or complete-video codec quality.

**Decision rule:** the proposed single-window pilot qualifies the correctness,
independent-image-judge path, physical byte ledger and measured cost only.
Source WASB recall must be at least 80% of visible labels within 10 pixels;
absence of negative examples leaves false-positive estimation unresolved. All
six frozen native points, registered frames and serialized packets must pass
substantive checks. The pilot cannot pass the old 6/8 rule, revise its +1%
frame-PSNR guard or become an unseen-data confirmation. A larger confirmation
needs fresh review and authority, with its protocol fixed before outcomes.

**Authority/budget:** the user approved the [bounded
card](../cards/k5c-independent-ball-pilot.md) at 900 GPU-s and 1,800 workload
CPU-s on 2026-11-01, conditional on pre-admission correctness, inherited
single-core lifecycle bounds, training-split/identity receipt and a complete
smoke-derived projection. Historical K5b budget is closed; W8/W9 balances cannot
be borrowed. The numbers are approved caps, not measured cost or assurance
that this path fits.

Root recovered the pinned WASB authors' dataset configuration, loader and
pretrained-model instructions at `923462cacdeb3353b84ddebdedb3f4b7a8553b0f` via
the GitHub contents API. The config specifies games 1–7 for training and 8–10
for testing, and the instructions map `wasb_tennis_best.pth.tar` to that tennis
configuration and step-1 inference. Its SHA256 and primary URLs are in
[`k5c-inputs/wasb-reported-split-provenance.json`](k5c-inputs/wasb-reported-split-provenance.json).
This is author-reported training provenance, not an independent audit of
checkpoint optimizer inputs. The actual pilot weights/code/windows and their
source image/label joins must match the reviewed receipts before execution.
Game8/Clip6 is historical development footage even if excluded from the judge's
reported training split.

A new sequential entrypoint and focused CPU checks are being prepared in
`experiments/tennis/k5c_independent_ball.py`. It reuses the real native codec and
ball-fitting/serialization primitives. It must be reviewed and its substantive
validator and resource admission completed before any fleet submission. No
mock timing, test fixture or prepared CLI is codec-quality evidence.

**Pre-admission review:** worker code `74cb91d4` has ten passing CPU checks,
full Ruff passes, and local mypy has seven existing environment/baseline errors
in unchanged files. Root reviewed the production native primitives, real
fit/render/packet parity, the independent image-input judge, source-first
qualification, all-frame labels and physical-byte validation. A once-only
DCVC attempt preserves failure evidence; YUV/NPY intermediates remain in
separate host-local scratch rather than the published archive. The substantive
smoke validator also requires a conservative 3x frame-count, 1.25-margin
command-time projection plus lifecycle reserves, within the stage and remaining
campaign budget. Crop LPIPS/DISTS are not implemented for this correctness pilot.

Fleet inspection covered all six hosts. The 1,560-record canonical job listing
returned after one bounded 60-second timeout and a successful 180-second retry;
no existing request is replayed or cancelled. gpu3 failed the atomic-visibility
read deadline and is excluded. Doctor passed for the complete remaining
compatible pool gpu2/gpu5/gpu6, whose workers are fresh. gpu6 was idle at the
capacity snapshot; the compatible host pool is not manually pinned to it.
An attended CPU-only probe of the deployed manager on gpu6 verified one selected
CPU inherited by a child and grandchild, using 0.120445 self CPU-s and 0.041506
child CPU-s. This supports cooperative containment of reviewed code, not an
adversarial security guarantee. Actual native paths/device and full cost still
require the authorized smoke.

Two new immutable input files (author-reported split and reviewed exposure
receipt) were written once to a fresh external run directory, read back and
hash checked. No weight or prior result was changed. Exact receipts and the
conservative **300 CPU-s preparation charge** are in
[`k5c-inputs/preflight/admission-summary.json`](k5c-inputs/preflight/admission-summary.json).
Control-plane CPU telemetry is incomplete; that allocation is not a claim of
measured exact aggregate use. Original W8/K5b budgets are untouched. The frozen
new spec is [`k5c-inputs/pilot-spec.json`](k5c-inputs/pilot-spec.json), with a
720-second single-CPU campaign, 240-second smoke, 420-second 48-frame stage,
no retry and deferred failed publication. Inclusive outer caps remain
900 GPU-s / 1,800 workload CPU-s.

**Jobs:** none yet; record the actual submission identity before waiting.
**New GPU use:** zero. PRESLEY is untouched; no training or automatic larger
confirmation is authorized.

## 2026-11-03: resumed before the first submission

Saved state and the canonical fleet listing confirm that no K5c request was
submitted and GPU use remains zero. The November 1 source/receipt publication
succeeded; it is preserved and is not replayed. The expired original spec is
retained. The new [spec](k5c-inputs/pilot-spec-20261103.json) changes only its
deadline and the verified healthy host pool. All six hosts were inspected;
doctor now passes for gpu2/gpu3/gpu5/gpu6 and all 1,560 requests were read, with
no currently pending/running/launching/preparing request. Foreign GPU activity
is untouched; at this snapshot the two gpu2 A6000 devices were idle.

An independent static review found no blocking native-interface mismatch.
The newer DCVC worker fixes a container-hash telemetry variable, while HT-L,
QP, checkpoints, color input and native acceptance guards retain their prior
settings. The actual target-device path remains subject to the authorized
smoke. A small source change persists source-judge qualification even if its
gate fails, so a failed oracle retains its measured denominator and predictions.
All ten focused CPU tests and full Ruff pass.

The resumed conservative preparation allocation is 300 CPU-s, taken from the
existing contingency: cumulative preparation **600 CPU-s**, leaving an envelope
of 840 single-core lifecycle +180 independent audit +120 monitoring +60
contingency =1,200 CPU-s. The original **1,800 CPU-s /900 GPU-s** ceilings do not
increase. These allocations are not invented exact control-plane CPU telemetry.
The refreshed records are in
[`k5c-inputs/preflight/resume-20261103.json`](k5c-inputs/preflight/resume-20261103.json).
No new budget, training, retry or larger confirmation is implied by resuming.

## 2026-11-03: authorized single pilot submission

Submitted once as fleet job `20261103T132157Z-f665bece`; initial state pending.
Normalized spec SHA256 `52abc84a4c65e5ae0cc5da9e276ca0dc58c2a235d2dc67917847fda8441ffa77`;
clean source commit `862d56c2a7d4499a0a0a9ed97f5a9f535b659cfb`, source archive
SHA256 `e24a6b252a79e3968f464d0eace460d05b7bac37a675229d070c571d050a2d1e`,
snapshot SHA256 `00e487604fda13590e832ed7aaa85e85e31f26303415dfca7e2f5c541b5b2d87`.
The frozen spec identifies the input checksums; large input verification is
charged to the campaign. The sole attempt reserves the existing 900 GPU-s cap;
600 CPU-s preparation is conservatively charged against 1,800 CPU-s total.

**Outcome:** failed at the smoke cost gate; no 48-frame stage was admitted.
The smoke command exited zero in 138.110 s, but its validator returned
`passed: false`: the conservative 3 x 1.25 projection plus 60 s is about
577.9 s, above the 420 s full-stage ceiling. The validator reached this cost
check after its saved-output checks; its failure is not a passed pilot or an
independent audit. No quality or 6/8-rule claim follows. The failure stops this
sole authorized attempt; no retry or promotion was made.

Execution was on gpu2, NVIDIA RTX A6000 index 0, UUID
`GPU-fc7e8fff-75be-4984-a185-fe8ce63edcd9`. Exact source and input hashes,
CPU affinity, native tool identities and staging are in the saved dispatch.
The supervisor reported start-to-end 172.10 s; this is not a verified
allocation-to-release measurement, so conservatively retain the approved 900
GPU-s reservation and 840 CPU-s lifecycle envelope until separately audited.
The saved exact-job status reports failed, no GPU compute processes and no
PointStream claim. Its terminal and decision events were delivered and
acknowledged. Read-only small-record exports are at
`/private/tmp/pointstream-k5c-20261103-receipts` and
`/private/tmp/pointstream-k5c-20261103-receipts2`; original records remain in
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261103T132157Z-f665bece`.

### Continuation admission review, before dispatch

Fresh read-only canonical inspection re-read the closed original request, all
six hosts and all 1,563 requests. No request is pending/running/launching/
preparing/resuming. The older gpu3 attention record
`20261024T082408Z-3cb3f467` has an absent execution owner and no assigned GPU
in this snapshot; its underlying state is unresolved and is left untouched.
gpu2 has two idle A6000 devices and gpu6 one idle Ada; foreign work on gpu3/5
and the other gpu6 device is preserved. Prior doctor receipts for the full
compatible gpu2/3/5/6 pool remain valid; canonical dispatch verifies workers
freshly. Raw inspection records are at
`/private/tmp/pointstream-k5c-continuation-inspection-20261103/`.

Root reviewed the controller-only extension, including a full-stage non-result
validator correction. All preexisting numerical helper ASTs are identical to
`862d56c2`; only run/validate/main admission and explicit completion handling
change. Sixteen focused CPU tests and full Ruff pass. Local mypy retains seven
known baseline/environment errors outside this worker. The 48-frame phase
retains the exact prior archive, input/model/native identity bindings, actual
source qualification and all six native/BALL paths. The zero-frame marker is
validated solely as completion bookkeeping, never scientific data.

### Continuation submission, 2026-11-03

Submitted exactly once as `20261103T141758Z-f5d52fac`, initially pending;
absolute deadline 2026-11-03 14:45 UTC. Clean source commit
`6040fe5c7da6d804ea52d3d190e6aeb04c0e20b2`, source archive SHA256
`775169b3a27a5dd42c80cf79e83540dbf6c6f5287ee9d16892616b59adbd865c`.
Submit receipt normalized spec SHA256
`161a958ffff2aba437dd1a32c56255e8dd6002be6554349108133e599dde5de1`;
this differs from the pre-admission expected hash and requires reconciliation.
Receipt: `/private/tmp/pointstream-k5c-continuation-inspection-20261103/submit.json`;
results: `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261103T141758Z-f5d52fac`.
The phase ceiling is 660 GPU-s / 650 CPU-s including preparation, lifecycle,
audit and monitoring; prior closed charge is 201 GPU-s / 1,101 CPU-s, with
900 GPU-s / 1,800 CPU-s overall caps. The only real pilot is the 48-frame smoke;
the zero-frame full marker is not evidence. No third attempt is authorized.

**Spec identity resolved:** fleet hashes sorted JSON using default separators;
the pre-admission `2be95...` used compact separators. Recomputing fleet's
serialization for the exact frozen spec yields the receipt's
`161a958ffff2aba437dd1a32c56255e8dd6002be6554349108133e599dde5de1`.
There was no configuration change.

**Continuation terminal:** `20261103T141758Z-f5d52fac` completed on gpu6 Ada
index 0. The 48-frame command exited 0 in 227.511 s without recorded contention;
its validator reports six persisted component points and verified prior 16, but
independent audit remains required. The zero-frame full marker exited 0 in
0.121 s and is not a result. Creation to terminal settlement was 261.330 s;
conservatively charge 262 GPU-s, including queue time. The saved terminal
status shows no GPU process or PointStream claim; a read-only gpu6 process
inspection found neither supervisor 2767698, child 2767715 nor process group
2767715. New cumulative conservative GPU charge is 463/900 GPU-s. The accepted CPU phase envelope is 312 s (20 preparation + 262 single-core
lifecycle + 20 audit + 10 monitoring); cumulative 1,413/1,800 CPU-s. Aggregate
administrative CPU telemetry remains incomplete, so this is not an exact measurement.

Read-only receipts are in `/private/tmp/pointstream-k5c-continuation-20261103-receipts/`;
closure evidence is in `/private/tmp/pointstream-k5c-continuation-inspection-20261103/closure-archive.txt`.
The smoke archive was copied to `/private/tmp/pointstream-k5c-continuation-20261103-published.tar`;
remote and local SHA256 both equal
`d40a9f7e0a7d2878d01010d4e128467bbdc174f9a899e007a555b9febe256d32`.
The canonical small-artifact exporter rejected the tar extension; the copy used
read-only SSH/SCP without republishing. Terminal event
`4c515c8fe195594f83453bd15f96a428` was acknowledged after the terminal report
was delivered. The [closure ledger](k5c-inputs/continuation-20261103/closure-ledger.json)
records identities, native tools, resource bounds and receipt hashes.
