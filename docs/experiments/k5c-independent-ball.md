# K5c — independent image judge for the ball component

## Current decision, 2026-11-03

The first request `20261103T132157Z-f665bece` is closed at its cost-promotion
gate; its 16-frame records passed independent CPU audit. No 48-frame result
exists yet. The exact initial source/spec and cost failure below are preserved.
The recorded independent audit and allocation-to-release reconciliation support
an explicit [operational amendment](../cards/k5c-48-frame-continuation.md) for
the already approved 48-frame target using those accepted controls. This is
not a claim that the initial 420-second gate passed or a new compute allocation.

**Accepted cumulative envelopes:** 201 GPU-s and 1,101 workload CPU-s, leaving
699 seconds of each within the original 900/1,800 limits. The GPU envelope is
the rounded 200.734-second persisted worker lifecycle, corroborated by absent
supervisor/campaign processes and claims; it includes admission time, rather
than claiming exact allocation telemetry. CPU conservatively includes 600
preparation +201 single-core lifecycle +180 audit +120 monitoring. Independent
audit itself used 0.061 CPU-s; control-plane aggregate telemetry remains
incomplete. Raw and accepted receipts are preserved under
[`k5c-inputs/first-attempt-20261103/`](k5c-inputs/first-attempt-20261103/).

The audit recomputed registered frame/label joins, 5/10-pixel judge counts,
physical native/BALL byte ledgers and hashes, framing and RGB-PSNR guards for
all six settings. Source WASB found 16/16 visible labels within 10 pixels;
rendered WASB found 16/16 at each setting versus baseline counts 0,1,7 (DCVC)
and 8,3,2 (SVT). All 16 labels are positive: negative-frame false positives
remain unmeasured. This is a development prefix, not an independent statistical
confirmation or complete-codec win. Label/source identities were checked
against recorded hashes, not all external original images reread in the audit.

**Before the next outcome:** retain the same one-window hypothesis, competing
explanation, six settings, source qualification and physical-byte/PSNR checks.
The next test is the original 48-frame development pilot, nested over the 16
controls; do not pool them as 64 independent frames. The new phase is bounded
at 660 GPU-s /650 CPU-s, so cumulative envelopes remain at most 861/900 GPU-s
and 1,751/1,800 CPU-s, reserving 39/49 for uncertainty. No third attempt, training,
W8 retry, new model or larger study is admitted. Identity, source, device,
packet, frame or runtime failure stops it. The retained 577.91-second complete
forecast fits the new phase ceiling; the real-work forecast 517.91 seconds fits
the 535-second command. This explicit reallocation corrects the insufficient
internal stage cap without changing user-approved totals or quality rules.

Frozen continuation spec:
[`k5c-inputs/continuation-48-spec-20261103.json`](k5c-inputs/continuation-48-spec-20261103.json).
The fleet `smoke` stage measures the actual 48-frame pilot; the `full` stage
writes only a zero-frame non-result marker. Only the actual measured records
may support a paper result. Controller review and source freeze precede dispatch;
record the new exact job ID here before one bounded fleet wait. No continuation
submission has occurred at this checkpoint.

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
