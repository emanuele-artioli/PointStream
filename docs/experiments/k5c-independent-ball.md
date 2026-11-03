# K5c — independent image judge for the ball component

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
