# Experiments: motion admission amendment

**Current status:** the user explicitly approved this amendment within the
original aggregate budget. [Live execution and cumulative accounting](player-motion-amendment-20261117/release.json)
include the initial stopped smoke. Both amended44/240-frame requests completed
and passed independent integrity checks; Player2's accepted publication and
actual accounting admitted Player1. Both jobs and GPU claims are closed. The
prepared proposal below is retained as its original
permission request, not a second budget envelope.

## Approved execution — November 17

The user's explicit reply was **"Approve within original budget"**. No training,
new source, scoring stage or further automatic retry is released.

| Request | Recorded state | Evidence |
|---|---|---|
| Player2 `20261117T094919Z-5a0cc722`, gpu6 | Complete:44-frame smoke and one contiguous240-frame run; independent integrity PASS | [Audit](player-motion-amendment-20261117/player-2-audit.json), [native/publication identities](player-motion-amendment-20261117/player-2-publication-evidence.json) |
| Player1 `20261117T102252Z-ccb04da2`, gpu5 | Complete:44-frame smoke and one contiguous240-frame run; independent integrity PASS | [Audit](player-motion-amendment-20261117/player-1-audit.json), [native/publication identities](player-motion-amendment-20261117/player-1-publication-evidence.json) |

Player2's repeated smoke forecast1,686.45s, below the amended1,800s full cap.
The full run retained240frames, with932.75s measured render and12,538.76MiB
peak GPU memory. Shared-GPU timing supports cost admission, not throughput.
Its conservative lifecycle charge is1,341GPU-s plus1,341CPU-s and the entire
750externalCPU-s reservation. Including the original stopped run, this is
**1,669GPU-s /3,919CPU-s**, leaving2,531GPU-s /3,281CPU-s before Player1.
That covers Player1's whole2,400GPU-s /3,150CPU-s reservation with131s margin
in each aggregate. [Teardown/accounting](player-motion-amendment-20261117/player-2-teardown-accounting.json)
records absent task processes and GPU claims; both publications are preserved
read-only and the delivered terminal event is acknowledged.

Player1's repeated smoke forecast1,710.79s, also under1,800s. Its full
publication retained240frames. A management-host RPC timeout interrupted the
first read-only wait; inspection confirmed the same live job and passed gate,
and one bounded recovery wait observed completion. No computation was retried.
The top-level status update remained older than the remote terminal timestamp;
accounting therefore uses request creation through observed absent task
processes/group/GPU claims, an intentionally conservative1,649s lifecycle bound.
The [closure record](player-motion-amendment-20261117/player-1-teardown-accounting.json)
records this basis and the entire750externalCPU-s reservation. Total charges
are **3,318GPU-s /6,318CPU-s (55.3GPU-min /105.3CPU-min)** within4,200/7,200.
The remaining882s in each envelope is unspent and not transferred to another
experiment. Both terminal events are acknowledged and all four amended
publication archives are retained read-only.

Integrity acceptance is not visual acceptance. The full human review uses
all240native ticks (4.8s at50Hz), with explicitly slowed playback if useful.
The user's [Player2 reply](player-motion-amendment-20261117/player-2-human-review.json)
is **"Acceptable"**; Player1's review remains pending. No detailed failure notes
were supplied. This tests source-aligned body/clothing, pose and stability
before compositing and supplies no matched-rate codec, independent-source or
combined-ball result. Preserve this author development judgment separately
from source-only annotations and any future population preference study.

### Next contract, conditional on the viewing result

Do not relabel the old W8 receiver as this candidate. A fresh source-code
check finds `w8_payload.unpack_window` fixes match5/CFG2, required match state
and every-third saved ticks; `w8_receiver_generate` also validates the fitted
guider and match checkpoint. This pilot instead uses the installed common
tennis model, CFG3.5 and every native tick. Reuse its isolation and transport
primitives through an explicit new profile, with paid references/geometry,
receiver-generated silhouettes, a native full-frame fallback and actual
every-frame output. Preserve the old format and evidence unchanged.

If the human motion review supports continuing, first cost that minimal
player receiver and matched player-only comparison. The same-source ball
stream and source-only player/ball references still need authentication before
a combined claim. Avoid another model grid. A failed viewing result instead
requires identifying the conditioning/context failure before a narrow repair;
neither new generation nor training follows automatically from this budget.
The [unreleased receiver design](player-common-receiver-design-20261117.md)
identifies reusable functions and the remaining input/cost checks; it is not
a compute proposal or an additional permission gate for local preparation.

## Historical proposal — prepared after the original cost gate stopped promotion

**Not authorized. No further GPU work.** This proposes an explicit change to
the [approved original stopping rule](player-motion-20261116.md), within its
unchanged **4,200 GPU-second / 7,200 CPU-second overall envelope**. The
[machine-readable proposal](player-motion-amendment-20261117/proposal.json)
retains the prior charge and conditional admission rule.

## Observed stop and surviving evidence

Player 2 job `20261117T083755Z-de05930d`, source revision `040500fc`, generated
44 consecutive frames with the real shared-tennis renderer. Its immutable
32,706,560-byte publication has SHA256
`f3e0e6dea0085ac3787e8b52a0a11bcb2c8763971818b0426d8adec1a646a1d4`.
Native and [independent frozen validation](player-motion-20261117/player-2-smoke-audit.json)
pass source/config/checkpoint, ticks/boxes, every crop, CUDA/kernel and memory
checks. Peak GPU memory was 11,295.73 MiB under the 18,000 MiB cap.

Actual worker wall was 255.5483 s, including 22.9334 s model load; the render
was 219.6516 s. The frozen cost formula yields **1,704.68 s (28.41 minutes)**,
above the full-stage **1,500 s (25 minutes)** cap. The validator therefore
correctly rejected promotion. No 240-frame output exists; Player 1 was not
admitted. Shared timing supports budget admission, not a throughput claim.

All task PIDs/groups and GPU claims are gone, and both terminal events are
acknowledged. Conservative accounting charges **328 GPU-s and 1,828 CPU-s**:
request creation through terminal worker reconciliation, plus the entire
1,500 external CPU-second reservation. Unspent amounts are **3,872 GPU-s /
5,372 CPU-s**; they have not been reallocated. The [closed release ledger](player-motion-20261117/release.json)
and teardown record retain the accounting basis.

The 44-frame source-aligned display is **smoke only**, 0.88 native seconds;
10 fps playback is explicitly slowed to 4.4 s. It supplies no longer-motion,
rate–distortion, stock-comparison or complete decoder result. Raw NPZ/RGB
remain separate from display proxies. Fresh LPIPS/DISTS remain unscored.

## Proposed change, without changing the candidate

Keep the exact source/units, weights, installed common checkpoint, references,
poses/boxes, seed rule, 512x784, 30 steps, CFG3.5 and context24/overlap4.
Each 240-frame sequence is still generated in one call. No training, guidance
sweep, new source or method tuning is proposed.

Make admission policy explicit. Missing `motion_admission_policy` retains
`original-1500-v1` and the original 450/1500/2100/60-second limits. The proposed
`extended-1800-v1` changes only full-stage and request limits:

| Limit | Original | Proposed amendment |
|---|---:|---:|
| Real 44-frame smoke | 450 s | 450 s |
| Full 240-frame stage | 1,500 s | 1,800 s |
| Validator | 60 s | 60 s |
| Whole request | 2,100 s | 2,400 s (40 min) |
| External CPU per new request | 1,500 s | 750 s |

The multiplier 1.25, 90 s startup/publication reserve, finite-clock checks and
60 s close reserve are unchanged. A single helper binds the policy to both
native specification and validator. Unknown policies fail; selecting a policy
does not authorize computation. The observed timing passes the proposed cap
only when enough native campaign time remains. A new real smoke is required;
the failed job and its original gate will not be relabeled or resumed.

The retained local fork suggests 3 context windows at 44 frames and 12 at
240, rather than linear frame-count scaling. Actual packed scheduler identity,
all non-window costs and full memory behavior were not independently qualified.
This is not a reason to replace the measured forecast or promise completion.
The amendment retains that conservative formula and changes an explicit cap.

## Aggregate budget and conditional second request

Initially admit **only Player 2**, after explicit amendment approval and fresh
normal fleet/source/config gates. Its new maxima are 2,400 GPU-s and 3,150
CPU-s (one-core lifecycle plus 750 external CPU-s), within the unspent envelope.

Player 1 is **not reserved or guaranteed**. Admit it only after Player 2's
accepted full publication and actual conservative accounting, when the remaining
aggregate envelope still covers **2,400 GPU-s and 3,150 CPU-s**. Both original
and amendment charges count toward 4,200/7,200. Otherwise stop with the completed
result. Never add both new maximum reservations as if the initial smoke were free.

Any further failed correctness/cost gate, ambiguous submission or exhausted
budget stops this continuation. No automatic retry or hidden cap reduction.
Full accepted output still goes to a source-aligned human viewing decision;
source-spread, complete paid player-and-ball decoding, matched ROI/DCVC curves
and confirmation remain subsequent work.

## Why approval is required

The original approved card says to stop on failed cost gates and forbids an
automatic retry. This amendment changes per-request limits and authorizes a
repeated smoke. It therefore needs an explicit new instruction even though
the overall GPU/CPU envelope is unchanged. No amended request has been submitted.
