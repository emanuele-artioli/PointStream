# Development campaign — 23–30 September 2026

Authorized by decision on 23 September 2026. This file is the plan for the
remaining week. The September 17 roadblock and the 20 September evidence freeze
are historical. Submission is 30 September. Held-out confirmation is skipped.
A second domain runs only after a claimable tennis point exists.

## Claim

Decision metric: weighted PSNR, `0.7 * foreground + 0.3 * background`, same
mask on PointStream and the anchor. Overall, foreground, and background PSNR
are reported beside it and do not decide.

A claimable point has weighted PSNR at least as high as the anchor and no more
bytes than that anchor. Points over the anchor's rate stay in the log and are
not wins. One development clip is enough to claim a regime; it is not a
held-out confirmation.

Runtime is recorded on every row and does not decide that point. A faster
decode with a worse picture is not a win. Once a point ties or beats the
anchor on weighted PSNR at no more bytes, latency is the next sentence:
whether the sender, the client, or both are faster than the baseline.
Sender time includes an offline plate build. Client time is decode plus the
render that arm requires. The two are reported separately.

Shared model weights stay out of the bitstream. The paper must say the tennis
model is pre-shared and can be fetched in downtime. Appearance, motion, and
residuals are charged.

## Instruments already in the tree

- `select_best_background_frame` in `src/components/background/still.py`.
  Background-only MSE, no warp. Ties take the lowest index, so a flat window
  selects frame 0. `still_frame0` stays the control.
- `residual_clip_fraction` in `src/pipeline/residual/lossy.py`. Fraction of
  foreground pixels whose signed error falls outside `[-128, 127]`, the range
  the clipped `uint8` residual can carry. Log this before any training run.

## Calendar

| Day | Work | Stop if |
|---|---|---|
| 23 Sep | Step 1 background encodes on Federer scene 007, 48 frames | A free GPU is unavailable for anything that needs one; encodes themselves are CPU |
| 24 Sep | Pick the background per rate band. Step 2 composite, residual, clip fraction | The background arm alone exceeds the anchor at every QP that still leaves a foreground budget |
| 24 Sep night – 27 Sep | Step 3 training, overnight, one family at a time | Two nights on a family produce no foreground gain over the warp |
| 28 Sep | Step 4 residual of the trained model, rate cap | Clip fraction on the trained residual is still high and another night cannot start |
| 29 Sep | Write the paper numbers if a claimable point exists. Second domain only then | No claimable tennis point: write the negative result, skip the second domain |
| 30 Sep | Submission | |

Do not reopen the WebP-plate ladder, the motion-packing pass, or held-out confirmation.

## Step 1 — Background

Measured. The record is
[the background campaign](20260923-background-campaign.md). Foreground starts
from the band chosen there: Perricard’s inpainted video when the court must
match, Alcaraz scene 000’s panorama when the background may sit 1.5 dB under
the source and leave about 40 kB.

## Step 2 — Foreground base, before training

On the chosen background, composite one AV1 intra QP 42 crop through bbox
motion, and a second arm through COCO-17 keypoints. These numbers already
exist on the unregistered WebP plate; recompute them on the chosen background
so the total is one bitstream.

Then the residual of `source - composite`, split:

- foreground residual, player mask only
- background residual, the complement
- either, both, or neither

Coarsen the foreground residual first, then the background residual, until
total bytes are at or under the anchor. Also keep the residual-off point.
Log `residual_clip_fraction` on the foreground residual before any training
command.

Branches:

- Clip fraction above 0.05: do not spend a finer quantizer on this base. The
  model has to shrink the error. Training still starts.
- Total bytes at or under the anchor and weighted PSNR at or above it: this
  is already a claimable point. Training still starts, to shrink the residual.
- Total bytes over the anchor after both residuals are off: the appearance
  and motion themselves do not fit. Cut appearance to the single crop. If
  that still does not fit, the background choice was wrong; return to Step 1
  and pick the next-cheapest arm.

## Step 3 — Training

The foreground model is the contribution we still have to try. It runs even
when Step 2 already fits, because a closer player makes the residual smaller.

Weights are pre-shared. They are not counted in the bitstream. Inference is
deterministic: fixed seed, no dropout. The server and the client run the same
checkpoint so the residual is the difference between ground truth and that
shared sample.

One family at a time, on one GPU, overnight. Check the machine for a free GPU
before each launch. Order:

1. The family that already has a training entry point and a client decode.
   Start there. Do not open a second family on night one.
2. Conditioning is the one appearance crop plus COCO-17 keypoints, the wire
   Step 2 already charges.
3. Training data is development tennis only. Do not touch a held-out match.

After each night, score foreground PSNR of the generated player pasted on the
Step 1 background, residual off, against the Step 2 warp.

- Gain of at least 2 dB foreground over the warp: stop this family and go to
  Step 4 with that checkpoint.
- Smaller gain, and a night remains before 27 Sep: one more night on the same
  family, not a new grid.
- Two nights without a 2 dB gain: switch once, to the next family that already
  loads in this tree. A third family is out of time.
- No family gains 2 dB by the morning of 28 Sep: stop training. The write-up
  is that current generators are not yet good enough on this task, and the
  claimable point, if any, is the Step 2 composite.

Do not search autoencoders against motion-vector fields in this week. That
comparison is a later campaign.

## Step 4 — Residual of the trained player

Same split and the same rate cap as Step 2. Log the clip fraction again.

- Clip fraction still above 0.05: the checkpoint did not fix saturation.
  Report it. Do not buy a finer residual.
- Clip fraction at or below 0.05, and total bytes at or under the anchor, and
  weighted PSNR at or above the anchor: claimable point. This is the one to
  put in the paper.
- Bytes fit and weighted PSNR does not: coarsen the background residual before
  the foreground residual, because the decision metric weights the player more.
  If the capped point still misses, keep the bytes and the scores, and do not
  call it a win.

## Step 5 — Second domain

Runs only after a claimable tennis point, and only if that point exists on
28 Sep so 29 Sep can hold one extra clip. Same frozen choices: the winning
background arm, the same appearance and motion, the same checkpoint, the same
rate cap, the same weighted metric.

First candidate: one egocentric clip from the demo, because that code and
footage are already in the tree. A second sport replaces it only if the
egocentric clip cannot be decoded with the tennis masks. One clip. No sweep.

If 28 Sep has no claimable tennis point, skip this step.

## Paper

Write numbers only from a claimable point, labeled as a development operating
point on Federer scene 007. The 14.2%–18.3% headroom figure stays the
motivation, measured on eight other scenes. The 65.4% and 32.5% sentences stay
withdrawn. If the week ends without a claimable point, the paper says the
inpainted-video budget and the generator result as measured, including the
negative one.
