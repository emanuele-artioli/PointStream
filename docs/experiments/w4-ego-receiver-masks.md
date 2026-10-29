# Experiments: W4

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W4: egocentric with masks found at the receiver (rule)
- Question. In the egocentric pipeline (H5), with the foreground mask F
  free, PointStream won weighted DISTS by 15% against the venue-adapted
  DCVC (H5q); billing F removed every win (H5p). H5p 4d(b) tracked F at the
  decoder with SAM 3.1 but then paid an exact correction stream (13–61
  kbps). Composites made with the *uncorrected* receiver masks were never
  scored. Does a zero-correction receiver mask keep the DISTS win?
- Setting: the four pilot items (P01_107, P02_02, P03_10, P09_106), H5p
  4a's foreground points and pairs, the `dcvc-venue` anchor at equal
  adaptation, the H5q/H5p metric code. Receiver mask: the pinned SAM 3.1
  tracker on the decoded foreground stream, initialized with the first
  frame's s=8 union mask coded intra (billed), no later prompts, no
  corrections. If H5p 4d(b)'s predicted masks were saved, reuse them and
  only composite and score.
- Decision rule (H5p's): weighted DISTS mean ≤−10% against `dcvc-venue`
  and ≥3 of 4 items below zero, with the seed mask billed. Report LPIPS and
  PSNR beside it (LPIPS lost by 6% even with F free). Report each item's
  IoU with the exact F.
- Hypothesis: DISTS tolerates small boundary errors; foreground and
  background agree near the boundary.
- Competing explanation: missed handled objects and mask leaks put
  background pixels on the hands, and DISTS loses.
- Budget: ≤1.5 GPU-hours (SAM on A6000 or Ada), every stage ≤45 min.
- Jobs: listed in the entries below (code `bfbe1d70` and later; Ada gpu5/gpu6).

### 2026-10-29 — W4 first job failed on its time cap (no evidence yet)
- What was saved. H5p 4d(b) kept the tracker's masks for one foreground
  point per item only (`dcvc@0/4`; P09_106 `dcvc@42/2`), not for the 30
  points of 4a's hull pairs. Their IoU against the exact F is 0.82 | 0.72 |
  0.38 | 0.65 on the reduced/upsampled grid (4d(b)'s audit, whole clip).
  Re-tracking every point costs about 30 × 90 s on top of 54 scored pairs,
  more than the 1.5 GPU-h. So the scoring job takes each item's saved mask
  and applies it to all of that item's hull pairs (at the saved point it is
  exactly the pinned configuration). Design call, flagged for root: the
  mask comes from the lowest-rate decode and is likely worse than a tracker
  run on a higher-rate point would give, so a pass is safe, a fail is not
  proof for the higher-rate points. The seed (90–112 bytes plus two
  configuration bytes) is the only mask byte billed, in the MASK segment.
- Code (`bfbe1d70`): `h5 score --receiver-masks/--receiver-seed`,
  `w4_receiver.py`, `w4_specs.py` (specs from 4a's, same pairs), and
  `w4_report.py` (BD-rate against `dcvc-venue`, rule verdict); tests in
  `tests/test_w4.py`.
- Job P03_10, gho coder, 7 pairs: `20261029T090812Z-2488daa0`, gpu6 RTX
  6000 Ada, code `bfbe1d70` (ready.json git_head confirmed). Smoke passed
  (24 frames, 2 pairs, 143 s, validator 10/10, seed equals the first union
  mask). The full stage was stopped by its own 550 s cap ("command exceeded
  its 550 s allowance"; the monitor's stage label "attention" is just the
  supervisor's attention flag, not a code stage) after 2 of 7 pairs. No
  score from it is cited.
- Where the time went (two finished pairs, seconds): decode_files 145 and
  117, DISTS 16 and 15, B2 harness 6.5 and 4.1, composite 2.6 and 2.7. Only
  decode_files (FFV1 reads of the background and foreground files and their
  RGB conversion) is slower than in 4a on the same GPU class (4a P03 gho on
  Ada: 6–13 s a pair, 40 s a pair in all); everything else matches. The
  inputs were local cache hits, so it is not staging. The supervisor
  flagged `timing_contaminated`; gpu6's other GPU carried another user's
  100% training job and `/local` is 98% full. The cause is not proven: it
  is host contention, not the receiver-mask code (which only adds an alpha
  built once per job).
- Charged: 755.4 s of GPU (stage started to ended) plus 3.3 s salvage =
  0.21 GPU-h of 1.5; remaining 4641 s (1.29 GPU-h). No other job submitted.
- Fix proposed (not yet run, waits for root). (1) Caps from measured cost:
  full = 300 + 200 s × pairs (P03 gho 1700 s), smoke unchanged (1 pair,
  24 frames; the 2-pair smoke took 143 s, projected under 120 s with one
  pair, far inside 600 s). (2) Resubmit P03 gho first on gpu5 alone and read
  the first pair's decode_files: if at most 30 s a pair, run the other seven
  jobs at 4a's cost (~45 s a pair, ~3000 s in all, inside the remaining
  budget); if it is again 100 s or more, cancel that job (own job, cost
  under 400 s) and thin to a fixed three or four hull pairs per item spread
  over the rate range (about 14 pairs, ~2500 s at 175 s a pair), reporting
  4a's F̂ composites on the same thinned pairs beside them so the comparison
  stays like for like. The decision is made before any receiver-mask score
  is seen.

### 2026-10-29 — W4 plan after the failed job (written before any resubmission)
- Scope of the claim (root, accepted). Each item's saved 4d(b) mask, tracked
  on that item's lowest-rate decode (P09_106: `dcvc@42/2`), is applied to all
  of its hull pairs. This is a conservative construction: a **pass supports**
  the claim that zero-correction receiver masks keep the DISTS win; a **fail
  does not reject receiver masks in general**, because a real receiver would
  track on its own decode of each point, which is better than the lowest-rate
  one. The seed and the two configuration bytes are the only mask bytes
  billed.
- Host notes. gpu6's `/local` was 98% full (6.3 of 6.9 TB, 193 GB free) and
  another user's training job held one of its GPUs at 100% during the failed
  job; nothing there is touched or deleted. gpu5's worker heartbeat was 835 s
  old (stale) when first checked and gpu6's 237 s old a few minutes later (the
  submit gate refuses a stale host); jobs go to whichever Ada host is fresh at
  submission, gpu5 at the first resubmission (`HOSTS` in `w4_specs.py`).
- Caps: full stage 300 s + 200 s per pair; smoke 24 frames, 2 pairs (the
  failed job's smoke took 143 s). W4 total stays within 1.5 GPU-h; 758.7 s
  are charged, 4641 s remain, and nothing is started that cannot finish
  inside them.
- Switch rule (fixed now). Resubmit P03_10 gho with its full 5 pairs (cap 1300 s; the first entry's
  "7 pairs" was the venue job's count). Read
  `decode_files` of its first finished pair from the job's checkpoint rows.
  - at most 30 s: run the other seven full-set jobs (all 54 hull pairs);
  - more than 30 s (including 100 s or more): cancel this job at once (own
    job only) and run the thinned set below.
- Thinned set (`w4_report.thin`, rate position only, no score looked at).
  Per item, the weighted-DISTS hull pairs sorted by their billed rate in 4a
  (the F-hat composites, mask bytes included), the pairs at ranks
  round(k (n - 1) / 3) for k = 0..3. That gives four pairs per item:
  P01_107: `enh-1080-63+dcvc-42_2` (gho), `scene-1080-63+dcvc-63_1` (venue),
  `scene-540-0+dcvc-0_4` (venue), `scene-540-13+dcvc-21_2` (gho).
  P02_02: `enh-1080-32+dcvc-63_1`, `enh-1080-59+dcvc-63_1`,
  `scene-540-0+dcvc-0_4` (all venue), `scene-540-25+dcvc-42_4` (gho).
  P03_10: `enh-1080-63+dcvc-42_4` (gho), `scene-1080-63+dcvc-63_1` (venue),
  `scene-540-0+dcvc-0_4` (gho), `scene-540-25+dcvc-42_4` (gho).
  P09_106: `enh-1080-32+dcvc-63_1`, `enh-1080-59+dcvc-63_1`,
  `scene-540-0+dcvc-0_4` (all venue), `scene-540-13+dcvc-42_2` (gho).
  The matching 4a F-hat composites (billed, and with the mask bytes removed)
  are taken from the same four pairs by `w4_report.py --thin`, and the
  verdict is judged on the thinned receiver curve with the same rule
  (weighted DISTS mean <= -10%, >= 3 of 4 items below zero, against
  `dcvc-venue`). A thinned curve is a coarser hull than the full one and is
  reported as such; if a thinned item has no BD-rate (no overlap), it is
  reported as better or worse at every rate and counts as not passing for
  the mean.
- Waits are handed to haiku-runner (`ps-fleet wait`, at most 45 minutes).

