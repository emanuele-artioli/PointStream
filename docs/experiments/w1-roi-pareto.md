# Experiments: W1

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W1: PointStream against every SVT bit allocation (rule)
- Question. At equal total rate, does PointStream reach a pair (whole-frame
  quality, player quality) that SVT-AV1 cannot reach by moving bits between
  players and background with a region-of-interest (ROI) quantizer map? This
  is the fair comparison for a codec whose claim is "spend the bits on what
  the viewer watches"; whole-frame BD-rate alone cannot credit it.
- Point under test: the existing audited hybrid on Al (Alcaraz–Hurkacz shot
  031 from frame 129), job `20261027T211451Z-f6a754eb`: total 22.737962 kbps,
  whole LPIPS 0.275787 / DISTS 0.169612, fixed-foreground LPIPS 0.345876 /
  DISTS 0.252551. No new PointStream output is generated in W1.
- Baseline grid (fixed before any quality is seen). SVT-AV1, the same
  native build, preset 4 and 240 native frames as the existing anchors.
  Resolutions: full (1920×1080) and half (960×540). ROI QP offset inside the
  player region R relative to the rest: 0, −20, −40 (the SVT ROI map; R from
  the same frozen source masks, encoder-only, which is legitimate because the
  offsets are signalled in the paid bitstream). For each (resolution,
  offset), three CRFs chosen on rate alone in a CPU preflight so that two
  bracket 22.74 kbps. That is 18 encodes. The existing plain-SVT points stay.
  Reuse `experiments/tennis/parallel_baselines.py`'s ROI path where it fits.
- Scoring: the same 80 ticks (every third frame), same fixed foreground R,
  same LPIPS/DISTS code and weights as the real-base judge
  (`experiments/tennis/k6b_real_base_judge.py`). Controls: the anchor points
  already scored must reproduce within 1e-4.
- Decision rule. For each policy, interpolate whole and foreground quality
  at 22.737962 kbps linearly in log-rate between its two bracketing points;
  a policy without a bracket is unresolved, never extrapolated. A policy
  *dominates* the hybrid on a metric if its whole score ≤ hybrid whole +
  0.001 AND its foreground score ≤ hybrid foreground + 0.001. Outcome A: no
  policy dominates on LPIPS and none on DISTS — PointStream is
  Pareto-better on Al; next, score the Med window's actual-base hybrid
  (never scored) and run the same grid on Med. Outcome B: some policy
  dominates on either metric — the player route does not beat a fair
  allocation even by its own thesis at this rate; report and stop.
  Also report, as the candidate headline number, the best ROI foreground
  score reachable at the hybrid's whole-frame score (interpolated across
  offsets at the same resolution), for each metric.
- Hypothesis: players move fast and are small; SVT cannot bring their LPIPS
  near 0.35 at ~23 kbps without the background collapsing.
- Competing explanation: R covers few blocks, so a −40 offset at half
  resolution buys player quality near 0.35–0.40 for a few kbps and keeps
  the whole frame near the envelope, dominating the hybrid.
- Budget: CPU encodes through the fleet (each stage ≤45 min, split by
  window/resolution); scoring ≤1 GPU-hour including smokes and failures.
- Jobs: see the entries below (masks `20261029T092537Z-88270f98`; encodes
  half `20261029T092919Z-a9e696a8`, full `20261029T093403Z-0a9c7eba`,
  quarter `20261029T094410Z-40e65a2f`). A first masks submission,
  `20261029T091730Z-c2a7bbd0`, was never published (its snapshot transfer to
  gpu2 timed out after the files had landed; it has no spec and cannot run);
  it was inspected, not replayed, and the masks job was submitted again with
  a reduced snapshot.

### 2026-10-29 — W1 implementation, rate-only preflight and substitution rule (before any run)
- Written before any W1 job ran and before any W1 quality number exists.
  The decision rule, grid and budget above are unchanged. Code:
  `experiments/tennis/w1_roi.py` (ROI map, CRF search, rule),
  `w1_masks.py` (fixed foreground S), `w1_encode.py` (CPU encodes),
  `w1_score.py` (GPU scoring), `w1_specs.py` (job specifications).
- ROI sign. The map is 0 outside and the offset (0, −20, −40) inside every
  64×64 block touched by the frozen source R; offset 0 is the plain encode
  with no map. Same blocks as the older comparator, which instead raised the
  background by +32; ours only adds bits inside R, so a CRF at the ceiling
  can still be lowered by the offset.
- CRF range. SvtAv1EncApp 4.2.0 accepts `--crf` 1–70, not 63. The plain
  half-resolution CRF 63 base already costs 33.685 kbps (40,422 bytes), above
  the 22.74 kbps target, so reaching it needs CRF above 63.
- Rate-only preflight. For each (resolution, offset) a search over integer
  CRFs finds adjacent CRFs whose full 240-frame rates straddle 22.737962 kbps,
  plus the neighbour CRF+2 (else CRF−1). Rate is the complete IVF file bytes
  over 9.6 s, as for the existing anchors. The search is the preflight and its
  encodes are the final ones; no quality is read by the encode job. Up to 8
  encodes per policy. A policy that has no point at or below the target at CRF
  70 (or none above it at CRF 1) is unresolved by the rule: never
  extrapolated, never a dominator.
- Substitution (root decision 2026-10-29). If full resolution is unresolved at
  all three offsets, it is replaced by quarter resolution (480×272 coded,
  scale 4; same offsets, same CRF search), where the existing SVT envelope
  sits at this rate. The preflight rates and the substitution are recorded in
  an entry below before anything is scored.
- Controls. The plain CRF 63 bases of scale 8/4/2 are re-encoded (their bytes
  must equal the existing 12,237 / 20,557 / 40,422) and rescored; their
  per-tick whole and foreground LPIPS/DISTS must equal the hybrid job's
  recorded scores within 1e-4 (the hybrid job's published scores, sha256
  `e346844d…`).
- Foreground S is the real-base judge's: R plus the warp-drawn player pixels,
  rebuilt from the saved K6 window, keypoints and source mattes, checked
  against the K6 window's recorded hashes of R and S on every tick.
- Scoring runs on an RTX A6000 (the controls were scored on one); the job is a
  shared-GPU job, so its timings are cost only.

### 2026-10-29 — W1 deviation: full resolution unresolved, quarter substituted (before any scoring)
- Written after the rate-only encode jobs and before any quality number. CPU
  jobs, no GPU time. Rates are complete IVF file kbps over 9.6 s, all 240
  frames, SvtAv1EncApp 4.2.0 preset 4, keyint −1; every stream passed an
  independent dav1d decode. Controls: the plain CRF 63 bases of scale 8/4/2
  re-encoded to exactly 12,237 / 20,557 / 40,422 bytes, as the existing anchors.
- Full resolution (1920×1080), job `20261029T093403Z-0a9c7eba`, gpu3: even at
  CRF 70 the rate stays above the target: offset 0: 34.15, −20: 37.94, −40:
  38.01 kbps (target 22.737962). All three offsets are unresolved by the rule.
  By the root decision (2026-10-29) full is replaced by quarter resolution
  (480×272 coded, scale 4); offsets and search unchanged.
- Half resolution (960×544), job `20261029T092919Z-a9e696a8`, gpu2, all three
  offsets bracket. Offset 0: CRF 64 28.01, 65 24.64, 66 22.28, 67 20.56.
  Offset −20: 66 25.79, 67 24.19, 68 22.47, 69 21.22. Offset −40: 66 25.85,
  67 24.10, 68 22.48, 69 21.09. Selected: 65–67 (offset 0), 67–69 (−20, −40).
- Quarter resolution (480×272), job `20261029T094410Z-40e65a2f`, gpu3, all three
  bracket. Offset 0: CRF 60 26.76, 61 24.11, 62 20.61, 63 17.13. Offset −20:
  60 30.60, 61 27.69, 62 24.26, 63 20.55, 64 18.70, 65 17.31. Offset −40:
  60 30.57, 61 27.80, 62 24.32, 63 20.59, 64 18.66, 65 17.31. Selected: 61–63
  (offset 0), 62–64 (−20, −40).
- Observation recorded before scoring: offsets −20 and −40 cost the same rate
  within 1% at every CRF and both resolutions (e.g. half CRF 67: 24.19 vs
  24.10). The grid is unchanged; either the encoder saturates the ROI offset
  or the two policies are near-duplicates. The scores will show which; if −40
  equals −20 in quality too, the two are one policy and the ROI curve has
  effectively two points.
- Scored set: 3 controls + 3 CRFs × 3 offsets × {half, quarter} = 21 points on
  the same 80 ticks. Masks job `20261029T092537Z-88270f98` (80 ticks rebuilt,
  every R and S hash equals the K6 window's record).

### 2026-10-29 — W1 addition: secondary re-billed rate (written before any score was read)
- Disclosure. The scoring job `20261029T095649Z-43181442` (gpu3, A6000) had
  already finished its smoke and full stages when this entry was written; its
  outputs were not opened. The addition below was fixed from the root
  message and W2's file, not from a W1 quality number.
- Basis. W2 (`claude/w2-pose-rate`, `docs/experiments/w2-pose-rate-2026-10-29.json`,
  code `experiments/tennis/w2_pose_rate.py`) found an exact (round-trip
  identical, zero keypoint error) re-coding of Al's keypoint packet: key 4,
  4 px, linear interpolation, previous-frame prediction, 2,174 bytes =
  1.811667 kbps against the original 4,236 bytes = 3.530000 kbps. The
  generated players are unchanged, so the hybrid's quality is identical at a
  total rate lower by 1.718333 kbps.
- Primary (as frozen): the rule applies at 22.737962 kbps.
- Secondary: the same rule (linear in log-rate between the two bracketing
  points, dominance within 0.001 on both axes, outcome A or B, best
  foreground at the hybrid's whole-frame score) applies at
  22.737962 − (3.530000 − 1.811667) = 21.019629 kbps, with the hybrid's four
  scores unchanged. A policy whose scored CRF points do not bracket 21.019629
  is unresolved there; no new encodes are made for it. The selected CRFs were
  chosen for the primary rate, so some policies will be unresolved at the
  secondary rate by construction.
- Full resolution: no retry is needed; its submission already ran (job
  `20261029T093403Z-0a9c7eba`) and is unresolved at all offsets (above).
