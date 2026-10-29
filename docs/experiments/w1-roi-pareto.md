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
- Jobs: (none yet)

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
