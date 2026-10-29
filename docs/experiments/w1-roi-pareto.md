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
