# Experiments: W5

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md), [W1](w1-roi-pareto.md).

### 2026-10-29 — W5: the W1 test on the Medvedev window (rule)
- Question. W1 found no SVT allocation reaching PointStream's (whole-frame,
  player) quality pair on Al. Does the same hold on Med (Medvedev–Struff
  shot 014 from frame 194, 240 frames at 50 fps, 4.8 s), with every bit
  paid as on Al, and against DCVC-UF too?
- Step 1, the hybrid point. Score Med's actual-base hybrid with the frozen
  real-base recipe of job `20261027T211451Z-f6a754eb` (scale 4 paid base,
  saved generator crops, receiver masks; the Med controls already
  reproduced there), on its 80 scored ticks, whole frame and fixed
  foreground R, LPIPS and DISTS. Rate: base + side terms as on Al, with the
  keypoint packet billed as KPS2 (2193 B) for the primary rate and KPS1
  (3542 B) reported beside it.
- Step 2, the anchors. Plain SVT-AV1 and SVT with +20/+40 QP outside R
  (the working convention, W1) at half and quarter resolution, CRFs chosen
  on rate alone to bracket the hybrid's rate, same judge and controls as
  W1. DCVC-UF points come from W3/W3b (Med triple); if they do not bracket
  the hybrid's rate, the DCVC verdict is unresolved, not extrapolated.
- Decision rule: W1's, per anchor family. A policy dominates the hybrid on
  a metric if whole ≤ hybrid + 0.001 AND foreground ≤ hybrid + 0.001 at
  the hybrid's rate (log-rate interpolation between bracketing points
  only). Outcome A: nothing dominates on LPIPS or DISTS. Outcome B: some
  policy dominates on either. A narrowing on B: report which metric and
  family, and the foreground margin left.
- Hypothesis: as on Al; Med's players are larger in the frame, so the
  player gain is larger and the whole-frame cost smaller.
- Competing explanation: at 50 fps the base costs more per second, the
  side share is smaller, and plain SVT at half resolution dominates.
- Budget: ≤1.5 GPU-hours including smokes and failures; CPU encodes via
  the fleet; every stage ≤45 min.
- Jobs: (none yet)
