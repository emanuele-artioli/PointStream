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

### 2026-10-29 — W5 implementation notes (written before any run; no deviation from the rule above)
- Step 1 reuses the frozen judge unchanged except for one switch, `--no-early-stop`
  (default off, so every earlier invocation behaves as before): job `20261027T211451Z-f6a754eb`
  stopped after Al by the registered arithmetic stop, so Med was never scored. The same spec
  (inputs, hosts reduced to gpu3, RTX A6000, `separated-v1` cost model) is re-run with the switch;
  it scores both windows, so Al doubles as a regression check against the recorded Al means.
  Spec: `docs/experiments/w5-specs/judge.json`.
- Med's hybrid rate = the scale 4 plain base file (20484 B, 34.140 kbps over 4.8 s) + side terms,
  with the keypoint packet as KPS1 (3542 B, 5.903 kbps) in the judge's `rates`; the KPS2 rate
  (primary) subtracts (3542 - 2193) B * 8 / 4.8 s = 2.2483 kbps. Same quality for both (pixels
  unchanged).
- Step 2 reuses W1's code parametrised by `--window med` (`w1_roi.Window`; Al is the default and
  its constants are unchanged): masks S (`w1_masks`), CRF search and encodes (`w1_encode`),
  scoring (`w1_score`). The CRF search brackets the KPS1 rate (the higher one); the usual three
  selected CRFs (bracketing pair + the next-lower-rate neighbour) then also bracket the KPS2
  rate, exactly as W1 did for Al's re-billed rate. First probes: quarter 60, half 70 (half-
  resolution plain CRF 63 is already 80.3 kbps, above the hybrid; if CRF 70 is still above the
  target the half-resolution family is unresolved there, never extrapolated).
- Grid: plain (offset 0, inside = no ROI) and outside-R +20/+40 (W1's convention honoured by
  SVT-AV1 4.2.0 CRF mode; the negative-inside convention was shown inert) at half and quarter
  resolution; the three plain CRF 63 bases of scale 8/4/2 are re-encoded and must match the
  base manifest's bytes (11734/20484/48152) and re-score within 1e-4 of the step 1 archive.
- Rule application: primary rate KPS2, KPS1 beside; hybrid quality from step 1's `quality.json`
  (scale 4 method means). DCVC: judged on W3b's Med triple if present; otherwise pending.
- Jobs (recorded at submission, 2026-10-29): step 1 judge `20261029T192002Z-9f6a6e45` (gpu3, A6000, ready.json git_head `ee3623a4`; client publish timed out, status pending afterwards); masks S `20261029T192450Z-bc8195a6` (CPU, started on gpu2; git_head from ready.json, to be confirmed). Encode and scoring job IDs follow below.
