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
- Jobs: (none yet)
