# Experiments: K3b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-17 — K3b: the best motion a keypoint warp could give (flow-warped reference)
- Question (user, 2026-10-17: are hands conclusively a bad keypoint
  foreground?). K3's bound gave the reference the hand's *true pixels*,
  so it granted appearance as well as motion; at a keypoint rate of
  about 1 kbps (H2 records 1.01 kbps at +11% 2D error on HOT3D: 10 Hz
  interpolated, PCA 12, steps ×8, 120 ms) it passed only barely (−10%,
  3 of 4). A keypoint warp can only move the previous decoded hand. This
  stage measures that motion's value with the keypoints free: if it does
  not pay unbilled, no keypoint rate makes keypoint-conditioned coding pay
  on hands.
- Arm F (`k3_ref`): as B (reset at every NAL to the previous decoded frame
  x̂(t₀−1)), but inside M = R(t₀−1) ∪ R(t₀) the reference is x̂(t₀−1)
  warped to t₀ by DIS optical flow computed on the *source* frames t₀ and
  t₀−1 (H3a's backward flow and bilinear warp, `h3.dis_flow`/`h3.warp`:
  medium preset, luma, on the two regions' box grown by 64 px; at half
  resolution for full-resolution points as H1/H3, at the coded resolution
  otherwise). Why it bounds a keypoint warp: the flow is estimated from the
  true frames at every pixel (fingers, forearm, the uncovered background),
  while a MANO warp follows the coded pose's mesh, which misses or
  overshoots about a tenth of the hand (H3b) and has no forearm or
  background motion. It is not a strict bound (DIS errs at occlusions and
  blur), so the reading names that limit; it grants no appearance.
  Arms A (calibration and anchor) and B (control) are recoded beside it.
- Items, points, quality, rates, comparison: K3's (4 pilot items, H3c's
  nine points, LPIPS_R deciding, per-NAL calibration, hull against hull).
- Decision rule (`k3.FLOW_DECISION`, fixed before the run). Gate: F
  *unbilled* against A on LPIPS_R, by `h3d.arm_verdict`.
  - Fails (mean ≥ 0 or ≤ 1 item below 0): hands are closed for keypoint
    conditioning at every keypoint rate; PointStream moves to tennis
    players as keypoints (user, 2026-10-17).
  - Passes (mean ≤ −10%, ≥ 3 of 4 below 0, none worse at every rate): then
    F billed at H2's 1.01 kbps point is checked by the same verdict; if it
    also passes, the MANO warp at that point is the next stage (its own
    rule), else hands close with the keypoint rate named as the obstacle.
  - Unclear: to the user.
  - Beside: F against B (the warp's value on a fixed path), F billed at
    1.01 and 1.93 kbps against A, and K3's C (from K3's jobs) against F
    (what appearance adds beyond motion).
- Hypothesis. Motion of decoded pixels is worth much less than true
  pixels: F against B about −15% (K3's C against B was −48%), and since
  the reset costs 39%, F unbilled against A about +15 to +25%: fails.
- Competing explanations. (1) DIS flow is poor on fast, blurred hands, so
  F understates motion: the flow's warp error is reported (mean absolute
  luma error inside R(t₀) of the warped source t₀−1 against the source t₀,
  against the unwarped one). (2) The reset's cost hides a gain a trained
  coder could keep (K3's reading): F against B shows it.
- Budget. Ada or A6000, shared. Smoke ≤ 600 s; pilot: 4 items × 2 jobs
  (arms A, B, F), each ≤ 45 min. Ceiling 2 GPU-hours.
- Jobs (shared; inputs, environment and points as K3; code per
  `ready.json`): at `5db475b`, P01_107 full resolution
  `20261017T154233Z-acae64e8` and ½ + ¼ `…154244Z-44785c5c`, P03_10 full
  resolution `…154318Z-fe06e743` (RTX A6000), P09_106 full resolution
  `…154339Z-2dfbdd07`; at `22e7fdf`, P02_02 `…155753Z-288d6bb2` and
  `…155802Z-73432d9d`, P03_10 ½ + ¼ `…155828Z-600d3fda`, P09_106 ½ + ¼
  `…155850Z-194b2c1a` (all RTX 6000 Ada but P03_10's full resolution).
  `22e7fdf` changes only the validator: P02_02's first submission
  (`…154253Z-98d49b18`) failed its smoke on the check that the flow warp
  lowers its own error, because in the smoke's 48 frames the hand stands
  still (0.03 luma levels between frames, resampling noise 0.0005 above
  it); the check now allows a quarter luma level. Its sibling and the
  three jobs that had not started were cancelled and resubmitted; one
  duplicate was cancelled before it ran. Every full stage passed the
  validator run locally. About 1.7 GPU-hours (smokes 0.27, full stages
  1.45), within the 2 allowed. Report (`k3 report`, code `22e7fdf`, run
  locally, K3's jobs passed as `--k3`):
  `pointstream-data/visor/k3-2026-10-16/report-k3b-22e7fdf/k3b-report.json`
  (`13436277…bbcc`).
- Outcome (pilot, 4 items). **K3b fails: hands are closed for
  keypoint-conditioned coding at every keypoint rate.** BD-rate on LPIPS_R
  (hull against hull; mean, 95% bootstrap CI, items below 0; per item in
  the order P01_107, P02_02, P03_10, P09_106):

  | test vs anchor | LPIPS_R | items < 0 | per item | DISTS_R per item | PSNR_R per item |
  |---|---|---|---|---|---|
  | **F unbilled vs A (gate)** | **+44% (+29 to +65)** | 0 | +76, +32, +40, +28% | +47, +11, +8, +19% | +63, +42, +43, +28% |
  | F vs B (the warp on a fixed path) | +4% (+1 to +8) | 0 | +10, +1, +2, +5% | +7, 0, −4, +7% | +13, +1, +3, +6% |
  | F billed at 1.01 kbps vs A | +85% (+67 to +97) | 0 | | | |
  | F billed at 1.93 kbps vs A | +118% (+94 to +137) | 0 | | | |
  | K3's C (true hand pixels) vs F | −50% (−66 to −32) | 4 | −71, −40, −60, −29% | | |

  - The flow is accurate: inside R(t₀) it brings the source's previous
    frame from 7.5 to 1.9 luma levels of the next one (mean over the
    pilot's F points). Competing explanation (1), a poor flow, is
    refuted.
  - Even with the keypoints free and the motion this good, the
    flow-warped reference codes R 4% *worse* than the unwarped previous
    frame on the same reset path (B), and 44% worse than the unmodified
    codec. Competing explanation (2), that the reset hides a gain a
    trained coder could keep, is refuted as far as motion goes: on the
    fixed path, motion gives nothing to keep.
- Reading. All of K3's gain was appearance, none of it motion: the true
  hand's pixels save 50% against the flow-warped hand at the same place.
  DCVC-UF already finds the hand's motion on its own (its 8-frame
  conditional coding is motion-aware without a motion stream); what it
  cannot predict is how the hand looks in the new frame, which no warp of
  the previous decoded hand supplies. A slightly worse result than no
  warp fits that: the warp resamples decoded pixels (blur) and moves the
  boundary, which the model would otherwise have predicted itself. This
  holds for every keypoint rate, since F bills nothing; the hypothesis
  (F vs B about −15%) was wrong in the same direction as the conclusion.
  Hands stay pixels (G: DCVC-UF + the generic enhancer); keypoints
  remain a description of the hand (H3c2: the pose for 69% less rate),
  not a coding aid for its pixels. Next (user, 2026-10-17): tennis
  players as keypoints.
- Scope. Four pilot items, H3c's nine points, LPIPS deciding; DCVC-UF
  HT-L's published weights with one reference per 8-frame group;
  DIS optical flow on the source as the stand-in for the best keypoint
  motion. A conditional coder trained with a motion input beside its
  memory is not measured, but F against B gives no gain for it to learn
  from motion; appearance would need sending.
