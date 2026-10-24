# Experiments: K1c

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-21 — K1c: K6's bound on egocentric hands (outline granted)
- Question ([PLAN K1c](../../PLAN.md)). H3b's hand bound drew MANO's mesh at
  the coded pose and charged MANO's outline (it misses 9.7% of the hand and
  overshoots 9.2%). K6's `warp` grants the outline: the frame's own pixels
  moved from the uncoded to the coded keypoints by a 2D skeleton warp, so
  only the coding error is charged. On bodies that was the difference
  between a loss and a 95% saving. Does the same bound beat the pixel
  coders on hands, at H2's coded keypoint rate, against H3d's envelope?
- Items. H3b's pilot: B2's four windows (`P01_107_0000003049`,
  `P02_02_0000006080`, `P03_10_0000000616`, `P09_106_0000006303`), 240
  frames each, with H3c2's staged inputs unchanged (window decode, fill
  masks and record, H2's VISOR poses and `code.json`, MANO, LPIPS and
  DISTS backbones; `h3c-2026-10-11/inputs.json` `da49a912…7a10`).
- Keypoints. H2's WiLoR rows, each track coded with H2's 100 ms choice
  (Gaussian 33 ms, 15 Hz interpolated, full pose, steps ×8; 1.93 kbps
  pooled, 1.927 in K1) and its median shape, decoded, and the 21 joints
  projected (`h3b.item_poses`, `h2.pinhole`, as `h3c.item_judge`). Source
  keypoints: the same row's uncoded fit (its own shape) projected the same
  way, so the warp at the uncoded keypoints is the identity.
- Warp. `k6.draw` with a hand skeleton instead of K4's body: the 20 bones
  of the 21-joint OpenPose/MANO hand (wrist to each finger's base, then
  along each finger), every joint confident. Per fitted hand (one H2 row
  per side and frame), the region moved is that side's fill-set class mask
  (hand and forearm, as R), σ = 0.08 of that mask's height (K6's constant,
  not tuned), the field on K6's 4 px grid. Hands are drawn farther first
  (mean coded joint depth). As H3b, drawn pixels never cover objects (the
  fill set's other foreground), and inside R the source is first replaced
  by K6's Telea fill (radius 5), so a hand with no fit in that frame, and
  every pixel of R the warp misses, is charged as fill. Variants:
  - `warp` (the bound): the uncoded to the coded keypoints.
  - `warp@uncoded` (diagnostic): the identity on fitted hands; what remains
    is the charge for unfitted hands.
- Judges. LPIPS on R exactly as H3c and H3d scored their points (`h3c.Scorer`:
  the variant inside R, the source outside; B2's AlexNet spatial map on the
  full frame averaged over R; mean over the window's frames with R
  non-empty), so the bound and the envelope share region, frames and
  network. **Decides.** Beside: DISTS on R (H3c's), PSNR on R, LPIPS over
  S = R ∪ every drawn pixel with the drawing kept outside R (overshoot
  charged, as K6), the share of R the warp misses and the share of its
  drawing outside R.
- Envelope. Per item, the upper concave hull (`g5.envelope`, linear in rate)
  of SVT-AV1's eleven points and DCVC-UF off the shelf (H3d's bar) together
  with DCVC-UF + G's six points (H3d's component, arm `G-L`), all on R, from
  H3d's component report (`h3d-2026-10-12/report-component/h3d-report.json`,
  `467be69f…6e00`); K1's `joint` hull. At 1.927 kbps it reads LPIPS 0.258
  (P01_107), 0.204 (P02_02), 0.155 (P03_10) and 0.184 (P09_106).
- Decision rule (`k1c.DECISION`, fixed before any run). Per item, the bound
  passes when its LPIPS on R is below the hull's at 1.927 kbps.
  - If it passes on ≥ 3 of 4 items, hands get K6b's ladder in PLAN
    (audit, zero-shot, overfit, match fit), as bodies do.
  - Otherwise hands are closed for every renderer at the keypoint stream's
    rate: no renderer can draw more of the true frame at the coded pose.
  - If LPIPS over S (overshoot charged) moves the count of passing items
    across 3, or DISTS on R disagrees with LPIPS on that count, the outcome
    goes to the user.
  - The hull's rate for the bound's quality (and the saving) is reported
    per item, as K1.
- Hypothesis. The bound passes on ≥ 3 items. H3b's `wilor:arm_uncoded` (MANO
  at the uncoded pose) was 0.168 on S and already above the hull on 2
  items; granting the outline removes MANO's 10% miss and spill, and the
  coding error (about 10 px on WiLoR's joints) costs less than that.
- Competing explanation. Hands are not bodies: fingers are thin and close,
  a 10 px joint error is a large share of a finger, and the warp blends
  neighbouring fingers' bones, so the coding error alone may cost more
  LPIPS than a ¼-scale DCVC-UF + G. `warp@uncoded` separates the coding
  error from the unfitted-hand charge; the missed share separates both
  from the outline.
- Budget. One fleet job (`k1c run`), shared Ada or A6000 (LPIPS and DISTS
  verified there): smoke ≤ 600 s (P01_107, first 48 frames), full ≤ 45 min
  (the four items; the warp on the CPU, two scorings per item). Ceiling
  1 GPU-hour, smoke included. The comparison (`k1c report`) is local, from
  the validated stage and H3d's report.
- Job `20261021T184040Z-70821cd0` (gpu5, RTX 6000 Ada, shared; `ab8c205`
  from `ready.json`; environment `pointstream-20261009T214002Z`; inputs
  H3c2's, `da49a912…7a10`). Smoke passed the fleet validator. The full
  stage (4 items, 240 frames) took 1,182 s and passes all 47 of
  `k1c validate`'s checks locally. Among them: the warp at the uncoded
  joints reproduces the source on every fitted hand, LPIPS and DISTS of the
  source against itself are 0, and H2's units hash-check. Rate: 1.927 kbps
  (WiLoR, `gauss:33.3|15Hz-linear|full|q8|previous`). Report
  `visor/k1c-2026-10-21/report/k1c-report.json` (`085bb948…ef90`, `k1c
  report` at `ab8c205` on `full/published.tar` `3207020e…898d`), with the
  local validation beside it.
- Outcome. On R, at 1.927 kbps:

  | item | `warp` LPIPS | hull LPIPS | hull's rate for `warp`'s LPIPS (saving) | LPIPS over S | `warp` DISTS / hull | `warp@uncoded` LPIPS | R unfitted | fitted R missed | spill |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | P01_107 | **0.087** | 0.258 | 45.9 kbps (96%) | 0.089 | 0.071 / 0.223 | 0.004 | 0.4% | 3.0% | 2.4% |
  | P02_02 | 0.272 | 0.204 | 0.96 kbps (−101%) | 0.276 | 0.191 / 0.251 | 0.192 | 39.6% | 8.0% | 3.0% |
  | P03_10 | **0.093** | 0.155 | 8.6 kbps (78%) | 0.104 | 0.104 / 0.203 | 0.010 | 1.4% | 5.2% | 3.5% |
  | P09_106 | 0.211 | 0.184 | 1.06 kbps (−82%) | 0.219 | 0.218 / 0.225 | 0.138 | 26.8% | 7.3% | 6.1% |

  (Hull: H3d's bar and DCVC-UF + G, K1's joint hull, read at the
  keypoint rate. R unfitted: the share of R in a hand that has no H2 fit
  in that frame, drawn as fill. Missed: the share of a fitted hand's mask
  the warp does not reach. Spill: the share of the drawing outside R.
  The coded joints move 7–12 px from the uncoded ones.)
  - LPIPS on R: the bound is below the hull on **2 of 4** items, so by
    its own gate this fails the 3-of-4 bar. LPIPS over S gives the same 2.
  - DISTS on R: the bound is below the DISTS hull on **4 of 4**. That
    moves the count across 3, so **by the rule the outcome goes to the
    user.**
- Diagnosis. The two losing items are the two where H2 left much of R
  unfitted (40% and 27% of R, against 0.4% and 1.4%). Those pixels are
  charged as Telea fill. `warp@uncoded` isolates that charge: it is 0.192
  and 0.138 there, already near the hull (0.204, 0.184), and the coding
  error (11–12 px) adds the rest. Where the hands are fitted, the outline
  granted turns H3b's loss into a 78–96% saving. H3b's MANO bound
  (`wilor:arm`, 0.17–0.31 on R through K1's bridge) lost on 3 of 4. So
  what decides hands at this rate is the estimator's coverage (a hand
  with no fit has no keypoints) and the coding error. The renderer's
  outline does not decide it.
- Hypothesis: refuted as stated (2 of 4, not ≥ 3), held where the hands
  are fitted. The competing explanation (finger blending makes the coding
  error dominate) does not hold on the fitted items: misses are 3–8%.
- Proposed reading, for the user. (a) By LPIPS's letter, hands are closed
  for every renderer at the keypoint stream's rate as H2 codes it today.
  (b) Or hands get K6b's ladder scoped to fitted hands, after a coverage
  fix: H2b's hybrid estimator, or hands without a fit sent as pixels (a
  hybrid). That would need its own bound first, charging the pixel
  coder's rate on unfitted hands.
- Scope. 4 pilot items, 240 frames. Appearance and outline granted, and
  no reference billed. The hull is H3d's recorded points. LPIPS decides,
  DISTS is reported, and they disagree on 2 items.
- Budget: 1,182 s of stage execution plus the smoke (under 0.5 GPU-hour),
  inside the 1-hour ceiling.
