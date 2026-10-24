# Experiments: H3b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-11 — H3b oracle: perfect appearance at the coded pose
- Question. Could a generator that draws the hand from the pose stream
  alone (about 2 kbps, 20× below SVT-AV1's lowest rate on the hand pixels)
  beat SVT-AV1 on VISOR's hand pixels, if its appearance were perfect? The
  oracle grants a perfect generator (the true hand's own pixels) and charges
  only what the pose cannot carry (PLAN H3, agreed 2026-10-11):
  - the coded pose's placement error, from estimation and from coding;
  - what MANO does not describe: the forearm and sleeve, objects in front
    of the fingers, blur outside the mesh.
- Poses. H2's VISOR fits (jobs `20261009T215322Z-188ed69c` and
  `20261009T220823Z-b309d9a2`: per hand-frame rotation, shape, translation,
  for WiLoR and HaMeR, on H1's boxes) coded per track with each model's
  100 ms choice from H2's `code.json` (WiLoR: Gaussian 33 ms, 15 Hz
  interpolated, full pose, steps ×8, 1.93 kbps; HaMeR: the same with PCA 12
  and steps ×4, 1.77 kbps), decoded with the track's median shape as H2's
  scheme sends it. The uncoded fit, each frame with its own shape, stands
  in for the true pose (VISOR has none).
- Render. Per frame, each fitted hand's MANO mesh is rasterized at the
  coded pose (painter's order by depth, both hands together). Each covered
  pixel takes the source's colour at the same surface point (face and
  barycentric coordinates) on the uncoded mesh's projection, sampled
  bilinearly; where that point falls outside the hand's mask, the nearest
  mask pixel's colour instead, so the render only ever shows hand. Rendered
  pixels never cover objects (VISOR's non-hand masks: the object stream is
  in front). The share of samples hidden in the uncoded render (self
  occlusion) is reported.
- Composites. Inside the hand region R (union of VISOR's hand masks, fill
  set, forearm included, as H3a), the source is first replaced by an
  OpenCV Telea fill of R (radius 5): the background a client would show
  where nothing is drawn. Then, per model m ∈ {WiLoR, HaMeR}:
  - `m:hand`: the render on top. Forearm, unfitted hands and every hand
    pixel the render misses stay filled (charged).
  - `m:arm`: the forearm's true pixels, then the render. A perfect arm
    model, granted free: the bound for a generator that also draws the arm.
  - `m:arm_uncoded`: `m:arm` at the uncoded pose (diagnostic: coding error
    against estimation error).
  - `m:hybrid@c`: forearm and unfitted hands as SVT-AV1's pixels at CRF c,
    then the render; rate = pose + H1's bits on forearm and hand-without-fit
    at c.
  The forearm is H1's split of each fitted hand's mask at WiLoR's uncoded
  wrist line (so H1's bits on each part apply); unfitted hands are masks
  with no H1 box. SVT-AV1 (`svt@c`): B2's streams at CRF 41, 48, 55, 62
  inside R, as H3a.
- Judges. Every method is scored per hand-frame on the same region S: R
  plus every rendered silhouette of either model, coded or uncoded,
  objects excluded.
  - LPIPS: B2's AlexNet spatial map on the full frame, averaged over S.
    Decides (user, 2026-10-10). SVT-AV1's LPIPS over R alone is kept as a
    check against H3a's numbers.
  - PSNR over S, and DISTS over S: G5c's masked DISTS (torchmetrics 1.9's
    alpha/beta on `Models/DISTS/vgg16-397923af.pth`, every layer's
    statistics weighted by S) on the crop around S padded by 64 px.
  - Temporal consistency: G5c's flicker on S over consecutive hand-frames
    (DIS flow on the source warps t−1 onto t; the method's temporal
    residual minus the source's, luma, mean absolute value).
  - On every 5th frame (judge frames), per fitted hand, for the source,
    `svt@c`, `m:hand` and `m:arm`:
    - pose fidelity: WiLoR re-run on the composite with H1's box, 2D joint
      error against WiLoR re-run on the source (share of the box);
    - realism: Inception-v3 pool features (torchvision's ImageNet weights,
      `Models/Inception/`; not the TF FID network, so values compare only
      across these methods) of the box crop grown 1.5× square; KID
      (polynomial kernel, unbiased) and FID against the source crops,
      pooled over items.
- Rates. The oracle's rate is the model's pose stream. Appearance is
  granted: a generator would add one reference per track, reported as
  tracks per minute. SVT-AV1's rate per item and CRF is H1's bits on hand +
  forearm + hand without fit (fill set), as H3a.
- Decision rule (fixed before any run; `h3b.DECISION`, applied by
  `h3b report`).
  - Gate: `wilor:arm` at the pose stream's rate against `svt@62`, LPIPS on
    S, per item; mean difference over items with a 95% paired bootstrap by
    item (1,000 resamples). Passes if the interval lies below zero: a
    perfect generator beats SVT-AV1's lowest point at a twentieth of its
    rate. Its equal-quality saving (the rate SVT-AV1 needs for the oracle's
    LPIPS, linear in log-rate between CRF points; "beyond CRF 41" if better
    than all) is reported beside it.
  - `wilor:hand`, the same test: whether the generator must draw the arm.
  - Hybrid: H3a's equal-rate test for each `wilor:hybrid@c` (SVT-AV1's
    LPIPS at the hybrid's rate, linear in log-rate, held at the end
    points); passes if below zero at some c.
  - Then: if the gate passes, H3b's generator is built (audit next); it
    must draw the arm unless `wilor:hand` also passes. If only the hybrid
    passes, the generator draws the hand and the forearm is coded. If
    nothing passes, the generator is not built and H3c is next, unless the
    gate fails only through placement (`wilor:arm_uncoded` passes) or
    `wilor:arm` matches `svt@62` on pose fidelity, DISTS and KID; then the
    outcome goes to the user before the route is dropped.
  - HaMeR's composites take the same tests, reported beside (swapping the
    estimator is cheap, H2).
- Hypothesis. `wilor:arm` passes the gate (true appearance, about 8 px of
  coding error) but would not beat CRF 55; `wilor:hand` fails, since the
  forearm is about half of R (H1) and a fill is not an arm; the hybrid
  passes at CRF 55 and 62. The coding error costs less LPIPS than the
  estimation error (`arm_uncoded` close to `arm`).
- Competing explanations. (1) The charged pixels show a Telea fill; a
  better background fill would charge less. The share of R the render
  misses and LPIPS on rendered pixels alone are reported. (2) Appearance is
  sampled at the uncoded pose, so estimation error is charged only through
  the silhouette; HOT3D's motion capture could test that proxy and is run
  only if the decision turns on it. (3) LPIPS charges a few pixels of
  displacement that a viewer would not notice without the source beside
  it; pose fidelity and DISTS are reported to show that.
- Budget. GPU (LPIPS, DISTS, WiLoR and Inception on Ada or A6000, all
  verified there): smoke ≤ 600 s (one item, 48 frames); pilot: B2's 4 pilot
  items, ≤ 45 min; full: the other 30 items in parts of 5 (about 5 min per
  item), ≤ 3 GPU-hours. Ceiling 4 GPU-hours.
- Jobs. Smoke `20261011T070541Z-cf8849a9` failed on its first hand frame
  (OpenCV's `remap` caps map sides at 32,767; fixed in `4bd4f51` with a
  test). Pilot `20261011T071143Z-d8e6dc58` (code `4bd4f51`; gpu3, RTX
  A6000; environment `pointstream-20261006T113321Z`; inputs and specs in
  `pointstream-data/visor/h3b-2026-10-11/`; B2's 4 pilot items, 240
  frames; 2,619 s, 11 min per item, peak 14.2 GiB; validator 14/14,
  including WiLoR's re-run reproducing H2's fits and the uncoded
  silhouette matching H2's IoU). Report (`h3b report`, code `4bd4f51`, on
  the pilot only): `pointstream-data/visor/h3b-2026-10-11/report-pilot-4bd4f51/h3b-report.json`
  (`448aecaf…1fec`).
- Outcome (pilot, 4 items). **The gate fails, by a wide margin, and every
  judge agrees.**

  | method | kbps | LPIPS | DISTS | PSNR dB | flicker | pose error | KID | FID |
  |---|---|---|---|---|---|---|---|---|
  | `svt@62` | 35.1 | 0.102 | 0.127 | 33.9 | 2.1 | 1.0% | −0.002 | 20 |
  | `svt@41` | 178.9 | 0.052 | 0.085 | 39.3 | 1.3 | 0.5% | −0.003 | 8 |
  | `wilor:arm` | 1.93 | 0.211 | 0.180 | 20.7 | 9.4 | 1.7% | 0.003 | 48 |
  | `wilor:hand` | 1.93 | 0.298 | 0.224 | 18.7 | 11.4 | 2.6% | 0.005 | 56 |
  | `wilor:arm_uncoded` | 1.93 | 0.168 | 0.163 | 22.8 | 6.4 | – | – | – |
  | `hamer:arm` | 1.77 | 0.248 | 0.192 | 19.8 | 9.4 | 2.3% | 0.005 | 60 |

  (LPIPS, DISTS, PSNR and flicker on S, means over items; pose error is
  WiLoR's 2D joint error against its fit on the source, as a share of
  the box, on judge frames; KID and FID on 255 hand crops per method.)
  - Gate: `wilor:arm` − `svt@62` = +0.109 LPIPS (95% CI +0.062 to
    +0.156), better on 0 of 4 items. `wilor:hand` +0.196. HaMeR is worse
    still (+0.146). Even with no coding error, `wilor:arm_uncoded` is
    +0.066 (CI +0.003 to +0.129), better on 1 item.
  - Hybrid (forearm coded by SVT-AV1): worse than SVT-AV1 at its own rate
    at every CRF (+0.059 to +0.072; at CRF 62 the hybrid is 18.1 kbps and
    LPIPS 0.166, against SVT-AV1's 0.102 at 35 kbps).
  - Equal quality: the oracle is worse than SVT-AV1's lowest point on
    every item, so the rate SVT-AV1 would need for its LPIPS lies below
    35 kbps and is not measured here.
  - Render: the coded pose moves WiLoR's joints 9.9 px (HaMeR 15.2); the
    render misses 9.7% of the hand pixels and 9.2% of what it draws falls
    outside them; 9.5% of samples land outside the region and 2.2% are
    hidden in the uncoded render. LPIPS on the rendered pixels alone is
    0.207, so the loss is in the hand itself, not in the fill.
  - Verdict by the rule: the generator is not built; H3c is next. No
    escalation applies: `arm_uncoded` fails the gate too, and `wilor:arm`
    is worse than `svt@62` on pose fidelity, DISTS and KID.
- Reading. A perfect-appearance generator placed by MANO loses to coded
  pixels at SVT-AV1's lowest rate on every judge, including those that
  ignore identity. About half of the gap remains with the uncoded pose
  (0.168 against 0.211), so it is not mainly the pose coding: MANO's
  silhouette misses or overshoots about a tenth of the hand, and LPIPS
  charges the misplaced edges and the displaced texture. The forearm
  matters as predicted (`hand` 0.298 against `arm` 0.211); the hypothesis
  that `arm` would pass is refuted. Competing explanation (1), the fill:
  LPIPS on the rendered pixels alone (0.207) is already twice SVT-AV1's,
  so a better fill (G5c's ProPainter) cannot close the gap.
- Scope. Four items; the full run was not needed for the decision (0 of
  4 items better, every judge agreeing). The realism judge uses
  torchvision's ImageNet Inception-v3, not the TF FID network, so its KID
  and FID compare methods within PointStream only; the paper must say so.
  The pose stream's rate leaves out one reference per track (163 tracks
  per minute on these items, since fits drop out between frames).
