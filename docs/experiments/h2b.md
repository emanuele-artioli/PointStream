# Experiments: H2b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-10 — H2b: a hybrid of the two estimators
- Question. H2 chose WiLoR 3 to 2, but the two models are better at
  different parts of the pose: WiLoR at the fingers (HOT3D PA-MPJPE 6.2
  against 9.2 mm), HaMeR at the whole hand's orientation (7.6° against
  14.7°). An exploratory combination of H2's recorded HOT3D outputs (CPU;
  the 54 clips H2 used, so not evidence) took WiLoR's articulation and
  shape and HaMeR's global orientation and wrist position: wrist-aligned
  MPJPE 18.2 mm against HaMeR 20.1 and WiLoR 29.7, PA-MPJPE 6.2, orientation
  7.6°; the reverse combination was worse than both (31.7 mm), and no
  per-frame or per-depth switch favoured WiLoR (HaMeR is closer on 72% of
  hand-frames, 67–82% in every depth quartile). The hybrid was chosen on
  those clips, so it is tested here on data it was not chosen on.
- Hybrid. Per hand, both regressors on the same box (H2's preprocessing).
  Sent: WiLoR's 15 joint rotations and shape, HaMeR's global orientation and
  translation, the translation corrected so the wrist stays where HaMeR put
  it (MANO turns about a shape-dependent rest wrist). Computed offline from
  the saved parameters with H2's numpy MANO; each pure model's saved output
  is reproduced from its parameters as a check (< 0.5 px, < 1 µm).
- Data. HOT3D: 54 new train_aria clips, the next 6 per participant in H2's
  content-blind order (none of H2's). HInt: test, rerun saving the
  parameters (H2 saved keypoints only); the hybrid was not chosen on HInt.
  VISOR: H2's recorded runs; the silhouettes rendered offline against the
  masks' hand side as WiLoR split it (the chosen model's, so the hybrid is
  not favoured), with HaMeR's split reported beside it.
- Decision rule (fixed before any run; `h2.HYBRID_DECISION`, applied by
  `h2 hybrid`). The hybrid replaces WiLoR when all three hold: (a) HOT3D
  wrist-aligned MPJPE below both models' (95% paired bootstrap interval of
  the difference below zero against each, 1,000 resamples by clip); (b)
  HInt PCK@0.05 (all, keypoint-extent normalisation) on VISOR frames and on
  New Days no worse than WiLoR's by more than 1.0 point (the interval's
  lower bound of hybrid − WiLoR above −1.0; by image); (c) VISOR hand-side
  IoU median no worse than WiLoR's by more than 0.01 (lower bound above
  −0.01; by item). Otherwise WiLoR stays. Reported beside: PA-MPJPE,
  orientation, wrist position, acceleration error (world axes), 2D error in
  the fisheye image, and the cost (both regressors: about 60 ms per hand at
  batch 1 on an A6000 against 30).
- Hypothesis. The hybrid passes all three: its HOT3D gain is about 2 mm
  over HaMeR and 11 over WiLoR; on HInt and VISOR, which score 2D
  keypoints and silhouettes, HaMeR's orientation changes little, so it lands
  within the margins of WiLoR.
- Competing explanation. The exploratory gain is selection on 54 clips; on
  new clips it shrinks below HaMeR's. Or the parts do not compose: WiLoR's
  finger angles are relative to its own (wrong) wrist frame, so under
  HaMeR's orientation the fingers point elsewhere and the 2D tests drop.
- Budget. GPU: HInt test (3,660 hands, ≈ 15 min) and 54 HOT3D clips (≈ 35
  min), each with a ≤ 600 s smoke; CPU: `h2 hybrid` ≤ 30 min. Ceiling 2
  GPU-hours.
- Jobs (environment `pointstream-20261006T113321Z`; inputs and specs in
  `pointstream-data/visor/h2b-2026-10-10/`). HInt `20261010T061701Z-ddbd03d5`
  (gpu1, Quadro RTX 8000; 3,660 hands; 870 s; validator 10/10; WiLoR
  43.73 against H2's 43.76 on the A6000). HOT3D `20261010T061706Z-04fd0abb`
  (gpu2, RTX A6000; 54 new clips, 14,718 hands; 2,076 s; validator 10/10).
  Scoring `h2 hybrid` (CPU): `20261010T065756Z-44c8f27f` failed (its smoke's
  two HInt blocks held one split only; fixed), `20261010T065944Z-2452074c`
  (code `ca092fd`; 96 s; validator 6/6; reproduction of each model from its
  parameters 0.0006 px on HInt, 0.05 µm on HOT3D, VISOR WiLoR IoU within
  0.0007 of H2's) is the result:
  `pointstream-data/visor/h2b-2026-10-10/report-ca092fd/hybrid.json`
  (`7dd0d8ed…c460e`).
- Outcome. **The hybrid is rejected; WiLoR stays.** (a) holds: on the new
  HOT3D clips its wrist-aligned MPJPE is 19.4 mm against HaMeR 20.7 (−1.2,
  CI −2.0 to −0.6) and WiLoR 30.1 (−10.7, CI −12.9 to −8.5); PA-MPJPE 6.0
  (WiLoR's), orientation 8.2° (HaMeR's 8.1°). (b) fails: HInt PCK@0.05 on
  VISOR frames 35.6 against WiLoR 43.7 (−8.1, CI −8.8 to −7.5) and HaMeR
  39.5; on New Days 44.4 against 47.5 (−3.1, CI −3.7 to −2.4) and 48.9.
  (c) fails: VISOR hand-side IoU median 0.739 against WiLoR's 0.768 (−0.029,
  CI −0.037 to −0.022; against HaMeR's split 0.755 against 0.783). Beside
  them on HOT3D: 2D error in the fisheye image 20.2 px against 17.3
  (HaMeR) and 17.4 (WiLoR); acceleration 6.6 against 5.7 and 7.2 mm/frame².
- Reading. The competing explanation holds: the parts do not compose. Each
  model's finger rotations are fitted together with its own global
  orientation to the same image; under the other model's orientation they
  point elsewhere, so the hybrid is worse than both parents wherever the
  image is the judge (HInt keypoints, VISOR silhouettes, HOT3D's 2D error),
  while the wrist-aligned 3D error, which weighs the orientation most, still
  improves. The exploratory 3D gain replicated (smaller: 1.2 mm over HaMeR
  against 1.9), which is the metric the hybrid was chosen on and not the one
  a renderer is judged by. Mixing estimators at the parameter level is not a
  route to a better pose; a per-frame choice was already ruled out by H2's
  data (HaMeR closer on 72% of HOT3D hand-frames in every depth quartile).
  What remains of the orientation gap is H3's to measure on renders.
