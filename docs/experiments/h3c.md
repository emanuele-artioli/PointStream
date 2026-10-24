# Experiments: H3c

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-11 — H3c: a perceptually tuned DCVC-UF for the hand pixels
- Question (PLAN H3c). G5b's method, DCVC-UF HT-L fine-tuned with β·LPIPS
  added to its masked distortion, turned the background decoder from blur
  to texture and transferred across kitchens. Fine-tuned on hands and
  forearms, does it code VISOR's hand pixels for fewer bits than SVT-AV1 at
  equal LPIPS? It is a foreground coder in its own right and the bar for
  H3d (and for any generator): H3b2 showed that SVT-AV1 downscaled already
  reaches the pose stream's rate on the hands.
- Region and items. R is the union of both hand classes of the fill set
  (VISOR's hand masks with B1b's SAM fill; they include the visible
  forearm), as in H3b's `svt_lpips_r`. Scored on B2's 4 pilot items
  (P01_107, P02_02, P03_10, P09_106), 240 frames each, on every frame with
  a hand.
- Quality, as H3b scored SVT-AV1 on R: the decode inside R, the source
  outside, full-frame LPIPS (AlexNet) averaged over R, mean over the
  window's hand frames. PSNR on R and masked DISTS on R's box plus 64 px
  beside it. Points coded below full resolution are upscaled bicubically to
  1920×1080 first (H3b2).
- Anchor: SVT-AV1's per-item front on R (lowest LPIPS at each rate) over
  B2's CRF 41/48/55/62 streams (rate on R: H1's libaom accounting) and
  H3b2's seven low points (CRF 63 at 1, ½, ¼, ⅛ and 1⁄16 of the
  resolution, CRF 55 at ¼ and ⅛; re-encoded with H3b2's code and checked
  against its stream sha256s).
- DCVC-UF points. HT-L codes the window's own 4:2:0 planes (full range as
  B2), at full resolution and downscaled by 2 and 4 (H3b2's per-plane area
  downscale and sizes), at QP 0, 21 and 42: nine points per arm and item.
  DCVC-UF's floor at full resolution is above the hands' interesting range
  (B2: QP 27 is 329 kbps for the whole frame), so the scales are what
  reaches it, as for SVT-AV1. Rate on R: each NAL's container bytes are
  split by DCVC-UF's own entropy model, per 16×16 latent cell (bits of y,
  plus z's spread over its cells), from a forward pass of the same model on
  the same frames with rounding in place of noise; each cell's bits are
  spread evenly over its pixels in the NAL's frames and summed over R's
  pixels (H1's rule for libaom's blocks; R downscaled with area ≥ 0.5 as
  for SVT-AV1's low points). Reported beside: the estimate's total against
  the actual bytes per NAL, and every point's whole-frame rate.
- Arms. *Reference*: HT-L off the shelf. *Upper bound* (the oracle):
  fine-tuned on the 4 pilot windows themselves. *Component*: fine-tuned on
  the 24 other items whose participants are not the pilot items'
  (P01, P02, P03 and P09 excluded): another person's hands in another
  kitchen. The component is one model for every video, delivered with the
  decoder and not billed, like any codec's weights; the upper bound is not
  deployable.
- Fine-tuning (`experiments/background/g5b_train.py` unchanged, through a
  new driver): G5b's recipe (HT-L stage 2, cascaded 33-frame sequences,
  batch 4, AdamW at 1e-5, a random QP per sample with λ from 1 to 768, the
  image model frozen, gradient norm clipped at 0.2), distortion on R only
  with DCVC's YUV/RGB weighting, optionally plus β·LPIPS on R. Training
  data: segments of 48 consecutive window frames at one scale (1, ½, ¼),
  each cropped to a fixed 512×512 box centred on R's union over the
  segment (the ¼ frames, 480×272, padded by edge replication outside the
  loss mask); the upper bound uses every segment of the 4 pilot windows,
  the component two segments per item and scale. β comes from a 40-step
  calibration of the upper bound at β = 0, by G5b's rule: β times the
  monitor set's mean LPIPS on R equals twice its masked distortion at
  step 0. The steps are G5b's 540 (30 minutes at its exclusive Ada
  measurement, 3.28 s per step at the same patch, batch and groups):
  H3c's GPU jobs run shared (fleet, 2026-10-11), so their own step times
  are not evidence. Training runs on Ada only (HT-L training has not run
  on an A6000; a first run there would have to be exclusive).
- Comparison. Per item, each arm's front over its nine points against
  SVT-AV1's: BD-rate on LPIPS_R, the mean log-rate difference over the
  LPIPS range both fronts cover (linear in log rate, 64 levels); none
  where they do not overlap (then "dominates" or "is dominated"). Saving =
  −BD.
- Decision rule (`h3c.DECISION`, fixed before the run).
  1. Upper bound, β = 0 and β > 0. It *passes* if at some β the mean
     BD-rate over the 4 items against SVT-AV1 is below −10% with at least
     3 of 4 items below 0. It *fails* if at both β the mean is at or above
     0 (or SVT-AV1 dominates): even fitted to the test frames, the tuned
     coder does not beat SVT-AV1, so H3c is not built and H3d's bar is
     SVT-AV1's front. Otherwise *unclear*: the component still runs (one
     run), flagged.
  2. Component, at the β of the better upper bound. *Adopted* (H3c is the
     foreground pixel coder and, with SVT-AV1, sets H3d's bar as the lower
     envelope of the two) if its mean BD against SVT-AV1 is below 0 with
     at least 3 of 4 items below 0, and its mean BD against the reference
     is below −10% (the tuning, not DCVC-UF itself, earns it). If the
     reference alone beats SVT-AV1 by the same test, it joins the bar.
  DISTS's and PSNR's BD-rates are reported beside; a sign disagreement
  between LPIPS and DISTS is stated with the visual check.
- Hypothesis. Off the shelf, DCVC-UF is worse than SVT-AV1 on LPIPS_R
  (G5b: +151% median on the background) and closer on PSNR (B2: −13% on
  weighted PSNR). β > 0 carries G5b's large LPIPS gain over: the upper
  bound passes by 50% or more, β = 0 is near parity. The component keeps
  about half of the upper bound's log-rate gain (G5b's scene model kept
  half, its cross-scene control most of the LPIPS gain) and beats SVT-AV1
  by 20–50% on LPIPS_R, at a PSNR cost. At ¼ scale DCVC-UF reaches a few
  kbps on R.
- Competing explanations. (1) β > 0 synthesizes texture that LPIPS rewards
  but that is not the source's (G5b's visual check): DISTS, PSNR and
  decoded frames of each arm, published by the coding jobs, expose it.
  (2) The rate split favours one codec: DCVC-UF's estimate and libaom's
  accounting are different instruments. The estimate's total against the
  actual bytes, and the whole-frame rates, are reported; a BD-rate sign
  that flips between the region rate and the whole-frame rate is stated.
  (3) The component knows the person or kitchen: excluded by participant.
- Budget. Smoke ≤ 10 minutes. Training data is built inside each training
  job (a window decodes in about 2 s; H3b2). Anchor and reference: one GPU job per
  pilot item ≤ 45 minutes. Calibration ≤ 10 minutes. Upper bound: two
  runs ≤ 45 minutes (30 minutes of training each), then their coding,
  split into jobs ≤ 45 minutes. Component: one run and its coding.
  Ceiling 8 GPU-hours on Ada (training: 47,000 MiB declared, as G5b
  peaked at 42.7 GiB allocated) and Ada or A6000 (coding: 32,000 MiB, after 24,000 was overrun at 26,098). Stage by stage; a stage that fails the rule stops what follows.
- Inputs: H3b's (`pointstream-data/visor/h3b-2026-10-11/inputs.json`:
  evaluation set, dense archive, fill masks and record, B2's SVT-AV1
  streams and results, LPIPS, DISTS, the videos), H1's report
  (`visor/h1-2026-10-08/report-6847a29`) and inspect binary, H3b's and
  H3b2's units for the cross-checks, DCVC-UF's image and HT-L checkpoints
  (G5b's). Environment `pointstream-20261009T214002Z` (`32e68872…4661`,
  G5b's; the H3b environment plus NVRC).
- Calibration (`20261011T215620Z-b6a1294d`, shared, RTX 6000 Ada, code
  `0ccb20d`; the upper bound's data: 4 windows × 3 scales × 5
  segments, 2,880 fitted frames, 960 sequence starts; 40 steps, β = 0;
  validator 8/8). At step 0 the monitor set's masked distortion is
  0.002376 and its LPIPS on R 0.0564, so **β = 0.084** (G5b's 0.026 came
  from LPIPS on the whole visible background, 0.203). Peak 42.72 GiB
  allocated, as G5b. Infrastructure, not evidence. Upper bound, 540 steps:
  β = 0 `20261011T230242Z-eed20579`, β = 0.084 `20261011T230251Z-97f6ec94`.
- Upper bound, outcome (stage 1). Training: β = 0 converged (monitor loss
  −2.4% over the last fifth); β = 0.084 did not by G5b's 3% (−3.5%), which
  can only understate an upper bound (training jobs at code `5002cf6`).
  Coding jobs (shared; validators passed; code per job's `ready.json`:
  anchor and reference `4e6bd8d`, P02_02's `5002cf6`; arms `ab86e31`,
  P09_106's β = 0 rerun `8c70699`): anchor and reference `20261011T220119Z-69f5f6d7`
  (P01_107), `…223528Z-6ecdc738` (P02_02), `…221908Z-8193ffb7` (P03_10),
  `…220719Z-19637920` (P09_106); β = 0 `20261011T235449Z-bee05804`,
  `…235458Z-ac5f6422`, `…235508Z-87306ff1`, `20261012T014347Z-7c545a0e`;
  β = 0.084 `20261011T235710Z-b7171bd9`, `…235719Z-d0c4b5b3`,
  `…235728Z-58b72909`, `…235737Z-4bdfd185`. Earlier attempts, failed and
  not evidence: a missing archive extraction and the estimate's padding
  (smokes), a CUDA out-of-memory when another user's process grew on gpu6,
  a memory overrun of the shared claim (26,098 MiB against 24,000 + 1,024;
  fixed by releasing the scorer's cache and declaring 32,000), and one
  decode mismatch of the β = 0 model on P09_106 at ½ scale, QP 42, on gpu6
  (`20261011T235517Z-dc8fe8c1`), which did not recur when the same job ran
  again on gpu3. Report (`h3c report`, code `a5b8a0a`):
  `pointstream-data/visor/h3c-2026-10-11/report-stage1/h3c-report.json`
  (`b78300039d74d31c…`). The anchor reproduces H3b: SVT-AV1's LPIPS on R
  matches H3b's and H3b2's units within 0.0002 on every frame, and every
  re-encoded low stream matches H3b2's sha256.

  BD-rate on LPIPS_R against SVT-AV1's front (mean over the 4 items, 95%
  bootstrap CI; items below 0), and beside it:

  | arm | LPIPS_R | items < 0 | vs reference | DISTS_R | PSNR_R | LPIPS_R, whole-frame rate |
  |---|---|---|---|---|---|---|
  | reference (off the shelf) | −63% (−68 to −59) | 4 | — | −55% to +52% | −51% to −73% | −59% to −73% |
  | upper bound, β = 0 | −43% (−52 to −33) | 4 | +66% | −27% to +66% | −58% to +10% | −73% to −87% |
  | upper bound, β = 0.084 | −61% (−75 to −48) | 4 | −14% (2 items worse) | −5% to −78% | −48% to +18% | −78% to −94% |

  By the rule the upper bound **passes** (β = 0.084), and the component
  runs at β = 0.084.
- Reading (stage 1). The hypothesis was wrong at its first step: DCVC-UF
  off the shelf already needs about 63% less rate than SVT-AV1 for the same
  LPIPS on the hand pixels, on every item, and PSNR and the whole-frame
  rate agree (B2 found −13% on whole frames: the hands at these low rates
  are where DCVC-UF gains most). So, by the rule's last clause, the
  reference joins H3d's bar. DISTS disagrees on two items (+52%, +16%):
  DCVC-UF's smooth hands lose texture statistics that DISTS weighs. Tuning
  with β = 0.084 removes that disagreement (DISTS −5% to −78% on every
  item) but adds only 14% against the reference on LPIPS_R, two items
  worse; tuning without LPIPS is 66% worse than the reference: with the
  rate charged on the whole patch and the distortion only on R, the model
  learned to starve the background (whole-frame rate −73% to −87%), not to
  code the hands for fewer bits. The upper bound's margin over the
  reference is close to the component's 10% bar, so the component is
  expected to miss it.
- Visual check (P02_02, where the reference loses on DISTS; frame 120,
  every coder at about 8 kbps on R, crops from the coding jobs'
  `published.tar`). SVT-AV1 at ¼ (CRF 63, 8.0 kbps) blurs and blocks the
  glove's edges and loses the glass's rim. DCVC-UF off the shelf at full
  resolution (QP 0, 8.2) keeps the outlines but renders the glove waxy and
  flat, without its folds: the lost texture DISTS charges. At ½ (QP 21,
  8.5) it keeps more folds and the rim, and the tuned model (β = 0.084,
  QP 0 at ½, 8.3) is close to it with slightly more fold detail. No
  synthesized or periodic texture in any. The source itself is motion
  blurred here.
- Component, outcome (stage 2). Training `20261012T021957Z-703553f1`
  (shared, RTX 6000 Ada, code `17b6dc0`; 24 items of 14 participants, two
  segments per item and scale, 6,912 fitted frames; β = 0.084, 540 steps;
  converged; validator 8/8; peak 46.69 GiB allocated, close to the 47,000
  MiB declared). Coding (code `26643b1`, validators passed): P01_107
  `20261012T030241Z-1217c42b`, P02_02 `…030324Z-534c7f95`, P03_10
  `…030418Z-3df2f007`, P09_106 `…030512Z-00191e69`. Report (code
  `26643b1`): `pointstream-data/visor/h3c-2026-10-11/report-stage2/h3c-report.json`
  (`a402322a11ff7d2d…`), over all 16 coding jobs.

  | item | LPIPS_R vs SVT-AV1 | vs reference | DISTS_R | PSNR_R | whole-frame rate |
  |---|---|---|---|---|---|
  | P01_107 | +19% | +219% | +31% | +169% | −31% |
  | P02_02 | −59% | +21% | −36% | −48% | −81% |
  | P03_10 | +55% | +286% | −24% | +236% | −56% |
  | P09_106 | −39% | +80% | −33% | −30% | −79% |
  | mean | −6% (95% CI −49% to +37%), 2 of 4 below 0 | +151% | | | |

  **Not adopted**: it does not beat SVT-AV1 on 3 of 4 items, and it is
  151% worse than DCVC-UF off the shelf.
- Reading. Fine-tuning DCVC-UF for hands does not pay. Fitted to the test
  windows themselves it gains only 14% over the published model; fitted to
  other people's hands it loses, by a factor of 2.5 on average in rate at
  equal LPIPS on R. As for β = 0 in stage 1, the whole-frame rate falls
  (−31% to −81% against SVT-AV1) while the rate on the hands rises: with
  the distortion counted only on R and the rate on the whole patch, the
  model learns to drop the background, and on unseen hands it has also
  lost some of what the published model knew. A loss with the rate on R
  only, or the hands coded without the background, would be a different
  design; nothing here suggests it would beat the published model by much,
  since even the upper bound barely did. **H3c's outcome is the reference**:
  DCVC-UF HT-L off the shelf codes VISOR's hand pixels for about 63% less
  rate than SVT-AV1 at equal LPIPS (CI −68% to −59%, 4 of 4 items; PSNR
  and the whole-frame rate agree, DISTS on 2 of 4). H3d's bar (and any
  generator's) is the lower envelope of SVT-AV1 and DCVC-UF off the shelf
  on R, per item. Scope: 4 pilot items, 240 frames each; one β; one
  trainer recipe (G5b's).
- GPU time: about 7 GPU-hours (calibration 0.1, three training runs 1.6,
  16 coding jobs about 4.5, failed attempts about 0.5), within the 8
  allowed. Shared jobs' timings are not evidence.
