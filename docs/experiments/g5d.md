# Experiments: G5d

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-11 — G5d oracle: a venue-trained enhancer of SVT-AV1 (user, 2026-10-11)
- Question: can knowledge of the venue held at the client make plain
  SVT-AV1's background better at no extra rate, keeping what AV1 does well
  (PSNR, steady frames), which the scene-adapted DCVC-UF loses? The
  stream stays standard AV1; a small network, sent once like the scene
  model, maps each decoded frame to a better one (NAS, Yeo et al., OSDI
  2018). Asked after step 4's round 1: we must beat AV1 by using what the
  client already has, not by a stronger conventional codec.
- Method (`experiments/background/g5d.py`). SVT-AV1 at G5's CRFs and preset
  on the excerpt's ProPainter fill (the prefill of step 2). An
  EDSR-style residual CNN (8 blocks, 64 channels, about 0.6 M parameters,
  starting as the identity), one per CRF, fitted on (decoded, source)
  patches of 128 px on V for 2,000 steps (Adam 2 × 10⁻⁴, cosine). Fits:
  *excerpt*, on the scored frames themselves (the oracle); *scene*, on the
  stretch's other frames coded at the same CRF (G5b's split, the same
  footage as recipe C). Losses: masked MSE, and MSE plus LPIPS on V at
  equal weight at step 0. Scored as every method (`g5c.full_score`);
  the rate is SVT-AV1's.
- Clips: the pilots P26_02 and P06_03 (calm, where a venue model helps
  most; the oracle needs the best case).
- Decision rule, fixed now. An arm passes when its BD-rate against
  SVT-AV1 (the anchors' envelope of frame and fill, `g5c.py rescore`) is
  −10% or better on at least 3 of PSNR_V, LPIPS_V, DISTS_V, flicker_V and
  KID_V, on both pilots. No excerpt arm passing closes the enhancer. A
  scene arm passing means a venue-footage enhancer (other recordings,
  as step 3) is built next and compared with the scene model.
- Hypothesis: the excerpt arms pass with PSNR_V among their metrics
  (−15% to −30%; an MSE-fitted network removes AV1's blocking and
  ringing), the scene arms reach about half of that, and flicker does
  not get worse (the network sees one frame but AV1's frames are
  steady). Competing explanation for a pass: memorizing the excerpt's
  frames (0.6 M parameters, 240 frames); the scene arms, which never see
  them, answer it.
- Budget: 2 jobs, each ≤ 45 minutes on Ada or A6000 (12 SVT-AV1 encodes,
  24 fits, 30 scored points), about 1.5 GPU-hours.
- Runs. Each first job ran out of time or was contended after 3–5 of
  the 6 CRFs (`20261011T182551Z-07be07f1` P26_02, A6000, contended after
  CRFs 32–48; `…182602Z-5a00c1bc` P06_03, Ada, 2,700 s after CRFs
  32–59; code `80ac357`). Their salvaged checkpoints were passed back
  (`--restore`) and the remaining CRFs ran on shared GPUs:
  `…213128Z-5d649cb7` (P26_02, Ada) and `…213117Z-60964490` (P06_03,
  A6000), code `86e541b` (the same enhancer code, plus `--restore` and
  the fleet merge). About 75–160 s per fit; 594,371 parameters (1.2 MB
  at fp16).
- Outcome, against the corrected anchor (entry below). BD-rate against
  SVT-AV1, P26_02 | P06_03 ("dom." = better at every rate, no overlap):
  | Arm | LPIPS_V | DISTS_V | PSNR_V | flicker_V | KID_V | Beats by 10% |
  |---|---|---|---|---|---|---|
  | plain SVT-AV1 (check) | 0% \| 0% | 0% \| 0% | +1% \| 0% | +1% \| +1% | 0% \| 0% | 0 \| 0 |
  | excerpt, MSE | −3% \| −4% | −1% \| −1% | 0% \| −2% | +1% \| +2% | −5% \| −4% | 0 \| 0 |
  | scene, MSE | −1% \| −3% | +2% \| 0% | 0% \| 0% | +1% \| +1% | −2% \| −3% | 0 \| 0 |
  | excerpt, MSE + LPIPS | dom. \| −69% | dom. \| −43% | +22% \| +12% | +123% \| +96% | −65% \| −31% | 3 \| 3 |
  | **scene, MSE + LPIPS** | **dom. \| −58%** | **dom. \| −20%** | +25% \| +20% | +122% \| +93% | **−58% \| −24%** | **3 \| 3** |
  - **By the rule a scene arm passes** (3 metrics on both pilots), so a
    venue-footage enhancer (other recordings only, as step 3) is built
    next and compared with the scene model.
  - Hypothesis wrong in kind: the MSE enhancer barely moves SVT-AV1
    (PSNR_V −2% to 0%, even fitted on the excerpt), so AV1's decoded
    frames hold little a small network can restore toward the source.
    The gain is all perceptual, and it costs PSNR_V (+20–25%) and
    flicker (+93–123% BD, about 12% more at equal CRF): the network
    synthesizes texture frame by frame. The scene arm keeps most of the
    oracle's gain (memorization is not the explanation).
  - Against the scene model (recipe L, step 4, the same pilots and
    anchors): L is stronger on LPIPS_V (−92% | −66%) and DISTS_V (−73% |
    −38%); the enhancer on PSNR_V (L +53% | +96%), flicker (L worse at
    every rate) and KID_V (L +12% | +3%), at 1.2 MB against 51 MB, on a
    standard AV1 stream. By the selection's count on these two stretches
    the enhancer beats SVT-AV1 on 3 metrics, L on 2.
  - GPU time: about 2.5 hours (4 jobs).

### 2026-10-12 — Correction: SVT-AV1's envelope was a staircase
- Found in G5d: plain SVT-AV1 on the fill, byte-identical to the anchor's
  own fill streams (same sha256), scored 7–15% *better* than "SVT-AV1's
  envelope". G5's anchor joins SVT-AV1 on the frame and on the fill, and
  `g5.bd_rate` kept their Pareto front: the frame encodes, 7–33% worse
  than the fill's, still put points barely above the fill's at much higher
  rate, and the PCHIP interpolation through that staircase ran below both
  of SVT-AV1's own curves. Every BD-rate against SVT-AV1 since G5 was
  therefore optimistic for the test, by 0–15% of the anchor's rate
  (median about 7%).
- Fix (`g5.envelope`): the anchor is the upper concave hull of its points
  (what switching between the encodes achieves), which here is the fill's
  curve within 1%; the test curve keeps its Pareto front. Regression test
  in `tests/background/test_g5c.py`. Every later number uses it.
- Recomputed G5c numbers (BD-rate against SVT-AV1, was → is); **no
  decision changes**:
  - Step 2, recipe C on the 10 stretches: LPIPS_V median −41% → −39% (8
    of 10 below 0, unchanged), DISTS_V +28% → +30%, PSNR_V +121% →
    +130%. Pilots (median of two): C LPIPS_V −78% → −75%, DISTS_V −55%
    → −51%.
  - Step 3, footage only (G): LPIPS_V −60%/−74% → −53%/−72% (P06_03 /
    P26_02), DISTS_V −40%/−45% → −32%/−46%; scene plus footage (H)
    −57%/−88%.
  - Oracle on the fast stretches (U): P04_13 −0.3% → +9%, P06_05 −11% →
    −6% (the scope conclusion holds, more strongly).
  - Step 4, development set: C LPIPS_V −55% → −49%, DISTS_V +3% → +5%;
    L −65% → −61% and −10% → −2% (L still adopted, 2 metrics against 1;
    its DISTS_V margin is thin); W −50% → −44%.
  - G5b's earlier numbers (Telea anchors, 2026-10-10) carry the same bias
    and are superseded by G5c's; they are not recomputed.

### 2026-10-12 — G5d step 2: the venue enhancer from other recordings; the fast-scene ceiling
- Questions (user, 2026-10-12: both routes for the stored knowledge go in
  the paper). (a) Does an enhancer fitted only on the kitchen's *other
  recordings*, made before the stream exists, still beat SVT-AV1, and
  how does it compare with the footage-only scene model? (b) Does
  feeding the previous decoded frame (a temporal enhancer) remove the
  flicker its texture synthesis adds? (c) Would better training settings
  rescue the fast stretches, or is the base codec the limit there?
- (a, b) Enhancer arms (`g5d.py enhance --arms`, MSE + LPIPS loss as the
  oracle's passing arm, the same network, steps and CRFs): footage-lpips
  and footage-lpips-t (step 3's footage archives: P26_02
  `20261011T062431Z-7d8edeb3`, 1,200 frames of P26_108; P06_03
  `…062441Z-421d23d7`, 2,400 frames of P06_05 and P06_10; each coded by
  SVT-AV1 at the CRF; target the fill on V, equal to the source there
  through 4:2:0), and scene-lpips-t beside the oracle's scene-lpips.
  Two jobs per pilot (CRFs 32, 40, 48 and 54, 59, 63).
  Scene model for the head-to-head: footage only (as step 3's G) with
  the step-4 recipe (learning rate 3 × 10⁻⁵), on both pilots.
- (c) The oracle of step 3 (fitted on the excerpt itself, U) with the
  step-4 recipe on the two fast stretches, P04_13 and P06_05.
- Decision rules, fixed now.
  (a) The venue enhancer stands if footage-lpips or footage-lpips-t
  beats SVT-AV1 by 10% or more on at least 3 of the 5 metrics on both
  pilots. The head-to-head with the footage-only scene model is
  reported metric by metric; it decides nothing (both go in the paper).
  (b) The temporal enhancer replaces the per-frame one if, for the same
  fit, flicker_V at matched rate falls by 10% or more and its LPIPS_V
  BD-rate against the per-frame one is +10% or better, on both pilots.
  (c) A training-settings round for fast scenes is worth running only if
  the oracle with the new recipe beats SVT-AV1 by 30% or more on
  LPIPS_V on either fast stretch (the gain step 3 needed on P06_05);
  otherwise the fast stretches are scoped out as the base codec's limit.
- Hypotheses. (a) The footage enhancer keeps most of the scene
  enhancer's gain (it learns the kitchen's textures, not the excerpt's
  frames) and passes; against the footage-only scene model it repeats
  the pattern of the pilots (weaker on LPIPS_V and DISTS_V, better on
  PSNR_V, flicker and KID_V). (b) The temporal input cuts the added
  flicker by a third or more at little LPIPS_V cost. (c) The oracle stays
  above −30% on both (the old one reached +9% and −6%): fast scenes are
  the base codec's limit.
- Competing explanation for (b) failing: the flicker is the network
  inventing new texture each frame from the same coarse input, which a
  previous *decoded* frame cannot pin down; a recurrent input (the
  previous *enhanced* frame) would then be the next step.
- Budget: 4 enhancer jobs (≤ 45 minutes, shared Ada or A6000) and 4
  training jobs (≤ 45 minutes, Ada), about 5.5 GPU-hours.
- Runs (code `6ab4c00` for all, shared GPUs). Enhancer: P26_02
  `20261012T062332Z-e7ab76cb` (CRFs 32–48, Ada) and `…062342Z-7b7f0358`
  (54–63, A6000); P06_03 `…062353Z-27439cd6` (32–48, Ada) and
  `…062402Z-09d868ee` (54–63, Ada); their SVT-AV1 streams are
  byte-identical to the oracle's. Scene model, footage only, recipe L
  (Ada): P26_02 `…062412Z-3a369103`, P06_03 `…062422Z-17b767b0`.
  Oracle with recipe L (Ada): P04_13 `…062435Z-0be6a398`, P06_05
  `…062459Z-4f8164a9`. The four training jobs were specified without
  `--inception` (step 3's spec helper predates KID), so they have no
  KID_V; their flicker verdicts are "unclear" only because their curves
  reach below SVT-AV1's lowest rate, and they flicker more wherever the
  two overlap.
- Outcome. BD-rate against SVT-AV1 (corrected anchor), P26_02 | P06_03:
  | Model, fitted on the other recordings only | LPIPS_V | DISTS_V | PSNR_V | flicker_V | KID_V | Beats by 10% |
  |---|---|---|---|---|---|---|
  | enhancer (footage-lpips) | −36% \| −50% | −47% \| −31% | +69% \| +14% | +54% \| +83% | +10% \| −19% | 2 \| 3 |
  | enhancer, temporal (footage-lpips-t) | −48% \| −51% | −45% \| −34% | +57% \| +15% | +51% \| +93% | +10% \| −26% | 2 \| 3 |
  | scene model, recipe L | −74% \| −65% | −55% \| −39% | +92% \| +95% | worse | — | 2 \| 2 (of 4) |
  - **(a) The venue enhancer does not stand by the rule** (3 metrics on
    both pilots): on P26_02 it beats 2 (KID_V +10%). It still beats
    SVT-AV1 on LPIPS_V and DISTS_V by 31–51% on both pilots, made only
    from other recordings. P26_02's only other recording (P26_108) is
    from the later collection, where the footage-only scene model was
    also weakest (step 3). Against the footage-only scene model the
    pattern of the pilots holds: the scene model is stronger on LPIPS_V
    and DISTS_V, the enhancer loses less PSNR_V and flicker. The
    footage-only scene model with recipe L improves on step 3's (LPIPS_V
    −6% and −24% against it).
  - **(b) The temporal enhancer is not adopted**: flicker at matched
    rate ×1.00–1.01 against the per-frame one, for both fits on both
    pilots (LPIPS_V 3–13% better). The competing explanation holds: the
    network re-invents texture each frame, and a previous *decoded*
    frame, as blurry as the current one, cannot pin it; a recurrent
    input (the previous *enhanced* frame) is the next candidate.
  - **(c) The fast stretches are not the base codec's limit alone.**
    The oracle with recipe L beats SVT-AV1 on LPIPS_V by 22% on P04_13
    and 31% on P06_05 (with step 2's recipe: +9% and −6%); DISTS_V and
    PSNR_V stay far behind (+71% and +107%; +262% and +281%). By the
    rule (30% on either) a training-settings round for fast scenes is
    worth running. Hypothesis (c) wrong: the learning rate moved the
    ceiling by about 30 points.
  - GPU time: about 5 hours (8 jobs).
