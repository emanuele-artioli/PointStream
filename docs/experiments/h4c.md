# Experiments: H4c

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-16 — H4c: handled objects as pixels, H3d's recipe
- Question ([PLAN H4c](../../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08);
  user, 2026-10-13). H4a and H4b found no motion representation for handled
  objects, so they stay coded pixels, but no component codes them yet: in
  the pipeline the background fills their region. Hands are coded by
  DCVC-UF HT-L off the shelf at ¼ and ½ plus G, a generic super-resolution
  network fitted on hand crops (594 k parameters; −22% at equal LPIPS on R,
  H3d). Does the same recipe code handled objects? Three arms, each scored
  on hand pixels and on handled-object pixels separately:
  1. **G as it is** (the hands' network applied to the objects);
  2. **G_O**, G's network refitted on object crops;
  3. **G_HO**, refitted on hands and objects together (one network, the same
     size).
  The user's concern: hands are alike and objects are not, so one enhancer
  may serve neither; if hands lose with the shared network, keep two.
  Fallback if all fail: generative completion of unseen sides, reported,
  not built.
- Regions. O is H1's handled-object part on the fill set: per frame, each
  object label's union of fill-mask instances painted in instance order onto
  free pixels, *handled* when a human labelled it in contact with a hand at a
  keyframe bounding or inside the window (H1's `contact_objects`), then the
  hand masks painted over (hands win). So O excludes R, and H1's per-part
  bit accounting of B2's SVT-AV1 streams on `handled_object` is the rate on
  O. R is H3c's (both hand classes of the fill set).
- Items: B2's 4 pilot items, 240 frames (H3d's). Handled objects there are
  small (fill set: 0.2–7% of the foreground's pixels per item, H1), so every
  score reports O's pixels per frame and the frames with O.
- Bar on O, per item: the upper concave hull (`g5.envelope`) on (kbps on O,
  −LPIPS_O) of SVT-AV1 (B2's CRF 41/48/55/62 streams, rate from H1's
  `handled_object` kbps; H3b2's seven low points re-encoded with H3b2's code,
  rate on O by H1's libaom accounting) and DCVC-UF HT-L off the shelf (full,
  ½ and ¼ resolution × QP 0, 21, 42; rate on O by its own entropy model, as
  H3c). Quality as H3c on R, with O for R: the decode inside O, the source
  outside, full-frame LPIPS averaged over O; PSNR_O; masked DISTS on O's box
  plus 64 px; flow-free flicker on O.
- Stage 1, one code path (`experiments/visor/h4c.py run`), every point
  scored on O and on R:
  - the bar's twenty points (base decodes);
  - at DCVC-UF's six ¼ and ½ points: **arm 1**, G's network for that point
    (H3d's generic jobs `20261012T221222Z-e514f83b`, `…225257Z-99094ccb`)
    applied on O's box (and on R's, reproducing H3d's G scores on R as the
    check); and the **oracle**: G5d's enhancer (L, H3d's settings, 2,000
    steps, MSE plus LPIPS) fitted on the window's own O, its bits free. It
    bounds arms 2 and 3 on objects: any network of that size fitted outside
    the window sees the scored frames less.
- Stage 2 (only if the oracle passes or is unclear; footage specified in
  this entry before its runs): G_O and G_HO fitted on recordings of
  participants other than P01, P02, P03 and P09, one network per point.
  Object masks there come from VISOR's dense train interpolations (objects
  a human marked in contact, O's definition), not SAM 3.1: SAM has no prompt
  for "the object in the hand", and the masks only shape the training loss.
  G_HO adds those object crops to G's own hand footage (SAM 3.1 hand and arm,
  H3d), same steps and size. Then one scoring job per item applies G, G_O
  and G_HO at the six points, on O and R.
- Test curves: each arm's Pareto front over its six points, as H3d. BD-rate
  per metric against the bar (PCHIP of log rate over the overlap, at least
  0.005 LPIPS; otherwise better or worse at every rate). Saving = −BD.
- Decision rule (fixed now; `h4c.DECISION`).
  - An arm *stands* on a region by H3d's rule (`h3d.DECISION`) against that
    region's bar: mean BD-rate on LPIPS ≤ −10% with at least 3 of 4 items
    below 0; *fails* at mean ≥ 0 or at most 1 item below 0; otherwise
    unclear. The oracle is judged the same way on O.
  - Stage 1. If the oracle fails on O, arms 2 and 3 are not built: objects
    are coded by DCVC-UF + G if arm 1 stands on O, otherwise by DCVC-UF off
    the shelf without an enhancer, and generative completion is reported as
    the remaining route. If the oracle passes or is unclear, stage 2 runs.
  - Stage 2. The object network is the first standing arm on O in the order
    G, G_HO, G_O (one network before two; a shared one before a second),
    unless a later standing arm beats it by 10% (its mean BD-rate against
    the earlier arm's points on LPIPS_O ≤ −10%). Hands keep G unless the
    object network is G_HO and G_HO keeps the hands: it stands on R and its
    mean BD-rate against G on LPIPS_R is below +10%; then G_HO serves both.
    If the object network is G_HO but it loses the hands, or is G_O, there
    are two networks (G on hands). If no arm stands on O, objects are coded
    by DCVC-UF off the shelf and generative completion is reported.
  DISTS and PSNR are reported beside and do not decide; a sign disagreement
  with LPIPS is stated with a visual check.
- Hypothesis. An enhancer restores local texture lost by DCVC-UF's
  downscaling (H3d: PSNR held on DCVC-UF, so it restores rather than
  invents) and G5d's enhancer transferred across kitchens, so: the oracle
  passes on O (about −50%, less than the hands' −77% since O is small and
  often blurred); G as it is helps objects a little (−5% to −15%, unclear);
  G_O stands (about −20%); G_HO matches G on hands within 5% and G_O on
  objects within 5%, so one network serves both, against the user's concern.
- Competing explanations. (1) The user's: objects vary (glass, packaging
  with print, metal, food), so a generic network gains little on them, or
  learns a hand prior that recolours them (K's gloves in H3d); the visual
  check of each arm on each item shows it. (2) O is small and fast (H4b:
  views die in 0.03 s), so LPIPS_O averages few pixels and per-item results
  swing; pixel counts and per-item BD-rates are reported, and the bar
  itself may be flat. (3) The rate split onto a small region is coarse
  (16×16 latent cells, 8×8 libaom blocks): the whole-frame rates are
  reported beside.
- Budget. Stage 1: per item two jobs (bar: SVT-AV1's eleven points and
  DCVC-UF at full resolution; dcvc: the six ¼ and ½ points with G and the
  oracle), 8 jobs, each ≤ 45 minutes shared on Ada or A6000 at 32,000 MiB
  (H3c's anchor jobs took 12–18 minutes, H3d's oracle jobs 11–28), about
  4 GPU-hours; smokes ≤ 600 s on one item, 48 frames. Stage 2: footage,
  two network jobs and four scoring jobs, about 4 GPU-hours. Ceiling 10
  GPU-hours.
- Stage 2's footage, as fixed above, before its runs (2026-10-16, after
  stage 1). The object footage takes recordings of participants other than
  P01, P02, P03 and P09 that have VISOR dense train interpolations, ordered
  by sha256 (`pointstream-h4c:objects:<video>`); the first 8 were staged
  (`h4c-2026-10-16/inputs/visor-train-objects.tar`, `1afb9c61…ac49`) and the
  first 6 with a placeable segment used: P06_11, P04_110, P35_109, P23_02,
  P06_103, P26_110 (5 participants). Segments of 48 frames lie inside a
  dense run in which a human marked an object in contact at a keyframe of
  the run, placed by sha256, every frame within VISOR's drift of 1 (the
  training admission of 2026-10-06); O is painted from the dense masks as on
  the evaluation windows. G_HO's hand crops are G's own footage
  (`20261012T213113Z-6ab41492`, SAM 3.1 hand and arm) unchanged. Both
  networks: G's size and settings (8 × 64, 594 k parameters, 2,000 steps,
  MSE plus LPIPS), one per point.
- Jobs (shared; validators passed on each smoke, and every full stage was
  validated locally on its `published.tar`, all checks passing; code
  `35378a6` from each `ready.json`, except P01_107's first job at `1148735`,
  whose stage-1 path is the same). Specs and inputs:
  `pointstream-data/visor/h4c-2026-10-16/` (`inputs-867fac311ae5.json`).
  - Stage 1, bar and arm 1 and the oracle: P01_107 ¼ and ½
    `20261016T090326Z-1acfc638`, SVT-AV1 points `…091919Z-4ef71e59`,
    full-resolution DCVC-UF `…091932Z-9daaf74e`; P02_02 ¼ `…091946Z-fa663164`,
    ½ `…091959Z-e415c0f4`, SVT-AV1 `…145448Z-0d0266a4`, full `…092206Z-c1a801de`;
    P03_10 ¼ `…092018Z-adafb13c`, ½ `…092033Z-64e75829` (two points; its
    `partial.tar`) and `…150851Z-693e0958` (QP 42), SVT-AV1 `…092140Z-7042c665`,
    full `…145524Z-da2bf9a5`; P09_106 ¼ `…092057Z-a6428a3e`, ½
    `…092112Z-97be19d1`, SVT-AV1 `…092152Z-ca41eff5`, full `…145545Z-1f898699`.
  - Stage 2: object footage `20261016T145818Z-74b36eb7` (CPU); networks ¼
    `…152552Z-c3d9ed14`, ½ `…153801Z-03f2a819`; scoring P01_107
    `…160611Z-c5a96d1f` (four points, `partial.tar`), `…184512Z-ceb4f695`,
    `…184525Z-29be19be`; P02_02 `…162052Z-cb586f84`; P03_10
    `…160637Z-a26a7146` (four points, `partial.tar`), `…203231Z-76b386fa`,
    `…203256Z-b78da953`; P09_106 `…160653Z-d4c46dfa` (three points,
    `partial.tar`), `…185017Z-99faa39c`, `…203243Z-ec8fa4db`,
    `…203218Z-c41d6e67`.
  - Failed attempts, not evidence: DCVC-UF decodes that disagreed with
    themselves (the code stops on them) at ½ resolution on shared RTX 6000
    Ada GPUs, eight times on gpu5 and gpu6 (QP 21 four times, QP 42 twice,
    QP 0 twice, one of them in the ½ networks job's footage); never on the
    A6000s, where the last
    points ran (as K5 recorded); two CUDA out-of-memory errors from another
    user's processes; one memory overrun (SVT-AV1 points of P02_02, 20,732
    MiB against 16,000 declared; rerun at 24,000); two smokes over 600 s
    on contended GPUs (rerun with 120-frame smokes). Jobs declaring 60 GiB
    of local storage (copied from H3d) were not admitted on the A6000 hosts,
    whose RAM roots are mostly the fleet's input cache; 20 GiB was admitted.
  - Report (`h4c report`, code `076ced7`, which only merges stage 2's rows
    into stage 1's units of the same name):
    `pointstream-data/visor/h4c-2026-10-16/report-076ced7/h4c-report.json`
    (`d4f10117…b9d8`), with the visual check `p02_dcvc21_4_f120.png`. On R
    the base points and G reproduce H3d's bar and G within the rule's
    tolerance (`cross_check_holds`), and G on R gives H3d's −22% exactly.
- Outcome. Handled objects in the pilot windows (H1's definition): soy milk
  (P01_107), glass and tap (P02_02), pancake and its package (P03_10), fork
  (P09_106, on 142 of 240 frames); 5,700–60,600 pixels per frame. BD-rate
  on LPIPS against each region's bar (mean over 4 items, 95% bootstrap CI;
  items below 0), DISTS and PSNR beside:

  | arm | LPIPS_O | items | DISTS_O | PSNR_O | LPIPS_R (hands) | items |
  |---|---|---|---|---|---|---|
  | oracle (fitted on the window) | −92% (−96 to −88) | 4 | −78% to −88% | −7% to −61% | — | |
  | G as it is | +10% (−19 to +45) | 1 | −32% to +38% | +25% to +147% | −22% (−35 to −10) | 4 |
  | G_O | −45% (−55 to −34) | 4 | −2% to −58% | +16% to +51% | −1% (−27 to +33) | 3 |
  | **G_HO** | **−39% (−48 to −28)** | **4** | −2% to −56% | +10% to +42% | **−33% (−48 to −18)** | **4** |

  Per item on O, G_HO: −23%, −44%, −36%, −51% (P01_107, P02_02, P03_10,
  P09_106); G_O −27%, −55%, −46%, −54%; G +5%, +58%, −27%, +6%. Pairwise
  (mean BD-rate on LPIPS): G_O against G_HO on O −1.9%; G_HO against G on
  O −28%, on R −8%. Flicker on O within 4% of the base's for every arm.
- Decision (by the rule): **stage 1** — the oracle passes, G as it is fails
  on objects, so stage 2 ran. **Stage 2** — the first standing arm on O in
  the order G, G_HO, G_O is G_HO; G_O does not beat it by 10% (−1.9%). G_HO
  stands on R and is better than G on the hands (−8%, below the +10%
  allowance): **one network, G_HO, codes hands and handled objects**, over
  DCVC-UF HT-L off the shelf at ¼ and ½. It replaces G on the hands too.
- Visual check (P02_02, frame 120, DCVC-UF ¼ at QP 21, 1.1 kbps on O). G as
  it is turns the chrome tap pink-brown, a skin prior from its hand-only
  training: that is its +58% on P02_02 and its PSNR loss. G_O and G_HO keep
  the tap grey and sharpen the glass's and the tap's edges toward the source,
  with no texture the source lacks; the oracle is closer still (it has seen
  this glass).
- Reading. The hypothesis held in direction and was wrong in size: G as it
  is does not help objects (+10%, not −5% to −15%) because it recolours them
  toward skin, and the refitted networks gain more (−39% to −45%, not about
  −20%). The user's concern (hands are alike, objects are not, so one
  enhancer may serve neither) does not hold on these items: a network that
  has seen objects as well as hands is as good on objects as one that has
  seen only objects (within 2%) and better on hands than G (8%), probably
  because the object footage adds more varied texture and colour than G's
  six recordings of hands alone. What does not transfer is a network that
  has seen only one of the two: G recolours objects, G_O loses the hands on
  P09_106 (+52%). As for the hands, most of the oracle's gain is fitting the
  scored window (−92% against −39%); and the gain is perceptual, PSNR on O
  is worse for every refitted arm (+10% to +51%), as G's was on hands.
  Competing explanation (2), small and fast objects: P09_106's fork (5,700
  pixels, 142 frames) is the smallest region and the arms rank there as
  elsewhere. Generative completion (the fallback) is not needed.
- Scope. 4 pilot items, 6 handled objects; one object footage (6
  recordings, 5 participants, 691 frames with O); masks from VISOR's dense
  interpolations for the footage and the fill set for scoring; one network
  size and schedule. The order G, G_HO, G_O in the rule preferred one network
  before two; G_O's 1.9% edge on objects is inside the noise of 4 items.
- GPU time: about 11 GPU-hours (7.0 in finished full stages, the rest smokes
  and the failed attempts above), **above the 10 allowed**: the excess is
  the reruns of non-deterministic decodes on shared Ada GPUs. Shared jobs'
  timings are not evidence.
