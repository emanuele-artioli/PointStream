# Experiments: H5q

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-20 — H5q 1: the venue model codes the foreground (equal adaptation)
- Question (PLAN H5q, lever 1). At equal adaptation, neither model billed,
  the venue-adapted whole-frame DCVC-UF beats PointStream on weighted
  LPIPS (+16%, 1 of 4; H5p 3). Where does PointStream lose, and does
  adapting its foreground coder the way the anchor is adapted close it?
- Where it loses (H5p 3's report, `matched`: PointStream's best pair at
  or under each anchor point's rate, per region, LPIPS on the fill masks):
  - Below about 200 kbps (the scene route) PointStream's foreground is
    already better than the venue anchor's (P01_107 at 92 kbps: 0.102
    against 0.154; P09_106 at 128 kbps: 0.175 against 0.263) and its
    background worse (0.336 against 0.197; 0.397 against 0.234): the
    hull spends the rate on F, as the 0.7 weight asks.
  - Above it, the foreground loses (P01_107 at 504 kbps: 0.062 against
    0.035; P02_02 at 1,145 kbps: 0.125 against 0.063). At QP 63 and full
    size, PointStream's foreground equals whole-frame DCVC-UF **off the
    shelf** at the same QP (0.038 | 0.036, 0.082 | 0.082, 0.096 | 0.096)
    and the venue model halves it (0.024, 0.044, 0.048). So the padding
    outside F costs nothing, and the high-rate gap is the adaptation alone:
    the anchor's model was fitted on the video's own frames, hands
    included, and PointStream's foreground coder was not.
- Method. The cheapest form of the lever needs no training: each item's
  venue model (H5p 3 (b): recipe L on the own video's whole frames outside
  the window, `train --fill none`, the very checkpoint the anchor codes
  with) replaces DCVC-UF HT-L's video model in PointStream's foreground
  coder (`h5 foreground --video-arm`, new; the intra model stays the
  public one, as in the anchor). Everything else is G_HO's PointStream:
  the ten foreground points, G_HO in both slots at ¼ and ½ (fitted on off-
  the-shelf decodes, so a mismatch here, stated), the backgrounds as
  published (scene model, enhancer). Scored by `h5 score --tag venuefg` on
  each item's pairs of G_HO's hull from H5p 3 (14–15 per item), the
  foreground swapped for the venue-coded one at the same point; fixed now,
  before any score. A subset of pairs gives a hull no better than the
  full grid would, so a pass is not inflated by the choice; neighbours
  are added only if the answer is unclear (lever 2).
  - This is the component itself, not an oracle: it costs less than the
    memorization oracle (no training), which is kept for the fine-tune on
    foreground input that would follow a partial result.
  - PointStream now holds two venue models (the scene model on filled
    footage, the whole-frame model) against the anchor's one. Venue models
    are sent beforehand and unbilled (user, 2026-10-20); payback hours are
    reported for both. A one-model PointStream (the whole-frame model also
    coding the filled background) is a later variant if this passes.
- Decision rule, fixed now: rule (a) on weighted LPIPS (fill), variant
  `venuefg` against `dcvc-venue` alone, neither billed: mean BD-rate −10%
  or better and 3 of 4 items below zero. Also reported, as the H5q target
  asks: against the fair pooled envelope (SVT-AV1 preset 1 with the
  ladder, DCVC-UF off the shelf with the ladder, `dcvc-venue`), unbilled;
  weighted DISTS, frame LPIPS and DISTS, weighted PSNR beside them.
  - Passes: H5q's target is met on LPIPS; the split wins at equal
    adaptation (still with F granted; H5p 4 next).
  - Fails, with the high-rate foreground gap closed (fg LPIPS at QP 63
    within 10% of the anchor's): what remains is the low-rate background,
    so lever 2 (allocation grid) or the background's own coder is next.
  - Fails, with the gap open: the adaptation does not reach the padded
    foreground input; the fine-tune on foreground input (lever 1 proper,
    memorization oracle first) decides whether the lever is dropped.
- Hypothesis: at full size the foreground's LPIPS falls to the anchor's at
  the same QP (about −40%); at ¼ and ½ less, since G_HO saw off-the-shelf
  decodes. Weighted LPIPS against `dcvc-venue` moves from +16% to between
  −5% and −15%: a likely pass on 3 of 4, P02_02 (+35%) the hardest.
- Competing explanation: the venue model's gain lives in what it memorized
  outside F (the kitchen), and the padded foreground input, mostly flat
  push-pull fill, gets little of it; or G_HO after a venue decode loses
  more at ¼ and ½ than the coder gains.
- Budget: 8 foreground jobs (per item: the six G_HO points; full size and
  QP 63 at ½), about 12 minutes each on a shared GPU; 4 scoring jobs of
  14–15 pairs, about 15 minutes each. About 3 GPU-hours, ceiling 5; every
  stage under 45 minutes.
- Runs (shared GPUs; every full stage validated locally; code `ef8870d`
  for the first job, `53c122a` for the rest, from each job's `ready.json`).
  - Foreground (`h5 foreground --video-arm`), the six G_HO points and the
    full-size points per item: P01_107 `20261020T185116Z-f7515340`,
    `…183312Z-66524402`; P02_02 `…185132Z-0ba747bd`, `…190910Z-92937293`;
    P03_10 `…185205Z-a530055a`, `…191021Z-85a8bed8`; P09_106
    `…185227Z-7a3a9678`, `…191028Z-ce3fdc03` (RTX A6000; the rest RTX 6000
    Ada). Local validation 11/11, except one sanity check on two jobs: F's
    PSNR at `dcvc@0/4` is 19.07 dB (P01_107, 3.6 kbps) and 19.45 dB
    (P09_106, 4.6 kbps), under the validator's 20 dB "a picture" floor.
    A P01_107 frame was inspected: a low-rate picture, not a broken decode.
    The venue model trades PSNR for LPIPS at every point (QP 21 full size
    on P01_107: 28.9 dB against the off-the-shelf 30.9). The floor was not
    changed.
  - **The venue model fails to decode at QP 63 at ½** on P02_02 (decode
    unlike its encode, 3 of 3 attempts, `…185151Z-2b315029`) and P03_10
    (decoder segfault, 3 of 3, `…185221Z-036bb1e3`); P01_107's passed. On
    no hull pair, the point was dropped and those full-size jobs rerun
    without it (P09_106's cancelled before reaching it,
    `…185234Z-5230ec95`). A coder that cannot decode one of its operating
    points is a defect to state if this route is kept.
  - Scoring (`h5 score --tag venuefg`, 14–15 pairs): P01_107
    `20261020T214217Z-bc5cddf0`, P02_02 `…214224Z-5179bf35`, P03_10
    `…214230Z-2d47185e` (RTX A6000), P09_106 `…214236Z-990c7806`; 9/9
    each, locally. The report is H5p 3's (its inputs reproduce H5p 3's
    table exactly) plus these rows.
- Outcome. BD-rate, P01_107 | P02_02 | P03_10 | P09_106, neither model
  billed:

  | PointStream against | weighted LPIPS | mean (CI) | < 0 | weighted DISTS | frame LPIPS | frame DISTS | weighted PSNR |
  |---|---|---|---|---|---|---|---|
  | `dcvc-venue`, G_HO (H5p 3) | +13 \| +35 \| +27 \| −9% | +16% (0 to +31) | 1 | −4% (2/4) | +15% (0/4) | −17% (4/4) | −11% (3/4) |
  | **`dcvc-venue`, venue-coded foreground (the rule)** | +22 \| +30 \| +30 \| −8% | **+19% (+2 to +30)** | **1** | −14% (3/4) | +18% (0/4) | −18% (4/4) | +3% (2/4) |
  | fair pooled, venue-coded foreground | +28 \| +29 \| +31 \| −5% | +21% (+3 to +30) | 1 | −8% (3/4) | +18% (0/4) | −14% (4/4) | +69% (0/4) |
  | `dcvc-venue`, either foreground per point (post hoc) | +8 \| +19 \| +12 \| −15% | +6% (−9 to +16) | 1 | −15% (4/4) | +12% (0/4) | −22% (4/4) | −11% (3/4) |

  - **The rule fails** (+19%, 1 of 4).
  - **The high-rate foreground gap is closed** on every item: at the venue
    anchor's QP 63, PointStream's foreground LPIPS is 0.026 | 0.044 |
    0.050 | 0.050 against the anchor's 0.024 | 0.044 | 0.048 | 0.050
    (G_HO's: 0.038 | 0.082 | 0.096 | 0.084).
  - **The low rates got worse.** At ¼ and ½, G_HO after a venue decode
    loses to G_HO after an off-the-shelf decode (P01_107 at 45 kbps:
    foreground 0.226 against 0.171; P09_106 at 128 kbps: 0.253 against
    0.175). G_HO was fitted on off-the-shelf decodes: the competing
    explanation, for the low rates only.
  - Post hoc, not the rule: an encoder choosing per point between the two
    foreground coders (both sent beforehand; venue-coded on the hull at
    full size, off the shelf at ¼ and ½) reaches +6% (1 of 4). Weighted
    DISTS −15% (4 of 4) and frame DISTS −22% (4 of 4).
  - Hypothesis partly right: the full-size foreground reached the anchor,
    the weighted-LPIPS margin did not move, because the foreground was not
    what remained.
- Where the loss is now (a diagnostic on the post hoc rows, not evidence
  of a component). Weighted LPIPS is exactly 0.7 × foreground + 0.3 ×
  background (to 1e-16). Giving PointStream the venue anchor's
  **background** LPIPS at the same total rate, keeping its own
  foreground: −12 | −4 | −8 | −30%, mean −13%, 4 of 4 (the rule would
  pass). Giving it the anchor's **foreground** instead: +1 | −7 | −2 |
  +3%, mean −1%, 2 of 4. At equal adaptation the split's loss on LPIPS is
  the background stream.
- The background stream is not worse per bit. Against the venue anchor's
  whole frame at the same rate, its background LPIPS is lower below
  about 300 kbps (P01_107 at 93 kbps: 0.164 against 0.196; P02_02 at 215
  kbps: 0.104 against 0.202) and within about 0.01–0.03 above. What the
  split loses is structural: it pays for the background stream and for F
  again in the foreground stream, where the anchor pays once. The per-item
  loss follows F's share of the frame: P02_02 50% (+19%), P03_10 19%
  (+12%), P01_107 14% (+8%), P09_106 2% (−15%).
- The fill lever, estimated before any run (CPU, `fillfree.py`; an
  estimate, not a bound, not evidence). The background stream codes F's
  ProPainter fill, which the foreground then overwrites. Cutting the
  background rate by its bits for F, taken as F's area share times a
  factor, quality unchanged: factor 0.5, +1% (1 of 4); factor 1 (F free),
  −5% (2 of 4: +2 | −7 | 0 | −17). Even a free fill leaves the rule
  unmet, so per the protocol it is not built for this decision. Lever 2
  (a denser grid) adds a few points at most; lever 3 matters only once F
  is billed.
- **Reading for H5q (2026-10-21):** at equal adaptation, with the levers
  measured or estimated, the split does not reach rule (a) on weighted
  LPIPS. It wins on DISTS (weighted −15%, frame −22%, 4 of 4, post hoc
  per-point choice) and on weighted PSNR (−11%, 3 of 4), and it wins
  where F is small (P09_106). What the egocentric claim becomes is the
  user's decision.
- GPU time: about 4.5 GPU-hours in all (12 jobs, the 3 failed or cancelled
  full-size attempts included), within the ceiling of 5.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5q1-53c122a/`
  (`h5 report` at `53c122a` over H5p 3's sources plus the four scoring
  stages; the job specs, the scoring stages' `h5.json`, and the post hoc
  `union.py`, `decompose.py` and `fillfree.py`).

### 2026-10-21 — H5q 2: what the background stream spends on the filled F (oracle)
- Question (user, 2026-10-21: measure the free fill before closing H5q).
  The background stream codes the whole frame with F filled by ProPainter,
  and the foreground then overwrites F. H5q 1's estimate took the fill's
  bits as proportional to F's area: even a free fill reached only −5% (2
  of 4). ProPainter's fill may flicker, though, and cost more than its
  area (P02_02: F is 50% of the frame). What does the fill really cost,
  and would a free one let the split pass at equal adaptation?
- Method (`g6 fill-cost`, new). Per item, every background point on the
  union hull of H5q 1 (9–10 per item: scene model at 540p and 1080p,
  SVT-AV1 with G6's enhancer), coded twice through G6's own coder: with
  ProPainter's fill (as published) and with a static plate in F (the
  window's per-pixel temporal median of the filled frames, constant in
  time, so the codec predicts it almost for free). Both decodes scored on
  V (`g5.score`, the dataset tier); SVT-AV1's also after the published
  enhancer of that CRF, re-applied. Then each hull row is recomputed:
  background rate × (static bytes / ProPainter bytes) of its point,
  background LPIPS + (static − ProPainter) LPIPS on V, foreground
  unchanged, weighted LPIPS = 0.7 fg + 0.3 bg (exact, H5q 1).
  - Why it is an oracle and not the component: it keeps the scored rows'
    composites and moves only the background's rate and its V quality by
    measured amounts; the composite's background region is V up to the
    8 px ring. The real component (the static fill in `format` and
    `enhance`, every pair rescored, about 7 GPU-hours) runs only if this
    passes.
  - The configuration is H5q 1's post hoc per-point choice of foreground
    coder (off the shelf with G_HO at ¼ and ½, the venue model at full
    size), the most favourable reading so far. It is fixed now, before
    this run, as the configuration a real run would be judged on.
- Decision rule, fixed now: rule (a) on weighted LPIPS (fill) against
  `dcvc-venue` alone, neither billed: mean −10% or better and 3 of 4 items
  below zero.
  - Passes: the static fill is built into G6's routes and every hull pair
    is rescored, judged by the same rule.
  - Fails: H5q closes on LPIPS; the egocentric claim is reframed (the
    user's decision), with the fill's measured cost reported.
- Hypothesis: the fill costs more than its area where F is large and
  moving; P02_02's background rate drops 40–60%, the others 10–25%; the
  oracle reaches −5% to −12%, a borderline pass at best.
- Competing explanation: the codecs already predict ProPainter's smooth
  fill cheaply, so it costs less than its area and the oracle stays near
  or above H5q 1's area estimate (−5%).
- Budget: 8 jobs (per item: the scene-model points; the SVT-AV1 points),
  each point coded twice, about 25 minutes each on a shared GPU; about 3
  GPU-hours, ceiling 4; every stage under 45 minutes.
- Runs (`g6 fill-cost`, code `37db6aa` per each job's `ready.json`; RTX
  6000 Ada; every full stage validated locally, 10/10). P01_107: scene
  540p `20261021T154938Z-b665eb20`, scene 1080p `…162327Z-c71e1490`,
  SVT-AV1 `…155730Z-18f6ee72`. P02_02: scene `…155736Z-5d7e5688`, SVT-AV1
  `…155717Z-f071aac2`. P03_10: scene `…155743Z-d88d653e`, SVT-AV1
  `…163549Z-e533eed5`. Not citable: a first P02_02 SVT-AV1 job stopped by
  a GPU-memory overrun (19.9 GB against 16 GB declared;
  `…154945Z-2eb030d6`, smoke passed), and two smokes over 600 s on the
  contended A6000 gpu2 (`…155723Z-7ead6dae`, `…155749Z-3a4b785c`), rerun
  on Ada. P09_106's two jobs were cancelled once the decision was made
  (below): `…163556Z-79197121`, `…163602Z-be5dbc75`.
- The fill's measured cost (static plate bytes over ProPainter's, at the
  same point; LPIPS on V moved by at most 0.007 anywhere):

  | Item (F share) | scene model | SVT-AV1 (enhancer re-applied) |
  |---|---|---|
  | P01_107 (14%) | 0.98–1.04 | 1.02–1.03 |
  | P02_02 (50%) | 0.83–0.87 | 0.61–0.71 |
  | P03_10 (19%) | 0.99–1.02 (0.90 at 1080p QP 63) | 0.93–1.01 |

  - The static plate is not free where F moves: pixels switching between
    moving background and a frozen plate cost bits. ProPainter's
    temporally coherent fill is cheap for the scene model; SVT-AV1 pays
    for it only where F is large (P02_02).
- Outcome. Oracle BD-rate on weighted LPIPS against `dcvc-venue`, neither
  billed (the same rows unchanged reproduce H5q 1's per-point reading
  exactly, +8 | +19 | +12%): P01_107 **+9%**, P02_02 **+11%**, P03_10
  **+13%**; P09_106 not measured (−15% unchanged, F 2%).
  - **The rule fails**: three items stay above zero, so at most 1 of 4
    can be below zero whatever P09_106 measures. Decided at 3 of 4 items;
    P09_106's jobs were cancelled.
  - Hypothesis wrong except on P02_02 (40–60% predicted, 13–39% measured);
    the competing explanation holds for the scene model everywhere and for
    SVT-AV1 where F is small. The free fill does not exist in this form,
    and even the area-proportional bound (H5q 1: −5%, 2 of 4) failed.
- **H5q closes on LPIPS (2026-10-21).** At equal adaptation the split
  does not beat the venue-adapted whole-frame DCVC-UF on weighted LPIPS:
  +16% as built, +6% at best per point, still above zero with the fill's
  measured cost removed. It wins on weighted DISTS (−15%, 4 of 4), frame
  DISTS (−22%, 4 of 4) and weighted PSNR (−11%, 3 of 4), and on LPIPS
  where F is small (P09_106, 2%: −15%). The egocentric claim is the
  user's to reframe.
- GPU time: about 2.5 GPU-hours (7 complete jobs, the failed and cancelled
  ones included), within the ceiling of 4.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5q2-37db6aa/`
  (the stages' `g6.json`, `oracle.py` and its `oracle.json`, the specs).
- Follow-up analyses (CPU, 2026-10-21; not new runs).
  - Every metric, the per-point configuration against `dcvc-venue` | the
    fair pooled envelope: weighted LPIPS +6% (1/4) | +8% (1/4); frame
    LPIPS +12% (0/4) | +11% (0/4); weighted DISTS −15% (4/4) | −9%
    (3/4); frame DISTS −22% (4/4) | −18% (4/4); weighted PSNR −11% (3/4)
    | +43% (1/4); frame PSNR +5% (1/4) | +85% (0/4). DISTS wins against
    every anchor; PSNR wins only against the venue codec (SVT-AV1 at
    preset 1 holds the pooled PSNR hull); LPIPS loses against every
    anchor.
  - F's share of the frame over all 34 windows of evaluation set v2
    (`visor_dense_sam_fill`, every 8th frame; archive `049977f0…`):
    median 24%; 1 window under 5% (P09_106, 2.3%), 3 under 10%, 12 over
    30% (up to 77%). The pilots' LPIPS reading crosses zero between F =
    2% (−15%) and 14% (+8%). A small-F scope would cover about 3 of 34
    egocentric windows; broadcast sports, where the players are a few
    percent of the frame, are the natural small-F domain (K6).
