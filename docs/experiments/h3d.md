# Experiments: H3d

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-12 — H3d oracle: a content-aware super-resolution of the hand pixels (NAS)
- Question (PLAN H3d; user, 2026-10-11). NAS (Yeo et al., OSDI 2018) codes
  the video at a low resolution and ships a small super-resolution network
  trained on that video, its bits charged once. Coded at ¼ or ⅛ of the
  resolution, the hand pixels reach the pose stream's rate (H3b2; DCVC-UF at
  ¼ reaches 0.3–3 kbps on R, H3c). Does a network trained on the video's own
  hands lift those decodes above H3c's bar: the per-item upper concave hull
  of SVT-AV1 and DCVC-UF off the shelf on R, at equal LPIPS?
- Oracle: the network fitted on the scored window itself, its bits free.
  The component fits it on the video it streams too, but must pay for it
  (amortized over the video, as PLAN I's payback) and sees more frames than
  the window, so it fits each scored frame less closely. A fit that starts
  as the identity on the scored frames' own (decode, source) pairs, without
  a size charge, is therefore a bound for any NAS-style network of the same
  size or smaller on these bases. It bounds only this configuration (one
  network per coded stream, one frame in, its size); what to change after a
  failure is stated with the outcome.
- Method (`experiments/visor/h3d.py`, `oracle`). Per pilot item (B2's four,
  240 frames, R as in H3c: both hand classes of the fill set, the visible
  forearm included):
  - Bases, re-coded with H3c's code and settings. *dcvc*: DCVC-UF HT-L off
    the shelf at ¼ and ½ of the resolution, QP 0, 21 and 42 (six points,
    rate on R by its own per-cell estimate). *svt*: SVT-AV1 at CRF 55 and 63,
    ¼ and ⅛ of the resolution (H3b2's four points, rate on R by H1's libaom
    accounting; streams checked against H3b2's sha256). Each decode is
    upscaled bicubically to 1920×1080, as H3c scores it.
  - Network, one per base point: G5d's enhancer (`g5d.enhancer`, EDSR-style,
    8 blocks of 64 channels, about 0.6 M parameters, 1.2 MB at fp16), which
    starts as the identity on the bicubic upscale, so step 0 is the base
    point itself. Fitted on (upscaled decode, source) patches of 128 px
    centred on R pixels of the window's hand frames, 2,000 steps, batch 16,
    Adam 2 × 10⁻⁴, cosine (G5d's settings), with G5d's passing loss: MSE on
    R plus LPIPS on R (the decode pasted into the source), at equal weight
    at step 0. MSE alone is not run: LPIPS decides, and G5d's MSE fits barely
    moved SVT-AV1.
  - Scoring: H3c's `Scorer`, unchanged (the output inside R, the source
    outside; LPIPS_R, PSNR_R, masked DISTS on R's box). Every base point is
    scored before and after its fit. The before scores and rates must
    reproduce H3c's units: within 0.0002 LPIPS for SVT-AV1 (byte-identical
    streams), and within 0.005 LPIPS and 5% rate for DCVC-UF (its streams
    differ across GPU classes). Beside, a flow-free flicker measure: on
    consecutive hand frames, the mean absolute luma of the output's change
    minus the source's change, on R in both frames.
- Bar, per item: the upper concave hull (`g5.envelope`) on (kbps on R,
  −LPIPS_R) of H3c's SVT-AV1 points (B2's CRF 41/48/55/62, H3b2's seven
  low points) and DCVC-UF off the shelf (nine points), from H3c's anchor and
  reference units. Each arm's curve is its Pareto front over its six (*dcvc*)
  or four (*svt*) fitted points, as the convention keeps the test's front.
- Comparison: BD-rate on LPIPS_R against the bar (PCHIP of log rate over
  the LPIPS range both span, at least 0.005; otherwise "better" or "worse at
  every rate"), and beside it on DISTS_R and PSNR_R, against the same
  points' hull on that metric. Saving = −BD. The weights' payback is
  reported per item: 1.2 MB over the arm's rate saving on R at the window's
  median LPIPS, against the item's video duration (ffprobe).
- Decision rule, fixed now (`h3d.DECISION`). Per arm (*dcvc*, *svt*):
  - **passes** if its mean BD-rate on LPIPS_R over the 4 items is −10% or
    better, with at least 3 of 4 items below 0. An item "better at every
    rate" counts below 0 and is left out of the mean; "worse at every rate"
    counts as not below 0 and blocks a pass. The mean is over the items with
    a number (at least two).
  - **fails** if its mean is at or above 0, or at most 1 item is below 0.
  - otherwise **unclear**.
  The oracle passes if either arm passes. Then the component is specified
  next (an entry of its own): the network fitted on the same video outside
  the window, its bits charged over the video, with network size as the
  pilot axis. If both arms fail, H3d is not built. If neither passes and
  either is unclear, the result goes to the user before anything is built.
  DISTS and PSNR do not decide. A sign disagreement with LPIPS is stated,
  with the visual check (crops of the source, the base and the fitted
  output at about the same rate).
- Hypothesis. *dcvc* passes narrowly. Fitted with LPIPS on the scored
  frames, the network restores fold and edge texture that DCVC-UF's ¼ and ½
  decodes smooth away: about −20% to −35% on LPIPS_R, PSNR_R near 0, and
  DISTS improving most (DCVC-UF's weak metric in H3c). *svt* improves on
  SVT-AV1's own points by a similar factor but stays behind the bar: the
  bar is DCVC-UF's, already 63% cheaper. Flicker rises (G5d: the network
  re-invents texture each frame).
- Competing explanations. For a pass: the network memorizes the window
  (0.6 M parameters, 240 frames of hand pixels). That is allowed in the
  oracle but not in the component, which is fitted outside the window. For a
  fail: at ¼ and ⅛ the decodes keep no hand detail a network could restore
  without inventing it. PSNR_R not moving, and LPIPS gains only at the
  highest-rate bases, would show it.
- Budget. Smoke ≤ 10 minutes (one item, one point of each base, 50 steps,
  asserting the fit's step 0 equals the base's scores). Oracle: one job per
  item and arm (8 jobs). Each is ≤ 45 minutes, shared Ada or A6000 at
  32,000 MiB, and checkpoints every finished point. That is about 5
  GPU-hours; ceiling 6.
- Inputs: H3c's (`h3c-anchor-<video>.json` staged inputs, re-hashed),
  manifest `pointstream-data/visor/h3d-2026-10-12/inputs-9b3b06a06e4e.json`;
  environment `pointstream-20261009T214002Z` (`32e68872…4661`).
- Jobs (code `852357d` for all, from each `ready.json`; shared). Validators
  12/12, including the step-0 network reproducing the base point exactly:
  on each smoke (the fleet's gate, which validates only the smoke) and,
  run locally on each full stage's `published.tar` and `h3d.json`, on the
  full run too (2026-10-13).
  *dcvc*: P01_107 `20261012T180806Z-2a4c0cf4` (RTX 6000 Ada), P02_02
  `…183021Z-d077eba0` (Ada), P03_10 `…183037Z-3ebf898f` (A6000), P09_106
  `…183051Z-3f80dcab` (Ada). *svt*: P01_107 `20261012T180816Z-08afa095`
  (A6000), P02_02 `…183107Z-daa8790f`, P03_10 `…183127Z-ce697c9e`, P09_106
  `…183140Z-5e2929db` (Ada). 11–28 minutes each; fits 67–122 s (shared,
  not evidence). Report (`h3d report`, code `fcf378b`, against H3c's anchor
  and reference units `20261011T220119Z-69f5f6d7`, `…223528Z-6ecdc738`,
  `…221908Z-8193ffb7`, `…220719Z-19637920`):
  `pointstream-data/visor/h3d-2026-10-12/report-oracle/h3d-report.json`
  (`12e02bc6…a71a9f`), with the crops below. The base points reproduce H3c's
  on every item: SVT-AV1's streams byte for byte (LPIPS within 7 × 10⁻⁵ per
  frame), DCVC-UF's with identical LPIPS and rate (6 of 24 streams differ
  in bytes, as across GPU classes).
- Outcome. BD-rate against H3c's bar (per-item concave hull of SVT-AV1 and
  DCVC-UF off the shelf on R):

  | arm | item | LPIPS_R | DISTS_R | PSNR_R | flicker (× base) | payback of 1.2 MB | video |
  |---|---|---|---|---|---|---|---|
  | DCVC-UF + SR | P01_107 | −88% | −80% | −28% | 1.04 | 286 s | 119 s |
  | | P02_02 | −69% | −72% | +3% | 1.08 | 717 s | 207 s |
  | | P03_10 | −89% | −83% | −3% | 1.06 | 443 s | 103 s |
  | | P09_106 | −64% | −60% | −19% | 0.94 | 2,340 s | 149 s |
  | | **mean** | **−77% (CI −89 to −66), 4 of 4** | | | | | |
  | SVT-AV1 + SR | P01_107 | −72% | −76% | +56% | 0.96 | 640 s | 119 s |
  | | P02_02 | +30% | −57% | +210% | 1.00 | — | 207 s |
  | | P03_10 | −52% | −80% | +226% | 1.01 | 3,016 s | 103 s |
  | | P09_106 | +19% | +4% | +132% | 0.92 | — | 149 s |
  | | mean | −19% (CI −62 to +24), 2 of 4 | | | | | |

  **The oracle passes** on the *dcvc* arm; *svt* is unclear. Payback: the
  seconds of the hand stream whose saving (at the test front's median
  LPIPS) equals 1.2 MB.
- Visual check (`p02_zoom.png`, `p09_zoom.png` beside the report; frame
  120 and 98). P02_02: DCVC-UF at ¼, QP 21 (2.75 kbps) with the fitted
  network is close to DCVC-UF at ½, QP 21 (8.5 kbps): the glove's folds and
  the glass rim come back, with no texture that the source does not have.
  P09_106: the network sharpens the sleeve's and the hand's outlines, with a
  faint contour at the sleeve's edge.
- Reading. Hypothesis wrong in size: a network fitted to the window lifts
  DCVC-UF's ¼ and ½ decodes by far more than expected (−77%, not −20% to
  −35%), and on DCVC-UF PSNR_R improves or holds too (−28% to +3%), so it
  restores the source rather than inventing texture; flicker is unchanged.
  On SVT-AV1 it is the other way round: PSNR_R gets much worse (+132% to
  +226%) and LPIPS_R gains on two items only, so it synthesizes texture over
  AV1's blocks, as G5d found on the background. Two things remain open
  for the component, both raised by the rule's competing explanation:
  - *Memorization.* 0.6 M parameters fitted on 240 frames of the scored
    hands. A fit outside the window answers it.
  - *Payback.* At 1.2 MB the network costs more than it saves within each
    pilot video (286–2,340 s against 103–207 s). It must shrink by about
    3–20 times, or be shared beyond one video (per person or kitchen, or
    one generic network sent with the decoder, as DCVC-UF's own weights are).
- GPU time: about 2.4 GPU-hours (8 jobs), within the 6 allowed.

### 2026-10-12 — H3d component: per-video, per-person and generic super-resolution of the hand pixels
- Question (user, 2026-10-12, after the oracle). The oracle's network
  (fitted on the scored window, bits free) beats H3c's bar by 77% on
  DCVC-UF's ¼ and ½ decodes, but at 1.2 MB it costs more than it saves
  within any pilot video. Which way of delivering it keeps the gain once it
  is fitted without the scored frames and its delivery is accounted for?
  Three ways (user, 2026-10-12):
  - **V, per video (NAS).** Fitted on the same video outside the window,
    sent with it: its bits are billed, over the video's duration.
  - **K, per person and kitchen (the hybrid).** Fitted on the same
    participant's other recordings (in EPIC-KITCHENS a participant films one
    kitchen), sent to the client once, before, and reused for every new
    video: not billed per video.
  - **G, generic.** Fitted on other participants' recordings, delivered
    with the decoder like DCVC-UF's own weights: not billed.
- Footage (`h3d footage`). Segments of 48 consecutive frames at the native
  rate, placed content-blind by sha256 (as `g5c.choose_segments`), at least
  5 s from a video's ends; for V also at least 5 s from the scored window.
  Only recordings of the evaluation's verified classes (1920×1080 at 50 or
  59.94 frames/s; 7 of the 179 videos on disk are not). V: 24 segments of
  the item's own video. K: 24 segments over the participant's other
  recordings on disk (P01: 9, P02: 18, P03: 17, P09: 4 videos), never the
  item's video. G: 24 segments over recordings of participants other than
  P01, P02, P03 and P09. K and G draw on at most 6 recordings each (chosen
  by sha256), 4 segments per recording (P09: 4 recordings, 6 each), so a
  job stages a few videos, not dozens. All arms
  therefore fit on the same 1,152 frames, about 4.8 windows. R in the
  footage is SAM 3.1's "hand" plus "arm" at G1's analysis size (as G1 and
  G5c step 3), upsampled to 1080p: what a sender's own segmenter would give
  it, since only the evaluation windows have dataset masks. Scoring stays on
  the dataset masks (H3c's R).
- Fit and score. As the oracle, on its passing arm only: DCVC-UF HT-L off
  the shelf at ¼ and ½, QP 0, 21 and 42, one network per point, 2,000
  steps, MSE plus LPIPS on R. The footage is coded at each point by the same
  code (its segments concatenated into one sequence per point) and upscaled
  bicubically. Sizes: G5d's network (L, 8 blocks × 64 channels, 594 k
  parameters, 1.19 MB at fp16) for every arm; for V also M (4 × 48, 169 k,
  0.34 MB) and S (4 × 24, 43 k, 0.09 MB), since its bits are billed. The
  scored window, its base points and the bar are the oracle's.
- Rate. K and G: the rate on R, as the bar. V: the rate on R plus the
  network's fp16 bits divided by the video's duration (NAS's amortization;
  the hand stream carries them). V without its bits is reported beside it,
  to separate memorization from payback.
- Decision rule, fixed now (`h3d.COMPONENT`). Each arm (V at its best size,
  K, G) *stands* by the oracle's rule against the bar: mean BD-rate on
  LPIPS_R −10% or better, with at least 3 of 4 items below 0. Then:
  - if G stands, H3d's coder is DCVC-UF with G's network (deployable
    everywhere, nothing per video); K is adopted instead only if it beats G
    by 10% or more (mean BD-rate of K against G on LPIPS_R), and V only if
    it beats both by 10%;
  - if G does not stand but K does, the coder is DCVC-UF with K's network
    (needs earlier footage of the person), V again only if 10% better;
  - if only V stands, V; if no arm stands, H3d is not built, and V without
    its bits says why (near the oracle: the bits; far: the oracle
    memorized).
  DISTS_R, PSNR_R and flicker are reported beside; a sign disagreement with
  LPIPS is stated with a visual check.
- Hypothesis. Without the scored frames, V-L keeps about two thirds of the
  oracle's gain (−50%) before its bits, but its bits (40–90 kbps at L, 3–7
  kbps at S over 1.7–3.5 minutes) exceed what it saves, so V fails at every
  size. K stands (−40%): same hands, sleeves and lighting. G stands too
  (−25%): most of the gain is undoing DCVC-UF's low-resolution blur, which
  is not specific to a person; so H3d's coder is DCVC-UF with G, unless K
  is 10% better.
- Competing explanations. For G failing while K stands: the gain is
  appearance-specific (skin, gloves, sleeves), which argues for the hybrid.
  For all failing: the oracle memorized the window, which V without bits
  shows. SAM's masks differ from VISOR's (forearm extent): they only shape
  the training loss; a K or G fitted on VISOR's train-split masks would be
  the check if they matter.
- Budget. Footage masks: 8 jobs (V and K per item) and 1 (G), SAM 3.1 at
  about 0.3 s per frame (about 6 minutes per arm), each ≤ 45 minutes on Ada
  or A6000, shared at 40,000 MiB (G5c's footage declaration). G's networks:
  2 jobs (the ¼ and the ½ points, 3 fits each). Scoring: 24 jobs, one per
  item and point (coding both footages at the point dominates: 2 × 1,152
  frames), each fitting V-L, V-M, V-S and K and applying G; each ≤ 45
  minutes, shared. About 10 GPU-hours; ceiling 14.
- Runs. Footage (code `35b864f`; V-P03_10 `2726bd2` after its smoke's
  two content-blind segments showed no hands and the old check failed it):
  G `20261012T213113Z-6ab41492`; K `…213122Z-705f4eeb` (P01_107),
  `…213132Z-4141a4f7` (P02_02), `…213147Z-e0d64447` (P03_10),
  `…213239Z-30a4ff37` (P09_106); V `…213300Z-9b9b1944`,
  `…213310Z-60e3c893`, `…223750Z-84cdb758`, `…213403Z-6fcf5062`. G's
  networks: ¼ `…221222Z-e514f83b` (`35b864f`), ½ `…225257Z-99094ccb`
  (`532ef12`). The first ½ job (`…221147Z-9760525e`) passed its smoke but
  fitted QP 21 on a non-deterministic DCVC-UF decode of the footage (shared
  gpu5); found by validating the full stage locally, discarded, and the
  footage coder now stops on such a decode, as H3c's does. Scoring: 24 jobs,
  one per item and point, P01_107, P02_02 and P09_106 at `5de4fd3` and
  P03_10 at `0b4bad3` (its first smoke had no footage with hands to fit; the
  smoke now takes the first segments with R); one P02_02 job was
  resubmitted after a non-deterministic footage decode stopped it. Job ids
  are in the report's `runs`. The fleet validates only each job's smoke;
  every full stage above was validated locally with the same checks (24 of
  24 scoring, 9 of 9 footage, 2 of 2 networks). Report (`h3d report
  --component`, code `9e3af03`):
  `pointstream-data/visor/h3d-2026-10-12/report-component/h3d-report.json`
  (`467be69f…6e00`). About 8–9 GPU-hours (from the jobs' execution
  records), within the 14 allowed.
- Outcome. BD-rate on LPIPS_R against the bar (V billed its network over
  the video; "free" is V without its bits):

  | arm | P01_107 | P02_02 | P03_10 | P09_106 | mean (95% CI) | items < 0 | verdict |
  |---|---|---|---|---|---|---|---|
  | **G, generic (with the decoder)** | −28% | −8% | −40% | −12% | **−22% (−34 to −10)** | 4 | **stands** |
  | K, per person (sent before) | −43% | +88% | −55% | −1% | −3% (−49 to +56) | 3 | unclear |
  | V-S, per video, billed | +99% | +113% | +177% | +237% | +156% | 0 | fails |
  | V-M, billed | +387% | +384% | +763% | +876% | +603% | 0 | fails |
  | V-L, billed | +1146% | +1201% | +2283% | +2925% | +1889% | 0 | fails |
  | V-L free | −60% | −11% | −63% | −16% | −37% (−61 to −13) | 4 | — |
  | V-S free | −38% | −2% | −52% | −10% | −26% (−45 to −6) | 4 | — |
  | no network (check) | +3% | +2% | −1% | +19% | +6% | 1 | — |

  G beside: DISTS_R −53%, −25%, −62%, −21%; PSNR_R +35%, +28%, +11%, +13%;
  flicker × 1.00–1.06. **By the rule, H3d's coder is DCVC-UF off the shelf
  at ¼ and ½ resolution with G's network** (594 k parameters, fitted on six
  recordings of five other participants, delivered with the decoder): about a fifth
  less rate than the best of SVT-AV1 and DCVC-UF for the same LPIPS on the
  hands, on every pilot item. K does not stand, so the hybrid is not adopted.
  V fails at every size: on 1.7–3.5-minute videos even 86 KB costs more than
  it saves.
- Visual check (`p02_component_f060.png`, `…f120.png`, `p02_G_zoom.png`
  beside the report; P02_02, QP 21 at ¼, rerun with crops as
  `20261013T020929Z-72c81eac`, code `9e3af03`, its scores equal to the
  first run's for the base and G and within 5 × 10⁻⁴ for V; K's 0.234
  against 0.248, its fit not bit-identical across GPUs). K recolours the
  yellow rubber gloves toward skin: fitted on P02's other recordings,
  mostly bare hands, it learned a skin prior, which explains its +88% and
  PSNR +177% on P02_02. G sharpens the glove's edges and folds a little and
  shifts its yellow slightly toward brown, without texture the source lacks;
  that shift is where its PSNR goes while LPIPS and DISTS improve.
- Reading. Hypotheses: V failing once billed, right; K standing, wrong
  (it fails on the one item whose hands look unlike the person's other
  footage); G standing, right (−22% against −25% predicted). Two lessons
  for the paper. Most of the oracle's −77% came from fitting the scored
  frames: a network fitted on the same video outside the window keeps −37%
  free of bits (memorization explains about half), and a generic one −22%.
  A per-person network is specific to appearance: better than G where the
  person looks as usual (−43%, −55%), much worse when they wear gloves; a
  person network would need to know what it has not seen (falling back to
  G), which is a design for later, not adopted now. G costs PSNR (+11% to
  +35%): it is a perceptual gain, as G5d's enhancer on the background.
  Scope: 4 pilot items; one generic footage of six recordings (five
  participants) and 1,152 frames; one schedule of 2,000 steps (cosine, its
  loss flat at the end); network size L only for G and K.

### 2026-10-16 — Does the learned codec compose better with a learned enhancer? (analysis of H3d's oracle)
- Question (PLAN I, user 2026-10-16): the paper may argue that a learned
  codec gains more than a conventional one when PointStream adds learned
  parts. H3d's oracle judged each enhanced coder against the shared bar
  (the hull of SVT-AV1 and DCVC-UF), which mixes the base's quality with
  the enhancer's gain. Here each enhanced coder is judged against its own
  base: the upper concave hull of that coder's points alone.
- Data: H3d's oracle report `pointstream-data/visor/h3d-2026-10-12/report-oracle/h3d-report.json`
  (`12e02bc6…`): the same enhancer (G5d's, 594k parameters), fitted on
  each scored window with its bits free, over DCVC-UF HT-L at ¼ and ½ and
  over SVT-AV1 at ¼ and ⅛; per item, its Pareto front against its coder's
  points in the bar (DCVC-UF 9 points, SVT-AV1 11). No compute: `h3d.compare`
  (code `56f1120`) on the recorded points. Output:
  `…/h3d-2026-10-12/analysis-own-base-56f1120/own-base.json` (`e261b6d3…`).
- Decision rule (written after the numbers were computed, so it is a
  reading aid, not a pre-registered test): the claim
  "the learned codec composes better" is supported if DCVC-UF's gain over
  its own base exceeds SVT-AV1's by at least 10 points of BD-rate on
  LPIPS, on at least 3 of 4 items; refuted if SVT-AV1's gain is at least
  as large on average; unclear otherwise.
- Outcome (BD-rate against the coder's own envelope; P01_107, P02_02,
  P03_10, P09_106; mean):

  | enhanced coder | LPIPS | DISTS | PSNR |
  |---|---|---|---|
  | DCVC-UF | −88, −66, −87, −64% (−76%) | −83, −92, −88, −60% (−81%) | −28, +1, −3, −19% (−12%) |
  | SVT-AV1 | −84, −59, −81, −70% (−73%) | −75, −57, −80, −59% (−68%) | −37, −27, −27, −49% (−35%) |

  DCVC-UF's LPIPS gain exceeds SVT-AV1's by 3 points on average (by ≥ 10
  points on 0 of 4 items; SVT-AV1's is larger on P09_106); on DISTS by 13
  points; on PSNR SVT-AV1 gains more.
- Decision (by the rule): **not supported** (unclear on the rule's
  wording, refuted on LPIPS in substance). The enhancer helps both coders
  about equally over their own base. H3d's "SVT-AV1 unclear (−19%)" was
  against the shared bar, which DCVC-UF's base already sets 63% lower; it
  measured SVT-AV1's weaker base, not a weaker composition.
- Reading. What the paper can claim from this: DCVC-UF is the better base
  (−63% at equal LPIPS on the hands, H3c), and a learned enhancer adds a
  similar large gain to either coder as an oracle. The PLAN I argument
  should be phrased as "learned parts directed by domain knowledge add
  large gains on top of a codec, and the learned codec is the better
  base", not "composes better". Two cautions: SVT-AV1's enhanced output
  invents texture (H3d's visual check), which LPIPS can reward, so its
  perceptual gain needs the visual check beside it; and this is the
  oracle (fitted on the window, bits free), not the generic network G,
  which was fitted on DCVC-UF decodes only.
