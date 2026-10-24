# Experiments: H5p

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-20 — H5p 1: the venue models' payback
- Question (PLAN H5p 1): H5's background routes use models fitted on the
  window's own video, never billed: the scene model (recipe L, the whole
  DCVC-UF HT-L fine-tuned) and the enhancer (G5d's network, one per CRF).
  Once their bits are paid, does PointStream (G_HO in the foreground
  slots, F granted) still pass H5's rule (a)? Billed over one video, or
  only over a kitchen's footage?
- Sizes.
  - Scene model: what the client needs is the 4-bit change from the public
    HT-L weights, LZMA-coded (G5c's coder `g5c.compress_state`, variant
    `delta-4`), measured on each pilot's model in one CPU job
    (`h5 model-size`). G5c found this variant keeps quality on recipe L's
    models (+0.4% and +2.2% BD-rate on LPIPS_V against the 16-bit model):
    assumed here for G6's models, not re-checked.
  - Enhancer: its fp16 file (1,198,115 bytes), one per operating point
    (the network of its CRF).
  - Not billed: DCVC-UF's public weights, G_HO and ProPainter, all
    generic and shipped with the decoder.
- Billing (`h5 report --billing`): a pair's rate gains its background
  model's bits divided by the seconds it serves:
  - *video*: the item's whole video (ffprobe: P01_107 119 s, P02_02 207
    s, P03_10 103 s, P09_106 149 s);
  - *kitchen*: the participant's recordings on disk (EPIC-KITCHENS
    participants film one kitchen each; P01 10 videos, 10,503 s; P02 19,
    10,661 s; P03 18, 8,702 s; P09 6, 2,330 s). The dataset holds more
    footage than is on disk, so this undercounts.
  Beside it, the payback: at each point of PointStream's hull inside the
  pooled anchors' range, the model's bits over the rate it saves at equal
  weighted LPIPS (seconds of stream).
- Decision rule, fixed now: H5's rule (a) on the billed curves (weighted
  LPIPS with fill, pooled anchors, mean BD-rate −10% or better, at least
  3 of 4 items below zero).
  - Passes billed per *video*: the routes stand as per-video models.
  - Else passes per *kitchen*: they stand as per-kitchen models. A model
    fitted on one video then has to serve the kitchen's other videos, so
    a cross-video check follows (a model fitted on other recordings of
    the kitchen, scored on the pilot window) before the paper claims it.
  - Else neither: the background needs a generic model, or one far
    smaller (G5c step 4's partial or low-rank adaptation), before any
    background claim.
- Hypothesis. Per video fails: the change is about 50 MB (G5c), 2–4 Mbps
  over these videos, above every hull point. Per kitchen it costs about
  40 kbps (P01–P03) and 170 kbps (P09), comparable to the hull's low end
  (14–60 kbps, all on the scene route), so the low rates are lost to the
  enhancer route or the anchors. Rule (a) passes per kitchen on P01–P03
  with a smaller margin than −29%, and P09 fails.
- Competing explanation. Rule (a) can pass per kitchen only because the
  undercounted footage is still long enough; the per-kitchen verdict also
  rests on the untested assumption that one model serves the kitchen. The
  rule's cross-video check separates the two.
- Budget: one CPU job (four 482 MB checkpoints against the public
  weights, under 20 minutes); the report runs locally. No GPU.
- Outcome. Size job `20261020T130040Z-e3034466` (code `811ae97`, CPU,
  119 s; validated locally 4/4). Each scene model ships as a 4-bit change
  of **50.1–50.6 MB** (8-bit: 111 MB; 79–80% of its levels nonzero, as in
  G5c). Report over H5's and the G_HO swap's 41 score stages with
  `--billing` (code `028bf81`), BD-rate on weighted LPIPS with fill against
  the pooled anchors (P01_107 | P02_02 | P03_10 | P09_106):

  | billing | weighted LPIPS | mean (CI) | < 0 | weighted DISTS | frame LPIPS | frame DISTS |
  |---|---|---|---|---|---|---|
  | none (H5 with G_HO) | −19 \| −20 \| −42 \| −34% | −29% (−38 to −19) | 4 | −23% | −46% | −53% |
  | per video | +5 \| −9 \| −10 \| −22% | −9% (−19 to +1) | 3 | +12% (1 of 4) | −31% | −40% |
  | per kitchen | −17 \| −16 \| −37 \| −29% | −25% (−33 to −16) | 4 | −15% (4 of 4) | −43% | −45% |

  - **Per video the rule fails** (−9%, short of −10%). **Per kitchen it
    passes** on 4 of 4. By the rule the routes stand as per-kitchen models,
    and the cross-video check comes before the paper claims them.
  - Billed per video, PointStream's curve is its enhancer route alone
    (enhancer route by itself: +5, −9, −10, −22%): 50 MB is 1.9–3.9 Mbps
    over these videos, so the scene model leaves the hull. Per kitchen,
    the enhancer route alone gives −17, −17, −31, −28%, so inside the
    anchors' range the scene model adds little once billed. Its own
    points lie at 14–140 kbps, below the anchors' lowest rate, where no
    BD-rate reads them; the low-rate anchors (H5p 3) are the test there.
  - Payback, seconds of stream at the hull's points inside the anchors'
    range: scene model 1,310–11,850 s (0.4–3.3 h; P09_106 1.3 k, P03_10
    3.5 k, P02_02 4.0 k, P01_107 2.3–11.8 k); enhancer 10–324 s, mostly
    under a minute. So each enhancer repays within one pilot video, and
    the scene model only within a kitchen (the participants' on-disk
    footage: 0.65–3.0 h).
  - Hypothesis half right. Per video fails, but narrowly, because the
    enhancer route nearly carries it (predicted: clearly). Per kitchen
    passes on all 4 (predicted: P09 fails): billed over 2,330 s the scene
    model adds 174 kbps, but the enhancer route covers P09's range.
  - Weighted PSNR stays worse (+27% unbilled, +31% per kitchen), as in H5.
- Report: `pointstream-data/visor/h5-2026-10-16/report-pay-028bf81/` (with
  the billing file, the durations and the size result).
- Superseded by H5p 3. The per-kitchen pass holds only against B2's
  anchors, which stop at 162–525 kbps. Against fair anchors that reach the
  scene route's 8–80 kbps, PointStream billed per kitchen needs 74% more
  rate (0 of 4).

### 2026-10-20 — H5p 2: F as a video (rate oracle, CPU)
- Question (user, 2026-10-20; moved ahead of the fair anchors): H5m's
  coder XORs each frame's bit planes with the previous frame's and
  compresses them losslessly, so a hand moving a few pixels pays for its
  whole edge every frame (36–116 kbps at s = 8). Does a motion-compensated
  video codec carry F at the same accuracy for a fraction of that?
- Method (`h5 mask-video`, one CPU job per pilot item). F's class map
  from the fill mask set, coded as raw 8-bit 4:2:0 at 1/s (s = 2, 4, 8;
  padded to a multiple of 8) through SVT-AV1 (B2's binary, preset 4, CRF
  20, 35, 50; rate = the AV1 payload) and decoded by dav1d.
  - *sdf*: Y is F's signed distance to its boundary, computed at 1080p,
    reduced by area, clipped to ±8 coded pixels, stored as 128 + 15·d.
    The client's F̂ is the bilinear upsample of Y at ≥ 128, so the codec
    sees a ramp instead of a step, and the zero crossing places the
    boundary below a cell. Ringing below half the ramp's height does not
    move the threshold.
  - *binary*: Y is 255 on F's reduced cells, 0 elsewhere (the step).
  - Classes: U is 255 on hand cells, 0 on object cells, 128 outside F,
    at chroma's half size; a pixel of F̂ is a hand where U's bilinear
    upsample is ≥ 128. V is 128.
  - Beside them, H5m's lzma coder at s = 4, 8, 16, recomputed in the same
    job.
  - Accuracy of F̂ against F at 1080p: IoU, the share of F's pixels
    wrong, the mean symmetric boundary distance (px) and the class
    accuracy on F ∩ F̂.
  - (yuva420p: no AV1 encoder here carries alpha; containers that do,
    such as VP9 in WebM, code it as a second monochrome stream, so a grey
    plane is the same thing. The foreground stream is DCVC-UF, which has
    no alpha plane.)
- Decision rule, fixed now. Per item, the cheapest video setting at least
  as accurate as lzma at s = 8 (IoU at least as high, boundary distance at
  most as large) sets the item's mask rate. H5m's s = 8 rows are re-billed
  at that rate (mask bytes replaced) and judged by H5's rule (a) (weighted
  LPIPS with fill, pooled anchors, mean −10% or better, 3 of 4 items below
  zero).
  - Passes: the video coder is adopted for F. Step 4 (the foreground
    re-encoded from F̂, GPU) uses it and rescores, which replaces this
    estimate.
  - Fails, but the video coder is at least 2× cheaper at equal accuracy:
    adopted as the mask coder; F carried in the foreground stream (no
    separate F) is tested next.
  - Otherwise: lzma stays, and F carried in the foreground stream becomes
    the main route.
- Hypothesis. *sdf* at s = 4–8, CRF 35–50, matches lzma s = 8's accuracy
  at 5–15 kbps (3–10× cheaper); *binary* costs about twice *sdf* at equal
  accuracy. Re-billed, rule (a) passes narrowly (about −10%).
- Competing explanation. Equal IoU and boundary distance do not mean an
  equal composite: the codec's errors may sit elsewhere, or flicker
  between frames where lzma's are stable. Step 4's rescoring tests that;
  this rule only estimates it.
- Budget: 4 CPU jobs, under 20 minutes each. No GPU.
- Points added (before reading the other items; rule unchanged): on
  P01_107 the video coder's rate still falls steeply at CRF 50 (*sdf* at
  1/8: 108 kbps at CRF 35, 44 at 50) while F̂'s accuracy barely moves, so
  CRF 56 and 63 at s = 8 and 16 run as a second job per item (same code).
- Outcome (CPU; code `028bf81`: P01_107 `20261020T130419Z-552bc357`,
  P02_02 `…130427Z-b3d147ab`, P03_10 `…130435Z-4fa52c2b`, P09_106
  `…130456Z-85922e9b`; `6806e31` for CRF 56/63: `…131116Z-19531c72`,
  `…131124Z-4e72e14f`, `…131132Z-49949d4c`, `…131140Z-cf56c11b`; SVT-AV1
  v4.2.0; every result validated locally 5/5). Per item, P01_107 | P02_02
  | P03_10 | P09_106, rate (kbps) and IoU:

  | coder | P01_107 | P02_02 | P03_10 | P09_106 |
  |---|---|---|---|---|
  | lzma, s = 8 (H5m) | 51, 0.977 | 116, 0.988 | 75, 0.983 | 36, 0.955 |
  | lzma, s = 16 | 16, 0.923 | 39, 0.970 | 23, 0.945 | 14, 0.833 |
  | *sdf*, s = 8, CRF 50 | 44, 0.981 | 133, 0.989 | 60, 0.985 | 31, 0.947 |
  | *sdf*, s = 8, CRF 63 | 16, 0.971 | 32, 0.983 | 20, 0.976 | 15, 0.889 |
  | *binary*, s = 8, CRF 63 | 25, 0.977 | 49, 0.987 | 33, 0.982 | 19, 0.942 |
  | cheapest at lzma s = 8's accuracy | *binary* s8 CRF 50, 73 | *sdf* s4 CRF 50, 282 | *sdf* s4 CRF 50, 126 | *sdf* s4 CRF 50, 57 |

  - **The rule fails.** On no item is a video setting at least as accurate
    as lzma at s = 8 for less: the cheapest costs 1.4–2.4× more. Re-billed
    at it, H5m's s = 8 rows give +9, +47, +3, −16% on weighted LPIPS
    (mean +11%, 1 of 4 below zero; lzma: +3, +21, −8, −18%). Neither
    clause holds, so **lzma stays the mask coder**, and F carried in the
    foreground stream becomes the main route for F.
  - Hypothesis wrong. Frame-to-frame XOR plus lzma is already a strong
    temporal coder for a mask that mostly stays still. SVT-AV1 at 50
    frames/s pays per-frame overhead, and a chroma plane of hard steps.
    *sdf* does buy accuracy: at the same scale it beats lossless lzma on
    IoU (s = 4, CRF 20: 0.990 against 0.986 on P01_107), but that costs
    more than the finer grid it would replace.
  - What video coding does buy: about half the rate for a slightly worse
    F̂. *binary* at s = 8 and CRF 63 keeps lzma s = 8's IoU within 0.013
    (boundary +0.2 to +3.7 px) at 0.42–0.52 of its rate. Re-billed there,
    assuming no loss of quality (optimistic, outside the rule), weighted
    LPIPS is −4, +10, −18, −20% (mean −8%, 3 of 4): still short of
    rule (a).
  - The boundary distance is dominated by small instances that F̂ drops
    or adds (P02_02: 28–45 px for every coder, lossless lzma included),
    so it compares coders but is not a size of edge error. The IoU and
    the share of F wrong carry the comparison.
  - Caveat for the next step: H5m's rows have G in the foreground slots.
    G_HO took the granted-F result from −15% to −29%, so coded F with
    G_HO has not been measured. That, with K taken from F̂, is step 4's
    rescoring (GPU).
- Results: `pointstream-data/visor/h5-2026-10-16/mask-video-6806e31/`
  (each job's `h5.json` and the estimate).

### 2026-10-20 — H5p 3: fair anchors (equal-time SVT-AV1 with the low-rate ladder; a venue-adapted DCVC-UF)
- Question (PLAN H5p 3): H5's anchors are B2's SVT-AV1 at preset 4 and
  DCVC-UF HT-L off the shelf, at 1080p only. PointStream gets an
  equal-time encoder budget (H5's timing: preset 1), models fitted to the
  venue, and rates down to 14–29 kbps, where the anchors' floor is
  162–525 kbps, so its scene-route points go unread. Does PointStream
  still pass rule (a) when the anchors get the same:
  (a) preset 1 and lower rates through lower resolutions (PLAN B2b 3);
  (b) the same adaptation, a whole-frame DCVC-UF fine-tuned on the venue?
  (b) also separates the paper's two possible claims: the object split,
  or the adaptation alone.
- Method.
  - (a) `h5 anchors`, per item, scored by H5's harness (B2's metrics on
    both mask sets plus masked DISTS; whole frames).
    - Downscale: area average in RGB, as G6's backgrounds. Upscale:
      bicubic to 1080p, as PointStream's client.
    - SVT-AV1 4.2.0 at preset 1: 1080p CRF 41, 48, 55, 62, 70; 720p, 540p
      and 360p CRF 48, 55, 62, 70.
    - DCVC-UF HT-L off the shelf: 1080p QP 0 and 13, below B2's 27–63,
      which stay in from H5's stages; 540p and 360p QP 0, 21, 42.
    - Points chosen from PointStream's rate range alone (14 kbps to
      10 Mbps), before any is scored.
  - (b) G6's pipeline with the fill switched off (`--fill none`, new):
    - footage of the item's own video outside the window, whole frames,
      no segmenter, no ProPainter;
    - recipe L's training (540 steps, lr 3e-5, LPIPS weight 0.005), the
      same as PointStream's scene model;
    - `format` on the unfilled window at PointStream's scene grid (540p QP
      0, 13, 25, 38; 1080p QP 13, 25, 38, 50, 63);
    - its published decodes scored by `h5 anchors --venue` as anchor
      `dcvc-venue`.
    - Billed as PointStream's scene model: its 4-bit change (sized by
      `h5 model-size`) over the kitchen's footage. PointStream billed per
      kitchen too (H5p 1's standing billing).
- Decision rule, fixed now. PointStream (G_HO in the foreground slots, F
  granted, venue models billed per kitchen) judged by rule (a) (weighted
  LPIPS with fill, mean −10% or better, 3 of 4 items below zero) against
  the fair pooled envelope: the hull of SVT-AV1 at preset 1 with the
  ladder, DCVC-UF off the shelf (B2's points and the ladder), and
  `dcvc-venue` billed per kitchen.
  - Passes: the background result stands against fair anchors (still
    pending F, H5p 4, and the cross-video check of H5p 1).
  - Fails: the paper does not claim PointStream beats a venue-adapted
    codec. The rule is also read against each family alone (which one
    closes the gap), and against `dcvc-venue` by itself (adaptation
    without the object split).
- Hypothesis. Preset 1 helps SVT-AV1 by 10–20%, but DCVC-UF holds the
  pooled hull at most rates, so (a) changes little above 160 kbps; below
  it the ladder gives the anchors points, and PointStream's scene route
  wins there on weighted LPIPS (an upscaled 360p frame is blurred
  everywhere). `dcvc-venue` is the binding anchor: it gets recipe L's
  background gain (G5c: −66 to −92% on LPIPS_V against SVT-AV1) on the
  whole frame. Rule (a) still passes, with a margin of −10 to −20%
  instead of −25%.
- Competing explanation. PointStream's gain is the venue adaptation, not
  the split: `dcvc-venue` matches or beats it, and the object-centric
  part adds nothing at equal adaptation. If so, the paper's contribution
  is the adaptation, and the split has to earn its place elsewhere (rate
  control by region, or the foreground's own coder).
- Budget. (a) SVT-AV1: 2 jobs per item (1080p; 720p to 360p), 32 CPU
  threads and a shared GPU for scoring, under 40 minutes each. DCVC-UF
  ladder: 1 job per item, shared GPU. (b) per item: footage (CPU), train
  (GPU, about 30 minutes), format (GPU), venue scoring (GPU); one CPU size
  job. About 11 GPU-hours (scoring-dominated), ceiling 14; every stage
  under 45 minutes.
- Runs (code `718a751` for the anchors and the first venue jobs, `f3d9915`
  from the P01_107 footage on; each job's revision is in its
  `ready.json`; every result validated locally). A first batch of the 12 anchor jobs (`b97bd58`)
  declared too little GPU memory: DCVC-UF used 16.9 GB against 14 GB, and
  scoring 11.3 GB against 10 GB. One DCVC-UF ladder decode also differed
  from its encode under contention. Those jobs failed or were cancelled;
  nothing from them is used.
  - Anchors, per item: DCVC-UF ladder, SVT-AV1 1080p, SVT-AV1 720–360p.
    - P01_107: `20261020T133257Z-cec6e430`, `…133410Z-62ee055c`,
      `…133526Z-db9053fe`.
    - P02_02: `…133315Z-2a68afca`, `…133430Z-3ab2c823`,
      `…133544Z-897b0831`.
    - P03_10: `…133334Z-62c658d3`, `…133448Z-b933edc3`,
      `…133602Z-fb6586ba`.
    - P09_106: `…133352Z-afdfeb53`, `…133507Z-0b5069ce`,
      `…133619Z-145acb01`.
    - Every DCVC-UF point decoded as encoded at the first attempt.
  - Venue model (`--fill none`): footage, training, format and scoring.
    - P02_02: `…132940Z-28353451`, `…133758Z-38561bef`,
      `…144639Z-b9b69f62`, `…153838Z-790fe184`.
    - P03_10: `…132959Z-2699320c`, `…133822Z-2953c84f`,
      `…144659Z-c395c5b5`, `…151657Z-e7ba25c1`.
    - P09_106: `…133017Z-0a633fa3`, `…133836Z-4195fc8e`,
      `…144717Z-bfd96dc6`, `…153859Z-f34a5a82`.
    - P01_107: footage `…144820Z-7d3642df`, after a first attempt
      (`…132849Z-8ca7dab5`) stalled on a loaded host converting the whole
      footage to 4:2:0 at once (now in 64-frame chunks, `f3d9915`,
      identical output); training `…151642Z-bfeaf628`, format
      `…155531Z-154db698`, scoring `…162325Z-9327cf35`.
    - Training converged on every item (LPIPS_V 0.21–0.33 → 0.11–0.17).
    - Size (`…145053Z-488101d3`; P01_107 `…155550Z-5bd310fb`): 50.1–50.7 MB
      per 4-bit change, the same as PointStream's scene models.
  - What the fair anchors reach: DCVC-UF off the shelf goes down to 8–9
    kbps at 360p QP 0, and to 38–39 kbps at 1080p QP 0. SVT-AV1 at
    preset 1 reaches 86–95 kbps at 1080p CRF 70. B2's anchors stopped at
    162–525 kbps.
- Outcome. BD-rate on weighted LPIPS (with fill), P01_107 | P02_02 |
  P03_10 | P09_106.

  | PointStream (G_HO, F granted) against | weighted LPIPS | mean (CI) | < 0 | weighted DISTS | frame LPIPS | frame DISTS | weighted PSNR |
  |---|---|---|---|---|---|---|---|
  | B2's anchors (H5p 1's reading), models billed per kitchen | −17 \| −16 \| −37 \| −29% | −25% | 4 | −15% | −43% | −45% | |
  | fair pooled, models not billed | +19 \| +32 \| +27 \| −9% | +17% (0 to +29) | 1 | +2% | +15% | −12% (4/4) | +43% |
  | **fair pooled, models billed per kitchen (the rule)** | +113 \| +46 \| +48 \| +89% | **+74% (+47 to +101)** | **0** | +66% | +42% | +35% | +123% |
  | the venue-adapted DCVC-UF alone, neither billed | +13 \| +35 \| +27 \| −9% | +16% (0 to +31) | 1 | −4% (2/4) | +15% (0/4) | −17% (4/4) | −11% (3/4) |
  | the venue-adapted DCVC-UF alone, both billed per kitchen | +8 \| +26 \| +13 \| −6% | +10% (−1 to +21) | 1 | −7% (4/4) | +5% (1/4) | −19% (4/4) | −12% (3/4) |

  - **The rule fails.** Against the fair anchors, with the venue models
    billed per kitchen as H5p 1 required, PointStream needs 74% more
    rate for the same weighted LPIPS (0 of 4 items below zero). The paper
    does not claim that PointStream beats a venue-adapted codec, nor
    fairly configured off-the-shelf ones once its models are paid for.
  - Why (a) alone already fails it. DCVC-UF off the shelf at QP 0–13 and
    on the 540p/360p ladder reaches the 8–80 kbps range where
    PointStream's scene route lives. There, the 50 MB scene model billed
    per kitchen adds 40–174 kbps.
    - With (a)'s anchors and no venue anchor: +59% (0/4). SVT-AV1 at
      preset 1 alone: +4% (3/4). DCVC-UF (B2's points and the ladder):
      +45% (0/4).
    - Spreading the model's bits over more footage (the same for every
      item), PointStream against (a)'s anchors: 3 h +33% (1/4); 10 h −4%
      (3/4, fails); 30 h −16% (3/4, passes); 100 h −20%; 300 h −22%.
      Break-even lies between 10 and 30 hours of stream per scene model;
      a pilot kitchen's footage on disk is 0.65–3 h.
    - H5p 1's per-kitchen pass was an artifact of B2's anchors stopping
      at 162–525 kbps, which kept the low rates out of the BD range.
  - The competing explanation holds on LPIPS. At equal adaptation,
    neither model billed, the whole-frame venue-adapted DCVC-UF beats
    PointStream on weighted LPIPS (+16%, PointStream better on 1 of 4) and
    on frame LPIPS (+15%, 0 of 4). So most of H5's LPIPS gain was the
    adaptation, not the object split. The metrics disagree, though:
    PointStream is better than the venue codec on frame DISTS (−17%, 4 of
    4), weighted PSNR (−11%, 3 of 4), and slightly on weighted DISTS
    (−4%). Billing both per kitchen changes little (+10% weighted LPIPS;
    weighted DISTS −7%, 4 of 4). The split costs LPIPS and buys DISTS and
    PSNR at equal adaptation; it is not a uniform gain.
  - Hypothesis wrong. Preset 1 and the ladder were predicted to change
    little and to leave rule (a) passing at −10 to −20%. Instead the
    ladder exposed the low rates where the billed scene model loses, and
    the venue anchor matches PointStream on LPIPS. Predicted right: the
    venue anchor is the binding one on LPIPS.
- Decided (user, 2026-10-20):
  - Venue models are sent beforehand and not billed; the venue-adapted
    anchor is then unbilled too (the "neither billed" row).
  - The hours of stream a venue model needs to repay its bits are reported
    in the paper as a metric (here 10–30 h against the fair anchors),
    whatever the dataset's video lengths; runtime use is assumed long
    enough.
  - Smaller venue models (partial or low-rank adaptation) are a later,
    secondary optimization, once there is a win.
- What this left for the paper (before the user's decision above):
  - The background's venue adaptation is not, by itself, an
    object-centric contribution. A whole-frame codec adapted the same way
    gets it.
  - A per-venue model must be about 10× smaller to repay within a
    kitchen's footage (G5c step 4: partial or low-rank adaptation), or be
    shared across kitchens, or be argued as sent beforehand. If it is sent
    beforehand, the adapted whole-frame anchor is also unbilled, and the
    comparison is the "neither billed" row above.
  - Where the split may still earn its place: DISTS and PSNR at equal
    adaptation, the foreground's own quality (G_HO), rate control by
    region, and F carried in the stream (H5p 4). Each needs its own test.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5p3-a06e886/` (code
  `a06e886`; with the billing files, the hours sweep's billing, every
  anchor job's `h5.json` and the interim notes).
- GPU time: about 12 GPU-hours (12 anchor jobs, 4 footage, 4 training, 4
  format, 4 venue scoring; the failed first batch about 0.5), within the
  ceiling of 14.

### 2026-10-22 — H5p 4a: F coded and billed, the foreground re-encoded from F̂ (G_HO in the slots)
- Question (PLAN H5p 4). Every equal-adaptation reading so far has F
  granted. H5m coded F (lzma at 1/s) on streams made for the exact F, with
  G in the slots; the loss it found was partly that bias. With F billed
  and the foreground built for F̂ (K, the ring and the enhancer slots from
  F̂), what is left of the split's wins against the venue-adapted
  whole-frame DCVC-UF, neither venue model billed?
- Configuration, fixed now: H5q 1's per-point choice of foreground coder
  (DCVC-UF off the shelf, or the window's venue model, per point; G_HO in
  both slots at ¼ and ½), the configuration H5q 2 fixed as the one a real
  run is judged on. Its F-granted reading against `dcvc-venue`: weighted
  DISTS −15% (4 of 4), frame DISTS −22% (4 of 4), weighted PSNR −11% (3 of
  4), weighted LPIPS +6% (1 of 4).
- Method.
  - `h5 foreground --mask-scale s` (new): the client's F̂ (`mask.reduce`,
    `encode`, `decode`, `upsample`, as the MASK segment carries it)
    replaces F in the foreground's input (K = F̂ dilated by the ring and the
    guard, the rest push-pull padded) and in the enhancer slots (the class
    of the nearest F̂ pixel on F̂ and its ring). F's PSNR is still taken
    on the true F.
  - `h5 score --mask-scale s` (H5m's): F̂ in the compositor's alpha, the
    MASK payload in the container, so the rate bills it. The score refuses
    a foreground coded for another s.
  - The backgrounds stay as published, filled for the exact F. A pipeline
    built for F̂ would code the true pixels where F̂ misses F, so this
    reading is conservative where F̂ under-covers F, and unchanged where it
    over-covers (the foreground covers that fill).
  - Pairs, fixed now, before any score: per item, the per-point
    configuration's hull pairs on weighted DISTS (fill) with F granted, 15
    | 13 | 12 | 14 (P01_107 | P02_02 | P03_10 | P09_106), 30 foreground
    points in all, each re-encoded with its own coder. A subset's hull is
    no better than the full grid's, so a pass is not inflated by the
    choice. The other metrics are read on the same pairs, beside their
    F-granted reading on those pairs (weighted LPIPS +16%, 1 of 4; frame
    DISTS −12%, 3 of 4; weighted PSNR +8%, 1 of 4), so each row compares
    like with like: what billing F costs at fixed pairs.
  - s = 8 first (H5m's best single scale on weighted LPIPS, 36–116 kbps).
    s = 16 runs only if s = 8 fails the rule, and is then judged as the
    hull over s.
- Decision rule, fixed now. The win to keep is weighted DISTS (fill)
  against `dcvc-venue` alone, neither billed, in rule (a)'s form: mean
  BD-rate −10% or better and 3 of 4 items below zero.
  - Passes: PointStream with every byte of F counted keeps its weighted
    DISTS win at equal adaptation; lzma at that s with the F̂-built
    foreground is the billed configuration the paper reports. 4b decides
    whether F in the foreground stream is worth building for margin.
  - Fails at s = 8: s = 16, then the hull over s, by the same rule.
  - Fails at both: billing F removes the split's DISTS win; 4b's oracle
    says whether a MASK-free F could restore it.
  - Reported beside it, on the same pairs, against `dcvc-venue` and the
    fair pooled envelope (SVT-AV1 preset 1 with the ladder, DCVC-UF off
    the shelf with the ladder, `dcvc-venue`): weighted LPIPS, frame LPIPS,
    frame DISTS, weighted PSNR, and the mask's kbps and F̂'s IoU.
- Hypothesis: the rule fails at s = 8. The mask costs 36–116 kbps, as much
  as PointStream's whole stream at the low end of its hull, where most of
  the DISTS hull's pairs lie (scene model at 540p); the F̂-built
  foreground recovers part of H5m's boundary loss but not the rate.
  Weighted DISTS moves from −15% to between −5% and +10%; P02_02 (F 50%,
  116 kbps) and P01_107 (−0% granted) go above zero.
- Competing explanation: the mask adds a near-constant 36–116 kbps, which
  shifts the curve's low end off the overlap with `dcvc-venue` (13–26
  kbps up to 2–4.6 Mbps) but is a small share of the rate over most of
  the overlapping quality range, where the BD-rate is integrated in log
  rate; DISTS judges texture more than F̂'s boundary, and the F̂-built
  foreground removes H5m's padding artefacts. Then the win survives with
  a smaller margin (−10% or better).
- Budget (s = 8): per item two foreground jobs (the off-the-shelf points;
  the venue-coded points, `--video-arm`) and two scoring jobs (the score
  stage takes one coder's manifest per key), 16 jobs. From H5q 1's runs
  (2.3 min per foreground point, about 50 s per scored pair, ~3 min
  overhead per job): about 2.7 GPU-hours, ceiling 3.5. s = 16, if needed,
  the same again: together over the card's 4 GPU-hours, so the user is
  asked before it is submitted. Every stage under 45 minutes; shared GPUs,
  ½-scale DCVC-UF points allowed on Ada and A6000 with the coder's three
  attempts (docs/fleet.md).
- Runs (s = 8; shared GPUs; code revision from each job's `ready.json`;
  every full stage validated locally: foreground 12/12, scoring 9/9).
  - Foreground (`h5 foreground --mask-scale 8`), off the shelf | venue
    model: P01_107 `20261022T131134Z-62a239a3` | `…131141Z-403dcbca`;
    P02_02 `…122639Z-cdff24e8` | `…125101Z-71de7c90`; P03_10
    `…122652Z-f2d94c0f` | `…125106Z-2b385f06`; P09_106
    `…122705Z-efd8ee14` | `…125110Z-2a8c63f0`. Code `fc0caf8` for the
    three first off-the-shelf jobs, `1ba09f2` for the others (it adds only
    `h5 crops`). RTX 6000 Ada.
  - Two venue-coded jobs pass 11/12: F's PSNR at `dcvc@0/4` (3.6–4.4
    kbps) is 19.09 dB (P01_107) and 19.33 dB (P09_106), under the
    validator's 20 dB floor, as H5q 1 found at the same point with F
    granted (19.07, 19.45). A low-rate picture, not a broken decode
    (every other point 22–40 dB); the floor was not changed.
  - Scoring (`h5 score --mask-scale 8`, tags `h5p4-gho` | `h5p4-venue`):
    P01_107 `20261022T172510Z-1e1e4ff1` | `…172612Z-33e8a123` (`8602763`);
    P02_02 `…125643Z-1ff8a9ab` | `…134121Z-3e1b0349`; P03_10
    `…125653Z-63c4d1dd` | `…134531Z-0d44d55c`; P09_106
    `…130311Z-60d8ed55` | `…133417Z-c7a01f55` (`1ba09f2`). RTX 6000 Ada
    and RTX A6000. Every planned pair scored: 15 | 13 | 12 | 14.
  - Not citable: the first P01_107 off-the-shelf job (`…122606Z-a7413b7b`)
    stopped at its smoke, whose only point was `dcvc@0/4` (F at 19.9999
    dB, the floor above); the smoke moved to the highest-QP ¼ point, as
    H5q 1's specs had it, and four not-yet-started venue jobs were
    cancelled and resubmitted with it (`…122629Z-3f8b7426`,
    `…122645Z-e4edbf2c`, `…122658Z-51047d6d`, `…122710Z-c94154f6`). Its
    rerun's smoke ran past 600 s on the contended A6000 gpu2
    (`…125053Z-fe60a43e`, as in H5q 2); P01_107's two jobs then ran
    without gpu2 (`…125057Z-79f22900` cancelled).
- The mask at s = 8, P01_107 | P02_02 | P03_10 | P09_106: 51 | 116 | 75 |
  36 kbps, F̂'s IoU with F 0.977 | 0.988 | 0.983 | 0.955 (H5m's).
- Outcome. BD-rate on 4a's pairs, the per-point configuration, neither
  venue model billed; P01_107 | P02_02 | P03_10 | P09_106, mean, items
  below zero:

  | against `dcvc-venue` | F granted (4b's oracle) | **F billed, s = 8 (the rule)** | F̂ stream, mask bytes removed | s = 16 bound |
  |---|---|---|---|---|
  | **weighted DISTS** | −0 \| −16 \| −22 \| −22: −15%, 4/4 | **+125 \| +146 \| +102 \| −0: +93%, 1/4** | +0 \| −16 \| −21 \| −30: −17%, 3/4 | +44 \| +47 \| +22 \| −18: +24%, 1/4 |
  | weighted LPIPS | +6 \| +31 \| +42 \| −16: +16%, 1/4 | +127 \| +176 \| +194 \| +12: +127%, 0/4 | +15%, 1/4 | +56%, 1/4 |
  | frame DISTS | −8 \| −28 \| −21 \| +10: −12%, 3/4 | +115 \| +131 \| +92 \| +41: +95%, 0/4 | −13%, 3/4 | +26%, 0/4 |
  | frame LPIPS | +26%, 0/4 | +131%, 0/4 | +27%, 0/4 | +65%, 0/4 |
  | weighted PSNR | +8%, 1/4 | +94%, 0/4 | +10%, 1/4 | +41%, 1/4 |

  Against the fair pooled envelope, F billed: weighted DISTS +117% (0/4),
  weighted LPIPS +153% (0/4), frame DISTS +107% (0/4), weighted PSNR
  +199% (0/4); F granted on the same pairs: −9% (3/4), +17%, −7%, +79%.
  - **The rule fails at s = 8** (+93%, 1 of 4), decided at 3 items (two
    above zero, whatever P01_107 read).
  - **s = 16 is not run**: its upper bound (s = 8's composites, a better
    F̂ than s = 16's, re-billed at s = 16's measured 16 | 39 | 23 | 14
    kbps from H5m) still fails (+24%, 1 of 4), so the real run cannot
    pass and the hull over s cannot either.
  - **F̂ costs nothing; the loss is the MASK segment's rate.** With the
    mask bytes removed, the stream built for F̂ reads as F granted, or
    better (weighted DISTS −17% against −15%; P09_106 −30% against
    −22%): the F̂-built foreground removed H5m's bias entirely. Adding
    36–116 kbps to every point moves the low end of PointStream's curve,
    where the venue codec reaches 13–26 kbps, by a factor of 2–10, and the
    BD-rate integrates in log rate.
  - Hypothesis right on the verdict, wrong on the size (+93%, not −5 to
    +10%). The competing explanation fails: the shift dominates the
    overlap because the venue anchor's curve starts at 13–26 kbps.
  - **Billing F removes every win of the split at equal adaptation**, on
    every metric, against both the venue codec and the pooled envelope.
    What 4b's oracle says about a MASK-free F follows.
- GPU time: about 2.6 GPU-hours (23 jobs, the failed, timed-out and
  cancelled ones included), within the ceiling of 3.5.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5p4-8602763.tar`
  (sha256 `1ecb4806…c9c4a2`, read-only; one archive, as the NFS home
  asks): the specs, every stage's `h5.json`, `pairs4a.py`,
  `analyse4a.py` with its `a4.json`, `bound16.py`, `breakeven.py` and
  their outputs in `out/`, and H5p 5's crops.

### 2026-10-22 — H5p 4b: oracle for F carried by the foreground stream (no MASK segment)
- Question (PLAN H5p 4, H5q lever 3). Could F carried by the foreground
  stream itself (a key outside F, an alpha plane, or F derived at the
  client from what it decodes), with no MASK segment, turn a verdict that
  H5q or 4a reads as a loss?
- Oracle: the exact F at zero bits, with today's foreground stream and its
  slots: the F-granted rows of 4a's pairs, already measured (H5q 1's
  `venuefg` and G_HO scoring stages). It dominates any such component:
  - no carried F̃ is more accurate than F, and F costs no bits here;
  - carrying F in the stream constrains the padding outside F (a key, a
    hard edge, or a second plane), which push-pull's smooth padding, built
    to cost the codec as little as possible, does not; so the stream does
    not get cheaper. Assumed, stated: a carried F that also made the
    foreground stream cheaper than push-pull padding would break the bound.
  - No GPU: the oracle's rows exist. A tighter oracle (the real cost of a
    key or alpha plane) runs only if this one moves a verdict.
- Operating point: where H5q loses, weighted LPIPS (fill) against
  `dcvc-venue`, neither billed, rule (a) (mean −10% or better, 3 of 4
  below zero). Then each metric of 4a's table where 4a's billed reading
  fails rule (a)'s form.
- Decision rule, fixed now. For each such verdict: if the oracle also
  fails it, F in the foreground stream cannot turn it and is not built for
  it. If the oracle passes where 4a fails, the component could move that
  verdict: a tighter oracle (the measured cost of the carrier) is next,
  with its own rule.
- Hypothesis: on weighted LPIPS the oracle fails (the per-point
  configuration with F granted reads +6%, 1 of 4, on its full grid and
  +16% on 4a's pairs), so the component is not built for H5q's loss.
  Whether it matters for DISTS depends on 4a: if 4a keeps the DISTS win,
  the oracle can only add margin; if 4a loses it, the oracle (−15%, 4 of
  4) passes and a tighter oracle follows.
- Competing explanation: none can rescue weighted LPIPS here, since the
  bound is the F-granted reading itself; the only open question is
  whether the bound's assumption (no cheaper stream) holds, which matters
  only if a verdict is close.
- Budget: CPU, minutes.
- Outcome (CPU; the oracle's rows are H5q 1's scoring stages, 4a's pairs;
  `analyse4a.py`, `breakeven.py` in 4a's report).
  - **Weighted LPIPS: the oracle fails** (+16%, 1 of 4 on 4a's pairs; +6%,
    1 of 4 on the per-point configuration's full grid). F carried by the
    foreground stream cannot turn H5q's loss; **not built for it.** The
    same holds for frame LPIPS (+26%, 0/4) and weighted PSNR (+8%, 1/4),
    which the oracle fails too.
  - **Weighted DISTS: the oracle passes where 4a fails** (−15%, 4 of 4,
    against 4a's +93%, 1 of 4), so by the rule a MASK-free F could move
    that verdict, and a tighter oracle is next. Frame DISTS: the oracle
    reads −12% (3 of 4), passing too.
  - A first tighter bound, computed post hoc on the CPU (not a rule
    written before its run, so a reading, not a decision): the oracle's
    rows with a constant rate c added for F, the most any carrier could
    spend and keep the win. Weighted DISTS against `dcvc-venue`: c = 2
    kbps −11% (3/4, passes), 5 kbps −5% (3/4, fails), 10 kbps +4% (2/4).
    Frame DISTS fails already at 2 kbps (−8%, 3/4). **A carrier must cost
    about 2 kbps or less**, about 5 bytes per frame at 50 frames/s.
  - Every F coder measured so far costs 7–20× that at usable accuracy:
    lzma at s = 16, 14–39 kbps (IoU 0.83–0.97, H5m); SVT-AV1 at s = 8 CRF
    63, 15–32 kbps (IoU 0.89–0.98, H5p 2); lzma at s = 8, 36–116 kbps.
    A carrier inside the foreground stream (a key outside F or an alpha
    plane) still codes F's boundary every frame as the hands move.
- Decision by the rule: **not built for LPIPS** (the oracle fails).
  **Unclear for DISTS**: the oracle passes, and the tighter oracle the
  rule names (the measured cost of a real carrier, a key or alpha plane in
  the foreground stream: about 2–3 GPU-hours, as 4a) is not run here. The
  2 kbps budget makes a pass unlikely, so the session recommends not
  building it; whether to run that measurement is for the root session
  and the user.

### 2026-10-22 — H5p 5: visual check, PointStream and the anchors at matched rates (CPU)
- Question (PLAN H5p 5): what do PointStream and the anchors look like at
  the same rate, for the user's review beside the BD-rates?
- Method (`h5 crops`, new; one CPU job per item). The client's output
  rebuilt on frames 60 and 180 from the published decodes: PointStream as
  billed in 4a (F coded at 1/8 and the stream built for F̂, the composite
  as `score` makes it), the venue-adapted DCVC-UF's published FFV1, and
  SVT-AV1 at preset 1 re-encoded at its ladder point. Each PNG shows F's
  box (padded, at least 384 px) and the whole frame, labelled with each
  stream's rate.
- Matches, fixed by rule before any image: on PointStream's billed
  weighted-DISTS hull, the pair nearest 200 kbps and the pair nearest
  1 Mbps (log rate); each anchor at its best weighted DISTS at or under
  that rate (H5's "matched" convention, which favours the anchors).
  PointStream | venue | SVT-AV1 (kbps):
  - P01_107: 164 | 132 (1080p QP 25) | 147 (720p CRF 62); 1027 | 670 |
    811 (1080p CRF 48).
  - P02_02: 238 | 202 | 221 (720p CRF 62); 968 | 669 | 886 (720p CRF 48).
  - P03_10: 179 | 93 | 138 (720p CRF 62); 1343 | 976 | 928 (1080p CRF 48).
  - P09_106: 182 | 128 | 161 (360p CRF 62); 1230 | 1084 | 1177 (720p CRF
    48).
- Runs (CPU, 32 threads; every full stage 6/6 locally; each rebuilt
  PointStream rate equals its scoring row's container): P01_107
  `20261022T173554Z-a7980f6d`, P09_106 `…172537Z-588cfb8e` (`8602763`);
  P02_02 `…173732Z-ef9b37e1`, P03_10 `…173739Z-517bd9ba` (`583a4fb`, the
  same `h5 crops`). Not citable: the first P02_02 and P03_10 jobs ran past
  the 600 s smoke on the loaded gpu2 (`…172523Z-0f87919b`,
  `…172530Z-0aea402f`), rerun without it.
- Outcome: 16 PNGs in 4a's report archive (`crops/`), for the user's
  review; no rule, no verdict. First look (the session's, not a judge):
  at about 200 kbps PointStream draws the hands sharpest, the venue codec
  keeps a little more texture on handled objects (P09_106's towel) and
  in the background, and SVT-AV1 at 360–720p blurs the hands; at about
  1 Mbps the three are close at this size.
- Item report (H5p 4–5, sent to the root session; docs/sessions.md):
  - Stopped because: goal reached (4a and 4b decided as far as their
    rules go, the crops of 5 ready); one decision is open for the root
    session and the user.
  - Decision by the rule: 4a fails (F billed at s = 8: weighted DISTS
    +93%, 1 of 4, against the venue-adapted DCVC-UF; s = 16's upper bound
    +24%, 1 of 4, so not run). 4b: not built for LPIPS (the oracle, exact
    F at zero bits, reads +16%, 1 of 4); unclear for DISTS (the oracle
    passes, −15%, 4 of 4; a post hoc bound puts any F carrier's budget at
    about 2 kbps).
  - Numbers, budget, branch and checks: in the three entries above;
    about 2.6 GPU-hours of the card's 4; CPU for 4b and 5.

### 2026-10-22 — H5p 4c: F carried by the foreground stream, a key outside K (measured carrier)
- Question (user, 2026-10-22: "measure the MASK-free F carrier ..., with
  its rule written first"; 4b's next oracle). 4b's zero-bit oracle keeps
  the weighted DISTS win and a post hoc bound gave any carrier about 2
  kbps. What does a real carrier inside the foreground stream cost, and
  does PointStream with it keep the win?
- Carrier (`--carrier key`, new), chosen so the compositor stays as built:
  - Encoder: the foreground input keeps the source on K (F dilated by the
    ring and the guard, `keep_radius`), exactly as today, and puts one
    flat key colour outside K instead of push-pull padding. The key is
    chosen per window from six saturated colours, the one farthest (in
    YCbCr, 1st percentile of distances) from the source pixels on K.
  - Client: K̃ is where the decoded foreground (at its coded size) lies
    farther than τ from the key, upsampled to 1080p; F̃ = K̃ eroded by
    `keep_radius`. The alpha ring and the enhancer slots then run on F̃
    as on F. Erosion after dilation fills only gaps narrower than the
    keep radius; those pixels are inside K, so they show coded source,
    not the fill.
  - G_HO sits in both slots, so the client needs no hand/object classes.
  - The encoder picks τ per point from a fixed set (the best IoU of F̃
    with F on every 8th frame) and sends the key and τ: a 5-byte CARRIER
    segment, billed.
  - Exact F stays the encoder's input (no F̂); the backgrounds stay as
    published (filled for F).
- Configuration and pairs: 4a's (the per-point configuration; the 54
  pairs of its weighted-DISTS hull, 30 foreground points), so each row
  compares like with like against F granted and against 4a.
- Two stages, the cheaper first.
  - Stage 1, the foreground jobs: the key-padded streams, their rate
    against the push-pull streams at the same point, and F̃'s IoU.
  - Gate (an optimistic bound, CPU): the F-granted rows with each pair's
    foreground bytes replaced by the key stream's, plus the CARRIER
    segment, quality unchanged. If this fails the rule below, stage 2 is
    not run and the carrier fails.
  - Stage 2, scoring: the composites with F̃ (no MASK), the CARRIER
    segment in the container.
- Decision rule, fixed now: weighted DISTS (fill) against `dcvc-venue`,
  neither venue model billed, rule (a)'s form (mean −10% or better, 3 of
  4 items below zero).
  - Passes: the key carrier is PointStream's F; the split keeps its
    weighted DISTS win at equal adaptation with every byte counted.
  - Fails (at the gate or in stage 2): no F coder tried keeps the win;
    H5p 4 closes as a fail.
  - Reported beside it, on the same pairs: weighted LPIPS, frame LPIPS
    and DISTS, weighted PSNR, against `dcvc-venue` and the fair pooled
    envelope; the carrier's rate per point and F̃'s IoU.
- Hypothesis: the gate fails. The flat key makes a hard edge along K's
  border every frame, which a smooth push-pull padding avoids; at the ¼
  points (4–30 kbps) that adds more than the 2 kbps the win allows.
- Competing explanation: a flat key is as cheap as push-pull (both are
  predicted almost for free; the edge along K's border is short and moves
  with the hands, which the codec already pays for), so the rate barely
  moves, F̃'s IoU is high, and the rule passes near F granted's −15%.
- Budget: stage 1, 8 foreground jobs (as 4a: about 1.3 GPU-hours);
  stage 2, 8 scoring jobs (about 0.9 GPU-hours), only if the gate passes.
  Ceiling 3 GPU-hours. With 4a's 2.6 this takes the item past the card's 4
  GPU-hours; the user asked for this run (2026-10-22), knowing the
  estimate (2–3 GPU-hours). Every stage under 45 minutes; gpu2 excluded
  (smokes past 600 s there all day).
- Runs, stage 1 (`h5 foreground --carrier key`, code `9b4b3ba` per each
  job's `ready.json`; RTX 6000 Ada; every full stage validated locally,
  13/13), off the shelf | venue model: P01_107
  `20261022T181434Z-92f9fe49` | `…181442Z-01f9444d`; P02_02
  `…181451Z-8e67f9a5` | `…181458Z-05f8e2dd`; P03_10 `…181506Z-d3a74345`
  | `…181512Z-2cb46248`; P09_106 `…181520Z-d8c8e56f` |
  `…181527Z-4ad9e659`.
  - Keys: magenta on P02_02, green on the others.
  - Two venue jobs miss the 20 dB F-PSNR floor at `dcvc@0/4` only:
    P01_107 18.96 dB, P09_106 18.58 dB (push-pull at the same point
    19.09 and 19.33). It is the low-rate point already known from 4a and
    H5q 1, and F̃ misses 1.8% and 4.9% of F there; every other point is
    21–40 dB. The gate uses only these streams' rates.
  - τ: every point chose 96, the top of the set; IoU is already 0.95–0.99
    above the lowest points, and τ does not change the rate.
- F̃ is accurate. IoU with F is 0.95–0.99 above the lowest points (0.88 at
  P09_106's `0/4`, where F is 2% of the frame); F̃ misses 0.1–1% of F and
  adds 1–6%, as coded source inside K. It is as accurate as lzma at s = 8
  (0.955–0.988) at no MASK cost.
- The key's cost, key stream minus the push-pull stream at the same point
  (granted F):
  - P02_02 (F 50%): +2.5 to +865 kbps, +36% to +93%.
  - P03_10 (F 19%): +2.1 to +625 kbps, +60% to +129%.
  - P01_107 (F 14%): +1.2 to +379 kbps, +26% to +58%.
  - P09_106 (F 2%): −175 to +9.7 kbps, −21% to +12% (cheaper than
    push-pull at three of seven points).
  The flat key's step along K's border costs bits in proportion to K's
  perimeter, every frame.
- Outcome: **the gate fails, so stage 2 is not run.** Even with quality as
  granted (an optimistic bound), weighted DISTS against `dcvc-venue` reads
  +18 | +5 | +9 | −30%: **+1% (1 of 4)**; against the pooled envelope +7%
  (1 of 4). Weighted LPIPS +39% (1/4), frame DISTS +1% (1/4), weighted
  PSNR +35% (1/4).
  - Hypothesis right on the verdict. The key edge costs far more than the
    2 kbps budget wherever F is large: 1–25 kbps at the ¼ points, and
    26% or more at every point.
  - The competing explanation holds only where F is small: on P09_106
    the key costs about what push-pull does, and the bound reads −30%
    (F granted: −22%).
  - **H5p 4 closes as a fail.** No F coder tried keeps the split's
    weighted DISTS win at equal adaptation on egocentric video: neither
    lzma at s = 8 or 16, nor video coding (H5p 2), nor a key in the
    foreground stream. Where F is a few percent of the frame (P09_106),
    the key carrier costs about nothing and the split still wins.
- GPU time: 1.38 GPU-hours (8 jobs), within the ceiling of 3; stage 2
  unspent. The item's total is about 4.0 GPU-hours.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5p4c-9b4b3ba.tar`
  (sha256 `982fe3f7…915ecd`, read-only): the specs, every stage's
  `h5.json`, `gate4c.py` and its output.

### 2026-10-24 — H5p 4d rules: binary shape and decoder-derived F
- Scope: the four H5 pilots only; no final runs or Wan-Animate. Root report
  address: `01a16cfc-1a88-7013-bd79-8489fe7c1eb2` (delegating Codex chat);
  prior root: `session_01StNQ6dvxYRHhnMbQymNgNf`.
- Ceiling: exact free masks preserve weighted DISTS (−15%, 4/4) but
  **do not fix the established LPIPS loss** (+16% on 4a's pairs).
- (a) Decision rule, before the run: at exactly lzma s=8's binary accuracy,
  the cheapest of intra, previous-frame context, and motion-predicted
  context must cost ≤2 kbps on at least 3/4 items. This is a feasibility
  screen for the stated rate target, not a measured BD-rate verdict.
  A pass proceeds to rebilling 4b's fixed pairs and the existing DISTS rule
  (mean ≤−10%, ≥3/4 below zero); a fail closes only these configurations.
- Hypothesis: binary contexts and motion reduce lzma's 36–116 kbps but
  cannot remove enough boundary innovation to reach 2 kbps.
  Competing explanation: contour coherence leaves almost no unpredictable
  boundary information after motion, so the target becomes feasible.
- Bound/permission to build: 4b's exact-free-F oracle passes DISTS, so
  these coders can move that verdict; they are not built to rescue LPIPS.
- Component: experimental adaptive binary arithmetic coder, not an MPEG-4
  CAE or JBIG2 implementation. Five causal spatial neighbours and five
  predicted temporal neighbours, adaptive counts; three modes (intra,
  previous frame, ±4-cell integer search in 32-cell blocks). Motion vectors
  are LZMA-compressed and billed, alongside the header and arithmetic bits.
  No free encoder-only motion. The current foreground stream's DCVC motion
  is latent rather than exposed vectors; this tests independent shape motion.
  Union F only, since both enhancer slots use G_HO; excluding class bits
  favours the candidate. Reduction uses the same majority cells as lzma,
  guaranteeing identical binary IoU and boundary distance after upsampling.
- Correctness smoke: real arithmetic encode/decode on moving shapes, full,
  empty, random masks, odd dimensions and partial blocks; 12 tests passed.
  Fleet smoke: 8 real pilot frames, all three modes and exact roundtrips;
  full stage: all 240 frames, same input and processing path.
- Budget (a): zero GPU-hours; four CPU jobs, each smoke ≤600 s, full ≤2400 s,
  total per-job allowance ≤3100 s. No scaled run beyond the four pilots.
- (b) Decision rule, before any segmentation run: use only already-decoded
  streams and decoder-available prompts; record any granted prompts as an
  oracle and bill any actual correction/initialization. Measure F accuracy
  beside lzma s=8 and score the fixed 4a pairs against `dcvc-venue` at equal
  adaptation: weighted DISTS mean ≤−10%, ≥3/4 below zero. Exact-free-F is
  the motivating ceiling; it does not prove every inaccurate mask produces
  a worse perceptual score. No gate will assume monotonicity without checking.
- Hypothesis (b): segmentation can find hands but lacks the labelled object
  identity and outlines that generated B1b's F. Competing explanation:
  the decoded streams retain enough semantic evidence to recover F without
  additional information. Audit that interface before allocating a GPU.
- Budget (b): at most 2 GPU-hours including smokes and failed attempts;
  every stage ≤45 min. Choose a representative smoke before estimating the
  number of pairs affordable. A missing decoder interface or oracle that
  cannot dominate its proposed correction family is a design blocker,
  not a measured failure of decoder segmentation.
- CPU submission (revision `f632732c`): P01_107
  `20261024T082551Z-b86db128` (acknowledged pending). Input mask archive
  SHA256 `049977f057a367a921255dbf894489b877d85c0f33e1baf2554e0ca1af96799a`;
  packed environment `pointstream-20261009T214002Z`, SHA256
  `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
  Command: `python -m experiments.visor.h5_shape --masks {staged:fill_masks}
  --item ITEM --frames {frames}`; frames 8 smoke / 240 full. No native
  video encoder or decoder is used by this CPU shape experiment.
- CPU submission P02_02: `20261024T082613Z-c3ff4ae0` (pending at
  acknowledgement); same revision and staged inputs as P01_107.
- CPU submission P03_10: `20261024T082655Z-eb51efd6` (pending at
  acknowledgement). P01_107's real smoke completed three exact roundtrips;
  smoke rates are infrastructure checks, not pilot evidence.
- CPU submission P09_106: `20261024T082741Z-0dec552f` (acknowledged).
- CPU full-stage results (240 frames, revision `f632732c`): intra |
  temporal | motion, kbps: P01_107 44.53 | 14.80 | 15.07;
  P02_02 50.79 | 24.79 | 25.83; P03_10 50.65 | 21.54 | 22.88;
  P09_106 15.25 | 11.76 | 10.85. All 12 full encode/decode roundtrips
  are exact on the reduced union. **Rule (a) fails, 0/4 ≤2 kbps.** This
  rejects these three configurations, not every possible CAE/motion coder.
- (b) Concrete configuration, fixed before GPU submission: the pinned
  B1b SAM 3.1 mask tracker on each saved 4a decoded foreground point,
  initialized with the first s=8 union mask, actually encoded and decoded
  with the intra shape coder. No future human prompt is granted. The
  predictor tracks the union as one object; it does not receive source
  images, object labels, later masks, or corrections as tracking prompts.
  Its full-size predictions are majority-reduced to s=8. Send the XOR
  against the target s=8 union through the temporal shape coder; bill seed,
  correction payload and two configuration bytes. Decoder XOR roundtrips
  must recover 4a's exact reduced union on every frame. G_HO is the same
  in both slots, so class identities cannot change the corrected image.
- This is a **real corrected-mask configuration**, not a bound on all
  segmentation/correction methods. Its unchanged binary masks allow the
  measured 4a composites/metrics to be reused point by point with MASK
  bytes replaced. Each foreground point has its own correction rate: never
  apply one point's predictor to other points for free. Uncorrected IoU is
  reported, not substituted for an unmeasured DISTS score. No claim about
  uncorrected segmentation's BD-rate is made without scoring it.
- GPU budget allocation: four jobs capped at 1800 seconds each (sum 2 h),
  each smoke ≤300 s (8 frames, first point), full ≤1320 s (240 frames,
  all saved foreground points), validator ≤60 s; remaining allowance for
  overhead. SAM 3.1 on Ada/A6000 is already in the model–GPU table;
  runtime/device and full checkpoint load are validated again. Per-point
  results are saved immediately, preserving completed work if time runs out.
- Local checks: pinned temporary Python 3.12 environment from `.[dev]`;
  ruff passed. Initial base-Mac checks were unsuitable (numpy 2 / missing
  stubs), so no result from them is treated as a project regression.
  Full pinned pytest: 828 passed, 25 skipped, two platform-sensitive failures
  (`ps` denied by sandbox; protected dangling `/home/.../Datasets` symlink
  resolution on macOS). Rechecking those outside the sandbox separately.
- Decoder submission P01_107: `20261024T083129Z-617103d1`, revision
  `94178029`, pending at acknowledgement; ≤1800 seconds (0.5 GPU-hours).
  No other decoder jobs submitted until this smoke is inspected.
- Review correction before promoting decoder work: saved `decoded.mkv`
  contains mask-placed G_HO enhancement, so using it directly is circular.
  Cancelled only our job `20261024T083129Z-617103d1` after 50.6 s
  (0.0141 GPU-hours), retaining its smoke artifacts but **excluding them
  from scientific evidence**. This is a methodological error, not a SAM
  failure. The corrected entrypoint independently decodes `stream.psdc`
  with the matching stock/venue weights before loading SAM. Both native
  decoder passes must agree. SAM then sees raw pre-enhancement RGB only.
  Decoder processes finish before SAM is loaded, avoiding simultaneous
  codec/tracker GPU memory. The saved enhanced frames supply no pixels.
- Remaining allocation: first corrected job ≤1740 s, other three ≤1800 s;
  plus the cancelled 50.6 s totals at most 7190.6 s (<2 GPU-hours). Full
  first job ≤1260 s; no retry may exceed this aggregate allowance.
- Corrected decoder submission P01_107: `20261024T083446Z-1d84eea5`,
  revision `cf692902`, pending at acknowledgement. Native stream decoding
  also requires the image checkpoint and matching stock/venue video weights;
  these are staged by SHA256 (full identities in the job's spec).
- Pinned local mypy now passes (183 files). The two platform-sensitive
  pytest checks were re-run outside the sandbox: process suspension passes;
  the existing dangling `/home/.../Datasets` symlink guard test still fails
  on this Mac. It is unrelated to shape coding and remains reported.
- Corrected raw-input job `20261024T083446Z-1d84eea5` stopped after
  60.8 seconds: smoke passed (27.7 s), full's first native DCVC double
  decode differed. No full segmentation numbers are evidence. This is the
  known native decoder determinism hazard in `docs/fleet.md`, not a mask
  quality failure. P02_02 `20261024T083702Z-14c3862e` was acknowledged
  before the full failure was inspected; its own determinism gate remains
  mandatory and no bad decode is promoted.
- Next revision saves every native attempt before deciding, retries at most
  three times as H5's codec path does, and requests exclusive GPUs. Remaining
  three item allocations ≤1700 seconds each; including P02_02's ≤1800 and
  the two ended attempts (~111.4 s), the aggregate upper bound is ~7012 s.
- Additional CPU reading (same 4a fixed pairs and identical binary masks):
  replacing MASK with the best shape payload gives weighted DISTS
  +40.9 | +26.2 | +19.9 | −20.5%, mean **+16.6%, 1/4 below zero**;
  weighted LPIPS +49.2% (1/4). Existing lzma baseline reproduces +93.35%
  weighted DISTS (1/4) before replacement. PCHIP and upper concave anchor
  envelope use `h5.compare`; every container segment overhead is billed.
  `experiments/visor/h5_mask_report.py` records SHA256 identities of its
  score, shape and correction inputs. This reading confirms failure under
  the full DISTS rule, beyond the ≤2 kbps feasibility screen.
- P01_107 exclusive retry: `20261024T083825Z-d68be64f`, code `e740088b`,
  cap 1700 seconds. P02_02 remains `20261024T083702Z-14c3862e` with its
  original 1800-second cap; no result is accepted after a differing decode.
- P02_02 `20261024T083702Z-14c3862e` stopped after ~103 s: a native
  decoder subprocess segfaulted in full; no full mask evidence. Native
  process failures now consume one of the same three attempts, with each
  error retained. The exclusive P01 retry passed smoke and completed its
  first full native point; it has not yet produced a full SAM result.
- Revised remaining allocation after P02 ended: P01's current ≤1700 s
  plus P02 retry, P03 and P09 each ≤1700 s =6800 s. Ended attempts
  ~215 s give <7015 s aggregate maximum, still under 7200 s. No further
  retries beyond these jobs without first re-auditing actual elapsed time.
- P02_02 exclusive retry: `20261024T084111Z-1939a70f`, revision
  `9e576c04`, ≤1700 s (pending at acknowledgement).
- P03_10 `20261024T084139Z-eed9df04` and P09_106
  `20261024T084208Z-f947299c`: code `9e576c04`, each ≤1700 s,
  acknowledged pending. P01 `20261024T083825Z-d68be64f` stopped after
  ~80.7 s: a full-stream point failed all three native determinism checks
  even on an exclusive Ada GPU. Its raw per-attempt reports are retained.
  This blocks a complete corrected-mask BD-rate curve for P01 at this
  configuration; **no segmentation verdict follows from it**.
- P02 exclusive retry `20261024T084111Z-1939a70f` also stopped at the
  three-attempt native gate in full. P03/P09 continue under their existing
  caps. **The requested full corrected-mask BD-rate result is blocked by
  native decoder reliability, not established as a pass or fail.**
- Revised bounded diagnostic, declared before full SAM results: first
  stock foreground point `dcvc@0/4`, 240 frames on each item, with the
  identical seed/tracker/correction pipeline. Four jobs ≤600 s each,
  smoke ≤180 s (8 tracked frames), full ≤300 s, validator ≤60 s.
  This tests whether that concrete decoder-mask configuration meets 2 kbps;
  it cannot rule out other points, segmenters, or correction coders, and
  cannot replace the missing full BD-rate curve. No extrapolated quality
  or rate is reported as a measured result. P09's stock point is not on
  4a's selected hull (its corresponding point used venue weights).
- Budget reserve: four diagnostic caps =2400 s, remaining P03/P09 caps
  =3400 s, plus ended jobs <450 s; total <6250 s (<2 GPU-hours). This
  diagnostic is cheaper than changing or re-encoding the codec ladder,
  which is outside this card's mask work.
- Diagnostic P01_107: `20261024T084521Z-bc72ff4d`, revision `74bd6d57`,
  ≤600 s; pending at acknowledgement.
- Final local checks at `9e576c04` implementation: ruff and mypy pass
  (184 files); full pytest outside sandbox: **830 passed, 24 skipped,
  1 failed**, the pre-existing macOS protected-symlink test. All 50
  PointStream tests pass. Linux CI is still required for the platform case.
- Diagnostics, same code and ≤600 s each: P02_02
  `20261024T084534Z-c700da3b`; P03_10 `20261024T084607Z-e00640d5`.
  P03's full-ladder job `20261024T084139Z-eed9df04` and P09's
  `20261024T084208Z-f947299c` both reached the native validation stop;
  neither supplies full mask-curve evidence.
- Diagnostic P09_106: `20261024T084640Z-668afb3e`, code `74bd6d57`,
  complete and validated, as are the other three diagnostics. P09's first
  saved **stock** point is actually `dcvc@42/2`, not `dcvc@0/4`; the
  earlier description assumed an existing point that its stock archive
  does not contain. Selection was by first saved point, before results.
  This limits its interpretation to that ½-resolution point.
- Full diagnostics (240 frames, Ada, exact corrected grid): P01_107
  27.30 kbps; P02_02 61.33; P03_10 30.00; P09_106 13.29. All exceed
  2 kbps. Unreduced SAM IoU against the s=8 reference mask is respectively
  0.827, 0.716, 0.383, 0.654; this is not IoU against original F.
- Actual GPU use from all eleven supervisors' ended−started times:
  **1141.39 seconds =0.3171 GPU-hours of 2**, including every failed and
  cancelled attempt. All jobs are terminal; no GPU work remains running.
- Final descriptive CPU accuracy audit: compare saved reduced predictions
  and corrected grids against original B1b F with the existing IoU and
  symmetric boundary metric. This does not change the decision rule or
  invent a missing full-curve result. Four items, 8-frame smoke / 240-frame
  full, no GPU; input NPZ files and original mask archive SHA256-identified.
