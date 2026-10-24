# Experiments: G5c

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-10 — G5c: scene model at scale, step 1 (checks without training)
- Question (PLAN G5c 1): before any more training, does G5b's LPIPS gain
  hold on a metric the model never trained on (DISTS), does the scene
  model flicker, and how small can the model be sent?
- Scope. G5b's pilot excerpts (P26_02, P06_03; the gate tier as in G5b)
  and every coded method there: SVT-AV1 (G5's 12 streams per stretch),
  DCVC-UF off the shelf (G5b's `dcvc` streams) and G5b's five fine-tuned
  arms per stretch (upper bound and scene at β = 0 and 0.026, control).
  The checks decode the published streams with the published checkpoints;
  nothing is re-encoded or retrained, and every stream's bytes are checked
  against the size its result recorded (`experiments/background/g5c.py`
  `checks`).
- **DISTS on V.** torchmetrics 1.9's DISTS (its alpha/beta, `weights.pt`
  `f5e65c96…8218`) on the VGG16 backbone `Models/DISTS/vgg16-397923af.pth`
  (`397923af…5bf0`, copied from the shared torch hub cache like LPIPS's
  AlexNet; the user approved the download, which turned out unneeded). As
  for LPIPS, the source foreground is pasted into the decoded frame; DISTS
  pools whole feature maps, so V enters as weights: every layer's means,
  variances and covariance are taken over V resized to that layer (area
  average). With V the whole frame this is DISTS exactly (a test checks
  it). BD-rate on q = −100 × DISTS_V, as LPIPS.
- **Flicker on V.** For each pair of consecutive excerpt frames, DIS flow
  on the source (G5's `oracle_flows`) warps frame t−1 onto t; the
  method's temporal residual (decoded t − warped decoded t−1) minus the
  source's (source t − warped source t−1), on luma, mean absolute value
  over V of both frames inside the image, averaged over pairs (all 240
  frames, the own tier's masks). Zero means the decoded video changes
  exactly as the source does; real motion and flow error cancel. Also
  reported: the decoded and source residual energies, and BD-rate on
  q = −flicker against SVT-AV1.
- **Model size** (`g5c.py` `compress`), on the two β = 0.026 scene models.
  Variants: the whole model at 16 bits (G5b's, re-coded as a check:
  same bytes) and at 8, 4 and 2 bits; the change from the public HT-L
  weights (which the client has) at 8, 4 and 2 bits, dense, and with only
  the largest 10% of changes per tensor kept. Tensors of 1,024 or more
  values are quantized per output channel (symmetric uniform, fp16
  scales); smaller ones stay fp16 (whole) or fp16 differences (change).
  Size is the bytes of an actual coder (LZMA, per tensor), with the
  zeroth-order entropy beside it; payback as in G5b. Each variant loads
  strictly into B2's worker and re-codes the excerpt at QP 9/27/45/63.
- Decision rules, fixed now (`g5c.DECISION`):
  - DISTS: the LPIPS gain *transfers* if the β = 0.026 scene model beats
    SVT-AV1 on DISTS_V (BD < 0) on P06_03, where it beat it on LPIPS_V.
    If not, the paper calls the gain LPIPS-specific and step 2's
    DISTS-in-the-loss arm runs first.
  - Flicker: *found* if, on either stretch, the β = 0.026 scene model is
    worse than SVT-AV1 at equal rate (BD on q = −flicker > 0, or no
    overlap with all its points worse). Then a temporal term joins the
    loss in step 2 (PLAN G5c 2).
  - Size: a variant *keeps quality* if its BD-rate on LPIPS_V against the
    16-bit model's curve is at most +5%. The smallest such variant sets
    the model size for steps 2–5 and the paper's payback; all variants
    are reported by size.
- Hypotheses. DISTS transfers but less: BD against SVT-AV1 on P06_03
  between −30% and −60% (DISTS tolerates resampled texture, which is what
  β > 0 makes). Flicker is found for β = 0.026 and not for β = 0: G5b's
  visual check saw synthesized lines and stains move or vanish. The 8-bit
  whole model keeps quality and 4-bit does not; the change from the
  public weights is small (540 steps at 1e-5), so the 4-bit change keeps
  quality at a few MB, and the sparse 4-bit change at 1–3 MB, a payback
  of minutes instead of hours.
- Competing explanations. DISTS and LPIPS are both deep-feature metrics,
  so DISTS agreeing is not a human judgement; the paper's side-by-side
  frames carry that. Flicker could come from the source's flow error,
  which the residual difference cancels only where the flow is wrong in
  the same way for both; SVT-AV1 and DCVC-UF off the shelf, measured the
  same way, give the floor. For size, quantization noise could act as a
  regularizer and *help* on the held-out excerpt; then the 16-bit model's
  curve is not an upper bound, and the variants are compared with
  SVT-AV1 too.
- Budget. Checks: 2 jobs (one per stretch), each ≤ 45 minutes on Ada or
  A6000 (36 decodes, LPIPS, DISTS, flicker). Size: 4 jobs (per stretch,
  the whole-model variants and the change variants), each ≤ 45 minutes
  (4 or 6 variants × 4 QPs, encode, decode and scoring). Smokes ≤ 10
  minutes.
  Ceiling 3.5 GPU-hours.
- Outcome, checks and size (code `7454618` for the first size job, `93257e4`
  for the other five; the report at `93257e4`;
  report `g5c-report.json` from `g5c.py report` over the six results).
  Jobs: checks P26_02 `20261010T194633Z-c888eb58` (A6000), P06_03
  `…194639Z-b316025e` (Ada); size P26_02 `…193016Z-68f20aa8` (whole,
  Ada) and `…194645Z-a3f9df45` (change, Ada), P06_03 `…194652Z-8711be56`
  and `…194658Z-fd240ad0` (Ada). Anchor: SVT-AV1 on Telea's fill, as G5b.
  - **DISTS: the gain transfers.** BD-rate on DISTS_V against SVT-AV1,
    β = 0.026 scene model: −53% (P06_03; −74% on LPIPS_V) and −76%
    (P26_02, where LPIPS_V is unclear by G5's verdict). Upper bound −68%,
    −76%; control −35%, −68%. The β = 0 arms and the unadapted model stay
    worse than SVT-AV1 on DISTS_V too (+21% to +162%). As predicted.
  - **Flicker: found.** Flicker on V never overlaps SVT-AV1's range:
    SVT-AV1 1.8–2.7 luma levels; DCVC-UF off the shelf 2.2–3.7; β = 0
    scene 2.2–3.9, as the unadapted model; β = 0.026 scene 2.4–4.8. At
    about 30–45 kbps: SVT-AV1 2.6–2.7, unadapted 3.05, β = 0.026 scene
    3.5–4.0. So DCVC-UF flickers more than SVT-AV1 on its own, and
    synthesized texture adds about a third. A temporal term joins the
    loss in step 2, as the rule says.
  - **Size: the 4-bit change keeps quality.** Against the 16-bit model's
    curve (BD on LPIPS_V): change at 8 bits ≤ +0.7%, at 4 bits +0.2%
    (P06_03) and +1.1% (P26_02), so the smallest variant that keeps
    quality is the 4-bit change, **51 MB** (LZMA; zeroth-order entropy
    about the same). The whole model keeps quality at 8 bits (109 MB, +2%
    and +4%) and breaks at 4 and 2 bits (LPIPS_V above 0.26 at every
    rate). The 2-bit change loses (+111%, +316%); keeping only the
    largest 10% of the change loses at every width (+239% to +1,056%):
    fine-tuning moved nearly every weight a little (81% of the 4-bit
    levels nonzero), so the change is not sparse. Hypothesis half right:
    the 4-bit change keeps quality, but it is 51 MB, not a few MB. A
    small model has to be trained small (step 4: partial or low-rank
    adaptation). Payback of the 51 MB change against SVT-AV1 at equal
    LPIPS_V: 0.6 h (P26_02, 10 kbps), 3.1 h and 1.1 h (P06_03, 12 and
    44 kbps), from G5b's 2.5, 12.9 and 4.8 h at 16 bits (218 MB LZMA).
  - Decoding stored streams. Re-encoding with the stored checkpoints
    reproduced every stored stream byte for byte on gpu6 (P06_03), and
    every stored stream decoded alone there. On gpu5 (Ada, same class)
    re-encoding differed by up to 1 byte with identical LPIPS_V, and one
    stored stream's decode alone segfaulted (P26_02, first smokes); on
    the A6000 (gpu3), Ada-encoded streams decoded alone gave two
    different passes or segfaulted (QP 63), and re-encoding there
    differed by up to 0.6%. DCVC-UF's decoder must therefore run on the
    encoder's GPU class, and even same-class hosts are not guaranteed
    bit-exact, though DCVC sets deterministic algorithms. Every score
    above and in G5b comes from an encode and its own decode that match
    (deterministic, intra frame equal to the encoder's); P26_02's checks
    are scored at the A6000 re-encodes' own rates. For the paper: a
    deployment caveat for every DCVC-UF arm, the scene model included.
- **Fill check** (`g5c.py` `fill`, added after ProPainter's audit, before
  its run). The holes every method codes (not V of the training masks) on
  the two pilot excerpts, filled by OpenCV Telea per frame (G5's
  `filled`) or by ProPainter (sczhou/ProPainter at `e870e79`, the
  official code chosen by the user over their fork; weights v0.1.0 from
  `Models/ProPainter`; the repository's inference steps and defaults, with
  edge padding to a multiple of 8 instead of resizing, so known pixels
  stay exact: `experiments/background/propainter_fill.py`). Each fill is
  coded by SVT-AV1 at G5's six CRFs and scored on V as above (LPIPS,
  DISTS, PSNR, flicker); the fill itself is never scored. Fill flicker:
  mean absolute luma change inside the holes of consecutive frames after
  warping by DIS flow on the filled video itself.
  - Rule: ProPainter's fill *wins* if SVT-AV1 coding it saves at least
    2% at equal LPIPS_V (BD-rate against Telea's fill) on both
    excerpts; then every later arm codes and trains on it. Otherwise
    Telea stays (cheaper: no GPU at the sender).
  - Hypothesis: it wins, by 5–15%: a temporally consistent fill is
    cheaper to predict across frames than a per-frame Telea smear that
    changes with the hand, and its fill flicker is a fraction of Telea's.
  - Competing explanation: the saving comes from fewer bits in the
    holes, where quality is not scored, rather than better prediction of
    V; that is still a real saving for every method coding the fill.
  - Budget: one job (both excerpts), ≤ 45 minutes on Ada.
  - Outcome (`20261010T193932Z-ccdfebd6`, Ada, code `c271736`):
    **ProPainter wins.** BD-rate of SVT-AV1 on ProPainter's fill against
    Telea's, on LPIPS_V: −11.2% (P26_02), −12.7% (P06_03); on DISTS_V
    −15.6%, −14.6%; PSNR_V −10.9%, −9.8%; flicker on V −15.7%, −13.7%.
    Fill flicker 2.4 against 6.1 (P26_02) and 2.5 against 4.4 (P06_03).
    ProPainter takes 81 s per 240-frame excerpt on Ada (18 GiB peak);
    Telea 42–57 s on one CPU core. As predicted. ProPainter's own output
    repaints a 4-pixel ring of V around each hole (its mask dilation); the
    source is pasted back on V, so every method still codes V as it is.
    Consequence: from here every arm codes and trains on ProPainter's
    fill; the scene frames (about 1,200 per stretch) are filled once per
    stretch and staged as an archive.

### 2026-10-10 — G5c step 2: the distortion–perception trade on ProPainter's fill
- Question (PLAN G5c 2): which loss should the scene model train with?
  The β sweep gives the curve from PSNR-faithful to LPIPS-optimal; a
  temporal term answers step 1's flicker; DISTS in the loss tests whether
  training on every scored metric helps (the user's call, 2026-10-10).
  Run overnight with the user's go-ahead for unblocked GPU work.
- Fill and anchors (`g5c.py prefill`). Step 1 switched every arm to
  ProPainter's fill, so each of the 10 stretches is filled once, whole
  (about 1,200 frames): the held-out excerpt alone, as in the fill check,
  and the scene frames in chunks of at most 200 frames with 20 context
  frames from the same side (never from the excerpt, so no held-out
  pixel reaches a training frame's fill). The 4:2:0 fill is published
  per stretch (sha256 in its meta) and every later job trains on and
  codes from it (`g5b.py finetune --prefilled`). The same job codes the
  new anchors on the excerpt: SVT-AV1 at G5's six CRFs on the frames and
  on the fill (anchor: the upper envelope, as G5), and DCVC-UF off the
  shelf on the fill (the reference), all scored with LPIPS, DISTS, PSNR
  and flicker on V.
- Arms (scene frames of the same stretch, P26_02 and P06_03, 540 steps,
  G5b's recipe otherwise): A β = 0.026 (G5b's choice, again, on the new
  fill), B β = 0.01, C β = 0.005, D β = 0, E β = 0.026 plus DISTS on V
  with its weight set at step 0 so that it equals the LPIPS term on the
  monitor set, F β = 0.026 plus the temporal term with γ = 0.5: the mean
  square over V of the change of the coding error between consecutive
  frames, ((x̂_t − x̂_{t−1}) − (x_t − x_{t−1}))², which is about twice
  the distortion when errors are independent between frames, so γ = 0.5
  gives it the distortion's own weight. Every arm is scored with LPIPS,
  DISTS, PSNR and flicker on V (flicker flow-compensated, unlike the
  loss).
- Decision rule, fixed now. An arm *beats* SVT-AV1 on a metric on a
  stretch if its BD-rate is below 0, or no BD-rate exists and none of its
  points inside SVT-AV1's rate range lies below SVT-AV1's envelope and at
  least one lies above it. (1) Candidates: arms A–D that beat SVT-AV1 on
  LPIPS_V and DISTS_V on both pilot stretches. (2) The β for steps 3–5 is
  the candidate's with the lowest mean PSNR_V BD-rate against SVT-AV1
  (the least distortion paid). (3) The temporal term is adopted if F's
  flicker on V, averaged over its four QPs, is at least 20% below A's on
  both stretches and F's LPIPS_V BD-rate against A is at most +10% on
  both. (4) DISTS in the loss is adopted if E's DISTS_V BD-rate against A
  is −10% or better and its LPIPS_V BD-rate against A at most +10%, on
  both stretches. Adopted terms join the chosen β in step 3; their
  combination is checked there, not assumed. If no arm is a candidate, β
  stays 0.026 and the gap is reported.
- Hypotheses. On the new anchor (SVT-AV1 also gains 11–13% from the
  fill), A still beats SVT-AV1 on both metrics, by a little less than
  G5b's −74% on P06_03. β = 0.01 keeps most of the LPIPS_V gain at about
  half the PSNR_V cost and is chosen; β = 0.005 is not a candidate on
  P26_02; β = 0 is not on either. F cuts flicker by 20–30% at under 10%
  LPIPS_V cost and is adopted. E gains 10–20% on DISTS_V at equal LPIPS_V
  and is adopted, but this one is the least certain.
- Competing explanations. A against G5b's β = 0.026 separates the fill's
  effect from the loss's. A lower flicker in F could come from blurrier
  output rather than steadier texture; LPIPS_V and DISTS_V against A
  tell. DISTS and LPIPS share deep features, so E improving both is not
  proof of better video; the side-by-side frames are.
- Budget. Prefill and anchors: 10 jobs, each ≤ 30 minutes on Ada or
  A6000 (ProPainter about 7 minutes per 1,200 frames at 18 GiB, 12
  SVT-AV1 and 4 DCVC-UF codings, scoring). Arms A–D and F: 10 jobs, each
  ≤ 45 minutes on Ada (G5b's 33 minutes plus DISTS and flicker scoring).
  E: a 40-step calibration at the real settings first (DISTS through
  VGG16 on every frame, activations recomputed), and E's runs only if
  540 steps fit in 45 minutes; otherwise DISTS on one frame per group,
  recorded. Ceiling 15 GPU-hours.
- Calibration of E (`20261010T214417Z-5328894d`, P06_03, 40 steps, Ada,
  code `22bc2b2`): DISTS on every frame costs 7.4 s per step (3.3 without)
  and peaks at 45.7 GiB, so 540 steps would take 67 minutes. As the budget
  says, E runs with DISTS on the first frame of each group (4 of 32
  frames, `--dists-every 8`), its weight 8 times the step-0 match so its
  share stays equal to the LPIPS term (δ = 0.038 per frame at step 0
  before the factor). Not evidence.
- The temporal weight was wrong. The same monitor set at step 0 has the
  temporal term at 0.00057 against a masked distortion of 0.0052: DCVC-UF's
  coding errors are strongly correlated between frames, so the term is
  about a tenth of the distortion, not twice it, and γ = 0.5 gives it about
  5% of the distortion's weight instead of an equal share. F runs as
  submitted (its answer: a weak term changes nothing, or does), and **F2**
  is added before its runs: β = 0.026 with γ = 9, so γ times the term
  equals the distortion at step 0, as intended. Rule (3) applies to F2
  as to F; if both qualify, the one with less flicker is adopted.
- Prefill and anchors, outcome (code `22bc2b2`; all 10 stretches,
  Ada and A6000): `20261010T212240Z-6154e13b` (P26_02),
  `…212247Z-c5d24f8b` (P06_03), `…212640Z-2da36c49` (P02_09),
  `…212648Z-69b17f32` (P03_10), `…212654Z-787a2fbd` (P04_13),
  `…212702Z-104550fc` (P06_05), `…212710Z-2b31ffcd` (P06_10),
  `…212716Z-fac3a3df` (P06_106), `…212722Z-96267071` (P09_106),
  `…212727Z-e0c6b3d2` (P37_102). ProPainter fills 1,200 frames in about
  6.5 minutes at 18.6 GiB; the excerpt's SVT-AV1 points reproduce the
  fill check's exactly. Archives: P26_02 `c2afe03a…df43`, P06_03
  `f6d0863e…60dd`.
- Outcome so far (code `22bc2b2`, Ada, 540 steps, all converged by G5b's
  rule). BD-rate against SVT-AV1 on ProPainter's fill, P06_03 | P26_02:
  | Arm | LPIPS_V | DISTS_V | PSNR_V | Jobs |
  |---|---|---|---|---|
  | A β = 0.026 | −73% \| −95% | −46% \| −70% | +130% \| +91% | `20261010T214349Z-754aa83e`, `…214555Z-c64ae187` |
  | B β = 0.01 | −71% \| −93% | −49% \| −73% | +98% \| +56% | `…214355Z-7ac53ed1`, `…214602Z-f727e078` |
  | C β = 0.005 | −66% \| −90% | −42% \| −67% | +81% \| +46% | `…214400Z-11e27fad`, `…214609Z-1f5596b6` |
  | D β = 0 | +61% \| +74% | +80% \| +184% | +46% \| +12% | `…214406Z-54cc0daf`, `…214615Z-fa6ff232` |
  | F γ = 0.5 | −72% \| −95% | −43% \| −68% | +128% \| +85% | `…214411Z-a63f4b78`, `…214622Z-f3dfad74` |
  - Rule (1)–(2): A, B and C are candidates; D is not. Mean PSNR_V
    BD-rate: A +110%, B +77%, C +63%, so **β = 0.005** for steps 3–5.
    The hypothesis picked β = 0.01 and expected C to fail on P26_02; it
    did not. A perceptual weight of any size flips the model from losing
    to SVT-AV1 (D) to winning by two thirds or more; past that, β trades
    PSNR_V for a few points of LPIPS_V. A β between 0 and 0.005 was not
    tried.
  - On the new fill, A (−73% on P06_03) keeps G5b's gain against
    SVT-AV1 (−74% on Telea's fill), and P26_02 now has a BD-rate
    (−95%): ProPainter's anchor reaches lower rates, so the curves
    overlap. D against DCVC-UF off the shelf: −3% PSNR_V on P06_03 (G5b
    on Telea's fill: −11%), −25% on P26_02.
  - Rule (3), F: flicker 2–4% below A's at every QP on both stretches,
    LPIPS_V within 2% of A's. Not adopted: at γ = 0.5 the term is about
    5% of the distortion (the calibration above), too weak to matter.
    F2 decides the temporal term.
- Outcome, E and F2 (code `7f13f6b`, Ada, 540 steps, converged). BD-rate
  against A, P06_03 | P26_02:
  | Arm | LPIPS_V | DISTS_V | PSNR_V | flicker (mean over QPs) | Jobs |
  |---|---|---|---|---|---|
  | E (DISTS, 1 frame in 8) | +10.5% \| +20.5% | −29% \| −18% | +8% \| +5% | 0% \| −1% | `20261010T230108Z-2dca65b9`, `…230132Z-dc29fffb` |
  | F2 (γ = 9) | +43% \| +40% | +46% \| +64% | −7% \| −13% | −13% \| −13% | `…230122Z-c0840bae`, `…230142Z-91a3f4be` |
  - Rule (4): E gains on DISTS_V (−29%, −18%) but loses more than 10% on
    LPIPS_V on both stretches (+10.5%, a near miss, and +20.5%). **Not
    adopted.** At an equal share the two deep-feature terms trade against
    each other rather than both improving: the competing explanation
    held, not the user's hope that training on every scored metric would
    lift all three. The DISTS weight set at step 0 was 0.30 and 0.19 per
    scored frame (with the factor 8); 3.7–3.9 s per step, 43.2 GiB.
  - Rule (3): F2 lowers flicker by 13% on both, short of 20%, and costs
    40% on LPIPS_V and 46–64% on DISTS_V while gaining 7–13% on PSNR_V:
    the steadier output is the smoother one (the competing explanation).
    **Not adopted.** Flicker stays a limitation of the scene model, most
    of it DCVC-UF's own (step 1).
  - **Step 2's loss for steps 3–5: β = 0.005, no DISTS term, no temporal
    term.** Hypotheses: β wrong (0.01 predicted), temporal wrong (adopted
    predicted), DISTS wrong (adopted predicted, flagged least certain).
  - GPU time for step 2: about 11.5 hours (prefill 10 × 0.25–0.3, arms
    14 × 0.55–0.6, calibration 0.15), within the 15 allowed.
- **The step-2 recipe on all 10 stretches** (written before its runs,
  with the user's overnight go-ahead for unblocked GPU work). G5b's
  deferred stage 3 with step 2's loss: scene frames of the same stretch,
  β = 0.005, ProPainter's fill, 540 steps, on the 8 stretches not yet
  fitted (pilots: arm C). It gives the gate on the current recipe and the
  baseline row that steps 3 (other days) and 4 (recipe) must beat; it
  does not replace step 5.
  - Rule (G5b's gate): BD-rate on LPIPS_V against SVT-AV1 on the new
    anchors, median over the 10 stretches below 0 (a stretch with no
    BD-rate counts by the verdict's sign, unclear ones are excluded and
    counted). DISTS_V and PSNR_V are reported the same way.
  - Hypothesis: passes, median between −60% and −85% on LPIPS_V; DISTS_V
    below 0 on at least 8 of 10; PSNR_V above 0 on all.
  - Budget: 8 jobs, each ≤ 45 minutes on Ada, about 5 GPU-hours.
  - Outcome (code `7b3db40`, Ada, all converged). Jobs: P02_09
    `20261011T012101Z-788838d3`, P03_10 `…012112Z-9685b85a`, P04_13
    `…012123Z-73e9f1c9`, P06_05 `…012134Z-fe2661d2`, P06_10
    `…012145Z-8b14a9c7`, P06_106 `…012155Z-1e3b2ae7`, P09_106
    `…012206Z-36d91d5e`, P37_102 `…012216Z-ff4f09a7`; pilots from arm C.
    BD-rate against SVT-AV1 (LPIPS_V / DISTS_V / PSNR_V), and LPIPS_V
    against DCVC-UF off the shelf:
    | Stretch | LPIPS_V | DISTS_V | PSNR_V | LPIPS_V vs DCVC-UF |
    |---|---|---|---|---|
    | P26_02 | −90% | −67% | +45% | −94% |
    | P37_102 | −72% | −40% | +20% | −84% |
    | P06_03 | −66% | −42% | +80% | −77% |
    | P06_10 | −44% | +47% | +162% | −84% |
    | P06_106 | −43% | −7% | +27% | −62% |
    | P03_10 | −39% | +29% | +213% | −80% |
    | P02_09 | −29% | +49% | +173% | −83% |
    | P09_106 | −29% | +27% | +33% | −52% |
    | P06_05 | +4% | +158% | +331% | −84% |
    | P04_13 | +22% | +123% | +287% | −80% |
    | **median** | **−41%** | **+28%** | **+121%** | |
    - **The gate passes** on LPIPS_V: median −41%, below 0 on 8 of 10.
      Hypothesis too optimistic (−60% to −85%): the pilots were the two
      best stretches but one. Against DCVC-UF off the shelf the scene
      model saves 52–94% on every stretch.
    - **DISTS_V does not follow**: median +28%, below 0 on 4 of 10
      (hypothesis: 8). This revises step 1's "the LPIPS gain transfers to
      DISTS", which rested on the two pilots: on most stretches the
      β > 0 model wins on the metric it trained on and loses on the one
      it did not. The paper reports both, with frames.
    - PSNR_V is above SVT-AV1's on every stretch, as predicted.
    - Payback of the 51 MB 4-bit change, where SVT-AV1 reaches equal
      LPIPS_V: 0.4–8 h on 8 stretches at QP 27–45 (17 h on P06_05; none
      on P04_13, which SVT-AV1 beats).
    - For step 3: the losing stretches (P04_13, P06_05) and the DISTS gap
      are what more scene footage and the recipe must close.

### 2026-10-11 — G5c: the oracle on the stretches that lose (data or method?)
- Question (the user, 2026-10-11): P04_13 and P06_05 lose to SVT-AV1 on
  LPIPS_V with the step-2 recipe, and they are the excerpts that move most
  (step 2's 10-stretch outcome: the scene model's gain tracks the
  excerpt's motion, ρ = 0.81, and SVT-AV1's rate, ρ = 0.87). Is the method
  the limit there, or the footage it was fitted on?
- Oracle: G5b's upper bound with step 2's recipe (β = 0.005, ProPainter's
  fill, 540 steps), fitted on the held-out excerpt itself: perfect footage
  of exactly the views it must code. It bounds what any footage could give
  the recipe, so it dominates the footage arms of step 3.
- Rule, fixed now: on each of P04_13 and P06_05, if the oracle beats
  SVT-AV1 on LPIPS_V (BD < 0) by at least 30%, the loss there is the
  footage's (the method can win with footage that shows the views), and
  the paper says so, with step 3 measuring how much real footage closes;
  if it does not beat SVT-AV1, the method is the limit on fast excerpts
  and the paper scopes to calm ones on that ground.
- Hypothesis: the oracle wins on both by 40–70% (G5b's upper bound at
  β = 0.026 won by 83% on P06_03), so the loss is the footage's.
- Competing explanation: the oracle sees the excerpt it is scored on, so
  part of its win is memorization of exactly those frames, which no real
  footage gives; the oracle's margin is an upper bound, not a forecast.
  Step 3's footage arms measure the realistic share.
- Budget: 2 jobs, each ≤ 45 minutes on Ada.
- Outcome, P04_13 (`20261011T063021Z-477693dc`, Ada, code `4a75b6e`,
  converged): BD-rate against SVT-AV1 on LPIPS_V −0.3% (DISTS_V +98%,
  PSNR_V +256%); the step-2 model C was +22% (+123%, +287%). **The method
  is the limit on P04_13**: even fitted on the excerpt it codes, the
  recipe only ties SVT-AV1 on LPIPS_V and loses on the rest, so no
  footage can make it win there. Its excerpt moves most of all ten (source
  flicker 5.4–7.2 against about 2–4 on the calm ones). The rule is the
  recipe's (β = 0.005); a higher β is step 4's to try.
- Outcome, P06_05 (`20261011T063032Z-02c80566`, Ada, code `4a75b6e`, converged): LPIPS_V
  −11% against SVT-AV1 (DISTS_V +138%, PSNR_V +315%); C was +4% (+158%,
  +331%). Short of the rule's −30%: **the method is the limit on P06_05
  too.** On both fast stretches, fitting on the very frames to be coded
  moves LPIPS_V by only 15–20 points against SVT-AV1 and leaves flicker
  at 4–7 luma levels (C's 4–7 too): the loss comes from DCVC-UF's coding
  of fast motion, not from what the model knows of the scene. The
  hypothesis (the oracle wins by 40–70%, so the footage is to blame) was
  wrong. For the paper: the scene model's gain is scoped to calm
  viewpoints on this evidence, and the fast excerpts are reported as the
  base codec's limit. P06_05's footage arms still run: a win is reachable
  there (−11%), so whether real footage gets near it is still asked.

### 2026-10-11 — G5c step 3: footage from the kitchen's other recordings
- Question (PLAN G5c 3): does footage of the same kitchen from other
  recordings make a better scene model than the stretch's own 90 s, and
  can it stand alone (a model made before the stream exists)?
- Scope, decided before any run. Four stretches: the pilots (P26_02,
  P06_03) and the two that lose to SVT-AV1 on LPIPS_V (P04_13, P06_05),
  since closing those is what more footage must do. Footage only from
  recordings that EPIC-KITCHENS' hand-object detections cover, so the
  foreground is G1's exactly (SAM 3.1 "hand" and "arm" plus the detector's
  boxes at score ≥ 0.5, dilated 16 px at 1080p); fetching detections for
  the rest would be a new download. Same kitchen checked by eye on two
  frames per recording: P06's `_101`–`_110` recordings are another home
  (dark counter, double sink) and are excluded for P06_03 and P06_05; P26
  and P04 are one kitchen each. Never the stretch's own recording. Some
  recordings may be from the same day, so "other recordings", not "other
  days". Recordings: P26_02 ← P26_108; P06_03 ← P06_05, P06_10; P06_05 ←
  P06_03, P06_10; P04_13 ← P04_06, P04_24.
- Dose (fewest settings first): 4 segments of 60 s per stretch at 10
  frames/s (2,400 frames, about 2.5 times the stretch's own scene frames),
  dealt round-robin over its recordings and placed by sha256 of the
  stretch, recording and number, at least 5 s from either end
  (`g5c.choose_segments`); P26_108 is 196 s long, so P26_02 gets 2
  segments (1,200 frames). ProPainter fills each segment by chunks within
  it (`g5c.py footage`). If footage helps, a larger dose follows for a
  dose–response curve.
- Arms (step 2's recipe: β = 0.005, ProPainter's fill, 540 steps; scored
  on the stretch's held-out excerpt as before): G, footage only (`--fit
  footage`); H, the stretch's scene frames plus footage. Baseline: arm C
  of step 2 on the same stretch.
- Decision rule, fixed now. Footage *helps* if H's BD-rate on LPIPS_V
  against C is −10% or better on at least 3 of the 4 stretches, or H turns
  a stretch that loses to SVT-AV1 (P04_13, P06_05) into one that beats it
  on LPIPS_V. Footage *stands alone* if G's BD-rate on LPIPS_V against C
  is +10% or better (at most 10% worse) on at least 3 of 4. DISTS_V and
  PSNR_V are reported beside it. If the user switches step 2's loss to
  the DISTS arm (open, 2026-10-11), the arms run with that loss instead
  and the rule is unchanged.
- Hypotheses. H helps by 10–25% on the calm stretches (P26_02, P06_03)
  and less on the moving ones, and turns neither loser; the gain grows
  with the dose. G lands within 15% of C: the kitchen is the same, but the
  excerpt's exact clutter is not in other recordings.
- Competing explanation: more frames help through diversity, not the
  kitchen; the cross-kitchen control of G5b (another stretch's footage at
  equal size) answers that and runs if H helps.
- Scope change (2026-10-11, before any step-3 training result): P04_13's
  footage job (`20261011T062500Z-011fc5d2`) ran out of its 2,700 s in
  ProPainter after SAM on a shared GPU, and P04_13's oracle (entry above)
  only ties SVT-AV1 even fitted on the excerpt itself, which bounds every
  footage arm there. As the protocol says for a failed oracle, P04_13's
  footage arms are not run. The rule applies to the 3 other stretches:
  "helps" needs H at −10% or better against C on 2 of 3, or P06_05 turned
  into a win; "stands alone" needs G within +10% of C on 2 of 3.
- Budget. Footage: 4 jobs, each ≤ 45 minutes on Ada or A6000 (SAM 3.1
  about 0.38 s and ProPainter about 0.32 s per frame, 2,400 frames).
  Training: 8 jobs, each ≤ 45 minutes on Ada. About 8 GPU-hours.
- Footage, outcome: P26_02 `20261011T062431Z-7d8edeb3` (2 segments of
  P26_108, 1,200 frames, foreground about 30% per frame, SAM 0.29 s per
  frame) and P06_03 `…062441Z-421d23d7` (P06_05 and P06_10, 2,400
  frames). The first attempts ran out of GPU memory: SAM 3.1 held about
  31 GiB while ProPainter ran, fixed in `ca79ee0` (`g5c.py` frees it; the completed jobs ran it),
  and declared 40,000 MiB. P04_13 `…062500Z-011fc5d2` and P06_05
  `…062451Z-417b2ef0` ran out of their 2,700 s in ProPainter on a shared
  GPU; P04_13's arms were dropped by the scope change above, P06_05's are
  not rerun (below).
- Outcome, arms (code `401ddf9`, Ada, β = 0.005, 540 steps, converged).
  BD-rate against C | against SVT-AV1:
  | Arm | Stretch | LPIPS_V | DISTS_V | PSNR_V | Job |
  |---|---|---|---|---|---|
  | G footage only | P26_02 | +226% \| −74% | +69% \| −45% | +16% \| +55% | `20261011T064151Z-62e85d9d` |
  | H scene + footage | P26_02 | +14% \| −88% | +4% \| −67% | +10% \| +56% | `…064200Z-7def1338` |
  | G footage only | P06_03 | +13% \| −60% | +2% \| −40% | −3% \| +77% | `…070725Z-599e0365` |
  | H scene + footage | P06_03 | +6% \| −64% | 0% \| −43% | −7% \| +73% | `…070738Z-f9747804` |
  - **Footage does not help** (rule: H at −10% against C on 2 of 3): H is
    worse than C on both completed stretches, so 2 of 3 cannot be met.
    The clause "H turns P06_05 into a win" is untested: its footage job
    ran out of time, and its ceiling is the oracle's −11%; not rerun,
    since the rule is decided and the clause's ceiling is small. At 540
    steps more frames dilute the steps spent on the frames nearest the
    excerpt; a longer schedule (step 4) could change that, or the other
    recordings may simply differ (lighting, clutter, P26_108 is from the
    later collection).
  - **Footage does not stand alone** (rule: G within +10% of C on 2 of
    3): +226% and +13% (P06_03 a near miss). Hypotheses: H helps by
    10–25% (wrong), G within 15% of C (right on P06_03, wrong on P26_02).
  - What holds for the paper: a model made only from other recordings of
    the kitchen, before the stream exists, still beats SVT-AV1 by 60–74%
    on LPIPS_V and 40–45% on DISTS_V on both calm stretches; the stream's
    own scene footage adds more on top.
  - GPU time: about 5 hours (footage 4 jobs and 4 failed attempts,
    arms 4 × 0.6, oracles 2 × 0.6).

### 2026-10-11 — G5c step 4: the recipe, round 1 (frame weights, learning rate)
- Question (PLAN G5c 4): which training change makes the scene model beat
  SVT-AV1 on more of the selection's metrics, at matched rate?
- Measured before any run, from recorded streams (no GPU):
  - *The first frame is cheap.* In recipe C's streams on the 10 stretches
    (step 2) the intra frame, which the image model codes, is 1.4–9.6%
    of the bytes (median about 3%; the most at the lowest QP), parsed from
    the published containers. Adapting the image model can save at most
    that share of rate directly (its oracle: an intra frame costing
    nothing), so it is dropped from step 4.
  - *Standing under the selection* (median BD-rate against SVT-AV1 over the
    10 stretches). DCVC-UF off the shelf loses on every metric on every
    stretch (LPIPS_V +39% to +451%). Recipe C beats it on LPIPS_V only
    (−41%); DISTS_V +28%, PSNR_V +121%, and flicker is worse at every rate
    on 7 of 10 stretches (+271% to +471% on the other 3): 1 of 4. On the
    pilots every β > 0 arm of step 2 scores 2 of 4 and C wins the tie
    (mean −23% against A's −11%).
  - *A mechanism for the flicker.* DCVC-UF trains with per-frame weights
    within each group of 8 (`mse_8frames_sum`): 1.5 on the last frame
    (the next group's reference), 0.4 on the odd and 0.16 on the even
    positions, so quality swings within every group by design.
- New judge, held out of every loss: KID on V, HiFiC's patch protocol
  (patches of 256 px, two per frame where V covers at least 75%, the
  foreground pasted back from the source, the same patches for every
  method; `g5c.kid_patches`), on torchvision's Inception-v3 ImageNet pool
  features (`Models/Inception/inception_v3_google-0cc3c7bd.pth`, sha256
  `0cc3c7bd…a5ef`, staged under `background/g5c-2026-10-11/inputs/`),
  unbiased MMD² with the cubic kernel, ×1000. It is not the TF FID
  network (torch-fidelity is not in the environment), so its values rank
  our codecs and are not comparable with published FID or KID. Also
  recorded: PSNR_V per position within the groups.
- Development set, fixed now: P26_02 and P06_03 (the pilots), P06_10 and
  P02_09 (4th and 7th–8th of 10 by C's LPIPS_V gain), so the median is
  not the pilots'. The fast losers (P04_13, P06_05) are step 5's scope.
  The chosen recipe then runs on all 10.
- Anchors: SVT-AV1 at G5's CRFs on frame and fill, and DCVC-UF off the
  shelf, re-coded on the ProPainter prefills and scored with KID
  (`g5c.py rescore`).
- Arms (β = 0.005, ProPainter's fill, 540 steps; `g5b.py finetune`): C
  again (the replicate, for training noise; GPU training is not
  bit-reproducible), W flat frame weights (0.4175 on every position, the
  same total; `--frame-weights`), L learning rate 3 × 10⁻⁵ (3×; `--lr`).
- Decision rule, fixed now. An arm's score is the number of PSNR_V,
  LPIPS_V, DISTS_V, flicker_V and KID_V on which its median BD-rate
  against SVT-AV1 over the 4 stretches is below 0 (worse at every rate
  counts as not beating), ties broken by the mean of the five medians,
  each BD-rate capped at +100% (worse everywhere counts +100%). An arm
  replaces C if it scores more than C's replicate, or the same with a
  mean lower by more than max(5 points, twice the difference between C
  and its replicate on the four metrics both have). Two winners combine
  in round 2.
- Hypotheses. W: C's PSNR_V spread over the 8 positions is at least
  1 dB and W's under 0.5 dB, and flicker_V at matched rate falls by 10%
  or more, with LPIPS_V within ±10% of C. L: LPIPS_V 5–15% better than C
  (540 steps at 10⁻⁵ under-fit the excerpt), PSNR_V unchanged.
- Competing explanation for the flicker: generated texture that changes
  from frame to frame (β > 0 added about a third, step 1), not the
  groups; then W flattens the profile but leaves flicker as it is.
- Budget: 4 rescore jobs (≤ 20 minutes each, Ada or A6000) and 12
  training jobs (≤ 45 minutes each, Ada), about 8.5 GPU-hours.
- Outcome (code `32d55ab` for every job, from each `ready.json`; training
  on Ada, 540 steps, all converged by G5b's rule, no non-finite steps).
  Anchors: P26_02 `20261011T085154Z-a3bb2f13` (Ada), P06_03
  `…085245Z-a0cd2c6b` (A6000), P06_10 `…085316Z-05ba2a56`, P02_09
  `…085329Z-34bbf0e8` (Ada). C: `…085350Z-4c056175`, `…085408Z-6602fb50`,
  `…085426Z-94ba7492`, `…085446Z-8605a9eb`. W: `…085505Z-c5bad6a9`,
  `…085521Z-32371297`, `…085538Z-e705ae6f`, `…085558Z-423565ea`. L:
  `…085614Z-ea0b9f98`, `20261011T163301Z-e47ae994` (P06_03; its first
  submission `…085631Z-ed932c65` never finished transferring, was
  inspected, never ran, and was submitted anew), `…085712Z-f0960193`,
  `…085733Z-d794a7ea`. Same order of stretches as the anchors.
  Median BD-rate against SVT-AV1 over the 4 stretches (each capped at +100%):
  | Arm | LPIPS_V | DISTS_V | PSNR_V | flicker_V | KID_V | Beats | Mean |
  |---|---|---|---|---|---|---|---|
  | DCVC-UF off the shelf | +100% | +100% | +78% | worse | +100% | 0 | +96% |
  | C (step 2, old anchors) | −55% | +3% | +90% | worse | — | 1 | |
  | C replicate | −55% | +3% | +91% | worse | +94% | 1 | +47% |
  | W flat weights | −50% | +9% | +94% | worse | +65% | 1 | +44% |
  | **L lr 3 × 10⁻⁵** | **−65%** | **−10%** | +84% | worse | +51% | **2** | **+32%** |
  - Training noise: C and its replicate differ by 0.2 points on the mean
    of the four shared medians, so the margin is the 5-point floor.
  - **L is adopted** (2 metrics against 1). Per stretch it improves
    LPIPS_V most where C was weakest (P06_10 −44% → −59%, P02_09 −29% →
    −39%) and turns DISTS_V's median below 0 (the calm pilots −73% and
    −46%; P06_10 and P02_09 still +26% and +37%); flicker at matched rate
    is unchanged (ratio to C 0.99–1.01). Hypothesis (LPIPS_V 5–15%
    better, PSNR_V unchanged): right; PSNR_V even improves a little.
  - **W is not adopted** (1 metric, mean 3 points better than C, under the
    5-point margin). The mechanism holds but is weak: C's PSNR_V spread
    over the group's 8 positions is 2.2–3.9 dB (DCVC-UF off the shelf
    2.9–3.5, SVT-AV1 0.9–1.3), W's 1.6–2.5 dB, not under 0.5; flicker at
    matched rate falls only 5–6%, at about 5 points of LPIPS_V. Both
    hypotheses wrong in size. DCVC-UF's hierarchical weights are its
    rate–distortion design (sharp references for the next group), and
    fine-tuning keeps much of the swing.
  - Flicker stays worse than SVT-AV1 at every rate on every stretch for
    every arm, and PSNR_V stays far behind: both are DCVC-UF's (off the
    shelf it loses on all five metrics on all four stretches), not the
    adaptation's, which improves LPIPS_V, DISTS_V and KID_V.
  - GPU time: about 9 hours (16 jobs).

### 2026-10-11 — G5c step 4, round 2: a higher learning rate
- Question: L (3 × 10⁻⁵) beat C (10⁻⁵); is the gain still rising?
- Arm L2: learning rate 10⁻⁴, otherwise L; same 4 stretches and anchors.
- Decision rule, fixed now: L2 replaces L if it beats SVT-AV1 on more
  metrics, or on as many with a mean lower by more than 5 points; a run
  with non-finite steps or not converged by G5b's rule is not adopted.
  Then the recipe runs on the other 6 stretches (their anchors are
  rescored with KID now, `g5c.py rescore`, independent of the choice).
- Hypothesis: L2 is within ±5 points of L (the gain flattens), with a
  risk of instability at 10⁻⁴ (the gradient is clipped at 0.2).
- Budget: 4 training jobs (Ada) and 6 rescore jobs (Ada or A6000),
  about 3.5 GPU-hours.
- Outcome (Ada; P26_02 `20261011T184234Z-31b797df` and P06_03
  `…184244Z-d33d60f6` at code `2ad8693`, exclusive; P06_10
  `…213150Z-8f4640c2` and P02_09 `…213201Z-fd642194` at `86e541b`, the
  first shared-GPU jobs, same training code), against the corrected
  anchor (entry below): L2 beats SVT-AV1 on 1 metric (LPIPS_V −58%;
  DISTS_V +6%), L on 2 (LPIPS_V −61%, DISTS_V −2%); L2 also failed to
  converge on P26_02 (the monitor still fell 5.9% over the last fifth).
  **L2 is not adopted; the recipe is β = 0.005, learning rate 3 × 10⁻⁵,
  ProPainter's fill, 540 steps.** Hypothesis (within ±5 points, with a
  risk of instability): right. Anchors of the other 6 stretches,
  rescored with KID: P03_10 `…184136Z-395aa792`, P06_05
  `…184156Z-92a0ebd2`, P09_106 `…184205Z-4838ab3f`, P37_102
  `…184215Z-cef9da1e`, P06_106 `…184225Z-fc129d92` (code `2ad8693`),
  P04_13 `…213139Z-48eb8982` (`86e541b`, after `…184146Z-e007b2f1` was
  contended with nothing to resume and `…201618Z-4cdfa689` was
  cancelled unstarted to run shared).
- Recipe on all 10 stretches (L; the 4 development runs above plus P03_10
  `20261011T233134Z-71d961b4`, P04_13 `…233143Z-06b131c9`, P06_05
  `…233152Z-26887ec8`, P09_106 `…233201Z-058c940e`, P37_102
  `…233211Z-d001e77e`, P06_106 `…233220Z-2f6b67a2`; code `d64d6ba`, Ada,
  shared GPUs, all converged). BD-rate against SVT-AV1 (corrected anchor),
  median over the 10 | stretches below 0, with step 2's C beside it:
  LPIPS_V **−50%** | 9 of 10 (C −39% | 8); DISTS_V +19% | 4 (C +30% | 4);
  PSNR_V +98% | 0 (C +130%); flicker worse on all 10; KID_V +37% | 1.
  By stretch, LPIPS_V: P26_02 −92%, P37_102 −68%, P06_03 −66%, P06_10
  −55%, P06_106 −51%, P03_10 −50%, P02_09 −31%, P09_106 −31%, P06_05 −8%,
  P04_13 +12%. DISTS_V beats SVT-AV1 on the four calmest (P26_02 −73%,
  P06_03 −38%, P37_102 −31%, P06_106 −11%). Over all 10 the recipe beats
  SVT-AV1 on 1 of the 5 metrics; on the calm stretches on 2 (LPIPS_V,
  DISTS_V).

### 2026-10-12 — G5c step 4, round 3: a longer schedule (fast scenes first)
- Question (G5d step 2's rule c): recipe L lifted the fast-scene oracle
  by about 30 points, so the scene model is under-fitted there; does
  training longer help the deployable model (fitted on the stretch's
  other frames) on fast and on calm stretches?
- Arm X: recipe L continued from its published checkpoint for another
  540 steps (1,080 in all; a fresh optimizer, `g5b.py finetune --init`),
  on the 4 development stretches (from L's jobs there) and on the two
  fast ones, P04_13 and P06_05 (from L's 10-stretch run). Scored with
  KID and the rest as round 1.
- Decision rule, fixed now: X becomes the recipe if on the development
  set it beats SVT-AV1 on more metrics than L, or on as many with a mean
  lower by more than 5 points. On the fast stretches every metric is
  reported against SVT-AV1 and a win is any BD-rate below 0, however
  narrow (user, 2026-10-12: a narrow win is still a win, not hidden).
- Hypotheses: on the calm development stretches within ±5 points of L
  (near their ceiling); on the fast ones LPIPS_V improves by 10 points
  or more (P06_05 −8% → −18% or better, P04_13 +12% → +2% or better);
  DISTS_V and PSNR_V stay lost there.
- Competing explanation if the fast stretches do not improve: the
  oracle's gain came from fitting the excerpt's own motion, which the
  stretch's other frames do not share, so more steps on them cannot
  reach it.
- Budget: 6 training jobs (≤ 45 minutes, Ada, shared), about 4
  GPU-hours.
- Runs (Ada, shared GPUs, all converged; every job continued from L's
  published checkpoint of the same stretch). P26_02
  `20261012T073147Z-28138c3c`, P06_03 `…073156Z-778f53cb`, P02_09
  `…073216Z-cbafaee6`, P06_05 `…073237Z-28b93c3c` (code `f9001ed`).
  P06_10 and P04_13 ran out of their 2,700 s while another user's
  process shared the GPU for 29 and 39 minutes (`…073205Z-48d358ad`,
  `…073226Z-5361a155`); their salvaged trainer state was continued
  (`g5b.py --resume`, full stage only): P06_10 `…085331Z-fc0602d1`,
  P04_13 `…085441Z-51ff410f` (code `489dcfa`, gpu6 only while gpu5's
  worker missed heartbeats under other users' load). A first resume
  (`…082655Z-fb3e5c39`) failed its smoke, which had loaded the late
  state and trained no step (fixed in `489dcfa`); `…083344Z-25cad55b`
  was cancelled before it could fail the same way.
- Outcome. BD-rate against SVT-AV1, L (540 steps) → X (1,080):
  | Stretch | LPIPS_V | DISTS_V | PSNR_V | KID_V |
  |---|---|---|---|---|
  | P26_02 | −92% → −93% | −73% → −70% | +53% → +58% | +12% → +10% |
  | P06_03 | −66% → −71% | −38% → −40% | +96% → +90% | +3% → −7% |
  | P06_10 | −55% → −63% | +39% → +24% | +173% → +161% | +154% → +140% |
  | P02_09 | −31% → −37% | +34% → +26% | +136% → +137% | +137% → +111% |
  | P06_05 (fast) | −8% → **−19%** | +154% → +135% | +331% → +325% | +124% → +117% |
  | P04_13 (fast) | +12% → **+0.3%** | +113% → +101% | +327% → +322% | +193% → +181% |
  - Development set: X beats SVT-AV1 on 2 metrics like L (LPIPS_V
    −67%, DISTS_V −8% median), mean 3 points lower than L's. **By the
    rule X does not replace L** (margin 5 points), although it is better
    on LPIPS_V on every stretch (1–11 points) and on DISTS_V on 5 of 6:
    longer training helps steadily but little per step. PLAN's "final
    results train to convergence" stands for the paper's runs.
  - Fast stretches: P06_05 now beats SVT-AV1 on LPIPS_V by 19%, a win
    (narrow, reported as one); P04_13 ties (+0.3%). DISTS_V, PSNR_V and
    KID_V stay lost there. Hypotheses: right (P06_05 −18% or better,
    P04_13 +2% or better), and the calm stretches moved 1–5 points.
    The oracle's −22% and −31% leave little room: on fast scenes the
    scene model reaches LPIPS_V parity or a small win, no more.
  - GPU time: about 5 hours (6 jobs, 2 timed out and continued).

### 2026-10-12 — G5c: model size of the final recipe
- Question (PLAN G5c, "done when": model size per variant and payback):
  how small does recipe L's scene model ship? Step 1 measured β = 0.026's
  model; the recipe's 3× learning rate may move the weights further from
  the public ones, so the change may compress worse.
- Method: `g5c.py compress` as step 1 (per-output-channel quantization,
  LZMA per tensor, the excerpt re-coded with each variant on the same GPU
  class), now coding the arm's own ProPainter fill (`--prefill`, new) and
  scoring KID. Variants: whole-16 (the reference), delta-8, delta-4,
  delta-2. Arms: L on the pilots (P26_02 `20261011T085614Z-ea0b9f98`,
  P06_03 `20261011T163301Z-e47ae994`).
- Decision rule (step 1's, `DECISION["keeps_quality_bd"]`): a variant
  keeps quality if its BD-rate on LPIPS_V against whole-16 is +5% or
  better; the smallest such variant is the shipped size, and payback is
  recomputed from recipe L's rate saving against SVT-AV1.
- Hypothesis: delta-4 still keeps quality (about 50 MB, as step 1) and
  delta-2 does not.
- Budget: 2 jobs, ≤ 45 minutes each on Ada (shared), about 1.5 GPU-hours.
- Outcome (code `f57259c`, Ada, shared; P26_02 `20261012T104916Z-b516e685`,
  P06_03 `…105023Z-6f638c97`). Size (LZMA) and BD-rate on LPIPS_V against
  whole-16, P26_02 | P06_03: whole-16 218 MB; delta-8 111 MB, +0.9% |
  +0.8%; **delta-4 50 MB, +2.2% | +0.4% (keeps quality)**; delta-2 11 MB,
  +338% | +115% (breaks). Re-encoded streams with whole-16 match the
  arm's published ones within 0.01% of bytes. **The recipe ships as its
  4-bit change from the public weights, 50 MB**, as step 1's model
  did (hypothesis right): the higher learning rate does not make the
  change less compressible. Payback at 50 kbps of SVT-AV1 (recipe L's
  LPIPS_V saving −92% | −66%): 2.4 | 3.4 hours of stream.
