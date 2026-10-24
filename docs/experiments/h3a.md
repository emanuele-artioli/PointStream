# Experiments: H3a

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-10 — H3a: texture-transfer oracle for hand rendering
- Question. Could a renderer that carries a hand's appearance from
  reference frames sent earlier beat SVT-AV1 on VISOR's hand pixels at equal
  rate? This covers a textured MANO mesh, a mesh with a learned residual,
  and a forearm drawn from the mask's extent. LPIPS decides, with PSNR
  reported beside it (user, 2026-10-10): the render replaces the pixels, as
  in GenStream. H1 bounds the pixel route: carried by homographies, a hand's
  reference falls below 30 dB within 0.04 s (median). So a render will not
  reach SVT-AV1's PSNR without a residual, and the open question is
  perceptual.
- Region. Per frame of evaluation set v2's 240-frame windows: the union of
  VISOR's left- and right-hand masks (mask set `visor_dense_sam_fill`, whose
  hands are B1's), forearm included, since H3 renders the arm too (H1:
  49.7% of the hand masks).
- SVT-AV1. B2's streams at CRF 41, 48, 55 and 62, decoded by dav1d. Each
  frame is composited: decoded pixels inside the region, source pixels
  outside, so only the hands differ. Per frame: PSNR and LPIPS inside the
  region, with B2's RGB view and LPIPS (AlexNet spatial map at full
  resolution, mean inside the region). Rate: H1's bits on the same pixels at
  that CRF (hand + forearm + hand without fit, fill set, per item).
- Oracle, per CRF c and margin δ. Frames are taken in order. The bank holds
  the frames sent so far as references, each as SVT-AV1 decoded it at c.
  - For a frame with a hand, each of the 6 most recent references is warped
    to it by DIS optical flow from the source frame to the reference's
    source frame. This is the oracle's advantage: the motion comes from the
    target itself.
  - The warp with the lowest squared error in the region is kept. If its
    LPIPS exceeds SVT-AV1's at c on that frame by more than δ, or the bank
    is empty, the frame is sent as a new reference and shows SVT-AV1's
    decoded pixels.
  - Rate: references × b_c + the pose stream. b_c is SVT-AV1's bits on the
    hand pixels at c per hand-frame: an inter-coded frame's cost, below
    what a reference sent alone would cost. The pose stream is H2's chosen
    coding of WiLoR on VISOR (`code.json`, 100 ms budget: 1.93 kbps), though
    the oracle does not use it.
  - δ ∈ {0.02, 0.05, 0.10}.
- Why it bounds the gated renderers. They place reference pixels by a
  warp driven by the decoded pose. The oracle has every advantage over them:
  true motion from the target, the target's own mask, references chosen
  using the target, and an optimistic reference cost. It does not bound
  renderers that synthesise appearance found in no reference
  (pose-conditioned generation, learned shading); a learned residual only
  partly.
- Decision rule (fixed before any run; `h3.ORACLE_DECISION`, applied by
  `h3 report`).
  - Per item and per operating point (c, δ), the oracle's mean region LPIPS
    is compared with SVT-AV1's at the oracle's rate. SVT-AV1's value is
    linear in log-rate between CRF points; below CRF 62's rate it is CRF
    62's value, and above CRF 41's it is CRF 41's.
  - The difference (oracle − SVT-AV1) is averaged over the 34 items, with a
    95% paired bootstrap interval (1,000 resamples by item).
  - **The oracle passes if the mean difference is below zero at some
    (c, δ).** The point estimate decides, because a bound should err toward
    building.
  - Pass: a textured-mesh renderer (hand and forearm) is built next, run on
    WiLoR's and HaMeR's coded poses (PLAN H3), and scored the same way.
  - Fail: no texture-transfer renderer is built. The outcome says whether
    refreshes or warp quality failed. Pose-conditioned generation would need
    its own oracle, and the user decides whether to pursue it.
  - Reported beside: PSNR at equal rate, the share of hand-frames sent as
    references, and the difference with references costing 3× and 5× b_c
    (closer to an intra frame).
- Hypothesis. The oracle passes at low rate. With CRF 41 or 48 references
  sent on 10–30% of hand-frames, the warps stay within δ of the
  references' LPIPS at a rate near CRF 62's, below CRF 62's LPIPS (B2's
  frame LPIPS: 0.101 at CRF 41, 0.165 at CRF 62). At 3× reference cost the
  margin halves.
- Competing explanation. Hands change appearance faster than references can
  be refreshed: articulation shows new surfaces, and blur and held objects
  occlude them. H1's references died in 0.04 s, so refreshes are needed on
  most frames and the oracle tracks SVT-AV1's curve without beating it.
- Budget. GPU only for LPIPS; flow and warps on 16 CPU threads. Smoke: one
  item, 48 frames, ≤ 600 s, including a CPU-against-GPU LPIPS check. Pilot:
  B2's four pilot items (P01_107, P09_106, P02_02, P03_10), ≤ 45 min. Full:
  the other 30 items in parts of ≤ 45 min. Ceiling 3 GPU-hours, 5 h wall.
- Jobs (code `ea5add8`, environment `pointstream-20261006T113321Z`; inputs
  and specs in `pointstream-data/visor/h3-2026-10-10/`; every validator
  10/10). Pilot `20261010T073916Z-5058a257` (gpu2, RTX A6000; B2's four
  pilot items; 1,355 s, 5.6 min per item; smoke 65 s, LPIPS CPU against GPU
  7e-6). Full, the other 30 items in five parts of six:
  `20261010T082506Z-54ab4623` (A6000, 2,001 s), `…082512Z-366ce9ad` (RTX 6000
  Ada, 2,099 s), `…082551Z-df6149fa` (A6000, 2,218 s), `…082528Z-70e1e80a`
  (A6000, 2,479 s), `…082534Z-e748833b` (A6000; it reached the 2,700 s cap
  after 5 of its 6 items, which were all saved in `partial.tar`; the
  validator passes on them), and `…100218Z-87604272` (Ada, 212 s; the sixth
  item, P37_102). `…082517Z-8338cee3` failed its transfer checksum and
  was never published, so it never ran. About 3.6 GPU-hours. Report
  (`h3 report`, all 34 items):
  `pointstream-data/visor/h3-2026-10-10/report-ea5add8/h3a.json`
  (`95ac0e39…d4c3`).
- Outcome. **The oracle passes.** SVT-AV1 on the hand pixels, as means over
  items: CRF 62 42.2 kbps, LPIPS 0.122, PSNR 34.9 dB; CRF 48 140.5 kbps,
  0.085, 38.8 dB; CRF 41 212.6 kbps, 0.077, 39.8 dB. The best oracle point,
  CRF 48 references with δ 0.02, sends 26% of hand-frames as references:
  41.9 kbps (pose stream 1.93 included), LPIPS 0.091.
  - Against SVT-AV1 at the same rate it is 0.025 better in LPIPS (CI
    −0.029 to −0.022), on all 34 items, and 0.9 dB better in PSNR (CI
    +0.6 to +1.2).
  - 7 of the 12 operating points win with the whole interval below zero:
    CRF 41, 48 and 55 references at δ 0.02 and 0.05, and CRF 41 at δ 0.10.
    The others tie or lose: references at CRF 62 are too poor, and at δ 0.10
    they are kept too long.
  - At δ 0.05 the oracle runs at 14–35 kbps, below SVT-AV1's lowest point,
    still better in LPIPS (−0.008 to −0.020) but 1–2 dB worse in PSNR.
  - Cost sensitivity: at 3× the reference cost the best point barely wins
    (CRF 55, δ 0.02: −0.005, CI −0.009 to −0.001, 23 of 34 items). At 5× no
    point wins.
- Reading. The hypothesis holds: references sent on about a quarter of
  hand-frames and carried forward by the true motion beat coded pixels at
  CRF 62's rate, in LPIPS and slightly in PSNR. Against the competing
  explanation, references need not be sent on most frames: about one
  frame in four suffices at δ 0.02. H1's 0.04 s reference life was measured
  at a fixed 30 dB with homographies, not against the codec's own quality
  with dense motion.
  - Two limits bound what the pass means. The warps here use the target's
    own motion, which a pose-driven renderer must approximate from a coded
    pose (H2: 17 px median 2D joint error on HOT3D). And the margin
    disappears when a reference costs 3–5× an inter-coded frame. References
    need not be coded on their own: they can be sent as their own
    low-frame-rate SVT-AV1 stream, each predicted from the last reference
    (user, 2026-10-10). Its cost (frames about four apart at 26% refresh)
    is measurable and has not been measured.
- Equal-quality view (computed after the rule from `h3a.json`; not part of
  the decision). The rule compares at equal rate and holds SVT-AV1 at CRF
  62's quality below CRF 62's rate, which understates the oracle. Measured
  as the rate SVT-AV1 needs for the oracle's LPIPS (per item, linear in
  log-rate between CRF points; points worse than CRF 62 have no SVT-AV1
  match and are counted apart), the median saving on hand pixels is:
  - CRF 48 references, δ 0.02: 65% at the inter-frame cost, 32% at 2×, −1%
    at 3×.
  - CRF 55 references, δ 0.02: 71%, 46%, 21%.
  - CRF 55 references, δ 0.05: 74%, 54%, 34% (28 items; 6 are worse than
    CRF 62).
  The oracle's rate is about 95% references and 5% pose. Between
  references the frames are warped by the true flow, not generated from the
  pose. So it bounds texture transfer with refresh, not a generator that
  renders from one reference and the pose alone. Such a generator would
  cost about the pose stream (2 kbps) and needs its own test.
