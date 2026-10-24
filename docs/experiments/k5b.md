# Experiments: K5b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-20 — K5b (1): the ball renderer's oracle (a drawn ball, billed)
- Question (PLAN K5b 1): K5's oracle pasted the source's own ball pixels
  and billed only the trajectory's entropy. Does a ball the decoder
  *draws* keep the win when every byte of what it needs is sent? Two
  renderers, both from the coded positions and a per-window appearance
  sent once: (a) a **sharp disk** (anti-aliased, radius r, colour c);
  (b) a **streak**, the same disk averaged over the exposure along the
  coded velocity (alpha = the fraction of the exposure each pixel spends
  covered, so a fast ball fades as it lengthens, as a camera blurs it).
  Velocity is the central difference of the coded positions (one-sided
  at a run's ends, zero for a lone frame), so it costs nothing beyond
  the positions and one frame of decoder delay. Still an oracle: the
  positions are TrackNet's labels (a tracker comes in K5b 2).
- What is sent (`src/pointstream/ball.py`, a `BALL` segment of the
  PointStream container, its tag and length bytes billed): per window,
  an on/off bit; when on, colour (3 × 8 bits), radius index (4 bits,
  1.5–6 px), exposure index (3 bits, 0–1 of a frame interval; 0 is the
  disk), phase index (3 bits, the streak's centre at −½…+½ of the
  frame's displacement from the labelled point), the Exp-Golomb order of
  the residuals; visibility as alternating run lengths (Exp-Golomb);
  each run's first position absolute (11 + 10 bits at 1280×720), then
  constant-velocity residuals (signed Exp-Golomb, K5's predictor). Rate
  = the segment's bytes per window, per window (K5 pooled an entropy; a
  real code bills more). Decoding the payload returns the positions and
  parameters exactly (a test).
- The encoder's fit (analysis by synthesis, as an encoder may: it holds
  the source and its own decode): per window and coder point, a grid
  over radius × exposure × phase (the disk: radius only), the colour in
  closed form (least squares of the composite against the source, then
  rounded to 8 bits), choosing the combination that most lowers the
  window's whole-frame squared error; off when none lowers it (then the
  segment is one byte of payload).
- Windows, coders, judges and comparison: K5's, unchanged (8 TrackNet
  windows, SVT-AV1 preset 4 at CRF 35–62, DCVC-UF HT-L at QP 0–63 on the
  A6000; ball-region PSNR within 8 px of the label, frame PSNR; BD-rate
  against the pooled envelope and each coder's own). Added: a
  *streak-region PSNR* (within 8 px of the segment the label's own
  velocity sweeps in one frame), reported, since K5's region misses most
  of a 20–40 px streak. Recall is not measured here: YOLO26x was not a
  valid judge (46% on the source); the tracker of K5b (2) becomes the
  judge. K5's pixel oracle is recomputed in the same job, so each
  renderer's distance to the bound is measured on the same decodes.
- Decision rule (`k5b.DECISION`, fixed before any run). A renderer is
  adopted when, over a coder underneath (SVT-AV1 or DCVC-UF), it beats
  the pooled envelope on ball-region PSNR on ≥ 6 of the 8 windows, each
  by BD-rate ≤ −25% or better at every rate (K5's bar, now with real
  bytes and a drawn ball), **and** its frame-PSNR BD-rate against its own
  coder's envelope is ≤ +1% on ≥ 6 of 8 (the ball mode does not hide a
  cost elsewhere in the frame). If both renderers pass, both stay
  available and the user's side-by-side preference picks the codec's
  default; a preferred renderer that fails the rule is not the default.
  If neither passes, the ball mode needs a richer appearance (a per-window
  sprite of the source's ball, its own oracle) and the reading says which
  half failed.
- Hypothesis: the streak keeps most of K5's gain (pooled BD on DCVC-UF
  −45% to −55%, K5's oracle −57%) and may beat the pixel oracle on fast
  windows, where the oracle pastes only 4 px of a long streak; the disk
  passes on DCVC-UF but by less (about −35%); both pass on SVT-AV1 by less
  than on DCVC-UF (K5: −38%). Frame PSNR moves by under 0.3%. The
  billed rate stays under 0.5 kbps per window (K5's entropy: 0.23).
- Competing explanations: (1) TrackNet's integer labels and their
  convention (which point of a streak is marked) misplace the drawn ball,
  costing the disk on fast frames; the fitted phase shows it. (2) The
  fitted colour mixes ball and court (a 3–5 px ball in 4:2:0), giving a
  dull ball that scores well and looks wrong; the side by side shows it.
  (3) At high rates the decode keeps the ball, and a second drawn ball
  doubles it; the per-point fit can switch the mode off, and the curves
  show where it does. (4) The streak wins on PSNR only because PSNR
  rewards blur; the user's preference is the check for display.
- Side by side (for the user's preference, K5b): per window, at DCVC-UF
  QP 21 (about 40 kbps, where the decode loses the ball) and QP 42, a
  2 × 2 clip at equal rate (source, the pixel coder, + disk, + streak;
  the ball mode adds under 1% of the rate), as a ball-following crop
  enlarged 3×, and a full-frame 2 × 2 at QP 21.
- Budget: DCVC-UF on the A6000 (K5's table), SVT-AV1 on 16 threads, no
  detectors. Smoke: one window, 48 frames, ≤ 600 s. Pilot: part 1, 4
  windows (K5's pilot windows first), ≤ 45 min; part 2, the other 4,
  ≤ 45 min, run only if part 1 leaves the rule open. Ceiling 2
  GPU-hours and 6 h wall.
- Failed attempt (infrastructure). `20261020T184724Z-58d02fd1`
  (`14d3892`): the smoke stopped writing the full-frame clip (the label
  was drawn on a read-only frame); fixed in `523cee1`.
- Job: part 1 `20261020T185255Z-43bcc85e` (`523cee1`; RTX A6000, gpu3,
  shared with another user's process; game3/Clip3, game7/Clip2,
  game10/Clip10, game3/Clip6; full stage 1,213 s; `published.tar`
  `6520ae01…b095`; inputs, environment, checkpoints and windows as K5;
  SVT-AV1 4.2.0, dav1d 1.5.4 from the environment). Validator passed on
  the smoke and, run locally, on the full stage (12 checks). Every DCVC-UF
  point decoded deterministically on its first attempt. About 0.4
  GPU-hours with the smoke.
- Report (`k5b report` on part 1; `k5b-report.json` `a4a556a0…dfc8`). Means
  over the 4 windows (kbps; ball-region PSNR dB; decode → pixel oracle /
  disk / streak):

  | Point | kbps | ball PSNR | streak-region PSNR | frame PSNR |
  |---|---:|---|---|---:|
  | DCVC-UF QP 0 | 12.95 | 18.73 → 22.94 / 19.32 / 19.70 | 20.98 → 24.10 / 21.48 / 21.88 | 29.42 |
  | DCVC-UF QP 21 | 36.60 | 19.89 → 24.11 / 20.28 / 20.63 | 22.19 → 25.36 / 22.53 / 22.90 | 33.74 |
  | DCVC-UF QP 42 | 117.7 | 23.89 → 27.56 / 23.89 / 23.92 | 26.01 → 28.64 / 26.01 / 26.05 | 37.19 |
  | SVT-AV1 CRF 62 | 73.94 | 20.21 → 24.41 / 20.56 / 20.87 | 22.54 → 25.73 / 22.86 / 23.18 | 35.31 |
  | SVT-AV1 CRF 55 | 176.0 | 23.62 → 27.74 / 23.67 / 23.77 | 25.91 → 28.97 / 25.95 / 26.04 | 38.35 |

  The ball segment costs 0.22–0.33 kbps when on (one byte of payload,
  0.003 kbps, when off); the fit turns it off at DCVC-UF QP ≥ 42 and
  SVT-AV1 CRF ≤ 48 on most windows. BD-rate on ball-region PSNR against
  the pooled envelope, per window (game10/Clip10, game3/Clip3,
  game3/Clip6, game7/Clip2): K5's pixel oracle on DCVC-UF −71.9%, −55.8%,
  −58.3%, −51.0% (K5: −57.3% median, reproduced); the disk on DCVC-UF
  +7.3%, +6.7%, +4.5%, +0.7%; the streak on DCVC-UF +3.2%, +0.8%, +0.8%,
  −3.9%; on SVT-AV1 the disk +21% to +30%, the streak +19% to +28%.
  Frame PSNR against each coder's own envelope: +0.0% to +0.8% (within
  the +1% bar). Fitted streaks: radius 1.5–4 px, exposure 0.43–1.0 of a
  frame, phase −1/6, colours dull (e.g. 182/182/172 on a white-ball
  window).
- Decision (by the rule): **no renderer is adopted.** Both fail the ball
  half on all 4 windows with either coder (the rule needs 6 of 8, so 4
  failures decide it); part 2 was not run. The frame half passes.
- Reading. The hypothesis fails: a per-window appearance keeps 10–20% of
  the pixel oracle's gain on the ball region (DCVC-UF QP 21: +0.4 dB for
  the disk, +0.7 dB for the streak, +4.2 dB for the oracle), not most of
  it. The review strips show why: fast balls are long, bright streaks;
  the drawn streak has the right direction but stays faint, and the
  fitted colour is dull (competing explanation 2: the least-squares fit
  hedges a ball it cannot place exactly). Competing explanation 3 holds
  in part: the decode keeps a faint ghost above the ball at QP 21, which
  the drawn ball does not cover.
  A diagnostic (`experiments/tennis/k5b_diagnostic.py`; not evidence:
  source JPEGs read by OpenCV, a temporal
  median as the true background, TrackNet's camera being still; 8
  windows, CPU; `k5b-diagnostic.json` `be8237c1…f5d5`) separates the causes. Ball-region PSNR over the median,
  per window: the per-window renderer recovers 15–60% of what pasting the
  source within 4 px recovers (+0.4 to +2.4 dB against +2.4 to +6.7 dB);
  the same streak shifted per frame by its best offset within 4 px (a
  position oracle; median offset 1.4–3.2 px from the label) adds about
  1.5 dB; a free setting and colour per frame at the label (an appearance
  oracle) adds about the same; each alone stays 0.1–2.3 dB under the
  paste. A sensor-saturation gain (alpha = min(1, g × coverage)) is
  chosen on one window only. So the drawn ball needs both: positions
  refined by the encoder to where the ball is (TrackNet's integer labels
  are 2–3 px off the best fit) and an appearance sent per frame (about
  35 bits, 1 kbps), each an encoder decision that is billed.
- Next (K5b 1b, its own oracle, proposed): per-frame position refinement
  (±4 px by fit, residuals billed through the same Exp-Golomb code) and a
  per-frame appearance (setting and colour, predicted from the previous
  frame, billed), on the same decodes and judges, with K5b 1's rule.
- Rendering preference (side by side sent to the user 2026-10-20: three
  windows' ball-following clips and one full frame at DCVC-UF QP 21):
  pending.

### 2026-10-20 — K5b (2): an encoder-side ball tracker (audit and rule)
- Question (PLAN K5b 2, ball stage): a tracker that finds the ball on the
  source for the ball mode's encoder, and that also judges the codecs'
  outputs (K5: YOLO26x finds 46% of the source's balls, SAM 3.1's text
  prompt 56%; neither is a valid judge).
- Audit (2026-10-20). **No TrackNet-family weights are in `Models`**
  (`YOLO/` holds COCO detectors and trackers' configs only). Candidates:
  - *WASB* (Tarashima et al., BMVC 2023; `nttcom/WASB-SBDT`, **MIT**):
    an HRNet heatmap tracker with tennis weights (`wasb_tennis_best.pth.tar`,
    Google Drive) and the TrackNetV2, ResTrackNetV2, MonoTrack, BallSeg and
    DeepBall baselines retrained by the same code. Its tennis config
    trains on TrackNet **games 1–7** and tests on games 8–10, so five of
    K5's eight windows (games 2, 3, 6, 7) are in its training set; only
    game8/Clip6 and game10/Clip10 are held out. Plain PyTorch (tested on
    Python 3.8, CUDA 11.3); no conflict expected with the pointstream
    environment's torch, checked in the smoke.
  - *RacketVision's BallTrack* (`OrcustD/RacketVision`, **MIT**):
    TrackNetV3 with a median background frame, checkpoints on Hugging Face
    (`linfeng302/RacketVision-Models`; the README names a badminton
    example, a tennis checkpoint is unconfirmed); torch 2.1.2 and
    mmengine.
  - *TrackNetV3* (`qaz812345/TrackNetV3`, **MIT**): shuttlecock only.
  - TrackNet v1 (the dataset's own model, no licence stated): not used.
- Labels on disk. TrackNet: 95 clips, 10 games, 1280×720, every frame
  (no licence stated; already used by K4–K5). RacketVision tennis
  (**MIT**): 431 rallies at 1920×1080, 21,544 labelled frames (20,066
  visible; about every 11th frame), split 350 / 38 / 43 with no match in
  two splits.
- Held-out labels for the decision: TrackNet's games of K5's windows
  (2, 3, 6, 7, 8, 10) and RacketVision tennis test. Training uses TrackNet
  games 1, 4, 5, 9 (44 clips) and RacketVision tennis train.
- Judge: *detection rate*, the share of visible labelled frames with the
  tracker's position within 10 px at 720p (15 px at 1080p; 5 px
  reported), plus false detections per frame without a visible ball.
- Downloads (user, 2026-10-20: "yes, all", after asking to download every
  candidate and pick the best by a cheap test). `Models/WASB`: the seven
  tennis checkpoints of WASB's model zoo (WASB 6.1 MB, TrackNetV2 45 MB,
  ResTrackNetV2 5.1 MB, MonoTrack 11.7 MB, BallSeg 50.9 MB, DeepBall
  0.3 MB, DeepBall-Large 4.1 MB) and the code at `923462c` as a tar;
  `Models/RacketVision`: `balltrack_best.pth` (46.6 MB; sha256 as
  published), a tensors-only copy loadable with `weights_only=True`, and
  the code at `c44af2a`. sha256 in each `MANIFEST.json`. BallTrack's
  checkpoint holds mmengine log objects that Hugging Face flags as unsafe
  pickles; it was read once with an unpickler that allows only tensors,
  numpy arrays and an inert stub for the log objects. It is one model for
  the three sports, trained on RacketVision only (epoch 47), so all of
  TrackNet is held out for it.
- Stage A, the comparison (no training; the user's cheap test). Nine
  settings, each with its authors' inference: WASB (online tracker,
  windows of 3 frames at steps 3 and 1), TrackNetV2, ResTrackNetV2,
  MonoTrack (no heatmap weighting, per-frame peak), BallSeg and the two
  DeepBalls (step 1), as WASB's README evaluates them; BallTrack with the
  clip's median frame and its 4 frames ending at the scored frame (its
  training alignment; its `inference.py` uses the 4 before it). Every
  frame at 1280×720 (RacketVision's 1080p area-downscaled, labels
  scaled). Sets: TrackNet games 8–10 (held out for all nine), RacketVision
  tennis test (held out for all nine), TrackNet games 1–7 (held out for
  BallTrack only; in-training for the zoo, reported), and TrackNet games
  8–10 through SVT-AV1 preset 8 at CRF 62 (K5's collapse point; a
  robustness probe). TrackNet visibility 3 (occluded) is excluded from
  both positives and negatives.
- Rule for the choice (fixed before the run): the candidate with the
  highest mean of its detection rates on the two sets held out for all
  (TrackNet 8–10, RacketVision test) is chosen; within 0.02 of it, the one
  with the higher rate on the SVT-AV1 probe, then fewer false detections.
  Adoption without training: if the chosen candidate holds out K5's games
  (BallTrack does) and meets stage B's bar on them (≥ 0.90 on TrackNet's
  K5 games, ≤ 0.05 false detections per ball-less frame, ≥ 0.80 on
  RacketVision test), it is adopted as tracker and judge with no training.
  Otherwise stage B trains the chosen architecture with K5's games held
  out (a WASB-zoo winner is retrained, since its weights saw four of K5's
  games).
- Stage B (only after A): smoke, a pilot of ≤ 45 min (one setting of the
  learning rate and schedule, the held-out rate tracked per epoch), then
  training in parts of ≤ 45 min that each save their checkpoint.
  Decision rule: the tracker is adopted when its detection rate on the
  held-out TrackNet games is ≥ 0.90 with ≤ 0.05 false detections per
  frame without a ball, and ≥ 0.80 on RacketVision tennis test; otherwise
  the negative is reported and the ball mode keeps labelled positions in
  its claims (scoped to labelled data).
- Hypothesis: WASB leads the zoo on TrackNet 8–10 (its paper: F1 about
  0.96) but falls to about 0.6–0.8 on RacketVision's broadcast footage;
  BallTrack, trained on RacketVision's broadcast tennis, leads there and
  reaches about 0.85 on TrackNet, so the choice is between these two, and
  the codec probe separates them (a model with a background frame should
  hold the ball better on a smoothed court).
- Competing explanation: TrackNet's games 8–10 share courts and players
  with games 1–7, so stage A overstates generalization; RacketVision
  test, from other matches, is the check.
- Budget: stage A ≤ 1.5 GPU-hours in parts of ≤ 45 min (one per set);
  stage B ≤ 6 GPU-hours.
- Failed attempts (infrastructure, no results): `20261020T191508Z-70549af5`,
  `…191515Z-e36c7b87`, `…191521Z-b6a436f6` (`c913734`: the packed
  environment has neither hydra nor omegaconf; WASB's postprocessors are
  now loaded by path and its configs read as an attribute dict);
  `20261020T214419Z-17cc0776`, `…214426Z-1d7002a9`, `…214432Z-fb693dbc`
  (`d5260a7`: `--limit 0` arrived as the string "0", so one full stage
  scored no clip and its validator passed on an empty list; cancelled or
  superseded; the validator now rejects an empty stage);
  `20261020T214842Z-bf85e989`, `…214855Z-a4f4e1df` (gpu2, smokes over
  600 s beside another user's process at 100%; rerun on Ada);
  `20261020T220040Z-fe792dc2` (the SVT-AV1 probe's smoke failed the check
  that some candidate finds the ball, which none does at CRF 62; the
  probe is now checked by its decode's frame PSNR, ≥ 25 dB).
- Jobs (all RTX 6000 Ada, gpu5/gpu6, shared; environment
  `pointstream-20261009T214002Z`; inputs `inputs-k5b2.json`
  `7cb55480…cd63`): TrackNet games 8–10 `20261020T214833Z-bf224666`
  (`7590a77`; 30 clips; 1,362 s; `published.tar` `6d2aed8e…2d38`);
  RacketVision tennis test in three parts `20261020T220058Z-fc8980da`
  (`cff4f60`, whose `k5b_trackers.py` is `7590a77`'s; 14 rallies; 680 s;
  `3dbecb66…eadb`), `20261020T214901Z-0d339486` (`7590a77`; 15; 1,240 s;
  `0c53b65d…33ec`), `20261020T214907Z-eefaac4b` (`7590a77`; 14; 634 s;
  `3fe00d05…f3cd`); the SVT-AV1 probe `20261020T220308Z-436c3db3`
  (`09f0cd3`; 30 clips; 986 s; `a4178a34…afe1`); TrackNet games 2, 3, 6, 7
  `20261020T221615Z-1a87ab2f` (`09f0cd3`; 30 clips; 1,135 s;
  `5ae5a714…d31f`). Validators passed on each smoke and, run locally, on
  each full stage. Every model's parameters on `cuda:0` (asserted in each
  `k5b-trackers.json`; torch 2.10.0+cu128). About 1.7 GPU-hours of full
  stages, 2 with smokes and failed attempts: over the 1.5-hour budget
  (the RacketVision rallies are 600 frames at 60 frames/s, longer than
  estimated, and gpu2's contention cost two smokes).
- Report (`k5b_trackers report` on the six jobs; `k5b-trackers-report.json`
  `f2cec565…a42c`). Detection rate within 10 px (5 px), false detections
  per frame without a ball:

  | Candidate | TrackNet 8–10 | RacketVision test | Mean (rule) | K5's games 2,3,6,7,8,10 | SVT-AV1 CRF 62 (8–10) |
  |---|---|---|---:|---|---:|
  | BallTrack | 0.938 (0.851), 0.030 | 0.881, 0.286 | **0.910** | 0.918 (0.771), 0.022 | 0.026 |
  | WASB, step 1 | 0.923 (0.911), 0.051 | 0.834, 0.379 | 0.879 | 0.966 (0.944)\*, 0.039 | 0.018 |
  | WASB | 0.922 (0.909), 0.066 | 0.831, 0.350 | 0.876 | 0.965 (0.945)\*, 0.041 | 0.017 |
  | MonoTrack | 0.898 (0.880), 0.112 | 0.797, 0.404 | 0.847 | 0.953\*, 0.047 | 0.021 |
  | ResTrackNetV2 | 0.865 (0.844), 0.076 | 0.745, 0.300 | 0.805 | 0.909\*, 0.037 | 0.018 |
  | TrackNetV2 | 0.842 (0.829), 0.046 | 0.745, 0.291 | 0.794 | 0.928\*, 0.024 | 0.010 |
  | BallSeg | 0.618 (0.598), 0.005 | 0.637, 0.123 | 0.627 | 0.759\*, 0.002 | 0.011 |
  | DeepBall-Large | 0.390 (0.357), 0.553 | 0.424, 0.773 | 0.407 | 0.704\*, 0.154 | 0.010 |
  | DeepBall | 0.396 (0.367), 0.497 | 0.193, 1.000 | 0.294 | 0.670\*, 0.363 | 0.011 |

  \* games 2, 3, 6, 7 are in WASB-zoo training (WASB 0.988 on them
  alone), so these are not held out. TrackNet 8–10: 5,472 ball frames and
  197 without; RacketVision test: 1,947 labelled ball frames and 203
  labelled without (false rates there are high for every candidate;
  those frames were not inspected);
  K5's six games: 11,496 and 534. WASB's F1 on 8–10 is 0.954 against
  0.956 in its paper, so the wrapper reproduces its authors' inference.
  The SVT-AV1 probe (preset 8, CRF 62, median 84 kbps, frame PSNR
  32.7–34.3 dB) leaves every candidate under 0.03.
- Decision (by the rule): **BallTrack is chosen** (0.910 against WASB's
  0.879; none within 0.02, so the probe did not enter) and **adopted
  without training**: it never saw TrackNet, and on K5's six games it
  finds 0.918 of the balls with 0.022 false detections per frame without
  a ball, and 0.881 on RacketVision test (bars 0.90, 0.05, 0.80). On K5's
  8 windows it finds 0.916 (0.006 false). Stage B is not needed.
- Reading. The hypothesis holds on the choice and half on the numbers:
  WASB leads the zoo on TrackNet (0.92) and falls on RacketVision's
  broadcast footage (0.83, not as low as predicted), while BallTrack,
  trained on RacketVision's other matches, leads both. The competing
  explanation (games 8–10 share courts with 1–7) is visible in the zoo
  (WASB 0.988 on its training games, 0.922 on 8–10) and does not touch
  BallTrack. Two limits for its use: (1) it places the ball coarsely (5 px
  rate 0.771 on K5's games against WASB's 0.944; it takes a blob's box
  centre at 512×288), so the ball mode's positions want refinement by the
  encoder (K5b 1's diagnostic: about 1.5 dB from per-frame position);
  (2) as a judge it is valid (0.92 on the source), and it confirms K5's
  regime with a valid detector: through SVT-AV1 at CRF 62 every
  candidate finds under 3% of the balls (YOLO26x's 0.02 in K5 was not a
  valid judge).

### 2026-10-21 — K5b (1b): a per-frame ball, judged by detection and DISTS (rule)
- Question (PLAN K5b 1b): K5b 1 drew the ball from TrackNet's labels with
  one appearance per window and kept 10–20% of the pixel oracle's
  ball-region PSNR gain; its diagnostic put the gap on position (labels
  1.4–3.2 px off the best fit) and on a per-window appearance. Does a
  ball the encoder places and styles per frame, every byte billed, keep
  the ball visible and recognisable at the low rates where the pixel
  coders lose it, end to end from a tracker's positions?
- Judges (user, 2026-10-21). K5b 1 decided on ball-region PSNR, which
  punishes a drawn ball twice for a 2 px misplacement and rewards a dull
  one; the ball mode's claim is that the ball can be seen and followed,
  not that its pixels are the source's. Deciding: (a) **detection**,
  BallTrack (K5b 2: never trained on TrackNet, so a clean judge on every
  K5 window) on the decoded or composited frames, a hit within 10 px of
  TrackNet's label, over the window's visible frames (5 px reported);
  (b) **DISTS** (masked DISTS of `src.codecs.quality`, VGG16, all-ones
  mask) on a 64 × 64 crop centred on the label (clamped to the frame),
  composite against source, averaged over the visible frames. Reported:
  ball-region and streak-region PSNR (K5b 1's), LPIPS on the same crop,
  frame PSNR, and the false detections per ball-less frame.
- What changes from K5b 1 (`src/pointstream/ball.py`, the same `BALL`
  segment, versioned): (1) per-frame position refinement: the encoder
  moves each coded position by the fit's best offset within ±4 px, and
  the refined positions go through the same constant-velocity
  Exp-Golomb code, so their cost is billed; (2) a per-frame appearance:
  radius, exposure and phase indices and the 8-bit colour per visible
  frame, each coded as the signed Exp-Golomb difference from the
  previous visible frame (the window's first is absolute), so a steady
  ball costs a few bits a frame. The fit alternates the two (position
  with the current appearance, then the appearance at the refined
  position; two rounds), minimising whole-frame squared error as in
  K5b 1, with the off switch per window kept. As built (`ball.fit_frames`):
  each frame's alpha at every setting is tabulated once, the appearance
  step is one batched product over all settings and decodes, the
  position step tries every second offset then the 8 neighbours of the
  best, velocities in the fit are the tracker's (the decoder's, from the
  refined positions, draw the scored frames), and a frame is sent only
  where its ball lowers the crop's error (so frames can be hidden, which
  the visibility runs code). On one window with 9 stand-in decodes the
  fit takes about 67 s per position source on a laptop CPU, and the
  payload is 0.9–1.4 kbps.
- Position sources (the encoder's tracker), each a variant: TrackNet's
  labels (the bound); **BallTrack** on the source (the encoder the codec
  would ship; misses become hidden frames, false detections are drawn);
  **WASB** (step 1) on the source, cheap at inference only and more
  precise (5 px 0.944 against 0.771), disclosed: it trained on games 2,
  3, 6, 7, so on six of the eight windows its positions are not held out
  (game8/Clip6 and game10/Clip10 are). The judge stays BallTrack, scored
  against the labels, so a WASB-placed ball is judged by a detector that
  did not place it.
- Windows, coders and rates: K5b 1's (8 TrackNet windows; SVT-AV1 preset
  4 at CRF 35–62, DCVC-UF HT-L at QP 0–63 on the A6000); rate = the
  coder's bytes plus the billed `BALL` segment; K5's pixel oracle
  recomputed, and judged by the same judges, as the bound.
- Decision rule (`k5b.DECISION_1B`, fixed before any run). A renderer
  (disk or streak) with **BallTrack's positions** is adopted when, over a
  coder underneath, on ≥ 6 of the 8 windows: (i) detection: the rate at
  which the composite first finds 80% of the balls BallTrack finds on the
  source (linear in log rate between points) is at most half the rate
  the pooled pixel envelope (upper hull of detection against rate over
  both coders) needs for the same, or the envelope never gets there;
  **and** (ii) DISTS-crop BD-rate ≤ −25% against the pooled envelope, or
  better at every rate; **and** (iii) frame-PSNR BD-rate ≤ +1% against
  its own coder's envelope (K5b 1's guard). BD overlap (`k5b1b.DECISION`):
  0.01 in 1 − DISTS and 1 − LPIPS, 0.5 dB in PSNR. Six of eight needs
  both parts; part 1 alone (four windows) decides nothing by itself. The
  judge's GPU path (median and downscale on the GPU) must agree with
  K5b 2's CPU path on ≥ 97% of the source's frames (a validator check).
  If the labels' variant passes
  and BallTrack's does not, the tracker is the gap (WASB's variant says
  whether its precision closes it, and WASB retraining with K5's games
  held out becomes the next step). If neither passes, the reading says
  which judge failed and by how much. If both renderers pass, the user's
  preference from this run's side by side picks the default.
- Hypothesis: the drawn ball restores most of the source's detections
  where the pixel coders lose them (DCVC-UF QP 21–63, SVT-AV1 CRF 55–62:
  under 10% found) for about 1 kbps, so (i) passes on DCVC-UF and SVT-AV1
  alike; the streak passes DISTS by more than the disk on fast windows,
  the disk detection by more (a crisp blob is what BallTrack's heatmap
  peaks on); BallTrack's positions lose under 0.05 of detection against
  the labels' after refinement.
- Competing explanations: (1) BallTrack detects a drawn disk because it
  is a bright blob, not because it looks like the ball; the false
  detections on ball-less frames and DISTS check it, and the side by
  side shows it. (2) With BallTrack's positions, the judge may agree
  with itself (it drew the ball where it already looked); scoring
  against the labels and WASB's variant check it. (3) DISTS on a
  64-pixel crop is dominated by the court around the ball, so the ball
  barely moves it; the oracle's DISTS gain (pasted pixels) bounds what it
  can show, and LPIPS is reported beside it.
- Side by side (for the user's preference): K5b 1's layout (source,
  pixel coder, + disk, + streak, at equal rate, ball-following 3× crops
  and one full frame), at DCVC-UF QP 42 and SVT-AV1 CRF 55, BallTrack's
  positions.
- Budget: GPU (BallTrack, WASB, DISTS, LPIPS, DCVC-UF) on the A6000;
  BallTrack, WASB and DISTS are new on that class, so the smoke asserts
  CUDA for each and the result goes into the model–GPU table. Smoke: one
  window, 48 frames, ≤ 600 s. Part 1: K5b 1's four windows, ≤ 45 min;
  part 2: the other four, run only if part 1 leaves the rule open.
  Ceiling 2 GPU-hours and 6 h wall.
- Job: `20261021T084417Z-048cc9d4` (`d4bed15`; RTX A6000, gpu3, shared
  with another user's process; smoke 247 s on game3/Clip3's first 48
  frames, then part 1: game3/Clip3, game7/Clip2, game10/Clip10,
  game3/Clip6 at 240 frames, 2,380 s, inside the 45 min stage;
  `published.tar` `b75adff4…c8d3`; inputs: K4's manifest `233a08b1…`,
  K5b 2's `7cb55480…cd63`, and `inputs-k5b1b.json` `32fb61ae…40f6` (the
  DISTS VGG16 `397923af…` and LPIPS AlexNet `7be5be79…` backbones);
  environment `pointstream-20261009T214002Z`, torch 2.10.0+cu128;
  SVT-AV1, dav1d and DCVC-UF as K5b 1). Every model on `cuda:0`. Validator
  passed on the smoke and, run locally, on the full stage (15 checks;
  BallTrack's GPU path agrees with its CPU path on every source frame).
  Per window: coding 4–7 min, trackers 15–20 s, the three sources' fits
  70–134 s (in parallel), judging the rest. About 0.75 GPU-hours.
- Report (`k5b1b report`; `k5b1b-report.json` `c1f24193…bbe3`). BallTrack
  finds 0.88–0.99 of the source's balls (target 0.71–0.79). Means over
  the 4 windows, BallTrack's positions (detection rate; 1 − DISTS shown
  as DISTS):

  | Point | kbps decode → + streak | detection decode / oracle / disk / streak | DISTS decode / oracle / disk / streak |
  |---|---|---|---|
  | DCVC-UF QP 0 | 13.0 → 13.9 | 0.01 / 0.92 / 0.90 / 0.91 | 0.246 / 0.204 / 0.209 / 0.200 |
  | DCVC-UF QP 21 | 36.6 → 37.6 | 0.15 / 0.91 / 0.90 / 0.92 | 0.199 / 0.165 / 0.171 / 0.164 |
  | DCVC-UF QP 42 | 117.7 → 118.9 | 0.67 / 0.92 / 0.91 / 0.92 | 0.140 / 0.125 / 0.131 / 0.128 |
  | SVT-AV1 CRF 62 | 73.9 → 75.0 | 0.19 / 0.90 / 0.89 / 0.91 | 0.175 / 0.142 / 0.148 / 0.140 |
  | SVT-AV1 CRF 55 | 176.0 → 177.2 | 0.59 / 0.92 / 0.91 / 0.92 | 0.127 / 0.107 / 0.114 / 0.109 |

  Per window (BallTrack's positions): on DCVC-UF, the rate to find 80%
  of the source's balls is 0.05–0.15 of what the pixel envelope needs
  (detection passes 4/4, disk and streak); DISTS BD −27% to −42% for the
  streak (4/4), −22% to −33% for the disk (2/4); **frame-PSNR BD
  +1.7% to +2.8% against DCVC-UF's own envelope (0/4; bar +1%)**. On
  SVT-AV1, whose lowest point is already 57–97 kbps, the detection ratio
  is 0.27–0.72 (2/4), as for K5's pixel oracle (identical ratios: the
  ball mode reaches the target at SVT's lowest point, so the ratio is
  bounded by it); frame BD +0.2–0.3% (4/4); streak DISTS 2/4. All three
  on one window at most. The three position sources score alike (labels,
  BallTrack, WASB within 0.01 detection and 0.002 DISTS at every point).
  The ball costs 0.79 kbps (disk) and 1.11 kbps (streak) on average
  (0.33–1.47), and raises frame PSNR by about 0.01 dB.
- Decision (by the rule): **no renderer adopted**, and part 1 decides it:
  on DCVC-UF the frame guard fails on 4/4 (at most 4/8), on SVT-AV1 every
  variant passes all three on ≤ 1/4 (at most 5/8); 6/8 is out of reach,
  so part 2 is not run.
- Reading. The hypothesis holds on what the ball mode is for and fails on
  its cost. A drawn ball at the encoder's positions restores detection
  where the pixel coders lose the ball (0.01 → 0.90 at 13 kbps; DCVC-UF
  alone needs 80–260 kbps for 80% of the source's), and on crop DISTS it
  closes 75–100% of the gap to the pixel oracle (the streak is within
  0.004 of the oracle at every DCVC-UF point). What fails is the frame guard, and it
  fails on rate, not quality: the per-frame payload (about 1 kbps, most
  of it colour differences) is 7% of DCVC-UF's 13 kbps lowest point,
  while frame PSNR rises by 0.01 dB, so the frame BD-rate is +2% to +3%.
  K5b 1's per-window ball cost 0.2 kbps and passed the guard; the guard
  (≤ +1%) was set for that payload. Competing explanation 1 (BallTrack
  detects a bright blob) is not excluded by these numbers; DISTS, which
  also passes, and the clips speak against it. Explanation 2 (the judge
  agrees with itself) is excluded: labels, BallTrack and WASB positions
  score the same. Explanation 3 (DISTS on 64 px dominated by the court)
  does not hold: DISTS moves by 0.04 at QP 0, as much as the oracle's.
  Positions: BallTrack places the ball within 5 px of the label on 0.78
  of frames, WASB 0.92 (disclosed overlap); after refinement the fit's
  positions sit 1.4 (labels), 2.3 (WASB) and 3.0 px (BallTrack) from the
  labels, which are themselves 1.4–3.2 px off the best fit (K5b 1's
  diagnostic), so precision is not what limits the ball mode.
- Side by side (sent to the user 2026-10-21): game3/Clip3 and game7/Clip2
  at DCVC-UF QP 42, game3/Clip3 at SVT-AV1 CRF 55 (ball-following 3×
  crops), game10/Clip10 full frame at DCVC-UF QP 42; BallTrack's
  positions. Rendering preference (user, 2026-10-21): **the streak**.
  The user's reading of the clips: the source's own ball is low quality
  and pixelated; DCVC-UF looks good but loses the ball at times,
  especially near the far player; the disk often looks unrealistic; the
  streak looks as good as DCVC-UF where DCVC-UF keeps the ball, and keeps
  it visible far more often.
- Decision (user, 2026-10-21): the ball mode is **kept**, with the streak
  as default and BallTrack's positions, although it fails the frame
  guard by the rule; the failure is on rate (about 1 kbps), it is
  disclosed wherever the result is reported, and the rule is not
  restated after the fact.
- A cheap check of the payload's rate (not evidence: game3/Clip3, three
  Gaussian-blurred stand-in decodes, the streak fit from the labels,
  laptop CPU; `rate_check.py` in the session's scratch). Coding each
  frame's colour at a coarser step: step 8 saves 20–22% of the payload
  (1,267–1,407 → 989–1,134 bytes) and keeps 99.7–99.9% of the fit's
  squared-error gain; step 16 saves 29–30% and keeps 98.5–99.7%. One
  colour per window halves it but leaves the ball worse than none. The
  rest (about 0.7 kbps) is positions and settings. So step-8 colour is a
  free default for the final runs (a payload change, set before them);
  reaching the guard's 0.2–0.4 kbps would need smoother positions and
  settings too, which the user judged not worth pursuing now.
- Next (proposed, for the user): the ball mode's rate. Either the encoder
  spends fewer bits (a rate-aware per-frame fit: colour quantised or held
  unless the error gain pays its bits; about 0.2–0.4 kbps would pass the
  guard at 13 kbps), or the guard is restated as what it was meant to
  check (frame PSNR at the same coder point not lower, which the
  encoder's last check already ensures), which is a rule change for the
  user to make, disclosed as such.
