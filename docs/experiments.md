# Experiments

## Protocol

GPU time and training are the constraint; code is cheap. Rank work by value to
the paper first and the demo second.

1. **Write it down first.** Add an entry below with the decision rule, the
   hypothesis, the competing explanation and the budget (GPU hours and wall
   time).
2. **Oracle first.** Before building a component (a representation, a model,
   a coder), measure the cheapest upper bound of what it could achieve, at the
   operating point the decision needs. An oracle may use what the component
   cannot have, such as the target frame, ground truth or unlimited bits.
   - *It must dominate the components it gates.* Gating means those
     components never run, so nothing would reveal a wrong bound later. The
     entry says why the oracle is a bound. If that cannot be argued from its
     construction, check it on dev samples against the cheapest gated
     component, and fix the oracle before it gates anything.
   - *It answers only its configuration.* The entry names what the oracle
     bounded. When it fails, the result says what to change next, for example
     several references instead of one, and that change gets its own oracle.
   - *Results.* An oracle's result is recorded like any other. The component
     is built only if the oracle passes the decision rule.
3. **Correctness smoke, in minutes.** The real component on a representative
   bounded input, through the same entrypoint as the full run. On a model's
   first run on a GPU class, the smoke also asserts device, execution provider
   and kernels ([fleet](fleet.md#modelgpu-table)).
4. **Bounded pilot on the one axis that matters.** Start a sweep with the
   fewest settings that give a rough answer, typically three spread over the
   range. Add intermediate settings only where the answer is unclear, and
   record why.
5. **Scaled run**, only after the pilot shows it can meet the target within
   budget.
6. **Stop when the decision is made.** Record the outcome below, including
   negative results.

**Waiting time.** The agent's context stays cheap to resume for about an
hour. Every stage the agent waits on is therefore sized to end within 45
minutes. Smokes meet this already (fleet caps them at 600 s), and oracles and
pilots are scoped to it. A full run that needs longer is split into parts of
at most 45 minutes: shards of its items, one job each, with the same
specification apart from the item selection. Each part saves every finished
item ([fleet](fleet.md#checkpoints)), so a crash or a GPU taken by another user
loses at most the item in progress. The agent reads each part's results as it
ends and decides whether the next part is still needed, so a decision reached
early also stops the run early.

Fleet runs follow [fleet](fleet.md): a smoke of at most 600 s, a validator with
substantive checks, sha256-identified inputs, a deadline and a budget.

**Evidence.** A number is evidence only if it names its job, code revision,
inputs and GPU, and comes from the real component, not a stand-in. Smoke runs
and stand-ins are not evidence.

**Evaluation basis.** Masks for scoring come from the datasets, never from
PointStream's own segmenter. Weighted PSNR is 0.7 foreground + 0.3 background
on dataset masks. Results are reported per provenance tier
([resources](resources.md#datasets)). Do not count correlated frames as
independent samples.

**Baselines.** SVT-AV1 and a state-of-the-art neural video codec, with exact
encoder and decoder paths and versions recorded. VVC only if a correct
invocation turns out to be needed.

**Two tracks: fidelity and experience** (user, 2026-10-22; the thresholds
marked *proposed* were accepted by the user the same day). PointStream answers two problems,
and each object's representation is judged on both:

1. *Fidelity: encode the original.* Full-reference judges on the object's
   region (LPIPS, DISTS, PSNR), as BD-rate against the anchor envelope, or at
   the method's rate against the anchor hull. Every entry so far is on this
   track, and its verdicts stand.
2. *Experience: the same experience as the original, despite pixel
   differences.* At the method's rate, an item passes when all three hold
   against the anchor hull at that rate:
   - **Realism.** KID of the object's crops (`h3b`'s Inception crops) against
     real crops of the same class from footage that neither the method nor
     the anchor was fitted on: for hands, held-out VISOR videos, and for
     players, other matches. *Proposed:* the method's KID is at most the
     anchor's upper 95% bound.
   - **Same content.** The object's estimator, re-run on the output, agrees
     with the source: WiLoR's 2D joints for hands, the body pose for
     players, the detected position for the ball. *Proposed:* the method's
     error is at most max(the anchor's error, 1.5 × the coded keypoints' own
     error against the source).
   - **Same over time.** Flicker (`h3b.flicker`: the method's temporal
     residual minus the source's, luma, in the region). *Proposed:* at most
     max(the anchor's, 1.25 × the source's own frame-to-frame change). An
     identity drift along a track (an appearance embedding's change against
     the source's) is reported beside it and does not decide.
   Anchors on this track: the strongest perceptual pixel anchors measured
   (DCVC-UF + G; H3c's perceptually tuned DCVC-UF where it exists) with the
   SVT-AV1 envelope. Criterion as on fidelity: the share of items passing,
   ≥ 75%. A user study (two-alternative forced choice) is optional, and
   nothing relies on it.

   Results judged only on fidelity are untested on experience, not passed.
   Re-judging them on this track is allowed, and the entry says when the
   judge was added. Queue, cheapest first: H3e (its generations are saved,
   no new generation), FoundHand (K6b hands rung 2, re-run keeping the
   crops), H3a/H3a2 if their stages kept the decodes, and K6's `ref` on
   players. Its rule is the entry "J: experience-track re-judging (rule)"
   (PLAN J).

## Entry template

```markdown
### YYYY-MM-DD — <short name>
- Decision rule:
- Hypothesis:
- Competing explanation:
- Budget:
- Jobs: the 16 listed in the [model–GPU table](fleet.md#modelgpu-table)
  (environment `pointstream-20261006T113321Z`), plus pre-checks on an earlier
  build (`20261006T105407Z-800aeeb2` contended, `20261006T105506Z-13aa98b2`,
  `20261006T112304Z-ea395e8f`) that found the missing `clip` and `dill`.
- Outcome: one environment adopted. Every component passes on Ada and A6000.
  The hand models and YOLOE pass on all four classes. DCVC-UF passes on the RTX
  8000, not on the GV100 (upstream assertion). SAM 3.1 falls back to math
  attention on the RTX 8000 (39.5 GiB, 2.2× slower) and runs out of memory on
  the GV100, so SAM jobs use Ada and A6000. No component needed a second
  environment. Wall time about 1.5 h of queue and run, mostly staging and
  waiting for GPUs other users held.
```

## Decisions

### 2026-10-05: Evaluate on labelled data
- Decision: evaluation and training targets come from labelled datasets
  (OpenTTGames, RacketVision, TrackNet, VISOR, EgoHOS, HOT3D). SAM 3.1 fills
  only missing classes, prompted from the labels where possible. No hand
  labelling.
- Scope: racket sports and egocentric hand-object video. Non-racket sports are
  future work.
- Foreground includes the ball and the handled object.

### 2026-10-06: Egocentric first, racket sports second
- Decision: phase 1 is egocentric hand-object video, with VISOR carrying the
  end-to-end evaluation. Phase 2 is racket sports, once phase 1 has a result.
  In phase 2, players and the ball are evaluated in separate stages, each on the
  dataset that labels it, to estimate performance on a fully labelled dataset.
  Every dataset is used for what it labels: segmentation datasets for
  segmentation, keypoint and pose datasets for pose and foreground encoding,
  ball datasets for the ball.
- Reason: the downloaded data (manifests in `Datasets/manifests/`) shows no
  racket dataset has video, dense foreground masks and ball positions together.
  OpenTTGames labels only windows of 4 frames before and 12 after each event,
  about 9% of frames, with 320×128 model-aided masks. RacketVision has ball
  points on 11–20% of frames, racket keypoints on 4–9%, and no masks. TrackNet
  has dense ball points but no masks and no video, only JPEG frames. VISOR has
  1080p video, and its dense masks cover hands with forearms plus active
  objects: 41% of frames in a measured clip, in runs of about 7.5 s.
- Risk recorded: the panorama/mosaic background is likeliest to win with
  OpenTTGames' fixed camera and least likely with egocentric head motion. Phase
  1 background results may therefore understate that method; phase 2 tests it
  where it should work.

### 2026-10-06: VISOR frame drift: exact frames for evaluation, small drift for training
- Context: VISOR's dense masks are numbered by VISOR's own frame extraction;
  only sparse keyframes are linked to the video (`frame_mapping.json`). Dense
  frames are placed by counting from their run's keyframes, and the *drift* of
  a keyframe-to-keyframe stretch (decoded frames minus VISOR frames) bounds how
  far a placement can be off. Metadata of the 43 val videos with dense files:
  EK-100 (16 videos, 50 fps) has drift 0 on all 2,383 stretches; EK-55 (27
  videos, 59.94/29.97 fps) has 0 on 72.9%, ±1 on 22.6%, ±2 on 4.2%, ≥3 on 0.4%
  of 3,358. On P32_07 a 1-, 2- and 4-frame error mislabels a median 2.3%, 6.7%
  and 12% of the hand area.
- Decision: evaluation scores only exactly placed frames, because drift moves
  mask edges off the hand and so tends to penalise PointStream, which spends
  its bits where the hand really is. B1's set uses the EK-100 videos, exact
  everywhere. B2 may add EK-55 clips: encoded whole, scored only on frames
  proven exact (keyframes, and stretches whose keyframes match the released
  JPEGs with no drift), via `visor.clip_masks(..., exact_only=True)`. Training
  (E1, F1) admits stretches with drift ≤ 1 (about 96% of EK-55 by the val
  survey; F1 measures it on train); every exported mask records its drift so
  the cutoff can change without re-exporting.
- Not scheduled: registering each dense frame to the video (nothing to
  validate it against between keyframes) and reproducing VISOR's frame
  extraction (exact for all of EK-55 if found, verifiable on the released
  JPEGs). Revisit the second if reviewers need full EK-55 clips or the exact
  frames are too sparse for stable B2/D1 numbers.

### 2026-10-07: VISOR evaluation set v2 and a SAM fill for missing hands
- Context: B1's first set (16 EK-100 items) was exact but excluded EK-55
  wholesale, and VISOR's dense masks leave out hands a human labelled
  (22–30% of dense val frames; [B1 run](#2026-10-06--b1-visor-frame-mapping-and-evaluation-set)).
- Decision (set v2, fixed before any item is scored): one item per validation
  video of a verified 1080p video class with one VISOR frame per decoded frame
  (EK-100 50 fps and EK-55 59.94 fps); the item is a 240-frame window, from
  the non-overlapping windows of each dense run, picked by
  sha256("pointstream-b1:<video>") among those that lie exactly on the video
  and have no hand gap (`visor.hand_gaps`: a hand labelled by a human at both
  ends of a run is in every dense frame of it, and one labelled at one end is
  in the dense masks at that end). Expected from the metadata survey: about 15
  EK-100 and 20 EK-55 items. Objects other than hands may still be missing;
  that is reported, not filtered.
- Decision (later step, PLAN B1b): fill the missing hands and objects with SAM
  3.1 tracking from the human keyframe masks, as tier `sam_from_label_prompt`.
  Adopt it if it passes validation on keyframes it never saw and on a random
  sample of in-between frames, checked by eye and against EPIC-KITCHENS
  hand-object boxes (a detector's output, independent of SAM). Every result is
  reported with and without the fill.
- Outcome: set v2 has 34 items (15 EK-100, 19 EK-55 59.94 fps; `eval_set.json`
  sha256 `0d25301a…ec2`, archive `a2d9e0cb…c3c`). Excluded: the two 29.97
  fps videos and the 720p P12_04 by class; P03_14, P03_22, P07_103, P26_01,
  P29_04 and P32_07 have no exactly aligned, hand-complete window. Conversion
  job `20261007T080749Z-8c75e456` failed its smoke: the check that each item
  contains a human-labelled frame is wrong for 24-frame ranges and for one
  window that lies between two labelled frames; it now requires every human
  frame in range to be compared. Job `20261007T081136Z-71468647` (gpu1, CPU
  only, `7adefe3`, 382 s full stage, published masks `0cf83b8c…37b5`):
  validator passed. At all 78 human-labelled frames inside the items, every
  hand is in the dense masks, IoU 0.902–0.997 (median 0.984, 144 hands);
  objects median 0.966 (253), and 108 human-labelled objects are absent from
  the dense masks there (reported, not filtered). Hold the first frame, mean
  over items: foreground J 0.459, F 0.370 (EK-100 0.403, EK-55 0.504; per item
  0.03–0.85). B1's done-when holds on set v2.

### 2026-10-05: Baselines
- Decision: SVT-AV1 plus one state-of-the-art neural video codec. VVC only if a
  correct invocation is needed.
- 2026-10-06: the neural codec is DCVC-UF (CVPR 2026, microsoft/DCVC `cbdae87`),
  the newest DCVC, with a real entropy-coded bitstream. HNeRV is dropped: it has
  no temporal model, and its latents cost as much as AV1 at 240p in the pre-reset
  project ([resources](resources.md#models)).

## Runs

### 2026-10-06 — Environment audit smokes, phase 1 first wave
- Decision rule: the single `pointstream` environment is adopted for a
  component on a GPU class when that component's smoke passes every check on
  that class (claimed device of the class, model on it, CUDA kernels launched,
  the expected attention family, and an output check against the dataset's own
  labels). A component that fails only on a class records that class as
  unsupported in the model–GPU table. A component that fails on every class
  because of a dependency conflict reopens the audit with a second environment
  for it.
- Hypothesis: SAM 3.1, YOLOE-26, HaMeR, WiLoR, hand_tracking_toolkit,
  DCVC-UF and SVT-AV1 run from one Python 3.12 / torch 2.10.0+cu128 prefix on
  the 535 driver, on Ada and A6000; on Turing and Volta the SAM 3.1 and DCVC
  kernels may fall back or fail.
- Competing explanation: a passing import hides a silent fallback (CPU, math
  attention, a DCVC extension built for another GPU); the kernel and device
  records exist to catch that, not the pass/fail bit.
- Budget: four groups (SAM 3.1; YOLOE + SVT-AV1 + DCVC-UF; HaMeR + HOT3D;
  WiLoR) on four GPU classes, 16 jobs, each at most 600 s smoke + 600 s full.
  Ceiling 5.5 GPU-hours wall, expected under 2.
- Not evidence: infrastructure smokes (`citable: false`). B1 starts in its own
  session.
- Jobs: the 16 listed in the [model–GPU table](fleet.md#modelgpu-table)
  (environment `pointstream-20261006T113321Z`), plus pre-checks on an earlier
  build (`20261006T105407Z-800aeeb2` contended, `20261006T105506Z-13aa98b2`,
  `20261006T112304Z-ea395e8f`) that found the missing `clip` and `dill`.
- Outcome: one environment adopted. Every component passes on Ada and A6000.
  The hand models and YOLOE pass on all four classes. DCVC-UF passes on the RTX
  8000, not on the GV100 (upstream assertion). SAM 3.1 falls back to math
  attention on the RTX 8000 (39.5 GiB, 2.2× slower) and runs out of memory on
  the GV100, so SAM jobs use Ada and A6000. No component needed a second
  environment. Wall time about 1.5 h of queue and run, mostly staging and
  waiting for GPUs other users held.

### 2026-10-06 — B1: VISOR frame mapping and evaluation set
- Pre-checks (not evidence; this session, no fleet job): the dense VISOR
  polygons are drawn on an 854×480 canvas (the file's `info`); scaled to 1080p,
  dense keyframes match the sparse human polygons at IoU 0.97–0.995 on P32_07.
  Dense frames are named in VISOR's own numbering, which `frame_mapping.json`
  covers only on sparse frames. Anchoring dense keyframes through the mapping
  and the EPIC time rule (metadata only, all 43 val videos with dense files):
  on the 16 EK-100 videos (HEVC, 50 fps) VISOR frame n is EPIC frame n with
  zero drift between keyframes; on the 27 EK-55 videos (H.264, 59.94 and
  29.97 fps) VISOR's numbering wanders up to 117 frames from the video and
  15–60% of keyframe-to-keyframe segments drift by 1–3 frames. A local decode of
  P32_07 (59.94 fps, frames repeated in pairs) matched the EPIC time rule on
  6/6 sparse JPEGs and VISOR n − 1 on 5/6.
- Decision rule: (1) *mapping*. For each video class (codec, size, nominal
  rate), a candidate rule holds when, on every checked sparse frame of every
  video in the class, its decoded frame is within 0.5 grey levels (mean
  absolute RGB difference) of the best match to the released JPEG, and that
  best match is below 3. Rules (`experiments/visor/b1.py` `RULES`): VISOR
  n − 1; VISOR n by time at 60 per second; through `frame_mapping.json`, EPIC
  k − 1, EPIC k by time at 60 per second, at the nominal integer rate, rounded
  up at 60 per second, and the reader's rule (60 per second for EK-55, 50 for
  EK-100). (2) *evaluation set*. Items come only from validation videos whose
  class satisfies both VISOR n − 1 and the reader's EPIC rule, from runs whose
  first 240 frames lie exactly on the video (no drift
  between keyframes); one run per video, picked by
  sha256("pointstream-b1:<video>") among the eligible runs, its first 240
  frames starting on a keyframe (`tools/datasets/visor_b1_inputs.py`). If no
  class satisfies both, the set is not fixed and B1 reports keyframe anchoring
  with its measured drift as the only option. (3) *done*: the set converts
  (every frame labelled, aligned exactly, dense sources match their sha256,
  median dense-keyframe vs human hand IoU ≥ 0.9) and the trivial candidate
  "hold the first frame" scores against it.
- Hypothesis: on EK-100 (50 fps) VISOR n − 1 and EPIC k − 1 (= the time rule)
  hold on every checked frame, so the 16 EK-100 validation videos give an
  exactly aligned set. On EK-55 (59.94 and 29.97 fps) the EPIC time rule holds
  but VISOR n − 1 does not, and 29.97 fps rgb frames were extracted at 30.
- Competing explanation: a match within 0.5 grey levels may hide an off-by-one
  on videos that repeat frames (ties are recorded per frame; a tie is the same
  image, which is all the masks need); a JPEG matching several decoded frames of
  a static scene would also pass, so each window's differences are recorded. The
  50 fps zero drift could be an artefact of the EPIC rule; the JPEG check tests
  both rules independently.
- Budget: CPU only (`"device": "cpu"`, [fleet](fleet.md#cpu-jobs)), no GPU.
  Job 1 (mapping): 10 videos, 325 sparse JPEGs (P32_07, P02_02, P03_22, P09_07,
  P18_02, P17_01 for EK-55; P07_103, P09_106, P01_107, P26_108 for EK-100);
  smoke 2 frames per video ≤ 300 s, full 40 per video ≤ 1,200 s, 16 threads
  (one process per video). Budget 1.5 h, deadline 3 h after submission.
  Job 2 (evaluation set): smoke 2 items × 24 frames ≤ 300 s, full all items ×
  240 frames ≤ 1,800 s, 16 threads. Each budget ≤ 1.5 h including staging the
  4.9 GB environment and 2–4 GB of inputs. Ceiling 3 h wall, no GPU hours.
- Inputs: `pointstream-data/visor/b1-2026-10-06/mapping-v2/`
  (`visor-mapping-check.tar` sha256 `4e32ce13…e0c0`, record
  `visor-mapping-check.json` `6ecee54e…c289`, videos hard-linked under
  `inputs/video_<id>/`); environment `pointstream-20261006T113321Z`.
- Jobs: `20261006T191914Z-10673c3a` (gpu3, `8cb7dfb`) failed its smoke: P09_07
  (29.97 fps, 1,655 decoded frames) has sparse frames up to VISOR and EPIC
  3025, so its frames were extracted at 60 per second, not at its own rate.
  Every rule then pointed past the end of the video and the check crashed
  instead of recording a miss. Locally on P09_07 the 60-per-second EPIC rule
  matched 28/32 JPEGs with ordinary rounding and 32/32 rounded up; the rules
  above add the rounded-up variant, and a miss is now recorded. The metadata
  drift survey assumed 30 per second for the two 29.97 fps val videos (52 of
  3,358 EK-55 stretches); drift is now measured against the extraction rate.
- Job `20261006T193326Z-64057023` (gpu3, CPU only, `efc516a`, 88 s full
  stage): validator passed. EK-100 (HEVC 50 fps, 4 videos): VISOR n − 1 and
  EPIC k − 1 hold on 139/139 JPEGs. 59.94 fps (3): EPIC nearest to
  (k − 1)/60 s holds on 86/86, VISOR n − 1 on 18/86 (off by up to 13). 29.97 fps
  (2): EPIC (k − 1)/60 s rounded up holds on 72/72, VISOR n − 1 on 0/72. 47.95
  fps (P17_01, train): no rule on every frame (26/28 nearest, 23/28 rounded
  up); the reader raises for that rate. Rule table in
  [resources](resources.md#datasets).
- Evaluation set: 16 items, one per EK-100 validation video, 240 frames, from
  `tools/datasets/visor_b1_inputs.py evalset` on that job's `mapping.json`
  (`eval_set.json` sha256 `b994c531…d9cb`, archive `cca6169a…020f`). The 27
  EK-55 validation videos are excluded.
- Job `20261006T194624Z-8b0d5b7e` (gpu3, CPU only, `2985306`, 160 s full
  stage, published masks `afd7370a…5799`): validator passed (every frame
  labelled and exactly aligned, sources match their sha256, dense keyframe vs
  human masks median IoU 0.973 for hands over 25 instances, 0.979 for objects
  over 35). Trivial candidate "hold the first frame", mean over items:
  foreground J 0.337, F 0.264; left hand J 0.349; right hand J 0.285; active
  object J 0.326. B1's done-when holds.
- Finding: the dense masks are incomplete. VISOR's README says the dense
  interpolations "are filtered and only high J&F scored interpolations are
  provided": each object's track between two keyframes is kept, cut short or
  dropped on its own score, and an object labelled at only one end is never
  interpolated. On the first frames of the 16 items, 5 lose a hand (P01_107,
  P07_110, P09_104, P30_110 one each; P09_103 both); where both have the
  hand, IoU is 0.84–0.99, so this is omission, not misalignment. (An earlier
  version of this entry gave only the one-end cause; P09_103 and P01_107 lose
  hands labelled at both ends.) Metadata survey of all val videos with dense
  files (this session, not a fleet job): 22% of EK-100 dense frames (37,708 of
  170,683) and 30% of EK-55 59.94 fps frames (105,755 of 354,513) miss a hand
  that a human labelled at an end of their run; hands dropped despite labels at
  both ends: 277 (EK-100), 711 (EK-55); labelled at one end: 320, 514. Any
  labelled object is missing on 87% and 90% of frames. A hand that comes and
  goes between keyframes is not counted. All 1,786 HInt frames in these videos
  are sparse keyframes, so HInt has no keypoints on the frames where hands go
  missing. For B2 a missing hand is scored as background; for D1 a segmenter
  that finds it is penalised; F1 background frames and E1 would learn
  unlabelled hands as background. Drift-free 240-frame windows: EK-100 16/16
  videos (557 windows), EK-55 59.94 fps 1080p 21/24 (444); hand-complete among
  them: 15 (291) and 20 (205); all objects complete: 4 (13) and 3 (13).
  Decision pending with the user (options in the VISOR frame drift report).

### 2026-10-07 — B2: baseline rate-distortion on VISOR
- Question: what rate and quality do SVT-AV1 and DCVC-UF reach on evaluation
  set v2 (34 windows of 240 frames, 15 EK-100, 19 EK-55), scored on the VISOR
  masks? These curves are the target PointStream must beat.
- Method (`experiments/visor/b2.py`, `src/codecs/svtav1.py`,
  `src/codecs/quality.py`). Each window is decoded from the source video at
  `first_video_index` and kept as the decoder's own 8-bit 4:2:0 planes (the
  sources are full-range `yuvj420p`; EK-100 tags BT.709, EK-55 is untagged).
  Both codecs take those planes: SVT-AV1 4.2.0 (`SvtAv1EncApp`, preset 4,
  one-pass CRF, random access, one keyframe: `--keyint -1`), decoded by dav1d;
  DCVC-UF through its YUV420 path (the input `test_video.py` uses for UVG, one
  intra frame), decoded from the container bytes in its own process on the
  same GPU. Rate is the AV1 payload of the IVF stream or the whole DCVC
  container, in kbps at the source rate. Every metric sees one RGB view of
  source and decoded frames (bilinear chroma, BT.709, the source's range,
  rounded to 8 bits): PSNR over the frame and inside the B1 dense masks'
  foreground (union of hands and active objects) and background, from exact
  integer error sums; weighted PSNR = 0.7 PSNR_fg + 0.3 PSNR_bg per frame (dB),
  then the mean over frames (pooled-error variant recorded too); LPIPS (AlexNet,
  torchmetrics 1.9.0 heads, backbone `Models/LPIPS/alexnet-owt-7be5be79.pth`)
  as a full-resolution map, so it splits by region the same way; MS-SSIM (RGB);
  VMAF `vmaf_v0.6.1` through the hosts' `/opt/local/bin/ffmpeg` (libvmaf 3;
  the environment's ffmpeg has no libvmaf); Y/U/V PSNR. Items are the samples:
  curves are means over items per rate point, per video type and overall.
  With every score, the share of each frame's labelled objects missing from
  the masks: objects a human labelled at both keyframes of the frame's run (or
  at the frame itself, if it is a keyframe) that the mask set lacks. Results
  are keyed by mask set (`visor_dense` now); streams are published, so B1b's
  fill is added later by scoring the same streams with a second mask set
  (`run --streams`), without re-encoding.
- Correctness gate (smoke and full): every released sparse JPEG of each item
  (B1 archive) matches the decoded frame the reader's rule names (best within
  ±2 frames, within 0.5 of the best, below 3 grey levels, as B1); each item's
  masks equal B1's published masks (`masks.rle` sha256 and decoded-mask hash
  from job `20261007T081136Z-71468647`); staged videos match the set's
  sha256; every frame decoded; no labelled hand missing; rate and quality rise
  together across points; for DCVC-UF, deterministic decode, decoder intra
  frame equal to the encoder's, encode and decode on one GPU, extension variant
  for the class, CUDA kernels launched, and CPU and GPU metrics agreeing (PSNR
  < 1e-3 dB, LPIPS < 1e-3, MS-SSIM < 1e-4), because SVT-AV1 is scored on CPU.
- Pilot (rate points, and DCVC-UF structure): 4 items, the two smallest
  videos of each type (staging cost; content-blind): `P01_107_0000003049`,
  `P09_106_0000006303` (EK-100), `P02_02_0000006080`, `P03_10_0000000616`
  (EK-55); 240 frames. SVT-AV1 CRF 13, 20, 27, 34, 41, 48, 55, 62; DCVC-UF
  QP 0, 9, 18, 27, 36, 45, 54, 63, for HT-S and HT-L.
- Decision rule (rate points, fixed before the pilot): on the pilot means of
  weighted PSNR (`visor_dense`) against rate, take the common quality range of
  SVT-AV1 and the chosen DCVC-UF structure, capped above by each codec's
  quality at its highest point below 3.2 Mbps (a third of the lowest pilot
  source bitrate, 9.7 Mbps: re-encoding near the source's own rate measures
  its artefacts, not the codec). Place 6 targets evenly across that range; per
  codec take the swept point whose pilot mean is nearest to each target,
  without duplicates, plus its next lower-rate swept point below the range
  (PointStream is expected to work at low rate). If the range is under 3 dB or
  a codec gets fewer than 4 points, refine the sweep on the pilot items first.
  DCVC-UF structure: HT-L if its pilot BD-rate against HT-S (weighted PSNR,
  mean over the 4 items) is below −3%, else HT-S (faster, and the structure
  the environment audit verified). LD is not run: SVT-AV1 runs random access.
- Decision rule (B2): done when both codecs' curves for all 34 items come from
  recorded jobs whose validators passed. A failed JPEG check stops the run: the
  item's frame placement, not the codec, is then in question.
- Hypothesis: DCVC-UF needs less rate than SVT-AV1 preset 4 for equal
  weighted PSNR on both video types (negative BD-rate), as it does against
  VTM on UVG; the two codecs differ less on the foreground than on the
  background, because hands move fast and blur.
- Competing explanation: egocentric kitchen video (head motion, motion blur,
  re-encoded H.264/HEVC sources with their own artefacts, full-range YUV) is
  far from DCVC-UF's training data, so SVT-AV1 may match or beat it; and any
  gap at high rate may reflect how each codec reproduces source artefacts. The
  per-type split and the rate cap separate these.
- Budget: pilot, SVT-AV1 one CPU job (32 threads; smoke 2 items × 24 frames ×
  2 points ≤ 600 s; full ≤ 5,400 s) and DCVC-UF two GPU jobs on Ada or A6000
  (8 threads; smoke ≤ 600 s, full ≤ 3,600 s each), together ≤ 2.5 GPU-hours.
  Full run, SVT-AV1 as two CPU jobs of 17 items (48 threads, ≤ 3 h each);
  DCVC-UF as three GPU jobs of 11–12 items on one GPU class (≤ 2.5 h each).
  Ceiling 10 GPU-hours and 12 h wall, including staging (73 GB of videos in
  all; each job stages only its items' videos).
- Pilot jobs (environment `pointstream-20261006T113321Z`; all validators
  passed, every sparse JPEG of the 4 items matched): SVT-AV1
  `20261007T091445Z-cb576ca3` (gpu6, CPU only, `49b72af`, 1,581 s full stage);
  DCVC-UF HT-S `20261007T094441Z-ba27412b` (gpu6, RTX 6000 Ada, `361067a`,
  991 s); HT-L `20261007T094711Z-0e696539` (gpu3, RTX A6000, `361067a`,
  1,709 s). Failed or stopped first attempts, none evidence: `…091007Z-91bbfae2`
  (the hosts' ffmpeg could not load libvmaf without its library path),
  `…091602Z-72ae5f5b`, `…091717Z-9b051cd0`, `…093027Z-71cc5ec2` (PyAV
  stalled decoding in a process that had loaded torchvision; windows are now
  decoded in a fresh process), `…093726Z-4e7676dc` (CPU and GPU PSNR differed
  by 5.6e-5 dB, from rounding in the float RGB view, against a check of
  exact equality; now < 1e-3 dB), and the cancelled `…091012Z-fb21885b`,
  `…091018Z-7bae2104`, `…093958Z-a67af345`.
- Pilot outcome (means over the 4 items, `visor_dense`). SVT-AV1 CRF 62 to 13:
  378 kbps 34.3 dB to 29.5 Mbps 44.5 dB weighted PSNR. DCVC-UF HT-S QP 0 to
  63: 78 kbps 27.2 dB to 3.2 Mbps 38.6 dB; HT-L: 60 kbps 28.5 dB to 3.0 Mbps
  39.1 dB. HT-L against HT-S: BD-rate −42.9% (per item −33 to −51%), so
  HT-L. GPU class does not explain it: the audit coded one clip with HT-S on
  Ada and A6000 at 0.00634 and 0.00633 bpp, PSNR within 0.08 dB. SVT-AV1
  beats HT-S at every rate (e.g. 1,403 kbps 37.7 dB against 1,823 kbps
  37.4 dB), also in luma PSNR on the raw planes on every item, so it is not an
  artefact of the RGB view or the masks; HT-L is close to SVT-AV1 (953 kbps
  37.0 dB against SVT-AV1 894 kbps 36.7 dB). DCVC-UF reaches at most 3.0–3.2
  Mbps at QP 63, so the common range ends at 38.6 dB. The rule gives the range
  34.3–38.6 dB (4.3 dB), SVT-AV1 CRF 41, 48, 55, 62 and DCVC-UF HT-L QP 27,
  36, 45, 54, 63; no refinement (`b2 choose`). SVT-AV1 has no point below the
  range, which its own lowest point sets. Missing labelled objects: 6% of
  each frame's on the EK-100 pilot items, 32% on EK-55. Caveat: DCVC-UF codes
  the sources' full-range YUV, while its training video is most likely
  limited range.
- Full run: SVT-AV1 two CPU jobs of 17 items; DCVC-UF HT-L three jobs of
  11–12 items, all on RTX A6000 (the pilot's class, so the 4 pilot items must
  reproduce bit for bit).
- Full jobs (`0ddfe0c`, environment `pointstream-20261006T113321Z`, every
  validator passed: 17 checks per SVT-AV1 job, 22 per DCVC-UF job): SVT-AV1
  `20261007T102311Z-ac4aa348` (gpu6, CPU only, 2,521 s) and
  `20261007T103838Z-8c2353e6` (gpu1, CPU only, 3,930 s); DCVC-UF HT-L
  `20261007T102323Z-389aed6e`, `20261007T102425Z-d8de2e33`,
  `20261007T102541Z-ad949516` (gpu3, RTX A6000, 3,394, 3,072 and 3,118 s).
  `20261007T102316Z-eb661959` was cancelled during staging: its 48-thread CPU
  claim on gpu3 kept the only free A6000 from the DCVC-UF jobs. All 34 items
  for both codecs; all 78 released sparse JPEGs match their decoded frames;
  every pilot item's streams reproduced bit for bit (SVT-AV1 also across
  hosts). GPU use about 4 GPU-hours with the pilots and failed smokes, wall time 5.7 h.
- Outcome (`b2 report`, means over the 34 items, `visor_dense`; report and
  figure in `pointstream-data/visor/b2-2026-10-07/report-c52b117/`,
  `b2-report.json` sha256 `e1a4ecd7…dedb3`, `b2-rd.png` `bc95aea3…66e9`):

  | Codec, point | kbps | wPSNR (dB) | fg / bg PSNR | VMAF | LPIPS |
  |---|---:|---:|---|---:|---:|
  | SVT-AV1 CRF 62 | 434 | 34.58 | 34.42 / 34.96 | 73.3 | 0.165 |
  | SVT-AV1 CRF 55 | 1,022 | 36.87 | 36.85 / 36.94 | 83.2 | 0.126 |
  | SVT-AV1 CRF 48 | 1,576 | 37.84 | 37.88 / 37.73 | 86.6 | 0.111 |
  | SVT-AV1 CRF 41 | 2,498 | 38.67 | 38.78 / 38.41 | 89.3 | 0.101 |
  | DCVC-UF QP 27 | 329 | 34.47 | 34.58 / 34.22 | 68.1 | 0.190 |
  | DCVC-UF QP 36 | 584 | 35.97 | 36.09 / 35.67 | 76.4 | 0.154 |
  | DCVC-UF QP 45 | 1,032 | 37.21 | 37.35 / 36.87 | 82.4 | 0.128 |
  | DCVC-UF QP 54 | 1,846 | 38.26 | 38.42 / 37.88 | 86.6 | 0.109 |
  | DCVC-UF QP 63 | 3,283 | 39.14 | 39.31 / 38.74 | 89.3 | 0.095 |

  BD-rate of DCVC-UF HT-L against SVT-AV1 (per item, cubic, over each item's
  common range; all 34 overlap), mean / median: weighted PSNR −13.3% / −18.3%
  (EK-100 −18.3 / −29.9, EK-55 −9.4 / −11.3; per item −55.5 to +76.5, 7 of
  34 positive); whole-frame PSNR −0.5% / −3.6% (EK-100 −9.7 / −20.4, EK-55
  +6.7 / +7.4); VMAF +7.5% / +5.0% (EK-100 −1.4 / −11.4, EK-55 +14.6 /
  +13.0). The labelled objects missing from the dense masks: 33% of each
  frame's, mean over items (EK-100 31%, EK-55 35%); no labelled hand missing.
- Reading: the hypothesis holds for the weighted score: DCVC-UF HT-L needs
  less rate than SVT-AV1 preset 4 for equal weighted PSNR, on both video
  types. Its second part is refuted: the gain is in the foreground. On the
  whole frame the two tie, and VMAF prefers SVT-AV1, by 13–15% on EK-55.
  DCVC-UF HT-S was worse than SVT-AV1 in the pilot, so the structure matters
  more than the codec family here. Per-item spread is wide, so curves are
  reported per item and per type, not as one number. The target PointStream
  must beat at equal rate is the better of the two per metric: DCVC-UF HT-L
  on weighted PSNR and foreground PSNR, SVT-AV1 on VMAF on EK-55. Limits:
  SVT-AV1's lowest point is 434 kbps (CRF 62) and DCVC-UF's highest is
  3.3 Mbps (QP 63); one preset and one structure each; the dense masks lack a
  third of the labelled objects, which B1b's fill is meant to supply (add it
  with `run --streams` on the published streams). B2's done-when holds.
- Timing (every job records its stage seconds in `execution.json`; every item
  and rate point in `b2.json` records encode, decode, scoring and VMAF
  seconds; no stage was contended or paused). SVT-AV1 used no GPU: CPU jobs of
  32 (pilot) and 48 threads, full stages 1,581 + 2,521 + 3,930 s. All GPU time
  was DCVC-UF: full stages 991 (HT-S) + 1,709 (HT-L pilot) + 3,394 + 3,072 +
  3,118 s = 3.4 GPU-hours, plus smokes and staging. Per 240-frame window,
  median: SVT-AV1 preset 4 encodes in 14.6 s (16 fps; 8 threads, up to 6
  windows at once on a shared host), dav1d decodes in 0.9 s; DCVC-UF HT-L on
  an RTX A6000 encodes in 12.5 s (19 fps) and decodes in 5.2 s (46 fps), each
  after a 5.8 s model load, and its times include the worker's CPU-side 4:2:0
  conversion and hashing, so they are not DCVC-UF's kernel speed. Scoring
  dominated SVT-AV1's jobs: 227 s per point on CPU (LPIPS and MS-SSIM at
  1080p) against 6.5 s on the GPU for DCVC-UF. These are pipeline timings,
  not codec benchmarks: hosts were shared and SVT-AV1 ran windows in
  parallel. Superseded as speed evidence by B2b: SVT-AV1's encodes were not
  confined to 8 threads (`--lp`, below), and DCVC-UF's include the worker's
  conversion and hashing. Use B2b J2 (SVT-AV1 per preset, 32 pinned cores,
  one window at a time) and J1 (DCVC-UF codec calls) instead. B2's rates and
  scores are unaffected.
- Foreground analysis (from the recorded per-item results, no new job): at a
  mid rate (SVT-AV1 CRF 48, DCVC-UF QP 45) foreground PSNR exceeds background
  PSNR by 0.48 dB for DCVC-UF and 0.16 dB for SVT-AV1, and DCVC-UF's gap is
  larger on 28 of 34 items; its BD-rate is better on weighted than on
  whole-frame PSNR on 30 of 34 items. Its weighted advantage is larger where
  the foreground is a smaller share of the frame (Spearman 0.42 between
  BD-rate and foreground fraction) and where the background is easy for
  SVT-AV1 (−0.42 with SVT-AV1's background PSNR). Neither codec is told
  where the foreground is: both encode the whole frame, and masks enter only
  the scoring.
- Correction (found in B2b): B2 passed `--lp 8` meaning 8 threads, but
  SVT-AV1 4.2.0's `--lp` is a parallelism level (0–6); 8 was clamped to 6, so
  every encode could use the whole host. Rate and quality are unaffected (the
  output is deterministic: pilot items reproduced bit for bit across hosts);
  the SVT-AV1 timings above are not "8 threads".

### 2026-10-07 — B2b: fair baselines (timing, equal time, ROI, range, complexity)
- Question: are B2's baselines as strong and as fairly configured as they
  should be before PointStream is compared with them? Code: `cdab5fc` and
  later (`experiments/visor/b2.py` variants, `src/codecs/svtav1.py` CPU
  confinement, DCVC-UF codec-call timing). Every SVT-AV1 encode now runs on
  its worker's own cores (CPU affinity), with wall and CPU seconds; every
  codec is scored on the GPU (B2's CPU scores stay valid). The low-rate
  resolution ladder (PLAN B2b item 3) waits for PointStream's operating
  rates; its points will be chosen by PointStream's rate range alone, never by
  how PointStream scores against them.
- J1, DCVC-UF codec timing and range check. HT-L on the 4 pilot items, 240
  frames, QP 27, 36, 45, 54, 63, each with full-range input (B2's
  configuration, also a reproduction check against B2's streams) and with
  limited-range input (source mapped to 16–235/16–240, output mapped back,
  scored against the same full-range reference); RTX A6000 on gpu3, 8 threads.
  Timing: wall seconds inside the codec calls (network and entropy coding,
  GPU-synchronized), apart from reading, 4:2:0 conversion, hashing and the
  second, verifying decode. Decision rule (range): if limited-range input
  changes DCVC-UF's BD-rate against its full-range self (weighted PSNR, mean
  over the 4 items) by more than 3%, B2's DCVC-UF curves are re-run with the
  better input and the paper reports both; otherwise full range stands.
  Hypothesis: limited range costs DCVC-UF little (within 3%). Competing
  explanation: its training on limited-range YUV makes full range out of
  distribution, so limited range is clearly better.
- J2, SVT-AV1 time per preset. Presets 1, 2, 3, 4, 5, 6, 8, 10, 12 on two
  pilot items (P01_107 EK-100, P02_02 EK-55), CRF 41, 48, 55, 62, one window
  at a time on 32 cores of gpu3 (half its 64; DCVC-UF had one GPU and 8
  cores there), after J1 finishes so the host is not shared with it; scored on
  its A6000. Decision rule (equal time): SVT-AV1's B2 preset becomes the
  slowest preset whose median encode wall time per window is at most
  DCVC-UF HT-L's median codec encode time per window from J1 (same host);
  if that is not preset 4, the B2 SVT-AV1 curves and the ROI run use it, and
  preset 4 stays as a reference. Hypothesis: DCVC-UF's codec time is far
  below B2's 12.5 s, so the fair SVT-AV1 preset is faster than 4 (and worse).
  Competing explanation: DCVC-UF's entropy coding on the CPU dominates and its
  time is near B2's, so preset 4 or slower fits.
- J3, ROI SVT-AV1 pilot. `--roi-map-file` with quantizer offsets 0 (control),
  −16, −32, −64 inside every 64×64 block that touches the VISOR foreground,
  per frame; preset 4, CRF 41, 48, 55, 62, the 4 pilot items, scored on the
  GPU. Decision rule: the offset with the most negative mean BD-rate on
  weighted PSNR against offset 0 is adopted for the full ROI run (all 34
  items, at the equal-time preset); if none is below −3%, ROI is reported as
  not helping. Whole-frame PSNR and VMAF BD-rates are reported beside it,
  since ROI trades background for foreground. Hypothesis: a moderate offset
  (−16 or −32) gains more than 5% on weighted PSNR. Competing explanation:
  64×64 blocks cover so much more than the hands (median foreground 20% of
  the frame) that the shift barely changes the weighting.
- J3 outcome (`20261007T174905Z-c6e9381d`, gpu6, RTX 6000 Ada for scoring,
  `aac0cb4`, 373 s full stage; validator passed): lowering the quantizer in
  foreground blocks does not work. BD-rate against no ROI, mean over the 4
  items, weighted PSNR: −16 → +1.7%, −32 → +5.4%, −64 → +15.1% (whole-frame
  PSNR +3.2, +6.0, +11.7%); at CRF 48, −64 adds 6% rate while foreground PSNR
  falls (37.59 → 37.25 dB). A CPU probe on 16 frames of P01_107 (gpu6, 4
  cores, not a fleet job, not evidence) shows why: the map is honoured
  spatially, since a +64 offset on the left half costs it 1.15 dB and saves 7%,
  but a −64 offset adds 8% bytes and leaves the region's PSNR unchanged
  (41.29 → 41.30 dB), with or without adaptive quantization (`--aq-mode 0`).
  In SVT-AV1 4.2.0's CRF mode, ROI can make a region cheaper, not better. The
  validator's ROI check (any positive mean margin) was too weak to catch this.
- J3b, ROI by raising the background: offsets +16, +32, +64 on every 64×64
  block with no foreground pixel, against 0; CRF 34, 41, 48, 55, 62 (lower
  CRFs keep the curves overlapping as the background loses quality); preset 4,
  the 4 pilot items, GPU scoring. Decision rule as for J3: the offset with the
  most negative mean BD-rate on weighted PSNR against 0, adopted if below −3%,
  for the full ROI run; otherwise ROI is reported as not helping SVT-AV1 here.
  Hypothesis: +16 or +32 gains on weighted PSNR (the foreground keeps its
  quantizer while the background pays). Competing explanation: blocks are so
  coarse that "background" blocks are a minority of what the weights favour,
  so the shift barely helps.
- J3b outcome (`20261007T185948Z-f7d9ed2b`, gpu6, RTX 6000 Ada for scoring,
  `6698a9b`, 466 s full stage; validator passed; a first attempt,
  `20261007T180417Z-38209348` on gpu1, ended `contended` near its end, since
  another user's process kept taking the RTX 8000, and published only
  `partial.tar`): raising the background works as designed: foreground PSNR
  holds (e.g. 37.58–37.60 dB at CRF 48 for every offset) while the background
  and the rate fall. BD-rate against no ROI, mean / median over the 4 items,
  weighted PSNR: +16 → −1.6% / −0.5%, +32 → −3.04% / −1.3%, +64 → −0.8% /
  +2.2%; whole-frame PSNR +3.0, +8.6, +31.2%; VMAF +2.6, +6.0, +19.4%. By the
  rule, +32 is adopted for the full ROI run, but only just: one item
  (P09_106, −10.4%) carries the mean, the others are −1.5 to +0.8%. The full
  run on 34 items decides whether the gain is real. `--enable-tf 0`, preset
  4, CRF 41–62, the 4 pilot items, against B2's pilot streams. Reported, no
  decision attached.
- J5, complexity (secondary, CPU): per frame, VCA v2 (`/opt/local/bin/vca`,
  E and h), ffmpeg `siti`, and Sobel gradient and frame-difference means
  inside and outside the foreground, for all 34 items; correlated (Spearman,
  over items) with B2's per-item BD-rates in `b2 report --complexity`. Run as
  two jobs on gpu6 and gpu1, where B2's SVT-AV1 jobs left those items' videos
  cached. Reported, no decision attached.
- J1 outcome (`20261007T174610Z-05be667d`, gpu3, RTX A6000, `aac0cb4`,
  2,402 s full stage; validator passed). DCVC-UF HT-L codec time per
  240-frame window, median over 20 codings: encode 1.90 s (126 fps), decode
  2.03 s (118 fps), the same at every QP; the worker's whole encode took
  13.1 s, so B2's DCVC-UF timings were about 85% pipeline (conversion,
  reading, hashing). All 20 full-range streams equal B2's pilot streams bit
  for bit. Limited-range input is worse: BD-rate against full range, mean over
  the 4 items, weighted PSNR +5.9%, whole-frame PSNR +5.9%, VMAF +1.5%, worse
  on every item; part of it is the range mapping's own rounding. Full range,
  B2's configuration, stands; no re-run (the rule's change exceeds 3% but in
  favour of what B2 used). The hypothesis is refuted, and not in the
  competing explanation's direction.
- J2 outcome (`20261007T174747Z-8a4e6d3f`, gpu3, 32 cores, one window at a
  time after J1 had finished, RTX A6000 for scoring, `aac0cb4`, 2,477 s full
  stage; validator passed; host load average 7–9 before each encode). Median
  SVT-AV1 encode time per 240-frame window (wall, CPU) and BD-rate against
  preset 4 on weighted PSNR, mean over the 2 items: preset 1 80.5 s (750 s
  CPU), −15.6%; 2 46.2 s, −12.0%; 3 30.3 s, −6.2%; 4 18.7 s (149 s CPU);
  5 12.8 s, +3.2%; 6 9.1 s, +12.3%; 8 4.6 s, +49.7%; 10 2.7 s, +110%;
  12 1.94 s (14.6 s CPU), +133%. DCVC-UF HT-L's codec encode on the same host
  takes 1.90 s, so no tested preset fits the equal-time rule; preset 12 is
  2% over. The rule did not say what happens when none fits, and the choice
  changes the baseline's strength by more than a factor of two in rate, so it
  is put to the user before the full runs (decision below).
- Decision (user, 2026-10-07): equal time stays the rule, but the time to
  match is PointStream's, not DCVC-UF's: PointStream will likely be slower
  than DCVC-UF alone. So the equal-time SVT-AV1 preset is chosen once
  PointStream's encode time per window is measured on the same host class,
  by J2's table (slowest preset at or under that time), and SVT-AV1 is
  re-encoded then. The full ROI run (+32, J3b) waits for the same preset, and
  the low-rate ladder for PointStream's rates, so all three run together.
  Until then B2's preset-4 curves are the reference. gpu5, RTX 6000 Ada for scoring,
  `aac0cb4`, 149 s full stage; validator passed): without temporal filtering
  SVT-AV1 is clearly worse. BD-rate against B2's pilot (same items and CRFs),
  mean over the 4 items: weighted PSNR +21.5%, whole-frame PSNR +27.6%, VMAF
  +27.1%, worse on every item. It costs the background slightly more than the
  foreground (−0.28 against −0.13 dB at CRF 48), so filtering explains at most
  a small part of DCVC-UF's foreground advantage.
- J5 outcome (`20261007T175147Z-16c6f429` gpu6, 104 s;
  `20261007T175326Z-67c86dcf` gpu1, 150 s; CPU only, `aac0cb4`; validators
  passed). Spearman correlation over the 34 items between B2's BD-rate of
  DCVC-UF against SVT-AV1 and each complexity measure (positive: DCVC-UF does
  relatively worse on more complex content): weighted PSNR, background Sobel
  gradient 0.56, VCA E 0.47, foreground gradient 0.42, background frame
  difference 0.32, foreground frame difference 0.28, VCA h 0.23; whole-frame
  PSNR, background gradient 0.63, VCA E 0.52, background difference 0.46.
  With 34 items, |ρ| above about 0.34 is significant at 5%. The DCVC-UF
  advantage is not explained by how much the foreground moves; it is largest
  on scenes with little spatial detail, especially in the background.
- Budget: J1 ≤ 1.5 h on one A6000; J2 ≤ 2.5 h (32 threads, GPU for scoring
  only); J3 ≤ 1 h and J4 ≤ 0.5 h on any Ada, A6000 or RTX 8000 (scoring
  only); J5 ≤ 1 h CPU. Ceiling 6 GPU-hours and 8 h wall; smokes ≤ 600 s.

### 2026-10-07 — B1b: SAM 3.1 fill of VISOR's missing hands and objects
- Question: can SAM 3.1, prompted with the human masks at a run's keyframes,
  supply the hands and objects VISOR's dense masks lack, accurately enough to
  score B2 and D1 with a second mask set? On evaluation set v2 the dense masks
  lack 33% of each frame's labelled objects (B2) and no labelled hand (by
  construction); elsewhere 22–30% of dense val frames lack a labelled hand.
- Method (`experiments/visor/b1b.py`, `src/segmentation/sam31_tracker.py`).
  SAM 3.1's public multiplex API takes no mask prompts at the pinned commit,
  so its own tracker is used directly (sam3's `build_sam3_multiplex_video_model`,
  the checkpoint's `tracker.model.*` and `detector.backbone.vision_backbone.*`
  weights; every key must load). Per item, the *span* runs from the last
  human-labelled frame at or before the window to the first at or after it
  (251–516 frames, 2–11 keyframes; 98 keyframe pairs over the 34 items; a CPU
  survey, not evidence), decoded as consecutive video frames from keyframes
  placed by B1's verified rules; the sparse JPEGs inside the span are checked
  as in B1. Every object a human labelled at a keyframe of the span is a SAM
  object (at most 14 per item).
  *Held out*: for each consecutive keyframe pair a < b, SAM is prompted with
  the human masks at a only, tracks forward, and is scored at b against the
  human masks it never saw (J, boundary F at 0.8% of the diagonal), beside the
  floor "hold the mask of a"; objects labelled at a but not at b score whether
  SAM lets them go (`released`: under 64 pixels). Objects the dense masks drop
  between a and b form the *hard subset*: the ones the fill is for.
  *Fill*: per gap, SAM is also prompted with the human masks at b and tracks
  backward to the gap's midpoint; each frame takes the prediction from its
  nearer keyframe (the held-out forward run for the first half). On each
  window frame, an object the dense masks lack (`visor.object_key`) and SAM
  finds (at least 64 pixels) is added as tier `sam_from_label_prompt`; dense
  masks are unchanged. Output: `masks.rle` per item in B1's format. (Changed
  after the first two smokes, before any evidence: the plan was one session
  prompted at every keyframe, but multiplex SAM 3.1 takes mask prompts only on
  the first frame of a fresh state, `20261007T204154Z-9531c0f7`.)
  *Hand boxes*: EPIC-KITCHENS-100 hand-object detections (Shan et al. 2020;
  `Datasets/manifests/EPIC-KITCHENS-hand-objects.json`, detector output, not
  labels). A hand mask agrees when a detected hand of its side (score ≥ 0.5)
  lies at least half inside the mask's bounding box (VISOR hands include the
  forearm, the detector's do not). Rates are read against the human hands at
  keyframes and the dense hands on the same frames.
  *Review*: two window frames per item that are not keyframes, picked by
  sha256("pointstream-b1b-review:<item>:<n>"), drawn with the dense masks, the
  fill and SAM's own masks, published as an artifact for review by eye.
- Decision rule (fixed before any run; `b1b.DECISION`, applied by
  `b1b report` over all 34 items). Hands are adopted when, over all held-out
  pairs: mean J ≥ 0.70, median J ≥ 0.80, mean J at least 0.15 above the hold
  floor; on the hard subset (if n ≥ 10) mean J ≥ 0.60; hands labelled at a and
  not at b released in ≥ 70% of cases (if n ≥ 5); and SAM's held-out hands
  agree with the detector at ≥ 0.9 times the rate of the dense hands on the
  same frames. Objects are adopted when mean J ≥ 0.60, median J ≥ 0.65, mean J
  at least 0.10 above the floor, and on the hard subset (if n ≥ 10) mean J ≥
  0.50. If only one group passes, the fill keeps only that group (SAM
  instances of the other class dropped from the published masks on CPU). The
  review by eye can veto adoption for a systematic failure; it is recorded
  here. Only if a group is adopted are B2's published streams rescored with
  both mask sets (`b2 run --streams`, GPU scoring), and every B2 number is
  reported with and without the fill, with the share of labelled objects still
  missing.
- Hypothesis: across keyframe gaps of 1.3–1.7 s, SAM 3.1 from human masks
  reaches hand J ≥ 0.8, far above the hold floor; objects score lower (small,
  occluded, handled) but pass; the fill cuts the missing-object share from
  33% to under 10%, and moves B2's weighted PSNR by under 1 dB and its BD-rates
  by a few percent.
- Competing explanations: (1) VISOR dropped exactly the tracks its
  interpolation found hard (fast motion, occlusion, small objects), so SAM's
  accuracy on all objects overstates it on the filled ones; the hard subset
  tests this. (2) One-sided tracking over a whole keyframe gap is harder than
  the fill (each frame at most half a gap from its prompt), so held-out scores
  understate the fill; the rule accepts that bias as conservative. (3)
  Detector agreement may track the detector's own failures on blurred frames;
  the dense hands on the same frames calibrate it. Pairs within an item are
  correlated; results are also given per video type and per item.
- Budget: GPUs Ada and A6000 only (model–GPU table). Smoke: one item
  (P30_110, span 251, one pair), ≤ 600 s. Pilot: the four B2 pilot items
  (P01_107, P09_106, P02_02, P03_10; 1,272 span frames, 11 pairs), ≤ 1 h.
  Full: the other 30 items as two jobs of 15, ≤ 2 h each. Rescoring (if
  adopted): B2's five full jobs' streams, SVT-AV1 scored on any Ada/A6000,
  DCVC-UF decoded on RTX A6000 (its encode class), ≤ 1.5 h per job. Ceiling 9
  GPU-hours and 12 h wall including staging.
- Smokes that failed (not evidence): `20261007T203825Z-0e10810c` (gpu6, Ada,
  `b64332b`): the standalone tracker asked its backbone for the detector's
  head, whose output the tracker cannot read; it now asks only for its own two
  heads. `20261007T204154Z-9531c0f7` (gpu3, A6000, `25db974`): a second mask
  prompt on a fresh multiplex state is refused (above). Every checkpoint key
  loaded in both (931 of 931).
- Pilot `20261007T205637Z-0ba26d70` (gpu3, RTX A6000, `4d7f28b`; smoke on
  P30_110 and validator passed, 18 checks; full stage 573 s, about 5 frames/s
  tracked including the backbone, peak 5.9 GiB): held out, hands mean J 0.90,
  median 0.96 (n = 22; hold floor 0.28); objects 0.88, 0.91 (n = 30; floor
  0.31), hard subset 0.83 (n = 10). Detector agreement: SAM's held-out hands
  0.73 (1,387 of 1,908), dense hands on the same frames 0.83 (1,592 of 1,929).
  Finding: on P09_106 57 of the 66 filled hands lay more than half on the
  other hand, which the dense masks already had (a track that lost its hand
  and latched onto the other; the detector called 32 of them the other side).
  Fill rule added before the full run (adoption rule unchanged): a SAM
  instance more than half on an instance already in the frame is not added
  (`MAX_OVERLAP`). The full run repeats the four pilot items with it.
- Full jobs (commit `1d33976`, environment `pointstream-20261006T113321Z`,
  inputs `pointstream-data/visor/b1b-2026-10-07/inputs/`: detections tar
  `949ea65d…97c3`, manifest `711870f2…1ee5`, SAM checkpoint `0567debe…1cb6`,
  B1 masks `0cf83b8c…37b5`; smokes and validators passed, 19 checks each, and
  the validator passes on both full outputs too): share A
  `20261007T211731Z-f7a21e9f` (17 items, gpu3, RTX A6000, 2,876 s full stage,
  `b1b.json` `b6009493…527a`) and share B `20261007T211914Z-9402b935` (17
  items, gpu6, RTX 6000 Ada, 2,018 s, `39744ab2…c2db`); 17,535 frames
  tracked, 2,344 s of tracking, peak 6.8 GiB; no contention. Report
  (`b1b report`, `pointstream-data/visor/b1b-2026-10-07/report-1d33976/`,
  `b1b-report.json` `b6567477…d2a5`):

  | Held out (prompt at a, score at b) | n | mean J | median J | mean F | hold floor J | hard n | hard mean J |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Hands | 178 | 0.931 | 0.969 | 0.965 | 0.360 | 1 | 0.974 |
  | Hands, EK-100 / EK-55 | 99 / 79 | 0.965 / 0.888 | 0.972 / 0.965 | 0.991 / 0.932 | 0.424 / 0.279 | | |
  | Objects | 407 | 0.828 | 0.911 | 0.905 | 0.252 | 158 | 0.704 |
  | Objects, EK-100 / EK-55 | 221 / 186 | 0.858 / 0.793 | 0.922 / 0.889 | 0.928 / 0.879 | 0.314 / 0.179 | 75 / 83 | 0.746 / 0.666 |

  Mean gap between keyframes 111–117 frames (about 2 s). Hands labelled at a
  and gone at b: SAM let 4 of 6 go (0.67); objects 24 of 49. Detector
  agreement (hand box at least half inside the mask's box): SAM's held-out
  hands 0.941 (13,471 of 14,317), dense hands on the same frames 0.958
  (13,705 of 14,313), ratio 0.98; human hands at keyframes 0.943 (232 of
  246). Frames where the detector sees a hand that the masks lack: 41 of
  14,544 detected hands (0.3%), the same with and without the fill.
- Decision (by the rule): **objects adopted, hands not**. Hands pass every
  accuracy check by a wide margin but fail one: SAM released 4 of 6 departed
  hands, under 0.70 (n = 6, so the failure rests on two cases). The fill
  therefore keeps objects only. On this set that costs little: the overlap
  rule had already left 9 filled hand instances (57 rejected), none of which
  the detector confirmed, and the review frames show the failure mode (a
  hand track moving onto the other hand). The hard subset (objects the dense
  masks drop) is 0.70 against a 0.10 floor. The review by eye
  ([artifact](https://claude.ai/artifact/GSkm3caejx3bNJYUJnQP8K), 68 frames;
  user, 2026-10-08) flagged 5 frames in 4 items (P01_107 w083, P04_24 w055
  and w096, P08_17 w128, P27_105 w169), all hands in SAM's own tracks: the
  hand kept but the forearm lost, or hand and forearm lost. None is in the
  adopted object fill, so there is no veto; they confirm that SAM's hand
  tracks are not fit to fill hands. Also visible there, not flagged: on
  P27_105 the filled fridge is fragmented into speckle, and large touched
  surfaces (sink, fridge, cupboard) enter the foreground as VISOR labels
  them.
- Fill (objects only; merge job `20261007T222646Z-3a7ac6ef`, CPU, `339c918`,
  validator passed; `published.tar` `049977f0…799a`, `merge.json`
  `1c705e4a…4bea`): 18,267 object instances added on 7,504 of 8,160 window
  frames in 33 of 34 items; 833 object instances rejected for overlap. The
  share of each frame's labelled objects missing falls from 33.3% (dense) to
  1.6% (with the fill), mean over items. Limit: on 8 items a prompt is not
  reproduced on its own frame (minimum IoU 0.00–0.76, every item's median ≥
  0.96), most likely thin or tiny masks lost at the tracker's mask-input
  resolution.
- B2 rescored with both mask sets (commit `339c918`; `b2 run --streams` on
  each B2 full job's published streams, every stream checked against B2's
  recorded sha256; all validators passed; no contention): SVT-AV1
  `20261007T223938Z-cc5aa830` and `20261007T224051Z-2891acbc` (gpu6, RTX 6000
  Ada for scoring; dav1d on CPU; 520 and 554 s), DCVC-UF
  `20261007T223639Z-74f60a0f`, `20261007T223737Z-ab077fea`,
  `20261007T223830Z-2139df4c` (gpu3, RTX A6000, B2's encode class; 2,306,
  2,044 and 2,094 s). Reproduction: `visor_dense` scores equal B2's, within
  9e-6 dB for SVT-AV1 (GPU now, CPU then) and exactly for DCVC-UF. Report
  `pointstream-data/visor/b1b-2026-10-07/b2-rescore-339c918/` (`b2-report.json`
  `ce234b4f…9f29`, `b2-rd.png` `83966155…eb80`), means over the 34 items:

  | Codec, point | kbps | wPSNR dense / fill (dB) | fg PSNR dense / fill | bg PSNR dense / fill | wLPIPS dense / fill |
  |---|---:|---|---|---|---|
  | SVT-AV1 CRF 62 | 434 | 34.58 / 34.18 | 34.42 / 33.75 | 34.96 / 35.20 | 0.160 / 0.162 |
  | SVT-AV1 CRF 55 | 1,022 | 36.87 / 36.47 | 36.85 / 36.19 | 36.94 / 37.11 | 0.120 / 0.121 |
  | SVT-AV1 CRF 48 | 1,576 | 37.84 / 37.44 | 37.88 / 37.26 | 37.73 / 37.86 | 0.105 / 0.106 |
  | SVT-AV1 CRF 41 | 2,498 | 38.67 / 38.28 | 38.78 / 38.18 | 38.41 / 38.51 | 0.094 / 0.094 |
  | DCVC-UF QP 27 | 329 | 34.47 / 33.95 | 34.58 / 33.76 | 34.22 / 34.41 | 0.175 / 0.179 |
  | DCVC-UF QP 36 | 584 | 35.97 / 35.45 | 36.09 / 35.28 | 35.67 / 35.84 | 0.142 / 0.145 |
  | DCVC-UF QP 45 | 1,032 | 37.21 / 36.70 | 37.35 / 36.56 | 36.87 / 37.02 | 0.119 / 0.121 |
  | DCVC-UF QP 54 | 1,846 | 38.26 / 37.76 | 38.42 / 37.64 | 37.88 / 38.02 | 0.102 / 0.104 |
  | DCVC-UF QP 63 | 3,283 | 39.14 / 38.63 | 39.31 / 38.53 | 38.74 / 38.87 | 0.090 / 0.091 |

  Labelled objects missing, mean over items: 33.2% dense, 1.6% with the fill
  (EK-100 30.8% / 1.8%, EK-55 35.2% / 1.4%). BD-rate of DCVC-UF HT-L against
  SVT-AV1 on weighted PSNR, mean / median: dense −13.3% / −18.3%, with the
  fill −10.2% / −16.0% (EK-100 −18.3 / −29.9 → −14.5 / −25.6; EK-55 −9.4 /
  −11.3 → −6.8 / −7.6). Whole-frame PSNR (−0.5% / −3.6%) and VMAF (+7.5% /
  +5.0%) use no masks and do not change.
- Reading: the hypothesis holds for objects. The fill cuts the missing
  objects from 33% to 1.6% and moves weighted PSNR by 0.4–0.5 dB (it lowers
  it: the added objects are handled, moving and harder to code than the
  background they were counted as). It narrows DCVC-UF's weighted advantage
  over SVT-AV1 by about 3 points of BD-rate without reversing it. For hands the
  rule held them back on a small-n release check, and the pilot found the
  failure the check guards against. B1b's done-when holds: the numbers from
  recorded jobs decided adoption. GPU use about 4 GPU-hours (fill 1.4,
  rescoring 2.1, smokes and pilot 0.3) of the 9 budgeted; wall time about 5.5 h.
  Not done: train-split filling for E1/F1 (hands need a better release
  rule first).
- Train split without a hand fill (CPU survey, gpu1, 2026-10-08, code
  `2eeb7a2`, not a fleet job; script and output in
  `pointstream-data/visor/b1b-2026-10-07/train-gap-survey-2eeb7a2/`,
  `train_gaps.json` `79beec63…db92`): of the 2,277,946 dense frames of the
  115 train videos, F1 admits 2,171,781 (drift ≤ 1; P17_01 at 47.95 fps has
  no frame rule). Excluding the frames where `hand_gaps` reports a missing
  hand keeps 1,631,454 (75%; EK-100 77%, EK-55 59.94 fps 73%), with 89% of
  the hand masks and 82% of the object masks, and 4,425 of 5,000 dense runs
  keep at least one frame. Per video the loss has median 26%, 90th
  percentile 40%, maximum 55%. Decision: no hand fill for now. F1 and E1
  exclude the gap frames; 1.6 million frames at 50–60 fps from 115 videos is
  far more than either will sample. Revisit only if E1 or F1 turns out data
  limited, or if their errors concentrate on fast motion and occlusion, the
  conditions under which VISOR's interpolation, and so the excluded frames,
  fail.
- Review by eye, fragmentation (CPU, same day, merged fill
  `049977f0…799a`, every 4th frame): 20% of fill instances have more than a
  tenth of their area outside their largest connected piece, against 16% of
  the dense instances (hands occluding objects split masks legitimately);
  12% have more than a quarter outside. Most often fragmented: cheese, bread,
  pot, mozzarella (small or translucent, often held). The speckled fridge on
  P27_105 is the extreme case, not the norm. No change to the fill.

### 2026-10-08 — G1: camera motion and coverage audit
- Question (PLAN G): does each dataset's camera only rotate (and zoom) about
  a fixed centre, so that a panorama built from a warm-up can rebuild its
  background, and how long must the warm-up be? G2–G5 need from it: the clips
  a background evaluation can use, the warm-up length, the panorama's
  projection, and whether the background work starts on VISOR (phase 1) or on
  racket sports.
- Clips (fixed content-blind before any run; `tools/datasets/g1_inputs.py`
  `RULE`; `clips.json` from metadata only: frame counts, splits, names).
  *VISOR windows*: all 34 items of evaluation set v2, every frame (240 at 50
  or 59.94 fps); foreground = the dense masks with B1b's object fill
  (`visor_dense_sam_fill`). *VISOR stretches*: per evaluation video, 120 s of
  decoded frames from the window's first frame (moved earlier to fit the
  video, or the whole video if shorter), at 10 frames/s. *OpenTTGames*: the 7
  test videos (staged whole; the 5 train videos are 4–11 GB each), 120 s from
  sha256("pointstream-g1:ott:<video>") mod the free range, 10 frames/s.
  *RacketVision*: per sport the 8 test clips with the smallest
  sha256("pointstream-g1:racketvision:<sport>/<match>_<rally>"), whole, 10
  frames/s. *TrackNet*: the 8 clips with the smallest
  sha256("pointstream-g1:tracknet:<game>/<clip>"), whole, 10 frames/s. VISOR
  frames are decoded frame indices placed by B1's verified rules; each
  window's released sparse JPEGs are checked against the decoded frames.
- Foreground out of the measurement: VISOR windows by their dense masks;
  everything else by SAM 3.1 text prompts, tier `sam_text` ("hand" and "arm"
  on VISOR, plus the EPIC-KITCHENS hand-object detector's hand and object
  boxes at score ≥ 0.5, since a held object has no fixed name to prompt;
  "person" and "racket" on racket sports, so spectators, umpires and ball
  kids are masked too). All masks dilated by 16 px at 1080p. Tier check: on
  each stretch's frames inside its window, the dense masks' recall by the
  `sam_text` masks, and the rotation residual measured with both.
- Method (`experiments/background/camera.py`, `g1.py`; CPU, OpenCV 4.10 and
  SciPy from the packed environment; 960×540 analysis frames, errors at
  1080p). SIFT on background, ratio test, MAGSAC. Homographies frame to frame
  and frame to reference; the rotation-and-zoom model with the clip's lens
  (focal length and one division-model distortion term, fitted jointly on
  pairs 0.5 s apart; a window uses its stretch's lens; when the rotation is
  too small to identify it, a 60° prior is used and recorded as
  unobservable). Each frame registers to the keyframe its predicted view
  overlaps most (a new keyframe below 60% overlap or 200 inliers). Residual
  of the reference warped into the frame, on the background of both: PSNR
  raw, after a gain and offset (exposure), and after re-warping by dense
  optical flow (DIS); the residual flow on textured background is the
  alignment error. Its squared error splits into exposure, parallax
  (misaligned pixels, > 1 px, whose correspondence obeys the epipolar
  geometry where GRIC prefers a fundamental matrix), independent motion (the
  other misaligned pixels: water, screens, people, mask leaks) and a
  remainder (lighting, blur, noise, disocclusion); it is also given by image
  radius (distortion), against rotation speed (rolling shutter) and on
  blurred frames (Laplacian variance under half the clip median).
  Translation: for frames 0.5 s apart, GRIC (homography against fundamental
  matrix, σ = 1 px at 1080p) and the parallax, the homography's transfer
  error on the fundamental matrix's inliers. Coverage: every frame's
  background is marked on an azimuth–elevation canvas (cell 4 px at 1080p)
  through the rotation model, with the frame that first saw it;
  coverage_t(w) is the share of frame t's background seen within the first w
  seconds, and C(w) its mean over the frames after w. In-job self-test: a real
  frame re-rendered by a known 3°/1° rotation and 2% zoom must come back
  within 0.05° and 0.5 px.
- Decision rule (`g1.DECISION`, fixed before any run). A frame is
  *explained* when the 90th percentile of residual flow on textured
  background, after the rotation-and-zoom warp to its reference, is ≤ 2 px at
  1080p (shares within 1 and 4 px reported beside it). A clip *holds* when
  ≥ 90% of its measured frames are explained, ≥ 95% register directly to a
  keyframe, ≥ 80% are measured (enough visible background) and it has one
  segment (no cut or loss). A dataset (VISOR judged on its stretches, windows
  reported beside them; RacketVision per sport as well) *holds* when ≥ 75% of
  its clips hold, holds *for a subset* at 25–75% (G2 gets the clips that
  hold), and *does not hold* below 25%. Translation is *confirmed* for a clip
  when GRIC prefers a fundamental matrix on ≥ 50% of its 0.5 s pairs and
  their median parallax is ≥ 2 px at 1080p. Warm-up: per clip, the first w on
  a 0.5 s grid with C(w) ≥ 0.90 (and 0.99), else "not reached"; per dataset,
  the median and range over the clips that hold. G2 can use a clip that
  holds and plays at least 10 s after its 90% warm-up. The background work
  starts on VISOR if VISOR holds, or holds for a subset of at least 10
  stretches with a median 90% warm-up ≤ 30 s; otherwise on racket sports if a
  racket dataset holds; otherwise the panorama is not the first candidate and
  G5 (a neural model of the background) leads, on whichever domain G2 can
  evaluate.
- Hypothesis: OpenTTGames' fixed camera holds trivially (no motion, lens
  unobservable), and its coverage is limited only by the players: 90% within
  seconds, 99% late or never. RacketVision and TrackNet broadcast cameras
  mostly hold (fixed, or slow pan and zoom); some clips may break on a cut.
  VISOR does not hold: the head translates while it turns, and at kitchen
  distances (0.3–1.5 m) a few centimetres give parallax well above 2 px within
  0.5 s; GRIC prefers a fundamental matrix on most pairs, and fewer than a
  quarter of the stretches hold. Coverage under the rotation approximation
  needs tens of seconds on VISOR. So the background work starts on racket
  sports.
- Competing explanations: (1) lens distortion of the wide head-mounted
  camera, not parallax, makes the residual: the fitted k1 and the residual
  by radius test it. (2) Rolling shutter under fast head rotation: residual
  that grows with rotation speed while GRIC still prefers a homography
  (Spearman of residual against speed). (3) Foreground mask leaks (hands the
  masks miss, unmasked handled objects, other people) show as independent
  motion near the foreground; the tier check compares the two mask sources on
  the same frames. (4) Blur and compression set a floor unrelated to
  geometry: the flow measure largely ignores photometric noise, blurred
  frames are reported apart, and OpenTTGames' static camera shows the floor.
  (5) Flow errors on flat surfaces: only textured pixels count, and the
  self-test checks the measure on known motion. Frames of a clip are
  correlated, so conclusions are drawn per clip, not per frame.
- Budget: the only GPU work is SAM 3.1 (Ada or A6000, model–GPU table): about
  47,000 analysis frames, two text passes each (VISOR stretches 34 × ≤ 1,200,
  OpenTTGames 7 × ≤ 1,200, RacketVision 24 and TrackNet 8 clips of about 70);
  the CPU analysis runs in the same jobs (16 threads). Smoke ≤ 600 s per job
  (clips capped at 12 s). Pilot: 2 VISOR videos (stretch and window), 2
  OpenTTGames, 1 RacketVision clip per sport, 2 TrackNet, one job ≤ 1 h; it
  measures SAM's frames per second and the analysis seconds per frame, and
  the full run is sized from them. Full: the VISOR videos in three jobs and
  the racket clips in one, ≤ 2.5 h each. Ceiling 10 GPU-hours and 12 h wall
  including staging. If the pilot shows the full run cannot fit, the
  stretches are shortened before anything else changes, and that is recorded
  here.
- Inputs: `pointstream-data/background/g1-2026-10-08/` (`inputs.json`
  `4d2026cd…79f7`, every file's sha256): `clips.json` `a8820a14…1f4c`
  (34 + 34 VISOR, 7 OpenTTGames, 24 RacketVision, 8 TrackNet; about 49,600
  SAM frames); VISOR and OpenTTGames videos hard-linked; RacketVision clips
  packed from the sport tars, each checked against
  `RacketVision.members.sha256` (`racketvision-g1.tar` `0dbada1c…8392`);
  TrackNet JPEGs from `Dataset.zip` (`tracknet-g1.tar` `4b4c4b8b…a4f0`);
  B1b fill `049977f0…799a`, B1 dense archive `a2d9e0cb…4c3c`, detections
  `949ea65d…97c3`, SAM checkpoint `0567debe…1cb6`; environment
  `pointstream-20261006T113321Z`.
- Before any fleet run (dev check on gpu1, CPU, not evidence): registration
  first used the 1 px MAGSAC threshold of the residual and a 200-inlier
  keyframe trigger, so on VISOR 20% of frames were lost and nearly every frame
  became a keyframe. The correspondence thresholds were loosened (MAGSAC 6 px
  at 1080p, inlier share ≥ 0.15, keyframe below 100 inliers); the residual
  criterion that judges a frame (p90 ≤ 2 px) is unchanged.
- Failed attempts (infrastructure, no results): `20261008T151121Z-14aeb6de`
  (`af1b66b`): staging the TrackNet tar failed on the worker, whose zip
  directory entries had been packed as empty files (now `.jpg` members
  only, `14b5a56`). `20261008T154749Z-1e8501a2` (gpu3, `14b5a56`): smoke and
  validator passed (19 checks), then the full stage failed on one clip, where
  OpenCV's USAC raised on a degenerate fundamental-matrix sample; such a fit
  now reads as no fit (`3edfedc`, with a test).
- Pilot `20261008T162927Z-d9372b99` (gpu3, RTX A6000, `3edfedc`; smoke 373 s
  and validator passed, 19 checks; full stage 2,592 s, validator run on the
  full output too, 19 checks; no contention; `g1.json` `b9b9a121…e3d3`,
  `published.tar` `8240a8ab…8447`). SAM 3.1 text passes (two prompts) run at
  2.3–3.3 frames/s at 960×540 (a 1,200-frame stretch in about 450 s), peak
  27.7 GiB (the full jobs declare 32,000 MiB); the CPU analysis takes about
  0.8 s per frame per process alongside. Self-test: 0.0026° and 0.09 px.
  Results (11 clips, report by the rule; one-clip groups are not verdicts):

  | Clip | holds | explained | p90 flow px (median) | f2f / f2r homography explained | GRIC prefers F (0.5 s) | parallax p50 / p90 px | w90 / w99 s |
  |---|---|---:|---:|---|---:|---|---|
  | VISOR stretch P25_101 | no | 0.006 | 21.2 | 0.26 / 0.02 | 0.87 | 2.06 / 5.5 | (11 segments) |
  | VISOR stretch P22_107 | no | 0.000 | 38.4 | 0.47 / 0.01 | 0.74 | 1.43 / 4.8 | (6 segments) |
  | VISOR window P25_101 | no | 0.067 | 9.8 | 0.98 / 0.10 | 0.55 | 1.08 / 2.6 | 0 / 4.0 |
  | VISOR window P22_107 | no | 0.013 | 10.1 | 1.00 / 0.02 | 0.72 | 1.45 / 3.4 | 0 / 2.0 |
  | OpenTTGames test_2 | yes | 0.930 | 1.29 | 1.00 / 0.84 | 0.00 | 0.23 / 0.96 | 0 / 9.0 |
  | OpenTTGames test_4 | yes | 1.000 | 1.15 | 1.00 / 1.00 | 0.00 | 0.19 / 0.77 | 0 / 5.0 |
  | RacketVision badminton | yes | 1.000 | 0.72 | 1.00 / 1.00 | 0.00 | 0.13 / 0.47 | 0 / 5.5 |
  | RacketVision table tennis | yes | 1.000 | 0.38 | 1.00 / 1.00 | 0.00 | 0.02 / 0.11 | 0 / 1.0 |
  | RacketVision tennis | yes | 0.949 | 0.86 | 1.00 / 0.95 | 0.00 | 0.16 / 0.58 | 0 / 1.5 |
  | TrackNet game5/Clip9 | no | 0.883 | 1.25 | 0.95 / 0.87 | 0.00 | 0.23 / 0.79 | 0 / 1.5 |
  | TrackNet game4/Clip3 | yes | 1.000 | 0.41 | 1.00 / 1.00 | 0.00 | 0.15 / 0.56 | 0 / 1.5 |

  Read with care. On VISOR the lens-free homography fails as well (frame to
  reference explained on 1–10% of frames, frame to frame at 50 fps on 98–100%),
  so the failure does not rest on the fitted lens; the lens itself is unstable
  (horizontal field of view 31.5° on P25_101, 76.7° on P22_107) and the
  rotation-and-zoom model takes a 2.5–3.5× focal range on a fixed-focus
  camera: translation leaks into it. The residual sits where depth changes
  (floor against counter); the attribution gives parallax 35–48% and
  independent motion 21–36% of the squared error. By the rule, translation is
  confirmed on 1 of the 2 stretches: GRIC prefers a fundamental matrix on 74%
  and 87% of pairs, but the median parallax of P22_107 is 1.4 px against the
  2 px bar (its p90 is 4.8 px). Stretches lose registration (29–48 frames,
  6–11 segments), so their coverage curves are not meaningful. Tier check:
  the `sam_text` masks with detector boxes recall 69–76% of the dense
  foreground pixels, and the rotation residual on the same frames is the
  same with either mask (explained 0.04 and 0.04, 0 and 0; p90 11.3 against
  10.3 px, 9.5 against 10.0 px). On racket sports the camera did not move
  (median rotation speed under 0.06°/s, lens unobservable, prior used), the first
  frame already shows 90% of every later frame's background, and the players
  set the 99% warm-up (1–9 s). TrackNet game5/Clip9 misses by unmasked
  player shadows; the "person" prompt also masks spectators and ball kids.
  RacketVision and TrackNet clips are 4–10 s rallies, so none plays 10 s
  after its warm-up (the G2 rule); a panorama shared across a match's rallies
  is outside G1.
- Full run sized from the pilot (same revision `3edfedc`, so the pilot's
  11 clips stand and are not repeated): the other 32 VISOR videos in three
  jobs (11, 11, 10 videos; SAM about 450 s per stretch, ≤ 9,000 s full
  stage each), and the other 32 racket clips in one (≤ 7,200 s). Estimated
  6.6 GPU-hours, 8.2 with the pilot and the failed attempts, inside the
  10-hour ceiling. Jobs `20261008T172332Z-8f85d3af`,
  `20261008T172338Z-c73de835`, `20261008T172343Z-cbfd15b8` (VISOR) and
  `20261008T172349Z-56d1bb1a` (racket), on gpu3, gpu5 and gpu6 (gpu2's
  worker had no fresh heartbeat).
- Full run, what happened. VISOR share 3 `20261008T172343Z-cbfd15b8` (gpu3,
  RTX A6000, `3edfedc`; 5,978 s, no contention; validator passed on the full
  output). Shares 1 and 2 (`…8f85d3af`, `…c73de835`) stopped at the smoke
  gate on a validator error, not on data: the 6 s smoke cap left P01_107's
  and P06_10's stretches before their windows, so the tier check had no
  frames; it now applies only where the analysed range reaches the window
  (`6a391dc`, which changes only the validator). The racket job
  (`…56d1bb1a`) was stopped as contended (another user's process held its GPU
  past the 900 s pause; no results kept). Reruns at `6a391dc`, smoke gates
  passed, no contention: VISOR share 1 `20261008T203704Z-c101d42a` (gpu6, RTX
  6000 Ada, 4,310 s), share 2 `20261008T203709Z-614dcdf0` (gpu6, 4,157 s),
  racket `20261008T203714Z-fd5d2f5b` (gpu5, RTX 6000 Ada, 2,413 s). The
  validator passes on every full output except share 1, which fails only
  `visor_window_jpegs_match_decoded_frames` because P04_24's window holds no
  released sparse JPEG (the check wants a non-empty list); every JPEG that
  was checked matched its decoded frame (26 of 26 in that share).
  About 7 GPU-hours in all, inside the 10-hour ceiling; the wall time (about
  21 h from staging) exceeded the 12 h ceiling because only one Ada/A6000
  GPU was free for most of the evening (gpu5 and gpu6 busy with other users,
  gpu2 unreachable).
- Report: `pointstream-data/background/g1-2026-10-08/report-8e79d9e/`
  (`g1-report.json` `aa1c334f…89cc`, figures included), all 107 clips from the pilot and the
  four full jobs. Review page with overlays and coverage maps:
  [artifact](https://claude.ai/artifact/LB8yoyyc3ebqb7Bii6X2RF).

  | Dataset | clips | hold | verdict | explained (median) | p90 flow px (median) | homography to ref. explained | translation confirmed | GRIC prefers F (median) | w90 / w99 s (holding) | G2 clips |
  |---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
  | VISOR stretches | 34 | 0 | does not hold | 0.006 | 30.2 | 0.02 | 17/34 | 0.87 | – | 0 |
  | VISOR windows | 34 | 0 | does not hold | 0.063 | 13.9 | 0.08 | 16/34 | 0.89 | – | 0 |
  | OpenTTGames | 7 | 6 | holds | 0.998 | 1.15 | 0.99 | 0/7 | 0.00 | 0 / 3.3 | 6 |
  | RacketVision | 24 | 13 | holds for a subset | 0.944 | 1.22 | 0.89 | 0/24 | 0.00 | 0 / 1.5 | 0 |
  | TrackNet | 8 | 4 | holds for a subset | 0.942 | 1.02 | 0.93 | 0/8 | 0.00 | 0 / 1.3 | 2 |

  RacketVision per sport: badminton 4/8, table tennis 5/8, tennis 4/8 (each
  a subset).
- Outcome by the rule. *VISOR does not hold* (0 of 34 stretches, 0 of 34
  windows). The failure does not rest on the fitted lens: a full homography
  to the reference explains 2% (stretches) and 8% (windows) of frames, while
  frame to frame at 50–60 fps explains 99% of window frames. Residual energy
  is parallax 39% and independent motion 32% on the stretches (42% and 28% on
  the windows), already 7.6 px at the image centre and rising to 14.9 px at
  the border (distortion, fitted, cannot be the main cause; the border also
  shows the nearest surfaces), and little related to rotation speed
  (Spearman 0.18), so rolling shutter does not explain it either. GRIC prefers
  a translating camera on a median 87% of 0.5 s pairs; translation is
  confirmed by the rule on 17 of 34 stretches (median parallax 2.0 px, at
  the rule's bar). The tier check holds: on the stretches' window frames the
  rotation residual is the same with the dense masks and with `sam_text`
  (p90 18.6 against 17.9 px, explained 0 and 0), although `sam_text` with
  detector boxes recalls only 53% of the dense foreground pixels. VISOR's
  fitted lens is unstable (median horizontal field of view 62°, varying
  widely), as expected when translation leaks into a rotation-only fit.
  *Racket sports: no camera translates* (0 of 39 clips; parallax medians
  0.07–0.20 px). OpenTTGames holds (6/7; test_3 misses at 0.72 explained).
  RacketVision and TrackNet hold for a subset; of their 15 failing clips, 4
  RacketVision clips pan or zoom (table tennis match10_011, match10_001,
  match11_007; tennis match132_000), and the rest are still cameras whose
  background moves: crowds, caption graphics sliding in (badminton
  match133_000), exposure changes (TrackNet game1/Clip12), unmasked player
  shadows. TrackNet game3/Clip6 (explained 0.03, still camera) has residual
  spread over the low-texture court with no visible moving object; its cause
  is not determined. *Warm-up*: wherever a clip holds, the first frame
  already shows at least 90% of every later frame's background (w90 = 0 s);
  the players set the 99% warm-up, median 3.3 s (OpenTTGames, 1–9 s), 1.5 s
  (RacketVision) and 1.3 s (TrackNet). *G2 can use 8 clips*: OpenTTGames
  test_1, test_2, test_4, test_5, test_6, test_7 and TrackNet game8/Clip3,
  game10/Clip1; RacketVision rallies (5–10 s) are too short for 10 s of play
  after a warm-up. *Start*: by the rule, the background work starts on racket
  sports, on OpenTTGames, the only racket dataset that holds outright.
- Reading for G2–G5. The hypothesis holds for VISOR and for OpenTTGames. It
  was too optimistic for the broadcast sets: their cameras turn at most, but
  a rotation-only panorama alone is not a background for them, because a
  quarter to half of their clips have moving background (crowds, graphics,
  lighting). A panorama is a static background plus a residual, and G2 must
  measure that residual's rate. For VISOR a rotation-only panorama is ruled
  out; the neural route (G5) or a 3D representation is the candidate for
  egocentric video, and the projection question (G3) only arises for the
  fixed and broadcast cameras, where a planar projection suffices (median
  rotation under 0.04°/s on every holding clip). Not measured: whether one panorama can
  serve all rallies of a broadcast match (outside G1).
- After review (2026-10-09). (1) P04_24 is not a gap in the data: VISOR
  releases sparse JPEGs only at its human-labelled keyframes, and P04_24's
  240-frame window lies between two of them; its frames are placed by B1's
  drift-free rule like every other window. The validator now accepts a
  window with no released JPEG inside it instead of failing on an empty list;
  nothing is removed from the evaluation set. (2) The four RacketVision clips
  that pan or zoom fail even a full homography to their reference, and two of
  them (table tennis match10_001 and match11_007: zooms of 1.9× and 3.7× within
  5 s) fail frame to frame too, so they are not a pure pan-and-zoom the model
  already covers; likely a moving camera rig, blur and LED boards, not
  established. (3) G1's job now checkpoints each finished clip and restores it
  on a declared resume ([fleet](fleet.md#checkpoints)); the analysis is
  unchanged. (4) `g1 report` writes the paper's figures from the recorded
  outputs (vector PDF and PNG: coverage per racket dataset, residual per clip,
  per-frame residual; one hash-picked overlay sheet per dataset).

### 2026-10-09 — G1d: depth-aware warps on VISOR's background
- Question: G1 found that no VISOR stretch or window holds rotation-only
  (residual energy about 39% parallax and 32% independent motion on the
  stretches). How much of that residual does a warp that knows depth remove?
  Does egocentric video get a 3D background (a G3 variant), or go straight to
  the neural route (G5)?
- Clips (fixed content-blind before any run). All 34 windows of evaluation
  set v2, every frame, with G1's foreground (dense masks with B1b's fill). The
  10 stretches with the smallest sha256("pointstream-g1d:visor-long:<video>"):
  P26_02, P03_10, P06_10, P06_03, P37_102, P06_05, P02_09, P04_13, P09_106 and
  P06_106. Each is G1's 120 s at 10 frames/s with G1's foreground: the
  published SAM 3.1 `sam_text` masks plus the detector boxes, dilated as in
  G1. Frames are G1's decoded analysis frames. One CPU job decodes them once
  into a lossless archive (FFV1) with the masks. The job checks that each
  frame's foreground share equals the one G1 recorded and that the windows'
  released JPEGs still match the decoded frames.
- Pairs. G1's registration is re-run with G1's recorded lens. Every
  `direct` frame t is warped from its G1 reference keyframe k by every
  method, so all methods are compared on the same pairs. The job checks that
  the references and G1's rotation residual (G1's own measure) reproduce.
  Depth methods use EPIC Fields' published COLMAP calibration of
  EPIC-KITCHENS (P28_101: fx 239.6, fy 243.1 at 456×256, so 86.7° horizontal
  field of view with their mean; distortion negligible). A per-clip
  self-calibration is reported beside it as a diagnostic: the grid of focal
  length and k1 that minimises the calibrated essential matrix's Sampson
  error.
- Methods, in order:
  `rot` (G1's rotation and zoom, reproduced);
  (a) `h1` (one homography on undistorted pixels) and `planes2/3/4`. These
  carry a label map on the keyframe, sent once. K planes come from
  sequential MAGSAC (1 analysis px, 2 px at 1080p) on dense correspondences
  from k to its first companion: DIS flow seeded by a homography, sampled
  every 6 px on textured background. Each pixel takes the plane that best
  predicts its flow, then a 31 px mode filter. Each frame sends K
  homographies, which the encoder fits to the dense flow from k to t, per
  label, falling back to `h1` below 30 samples. The renderer forward-splats
  keyframe pixels, and the larger displacement wins a collision, because
  nearer surfaces move more;
  (b) `epi`: the dense flow from t to k, projected onto the pair's epipolar
  lines (fundamental matrix from the pair's SIFT matches). It removes the
  motion a rigid scene can explain, but it inherits every error of the
  per-pixel flow, so it is *not* a ceiling: the dev check below found `tri`
  above it. It is not a representation.
  `tri`, the triangulated upper bound: the keyframe's depth comes from dense
  flow to two companions. They are the frames nearest 0.5 s and 1.0 s after k
  (or before, at a clip's end; within 0.3–1.5 s, same segment, at least 30
  homography inliers). Each companion is posed by the essential matrix, and
  each pixel is triangulated (angle ≥ 0.5°, reprojection ≤ 1.5 analysis px).
  The primary is the companion with the most pose inliers. A second joins
  only if, scale-matched, it agrees on their shared pixels (median |log
  ratio| ≤ 0.05); where both measure, the larger angle wins. Background holes
  are filled by local plane fits of inverse depth (windows 31–241 px, at
  least 15% measured). Inverse depth is coded in 8 bits. Each frame's pose
  comes from PnP (RANSAC, then LM) on the keyframe's 3D points at the pair's
  matches. The renderer forward-splats the keyframe's background into t with
  a z-buffer, fills cracks of at most 3×3, then warps colour backwards.
  `tri_raw` renders only the triangulated pixels, so its holes show how much
  of `tri` rests on the fill. A frame t is excluded only when it is a
  companion of its own reference, the circular case;
  (c) `da3`, Depth Anything 3 depth of the keyframe, monocular. It uses the
  same PnP and renderer, runs only if the rule below reaches (c), and is
  audited into the environment first;
  (d) `kf`, depth-augmented keyframes as the representation: the better of
  `tri` and `da3`, rendered from the reference and the nearest other
  keyframe, merged by z-buffer. 3DGS is tried only if (d) leaves holes (below).
- Measures, per method and frame, are G1's `camera.residual` on the rendered
  region of the background: p90 residual flow on textured background at
  1080p (≤ 2 px), shares within 1/2/4 px, PSNR raw / after gain / after
  flow, and the attribution (exposure, parallax by the pair's epipolar test,
  independent motion, remainder). Before the flow is measured, each warp's
  map is extrapolated into its invalid pixels (`depth.complete`), so that
  holes inside the warped image do not drag the flow at their edges; the
  scored region is unchanged. Beside them: the hole share (the background
  both views see, `h1`'s region, that the method does not render), the share
  of > 2 px pixels within 32 px (1080p) of the foreground, the rate in bytes
  (`rot` 4 float32 per frame; `h1` 8; planes 8K per frame plus the PNG label
  map per keyframe; `tri`/`da3` 6 float32 per frame plus the PNG of the
  8-bit inverse depth per keyframe, its 16-bit size beside it; `kf` also the
  keyframe's colour as JPEG q90, which every keyframe representation pays),
  and the render time per frame (one CPU thread, warp construction only). A
  frame is *explained* when its p90 is ≤ 2 px **and** its hole share is
  ≤ 10%. Per clip, shares are over G1's measured frames, and a frame a
  method cannot render counts as not explained.
- Decision rule (`g1d.DECISION`, fixed before any fleet run). A method
  brings a group under 2 px *on a meaningful share* when its median clip
  explains at least 50% of frames. Windows and stretches are judged
  separately; clips with ≥ 90% (G1's clip bar) are reported beside. (1) If
  neither `epi` nor `tri` has a meaningful share in either group, no
  static-scene warp at hand brings VISOR under 2 px. Then (c) and (d) are
  not run, and egocentric goes to G5. (2) Otherwise (c) and (d) run. The
  chosen representation is the realizable method (`planes*`, `tri`,
  `tri_raw`, `da3`, `kf`) with the fewest bytes that has a meaningful share
  in both groups: egocentric video then gets that background in G3. If none
  qualifies, G5 leads, and the gap between the best measure (`epi` or `tri`)
  and the best realizable method is recorded. (3) 3DGS is tried only if the
  chosen method's median hole share exceeds 5%. In every case, the
  composition of the residual left by the best methods answers what the
  remaining residual is made of.
- Hypothesis: planes help little (median window explained ≤ 20% at K = 4).
  Depth reaches a meaningful share on the windows but not on the stretches,
  where references are older and other people, water and lighting change.
  So no realizable representation qualifies on both groups, and G5 leads for
  egocentric video. What remains is mostly independent motion near the
  hands and a photometric remainder.
- Competing explanations: (1) a measurement floor: DIS flow between views
  far apart in time or lighting reports misalignment where the geometry is
  right. Tested by the residual against the time to the reference, by the
  PSNR after flow, and against OpenTTGames' static floor (G1: 1.15 px p90).
  (2) Rolling shutter: no single pose fits, so the residual grows with
  rotation speed (Spearman). (3) Mask leaks: residual next to the foreground
  (the near-foreground share). (4) Calibration: `epi` and the planes do not
  need the focal length; `tri` does, and the self-calibration's error at
  EPIC Fields' lens shows how well that lens fits each clip.
- Budget: preparation CPU job (stage the 34 videos, 73 GB, plus the masks;
  decode 34 windows and 10 stretches) ≤ 2 h at 16 threads. Evaluation CPU
  jobs: smoke ≤ 600 s (2 windows and 1 stretch, 40 frames each), pilot on
  the same clips in full ≤ 1 h to measure seconds per frame, full ≤ 3 h at
  48 threads. DA3, if reached: one GPU job ≤ 1 GPU-hour on Ada or A6000,
  including its first-run kernel check. Ceiling 2 GPU-hours, 8 CPU-job-hours,
  14 h wall.
- Before any fleet run (dev checks on gpu6 with the packed environment, the
  pilot clips capped at 40 frames; not evidence). The preparation round trip
  is lossless, and its foreground equals G1's on every frame. Five changes
  followed. (i) The first lens was the per-clip self-calibration. It drifted
  to the grid's edge (37° and pincushion k1 on two of three clips), because
  near-rotational pairs let a long focal length fit the essential matrix
  too; EPIC Fields' calibration replaced it. (ii) Holes inside a splatted
  image are black, and DIS flow dragged at their edges: `tri` measured
  12.5 px on a pair whose own correspondences it reproduced within 1.1 px
  (p90). Hence `depth.complete`. G1 never met this, because its warps leave
  holes only at the border. (iii) Telea inpainting of inverse depth put
  8–15 px (p50) of error into the 16% of textured pixels it filled. Local
  plane fits brought them to 2.7–4.9 px, and `tri` on that pair from 18 to
  4.5 px. A second companion with 129 pose inliers had also overridden good
  pixels, hence the agreement test. (iv) Sparse SIFT (about 500 matches)
  supports two planes at most, so `planes3/4` equalled `planes2`; dense
  fitting separates them (P02_12 window: 7.7, 4.4 and 3.6 px p90 for K = 2,
  3 and 4, against 13.5 for `h1`). (v) `epi` was meant as the ceiling, but
  `tri` explained all 37 frames of P03_120 where `epi` explained 57%, so the
  rule's step 1 now asks of both. Reproduction of G1: references and rotation
  p90 matched on every frame (tolerance 0.05 px).
- Preparation `20261009T075810Z-295529d4` (gpu6, CPU, `499998c`; smoke 70 s,
  full 312 s; validator passed on smoke and full output, 8 checks): 44
  clips, 7.0 GB of FFV1 (`published.tar` `5a9c47db…5d62`). The round trip
  is lossless, the foreground equals G1's on every frame, and the window
  JPEGs match. Hosts gpu1, gpu3, gpu5 and gpu6: gpu2 refused SSH (no worker
  heartbeat), and gpu4 is left out because it is locked for measurements.
- Pilot `20261009T080919Z-2aa14966` (gpu1, CPU, `0a5bb1f`; smoke gate
  passed, 10 checks; full 623 s at 32 threads, about 950 frames; validator
  passed on the full output). G1's references and rotation p90 reproduce on
  every frame. Explained share (p90 ≤ 2 px, holes ≤ 10%):

  | Clip | rot | h1 | planes2 / 3 / 4 | epi | tri | tri_raw (ignoring holes) |
  |---|---:|---:|---|---:|---:|---:|
  | window P02_12 | 0.07 | 0.17 | 0.22 / 0.22 / 0.23 | 0.35 | 0.15 | 0.20 |
  | window P03_120 | 0.11 | 0.14 | 0.19 / 0.20 / 0.20 | 0.70 | 0.20 | 0.70 |
  | stretch P26_02 | 0.01 | 0.03 | 0.05 / 0.04 / 0.05 | 0.08 | 0.02 | 0.10 |

  Over a whole window, every residual grows with the time to the reference
  (Spearman 0.76–0.93; the windows' references are their first frame).
  Depth triangulated 0.5–1 s after the keyframe does not hold seconds
  later, so `tri` falls far below its 40-frame dev check. The full run of
  the other 41 clips is sized from this (about 18,500 frames, about 1 h at
  48 threads): `20261009T082847Z-36c4a84c`, same revision, so the pilot's
  clips stand.
- Full run `20261009T082847Z-36c4a84c` (gpu6, CPU, `0a5bb1f`; smoke gate
  passed, 10 checks; full 2,137 s at 48 threads, no contention; validator
  passed on the full output). G1's references and rotation p90 reproduce on
  every frame of all 44 clips. Report:
  `pointstream-data/background/g1d-2026-10-09/report-b009870/`
  (`g1d-report.json` `3cf1d238…07e7`, from the pilot's and this job's
  `g1d.json`). Medians over clips (explained: p90 ≤ 2 px and holes ≤ 10%;
  energy: share of the raw squared error, mean over frames):

  | Group | Method | explained (median clip) | clips ≥ 50% | p90 px | holes | bytes/frame | render s | parallax / independent / remainder / exposure |
  |---|---|---:|---:|---:|---:|---:|---:|---|
  | windows (34) | rot | 0.06 | 0 | 13.9 | 0.01 | 16 | 0.11 | 0.42 / 0.28 / 0.24 / 0.06 |
  | windows | h1 | 0.08 | 1 | 8.2 | 0.01 | 32 | 0.09 | 0.36 / 0.27 / 0.31 / 0.06 |
  | windows | planes2 / 3 / 4 | 0.14 / 0.14 / 0.16 | 1 / 3 / 3 | 5.7 / 4.8 / 4.4 | 0.02–0.03 | 343 / 451 / 535 | 0.44–0.46 | (K = 4) 0.28 / 0.24 / 0.42 / 0.06 |
  | windows | epi | 0.34 | 8 | 2.7 | 0.03 | – | 0.40 | 0.12 / 0.29 / 0.51 / 0.07 |
  | windows | tri | 0.09 | 2 | 5.5 | 0.03 | 4,042 | 0.49 | 0.28 / 0.28 / 0.37 / 0.07 |
  | windows | tri_raw | 0.00 (holes) | 1 | 2.7 | 0.35 | 4,042 | 0.33 | 0.25 / 0.14 / 0.53 / 0.08 |
  | stretches (10) | rot | 0.005 | 0 | 28.9 | 0.01 | 16 | 0.11 | 0.43 / 0.31 / 0.17 / 0.09 |
  | stretches | h1 | 0.016 | 0 | 22.3 | 0.01 | 32 | 0.09 | 0.36 / 0.34 / 0.22 / 0.09 |
  | stretches | planes2 / 3 / 4 | 0.04 / 0.04 / 0.04 | 0 | 16.9 / 14.4 / 13.2 | 0.02–0.04 | 888 / 1,224 / 1,465 | 0.51–0.53 | (K = 4) 0.28 / 0.33 / 0.30 / 0.09 |
  | stretches | epi | 0.09 | 0 | 6.6 | 0.02 | – | 0.40 | 0.11 / 0.37 / 0.42 / 0.10 |
  | stretches | tri | 0.02 | 0 | 14.8 | 0.04 | 12,406 | 0.57 | 0.29 / 0.33 / 0.29 / 0.09 |
  | stretches | tri_raw | 0.00 (holes) | 0 | 6.1 | 0.42 | 12,389 | 0.35 | 0.31 / 0.20 / 0.38 / 0.10 |

  Per clip, `epi` explains 2–100% of a window's frames (8 of 34 at least
  half, one all) and 5–27% of a stretch's. `tri` explains at most 68% of a
  window and 12% of a stretch. Bytes per frame count the per-frame
  parameters plus each used keyframe's map, spread over the frames that use
  it; stretches have a keyframe every few frames, hence their larger rates.
  Render times are one CPU thread in numpy, warp construction only.
- Outcome by the rule: **step 1**. Neither `epi` (median clip 0.34 on the
  windows, 0.09 on the stretches) nor `tri` (0.09, 0.02) reaches a
  meaningful share in either group. No depth-aware warp at hand brings
  VISOR's background under 2 px on a meaningful share of frames, so (c)
  (Depth Anything 3) and (d) (depth-augmented keyframes) were not run, and
  3DGS does not arise. **Egocentric video goes to G5**, the neural
  background model. Code for (c) and (d) (`experiments/background/da3.py`,
  methods `da3`, `da3_tri` and `kf`) is in the repository but has never run.
  DA3 was not audited into the environment and no weight was downloaded.
  The plan for that audit is a pure-Python overlay (DA3 at `3d835ec`,
  `addict`, `omegaconf`) with import stubs for `pycolmap` and `evo`, which
  monocular inference never calls.
- What depth removes and what remains. Depth does what it should to the
  geometric part. A warp that follows each pixel's epipolar line (`epi`)
  cuts parallax from 42% to 12% of the error on the windows (43% to 11% on
  the stretches), and the median p90 from 13.9 to 2.7 px (28.9 to 6.6 px).
  Four planes reach 4.4 px (13.2), so planes take a third to half of the way
  at about 535 bytes per frame. A representation built once from a keyframe
  does not hold, though. Depth triangulated 0.5–1 s after the keyframe
  (`tri`) is right where it was measured (`tri_raw`: p90 2.7 px on the
  windows), but that is 58–65% of the background, and the plane-filled rest
  pulls it back to 5.5 px. Every residual grows with the time to the
  reference (Spearman about 0.55–0.69 on the windows, whose reference is
  their first frame, and 0.36–0.42 on the stretches), and barely with the
  rotation speed (0.11–0.28), so rolling shutter is not the cause. What
  remains after `epi`, the most generous static-scene warp, is not geometry
  a better depth could fix. On the windows (stretches) it is 29% (37%)
  independent motion, 51% (42%) remainder (lighting, blur, noise,
  disocclusion), 7% (10%) exposure, and 12% (11%) parallax. Mask leaks are
  not it: only 4–7% of the misaligned pixels lie within 32 px of the
  foreground. PSNR after flow stays at 31.9 dB (30.2 on the stretches).
  Competing explanation (1), a measurement floor at wide time gaps, is not
  excluded: the residual's growth with the time gap fits both a scene and
  appearance that change and a flow that degrades. OpenTTGames' static floor
  (1.15 px p90) shows the measure can reach the bar on a static scene.
  Calibration (4): the self-calibration ranges 41–96° per clip against EPIC
  Fields' 86.7°. It is a diagnostic, but `tri`'s focal length is the one
  untested assumption. `epi` does not depend on it, and the decision rests
  on `epi` failing too.
- Hypothesis: partly right. Planes helped little (window median 0.16 at
  K = 4, inside the ≤ 0.20 predicted). Depth did *not* reach a meaningful
  share on the windows (predicted it would): the windows' reference is their
  first frame, up to 4.8 s back. The remainder is larger than predicted, and
  independent motion near the hands is not where it sits.
- Budget: CPU only, 0 GPU-hours (ceiling 2). CPU jobs: preparation 312 s,
  pilot 623 s, full 2,137 s, so about 50 min of CPU-job wall (ceiling 8
  CPU-job-hours). Wall time from the first submission to the report: about
  1.5 h (ceiling 14 h).
- After review (2026-10-09). (1) *Scope of the verdict.* The rule judged G1's
  pairs, where a frame keeps one reference for up to 4.8 s on the windows
  and seconds on the stretches. It rules out a background sent once and
  reused for seconds *on VISOR*. Racket sports are outside it: G1 found
  their fixed cameras hold a panorama sent once (OpenTTGames 6 of 7 clips,
  first frame already 90% of every later background). It says nothing about
  references kept fresh. Binned by
  the age of the reference (`g1d ages`, from the two runs' published
  results, no new computation; `ages-fc4ccdf/g1d-ages.json` `ad72bbe5…f6ef`):

  | Windows: reference age | frames | rot | planes4 | epi | tri |
  |---|---:|---:|---:|---:|---:|
  | < 0.1 s | 1,186 | 37% | 49% (p90 1.5 px) | 70% (1.3 px) | 33% |
  | 0.1–0.25 s | 922 | 10% | 35% | 43% | 19% |
  | 0.25–0.5 s | 883 | 4% | 29% | 37% | 16% |
  | 0.5–1 s | 1,203 | 2% | 20% | 33% | 9% |
  | 1–2 s | 1,601 | 1% | 8% | 24% | 6% |
  | 2–5 s | 1,742 | 2% | 11% | 25% | 10% |

  On the stretches, the best bin with many frames (0.1–0.25 s, 2,166 frames)
  reaches 18% (`epi`) and 10% (`planes4`). A bin's share also depends on
  which clips have frames at that age, so this is indicative. It moves the
  question to how often a reference must be refreshed and what the refreshes
  cost, which is G1e. G5 is not decided yet. (2) *The oracle was not a bound.*
  `epi` was meant to bound every static-scene warp, but `tri` beat it on one
  pilot clip, because `epi` inherits the per-pixel flow's errors. The
  protocol now requires an oracle to dominate what it gates (PR #180). The
  verdict still stands for G1's pairs, since `tri` and the planes failed
  too. (3) *A mask confound on the stretches.* Their foreground is G1's
  `sam_text` tier, which recalls about half of the dense masks' hand pixels
  (G1 tier check). Unmasked hands then count as background motion. That may
  be why the stretches stay far below the windows even with fresh
  references, and G1e checks it.

### 2026-10-09 — G1e: reference refresh on VISOR
- Question: G1d ruled out a VISOR background reference that is sent once and
  reused for seconds. Binned by age, G1d's records explain far more with
  fresh references (windows under 0.1 s: 70% with `epi`, 49% with four
  planes). Can a reference refreshed often enough reach the 2 px bar, and
  how often must it be refreshed? If a sendable warp passes at a useful age,
  G2 prices the refreshes in bytes. If not, egocentric video goes to G5.
- Why a new measurement and not G1d's bins. In G1d, a frame's age was G1's
  choice: G1 started a keyframe where tracking failed, so fresh-reference
  frames are not a random sample. A bin also mixes clips. G1e fixes the
  targets and forces the age, so every age is scored on the same frames.
- Clips: G1d's 34 windows and 10 stretches, from G1d's prepared lossless
  archive (`20261009T075810Z-295529d4`, `published.tar` `5a9c47db…5d62`),
  with the same foreground. Measurement floor: the 7 OpenTTGames test clips
  of G1 (120 s at 10 frames/s, G1's `sam_text` masks of persons and rackets),
  prepared the same way in one more CPU job.
- Pairs. Targets are fixed before any run, on a stride, so that correlated
  neighbours do not count as separate samples. Windows: every 10th frame from
  1 s (about 20 per window, 0.2 s apart). Stretches and OpenTTGames: one per
  second from 1 s (119 per clip). Each target t is warped from the reference
  r = t − g for each age g in {1 frame, 0.1 s, 1 s}. On windows (50 or 59.94
  frames/s) these are 1, 5 or 6, and 50 or 60 frames. On stretches and
  OpenTTGames (10 frames/s) one frame is 0.1 s, so they get two ages. Every
  age uses the same targets. A pair is matched by SIFT and a homography with
  at least 30 inliers, as in G1. A pair that does not match counts as not
  explained, and the matched share is reported.
- Methods, all from G1d (`depth.py`), with the reference r as the keyframe:
  `h1` (one homography; also its region defines holes, as in G1d), `planes4`
  (the sendable warp: a label map on the reference, fitted to the dense flow
  to the reference's companion 0.5 s later, then four homographies per
  frame), `epi` (dense flow projected onto epipolar lines; not sendable) and
  `tri` (depth of the reference triangulated from its companions, PnP per
  frame). Companions are chosen as in G1d, without G1's registration (the
  six nearest candidates to each companion time). An encoder that refreshes
  references can see 0.5–1 s ahead at the cost of that much latency, which
  the rate step (G2) would have to state.
- The oracle, and why it bounds what it gates. It gates refreshed
  static-scene warps: `planes4`, `tri`, and the depth representations G1d did
  not run (Depth Anything 3 depth, depth-augmented keyframes, 3DGS). G1d
  found that `epi` alone does not dominate (`tri` beat it on P03_120). So
  the oracle is the per-pair best of `epi`, `planes4` and `tri`: a pair is
  explained if any of the three explains it. By construction it dominates
  `planes4` and `tri`. It does not dominate unmeasured depth (DA3, 3DGS)
  by construction. The argument for them is that after `epi`, 87–89% of
  G1d's residual energy was independent motion, remainder and exposure, and
  a static-scene warp of any depth cannot remove those. Per pair, the
  share where `epi` alone matches the oracle is reported, as a check of
  that argument at fresh ages.
- Measures per pair and method: G1d's (`camera.residual` with
  `depth.complete`): p90 residual flow on textured background at 1080p,
  hole share against `h1`'s region, shares within 1/2/4 px, PSNR after flow,
  energy attribution, near-foreground share. A pair is *explained* when
  p90 ≤ 2 px and holes ≤ 10%. Rate is per frame: `planes4` 128 bytes plus its
  label map PNG per reference, and `tri` 24 bytes plus its 8-bit
  inverse-depth PNG per reference. The reference image itself is G2's to
  price. Render time: one CPU thread, warp construction. Two checks beside it:
  - *Measurement floor, two ways.* (i) OpenTTGames, a static camera, at
    0.1 s and 1 s: `h1` p90 and the explained share. (ii) On every VISOR
    pair, the forward-backward consistency of the DIS flow on textured
    background: the p90 at 1080p of |f(x) + b(x + f(x))|, where f is the
    flow from the target to the warped reference and b the flow back. It is
    computed for every warp, and the rule reads `epi`'s, the closest
    alignment. A pair whose own flow disagrees with itself by more than 2 px
    cannot certify 2 px.
  - *Mask confound.* The one-per-second targets rarely fall inside a
    stretch's evaluation window, so this check takes every stretch frame
    there (10 per second, from 1 s). VISOR's dense masks exist there, read
    from the B1b fill already staged for G1d. Each pair is scored twice:
    with G1's `sam_text` foreground and with the dense masks (dilated as in
    G1). The reference also needs a dense mask, so at 1 s only targets at
    least 1 s into the window count.
- Decision rule (`g1e.DECISION`, fixed before any fleet run). As in G1d, a
  method is meaningful in a group at an age when its median clip explains at
  least 50% of targets. A method's *refresh age* in a group is the largest
  tested age at which it is meaningful. Windows and stretches are judged
  separately.
  1. *Oracle gate.* If the oracle's refresh age on the windows is below
     0.1 s (only one frame, or none), a refreshed reference is no cheaper
     than coding every frame. No refresh scheme is built, and egocentric
     video goes to G5.
  2. *Sendable warp.* Otherwise, if `planes4` (or `tri`) has a refresh age of
     at least 0.1 s on the windows *and* on the stretches, G2 prices that
     warp refreshed at that age against SVT-AV1 and DCVC-UF on the same
     background frames. If several qualify, the one with fewer bytes per
     frame goes forward. The stretch verdict stands only if the mask check
     agrees: the dense and `sam_text` explained shares on the same pairs
     differ by at most 10 points. Otherwise the stretch verdict is the
     dense-mask one, on those pairs only, and is flagged as resting on
     fewer frames.
  3. If the oracle qualifies at ≥ 0.1 s but no sendable warp does, the gap
     is recorded, and the next step is a sendable warp between them, with
     DA3 or depth-augmented keyframes from G1d's (c) and (d), at the oracle's
     refresh age. That step gets its own entry.
  4. *Refinement.* If `planes4` or the oracle is meaningful at 0.1 s but
     not at 1 s on the windows, one more age, 0.3 s, is run on the same
     targets. This places the refresh age within a factor of about 3, since
     G2's rate scales with the refresh rate. No other ages are added.
  5. *Floor.* The growth with age counts as real only if OpenTTGames'
     median `h1` p90 stays ≤ 2 px at 1 s and VISOR's forward-backward p90
     stays ≤ 2 px at the age in question. Where it does not, that age's
     verdict is reported as *not measurable*, and it is not counted as a
     failure.
- Hypothesis: on the windows the oracle is meaningful at one frame and at
  0.1 s (about 65–75%), but not at 1 s. `planes4` is meaningful at one frame
  and borderline at 0.1 s (40–55%), so step 4 runs. The stretches stay below
  the windows at 0.1 s. The mask check closes part of that gap, not all of
  it. The floor holds at 0.1 s, and OpenTTGames holds at 1 s.
- Competing explanations: (1) G1d's fresh bins were easy frames chosen by
  G1's keyframing. Then forced ages give lower shares than G1d's bins at
  the same age. (2) A measurement floor: DIS between distant views degrades.
  Then forward-backward error grows with age as fast as the residual does.
  (3) Masks: the stretches' residual is unmasked hands. Then dense masks
  raise the stretch share on the same pairs.
- Budget: CPU only, 0 GPU-hours. OpenTTGames preparation ≤ 20 min. Dev check
  on gpu6 (packed environment, one window and one stretch, a few targets;
  not evidence) to time the pairs and check forward-backward consistency on
  synthetic and real pairs. Smoke ≤ 600 s. Pilot (2 windows, 1 stretch,
  1 OpenTTGames clip, every target) ≤ 20 min. Full run of the rest sized
  from the pilot and split into shards of ≤ 45 min each, checkpointed per
  clip (about 2,100 window pairs, 2,400 stretch pairs and 1,700 OpenTTGames
  pairs; G1d spent about 5.5 CPU-s per frame for eight methods with shared
  keyframes, and here every pair has its own reference). Refinement (0.3 s),
  if reached: one more job of ≤ 45 min. Ceiling 6 CPU-job-hours and 6 h
  wall.
- Before any fleet run (dev check on gpu6, packed environment, `9ae7334`;
  window P02_12 and stretch P26_02, three targets each; not evidence). The
  run works end to end, and the oracle is at or above each of its parts on
  every pair. A VISOR pair takes about 4.5 CPU-s, so the full run is about
  5,500 VISOR pairs (25,000 CPU-s), and fits one job of under 45 min at 48
  threads without shards. On P02_12, at one frame all four warps explain all
  three targets; at 0.1 s `epi`, `planes4` and `tri` do; at 1 s none does
  (`planes4` p90 5–18 px). Forward-backward p90 is 0.1–0.2 px at one frame,
  about 0.3 px at 0.1 s and 1.1–1.9 px at 1 s on the window (`epi`). On the
  stretch it is 0.8–2.5 px at 0.1 s and 3–14 px at 1 s, so step 5 may well declare the
  stretch's 1 s not measurable. `tri` has no reference depth on 7 of the
  stretch's 16 pairs: the relative pose to the companions fails (a near
  rotation), as in G1d. Inputs: `pointstream-data/background/g1e-2026-10-09/`
  (`inputs.json` `3d29aea5…b4e1`). It holds the seven OpenTTGames results and
  masks from G1's jobs `20261008T162927Z-d9372b99` and
  `20261008T203714Z-fd5d2f5b`, and G1d's prepared archive (`5a9c47db…5d62`).
  Another session had moved the fleet directory to
  `jobs/fleet-archive-20261009/`. It was moved back (doctor passed, workers
  restarted), and the recorded paths in `inputs.json` were corrected, which
  is why its sha256 changed from the first build (`643ee3b1…991f`).
- OpenTTGames preparation `20261009T194103Z-e71866e4` (gpu1, CPU, `9b0f8de`;
  smoke gate passed; full 211 s at 16 threads; validator passed, 9 checks):
  7 clips (300–1,200 frames; test_2, test_3, test_5 and test_6 are shorter
  than 120 s), lossless, foreground equal to G1's on every frame.
  `published.tar` `52d46d36…bffc`.
- Pilot `20261009T195037Z-d2a08aa1` (gpu1, CPU, `9b0f8de`; smoke gate
  passed; full 123 s at 32 threads; validator passed, 10 checks, including
  the oracle at or above each part on every pair): windows P02_12 and
  P03_120, stretch P26_02, OpenTTGames test_3. Explained share per clip
  (one frame / 0.1 s / 1 s):

  | Clip | planes4 | epi | tri | oracle | epi forward-backward p90 |
  |---|---|---|---|---|---|
  | window P02_12 | 1.00 / 1.00 / 0.11 | 1.00 / 1.00 / 0.39 | 0.94 / 0.89 / 0.11 | 1.00 / 1.00 / 0.39 | 0.13 / 0.21 / 0.75 px |
  | window P03_120 | 1.00 / 1.00 / 0.26 | 1.00 / 0.95 / 0.63 | 0.95 / 0.79 / 0.21 | 1.00 / 1.00 / 0.68 | 0.22 / 0.39 / 0.66 px |
  | stretch P26_02 (0.1 s = one frame) | 0.40 / 0.08 | 0.41 / 0.08 | 0.11 / 0.00 | 0.49 / 0.10 | 1.20 / 2.60 px |

  On the stretch, 31% of the 0.1 s pairs and 42% of the 1 s pairs do not
  match (fewer than 30 homography inliers), and by the rule they count as
  not explained. On the pairs that do match, `planes4` explains 58%
  at 0.1 s. Mask check on its 30 window-frame targets at 0.1 s: `planes4`
  explains 0.50 with the dense masks and 0.60 with `sam_text`, and the
  oracle 0.57 and 0.77, so the automatic masks do not hold this stretch
  down. OpenTTGames test_3: `h1` explains 97% at both ages, with p90 0.94
  and 1.20 px, so the measure holds on a static camera at 1 s. Nothing in
  the pilot changes the run's settings, so the other 41 clips run as
  specified: `20261009T195921Z-a5d08578`.
- Full run `20261009T195921Z-a5d08578` (gpu5, CPU, `9b0f8de`; smoke gate
  passed; full 903 s at 48 threads, no contention; validator passed, 10
  checks). Refinement at 0.3 s, reached by step 4: `20261009T201916Z-9abe4e90`
  (gpu5, CPU, `e5099d7`, all 51 clips; 441 s; validator passed). Report:
  `pointstream-data/background/g1e-2026-10-09/report-1ae0ded/`
  (`g1e-report.json` `fdd2ede3…8ff1`, from the pilot's, the full run's and
  the refinement's `g1e.json`: `a5727e3f…bd19`, `5a71bc95…a96b` and
  `6174d3d2…7e73`). Median clip's explained share (p90 ≤ 2 px and holes
  ≤ 10%; an unmatched pair counts as not explained):

  | Group | Age | h1 | planes4 | epi | tri | oracle | oracle p90 | epi forward-backward p90 |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | windows (34) | one frame (0.02 s) | 1.00 | 1.00 | 1.00 | 0.68 | 1.00 | 0.43 px | 0.22 px |
  | windows | 0.1 s | 0.45 | **0.78** | 0.84 | 0.46 | **0.92** | 0.93 px | 0.41 px |
  | windows | 0.3 s | 0.11 | 0.36 | 0.44 | 0.11 | **0.54** | 1.80 px | 0.65 px |
  | windows | 1 s | 0.00 | 0.06 | 0.16 | 0.00 | 0.17 | 3.47 px | 1.30 px |
  | stretches (10) | 0.1 s (one frame) | 0.31 | 0.39 | 0.49 | 0.27 | **0.51** | 1.53 px | 0.55 px |
  | stretches | 0.3 s | 0.06 | 0.18 | 0.28 | 0.08 | 0.31 | 2.87 px | 1.08 px |
  | stretches | 1 s | 0.01 | 0.03 | 0.09 | 0.03 | 0.10 | 4.69 px | 1.99 px |
  | OpenTTGames (7), `h1` | 0.1 / 0.3 / 1 s | 1.00 / 1.00 / 0.99 | | | | | 0.84 / 0.97 / 1.03 px | 0.51 / 0.58 / 0.66 px (`h1`) |

  Clips with at least half explained, `planes4` / oracle: windows at 0.1 s
  22 / 27 of 34, stretches at 0.1 s 4 / 6 of 10. Per clip at 0.1 s,
  `planes4` explains 0–100% of a window's targets (11 windows all of them)
  and 16–88% of a stretch's. The stretches' pairs mostly match (median
  clip 95% at 0.1 s, 75% at 1 s). Their shortfall is residual on matched
  pairs: `planes4` explains 49% of them and the oracle 65%. `epi` alone
  gives the oracle's verdict on 95–100% of a median clip's pairs, so the
  best-of construction mattered little here. `planes4` costs
  128 bytes per frame plus a label map per reference.
- Outcome by the rule: **step 3**.
  - **Refresh ages.** The oracle's are 0.3 s on the windows and 0.1 s on
    the stretches. `planes4`'s is 0.1 s on the windows (78%) and none on
    the stretches (39% at 0.1 s). `tri`'s is one frame on the windows, and
    none on the stretches.
  - **No sendable warp qualifies in both groups**, so G2 does not yet
    price a refresh scheme. The rule's next step is a sendable warp
    between `planes4` and the oracle, at the oracle's refresh age (DA3 or
    depth-augmented keyframes), with its own entry. Its ceiling is the
    oracle's: on the stretches, 51% at 0.1 s, only just above the bar, and
    31% at 0.3 s.
  - **Floor (step 5).** It holds at every age: OpenTTGames' `h1` p90 is
    1.03 px at 1 s, and VISOR's forward-backward p90 is at most 1.99 px.
    The growth with age is therefore real. The residual grows from 0.93
    to 3.47 px on the windows between 0.1 s and 1 s, while the flow's
    self-disagreement grows from 0.41 to 1.30 px.
  - **Mask check.** It holds: dense and `sam_text` masks give the same
    shares within 2 points, so the stretches' gap to the windows is
    content, not masks.
- What the remaining residual is made of (`epi`, energy share of the raw
  squared error, mean over pairs). On the windows at 0.1 s: remainder 70%,
  independent motion 21%, parallax 7%, exposure 2%. At 1 s the shares are
  46%, 34%, 13% and 7%. On the stretches at 0.1 s: 63%, 27%, 7% and 2%.
  Misaligned pixels near the foreground: 5–7%. A fresh reference removes the
  geometric part almost entirely. What stays is photometric (blur, noise,
  lighting, disocclusion) and things that move by themselves, which no
  static-scene warp can remove.
- Hypothesis: partly right. The oracle was meaningful at one frame and at
  0.1 s on the windows, and not at 1 s, as predicted. It was stronger than
  predicted (92% at 0.1 s, against 65–75%), and so was `planes4` (78%,
  against 40–55%). The stretches stayed below the windows, as predicted.
  The mask check closed none of the gap, where part was predicted. The
  floor held, as predicted.
- Competing explanations. (1) Selection by G1's keyframing: rejected. Forced
  ages give *higher* shares than G1d's age bins (window bin under 0.1 s:
  `epi` 70%, four planes 49%). (2) A measurement floor: rejected. The flow's
  self-disagreement stays under 2 px and grows a third as fast as the
  residual, and OpenTTGames holds at 1 s. (3) Masks: rejected (above).
- Budget: CPU only, 0 GPU-hours. CPU jobs: preparation 211 s, pilot 123 s,
  full 903 s, refinement 441 s, so about 28 min of CPU-job wall (ceiling 6
  CPU-job-hours). Wall time from the first submission to the report: about
  50 min (ceiling 6 h).
- After review (user, 2026-10-09): the rule's step-3 component is *not*
  built. Its ceiling on the stretches is the oracle's 51% at 0.1 s, a
  narrow pass at best, for a reference refreshed ten times a second.
  Egocentric background work moves to G5. The parked options and what
  would reopen them are in PLAN (G1e).

### 2026-10-09 — G5 oracle: the best a neural background model could do on VISOR
- Question (PLAN G5, after G1e): could a neural background model beat
  SVT-AV1 at equal rate on VISOR's visible background? G5's candidates store
  the background in a per-scene model's weights and send one latent per
  frame. G1e adds a second one: a model conditioned on a reference that is
  refreshed every 0.1 s and warped to the frame, which then only has to
  code what G1e found left after a fresh reference (photometric change and
  independent motion). Training a G5 model is gated on this oracle.
- Clips. G1d's prepared archive (`20261009T075810Z-295529d4`,
  `published.tar` `5a9c47db…5d62`; 960×540, lossless). *Windows*: all 34,
  every frame (240 frames, 4.0 s at 59.94 or 4.8 s at 50 frames/s).
  *Stretches*: the 10 of G1d, each cut to a 24 s excerpt (240 frames at
  10 frames/s) that starts at the stretch's first evaluation-window frame,
  moved earlier to fit. This rule is fixed before any run and is blind to
  content. It keeps the window's frames, which carry VISOR's dense masks,
  inside the excerpt. A whole 120 s stretch would cost five times the GPU
  time, and a model has to cover new background as the clip goes on (G1,
  G1d), so its bytes grow with length too.
- Operating point: 960×540, as stored. Every method codes the same frames
  and is scored against the same RGB frames. This is G1's analysis
  resolution, not VISOR's 1080p. A pass here would be checked at 1080p
  before anything is built on it.
- Protocol, from G2's (branch `claude/pointstream-g2-background-protocol`
  at `c99aaab`; the G2 session could not be reached, so its text is taken as
  written and adapted to VISOR as G2 anticipates). *Rate* over the clip's
  length: every byte a method sends for the clip, divided by its duration.
  Egocentric video has no warm-up that pays off (G1, G1d), so every method
  codes the clip from its first frame, and the baselines pay for their
  intra frame. *Quality* on visible background V, the complement of the
  dilated foreground, as G1d stored it. On the windows that is VISOR's
  dense masks with B1b's fill. On the stretches it is G1's `sam_text`
  foreground, plus VISOR's dense masks on the window frames (G1e's
  `window_dense`, dilated as in G1). PSNR_V comes from exact
  squared-error sums in RGB. LPIPS (AlexNet) is averaged over V, with F
  pasted back from the source (G2). A clip's score is the mean over its
  scored frames. On the windows all 240 frames are scored, against the
  dataset masks. On the stretches the window frames are scored against
  the dataset masks, which the gate reads. All 240 frames are also scored
  against `sam_text`, as a check. Clips are the samples. *Render time* per
  frame at the client: NVRC's decode from its bitstream (model forward),
  B's sequential rollout including the warp, and dav1d for SVT-AV1, each
  on one host class.
- Baselines (CPU), on the same frames: SVT-AV1 4.2.0 preset 4, random
  access, one keyframe (`--keyint -1`), CRF 32, 40, 48, 54, 59 and 63, 4:2:0
  8-bit full range (BT.709), on two inputs. *Frame* is the source as is.
  *Filled* has the foreground inpainted per frame (OpenCV Telea, radius 5),
  which is smooth and cheap to code. The gate's baseline at each rate is
  the better of the two inputs: the upper envelope of both inputs' points.
  The RGB→4:2:0→RGB round trip of the source is scored as the baselines'
  ceiling (the oracles output RGB directly). DCVC-UF HT-L (B2's
  configuration, QP 9, 27, 45 and 63) runs on the pilot clips' filled input
  for the comparison, not for the gate.
- **Arm A, per-scene implicit model: NVRC** (Kwan et al., NeurIPS 2024,
  `hmkx/NVRC` @ `ccc432d`, MIT). It is the strongest public implicit
  video codec: a HiNeRV-v2 representation trained end to end for rate and
  distortion, with entropy-coded weights and grids written to a real rANS
  bitstream. Config: HiNeRV-v2 at the size of NVRC's smallest model,
  channels 224/112/56/28. Upsampling 5·3·2·2 (540 and 960 divide only by
  multiples of 60) and a base grid of 80×9×16×6, its value count scaled
  from the 1080p configuration by pixels times frames. NVRC's own schedule:
  360 epochs, then 30 with rate fine-tuning (`s1-360e`, `s2-30e`), with
  180×120 patches in batches of 96 (2.1 Mpx per step, as its 120×120 × 144
  at 1080p). The loss is mean squared error on V only. A fourth PNG
  channel marks V, and F takes the output's own value, so it adds no
  error (`env/patches/nvrc-pointstream.patch`). Rate = the bitstream's
  bytes. Quality is scored on the frames decoded from that bitstream.
  Three settings, λ = 50, 200 and 800 (×4 apart). Dev check: SVT-AV1
  spends about 0.01–0.02 bits per pixel here, and λ in that range targets
  it.
- **Arm B, model conditioned on a refreshed warped reference** (written for
  G5: `experiments/background/g5.py`, PyTorch only). The reference is the
  model's own output at the last refresh point before the frame. Refresh
  points are every 0.1 s, so on the windows a frame's reference is 1–6 frames
  old, and on the stretches (10 frames/s) it is the previous frame. The
  reference is warped by the dense DIS flow from the source frame to the
  source reference, and priced as `planes4`, 128 bytes per frame. A
  sendable warp can only be worse (G1e: `epi`, the dense flow on epipolar
  lines, matches the oracle on 95–100% of pairs). A decoder upsamples a
  per-frame latent (45×80×c, c = 4) by 3·2·2, with features of the warped
  reference concatenated at every scale, and blends a synthesised image
  with the warped reference through a predicted weight. The first frame
  has no reference. Its widths (32, 24, 16, 12 at λ = 200 on 240 frames;
  about 110k parameters) scale with the square root of the frame count and
  the fourth root of λ, because its weights are priced over the clip's
  frames and so belong to the rate point. Latents are fitted directly (an auto-decoder, which
  bounds any encoder for the same decoder) under uniform-noise
  quantization and a learned factorized Laplace prior per channel. Rate =
  latents (their ideal code length under that prior, plus the prior at 32
  bits per value) + weights (8-bit uniform quantization per tensor, priced
  at its empirical entropy plus a 32-bit scale per tensor; scored with the
  quantized weights) + motion. Training runs closed loop, as the decoder
  runs: refresh groups go in time order every epoch, with the current
  outputs as references (detached), for 300 epochs. The loss is λ·MSE on V
  plus rate, with λ = 50, 200 and 800 (rate per pixel, as in NVRC).
  Evaluation is a fresh sequential rollout with the quantized latents and
  weights.
- Why these are bounds, and what each bounds. Both arms are fitted to the
  very frames they are scored on. A G5 model would be fitted to a warm-up
  and then render later frames, while G1 and G1d show that egocentric
  background keeps changing. Both see a loss only on V, so they need no
  clean frames (G4) and spend nothing on the foreground. Neither pays for
  a foreground fill, which the filled baseline does. Both output 4:4:4 RGB.
  - A bounds G5's per-scene model class (HNeRV-style "weights once, one
    latent per view"): it is that class's strongest published member,
    with its own rate-distortion training and a real bitstream. It does
    not bound a future implicit architecture that beats NVRC.
  - B bounds a reference-conditioned model with a sendable warp and a
    0.1 s refresh. Its motion is the dense flow of the source frames,
    priced as the cheapest sendable warp, and it is fitted to the scored
    frames. It does not bound other conditional architectures. DCVC-UF
    given a render as its reference is G2's residual arm (b), tested
    under G3.
  - DCVC-UF fine-tuning (a PLAN G5 candidate) is not gated here. Its
    starting point is the DCVC-UF baseline above.
  - *Convergence check*: NVRC's log of rate and PSNR at every 30-epoch
    evaluation, and B's every 30 epochs. A run whose last interval still
    gains more than 0.3 dB at equal rate (or saves more than 10% rate at
    equal PSNR) is flagged as not converged. A loss from a flagged run does
    not count toward a verdict of "ruled out".
- Decision rule (`g5.DECISION`, fixed before any fleet run). Per clip and
  arm, BD-rate on PSNR_V (Bjøntegaard, piecewise cubic in log rate,
  over the overlapping quality range) against the baseline envelope. With
  less than 0.5 dB of overlap, the arm is *dominated* if every arm point
  lies below the envelope (or below its lowest-rate point, at a higher
  rate), and dominated counts as +∞. Otherwise the clip is *unclear*, and
  one more λ is added toward the overlap (protocol step 4).
  1. *Pilot* (G1d's pilot clips: windows P02_12 and P03_120, stretch
     P26_02; three λ per arm). Per arm:
     - *Ruled out*: BD-rate ≥ +100% (or dominated) on all three pilot
       clips, from converged runs. The arm is not scaled, and its G5
       component is not trained. Why three clips can decide: two codecs'
       per-item BD-rates on VISOR spread from −55% to +77% around their
       median (B2). A median below 0 over 44 clips is implausible when
       three clips all sit at +100% or worse, against a generous oracle.
     - *Candidate*: BD-rate < 0 on at least one pilot clip, or between 0
       and +100% on any. The arm's oracle runs on the other 41 clips
       ("the rest"). Its cost, about 0.7 GPU-hours per clip and setting
       for A, is beyond this entry's budget, so it needs the user's
       authorization first.
  2. *Gate* (after the rest): per arm and group (windows, stretches), the
     median clip's BD-rate < 0. Only then is the arm's real G5 component
     built: fitted on warm-up frames and rendering later frames, with a
     smoke and a pilot of its own. An arm that passes in one group only
     scopes G5 to that group.
  3. LPIPS on V, render time and DCVC-UF are reported beside the gate. They
     do not change it.
- Hypothesis. A, NVRC, is ruled out on the windows. A 4 s clip leaves its
  weights nothing to amortize over, and its published advantage is on
  600-frame sequences of slower content. Every window will need more than
  twice SVT-AV1's rate. The stretch excerpt (24 s at 10 frames/s) does
  better, but still above +50%. B is a candidate on the windows (between
  0 and +100%), because the reference carries most of each frame. On the
  stretch it is worse: one frame is 0.1 s there, and G1e explained only
  51% of stretch pairs at that age.
- Competing explanation. If the arms lose, the cause could be their
  training rather than the class: too few iterations for 240 frames (the
  convergence check tests this), the 540p operating point, or drift in B's
  closed loop (its rollout is reported against the training loss). If an
  arm wins, the cause could be its generosity rather than a real model's
  ability: fitting to the scored frames, the loss on V only, oracle
  motion, 4:4:4 output. That is why a pass leads to the real component's
  own smoke and pilot, not to its adoption.
- Environment: NVRC is audited into the one environment
  ([resources](resources.md#environments)): `compressai` 1.2.8,
  `accelerate` 1.3.0, `pytorch-msssim` 1.0.0 and `compressai`'s
  `torch-geometric` 2.8.0.post1 (with `pandas` 3.0.6 and `xxhash` 4.0.1),
  installed without dependencies over torch 2.10.0+cu128, plus NVRC
  vendored in `opt/NVRC` with the patch. Dev checks on gpu6 (RTX 6000 Ada,
  not evidence): NVRC trains P02_12 at 31 frames/s with a 5.2M-parameter
  model (fp16, Inductor), which puts its schedule at about 50 min per run.
  It writes and decodes its bitstream (49.9 Mbit after 6 of 390 epochs,
  at 22.8 dB), and decodes at about 120 frames/s. On P02_12, SVT-AV1 gives
  35.2 dB (whole frame) at 317 kbps (CRF 60) and 37.5 dB at 690 kbps
  (CRF 50).
- Budget. Baselines: one CPU job of ≤ 45 min (44 clips × 2 inputs × 6 CRF),
  plus DCVC-UF on the 3 pilot clips (≤ 20 min, one GPU). Pilot: 9 A runs
  and 9 B runs, one GPU each, every job ≤ 45 min. Ceiling 12 GPU-hours and
  6 h wall for this entry. The rest is outside this ceiling.
- Before any fleet run (dev checks on gpu6, RTX 6000 Ada, packed
  environment `pointstream-20261009T214002Z`; not evidence). The three
  commands run end to end on the pilot clips, and their validators pass.
  - *Baselines.* On 48 frames, P26_02's excerpt starts at its first frame
    and holds 40 dense-mask frames. The 4:2:0 round trip of the source
    scores 49–52 dB on V.
  - *A's timing.* The 2.26M-parameter model trains at 40 frames/s compiled
    (Inductor) and 25 frames/s eager. Every compile costs about 5 min per
    stage. One run (240 frames × 390 epochs) is therefore about 48 min
    compiled and 62 min eager, beyond the 45-minute limit. So each A run is
    two jobs. *Part 1* is stage 1 (360 epochs, compiled, about 41 min),
    on RTX 6000 Ada only: an A6000 is slower and would overrun. It
    publishes its checkpoint and its training log. *Part 2* is stage 2
    (30 epochs, eager: they do not repay a compile), the bitstream, its
    decode and the scores, on Ada or A6000, from part 1's checkpoint.
    Smokes: part 1 on 48 frames for 2 epochs (about 6 min, mostly
    compiling); part 2 on the full 240 frames from part 1's checkpoint for
    1 epoch.
  - *B's timing.* About 6 s per epoch on a stretch (240 refresh groups of
    one frame), so 300 epochs take about 30 min. Windows (40–48 groups) are
    faster. Its first dev size scaled with duration and gave the stretch
    560k parameters, 160 kbps of weights alone. The size now scales with
    the frame count (above).
- Jobs. Baselines (CPU, all 44 clips): `20261009T221753Z-64ab816f`. DCVC-UF
  (pilot clips): `20261009T221918Z-c483803d`. A, part 1:
  `20261009T221725Z-c979689a` failed its smoke gate, but only in the
  validator: it kept only the log line of an epoch's last step, which NVRC
  does not always log. The run itself was sound (406 s, checkpoint
  published), and the fixed parser reads its curve. Resubmitted with the
  fix (`6ed4410`) as one job per clip and λ.
- **Arm B, first version: not a bound** (9 pilot jobs, `3c15191`, all
  complete, validators passed; for example `20261009T221736Z-7cb43081`).
  As built, its references were only its own outputs, and nothing ever sent
  a refreshed reference. The first frame came from the 45×80×4 latent
  alone, and every later frame inherited it through the warp. It reached
  15–18 dB PSNR_V on the windows at 185–886 kbps, and 22–25 dB on the
  stretch at 46–158 kbps, flat over its last 150 epochs. DCVC-UF reaches
  31 dB at 86 kbps on P02_12. This measures a model that has to code a
  whole frame from a 1/12-resolution latent and a 60–200k-parameter
  decoder. It does not measure a reference refreshed every 0.1 s, so these
  runs do not count toward the gate.
- **Arm B, as G1e meant it** (before any of its runs). The refreshes are
  sent. Every 0.1 s a frame (every 5th at 50 frames/s, every 6th at 59.94)
  is coded by SVT-AV1 (preset 4, its CRF the arm's setting) as its own
  stream at a tenth of a second per frame, and its bytes count. Each frame
  between refreshes is its refresh's decoded image, warped by the oracle
  flow (128 bytes per frame) and corrected by the per-clip model above
  (latents and weights priced). It is trained on the frames between
  refreshes only, against fixed references, so no error carries over. Two
  variants per setting: *warp* (no model: refreshes plus motion only) and
  *cond* (with the model). Settings, three pairs of refresh CRF and model
  λ, matched so that each refresh quality gets a correction of similar
  slope: (59, 200), (50, 800), (40, 3200). It runs on the windows only. On
  the stretches (10 frames/s) every frame is a refresh, so the arm *is*
  SVT-AV1 at 10 frames/s, which the baseline already is. It cannot beat
  the baseline there, by construction. Why it bounds the class: the refresh
  stream is a real codec's, the motion is oracle motion at the sendable
  price, and the model is fitted to the scored frames with a loss on V.
  The bound is for this correction model; a stronger conditional coder is
  DCVC-UF given the warped refresh (G2's arm (b)), not tested here.
- Runs so far (`33cd300` and later; environment `pointstream-20261009T214002Z`;
  every validator passed unless said).
  - *Baselines.* `20261009T221753Z-64ab816f` (CPU, all 44 clips) ran past its
    2,700 s allowance after 14 windows (about 3.2 min per clip, mostly
    inpainting and scoring). Its per-clip checkpoint kept them; they are
    read from its `partial.tar` (`report` accepts one clip's
    `baselines.json`). The other 30 clips ran in four CPU shards by explicit
    clip list: `20261009T230937Z-17f60d9f`, `20261009T230943Z-a6010e7f`,
    `20261009T230949Z-be5deada` and `20261009T230954Z-4c28fc16`. CPU jobs
    cannot declare a resume, so the shards replace one.
  - *DCVC-UF* `20261009T221918Z-c483803d` (gpu6, Ada; deterministic decode,
    intra matches). This job processes 3 clips without a per-clip
    checkpoint, against the fleet rule; it ran 10 min.
  - *Arm B* (`33cd300`; P02_12 and P03_120, three settings each):
    `20261009T230416Z-0e6c0458`, `…230421Z-60b64028`, `…230427Z-c83b3a52`,
    `…230433Z-991477d7`, `…230438Z-39ed6c81` and `…230443Z-475686e3` (gpu2
    A6000 and gpu6 Ada, 4–7 min each). On P02_12, against SVT-AV1 on the
    filled input (127, 271 and 425 kbps for 31.0, 33.6 and 34.9 dB):

    | Refresh CRF, λ | warp (kbps → PSNR_V) | cond (kbps → PSNR_V) | cond's weights, latents |
    |---|---|---|---|
    | 59, 200 | 191 → 31.03 | 369 → 31.02 | 172, 6 kbps |
    | 50, 800 | 310 → 32.64 | 664 → 32.63 | 301, 53 kbps |
    | 40, 3200 | 473 → 33.58 | 1,166 → 33.61 | 552, 141 kbps |

    The model never helps: at most 0.03 dB for 180–690 kbps. Its curve is
    flat from epoch 30, and every run passes the convergence check. What a
    warped 0.1 s refresh leaves (G1e: photometric change and independent
    motion) is not low-frequency enough for a 1/12-resolution latent to
    correct cheaply, and the per-clip weights alone cost more than SVT-AV1's
    whole stream at that quality.
  - *Arm A, part 1.* P02_12 at λ = 200: `20261009T222809Z-f3873f4a` (gpu6
    Ada, 2,322 s, converged: +0.06 dB over its last 30 epochs). Part 2:
    `20261009T235421Z-d60d9f80` (gpu2 A6000, 571 s): **2,129 kbps for
    35.45 dB PSNR_V** (LPIPS 0.072), decoded from its 1,065,336-byte
    bitstream; render 25.8 ms per frame (A6000, PNG writing included).
    SVT-AV1 needs 594 kbps for 35.65 dB, so NVRC needs about 3.6 times the
    rate. Only one RTX 6000 Ada was free (the others hold other users'
    processes), so the 7 pending part-1 jobs were cancelled before they
    started and resubmitted with the decision's order first (λ = 200 on
    P03_120 and P26_02, then λ = 50, then λ = 800) and a 12:00 deadline.
    The wall time thus exceeds this entry's 6 h ceiling. GPU-hours stay
    within it.
  - *Arm A, the rest of the pilot.* Part 1 / part 2: P02_12 λ = 50
    `20261009T222831Z-c990cab8` / `20261010T004119Z-ea3c26a3`; P03_120
    λ = 200 `20261010T001154Z-e839e37b` / `20261010T012618Z-240f9af5`;
    P03_120 λ = 50 `20261010T001209Z-afcabd94` / `20261010T030026Z-2fc38d62`;
    P26_02 λ = 200 `20261010T001201Z-d7ddb776` / `20261010T021325Z-d383710e`;
    P26_02 λ = 50 `20261010T001216Z-b695080f` / `20261010T034511Z-482b5d1b`.
    Part 1 always ran on gpu6 (Ada), 2,320–2,360 s; part 2 on gpu2 (A6000).
    Every run passes the convergence check (+0.04 to +0.07 dB over its last
    30 epochs). The three λ = 800 runs were cancelled before they started:
    their points lie above SVT-AV1's quality range, and they could not
    change the verdict below.
- Report `pointstream-data/background/g5-2026-10-09/report-a9d3806/`
  (`g5-report.json` `4b1ae6b8…047f`, from 31 result files: the 14 clips of
  the stopped baseline job, copied beside it, the 4 shards, DCVC-UF, the 6
  arm B runs and the 6 arm A runs). BD-rate on PSNR_V against the
  baseline envelope (positive: more rate for equal quality):

  | Clip | A, NVRC | B, cond | B, warp only | DCVC-UF (comparison) |
  |---|---:|---:|---:|---:|
  | window P02_12 | +347% | +184% | +34% | −17% |
  | window P03_120 | **+2.7%** | +409% | +103% | +27% |
  | stretch P26_02 | +133% | — (is SVT-AV1) | — | +41% |

  NVRC's points (kbps → dB PSNR_V): P02_12 941 → 31.93 and 2,129 → 35.45;
  P03_120 736 → 36.50 and 1,497 → 37.90; P26_02 108 → 37.22 and 240 →
  39.69. On P03_120 the overlap with SVT-AV1 is only 36.5–37.6 dB, at
  0.7–1.5 Mbps. Below that NVRC has no points: its lowest λ already costs
  736 kbps there, where SVT-AV1 gives 32.8 dB for 74 kbps.
- Outcome by the rule.
  - **Arm B is ruled out**: +184% and +409% on the windows, from converged
    runs. On the stretches it is SVT-AV1 by construction. Its correction
    model never helps, and even with no model (warp only) refreshing every
    0.1 s with oracle motion costs +34% and +103%. That also answers G1e's
    parked options: a refreshed reference with any sendable warp loses to
    SVT-AV1 on these windows.
  - **Arm A is a candidate** (step 1): P03_120 lies between 0 and +100%.
    The other two clips are well beyond +100%, and the median pilot clip is
    +133%. By the rule, the next step is A's oracle on the other 41 clips,
    at about 0.85 GPU-hours per clip and λ (two λ: about 70 GPU-hours), and
    that needs the user's authorization. Nothing of G5's component is
    trained until then.
  - *Render time* per frame at 960×540: dav1d median 1.05 ms (CPU, loaded
    host), DCVC-UF 2.5–3.5 ms (Ada), arm B 2.6–3.0 ms (A6000), NVRC 25.8 ms
    (A6000, PNG writing included).
  - LPIPS on V was scored for the GPU arms and DCVC-UF but not for SVT-AV1:
    its job ran on CPU without the LPIPS network. It does not enter the
    rule.
- Hypothesis: mostly right. A was not ruled out on every window (P03_120
  is at parity in its high-quality overlap), but it was far beyond +100%
  on P02_12 and on the stretch, as predicted. B was predicted to be a
  candidate on the windows; it was ruled out, because its per-clip
  correction model buys nothing and its weights cost more than SVT-AV1's
  whole stream.
- Competing explanations. *Training*: rejected for both arms. Every run is
  converged by the rule's check, and arm A used NVRC's own schedule.
  *Generosity*: A's parity on P03_120 is despite being fitted to the scored
  frames with a loss on V only. A real G5 model, fitted to a warm-up, can
  only do worse. *Operating point*: everything is at 960×540, and the clips
  are short (4–4.8 s windows; 24 s stretch excerpts).
- Budget as run: about 6.8 GPU-hours (A part 1 6 × 0.65, part 2 6 × 0.16,
  B 15 × 0.1 including the first version, DCVC-UF 0.2, failed smoke 0.1)
  and 5 CPU jobs (about 2.6 h of job wall). Wall time from the first
  submission to the report: about 4 h, so within the 6 h ceiling after all.
- After review (user, 2026-10-10). The scene model is delivered beforehand
  and not billed: the rate is what is sent at run time. That needs a
  held-out segment, and NVRC cannot render frames outside its fitted range,
  so NVRC's oracle on the other 41 clips is dropped. Its pilot stays as the
  paper's intermediate step (a fully billed per-clip neural codec). The
  gate metric becomes LPIPS on V, and training may optimize it directly;
  PSNR on V is reported. Long clips come first. The next step is PLAN G5b,
  DCVC-UF fine-tuned on the scene, with its own entry.

### 2026-10-10 — G5b: scene-adapted DCVC-UF on VISOR's stretches
- Question (PLAN G5b): with the scene model delivered beforehand and not
  billed, does DCVC-UF fine-tuned on the scene code a held-out stretch of the
  same scene for fewer bits than SVT-AV1 (and than DCVC-UF unadapted) at
  equal LPIPS on the visible background?
- Held-out segment. On each of G1d's 10 stretches (120 s at 10 frames/s;
  P03_10 103 s) it is G5's 240-frame excerpt (24 s), not 96–120 s as PLAN
  first said: VISOR's dense masks lie in 96–120 s on only one stretch
  (P09_106), and the gate needs them. The excerpt holds the stretch's
  dense-mask window (40–48 frames), and G5's SVT-AV1 streams and DCVC-UF
  setup already code exactly these frames. *Scene frames*: the rest of the
  stretch, less 5 s on each side of the excerpt (no near-duplicate of a
  held-out frame is fitted); 910 frames on most stretches. On 4 stretches
  they lie on both sides of the excerpt; that is the deployment's
  assumption (a model of the scene, made from other footage of it), not
  causal coding.
- Protocol. As G5 (`experiments/background/g5.py`): 960×540, every method
  codes the excerpt's *filled* input (foreground inpainted) or, for SVT-AV1,
  the better of frame and filled; quality on V with the foreground pasted
  back; rate = every byte sent for the excerpt over its 24 s. The fine-tuned
  model is not billed; its fp16 size is reported, with the playback after
  which it would be repaid at each operating point (8 × bytes / rate saved
  against SVT-AV1 at equal LPIPS_V).
- **Gate metric**: LPIPS (AlexNet) on V of the dataset tier (the dense
  frames), mean over frames; BD-rate is G5's (PCHIP, Pareto envelopes) on
  q = −100 × LPIPS_V, so G5's 0.5 overlap is 0.005 LPIPS. PSNR_V on the
  same frames, and LPIPS_V on all 240 frames against G1's `sam_text`
  masks, are reported beside it.
- Arms. *Anchor*: SVT-AV1 (G5's streams, decoded again and scored with
  LPIPS). *Reference*: DCVC-UF HT-L off the shelf, QP 9/27/45/63 (G5's
  `dcvc`). *Upper bound*: HT-L fine-tuned on the excerpt itself. *Component*:
  fine-tuned on the scene frames. *Control*: fine-tuned on another
  stretch's scene frames (P26_02 ↔ P06_03), to tell the scene from
  egocentric footage, the mask or the loss. Fine-tuning
  (`experiments/background/g5b_train.py`, in DCVC's tree): HT-L stage 2's
  recipe (cascaded, 33 frames = I + 4 groups of 8, 512×512 patches, batch
  4, AdamW at 1e-5, a random QP per sample with DCVC's λ from 1 to 768, the
  image model frozen), distortion on V only with DCVC's YUV/RGB weighting,
  optionally plus β·LPIPS_V. β is set once from the smoke so that β times
  the monitor set's mean LPIPS_V equals twice its mean masked distortion
  at step 0. Steps are set from the smoke so that training takes at most
  30 minutes on Ada. A fixed monitor set (from the fitted frames only,
  4 QPs) gives the curve.
- Decision rule (`g5b.DECISION`), on the pilot stretches P26_02 (G5's pilot
  stretch) and P06_03 (the best `sam_text` recall, 0.84). An arm *passes*
  if on some pilot stretch, from a converged run (monitor loss down less than
  3% over the last fifth of the steps), it beats SVT-AV1 (BD < 0) and the
  unadapted DCVC-UF (BD < −10%). It *fails* if on both it does neither.
  Otherwise it is *unclear* (more steps, once). Order: (1) the upper bound,
  β = 0 and β > 0. If it fails, the component is not built and G5b stops.
  (2) The component and the control, with the better β. If the component
  passes and the control reaches at least 2/3 of its gain against the
  unadapted model, the gain is not the scene's, and G5b reports an
  egocentric-adapted DCVC-UF instead. (3) The component on the other 8
  stretches: the gate passes if the median stretch's BD-rate against
  SVT-AV1 is below 0.
- Hypothesis: the upper bound passes with large margins (BD −40% or better
  against the unadapted model), because the decoder and the entropy model
  absorb the excerpt's textures. The component keeps a third or less of
  that gain: the excerpt's background is mostly seen elsewhere in the
  stretch, but lighting, clutter and viewpoint differ. Against SVT-AV1 on
  LPIPS the component is near parity: DCVC-UF off the shelf was +41% on
  PSNR_V on P26_02, and DCVC-UF's LPIPS is usually better than its PSNR
  rank. β > 0 helps LPIPS by 10–30% at a small PSNR cost.
- Competing explanation: whatever the component gains comes from adapting
  to egocentric kitchen footage at 960×540, to the filled input or to the
  loss on V, not from knowing the scene. The control measures that. A
  second one: LPIPS gains with β > 0 come from texture that LPIPS rewards
  but that is wrong, which PSNR_V beside it and a visual check expose.
- Side check (PLAN G5b, cheap): G5's warp-only arm (refreshes coded by
  SVT-AV1, oracle DIS flow at 128 B per frame between) with refreshes every
  0.3 s and 1 s on G5's pilot windows (P02_12, P03_120), CRF 40/50/59.
  If either is below 0 BD on PSNR_V or LPIPS_V against SVT-AV1 on either
  window, G1e's refreshed reference reopens as conditioning for G5b;
  otherwise it stays closed. Hypothesis: both stay above +30%, since
  SVT-AV1's inter coding already uses what a refresh carries, and the warp
  error grows with the reference's age (G1e).
- Budget. Rescore ≤ 1 GPU-hour (one job, all 44 clips). Reference ≤ 0.5
  (one job). Smoke ≤ 10 minutes. Pilot ≤ 6 GPU-hours: the upper bound
  (2 stretches × 2 β) and the component and control (2 + 2), each run
  ≤ 45 minutes on Ada. The rest ≤ 6 GPU-hours (8 runs). The warp check ≤
  0.5 (two jobs). Ceiling 14 GPU-hours, stage by stage; a stage that
  fails the rule stops what follows.
- Inputs: as G5 (`pointstream-data/background/g5-2026-10-09/inputs.json`
  `fe56577d…baecf`), the DCVC-UF checkpoints (image `b3b900de…3302`, HT-L
  `934bde4a…b1a6b3`, the latter 482 MB at fp32), and
  G5's SVT-AV1 streams: `20261009T221753Z-64ab816f` `partial.tar`
  (`e8b27b50…245b`, 14 windows), `20261009T230937Z-17f60d9f`
  (`825625d6…9598`), `…230943Z-a6010e7f` (`e317bd7c…1244`),
  `…230949Z-be5deada` (`8523c9b4…6bc82`), `…230954Z-4c28fc16`
  (`ba0b804f…85b87`). Environment `pointstream-20261009T214002Z`
  (`32e68872…4661`): DCVC-UF's training code needs nothing new
  ([resources](resources.md), DCVC-UF).
- Calibration (before the pilot). Smoke and a 40-step run at the pilot's
  settings, upper bound on P26_02, β = 0 (`20261010T062353Z-1ed3e72b`,
  Ada, code `60ad72e`): HT-L has 120.5 M parameters (241 MB at fp16); the
  fine-tuned checkpoint loads strictly into B2's worker, decodes
  deterministically and matches the encoder's intra frame. 3.28 s per step,
  42.7 GiB peak. At step 0 the monitor set's masked distortion is 0.00264
  and its LPIPS_V 0.2034, so **β = 0.026**; 30 minutes of training is
  **540 steps**. Infrastructure, not evidence. Warp check smokes and runs:
  `20261010T062342Z-5d5103f4` (P02_12), `…062347Z-917cfaaa` (P03_120).
- Anchors and reference (code `60ad72e`). SVT-AV1 rescored with LPIPS:
  `20261010T062332Z-92f06774` (exceeded its 2,700 s after 23 windows,
  kept in its `partial.tar` `4d18c877…a927`), `…071155Z-7cdba7a4` (the
  10 stretches), `…071200Z-882d6d58` (the other 11 windows). DCVC-UF off
  the shelf on the 10 stretches: `20261010T062337Z-78a7cdcc`. Against
  SVT-AV1 it is worse on LPIPS_V than on PSNR_V: median BD +151% on
  LPIPS_V (range +26% to +545%), against about +100% on PSNR_V. The
  hypothesis that DCVC-UF ranks better on LPIPS was wrong.
- Warp check, outcome: **closed**. Against SVT-AV1 on LPIPS_V: 0.3 s
  +52% (P02_12) and +51% (P03_120); 1 s +42% and +54%. On PSNR_V: 0.3 s
  +34% and dominated; at 1 s, unclear (the cheapest points lie below
  SVT-AV1's cheapest rate, and every point in its range lies below it).
  As predicted.
- Upper bound, outcome (540 steps, all converged by the rule; Ada).
  β = 0: P26_02 `20261010T063842Z-29a4262e`, P06_03 `…063853Z-c1d2154d`.
  β = 0.026: P26_02 `…063847Z-3990302d`, P06_03 `…063858Z-93923e53`.
  BD-rate on LPIPS_V (PSNR_V) against SVT-AV1 | against DCVC-UF:
  - β = 0, P26_02: +37% (−1%) | −37% (−33%).
  - β = 0, P06_03: +11% (+11%) | −36% (−31%).
  - β = 0.026, P26_02: unclear (+82%) | dominates (+15%). Its three
    points inside SVT-AV1's rate range all lie far below SVT-AV1's best
    LPIPS_V (0.063 at 31 kbps against 0.132 at 264 kbps); the cheapest
    (8.7 kbps) lies below SVT-AV1's range, which G5's verdict calls unclear.
  - β = 0.026, P06_03: −83% (+72%) | −92% (+7%).
  By the rule the upper bound **passes** (β = 0.026 on P06_03); β = 0
  alone would be unclear (it beats the reference on both stretches,
  SVT-AV1 on neither). Scene adaptation buys a steady third of the rate
  against the unadapted model on both metrics. β > 0 trades PSNR_V for a
  large LPIPS_V gain, as the second competing explanation (texture that
  LPIPS rewards) would also produce; the visual check decides that, from
  decoded frames that the component's jobs now publish. So the component
  runs at both β (beyond the rule's "better β", within the pilot budget),
  and the control at β = 0.026. Pilot GPU time so far: about 2.3 hours.
- Component and control, outcome (540 steps, all converged by the rule;
  Ada; code `fb39288`). Scene frames, β = 0: P26_02
  `20261010T080746Z-23963f61`, P06_03 `…080758Z-032b73ee`; β = 0.026:
  `…080752Z-37856ad4`, `…080803Z-38efc9c3`. Control (β = 0.026, fitted
  on the other pilot stretch's scene frames): P26_02 from P06_03
  `…080809Z-e1acbcd8`, P06_03 from P26_02 `…080814Z-72742e68`. BD-rate on
  LPIPS_V (PSNR_V) against SVT-AV1 | against DCVC-UF:
  - Component β = 0: P26_02 +82% (+16%) | −20% (−21%); P06_03 +44%
    (+36%) | −11% (−11%). Unclear by the rule: it beats the reference on
    both, SVT-AV1 on neither. It keeps about half the upper bound's gain
    (−37%, −36%), more than the third predicted.
  - Component β = 0.026: P26_02 unclear (+87%) | dominates (+20%), every
    point in SVT-AV1's range below SVT-AV1's best LPIPS_V; P06_03 −74%
    (+111%) | −87% (+32%). **Passes** by the rule. Payback of the 241 MB
    model against SVT-AV1 at equal LPIPS_V: 2.8 h (P26_02, 10 kbps), 14 h
    and 5 h (P06_03, 12 and 44 kbps); at higher rates SVT-AV1 never
    reaches its LPIPS_V.
  - Control: P26_02 −85% (+117%) | −90%; P06_03 −41% (+81%) | −69%.
    The scene model against its control: −54% (P26_02) and −65% (P06_03).
    The control's share of the component's gain against the unadapted
    model, in log rate: 75% on P26_02 (over the rule's 2/3) and 52% on
    P06_03 (under it). The rule fixed no aggregation over the pilot
    stretches, so it does not settle this: most of the LPIPS gain comes
    from fine-tuning for LPIPS on egocentric footage (the first competing
    explanation), and knowing the scene halves the rate again on both
    stretches.
  - Visual check (P26_02, frame 20, foreground pasted back; frames in the
    jobs' `published.tar`). At equal or lower rate the β = 0.026 output
    keeps the countertop's grain and the jar's inner structure where
    β = 0 shows a smooth wash; no periodic or grid artifact. The grain is
    synthesized, not the source's (lines and stains move or vanish), which
    is the PSNR_V cost. A perception–distortion trade, not an adversarial
    one; the paper has to say the background texture is generated.
- Report: [G5b Scene Model Pilot](https://claude.ai/artifact/H9yE6iBUGWau6K1HQBLcek)
  (curves, attribution, frames, payback, review answers).
- Pilot GPU time: about 5.6 hours (calibration 0.1, upper bound 4 × 0.55,
  component and control 6 × 0.55), within the 6 allowed. The rest (8
  stretches) waits for the user: the result changes what the rest should
  measure (per-scene against a pooled egocentric model).

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

### 2026-10-12 — G6 steps 2–3: the background at H5's format (resolution, frame rate)
- Question (PLAN G6): G5c's evidence is at 960×540 and 10 frames/s; H5
  is scored by B2's harness at 1080p and the windows' native 59.94
  frames/s. (2) Should the background be coded at 1080p, or at 960×540
  and upsampled at the client? (3) Does the scene model trained on 10
  frames/s footage keep its gain at the native rate, or does it need
  native-rate training?
- Method (`experiments/background/g6.py format`): the pilots' evaluation
  windows (P26_02_0000001071, P06_03_0000005110; 240 frames, 4 s),
  decoded as B2 does (its JPEG gate checked). Foreground: B1b's
  `visor_dense_sam_fill`, dilated 16 px at 1080p as G1; filled by
  ProPainter at 960×540, upsampled into the holes only. Methods on the
  filled frames: SVT-AV1 (G5's preset and CRFs) at 1080p and at 960×540
  upsampled (bicubic); DCVC-UF off the shelf at 1080p; recipe L's scene
  model of the window's stretch (trained at 960×540, 10 frames/s; the
  window lies inside its held-out excerpt) at 1080p and at 960×540
  upsampled. Scored at 1080p on the background (not B1b's foreground),
  `g5c.full_score`: PSNR, LPIPS, DISTS, flicker and KID on V.
- Decision rules, fixed now. (2) Per codec, the coded size is 960×540
  upsampled if its BD-rate on LPIPS_V against the same codec at 1080p is
  0% or better on both windows, else 1080p; PSNR_V and DISTS_V are
  reported beside it. The anchor for H5 is then SVT-AV1's envelope over
  both sizes (concave hull). (3) The 10 frames/s scene model transfers if,
  at its chosen size, its BD-rate on LPIPS_V against that SVT-AV1 envelope
  is within 20 points of its stretch's (P26_02 −92%, P06_03 −66%) on both
  windows; otherwise native-rate training (G6 step 3) runs before step 4.
- Hypotheses. (2) Downscaled coding wins for both codecs at these rates
  (the background is out of focus and noisy at 1080p, so its fine detail
  costs bits that LPIPS does not reward), most at the low end. (3) The
  gain transfers: at 59.94 frames/s each frame moves 6× less, which
  favours DCVC-UF's temporal prediction more than SVT-AV1's.
- Competing explanation for a failure of (3): the model learned the
  stretch's texture at 960×540 scale, and at 1080p (or after resampling)
  the textures it synthesizes no longer match.
- Budget: 6 jobs (per window: SVT-AV1 at 1080p; SVT-AV1 and the scene
  model at 960×540; the scene model and DCVC-UF at 1080p), each ≤ 45
  minutes on Ada (shared), about 4 GPU-hours.
- Runs (Ada, shared GPUs). P26_02: SVT-AV1 1080p `20261012T122424Z-f94ef019`,
  960×540 methods `…131334Z-d125ea07` (code `43086a7`, gpu6), scene model
  and DCVC-UF at 1080p `…122446Z-943a1dc5`. P06_03: `…122457Z-a660ab4c`,
  `…122506Z-f7bc558e`, `…122515Z-439a790f`. Code `915bae2` unless noted.
  P26_02's first 960×540 job (`…122435Z-307fb4ad`, gpu5, shared with
  another user's process) failed when DCVC-UF's decoder segfaulted after
  its own encode, and two of its saved scene-model points had failed the
  decode check (not deterministic, the decoder's intra frame not the
  encoder's): not reused; the job ran again from scratch on gpu6, and a
  restored attempt (`…130446Z-9e73b9bf`) failed its smoke, which had
  loaded the full run's points (fixed in `43086a7`). Every reported
  DCVC-UF point decodes deterministically and matches the encoder's
  intra frame. Foreground (B1b) 35% and 29% of the pixels, filled holes
  38% and 34%.
- Rates. SVT-AV1 at 1080p spends 130–3,070 kbps on the background (CRF
  63–32); the scene model at 960×540 12–400 kbps, mostly below SVT-AV1's
  lowest rate (CRF 63 is its floor), so BD-rates use the overlap.
- Outcome (2), resolution. BD-rate on LPIPS_V of 960×540 upsampled
  against 1080p, P26_02 | P06_03: SVT-AV1 +54% | +100% (DISTS_V +40% |
  +171%): **SVT-AV1 codes at 1080p**. Scene model −35% | −28% (PSNR_V
  −60% | −35%, DISTS_V −68% | −53%, KID_V −82% | −66%): **the scene
  model codes at 960×540 and is upsampled at the client**. Hypothesis
  half wrong: downscaling helps the model trained at that scale, not
  SVT-AV1. H5's anchor is SVT-AV1's envelope over both sizes, which is
  its 1080p curve.
- Outcome (3), frame rate. The scene model at 960×540 against the SVT-AV1
  envelope at 1080p, P26_02 | P06_03: LPIPS_V **−91% | −45%**, DISTS_V
  −93% | −12%, KID_V −77% | −6%, PSNR_V +72% | +25%, flicker unclear
  (too little rate overlap). Its stretches gave LPIPS_V −92% | −66%: on
  P26_02 the gain transfers, on P06_03 it is 21 points smaller, 1 point
  outside the rule's 20. **By the rule, native-rate training (G6 step
  3) runs before step 4.** Beside the rule: at the native rate the model
  beats SVT-AV1 on 3 of 5 metrics on both windows (on the stretches, 2
  of 5), and loses far less PSNR_V (+25% against +96% on P06_03); DCVC-UF
  off the shelf at 1080p comes close to SVT-AV1 there (LPIPS_V −5%,
  PSNR_V −7% on P06_03; +48% and +10% on P26_02).
- Decision (user, 2026-10-12): the frame-rate rule is treated as
  passed. It missed by 1 point on P06_03 (21 against 20), and the model
  already beats SVT-AV1 on more metrics at the native rate than on the
  stretches. Native-rate training is not run now; retraining at 1080p
  and at the native rate are ablations for the paper's final runs.
- GPU time: about 5 hours (8 jobs, 2 failed).

### 2026-10-12 — G6 steps 4–5, pilot: every stage per window on the pilots' windows
- Question (PLAN G6 4–5): the pipeline that makes both routes' background
  for every window of evaluation set v2, at the interface agreed with the
  foreground session, run end to end on the pilots' windows
  (P26_02_0000001071, P06_03_0000005110) before the other 32. Two things
  change from steps 2–3 and are checked here: the scene model's footage
  (the window's own video, not G1d's stretch), and the enhancer at H5's
  format (1080p and the native rate; G5d fitted it at 960×540, 10
  frames/s).
- Method, per window (`experiments/background/g6.py`):
  1. `footage --set scene`: G5b's split of a stretch applied to the
     window, so no window needs a prepared stretch: the 120 s of its own
     video around it (shifted to stay 5 s from the video's ends), less the
     window and 5 s on each side, at 10 frames/s and 960×540 (about 1,060
     frames). Foreground as G5c's footage (SAM 3.1 "hand" and "arm" plus
     the EPIC detector's boxes at score ≥ 0.5, dilated 16 px at 1080p;
     the detections cover all 34 videos, G1's `hand-objects.tar`), filled
     by ProPainter.
  2. `footage --set native`: 4 segments of 2 s at the window's rate and
     1080p from the same parts (starts spread evenly, blind to content),
     the same foreground, ProPainter at 960×540 upsampled into the holes
     only (as the window's fill). The enhancer's training pairs.
  3. `train`: recipe L (β 0.005, learning rate 3 × 10⁻⁵, 540 steps) on
     step 1's footage, from the public HT-L weights.
  4. `format --publish`: the window decoded as B2 does, F = B1b's
     `visor_dense_sam_fill` filled *exactly* (no dilation, as agreed),
     the whole frame coded: the scene model at 960×540 upsampled, QP 0,
     13, 25, 38, 50, 63, plus at 1080p QP 50 and 63 for the high end.
  5. `enhance --publish` (two jobs, CRFs 32–48 and 54–63): SVT-AV1 at
     1080p (G5's preset and CRFs; the anchor, published too), and G5d's
     enhancer (8 blocks, 64 channels, MSE + LPIPS on V, 2,000 steps)
     fitted per CRF on step 2's footage coded by SVT-AV1 at that CRF,
     applied to the window's SVT-AV1 decode.
  Scored at 1080p on V (`g5c.full_score`: PSNR, LPIPS, DISTS, flicker,
  KID). Published per window: each point's decoded 1080p frames (FFV1,
  4:2:0, BT.709 full range; checked by decoding them again), bitstream,
  enhancer weights, and a manifest (window, frame range, F's mask-set
  sha256, per point bytes, bitstream sha256, per-frame decoded sha256s,
  decode ms per frame, GPU class, job id and code revision). Points that
  the client upsamples are published after upsampling and scored as
  published (through 4:2:0, unlike steps 2–3).
- Decision rules, fixed now. (a) *The window's own footage replaces the
  stretch* if the scene model's BD-rate on LPIPS_V against SVT-AV1 at
  1080p is within 20 points of steps 2–3's (−91% | −45%) on both
  windows; otherwise the footage protocol is revised before scaling.
  (b) *The enhancer works at H5's format* if it beats SVT-AV1 at 1080p by
  10% or more on LPIPS_V on both windows; otherwise its fit at 960×540
  (SVT-AV1 at 540p, enhanced, upsampled) is tried on the pilots before
  scaling. (c) *The interface holds* if every manifest is complete and
  the foreground session composites one pilot window from it without a
  change. All three pass → the other 32 windows run with the same
  settings. Every metric is reported as BD-rate against SVT-AV1 at
  1080p (concave-hull envelope), the rate allocation with the
  foreground is the foreground session's.
- Hypotheses. (a) Passes: the window's own footage is closer to the
  window in time than most of a stretch's frames; filling F exactly
  (smaller holes) helps both codecs alike. (b) Passes with a smaller
  margin than at 960×540 (G5d's footage enhancer −36% | −50% on
  LPIPS_V): at 1080p and the native rate SVT-AV1's artifacts are finer
  and the enhancer's receptive field covers less of the scene.
- Competing explanations. (a) A failure would mean the stretch's frames
  far from the window carried the gain (more viewpoints), not the
  window's neighbourhood; footage from the whole video would test it.
  (b) A failure at 1080p could be the fit's size (4 segments of 2 s at
  1080p, 400–480 frames, hold fewer distinct views than G5d's 1,200–2,400).
- Budget: pilot 12 jobs, each ≤ 45 minutes on Ada or A6000 (shared),
  about 5 GPU-hours. The other 32 windows at the same per-window cost:
  about 80 GPU-hours, above PLAN's 65 (steps 4–5); the scale-up waits
  for the user's approval with the pilot's measured cost.
- Runs (shared GPUs; code `a103325`, with `f2e4aa9` for the enhancer
  jobs and `9e49c5e` for P26_02's native footage). Footage, scene:
  P26_02 `20261012T144606Z-fe5e8ef5` (1,061 frames: 79 before the window,
  982 after, since it starts 18 s into the video), P06_03
  `…144634Z-91fdde7f` (1,060). Footage, native: P06_03
  `…144704Z-1b330ff2`, P26_02 `…153200Z-a9414105` (480 frames each,
  foreground 9–25%). Training (Ada, both converged, monitor −0.3% and
  −0.4% over the last fifth): P26_02 `…150907Z-d03a60c9`, P06_03
  `…151248Z-bc31027a`. Scene route: P26_02 `…161128Z-5999b792`, P06_03
  `…160501Z-6cd65382`. Enhancer route: P06_03 `…154011Z-5c4135b3`,
  `…154024Z-a4e8259d`; P26_02 `…155052Z-9ad2482e`, `…155102Z-82b6caf5`.
  Failures, none citable: two SAM 3.1 and one ProPainter out of memory
  when another session's process grew on the shared GPU after
  admission (`…144625Z-c117ffa3`, `…160057Z-2eb07276`); P26_02's native
  footage hung twice after SAM (`…144950Z-e96dde14` ran out of its
  smoke time, `…150658Z-eef26194` was cancelled after 20 idle minutes):
  PyAV's threaded H.264 decoder deadlocked on close inside the process
  holding SAM 3.1, fixed in `9e49c5e` (the 1080p decode runs in the
  spawned decoder process); the enhancer's footage encode had no work
  directory (`…150734Z-796c8fa3`, fixed in `f2e4aa9`); the enhancer's
  declared 20,000 MiB was too low at full size (21.3 GB,
  `…152624Z-b5bff301`), now 30,000.
- Outcome. BD-rate against SVT-AV1 at 1080p (concave hull, 174–4,470
  kbps), P26_02 | P06_03:
  | Route | LPIPS_V | DISTS_V | PSNR_V | flicker_V | KID_V |
  |---|---|---|---|---|---|
  | enhancer (enh-1080) | better at every rate \| −83% | better at every rate \| −82% | +53% \| +26% | worse at every rate (both) | −72% \| −68% |
  | scene model, 540p | −97% \| **+20%** | −95% \| +116% | +97% \| +97% | worse at every rate (both) | −82% \| worse at every rate |
  | scene model, 1080p | better at every rate \| −61% | −79% \| +125% | +176% \| +94% | worse at every rate (both) | +35% \| +435% |
  - **(b) passes**: the enhancer beats SVT-AV1 on LPIPS_V, DISTS_V and
    KID_V on both windows, far beyond the 10% needed (at the same rate
    LPIPS_V falls 42–59%, PSNR_V 0.4–0.7 dB), and loses PSNR_V and
    flicker as at 960×540 (G5d). Hypothesis (a smaller margin than G5d's
    −36% | −50%) wrong: fitted on the window's own video at its size and
    rate, it gains more. Applying it took about 160 ms per frame on a
    shared A6000: not real time at 59.94 frames/s.
  - **(a) passes on P26_02 and fails on P06_03** (−97% against −91%;
    +20% against −45%, 65 points off). Two things changed at once from
    steps 2–3: the footage (the window's own 120 s, not the stretch) and
    F (filled exactly, not dilated by 16 px, so V now includes the
    pixels at the hands' edges, which neither model was trained to
    reproduce; P06_03's foreground is 27% of the frame). Beside it, on
    P06_03 the model now codes better at 1080p than at 540p (−61%
    against +20%), the reverse of step 2.
  - (c) Every manifest is complete (FFV1 frames re-decoded and checked,
    bitstreams, enhancer weights); the foreground session has not yet
    composited one.
  - GPU time: about 5.2 hours (12 completed jobs, 7 failed or cancelled).
- Diagnosis of (a) on P06_03, decided before its results (two jobs,
  about 1 GPU-hour): A, the stretch's recipe-L model of steps 2–3
  (`20261011T163301Z-e47ae994`) with F filled exactly
  (`20261012T180309Z-48fb21c5`); B, the window-footage model dilated by
  16 px as steps 2–3 (`…180332Z-92a1e304`), against steps 2–3's SVT-AV1
  anchor (`…122457Z-a660ab4c`, the same dilation). If A also loses
  (LPIPS_V at 540p within 20 points of +20%) and B recovers (within 20
  points of −45%), the exact fill is the cause and the footage stands;
  if A keeps −45% ± 20 and B does not recover, the footage is the cause
  and the footage protocol is revised (the whole video, as the
  competing explanation says). Mixed results are reported as they are
  and decided with the user.
- Diagnosis, outcome (code `9e49c5e`; A gpu6, B
  gpu5, Ada, shared; both 10 of 10 checks). BD-rate on LPIPS_V at 540p
  against SVT-AV1 at 1080p with the same fill: A, the stretch model with
  F exact, **+4%** (steps 2–3, F dilated: −45%); B, the window-footage
  model with F dilated, **+22%** against steps 2–3's own anchor
  (`…122457Z-a660ab4c`). By the rule this is mixed: A moves 49 points
  with the fill, and B does not recover with it.
  - The points explain both. On P06_03 the scene model's 540p curve
    overlaps SVT-AV1's only in a thin band of quality: its top one or two
    points (QP 50 and 63, 140–540 kbps) against SVT-AV1's lowest CRF
    (138–174 kbps). There the BD-rate is a PCHIP integral over a few
    thousandths of LPIPS and swings with them: steps 2–3's −45% was as
    fragile as the pilot's +20%. At matched rate (log-rate
    interpolation) the window-footage model equals the stretch model
    with the same dilated fill: LPIPS_V 0.225 against 0.219 at 27 kbps,
    0.163 | 0.164 at 59, 0.129 | 0.128 at 139, 0.108 | 0.106 at 394
    kbps (within ±3%, through 4:2:0 or not). The footage is not the
    cause, and the exact fill raises both codecs' cost (SVT-AV1's lowest
    point moves from 138 to 174 kbps and 0.135 to 0.143 LPIPS_V).
  - Rule (a) as written measures the overlap, not the model, on a
    window where the curves barely meet. What holds: the window's own
    footage makes the same scene model as the stretch (both windows);
    on P06_03 the scene model at 540p is SVT-AV1's equal at its lowest
    rates and worse above, the 1080p model wins on LPIPS_V there (−58%
    to −66%), and the route's hull (540p and 1080p) gives −22% (fill
    dilated) and +3% (exact). Decision with the user (the rule's
    clause for mixed results).
  - GPU time: about 1 hour (2 jobs).
- Decision (user, 2026-10-12): the other 32 windows do not run now.
  About 80 GPU-hours is worth spending once, on the final experiments,
  when the pipeline is settled. First the rest of the pipeline is found,
  built and optimized with cheap tests; then the paper's first draft
  fixes the figures and the experiments missing for them; then one
  heavy run produces every result the paper needs. The pilot's code
  (`g6.py` footage, train, format, enhance) and its two published
  windows stay as they are for the foreground session's compositor.


### 2026-10-12 — G6 item 1: the scene model at the exact fill, and its coded size
- Question: the pilot (above) put the window's F at the agreed exact fill
  (no dilation), so V now holds the pixels beside the hands' edges. The
  scene model was trained on footage whose foreground was G1's (dilated
  16 px, hand boxes), so it never saw that band. On P06_03 its 540p
  curve fell to SVT-AV1's level, while the 1080p model won (−58% to −66%
  on LPIPS_V), the reverse of step 2's choice of 540p (made with the
  dilated fill). Two questions. (a) Does footage masked like F teach the
  model the band? (b) Should the model code at 540p, at 1080p, or both
  by rate?
- Method (`experiments/background/g6.py`, pilot windows P26_02_0000001071
  and P06_03_0000005110):
  1. `footage --set scene --mask exact`: the pilot's footage protocol with
     the foreground shaped like F: SAM 3.1 "hand" and "arm" masks
     undilated, plus the EPIC detector's *object* boxes (score ≥ 0.5),
     undilated. The hand boxes are dropped because a box covers the band
     beside the hand that exact F leaves in V; the object boxes stand in
     for VISOR's object masks, which the footage has no annotations for.
     ProPainter fill as before. Model E: recipe L (β 0.005, learning rate
     3 × 10⁻⁵, 540 steps) on it, from the public HT-L weights. Model D:
     the pilot's trained model on G1's footage (`20261012T150907Z-d03a60c9`,
     `…151248Z-bc31027a`).
  2. `format` for D and E on each window with F exact, the same grid:
     540p QP 0, 13, 25, 38, 50, 63 (upsampled through 4:2:0), 1080p QP 13,
     25, 38, 50, 63 (the pilot had only 50 and 63, at 620 kbps and up,
     above most of SVT-AV1's range). Not published: this is a test.
  3. Anchor: the pilot's SVT-AV1 at 1080p with the same exact fill (the
     enhance jobs, 174–4,470 kbps). Scored at 1080p on V as the pilot.
- Decision rules, fixed now.
  (a) *E replaces D* if E's BD-rate against D on LPIPS_V (each model's
  hull of 540p and 1080p, same window, same F) is −5% or better on
  P06_03 and no worse than +2% on P26_02. Otherwise D stays (G1's
  footage, unchanged code path). Model against model: both curves span
  the same rates, so this BD-rate does not have the thin-overlap
  fragility that rule (a) of the pilot had.
  (b) *Coded size*, for the model (a) picks: BD-rate against SVT-AV1 at
  1080p (concave hull) on LPIPS_V for 540p alone, 1080p alone and their
  hull. If 1080p alone is within 5 points of the hull on both windows,
  the model codes at 1080p in SVT-AV1's range. Otherwise, if 540p alone
  is within 5 points of the hull, it stays at 540p. Otherwise the route
  ships the hull and switches size at the crossing rate. 540p stays
  available below SVT-AV1's lowest rate either way, if it is the hull
  there. Beside each BD-rate: the overlap in quality, and LPIPS_V at
  matched rate (log-rate interpolation) at SVT-AV1's three lowest CRFs.
  Every other metric (DISTS_V, KID_V, PSNR_V, flicker_V) is reported.
- Hypotheses. (a) E gains 5–15% on P06_03 (foreground 27%, so a long edge
  band) and stays level on P26_02. (b) 1080p is within 5 points of the
  hull in SVT-AV1's range on both windows: there, 540p's upsampling
  costs more than 1080p's extra bits. 540p wins only below about
  150 kbps.
- Competing explanations. (a) The band's cost is intrinsic: shadows,
  motion blur and VISOR's mask moving from frame to frame make the seam
  between fill and frame move, which no training removes (SVT-AV1 paid
  it too: its lowest point moved from 138 to 174 kbps). Then E ≈ D.
  SAM's undilated masks may also leave hand pixels in the footage's V,
  spending the model's capacity on them, so E could lose. (b) The
  pilot's 1080p win could come from its points sitting at rates where
  SVT-AV1 is weakest, not from the size; the new low QPs at 1080p test
  this.
- Budget: 8 jobs on shared Ada or A6000, each under 45 minutes: 2
  footage, 2 training (Ada), 2 format for E, plus 2 format for D. About
  3 GPU-hours.
- Runs (code `471f211`, shared GPUs, every job's validator passed).
  Exact footage: P26_02 `20261012T191458Z-ec41db31` (1,061 frames,
  foreground 23% | 14% per segment against G1's 28% | 17%), P06_03
  `…191508Z-059d63ab` (1,060; 23% | 17%). Model E, trained (both
  converged, monitor within 1% over the last fifth): P06_03
  `…193354Z-1c40a5f2` (Ada); P26_02 `…193114Z-78bb999e` ran out of its
  2,700 s on a GPU shared at 100% and saved its state, resumed to step 540
  by `…211954Z-b60afc24` (A6000; DCVC-UF trains on both classes, peak
  42 GiB). Format, D: P26_02 `…191518Z-6a62cbbb`, P06_03
  `…191526Z-32f31a5f`; E: P26_02 `…213720Z-eea1154c`, P06_03
  `…203355Z-54009c15`. Failed, not citable: E's P26_02 format
  `…212847Z-8fedd1fc`, whose smoke had one of eleven points (540p QP 38)
  decode unlike its encoder on gpu5 under contention; the same spec,
  resubmitted unchanged, passed every check.
- Outcome. Model against model (each model's concave hull, the same F),
  P26_02 | P06_03:
  | E against D | LPIPS_V | DISTS_V | PSNR_V | KID_V |
  |---|---|---|---|---|
  | hull (540p and 1080p) | **−11% \| −12%** | +32% \| −9% | −3% \| −5% | +17% \| −8% |
  | 540p | 0% \| −7% | +31% \| −7% | 0% \| −3% | +17% \| −7% |
  | 1080p | −28% \| −23% | +15% \| −14% | −15% \| −11% | −27% \| −23% |
  Against SVT-AV1 at 1080p (concave hull), LPIPS_V BD-rate and LPIPS_V
  at matched rate at SVT-AV1's three lowest CRFs (P26_02 189, 471, 787
  kbps: SVT-AV1 0.241, 0.218, 0.206; P06_03 174, 417, 708 kbps: 0.142,
  0.100, 0.087):
  | Model, size | P26_02 | P06_03 |
  |---|---|---|
  | D 540p | −97%; 0.146, 0.128, 0.126 | +20%; 0.137, 0.119, 0.113 |
  | D 1080p | −97%; 0.143, 0.109, 0.095 | −18%; 0.143, 0.099, 0.076 |
  | D hull | −96% | −4% |
  | E 540p | −97%; 0.149, 0.134, 0.130 | +26%; 0.138, 0.119, 0.113 |
  | E 1080p | −98%; 0.127, 0.102, 0.097 | **−38%**; 0.128, 0.087, 0.066 |
  | E hull | −97% | −33% |
  On the other metrics the E hull against SVT-AV1 is DISTS_V −92% |
  +87%, KID_V −66% | +218%, PSNR_V +126% | +80%, and flicker_V worse at
  every rate on both.
  - **(a) passes: E replaces D** (−11% on P26_02, at most +2% needed;
    −12% on P06_03, −5% needed). Hypothesis right on P06_03 (−12%,
    within the predicted 5–15%) and wrong on P26_02, which gains as much
    instead of staying level. The gain is at 1080p (−23% to −28% on
    LPIPS_V, and on KID_V and PSNR_V too); at 540p it is 0 to −7%, so the
    band the model learnt is detail that 540p does not keep. Against it,
    on P26_02 E loses DISTS_V (+32%) and KID_V (+17%) on the hull, all
    of it at 540p. The competing explanation (the band costs what it
    costs) is refuted for LPIPS_V.
  - **(b) the model codes at 1080p in SVT-AV1's range**: 1080p alone is
    within 5 points of the hull on both windows (−98% against −97%;
    −38% against −33%), and at matched rate 1080p is better than 540p at
    every one of SVT-AV1's three lowest rates on both. Hypothesis right.
    540p stays for rates below SVT-AV1's lowest (the hull is 540p below
    about 60 kbps on P26_02 and 70 kbps on P06_03). The pilot's
    1080p −61% on P06_03 had two points overlapping SVT-AV1 in a thin
    band; with QP 13–38 it is −18% for D, and the competing explanation
    (the pilot's 1080p points sat where SVT-AV1 is weakest) held for D.
    Caveat: on P26_02, 1080p's LPIPS_V overlap with SVT-AV1 is thin (its
    lowest point, 61 kbps, already beats SVT-AV1's best), so its −98%
    rests on matched rate, which agrees. On DISTS_V and KID_V, 540p is
    better than 1080p on both windows (E 1080p against 540p: DISTS_V
    +178% | +87%, KID_V +180% | +163%): the 1080p decode carries LPIPS
    detail that the distribution metrics do not credit. The route's size
    follows the headline metric, LPIPS_V, as G5c's selection did.
  - GPU time: about 4 hours (10 jobs, 2 of them failed or timed out).
- Decision: G6's scene model is trained on footage masked like F
  (`footage --mask exact`) and codes at 1080p in SVT-AV1's range, with
  540p below it. The final run's `format` grid becomes 540p QP 0, 13, 25,
  38 and 1080p QP 13, 25, 38, 50, 63 (passed as `--mask exact` and
  `--values`; the defaults stay the pilot's, so its specs re-run as
  they were). The pilot's published windows (D at
  540p and two 1080p points) stay for the foreground session's
  compositor test, rule (c), whose interface does not change.


### 2026-10-08 — H1: foreground motion and representation audit
- Question: what is VISOR's foreground made of, and how much of it could
  compact per-object motion parameters (hand pose, an object's rigid motion)
  explain instead of coded pixels? Answers H2–H5's questions: whether hands
  are worth MANO at all, how much of a VISOR hand is forearm (H3 must render
  it), which objects a rigid model holds for (H4), and what the parameters
  cost against B2's bits on the same pixels.
- Data: evaluation set v2 (34 windows of 240 frames), mask set
  `visor_dense_sam_fill` (B1b, objects filled), with `visor_dense` reported
  beside it for composition. Frames decoded as B2 does, sparse-JPEG gate
  included. Code `experiments/visor/h1.py`.
- Parts, per frame (each pixel in one part; earlier parts win where masks
  overlap): *hand* (a VISOR hand mask on the fingers' side of the MANO wrist
  line: the line through the fitted wrist joint perpendicular to wrist →
  middle-finger knuckle), *forearm* (the same mask beyond that line),
  *hand, no fit* (a hand mask without a MANO fit), *handled object* (an active
  object a human labelled in contact with a hand, `in_contact_object`, at a
  keyframe bounding or inside the window), *other object* (every other active
  object), background.
- Method.
  1. Composition: each part's share of the frame's and of the foreground's
     pixels. Bits: B2's published SVT-AV1 streams (all 34 items, CRF 41–62)
     decoded by libaom 3.12.1's `inspect` built with bit accounting
     (`CONFIG_ACCOUNTING`, `CONFIG_INSPECTION`; `tools/aom/build-inspect.sh`,
     binary `aom-inspect-v3.12.1-ps1`, sha256 `9b2e75a2…3959`). The patch
     (`env/patches/aom-inspect-ps1.patch`, `cbc0eb0a…50bf`) prints each
     frame's order hint, so bits map to display frames, and every symbol:
     upstream prints a block's context in place of its first symbol, dropping
     its bits. Every entropy-coded symbol is attributed to the
     block being decoded; a block's bits are spread evenly over its pixels and
     split by the parts' pixel shares in the frame it displays as. Frame
     headers are outside the accounting; their share is reported. DCVC-UF's
     rANS stream has no per-position bits, so its per-part split is not
     measured; only its totals are compared.
  2. Hands: WiLoR (its own YOLO hand detector, conf 0.3; quicker than HaMeR,
     with a detector; H2 decides between them). Each VISOR hand takes the
     detection most of whose box lies on the mask (≥ 0.5 of the box);
     handedness comes from VISOR. The fitted MANO mesh is projected with
     WiLoR's camera and rasterized as a silhouette. Per hand and frame: IoU
     with the whole mask, IoU with the hand-side part, share of the mask
     beyond the wrist (forearm), share of the silhouette outside the mask.
     Stability over consecutive fitted frames: root-relative 3D joint
     acceleration (mm/frame²), 2D keypoint acceleration (share of box size),
     global-orientation change (degrees per frame, flips over 45°). Failure
     conditions, flagged per hand-frame: no detection; detector box at the
     image edge; occlusion (≥ 25% of the box covered by object masks); blur
     (Laplacian variance in the box under half the hand's median, or box
     motion over 10% of its size per frame).
  3. Objects (and, as rigid baselines, the hand-side part and the forearm):
     backward DIS optical flow per frame pair (OpenCV, half resolution), and
     per object a similarity and a homography fitted to the flow inside its
     mask (RANSAC). Frame to frame: PSNR inside the object's mask at t + 1 of
     the previous frame warped by no motion, the similarity, the homography,
     and the dense flow (the most any motion model can explain). From a
     reference (what PointStream would send): the object's first frame,
     carried by the chained homographies; it *holds* at a frame when PSNR on
     the covered pixels is ≥ 30 dB and at most 20% of the mask is uncovered;
     a new reference starts where it fails; reported as the share of frames
     held and the reference life (s). 33.75 dB (B2's foreground PSNR at
     SVT-AV1 CRF 62 with the fill) is reported beside 30 dB. The homography's
     squared error is split by pixel into new appearance (the source falls
     outside the object's mask at t), occlusion by hands (it falls on a hand),
     deformation (homography error minus flow error on the remaining pixels),
     appearance change (the flow's own error there: turning, lighting, source
     noise), with frames whose object moves more than 10 px per frame counted
     as motion blur.
  4. Parameter rate: hands as MANO pose (15 joints, axis-angle), global
     orientation, and camera (image position of the MANO origin and log
     depth), shape once per track; objects as the homography (4 bounding-box
     corner displacements). Quantized (angles 1°, positions 0.25 px, log depth
     0.002; corners 1/8 px; the coarsest steps keep the projected joints
     within 1 px on average, checked in the job), coded as first differences;
     the rate is the empirical entropy of those symbols pooled over items,
     times the frame rate, with a fixed 16 bits per value as the upper bound.
     Compared with SVT-AV1's bits on the same part (step 1).
  5. Review: two content-blind frames per item (sha256 of
     "pointstream-h1-review:<item>:<n>") drawn with the parts, the rendered
     MANO hands and the homography-warped objects with their error.
- Decision rule (fixed before any run; `h1.DECISION`, applied by `h1 report`
  on the means over the 34 items, fill set). A part is worth a parametric
  model when all three hold: (a) it carries ≥ 10% of the foreground's
  SVT-AV1 bits at both CRF 48 and CRF 62; (b) the model explains it: hands
  fitted on ≥ 90% of hand-frames, median hand-side IoU ≥ 0.70 and ≥ 70% of
  hand-frames at ≥ 0.60; rigid parts held from a reference on ≥ 70% of their
  frames with median reference life ≥ 0.5 s; (c) its parameter rate (entropy
  estimate) is ≤ 10% of SVT-AV1's bits on it at CRF 62. Otherwise it stays
  pixels; an object class or condition that passes (b) on its own is named
  for H4 as a candidate. The forearm is H3's to render if it is ≥ 20% of the
  hand masks' pixels; (b) decides rigid layer or pixels. Order of H2–H4:
  passing parts by expected saving (bit share × share explained).
- Hypothesis: hands with forearms are most of the foreground's pixels and
  bits; forearms are 30–40% of VISOR's hand masks; WiLoR covers the hand side
  at median IoU ≥ 0.70, failing mostly under occlusion by held objects and
  blur; handled objects are poorly explained by rigid 2D motion (under half
  their frames held: they turn, are occluded by hands, deform), while other
  active objects (touched surfaces, static in the world) are held like the
  background; parameters cost a few kbps, under 5% of SVT-AV1's bits on the
  same pixels. Order: hands (H2, H3, forearm included), then handled objects
  only where rigid (H4).
- Competing explanations: (1) large touched surfaces (sink, fridge,
  cupboard) that VISOR labels active objects dominate the foreground's
  pixels and bits; they are static in the world, so G's background, not H,
  should carry them. The handled/other split tests this. (2) MANO covers the
  masks (IoU) but jitters, so its rate and renders are worse than the
  per-frame numbers suggest; the stability measures test this. (3) Rigid
  fits fail because the flow fails on textureless objects, not because the
  objects are non-rigid; the dense-flow bound separates the two.
- Budget: WiLoR runs on all four GPU classes (model–GPU table); flow and
  fits on 16 CPU threads. Smoke: one item, 48 frames, ≤ 600 s. Pilot: the four
  B2 pilot items (P01_107, P09_106, P02_02, P03_10), ≤ 1 h. Full: the other
  30 items as two jobs of 15, ≤ 2 h each. Ceiling 5 GPU-hours and 10 h wall
  including staging.
- Jobs (code `6847a29`, environment `pointstream-20261006T113321Z`; inputs
  `pointstream-data/visor/h1-2026-10-08/inputs/`, the B1/B1b/B2 records named
  in the method, specs `h1-pilot.json`, `h1-full-a.json`, `h1-full-b.json`
  beside them): pilot `20261008T072619Z-0d09640b` (gpu3, RTX A6000; the four
  pilot items; 1,106 s full stage); full B `20261008T083835Z-5dafce84` (gpu1,
  Quadro RTX 8000; 15 items; 5,387 s); full A1 `20261008T103118Z-a358e4ad`
  (gpu1, Quadro RTX 8000; ended `contended` after its 10th item, so only
  `partial.tar` was published: P02_09, P03_120, P04_13, P06_03, P06_10,
  P06_108, P07_110, P09_02, P09_104, P21_01, all finished before the foreign
  process appeared; its validator was run afterwards on `partial.tar` with
  the job's own smoke-stage WiLoR and device record, every check passing);
  full A2 `20261008T150618Z-fb44e5a3` (gpu3, RTX A6000; the other 5 of
  share A; 1,693 s). Validators passed on all four full outputs (18 checks
  each), so every item met the JPEG gate, matched both mask records and
  B2's stream hashes, and had each window frame's bits once. Not evidence:
  `…070724Z-fb6fa79e` (OpenCV `remap` limit), `…071805Z-646a9841` (staged
  binary not executable), `…083438Z-6040a9d6` (contended on gpu6),
  `…101514Z-20d1d0da` (cancelled during contention on gpu3),
  `…145714Z-d310293f` (smoke item not in the share). About 3.5 GPU-hours
  in the four full stages, 8.5 h wall. Report (`h1 report`, all 34 items):
  `pointstream-data/visor/h1-2026-10-08/report-6847a29/h1-report.json`
  (`7e01635a…3fee`). Review overlays (102 frames, 3 per item):
  [artifact](https://claude.ai/artifact/7ir8Smh5tyWLFDejPppj6k).
- Provenance note: another agent wrote a `published.tar` (a copy of
  `partial.tar`) and an `h1.json` (device record and peak memory copied from
  full B) into A1's job directory so the report would run. Both were moved
  to `visor/h1-2026-10-08/a1-reconstructed-outside-the-job/` with a README;
  the job directory again holds only what the fleet wrote. Its report agreed
  with the reviewed one on every decision value.
- Outcome, composition (means over items, fill set). The foreground is 26.5%
  of the pixels (hand 3.6%, forearm 4.2%, hand without fit 0.4%, handled
  objects 5.5%, other objects 12.8%) and 35% of SVT-AV1's bits at every CRF
  (35.1–35.6%; accounting covers 93–99% of each stream's payload, the rest
  is headers). Of the foreground's bits at CRF 48 / 62: hands 19.4 / 20.8%,
  forearms 11.7 / 12.3%, hands without fit 1.5 / 1.6%, handled objects
  29.1 / 27.0%, other objects 38.2 / 38.3%. In kbps at CRF 48 / 62: hands
  77.4 / 23.4, forearms 56.9 / 16.9, handled objects 176 / 41.3, other
  objects 232 / 62.5. Without the fill the foreground is 20.5% of the
  pixels and objects are a smaller share (other 40%, handled 9% of the
  foreground's pixels). DCVC-UF is not split by part.
- Outcome, hands (WiLoR, 14,852 hand-frames). Fitted on 91.5% of
  hand-frames; the unfitted ones are mostly small, partly visible hands
  (median mask 19k pixels against 76k). Hand-side IoU median 0.767 over
  fitted frames (mean of per-item medians 0.753), below 0.5 on 2.0%; whole
  mask IoU median 0.50, because the forearm is half the mask: 49.7% of the
  hand masks' pixels lie beyond the wrist (43% per fitted hand-frame; 7%
  when the hand is at the image edge, where the wrist line is least
  reliable). By condition: no flag 41% of frames, IoU 0.768, under 0.5 on
  0.2%; occlusion ≥ 25% of the box 35%, 0.741, 4.8%; image edge 24%, 0.803,
  2.8%; blur 10%, 0.756, 5.0%. Stability: root-relative joint acceleration
  median 4.7 (left) / 5.0 (right) mm/frame², 2D keypoints 2.1 / 2.6% of the
  box per frame², orientation change 1.9 / 2.1° per frame, 7 flips over 45°
  in 13,404 consecutive pairs. Quantization (1°, 0.25 px, 0.002 log depth)
  moves the projected joints by 1.00 px on average (95th percentile 1.73),
  at the 1 px bound the method set, not under it. Parameter rate (entropy
  of first differences): 165 bits per hand-frame, about 3.2 bits per value,
  so 15.2 kbps per item against SVT-AV1's 23.4 kbps on hand pixels at
  CRF 62 (65%) and 77.4 kbps at CRF 48 (20%).
- Outcome, rigid motion (homography from flow; PSNR medians, means over
  items). Frame to frame: other objects copy 29.1, homography 36.0, dense
  flow 37.0 dB; handled objects 26.9, 32.7, 34.3; forearm 34.6, 40.0, 41.4;
  hand 31.8, 37.8, 39.7. From a reference, at 30 dB with ≤ 20% uncovered:
  held 70.8% (other), 49.9% (handled), 78.8% (forearm), 71.9% (hand) of
  frames, but the chains are refreshed every few frames: median reference
  life 0.04 s (other), 0.02 s (handled), 0.04 s (forearm and hand); weighted
  by frames 0.10, 0.04, 0.24, 0.12 s; only 7.5% (other), 7.4% (handled),
  32% (forearm) and 17% (hand) of frames lie in references that live 0.5 s.
  No object label reaches a 0.5 s median life (best: toaster 0.21 s, one
  item; tray 0.15 s; chopping board 0.08 s, sink 0.07 s), though some are
  often held (toaster 93%, tray 91%, pot 88%, chopping board 85%). Error of
  the frame-to-frame homography, as shares of its squared error: other
  objects appearance 36%, deformation 33%, new pixels 14%, hand occlusion
  10%, blur 7%; handled objects 28%, 27%, 17%, 10%, 18%. Caveat: EK-55
  repeats some frames (10.5% of its frame pairs are near-identical, 44% on
  one item), which flatters its frame-to-frame scores slightly.
- Decision (by the rule; `h1-report.json`): **no part is worth a parametric
  model by H1's thresholds.** Hands pass (a) bits and (b) fit but fail (c)
  rate: their parameters cost 65% of SVT-AV1's bits on the same pixels at
  CRF 62, against the 10% allowed. Forearms, handled and other objects pass
  (a) and fail (b): references die within a tenth of a second (median life
  0.02–0.04 s, rule 0.5 s), and handled objects are held on only half their
  frames. The forearm is 49.7% of the hand masks, so by the rule H3 renders
  it with the hand. No object class passes (b) on its own, so none is named
  for H4.
- Reading. The hypothesis is refuted on its main point: hands with forearms
  carry a third of the foreground's bits, not most of them; objects carry
  two thirds, and other (not handled) objects alone 38%. The competing
  explanation (1) holds: large touched surfaces (pan, sink, hob, chopping
  board, plate) that VISOR labels active objects are the largest part of
  the foreground; their image motion is mostly the camera's, which G's background
  should carry rather than H. As predicted, the forearm is large (half,
  above the 30–40% guessed), WiLoR covers the hand side (median 0.77) and
  fails mostly under occlusion and blur, and handled objects are poorly
  explained (held 50%, life 0.02 s). Competing explanation (3) does not
  hold: dense flow explains only 1–2 dB more than the homography, and most
  of the error is appearance change and new pixels, which no motion model
  explains. Hands are the only part close to passing, and the gap is rate,
  not fit: WiLoR's per-frame estimates jitter (5 mm/frame²), so first
  differences cost 3.2 bits per value; smoothing, coarser angles or a pose
  subspace could plausibly bring the rate down severalfold, and at CRF 48
  it is already 20% of the codec's bits on hands. That is H2's to measure
  (temporal stability is one of its axes), not something H1 shows.
- Scope. The rigid test is a 2D warp (similarity, homography) of the
  object's pixels from a reference: exact for planar surfaces and for a
  camera rotating in place, an approximation for everything else. Its
  failure says a 2D warp of one reference view cannot carry these parts; it
  does not test a 3D object pose (6DoF), an articulated arm, a pose relative
  to the hand, or a bank of reference views. Rule (c)'s 10% is a screen
  that leaves room for the appearance reference and the residual; whether
  hands beat coded pixels at equal rate is H3's measurement.
- Order for H2–H4 (a judgement, since no part passed): H2 as planned, with
  pose rate after temporal smoothing added to its stability axis; H3 hand
  and forearm together, compared against coded pixels at equal rate, the
  forearm rendered from the hand's pose and the mask's extent rather than
  as a rigid layer (forearm references last 0.24 s frame-weighted); H4 as
  briefed (rigid objects from a reference) has no support from H1: handled
  objects stay pixels, and the static surfaces belong to G. H4 should be
  rescoped or dropped before it runs; the user decides.

### 2026-10-09 — H2: hand-pose estimators (HaMeR against WiLoR) and pose coding
- Question. Which MANO regressor PointStream uses on egocentric video, with
  numbers; and how cheaply its pose stream can be sent, as rate against
  distortion and latency, for each coding technique and their combinations
  ([PLAN](../PLAN.md#h2-hand-pose-estimators-hamer-against-wilor-done-2026-10-10)).
- Estimators. HaMeR (`3a01849`, ViT-H) and WiLoR (`fcb9113`) as the
  environment audit loads them; MANO v1.2 (the dechumpied copies). Both get
  the same box per hand, enlarged 2.0× (`ViTDetDataset`, as H1 and HaMeR's
  own HInt evaluation), and the same handedness. Joints for every metric are
  computed the same way from each mesh, predicted or ground truth: MANO's
  joint regressor on the 778 vertices plus the five fingertip vertices
  (744, 320, 443, 554, 671), in OpenPose order; so no metric depends on a
  model's own joint convention.
- Benchmarks.
  1. *HInt* (2D, egocentric frames, `HInt_annotation_partial.zip`,
     `ac42d9f8…c7fe`): TEST_epick (VISOR frames, 1,906 hands) and
     TEST_newdays (1,754), boxes from the labels. HaMeR's protocol
     (`hamer/utils/pose_utils.py`): PCK at 0.05, 0.10 and 0.15 of the box's
     longer side, over in-frame joints (*all*), unoccluded (*visible*) and
     occluded ones. HaMeR trains on HInt train, so it is in domain here and
     WiLoR is not. Anchor: HaMeR's published PCK@0.05 (all) is 43.0 on VISOR
     and 48.0 on New Days (a later re-run: 44.4, 49.4).
  2. *HOT3D-Clips* (3D, motion capture): 54 train_aria clips, 6 per
     participant (all 9), chosen content-blind (lowest sha256 of
     "pointstream-h2-hot3d:<clip>"), 150 frames at 30 fps. Each hand with a
     MANO label is warped from the 1408² fisheye stream `214-1` into the
     dataset's own pinhole crop camera (`hand_crops.json`, the toolkit's
     `warp_image`; the protocol of HOT3D's hand-tracking challenge) at half
     its focal length, 512²: the dataset's crops frame the hand so tightly
     that its mesh leaves the crop, and the regressors' 2× box would see
     padding (dev check on gpu6: boxes reached x = −15 and y = 551 in a
     512² crop; at half focal they sit inside with real context). The box
     is the projected ground-truth mesh's. Metrics:
     MPJPE after aligning the wrist and PA-MPJPE (mm), MPVPE after aligning
     the wrist, 2D joint error in the crop (share of box), and acceleration
     error against the ground truth (mm/frame², root-relative joints rotated
     into world axes, so the head's motion cancels). Neither model trains on
     HOT3D; hands wear motion-capture markers.
  3. *VISOR* (video, evaluation set v2, 34 windows, 240 frames, fill set):
     H1's recorded hand boxes (WiLoR's detector, matched to VISOR's masks),
     so both models fit the same hand-frames as H1. Per hand-frame: IoU of
     the rendered silhouette with the mask's hand side (H1's wrist split,
     each model splitting by its own wrist), 2D keypoint acceleration (share
     of box), and orientation flips over 45° per frame.
  4. *Speed* on one GPU class (RTX A6000), fp32, after warm-up, CUDA
     synchronised: ms per hand at batch 1 (the codec's case) and per hand at
     batch 32, regressor only; WiLoR's detector timed once per frame,
     since both models need a detector in PointStream.
- Pose coding (`h2 code`, CPU, on the saved per-frame parameters of both
  estimators). Sent per hand-frame: global orientation (axis-angle),
  articulation (15 joints axis-angle, or k coefficients of MANO's pose
  PCA), root as image position (px) and log depth; shape once per track
  (the track's median betas), not counted. On HOT3D the parameters are
  expressed in the fisheye camera's frame and the root's position is that
  camera's pinhole projection; on VISOR they are H1's (nominal focal). Encoder: smoothing, then temporal subsampling, then the
  subspace, then quantization, then prediction; rate = empirical entropy of
  the prediction residual symbols pooled per parameter group over items
  (H1's estimate), × frames sent per second. Decoder: dequantize, predict,
  and fill skipped frames by linear interpolation (needs the next sent
  frame) or by holding (no lookahead). Axes:
  - smoothing: none; One-Euro (causal; min cutoff 0.5, 1, 2 Hz × beta 0,
    0.5); centred Gaussian looking ahead L = 33, 67, 133, 267 ms (1, 2, 4,
    8 frames at HOT3D's 30 fps, 2, 3, 7, 13 at VISOR's 50; σ = L/2);
  - send rate: every frame, 15, 10, 7.5 Hz (every 2nd, 3rd, 4th frame at
    30 fps; every 3rd, 5th, 7th at 50);
  - subspace: none (45 values), PCA k = 6, 12, 24;
  - quantization: H1's steps (1°, 0.25 px, 0.002) × 0.5, 1, 2, 4, 8;
  - prediction: previous decoded frame (H1), constant velocity.
  Latency = frames of lookahead (the smoother's L, plus the gap to the next
  sent frame when interpolating), reported in ms. Distortion: on HOT3D
  against the motion-capture truth (wrist-aligned MPJPE, mm; 2D joint error
  in the fisheye image, px), and against the uncoded estimate; on VISOR
  against the uncoded estimate (2D px at 1920×1080) and, for the points on
  each Pareto front, the silhouette's hand-side IoU. Every combination of
  the axes is evaluated (3,080 per estimator); the curves are the Pareto
  fronts of rate against distortion per latency budget (0, 33, 100,
  267 ms) and per technique alone.
- Decision rule (fixed before any run; `h2.DECISION`, applied by
  `h2 report`).
  - Estimator. Five accuracy contests: HInt VISOR PCK@0.05 (all), HInt New
    Days PCK@0.05 (all), HOT3D PA-MPJPE, HOT3D acceleration error, VISOR
    median hand-side IoU. A model wins a contest when the 95% bootstrap
    interval of the paired difference excludes zero (1,000 resamples, by
    image for HInt, by clip for HOT3D, by item for VISOR). The model with
    more wins is chosen; on a tie (including no wins) the faster at batch 1.
    If HaMeR's HInt VISOR PCK@0.05 (all) falls outside 40–47, the protocol
    does not reproduce the paper, and the run stops there.
  - Coding, for the chosen estimator. Per latency budget (a combination's
    latency is the larger of its HOT3D and VISOR values), H3's pose input is
    the combination with the lowest VISOR rate among those whose HOT3D
    errors against the truth (both MPJPE and 2D) are at most 5% above the
    uncoded estimate's; its VISOR rate
    (kbps over the 34 items, as H1) and VISOR IoU change are reported beside
    it, and against H1's screen (≤ 10% of SVT-AV1's bits on hands at CRF 62:
    2.3 kbps against 23.4). The smallest budget whose choice passes the
    screen is named for H3; if none does, the lowest-rate choice at 100 ms.
- Hypothesis. HaMeR wins on HInt (in domain) by a few points, WiLoR on
  HOT3D or neither; no clear winner, so WiLoR is chosen on speed. Most of
  the pose rate is the estimator's jitter: centred smoothing with ≤ 4 frames
  of lookahead plus 10–15 Hz cuts the rate at least 4× without moving the
  error against the truth (smoothing lowers it), and the 100 ms choice
  passes H1's screen; PCA below 24 components costs accuracy on grasps;
  constant-velocity prediction does not help once smoothed.
- Competing explanations. (1) HOT3D's lab scenes and markers favour one
  model for reasons that do not carry to kitchens; HInt and VISOR test
  egocentric kitchens directly. (2) Errors against the truth are dominated
  by the estimator's bias (depth, global rotation), so coding changes are
  invisible against the truth; the distortion against the uncoded estimate
  separates the coding error. (3) The PCA subspace, learnt from MANO's scans
  of free hands, misses grasp poses; its error against the truth at small k
  tests it.
- Budget. GPU (A6000 preferred; both models pass on every class): smoke
  ≤ 600 s each; pilot: 200 HInt hands, 9 HOT3D clips (one per
  participant), the 4 B2 pilot items, ≤ 1 h; full: HInt test (3,660 hands)
  with the speed runs, 54 HOT3D clips, 34 VISOR items, ≤ 4 GPU-hours.
  CPU coding: one host, 16 threads, ≤ 2 h. Ceiling 6 GPU-hours, 10 h wall.
- Pilots (code `5d1ff5e` for HInt and VISOR, `d8e00e4` for HOT3D (the same
  `h2.py`); environment
  `pointstream-20261006T113321Z`; inputs and specs in
  `pointstream-data/visor/h2-2026-10-09/`). HInt `20261009T215317Z-ebff7963`
  (gpu2, RTX A6000; 50 hands per split; validator 10/10): PCK@0.05 (all)
  on VISOR frames HaMeR 52.6, WiLoR 56.2; New Days 58.1, 57.6. HaMeR's 52.6
  is above the anchor range (40–47) on 50 hands; the full HInt run decides
  it. Speed (A6000, fp32, regressor only): HaMeR 30.2 ms per hand at batch
  1 and 16.0 at batch 32, WiLoR 30.1 and 16.6 (both ViT-H, 672 M and 641 M
  parameters), WiLoR's detector 25.5 ms per frame: the speed tie-break does
  not separate them. HOT3D `20261009T215354Z-00426e36` (gpu2, RTX A6000;
  9 clips, one per participant, 2,562 hands; validator 10/10; the numpy
  MANO reproduces both models after the change of frame to < 1 µm):
  PA-MPJPE HaMeR 9.5 mm, WiLoR 6.4; acceleration error 4.4 and 5.8
  mm/frame²; but wrist-aligned MPJPE 19.3 and 31.2 mm, because WiLoR's
  global orientation is worse (median 16.9° against 7.0°, both hands
  alike, so not a mirroring error), while its articulation and size are
  better (size ratio 0.98 against 0.89). PA-MPJPE removes the orientation,
  which a renderer needs; the report gives orientation error and
  wrist-aligned MPJPE beside the pre-registered contests.
- Full HInt `20261009T220226Z-958cda4e` (gpu2, RTX A6000; all 3,660 test
  hands; 849 s; validator 10/10). PCK@0.05 / 0.10 / 0.15, all joints, as
  pre-registered (distance over the longer side of HInt's `bbox` grown to
  3:4): VISOR frames HaMeR 51.3 / 82.3 / 92.5, WiLoR 55.2 / 84.7 / 93.9
  (difference −3.8, 95% CI −4.5 to −3.2: WiLoR wins); New Days HaMeR 58.1 /
  84.3 / 92.4, WiLoR 56.6 / 83.9 / 92.6 (+1.6, CI +1.0 to +2.2: HaMeR wins).
  **The anchor fails**: HaMeR's 51.3 is outside 40–47, so by the rule the
  estimator decision stops here. Every published HaMeR number is 7–9
  points lower (all, visible and occluded; both sets), a uniform shift
  that points at the normalising box, not the model: HInt's `bbox` is
  padded, 1.24× the extent of the labelled keypoints (median, both axes).
  Normalised by the keypoints' extent instead (same 3:4 growth), HaMeR
  scores 39.6 on VISOR frames (paper 43.0) and 48.9 on New Days (paper
  48.0), WiLoR 43.8 and 47.5; the remaining VISOR gap is plausibly the
  input crop (HaMeR's evaluation crops around its own box; this run feeds
  HInt's to both models). Under either normalisation the order is the
  same: WiLoR ahead on VISOR frames by about 4 points, HaMeR ahead on New
  Days by about 1.5. How to proceed is the user's decision.
- Full HOT3D `20261009T220759Z-be2af4a6` (gpu2, RTX A6000; 54 clips, 9
  participants, 14,585 hands; 1,971 s; validator 10/10). PA-MPJPE HaMeR
  9.2 mm, WiLoR 6.2 (difference +3.1, 95% CI +2.7 to +3.6: WiLoR wins);
  acceleration error 5.1 and 6.7 mm/frame² (−1.7, CI −2.0 to −1.3: HaMeR
  wins). Reported beside them: wrist-aligned MPJPE 20.1 and 29.7 mm, MPVPE
  19.2 and 28.2 mm, 2D error 6.8% and 7.0% of the box, median global
  orientation error 7.6° and 14.7°.
- Coding dev check (not evidence: `h2 code` on the two pilots' outputs, gpu6
  CPU, code `2611942`; 45 s). It found a crash on tracks shorter than the
  Gaussian window (fixed, with a test). Values: H1's coding (q1, previous,
  every frame) costs 12.6 kbps (HaMeR) and 12.4 (WiLoR) on the four VISOR
  pilot items (H1: 15.2 on all 34). For HaMeR the rule chooses 6.5 kbps at
  0 ms (q4) and 1.7 kbps at 100 ms (Gaussian 33 ms, 15 Hz linear, q8; VISOR
  IoU 0.790 uncoded → 0.781). For WiLoR it chooses nothing at any budget:
  the shape is sent once (the track's median betas) while the uncoded
  reference keeps each frame's, and WiLoR's betas vary much more than
  HaMeR's, so even at half H1's step its 2D error against the truth is
  5.4% above the uncoded one (HaMeR 0.1%), past the 5% limit before any
  pose coding. One-Euro at these cutoffs lags fast hands (2D error 31 px
  against 14); centred smoothing lowers the 3D error against the truth
  (HaMeR 19.16 against 19.34 mm), as hypothesised. The reference for the
  5% limit is the user's decision.
- Decisions (user, 2026-10-10, after the HInt full run and the coding dev
  check; recorded in `h2.DECISION`). (1) HInt is normalised by the
  labelled keypoints' extent grown to 3:4, and the anchor is HaMeR within
  4 points of the paper on both sets: 39.6 against 43.0 on VISOR frames
  (−3.4, the remaining gap read as the input crop), 48.9 against 48.0 on
  New Days (+0.9); it holds. The pre-registered bbox normalisation is
  reported beside it. With it, VISOR frames: WiLoR 43.8, HaMeR 39.6
  (difference −4.2, CI −4.8 to −3.5: WiLoR wins); New Days: HaMeR 48.9,
  WiLoR 47.5 (+1.4, CI +0.8 to +2.0: HaMeR wins). One hand has fewer than
  two labelled keypoints and is left out. (2) The coding stage's 5% limit
  is measured against the uncoded pose with the shape the scheme sends
  (each track's median betas); the cost of sending the shape once is
  reported apart for each model.
- Full VISOR `20261009T220823Z-b309d9a2` (gpu1, Quadro GV100; the other 30
  items; 3,296 s; validator 8/8) with the pilot `20261009T215322Z-188ed69c`
  (gpu1, GV100; 4 items): 34 items, 13,592 hand-frames. Hand-side IoU
  median HaMeR 0.761, WiLoR 0.767 (H1's value exactly; difference −0.006,
  CI −0.011 to −0.0001: WiLoR wins, narrowly); ≥ 0.6 on 95.0% and 94.7%;
  2D keypoint acceleration 1.5% of the box for both; orientation flips over
  45° 0.9 and 0.5 per 1,000 frame pairs.
- Coding jobs (CPU; inputs the full HOT3D archive and both VISOR archives):
  `20261010T051916Z-ab13c92e` failed (its smoke ran the whole grid and
  passed 600 s); `20261010T053138Z-f0470dcf` (gpu2, code `52f6ae5`, smoke on
  4 clips and 4 items) completed, but its report exposed two defects, so it
  is not evidence: sending the shape once also moved the wrist (the root
  was MANO's origin, and the wrist sits at a shape-dependent offset from
  it), which inflated WiLoR's reference by 8.7 px on HOT3D and let PCA-6 at
  8° pass the 5% limit with VISOR IoU 0.67; and HaMeR's HOT3D 2D mean was
  10¹⁶ px from a few joints at the fisheye camera's plane. Fixed (`997432d`:
  the root sent keeps the wrist where the frame's own shape put it; each
  joint's 2D error is capped at the image width; both checked by the
  validator). `20261010T054156Z-d9821c8e` (code `997432d`; 369 s; validator
  9/9 on smoke and full) is the result. Report (`h2 report`, code
  `997432d`): `pointstream-data/visor/h2-2026-10-09/report-997432d/`
  (`h2-report.json` `dfbee96d…10f1`, `code.json` `0c357de5…1b24`).
- Outcome, estimator. Anchor holds (gaps −3.4 and +0.9). Contests: WiLoR
  wins HInt VISOR frames, HOT3D PA-MPJPE and VISOR IoU; HaMeR wins HInt New
  Days and HOT3D acceleration. **WiLoR, 3 to 2.** Speed does not separate
  them (30 ms per hand each at batch 1).
- Outcome, coding (WiLoR; rates on VISOR's 34 items as H1, errors on HOT3D
  against the truth, reference = uncoded pose with the track's shape:
  29.6 mm, 17.3 px). H1's coding: 15.22 kbps (H1: 15.2). Alone, each
  technique on H1's: constant-velocity prediction costs more (18.1 kbps);
  quantization steps ×2 / ×4 / ×8: 11.1 / 7.7 / 5.1 kbps, 2D error against
  the truth +0.6% / +2% / +7%; PCA 24 / 12 / 6: 14.4 / 8.2 / 5.2 kbps at
  +3% / +8% / +15% 2D; centred smoothing 33 / 133 ms: 11.4 / 7.7 kbps,
  2D −0.7% / +3% (MPJPE −0.1% / −0.4%); One-Euro (2 Hz, β 0.5): 9.4 kbps,
  +10%; 15 Hz held / interpolated: 5.7 kbps, +9% / −0.3% (interpolation
  costs 50 ms); 7.5 Hz interpolated: 3.2 kbps, +2.5%, 120 ms. Choices (lowest
  VISOR rate within 5% of the reference on both HOT3D errors): 0 and 33 ms:
  every frame, steps ×4 (7.7 kbps; VISOR IoU 0.768 → 0.762); 100 ms:
  Gaussian 33 ms, 15 Hz interpolated, full pose, steps ×8, previous (1.93
  kbps, 83 ms; HOT3D 29.8 mm, 18.2 px; IoU 0.752); 267 ms: Gaussian 67 ms,
  7.5 Hz interpolated, ×4 (1.70 kbps, 184 ms; IoU 0.752). H1's screen
  (2.34 kbps): the 100 ms choice passes and is named for H3; nothing at 0
  or 33 ms passes. HaMeR's numbers are within 1% of WiLoR's on every rate.
- Reading. The hypothesis holds on coding and half holds on the estimator.
  WiLoR is chosen by the rule, but the two trade places by criterion: WiLoR
  articulates fingers better (PA-MPJPE 6.2 against 9.2 mm) and lands
  slightly better on VISOR's masks; HaMeR orients the whole hand better on
  HOT3D (median 7.6° against 14.7°, wrist-aligned MPJPE 20 against 30 mm)
  and jitters less (acceleration 5.1 against 6.7). PA-MPJPE, the field's
  standard, removes exactly the orientation a renderer needs, so H3 should
  measure both on its own renders before committing; swapping estimators
  is cheap (same boxes, same MANO, same coding). The pose stream is cheap
  once the estimator's jitter is not coded: interpolating 15 Hz samples of
  a lightly smoothed pose at 8° steps is 7.9× below H1's rate and 8% of
  SVT-AV1's bits on hands at CRF 62, at 83 ms of latency; at zero latency
  quantization alone reaches only 7.7 kbps. Smoothing and interpolation
  remove jitter rather than motion: against the truth they cost nothing or
  help, while against the uncoded estimate they look like large errors (8
  px), which is competing explanation (2): the distortion that matters is
  against the truth, and the rule measured it there. MANO's PCA subspace
  hurts accuracy for its rate (competing explanation (3) holds), and
  constant-velocity prediction amplifies the jitter it predicts.
- Scope. HOT3D is a lab with markers on the hands; VISOR has no 3D truth,
  so the coding's effect on kitchens is measured only against the uncoded
  estimate and by IoU. Rates are entropy estimates pooled over items, not
  an implemented arithmetic coder; track starts (shape, first pose) are
  counted apart, as in H1. The latency is lookahead only (no encode or
  network time).

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

### 2026-10-10 — H3a2: the texture-transfer oracle with a measured reference stream
- Question. H3a charged each reference at SVT-AV1's cost of an inter-coded
  frame, and its equal-quality saving falls from 65% to 32% to none as that
  cost doubles and triples. The references can be sent as their own
  SVT-AV1 stream, each predicted from the previous reference (user,
  2026-10-10). What does that stream cost on the hand pixels, and what
  saving remains? The reference choice is also tuned (user: "worthy
  efforts"): finer margins, eight references tried per frame instead of
  four, and CRF 62 references dropped (they lost in H3a).
- Method. H3a's oracle (`h3 oracle`, same region, flow, warps and SVT-AV1
  baseline) with CRF {41, 48, 55} references, δ ∈ {0.01, 0.02, 0.03, 0.05}
  and the 8 most recent references tried. After the window, per run:
  - The reference frames (source frames, in order) are encoded as one
    SVT-AV1 stream with B2's settings (preset 4, CRF of the run, one
    keyframe) and decoded by dav1d.
  - Its bits on the hand pixels are accounted as H1 did (libaom inspect
    `aom-inspect-v3.12.1-ps1`; each block's bits spread over its pixels).
  - Every hand-frame is rescored with these decoded references: a reference
    frame shows itself, every other frame its reference warped by the same
    flow.
  - The refresh decisions stay those made against the full-rate stream's
    pixels (the decoded references differ only by coding noise at the same
    CRF).
  - The whole window is re-encoded by the same encoder at each run CRF and
    accounted the same way, a check against H1's bits on B2's streams.
- Rate: the reference stream's hand bits over the window plus H2's pose
  stream (1.93 kbps). SVT-AV1's curve is H3a's (B2's streams, H1's bits).
- Decision rule (fixed before any run; `h3.STREAM_DECISION`, applied by
  `h3 report`).
  - H3a's rule, at the measured rate: the oracle passes if its mean LPIPS
    difference from SVT-AV1 at equal rate is below zero at some point.
  - Beside it, the equal-quality saving per item (the rate SVT-AV1 needs
    for the oracle's LPIPS, linear in log-rate between CRF points, none
    beyond them), as a median with a bootstrap interval of the mean.
  - The point with the largest median saving among those matched on at
    least 30 items is the texture-transfer fallback's operating point (PLAN
    H3), and its saving is what a generator must beat at its own rate.
- Hypothesis. References about four frames apart cost 1.5–2× an inter-coded
  frame, so the best point keeps a 30–50% equal-quality saving. Finer
  margins at CRF 55 do best.
- Competing explanation. Hands change so much between references that the
  stream costs nearly intra-frame rates (3× or more), and no point saves
  more than 20%.
- Budget. H3a measured 5.6 min per item. Encoding and accounting add about
  1 min per run, and rescoring 30 s per run: about 15–20 min per item.
  Smoke: one item, 48 frames, ≤ 600 s. Pilot: B2's four pilot items, one
  job. Full: the other 30 items in parts of at most 45 min. Ceiling 12
  GPU-hours (the GPU only for LPIPS), overnight on hosts G5c leaves free.
- Jobs (code `4f1c429`, environment `pointstream-20261006T113321Z`; every
  validator 15/15). Pilot `20261010T213310Z-134c06e8` (RTX A6000; P01_107,
  P09_106; smoke 190 s; items 14.4 and 11.6 min). Full, the other 32 items
  in 16 jobs of two, `20261010T222727Z-11fee0de` to `…222849Z-252b6963`, on
  RTX A6000, RTX 6000 Ada, Quadro RTX 8000 and Quadro GV100 (LPIPS on the
  GPU matched the CPU within 1e-3 on every class), 1,095–2,309 s each. About
  8.5 GPU-hours. Report (`h3 report`, all 34 items):
  `pointstream-data/visor/h3-2026-10-10/report-h3a2-4f1c429/h3a2.json`
  (`636eced3…2d14`). The same encoder re-encoding the whole window
  reproduces H1's hand bits on B2's streams (median ratio 1.00 at every
  CRF).
- Outcome. **With measured references the oracle fails: no operating point
  beats SVT-AV1.**
  - Every one of the 12 points is worse in LPIPS at equal rate (+0.002 to
    +0.029; every interval above zero) and saves nothing at equal quality
    (median −0.4% to −19%).
  - The closest are the finest margins: CRF 48 references with δ 0.01
    (39% of hand-frames sent, 86.5 kbps, LPIPS 0.100 against SVT-AV1's
    0.098 at that rate, better on 10 of 34 items) and CRF 55 with δ 0.01
    (54.1 kbps, +0.004).
  - Two effects compound. A reference sent in its own stream costs 1.4–2.1×
    an inter-coded frame (median per point; more as references grow
    sparser). And it is decoded worse than the same frame in the full-rate
    stream at the same CRF: CRF 48 references with δ 0.02 go from H3a's
    41.9 kbps and LPIPS 0.091 to 66.6 kbps and 0.109.
- Reading. The hypothesis held on cost (1.5–2×) and failed on outcome; the
  competing explanation held, by a different route than expected. Sparse
  references are cheap enough, but they look worse than the same frames
  coded at full rate, because SVT-AV1 spends its quality where frames
  predict each other. H3a's pass relied on references with the full-rate
  stream's quality at the inter-frame price.
  - Texture transfer with refresh does not beat coded pixels on VISOR's
    hands, even with the target's own motion. The textured-mesh fallback is
    therefore not built.
  - What could still win must spend far fewer bits on appearance. A
    generator from one reference (or none) and the pose stream, at about
    2 kbps, is H3b. The saving it must beat is zero: SVT-AV1's own curve.

### 2026-10-11 — H3b oracle: perfect appearance at the coded pose
- Question. Could a generator that draws the hand from the pose stream
  alone (about 2 kbps, 20× below SVT-AV1's lowest rate on the hand pixels)
  beat SVT-AV1 on VISOR's hand pixels, if its appearance were perfect? The
  oracle grants a perfect generator (the true hand's own pixels) and charges
  only what the pose cannot carry (PLAN H3, agreed 2026-10-11):
  - the coded pose's placement error, from estimation and from coding;
  - what MANO does not describe: the forearm and sleeve, objects in front
    of the fingers, blur outside the mesh.
- Poses. H2's VISOR fits (jobs `20261009T215322Z-188ed69c` and
  `20261009T220823Z-b309d9a2`: per hand-frame rotation, shape, translation,
  for WiLoR and HaMeR, on H1's boxes) coded per track with each model's
  100 ms choice from H2's `code.json` (WiLoR: Gaussian 33 ms, 15 Hz
  interpolated, full pose, steps ×8, 1.93 kbps; HaMeR: the same with PCA 12
  and steps ×4, 1.77 kbps), decoded with the track's median shape as H2's
  scheme sends it. The uncoded fit, each frame with its own shape, stands
  in for the true pose (VISOR has none).
- Render. Per frame, each fitted hand's MANO mesh is rasterized at the
  coded pose (painter's order by depth, both hands together). Each covered
  pixel takes the source's colour at the same surface point (face and
  barycentric coordinates) on the uncoded mesh's projection, sampled
  bilinearly; where that point falls outside the hand's mask, the nearest
  mask pixel's colour instead, so the render only ever shows hand. Rendered
  pixels never cover objects (VISOR's non-hand masks: the object stream is
  in front). The share of samples hidden in the uncoded render (self
  occlusion) is reported.
- Composites. Inside the hand region R (union of VISOR's hand masks, fill
  set, forearm included, as H3a), the source is first replaced by an
  OpenCV Telea fill of R (radius 5): the background a client would show
  where nothing is drawn. Then, per model m ∈ {WiLoR, HaMeR}:
  - `m:hand`: the render on top. Forearm, unfitted hands and every hand
    pixel the render misses stay filled (charged).
  - `m:arm`: the forearm's true pixels, then the render. A perfect arm
    model, granted free: the bound for a generator that also draws the arm.
  - `m:arm_uncoded`: `m:arm` at the uncoded pose (diagnostic: coding error
    against estimation error).
  - `m:hybrid@c`: forearm and unfitted hands as SVT-AV1's pixels at CRF c,
    then the render; rate = pose + H1's bits on forearm and hand-without-fit
    at c.
  The forearm is H1's split of each fitted hand's mask at WiLoR's uncoded
  wrist line (so H1's bits on each part apply); unfitted hands are masks
  with no H1 box. SVT-AV1 (`svt@c`): B2's streams at CRF 41, 48, 55, 62
  inside R, as H3a.
- Judges. Every method is scored per hand-frame on the same region S: R
  plus every rendered silhouette of either model, coded or uncoded,
  objects excluded.
  - LPIPS: B2's AlexNet spatial map on the full frame, averaged over S.
    Decides (user, 2026-10-10). SVT-AV1's LPIPS over R alone is kept as a
    check against H3a's numbers.
  - PSNR over S, and DISTS over S: G5c's masked DISTS (torchmetrics 1.9's
    alpha/beta on `Models/DISTS/vgg16-397923af.pth`, every layer's
    statistics weighted by S) on the crop around S padded by 64 px.
  - Temporal consistency: G5c's flicker on S over consecutive hand-frames
    (DIS flow on the source warps t−1 onto t; the method's temporal
    residual minus the source's, luma, mean absolute value).
  - On every 5th frame (judge frames), per fitted hand, for the source,
    `svt@c`, `m:hand` and `m:arm`:
    - pose fidelity: WiLoR re-run on the composite with H1's box, 2D joint
      error against WiLoR re-run on the source (share of the box);
    - realism: Inception-v3 pool features (torchvision's ImageNet weights,
      `Models/Inception/`; not the TF FID network, so values compare only
      across these methods) of the box crop grown 1.5× square; KID
      (polynomial kernel, unbiased) and FID against the source crops,
      pooled over items.
- Rates. The oracle's rate is the model's pose stream. Appearance is
  granted: a generator would add one reference per track, reported as
  tracks per minute. SVT-AV1's rate per item and CRF is H1's bits on hand +
  forearm + hand without fit (fill set), as H3a.
- Decision rule (fixed before any run; `h3b.DECISION`, applied by
  `h3b report`).
  - Gate: `wilor:arm` at the pose stream's rate against `svt@62`, LPIPS on
    S, per item; mean difference over items with a 95% paired bootstrap by
    item (1,000 resamples). Passes if the interval lies below zero: a
    perfect generator beats SVT-AV1's lowest point at a twentieth of its
    rate. Its equal-quality saving (the rate SVT-AV1 needs for the oracle's
    LPIPS, linear in log-rate between CRF points; "beyond CRF 41" if better
    than all) is reported beside it.
  - `wilor:hand`, the same test: whether the generator must draw the arm.
  - Hybrid: H3a's equal-rate test for each `wilor:hybrid@c` (SVT-AV1's
    LPIPS at the hybrid's rate, linear in log-rate, held at the end
    points); passes if below zero at some c.
  - Then: if the gate passes, H3b's generator is built (audit next); it
    must draw the arm unless `wilor:hand` also passes. If only the hybrid
    passes, the generator draws the hand and the forearm is coded. If
    nothing passes, the generator is not built and H3c is next, unless the
    gate fails only through placement (`wilor:arm_uncoded` passes) or
    `wilor:arm` matches `svt@62` on pose fidelity, DISTS and KID; then the
    outcome goes to the user before the route is dropped.
  - HaMeR's composites take the same tests, reported beside (swapping the
    estimator is cheap, H2).
- Hypothesis. `wilor:arm` passes the gate (true appearance, about 8 px of
  coding error) but would not beat CRF 55; `wilor:hand` fails, since the
  forearm is about half of R (H1) and a fill is not an arm; the hybrid
  passes at CRF 55 and 62. The coding error costs less LPIPS than the
  estimation error (`arm_uncoded` close to `arm`).
- Competing explanations. (1) The charged pixels show a Telea fill; a
  better background fill would charge less. The share of R the render
  misses and LPIPS on rendered pixels alone are reported. (2) Appearance is
  sampled at the uncoded pose, so estimation error is charged only through
  the silhouette; HOT3D's motion capture could test that proxy and is run
  only if the decision turns on it. (3) LPIPS charges a few pixels of
  displacement that a viewer would not notice without the source beside
  it; pose fidelity and DISTS are reported to show that.
- Budget. GPU (LPIPS, DISTS, WiLoR and Inception on Ada or A6000, all
  verified there): smoke ≤ 600 s (one item, 48 frames); pilot: B2's 4 pilot
  items, ≤ 45 min; full: the other 30 items in parts of 5 (about 5 min per
  item), ≤ 3 GPU-hours. Ceiling 4 GPU-hours.
- Jobs. Smoke `20261011T070541Z-cf8849a9` failed on its first hand frame
  (OpenCV's `remap` caps map sides at 32,767; fixed in `4bd4f51` with a
  test). Pilot `20261011T071143Z-d8e6dc58` (code `4bd4f51`; gpu3, RTX
  A6000; environment `pointstream-20261006T113321Z`; inputs and specs in
  `pointstream-data/visor/h3b-2026-10-11/`; B2's 4 pilot items, 240
  frames; 2,619 s, 11 min per item, peak 14.2 GiB; validator 14/14,
  including WiLoR's re-run reproducing H2's fits and the uncoded
  silhouette matching H2's IoU). Report (`h3b report`, code `4bd4f51`, on
  the pilot only): `pointstream-data/visor/h3b-2026-10-11/report-pilot-4bd4f51/h3b-report.json`
  (`448aecaf…1fec`).
- Outcome (pilot, 4 items). **The gate fails, by a wide margin, and every
  judge agrees.**

  | method | kbps | LPIPS | DISTS | PSNR dB | flicker | pose error | KID | FID |
  |---|---|---|---|---|---|---|---|---|
  | `svt@62` | 35.1 | 0.102 | 0.127 | 33.9 | 2.1 | 1.0% | −0.002 | 20 |
  | `svt@41` | 178.9 | 0.052 | 0.085 | 39.3 | 1.3 | 0.5% | −0.003 | 8 |
  | `wilor:arm` | 1.93 | 0.211 | 0.180 | 20.7 | 9.4 | 1.7% | 0.003 | 48 |
  | `wilor:hand` | 1.93 | 0.298 | 0.224 | 18.7 | 11.4 | 2.6% | 0.005 | 56 |
  | `wilor:arm_uncoded` | 1.93 | 0.168 | 0.163 | 22.8 | 6.4 | – | – | – |
  | `hamer:arm` | 1.77 | 0.248 | 0.192 | 19.8 | 9.4 | 2.3% | 0.005 | 60 |

  (LPIPS, DISTS, PSNR and flicker on S, means over items; pose error is
  WiLoR's 2D joint error against its fit on the source, as a share of
  the box, on judge frames; KID and FID on 255 hand crops per method.)
  - Gate: `wilor:arm` − `svt@62` = +0.109 LPIPS (95% CI +0.062 to
    +0.156), better on 0 of 4 items. `wilor:hand` +0.196. HaMeR is worse
    still (+0.146). Even with no coding error, `wilor:arm_uncoded` is
    +0.066 (CI +0.003 to +0.129), better on 1 item.
  - Hybrid (forearm coded by SVT-AV1): worse than SVT-AV1 at its own rate
    at every CRF (+0.059 to +0.072; at CRF 62 the hybrid is 18.1 kbps and
    LPIPS 0.166, against SVT-AV1's 0.102 at 35 kbps).
  - Equal quality: the oracle is worse than SVT-AV1's lowest point on
    every item, so the rate SVT-AV1 would need for its LPIPS lies below
    35 kbps and is not measured here.
  - Render: the coded pose moves WiLoR's joints 9.9 px (HaMeR 15.2); the
    render misses 9.7% of the hand pixels and 9.2% of what it draws falls
    outside them; 9.5% of samples land outside the region and 2.2% are
    hidden in the uncoded render. LPIPS on the rendered pixels alone is
    0.207, so the loss is in the hand itself, not in the fill.
  - Verdict by the rule: the generator is not built; H3c is next. No
    escalation applies: `arm_uncoded` fails the gate too, and `wilor:arm`
    is worse than `svt@62` on pose fidelity, DISTS and KID.
- Reading. A perfect-appearance generator placed by MANO loses to coded
  pixels at SVT-AV1's lowest rate on every judge, including those that
  ignore identity. About half of the gap remains with the uncoded pose
  (0.168 against 0.211), so it is not mainly the pose coding: MANO's
  silhouette misses or overshoots about a tenth of the hand, and LPIPS
  charges the misplaced edges and the displaced texture. The forearm
  matters as predicted (`hand` 0.298 against `arm` 0.211); the hypothesis
  that `arm` would pass is refuted. Competing explanation (1), the fill:
  LPIPS on the rendered pixels alone (0.207) is already twice SVT-AV1's,
  so a better fill (G5c's ProPainter) cannot close the gap.
- Scope. Four items; the full run was not needed for the decision (0 of
  4 items better, every judge agreeing). The realism judge uses
  torchvision's ImageNet Inception-v3, not the TF FID network, so its KID
  and FID compare methods within PointStream only; the paper must say so.
  The pose stream's rate leaves out one reference per track (163 tracks
  per minute on these items, since fits drop out between frames).

### 2026-10-11 — H3b2: SVT-AV1 down to the pose stream's rate (equal quality)
- Question. H3b's oracle (LPIPS 0.211 at 1.93 kbps) is worse than
  SVT-AV1's lowest point (CRF 62, 35 kbps, 0.102), so its equal-quality
  saving is not measured. The user prefers equal quality to equal rate
  (2026-10-11): what rate does SVT-AV1 need on the hand pixels to reach
  the oracle's LPIPS, going down towards the pose stream's rate?
- Method (`h3b oracle --low-rate`, the oracle's code with only these
  composites). The window's source is downscaled per plane (OpenCV area;
  sides rounded to even) by s ∈ {1, 2, 4, 8, 16}, encoded with B2's
  settings (SVT-AV1 preset 4) at CRF 63 (AV1's coarsest) and, at s = 4 and
  8, CRF 55; decoded by dav1d, upscaled to 1920×1080 (bicubic) and
  composited inside R as `svt@crf/s`. Rate: the stream's bits on R
  downscaled to the encode (area ≥ 0.5), by H1's libaom accounting, over
  the window. Scored exactly as the oracle (same region S, re-rendered by
  the same code; checked against the oracle's units frame by frame), with
  every judge.
- Decision rule (fixed before the run; `h3b.LOW_DECISION`, applied by
  `h3b report --low`). Per item, SVT-AV1's Pareto front over H1's four
  points and these seven (lowest LPIPS at each rate); the rate it needs
  for each oracle variant's LPIPS (linear in log-rate along the front;
  none beyond it); saving = 1 − oracle rate / that rate. The generator
  route reopens (to the user, with H3c still next) if `wilor:arm`'s mean
  saving has a 95% bootstrap interval (by item) above 50%: even granted
  perfect appearance, a generator must at least halve SVT-AV1's rate at
  its own quality to pay for the references and the generator's own
  error. DISTS's equal-quality saving is reported beside it.
- Hypothesis. SVT-AV1 reaches LPIPS 0.21 at a quarter or an eighth of the
  resolution, around 5–10 kbps, so the oracle's saving is 60–80% and the
  route reopens on rate; but at that quality (about 21 dB on the hands)
  neither is usable, which is the reading to give.
- Competing explanation. Upscaled low-resolution video is blurred, which
  LPIPS punishes more than DISTS; the two savings may disagree.
- Budget. B2's 4 pilot items; one GPU job ≤ 45 min (seven encodes per item
  on the CPU, mostly small). Ceiling 1 GPU-hour.
- Jobs. `20261011T174427Z-19719582` (P01_107, P02_02; RTX A6000; 1,542 s)
  and `20261011T174436Z-787fa437` (P03_10, P09_106; RTX 6000 Ada;
  1,063 s); code `85c217e`; validator 11/11 each, including bits falling
  and LPIPS rising with every halving of the resolution. The region S
  matches the oracle's frame by frame on all 4 items. Report (`h3b report
  --low`, code `85c217e`):
  `pointstream-data/visor/h3b-2026-10-11/report-h3b2-85c217e/h3b2-report.json`
  (`16d7fec6…a685`).
- Outcome. SVT-AV1 on the hand pixels, means over the 4 items (LPIPS and
  DISTS on S):

  | point | kbps | LPIPS | DISTS |
  |---|---|---|---|
  | CRF 62 (B2's stream) | 35.1 | 0.102 | 0.127 |
  | CRF 63 | 26.6 | 0.118 | 0.136 |
  | CRF 63, ½ | 11.5 | 0.167 | 0.163 |
  | CRF 55, ¼ | 14.8 | 0.177 | 0.173 |
  | CRF 63, ¼ | 4.7 | 0.233 | 0.207 |
  | CRF 55, ⅛ | 5.8 | 0.243 | 0.222 |
  | CRF 63, ⅛ | 1.9 | 0.293 | 0.253 |
  | CRF 63, 1⁄16 | 0.6 | 0.342 | 0.301 |

  SVT-AV1 reaches the pose stream's rate at an eighth of the resolution.
  Equal-quality saving of each oracle variant against SVT-AV1's front:
  - `wilor:arm`: SVT-AV1 needs 19.4, 4.8, 4.2 and 1.7 kbps for its LPIPS
    on the four items; saving mean 48% (95% CI 6% to 81%). **Not above
    50%: the route does not reopen.** On DISTS 54% (CI 13% to 90%).
  - `wilor:hand` (no forearm granted): 3% (CI −121% to 75%), matched on
    3 items (on P02_02 it is worse than every SVT-AV1 point). At the
    same rate it ties SVT-AV1 at an eighth of the resolution (0.298
    against 0.293).
  - `wilor:arm_uncoded`: 74% (CI 56% to 93%); `hamer:arm`: −12%.
- Reading. Measured at equal quality, the oracle's advantage depends on
  what is granted. With a free, perfect forearm and the pose uncoded, a
  generator would save about three quarters; with the coded pose about
  half, uncertainly; drawing only the hand, nothing: SVT-AV1 at an eighth
  of the resolution matches it at the same 2 kbps. The hypothesis held on
  the rate (SVT-AV1 needs about 5 kbps at the oracle's quality) and on the
  reading: at LPIPS 0.21 to 0.29 both are far below usable quality.
  Competing explanation (blur): DISTS, which tolerates blur better,
  gives the same verdict.
- Outline check (user, 2026-10-11: would sending the hand's outline be
  too costly?). On VISOR's 34 items (fill set, both hands, CPU on gpu6,
  not a fleet job), sending each hand mask's outline in every frame, with
  no prediction from the previous frame, costs 170 kbps as a lossless
  chain code (1.14 bits per boundary pixel), 158 kbps as a polygon within
  1 px and 72 kbps within 2 px (entropy of the vertex deltas). That is
  above SVT-AV1's CRF 62 on the same pixels. Prediction across frames
  would have to cut it more than tenfold to fit the pose stream's budget.
  The outline is not a cheap fix.

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

### 2026-10-12 — H3c2: pose fidelity at rate (pixels against keypoints)
- Question (user, 2026-10-12). If the decoded hand only has to carry its
  pose, with identity allowed to change (as for robot learning, the
  coding-for-machines view), H2's coded keypoints (1.93 kbps) plus a hand
  generator delivered with the decoder, and billed nothing, compete with
  the pixel coders. H3c measured DCVC-UF off the shelf at 0.7–2 kbps on R
  at ¼ and ½ scale. At equal pose fidelity, which needs fewer bits: the
  keypoints or the pixels? This decides whether a generative-hand path
  (keypoints and a client-side generator, as generative face video coding
  does with facial keypoints) is worth building beside the pixel route.
- Pose error. On every 5th window frame with a fitted hand (H3b's judge
  frames), WiLoR is re-run with H2's box on the decode inside R (source
  outside, as H3c scores) and on the source; per hand, the mean 2D
  distance over the 21 joints between the two, over the box size (H3b's
  judge). For the keypoint stream, the coded joints (H2's coding, each
  track with its median shape, decoded and projected as H3b) against WiLoR
  on the source, over the same box size. Mean over the item's judged hands.
- Points, per pilot item (B2's four). Pixels: SVT-AV1's eleven points and
  DCVC-UF off the shelf at 3 scales × 3 QPs, coded again exactly as H3c
  (rates and LPIPS must reproduce H3c's units). Keypoints: H2's choices for
  WiLoR at the 0, 100 and 267 ms budgets (7.70, 1.93 and 1.70 kbps over the
  34 items; H2's rates are dataset means, so the same rate is used on
  every item).
- An oracle for the keypoint route: it grants a perfect generator, which
  renders exactly the coded pose, the forearm and the occlusions, for
  free. A real generator only adds error.
- Decision rule (`h3c2` in `h3c.DECISION`, fixed before the run). Per item,
  the pixel envelope (SVT-AV1 and DCVC-UF off the shelf together, Pareto
  front on rate and pose error); the rate it needs for the 100 ms
  keypoints' pose error (linear in log rate along the front; none beyond
  it); saving = 1 − 1.93 / that rate. The generative-hand path **opens**
  (planned beside the pixel route, judged identity-free: pose fidelity,
  KID and flicker) if the mean saving's 95% bootstrap interval (by item)
  lies above 50% (H3b2's bar: halving the pixels' rate, to pay for a real
  generator's own error). It **closes** if the mean saving is at or below
  0: a pixel coder carries the pose as well as the keypoints at their
  rate, and also the appearance. Otherwise it is unclear and goes to the
  user. The rate at which each pixel coder matches the 0 ms keypoints'
  error, and the keypoints' rate needed to beat the pixel envelope at
  1.93 kbps, are reported beside.
- Hypothesis. Below 2 kbps the pixel coders blur the hand (H3c: LPIPS
  about 0.37 at ¼ scale), so WiLoR misreads the fingers; matching the coded
  keypoints (about 9 px, H3b) needs 5–10 kbps, a saving of 60–80%: the path
  opens.
- Competing explanation. WiLoR's error on a blurred hand may stay small
  because it leans on its hand prior, which favours the pixels; the
  keypoints' error includes H2's smoothing and 15 Hz interpolation, which a
  generator would also inherit, so it is the fair one to compare.
- Budget. One shared GPU job per pilot item (H3c's anchor jobs plus WiLoR
  on 48 judge frames) ≤ 45 minutes; about 1.5 GPU-hours.
- Validator change (after P02_02's smoke `20261012T063140Z-ae4d2017`
  failed it, before any full result). The check that WiLoR re-run on the
  source reproduces H2's fit took H3b's 0.5% of the box; per item it is
  now the median hand within 1% (the smoke's 7 hands: median 0.61%, max
  1.6%). The check guards the judge's input path (a wrong frame or box
  moves the joints by more than 10% of the box); the pose errors compare
  every coder against the same re-run, so the decision does not depend on
  it.
- Jobs (shared; validators passed): P01_107 `20261012T063131Z-f8c9c61d`
  (RTX A6000, code `04e7f60`), P02_02 `20261012T070658Z-4e75fd43` (Ada,
  `fd36976`), P03_10 `20261012T063150Z-8b5b1a07` (Ada, `04e7f60`), P09_106
  `20261012T070806Z-8306d837` (Ada, `fd36976`). Earlier attempts failed
  only at the validator (a numpy boolean it could not serialize; the
  re-run tolerance above) and are not evidence. Report (code `fd36976`):
  `pointstream-data/visor/h3c-2026-10-11/report-h3c2/h3c-report.json`
  (`7ca4965ca9ca15ea…`). Reproduction of H3c: P02_02 and P09_106 (same GPU
  class as their H3c jobs) are identical in every stream and score; P01_107
  and P03_10 ran on the other class, where DCVC-UF's CUDA extension differs
  (`sm80` against `sm89`): its streams differ by up to 0.2 kbps on R and
  per-frame LPIPS by up to 0.03 on single frames (also SVT-AV1's, so the
  metric's own numerics), item means equal to 4 decimals.
- Outcome. Pose error (mean joint distance to WiLoR on the source, over
  the box), the 100 ms keypoints at 1.93 kbps against the pixel envelope:

  | item | judged hands | keypoints at 1.93 kbps | pixels at 1.93 kbps | pixels' rate for the keypoints' error | saving |
  |---|---|---|---|---|---|
  | P01_107 | 95 | 0.76% | 0.81% | 5.6 kbps | 66% |
  | P02_02 | 49 | 1.61% | 6.6% | 89 kbps | 98% |
  | P03_10 | 83 | 0.94% | 1.6% | 7.1 kbps | 73% |
  | P09_106 | 28 | 2.26% | 3.0% | 3.2 kbps | 39% |

  Mean saving 69% (95% CI 47% to 90%): above 50%, but the interval is not,
  so by the rule it is **unclear** and goes to the user. The 0 ms keypoints
  (7.7 kbps) have 0.4–0.7% error, which the pixels reach only at 14–120
  kbps. DCVC-UF off the shelf dominates SVT-AV1 at the low end of every
  pixel front; at the keypoints' rate both blur the hand.
- Reading. Granted a perfect generator, the keypoints carry the pose for a
  third of the pixels' rate or less on every item but P09_106 (39%), and
  P02_02 (gloved hands moving fast) is where pixels lose most. The
  hypothesis (60–80%) held on the mean. What the rule leaves open is
  whether a real generator keeps enough of that margin: its own pose error,
  the forearm and the occlusions by held objects are not charged here.

### 2026-10-12 — H3e: a pretrained pose-conditioned generator renders the hand from the coded keypoints
- Question (user, 2026-10-12, after H3c2). H3c2 granted a perfect
  generator: the keypoints (1.93 kbps) carry WiLoR's pose for 69% less rate
  than the pixel envelope. Does a real, pretrained generator delivered with
  the decoder (billed nothing) keep a saving above 50% once its own pose
  error, the forearm, held objects and the boundary with the background are
  charged as they are rendered? Identity may change (constant per track);
  pose error decides (user, 2026-10-12).
- Audit (2026-10-12; methods, code and weights; no hand image generator was
  in `Models`). Criteria: conditioned on keypoints or MANO; egocentric
  hands with objects; temporal consistency; released weights and licence;
  fits a 48 GB card.

  | method | conditioning | domain | video | weights, licence | verdict |
  |---|---|---|---|---|---|
  | [JointControlVideo](https://github.com/ZhangCYG/JointControlVideo) (ECCV 2026) | first frame, 42 camera-frame 3D hand joints and intrinsics, text | egocentric hand–object (Ego4D) | yes: Wan2.1-I2V-14B-480P with LoRA and an occlusion-aware joint module, 81 frames | released, Apache-2.0 (Wan2.1 Apache-2.0) | **chosen** |
  | [FoundHand](https://github.com/arthurchen0518/FoundHand) (CVPR 2025) | reference image with its keypoints and mask, target 2D keypoints | hands, 10M images incl. Ego4D and **EPIC VISOR** | per frame, autoregressive (no video training) | released (notebooks only), GPL-3.0 code, CC BY-NC data | second: 256² crops, trained on our test source |
  | [HandRefiner](https://github.com/wenquanlu/HandRefiner) | inpainting with a mesh depth ControlNet (SD 1.5) | generated images | no | released, MIT | per-frame flicker; no temporal model |
  | HOGAN / [HOIG](https://github.com/play-with-HOI-generation/HOIG) (NeurIPS 2022) | hand and object poses | HO3D, DexYCB | no | code; needs object meshes | no object poses in our stream |
  | ManiVideo, RHanDS, HanDiffuser, Generated Reality | MANO/mesh or skeleton | various | some | no released weights found | not usable |

  JointControlVideo fits best: its input is what H2 codes (MANO joints in
  the camera frame, with the camera), it is trained on egocentric hand–object
  video that does not include EPIC-KITCHENS, and it models time. Its joint
  order is not documented (its dataset is gated); the order is settled on a
  development window below, not on the pilot items.
- Oracle. Per pilot item (B2's four), one generation over the 240-frame
  window: the source's frame 0 as the first frame (granted uncoded), and
  H2's coded 100 ms WiLoR joints (each track with its median shape, decoded
  as H3b) at window frames 0, 3, …, 240 (81 frames, about 17–20 fps; frame
  240 repeats 239), in camera coordinates with WiLoR's pinhole camera,
  resized and centre-cropped to 832×480 as the model's loader does; left
  hand in joints 0–20, right in 21–41, absent hands zero. A fixed prompt
  ("First-person view of hands handling objects in a kitchen."), sent with
  the decoder; cfg 5, seed 0, the released sampler. The generated frames are
  upscaled back to the window's resolution (Lanczos); they stand for window
  frames 3k, and frames in between are not generated (interpolation granted).
  Two compositions, inside R only (source outside R, as H3c scores):
  - **V1, hand generated** (user's variant 1): the generated pixels inside
    each coded hand's MANO silhouette, dilated by 10% of its box and
    feathered, over DCVC-UF off the shelf decoded at each of H3c's nine
    points (scales 1, ½, ¼ × QP 0, 21, 42). Rate on R = 1.93 kbps + frame 0's
    bits on R at that point + frames 1–239's bits on R outside the pasted
    masks (H3c's split by DCVC-UF's own entropy model, by area). The
    decoder has the coded pose, so the mask costs nothing.
  - **V2, hand and forearm generated** (user's variant 2): the generated
    pixels in all of R (dilated as V1), over the same decodes. Rate =
    1.93 kbps + frame 0's bits on R. R is the dataset's outline, which the
    decoder does not have (H3b2: 72–170 kbps): V2 is a bound, not a design.
- Why it bounds the generator route at this operating point. It grants the
  uncoded first frame while billing it at the coarse point, drops the hand's
  bits while DCVC-UF keeps the hand as context for its other cells, grants
  frame interpolation, and V2 grants the outline. It does not bound a
  generator fine-tuned on EPIC-KITCHENS or one at higher resolution: if it
  fails because the 480p frame loses the fingers (generated hands under
  about 100 px), FoundHand at 256² crops is the next oracle.
- Judges, as H3c2, on window frames that are both generated and judged
  there (t = 15, 30, …, 225): WiLoR re-run with H2's box on the composite;
  pose error = the mean joint distance to WiLoR on the source over the box,
  mean over the item's hands. The pixel envelope (SVT-AV1 and DCVC-UF off
  the shelf, Pareto on rate and pose error) is recomputed on the same
  frames from H3c2's saved joints (no new pixel coding). Reported beside:
  the generator's own error against the coded joints (does it follow its
  input), KID on hand crops and flicker (mean absolute frame difference
  inside R against the source's, over generated frames), for the composite
  and for DCVC-UF at the nearest rate; a visual check of every item.
- Decision rule (`h3e` in `DECISION`, fixed before the run). Per item and
  variant, the saving at equal pose error = 1 − rate / (the pixel
  envelope's rate for its pose error, linear in log rate along the front),
  maximized over V1's nine points (the encoder can choose). A pose error
  below every pixel point counts at the envelope's highest rate (a lower
  bound on the saving); one above every pixel point at its lowest rate.
  V1 decides. **Pass** (build the route: next its real coded first frame
  and the decoder's own forearm mask) if V1's mean saving's 95% bootstrap
  interval (by item) lies above 50%. **Fail** (no generator now; H3d next)
  if V1's mean saving is at or below 50%. Otherwise unclear, to the user.
  If only V2 clears 50%, that goes to the user too: it would need a
  decoder-side forearm mask, which is untested.
- Joint order and settings, before the pilot items (development window: an
  H2 VISOR item outside B2's four). One generation each with the joints in
  OpenPose order (as `mano_np` emits) and in MANO order (16 kinematic joints,
  then the tips); the order whose generated frames give WiLoR the lower
  error against the coded joints is used. The number of sampling steps is
  the released 50 if a window fits in 40 minutes on an RTX 6000 Ada,
  otherwise the most that fit, not below 30, recorded here before the
  oracle.
- Revision (2026-10-12, after the prepare smoke `20261012T102513Z-caafd73e`
  failed its depth check, before any generation). H2's joints are in WiLoR's
  camera (focal 5000/256 of the frame, 37,500 px at 1920: near-orthographic,
  hands about 20 m away). JointControlVideo's module reads inverse depth
  (clamped at 0.2 m) and orders occlusions by it, as in its metric Ego4D
  training data. The generator gets a stated camera instead: a 90° horizontal
  field of view (960 px at 1920), each hand's root at the depth that camera
  implies (0.026 × WiLoR's, about 0.5 m), MANO's metric depth offsets between
  joints, and every joint back-projected through its coded 2D position, so the
  projection the generator sees is exactly H2's coded keypoints.
- Revision (2026-10-12, project convention adopted on main, before any
  generation). Comparisons are BD-rates against an achievable anchor
  envelope. The pixel envelope (SVT-AV1 and DCVC-UF at three scales, several
  configurations) is their upper concave hull in quality (lower convex hull in
  pose error over rate), not a Pareto staircase, which understates a mixed
  anchor (G5d). V1's statistic per item is its BD-rate on pose error against
  that hull (PCHIP of log rate over the pose range both span, at least 0.1% of
  the box); where they do not overlap, the item is reported as better or worse
  at every rate, and its best point's rate at equal pose error stands in.
  V2 (one point) keeps the rate at equal pose error on the hull. The
  thresholds are unchanged: V1 passes with the mean saving (−BD-rate) above
  50% across its 95% interval and fails at or below 50% mean.
- Smoke (exclusive, Ada; `20261012T123958Z-7b531e66`, code `be04b54`, validator
  passed): 9 frames at 4 steps, then 81 frames at 10 steps on P22_107. 32.8 s
  per step at 81 frames, so the released 50 steps (≈ 27 min per window) fit
  the 40 minutes: **50 steps**, one job per generation. Peak 39.7 GB
  reserved; later generations run shared at 42,000 MiB. The T5 tokenizer's
  warning about a "Mistral regex" does not apply: the prompt's ids equal
  sentencepiece's own encoding and decode back to the prompt. Earlier
  attempts: `20261012T114547Z-8cecd105` (inputs read from the wrong level of
  the prepare archive) and `20261012T122341Z-9a1a31a1` (stopped as contended
  by another user's process). Not evidence.
- Joint order (development window P22_107, 30 judged hands; generations
  `20261012T132045Z-f518e2e7` OpenPose and `20261012T132054Z-829ed143` MANO,
  code `1b1ba9e`; judges `20261012T140150Z-f713de00` and
  `20261012T135958Z-806a82d3`, code `5204ab3`, validators passed). WiLoR on the
  generated frames against the coded joints: **OpenPose order 3.67%** of the
  box, MANO order 5.60%; OpenPose better on 27 of 30 hands (difference −1.9
  points, 95% CI −2.7 to −1.3). The pilot items use OpenPose order (as
  `mano_np` emits). For scale, on this window the coded keypoints themselves
  are 1.78% from WiLoR on the source, so the generator adds about 2 points.
  Development numbers, not evidence for the rule.
- Hypothesis. The joint module follows its input closely (the paper reports
  joint adherence), so the composite's pose error is the keypoints' (0.8–2.3%
  of the box) plus about half a percent, and V1 saves 50–70% (frame 0 and
  the forearm cost most of its rate).
- Competing explanations. At 480p the generated hands are 60–150 px tall
  and WiLoR misreads upscaled fingers, so the error grows past the pixel
  coders' at a few kbps; or the 81-frame clip drifts from the first frame
  (identity or objects change), which pose error does not see but KID,
  flicker and the visual check do; or frame 0's bits alone exceed the
  savings, since every window pays one.
- Budget. CPU: the Wan2.1 backbone and JointControlVideo weights (about
  80 GB) into `Models`, one new environment. GPU, RTX 6000 Ada: smoke ≤ 600 s
  (few frames and steps; exclusive, the model's first run on the class);
  joint-order check 2 jobs; oracle 4 jobs, one per item, each ≤ 45 minutes;
  about 5 GPU-hours.
- Jobs (Ada; validators passed). Generations (OpenPose order, 50 steps, code
  `9a67692`): P01_107 `20261012T141546Z-18c30e4a`, P02_02
  `20261012T141556Z-5b185f3a`, P03_10 `20261012T141605Z-ff439104`, P09_106
  `20261012T141614Z-326ad02f`. Judges (nine DCVC-UF points each): P01_107
  `20261012T145802Z-69fab945`, P02_02 `20261012T145812Z-5e63a0e4`, P09_106
  `20261012T145830Z-93a84904` (code `9a67692`), P03_10 `20261012T152219Z-20d57d2d`
  (code `e5e212c`, which changes only the spec writer). P03_10's first judge
  `20261012T145821Z-f99e456a` stopped when DCVC-UF's two decodes of one stream
  disagreed (scale ½, QP 42, on a GPU shared with another user's process); the
  resubmission decoded deterministically, as in H3c. Not evidence. Report (code
  `8b17aa5`, from the judges' and H3c2's archives on the server):
  `pointstream-data/visor/h3e-2026-10-12/report/h3e-report.json`
  (`49af833b24604d67…`).
- Outcome: **fails**. V1 against the pixel hull, at equal pose error:

  | item | judged hands | V1 | V1's pose error | keypoints' | generator vs its input | V2 |
  |---|---|---|---|---|---|---|
  | P01_107 | 30 | −171% (worse at every rate; flat) | 1.77% from 2.06 kbps | 0.77% | 3.0% | −175% |
  | P02_02 | 16 | **+29%** (BD-rate) | 5.2–5.5% | 1.67% | 4.9% | +76% |
  | P03_10 | 26 | −197% (worse at every rate; flat) | 2.0% from 1.95 kbps | 0.97% | 2.7% | −190% |
  | P09_106 | 9 | −848% (BD-rate) | 5.5–6.5% | 4.5% | 10.7% | −610% |

  V1's mean saving −297% (95% CI −679% to −21%), at or below 50%: **fail**.
  V2 (granted the outline) −225% (CI −501% to +9%). The pose error of V1 is
  flat across its nine points: it is the generator's, whatever the forearm
  and objects cost. The pixel hull reaches it at 0.3–1 kbps on P01_107 and
  P03_10. Flicker (luma, against the source's own change, between generated
  frames): V1 5–23, DCVC-UF 3–7 on the same frames. KID on hand crops, pooled:
  V1 0.008–0.009 (interval to 0.03), DCVC-UF 0.0 at QP 42 full scale to 0.031
  at QP 0 ¼ scale: realism is not where it loses.
- Visual check (the judges' crops at one judged frame per item, and the
  generated videos). The generator follows the joints but not the camera: it
  receives the first frame and the hand joints, not the head's motion, so the
  generated scene drifts from the real one (a toaster comes into view on
  P03_10; on P09_106 the frame shows a towel and a bowl that are not there).
  Pasted inside the coded silhouette, its hands then sit in a different scene
  from the coded pixels around them: smeared boundaries (P09_106), a forearm
  with a tattoo the source does not have (P01_107), which WiLoR misreads.
  P02_02, the one positive item, has a nearly still camera and fast gloved
  hands that the pixel coders blur.
- Reading. H3c2's margin (keypoints 69% cheaper with a perfect generator)
  does not survive a real one: its hands land 2.7–10.7 points (of the box)
  from the joints it is given, so the composite's pose error is 1.0–3.7
  points above the keypoints' own (0.8–4.5), and DCVC-UF reaches that
  fidelity below 1 kbps on three of four items. The competing explanation that held is the
  drift, not the 480p resolution (the generated hands are large enough for
  WiLoR on P01_107 and P03_10, where the error is still 3 points). What would
  change next, and needs its own oracle: a generator conditioned on the
  decoded pixels around the hand each frame, rather than on a first frame
  alone (the proto-hand-conditioned DCVC-UF candidate in PLAN, or hand
  inpainting into the coded frames), or one conditioned on camera motion
  (Generated Reality conditions on head pose). Not built now: H3d is next.

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

### 2026-10-13 — H4a: grip pose on HOT3D (is a held object still in the hand's frame?)
- Question ([PLAN H4](../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08),
  item 1). H1 found that no 2D warp of one reference view carries a handled
  object for more than 0.02 s (median), and that handled objects are 27–29%
  of the foreground's SVT-AV1 bits. Grip pose proposes a 3D motion model
  instead: while an object is held, its pose is the hand's pose (already in
  the stream) times one fixed transform G, resent only when the grip
  changes. This entry measures, on motion capture, whether that model holds
  and how often G must be resent. Bits are not the question: any 6DoF
  pose costs at most ~1 kbps however it is sent (6 values per frame), against
  41.3 kbps of SVT-AV1 on handled objects at CRF 62 (H1). The question is
  whether an object's motion can be *derived from the hand's*, so that the
  encoder needs an object pose only at the few frames where G changes
  rather than a 6DoF tracker on every frame (VISOR has no object models
  and no object-pose labels).
- Data: HOT3D-Clips `train_aria` (`Datasets/HOT3D`, 1,516 clips of 150
  frames at 30 fps; Aria only, as H2), 12 clips per participant (all 9, so
  108 clips), content-blind (lowest sha256 of
  "pointstream-h4:hot3d:<clip>"). Per frame: `objects.json`
  (`T_world_from_object` per object), `hands.json` (UmeTrack
  `T_world_from_wrist` per hand, motion capture), `cameras.json` (the RGB
  fisheye `214-1`, 1408²). Object geometry: the decimated meshes in
  `object_models_eval` (their vertices; no texture is used). Code
  `experiments/hot3d/h4a.py` (to be written after this entry).
- Method.
  1. Error of a predicted pose P̂ against the true pose P of an object at a
     frame: the mean over the mesh's vertices of their distance in the
     image `214-1` when projected with P̂ and with P (the toolkit's
     fisheye camera; px), and the same distance in 3D (mm) beside it.
  2. Predictors, each starting from a *refresh* (a frame at which the true
     pose is sent, exactly; the oracle's simplification): *world* (the
     object stays where it was in the world: P̂(t) = P(r)); *camera* (it
     stays where it was in the camera, H1's copy in 3D); *grip-left* and
     *grip-right* (P̂(t) = W(t) · G, G = W(r)⁻¹ · P(r), W the hand's
     motion-capture wrist pose). A predictor fails at the first frame whose
     error exceeds τ, or where its hand is missing; a refresh is sent there.
  3. Coders, replayed over every object of every clip: *world only*,
     *grip only* (the nearer hand at the refresh), and *hybrid* (at each
     refresh the predictor that lasts longest; an oracle choice by
     look-ahead, a 2-bit mode). A frame is *held* when it is not a refresh;
     a reference's *life* is the time from one refresh to the next.
  4. Scored frames: an object's *moving* frames, where its mean vertex
     displacement per frame (central difference over ±2 frames) exceeds
     0.5 mm, dilated by 5 frames. Objects at rest are carried by *world*
     anyway; the moving frames are where the hand must explain the motion.
     Lives are cut at the end of a moving segment and of the clip; cut lives
     count at their observed length (conservative), and their share is
     reported.
  5. Reported beside: the noise floor (*world*'s error on objects that never
     move, frame to frame), the grip's drift within a life (rotation and
     translation of W(t)⁻¹ P(t) against G), how often G changes (refreshes
     per second of moving time, per object), τ ∈ {1, 2, 4} px, and the rate
     of refreshes (6 values at 16 bits plus the mode) against SVT-AV1 on
     handled objects.
- Decision rule (fixed before any run; τ = 2 px, means over clips of
  per-clip values, pooled over each clip's moving frames). Grip pose is
  worth building on when the *hybrid* coder, on moving frames, (a) holds
  ≥ 70% of them and (b) has median reference life ≥ 0.5 s (H1's thresholds
  for a rigid part), and (c) *grip* is what explains them: the hybrid's
  median life is at least twice *world only*'s on the same frames. Rate
  ((refreshes × 98 bits) per second) must be ≤ 10% of SVT-AV1's 41.3 kbps
  on handled objects (H1's (c)); it is reported but not expected to bind.
  Pass: grip pose is a viable motion model for held objects, and the next
  step is an appearance oracle (the object's scanned model rendered at the
  true pose on HOT3D, against the pixel coders at equal quality), since a
  pose alone renders nothing. Fail: grip pose is dropped and H4 moves to
  item 2 (a bank of reference views on VISOR). Unclear (passes at 4 px but
  not 2 px, or (a)–(b) pass without (c)): reported, and the user decides.
- Hypothesis: rigid objects in HOT3D are held in a stable grip. Within a
  grasp G drifts by a few mm and a few degrees; refreshes come at grasp,
  release and regrasp, about once per second of motion. The hybrid holds
  ≥ 85% of moving frames with median life ≥ 1 s at 2 px, and *world only*
  lives a few frames.
- Competing explanations. (1) Objects move in the fingers, not with the
  palm (turning a mug, pouring, in-hand rotation), so G drifts past 2 px
  within a few frames; the drift measure separates a stable grip with
  finger slip from no grip at all. (2) The hand's and the object's motion
  capture each have their own noise, so G jitters even in a fixed grip; the
  noise floor and τ = 4 px test whether 2 px is above it. (3) HOT3D's lab
  tasks with rigid objects favour grip pose against VISOR's kitchens (food,
  cloth, deformable packaging); a pass is an upper bound for VISOR, which
  this entry does not test.
- Budget: CPU only (`"device": "cpu"`, 16 threads); reading a clip's poses
  means streaming its ~100 MB tar from NFS. Smoke: 2 clips, ≤ 600 s. No
  pilot: nothing is fitted, and τ is swept in the same pass. Full: 108
  clips, ≤ 45 min. Ceiling 2 CPU-hours.
- Jobs: `20261013T064014Z-04a44b1c` (code `b003be8`, environment
  `pointstream-20261009T214002Z`; CPU only, 16 threads on gpu1; inputs
  `pointstream-data/hot3d/h4a-2026-10-13/inputs-26fb572756ed.json`, spec
  `h4a.json` beside it). Smoke 2 clips (3 s), full 108 clips of 9
  participants (15 s of compute; 8.5 min with staging). Validators passed on
  the smoke and, locally, on the full stage's `published.tar`
  (`939b43ef…48bb3`; 8 checks: every clip published, eval meshes within
  2 mm of `models_info` (the keyboard's decimated mesh is 1.03 mm off, so
  the bound set before the run at 1 mm was raised to 2 mm after the dev
  run), true poses projecting inside the image, resting objects within 1 px,
  every scored frame held or refreshed once). Report (`h4a report`):
  `pointstream-data/hot3d/h4a-2026-10-13/report-b003be8/h4a-report.json`
  (`cb3b135a…463d`).
- Outcome (means over the 108 clips, all with moving frames; 189 moving
  objects, 15,558 scored object-frames). Noise floor: resting objects hold
  within 0.08 px (median; 95th percentile 0.55 px), so 2 px is well above
  the motion capture's noise. On moving frames:

  | coder | τ | held | median life | frame-weighted life | refresh rate |
  |---|---|---|---|---|---|
  | world | 2 px | 18.4% | 0.035 s | 0.068 s | 2.3 kbps |
  | grip | 2 px | 37.8% | 0.040 s | 0.103 s | 1.8 kbps |
  | hybrid | 2 px | 49.4% | 0.052 s | 0.129 s | 1.4 kbps |
  | hybrid | 4 px | 69.0% | 0.098 s | 0.246 s | 0.9 kbps |

  The hybrid's held frames come from the grip on 53% (left 22%, right 30%)
  and from the world on 46%. Grip spread over a segment's moving frames
  (201 segments, quartiles): G turns by 4.0 / 10.1 / 22.6° and shifts by
  6.7 / 13.1 / 30.4 mm from its middle frame (median over frames), against
  the world pose's 4.0 / 21.0 / 53.9° and 19 / 82 / 155 mm. Per-clip held
  share of the hybrid at 2 px: 28% / 50% / 70% (10th, 50th, 90th
  percentile).
- Decision (by the rule; `h4a-report.json`): **grip pose fails.** At 2 px
  the hybrid holds 49% of moving frames (rule 70%) with a median life of
  0.05 s (rule 0.5 s), 1.5× *world only*'s (rule 2×); at 4 px it holds
  69% with 0.10 s. Rate passes (1.4 kbps against 4.1) and does not bind.
  By the rule, grip pose is dropped and H4 moves to item 2, a bank of
  reference views on VISOR.
- Reading. The hypothesis is refuted: a held object is not still in the
  wrist's frame. The grip explains most of the object's translation
  (median shift 13 mm against 82 mm in the world) but only half of its
  rotation (10° against 21°); these are medians of per-segment spreads,
  not a variance split. What remains, mostly rotation (10° turns a 10 cm
  object's edge by tens of pixels at arm's length), is far above what a
  rendered object can absorb: a reference carried by the wrist dies in a
  tenth of a second, no longer than H1's 2D warps on VISOR. Competing
  explanation (1) holds: objects move in the fingers (regrasps, in-hand
  turning, two-handed handling such as the keyboard in the dev clip) and
  the grip drifts continuously, not in a few discrete changes. (2) does
  not hold: the noise floor is 0.08 px. (3) makes this a stronger
  negative: HOT3D's rigid objects in a lab are the favourable case, and
  the wrist is the motion-capture truth, not an estimate.
- Scope. The hand frame is the wrist (UmeTrack's); a frame fitted to the
  fingertips in contact, or a pose predicted from the whole hand's
  articulation, was not tested and could hold longer. The test is
  geometric only (vertex reprojection), so it bounds any renderer that
  needs the object's pose to within 2–4 px. Clips are 5 s long, which
  cuts 4% of lives; the decision does not rest on long lives.

### 2026-10-13 — H4b: a bank of reference views for handled objects (oracle, VISOR)
- Question ([PLAN H4](../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08),
  item 2; H4a dropped grip pose). H1 carried each object from one
  reference by chained homographies and refreshed it where it stopped
  holding: handled objects held on 49.9% of frames, median reference life
  0.02 s. A bank keeps the views already sent: each frame is predicted from
  whichever view predicts it best, and a new view is sent only when none
  does. An object turned back, or reappearing from under a hand, then costs
  no new view. This oracle bounds how much a bank can save before any bank
  is coded: if even an unlimited bank, choosing views with hindsight,
  needs a new view about as often as H1's single reference, the component
  is not built and handled objects stay pixels (item 3).
- Data: evaluation set v2 (34 windows of 240 frames), fill masks
  (`visor_dense_sam_fill`, as H1), H1's objects: each object label's union
  of instances per frame, *handled* when a human labelled it in contact
  with a hand at a keyframe bounding or inside the window (H1's
  `contact_objects`). Frames decoded as H1 (sparse-JPEG gate), full
  resolution. Code `experiments/visor/h4b.py` (to be written after this
  entry), reusing H1's flow, homography fit and hold test.
- Method. Per object track, frame by frame, as H1: DIS flow (medium preset,
  half resolution), a homography fitted to it in the object's mask
  (RANSAC 2 px). A view r *predicts* frame t through a homography H
  (r → t) when, warping view r's pixels onto the object's mask at t, at
  most 20% of the mask is uncovered and PSNR on the covered pixels is
  ≥ 30 dB (H1's test; 33.75 dB beside). Two ways to reach a view:
  *chained* (the frame-to-frame homographies since r, broken where a step
  has no fit or the object is absent, as H1), and *direct* (SIFT features
  inside each mask, dilated 8 px; ratio test 0.75; homography by RANSAC
  3 px with ≥ 15 inliers; the 4 views with most inliers are tried). Coders:
  1. *single*: H1's single reference, chained only (must reproduce H1's
     recorded counts; the correctness check);
  2. *single + direct*: one reference, reached by either way;
  3. *bank-4*: the last 4 views used, either way;
  4. *bank*: every view sent so far, either way (the oracle).
  At each frame the coder takes the view and way with the highest PSNR
  among those that predict t; if none does, t becomes a new view. When a
  coder has more than 4 candidates (chained views plus direct tries), they
  are first screened on every 4th mask pixel in each direction and only
  the 4 best get the full test (added with the code, before any run: an
  unlimited bank can hold a hundred live chains; *single* always gets the
  full test, as H1). Environment and inputs are H1's
  (`pointstream-20261006T113321Z`), so *single* is checked against H1's
  recorded chains per track (within 2% of frames on ≥ 95% of tracks). Per
  track: frames *held* (predicted) and the new views, with the interval
  between consecutive new views (s). Other objects are run and reported
  beside; the decision is on handled objects.
- Decision rule (fixed before any run; 30 dB; means over the 34 items of
  per-item values pooled over the item's handled tracks). The bank is
  worth building when the *bank* oracle (a) holds ≥ 70% of handled-object
  frames, (b) has a median interval between new views ≥ 0.5 s (H1's
  thresholds for a rigid part), and (c) the bank is what helps: it sends at
  most half the new views of *single + direct* (a single reference that
  can also re-register). Pass: build the bank component (finite views,
  coded views and residual) and score it on handled-object pixels against
  SVT-AV1 and DCVC-UF at equal quality (BD-rate per metric against their
  envelope). Fail: handled objects stay coded pixels (item 3), and H4 is
  done. Unclear ((a) and (b) pass without (c), or a pass at 30 dB that
  fails at 33.75 dB): reported, and the user decides.
- Hypothesis: the bank helps where H1's error split says it can, with new
  pixels (17% of the homography's error on handled objects) and occlusion
  by hands (10%), but not with appearance change (28%), deformation (27%)
  and motion blur (18%). Held frames rise from 50% to about 60%, the
  interval between new views stays near 0.1 s, and the bank fails (a) and (b).
- Competing explanations. (1) H1's single reference fails because its chain
  breaks (no fit for a frame, the object absent for a frame), not because
  the appearance changes. Then *single + direct* recovers most of the gain
  and (c) fails; the bank itself is not what helps. (2) 30 dB on covered
  pixels is the wrong bar against coded pixels; 33.75 dB (B2's foreground
  PSNR at SVT-AV1 CRF 62) is reported beside. (3) A pixel codec already
  predicts from several past frames (SVT-AV1 up to 7 references, DCVC-UF
  from its decoded past), so a pass bounds the representation, not its
  gain over the codecs; the BD-rate stage decides that.
- Budget: CPU only (`"device": "cpu"`, 16 threads); decode, flow and SIFT
  per item. Smoke: one item, 48 frames, ≤ 600 s. Pilot: B2's 4 pilot items,
  ≤ 45 min, which also times an item. Full: the other 30 items, in parts of
  at most 45 min each. Ceiling 8 CPU-hours.
- Jobs: pilot `20261013T074546Z-821ee185` (code `2880e27`, environment
  `pointstream-20261006T113321Z` as H1; CPU only, 16 threads on gpu4;
  inputs `pointstream-data/visor/h4b-2026-10-13/inputs-7f2aebab5a6a.json`,
  spec `h4b-pilot.json` beside it). Smoke P01_107, 48 frames (98 s); full
  stage B2's 4 pilot items at 240 frames (1,260 s; per item 188–1,257 s,
  the bank's candidates dominating). Validators passed on the smoke and,
  locally, on the full stage's `published.tar` (`e14d7f86…`; 6 checks,
  including *single* against H1's recorded chains: exact on all 28
  track-thresholds). Report: `…/h4b-2026-10-13/report-pilot-2880e27/h4b-report.json`
  (`1e10dfb0…`). Not evidence: `20261013T074250Z-24ac003d` (the run
  finished its smoke; the validator was given a `{staged:…}` placeholder,
  which the fleet does not expand for validators).
- Outcome (pilot, means over the 4 items; 6 handled tracks, 1,342
  handled-object frames):

  | coder | held at 30 dB | median interval | new views per s | held at 33.75 dB |
  |---|---|---|---|---|
  | single (H1) | 54.6% | 0.032 s | 31.8 | 26.2% |
  | single + direct | 56.2% | 0.032 s | 31.0 | 26.4% |
  | bank-4 | 56.7% | 0.032 s | 30.6 | 26.4% |
  | bank (every view) | 55.9% | 0.032 s | 31.1 | 26.3% |

  The unlimited bank sends 1.00× the new views of *single + direct*
  (rule ≤ 0.5×). Per handled track (frames; new views single / single +
  direct / bank): soy milk 240; 140 / 125 / 130, glass 240; 77 / 77 / 77,
  tap 240; 60 / 59 / 56, pancake 240; 63 / 63 / 63, pancake package 240;
  128 / 128 / 128, fork 142; 78 / 78 / 78. Direct registration reaches 8%
  of the bank's held frames. Other objects behave the same (held 66.7% →
  67.3%, ratio 1.02).
- Decision (by the rule, on the pilot): **the bank fails every check**:
  held 56% (rule 70%), median interval 0.03 s (rule 0.5 s), and it sends
  as many new views as one re-registering reference (rule half). The rule's
  unit is the 34 items; the other 30 were not run, because no mean over
  them could pass: (c) needs the bank to halve the new views, and on every
  pilot track it changes them by 0–7%. H4b stops at the pilot; handled
  objects stay coded pixels (H4 item 3).
- Reading. The hypothesis held, and more strongly than predicted: the bank
  adds nothing, not even the ~10 points expected from occlusion and new
  pixels. What *held* demands: the warped view alone, with no residual,
  within 30 dB of the frame; a frame that is not held falls back to its
  pixels. In frames, the median gap between new views per handled track
  is 1, 2, 4, 3, 2 and 2 (soy milk, glass, tap, pancake, pancake package,
  fork); the mean over items of 0.032 s is about two frames at 50–60 fps,
  so these objects need their pixels on about every other frame, and the
  held frames are below the codec's own quality (at 33.75 dB, B2's
  foreground at CRF 62, 26% are held). The coders agree because they share
  the latest view and differ only in older ones: a view sent earlier
  almost never predicts a later frame better than the latest view does
  (the unit tests show the bank does reuse views when an object turns
  back and forth, and *single* reproduces H1 exactly): within a 4.8 s window the objects' appearance
  changes one way (turning, being cut, filled, lit differently, blurred),
  so the most recent view is always the best one and older views are not
  revisited. Competing explanation (1) does not hold either: direct
  registration recovers 1.6 points over H1's chain, so broken chains are
  not why the reference dies. What fails is appearance: at 30 dB a view
  predicts the next frame on half the frames and is resent every 0.03 s,
  which is what a pixel codec's own inter prediction already does.
- Scope. 4 pilot items, 6 handled tracks. Views are warped by homographies;
  a richer warp (dense flow, a 3D model) was H1's flow bound (1–2 dB over
  the homography) and is not a bank question. The screen (4 candidates
  kept from a 1/16-pixel test) could miss a view; it never applies to
  *single*, and *single + direct* differs from the bank by under 1 point.

### 2026-10-16 — H5: the first working PointStream (pilot)
- Question (PLAN H5): with the components chosen so far, does PointStream,
  one bitstream with every byte counted, beat B2's anchors on evaluation
  set v2, and how long do its encoder and decoder take? This entry covers
  the pipeline and its pilot on B2's 4 pilot items (P01_107_0000003049,
  P02_02_0000006080, P03_10_0000000616, P09_106_0000006303); the 34-item
  run is planned from it and waits for a settled pipeline.
- Components (as chosen; nothing is fitted here).
  - *F*, the foreground: B2's dataset foreground, the union of B1b's
    `visor_dense_sam_fill` instances, as agreed with G (PLAN G6). Its
    classes: hands (VISOR's hand classes, with the forearm; R) and objects
    (every other instance). F and the class map are **granted** to the
    decoder in this run: not in the rate, measured as their own line (each
    window's 3-level map at 1080p coded losslessly with FFV1, and with lzma
    over the frame-to-frame XOR of the packed bits; the smaller is
    reported).
  - *Background*: G6's per-window pipeline (`experiments/background/g6.py`,
    merged from `claude/g6-background-h5`) with G6 item 1's settings: the
    whole 1080p frame with F filled exactly by ProPainter, coded by two
    routes. Scene model: recipe L trained on the window's own-video footage
    masked like F (`footage --mask exact`), coded at 540p (QP 0, 13, 25, 38,
    upsampled at the client) and 1080p (QP 13, 25, 38, 50, 63). Enhancer:
    SVT-AV1 at 1080p (G5's preset 4, CRF 32, 40, 48, 54, 59, 63) and G5d's
    enhancer fitted per CRF on the window's own native-rate footage. Each
    venue model is shipped once (G6's interface): not in the rate, its size
    reported. The plain SVT-AV1 background (the enhance jobs' anchor) is an
    ablation.
  - *Foreground stream* (new, `src/pointstream/`): one DCVC-UF HT-L stream
    of the window at a quarter, half or full resolution (H3d's coder),
    off the shelf. Its input is the source with everything outside
    K = F dilated by the 8 px ring plus two coded pixels (2 × scale px, a
    bicubic tap's reach) replaced by
    a smooth push-pull extrapolation of K's own pixels (as MPEG-4's
    arbitrary-shape coding padded outside the shape), so DCVC-UF spends
    little outside F and no edge forms at K's border. H3c's rate on R was
    an attribution inside whole-frame codes; this stream is real and every
    byte of it is billed. Points: ¼ and ½ at QP 0, 21, 42 (each with H3d's
    G network for that point) and full resolution at QP 21, 42 (no
    network). *Amended after the first full score stage (P02_02, CRF
    32–48, `20261016T171116Z-d058a70a`), before any comparison with an
    anchor:* the foreground capped the hull's quality (LPIPS on F 0.12 at
    its best point, against 0.04–0.07 on the background), so QP 63 at ½
    and full resolution (no network: G has none for QP 63) are added, as
    separate foreground jobs per item. Decoder: bicubic upsampling to 1080p, then per class an
    enhancer slot: hands get G on R's box, objects the identity (the slot
    H4c may fill).
  - *Compositing* (G6): out = α·fg + (1 − α)·bg, α = 1 on F and falling
    linearly to 0 over 8 px outside F (Euclidean distance); bg elsewhere.
  - *Bitstream* (`src/pointstream/bitstream.py`): a header (magic, version,
    size, frame rate, frame count) and tagged, length-prefixed segments:
    background (route, coded size, codec, the codec's own stream), each
    foreground stream (scale, enhancer ids per class), and the venue-model
    and decoder-network references (sha256 prefixes). The rate is the
    container's whole size. The mask segment is absent here (granted).
- Scoring: B2's harness (`src.codecs.quality.score`, B2's regions):
  weighted PSNR (0.7 fg + 0.3 bg, per frame then the mean) with
  `visor_dense` (without the fill) and `visor_dense_sam_fill` (with it),
  weighted LPIPS likewise, and weighted DISTS (masked DISTS per region,
  `quality.masked_dists`, on every 4th frame for every method, to keep a
  pair's scoring near 15 s) beside; whole-frame PSNR too. B2's anchors (SVT-AV1
  preset 4, CRF 41–62; DCVC-UF HT-L, QP 27–63) are rescored from B2's
  published streams by the same code, so DISTS exists for them (SVT-AV1:
  B2's streams decoded; DCVC-UF: B2's points re-encoded on the same GPU
  class, since a stored DCVC-UF stream never decodes in a later job,
  docs/fleet.md; byte equality with B2's stream reported); the
  rescoring must reproduce B2's recorded weighted PSNR within 0.02 dB on
  every point (a check, not a choice). Every job that codes or decodes
  DCVC-UF runs on the RTX A6000 (unreliable on contended Ada, K5).
- Rate allocation: PointStream's curve per item and metric is the upper
  concave hull (`g5.envelope`) over every (foreground point, background
  point) combination of both routes. Anchors are each codec's upper concave
  hull. BD-rate: PCHIP of log rate over the overlapping quality range
  (`g5.bd_rate`'s method), against SVT-AV1, DCVC-UF and their pooled hull;
  "better/worse at every rate" when the curves do not overlap. Also
  reported: the hull of each background route alone, the plain-SVT-AV1
  background, and on the hull's points the split of bytes between
  foreground and background.
- Timing (one exclusive job, RTX A6000 on gpu3 with 32 cores, B2b J2's
  host class): per pilot window, encoder = ProPainter's fill + background
  encode (scene model's DCVC-UF encode; SVT-AV1 preset 4 on 32 threads) +
  the foreground's padding and DCVC-UF encode; decoder = background decode
  (DCVC-UF with the scene model; dav1d plus the enhancer) + foreground
  decode, upsampling and G + compositing. GPU stages synchronized, median
  of 3 runs. Not timed: F (granted: dataset masks) and venue-model training
  (offline, once per venue; G6's training jobs' times reported beside).
- Decision rules, fixed now.
  (a) *PointStream beats the anchors* if, against the pooled hull of
  SVT-AV1 and DCVC-UF, its mean BD-rate on weighted LPIPS
  (`visor_dense_sam_fill`) is −10% or better with at least 3 of 4 items
  below 0. Weighted DISTS, both weighted PSNRs and whole-frame PSNR are
  reported beside with the same rule's numbers, and a sign disagreement
  with LPIPS is stated. A pass makes the pipeline the candidate for the
  34-item run (planned, not run here).
  (b) *Where a failure lies* (or a margin comes from): the same BD-rates
  with the background replaced by plain SVT-AV1 (the background's share),
  and the region scores at matched rate (fg and bg separately).
  (c) *Equal-time SVT-AV1 preset* (B2b's decision, user 2026-10-07):
  PointStream's encode time per window is the median over the 4 pilot
  windows, for each background route; the slowest route's median sets the
  time, and the preset is the slowest in J2's table whose median encode
  time is at or under it (J2: preset 1 80.5 s, 2 46.2 s, 3 30.3 s, 4 18.7 s,
  5 12.8 s, 6 9.1 s, 8 4.6 s, 10 2.7 s, 12 1.94 s). This gives the anchor
  the most time PointStream uses.
  (d) *The bitstream holds*: every combination's container parses back to
  its segments byte for byte and its rate is its length; on one
  combination per item and route, the streams unpacked from it are the
  published ones whose decodes were composited (AV1 rebuilt into IVF with
  the same temporal units). The timing job decodes from containers.
- Hypotheses. (a) passes on LPIPS and DISTS by a wide margin (−30% or
  more against the pooled hull): the background models win −33% to −97%
  on LPIPS_V against SVT-AV1 (G6) and G wins −22% on the hands; weighted
  PSNR loses (+20% or worse), since both enhancers and the scene model
  trade PSNR for perceptual quality. (c) ProPainter dominates the encode
  (tens of seconds per window), so the equal-time preset is 3 or slower,
  a stronger anchor than B2's preset 4.
- Competing explanation for a failure of (a): splitting the frame costs
  more than the parts save. The foreground stream pays for F's boundary
  and its own intra frame; the background pays for the fill's seams; and
  the 8 px ring blends two decodes. B2's weighted LPIPS is 70% foreground,
  where the gain measured so far (G, −22%) is smallest and was measured on
  hands only, not objects. Measured by (b).
- Budget. Background for the 4 windows (G6's code, 6 jobs per window:
  scene footage, training, format; native footage, two enhance jobs): 24
  jobs, about 12 GPU-hours (G6's pilot: 5.2 for 2 windows with failures).
  Foreground: 4 jobs, about 2. Scoring and B2's rescoring: 8 jobs, about 5.
  A smoke on G6's published P06_03 window (its background exists): about 1.
  Timing: 1 exclusive job, under 45 minutes. About 21 GPU-hours, ceiling
  28; every job under 45 minutes, shared GPUs except timing.
- Runs (shared RTX 6000 Ada and RTX A6000; code revisions per job from
  each job's `ready.json`, recorded in its result; every full stage
  validated locally with the module's own validator: G6 stages 9–10 of
  9–10 checks, foreground 9/9, scoring 8/8, anchors 11/11).
  - Background (`experiments/background/g6.py`; per window: scene
    footage `--mask exact`, recipe-L training, format at 540p QP 0–38 and
    1080p QP 13–63; native footage, enhance CRF 32–48 and 54–63).
    P01_107: train `20261016T145531Z-1fe7ccb5`, format
    `20261017T093325Z-add15e57`, enhance `20261016T090743Z-5ebf7dea`,
    `…090801Z-18e827b5`. P02_02: train `20261016T092029Z-709f45bd`,
    format `20261016T145558Z-40cba9fc`, enhance `…090505Z-c04cf958`,
    `…090523Z-1d2c91cc`. P03_10: train `20261016T145706Z-3c7a9b4c`,
    format `20261017T093336Z-e59db56b`, enhance
    `20261016T145616Z-b65e6889`, `…145634Z-99b8bcb9`. P09_106: train
    `20261016T145724Z-feff65ee`, format `20261017T093347Z-2ba6b75d`,
    enhance `20261016T091551Z-14a0503c`, `…091616Z-95b1e7e8`. Every
    scene model converged (last fifth within 1.1%). Scene route
    10–5,130 kbps, enhancer route 126–7,470 kbps on the background.
  - Foreground (`h5 foreground`): P01_107 `20261016T185704Z-d332b01b`
    (10 points); P02_02 `20261016T091421Z-58f2a695` (8) and
    `…180928Z-ec25bbb9` (QP 63); P03_10 `…091448Z-b1805c33` and
    `20261017T093212Z-bfc50ee1`; P09_106 `20261016T091510Z-ff6fbf22`
    and `…181001Z-889ca36a`; P06_03 (smoke window)
    `20261017T094147Z-5770b26c`. 3.5–2,030 kbps on F.
  - Scoring (`h5 score`, 150 pairs per item in shards of 12–30, plus
    P09_106's 48 pairs with the plain SVT-AV1 background): the job ids
    are in the report's `sources`. Anchors (B2's 4 SVT-AV1 and 5
    DCVC-UF points per item): P01_107 `20261017T111944Z-c988e876`,
    P02_02 `…111915Z-66ad2183`, P03_10 `…111957Z-7c2a1eaf`, P09_106
    `…111928Z-60c2f00c`: every point reproduces B2's weighted PSNR within
    7 × 10⁻⁶ dB and its rate exactly, and every re-encoded DCVC-UF
    stream is byte-identical to B2's.
  - Report (`h5 report`): `h5-report.json`, `h5-rd.png` beside the jobs
    at `pointstream-data/visor/h5-2026-10-16/report-7ffa414/` (code
    `7ffa414`, per-key bootstrap; the first, `report/` from `c7e9e91`,
    is kept as it was).
  - Failed or replaced, none citable: GPU memory overruns of the
    foreground (33–35 GB on windows with large F; fixed by releasing the
    main process's cache before each DCVC-UF call and declaring 40 GB);
    non-deterministic DCVC-UF decodes on contended Ada (P06_03, QP 0 at
    ½) and on a contended A6000 (P03_10, QP 63 at ½), stopped by the
    per-point check and re-run; the scorer's first CPU compositor (2 min
    per pair; replaced by the same arithmetic on the GPU, checked to
    within 1 grey level, differing on at most 0.002% of samples); anchor jobs that
    required `dav1d` on the host's PATH (it ships in the environment); a
    timing smoke whose two DCVC-UF reports collided.
- Outcome, rate and quality (BD-rate against the pooled hull of
  SVT-AV1 preset 4 and DCVC-UF HT-L; negative is less rate; mean and
  95% bootstrap CI over the 4 items; P01_107 | P02_02 | P03_10 |
  P09_106):

  | Metric | PointStream | mean (CI) | items < 0 |
  |---|---|---|---|
  | **weighted LPIPS (fill)** | −12% \| +0% \| −26% \| −23% | **−15% (−25 to −5)** | 3 |
  | weighted LPIPS (no fill) | −13% \| +1% \| −26% \| −36% | −18% (−34 to −5) | 3 |
  | weighted DISTS (fill) | −12% \| +31% \| −1% \| −23% | −1% (−18 to +20) | 3 |
  | weighted PSNR (fill) | +78% \| +52% \| −4% \| −22% | +26% (−13 to +65) | 2 |
  | weighted PSNR (no fill) | +79% \| +53% \| −1% \| −47% | +21% (−24 to +66) | 2 |
  | frame LPIPS | −57% \| −16% \| −43% \| −37% | −38% (−52 to −23) | 4 |
  | frame DISTS | −61% \| −33% \| −51% \| −48% | −48% (−57 to −37) | 4 |
  | frame PSNR | +136% \| +83% \| +39% \| +21% | +70% (+30 to +123) | 0 |

  - **(a) passes**: weighted LPIPS −15% (rule −10%) with 3 of 4 items
    below 0. Weighted DISTS agrees in sign on 3 items but its mean is
    −1%; weighted PSNR disagrees on P01_107 and P02_02 (+78%, +52%) and
    agrees on the other two; whole-frame PSNR loses everywhere. The
    hypothesis was right in sign on LPIPS and wrong in size (−15%, not
    −30% or more), and half wrong on weighted PSNR (two items gain).
  - Routes: the enhancer route alone gives −14% on weighted LPIPS
    (−22 to −4), the scene route alone −4% (−18 to +16); together −15%.
    The scene route carries PointStream below the anchors' rates
    (PointStream reaches 15–30 kbps, the anchors start at 160–525
    kbps), where BD-rate cannot judge: B2b's low-rate ladder (pending)
    is what can.
  - **(b)** The plain SVT-AV1 background (P09_106) loses on every metric
    (weighted LPIPS +20%): the neural background is what wins. At each
    anchor point's rate (PointStream's best pair on weighted LPIPS at or
    under it; 9 points per item), PointStream's background has the lower
    LPIPS on 27 of 36 points (8 of 9 on P01_107, P02_02 and P03_10; 3 of
    9 on P09_106) and its foreground on 16 of 36 (2, 1, 5 and 8 of 9).
    On P01_107 and P02_02 (F 50% of the frame on P02_02) the foreground
    is worse by 0.01–0.04 LPIPS_F and the background carries the gain;
    on P09_106 (F 2%) it is the reverse. The competing explanation holds
    for the foreground where F is large: coding F as its own stream (its
    own intra frame, padding outside K) costs more per bit on F than
    whole-frame DCVC-UF.
  - **F billed** (the granted mask at its lossless cost, 0.5–2.9 Mbps
    per window): every comparison flips (weighted LPIPS +279%, from +26%
    on P09_106 to +429% on P02_02). The claim stands only with F
    granted; a mask coder costing a few kbps is a precondition of any
    deployed PointStream.
- Outcome, time (`h5 time`, code `33f4cdb`, exclusive RTX A6000 on gpu2,
  32 threads, median of 3 runs per stage; one job per window and route:
  scene `20261017T120926Z-b72736c7` P02_02, `…120948Z-2afd3bc4` P01_107,
  `…121010Z-8834efe5` P03_10, `…121031Z-27beb50a` P09_106; enhancer
  `…120937Z-f5b4fa31`, `…120959Z-cef259d9`, `…121020Z-943758c2`,
  `…121041Z-9cc7af81`; every job 7/7 checks locally, every container
  reproducible across its 3 runs). Operating points: scene model at
  1080p QP 38 with the foreground at ½ QP 21; SVT-AV1 CRF 48 plus the
  enhancer with the foreground at ¼ QP 21. Seconds per 4 s window,
  median over the 4 windows (range):

  | Stage | scene route | enhancer route |
  |---|---|---|
  | ProPainter fill (encoder) | 127 (123–138) | 128 (124–131) |
  | background encode | 1.9 (DCVC-UF codec calls) | 15–25 (SVT-AV1 preset 4, 32 threads) |
  | foreground padding (CPU) + DCVC-UF encode | 31–36 | 19–21 |
  | **encoder** | **163 (156–174)** | **166 (159–173)** |
  | **decoder** | **95 (80–152)**, 1.6–3.0 frames/s | **133 (124–151)**, 1.6–1.9 frames/s |

  The codecs themselves are fast (DCVC-UF 0.5–2 s per stream per
  window); the encoder is ProPainter (77%) and the decoder this
  pipeline's own CPU stages: compositing and 4:2:0 output (47–101 s,
  the same arithmetic takes about 7 s per window on the GPU in the
  scorer), G on the hands (7–25 s), the enhancer (37 s) and F's distance
  transform (11–12 s). Not timed: F (granted) and the venue models'
  training (G6 training jobs: 1,760–1,784 s per window, once per
  venue). The SVT-AV1 preset-4 encode inside the enhancer route took
  15–25 s on gpu2 (17.8 s on P02_02) against J2's median 18.7 s on gpu3
  (32 cores each), so J2's table applies.
  - **(c) Equal-time preset: SVT-AV1 preset 1.** The slowest route's
    median encode (166 s) is above every preset in J2's table; the
    slowest at or under it is preset 1 (80.5 s; −15.6% against preset 4
    on weighted PSNR in J2). PointStream's encoder is twice as slow even
    so, all of it ProPainter; presets 0 and −1 were not in J2.
    Hypothesis right (ProPainter dominates; 3 or slower).
- Reading. PointStream as built beats B2's preset-4 SVT-AV1 and DCVC-UF
  envelope on weighted LPIPS by 15% on the pilot items with F granted,
  and on frame LPIPS and DISTS by 38–48%; it loses whole-frame PSNR. Its
  gain is the neural background's; its foreground stream is the weaker
  part where F is large. Three things stand between this and a
  paper claim: (1) the anchor at equal time is preset 1 (and B2b's
  low-rate ladder and ROI variant), not preset 4; (2) F must be coded
  (lossless costs 0.5–2.9 Mbps, more than the rates where PointStream
  wins); (3) the foreground stream's per-bit cost on large F.
- Plan for the 34-item run (not run; final runs wait for a settled
  pipeline and the paper's figures, memory "final runs once"): per window
  6 G6 jobs (about 3 GPU-hours with queueing failures, G6's measure), 1–2
  foreground jobs, 7 scoring shards (12–30 pairs, 10–31 minutes each) and
  1 anchor job; about 34 × 6 ≈ 200 GPU-hours with today's grid, most of
  it G6's background (80) and scoring (about 60 at 30–60 s per pair).
  Before it: B2b's equal-time re-encode at preset 1 with the low-rate
  ladder and ROI, which the 34 items need anyway; a mask coder; and the
  foreground's per-bit cost. Scoring could drop the plain SVT-AV1
  background ablation and DISTS's five-region pass to cut its share.
- GPU time: 22.5 GPU-hours over 149 smoke and full stages (the jobs'
  execution records, failures included), within the 28 ceiling.

### 2026-10-17 — H5 with G_HO in both foreground slots (after H4c)
- Change (H4c's decision, user 2026-10-17, relayed by the H4c session): H4c's
  G_HO (G refitted on G's hand footage plus other participants' object
  crops, 8 × 64; −39% on handled-object pixels and −33% on hands at ¼ and ½,
  4 of 4 pilot items) replaces G in the hand slot and the identity in the
  object slot, at H3d's six points (`h5 foreground --slots G_HO`; networks
  from H4c's jobs `20261016T152552Z-c3d9ed14` and `…153801Z-03f2a819`). QP 63
  and the full-resolution points keep no network.
- Scope stated: H4c measured G_HO on H1's handled-object region O; H5's
  object class is all of F that is not a hand (VISOR's active objects,
  including large surfaces such as pans and boards), so G_HO on it goes
  beyond what H4c measured.
- Method: the foreground jobs of the 4 pilot items rerun at the six points;
  every pair of the H5 hull rescored as in H5 (F granted), and H5's report
  run with these rows in place of G's. No new rule: this is a component
  swap; reported as BD-rate of PointStream with G_HO against PointStream
  with G on the same pairs, and against the anchors as in H5.
- Budget: 4 foreground jobs and 4 scoring jobs, about 3 GPU-hours.
- Runs (shared Ada; every full stage validated locally: foreground 10/10,
  scoring 9/9). Foreground (`h5 foreground --slots G_HO`, one job per item
  and scale, code `470ac5f`): P01_107 `20261017T180513Z-ee4cbe4b`,
  `…180524Z-81d23687`; P02_02 `…180535Z-1db7d33b`, `…180547Z-2349b02b`;
  P03_10 `…180559Z-40098c01`, `…180613Z-9bd442dd`; P09_106
  `…180623Z-911e8a1a`, `…180636Z-0d751fa1`. The DCVC-UF streams are the
  same bytes as H5's (only the slots differ). Scoring (`--tag gho`, the six
  G points against every background on H5's hull, 54–60 pairs per item,
  code `7b54dfd`): P01_107 `20261017T182218Z-84e94d46`,
  `…182230Z-e960b121`; P02_02 `…183014Z-80e5c530`, `…183028Z-cbb75972`;
  P03_10 `…182855Z-f458eae1`, `…182908Z-7e362419`; P09_106
  `…183728Z-f10b2cf0`, `…183742Z-ddea296c`. Pairs whose foreground has no
  network (QP 63, full size) keep H5's rows. Failed or replaced, none
  citable: stage timeouts (two network passes per point; then the A6000s
  saturated by another user) and a 48.5 GB GPU overrun (the caching
  allocator kept every crop size), fixed in `ff98698` and `470ac5f`.
- Outcome (F granted; against the pooled anchors as in H5; P01_107 |
  P02_02 | P03_10 | P09_106):

  | Metric | G_HO | mean (CI) | < 0 | H5 with G |
  |---|---|---|---|---|
  | **weighted LPIPS (fill)** | −19 \| −20 \| −42 \| −34% | **−29% (−38 to −19)** | **4** | −15% (3) |
  | weighted DISTS (fill) | −19 \| −8 \| −39 \| −25% | −23% (−34 to −12) | 4 | −1% (3) |
  | weighted PSNR (fill) | +78 \| +54 \| −4 \| −22% | +27% (−17 to +66) | 2 | +26% (2) |
  | frame LPIPS | −61 \| −33 \| −49 \| −40% | −46% (−55 to −37) | 4 | −38% (4) |
  | frame DISTS | −63 \| −47 \| −53 \| −47% | −53% (−59 to −47) | 4 | −48% (4) |
  | frame PSNR | +136 \| +85 \| +40 \| +21% | +70% (+31 to +112) | 0 | +70% (0) |

  G_HO against G on the same pairs: weighted LPIPS −16%, −37%, −34%,
  −33%; weighted DISTS −17%, −47%, −49%, −23%; weighted PSNR +7%, +14%,
  +9%, +2% (G_HO trades PSNR for perceptual quality, as G did). With G_HO
  the foreground contributes: rule (a) passes on 4 of 4 items, and
  weighted DISTS, which disagreed with LPIPS under G, now agrees.
  Report: `pointstream-data/visor/h5-2026-10-16/report-gho-7b54dfd/`.
- Not yet a fair claim for the paper (user, 2026-10-20), to settle on the 4
  pilot items before any 34-item run: (1) the venue models (scene model,
  enhancer) are fitted on the window's own video and not billed, and their
  payback was never computed (H3d: per-video networks fail once billed on
  these short videos); (2) the anchors get no such adaptation: a
  whole-frame DCVC-UF adapted the same way, and SVT-AV1 at the equal-time
  preset 1, are the fair anchors; (3) F is granted (H5m). Metrics (user,
  2026-10-20): training a model on LPIPS (or any metric or combination) and
  scoring on the same metrics is legitimate and is not a gate; the scores
  stay LPIPS, DISTS and PSNR, with a visual check beside them.

### 2026-10-17 — H5m: coding F (the mask PointStream's decoder needs)
- Question (after H5's pilot, user 2026-10-17): H5's gain holds only with
  F granted; sent losslessly at 1080p, F and its class map cost 0.5–2.9
  Mbps per window and flip every comparison. Can F be coded cheaply
  enough that PointStream still beats B2's anchors with every byte,
  F's included, counted?
- Candidate (cheapest first): the class map (0 background, 1 hand, 2
  object) at 1/s of the frame size (s = 2, 4, 8, 16; F̂ where at least
  half of a cell's area is F, hand where hand pixels outnumber object
  pixels there), its two bit planes XOR-ed with the previous frame and
  compressed with lzma, losslessly; the client upsamples F bilinearly
  and thresholds at 0.5, and gives each pixel of F̂ the class of the
  nearest low-resolution F cell (`src/pointstream/mask.py`). F̂ replaces
  F in the compositor (the ring blend, G's slot); the bytes go into the
  container's MASK segment, so the rate counts them.
- Method (`h5 score --mask-scale s`): the published foreground and
  background decodes of H5's pilot re-composited with F̂ and scored as
  in H5 (B2's harness on the dataset masks, which stay the reference).
  Pairs: each item's H5 hull pairs on weighted LPIPS (12–17 per item),
  at each s. Bias, stated: the streams were coded for the exact F (the
  background filled F, the foreground kept K = F ⊕ (8 + 2·scale) px), so
  where F̂ leaves K the foreground shows padding; a pipeline built for
  F̂ would code K and the fill from F̂. This measures the candidate on
  today's streams, an estimate of the built component, not a bound.
  The H5 hull's pairs are chosen with F granted; with F̂ another pair
  could win, which only helps the candidate (the hull here is a lower
  bound of the candidate's own hull).
- Decision rule, fixed now. The candidate is adopted if, at some s (or
  the hull over s), PointStream with F̂ billed passes H5's rule (a)
  against the same pooled anchors: weighted LPIPS (fill) BD-rate −10% or
  better with at least 3 of 4 items below 0. Then the mask coder is
  built at that grid (encoder codes K and the fill from F̂). Otherwise
  the next candidate, with its own rule: F derived at the client from
  what it already decodes (the foreground stream coded with a key
  outside F), or contours with temporal prediction.
- Hypotheses. At s = 8 the class map costs 10–60 kbps per window and
  costs little quality (F̂'s boundary errors of up to 8 px fall in the
  ring's blend); the rule passes at s = 8 or 16, with a smaller margin
  than H5's −15% (the mask rate matters most at PointStream's low rates,
  where most of its hull lies but the anchors do not reach).
- Competing explanation for a failure: F's boundary is where the
  composite is judged hardest (hands against the background, the 70%
  weight on F); missing F pixels show the background's fill (inpainted,
  wrong content) and extra ones show the foreground's padding, so even
  small boundary errors cost more than the bits they save.
- Budget: 4 scoring jobs (one per item, about 60 pairs at 30–40 s), about
  3 GPU-hours, shared GPUs; each under 45 minutes.
- Runs (`h5 score --mask-scale s --pairs <H5 hull>`, code `802be6c`, shared
  Ada and A6000; every full stage 9/9 checks locally, every MASK payload
  decoding to its reduced grid exactly). P01_107 s = 2, 4, 8, 16:
  `20261017T150931Z-e433c92d`, `…150940Z-83ccfcf4`, `…150949Z-43d7a289`,
  `…150958Z-db776057`; P02_02: `…151010Z-7a75681d`, `…151019Z-494b6010`,
  `…151028Z-d9ada6bd`, `…151038Z-dd172972`; P03_10: `…151052Z-d0800d65`,
  `…151104Z-c9ed195b`,
  `…151116Z-ddb03427`, `…151127Z-ad0e1bbc`; P09_106: `…151140Z-c5e09039`,
  `…151149Z-33f8ba99`, `…151158Z-6374d0b3`, `…151207Z-d2be1d00`.
- Mask cost (kbps) and IoU of F̂ with F, P01_107 | P02_02 | P03_10 |
  P09_106: s = 2: 529 | 971 | 662 | 219 (IoU 0.98–0.99); s = 4: 166 | 320
  | 229 | 89 (0.97–0.99); s = 8: 51 | 116 | 75 | 36 (0.96–0.99); s = 16:
  16 | 39 | 23 | 14 (0.83–0.97). F costs about 1/s² of its lossless
  cost; at s = 8 it is still as much as PointStream's whole stream at
  its low end on the items where F is large.
- Outcome (BD-rate against the pooled anchors, as H5; per item P01_107 |
  P02_02 | P03_10 | P09_106):

  | F | weighted LPIPS | mean (CI) | < 0 | frame LPIPS | frame DISTS |
  |---|---|---|---|---|---|
  | granted (H5) | −12 \| +0 \| −26 \| −23% | −15% (−25 to −5) | 3 | −38% | −48% |
  | s = 2 | +123 \| +150 \| +122 \| −2% | +98% | 1 | | |
  | s = 4 | +33 \| +52 \| +27 \| −14% | +25% | 1 | +11% | +19% |
  | s = 8 | +3 \| +21 \| −8 \| −18% | −1% (−16 to +13) | 2 | −15% | −16% |
  | s = 16 | +21 \| +13 \| −11 \| +11% | +9% | 1 | −21% | −28% |
  | hull over s | +3 \| +13 \| −14 \| −18% | −4% (−17 to +8) | 2 | −21% (4/4) | −28% (4/4) |
  | lossless 1080p | | +279% | 0 | +261% | +348% |

  - **The rule fails**: no s, nor the hull over s, reaches −10% with 3 of
    4 items below 0 on weighted LPIPS; the candidate is not adopted as
    it is. Hypothesis wrong in cost (s = 8 costs 36–116 kbps, not 10–60)
    and in quality (the margin vanishes rather than shrinking).
  - The competing explanation holds on the F-weighted metric: F̂'s
    boundary errors (the fill showing where F̂ misses F, padding where it
    overshoots K) cost the 70%-weighted foreground more than the mask's
    bits save. On the whole frame, where F's boundary is a small part,
    PointStream with F coded at s = 16 still needs 21% less rate for the
    same LPIPS and 28% less for DISTS, on 4 of 4 items.
  - Bias recalled: the streams were coded for the exact F, so this
    under-states a pipeline built for F̂ (K and the fill from F̂); that
    component is the obvious next test of this candidate, cheaper than a
    new coder: re-encode the foreground with K from F̂ at s = 8 and 16.
  - Next, by the rule: F derived from what the client already decodes
    (no MASK segment: the foreground stream carries F, e.g. a key outside
    F or an alpha plane coded with it), or contours with temporal
    prediction; each with its own rule, after the F̂-consistent re-encode
    above says how much of the loss was the bias.
- Code: the report's per-key bootstrap (a comparison's CI no longer
  depends on what else is in the report; H5's headline CI is −25 to −5,
  first printed as −25 to −3), rows keyed by pair and mask scale.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5m-7ffa414/`.
- GPU time: about 2.5 GPU-hours (16 jobs).

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

### 2026-10-13 — K4: tennis audit (H1's method on broadcast tennis)
- Question (PLAN K4): what is broadcast tennis made of, part by part, and
  can a body-pose estimator's keypoints carry the players? Per part
  (players, racket, ball, background): its share of the pixels and of
  SVT-AV1's bits, how often the camera is still, and the pose estimator's
  fit and parameter rate against SVT-AV1's bits on the same pixels. Decides
  whether players pass H1's screen (and so whether K6 runs) and what the
  ball costs (K5 decides the ball mode).
- Windows (fixed content-blind before any run; `tools/datasets/k4_inputs.py`
  `RULE`, from metadata only). *Broadcast*: the archived `tennis_games`
  matches with scene clips (`alcaraz_hurkacz` 1080p25, `medvedev_struff`
  1080p50, `rublev_auger-aliassime` 1080p50, `djokovic_federer` 2160p50,
  `djokovic_zverev` 2160p60; 2160p downscaled to 1920×1080, area filter);
  per match the 2 clips of at least 240 frames with the smallest
  sha256("pointstream-k4:broadcast:<match>/<clip>"), first 240 frames: 10
  windows. *TrackNet*: the 8 clips of at least 120 frames with the
  smallest sha256("pointstream-k4:tracknet:<game>/<clip>"), first ≤ 240
  frames at 1280×720, 30 frames/s, with their ball labels. *Camera*: the
  one whole broadcast in the archive (`raw_1080p/nadal_kyrgios.mp4`,
  Wimbledon, 2,310 s at 1080p50, close-ups and replays included), every
  0.5 s. Every window is converted once by ffmpeg to 8-bit 4:2:0 (limited
  range), and every analysis reads that file, so masks, poses and codec see
  the same pixels.
- Parts, per frame (each pixel in one part; earlier parts win): *ball*,
  *racket*, *player*, background. SAM 3.1 text prompts ("tennis ball",
  "tennis racket", "tennis player"; offline bidirectional, G1's segmenter,
  JPEG chunks at 960×540, masks back to the window's size). On TrackNet the
  ball is its label instead (visibility 1 or 2): the SAM ball mask that
  contains the labelled point, else a disk of radius 4 px. The foreground
  is ball + racket + players; spectators, umpires and ball kids are
  background (a prompt that catches them is reported from the per-frame
  instance counts and the review frames).
- Method.
  1. Bits: SVT-AV1 4.2.0 preset 4, CRF 41, 48, 55, 62, one keyframe (B2's
     settings), decoded by H1's libaom `inspect` with bit accounting
     (`aom-inspect-v3.12.1-ps1`, `9b2e75a2…3959`); each block's bits spread
     evenly over its pixels and split by the parts' pixel shares in the
     frame it displays as (H1's `stream_bits`). Also reported: the bits
     within 16 px of the ball (its neighbourhood, which the block
     spreading can under-count).
  2. Pose: ultralytics 8.4.6 YOLO26x-pose and YOLO11x-pose (COCO 17
     keypoints, image size 1280, confidence 0.25), audited side by side;
     the estimator with the higher fitted share on the pilot is the one the
     rule reads (ties: keypoints inside the mask). Each SAM player instance
     takes the detection with at least 0.5 of its box inside the mask's box
     (H1's matching); unmatched = not fitted. Per player-frame: fitted;
     share of confident keypoints (≥ 0.5) inside the mask dilated by 8 px;
     box height. Stability over consecutive fitted frames of a track: 2D
     keypoint acceleration as a share of the box height per frame².
  3. Parameter rate: per fitted player-frame the 17 keypoints quantized at
     0.25 px (H1's position step) plus one visibility bit each, coded as
     first differences along the track; the empirical entropy of those
     symbols pooled over a dataset's windows, times the frame rate and the
     mean number of players per frame; 1 px reported beside it; 16 bits
     per value as the upper bound. Compared with SVT-AV1's bits on the
     player pixels. The ball's trajectory rate (TrackNet labels, 1 px):
     a visibility symbol per frame and, when visible, the residual of a
     constant-velocity prediction (absolute position after a gap), empirical
     entropy pooled.
  4. Camera: on frames 0.5 s apart (960×540, outer 12% border excluded,
     where the score and channel graphics sit), SIFT with MAGSAC; a pair is
     a *cut* below 40 inliers, *still* when the homography moves no
     corner of the inner frame by more than 2 px at 1080p (G1's bar), else
     *moving*. Each sampled frame is *wide* when YOLO26x-pose finds no
     person taller than 30% of the frame, else *close*. Reported: shares of
     time still, still-and-wide, moving, cut; median duration of still
     runs. The same measure on every window.
  5. Review: two content-blind frames per window (sha256
     "pointstream-k4-review:<window>:<n>") with the parts and both
     estimators' skeletons.
- Decision rule (`k4.DECISION`, fixed before any run; per dataset, means
  over its windows). Players pass H1's screen when (a) they carry ≥ 10% of
  the foreground's SVT-AV1 bits at CRF 48 and 62; (b) the chosen
  estimator fits ≥ 90% of player-frames and the median share of confident
  keypoints inside the mask is ≥ 0.9 (a 2D skeleton has no outline, so
  H1's IoU has no analogue here; K6 measures the outline); (c) the
  keypoint rate is ≤ 10% of SVT-AV1's bits on the player pixels at CRF 62.
  K6 (the player oracle) runs if (a) and (b) pass on broadcast and the
  keypoint rate is ≤ 50% of those bits (H3b's bar: perfect appearance at
  the coded pose must still save half); otherwise players stay pixels and
  K6 is not run. The ball and the camera are measured, not gated: the
  ball's cost feeds K5, the still share feeds G's per-shot background.
- Hypothesis: players are 1–3% of the pixels, 60–90% of the foreground's
  bits and 5–15% of all bits; the racket under 0.3% of the pixels; the
  ball under 0.02% of the pixels and under 0.5% of the bits (hundreds of
  bits per second at CRF 62); YOLO pose fits ≥ 95% of player-frames on
  broadcast, fewer on the far player and on 720p TrackNet; keypoints cost
  3–8 kbps for two players against 20–60 kbps of SVT-AV1 at CRF 62, so (c)
  fails at 10% but the K6 gate (50%) passes. The camera is still 60–80% of
  broadcast time and on ≥ 95% of rally frames.
- Competing explanations: (1) block spreading moves bits between small
  parts and the large blocks around them (the ball and far player in 64 px
  blocks); the ball neighbourhood brackets it. (2) "tennis player" also
  masks ball kids and umpires, inflating players; instance counts and the
  review frames check it. (3) Keypoint jitter, not motion, sets the rate
  (H1's hands); the acceleration measure and the 1 px rate show it. (4)
  Static graphics make a panning camera look still; the border exclusion
  and the review of moving runs check it.
- Budget: SAM 3.1 on Ada or A6000 (model–GPU table), about 13,000
  frame-prompts at G1's 5–6 per second (≈ 40 min); YOLO pose and SVT-AV1
  (72 encodes, 16 threads) and `inspect` in the same jobs. Smoke: one
  TrackNet window, 48 frames, ≤ 600 s. Pilot: 2 TrackNet and 2 broadcast
  windows, ≤ 45 min. Full: the other 14 windows in parts of ≤ 45 min.
  Camera: one job on the whole match, ≤ 45 min. Ceiling 4 GPU-hours and
  10 h wall.
- Amendment after the first pilot, before any full run (2026-10-13).
  Pilot `20261013T182436Z-d39c5b2c` (gpu3, RTX A6000, `0006071`; its full
  output passes the validator's 14 checks locally) showed competing
  explanation (2): "tennis player" returns 3–4 instances per frame, and
  the review frames show ball kids and line judges among them, which
  inflates the player part and pulls the fit share down (YOLO is never
  matched to a kneeling ball kid). *Players* are now the 2 "tennis player"
  tracks present on ≥ 50% of the frames whose mask centroid travels
  furthest, in median mask heights (`k4.choose_players`); the other tracks
  are a part of their own, *other person*, counted with the background,
  not the foreground. The rule is unchanged otherwise and reads the
  amended parts; the numbers with every instance a player are reported
  beside them (`as_first_defined`). The pilot is re-run with the
  amendment; the first pilot is not used for the decision. The decision
  rule (`k4.DECISION`: thresholds, CRFs, the K6 gate) is unchanged; only
  the parts it reads changed, as above.

  The first pilot's numbers, kept on record (every "tennis player"
  instance a player; `published.tar` `41c877af…0919`):

  | Window | players' pixels | foreground's share of all bits, CRF 62 | players' share of the foreground's bits, CRF 48 / 62 | fitted, YOLO11x / YOLO26x |
  |---|---:|---:|---:|---|
  | broadcast djokovic_zverev/008 | 3.90% | 30.1% | 88% / 92% | 0.991 / 0.990 |
  | broadcast rublev_auger-aliassime/052 | 3.55% | 19.2% | 95% / 96% | 0.827 / 0.794 |
  | tracknet game3/Clip3 | 0.84% | 12.0% | 95% / 97% | 0.333 / 0.333 |
  | tracknet game7/Clip2 | 1.02% | 22.8% | 92% / 94% | 0.578 / 0.590 |

  "Tennis player" instances per frame: 4.0, 2.8, 3.0 and 2.9 (mean).
  Ball under 1 kbps of SVT-AV1 at CRF 62 on every window; the broadcast
  windows' camera moved on 100% and 89% of their 0.5 s pairs.

- Jobs (all RTX A6000, gpu3, shared; inputs
  `pointstream-data/tennis/k4-2026-10-13/` `inputs-233a08b158e2.json`,
  `windows.json` `d79bb01c…e516`; environment `pointstream-20261009T214002Z`:
  SAM 3.1 at the pinned checkpoint with flash attention, ultralytics 8.4.6,
  SVT-AV1 4.2.0, ffmpeg 8.1.2, H1's `inspect`). Camera
  `20261013T134112Z-f78909bc` (`328bb79`; the whole match, 4,620 frames;
  1,576 s; `k4-camera.json` `bbbb30cc…44b2`). Amended pilot
  `20261013T184729Z-d7d08c01` (`0e47f61`; `published.tar` `63abd158…4ffa`);
  parts `20261013T222144Z-453bb877` (`78e7beb1…`), `20261013T222156Z-efa3c25e`
  (`fd17c8d5…`), `20261013T222209Z-5ad5967e` (`28bfe59d…`), all at
  `4fd8e10`, whose `k4.py` is the amended pilot's. Not evidence: the first
  pilot `20261013T182436Z-d39c5b2c` (the definition the amendment replaced;
  numbers above) and `20261013T134021Z-c5b03bb1` (smoke: `inspect` without
  its execute bit). Validators passed on the smokes and, run locally, on
  every full output (15 checks) except part 2 on one window:
  `broadcast/medvedev_struff/042` is the channel's "Watch more / Subscribe"
  outro card, no tennis and no person in any frame, so it fails
  `players_found_on_half_the_frames`, `players_chosen`, and the accounting
  check (headers are 22% of its nearly static 41 kbps stream at CRF 62);
  the other 4 windows of part 2 pass all 15. The card is left out of the
  report (`--exclude`, reason recorded): 9 broadcast and 8 TrackNet
  windows. About 2.5 GPU-hours with the smokes and the first pilot, inside
  the 4-hour ceiling; about 10 h wall from the rules to the last part, at
  the 10 h ceiling (the two session outages and queueing for gpu3, the
  one free A6000, took most of it).
- Report (`k4 report`, 17 windows; `k4-report.json` `a6a76af3…d16d`).
  Estimator by the rule: YOLO11x-pose (pilot fit 0.773 against YOLO26x's
  0.765; keypoints inside the mask 1.0 for both). Means over windows:

  | | broadcast (9) | TrackNet (8) |
  |---|---:|---:|
  | pixels: player / racket / ball / other people | 3.99 / 0.069 / 0.007 / 0.77% | 0.63 / 0.031 / 0.006 / 0.39% |
  | SVT-AV1 kbps, CRF 48 / 62 (whole frame) | 958 / 248 | 300 / 76 |
  | foreground's share of all bits, CRF 48 / 62 | 20.0 / 19.0% | 13.7 / 12.2% |
  | players' share of the foreground's bits, CRF 48 / 62 | 86.0 / 88.6% | 94.0 / 96.0% |
  | players, kbps at CRF 48 / 62 | 155 / 41.9 | 34.6 / 7.4 |
  | racket / ball, kbps at CRF 62 | 2.8 / 0.29 | 0.27 / 0.05 |
  | within 16 px of the ball, kbps at CRF 48 / 62 | 5.4 / 1.1 | 2.4 / 0.36 |
  | other people's share of all bits, CRF 62 | 5.8% | 1.9% |
  | (b) player-frames fitted (YOLO11x-pose) | 0.875 | 0.547 |
  | keypoint rate, kbps (0.25 px; 1 px) | 9.4; 9.3 | 4.2; 4.0 |
  | keypoint rate / players' SVT-AV1 bits, CRF 48 / 62 | 6.1 / 22.4% | 12.1 / 56.9% |
  | camera still (0.5 s pairs of the windows) | 41% | 97% |

  As first defined (every "tennis player" instance a player): players are
  92–94% of the foreground's bits on broadcast and 95–97% on TrackNet, and
  YOLO11x-pose fits 0.8999 (broadcast) and 0.28 (TrackNet) of the
  instance-frames. Fit by the player's height in the frame (YOLO11x-pose;
  broadcast, TrackNet): 6–10% 0.50, 0.00; 10–15% 0.91, 0.21; 15–25% 0.97,
  1.00; ≥ 25% 1.00, 1.00; keypoints, once fitted, lie inside the mask
  (median 1.0). Keypoint stability: acceleration 0.021 (broadcast) and
  0.029 (TrackNet) box heights per frame²; 3.05–3.14 bits per value. The
  keypoints come at 1–2 px precision (88–100% of first differences whole
  pixels, 27–67% zero), so 0.25 and 1 px cost the same. Ball trajectory
  (TrackNet labels): 7.6 bits per frame, 0.23 kbps. The whole match:
  still 34.9%, moving 41.0%, cut 24.1% of 0.5 s pairs; wide views 57.5%
  of pairs, and of those 57.5% still; median still run 1.5 s.
- Decision (by the rule, `k4-report.json`). **Players do not pass H1's
  screen, and K6 is not run as gated.** On broadcast they pass (a) (86% and
  89% of the foreground's bits) and fail (b) (fitted 0.875, rule 0.90) and
  (c) (keypoints 22% of SVT-AV1's player bits at CRF 62, rule 10%); the K6
  gate needs (a) and (b). TrackNet fails (b) (0.547) and (c) (57%). The
  definition does not decide it: as first defined, broadcast fits 0.8999,
  also under the bar. The ball costs 0.29 kbps of SVT-AV1 at CRF 62 on
  broadcast (1.1 kbps within 16 px), about its trajectory's 0.23 kbps; K5
  decides its mode (a ball mode is built). The camera is still on 35% of
  the broadcast match and 41% of the broadcast windows (TrackNet 97%).
- Reading. The hypothesis holds for the composition (players 0.6–4% of the
  pixels and 86–96% of the foreground's bits; racket and ball tiny) and
  fails on fit and camera. (b) fails on one thing: the far player. Players
  15% of the frame height or taller are fitted on 97–100% of frames at
  either resolution; below 10% (the far player at 720p, and often at
  1080p) YOLO's whole-frame inference finds them on 0–50%. That is the
  estimator's setting (one image at 1280 px for the whole frame), not a
  limit of keypoints, so the change to test next is a top-down estimate on
  each SAM player crop, enlarged; if that clears 0.90, the K6 gate is
  re-read with it. (c) fails by more than the jitter H1 saw on hands:
  YOLO's keypoints are already coarse and stable (3.1 bits per value), and
  the margin comes from how cheap the players' pixels are (42 kbps for
  both at CRF 62 on broadcast, 7.4 kbps at 720p), not from the keypoints'
  cost; at CRF 48 the keypoints are 6% of the player bits on broadcast,
  inside the 10%. Competing explanations: (2) held and was amended
  (officials and ball kids are 5.8% of the broadcast bits, which the first
  definition gave the players); the motion rule then misses a player who
  stands still in 2 of 9 broadcast windows (the server in
  `djokovic_federer/038`, seen in its review frame; in
  `djokovic_zverev/012` the largest person, 291 px tall at 540p, who
  travels 1.0 body heights), each replaced by a smaller person, with the
  fit there at 1.0, so the misses do not explain the failure (kept as
  known labelling errors; the definition is not changed again). Without
  them (same report, those windows excluded too): without
  `djokovic_federer/038`, 8 broadcast windows, fitted 0.859 (as first
  defined 0.888), keypoints 20.6% of the player bits at CRF 62; without it
  and `djokovic_zverev/012`, 7 windows, 0.839 (0.872), 17.7%; (a) passes
  and (b) and (c) fail in every case, so the decision does not depend on
  them; (3) does not hold (stable, coarse
  keypoints); (4) does not hold: broadcast cameras pan and zoom during
  rallies (review of the match's moving wide pairs), and the 12% border
  exclusion keeps the score graphics out. The camera is the main surprise:
  G1's still broadcast cameras (TrackNet, RacketVision) are not the
  archived matches', whose main camera follows play.
- Scope. 9 broadcast windows from 5 matches (scene clips of 4–10 s,
  close-ups included where the content-blind rule took them) and 8
  TrackNet windows; the camera from one match. Parts by SAM 3.1 text
  prompts at 960×540 (the far player and the ball are small there). The
  keypoint rate is of YOLO's 17 COCO keypoints at its own precision, with
  no shape or outline; a body model with an outline (K6) costs more.

### 2026-10-13 — K5: ball oracle (the true ball at its coded position)
- Question (PLAN K5): would a ball mode (the ball sent as its position and
  drawn by the decoder) beat the pixel coders on the ball? Oracle: the
  true ball's pixels drawn at its coded position over each pixel coder's
  decode, so the only cost added is the trajectory. It bounds every ball
  mode that codes the position at that precision, because it renders the
  ball perfectly (the source's own pixels) and bills nothing for its
  appearance. If it does not win, no ball mode is built.
- Windows: K4's 8 TrackNet windows (1280×720, 30 frames/s, ≤ 240 frames,
  the only dense ball labels), the same 4:2:0 files.
- Coders: SVT-AV1 4.2.0 preset 4, CRF 35, 41, 48, 55, 62 (B2's four and one
  higher rate); DCVC-UF HT-L off the shelf (B2's worker and checkpoints),
  QP 0, 21, 42, 63. Rate: the whole stream in kbps.
- Oracle: on every labelled frame with visibility 1 or 2, the source's
  pixels inside a disk of radius 4 px around the labelled position (TrackNet
  labels are integers, so the coded position is the label at 1 px) replace
  the decode's; rate = the coder's + K4's trajectory rate (empirical
  entropy, pooled over the 8 windows; the 21 bits per frame of a raw
  position as the upper bound, reported beside it).
- Judges, on visible frames (visibility 1–2): (1) *ball-region PSNR*,
  RGB, within 8 px of the label (so the decoded ring 4–8 px around the
  drawn ball counts), squared error pooled per window; (2) *a ball
  detector's recall*: the COCO "sports ball" class of ultralytics YOLO26x
  and YOLO11x (image size 1280, confidence ≥ 0.1); a frame is a hit when a
  detection centre lies within 10 px of the label (5 px reported); false
  detections per frame reported. The detector with the higher recall on
  the source frames (pilot) is the judge; it is a decision metric only if
  that recall is ≥ 0.5, otherwise it is reported and (1) decides alone.
  (3) Whole-frame PSNR, reported (the oracle should not change it).
- Comparison per window and judge: the oracle on each coder against that
  coder's anchor envelope (upper concave hull of its encodes), and against
  the envelope of both coders' encodes pooled (the best pixel coder);
  BD-rate with PCHIP over the overlapping range (overlap ≥ 0.5 dB for PSNR,
  ≥ 0.05 for recall), else "better/worse at every rate" (or unclear).
  Also: the rate below which each coder's recall falls under 90% of its
  source recall (where pixel coders collapse, K1's regime).
- Decision rule (`k5.DECISION`, fixed before any run). A ball mode is built
  (PLAN K, a trajectory and a rendered ball) when the better of the two
  oracles (on SVT-AV1 or on DCVC-UF) beats the pooled envelope on ≥ 6 of
  the 8 windows, each by BD-rate ≤ −25% or better at every rate, on
  ball-region PSNR and, if the detector is valid, on recall. The margin is
  25%, not H3b's 50%: the oracle bills no appearance, but a ball is a
  small sphere with little appearance to render. Otherwise the ball stays
  pixels.
- Hypothesis: both coders smear the ball at low rates (SVT-AV1 from about
  CRF 55, DCVC-UF at QP ≤ 21: learned coders smooth small, fast objects),
  so their recall falls well below the source's, while the oracle keeps
  the source's recall for under 1 kbps more; BD-rates under −50% on both
  judges; a ball mode is built.
- Competing explanations: (1) the detector misses the ball on the source
  (small, blurred), so recall is no judge; the validity check covers it.
  (2) The pixel coders already keep the high-contrast ball at every tested
  rate, so the oracle gains little. (3) A fast ball is a streak longer
  than the 4 px disk, so the drawn ball leaves a ghost of the decoded
  streak; review frames of the oracle on the fastest frames check it.
- Budget: DCVC-UF and YOLO on Ada or A6000, SVT-AV1 on 16 threads. Smoke:
  one window, 48 frames, ≤ 600 s. Pilot: 2 windows, ≤ 45 min (fixes the
  detector). Full: the other 6 windows in 2 parts of ≤ 45 min. Ceiling 3
  GPU-hours and 8 h wall.
- Failed attempts (infrastructure, no results). `20261013T134053Z-43fa3fb2`
  (`328bb79`): ultralytics rewrote CUDA_VISIBLE_DEVICES to `0` and the
  DCVC-UF worker refused to run without the claimed GPU. Two smokes on
  gpu6 (RTX 6000 Ada, shared with another user's process at 89%
  utilization) decoded DCVC-UF HT-L wrongly on some points:
  `20261013T182455Z-ec6f0505` on all four QPs, `20261013T183324Z-70873dcb`
  (code before ultralytics loaded) on QP 0 and 42 (9 and 13 dB frame PSNR;
  the two decodes disagreeing with each other and with the encoder's I
  frame), while QP 21 and 63 were right (34 and 40 dB). The plan, the
  checkpoint pair (intra + HT-L), the 4:2:0 limited-range input and the
  QPs are H3c's; on the RTX A6000 the same code and window decode every
  point right on the first attempt, with SVT-AV1 loading the CPUs beside
  it, so neither ultralytics nor CPU load is the cause. Since then a point
  is re-coded (up to 3 attempts, all recorded) until its decode is
  deterministic, matches the encoder's I frame and has luma PSNR ≥ 20 dB,
  the validator rejects a decode under 20 dB, and K5 runs on the A6000
  only. No attempt in the runs below needed a second try.
- Jobs (all RTX A6000, gpu3, shared; inputs
  `pointstream-data/tennis/k4-2026-10-13/` `inputs-233a08b158e2.json`,
  `windows.json` `d79bb01c…e516`; environment `pointstream-20261009T214002Z`;
  SVT-AV1 4.2.0 and dav1d 1.5.4 from it): pilot `20261013T184015Z-8a7e37ca`
  (`5a7ac24`; game3/Clip3, game7/Clip2; full stage 1,049 s;
  `published.tar` `ce4cf778…e2f6`), part 1 `20261013T190749Z-83c3c15b`
  (`4fd8e10`, whose `k5.py` is the pilot's; game10/Clip10, game3/Clip6,
  game6/Clip3; 2,012 s; `d8b66fe7…a0c3`), part 2
  `20261013T190805Z-c78ad0f2` (`4fd8e10`; game2/Clip5, game2/Clip8,
  game8/Clip6; 1,862 s; `02fed3d7…abac`). Validators passed on the smokes
  and, run locally, on all three full outputs (10 checks each). About
  1.4 GPU-hours with the smokes, inside the 3-hour ceiling.
- Report (`k5 report`, all 8 windows; `k5-report.json` `c1644eff…d2ac`).
  Judge: YOLO26x by the rule (pilot source recall 0.478 against YOLO11x's
  0.283); 0.478 is under the 0.5 validity bar, so recall is reported and
  ball-region PSNR decides alone. Over all 8 windows YOLO26x finds the
  ball on 46% of the visible source frames (0.37 false detections per
  frame). Trajectory: 7.6 bits per frame (visibility 0.40 bits, residual
  7.7 bits on 1,509 visible frames, 14 absolute restarts), **0.23 kbps**
  (raw positions 0.61 kbps).

  Means over the 8 windows (kbps, then ball-region PSNR decode → oracle,
  frame PSNR, YOLO26x recall decode → oracle):

  | Coder, point | kbps | ball PSNR dB | frame PSNR dB | recall |
  |---|---:|---|---:|---|
  | DCVC-UF QP 0 | 13.8 | 19.2 → 23.2 | 29.2 | 0.00 → 0.45 |
  | DCVC-UF QP 21 | 40.4 | 20.4 → 24.4 | 33.6 | 0.08 → 0.63 |
  | DCVC-UF QP 42 | 133 | 24.0 → 27.4 | 37.0 | 0.35 → 0.58 |
  | DCVC-UF QP 63 | 483 | 28.0 → 31.0 | 39.9 | 0.50 → 0.57 |
  | SVT-AV1 CRF 62 | 76.2 | 20.5 → 24.5 | 34.9 | 0.02 → 0.56 |
  | SVT-AV1 CRF 55 | 181 | 23.3 → 27.2 | 37.8 | 0.13 → 0.49 |
  | SVT-AV1 CRF 48 | 300 | 26.6 → 30.0 | 39.4 | 0.23 → 0.45 |
  | SVT-AV1 CRF 41 | 484 | 29.5 → 32.4 | 40.8 | 0.33 → 0.45 |
  | SVT-AV1 CRF 35 | 728 | 32.1 → 34.6 | 42.2 | 0.36 → 0.44 |

  BD-rate on ball-region PSNR, against the pooled envelope of both coders
  (median, range over windows): oracle on DCVC-UF −57.3% (−71.9% to
  −51.0%), oracle on SVT-AV1 −37.6% (−61.4% to −31.8%); against each
  coder's own envelope −68.0% and −43.8%. Frame PSNR is unchanged against
  each coder's own envelope (median +0.29% on DCVC-UF, −0.23% on SVT-AV1;
  the oracle adds 0.23 kbps and changes a 4 px disk); against the pooled
  envelope it reads +1.5% and +7.1%, which is the other coder's frame
  PSNR, not the oracle's. Recall,
  against the pooled envelope: the oracle on DCVC-UF is better at every
  rate or −92% to −96% on every window; on SVT-AV1 unclear on 7 (the
  curves cross), better at every rate on 1. Each coder's recall stays
  under 90% of the source's below 91–274 kbps (DCVC-UF; game8/Clip6
  never reaches it), and SVT-AV1 reaches it only on game7/Clip2 (at
  524 kbps).
- Decision (by the rule): **a ball mode is built.** The oracle beats the
  best pixel coder's envelope on ball-region PSNR on 8 of 8 windows (rule:
  6), each by more than 25% (the smallest margin is −31.8%), with either
  coder underneath.
- Reading. The hypothesis holds on its main point and in size (BD under
  −50% on DCVC-UF, on SVT-AV1 −32% to −61%). As predicted, the pixel
  coders lose the ball at low rate: at DCVC-UF's lowest points the
  detector finds it on 0–8% of frames, against 46% on the source, and the
  review strips (each window's fastest frame) show the decodes erasing a
  fast ball entirely, so there is no ghost to draw over (competing
  explanation 3). The learned coder loses it more than SVT-AV1 at equal
  rate, so the ball mode matters most on top of the pixel coder that
  PointStream is choosing for everything else. Competing explanation (1)
  holds in part: an off-the-shelf COCO detector is a weak judge of a 720p
  broadcast ball (46% recall on the source), so recall does not decide.
  It also rises *above* the source's with the oracle (0.63 against 0.46
  at DCVC-UF QP 21): a sharp disk on a smoothed court is easier to detect
  than the source's blurred streak, so the oracle's recall is not a
  fidelity measure. (2) does not hold: neither coder keeps the ball at
  any tested rate below 0.5 Mbps.
- Scope. The oracle draws the source's own pixels inside 4 px of the
  label and bills nothing for them. On fast frames the source ball is a
  streak of 20–40 px (review strips), which the oracle does not draw: it
  bounds the position and a dot of appearance, not the streak. A ball
  mode has to render the blur (a streak along the coded velocity) and
  pay for what it sends; its own oracle and rendering come next. TrackNet
  only (720p, 8 windows of ≤ 8 s); positions as labelled (integers).

### 2026-10-16 — K1: do keypoints win on egocentric hands at very low rates?
- Question ([PLAN K1](../PLAN.md#k-where-keypoints-win-analysis-user-2026-10-13)).
  H3b2 found that perfect appearance at the coded pose (`wilor:arm`, LPIPS
  0.21 at 1.93 kbps, forearm granted) needs 48% less rate than SVT-AV1 at
  equal LPIPS. Since then the pixel bar fell twice: DCVC-UF off the shelf
  (H3c, −63% against SVT-AV1) and G on top of it (H3d, −22% more). Does any
  keypoint point still lie above DCVC-UF + G's curve at its own rate, on the
  same items and region? From recorded reports only; no new coding.
- Inputs. H3b2's report (`h3b-2026-10-11/report-h3b2-85c217e/h3b2-report.json`,
  `16d7fec6…a685`; it carries H3b's oracle units and SVT-AV1's eleven points on
  S) and H3d's component report (`h3d-2026-10-12/report-component/h3d-report.json`,
  `467be69f…6e00`; H3c's bar and G's six points on R). Script
  `experiments/visor/k1.py`.
- Region and LPIPS variant. Both use B2's AlexNet LPIPS spatial map on the
  full frame, averaged over a region, mean over the window's hand frames.
  They differ in the region: H3b scored on S (R plus every rendered MANO
  silhouette, objects excluded), H3c/H3d on R (the fill set's two hand
  classes, forearm included). The keypoint judges of H3c2 and H3e (WiLoR's
  pose error) were never run on G's outputs, so only LPIPS can be compared.
  The bridge: SVT-AV1's eleven points are the same streams in both (H3c
  checked their sha256 and reproduced H3b's LPIPS on R within 0.0002), so
  per item they give a monotone map from LPIPS on S to LPIPS on R (linear
  between points; R reads 0–9% higher than S at the same point). The
  oracle's LPIPS on S becomes the LPIPS on R that SVT-AV1 has where it
  scores the same on S. The comparison: at the oracle's rate, the
  achievable hull (`g5.envelope`, linear in rate) of G's points alone and
  of G with H3c's bar (SVT-AV1 and DCVC-UF off the shelf); and the rate
  that hull needs for the oracle's quality. Reading S as R directly gives
  the same signs on every item.
- Decision rule (PLAN K1, fixed before reading the numbers): if any
  keypoint point lies above DCVC-UF + G's curve at its rate, the paper shows
  the crossover; if none does, the low-rate claim is dropped for egocentric
  hands.
- Outcome (code `k1.py` at this entry's commit; output
  `pointstream-data/visor/k1-2026-10-16/k1.json`). LPIPS on R at the
  oracle's rate (1.93 kbps; HaMeR 1.77):

  | item | `wilor:arm` (R≈) | G's hull | G + bar hull | hull's rate for its quality | `arm_uncoded` | `wilor:hand` | `hamer:arm` |
  |---|---|---|---|---|---|---|---|
  | P01_107 | **0.172** | 0.258 | 0.258 | 7.9 kbps (saving 76%) | above (0.104) | below (0.274) | above (0.221; saving 55%) |
  | P02_02 | 0.235 | 0.216 | 0.204 | 1.1 kbps (−71%) | below (0.222) | below (0.44 on S, off the map) | below |
  | P03_10 | 0.183 | 0.155 | 0.155 | 1.2 kbps (−65%) | above (0.124; saving 48%) | below (0.184) | below (worse than every G point) |
  | P09_106 | 0.307 | 0.184 | 0.184 | 0.3 kbps (−504%) | below (0.263) | below (0.354) | below |

  Items where the keypoint point lies above DCVC-UF + G: `wilor:arm` 1 of
  4, `wilor:arm_uncoded` 2, `wilor:hand` 0, `hamer:arm` 1. The hull with the
  bar changes no sign; on P02_02 the bar (DCVC-UF off the shelf) is above G
  at 1.9 kbps. A mean saving is not a useful summary here: `wilor:arm`'s
  per-item savings against the joint hull span +76% to −504%.
  The real generator (H3e) was judged on pose error, not LPIPS; it lost to
  the pixel hull without G on 3 of 4 items, and whether G changes pose error
  was not measured.
- Reading. A keypoint point lies above DCVC-UF + G on one item, P01_107,
  and only as an upper bound: appearance perfect and free, the forearm free,
  no reference billed. Drawing the hand alone (`wilor:hand`, what a
  generator without an arm model would send) is below the curve on every
  item, and the real generator is below the pixel coders even without G.
  By the letter of the rule a crossover exists, but on a single item and
  only for a bound; P01_107 is where G gains least at low rate (its ¼
  points are 0.26–0.36). Proposed reading, for the user to confirm:
  **the low-rate claim is dropped for egocentric hands as a property of the
  method**; what survives is "with perfect appearance and a free forearm,
  1 of 4 items", which the paper can state in a sentence, not as a
  crossover figure.
  **Agreed (user, 2026-10-17): the low-rate claim is dropped for egocentric
  hands.**
  K1's other half (a judge where keypoints carry what matters) stays with
  K2: re-judging DCVC-UF + G on pose error is a GPU step (WiLoR on G's
  outputs at the low points), not recorded data.
- Scope. 4 pilot items; one LPIPS bridge per item through SVT-AV1's
  points; H3e has no LPIPS, so the generator enters only through its pose
  result.

### 2026-10-16 — H4c: handled objects as pixels, H3d's recipe
- Question ([PLAN H4c](../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08);
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

### 2026-10-16 — K3: keypoint-conditioned pixel coding (oracle: a perfect hand in DCVC-UF's reference)
- Question (PLAN K3, H3's proto-hand candidate). Do the hand keypoints save
  pixel rate on the hand region R when the decoder has them? If DCVC-UF's
  reference already showed the hand where the keypoints put it, would it
  code R for fewer bits at equal quality, with the keypoints billed?
- How DCVC-UF HT-L consumes its reference (microsoft/DCVC `cbdae87`,
  `src/models/video_model_ht.py` and the CUDA proxy
  `dmc_htl_proxy.cpp`), checked before anything was built:
  - *There is no motion stream.* HT-L codes 8 frames per NAL jointly
    (`g_frame_delay = 8`), conditioned on a context `ctx` extracted from a
    recurrent memory (`feature_adaptor_m(memory, ref_feature)`), where
    `ref_feature` is the decoder's 512-channel feature of the previous NAL.
    Its bits are y (the latent) and z (its hyperprior) only; motion is
    implicit in y. So the premise "the gain is bounded by its motion bits"
    has no counterpart: no motion-bit share exists to report. Its
    replacement here is the share of R's bits that a perfect reference
    removes (arm E below), the bound on anything reference-side.
  - *A frame can be the reference without retraining.* At a memory reset
    the reference is a single frame: after an I frame,
    `add_ref_feature_from_frame(x̂)` sets the reference to the
    pixel-unshuffled frame and restarts the memory through
    `feature_adaptor_i`; the periodic reset (`reset_interval`, upstream
    default 32) does the same with `recon_head(feature, for_reset=True)`,
    which is the previous NAL's last decoded frame. So a frame of our
    choosing (a warped one) can replace the previous decoded frame at
    every NAL through DCVC's own public path, with no new weights: the
    model sees exactly the input it sees after every I frame and reset.
  - *Costs of that substitution.* (1) Resetting every NAL drops the
    recurrent memory, which the unmodified codec keeps for the whole
    window (PointStream's worker sets no reset); arm B measures that cost.
    (2) One reference serves 8 frames (133–160 ms at 50–60 frames/s), so a
    warp can align the hand at one instant only; arms C and D place it at
    the group's first and middle frame.
  - *Instrument.* The CUDA proxy cannot run an encoder and an in-loop
    decoder in one process (shared global tensor pool), and the encoder
    needs the decoded previous frame to build the next reference. The
    oracle therefore runs DCVC's own PyTorch path (`forward_one_frame`,
    the same weights, fp32, symbols rounded as in H3c's bit estimate), a
    complete encoder and decoder: rate from the model's own entropy model
    per 16×16 cell (H3c's split, `h3c_bits`), quality from its x̂. Each
    NAL's estimated bits are calibrated by the unmodified codec's real
    bytes for the same NAL (B2's worker, encode only, at every point),
    the same factor for every arm; raw estimates are reported beside. The
    unmodified arm through this path is cross-checked against H3c's real
    decodes (rate and LPIPS_R per point, H3d's tolerance: rate within 5%,
    mean LPIPS_R within 0.005), and its real stream against H3c's sha256.
- Oracle, and why it bounds the keypoint warp. The component would warp
  the previous decoded frame by the hand's MANO mesh motion from t−1's
  coded pose to t's (H3b's correspondence, H2's coded WiLoR stream). Its
  best possible output is the hand exactly where it is at the target
  frame, with the source's appearance, and the background it uncovers
  shown correctly. The oracle gives the reference exactly that: inside
  M = R(t₀−1) ∪ R(t), the source's pixels of frame t; elsewhere the
  previous decoded frame. Nothing a warp of decoded pixels can produce is
  closer to the frames being coded (it adds coding error and MANO's
  silhouette error, H3b). The bound holds if the network codes better
  from a reference closer to its target; arm E (the whole source frame)
  against B checks that it does. If the oracle fails, the warp is not
  built and the trained version is not pursued (the user's rule).
- Arms (DCVC-UF HT-L off the shelf, H3c's checkpoints; per P NAL coding
  frames t₀…t₀+7; references in 8-bit 4:2:0 converted as the reader does):
  - A: unmodified (recurrent memory, no reset), as H3c's reference.
  - B: reset at every NAL to the previous decoded frame x̂(t₀−1) (the
    control: the substitution path with nothing substituted).
  - C: B with M's pixels taken from the source at t = t₀.
  - D: B with M's pixels taken from the source at t = the group's middle
    frame (t₀+4, or the middle of a shorter last group).
  - E: the whole source frame t₀ (a perfect reference; diagnostic, not
    billed for anything, not deployable).
- Items and points: B2's 4 pilot items (P01_107, P02_02, P03_10,
  P09_106), 240 frames each; H3c's nine points (QP 0, 21, 42 at full
  resolution, ½ and ¼; region R and its downscaled mask as H3c).
- Quality: H3c's `Scorer` (decode inside R, source outside; full-frame
  LPIPS averaged over R, mean over hand frames; DISTS on R's box plus
  64 px; PSNR on R).
- Rates. R's rate per arm and point (calibrated as above). C and D are
  billed H2's coded WiLoR stream, 1.93 kbps (the 100 ms choice, its VISOR
  rate from `h2-code.json`, as H3c2): the keypoints a warp needs. A and B
  bill nothing more; E is never compared billed.
- Comparison. Per item, each arm's upper concave hull against the
  anchor's, BD-rate on LPIPS_R by PCHIP of log rate over the shared range
  (at least 0.005; else "better/worse at every rate"; `h3d.compare`).
  Amended before any result: H3d's front against hull, dry-run on H3c's
  recorded points relabelled as K3 arms, charged arm A +13% against its
  own hull, which would bias every arm against the anchor by more than
  the rule's margin; both sides' hulls are achievable (switching between
  encodes). Anchors: A (the unmodified codec, same instrument), and
  the pixel envelope (H3c's bar: SVT-AV1's eleven points and DCVC-UF's
  nine real points, H3c's units). DISTS and PSNR beside.
- Decision rule (`k3.DECISION`, fixed before any run). Gate: the better
  of C and D (by mean BD) against A, billed, on LPIPS_R, scored with
  H3d's verdict (`h3d.arm_verdict`): *passes* at mean BD ≤ −10% with ≥ 3
  of 4 items below 0 and none worse at every rate; *fails* at mean ≥ 0 or
  ≤ 1 item below 0; otherwise *unclear*.
  - Passes: the keypoint warp itself is built as the next stage (H3b's
    mesh correspondence, the same arms with the warp in place of the
    source pixels), with its own rule, before any training.
  - Fails: K3 is closed; neither the warp nor a trained conditional coder
    is pursued. If C/D fail billed but pass unbilled, the outcome names
    the keypoint rate at which they would pass (the keypoint rate is
    deferred to optimization, PLAN H3); that is not a reopening.
  - Unclear: to the user with the numbers.
  - Reported beside, without deciding: B against A (the memory's worth),
    C and D against B unbilled (the hand pixels' value on a fixed path),
    E against B (the most any reference could save: the counterpart of
    the motion-bit share), C/D against the pixel envelope.
- Hypothesis. A perfect hand in the reference saves rate on R at middle
  and high rates (hands are the fastest-moving content in egocentric
  video, so they are where temporal prediction fails), about 10–25% of
  R's bits against B; resetting every NAL costs 5–15% against A; billed,
  1.93 kbps erases the gain at DCVC-UF's ¼ points (0.3–9 kbps on R) and
  part of it at ½. Expected: C/D billed against A near 0, so the gate
  fails or is unclear; E against B saves 30–50%.
- Competing explanations. (1) Source pixels are sharper than every
  reference the model was trained on, so it may not exploit them, or may
  even pay for the mismatch: E against B shows whether a better reference
  helps at all, and a failure then speaks for DCVC-UF's present weights,
  not for a trained conditional coder (stated as the scope). (2) One
  reference for 8 frames cannot follow the hand within the group: D
  against C shows the sensitivity to the target time. (3) The PyTorch
  path differs from the CUDA codec (fp32 against fp16; estimated against
  real bits): the cross-check on A bounds that.
- Budget. Ada or A6000, shared GPU (the PyTorch path ran in H3c's coding
  jobs on both). Smoke ≤ 600 s: P01_107, 48 frames, QP 0 at scales 1 and
  4, all arms. Pilot: 4 items, two jobs per item (full resolution; ½ and
  ¼), each ≤ 45 min (H3c scored nine points in 13–18 min; here 45 points
  per item). Ceiling 5 GPU-hours, one day wall.
- Jobs (shared GPU; environment `pointstream-20261009T214002Z`; inputs
  `pointstream-data/visor/k3-2026-10-16/inputs.json`, `1e282b74…164e`;
  code per `ready.json`): P01_107 full resolution
  `20261016T090616Z-df01caf0` (`39bc160`, RTX 6000 Ada); the other seven at
  `b330f8f` (the change since `39bc160` is the report's comparison only):
  P01_107 ½ and ¼ `20261016T090907Z-67e990f8` (Ada), P02_02
  `…090937Z-37023fb0` (A6000) and `…091001Z-d6d10f62` (Ada), P03_10
  `…091015Z-f398547c` and `…091030Z-bcdf1371`, P09_106
  `…091045Z-a87c9f00` and `…091106Z-f64cba4d` (all four A6000). Every
  smoke passed the fleet's validator; every full stage passed the same
  validator run locally (15/15 checks each). About 2.4 GPU-hours (smokes
  0.35, full stages 2.0), within the 5 allowed; shared jobs' timings are
  not evidence. Report (`k3 report`, code `8855c4e`, run locally):
  `pointstream-data/visor/k3-2026-10-16/report-pilot-8855c4e/k3-report.json`
  (`3b78c1a1…8bfb`).
- Instrument. Arm A through the PyTorch path reproduces H3c's real codec:
  35 of 36 points within H3d's tolerance (calibrated rate within 1.1%,
  mean LPIPS_R within 0.004); the exception is P09_106 at ¼, QP 0 (rate
  +4.4%, LPIPS_R −0.007). 21 of the 36 real streams are byte-identical to
  H3c's (P03_10 all nine, P09_106 none); the others' sizes differ by 1.1%
  or less except that point (DCVC-UF's encoder is bound to the GPU; H3c
  ran on other cards). The estimate's ratio to the real bytes stays within
  the validator's 0.5–2 at every point (H3c: 0.78–1.27).
- Outcome (pilot, 4 items). **K3 fails by the rule; it is closed.**
  BD-rate on LPIPS_R (hull against hull; mean over items, 95% bootstrap
  CI; items below 0), DISTS and PSNR per item beside:

  | test vs anchor | LPIPS_R | items < 0 | DISTS_R per item | PSNR_R per item |
  |---|---|---|---|---|
  | **C billed vs A (gate)** | **+9% (−23 to +51)** | 2 | −60, −49, −38, +42% | 0, +47, +63, +91% |
  | D billed vs A | +66% (+15 to +144) | 0 | −46, −53, −22, +139% | +66% to +185% |
  | C unbilled vs A | −34% (−50 to −14) | 4 | −26% to −73% | −23% to +14% |
  | B vs A (reset every NAL) | +39% (+25 to +53) | 0 | +10% to +37% | +24% to +44% |
  | C unbilled vs B | −48% (−64 to −30) | 4 | −34% to −78% | −11% to −43% |
  | D unbilled vs B | −17% (−42 to +22) | 3 | | |
  | E vs B (perfect reference) | −51% (−68 to −34) | 4 | −41% to −82% | −13% to −48% |
  | E vs A | −38% (−55 to −19) | 4 | | |
  | C billed vs the pixel envelope | +7% (−25 to +51) | 2 | | |

  (Items in the order P01_107, P02_02, P03_10, P09_106.)
  - Gate: the better arm, C, is +9% against the unmodified codec with the
    keypoints billed: by the rule, *fails* (mean ≥ 0). P01_107 saves 31%,
    P02_02 and P03_10 are at parity, P09_106 loses 68%. D is worse than C
    on every item: a reference aimed at the group's middle frame codes the
    group's first frames worse.
  - Keypoint rate: C passes unbilled (−34%, 4 of 4) and still passes at a
    keypoint rate up to 1.02 kbps (−10% at 1.0 kbps, 3 of 4); it is unclear
    from 1.1 to 1.4 kbps and fails from 1.5 kbps. H2's cheapest WiLoR choice
    (266.7 ms) costs 1.70 kbps, so no recorded keypoint stream passes. The
    keypoint rate is deferred to optimization (PLAN H3); per the rule this
    is reported, not a reopening.
- Reading. A reference that shows the true hand helps DCVC-UF's present
  weights a lot: against the same reset path (B) it saves 48% of R's rate
  at equal LPIPS, nearly as much as the whole true frame (E, −51%), so
  almost all of a better reference's value is in the hand region (the
  hypothesis put E at 30–50% and C at 10–25%: both higher). Two things
  remove that gain. (1) The substitution path resets DCVC-UF's recurrent
  memory every 8 frames, which costs 39% (B against A, every item): the
  memory carries more than the previous frame does. (2) The keypoints
  cost 1.93 kbps, while DCVC-UF codes R at 0.3–9 kbps at its ¼ and ½
  points (P03_10 and P09_106 never exceed 31 kbps): over the shared LPIPS
  range, the low end decides, and there the keypoints alone cost more than
  the pixels. The oracle grants the hand's true pixels at the target
  frame, so it bounds every keypoint warp (which can only move decoded
  pixels): the warp is not built. Competing explanation (1), that the
  model cannot use a sharper reference, is refuted (E and C against B);
  (2), the one-reference-per-group limit, shows in D against C;
  (3), the instrument, holds (cross-check above).
- What would change it, each needing its own oracle: keypoints that are in
  the stream anyway (a mode that sends them for a machine task makes them
  free; C unbilled passes, but only as the loose appearance bound, and
  the warp would then need its own test, e.g. the decoded frame warped by
  the source's optical flow as a tighter bound); keypoints at or below
  about 1 kbps; or a conditional coder trained to take the keypoints
  beside its memory instead of through a reset (C against B bounds what
  the hand's pixels are worth there, −48%, before billing). The user's rule
  closes the trained version now, since the oracle saves nothing billed.
- Scope. Four pilot items (240 frames each, 50–60 frames/s), H3c's nine
  points, LPIPS deciding; DCVC-UF HT-L's published weights; one reference
  per 8-frame group. The full run was not needed: the gate fails with the
  pilot's mean above 0 and only 2 of 4 items below 0.

### 2026-10-17 — K3b: the best motion a keypoint warp could give (flow-warped reference)
- Question (user, 2026-10-17: are hands conclusively a bad keypoint
  foreground?). K3's bound gave the reference the hand's *true pixels*,
  so it granted appearance as well as motion; at a keypoint rate of
  about 1 kbps (H2 records 1.01 kbps at +11% 2D error on HOT3D: 10 Hz
  interpolated, PCA 12, steps ×8, 120 ms) it passed only barely (−10%,
  3 of 4). A keypoint warp can only move the previous decoded hand. This
  stage measures that motion's value with the keypoints free: if it does
  not pay unbilled, no keypoint rate makes keypoint-conditioned coding pay
  on hands.
- Arm F (`k3_ref`): as B (reset at every NAL to the previous decoded frame
  x̂(t₀−1)), but inside M = R(t₀−1) ∪ R(t₀) the reference is x̂(t₀−1)
  warped to t₀ by DIS optical flow computed on the *source* frames t₀ and
  t₀−1 (H3a's backward flow and bilinear warp, `h3.dis_flow`/`h3.warp`:
  medium preset, luma, on the two regions' box grown by 64 px; at half
  resolution for full-resolution points as H1/H3, at the coded resolution
  otherwise). Why it bounds a keypoint warp: the flow is estimated from the
  true frames at every pixel (fingers, forearm, the uncovered background),
  while a MANO warp follows the coded pose's mesh, which misses or
  overshoots about a tenth of the hand (H3b) and has no forearm or
  background motion. It is not a strict bound (DIS errs at occlusions and
  blur), so the reading names that limit; it grants no appearance.
  Arms A (calibration and anchor) and B (control) are recoded beside it.
- Items, points, quality, rates, comparison: K3's (4 pilot items, H3c's
  nine points, LPIPS_R deciding, per-NAL calibration, hull against hull).
- Decision rule (`k3.FLOW_DECISION`, fixed before the run). Gate: F
  *unbilled* against A on LPIPS_R, by `h3d.arm_verdict`.
  - Fails (mean ≥ 0 or ≤ 1 item below 0): hands are closed for keypoint
    conditioning at every keypoint rate; PointStream moves to tennis
    players as keypoints (user, 2026-10-17).
  - Passes (mean ≤ −10%, ≥ 3 of 4 below 0, none worse at every rate): then
    F billed at H2's 1.01 kbps point is checked by the same verdict; if it
    also passes, the MANO warp at that point is the next stage (its own
    rule), else hands close with the keypoint rate named as the obstacle.
  - Unclear: to the user.
  - Beside: F against B (the warp's value on a fixed path), F billed at
    1.01 and 1.93 kbps against A, and K3's C (from K3's jobs) against F
    (what appearance adds beyond motion).
- Hypothesis. Motion of decoded pixels is worth much less than true
  pixels: F against B about −15% (K3's C against B was −48%), and since
  the reset costs 39%, F unbilled against A about +15 to +25%: fails.
- Competing explanations. (1) DIS flow is poor on fast, blurred hands, so
  F understates motion: the flow's warp error is reported (mean absolute
  luma error inside R(t₀) of the warped source t₀−1 against the source t₀,
  against the unwarped one). (2) The reset's cost hides a gain a trained
  coder could keep (K3's reading): F against B shows it.
- Budget. Ada or A6000, shared. Smoke ≤ 600 s; pilot: 4 items × 2 jobs
  (arms A, B, F), each ≤ 45 min. Ceiling 2 GPU-hours.
- Jobs (shared; inputs, environment and points as K3; code per
  `ready.json`): at `5db475b`, P01_107 full resolution
  `20261017T154233Z-acae64e8` and ½ + ¼ `…154244Z-44785c5c`, P03_10 full
  resolution `…154318Z-fe06e743` (RTX A6000), P09_106 full resolution
  `…154339Z-2dfbdd07`; at `22e7fdf`, P02_02 `…155753Z-288d6bb2` and
  `…155802Z-73432d9d`, P03_10 ½ + ¼ `…155828Z-600d3fda`, P09_106 ½ + ¼
  `…155850Z-194b2c1a` (all RTX 6000 Ada but P03_10's full resolution).
  `22e7fdf` changes only the validator: P02_02's first submission
  (`…154253Z-98d49b18`) failed its smoke on the check that the flow warp
  lowers its own error, because in the smoke's 48 frames the hand stands
  still (0.03 luma levels between frames, resampling noise 0.0005 above
  it); the check now allows a quarter luma level. Its sibling and the
  three jobs that had not started were cancelled and resubmitted; one
  duplicate was cancelled before it ran. Every full stage passed the
  validator run locally. About 1.7 GPU-hours (smokes 0.27, full stages
  1.45), within the 2 allowed. Report (`k3 report`, code `22e7fdf`, run
  locally, K3's jobs passed as `--k3`):
  `pointstream-data/visor/k3-2026-10-16/report-k3b-22e7fdf/k3b-report.json`
  (`13436277…bbcc`).
- Outcome (pilot, 4 items). **K3b fails: hands are closed for
  keypoint-conditioned coding at every keypoint rate.** BD-rate on LPIPS_R
  (hull against hull; mean, 95% bootstrap CI, items below 0; per item in
  the order P01_107, P02_02, P03_10, P09_106):

  | test vs anchor | LPIPS_R | items < 0 | per item | DISTS_R per item | PSNR_R per item |
  |---|---|---|---|---|---|
  | **F unbilled vs A (gate)** | **+44% (+29 to +65)** | 0 | +76, +32, +40, +28% | +47, +11, +8, +19% | +63, +42, +43, +28% |
  | F vs B (the warp on a fixed path) | +4% (+1 to +8) | 0 | +10, +1, +2, +5% | +7, 0, −4, +7% | +13, +1, +3, +6% |
  | F billed at 1.01 kbps vs A | +85% (+67 to +97) | 0 | | | |
  | F billed at 1.93 kbps vs A | +118% (+94 to +137) | 0 | | | |
  | K3's C (true hand pixels) vs F | −50% (−66 to −32) | 4 | −71, −40, −60, −29% | | |

  - The flow is accurate: inside R(t₀) it brings the source's previous
    frame from 7.5 to 1.9 luma levels of the next one (mean over the
    pilot's F points). Competing explanation (1), a poor flow, is
    refuted.
  - Even with the keypoints free and the motion this good, the
    flow-warped reference codes R 4% *worse* than the unwarped previous
    frame on the same reset path (B), and 44% worse than the unmodified
    codec. Competing explanation (2), that the reset hides a gain a
    trained coder could keep, is refuted as far as motion goes: on the
    fixed path, motion gives nothing to keep.
- Reading. All of K3's gain was appearance, none of it motion: the true
  hand's pixels save 50% against the flow-warped hand at the same place.
  DCVC-UF already finds the hand's motion on its own (its 8-frame
  conditional coding is motion-aware without a motion stream); what it
  cannot predict is how the hand looks in the new frame, which no warp of
  the previous decoded hand supplies. A slightly worse result than no
  warp fits that: the warp resamples decoded pixels (blur) and moves the
  boundary, which the model would otherwise have predicted itself. This
  holds for every keypoint rate, since F bills nothing; the hypothesis
  (F vs B about −15%) was wrong in the same direction as the conclusion.
  Hands stay pixels (G: DCVC-UF + the generic enhancer); keypoints
  remain a description of the hand (H3c2: the pose for 69% less rate),
  not a coding aid for its pixels. Next (user, 2026-10-17): tennis
  players as keypoints.
- Scope. Four pilot items, H3c's nine points, LPIPS deciding; DCVC-UF
  HT-L's published weights with one reference per 8-frame group;
  DIS optical flow on the source as the stand-in for the best keypoint
  motion. A conditional coder trained with a motion input beside its
  memory is not measured, but F against B gives no gain for it to learn
  from motion; appearance would need sending.

### 2026-10-16 — K8: court-view shots and camera registration (rule)
- Question (PLAN K8). A broadcast match mixes court views with close-ups,
  crowd shots, replays and graphics. PointStream should get only the court
  views, and within each, a per-frame registration of the panning camera
  (K4: still on 35% of a match). Both come from one measurement: a frame
  is a court view when its court lines register to the court model with a
  homography. The same homography gives the court outline (K4b) and the
  camera's motion (G's per-shot background).
- Data: the archived broadcast matches (`tennis_games`; the whole
  `raw_1080p/nadal_kyrgios.mp4`, 2,310 s, and the scene clips K4 drew
  from), the court model in `Datasets/Tennis_courts` (`reference/`), and
  the old pipeline's shot code (`9cf3012`, `scene_classification.py`)
  for reference only (it called low-motion scenes points, which let a
  close-up and an outro card into K4's windows).
- Method. (1) Cuts: ffmpeg's scene score, threshold chosen on the review
  below; shots are the spans between cuts. (2) Registration per frame:
  court keypoints or lines to the court model by a homography (RANSAC),
  either from detected white lines or from a pretrained court-keypoint
  model (audit first; weights into `Models`, never into the checkout).
  (3) A shot is a court view when at least 80% of its frames register with
  a reprojection error under the bar below. (4) Review: a local page
  showing every shot of the whole match (first, middle and last frame,
  predicted class, the projected court drawn on the middle frame), where
  the user marks the true class; these marks are the labels.
- Decision rule (fixed before any run). Adopted when, on the whole match's
  shots as the user marks them: court-view precision ≥ 0.95 (a non-court
  shot sent to PointStream is the costly error), court-view recall ≥ 0.90
  weighted by shot duration, and registration on court-view frames with
  median reprojection error ≤ 2 px at 1080p (projected court lines against
  detected line pixels, as a chamfer distance), 95th percentile ≤ 6 px.
  Cut errors count through the shots they corrupt (a shot mixing two
  classes is wrong for both). Otherwise the misses are reported by class
  and the user decides.
- Hypothesis: court lines register on nearly every court view; misses are
  extreme zooms on a player (few lines visible) and fast pans with motion
  blur; precision is easy, recall loses the zooms.
- Competing explanation: the homography registers court-like structures in
  replays and graphics (a picture-in-picture court), which precision
  catches.
- Budget: CPU first (cuts and line detection); GPU only for a court-keypoint
  model, on the fleet with the usual smoke, ≤ 45 min per stage.

- Implementation/run preparation (2026-10-16, `codex/k8-court-shots`).
  The supplied `Tennis_courts/reference/` actually contains 966 PNG court
  photographs, not a geometric model; adjacent `bounding_boxes/` files are
  YOLO boxes. The user authorized regulation court geometry instead. The
  ITF model is 23.77 × 10.97 m, singles width 8.23 m, service lines 6.40 m
  from the net ([ITF](https://www.itftennis.com/en/about-us/organisation/tennis-glossary/));
  the elevated net is excluded from the planar painted-line model.
  No pretrained model, new environment or weights are needed: CPU OpenCV
  4.10.0, NumPy 1.26.4 and SciPy from the existing packed environment.
  The rule above is unchanged. White-line Hough hypotheses initialize
  RANSAC against all nine painted line segments, with the preceding frame
  as a warm start. Every decoded frame is measured at 1080p after K4's
  limited-range 8-bit YUV420 conversion. Directed chamfer is the mean
  distance from model-line samples at ≤1 projected-pixel spacing to
  detected white-line pixels; a frame registers at ≤2 px with support
  on at least six segments. Chamfer measures only in-frame projected line
  portions (at least half the projected samples visible); failed fits
  remain missing, never dropped from the error distribution. Summary uses
  the ordinary median and linearly interpolated 95th percentile, with
  infinite failures retained (fixed before user labels).
  Initial ffmpeg scene threshold 0.25 is provisional for the user's
  review (candidate scores ≥0.10 retained); the decision thresholds are fixed.
  Mixed shots are false positives if predicted court and their duration
  counts as missed court time. Registration is summarized across all
  user-labelled court frames, with missing fits treated as infinite error.
  CPU smoke: first 1,500 frames (30 s), ≤600 s; substantive validator
  checks frame coverage, shot partition, real registered frames, finite
  homographies, 80% classification and review images. Pilot/whole-match
  stages ≤2,700 s; initial ceiling 3,500 CPU wall seconds per job,
  16 CPU threads, no GPU allocation. The smoke establishes scale before
  any whole-match run. Focused synthetic checks recover a known court
  homography and reject blank frames; these are correctness checks, not
  match evidence. Outcome is pending the user's shot labels.
  Initial pilot `20261016T090118Z-23f79a90` (gpu1, CPU, `7349881`)
  was stopped during its smoke: `ffprobe -count_frames` was decoding the
  whole 2,310 s match on one CPU before the selected 30 s could run.
  No registration result or decision evidence was produced; task artifacts
  are retained. The revision uses K4's `-count_packets` convention instead,
  and checks each requested decoded frame while processing. Chamfer
  sampling was also corrected to uniform projected-pixel spacing (covered
  by a focused regression test) before any decision run.
  Corrected pilot `20261016T090621Z-91db7939` (gpu1, CPU, `ddd457c`)
  failed its timing guard in 3.2 s: the source is CFR 50/1 but starts at
  PTS 0.020 s. The implementation now explicitly rebases scene-score PTS
  with `setpts=PTS-STARTPTS`; shot indices remain zero-based presentation
  order, and ffmpeg seeks are relative to stream start. The source offset
  is recorded rather than requiring zero PTS. No registration result was
  produced. Neither diagnostic run supports the decision.
  Third diagnostic `20261016T090920Z-467b9cc2` (`999b899`, gpu1, CPU)
  exposed an ffmpeg scope bug before registration: `-frames:v` counts
  selected scene frames, so it did not bound input decoding. Stopped the
  owned job and retained its artifacts; scene scanning now has an input
  `-t` bound. An actual ffmpeg regression fixture checks that a cut beyond
  the selected input interval is excluded. No registration evidence yet.

- Registration diagnostic `20261016T091435Z-0a586c9d` (`84dbe11`,
  gpu1, CPU) completed all 1,500 frames / 4 shots in 228.2 s, predicted
  zero court shots. Its validator failed because the fleet archives and
  removes scratch before validation; the corrected validator reads
  `published.tar` as K4 does (regression-tested). No full stage ran.
  Visual QA found clipped doubles lines and grass texture in the initial
  colour mask. Before the next smoke, the detector now requires coherent
  long straight support, measures in-frame projected portions (rather
  than rejecting a clipped court), and retries independent hypotheses
  when a previous fit is poor. These are implementation corrections;
  the decision thresholds are unchanged and this diagnostic is not used
  for adoption. At this throughput the match is split into four balanced
  groups of whole shots, each saving every finished shot, ≤45 min per
  stage, then assembled into a new review directory. Assembly checks
  identical inputs/cuts and every source frame exactly once; completed
  archives remain unchanged. Initial ceiling remains 3,500 wall seconds
  per CPU job, 16 CPU threads, no GPU allocation.

- Revised smoke `20261016T092749Z-37f37559` (`b96ad65`, gpu5,
  CPU, no GPU) passed all ten substantive checks: all 1,500 frames and
  4 shots covered, images and finite homographies present, actual frames
  registering under 2 px, classification exactly at the fixed 80% share.
  Runtime 245.6 s; zero of four shots predicted court. This negative
  prediction result is retained; it is not accuracy without the user's
  labels. The 6,000-frame pilot completed in 313.3 s: 15 shots, 2
  predicted court. All requested frames were processed. A four-part
  full run is estimated at about 33 min per part including the repeated
  whole-match cut scan, within the 45 min stage cap (16 CPU threads,
  no GPU). Before the decision run,
  reports also record resolved ffmpeg/ffprobe paths and versions plus
  stream colour/codec/timestamp metadata. No model or threshold changes.

- Whole-match run submitted (CPU only, `43d32d9`; pending outputs).
  Four groups of balanced whole shots, same complete source and scene
  score scan, each its own ≤600 s smoke, ≤2,700 s full stage and 3,500 s
  total budget. Native tool/codec/colour metadata are in each report.

  | Part | Job | Host (CPU; GPU none) |
  |---|---|---|
  | 0 | `20261016T093935Z-caed3e5a` | gpu1 |
  | 1 | `20261016T093936Z-9d24b59f` | gpu1 |
  | 2 | `20261016T093938Z-c7c96b10` | gpu5 |
  | 3 | `20261016T093938Z-90023e70` | gpu5 |

  Source snapshot SHA256 `1a7e221992ff1d5851c298ede352f7489ec701126fea73b132fb81b58ce645f4`;
  clean selected patch SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  Input manifest `inputs-4d99d200bcbc.json`, SHA256
  `4d99d200bcbcc4279ac74781b6529cfbce1835a8afe47a4e94c92b76616327b1`;
  video SHA256 `676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66`.
  Packed environment `pointstream-20261009T214002Z.tar.gz`, SHA256
  `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
  Commands/specifications are preserved in each job's `dispatch.json`,
  stage metadata and `tennis/k8-2026-10-16/specs-v2/k8-full-<part>.json`.
  Review serving uses `tools/datasets/k8_review_server.py` (read-only,
  loopback only, byte ranges for seeking the source); marks stay in the
  browser until downloaded. UI layout/server changes after `43d32d9`
  do not change registration, cuts, predictions or archived evidence.

- Full-run resource failure and bounded recovery (2026-10-16).
  All four full stages reached their enforced 2,700 s allowance; none
  produced a complete whole-match report. Preserve their `partial.tar`
  files unchanged. They contain 232 complete shots / 70,383 frames of
  the 360 shots / 115,505 frames in the retained cut table. Exactly
  128 shots / 45,122 frames remain. No adoption result is inferred.
  The initial pilot underestimated hard line-search frames and Python
  thread contention. Recovery reuses the SHA256-identified cut table
  and completed archives, processes only missing frames, uses separate
  CPU processes, and checkpoints chunks of ≤240 frames. Homographies
  reinitialize at chunk boundaries; the same line detector, court model,
  RANSAC, chamfer and decision thresholds are retained. Per-item code
  and archive provenance distinguish the original and recovery results.
  Whole-shot classification is recomputed from every frame after merge;
  review images remain the original shot's exact first/middle/last.
  Recovery budget: 12 balanced missing-frame groups (~3,760 frames each),
  16 CPU threads / 8 processes, no GPU; smoke ≤300 s on 120 frames of
  known court shot 9; pilot one complete missing group, full ≤1,800 s,
  total ≤2,400 s per job. Only after that pilot passes are the remaining
  groups submitted. Validator checks frame coverage, bounded chunks,
  real registration, homographies, review images and unchanged shot rule.
  Tests verify preservation, exact chunk coverage and fail-closed assembly.

- Recovery pilot `20261016T110218Z-1bd79811` (`5432db0`, gpu4,
  CPU) passed its smoke (120 frames, 41.3 s processing / 44.5 s command)
  and completed group 0: 22 chunks / 3,756 frames in 246.7 s processing
  / 249.4 s command. This establishes the actual chunk/process path at
  the full group scale, with about 7× margin to the 1,800 s cap. Hardware
  and source identities are revalidated by each remaining job's own gate.
  The retained evidence and manifest are read-only. Recovery manifest
  SHA256 `5af2f0a3c4bf6b1f04b7ed145121f3989c0a9c57a6ff3a7ee303c70af577341f`.
  Remaining jobs select the committed `experiments/`, `src/` and
  `pyproject.toml` runtime dependency set, avoiding unnecessary NFS file
  creation; each records that source selection and its snapshot checksums.

- Remaining recovery requests (CPU only; `16b19ab`, same worker/court
  file contents as the successful pilot; runtime-only source snapshot
  SHA256 `c1bc200d99b2c1b4155c08cf7081a0382da99f52d991536b60801556182c0580`).

  | Group | Job |
  |---|---|
  | 0 (pilot) | `20261016T110218Z-1bd79811` |
  | 1 | `20261016T112031Z-242208de` |
  | 2 | `20261016T112031Z-139b7c17` |
  | 3 | `20261016T112032Z-8cd85120` |
  | 4 | `20261016T112033Z-cea7591b` |
  | 5 | `20261016T112050Z-f71ef62e` |
  | 6 | `20261016T112056Z-8d34945b` |
  | 7 | `20261016T112057Z-b93576b9` |
  | 8 | `20261016T112057Z-27c43737` |
  | 9 | `20261016T112101Z-863308e5` |
  | 10 | `20261016T112058Z-fed578eb` |
  | 11 | `20261016T112059Z-5522e2a8` |

  Concurrent source transfers suffered connection failures. Canonical
  status and shared-inbox inspection confirmed groups 2–11 had no ready
  record, owner, state or run; no workload had executed. Their original
  requests were continued, not resubmitted: the surviving complete tar
  matched every original snapshot checksum, partial source directories
  were preserved, restored source identities were checked against the
  already-published group 1, and the frozen fleet publisher completed
  publication under the original IDs/specifications. Source identity
  `df2516b1506a553ade44b637c471714c5cd040a261aa75c812d788322fcaba32`.
  Receipt: `recovery-01/publication-repair-receipt.json`; helper
  `tools/datasets/k8_publish_existing.py` refuses any ready/claimed/run
  state. No duplicate job was launched and all normal smoke gates remain.

- Review ready (2026-10-16; assembly revision `edcf474`). All twelve
  recovery jobs above completed their gates and full stages on CPU.
  Processing time per group 0–11: 246.7, 322.9, 233.2, 228.4, 203.0,
  288.1, 269.7, 461.4, 485.8, 468.6, 431.2, 507.5 seconds. Assembly
  verified the preserved archive hashes, recovery manifest identity,
  source-video SHA256 and exact non-overlapping coverage of all 115,505
  frames / 360 original shots. No completed shot was rerun. Each shot
  has its exact first, middle and last image (1,080 JPEGs); the middle
  carries the projected court when a homography exists. Predictions:
  41 court, 319 non-court. These are predictions, not measured accuracy.
  The browser was opened and verified at 0/360 marks; no labels supplied
  by the agent. Source video seeking uses a read-only loopback server.
  Final page, relative to the run root:
  `review/review.html`; review ID `63f9a0d78069c771`.
  Absolute run root:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k8-2026-10-16/`.
  `review/k8-report.json` retains every per-frame homography/error,
  conversion command, native-tool path/version, input/archive checksum,
  original and recovery job identity, and selected worker checksums.
  FFmpeg/ffprobe 8.1.2 from the packed environment supplied the native
  decode/scene paths; no encoder or GPU was used. The full scene command
  is explicitly reconstructed from the retained full cut scan, rather
  than mislabeled with the base smoke's 30-second invocation.
  Report SHA256:
  `530696e0c6b70dbbbe0939de295faa4a9c83a07140e7586038784bc2e44ce4cf`;
  HTML SHA256:
  `4a16e6f746fc1de6442df76e8334df2b7ac091a3fd082d3096205beee8af8a1e`.
  Required checks passed: ruff, mypy, pytest (574 passed, 19 skipped).
  **Outcome by the rule: pending the user's complete marks JSON.**
  Precision, duration-weighted recall, registration median/p95 and
  per-class misses will be scored only after those marks return.
  No adoption decision or threshold change has been made.

- Amendment 1: a stratified sample of shots to mark (2026-10-20, written
  before any mark; Claude takes over from `codex/k8-court-shots` on
  `claude/k8-court-shots`). The user asked for cheaper marking than all
  360 shots. **What changes:** the user marks (a) every predicted-court
  shot (all 41) and (b) a Poisson sample of the 319 predicted non-court
  shots, drawn with probability proportional to duration:
  π_i = min(1, n·d_i / D), with D the stratum's duration and the cap
  applied iteratively. Shot i is in the sample when u_i < π_i. Here
  u_i = sha256("pointstream-k8:<shot id>") / 2^256 is fixed and blind to
  content. The expected size is n = 50, about 90 shots in all, at about
  10 s per shot, or roughly 15 minutes. The sample is nested in n: a
  larger n keeps every shot already drawn. **Why:** precision needs only
  the predicted-court shots, and recall needs only an estimate of the
  court time hidden among the predicted non-court shots. **Estimators.**
  Precision is exact, from all 41 shots. Duration-weighted recall is
  R = A / (B + Y). A is the duration of predicted-court shots marked
  court. B is the duration of predicted-court shots marked court or
  mixed. Y is the duration of predicted non-court shots that are court or
  mixed, estimated by Horvitz–Thompson as Ŷ = Σ_sample y_i/π_i. Because
  π_i = d_i/c below the cap, each sampled positive adds exactly c seconds,
  so Ŷ = Y_capped + c·k for k sampled positives. The 95% interval on Y
  takes the exact Poisson (Garwood) interval for E[k], scaled by c. k is
  a sum of independent Bernoullis, less dispersed than a Poisson, so the
  interval errs wide. R's interval follows by monotonicity:
  [A/(B+Y_hi), A/(B+Y_lo)]. Registration error is measured on all true
  court frames. Frames of sampled shots carry weight 1/π_i, and missing
  fits stay infinite. The median and 95th percentile use the weighted
  linear interpolation that reduces to the ordinary median and the
  linearly interpolated percentile when all weights are equal. Misses
  are reported by class, both as sampled counts and as weighted estimates
  of count and duration. **Unchanged:** precision ≥ 0.95, duration
  recall ≥ 0.90, median ≤ 2 px, 95th percentile ≤ 6 px, the 80%
  shot rule, the classes, and mixed shots counted wrong for both.
  **Reading the recall interval.** If the interval lies wholly above or
  wholly below 0.90, the recall condition is decided. If it straddles
  0.90 while the other conditions pass, stage 2 extends the same draw to
  n = 150. Stage-1 marks are kept, and about 85 more shots add about
  15 minutes. Recall is then decided on the point estimate R̂, with the
  interval reported, and a still-straddling interval is stated as
  "within sampling error of the bar". The arithmetic behind this: with
  A ≈ 449 s and no positive in the sample, n = 50 gives a recall lower
  bound near 0.77, and n = 150 gives about 0.91. If precision or
  registration already fails, no second stage runs. The 360-shot review
  (`review/`, ID `63f9a0d78069c771`) and its report stay unchanged. The
  sample page is a new directory beside it, `review-sample-01/`.

- Sample review ready (2026-10-20, `e43215b`, CPU, no re-registration).
  Command: `python -m experiments.tennis.k8 sample --report review/k8-report.json
  --out review-sample-01`, run on the published report (SHA256 `530696e0…`,
  checked). The draw gives 41 + 41 = 82 shots to mark. Of the predicted
  non-court shots, 41 are sampled (expected 50; Poisson sampling varies the
  count), covering 548 of their 1,861 s. Scale c = 37.2 s, largest π = 0.94.
  Sample review ID `fb6cfe8a81f9a440`. `k8-sample.json` SHA256
  `91738f886955d25c72f2a7c04f9d723bc82a6def5a4cd2afa52bd69e9dbaa3b1`,
  `review.html` SHA256
  `92b2ed13234bf4bf78de53a200ce57ebe1db90fd111dfa2d1df64c59b4ef1dd4`.
  Images and source are relative links into `review/`. The page is served
  read-only on gpu1 loopback (port 17689) and reached through an SSH
  tunnel. It was checked at 0 / 82 marks; the agent supplied no labels.

- **Outcome (2026-10-20): not adopted**, decided at stage 1 with no second
  stage. Scored with `python -m experiments.tennis.k8 score` (scoring code
  as in `e43215b`) on the user's marks, `k8-marks-fb6cfe8a81f9a440.json`
  (SHA256 `eeba025b…cb4a2`, 82/82 marked), against the published report
  `530696e0…`. Score `review-sample-01/k8-score-sample-01.json`, SHA256
  `2ac7bebb4a7fbed8e7a6174cc7582cb017ff2df81f245b3861a0256e27b0c405`.
  The registration jobs are the four original parts and twelve recovery
  groups listed above (CPU only, no GPU).

  | Condition | Bar | Measured | Passes |
  |---|---|---|---|
  | Precision (exact, 41 shots) | ≥ 0.95 | 1.00 (41/41) | yes |
  | Duration recall (HT, 95% CI) | ≥ 0.90 | 0.34 [0.26, 0.45] | no |
  | Registration median, true court frames | ≤ 2 px | 1.43 px | yes |
  | Registration 95th percentile | ≤ 6 px | 24.5 px | no |

  The interval lies wholly below the bar, so no stage 2 runs. On predicted
  court shots alone the errors are median 0.91 px and p95 2.52 px. The p95
  fails because of the true court shots the classifier missed. Of the 41
  sampled non-court shots, 23 are court or mixed. The weighted estimate is
  about 856 s of court time hidden among the predicted non-court shots
  (633 s court, 223 s mixed), against 449 s found. **Misses by class**
  (sampled shots):
  1. *Near misses, the 80% shot rule* (11 court shots: 0000, 0002, 0012,
     0017, 0036, 0048, 0088, 0139, 0247, 0290, 0294). A homography exists
     on almost every frame, with a median chamfer of 0.9–2.0 px, but 20–50%
     of frames exceed 2 px (p90 3–4.6 px). These are per-frame errors in
     a shot that does register.
  2. *Wrong or no fit* (6 court shots: 0100, 0123, 0172, 0201, 0223,
     0259). The median is 4–20 px, or there is no fit (0201).
  3. *Missed cuts* (6 mixed shots: 0031, 0062, 0079, 0135, 0181, 0203).
     The 0.25 scene threshold merged a court and a non-court shot.
  The user decides how to proceed. Any change of method needs a new rule
  judged on shots not yet marked; these 82 marks have now been seen.
  Close-up, crowd and other shots were never predicted as court, so the
  competing explanation (court-like structures in non-court shots) is not
  observed.
- **Decode offset found by the user (2026-10-20).** Each shot's first
  review image was its predecessor's last frame. The cause: ffmpeg's input
  `-ss` counts from the container start (0, the audio), while cut indices
  count from the video's start (0.020 s, `setpts=PTS-STARTPTS`). So
  `-ss start/fps` decoded from frame start−1. Verified on the broadcast:
  the first cut is n = 593, and the old seek decoded n = 592. In the
  published report the registration row labelled f is therefore source
  frame f−1 for every shot except shot 0. Within each recovery chunk
  that starts inside a shot, one frame is measured twice and one is never
  measured. The effect on this outcome is negligible: a corrected shift
  flips one prediction (shot-0288, a one-frame shot), and recall and p95
  stay far from their bars. The archived evidence is unchanged.
  The decode now seeks to (video start − container start) + (start − ½)/fps
  (`seek_offset`, `seek_seconds`). A regression test with real ffmpeg
  (an MKV whose audio starts at 0 and video at 0.02 s) passes with the
  packed ffmpeg 8.1.2 on gpu1. Any later registration run uses the fix.
  Until then, per-frame homographies from this run must be read at
  frame f−1.
- **Partial court-shot list v0 for K4b (d)** (2026-10-20, the user's choice
  of options B and A). `court-shots-v0/k8-court-shots-v0.json` under the
  run root (SHA256 `f1efc61dafe6ebd75303a6448ddb5180eaec70dee9e329fde8517674088dad7a`):
  the 41 predicted-court shots, all marked court, 22,460 frames.
  Per-frame homographies are re-indexed for the decode offset; the row
  for a shot's last frame is taken from the next shot. It covers about a
  third of court time, so absence from the list does not mean non-court.
  K8b (revised cuts and registration) follows under its own rule.

### 2026-10-20 — K8b: court recall in shade and across venues (rule)
- Question (PLAN K8, after K8 was not adopted). Can a revised registration
  meet K8's unchanged decision rule on footage the method was not
  designed on?
- Diagnosis on dev data: K8's 82 marked shots, which may now be used for
  design. K8's line detector keeps only pixels with value > 170 and
  saturation < 80. In shade (court grass at value ≈ 90, a line at ≈ 133)
  it finds no lines. Late in the match the near half of the court is in
  shade, so those shots get a wrong fit (one line off) or none (0172,
  0201, 0223, 0259). The remaining misses fail only the per-frame 2 px
  bar of the shot rule (2–4 px during pans). Four of the six mixed shots
  switch from no-fit to fit with no scene-score cut, which looks like a
  gradual transition.
- Method (each piece checked on one clean frame from each of the 82 dev
  shots, decoded without warm start):
  1. *Relative line detector.* A line pixel is brighter than its local
     background (morphological opening, 31 px) by at least
     max(10, 0.10·background), and less saturated than its surroundings
     (closing, ≤ 0.75 of them). Support from long Hough segments is kept
     as in K8.
  2. *Hypotheses.* The previous frame (warm start, reset at cuts), then,
     when it fails or exceeds 2 px, the seeds from both K8's detector and
     the relative one. Each is refined against both masks, and the fit
     with the lowest error is kept. A bank of the match's own good
     homographies added one frame at twice the cost, so it is not used.
  3. *Ruler.* The error is the directed chamfer from projected model
     lines to the 1 px centerlines of the detected Hough segments.
     Measuring against a mask's full width would let a dense mask
     flatter the fit; on the dev frames the two agree within 0.7 px.
  4. *Spurious-fit guard.* Clutter is the share of the projected court's
     interior, more than 6 px from any model line, that is covered by
     detected pixels. A fit needs clutter ≤ 0.25, at least 6 supported
     segments and plausible geometry (as in K8). On the dev frames, court
     fits had clutter ≤ 0.085, and the two spurious fits (a close-up and
     a crowd shot) had 0.52–0.66.
  5. *Shot rule.* A frame *fits* at error ≤ 6 px, the p95 bar. A shot is
     court when at least 80% of its frames fit. The per-frame 2 px bar
     no longer gates the shot; registration accuracy is judged only by
     the decision rule below.
  6. *Cuts.* Scene score ≥ 0.25 as in K8. In addition, a shot is split
     where its per-frame fit state changes and the new state lasts at
     least 1 s. Shorter runs are absorbed into their neighbours.
  7. *Decoding.* Each clip is decoded sequentially from its start, with
     no seeks, so the decode-offset bug cannot arise.
  On the dev frames, 53/58 court frames fit within 2 px and 56/58 within
  6 px; K8's method got 42 and 52. The 18 non-court frames had 0 fits.
  These dev numbers are design evidence only.
- Held-out data: the 70 scene clips of the three other archived
  broadcasts (`tennis_games/alcaraz_hurkacz`, 21 clips, 6,968 frames at
  1080p50; `djokovic_zverev`, 8 clips, 20,320 frames at 2160p60, converted
  to 1080p by K4's conversion; `medvedev_struff`, 41 clips, 10,854 frames
  at 1080p50). These are different venues, not seen in design, and they
  are the clips K4b's windows come from.
  nadal_kyrgios is not re-scored: its marks were used for design.
- Marks: a review page, as in K8, of every shot the method produces in
  those clips (a census, so no sampling; about 100 shots, about 15–20
  minutes). The user marks the true class with K8's classes. The page
  shows predictions and the projected court as before.
- Decision rule (K8's thresholds, unchanged): adopted when, on the
  held-out shots as marked, court-view precision ≥ 0.95, court-view recall
  ≥ 0.90 weighted by duration, and on all true court frames (missing fits
  infinite) a median error ≤ 2 px and a 95th percentile ≤ 6 px with the
  ruler above. Mixed shots count wrong for both. Otherwise the misses are
  reported by class and the user decides. Limitation, stated beforehand:
  the clips were cut by the old pipeline around points and are mostly
  court views, so precision is weakly tested there. The 0/18 dev
  non-court fits are the only other precision evidence. If K8b is
  adopted, the nadal_kyrgios whole-match run (v1 of the court-shot list)
  comes next, and its predicted-court shots are spot-checked.
- Hypothesis: the shade failure and the 2 px gate cause most of K8's
  misses. Recall and registration pass on other venues, whose lighting
  and surfaces differ.
- Competing explanation: the relative detector is tuned to Wimbledon
  grass. Hard or clay courts with logos, shadows or different line
  contrast fail registration, or spurious fits appear on textured
  surroundings that the clutter guard does not catch.
- Budget: CPU only, on the fleet. At about 0.5–1.6 s per frame per
  process, 38k frames take about 20–60k CPU seconds. Parts are whole
  clips, 8 processes, checkpointing every finished clip. Smoke ≤ 600 s
  on one short clip; full stages ≤ 2,700 s; parts sized from the smoke's
  measured rate. No GPU.
- Pilot smoke `20261020T215039Z-fdd6bf8c` (`78f6fa0`, gpu4, CPU).
  An earlier submission, `20261020T215018Z-5357cd95`, snapshotted the
  main checkout (`dabfa5c`, which has no K8b code) and was cancelled
  before it ran anything. The smoke processed 240 frames of
  `djokovic_zverev/000` in 524 s and failed its validator: no frame fit.
  That clip is a court-level practice camera (4K60). Diagnosis: seven
  model lines fit within 2.4 px, but both singles sidelines were off by
  39 px. The fit had locked the model's doubles sidelines onto the court's
  singles lines. The real doubles lines leave the frame, the seed only
  takes sidelines that cross 60% of the frame height, and K8's geometry
  guard (corners within ±10% of the width) rejected the correct fit,
  whose near corners lie 25% beyond each side. Unguarded, the correct fit
  is 1.22 px on all nine lines. It is not evidence.
- Amendment 1 (2026-10-21, before any held-out shot is reviewed).
  (a) K8b's geometry guard allows corners up to 50% of the frame width
  beyond either side; K8's guard is unchanged. (b) When the best fit
  exceeds 2 px, it is re-seated on the model's four line-swap pairings:
  singles↔doubles sidelines and service lines↔baselines, as model-space
  scalings about the court centre. This is the same one-line-off
  ambiguity as K8's class-2 misses. (c) Each frame computes the distance
  transform once per mask, which changes speed only. No threshold
  changes. Because the fix was found on a held-out clip,
  `djokovic_zverev/000` (3,975 frames, 66 s) becomes dev: it is still
  registered (K4b needs it) but is neither reviewed nor scored. The
  held-out census is the other 69 clips.
- Runs (CPU only, no GPU; packed environment `32e68872…`; manifest SHA256
  `82a582e0246f1a3a0fce2fbe1f694097fe7b47dca8cf774cb3f897b3a51ae00a`).
  The corrected 4-part pilot `20261020T220814Z-c4b9d013` (`0465976`,
  gpu1) passed its smoke (240 frames, 524 s, about 2.2 s per frame per
  process) and reached its 2,700 s full-stage cap at 25 of 34 chunks.
  That code published clips only at the end, so `partial.tar` is empty;
  its `checkpoint.tar` keeps the finished chunks. The artifacts are
  retained and nothing was replayed. Resized to 8 parts × 16 processes
  with 200-frame chunks; each clip is finalized and published as soon as
  its chunks finish, longest clips first (`913b815`). Pilot part 0,
  `20261020T225829Z-3cb8b9aa` (gpu6): smoke 160 frames in 183 s; full
  stage 6 clips, 4,484 frames, in 499 s. Its validator passed on the
  smoke (fleet) and on the full stage (locally, from `published.tar`).
  Parts 1–7 were then submitted on the same revision (`git_head` checked
  in each `ready.json`):

  | Part | Job | Archive SHA256 |
  |---|---|---|
  | 0 | `20261020T225829Z-3cb8b9aa` | `3910621c…` |
  | 1 | `20261020T231141Z-bff7fd07` | `8e61d922…` |
  | 2 | `20261020T231147Z-a44fc8b0` | `86c30e3e…` |
  | 3 | `20261020T231153Z-d859538d` | `8e7d98c6…` |
  | 4 | `20261020T231200Z-23e16ed2` | `d1652905…` |
  | 5 | `20261020T231206Z-50159441` | `db7ad8b0…` |
  | 6 | `20261020T231213Z-19ea5d2d` | `c0317166…` |
  | 7 | `20261020T231220Z-a7110d20` | `0e4df826…` |

  All eight full stages completed (CPU, no GPU). They took 485–564 s on
  gpu5 and gpu6, and 2,446–2,599 s for parts 5–7, which shared gpu3
  (64 cores). Each passed its validator on the smoke (fleet) and on the
  full stage (locally). Assembled with `k8b assemble` (`a87dd2e`): 70
  clips and 38,142 frames, each clip exactly once, every review image
  present. Review `review/` under the run root, ID `13f69f53873866a2`,
  `k8b-report.json` SHA256 `8230bf1a2239978dfb7912bd5f3ed8367f5c85d1ea9e75c42e8ab8fb3fdc3314`,
  `review.html` SHA256 `a02a45a1a3a064fac69a95693699869650bdc2da3cd029b61d11685988b498f7`.
  Every clip forms a single shot: no cut at ≥ 0.25 and no persistent
  change of fit state inside a clip. 69 shots go to the user (the dev
  clip is excluded). 44 are predicted court (723 s) and 25 non-court
  (45 s, all medvedev_struff). These are predictions, not accuracy. The
  page is served read-only on gpu1 loopback (port 17690) through a
  tunnel, and was checked at 0 / 69 marks; the agent supplied no labels.
- **Outcome (2026-10-21): adopted.** Scored with `k8b score`
  (`76a7db2`) on the user's marks `k8b-marks-13f69f53873866a2.json`
  (SHA256 `8f7f665c…3c0dcd`, 69/69 marked) against `k8b-report.json`
  (`8230bf1a…`). Score `k8b-score.json` under the run root, SHA256
  `a914eceed27daf30c46881ee336b4c5505695afa9093830c4db990d2d00b2675`.
  The jobs, inputs and revision are in the table above (CPU only).

  | Condition | Bar | Measured | Passes |
  |---|---|---|---|
  | Precision (44 predicted court shots) | ≥ 0.95 | 1.00 (44/44) | yes |
  | Duration recall (723 s of true court) | ≥ 0.90 | 1.00 | yes |
  | Registration median, 31,914 true court frames | ≤ 2 px | 0.77 px | yes |
  | Registration 95th percentile | ≤ 6 px | 2.71 px | yes |

  Every shot matches the user's class. alcaraz_hurkacz has 21 court
  shots and djokovic_zverev 7. medvedev_struff has 16 court shots plus
  21 close-ups, 2 crowd shots and 2 graphics, all predicted non-court.
  The hypothesis holds: shade and the per-frame 2 px gate caused K8's
  misses. The competing explanation (failure on other surfaces, or
  spurious fits on texture) is not observed: the method held on hard
  courts and a court-level camera. Scope, as stated in the rule: precision
  rests on 44 shots (one-sided 95% lower bound 0.934, Clopper–Pearson).
  Only 25 held-out shots are non-court, and no replay or picture-in-picture
  appears. The clips were pre-cut by the old pipeline, so they test
  in-clip cuts and fit-state splits only weakly. Registration is measured
  with the centerline ruler (rule, item 3).
  **What K4b's candidate (d) can use now:** per-frame court homographies
  (ITF metres to 1080p pixels; null where there is no fit) and the court
  outline for every frame of these 70 clips, including the dev clip
  `djokovic_zverev/000`. They are in `clips/*.json` inside each part's
  `published.tar` and in the assembled `review/k8b-report.json`
  (`registration[n].homography`, n = the clip's decoded frame index).
  Still to run with the same adopted method: the clips of
  djokovic_federer and rublev_auger-aliassime (the two other matches K4b
  draws from), and nadal_kyrgios's whole match (v1 of the court-shot
  list).
- Application to K4b's other two matches (2026-10-21, the adopted method
  unchanged; a production run, not new evidence for the rule).
  djokovic_federer (23 clips, 4K) and rublev_auger-aliassime (66 clips),
  `<match>/videos/*.mp4`: 89 clips, 31,390 frames. Manifest
  `k8b-2026-10-20/k4b-matches/manifest.json` (SHA256 `6c2be1a4…`). Eight
  parts of whole clips (`3aca368`; CPU only, 16 processes each):
  `20261021T092159Z-9b8ef899`, `…092204Z-1ee9605a`, `…092208Z-804e7d30`,
  `…092315Z-1a7be77c` (part 3), `…092233Z-e808e648`, `…092238Z-019a88a8`,
  `…092243Z-41be5f46`, `…092247Z-91b9e709`. The first submission of part 3
  (`20261021T092213Z-112d5db2`) stopped during source upload: the
  directory holds only a partial `source.tar`, with no spec, ready
  record, owner or run. No worker could claim it and nothing executed,
  so part 3 was submitted once more. The partial directory is left as
  it is.
  All eight full stages completed and passed the validator on the smoke
  (fleet) and on the full stage (locally). Assembled at
  `k4b-matches/registration/k8b-report.json` (SHA256
  `eb7b34b9f8117f5926fb9e15d34bde9a209ce43213e19dc9d9641b38b552e3e0`):
  89 clips, 92 shots, 31,390 frames. Frames that fit: djokovic_federer
  13,710/13,901, rublev_auger-aliassime 10,264/17,489. The six
  rublev_auger-aliassime clips K4b needs that do not fit (003, 006, 039,
  054, 062, 063) were inspected visually. They are close-ups, a tunnel,
  and a player in front of a wall or by the sideline: not court views,
  so the absence of a fit is correct. 061 (fits on 34% of frames) is a low
  courtside view. These shots are not marked (the method is adopted), so
  this is an inspection, not evidence. Consumers use frames whose `fits`
  flag is true; a non-null homography above 6 px (e.g. in 003) is kept
  for inspection only.

### 2026-10-16 — K4b.1: identifying the two players (rule)
- Question (PLAN K4b, item 1). K4's motion rule (the two tracks that travel
  furthest) mislabels a player who stands still (2 of 9 broadcast windows).
  Which content-blind rule picks the two players, judged against players
  marked by hand rather than by the rule's own output?
- Data: frames from broadcast scene clips K4 did not measure, chosen
  content-blind (lowest sha256 of "pointstream-k4b:<match>/<clip>/<frame>"),
  at least 48 frames from at least 8 clips across the 5 matches, court
  views only once K8 is adopted. Person instances from SAM 3.1's "person"
  and "tennis player" prompts, as K4's segmenter.
- Marks: a local review page shows each frame with its person instances
  numbered; the user clicks the two players (or marks one or none visible).
  These marks are the labels; no other labels exist for broadcast players.
- Candidates (content-blind, each a fixed function of the clip):
  (a) motion in body heights (K4's amendment 1); (b) horizontal motion,
  with the clip's last second discounted; (c) a racket mask touching the
  person on most frames; (d) inside the court outline (K8's homography;
  evaluated once K8 is adopted); (e) keypoint motion; (f) the combinations
  (c)+(d), (a)+(d), (a)+(c).
- Decision rule (fixed before marking). Per candidate: the share of marked
  frames where both marked players are its two picks (mask IoU ≥ 0.5 each;
  frames with one marked player count on that one). The candidate with the
  highest share is adopted if it reaches 0.95, ties to the simpler rule;
  it is then applied once to every K4 window as amendment 2, with both
  earlier definitions' numbers kept. If none reaches 0.95, the table is
  reported and the user decides.
- Hypothesis: (d) or (c)+(d) wins: players are the people inside the court
  and holding rackets, while ball kids and line judges stand outside the
  lines or hold none.
- Budget: SAM 3.1 instances on the fleet (smoke ≤ 600 s, ≤ 45 min per
  stage); candidates on CPU; the user's marking about 15 minutes.
- Marking-stage implementation (2026-10-16, `codex/k4b-players`; rule
  above unchanged). Zero-based decimal frame indices and clip stems as in
  K4. Enumerate all frames in all 159 scene clips in the five `MATCHES`
  directories; exclude K4's ten broadcast clips in their entirety, including
  the outro card. Take the smallest **global** hash prefix meeting all
  three minima, without a content/court filter: 48 frames, 40 clips,
  all five matches (alcaraz/hurkacz 3, medvedev/struff 7, rublev/auger 20,
  djokovic/federer 7, djokovic/zverev 11). Selection is metadata-only;
  `tools/datasets/k4b_inputs.py` preserves the complete inventory, exclusion
  list, hashes, original clip identities and conversion commands.
  Outputs: `Datasets/pointstream-data/tennis/k4b-2026-10-16/`.
- This stage obtains review instances only through K4's
  `Sam31SequenceSegmenter`, offline bidirectional,
  both prompts, K4's limited-range 1080p 4:2:0 conversion and 960×540
  quality-95 JPEG path. It does not estimate clip-level motion from sparse
  review frames. Full temporal clip features for candidates (a), (b), (c),
  (e), (a)+(c) and scoring await the marks; (d) and its combinations await
  adoption of K8. No candidate or K4 amendment is applied at this stop.
  Raw masks of both prompts survive; overlapping prompt masks (IoU ≥ .8)
  share a review number, with the first mask as its scoring mask and all
  source IDs recorded. Page has an original-image toggle, explicit visible
  count, missing-instance flag, per-frame confirmation, browser storage and
  JSON export/import bound to selection and instance-result SHA256s.
- Marking-stage budget: one fleet job, smoke first 3 hash-ranked frames
  through the same path ≤ 600 s, full 48 frames ≤ 2,700 s, validator ≤ 300 s,
  total ceiling 3,600 s, compatible Ada/A6000 pool, shared execution (no
  timing claim). GPU outputs and frame/mask identities must pass the module
  validator for **both** smoke and full before the page is cited. The
  correctness smoke gates production of the page, not scientific adoption.
  Initial CPU preparation failed before conversion because an assumed
  ffmpeg path was absent; its empty archive is retained. Preparation v2
  uses the probed `/opt/local/bin/ffmpeg` and a separate `inputs-v2/`.
  First fleet request `20261016T085803Z-4152bf09` (`32ec13c8`, not evidence)
  was cancelled while pending after review found its validator expected
  local scratch to survive publication. The replacement validator reads
  `published.tar`, like K4's, and is tested after removing scratch.
  Request `20261016T090018Z-00e9fcb6` (`bd38cad9`) was also cancelled
  before allocation: its conservative 24,000 MiB reservation plus the
  sharing margin no longer fitted the available GPUs. The same revision
  and inputs were submitted at 16,000 MiB (recorded SAM smoke peak 11.3 GiB;
  this stage has one frame per session), without increasing the budget.
  Both cancelled requests' revisions were read from their `ready.json`.
- **Rejected marking attempt** `20261016T090231Z-64b221c7`, gpu5 RTX 6000 Ada, shared;
  revision **`bd38cad9bfdc4f082a3a39621a88083851dc500e`**, read from
  `ready.json` (no selected changes; empty patch SHA256 `e3b0c442…b855`),
  snapshot `87f25e2a…bacb4`, source identity `028dc031…da0`, spec
  `be5462a7…b5ce4`. Exact spec `marking-job-16g.json`; selection
  `inputs-v2/selection.json` SHA256 `5e79df8406407fbeeab6ef1ed2a990f39dad6393ecb1109d1fdbe65693fb4088`,
  selected converted frames archive `3af558001d71fbd06eff3d366aafe853de9b1e84c8775ab96c30d30dd3bcf161`.
  Environment: K4's `pointstream-20261009T214002Z.tar.gz`, SHA256
  `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`;
  SAM checkpoint `0567debe…41cb6`. CPU preparation used ffmpeg
  `n7.1.1-56-gc2184b65d2` at `/opt/local/bin/ffmpeg`; exact preparation
  sources, checksums, command and tool versions are retained in
  `preparation-source.tar` and `preparation-provenance.json`. There are
  no native video encodes in this marking stage.
- This attempt's one-frame sessions produced **zero instances on all 48
  frames** (peak 6.17 GiB). The smoke and local full validators passed
  integrity checks but failed to detect this functional failure; the
  subsequent output review caught it. It is not a usable marking page
  or scientific evidence. The archived output and the read-only root
  `index.html` are retained as failed artifacts, not offered for marking.
  SAM's pinned `sam3/model_builder.py:1185` enables masklet confirmation;
  `sam3_multiplex_base.py:251` requires three consecutive detections.
  One-frame video sessions cannot satisfy that gate.
- Correction, with sampling and decision rule unchanged: `inputs-v3/`
  reuses v2's frozen inventory and selection. Around each selected frame,
  eight **real consecutive** frames are decoded (start `frame - 4`,
  clamped to the clip's valid span), through exactly K4's conversion and
  JPEG path. Both prompts propagate on those short sequences with the
  original confirmation settings; only the originally selected frame's
  instances enter the page. All-empty smoke/full outputs now fail
  `nonempty_model_output`, and real contiguous context is checked. This
  supplies masks for manual marking, not the later full-clip candidate
  features. A regression covers published-archive validation after scratch
  removal, mask tampering, and rejection of all-empty model output.
  Replacement budget ≤ 3,000 s (smoke ≤ 600, full ≤ 2,100, validator ≤ 300),
  keeping total GPU execution within the original 3,600 s ceiling.
- Corrected job `20261016T091155Z-7b19ac9a`: revision
  **`dc4d4fdd8421cb9dd53178cf39e922e99e7802ef`**, from `ready.json`,
  no selected patch (empty patch SHA256 `e3b0c442…b855`), snapshot
  `7b9a32f7…3d38a`, source `836c4a3f…7f40b`, spec `fa2ec81f…8b5a0`.
  Inputs `inputs-v3/selection.json` SHA256
  `7d1a22f62ac03c514b5f47dbc8c4c3864d9e6b92d461ab90175d2b520c6c23ef`,
  context archive `3bf99973505a7fb77c68d7155fd56b584c62e3751489f87a0a161a2b46f269e6`;
  original frame IDs, ranks and source-video SHA256s checked equal to v2.
  Exact spec `marking-context-job.json`; preparation source/checksums and
  command `preparation-v3-source.tar`, `preparation-v3-provenance.json`.
  Environment, checkpoint, GPU compatibility and original sampling rule
  unchanged. Page to mark is the corrected `review/index.html`, not the
  failed attempt's root `index.html`.
- **Marking page ready; stopped for user marks.** Corrected job completed
  on gpu5 RTX 6000 Ada (shared). Both the 3-frame smoke and **48-frame
  full output** pass the module validator locally, all 16 checks. Full
  `published.tar` SHA256
  `2dc9acb6541224f9f72af8ceb9c781308c3e62d19fcb819515736e4b7da261ee`;
  `instances.json` `89b78156e85b62741326a5ff2dc8a020ac58934f5dba0ad33a158c887b40f731`;
  self-contained page `2a05d0eaf6a0a480e2caddd5d7714df72376c2f1606847506f22bfbfd49f0223`.
  334 numbered instance-frames across 48 review frames (none empty);
  all 48 original review JPEGs byte-identical to the frozen one-frame
  attempt's originals. Peak allocated GPU memory 10.70 GiB, within the
  16,000 MiB declaration. This is resource/marking validation, not evidence
  that a player-identification rule reaches 0.95; no timing claim.
  Durable page:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k4b-2026-10-16/review/index.html`.
  Companion `instances.json` and `provenance.json` identify the raw masks
  in the job's full `published.tar`. Local validation receipts are
  `context-smoke-local-validation.json` and `context-full-local-validation.json`.
  Browser QA on synthetic frames covered two/one/none, confirmation and
  persistence; the real complete page was checked at 0/48 confirmations.
  No real frames were marked by the agent. Ruff, mypy and the full test
  suite pass (560 passed, 19 skipped). Monitoring paused after completion.
  Next: freeze exact candidate implementations before reading the returned
  labels, compute their clip features and score by the unchanged rule.
  No rule is selected or scored yet; (d) waits for adopted K8 homography.
  Any winner is reported for review of K4's amendment before applying it
  to K4's windows, as required by this task.
- Labels returned (2026-10-16): immutable
  `marks-05f513052625.json`, SHA256
  `05f5130526254316c078003a3a66398c97aa33027ac56519e0f267429eba50ba`.
  Only schema, page identities and completeness were inspected initially:
  correct review/selection identities, 48 entries, 48 confirmed. Individual
  selections, visibility, missing flags and notes were not inspected before
  freezing `experiments/tennis/k4b_rules.py` and its focused tests.
- Candidate implementation freeze (before opening selections). All use
  tracks present on ≥ half the clip frames and choose at most two.
  (a) K4's sum of consecutive centroid distances / median mask height.
  (b) sum of consecutive absolute horizontal displacements in SAM pixels,
  zero weight for the last second. (c) one-pixel dilation of the racket
  mask at 960×540 for contact; require contact on **more than half all clip
  frames**, rank by contact fraction and abstain for unqualified tracks.
  (e) sum of mean displacements of jointly confident (≥ .5) YOLO11x-pose
  joints on consecutive fitted frames / median mask height (K4's native
  whole-frame pose, image size 1280). (a)+(c) ranks first by contact on
  most frames, then (a), filling remaining picks with (a). No other
  combinations without (d) are prescribed. Track ties use canonical
  prompt/native-ID order; candidate ties follow (a), (b), (c), (e), (a)+(c),
  putting the simpler alternatives first. Duplicate prompt tracks are
  matched one-to-one when IoU ≥ .8 on > half their co-visible frames,
  covering ≥ half the shorter track; prefer the person mask.
  Scoring uses injective IoU ≥ .5 matches for two marked players, only
  the marked one for one-player frames, and no visible picked masks on
  zero-player frames. A visible player without a labelled instance cannot
  earn an IoU match and is never silently excluded. The 0.95 bar, sample,
  metric, candidate list and deferral of (d) are unchanged. These functions
  do not load annotations to select any constants.
  A cheap proposal oracle is run first; it bounds **only the review's
  existing masks**, not re-segmentation of the full clips, which could
  return a previously missing instance. Full-clip feature extraction
  covers 30,825 frames across the same 40 clips; it is not approximated
  by the review's eight-frame contexts.
- Label audit after the freeze (`466860d`): 37 two-player, 10 one-player,
  one zero-player frame; no missing-instance flags or notes. All 48 pass
  schema, ID, confirmation and visibility checks. The proposal oracle is
  48/48, scoped to selecting the review's exact proposals. Audit artifact
  `label-audit-05f513052625.json` preserves this check and its per-frame rows.
- Full-clip feature stage preflight (before GPU runs). Hypothesis: contact
  plus motion improves over centroid travel when a player stands still;
  competing explanations are SAM racket misses, camera motion, pose misses
  and track fragmentation. Frozen formulas and human marks are not tuned.
  Every one of the 30,825 native source frames is converted once by K4's
  ffmpeg path and used for SAM person/player/racket and YOLO11x-pose.
  The input manifest excludes human selections. Pose runs before SAM and
  releases its model; SAM's official CPU video/state offload is enabled
  for long clips. The smoke compares all class/frame unions to native
  non-offloaded inference at IoU ≥ .999; offload is part of config provenance.
  Defaults for existing SAM callers remain unchanged. Full-clip native
  SAM IDs are retained; cross-prompt aliases and scalar features are CPU
  operations, with raw SAM masks, raw pose detections and scoring-frame
  masks saved so the validator can recompute them independently.
  Pilot axis is clip length, not quality or thresholds: minimum
  `medvedev_struff/007` (64 frames), median `djokovic_zverev/004` (407),
  maximum `djokovic_zverev/000` (3,975). Pilot promotion requires valid
  full artifacts and evidence the longest clip fits one ≤45-minute stage
  and the remaining clips fit the phase budget. No clip is truncated for
  the final score. Each job has a ≤600 s prefix smoke, substantive
  validator, SHA256-identified original video/weights/environment/manifest,
  ≤2,700 s full stage, 300 s validation reserve, and a deadline.
  GPU compatibility: the already-verified A6000/Ada pool, all six hosts
  eligible by model filter, 12,500 MiB allocator limit (recorded SAM peak
  11.3 GiB; CPU-offload pilot must substantiate long-clip use), shared
  execution with no timing claims, 40 GiB scratch. Phase ceiling 8 GPU-hours
  and 10 h wall, including pilots/smokes; record actual usage and reserve
  remaining job allowances before subsequent submissions. Failures or an
  unattainable stage/phase bound stop promotion, without enlarging the budget.
  Complete focused tests cover offload flags/provenance, the five rules,
  strict majority, gaps and tail discount, joint visibility, duplicate
  prompts, target-mask export, injective IoU matching and annotation identity.
- Feature pilot inputs: `features-v1/manifest.json`, SHA256
  `4db6651100efc5c450525b154d7f82e9b1b53de47e6699f02aaf2a25739a48b5`;
  40 immutable original-video identities (3,269,018,181 bytes total),
  hard-linked under `features-v1/inputs/`. Exact per-clip specs are alongside
  the manifest. Environment/weights reuse K4's pinned artifacts. Deadline
  `2026-10-17T01:00:00Z`, inside the 10 h wall ceiling from preparation.
  Submitted first, before any long-clip promotion:
  `20261016T151219Z-f30bd26d` (`medvedev_struff/007`, full allowance 300 s,
  total 1,200 s), and `20261016T151309Z-259d7651` (`djokovic_zverev/004`,
  full 446 s, total 1,346 s). Both `ready.json` records identify
  **`ebd0024d79c591fd70bcf868210fe609152ef498`**, snapshot
  `2d882a9c…257e8`, source `dc7ca0b3…b1235`, empty selected patch
  `e3b0c442…b855`; specs `342753ea…53840` and `98efd543…157a7`.
  The then-untracked scorer was explicitly excluded from both snapshots;
  it is not a dependency of feature extraction. No feature result is cited
  until its full output passes the module validator locally.
- Both first pilots failed in the **native comparison** before CPU-offload
  propagation or full promotion. Same diagnosed cause on gpu3 and gpu2:
  a 324 MiB activation allocation exceeded the 12,500 MiB per-process cap
  (11.34 GiB already allocated plus 619 MiB reserved; 12.21 GiB allowed),
  while physical memory remained available. These are failed configuration
  smokes, not evidence about any candidate. Saved logs and artifacts remain
  in the original jobs. Correction: 16,000 MiB allocator cap and 4,096 MiB
  sharing margin, giving a 20,096 MiB admission reservation versus the old
  20,692 MiB; more actual allocator headroom with physical sharing headroom
  still reserved. All input pixels, models, parity threshold, candidate
  formulas, stage timeouts, deadline and 8 GPU-hour phase ceiling unchanged.
  Revised specs are preserved separately in `features-v1/specs-16g.tar`.
  Re-run the short pilot alone first, then promote only from valid evidence.
- The 16,000 MiB short attempt `20261016T152206Z-e078b538` also failed
  during the native comparison (gpu2, before full promotion). Its
  `ready.json` revision is `04808b97db8b1137c0490f109a7695d95081af1a`,
  snapshot `54e80bea…96346`, spec `edd289ea…ba6fc`, empty selected patch.
  A 1.27 GiB detection-head activation exceeded the allocator cap; physical
  memory was available. No candidate outcome was produced. Revised-spec
  archive SHA256 `7773a5949ddbfba8ccbbeb9c4927fa9412840f3da24e1dd434c9307dd88d6701`.
- Revised memory approach, before another smoke: upstream
  `model_builder.py:1191` batches **16** frames for grounding; those
  detector activations caused both cap failures. Explicitly use grounding
  batches of eight, with video/state offload and the same 16,000 MiB cap.
  This changes batching, not temporal sampling: every native source frame
  still runs through the same detector/tracker and fixed candidate formulas.
  The official attribute is asserted effective and included in config
  provenance; default callers retain the original batch. The production
  smoke is still all first 48 frames. Its parity control uses the first 12
  real frames, default batch 16 (effective 12) versus configured batch 8,
  comparing every class/frame union at the unchanged IoU ≥ .999 threshold.
  Both parity mask sets and the comparison are preserved even if it fails.
  No memory cap, time allowance, deadline, phase budget, decision rule or
  human mark is changed by this revision.
- Before admission, cancelled `20261016T153440Z-1b94df2e` (terminal
  cancelled; no stage ran). Snapshot revision `2d2c5fda4203e8dfa499a404fcaa1d000cb4527b`;
  the queued request was also rejected at gpu5 admission for insufficient
  shared-memory headroom. Source inspection found that the pinned top-level
  multiplex `init_state` supports video offload but not tracker-state offload.
  The compatibility adapter had silently filtered the latter flag. Corrected
  it to reject unsupported true requests explicitly (regression tested), and
  request **video offload only**, retaining GPU tracker state. The underlying
  tracker's separate CPU-state option is not exposed by the top-level API;
  no adapter is added to change it. Therefore the longest-clip pilot must
  establish state-memory feasibility before promotion. Earlier references to
  video/state offload describe the requested, not effective, placement.
  Corrected specs preserved as `features-v1/specs-video-only.tar`, SHA256
  `3df90280f2308b23b8ed305a7f2794f4253cc1d643a5d1efa4a13cd681141c6a`.
  Grounding batch eight, parity IoU ≥ .999, all-frame sampling and all budget
  ceilings remain unchanged. Cancellation consumes no GPU-stage allowance.
- Corrected short pilot `20261016T154053Z-91a66242`, gpu3 RTX A6000:
  actual `ready.json` revision `84ec4a3edfa98c0f5778e523e9f0f8b1ef7fe9a3`,
  snapshot `3a1880ea…367e4`, source `97832f14…6c37e`, spec
  `3ee88af7…0459b`, empty selected patch. The immutable spec's generic
  “CPU offload” basis is unchanged from the batch-eight archive; the code
  and recorded effective configuration explicitly request video only.
  No full output or score is cited yet. Local checks: ruff and configured
  mypy pass; pytest 572 passed, 19 skipped (rerun with process-inspection
  permission after the sandbox denied `ps` in an existing contention test).
- The corrected smoke failed its parity gate, before production/full:
  minimum class/frame union IoU **0.9265116431832364**, below .999
  (123 native reference instances on 12 frames). Both RLE sets and
  `parity.json` survive in the job's `smoke/partial.tar`; the failure is
  not a candidate score. Batch eight is rejected. Restore **native batch
  16**, retain video-only CPU offload and the identical parity gate.
  Reserve 24,000 MiB allocator memory plus the unchanged 4,096 MiB sharing
  margin, admitting only GPUs with sufficient headroom; this targets the
  diagnosed native detector activation peak. Timeouts, deadlines, the
  8 GPU-hour phase ceiling, formulas and human marks remain unchanged.
  Specs saved separately as `features-v1/specs-native16.tar`, SHA256
  `a5d55d0b9e56b3765bce7fee955cb0641981d1a49827a6ebbfa43b3a01631264`.
  Re-run short alone; no medium/long/full-corpus promotion from failures.
- Native-batch pilot `20261016T154524Z-9d4d461c` (gpu3 A6000), actual
  ready revision `796933bb91d3828075eb2cfe9a1e0875ced2606b`, passed the
  12-frame offload parity comparison exactly (minimum union IoU **1.0**,
  123 reference instances); failed in the subsequent 48-frame production
  path, before full. A 1.27 GiB request hit the 23.44 GiB allocator cap
  with 20.35 GiB allocated and 3.07 GiB reserved but unused. Preserve its
  partial masks/parity and logs. Test the diagnosed fragmentation remedy
  `PYTORCH_ALLOC_CONF=expandable_segments:True`, with the same native
  batch, 24,000 MiB cap, input pixels, model, gate and all time ceilings.
  Record actual expandable CUDA segments and require them in validation;
  a successful short full and length pilots remain prerequisites to scaling.
- Expandable-allocator short pilot `20261016T154944Z-4081668c`, gpu3
  A6000, actual ready revision `3494a8b1c888c6daeb90c0cabd748fa3fa655754`:
  smoke and full complete. Full **locally validated**, all 17 checks true:
  64/64 frames, original source coverage, raw SHA256s, pinned CUDA SAM and
  pose, effective native batch, actual expandable segments, recomputed
  features and exact target masks. Full peak allocation 22.1047 GiB.
  Smoke parity min IoU 1.0. Smoke execution 154.750 s, full 120.802 s;
  GPU-claim interval 325.555 s. Shared timing is not a performance claim.
  Local receipt SHA256 `bb3cc62943dd42e6a761920d48b239e46358cc056d86afee0f20b95eca5f1be8`.
  These are configuration/full-coverage checks, not a candidate table.
- Length promotion is median only: 407 frames of `djokovic_zverev/004`.
  The initial 446 s full allowance has inadequate margin given the short
  stage's fixed loading cost plus per-frame propagation. Use a separately
  preserved median spec with full 900 s, smoke 600 s, validator 300 s,
  total 1,800 s; the 8 GPU-hour phase ceiling and deadline do not change.
  Do not submit the maximum or remaining corpus until median evidence
  establishes feasible stage and phase bounds. Prior failed smoke stage
  time is 390.260 s; charge complete GPU-claim intervals in the phase ledger.
- Median `20261016T155820Z-40b6c9d9`, gpu3 A6000, actual ready revision
  `f58bb8f1890aba12bffa6a7c2bfb8d079f8195be`: smoke passes, full fails
  at frame 160/407 of the first prompt, after 164.207 s. A 1.27 GiB
  request exceeds the 23.44 GiB cap with 21.92 GiB allocated and 344.50 MiB
  unused reserve. This demonstrates growing state pressure; no full result
  or candidate score is cited.
- Next memory-only correction uses the pinned lower-level tracker's
  **existing** `init_state(offload_state_to_cpu=True)` option. The top-level
  adapter still rejects unsupported requests; a separate narrow worker
  adapter forwards the native tracker arguments, verifies each returned
  state's `storage_device == cpu` and flag, and records counts and API
  version. No compute kernels, masks, frame sampling or tracking rules are
  altered. Effective config SHA256 includes this adapter. The existing
  native-versus-offloaded IoU ≥ .999 smoke now also gates tracker-state
  placement; native and offloaded mask archives are retained. Reset the
  initialization counters before production, so parity alone cannot pass
  the production-placement check. Tests verify forwarding, actual storage,
  unsupported API rejection and silently ignored requests. Repeat the
  bounded median pilot directly: its 48-frame correctness smoke precedes
  the 407-frame full stage, with identical 600/900/300 s limits and phase
  ceiling. No further corpus promotion until it and the scale bound pass.
  Source inspection also finds top-level `cached_frame_outputs` retains
  per-frame **boolean** masks on GPU independently of tracker-state storage.
  The same placement adapter stores this cache's already-computed masks on
  CPU after the native cache filter, preserving the original GPU output
  dictionary used by propagation/postprocessing. It changes storage only;
  no masks are recomputed. Record and validate every cached mask's CPU
  placement; adapter version 2 identifies both routes in the effective
  config. Regression test confirms the native output is unmodified and the
  cache copy moves to CPU. The native-mask parity gate covers this route too.
  gpu3 has 384 GiB CPU RAM available at inspection, for long-clip video/state
  storage; the maximum-length pilot must substantiate actual feasibility.
  Parity is strengthened before this promotion: require identical native
  class/track-ID sets and per-instance IoU ≥ .999, as well as class-union
  IoU ≥ .999. The validator recomputes these from both saved RLE archives;
  a regression covers an identity swap with identical union masks. This
  protects motion-based candidates from an offload route that might preserve
  only aggregate foreground. Decision/scoring thresholds remain unchanged.
  Median CPU-state spec SHA256
  `e02b425dfe10a660966f21941b21236b2ddb8955232bdde1eb7053279f3c34cd`.
- CPU-state attempt `20261016T161135Z-a8b70442`, gpu3 A6000, actual
  ready revision `58e9b2215203fbacffc1c8d5f59ea8bb04d3f18c`, failed the
  smoke before parity/full: upstream `video_tracking_multiplex._merge`
  casts the new CUDA tensor's dtype but omits its CPU destination device.
  The neighboring `_append` similarly concatenates without a device transfer.
  Add two guarded, in-memory compatibility replacements: transfer the new
  tensor to the destination tensor's device before assignment/concatenation.
  `_merge` retains its original dtype cast; `_append` retains native dtype
  promotion. No package/environment file is modified. The exact original
  expressions must occur once or installation fails. Save both unified
  diffs, original/revised source SHA256s and patch checksums before GPU
  propagation; include them in effective configuration identity and full
  validation. CPU-only regression reproduces mixed-device rejection and
  checks repaired values, dtype semantics and destination. The native
  mask/track parity gate remains mandatory, followed by the bounded median
  full stage. No candidate formula, score threshold or phase budget changes.
- Corrected CPU-state pilot `20261016T161909Z-5e827d31`, gpu3 A6000,
  actual ready revision `079bf6186f283ca1088817bee0016633481c8a1b`:
  smoke passes fleet validation and full median processing starts. Native
  versus CPU-storage parity: both minimum union and per-instance IoU 1.0,
  identical track-ID sets, 132 native instances. Smoke production records
  3/3 initialized tracker states and 1,125/1,125 cached masks on CPU.
  Compatibility-patch set SHA256
  `1692bdc34ece9f1dc48ec7b6b9354f97c69ba6f282c3648477651a0588f41019`.
  Native package archive remains immutable. No full result is cited before
  local validation. Local full suite after this correction: 575 passed,
  19 skipped; ruff and configured mypy pass.
- Preparing maximum-length confirmation only after local median validation:
  the shared A6000's observed propagation pace does not establish a ≤45 min
  all-prompt bound for 3,975 frames. Prefer the verified Ada class for that
  pilot, checking all hosts and claims again. Add an operational (not scoring)
  smoke scale gate: separately time imports, actual production preparation,
  SAM loading, all-prompt propagation, and CPU export. Exclude the native
  parity control; scale per-frame work to the full clip with 25% margin,
  retain measured fixed load/import costs, add 60 s fixed reserve. Save the
  estimate and its components; validator recomputes it and blocks promotion
  if it exceeds an explicitly supplied full allowance (at most 2,700 s).
  This changes neither masks nor the frozen identification rule. A passing
  estimate only permits the bounded maximum pilot, whose complete output
  still needs local validation and a feasible remaining-phase budget before
  any corpus run. All phase ceilings and deadlines remain unchanged.
- Median full `20261016T161909Z-5e827d31` is now **locally validated**:
  all 22 checks true, 407/407 native frames, recomputed scalar features and
  scoring masks, actual CPU states/cache and checksummed compatibility
  receipts. Full peak 21.5914 GiB; full execution 638.061 s; complete GPU
  claim 850.956 s. Receipt SHA256
  `f15fea18e714481826c327208e30bb011a92fbcfa0f9ddfda48cecf6cc597bc4`.
  Total completed phase claims 2,306.386 s; no other phase job is running.
  Full corpus and maximum remain unsubmitted at this point.
- Interruption: after median completion, automatic approval review could
  not run further read-only inspections because its account usage limit
  was exhausted. No command rejected by that review executed. After the
  stated reset time, the authorized fleet watch succeeded, confirmed all
  jobs terminal, and local validation resumed. The idle monitor was paused
  and the completion event acknowledged; no job was replayed. Absolute
  deadline `2026-10-17T01:00:00Z` and 8 GPU-hour ceiling remain unchanged.
- Maximum pilot specification file `features-v1/maximum-ada.json`, file
  SHA256 `8089be02891006995b91f314dbdc70df91672028b835c874e0b57b8f5a15fe51`:
  original 600/2,700/300 s stage/validation limits and 3,600 s reservation,
  Ada filter only, every host checked. gpu5's two Ada GPUs had 28,521 MiB
  free each and no PointStream claims; gpu5/gpu6 had 504/439 GiB available
  CPU RAM. GPU time reservation plus completed usage is 5,906.386 s,
  below the unchanged 28,800 s ceiling. The smoke forecast separates YOLO
  loading and the first eight-frame prediction batch from subsequent
  steady work, so one-time warmup is not multiplied by clip length. A
  regression checks this distinction and rejects a prefix without any
  post-warmup pose batch. The 25% margin and 60 s reserve stay fixed.
- Maximum pilot `20261016T200442Z-17bfb750` (recorded at takeover; the
  fleet's own record): gpu5 RTX 6000 Ada, shared, revision `6754141`
  from `ready.json`. Its 48-frame smoke passed every check but its own
  scale gate: forecast 5,348 s for the 3,975-frame clip against the
  2,700 s allowance (SAM 43.8 s for 48 frames with three prompts, peak
  21.46 GiB), so it was not promoted. Not evidence. Phase claims so far
  about 2,460 s of the 8 GPU-hour ceiling.
- Takeover (2026-10-20). Codex stopped here; a Claude session continues
  on `claude/k4b-players` (from `codex/k4b-players`, `origin/main`
  merged). Everything above stays as recorded.
- **Amendment 1 (user, 2026-10-20): features over a bounded window per
  marked frame, not the whole clip.** Written before any windowed run.
  - *Window.* 240 frames, K4's window length (K4 measured the first 240
    frames of each clip, and amendment 2 applies the chosen rule to those
    windows). One window per marked frame `f` of a clip of `n` frames:
    start `min(max(f − 120, 0), max(n − 240, 0))`, length `min(240, n)`.
    48 windows, 10,444 frames instead of 30,825; the 10 clips shorter
    than 240 frames are taken whole. Placement reads only the marked
    frame's index, chosen by hash before marking, and the clip's length,
    so it is content-blind and the same for every candidate. Every frame
    of the window is processed; nothing is subsampled.
  - *Why the candidates measure the same thing.* Each candidate is a fixed
    function of the frames it is given: presence on at least half of
    them, motion and contact summed over them. Amendment 2 will give it
    K4's 240-frame windows, so scoring it on 240-frame windows measures
    the rule as it will be applied. The whole-clip features (up to 3,975
    frames) measured an input the rule never sees in K4. Candidate (b)'s
    "last second discounted" becomes the window's last second, the only
    end a rule sees on a K4 window; its frozen formula already zeroes the
    last second of its input, so `k4b_rules` is unchanged. The frozen
    formulas (`RULES`, revision `466860d`), the duplicate-prompt merge,
    the injective IoU ≥ 0.5 scoring, the 0.95 bar, the marks and the
    deferral of (d) and its combinations to K8 are unchanged.
  - *SAM runs natively.* 240 frames is what K4 ran with no offload (three
    prompts, offline bidirectional; peaks 23.9–25.9 GiB under a 32,000
    MiB declaration on A6000). This path therefore drops video and
    tracker-state offload, the tracker-state adapter, the two in-memory
    replacements of upstream SAM functions, the grounding-batch control
    and the scale forecast (they stay in git history, `6754141`). SAM
    runs at its defaults (grounding batch 16), as in K4; the
    expandable-segment allocator setting stays (it changes storage, not
    results). YOLO11x-pose (1280) runs on every window frame first and is
    released before SAM loads, as before.
  - *Jobs.* One run per part; SAM and YOLO load once per part; each
    window's masks, poses and features are published as soon as it ends,
    so a stop keeps the finished windows. The smoke is the part's first
    window whole (240 frames), so it measures the real peak; the full
    stage runs every window of the part. Pilot: three 240-frame windows,
    one per source format (`djokovic_zverev/004@381`, 2160p60;
    `djokovic_federer/069@305`, 2160p50; `alcaraz_hurkacz/006@387`,
    1080p25). Its windows count in the score (same code and
    specification apart from the selection, as in K4). Promotion: the
    pilot's full output validates locally, its peak is within the
    declaration, and its rate puts each remaining part under 2,700 s.
    Then the other 45 windows in parts sized from that rate, in parallel
    on compatible GPUs.
  - *Validator.* Window placement recomputed from the manifest; every
    window frame masked and posed; SAM pinned, flash attention, default
    configuration (no offload, batch 16); person masks found; features and
    target masks recomputed from the saved masks and poses; frozen rules.
  - *Budget.* About 10,444 frames at the maximum smoke's 0.9 s per frame:
    2.6 GPU-hours of SAM, plus loading and pose. Ceiling 4 GPU-hours and
    6 h wall for this stage, within the 8 GPU-hour phase ceiling with
    what is spent. Shared mode, Ada and A6000 only, 32,000 MiB declared
    with a 4,096 MiB margin; every stage ≤ 45 min.
  - *Pilot* `20261020T184104Z-0af937d9`: gpu6 RTX 6000 Ada, shared
    (`timing_contaminated`), revision `fa7a4fb` from `ready.json`; specs
    and plan in `features-v2/` (`pilot.json` `ae7d56e7…`, `windows.json`
    `dc0e83ed…`), manifest `features-v1/manifest.json` (`4db66511…`). The
    smoke (one whole window) passed the fleet validator; the full stage
    (3 windows, 720 frames) passes all 18 checks locally. SAM native
    (flash attention, batch 16, default configuration); peak 27.3 GiB
    against the 31.25 GiB declaration; SAM 95–107 s per window and pose
    plus JPEGs about 21 s (shared timings, not evidence). Promoted: six
    parts of 7–8 windows (1,564–1,660 frames, about 16 min each at the
    pilot's rate), split longest-first by `tools/datasets/k4b_feature_inputs.py`.
  - *Parts*, all at `fa7a4fb` (`ready.json`). Part 3
    `20261020T185418Z-4dfa86ac` (8 windows, 1,612 frames, peak 27.8 GiB)
    and part 6 `20261020T185438Z-129dc1a3` (7 windows, 1,660 frames, peak
    27.3 GiB) completed on gpu6 RTX 6000 Ada; both full stages pass all
    18 checks locally. Parts 1, 2, 4 and 5 (`…185405Z-f78cf558`,
    `…185411Z-28392fd4`, `…185424Z-4a3e41d9`, `…185430Z-46b3f755`) were
    admitted one after another on gpu2 (A6000) and each ran out of
    physical memory in its smoke: another user's process grew to 34.4
    GiB after admission, leaving about 13 GiB (our allocator cap was 31.25
    GiB). Contention, not evidence. Resubmitted with the same
    specification apart from the host pool (gpu2 left out while that
    process holds 35 GB on both its GPUs): `part{1,2,4,5}-retry.json`,
    jobs `20261020T214345Z-b3c719eb`, `…214351Z-9fe1e191`,
    `…214357Z-3cf9965f`, `…214404Z-c4775c13` at `0477bfb` (code as
    `fa7a4fb`; only this entry changed). All four completed (gpu5 and
    gpu6, RTX 6000 Ada); each full stage passes all 18 checks locally.
    Every one of the 48 windows (10,444 frames) is now covered by a
    locally validated full stage; peaks 27.3–27.8 GiB.
- **Outcome (2026-10-21), amendment 1, the user's 48 marks** (37
  two-player, 10 one-player, 1 no-player frames). Jobs: pilot
  `20261020T184104Z-0af937d9` and parts 3, 6 at `fa7a4fb`; parts 1, 2,
  4, 5 (retries) at `0477bfb`, all from `ready.json`. Report
  `features-v2/score-2026-10-20/k4b-report.json` (`2a9a26fd…`), index
  `index-full.json` (`10e61aa5…`), local validation receipts beside
  them; marks `05f51305…`, review `instances.json` `89b78156…` and masks
  `published.tar` `2dc9acb6…` as recorded above.

  | candidate | correct / 48 | share |
  |---|---:|---:|
  | (a) motion in body heights | 30 | 0.625 |
  | (b) horizontal motion, window's last second discounted | 39 | 0.813 |
  | (c) racket contact on most frames | 32 | 0.667 |
  | (e) keypoint motion | 26 | 0.542 |
  | (a)+(c) | 42 | **0.875** |
  | (d), (c)+(d), (a)+(d) | pending K8 | |

  The window features' proposal oracle is 48/48: on every marked frame
  the window's SAM masks contain each marked player (IoU ≥ 0.5), so the
  misses are the rules' choices, not missing masks.
- Decision (by the rule): **no candidate reaches 0.95**; nothing is
  adopted and K4 is not amended. The table goes to the user. (d) and its
  combinations are pending: K8's v0 court list covers `nadal_kyrgios`
  only, and none of the marked frames come from it, so they wait for
  K8's homographies on the five marked matches.
- Reading (not a test; the rules stay frozen). (a)+(c)'s six misses: on
  five two-player frames it picks one marked player (IoU 0.97–0.98) and
  another person as the second, or on one frame two other people; the
  sixth is the one frame with no player visible, which any rule that picks
  tracks fails. (b) is right on 2 of those 6 and (a)+(c) on 5 of (b)'s
  9 misses (44 of 48 frames have one of the two right). Misses
  concentrate on rublev_auger-aliassime and djokovic_federer. The
  pattern, a second person mistaken for the far player, is what the
  court outline (d) is meant to remove, and the no-player frame is a
  shot K8 should drop. Hypothesis "(d) or (c)+(d) wins" stays open.
- Budget: 7,209 s of stage execution (2.0 GPU-hours: pilot, six parts
  and the four failed smokes, from each stage's `execution.json`), inside
  the 4-hour ceiling; about 4.5 h wall, most of it queueing for a GPU
  with 36 GB free.
- (d) and its combinations, defined before any (d) score is computed
  (2026-10-21; K8b adopted, homographies for the five matches:
  `k8b-2026-10-20/review/k8b-report.json` `8230bf1a…3314` for
  alcaraz_hurkacz, djokovic_zverev and medvedev_struff,
  `k8b-2026-10-20/k4b-matches/registration/k8b-report.json` `eb7b34b9…e3e0`
  for djokovic_federer and rublev_auger-aliassime; model metres, x across
  the court 0–10.97, y along it 0–23.77, to 1080p pixels). Frozen as
  `k4b_court.RULES_D` (version 1), on the same windows, canonical tracks
  and SAM masks as the other candidates, CPU only:
  - A track-frame is *registered* when K8's entry for that clip frame
    (window start + t) has `fits` true (error ≤ 6 px); other frames are
    not read. The *foot point* is the mask's lowest row at its centroid's
    x, at 1080p; it maps through the inverse homography to the court
    plane. *Inside*: x in [−1, 11.97] m and y in [−4, 27.77] m (the doubles
    court, 1 m beyond each sideline and 4 m behind each baseline; ITF's
    minimum run-off is 3.66 m at the sides and 6.40 m at the ends, where
    line judges and ball kids stand). A point behind the camera is outside.
  - Feature d: the share of the track's registered frames that are
    inside (0 with no registered frame); *inside on most*: d > 0.5.
  - (d): eligible tracks (presence ≥ 0.5, as every candidate) inside on
    most, by d, ties by track key (as v1); fewer than two qualify, fewer
    picks, as (c). (c)+(d): eligible tracks by inside on most, then racket
    on most, then d, then the racket share c. (a)+(d): by inside on most,
    then (a).
  - Scored as before (injective IoU ≥ 0.5 on the marked frame; bar 0.95;
    ties to the simpler rule: a, b, c, d, e, then (a)+(c), (c)+(d), (a)+(d)).
    Two tables: all 48 marked frames, and the marked frames K8 registers
    (court views, as this entry's data rule asks once K8 is adopted).
    The court-view table decides; the all-frames table is reported beside
    it.
- Outcome with (d) (`experiments.tennis.k4b_court`, CPU, local; the
  validated features of the seven jobs above; `features-v2/court-score-2026-10-21/`
  `k4b-court-report.json` `d593c879…`). K8 registers 37 of the 48 marked frames as
  court views (36 with two players, 1 with one); the 11 others are the
  no-player frame, five medvedev_struff one-player close-ups and five
  rublev_auger-aliassime shots (003, 006, 054, 062 with one player, 063
  with two).

  | candidate | court views (37) | all frames (48) |
  |---|---:|---:|
  | (a) | 23 (0.622) | 30 (0.625) |
  | (b) | 31 (0.838) | 39 (0.813) |
  | (c) | 25 (0.676) | 32 (0.667) |
  | (d) inside the court | 34 (0.919) | 36 (0.750) |
  | (e) | 19 (0.514) | 26 (0.542) |
  | (a)+(c) | 33 (0.892) | 42 (0.875) |
  | **(c)+(d)** | **36 (0.973)** | 45 (0.938) |
  | (a)+(d) | 35 (0.946) | 42 (0.875) |

  The v1 candidates' all-frames counts reproduce the table above exactly.
- Decision (by the rule): **(c)+(d) is adopted** (0.973 ≥ 0.95 on court
  views; no other candidate reaches the bar). The hypothesis held. Its
  application to K4's windows is amendment 2 (below, not yet run).
- Reading. (c)+(d)'s one court-view miss, rublev_auger-aliassime/002@422:
  only one track's foot point falls inside on most registered frames, so
  the second pick is another person. (d) alone misses it and two windows
  where a marked player stands inside on 34–42% of frames
  (rublev_auger-aliassime/040@363, 048@481); the racket rule recovers those
  two. On all 48 frames (c)+(d) misses three: that frame, the no-player
  frame and rublev_auger-aliassime/063@337, a two-player shot K8 does not
  register, so no track is inside and the racket fills the picks. Both
  are shots K8 sends to the pixel coder, where no identification is
  needed. Scope: 37 court-view frames from five matches; the margins
  (1 m, 4 m) were fixed from ITF's run-off before scoring and not tuned.
- Amendment 2 (to K4, not yet run): K4's broadcast windows need K4b's
  window features (SAM 3.1 with the person prompt; K4 kept no masks) and
  K8b's registration, which covers all 10 of K4's broadcast clips
  (medvedev_struff/042, the outro card, and rublev_auger-aliassime/038
  register on no frame, rublev_auger-aliassime/052 on 32 of 240); K4's
  players, bits and fit are then recomputed with (c)+(d)'s players, with
  amendment 1's and the first definition's numbers kept. TrackNet has no
  court registration (other courts, K8 not run there), so (d) cannot be
  applied to it as defined.
  Folded into K6 (2026-10-21; the user left the order to the session):
  after K4b.2 and K4b.3 every K6 gate condition passes whatever K4's two
  mislabelled windows hold, so a standalone re-run decides nothing; K6
  includes K4's 9 broadcast windows and reports K4's numbers with the
  adopted players there (PLAN K4b, K6).

### 2026-10-21 — K4b.2: pose on the far player (crop-size sweep)
- Question (PLAN K4b, item 2). K4's (b) failed on the far player alone:
  YOLO11x-pose on the whole frame at 1280 px fits players under 10% of
  the frame height on 0–50% of frames and those 15% and taller on
  97–100%. Does a top-down estimate on each player's crop, enlarged, fit
  the far player, and at which crop size?
- Players: identified by hand, not by a rule, so this does not wait for
  K4b.1's decision or K8. K4b.1's 48 windows (240 frames per marked
  frame, 10,444 frames, the five broadcast matches) already carry SAM 3.1
  masks and canonical tracks (`publish/<window>/sam-masks.rle`,
  `features.json`, from the seven locally validated jobs). On each
  window's marked frame, each player the user marked takes the canonical
  track whose mask matches it (IoU ≥ 0.5, injective; 48/48 match, K4b.1's
  proposal oracle), and that track is the player on every frame of the
  window (its person mask where both prompts have one, as K4b.1). The
  no-player frame's window has no players. *Far player*: in a window with
  two marked players, the one with the smaller median mask height; the
  other is the *near player*.
- Settings (YOLO11x-pose, ultralytics 8.4.6, confidence 0.25, as K4).
  *Whole frame*: K4's setting (1280 px), read from K4b.1's published
  detections; no GPU. *Crop ×s*, s = 1, 2, 4: the player's mask box at
  1920×1080, grown by 20% of its height on every side and clipped to the
  frame, resized by s (bicubic; s = 1 untouched), its long side capped at
  1280 px (K4's input; only near players reach it), padded to a multiple
  of 32 and run at that size, so ultralytics does not rescale it.
  Keypoints map back to the frame.
- Measures, per setting and player-frame, as K4 (`k4.pose_analysis`'s
  definitions): *fitted* when a detection has ≥ 0.5 of its box inside the
  mask's box (H1's matching; for a crop, only that crop's detections);
  *inside*: share of confident keypoints (≥ 0.5) inside the mask dilated
  by 8 px. Reported for all player-frames, the far and near players, and
  K4's height bins (6–10%, 10–15%, 15–25%, ≥ 25% of the frame height;
  under 6% beside them). *Synthetic far player* (fit says nothing of
  accuracy): every 8th frame of each near player whose median height is
  ≥ 15% of the frame and whose ×1 crop is fitted, reduced (area filter)
  to the same window's far player's median height, then run through each
  crop setting; OKS (COCO sigmas, the native mask's area) against the
  near player's own ×1 keypoints confident ≥ 0.5; unfitted counts as
  OKS 0. *Latency*: GPU milliseconds per crop at each setting (batch 16,
  synchronized), the whole frame beside it. Raw keypoints are saved (K4b.3
  reads them).
- Decision rule (fixed before any run). A crop setting *passes* when, on
  the 48 windows' hand-identified players: (1) fitted ≥ 0.90 of all
  player-frames and ≥ 0.90 of far-player frames; (2) median inside ≥ 0.9
  for both; (3) ≥ 0.90 of synthetic far players at OKS ≥ 0.5. The
  smallest passing s is chosen (cheapest). If one passes, PLAN K4b.2 is
  done: K4's (b) is re-read with it (here on hand-identified players; on
  K4's windows once K4b.1's identification is adopted), and K4b.3 runs on
  its keypoints. If none passes, stage 2 runs DWPose (`Models/DWPose`,
  `dw-ll_ucoco_384.onnx`, body 17 of COCO-WholeBody) on the same crops
  under the same three conditions, after an environment audit and a
  first-run smoke of its execution provider; then super-resolution of the
  crop; fine-tuning only if all fail.
- Hypothesis: whole frame fits about 0.5 of far-player frames and ≥ 0.97
  of near ones (K4); ×2 lifts the far player to ≥ 0.9 and ×4 adds nothing;
  near players are unchanged; inside stays 1.0; synthetic OKS ≥ 0.5 on
  most samples at ×2 and ×4, fewer at ×1.
- Competing explanations: (1) the far player is missed for occlusion or
  motion blur, not size: fit would not rise with s, and the height bins
  show it. (2) Enlarged crops give confident wrong keypoints: inside and
  the synthetic OKS check it; the reduction does not reproduce broadcast
  compression of a real far player, so OKS is an upper bound there. (3)
  A SAM track drifts to another person within the window: review images
  (the marked frame and one content-blind frame per window, every
  setting's skeleton on the crop) check it. (4) The crop's margin admits
  another person: the matching to the player's own mask decides, as K4.
- Budget: no SAM; one window conversion per window (K4b.1's, 83 s for
  all), YOLO on about 20,000 player crops per setting and the synthetic
  samples. Smoke: the part's first window (≤ 600 s). Pilot: K4b.1's three
  pilot windows, ≤ 45 min. Full: the other 45 windows in two parts of
  ≤ 45 min. Ada or A6000 (YOLO11x-pose's table rows; Ada asserted in
  K4b.1's jobs), shared, 12 GB declared. Ceiling 1 GPU-hour and 4 h wall.
- Jobs (all RTX 6000 Ada, gpu6, shared; environment
  `pointstream-20261009T214002Z`, ultralytics 8.4.6, YOLO11x-pose
  `013c4354…85f8`; inputs `pointstream-data/tennis/k4b-2026-10-16/pose-v1/`
  `inputs.tar` `3ae0106e…5e81`, built by `tools/datasets/k4b_pose_inputs.py`
  from K4b.1's seven validated jobs and the marks `05f51305…`). Pilot
  `20261021T090639Z-c2c32f37` (`1e9dc6b`; 3 windows, 720 frames; 177 s;
  `published.tar` `8bba8170…`); part 1 `20261021T091816Z-6a1e0668` (`c5ac3b5`;
  22 windows, 4,832 frames; 1,326 s; `552f3109…`); part 2
  `20261021T091821Z-1b022f25` (`c5ac3b5`; 23 windows, 4,892 frames; 1,245 s;
  `32e79fdd…`). `c5ac3b5` differs from `1e9dc6b` only in the validator's OKS
  comparison (within 1e-9: an ulp differs between the hosts' and macOS's
  numpy) and the `report` command. Peak 8.75 GiB of 12 GB. Every full output
  passes the validator's 9 checks locally (`pose-v1/report-2026-10-21/local-validation/`):
  each setting's fit and inside, and every synthetic OKS, recomputed from the
  published masks and keypoints. Both parts saw GPU contention from other
  users (action continue), so the latencies are indicative only.
- Outcome (`k4b_pose report`, 48 windows, 19,062 hand-identified
  player-frames: 8,860 far, 8,860 near, 1,342 in one-player windows;
  `k4b2-report.json` `bd87835b…91fb`):

  | setting | fitted, all | fitted, far | under 6% (113) | 6–10% (3,257) | 10–15% (3,609) | ≥ 15% (12,083) | synthetic OKS ≥ 0.5 (1,109) | median OKS |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | whole frame (K4) | 0.812 | 0.595 | 0.009 | 0.036 | 0.916 | 0.997 | | |
  | crop ×1 | **0.997** | **0.993** | 0.584 | 0.995 | 1.000 | 1.000 | 0.991 | 0.967 |
  | crop ×2 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.994 | 0.982 |
  | crop ×4 | 0.996 | 0.999 | 0.991 | 0.997 | 0.997 | 0.995 | 0.994 | 0.974 |

  Median keypoints inside the mask: 1.0 in every setting and group.
  Latency (shared, contended GPU): 21.6 ms per crop averaged over the three
  sizes, 39.6 ms per whole frame (median).
- Decision (by the rule): **crop ×1 passes** (all three sizes pass; the
  smallest is chosen). The far player is fitted: K4's (b) re-read on
  hand-identified players goes from 0.812 (whole frame) to 0.997 with
  median inside 1.0, above the 0.90 bar, so K6's gate condition (b) passes
  here; on K4's own windows it is re-read once K4b.1's identification is
  adopted. DWPose, super-resolution and fine-tuning are not needed.
  K4b.3 (keypoint rate) runs on the ×1 keypoints.
- Reading. The hypothesis held except that ×1 was already enough: the
  failure was the whole-frame input (1920 px shown to YOLO at 1280, so a
  60–100 px far player became 40–70 px), not the player's size in the
  source. ×2 adds the 113 player-frames under 6% of the height (0.58 →
  1.0). Competing explanation (1) does not hold (fit rises from 0.04 to
  0.99 with the crop alone in the 6–10% bin). (2), confident wrong
  keypoints: the synthetic far player agrees with its own native keypoints
  (median OKS 0.97–0.98), and on the far player 99.6% of ×1 detections are
  at least half the mask's height and 96.6% have ≥ 12 confident keypoints
  (whole frame 95.0%). The fit measure itself is loose on large players:
  enlarged crops of near players and of close-up players (one-player
  windows) get partial detections more often (near at ×4: 33% of detections
  under 0.7 of the mask's height, whole frame 8%; one-player windows at ×1
  25% against 0.5%), while K4's matching still counts them fitted; a
  review image (`rublev_auger-aliassime/002@422`, t = 175, ×4) shows a
  small skeleton inside the player's box. So the estimator to carry
  forward is ×1 crops for players under 15% of the height and either
  setting above (whole frame is as good there); K4b.3 and K6 read the
  keypoints, not this fit count. (3), track drift, and (4), others in the
  crop: the review images checked (the pilot's 12 and four from
  rublev_auger-aliassime) show the marked player each time, including with
  an umpire inside the crop. Scope: 48 broadcast windows from five
  matches; the synthetic check reduces near players without broadcast
  compression of a real far player, so its OKS is an upper bound.
- Budget: 2,905 s of stage execution (0.81 GPU-hours with the three
  smokes), inside the 1 GPU-hour ceiling; about 2 h wall from the rule.

### 2026-10-21 — K4b.3: keypoint rate against keypoint error
- Question (PLAN K4b, item 3). What does it cost to send the players'
  keypoints at each precision, and how far do the decoded keypoints move?
  K4 measured one point (9.4 kbps for two players, first differences at
  0.25 px, 22% of SVT-AV1's player bits at CRF 62) and kept no keypoints;
  K6 needs the curve to choose its operating point.
- Data: K4b.2's ×1 crop keypoints (the chosen setting), the 48 windows'
  hand-identified players, from the three locally validated K4b.2 stages
  (19,062 player-frames; 8,860 far, 8,860 near). CPU only, local.
- Codes (per player track, YOLO's 17 COCO keypoints, x and y): quantization
  step q = 1, 2, 4, 8 px at 1080p; prediction *previous* (first
  differences, K4's) or *constant velocity* (2·previous decoded −
  the one before; previous on a track's second frame); keypoint rate
  every frame, every 2nd or every 4th frame, the skipped frames decoded by
  linear interpolation between sent frames (held after the last). Closed
  loop: residuals are taken against decoded values. A frame where the
  player is not fitted ends the track; the next fitted frame starts a new
  one, its values sent absolutely (log2 of 1920/q and 1080/q bits). Per
  keypoint and sent frame, one visibility symbol (confidence ≥ 0.5). Rate:
  empirical entropy of the residual symbols and of the visibility symbols,
  each pooled over all windows per code (adaptive coding's optimistic
  bound, as K4), plus the absolute starts, in kbps for two players at each
  window's frame rate.
- Error, against the uncoded keypoints, over every fitted frame and the
  keypoints confident there: displacement in px at 1080p (median, p95) and
  in shares of the player's box height; OKS against the uncoded keypoints
  (COCO sigmas, the mask's area), for far and near players.
- Decision rule (fixed before computing). The code handed to K6 is the
  lowest-rate code whose median OKS against the uncoded keypoints is
  ≥ 0.95 for both far and near players (the coding error then sits well
  inside COCO's tolerance at the player's own scale); its rate is reported
  against K4's SVT-AV1 player bits on broadcast (41.9 kbps at CRF 62,
  155 kbps at CRF 48; K4's windows, not these, so a reference, not a
  test). The whole curve goes to K6, which judges the error on the
  rendering. K4's gate (c) is not re-read here: its bits come from other
  windows.
- Hypothesis: q = 2 px is free in error (YOLO's keypoints come at 1–2 px
  precision, K4) and costs a third less than 1 px; constant velocity saves
  little (K4: 27–67% of differences already zero) and loses on jitter;
  every 2nd frame passes at ≥ 50 fps; the chosen code costs 3–6 kbps for
  two players.
- Competing explanation: YOLO's frame-to-frame jitter, not motion, sets
  both the rate and the error floor of interpolation; the error at q = 1
  every frame against the uncoded keypoints, and the share of zero
  residuals, show it.
- Budget: CPU minutes, no fleet.
- Outcome (`experiments.tennis.k4b_rate`, local, 13 s; the three
  K4b.2 stages above, each locally validated; 24 codes;
  `pose-v1/rate-2026-10-21/k4b3-report.json` `89276994…`). kbps for two
  players (1.82 players per frame on average); OKS median and 5th
  percentile against the uncoded keypoints; displacement in px at 1080p:

  | code | kbps | far OKS (p05) | near OKS (p05) | px median / p95 |
  |---|---:|---:|---:|---:|
  | q 1, previous, every frame (K4's kind) | 15.0 | 0.998 (0.995) | 1.000 (0.999) | 0.40 / 0.60 |
  | q 2, velocity, every frame | 11.7 | 0.991 | 0.999 | 0.80 / 1.20 |
  | q 4, velocity, every frame | 9.5 | 0.963 (0.925) | 0.994 (0.986) | 1.59 / 2.40 |
  | q 2, velocity, every 2nd | 6.6 | 0.989 (0.913) | 0.998 (0.940) | 0.90 / 4.39 |
  | **q 4, velocity, every 4th** | **3.3** | 0.958 (0.813) | 0.988 (0.858) | 1.79 / 8.07 |
  | q 8, velocity, every 4th | 2.6 | 0.891 | 0.974 | 3.03 / 8.56 |

  18 of the 24 codes pass the bar; every q = 8 code fails on the far
  player. Constant velocity saves 2–8% over first differences at equal
  error. Absolute starts are 1–5% of the bits.
- Decision (by the rule): **q = 4 px, constant velocity, every 4th frame,
  3.3 kbps for two players** goes to K6 as the cheapest code within the
  bar, with the whole curve. Against K4's SVT-AV1 player bits on broadcast
  (other windows, a reference) it is 7.8% at CRF 62 and 2.1% at CRF 48;
  K4's 22% came from 9.4 kbps.
- Reading. The bar holds the median, not the tail: skipping frames
  costs the fast limbs. At every 4th frame the p95 error is 8 px (near
  player 9.5 px; players in one-player windows, mostly close-ups, 46 px),
  against 2.4 px for the same quantization every frame (9.5 kbps). So the
  frame rate is the lever whose error K6 must judge on the rendering:
  swinging arms are what a viewer watches. Hypothesis: q = 2 is not free
  but cheap in error (0.8 px median, 22% fewer bits than q = 1); velocity
  saves little, as expected; every 2nd frame passes; the chosen code is at
  the low end of 3–6 kbps. Competing explanation: jitter sets the floor
  at q = 1 (18% zero residuals, 4.2 bits per value; the crops give YOLO
  finer keypoints than K4's whole frames, 27–67% zeros there), so
  keypoints at full precision cost more here (15.0 kbps) than K4's 9.4.
  Scope: YOLO11x-pose's 17 keypoints on ×1 crops of hand-identified
  players in 48 broadcast windows; the rate is a pooled-entropy bound
  without a real entropy coder.

### 2026-10-21 — K6: player oracle (H3b for bodies, on broadcast tennis)
- Question (PLAN K6). Would a player sent as keypoints plus an appearance
  model beat the pixel coders on the players? The total (keypoints, with
  whatever the pose cannot describe charged) against the achievable
  envelope of SVT-AV1 and DCVC-UF swept over the coding resolution, and
  the keypoint frame rate judged on the rendering (K4b.3: every 4th frame
  has a p95 error of 8 px on fast limbs). Also K4b.1's amendment 2: K4's
  numbers with the adopted players.
- Pipeline (the adopted one, nothing re-chosen here).
  1. Court views: K8b's per-frame homographies
     (`k8b-2026-10-20/review/k8b-report.json` `8230bf1a…3314`,
     `k8b-2026-10-20/k4b-matches/registration/k8b-report.json`
     `eb7b34b9…e3e0`); a frame is *scored* only when its entry has
     `fits` true.
  2. Players: K4b.1's (c)+(d) (`k4b_court.choose_d`, `RULES_D` v1) on
     K4b.1's window features (SAM 3.1 person/player/racket, canonical
     tracks).
  3. Pose: YOLO11x-pose as K4b.2. A player whose median mask height is
     under 15% of the frame takes ×1 crops (K4b.2's path at s = 1); a
     taller one takes the whole-frame detections K4b.1 published (K4's
     setting), since enlarged crops give partial detections there
     (K4b.2's reading). Fitted by K4's matching.
  4. Keypoints: K4b.3's `code` (closed loop, linear interpolation of
     skipped frames). Codes: q 4 px constant velocity every frame, every
     2nd and every 4th (K4b.3's choice), and q 1 px previous every frame
     (K4's kind). Tracks run over fitted scored frames; a gap starts a
     track. Rate: each symbol's −log2 of its pooled frequency over all of
     K6's windows (K4b.3's optimistic adaptive bound), plus the absolute
     starts, over the window's scored seconds.
- Windows. K4's 9 broadcast windows (the outro card left out, as K4) and
  K4b.1's 48. K4's windows get K4b.1's features by K4b.1's code unchanged
  (`k4b_features run`), through a manifest that lists each K4 clip with
  frame 120 as its only "review frame", which `window_span` places at
  frames 0–239, K4's window (no labels involved). The oracle scores the
  45 windows with ≥ 24 scored frames: 8 of K4's (`rublev_auger-aliassime/038`
  registers on no frame, `/052` on 32) and 37 of K4b.1's (8,649 scored
  frames). Overlapping windows of one clip are correlated; the bootstrap
  is by window and the paper counts matches (5).
- Region. R(t) at 1080p: the two players' SAM masks (person mask first,
  as K4b.1) and every racket instance touching one (K4b.1's (c) contact:
  1 px dilation at 960×540), upsampled. Broadcast footage has no dataset
  masks, so R comes from SAM 3.1, as in K4 (scoped: the judge and the
  oracle read the same masks).
- Oracle variants (1080p composites; per code k). Each is H3b's
  construction with MANO replaced by a 2D skeleton warp. Bones: K4's
  `SKELETON` pairs with both ends confident (≥ 0.5) in the keypoints the
  pixels come from and in the target frame's. Each bone maps the target's
  coded segment onto the source's segment by a 2D similarity; an output
  pixel's source position is the blend of the bones' maps with weights
  exp(−d²/2σ²), d its distance to the coded bone, σ = 0.08 of the
  player's mask height (fixed, not tuned). No qualifying bone: a
  translation by the mean displacement of confident keypoints; no
  confident keypoint: nothing drawn. A pixel is drawn when its source
  position falls inside the source player's region (nearest); its colour
  is the source sampled bilinearly. The far player is drawn first. Inside
  R the source is first replaced by an OpenCV Telea fill (radius 5; H3b's
  fill); drawn pixels go on top, also outside R (overshoot is charged).
  - `warp@k` (the bound): the frame's own pixels, moved from the uncoded
    to the coded keypoints. Perfect appearance at the coded pose, with the
    outline, cloth and racket granted: only the coding error is charged.
    It dominates every renderer that places the player at the coded pose,
    because no renderer can draw more of the true frame.
  - `ref@k` (what the pose cannot describe charged): the pixels of a
    reference frame of the same player moved to the coded keypoints. The
    reference, per player and frame, is the player's fitted scored frame
    at least 1 s away whose uncoded keypoints are nearest to the coded ones
    (mean distance over jointly confident keypoints, each set centred on
    its confident mean and divided by its mask height; ties by the nearer
    frame). Decoder-side information only (the coded keypoints and a bank
    sent once); the bank is granted, as a per-player appearance model
    amortized over a match (H3d's condition). Charged: outline, cloth,
    hair and the racket, which moves with the forearm as a fixed grip
    (re-tests H4a's grip pose).
  - `ref@uncoded`: diagnostic, coding against appearance error.
- Anchors ("swept over the rendering resolution": the coding resolution,
  as H3b2). SVT-AV1 4.2.0 preset 4 (K4's settings) at CRF 48 and 62 at
  1080p, CRF 63 at ½, CRF 55 and 63 at ¼, CRF 63 at ⅛; DCVC-UF HT-L off the
  shelf (B2's worker and checkpoints) at QP 0 at 1, ½ and ¼ and QP 21 at
  ½. Downscaled per plane (area), decoded, upscaled bicubically to 1080p
  and composited inside R (source outside), as H3b2 and H3d. Rate: the
  bits on R over the scored frames (SVT-AV1: K4's block-spread accounting
  through `aom inspect`; DCVC-UF: H3c's own-entropy split onto R, R empty
  on unscored frames) per scored second.
- Judges, on scored frames with R non-empty, over S(t) = R(t) plus every
  oracle variant's drawn pixels at t: LPIPS (B2's AlexNet spatial map on
  the full frame, averaged over S) **decides**; masked DISTS (H3b's, crop
  of S plus 64 px) and PSNR over S are reported. Mean over the window's
  scored frames.
- Decision rule (`k6.DECISION`, fixed before any run). Per window, the
  anchor envelope is the upper concave hull (`g5.envelope`) of the ten
  anchor points on (kbps on R, −LPIPS). The rate it needs for a variant's
  LPIPS is linear along the hull (time-sharing between encodes); better
  than every anchor point: the best point's rate (the saving is
  understated); worse than every point: the saving is min(0, 1 − r / r₀)
  with r₀ the lowest anchor rate (unmeasured counts as no saving).
  Saving = 1 − r_keypoints / r_envelope. Mean over windows, 95% bootstrap
  CI by window (1,000 resamples, seed "pointstream-k6").
  1. Bound: `warp` at K4b.3's code (q 4, velocity, every 4th). It passes
     if the CI's lower end is above 50% (H3b2's bar: perfect appearance
     must still save half). If it fails there but passes at every 2nd
     frame or every frame, the operating point moves to the lowest-rate
     passing frame rate (the frame rate judged on the rendering). If it
     fails at every code, no player mode is built: players stay pixels.
  2. Charged: `ref` at the operating point. It passes if the CI's lower
     end is above 25% (K5's margin: rendered appearance, unbilled bank).
     Then a player renderer is built (K6b, its own entry; retrieval and
     warp first, since that is what `ref` measures). If 1 passes and 2
     fails, the outcome goes to the user with the diagnosis (coding
     against appearance from `ref@uncoded`, the outline shares).
  If DISTS's mean saving disagrees in sign with LPIPS's on a deciding
  variant, the outcome goes to the user.
  Stopping (sequential looks): the decision is read after K4's windows
  and after each later part; it is taken once every gate it reads has its
  CI wholly above or wholly below its bar on ≥ 8 windows from ≥ 4
  matches. The looks taken are recorded.
- Reported beside. The outline: share of R that `ref`'s drawing misses
  and share of its drawing outside R, by player height (under 10%,
  10–15%, 15–25%, ≥ 25%), against H3b's 9.7% and 9.2% on hands. The
  racket: LPIPS on racket pixels for `warp` and `ref`. Fast limbs: `warp`
  at every 4th against every frame on the scored frames in the top decile
  of wrist speed. BD-rate of each variant's curve over the codes against
  the envelope where the LPIPS ranges overlap, else better/worse at every
  rate.
- Amendment 2 (K4's 9 windows, all 240 frames, as K4 measured them),
  with (c)+(d)'s players: (a) the players' share of the foreground's
  SVT-AV1 bits at CRF 48 and 62 (players and rackets from K6's streams;
  the ball's bits from K4's record of the window, ≤ 1% of the
  foreground); (b) the fitted share (step 3's estimator, and whole frame
  alone as K4); (c) the keypoint rate at K4b.3's code and at q 1 previous
  against the players' bits at CRF 48 and 62. Beside K4's first
  definition and amendment 1 (K4's report), and K4's whole-frame kbps
  against K6's (same settings, same conversion).
- Hypothesis. `warp` passes the bound at every 4th frame: a 2 px median
  displacement costs little LPIPS, while the pixel coders reach 3 kbps on
  the players only at ¼–⅛ resolution. `ref` fails: cloth, racket and
  outline differ between frames, and LPIPS charges a wrong outline as in
  H3b. The outline error shrinks with the player's size in pixels but stays
  above 5%.
- Competing explanations. (1) LPIPS at 1080p charges misplacements a
  viewer would not see on a 60–100 px player; DISTS is reported. (2) The
  Telea fill is a poor background; LPIPS on the drawn pixels alone is
  reported. (3) Block spreading (64 px blocks, 16 px DCVC-UF cells)
  misplaces bits around a small far player; whole-frame rates are
  reported. (4) The bank from the window itself is optimistic (same light,
  same rally); a bank from elsewhere in the match would do worse, so `ref`
  is still an upper bound of a retrieval renderer.
- Budget. Features for K4's 9 windows: one job, smoke ≤ 600 s (one whole
  window), full ≤ 45 min. Oracle: about 10 min per window (SVT-AV1 on the
  CPUs beside DCVC-UF and its bit estimate, then rendering and judges);
  smoke the part's first window, first 48 frames (≤ 600 s); pilot 2 of
  K4's windows; parts of ≤ 4 windows (≤ 45 min). Shared Ada/A6000,
  memory from the smoke. Ceiling 8 GPU-hours and 16 h wall, smokes
  included.
- Features for K4's windows: job `20261021T112104Z-ede5db59` (gpu6, RTX
  6000 Ada, shared; `3d53e9b` from `ready.json`; manifest
  `k6-2026-10-21/features/manifest.json` `a696d2ba…9726`). Its full
  stage (9 windows, 2,160 frames) passes all 18 of `k4b_features
  validate`'s checks locally, the manifest's identity included. Oracle
  inputs: `k6-2026-10-21/oracle/inputs.tar` `20e595d5…e703`
  (`tools/datasets/k6_inputs.py oracle`; the seven K4b.1 feature jobs and
  this one, each identity checked against its index).
- Pilot attempts (infrastructure, no score read). `20261021T155104Z-1858221f`
  (`1f8ddc8`): the DCVC-UF worker refused to run, since ultralytics had
  rewritten CUDA_VISIBLE_DEVICES (K5's known issue; fixed in `d57efa2`).
  `20261021T155739Z-654cf647` (`d57efa2`): the staged `aom-inspect` had
  no execute bit (K4's; fixed in `5c83149`). `20261021T160249Z-c714d1f3`
  (`5c83149`, gpu3 RTX A6000 shared with another user's training): the
  smoke passed all 18 validator checks; the full stage finished
  `djokovic_federer/038@120` in 1,479 s, then stopped when DCVC-UF QP 0
  at ½ on `/072` decoded non-deterministically on 3 of 3 attempts (6–7
  dB; the fleet's known DCVC-UF failure under contention). Its scores
  were not opened.
- **Amendment 1 (2026-10-21, before any score was read).** Three changes,
  all for cost and robustness; the rule's gates, bars, variants and
  anchors are unchanged.
  1. *Judged frames.* A window took 1,479 s, so 45 windows would need
     about 19 GPU-hours against the 8-hour ceiling. The judges now run on
     every 3rd scored frame (t mod 3 = 0). Three is coprime with the
     codes' 2- and 4-frame sending phases, so sent and interpolated frames
     are judged in their true proportions. Rates still count every scored
     frame.
  2. *Warp field.* The skeleton warp's source positions are computed on a
     4 px grid and the displacement upsampled bilinearly. The field is
     smooth, and the result is exact for the identity and any translation
     (tested).
  3. *A DCVC-UF point that fails K5's guard on all 3 attempts* is
     recorded (its attempts kept) and left out of that window's envelope,
     instead of stopping the part. That can only weaken the envelope, in
     the keypoints' favour; the count of left-out points is reported.
  Each finished window is now checkpointed (summary and archive), so a
  stop keeps it.
- Jobs (all at `22992f7`, from each `ready.json`; environment
  `pointstream-20261009T214002Z`; inputs `20e595d5…e703`; shared GPUs,
  timings not evidence). Pilot `20261021T164820Z-f32fa989` (gpu3, RTX
  A6000; `djokovic_federer/038`, `/072`); K4 part 1
  `20261021T171929Z-fcf1e66d` and part 2 `20261021T171936Z-50ce2f74`
  (gpu6, RTX 6000 Ada); part 3 `20261021T171943Z-b9d3ff04` (gpu3, RTX
  A6000). Every smoke passed the fleet validator, and every full stage
  passes all 20 of `k6 validate`'s checks locally (players and keypoint
  codes recomputed, the warp at the uncoded keypoints reproducing the
  source exactly, anchors' LPIPS rising as they coarsen). No DCVC-UF
  point failed, so amendment 1's exclusion was never used. Peak 15.0 GiB.
  K6's whole-frame SVT-AV1 rates at CRF 48 and 62 equal K4's on all 9
  windows. Report: `k6-2026-10-21/oracle/report-2026-10-21/k6-report.json`
  (`22cb71c7…e0ad`, `k6 report` at this entry's commit), with the
  local validation receipts beside it.
- Outcome (K4's 9 windows; 8 scored, from all 5 matches; 1,710 scored
  frames, 565 judged). The stopping rule is met at the first look (8
  windows, 5 matches, every gate's CI on one side of its bar), so K4b.1's
  37 windows were not run. Means over the 8 windows; kbps on R (both
  players and their rackets) over scored seconds; LPIPS, DISTS and PSNR
  over S:

  | point | kbps | LPIPS | DISTS | PSNR dB |
  |---|---:|---:|---:|---:|
  | SVT-AV1 CRF 48 | 174.3 | 0.039 | 0.072 | 32.0 |
  | SVT-AV1 CRF 62 | 42.0 | 0.130 | 0.141 | 26.0 |
  | SVT-AV1 CRF 63, ½ | 9.3 | 0.277 | 0.222 | 21.7 |
  | SVT-AV1 CRF 63, ¼ | 2.5 | 0.384 | 0.282 | 19.3 |
  | SVT-AV1 CRF 63, ⅛ | 0.6 | 0.456 | 0.339 | 17.6 |
  | DCVC-UF QP 21, ½ | 5.6 | 0.253 | 0.212 | 23.3 |
  | DCVC-UF QP 0 | 4.3 | 0.294 | 0.239 | 22.0 |
  | DCVC-UF QP 0, ½ | 1.4 | 0.379 | 0.293 | 19.7 |
  | DCVC-UF QP 0, ¼ | 0.3 | 0.452 | 0.351 | 17.5 |
  | `warp`, q 1 previous, every frame | 13.8 | 0.011 | 0.021 | 34.3 |
  | `warp`, q 4 velocity, every frame | 8.5 | 0.037 | 0.045 | 24.3 |
  | `warp`, every 2nd | 5.3 | 0.049 | 0.053 | 22.9 |
  | **`warp`, every 4th (K4b.3's code)** | **3.6** | **0.075** | 0.068 | 20.8 |
  | **`ref`, every 4th** | **3.6** | **0.461** | 0.247 | 12.5 |
  | `ref`, uncoded keypoints | – | 0.460 | 0.247 | 12.6 |

  (SVT-AV1 CRF 55 at ¼, 11.8 kbps, 0.300, is in the report.)
  - Bound: `warp` at every 4th frame saves 0.950 of the envelope's rate
    (95% CI 0.914 to 0.975; every window above 0.83). Every 2nd frame
    0.956 [0.946, 0.970], every frame 0.933 [0.917, 0.953]. **Passes at
    K4b.3's code, which stays the operating point.** DISTS agrees (0.97).
  - Charged: `ref` at that code saves −6.58 (CI −9.91 to −3.63). It is
    worse than every anchor point, SVT-AV1 at ⅛ included, on 4 of 8
    windows, and worse than the envelope at its own rate on the other 4.
    **Fails.** DISTS agrees (−1.02 [−2.08, −0.06]).
- Decision (by the rule): **the bound passes and the charged oracle
  fails, so the outcome goes to the user.** No player renderer is built
  on this evidence.
  - The keypoint rate is not the problem. Perfect appearance at the coded
    pose costs 3.6 kbps for two players at LPIPS 0.075. SVT-AV1 spends
    42 kbps for 0.130 at CRF 62, and DCVC-UF at 4.3 kbps reaches 0.294.
  - What the pose cannot describe is the problem. `ref` is no better at
    the uncoded keypoints (0.460 against 0.461), so the coding error does
    not matter there. All of the loss comes from drawing a different
    frame's appearance at this pose.
- Reported beside.
  - Outline. `ref` misses 16–26% of R and draws 32–46% of its pixels
    outside R in every height bin (under 10%: 16% / 32%; ≥ 25%: 18% /
    45%). `warp` misses 3–9% and overshoots 3–13%. On hands, H3b's MANO
    render missed 9.7% and overshot 9.2%. A body warped from another
    frame's pose has a worse outline than MANO had on hands, and it does
    not improve with the player's size in pixels.
  - Racket: LPIPS on racket pixels is 0.474 for `ref` (the fixed grip
    moving with the forearm), 0.105 for `warp`, and 0.219 for SVT-AV1 at
    CRF 62. The fixed grip fails, as H4a's grip pose did.
  - Fast limbs: every 4th frame against every frame adds 0.074 LPIPS on
    the scored frames in the top decile of wrist speed, against 0.026 on
    the others. The 8 px p95 of K4b.3 shows on swinging arms. It does not
    move the bound: every 4th frame still saves 95%.
  - The fill: LPIPS on `ref`'s drawn pixels alone (0.474) matches its
    LPIPS over S. The loss is in the drawn player, not in the Telea
    background (competing explanation 2 does not hold).
- Amendment 2 (K4's 9 windows, all 240 frames, (c)+(d)'s players; means
  over windows, as K4):

  | | first definition (K4) | amendment 1 (K4) | (c)+(d) (K6) |
  |---|---:|---:|---:|
  | (a) players' share of the foreground's bits, CRF 48 / 62 | 92–94% | 86.0 / 88.6% | 90.8 / 92.7% |
  | (b) player-frames fitted | 0.8999 (YOLO11x, whole) | 0.875 (whole) | **0.989** (×1 crop under 15%); 0.878 whole frame alone |
  | (c) keypoints / players' SVT-AV1 bits, CRF 62 | – | 22.4% (9.4 kbps) | **10.3%** (K4b.3's code); 46.1% (q 1 previous) |
  | the same at CRF 48 | – | 6.1% | 2.4%; 10.7% |

  (a) and (b) pass K4's screen with the adopted players and estimator,
  and (c) misses H1's 10% bar by 0.3 points at CRF 62. K6's own
  comparison supersedes that screen. `djokovic_federer/038`, where K4's
  motion rule missed the standing server, now has both players fitted on
  every frame, and so does `djokovic_zverev/012`. On the two windows
  K8b does not register, the rule falls back to racket contact
  (`rublev_auger-aliassime/038`, a close-up, and most of `/052`). The
  ball's bits come from K4's record of each window, whose streams K6's
  reproduce.
- Reading.
  - The hypothesis held: `warp` passes and `ref` fails.
  - "The outline error shrinks with size" is refuted for a reference warp:
    it stays at 16–26% missed at every height.
  - K6 answers PLAN's question on rate and opens one on appearance.
    Keypoints carry the players' pose for a tenth of what pixels spend at
    a far better quality, if the decoder can draw the true appearance.
    Drawing it from another frame of the same player through a 2D
    skeleton warp is worse than any pixel coder, even at an eighth of the
    resolution.
  - What to change next (the oracle protocol asks for it): a renderer
    that is not a single-reference 2D warp. Candidates: a pose-conditioned
    generator learned per player over a match, or keypoints plus a coded
    residual (the hybrid H3b had). Each needs its own oracle, and
    choosing between them is the user's call.
- Scope.
  - K4's 9 broadcast windows (8 scored, 5 matches, court views only).
  - Masks from SAM 3.1, not dataset labels (none exist for broadcast).
  - LPIPS decides, and DISTS agrees on every gate.
  - The keypoint rate is K4b.3's pooled-entropy bound.
  - `ref`'s reference bank is the window's own frames, which is
    optimistic, so retrieval and a 2D warp are ruled out, not learned
    generators.
  - Appearance is granted unbilled in both variants.
- Budget: 8,625 s of stage execution (2.4 GPU-hours, with the features
  job, the three failed pilot attempts and the smokes), inside the
  8-hour ceiling; about 5 h of wall time from the rule, including an
  outage of the session.

### 2026-10-21 — K1c: K6's bound on egocentric hands (outline granted)
- Question ([PLAN K1c](../PLAN.md)). H3b's hand bound drew MANO's mesh at
  the coded pose and charged MANO's outline (it misses 9.7% of the hand and
  overshoots 9.2%). K6's `warp` grants the outline: the frame's own pixels
  moved from the uncoded to the coded keypoints by a 2D skeleton warp, so
  only the coding error is charged. On bodies that was the difference
  between a loss and a 95% saving. Does the same bound beat the pixel
  coders on hands, at H2's coded keypoint rate, against H3d's envelope?
- Items. H3b's pilot: B2's four windows (`P01_107_0000003049`,
  `P02_02_0000006080`, `P03_10_0000000616`, `P09_106_0000006303`), 240
  frames each, with H3c2's staged inputs unchanged (window decode, fill
  masks and record, H2's VISOR poses and `code.json`, MANO, LPIPS and
  DISTS backbones; `h3c-2026-10-11/inputs.json` `da49a912…7a10`).
- Keypoints. H2's WiLoR rows, each track coded with H2's 100 ms choice
  (Gaussian 33 ms, 15 Hz interpolated, full pose, steps ×8; 1.93 kbps
  pooled, 1.927 in K1) and its median shape, decoded, and the 21 joints
  projected (`h3b.item_poses`, `h2.pinhole`, as `h3c.item_judge`). Source
  keypoints: the same row's uncoded fit (its own shape) projected the same
  way, so the warp at the uncoded keypoints is the identity.
- Warp. `k6.draw` with a hand skeleton instead of K4's body: the 20 bones
  of the 21-joint OpenPose/MANO hand (wrist to each finger's base, then
  along each finger), every joint confident. Per fitted hand (one H2 row
  per side and frame), the region moved is that side's fill-set class mask
  (hand and forearm, as R), σ = 0.08 of that mask's height (K6's constant,
  not tuned), the field on K6's 4 px grid. Hands are drawn farther first
  (mean coded joint depth). As H3b, drawn pixels never cover objects (the
  fill set's other foreground), and inside R the source is first replaced
  by K6's Telea fill (radius 5), so a hand with no fit in that frame, and
  every pixel of R the warp misses, is charged as fill. Variants:
  - `warp` (the bound): the uncoded to the coded keypoints.
  - `warp@uncoded` (diagnostic): the identity on fitted hands; what remains
    is the charge for unfitted hands.
- Judges. LPIPS on R exactly as H3c and H3d scored their points (`h3c.Scorer`:
  the variant inside R, the source outside; B2's AlexNet spatial map on the
  full frame averaged over R; mean over the window's frames with R
  non-empty), so the bound and the envelope share region, frames and
  network. **Decides.** Beside: DISTS on R (H3c's), PSNR on R, LPIPS over
  S = R ∪ every drawn pixel with the drawing kept outside R (overshoot
  charged, as K6), the share of R the warp misses and the share of its
  drawing outside R.
- Envelope. Per item, the upper concave hull (`g5.envelope`, linear in rate)
  of SVT-AV1's eleven points and DCVC-UF off the shelf (H3d's bar) together
  with DCVC-UF + G's six points (H3d's component, arm `G-L`), all on R, from
  H3d's component report (`h3d-2026-10-12/report-component/h3d-report.json`,
  `467be69f…6e00`); K1's `joint` hull. At 1.927 kbps it reads LPIPS 0.258
  (P01_107), 0.204 (P02_02), 0.155 (P03_10) and 0.184 (P09_106).
- Decision rule (`k1c.DECISION`, fixed before any run). Per item, the bound
  passes when its LPIPS on R is below the hull's at 1.927 kbps.
  - If it passes on ≥ 3 of 4 items, hands get K6b's ladder in PLAN
    (audit, zero-shot, overfit, match fit), as bodies do.
  - Otherwise hands are closed for every renderer at the keypoint stream's
    rate: no renderer can draw more of the true frame at the coded pose.
  - If LPIPS over S (overshoot charged) moves the count of passing items
    across 3, or DISTS on R disagrees with LPIPS on that count, the outcome
    goes to the user.
  - The hull's rate for the bound's quality (and the saving) is reported
    per item, as K1.
- Hypothesis. The bound passes on ≥ 3 items. H3b's `wilor:arm_uncoded` (MANO
  at the uncoded pose) was 0.168 on S and already above the hull on 2
  items; granting the outline removes MANO's 10% miss and spill, and the
  coding error (about 10 px on WiLoR's joints) costs less than that.
- Competing explanation. Hands are not bodies: fingers are thin and close,
  a 10 px joint error is a large share of a finger, and the warp blends
  neighbouring fingers' bones, so the coding error alone may cost more
  LPIPS than a ¼-scale DCVC-UF + G. `warp@uncoded` separates the coding
  error from the unfitted-hand charge; the missed share separates both
  from the outline.
- Budget. One fleet job (`k1c run`), shared Ada or A6000 (LPIPS and DISTS
  verified there): smoke ≤ 600 s (P01_107, first 48 frames), full ≤ 45 min
  (the four items; the warp on the CPU, two scorings per item). Ceiling
  1 GPU-hour, smoke included. The comparison (`k1c report`) is local, from
  the validated stage and H3d's report.
- Job `20261021T184040Z-70821cd0` (gpu5, RTX 6000 Ada, shared; `ab8c205`
  from `ready.json`; environment `pointstream-20261009T214002Z`; inputs
  H3c2's, `da49a912…7a10`). Smoke passed the fleet validator. The full
  stage (4 items, 240 frames) took 1,182 s and passes all 47 of
  `k1c validate`'s checks locally. Among them: the warp at the uncoded
  joints reproduces the source on every fitted hand, LPIPS and DISTS of the
  source against itself are 0, and H2's units hash-check. Rate: 1.927 kbps
  (WiLoR, `gauss:33.3|15Hz-linear|full|q8|previous`). Report
  `visor/k1c-2026-10-21/report/k1c-report.json` (`085bb948…ef90`, `k1c
  report` at `ab8c205` on `full/published.tar` `3207020e…898d`), with the
  local validation beside it.
- Outcome. On R, at 1.927 kbps:

  | item | `warp` LPIPS | hull LPIPS | hull's rate for `warp`'s LPIPS (saving) | LPIPS over S | `warp` DISTS / hull | `warp@uncoded` LPIPS | R unfitted | fitted R missed | spill |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | P01_107 | **0.087** | 0.258 | 45.9 kbps (96%) | 0.089 | 0.071 / 0.223 | 0.004 | 0.4% | 3.0% | 2.4% |
  | P02_02 | 0.272 | 0.204 | 0.96 kbps (−101%) | 0.276 | 0.191 / 0.251 | 0.192 | 39.6% | 8.0% | 3.0% |
  | P03_10 | **0.093** | 0.155 | 8.6 kbps (78%) | 0.104 | 0.104 / 0.203 | 0.010 | 1.4% | 5.2% | 3.5% |
  | P09_106 | 0.211 | 0.184 | 1.06 kbps (−82%) | 0.219 | 0.218 / 0.225 | 0.138 | 26.8% | 7.3% | 6.1% |

  (Hull: H3d's bar and DCVC-UF + G, K1's joint hull, read at the
  keypoint rate. R unfitted: the share of R in a hand that has no H2 fit
  in that frame, drawn as fill. Missed: the share of a fitted hand's mask
  the warp does not reach. Spill: the share of the drawing outside R.
  The coded joints move 7–12 px from the uncoded ones.)
  - LPIPS on R: the bound is below the hull on **2 of 4** items, so by
    its own gate this fails the 3-of-4 bar. LPIPS over S gives the same 2.
  - DISTS on R: the bound is below the DISTS hull on **4 of 4**. That
    moves the count across 3, so **by the rule the outcome goes to the
    user.**
- Diagnosis. The two losing items are the two where H2 left much of R
  unfitted (40% and 27% of R, against 0.4% and 1.4%). Those pixels are
  charged as Telea fill. `warp@uncoded` isolates that charge: it is 0.192
  and 0.138 there, already near the hull (0.204, 0.184), and the coding
  error (11–12 px) adds the rest. Where the hands are fitted, the outline
  granted turns H3b's loss into a 78–96% saving. H3b's MANO bound
  (`wilor:arm`, 0.17–0.31 on R through K1's bridge) lost on 3 of 4. So
  what decides hands at this rate is the estimator's coverage (a hand
  with no fit has no keypoints) and the coding error. The renderer's
  outline does not decide it.
- Hypothesis: refuted as stated (2 of 4, not ≥ 3), held where the hands
  are fitted. The competing explanation (finger blending makes the coding
  error dominate) does not hold on the fitted items: misses are 3–8%.
- Proposed reading, for the user. (a) By LPIPS's letter, hands are closed
  for every renderer at the keypoint stream's rate as H2 codes it today.
  (b) Or hands get K6b's ladder scoped to fitted hands, after a coverage
  fix: H2b's hybrid estimator, or hands without a fit sent as pixels (a
  hybrid). That would need its own bound first, charging the pixel
  coder's rate on unfitted hands.
- Scope. 4 pilot items, 240 frames. Appearance and outline granted, and
  no reference billed. The hull is H3d's recorded points. LPIPS decides,
  DISTS is reported, and they disagree on 2 items.
- Budget: 1,182 s of stage execution plus the smoke (under 0.5 GPU-hour),
  inside the 1-hour ceiling.

### 2026-10-21 — K1d: K1c's bound with every masked hand fitted (coverage fix)
- Question (user, 2026-10-21, after K1c: "if we still have a chance to win
  on hands, let's try it"). K1c lost on P02_02 and P09_106 because H2 has
  no fit for 40% and 27% of R. On P02_02, 53 hand frames have no fit and
  131 have one. On P09_106 it is 66 and 114. Whole hands were never boxed
  by H1's detector, so H2b's hybrid (same boxes) cannot help. The encoder
  has the hand masks, though. If every masked hand is fitted, does the
  bound pass?
- Change against K1c (nothing else changes: items, warp, judges, hull,
  bars). For every window frame and side whose fill-set class mask is
  non-empty (box sides ≥ 16 px) and that has no H2 WiLoR row, WiLoR (H2's
  `Regressor`, same checkpoint) is run on that mask's bounding box, with
  the side from the mask's class. The fit becomes a row as in `h2.run_visor`
  (rotation, shape, full-frame translation at H2's focal). It joins H2's
  rows, and the item's WiLoR tracks are coded again with H2's 100 ms choice
  (`CodingSet`, so a new row joins its side's track). The mask box includes
  the forearm, so these fits can be worse than H1-boxed ones. That cost is
  charged through the coding error of the track.
- Rate. H2's pooled rate (1.927 kbps for H2's rows) scaled per item by the
  ratio of coded rows after to before. This assumes the new rows cost what
  H2's rows cost on average, and leaves out extra track starts. Both new
  counts are reported, and the hull is read at each item's own rate.
- Decision rule (`k1c.DECISION_K1D`, fixed before any run). This is K1c's
  rule on the covered fits. The bound passes on an item when LPIPS on R is
  below the hull at that item's rate.
  - ≥ 3 of 4: hands get K6b's ladder (PLAN K6b: audit, zero-shot, overfit,
    match fit), with the mask-box fit as the coverage path.
  - < 3: hands are closed for every renderer.
  - LPIPS over S or DISTS on R crossing 3 differently: to the user.
- Hypothesis. Passes on 4 of 4. With full coverage, `warp@uncoded` falls
  to about 0.01 everywhere (as on P01_107 and P03_10). The coding error
  of the new rows keeps P02_02 and P09_106 below their hulls (0.204,
  0.184), as it did at 0.087–0.093 on the fitted items.
- Competing explanation. Mask boxes include the forearm, so WiLoR's fits
  there jitter. H2's Gaussian smoothing then moves the coded joints far
  from the uncoded ones, and the coding error eats the gain. The mean coded
  shift of new rows against H2's rows is reported to show this.
- Budget. One fleet job, the K1c job with WiLoR added (WiLoR verified on
  Ada/A6000 by H3b/H3c2): smoke ≤ 600 s (P01_107, 48 frames), full ≤ 45
  min. Ceiling 1 GPU-hour.
- Job `20261021T200019Z-e6b9c107` (gpu6, RTX 6000 Ada, shared; `1aa1b43`
  from `ready.json`; inputs H3c2's). Smoke passed the fleet validator. The
  full stage took 1,019 s and passes all 59 of `k1c validate`'s checks
  locally, including that under 1% of R is left unfitted on every item
  (0.0% on all four). Report `visor/k1d-2026-10-21/report/k1c-report.json`
  (`6427ea8f…7c68`, on `full/published.tar` `a8249947…1f42`).
- Outcome. On R, each item at its own rate:

  | item | new rows (H2's) | kbps | `warp` LPIPS | hull LPIPS | result | DISTS `warp` / hull | coded shift, H2's rows / new rows |
  |---|---:|---:|---:|---:|---|---:|---:|
  | P01_107 | 6 (474) | 1.95 | **0.087** | 0.257 | saves 96% | 0.070 / 0.222 | 7.4 / 9.3 px |
  | P02_02 | 237 (243) | 3.81 | 0.232 | 0.162 | loses (−214%) | 0.163 / 0.234 | 20.8 / 64.2 px |
  | P03_10 | 15 (416) | 2.00 | **0.083** | 0.153 | saves 96% | 0.097 / 0.201 | 8.9 / 16.1 px |
  | P09_106 | 123 (146) | 3.55 | 0.141 | 0.138 | loses (−3%) | 0.177 / 0.197 | 15.6 / 24.2 px |

  - LPIPS on R: below the hull on **2 of 4**. Over S: 2 of 4. DISTS on R:
    4 of 4. **By the rule, the outcome goes to the user again.**
  - Coverage worked: `warp@uncoded` is ≤ 0.0001 on every item (it was
    0.192 and 0.138). P02_02 improved from 0.272 to 0.232 and P09_106
    from 0.211 to 0.141.
  - The competing explanation holds. The mask-box fits jitter: their coded
    joints move 24–64 px from the uncoded ones. They also corrupt the H2
    rows they join (P02_02's H2 rows moved 10.9 px in K1c and 20.8 px
    here). Doubling the rows doubles the rate, too, so the hull is read at
    3.5–3.8 kbps, where it is lower. P09_106 misses by 0.003; P02_02 by
    0.07.
- Reading. Hands win where the hand is boxed well (P01_107 and P03_10, 96%
  each, at H2's rate). They lose where a good fit is missing, now through
  bad fits instead of no fits. Hypothesis (4 of 4) refuted.
- Scope. As K1c. The rate scaling assumes a new row costs an average H2
  row. Mask boxes include the forearm (the stated limitation).
- Budget: 1,019 s plus the smoke. K1c and K1d together used under 1
  GPU-hour.

### 2026-10-21 — K1e: K1d with hand-only boxes (and why two items fit badly)
- Question (user, 2026-10-21: "if it's cheap, try to box the hand first";
  "can we figure out why 2 pilots get good fit, and 2 don't?"). K1d's
  mask-box fits include the forearm and jitter (coded shift 24–64 px on
  new rows against 7–9 px on H2's). Does fitting on the hand alone remove
  that, and does the bound then pass?
- Change against K1d (nothing else changes). Each mask-box row is fitted
  twice. WiLoR on the mask's box gives a wrist and middle knuckle. H1's
  `wrist_split` cuts the mask there, and WiLoR runs again on the hand
  part's box (the first box if the part is under 16 px a side). The
  second fit is the row.
- Measured beside, from the run's arrays (CPU, in `k1c report`):
  - Jitter: the median frame-to-frame acceleration of the uncoded joints
    (px, mean over 21 joints, consecutive frames of a side), for H2's rows
    and the new rows, for K1d's fits (K1e's first pass, the same box and
    network, saved beside) and for K1e's second fits. This tests the competing explanation
    directly: K1d's new rows should jitter more than H2's, and K1e's less
    than K1d's.
  - Scene predictors per item, measurable at the encoder: the share of the
    hand frames H1's detector boxed (H2's rows over all rows), the mean
    hand-region size, and the median joint speed. They are measured on
    the pilot and named as candidates only. A scope chosen from 4 items
    would have to be tested on items it was not chosen from (B2's other
    30) before any headline uses it.
- Decision rule (fixed before any run). K1c's: the bound passes when
  LPIPS on R is below the hull at the item's rate on ≥ 3 of 4 items. Then
  hands get K6b's ladder with the two-pass fit as the coverage path.
  Otherwise hands are closed, and the outcome goes to the user if LPIPS
  over S or DISTS on R crosses 3 differently.
  - The user's proposed criterion (2026-10-21, made after K1c's and K1d's
    results): pass when the mean over {LPIPS on R, DISTS on R} of the
    share of items below the hull is ≥ 75%. K1c and K1d already meet it
    (2 of 4 and 4 of 4). It is reported beside, and the user decides which
    governs. The concerns: it was proposed after the results, it would
    turn both earlier runs into passes, and DISTS compares texture
    statistics, so it is lenient on the smooth fill and misplaced texture
    that are this bound's failure modes. The paper reports both metrics
    either way.
- **Amendment (user, 2026-10-21, before K1e's results; after K1c's and
  K1d's).** The user's criterion governs K1e and every later run. It
  passes when the mean over {LPIPS on R, DISTS on R} of the share of items
  below the hull is ≥ 75%, so LPIPS must win on at least half the items.
  It also applies **retroactively** to K1c and K1d. That is a post-hoc
  change and the paper must disclose it: both runs failed the rule written
  before them (LPIPS 2 of 4) and pass under this one (2/4 and 4/4, mean
  75%). Consequence: hands go to K6b's ladder. The held-out test (K1f)
  is judged by this criterion, fixed before it runs.
- Hypothesis. The hand-only fits jitter about as little as H2's rows, so
  P09_106 (0.003 short in K1d) passes and P02_02 (0.07 short, rubber
  gloves) still fails: 3 of 4.
- Competing explanations. (1) The jitter comes from the hands themselves
  (blur, gloves, occlusion), not the box, so the second fit jitters as
  much as the first. (2) The rate, not the fit, decides P02_02: doubling
  the rows reads the hull at 3.8 kbps, where it is 0.162.
- Budget. One fleet job, K1d's with a second WiLoR pass (≤ 45 min; K1d
  took 17 min). Ceiling 1 GPU-hour.
- Job `20261021T205023Z-a7bda6e6` (gpu6, RTX 6000 Ada, shared; `be8251e`
  from `ready.json`). Smoke passed the fleet validator. The full stage
  (about 17 min) passes all 59 of `k1c validate`'s checks locally. Report
  `visor/k1e-2026-10-21/report/k1c-report.json` (`7e6f8f69…d769`, on
  `full/published.tar` `c9287130…1524`).
- Outcome (on R, each item at its own rate; LPIPS bound / hull):

  | item | kbps | LPIPS | DISTS | K1c → K1d → K1e LPIPS | joint acceleration, px: H2's rows / new rows, mask box / new rows, hand box |
  |---|---:|---:|---:|---|---:|
  | P01_107 | 1.95 | **0.086** / 0.257 | **0.069** / 0.222 | 0.087 → 0.087 → 0.086 | 5.1 / 9.0 / 9.1 |
  | P02_02 | 3.81 | 0.196 / 0.162 | **0.142** / 0.234 | 0.272 → 0.232 → 0.196 | 20.3 / 56.0 / 40.9 |
  | P03_10 | 2.00 | **0.083** / 0.152 | **0.097** / 0.201 | 0.093 → 0.083 → 0.083 | 6.5 / 28.2 / 28.5 |
  | P09_106 | 3.55 | 0.139 / 0.138 | **0.175** / 0.197 | 0.211 → 0.141 → 0.139 | 8.0 / 20.9 / 20.1 |

  - LPIPS: 2 of 4 below the hull (P09_106 misses by 0.001). DISTS: 4 of 4.
    **The amended criterion passes (mean 75%). The rule written before
    K1e fails (LPIPS 2 of 4).**
  - Jitter. The new rows jitter 2–9× more than H2's rows on the same item.
    The hand-only box cuts P02_02's jitter from 56 to 41 px, and P02_02's
    LPIPS from 0.232 to 0.196. It changes nothing on P01_107, P03_10 or
    P09_106. So the competing explanation (1) holds in part: the jitter
    there comes from the hands, not the box. P02_02's own H2 rows jitter
    20 px, against 5–8 px on the other items (rubber gloves).
  - The hypothesis (3 of 4 by LPIPS) is refuted by 0.001 on P09_106.
- Decision. Under the user's criterion, hands go to K6b's ladder with
  the two-pass fit as the coverage path. K1f, the held-out test, uses the
  same path.

### 2026-10-21 — K1f: the hand bound's scope on held-out items
- Question (user, 2026-10-21). On the pilot the bound wins big on two
  items (96%) and loses narrowly on the two where H1's detector boxed
  about half of the hand frames. Is "scenes where the detector tracks the
  hands" a scope where PointStream wins? Does it hold on items the scope
  was not chosen from?
- Items. B2's other 30 items, in a content-blind order (sha256 of the id;
  `tools/datasets/k1f_inputs.py`), 240 frames each. Two come from videos
  in G's training footage (P07_110, P22_107). They are reported apart,
  and the clean subset is reported beside the pooled one.
- Bound. K1e's path, unchanged: H2's WiLoR rows plus the two-pass
  hand-box fit for every boxable hand without one, coded with H2's 100 ms
  choice, K1c's warp, judged on R. Rate: K1d's per-item scaling.
- Envelope, coded per item as the pilot's was (`h3d.code_point`): H3b2's
  seven SVT-AV1 low points, DCVC-UF HT-L off the shelf at QP 0 at full
  size and QP 0, 21 and 42 at ½ and ¼, and H3d's G-L network on each ½
  and ¼ point. Points above about 20 kbps are left out. On the pilot, the
  hull with and without them is identical at every rate up to 5 kbps
  (checked on H3d's report). Savings that need a higher rate are capped.
  B2's CRF 41–62 streams are not used (all ≥ 35 kbps).
- Scope, fixed now from the pilot. An item is in scope when H1/H2's
  coverage, H2's rows over the side-frames with a boxable hand mask, is
  ≥ 0.8. On the pilot the two winners have 0.99 and 0.97, and the two
  losers 0.51 and 0.54. The encoder can measure it.
- Decision rule (`k1f.DECISION`, the user's criterion). A group passes
  when the mean over {LPIPS on R, DISTS on R} of the share of its items
  whose bound is below the hull at its rate is ≥ 75%.
  - In scope passes and out of scope fails: the scope is confirmed.
    Headline: on scenes in scope (their share of items and of hand frames
    reported), the hand bound saves a median X% of the envelope's rate at
    about 2 kbps. It is a bound (appearance granted), and K6b's ladder
    decides what a renderer keeps.
  - All items pass: no scope is needed, and the claim is over all items.
  - In scope fails: the scope claim is dropped, and hands' place on K6b's
    ladder goes back to the user.
- Hypothesis. In scope passes (both metrics ≥ 75%). Out of scope sits
  near the bar, as P09_106 does.
- Competing explanations. (1) The pilot's two winners are atypical
  (still, close views), and coverage ≥ 0.8 does not predict a win
  elsewhere. Then in scope fails. (2) Coverage is a proxy for motion. The
  joint speed per item is reported beside to tell the two apart.
- Budget. Parts of 2 items, each a fleet job of ≤ 45 min (the first one
  times an item; a smoke of 48 frames with two DCVC-UF points). Ceiling 8
  GPU-hours (user, 2026-10-21) for all of K1f. If it runs out, the items
  done in the fixed order are reported, and so is the shortfall.
- Attempts (infrastructure, no score read). Part 0
  `20261021T211220Z-3d431acd` (`eca0d32`, gpu6): the smoke passed (159 s),
  then the full stage overran its declared 36,000 MiB at 38,200 (PyTorch's
  cache held across points, with WiLoR and G resident) and was stopped
  before finishing an item. Fixed: the cache is released after each point,
  and 40,000 MiB is declared.
  Part 6 `20261021T220155Z-2e7fdeb3` ran past the 2,700 s stage limit
  (two slow items). Its `partial.tar` (`d34d41fd…7619`) keeps P06_03
  finished and P26_108's envelope. Rerun as part 6b with `--restore`:
  units of the same items and window length, sha256-checked, are kept, and
  only P26_108's bound is computed. Parts 8–10 had not started. They were
  cancelled and resubmitted as one item per job, so no part nears the
  limit. Items and order are unchanged. Part 8 (`20261021T220208Z-fda99e47`)
  had begun its smoke when it was cancelled, and its two items were
  resubmitted the same way.
- Reproduction check (added after the held-out scores were read, before
  they are recorded as evidence). K1f's envelope path has not been shown
  to reproduce H3d's recorded points. One job runs `k1f run` on the pilot
  item P01_107 (same flags, same inputs). Every point it shares with
  H3d's component report (SVT-AV1's seven low points, DCVC-UF's seven,
  G-L's six) must match within 2% in rate on R and 0.003 in LPIPS on R.
  If any point fails, the held-out result is not evidence until the
  difference is explained.
  Job `20261022T013208Z-9e048ecf` (gpu6, `3f530f1`): all 20 points pass,
  18 exactly. G-L at QP 0 ½ differs by 1.05% in rate and 0.0011 in LPIPS,
  and at QP 42 ½ by 0.34% and 0.0001. The bound reproduces K1e's 0.0865.
- Jobs (RTX 6000 Ada, shared, gpu6). Parts 0–5 and 7 (`d1a47c3`), 6b and
  the six single-item parts (`b9e1d58`), each revision from its
  `ready.json`: `20261021T211923Z-08179c83`, `20261021T220120Z-f7cbc6ed`,
  `20261021T220127Z-381c7366`, `20261021T220134Z-c0b4a121`,
  `20261021T220141Z-eff4e8be`, `20261021T220148Z-9e99d6a7`,
  `20261022T000642Z-3f9256ae`, `20261021T220201Z-9337c526`,
  `20261022T000649Z-adac9f11`, `20261022T000656Z-1aea7c30`,
  `20261022T000702Z-d16bf68f`, `20261022T000709Z-f303e32f`,
  `20261022T000742Z-2efab10c`, `20261022T000749Z-e723338c`. Every smoke
  passed the fleet validator, and every full stage passes `k1f validate`
  locally (44 checks for two-item parts, 26 for single, 46 for 6b).
  Report `visor/k1f-2026-10-21/report/k1f-report.json` (`2f6e4291…888e`),
  with the validation receipts in `k1f-validation.tar`. GPU time: 22,488 s
  of stage execution, plus the failed and cancelled attempts and the
  reproduction check, about 7.6 GPU-hours, within the 8.
- Outcome. 22 of the 30 held-out items were run (the budget's
  content-blind order; positions 22–29 were not run). On R, each item at
  its own keypoint rate (1.93–2.72 kbps):
  - **The bound is below the hull on 22 of 22 items by LPIPS and 22 of 22
    by DISTS.** Every group passes the criterion at 100%: all (22), in
    scope (19), out of scope (3), clean of G's training (21), and in
    scope and clean (18).
  - LPIPS on R: the bound's median is 0.084 (0.059–0.194), and the hull's
    at the same rate is 0.175 (0.076–0.410). On 14 items the bound beats
    every pixel point coded (up to about 45 kbps, SVT-AV1 CRF 63 at full
    size), so their saving is censored. On the 8 others the median
    saving is 84% (9–91%). The narrowest margin is P04_24, at 0.075
    against 0.076.
  - Scope. 19 of 22 items (86% of items and of hand frames) have coverage
    ≥ 0.8. The 3 below (0.71–0.73) also win once the hand-box fit fills
    them.
- Decision (by the rule). **All items pass, so no scope is needed. The
  claim is over all held-out items:** with appearance granted, egocentric
  hands sent as 21 coded keypoints (about 2 kbps) beat SVT-AV1, DCVC-UF
  and DCVC-UF + G on the hand region, on every held-out item, on both
  metrics.
- Reading. The pilot was harder than the held-out set. With the
  hand-box path, P02_02 (rubber gloves, jittery fits) and P09_106 lose
  by LPIPS, and on the held-out set nothing does. Pooled with the pilot,
  the bound wins 24 of 26 items by LPIPS and 26 of 26 by DISTS. The
  pilot's coverage predictor was not needed. Coverage is fixed by fitting
  every masked hand, and what is left on P02_02 is fit quality, not
  missing fits.
- Hypothesis. In scope passed as predicted. "Out of scope sits near the
  bar" is refuted: all 3 win clearly.
- Scope and caveats. This is a bound: the hand's own pixels are moved,
  so appearance and outline are granted. K6b's ladder decides what a
  renderer keeps. 22 of 30 held-out items were run. The rate is K1d's
  per-row scaling (new rows assumed to cost an average H2 row). G's
  training overlaps one item (P22_107). Judges are LPIPS and DISTS on R;
  pose fidelity was not judged. Savings are censored above about 45
  kbps.
- Completion (user, 2026-10-22: "yes", about 2.6 more GPU-hours). The 8
  items not run (positions 22–29 of the fixed order) run as single-item
  jobs, unchanged. The result is read over all 30.
  Jobs (`ffd6ccd`, gpu6, RTX 6000 Ada): `20261022T072405Z-ab09c0af`,
  `20261022T072414Z-a0071b24`, `20261022T072430Z-d94d8b6e`,
  `20261022T072439Z-e7c8ed8c`, `20261022T072448Z-b850bbac`,
  `20261022T072458Z-9cc87e86`, `20261022T072508Z-128ba164`,
  `20261022T072522Z-e7c0f8bc`. Each passes all 26 checks locally (9,378
  s together). Report over all 30:
  `visor/k1f-2026-10-21/report/k1f-report-30.json` (`d31b1fc2…0eb8`), with
  `k1f-validation-30.tar`. K1f's total: about 10.2 GPU-hours (8 approved,
  then 2.6 more, user, 2026-10-22).
- **Outcome over all 30 held-out items.** The bound is below the hull on
  **30 of 30 by LPIPS and 30 of 30 by DISTS**, and every group passes at
  100%: all (30), in scope (26), out of scope (4), clean of G (28), and
  in scope and clean (24). LPIPS on R: the bound's median is 0.086, the
  hull's 0.174. On 17 items the bound beats every pixel point coded (up
  to about 45 kbps). On the 13 others the median saving is 83% (10–93%).
  The narrowest margin is still P04_24 (0.075 against 0.076). In scope:
  87% of items and of hand frames. Pooled with the pilot (K1e's path),
  the bound wins 32 of 34 items by LPIPS and 34 of 34 by DISTS.


### 2026-10-22 — K6b (hands) 1: audit of hand renderers
- Question ([PLAN K6b](../PLAN.md), hands after K1f). K1f's bound moves
  the hand's own pixels to the 21 coded keypoints. A renderer has to draw
  them from a reference instead. Which released models can do that, and
  which one does the zero-shot rung test?
- Interface, fixed by K1f. Per hand and frame the renderer gets a
  reference crop of that hand and the 21 coded 2D keypoints (H2's 100 ms
  code, 1.93–2.72 kbps per item). Its pixels are composited inside R, the
  granted outline, with the source outside R (`k1c.compose`). So it has to
  draw a hand crop, not a scene. H3e's failure was that the scene drifted:
  JointControlVideo generated the whole frame from the first frame and
  never saw the camera's motion. The local interface removes that failure.
- Criteria: conditioning on a reference image and 2D (or 3D) hand
  keypoints; released weights; licence for research; inference cost per
  frame; whether it has seen VISOR (overlap is allowed, recorded, and the
  clean subset reported, per AGENTS.md).

  | model | conditioning | weights, licence | cost | verdict |
  |---|---|---|---|---|
  | [FoundHand](https://github.com/arthurchen0518/FoundHand) (CVPR 2025) | reference image (mask optional), 2D keypoints as heatmaps (OpenPose 21 per hand); per frame, with a released zero-shot video procedure (first frame plus recent generated frames) | released (Google Drive), GPL-3.0 code, CC BY-NC data | DiT on SD-VAE latents, 256² (32² latent); timing not published | **chosen** |
  | K6's `ref` (no learning) | the reference crop's pixels warped from its keypoints to the coded ones | in repo | CPU | **floor**: run beside, so the generator has to beat a warp of the same reference |
  | Moore-AnimateAnyone, original and tennis fine-tune (in `Models`) | reference image, DWPose maps of the full body | in `Models`, Apache-2.0 | SD 1.5, about 1 s per frame | rejected for hands: whole-body model, and FoundHand's paper reports it at LPIPS 0.35 against FoundHand's 0.17 on hand videos (their Table 3b, 12 phone videos). It stays K6b's body candidate |
  | SD 1.5 + ControlNet OpenPose (in `Models`) | text, OpenPose hand map; a reference only through an adapter | in `Models` | per frame | reserve: no reference conditioning trained for hands |
  | JointControlVideo (H3e, in `Models`) | first frame, 3D joints, camera intrinsics | Apache-2.0 | 27 min per 81 frames | rejected: H3e failed (−297%), and the cause was the scene drift |
  | [Generated Reality](https://github.com/LafouCC/generated-reality) (2026) | first frame, 21 3D joints per hand in camera space, head pose | MIT code; trained checkpoint not released (base Wan2.2-Fun-5B-Control) | 5B video DiT | not zero-shot: needs training on HOT3D/GigaHands first |
  | Re-HOLD (CVPR 2025), iDiT-HOI (2025), ManiVideo, RHanDS | hand–object layouts or meshes | no released weights found | | not usable |

- Overlap. FoundHand's repository lists VISOR among FoundHand-10M's
  sources, and VISOR's public annotations are its train and val splits,
  which is where B2's items come from. Before any score is read, the
  zero-shot rung checks the released data lists for each item's video.
  Items whose video is listed are reported as seen. If the lists cannot
  tell, every item counts as seen, and the claim is scoped to "a renderer
  that may have seen this footage". The judges (LPIPS's AlexNet, DISTS's
  VGG) have not seen it.
- Decision: FoundHand is the zero-shot candidate, with `ref` as the floor.
  Rung 2 needs its source at a pinned commit and its weights in
  `Models/FoundHand` (with a sha256 manifest), and an environment check.
  Its pins (Python 3.9, torch 2.3.0, lightning 2.3.0, timm 1.0.7) are older
  than the main environment's (torch 2.10, timm 1.0.30). It is plain DiT
  code, so the main environment plus lightning is tried first. A second
  environment is made only if an import or the smoke's output check fails.
- Searches (2026-10-22): FoundHand's repository and paper, Generated
  Reality's project page and repository, Re-HOLD's project page, and web
  searches for hand reenactment with released weights. Nothing newer than
  H3e's audit has released weights for a reference-plus-keypoint hand
  renderer, other than the fact that Generated Reality's code is out.

### 2026-10-22 — K6b (hands) 2: zero-shot oracle (rule)
- Question. With one unbilled reference crop per hand and the coded
  keypoints, does FoundHand, off the shelf, keep K1f's win over the pixel
  hull? K1f's bound (0.084 median LPIPS on R over its first 22 items;
  0.086 over all 30) is the ceiling, and the hull at each item's keypoint
  rate (median 0.175; 0.174 over 30) is the bar.
- Items. Positions 0–7 of K1f's content-blind order
  (`tools/datasets/k1f_inputs.py`), 240 frames each. All 8 have K1f's
  bound and hull recorded, so nothing is re-coded. Items in G's training
  footage and items whose video FoundHand has seen are reported apart,
  beside the pooled numbers. Development (crop rule, keypoint order,
  sampler settings) is done on K1c's pilot item P01_107 only. It is not
  among the 8.
- Keypoints and fits. K1f's, unchanged: H2's WiLoR rows plus the two-pass
  hand-box fit, coded with H2's 100 ms choice and projected to 21 joints.
- Reference (granted, unbilled, as K6b's step 2). For each hand side, the
  window's first frame where that side is fitted. The crop is the same as
  for the targets.
- Crop. Per hand and frame, a square around the union of the coded
  keypoints' box and that side's mask, ×1.25, resized to 256². The
  keypoints are mapped into the crop, in OpenPose order, with the right
  hand in joints 0–20 as FoundHand's labels have it. The output is
  resized back to the box (Lanczos).
- Variants, each composited by `k1c.compose`: inside that side's R the
  rendered pixels, Telea fill where no fit exists (K1c's charge), and the
  source outside R. R is the generator's matte, granted, as `warp`
  granted the outline.
  - `gen`: FoundHand with its released video procedure (the reference
    as first frame, then recent generated frames), at its released
    sampler settings.
  - `gen@frame` (diagnostic): FoundHand per frame from the reference
    alone. It separates the autoregressive drift from the per-frame
    error.
  - `ref` (floor): K6's warp of the reference crop's pixels from its
    uncoded keypoints to the coded ones.
  - `warp` reproduced (check): K1f's bound. It must match K1f's recorded
    LPIPS on R within 0.002 per item, or nothing is read.
- Judges, as K1f: LPIPS on R and DISTS on R (`h3c.Scorer`), mean over the
  window's frames with R non-empty. Reported beside: PSNR on R, flicker
  (mean absolute luma change between consecutive frames inside R,
  against the source's), WiLoR on the rendered crop against the coded
  keypoints (does it follow its input), and a visual check of one frame
  per item.
- Decision rule (`k6bh.DECISION`, fixed before any run; the user's
  criterion). An item is won on a metric when `gen`'s value on R is below
  the hull's at that item's keypoint rate (K1f's report). Score = mean
  over {LPIPS, DISTS} of the share of the 8 items won.
  - **Pass** (score ≥ 75%): the zero-shot route holds with the reference
    granted. The overfit rung is skipped. Next is the same oracle with
    the reference billed (coded once per window and its bits added), the
    interface K6c would use.
  - **Fail** (score < 75%): the overfit rung (a fit of FoundHand, or a
    LoRA on it, to the window's own frames) gets its rule and budget,
    with the gap from `gen` to the bound reported per item.
  - If `ref` scores ≥ 75%, a warp of one reference already suffices.
    That goes to the user, with the better of `ref` and `gen` per item.
  - If the score is under 75% on the unseen items alone but at or above
    on the seen ones (or the reverse), the outcome goes to the user.
- Hypothesis. `gen` lands near the bar, with a score around 50%. FoundHand
  reports LPIPS 0.17–0.20 on reposing at 256², which is about the hull's
  median, and DISTS forgives texture more than LPIPS does. `ref` scores
  lower: it is K6's failure mode (what the pose cannot describe).
- Competing explanations. (1) Having seen VISOR makes FoundHand better
  here than on new footage. The seen/unseen split shows it if the lists
  allow one. (2) The granted matte hides the generator's outline errors,
  as it did for `warp`, so a pass is still a bound on the matte. (3) The
  256² crop is the limit, not the generator. Large hands (box > 512 px)
  are compared with small ones beside the result. (4) Per-frame metrics
  ignore flicker, so flicker is reported, and a visual check is made of
  any item that passes with flicker above the source's.
- Before the run (CPU): the FoundHand source at a pinned commit and its
  weights into `Models/FoundHand` with a sha256 manifest; its import in
  the main environment (plus lightning); the overlap check above.
- Budget. GPU, RTX 6000 Ada (FoundHand's first run on the class). A smoke
  of at most 600 s on P01_107, 24 frames: it asserts CUDA, the dtype,
  the attention kernel and the output's range, and times one frame. The
  timing is recorded here before the pilot. The released step count is
  kept if one 240-frame item (both hands, `gen` and `gen@frame`) fits in
  40 minutes; otherwise the most steps that fit, not below half the
  released number. Then a pilot on P01_107 (development, not scored for
  the rule), and the 8 items in parts of at most 45 minutes, each saving
  every finished item. Ceiling 3 GPU-hours for the rung, smoke included.
  If the budget runs out, the items finished in the fixed order are
  reported, and so is the shortfall.
- Revision (2026-10-22, after reading FoundHand's source, before any run).
  - Source and weights. FoundHand at `ae0e980` (archive
    `visor/k6bh-2026-10-22/inputs/foundhand-src-ae0e980.tar.gz`,
    `fe8ff8a0…7f35`). The released checkpoint
    `DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt` (11.6 GB, `66434caf…62d7`)
    and SD's `vae-ft-mse-840000-ema-pruned.ckpt` (`c6a580b1…3dab`, as Hugging
    Face lists it) are in `Models/FoundHand` with a `MANIFEST.json`,
    hard-linked into the run's `inputs/`. Inference uses the DiT-XL/2's
    `ema_state_dict` (as the demos do) and needs only torch, numpy and
    timm. Lightning, SAM and mediapipe serve training and the demos' own
    keypoints and masks, which K1f supplies. So the main environment
    (`pointstream-20261009T214002Z`) runs it, and no environment is added.
    The DiT runs under bf16 autocast, and the VAE in float32.
  - Steps. The released demos sample with 250 respaced DDPM steps and
    guidance 2.5. At that count the 8 items cannot fit the 3 GPU-hours. A
    pilot on P01_107 (development, not scored) runs all four variants at
    50 steps and records seconds per step. The step count for the 8 items
    is the largest of {250, 100, 50} whose projected cost (both FoundHand
    variants) fits the remaining ceiling with a 25% margin. If 50 does not
    fit, `gen@frame` runs on the first 2 items only. If `gen` alone does not
    fit at 50, the rung goes back to the user. The count is recorded here
    before the 8 items run.
  - Keypoints. Read from K1f's published rows (`row_t`, `row_right`,
    `uncoded_kp2d`, `coded_kp2d` of each item's K1c unit; P01_107 from the
    reproduction job `20261022T013208Z-9e048ecf`). They are exactly the
    keypoints K1f scored, with no second WiLoR run. K1f did not publish
    joint depths, so overlapping hands are drawn smaller first (smaller
    taken as farther). The `warp` check against K1f's recorded value
    tests that too.
  - Overlap check. FoundHand-10M's file lists are only on Globus, which
    needs a login, so the lists cannot tell. Every item counts as seen, and
    the claim is scoped to a renderer that may have seen this footage
    (VISOR's annotated frames, not necessarily these windows).
  - Dropped before any run: "WiLoR on the rendered crop" (reported beside).
    It would add WiLoR to the job for a number that decides nothing.
    Flicker and the visual check stay.
  - Handedness. The README and the demos put the true right hand in slots
    0–20, un-mirroring MediaPipe, so no ordering check is needed.
- Attempts and pilot (development on P01_107, not scored for the rule;
  timings from a shared GPU, contaminated, not evidence).
  - `20261022T080212Z-35880d37` (`2344973`, gpu5): the smoke stopped while
    loading the checkpoint. It pickles the training script's `HandDiffOpts`
    from `__main__`. Fixed in `796e6e0` by loading with `__main__` classes
    stubbed (inference reads only `ema_state_dict`). Infrastructure, no
    score.
  - `20261022T080535Z-df3082c3` (`796e6e0`, gpu5, RTX 6000 Ada, shared):
    the smoke and the full stage pass (the full stage locally, 17 checks).
    First run on Ada: flash attention, bf16 CUTLASS GEMMs on sm80 kernels,
    35 ms per guided step (two samples), 675.6 M parameters, no missing
    keys. `warp` reproduces K1f exactly (LPIPS on R 0.08647 against
    0.08647). The full item took 2,045 s: `gen` 1,008 s, `gen@frame`
    488 s, and decoding, `ref`, `warp` and scoring about 550 s.
  - Pilot values on R (P01_107, 1.95 kbps; K1c's hull there reads 0.258
    LPIPS and 0.223 DISTS): `gen` 0.496 / 0.255, `gen@frame` 0.463 /
    0.258, `ref` 0.575 / 0.279, `warp` 0.086 / 0.069. Flicker (luma, R):
    source 6.6, `gen` 10.8, `gen@frame` 31.0. On its first 24 frames
    (the smoke), `gen` read 0.314 / 0.201. The later frames are much worse
    than the first ones.
  - Visual check (frame 120). Where a crop ran past the frame's edge (the
    forearms at the bottom), the zero padding showed up as a black band in
    FoundHand's output and was pasted onto R. Fixed in the next commit:
    the square crop is shifted inside the frame. The generated hands are
    plausible but not the source's: the fingers and the forearm's position
    differ.
  - Budget. Spent so far: about 0.7 GPU-hours of the 3. At 50 steps, `gen`
    with `ref` and `warp` costs about 26 min per item, so the 8 items need
    about 3.5 GPU-hours. **`gen` alone does not fit at 50 steps. By the
    rule, the rung goes back to the user.**
- Decision (user, 2026-10-22, before any scored run): **4 items at 50
  steps**: positions 0–3 of K1f's order (P09_103, P25_09, P22_107 (G's
  training footage), P06_108), with `gen`, `ref` and `warp`, and without
  `gen@frame`. One item per job, about 1.7 GPU-hours, within the 3. The
  rule is unchanged otherwise: the score is the mean over {LPIPS, DISTS}
  of the share of the 4 items won, and it passes at ≥ 75% (3 of 4 on
  both, or 4 and 2). Code `466f73f` (crops inside the frame).
- Jobs (RTX 6000 Ada, shared, so timings are contaminated; code from each
  job's `ready.json`). P06_108 `20261022T084910Z-abd3eff7` (`1320465`).
  P22_107 `20261022T084905Z-09c2f882` (`1320465`). P25_09
  `20261022T090959Z-8ccce208` (`7e75675`: the GPU cache is released
  between rendering and scoring, and 32,000 MiB is declared). The
  rendering is the same in both revisions. Every smoke passed the fleet
  validator, and every full stage passes `k6bh validate` locally. Each
  stage's `warp` reproduces K1f's recorded LPIPS on R within 0.0001.
  P25_09's `ref` covers 49% of R, under the validator's first coverage
  threshold (50%), which was meant to catch an empty drawing. It was
  lowered to 10% for `ref` after that stage was read (`1f47d1d`). The
  decision reads `gen` only, which covers 100%.
  Infrastructure failures (no score): P25_09 `20261022T084900Z-7e08ab8f`
  overran its declared 16,000 MiB (24,886), and P09_103
  `20261022T084856Z-539368dd` overran it too (22,058). P09_103's rerun
  `20261022T092110Z-d37eedfb` overran 32,000 MiB (38,926) while scoring.
  Report `visor/k6bh-2026-10-22/report/k6bh-report.json` (`e32546d7…2b4f`),
  with the local validations in `k6bh-validation.tar` (`8c97643f…0a0f`).
  GPU time: about 2.7 GPU-hours of the 3, failed attempts included.
- Outcome. 3 of the 4 items were scored. P09_103 was not: by then it
  could not change the decision, and the budget was nearly spent. On R,
  at each item's keypoint rate:

  | item | kbps | `gen` LPIPS / hull | `gen` DISTS / hull | `ref` LPIPS / DISTS | `warp` LPIPS / DISTS | crop (median px) | flicker `gen` / source |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | P06_108 | 1.94 | 0.308 / 0.172 | 0.273 / 0.206 | 0.391 / 0.307 | 0.091 / 0.123 | 550 | 19.7 / 14.4 |
  | P22_107 (G) | 1.93 | 0.281 / 0.236 | **0.200** / 0.204 | **0.191** / **0.125** | 0.059 / 0.049 | 742 | 6.5 / 3.8 |
  | P25_09 | 2.14 | 0.524 / 0.260 | 0.353 / 0.240 | 0.429 / 0.297 | 0.097 / 0.102 | 1,067 | 11.0 / 3.2 |

  (Bold: below the hull. G: in G's training footage. FoundHand may have
  seen every item, so the overlap could not be resolved.)
  - `gen` wins 0 of 3 items by LPIPS and 1 of 3 by DISTS: **score 17%**.
    With P09_103 won on both, it would be at most 37.5%.
  - `ref` (a warp of one reference) wins 1 of 3 on both metrics (33%), and
    beats `gen` on P22_107 and P25_09 by LPIPS. The pilot item
    P01_107 (development) read `gen` 0.496 / 0.255 against a hull of
    0.258 / 0.223.
  - `warp`, the bound, wins 3 of 3 on both (it reproduces K1f).
- Decision (by the rule): **fail.** FoundHand off the shelf does not keep
  K1f's win. A zero-shot renderer with one unbilled reference loses to
  the pixel hull at about 2 kbps on every scored item by LPIPS. The next
  rung (overfit) gets its rule and budget below. The decision does not
  depend on the items FoundHand may have seen, since none could be
  separated.
- Reading. The gap from `gen` to the bound (0.19–0.43 LPIPS) is much
  larger than the gap from the bound to the hull. FoundHand's hands are
  plausible but not the source's: the fingers and the forearm are drawn
  where its 256² crop puts them, and it flickers more than the source
  (6.5–19.7 against 3.2–14.4). That `ref` sometimes beats it says that
  where one reference is close in pose, real pixels beat generated ones.
  The forearm, which is in R, is a large part of what neither renders
  (K1c's R includes it). The largest crops (P25_09, about 1,070 px down
  to 256) lose the most.
- Hypothesis ("`gen` lands near the bar, score around 50%"): refuted.
  The score is 17%. Competing explanation (3), the 256² crop, fits the
  ordering by crop size but was not isolated. (1), having seen VISOR,
  cannot be tested, and the result is a fail despite it.

### 2026-10-22 — K6b (hands) 2b: a bank of free references (rule)
- Question (user, 2026-10-22, after rung 2). Rung 2's `ref`, one free
  reference warped to the coded keypoints, beat FoundHand on 2 of 3 items
  and the hull on 1. Can a few real references, each frame warped from
  the one nearest in pose, carry the hand's appearance well enough to beat
  the hull at the keypoint rate? H3a2 already showed that refreshed
  references coded as their own stream lose to SVT-AV1 at 40–140 kbps.
  This tests the corner H3a2 left open: very few references, at about
  2 kbps.
- Why it bounds. The references are the window's own frames, uncoded and
  unbilled. A real bank (coded, billed, or sent once per video from
  outside the window) can only do worse. If this fails, the reference
  route for hands closes at every rate this side of H3a2. If it passes,
  the next test is the buildable form: a bank chosen outside the window,
  coded, and billed once per video.
- Variants, on rung 2's 3 scored items (P06_108, P22_107, P25_09), with
  P01_107 for the smoke. K1f's rows, R as the matte, and Telea fill where
  nothing is drawn, as rung 2.
  - `bank@K`, K ∈ {1, 4, 16} references per hand side. Chosen at the
    encoder by farthest-point sampling on the uncoded poses (wrist-centred,
    scaled by the pose's extent), starting from the side's first fitted
    frame. Per frame, the decoder picks the reference whose coded pose is
    nearest, and warps its side mask's pixels from the reference's coded
    keypoints to the frame's (`k6.draw`). Only what a decoder holding the
    references would have is used.
  - `warp`: K1f's bound, the check (within 0.002 of K1f's LPIPS on R).
- Decision rule (`k6bh.BANK_DECISION`, fixed before any run; the user's
  criterion). Per K, an item is won on a metric when `bank@K` on R is
  below K1f's hull at the item's keypoint rate (the references free).
  Score = mean over {LPIPS, DISTS} of the share of the 3 items won.
  - **Pass** (some K scores ≥ 75%, i.e. all 3 items on one metric and at
    least 2 on the other): the next test is the buildable bank (outside
    the window, coded, billed once per video), with its own rule.
  - **Fail** (no K reaches 75%): no bank of real references beats the
    hull at the keypoint rate. With H3a2, the reference route for
    egocentric hands closes. Rung 3 (a generator fitted per clip) is then
    the last open rung, and it waits for the user.
- Hypothesis. `bank@16` passes, and `bank@1` does not (rung 2's `ref`
  was 1 of 3). Hands show few distinct poses within 4.8 s, so 16
  references cover most frames.
- Competing explanation. The coding error of the keypoints (7–12 px)
  plus the warp's blending of fingers, not the reference's pose, keeps
  the warp above the hull. Then `bank@16` stays near `ref` and far above
  `warp`, even with nearer references.
- Budget. No generation. GPU only for LPIPS and DISTS (four scorings
  per item). One job per item, ≤ 45 min each, declared at 40,000 MiB (rung
  2's scoring peaked at 38,926). Each job's smoke (at most 600 s) runs
  24 frames of its own item, so P01_107 is not needed. Ceiling 0.75
  GPU-hours.
- Jobs (RTX 6000 Ada, gpu6, shared; code `e0d4df5` from `ready.json`):
  P06_108 `20261022T105912Z-875474c1`, P22_107 `20261022T105916Z-287e2bd3`,
  P25_09 `20261022T105921Z-160d3f85`. Every smoke passed the fleet
  validator, and every full stage passes `k6bh validate` locally, with
  `warp` reproducing K1f. Report
  `visor/k6bh-2026-10-22/report/k6bh-bank-report.json` (`c4c0b472…5b70`),
  with the validations in `k6bh-bank-validation.tar` (`377b2202…7c39`).
  GPU time: 2,080 s of stages, about 0.6 GPU-hours with the smokes,
  within the 0.75.
- Outcome. On R, at each item's keypoint rate, the references free:

  | item | hull LPIPS / DISTS | `bank@1` | `bank@4` | `bank@16` | `warp` |
  |---|---:|---:|---:|---:|---:|
  | P06_108 | 0.172 / 0.206 | 0.416 / 0.322 | 0.344 / 0.276 | 0.227 / 0.209 | 0.091 / 0.123 |
  | P22_107 (G) | 0.236 / 0.204 | **0.198 / 0.128** | **0.173 / 0.116** | **0.100 / 0.078** | 0.059 / 0.049 |
  | P25_09 | 0.260 / 0.240 | 0.429 / 0.296 | 0.293 / **0.221** | **0.190 / 0.159** | 0.097 / 0.102 |

  Scores (mean over {LPIPS, DISTS} of the share of items won): `bank@1`
  33%, `bank@4` 50%, `bank@16` 67%. Clean of G's footage (P06_108,
  P25_09): 0%, 25%, 50%.
- Decision (by the rule): **fail.** No K reaches 75%. `bank@16` misses by
  one item-metric: P06_108's DISTS, 0.209 against 0.206.
- Reading. The rule expected a flat curve if the reference route were
  closed. Instead quality improves steeply with K: from K = 1 to 16, LPIPS
  on R falls by 45–56% on every item, and 16 references reach the hull on
  2 of 3 items. The competing explanation (the coding error and the warp's
  blending, not the reference's pose, keep the warp above the hull) is
  refuted. Nearer references close most of the gap to `warp`, so the
  pose distance to the reference is what costs. The hypothesis held for
  `bank@1`, and fell one item-metric short for `bank@16`.
  So the route does not close as the rule's fail branch states. What is
  open is a bank larger than 16, which only makes sense if it is
  affordable: sent once per video (VISOR's videos run for minutes) and
  chosen outside the window. That is the buildable test, with K up to 64
  and its billed rate. It needs keypoints for frames outside the windows
  (H2's poses cover the windows only), so it is a task of its own, not
  run here. Rung 3 (a generator fitted per clip) waits behind it.

### 2026-10-22 — K6b (hands) 3: overfit oracle (rule, not run)
- Question. Can FoundHand, fitted on the window's own frames, draw that
  hand well enough to beat the hull at about 2 kbps? It has seen the
  frames it is judged on (H3d's oracle form), so it bounds every renderer
  fitted elsewhere in the same video. If it fails, the generative route
  closes for egocentric hands at the keypoint stream's rate: K1f's win
  needs the hand's own appearance per frame, which only pixels carry.
- Model. FoundHand's EMA weights, fine-tuned per item (full DiT, or a
  LoRA if the smoke shows full tuning cannot fit 48 GB) with its own
  diffusion loss (`training_losses` in its `gaussian_diffusion`; no
  training script is released, so the loop is written here). Pairs
  (reference crop, target crop) come from the window's own frames and
  the same crops as rung 2 (K1f's uncoded keypoints for training, coded
  ones at test).
- Items. The 3 items scored in rung 2 (P06_108, P22_107, P25_09), with
  P01_107 for the smoke and the pilot. The same judges, hull and
  criterion. Pass (≥ 75%): rung 4 (a fit on other shots of the same
  video, billed). Fail: the generative route closes for hands, and K1f's
  bound stays a bound.
- Training discipline (PLAN K6b). A smoke of at most 10 min: the loss
  falls over 200 steps, and a sample changes towards the target. Then a
  learning-curve pilot on P01_107: fit on frames 0–179 at 25%, 50% and
  100% of a 30-minute step budget, judge `gen` on the held-out frames
  180–239. The scaled fit runs only if the curve, extrapolated, reaches
  the hull's LPIPS on R at P01_107's rate within 45 minutes per item.
  Every stage is at most 45 minutes and checkpointed.
- Hypothesis. It fails. Rung 2's gap is mostly pose-to-pixel error and
  the forearm, not identity, and fitting 240 frames of one hand helps
  identity most.
- Competing explanation. Most of the gap is appearance (skin, sleeves,
  lighting), which the fit learns. Then the held-out frames improve
  steeply in the pilot.
- Budget. A ceiling of 4 GPU-hours (smoke, pilot, 3 fits), RTX 6000 Ada,
  declared at ≥ 40,000 MiB. **Waits for the user's approval:** it is
  training, and rung 2's ceiling was nearly spent.

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


### 2026-10-22 — J: experience-track re-judging (rule)
- Question (user, 2026-10-22; protocol "Two tracks"). Four routes closed on
  fidelity only. Does any of them give the same experience as the anchors
  at its own rate: realistic, the same content, as steady over time? A pass
  reopens that route on the experience track only (its next step goes to
  the user); a fail closes it on both tracks. Fidelity verdicts stand
  either way.
- Results, cheapest first. Each is judged only from outputs that exist,
  except FoundHand, whose re-run is budgeted below. Nothing else is
  regenerated under this entry.
  1. **H3e** V1 (JointControlVideo; 4 pilot items, generations and judge
     archives saved; `h3e-report.json` `49af833b…`). Its nine V1 points
     differ only in the DCVC-UF context, so the encoder may choose: the
     item's operating point is the lowest-rate V1 point that passes all
     three checks, else its lowest-rate point.
  2. **FoundHand** `gen` (K6b hands rung 2; positions 0–3 of K1f's order).
     The crops were not kept, so `gen` is re-run with rung 2's code and
     settings (50 steps, seed as recorded), keeping the composites and
     crops, on all 4 items (P09_103 included). Its rate is the item's
     keypoint rate (reference unbilled, as rung 2). `ref` and `warp` are
     judged from the same job, beside it.
  3. **H3a / H3a2** (texture transfer; mesh renders against SVT-AV1). Only
     if their stages' renders or composites are on the server. If not, the
     entry records "not re-judged: outputs not kept", and they stay
     untested on experience.
  4. **K6 `ref`** on players (K4's 8 scored windows, every 4th frame code,
     3.6 kbps for two). Its renders are recomputed if not kept: a CPU warp,
     no generator.
- Judges, per item at the method's rate, against the anchor hull at that
  rate. Anchors: DCVC-UF + G and SVT-AV1 for hands (H3c's perceptually
  tuned DCVC-UF where it exists), K6's envelope for players. Each anchor
  quantity is read on its own hull (lower convex hull of the quantity over
  log rate), linear in log rate at the method's rate, clamped to the
  anchor's range (an item outside the range is flagged).
  - **Realism.** KID (`h3b`'s Inception crops and `kid_from`, bootstrap
    95% interval) of the object's crops against a fixed reference set of
    real crops of the class from footage that neither the method nor the
    anchor was fitted on. Hands: VISOR videos outside the 34-item
    evaluation set and outside G's and G_HO's training footage, with the
    list and its sha256 recorded before any score (FoundHand has seen
    VISOR, so its overlap is recorded, as rung 2). Players: K4b's matches
    other than the window's. Pass: the method's KID ≤ the anchor's upper
    95% bound.
  - **Same content.** The object's estimator on the output against the
    same estimator on the source: WiLoR's 2D joints for hands (mean joint
    distance, % of the box, H3c2's judge), YOLO11x-pose for players (K4b.2's
    path), as a share of the box. Pass: the method's error ≤ max(the
    anchor's, 1.5 × the coded keypoints' own error against the source).
  - **Same over time.** On the region, over consecutive judged frames.
    The accepted threshold names two quantities, so each is compared in
    its own form: pass if the method's `h3b.flicker` (its temporal residual
    minus the source's) ≤ the anchor's, **or** the method's own residual
    energy (mean absolute motion-compensated luma change, `g5c.flicker`'s
    energies) ≤ 1.25 × the source's. *This reading is the drafting
    session's; the user may change it before the first score.* Identity
    drift along a track is reported beside it and does not decide.
- Decision rule, per result (fixed before any score). An item passes when
  all three checks pass. **Pass**: at least 75% of the result's scored
  items pass (3 of 4 for H3e and FoundHand; 6 of 8 for K6 `ref`). **Fail**
  otherwise. Items in a judge's or generator's training footage are
  reported apart beside the pooled share. Each check's per-item pass rate
  is reported too, so a fail names the check that fails.
- Hypothesis. All four fail. H3e on same content (V1's pose error 1.8–6.5%
  of the box, where the pixel hull is at 0.3–1 kbps for that error) and on
  time (flicker 5–23 against DCVC-UF's 3–7); FoundHand on time (its change
  19.7 against the source's 14.4 on P06_108) and on content; K6 `ref` on
  content (it draws 32–46% of R from a fixed reference).
- Competing explanation. The thresholds are lenient where the anchor is
  poor: at 2–4 kbps the pixel anchors blur hands and limbs, so their pose
  error and KID are high and a generator clears them while looking wrong.
  The visual check of one frame per item is reported beside each pass for
  that reason. Or the reference set differs in domain (lighting, kitchens)
  enough that KID ranks every method alike.
- Budget. CPU: the reference set, the hulls, the reading of saved
  archives. GPU: Inception and WiLoR or YOLO on saved outputs, ≤ 1 GPU-hour
  for H3e, H3a/H3a2 and K6 `ref` together; FoundHand's re-run ≤ 3
  GPU-hours (rung 2 took about 0.45 GPU-hours per item at 50 steps on
  Ada, plus its failed attempts), declared at 40,000 MiB, which rung 2's
  scoring needed. Each stage ≤ 45 minutes.

### 2026-10-22 — K6b (hands) 2c: a buildable bank, chosen outside the window and billed (rule)
- Question ([PLAN K6b](../PLAN.md), hands, after 2b). Rung 2b's free,
  in-window bank reached the hull on 2 of 3 items with 16 references and
  improved steeply with K. Does a bank a codec could send keep that: chosen
  from the video **outside** the scored window, coded, and billed once per
  video, with the rate amortized over the video?
- Items. Positions 0–3 of K1f's content-blind order: P09_103, P25_09,
  P22_107 (G's training footage) and P06_108, 240 frames each, with K1f's
  rows, R, bound and hull (nothing in the window is re-coded). All four are
  planned. If P09_103 cannot run for infrastructure reasons, the rule reads
  the other three, as 2b did.
- Candidates for the bank (encoder side). VISOR's dense masks (the
  `visor_dense` interpolations; B1b's fill exists only inside windows) on
  every frame of the item's video outside the window and outside a 1 s guard
  on each side, at a stride of 5 frames, exact frames only on P25_09 (59.94
  fps, `visor.frame_alignment`). Per side with a mask whose box has both
  sides ≥ 16 px: WiLoR's two-pass hand-box fit (K1e's `mask_box_rows`,
  unchanged) and its 2D joints. A candidate is kept when at least 15 of its
  21 joints fall inside its mask dilated by 10% of the mask box's longer
  side (drops fits off the hand). This is why rung 2c is its own task: H2
  and K1f fitted the windows only.
- Selection. Per side, greedy k-medoids on 2b's pose vector
  (wrist-centred, scaled by extent): each step adds the candidate that most
  lowers the mean distance from every candidate to its nearest reference. It
  is nested, so the bank of 16 is the first 16 of the bank of 64, and it
  weights the poses the video shows often, where 2b's farthest-point
  sampling (one window, every pose scored) chased the extremes. K ∈ {16, 32,
  64} per side.
- What is sent per reference, and billed: the side mask's box (padded by 8
  px, even sides ≥ 64 px) as one SVT-AV1 still (CRF 40, preset 4, 4:2:0 full
  range; the box outside the mask filled smoothly first, so the background
  costs few bits), its mask as polygons (OpenCV contours simplified at
  1.5 px, coordinates in the box at ⌈log2 side⌉ bits each, plus a count per
  polygon), its 21 joints in the box at the same precision, the box's
  position (4 × 11 bits) and side (1 bit). Payload bits only (the AV1 OBUs,
  as `svtav1` counts them). The decoder uses the decoded pixels and the
  decoded polygon mask. CRF 40 is fixed now, before any run, from a probe of
  two sparse-frame hand crops (P06_108: CRF 35 0.18–0.36 bpp, CRF 45
  0.11–0.22 bpp). A lossless mask (bilevel PNG, about 750 bytes) cost as
  much as the image, hence polygons.
- Decoder, per window frame and fitted side (K1f's rows): the reference
  whose sent joints are nearest the frame's coded joints in pose (2b's
  matching), warped from its joints to the coded ones with `k6.draw`, inside
  R (granted, as K1f and 2b), Telea fill where nothing is drawn, objects
  kept.
- Rate. Item rate = K1f's keypoint rate for the item + the bank's bits
  (both sides, K references each) / the video's duration (ffprobe's, the
  whole video). Reported beside: the rate with the bank amortized over one
  minute, and the payback per metric, the shortest video for which the
  bank's quality still sits below the hull (T* = bank bits / (hull's rate
  at the bank's quality − keypoint rate); an upper bound where the bank
  beats every hull point; none where it never does).
- Variants. `bank2c@K` (coded, billed: the decision) and `free2c@K` (the
  same references uncoded and unbilled: separates the coding and billing
  from the move outside the window), K ∈ {16, 32, 64}, and `warp` (K1f's
  bound, the check: within 0.002 of K1f's LPIPS on R, or nothing is read).
- Decision rule (`k6bh.BANK2C_DECISION`, fixed before any run; the user's
  criterion). Per K, an item is won on a metric when `bank2c@K` on R is
  below K1f's hull at the item's rate. A rate above the hull's highest point
  counts as lost (the hull is not known there). Score = mean over {LPIPS,
  DISTS} of the share of items won.
  - **Pass** (some K scores ≥ 75%): hands have a buildable renderer at the
    keypoint rate plus the bank. Rung 3 is not needed on fidelity. Next: the
    passing K on K1f's other held-out items, then the component.
  - **Fail** with `free2c` at some K ≥ 75%: the bank's bits are what fail.
    Next goes to the user (a cheaper bank: lower resolution, one stream).
  - **Fail** with `free2c` failing too: references from elsewhere in the
    video do not carry the hand well enough. With 2b, the reference route
    closes at a buildable rate, and rung 3 is the last open rung for hands
    (it waits for the user).
- Hypothesis. Fails, with the score at 50% or below at every K. The bank's
  rate (estimated 1–9 kbps at K = 16–64) lifts the bar well above 2b's,
  and references from other activities in the video differ in lighting and
  sleeves. `free2c@64` comes near 2b's `bank@16`.
- Competing explanation. The same person in the same kitchen looks alike
  across the video, and 64 references from thousands of candidates sit
  nearer in pose than 2b's 16 from one window. Then `free2c` beats 2b's
  `bank@16` and `bank2c@16` or `@32` passes, with the bank under 3 kbps.
  The share of references within 10 s of the window, and of window frames
  matched to them, are reported, to tell a near-duplicate bank apart.
- Budget. One job per item, ≤ 45 min each, declared at 44,000 MiB (2b's
  scoring peaked at 38,926 on P09_103, here with seven scorings). Each job's
  smoke (≤ 600 s) runs 24 window frames and candidates at a stride of 200 on
  its own item. Estimate: about 5 min of decoding and WiLoR, then 7 scorings
  (about 150 s each in 2b) per item, so about 30 min, and about 2.6
  GPU-hours with the smokes. Ceiling 3.5 GPU-hours, failed attempts
  included. If it runs out, the items finished in the fixed order are
  reported, and so is the shortfall.
- Revision (2026-10-22, before any run; infrastructure only). The first job,
  `20261022T122902Z-6db2cc7e` (P06_108), stayed pending: other users held
  15–20 GB on every Ada GPU, so none had 44 GB free. It was cancelled before
  it started. The 38,926 MiB was rung 2's, with FoundHand resident, and the
  model–GPU table records 14.2 GiB for WiLoR, LPIPS and DISTS together. So
  PyTorch is capped at 27 GiB (`set_per_process_memory_fraction`, with
  expandable segments: the allocator frees its cache before growing, and a
  real need fails loudly), 30,000 MiB is declared, and the peaks are
  recorded in `k6bh2c.json`. Nothing in the rule changes.
- Revision 2 (2026-10-22, before any score was read; infrastructure). Job
  `20261022T125017Z-174bee4e` (P06_108, `ec72be4`) passed its smoke (28
  checks), but in the full stage the candidates alone took 33 min: 3,121
  frames at stride 5 and 5,361 fitted hands, with the fits waiting for the
  whole decode (about 17,000 1080p HEVC frames). Bank, render and seven
  scorings could not fit the same 45 min. So each item runs as two stages:
  **extract** (bank frames decoded in chunks of 256 in fresh processes while
  WiLoR fits the earlier ones; publishes the candidates unit) and **score**
  (restores it, decodes only the chosen references again, then codes,
  renders and scores). The stride is 10, not 5: about 1,500 candidates per
  video, still over 20 times K = 64. A stride-5 unit is cut to stride 10 on
  restore (the frames whose VISOR number is a multiple of 10, with their
  fits, each fitted on its own), so P06_108's 33 min are kept if that stage's
  `partial.tar` holds them.
- Revision 3 (2026-10-22, validator only, before P22_107 was scored).
  P22_107's score smoke `20261022T135926Z-ea68c004` failed one check: one
  of its 128 references (right hand, rank 55) decodes its polygon mask at
  IoU 0.874 against the source mask, under the check's 0.9 (outer contours
  only, so a hole is filled). That is lossy coding, not a fault: the
  decoder warps with the decoded mask, and its bits are billed. The check,
  meant to catch broken polygon coding, is now median IoU ≥ 0.95 and every
  reference ≥ 0.8. The three items already scored pass it (37 of 37 checks
  each).
- Jobs (RTX 6000 Ada, shared, gpu5 and gpu6; code from each `ready.json`).
  Extract: P09_103 `20261022T133541Z-972661c4`, P25_09
  `20261022T133548Z-9c159ae6`, P22_107 `20261022T133554Z-69d44a0b` (all
  `297927f`); P06_108's stride-5 fits from `20261022T125017Z-174bee4e`
  (`ec72be4`, its `partial.tar` `6ba9ac36…349b`, cut to stride 10). Score:
  P06_108 `20261022T134036Z-c2a4a791`, P09_103 `20261022T134327Z-87c2fe46`,
  P25_09 `20261022T140213Z-6e5dedbc` (all `4f2cb8c`), P22_107
  `20261022T172554Z-d766d9e1` (`83aa02d`; the validator change only). Every
  smoke that ran to the end passed the fleet validator. Every full stage
  passes `k6bh2c validate` locally (11 checks per extract, 37 per score), and
  `warp` reproduces K1f on every item. Candidates (stride 10): 529–1,723
  frames per video, 870–2,529 kept fits. References: SVT-AV1 at CRF 40,
  median PSNR on the mask 37–40 dB, median 10–14 kbit per reference with
  its polygons and joints, median polygon IoU 0.99. Peak GPU memory
  reserved 4.9 GB (P06_108). Report
  `visor/k6bh-2026-10-22/report/k6bh2c-report.json` (`3e46782b…c0b`). GPU
  time: 9,269 s of stages, failed and timed-out attempts included (P06_108's
  first full stage counted at its 2,700 s), about 2.6 GPU-hours of the 3.5.
- Outcome. On R, at each item's rate (keypoints plus the bank amortized over
  the whole video), against K1f's hull there:

  | item (video s) | kp kbps | `bank2c@16` | `bank2c@32` | `bank2c@64` | `free2c@64` (free) | `warp` |
  |---|---:|---:|---:|---:|---:|---:|
  | P09_103 (744) | 2.03 | 0.334 / 0.256 at 2.56 kbps (hull 0.197 / 0.213) | 0.330 / 0.253 at 3.16 (0.186 / 0.204) | 0.331 / 0.251 at 4.43 (0.163 / 0.186) | 0.320 / 0.238 (0.206 / 0.220) | 0.100 / 0.102 |
  | P25_09 (876) | 2.14 | 0.500 / 0.334 at 2.48 (0.249 / 0.237) | 0.489 / 0.327 at 2.86 (0.237 / 0.234) | 0.483 / 0.325 at 3.60 (0.233 / 0.233) | 0.467 / 0.302 (0.260 / 0.240) | 0.097 / 0.102 |
  | P22_107 (581, G) | 1.93 | 0.262 / **0.169** at 2.62 (0.221 / 0.197) | 0.237 / **0.156** at 3.31 (0.206 / 0.191) | 0.233 / **0.153** at 4.48 (0.179 / 0.180) | **0.208 / 0.134** (0.236 / 0.204) | 0.059 / 0.049 |
  | P06_108 (729) | 1.94 | 0.374 / 0.300 at 2.53 (0.161 / 0.199) | 0.358 / 0.289 at 3.09 (0.150 / 0.194) | 0.364 / 0.290 at 4.30 (0.137 / 0.187) | 0.366 / 0.286 (0.172 / 0.206) | 0.091 / 0.123 |

  (LPIPS / DISTS on R; bold: below the hull. G: in G's training footage.)
  Scores: `bank2c@16`, `@32` and `@64` all 12.5% (LPIPS 0 of 4, DISTS 1 of
  4); `free2c` 25% at every K. Clean of G (3 items): 0% for every variant
  but `warp`. The bank costs 0.3–2.5 kbps (0.29–1.78 Mbit), and on P22_107's
  DISTS it pays back after 109–234 s of video.
- Decision (by the rule): **fail, and `free2c` fails too.** A bank chosen
  outside the window, at any K up to 64, does not keep K1f's win: the
  references from elsewhere in the video do not carry the hand well enough,
  coded or not. With 2b, the reference route for egocentric hands closes
  at a buildable rate. Rung 3 (a generator fitted per clip) is the last
  open rung for hands on fidelity, and it waits for the user.
- Reading.
  - Coding and billing are not what fail. `free2c` is within 0.02 LPIPS of
    `bank2c` at the same K, and the bank adds at most 2.5 kbps.
  - K barely matters. From 16 to 64 references the mean pose distance falls
    by 15–27% on every item, but LPIPS on R moves by 0.00–0.03. In 2b, from 1
    to 16 in-window references, it fell by 45–56%. So the nearest pose is
    not what is missing. What is missing is the rest of the hand's
    appearance: the forearm's position and extent, the sleeve, the lighting
    and the hand's place in the frame.
  - The warped references cover 46–91% of R (K1f's own pixels: 95–97%),
    and the order follows the result: P22_107 0.83–0.91, P06_108 0.79,
    P09_103 0.72–0.76, P25_09 0.46–0.49. Where a reference's mask does not
    reach R's shape, Telea fills the rest. Against 2b's in-window `bank@16`
    (free), the gap is 0.10 to 0.23 LPIPS: P22_107 0.100 → 0.208, P25_09
    0.190 → 0.467, P06_108 0.227 → 0.366.
  - It is not a near-duplicate effect either way. On P09_103 and P22_107,
    50–72% of the matches use references within 10 s of the window, and
    they still lose. On P06_108 and P25_09, 0–19% do.
- Hypothesis ("fails, score ≤ 50% at every K; the bank's rate lifts the
  bar"): the fail held, at 12.5%. The stated reason did not: the bar moves
  little (0.3–2.5 kbps), and the free bank fails as well. The competing
  explanation (the same person and kitchen look alike across the video)
  is refuted: `free2c` is far above 2b's in-window bank on every item.
- Scope and caveats. 4 items (positions 0–3 of K1f's order), one of them in
  G's footage. One reference quality (CRF 40), but `free2c` bounds every
  quality. Candidates come only from VISOR's dense spans, at stride 10, with
  VISOR's dense masks (480p polygons) as the references' mattes, where the
  window uses B1b's fill. Matching is by the 2D pose shape alone (2b's
  matcher), not by position or scale in the frame. R is granted, as in K1f.
- Report (2026-10-22, to the root session).
  - Stopped because: goal reached.
  - Decision by the rule: fail, with `free2c` failing too. The reference
    route for egocentric hands closes at a buildable rate.
  - Numbers: score 12.5% for `bank2c@16`, `@32` and `@64`, 25% for
    `free2c` at every K, 0% clean of G. Table above. Jobs above, code
    `297927f` (extract), `4f2cb8c` and `83aa02d` (score), report
    `visor/k6bh-2026-10-22/report/k6bh2c-report.json` `3e46782b…c0b`.
  - Budget: about 2.6 GPU-hours of 3.5.
  - Deviations: stride 10, not 5, and two stages per item (time); the GPU
    memory cap; one validator threshold (one lossy mask). All recorded
    above before the affected scores were read. None changes the rule.
  - Open for the user: whether to approve rung 3 (overfit oracle, 4
    GPU-hours), the last open rung for hands on fidelity.
  - Proposed next step: rung 3 as written, or close hands on fidelity and
    leave them to the experience track (item J).

### 2026-10-22 — K6b (hands) 3: overfit oracle (run)
- Approved (user, 2026-10-22: "Rung 3 approved: run the overfit oracle as
  written, within 4 GPU-hours"), after rung 2c failed. The rule is the
  entry "K6b (hands) 3: overfit oracle (rule, not run)", unchanged: the same
  3 items, judges, hull and criterion; P01_107 for the smoke and the pilot.
- Fit (`experiments/visor/k6bh3.py`, written before any run). FoundHand's
  EMA weights, the full DiT fine-tuned per item: AdamW, learning rate 1e-5
  (the released checkpoint's), batch 8, bf16 autocast over float32 weights,
  FoundHand's own `training_losses` (MSE on the noise plus its learned
  variance term, 1,000 steps). Pairs: a target crop and another crop of the
  same hand from the fitting frames, both with K1f's uncoded keypoints and
  rung 2's crop rule. The reference's mask is dropped half the time (the
  video procedure's later references carry none), and 10% of pairs train
  the sampler's unconditional branch (zero conditions, `nvs` 2). The
  sin-cos position embedding stays frozen. The fitted weights are sampled
  directly (no EMA over a fit this short). Judged as rung 2's `gen` (50
  steps, guidance 2.5, from the window's first fitted frame, on the coded
  keypoints).
- Smoke checks (the rule's): at least 200 steps; the loss on a fixed probe
  (16 pairs, fixed timesteps and noise) falls; a sample of the probe's
  training pairs moves towards its target (LPIPS of the crops falls).
- Deviations, before any run:
  - The pilot judges the held-out frames 180–209 (30, not 60) at 25%, 50%
    and 100% of the 30-minute fit (0% keeps only the probes). By rung 2's
    timing (about 2.5 s per generated hand), four judgements of 60 frames
    plus the fit would not fit a 45-minute stage.
  - Both hands' chains are sampled in lockstep (one batch), so the video
    procedure's per-step random choice of reference is shared by the two
    hands.
  - Declared at 30,000 MiB with PyTorch capped at 27 GiB, not ≥ 40,000,
    as in 2c. Other users hold 15–20 GB on every Ada GPU.
  - The scaled fits' length is the pilot's extrapolated time to the hull,
    at most 45 minutes. If that and the judging of 240 frames do not fit one
    stage, fitting and judging are split, with the weights published
    between them.
- P01_107's bar: K1f's hull at its rate, from the reproduction stage
  `20261022T013208Z-9e048ecf` (K1c recorded 0.258 LPIPS and 0.223 DISTS).
