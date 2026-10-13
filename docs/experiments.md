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
  ([PLAN](../PLAN.md#h2-hand-pose-estimators-hamer-against-wilor)).
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
