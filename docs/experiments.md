# Experiments

## Protocol

GPU time and training are the constraint; code is cheap. Rank work by value to
the paper first and the demo second.

1. **Write it down first.** Add an entry below with the decision rule, the
   hypothesis, the competing explanation and the budget (GPU hours and wall
   time).
2. **Correctness smoke, in minutes.** The real component on a representative
   bounded input, through the same entrypoint as the full run. On a model's
   first run on a GPU class, the smoke also asserts device, execution provider
   and kernels ([fleet](fleet.md#modelgpu-table)).
3. **Bounded pilot on the one axis that matters.**
4. **Scaled run**, only after the pilot shows it can meet the target within
   budget.
5. **Stop when the decision is made.** Record the outcome below, including
   negative results.

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
