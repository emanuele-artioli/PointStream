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
