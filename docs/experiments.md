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
  best match is below 3. Rules: VISOR n − 1, the time rule on VISOR n, the EPIC
  time rule (extraction at the nominal integer rate) through
  `frame_mapping.json`, and EPIC k − 1. (2) *evaluation set*. Items come only
  from validation videos whose class satisfies both VISOR n − 1 and the EPIC
  time rule, from runs whose first 240 frames lie exactly on the video (no drift
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
  smoke 2 frames per video ≤ 300 s, full 40 per video ≤ 1,200 s, 8 threads.
  Job 2 (evaluation set): smoke 2 items × 24 frames ≤ 300 s, full all items ×
  240 frames ≤ 1,800 s, 16 threads. Each budget ≤ 1.5 h including staging the
  4.9 GB environment and 2–4 GB of inputs. Ceiling 3 h wall, no GPU hours.
- Inputs: `pointstream-data/visor/b1-2026-10-06/mapping-v2/`
  (`visor-mapping-check.tar` sha256 `4e32ce13…e0c0`, record
  `visor-mapping-check.json` `6ecee54e…c289`, videos hard-linked under
  `inputs/video_<id>/`); environment `pointstream-20261006T113321Z`.
