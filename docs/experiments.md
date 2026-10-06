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
- Jobs:
- Outcome:
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
