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
  (22–30% of dense val frames; [B1 run](experiments/b1.md#2026-10-06--b1-visor-frame-mapping-and-evaluation-set)).
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

Run entries live in [experiments/](experiments/), one file per PLAN item,
named by its code (`h5p.md`, `k6b-hands.md`, `environment-audit.md`), oldest
entry first. To start an item, create its file with the header the others
use; to add an entry, append it to the item's file. Parallel sessions then
touch different files and do not conflict.

Find an entry by its code or date:

```bash
grep -n "^### " docs/experiments/*.md          # every entry, by file
grep -ln "K4b" docs/experiments/*.md            # files that mention K4b
```

Entries written before 2026-10-24 were moved here unchanged from this file;
their anchors are the same, under the new file name.
