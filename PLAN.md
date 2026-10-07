# Plan

Each workstream is scoped to one session. Work runs in two phases
([decision](docs/experiments.md#2026-10-06-egocentric-first-racket-sports-second)):

- **Phase 1, egocentric hand-object video.** VISOR is the only adopted dataset
  with native video and dense human masks of the whole foreground: hands with
  forearms, and active objects. It carries the end-to-end codec evaluation.
- **Phase 2, racket sports**, once phase 1 has a result. No racket dataset has
  dense foreground masks, ball positions and video together. Players and the
  ball are evaluated in separate stages, each on the dataset that labels it.
  Together the stages estimate the performance on a fully labelled dataset.
  If phase 2 is not finished by submission, the paper presents it as future
  work.

Every dataset is used for what it labels ([resources](docs/resources.md#what-each-dataset-is-used-for)).
The environment audit (step 0) comes before the first component that needs a
new dependency.

## Done

- **A. Data acquisition and archive** (2026-10-06). All six datasets are under
  `Datasets`, with immutable manifests and smoke reads in `Datasets/manifests/`.
  The pre-reset data is in `Datasets/archive/pre-reset-2026-10-05/` with its
  own manifest. Tooling: `tools/datasets/`.
- **0. Environment audit, first wave** (2026-10-06). One environment for
  SAM 3.1, YOLOE-26, the VISOR and HOT3D-Clips readers, HaMeR and WiLoR,
  SVT-AV1 and DCVC-UF (the chosen neural codec), with the conflicts, locks and
  packed archive in [resources](docs/resources.md#environments) and per-GPU
  results in the [model–GPU table](docs/fleet.md#modelgpu-table). Open for H:
  HaMeR or WiLoR, decided on HInt test and HOT3D.

## Phase 1: egocentric

### B1. VISOR adapter and evaluation set

A VISOR reader in `src/segmentation` producing `ClipMasks` from the dense
interpolations. Frames come from the video through `frame_mapping.json`, which
names EPIC rgb frames, not decoded frames: on P32_07 rgb frame k is the frame at
(k − 1)/60 s, and `k − 1` as an index is up to 2 frames wrong
([resources](docs/resources.md#datasets), `experiments/audit/env_smoke.py`
`epic_frame_to_video_index`). B1 verifies the rule on more videos, including
the 50 fps EK-100 ones, against the released sparse JPEGs, and records it. It records native classes (left hand, right hand, active
objects), per-frame labelled flags, and a provenance tier per mask: `human` for
the sparse ground truth, `interpolated` for the dense frames. Frames are decoded
from the EPIC-KITCHENS videos through `frame_mapping.json`. The evaluation set
is a fixed list of dense runs from the validation split, recorded with its
sha256s.
**Done when** the set converts and a trivial candidate scores against it.

### B1b. Filling VISOR's missing hands and objects

VISOR's dense masks drop an object's track when its interpolation scored poorly,
so 22–30% of dense validation frames lack a hand a human labelled
([decision](docs/experiments.md#2026-10-07-visor-evaluation-set-v2-and-a-sam-fill-for-missing-hands)).
SAM 3.1, prompted with the human masks at a run's keyframes, tracks each
missing object through the run; the result is tier `sam_from_label_prompt`.
Validation: prompt at one keyframe and score at the next, which SAM never saw;
review a random sample of in-between frames by eye; check hands against the
EPIC-KITCHENS hand-object boxes (detector output, to be acquired and
manifested like the other datasets). Every B2 and D1 result is reported with
and without the fill. It also removes unlabelled hands from F1's background
frames and E1's training targets.
**Done when** the validation numbers from a recorded job decide adoption.

### B2. Baseline rate-distortion on VISOR

SVT-AV1 and DCVC-UF on the evaluation set: rate against weighted PSNR
(0.7 foreground + 0.3 background on VISOR masks) and the perceptual metrics.
This is the target PointStream must beat, and it needs no PointStream
component. Only exactly placed frames are scored
([drift decision](docs/experiments.md#2026-10-06-visor-frame-drift-exact-frames-for-evaluation-small-drift-for-training)):
the B1 set, optionally EK-55 clips scored on frames whose keyframes match the
released JPEGs.
**Done when** the curves come from a recorded job.

### D1. Segmentation benchmark on VISOR and EgoHOS

SAM 3.1 (text and prompted) and YOLOE-26 against the labels, per class, accuracy
against speed. EgoHOS is scored on single images only.
**Done when** the table is reproducible from a recorded job.

### E1. Handled-object proposer

A hand-object proposer trained on VISOR (contact relations) or EgoHOS (object
orders) prompts SAM 3.1. Cross-tested on the other dataset and on HOT3D object
masks.
**Done when** cross-dataset numbers decide whether to commit to it.

### F1. Training-data export

VISOR foreground crops and masks per instance, plus background frames with the
foreground removed, written by `python -m src.segmentation dataset` with
provenance and each mask's drift; training admits drift of at most 1 frame.

### G. Background encoding

Design session ([components](docs/components.md#3-background)). Candidates are
DCVC-UF, a panorama/mosaic, and SVT-AV1. Egocentric video is the hard
case for a panorama: the head moves constantly and the scene is close, so there
is parallax.

### H. Foreground encoding

Design session ([components](docs/components.md#4-foreground)): an appearance
vector plus keypoints per object. Hands are evaluated twice. On HOT3D the
keypoints come from motion capture (an oracle upper bound). On VISOR they come
from a hand-pose estimator, HaMeR or WiLoR (what deployment sees). HaMeR and WiLoR are compared first; no published work compares them on
egocentric video, so the result is a contribution and a paper table: 2D PCK on HInt
VISOR test (and New Days), 3D error against HOT3D motion capture, speed, and
the foreground reconstruction quality each gives. The literature does not
settle it for egocentric video ([resources](docs/resources.md#models)). The gap between the two
measures the cost of pose estimation.

## Phase 2: racket sports

### C. SAM 3.1 from label prompts

Players and rackets from RacketVision boxes and keypoints; the ball from point
prompts. OpenTTGames masks check the objects: whether SAM finds the right
players, table and scoreboard. Their 320×128 resolution cannot grade borders.

### Players stage

Foreground and background encoding with players as the foreground, scored on
OpenTTGames. Its camera is fixed, so this is the favourable case for a
panorama background.

### Ball stage

Ball tracker trained or fine-tuned on RacketVision, cross-tested on OpenTTGames
and TrackNet (the only dense ball labels). A parametric ball trajectory is the
candidate encoding.

### D2. Segmentation benchmark on racket sports

As D1, per dataset and class, with the ball scored by point metrics.

## Throughout

### I. Paper

Rescope with a fresh paper repository state: egocentric first, racket sports as
the second domain or as future work. Write the dataset and evaluation-protocol
sections (weighted PSNR on dataset masks, provenance tiers) and the hand-pose
estimator comparison from H. Set the venue and
submission date.
