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

### B1. VISOR adapter and evaluation set (done 2026-10-07)

Result: `src/segmentation/visor.py`, the verified frame rules, and evaluation
set v2 (34 windows). How every later step uses VISOR:
[Using VISOR](docs/resources.md#using-visor). The text below is the original
brief.

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

### B2b. Baseline follow-ups (agreed 2026-10-07, after B2)

B2's baselines made fair and stronger before anything is compared with them.
Each item gets an experiments.md entry before its fleet run.
1. **Scoring on the GPU for every codec.** SVT-AV1 jobs encode on the host's
   CPU but claim a GPU for scoring (B2 scored them on CPU: 227 s per point
   against 6.5 s). B2's CPU scores stay valid; they are not redone.
2. **Clean timing, then an equal time budget.** Time DCVC-UF's encode and
   decode as upstream does (CUDA events around the model calls), apart from
   the worker's verification (second decode, per-frame hashes) and its CPU
   4:2:0 conversion, which move out of the timed region. Time SVT-AV1 per
   preset on the same host class, isolated, with a fixed thread count. Then
   SVT-AV1 gets the slowest preset whose encode time per window fits DCVC-UF's
   on the same host, and that preset replaces preset 4 if it differs. CRF only
   sets the rate. SVT-AV1 has no GPU path; NVENC AV1 (Ada only) is a separate
   hardware encoder, at most an extra reference point.
3. **Lower rates by resolution.** CRF stops at 70, so SVT-AV1 also encodes
   720p, 540p and 360p versions, upscaled to 1080p by a fixed filter before
   scoring; each item's curve is the convex hull over resolutions. DCVC-UF
   gets the same ladder if its QP 0 floor is not low enough.
4. **ROI SVT-AV1.** `--roi-map-file` lowers the quantizer inside the VISOR
   foreground (an oracle: the dataset's masks, not a segmenter's). A pilot
   picks the offset. A segmenter-mask variant follows D1.
5. **DCVC-UF full-range check.** One pilot item coded with limited-range
   input, scored against the same reference.
6. **Secondary studies (CPU).** Spatial and temporal complexity per frame and
   inside the foreground (VCA v2, `/opt/local/bin/vca`; ffmpeg's `siti` as a
   cross-check) against each item's DCVC-UF advantage; SVT-AV1 with temporal
   filtering off (`--enable-tf 0`) on a few items.
7. **Why SVT-AV1's ROI cannot raise a region's quality** (deferred; a
   possible secondary contribution, after the main path). In 4.2.0's CRF
   mode a negative quantizer offset adds bytes without raising the region's
   PSNR, while a positive one lowers it as expected (B2b J3, J3b, probe).
   Hypothesis: the encoder derives its rate-distortion trade-off (lambda)
   from the frame's quantizer, not the segment's, so a finer segment
   quantizer buys coefficients that rate-distortion optimization does not
   turn into quality. First read SVT-AV1's ROI/segmentation and lambda code
   (minutes); then one bounded probe that tests what the code suggests (e.g.
   offsets with a matched lambda, or another encoder's ROI); write it up only
   if the mechanism is confirmed.
**Done when** the B2 curves are redrawn with the fair SVT-AV1 preset, the
low-rate ladder and the ROI variant, from recorded jobs.

Status (2026-10-07): items 1, 2 (timing), 5 and 6 are done, and item 4's pilot
chose the ROI offset (+32 on background blocks; lowering the foreground's
quantizer does nothing in SVT-AV1's CRF mode)
([outcome](docs/experiments.md#2026-10-07--b2b-fair-baselines-timing-equal-time-roi-range-complexity)).
DCVC-UF HT-L encodes a window in 1.9 s, faster than any SVT-AV1 preset on 32
cores. By the user's decision the equal-time preset is matched to
PointStream's encode time instead, so the equal-time re-encode, the full ROI
run and the low-rate ladder wait for PointStream's first timing and rates, and
then run together from J2's time-per-preset table.

### B1b. Filling VISOR's missing hands and objects (done 2026-10-08)

Result: by the rule fixed beforehand, the SAM 3.1 fill is adopted for objects
and not for hands (held out: objects mean J 0.83, hard subset 0.70; hands
0.93 but only 4 of 6 departed hands released). Mask set
`visor_dense_sam_fill` (objects only) for evaluation set v2: the labelled
objects missing fall from 33% to 1.6% of each frame's
([outcome](docs/experiments.md#2026-10-07--b1b-sam-31-fill-of-visors-missing-hands-and-objects)).
The review by eye can still veto. Open: train-split filling for E1/F1 (hands
need a better release rule first). The text below is the original brief.

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

### B2. Baseline rate-distortion on VISOR (done 2026-10-07)

Result: `experiments/visor/b2.py`; SVT-AV1 preset 4 (CRF 41–62) and DCVC-UF
HT-L (QP 27–63) on all 34 items, from recorded jobs
([outcome](docs/experiments.md#2026-10-07--b2-baseline-rate-distortion-on-visor)).
DCVC-UF HT-L needs 13% less rate (mean BD-rate; median 18%) for equal weighted
PSNR, ties on whole-frame PSNR, and loses on VMAF. B1b's fill is added by
rescoring the published streams. The text below is the original brief.

SVT-AV1 and DCVC-UF on the evaluation set: rate against weighted PSNR
(0.7 foreground + 0.3 background on VISOR masks) and the perceptual metrics.
This is the target PointStream must beat, and it needs no PointStream
component. Use evaluation set v2 as [Using VISOR](docs/resources.md#using-visor)
describes. Only exactly placed frames are scored
([drift decision](docs/experiments.md#2026-10-06-visor-frame-drift-exact-frames-for-evaluation-small-drift-for-training)):
the B1 set, optionally EK-55 clips scored on frames whose keyframes match the
released JPEGs.
**Done when** the curves come from a recorded job.

### D1. Segmentation benchmark on VISOR and EgoHOS

SAM 3.1 (text and prompted) and YOLOE-26 against the labels, per class, accuracy
against speed. EgoHOS is scored on single images only. On VISOR: evaluation set
v2, the "hold the first frame" floor (foreground J 0.459) as the table's first
row, and every number with and without B1b's fill
([Using VISOR](docs/resources.md#using-visor)).
**Done when** the table is reproducible from a recorded job.

### E1. Handled-object proposer

A hand-object proposer trained on VISOR (contact relations) or EgoHOS (object
orders) prompts SAM 3.1. Cross-tested on the other dataset and on HOT3D object
masks. VISOR training follows [Using VISOR](docs/resources.md#using-visor):
train split only, drift ≤ 1, no frames with a hand gap as targets.
**Done when** cross-dataset numbers decide whether to commit to it.

### F1. Training-data export

VISOR foreground crops and masks per instance, plus background frames with the
foreground removed, written by `python -m src.segmentation dataset` with
provenance and each mask's drift; training admits drift of at most 1 frame.
Background frames skip frames with a hand gap until B1b fills them
([Using VISOR](docs/resources.md#using-visor)).

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
