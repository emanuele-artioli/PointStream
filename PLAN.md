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
The review by eye (5 of 68 frames flagged, all SAM hand tracks losing the
forearm or the hand) does not veto it. No train-split hand fill: excluding
the frames with a hand gap keeps 75% of the admitted train frames (1.6
million), enough for E1/F1. The text below is the original brief.

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

### G. Background (user, 2026-10-08)

The thesis: a viewer barely looks at the background, and most of it repeats.
After a warm-up stretch of video, the client receives one representation of
everything the camera has seen; from then on each background frame is
rebuilt from it with only the camera's pose (GenStream did this with a
hand-made 3D model; PointStream builds the representation from the video
itself). With a camera that rotates in place there is no parallax, so the
representation is a panorama (planar, cylindrical or spherical) rather than
3D Gaussian splatting. A second candidate stores the warm-up background in
the weights of a neural codec and then sends one latent per view. The
foreground occludes the background: a panorama merges views and fills what
one frame hides; a neural model needs clean frames, from the panorama or from
generative inpainting. PRESLEY's lesson stands: the background only has to be
good enough not to distract. Each step below is one session; its outcome
decides the next.

#### G1. Camera motion and coverage audit (next)

Does the camera only rotate, and how fast does the background stop being new?
Per dataset and clip, with the foreground masked out: fit frame-to-frame and
frame-to-reference homographies (rotation and zoom about a fixed centre) and
measure what they leave unexplained (alignment error on background pixels,
the share of frames a homography explains, residual motion that indicates
parallax or moving background such as water, screens or other people); and
the coverage curve, the share of each frame's background already seen after
t seconds. VISOR (evaluation set v2 and longer dense runs; head-mounted, close
scene, so translation and parallax are expected) and the racket-sports sets
(fixed or pan-tilt-zoom cameras). CPU or light GPU.
**Done when** a recorded job says, per dataset, whether a rotation-only
background holds, how long a warm-up it needs, and which clips a background
evaluation can use; and whether phase 1's dataset suits it or the background
work should start on racket sports.

#### G2. Background evaluation protocol

From G1: the clip set (long enough for a warm-up to pay off), the warm-up
length, rate accounting (the one-time representation counted once, plus the
per-frame camera pose and any residual, reported against how long the clip
plays after the warm-up), quality on visible background only (dataset masks;
occluded pixels have no truth), and render time per frame at the client.
Baselines on the same background frames: SVT-AV1 and DCVC-UF, plus
PRESLEY-style degradation (VVC is an extension, after a working pipeline).
**Done when** the protocol is fixed in experiments.md and the baselines'
background curves come from recorded jobs.

#### G3. Panorama background

Build the panorama from the warm-up frames with the foreground masked out
(views merged so that what one frame hides another fills; G1 picks the
projection), code it once, send each frame's camera parameters, and render the
view at the client. Measured by G2's protocol: rate over clip length, quality
on visible background, render time, and the failure cases (residual motion,
lighting changes, background not seen during the warm-up).
**Done when** its curves and render time sit beside G2's baselines.

#### G4. Clean background frames

The neural route needs frames without the foreground: panorama-filled frames
from G3 against generative video inpainting (to be audited into the
environment), compared on the visible pixels around the hole and by eye.
**Done when** one source of clean frames is chosen for G5, with numbers.

#### G1d. Depth-aware warps on VISOR (done 2026-10-09)

After G1 ruled out a rotation-only panorama, G1d asked whether knowing depth
fixes it. On G1's pairs, where one reference serves for seconds, no
depth-aware warp brings VISOR's background under 2 px on a meaningful share
of frames. That holds for a few planes, triangulated keyframe depth, and even
dense flow on epipolar lines: median clip 34% of window frames and 9% of
stretch frames. So DA3 and depth-augmented keyframes were not run. For
egocentric video, a background sent once and reused for seconds is ruled out.
Racket sports are unaffected: G1 found their fixed cameras hold a panorama
(OpenTTGames 6 of 7 clips, with the first frame already showing 90% of every
later background). With a fresh
reference (under 0.1 s), though, the best static warp explains 70% of window
frames and four planes 49%. So the open question is the refresh rate and its
cost, not depth ([experiments](docs/experiments.md)).

#### G1e. Reference refresh (done)

A reference refreshed every 0.1 s brings VISOR's background under 2 px on
most frames: the best static warp explains 92% of window targets and 51%
of stretch targets. Four planes, the sendable warp, explain 78% and 39%.
At 1 s, neither group passes. The growth with age is real (the measurement
floor holds), and the masks are not the cause. What is left after a fresh
reference is photometric change and independent motion, not geometry. By
the rule (step 3), no sendable warp qualifies in both groups. Details in
[experiments](docs/experiments.md) (G1e entry).

Decision (user, 2026-10-09): egocentric background work moves to G5. The
rule's own next step is parked, because its best case is a narrow pass:
a sendable warp between four planes and the oracle (DA3 or depth-augmented
keyframes) at the oracle's refresh age, capped on the stretches by the
oracle's 51% at 0.1 s. A reference refreshed ten times a second is close to
ordinary video coding anyway. Parked options, with what would reopen them:

- **G1f, a better static warp at 0.1 s** (DA3 or depth-augmented keyframes,
  code in `experiments/background/{g1d,da3}.py`, never run). Reopen if G5
  fails, or if a cheaper refresh appears. It can only add parallax, which
  is 7% of what remains at 0.1 s.
- **Four planes refreshed at 0.1 s on annotated windows only** (78% of
  window targets). A G2 candidate only if the paper scopes egocentric
  results to annotated windows, which G1d's and G1e's content-blind
  stretches argue against.
- **Model the residual, not the geometry.** After a fresh reference, 63–70%
  of what remains is photometric (blur, noise, lighting, disocclusion) and
  21–27% is independent motion. A G5 model that is conditioned on the
  warped reference could target exactly this. That is the bridge from G1e
  to G5.

#### G5. Neural background model

A neural codec fitted or fine-tuned on the warm-up's clean frames, sent once
(its bytes counted), then one latent per view; DCVC-UF fine-tuning and an
implicit per-scene model (HNeRV-style, earlier rejected as a general codec)
are the starting candidates. Measured by G2's protocol against G3 and the
baselines, render time included.
**Done when** G3 and G5 are compared on rate, quality and render speed, and one
background is chosen for PointStream.

### H. Foreground (user, 2026-10-08)

The thesis: a pixel codec spends most of its bits on what moves, and people
and the objects they handle move in ways computer vision already models.
GenStream sent a skater as skeleton keypoints and rendered the person at the
client. PointStream does the same per object: an appearance reference sent
once (like the background's warm-up), then compact motion parameters per frame
(hand pose, an object's rigid motion), rendered and composited over the
background at the client. Whatever the model cannot explain (an object first
seen, food being cut, a failed estimate) falls back to coded pixels, so every
frame decodes. VISOR's hands include the forearm, which MANO does not model,
and the hands carry most of the viewer's attention. Each step is one session.
G and H are independent until H5, so their sessions can run side by side.

#### H1. Foreground motion and representation audit (done 2026-10-08)

Result: `experiments/visor/h1.py` on all 34 items of evaluation set v2, from
recorded jobs (pilot `20261008T072619Z-0d09640b`, full `…083835Z-5dafce84`,
`…103118Z-a358e4ad` (10 items, salvaged `partial.tar`), `…150618Z-fb44e5a3`;
[outcome](docs/experiments.md#2026-10-08--h1-foreground-motion-and-representation-audit),
[overlays](https://claude.ai/artifact/7ir8Smh5tyWLFDejPppj6k)). By the rule
fixed beforehand, no part is worth a parametric model. Hands (20% of the
foreground's SVT-AV1 bits) pass on bits and fit (WiLoR fits 91.5% of
hand-frames, hand-side IoU median 0.77) but fail on rate: 15.2 kbps of pose
parameters against 23.4 kbps that SVT-AV1 spends on hand pixels at CRF 62.
Forearms (12%), handled objects (27–29%) and other objects (38%) fail on fit:
a reference carried by homographies dies within 0.02–0.04 s (median). The
forearm is half of VISOR's hand masks (49.7%), so H3 renders it with the hand.
Objects, not hands, carry two thirds of the foreground's bits, most of them
large touched surfaces whose motion is the camera's (G's, not H's). Order:
H2 with pose rate after smoothing added to its stability axis; H3 hand and
forearm; H4 as briefed is unsupported and waits for the user's decision.
The text below is the original brief.

What the foreground is and how much of it a parametric model could explain.
On evaluation set v2: the foreground's share of pixels and of B2's bits
(hands, forearms, handled objects, other objects); per object, how well a
rigid 2D or 3D motion explains it frame to frame, and how much is
deformation, appearance change, occlusion and motion blur; per hand, how
well a hand-pose estimate re-projects onto the mask, and how much of the
mask is forearm. An estimate of the parameter rate per object per second.
**Done when** a recorded job says which foreground parts are worth a
parametric model, which should stay pixels, and in what order to build them.

#### H2. Hand-pose estimators: HaMeR against WiLoR

The comparison already planned, a paper table on its own: 2D PCK on HInt
VISOR test (and New Days), 3D error against HOT3D motion capture, speed, and
temporal stability on VISOR video. No published work compares them on
egocentric video ([resources](docs/resources.md#models)). HOT3D's motion
capture is the oracle that measures what pose estimation costs.

Pose coding (user, 2026-10-08; a candidate secondary contribution). H1 coded
WiLoR's MANO parameters as quantized first differences: 165 bits per
hand-frame, 3.2 bits per value, mostly the estimator's jitter (H1 outcome).
H2 measures the pose stream's rate against distortion (projected joint error,
and the rendered silhouette's IoU) and latency (frames of lookahead), as
curves, for each technique alone and in combination: temporal subsampling
with interpolation at the client (pose sent at 10–30 Hz), smoothing at the
encoder (a One-Euro filter or a small temporal model), prediction beyond the
previous frame (constant velocity), a pose subspace (MANO's PCA, 6–45
components), and the quantization step per joint. The optimal combination
per latency budget is the result; it decides H3's pose input.
**Done when** one estimator is chosen for PointStream, with numbers, and the
pose stream's rate-distortion-latency curves name the coding H3 uses.

#### H3. Hand and arm rendering

From an appearance reference and the per-frame pose (H2), render the hand and
forearm at the client: a textured MANO mesh, pose-conditioned generation, or
a mesh with a learned residual; the forearm from the mask's extent or a
simple arm model. Scored on VISOR foreground quality against coding the same
pixels with SVT-AV1 at equal rate, with HOT3D's oracle poses as the upper
bound, and on render time.
**Done when** a hand renderer beats coded pixels at some rate, or the session
shows why not.

#### H4. Handled objects (rescoped after H1, user 2026-10-08)

H1 found that no 2D warp of one reference view carries an object for more
than a few frames (median reference life 0.02–0.04 s): objects turn, show new
sides, are occluded by hands and deform. The large touched surfaces VISOR
labels active objects (pan, sink, hob, chopping board: 38% of the
foreground's bits) move with the camera; they go to G's background, and the
G section takes them up once G1 has finished. H4 covers the handled objects
(27–29% of the foreground's bits), cheapest idea first:
1. *Grip pose.* A held rigid object moves with the hand: in the wrist's frame
   it is still. While grasped, the object is sent as one fixed transform
   relative to the hand's 3D pose (already in the stream), resent only when
   the grip changes or the object is released. First measured on HOT3D,
   which has motion-capture poses of hands and objects: how constant the
   transform is during a grasp, and how often it changes.
2. *A bank of reference views.* Each object keeps a few past views; every
   frame is predicted from the nearest one (by pose, or by prediction
   error), and a new view is sent only when none predicts well enough. An
   object turned back and forth, or reappearing from under a hand, costs no
   new reference. Measured on VISOR against H1's single refreshed reference.
3. Whatever neither explains stays coded pixels.
Scored on VISOR foreground quality against SVT-AV1 on the same pixels at
equal rate, with LPIPS beside weighted PSNR.
Later candidates, once 1–3 have numbers: a per-object 3D model built from the
views seen so far (e.g. Gaussian splats) driven by a 6DoF pose, which
handles rotation; and generative completion of sides not yet seen, which
costs nothing to send and may score well on LPIPS while scoring badly on
PSNR, until the real side appears and must be corrected.
**Done when** handled objects have a representation chosen by numbers, or the
session shows they should stay pixels.

#### H5. First working PointStream

G's background and H's foreground in one bitstream with every byte counted:
references once, then camera pose, hand pose and object motion per frame,
plus the fallback pixels. Scored with B2's harness on evaluation set v2
(weighted PSNR on dataset masks, with and without B1b's fill) against B2's
curves, and timed end to end, which picks B2b's equal-time SVT-AV1 preset.
**Done when** PointStream's curves and encode and decode times on all 34
items come from recorded jobs, beside B2's.

### After a working pipeline

Optimisation, training and extensions, ordered by what G and H show they need.
VVC (VVenC, `/opt/local/bin/vvencapp` on the hosts) joins the baselines here
(user, 2026-10-08: SVT-AV1 is enough until then).

#### D1. Segmentation benchmark on VISOR and EgoHOS

Moved after a working pipeline (user, 2026-10-08): SAM 3.1 already segments
well enough to build on, and choosing a faster segmenter is an optimisation that needs the
pipeline's own timing.

SAM 3.1 (text and prompted) and YOLOE-26 against the labels, per class, accuracy
against speed. EgoHOS is scored on single images only. On VISOR: evaluation set
v2, the "hold the first frame" floor (foreground J 0.459) as the table's first
row, and every number with and without B1b's fill
([Using VISOR](docs/resources.md#using-visor)).
**Done when** the table is reproducible from a recorded job.

#### E1. Handled-object proposer

A hand-object proposer trained on VISOR (contact relations) or EgoHOS (object
orders) prompts SAM 3.1. Cross-tested on the other dataset and on HOT3D object
masks. VISOR training follows [Using VISOR](docs/resources.md#using-visor):
train split only, drift ≤ 1, no frames with a hand gap as targets.
**Done when** cross-dataset numbers decide whether to commit to it.

#### F1. Training-data export

VISOR foreground crops and masks per instance, plus background frames with the
foreground removed, written by `python -m src.segmentation dataset` with
provenance and each mask's drift; training admits drift of at most 1 frame.
Background frames and segmenter targets skip frames with a hand gap (B1b
found no hand fill fit for them; excluding keeps 75% of admitted train
frames, [Using VISOR](docs/resources.md#using-visor)).

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
