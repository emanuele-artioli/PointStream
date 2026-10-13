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

Oracle pilot on VISOR (2026-10-10, [experiments](docs/experiments.md), G5 oracle
entry). At 960×540, fitted to the scored frames, against SVT-AV1 on the visible
background: NVRC, the strongest public per-video neural codec, needs +347% and
+133% rate on two pilot clips and is at parity (+2.7%) on the third, but only at
0.7–1.5 Mbps. A model conditioned on a warped reference refreshed every 0.1 s is
ruled out (+184%, +409%; even with no model, the warped refresh costs +34% and
+103%). DCVC-UF, for comparison: −17%, +27%, +41%. By the rule NVRC is a
candidate, but after review (user, 2026-10-10) G5 moves to a scene model that is
not billed (G5b), which NVRC cannot be: it renders only the frames it was
fitted to. So NVRC's run on the other 41 clips is dropped. Its fully billed pilot
stays in the paper as the intermediate step from a per-clip neural codec to
PointStream's scene-adapted background.

#### G5b. Scene-adapted neural background (user, 2026-10-10)

The deployment PointStream targets shows one scene for a long time (a kitchen,
a court). A model of that scene is delivered to the client beforehand, and its
bytes are not billed: the rate is only what is sent at run time (foreground
appearance and keypoints, background latents and corrections). This replaces
G5's "sent once (its bytes counted)" and, for the scene model, G2's one-time
representation B_once. The paper states the assumption and reports beside it
the playback time after which the model's bytes would be repaid.

- *Held-out evaluation.* With free weights, a model fitted to the frames it is
  scored on would simply memorize them. So the model is fitted on a video's
  earlier part and scored on a held-out final segment.
- *Candidate*: DCVC-UF, its decoder and entropy model fine-tuned on the scene;
  the run-time rate is its own bitstream on the held-out segment. Its training
  code is in the pinned DCVC revision (`train_video.py`, `training.md`) and is
  audited into the environment first.
- *Clips*: long ones first, since they are the ones the paper will present.
  The 10 VISOR stretches (120 s at 10 frames/s): score G5's 24 s excerpt,
  which holds the stretch's dense-mask window (96–120 s has dense masks on
  one stretch only), and fit on the rest of the stretch less 5 s on each side.
  The 34 windows (4–4.8 s) are too short to split and are secondary.
- *Metric*: LPIPS on the visible background (foreground pasted back, as in G2)
  is the gate, and training may optimize it directly, as codecs optimize PSNR.
  PSNR on V is reported beside it.
- *Order*: LPIPS rescoring of the stored SVT-AV1 streams; DCVC-UF off the
  shelf on the held-out segments (the floor); the oracle, fine-tuned on the
  held-out segment itself (the upper bound) and on the first 96 s (the
  component), as a smoke, a two-stretch pilot (≤ 6 GPU-hours) and then the
  rest. Beside it, the cheap check of G1e's refresh at 0.3 s and 1 s (warp
  only, on the windows).

**Done when** the fine-tuned DCVC-UF's held-out curves on the stretches sit
beside SVT-AV1's and the unadapted DCVC-UF's, on LPIPS and PSNR on V, from
recorded jobs, and G5 can be compared with G3.

Pilot (2026-10-10, [experiments](docs/experiments.md), G5b entry;
[report](https://claude.ai/artifact/H9yE6iBUGWau6K1HQBLcek)). On P26_02 and
P06_03, fitted on the rest of the stretch: with β·LPIPS added to DCVC's own
distortion (β = 0.026) the scene model passes the rule, −74% against
SVT-AV1 on P06_03 and better than any SVT-AV1 point on P26_02 at equal
LPIPS_V, but +87% and +111% at equal PSNR_V; with distortion alone it never
beats SVT-AV1 (+82%, +44%). A model fine-tuned the same way on the other
kitchen gets 52–75% of the gain against the unadapted model (in log rate): the
loss teaches the decoder to rebuild texture instead of blur, which transfers
across kitchens. Knowing the scene halves the rate again (−54%, −65%). The
texture is plausible rather than the source's (visual check), and 241 MB of
weights repay themselves after 2.8–14 h.

#### G5c. Scene model at scale (agreed with the user, 2026-10-10)

Same code and protocol as G5b, cheapest first; each item is gated by the one
before only where noted.
1. *Checks on the pilot's models, no training.*
   - DISTS on V beside LPIPS: a metric the model never trained on (VGG16
     weights into `Models/`, approved), on all G5b arms and SVT-AV1.
   - Flicker on the visible background: the decoded frame's change between
     frames, after warping by the source's own flow, against the source's
     change; SVT-AV1, DCVC-UF and both scene models.
   - Model size against quality, every variant kept so the paper can offer
     them by latency budget: the whole model at 16, 8, 4 and 2 bits, and
     only the changed weights (the difference from the public weights,
     which the client already has) at 8, 4 and 2 bits, dense and with only
     the largest changes kept; each entropy-coded, each re-coding the
     excerpts.
   - The inpainted fill: OpenCV Telea per frame (now) against a temporal
     video inpainter (ProPainter, `Models/ProPainter.pth`, audited into the
     environment first), on the pilot excerpts, as rate at equal quality on V
     and flicker of the fill. The fill is never scored, but every method codes
     it, and the scene model trains on it. G4's comparison with panorama fills
     comes later; if the temporal fill wins here, all arms switch to it.
   *Done (2026-10-10, [experiments](docs/experiments.md), G5c):* the LPIPS
   gain transfers to DISTS on the two pilots (−53% and −76% against SVT-AV1;
   not on most other stretches, see step 2); DCVC-UF
   flickers more than SVT-AV1 and β > 0 adds about a third; the model
   ships as its 4-bit change from the public weights, 51 MB at equal
   quality (payback 0.6–3.1 h; the change is dense, so smaller needs
   step 4); ProPainter's fill beats Telea's by 11–13% for SVT-AV1 and
   replaces it everywhere; DCVC-UF decodes reliably only on the
   encoder's GPU class.
2. *The distortion–perception trade.* The loss already holds both terms; a
   β sweep (adding 0.005 and 0.01 to 0 and 0.026) gives the curve between the
   PSNR-faithful and the LPIPS-optimal model. If flicker is found, a
   temporal term joins the loss here. DISTS joins the loss as one more arm,
   so the model learns from every metric it is scored on (user, 2026-10-10):
   a model good on PSNR, LPIPS and DISTS at once is unlikely to game all
   three, and the paper shows side-by-side frames so readers can check by
   eye.
   *Done (2026-10-11, [experiments](docs/experiments.md), G5c step 2):*
   β = 0.005 (any perceptual weight flips the model from losing to beating
   SVT-AV1 by two thirds or more on LPIPS_V and DISTS_V; larger β only
   costs PSNR_V); DISTS in the loss trades LPIPS_V for DISTS_V and is not
   adopted; a temporal term buys 13% less flicker at 40% LPIPS_V and is
   not adopted. ProPainter's fill of all 10 stretches is published. The
   recipe on all 10 stretches passes G5b's gate on LPIPS_V (median −41%
   against SVT-AV1, 8 of 10 below 0) but not on DISTS_V (median +28%, 4
   of 10): the pilots were among the best stretches. Step 3 starts from
   this row.
3. *Scene footage from other days.* EPIC-KITCHENS recorded each kitchen on
   several days (3–19 videos per pilot kitchen, 179 videos on the hosts). The
   scene model is fitted on the kitchen's other videos and scored on the
   stretch's excerpt: hours of footage and a true hold-out across days. The
   cross-kitchen control of G5b stays as the attribution check.
   *Done (2026-10-11, [experiments](docs/experiments.md), G5c step 3):*
   other recordings of the kitchen do not improve the stretch's own model
   at 540 steps, and alone they fall short of it, but a model made only
   from them still beats SVT-AV1 by 60–74% on LPIPS_V on the calm pilots.
   The oracle on the two fast stretches only ties or barely beats
   SVT-AV1 (−0.3%, −11%): there the base codec's motion coding is the
   limit, not the footage.
4. *Recipe.* The image model that codes each stream's first frame adapted
   too, on the same scene frames; adapting only part of the network (decoder
   and entropy model, or low-rank adapters); learning rate, crop size and
   steps, a bounded sweep. Speed over polish until the baselines are beaten;
   final results train to convergence.
   Selection (user, 2026-10-11): the recipe is chosen by how many of
   PSNR_V, LPIPS_V, DISTS_V and flicker it beats SVT-AV1 on (median
   BD-rate over the stretches), ties broken by the mean BD-rate; visual
   comparisons are made at matched rate (each model re-coded at the QP
   nearest the target rate), never at matched QP. The user's blind look
   at equal QP (2026-10-11) could not tell β = 0.026 with and without
   DISTS apart, and preferred both over β = 0.005 and β = 0, which also
   spent fewer bits.
   *Round 1 done (2026-10-11, [experiments](docs/experiments.md), G5c
   step 4):* the image model is not adapted (the intra frame is 1.4–9.6%
   of the bytes); KID on V joins the selection as a held-out judge; a 3×
   learning rate is adopted (beats SVT-AV1 on LPIPS_V −65% and DISTS_V
   −10% over the 4 development stretches, against C's 1 metric); flat
   frame weights flatten DCVC-UF's quality swing only partly and cut
   flicker 5%, not adopted. PSNR_V and flicker stay DCVC-UF's losses.
   *Round 2 done (2026-10-12):* 10⁻⁴ is not better; **the recipe is
   β = 0.005, learning rate 3 × 10⁻⁵, ProPainter's fill, 540 steps**,
   on all 10 stretches it beats SVT-AV1 on LPIPS_V by a median 50%
   (9 of 10), and on DISTS_V on the four calmest; PSNR_V, flicker and
   KID_V are lost. All BD-rates against SVT-AV1 were
   corrected on 2026-10-12 (its envelope was a staircase, optimistic by
   about 7%; no decision changed; [experiments](docs/experiments.md)).
   *Round 3 (2026-10-12):* 1,080 steps improve every stretch a little
   (LPIPS_V 1–11 points) but by less than the rule's margin, so the
   recipe stays at 540 steps for development and the paper's final runs
   train to convergence; the fast stretches reach a narrow LPIPS_V win
   (P06_05 −19%) and a tie (P04_13). G5d step 2: the enhancer from
   other recordings beats SVT-AV1 on LPIPS_V and DISTS_V on both pilots
   but misses the 3-metric rule on P26_02; a temporal (decoded-frame)
   input does not cut its flicker.
   *Model size (2026-10-12):* the recipe ships as its 4-bit change from
   the public weights, 50 MB at equal quality (2-bit breaks); payback
   2.4–3.4 hours of stream at 50 kbps on the pilots.
   *G5d oracle (user 2026-10-11, done 2026-10-12):* a venue-trained
   enhancer after plain SVT-AV1 (NAS-style, 1.2 MB), fitted on the
   stretch's other frames with LPIPS in its loss, beats SVT-AV1 by 10%
   or more on LPIPS_V, DISTS_V and KID_V on both pilots, and loses less
   PSNR_V and flicker than the scene model; fitted for MSE alone it does
   nothing. By its rule a venue-footage enhancer is next, compared with
   the scene model.
5. *Scope: calm and moving viewpoints.* The scene model's gain tracks the
   excerpt's motion and SVT-AV1's rate (step 2). More calm stretches,
   chosen by G1's camera-motion audit with a threshold fixed before any
   coding, test that the gain holds on calm scenes; the moving ones stay
   in the paper. The oracle on the losing stretches (fitted on the
   excerpt itself) tells whether their loss is the footage's or the
   method's, and step 3 how much real footage closes it.
6. *The rest.* The chosen recipe on the 8 untouched stretches, against
   SVT-AV1 also at its visual-quality tune with film-grain synthesis.
Deferred to after a working pipeline: a general model pooled over many
kitchens with per-kitchen fine-tuning on top (the generalization claim); why
DCVC-UF off the shelf ranks worse on LPIPS than on PSNR (grain removed by an
MSE decoder, quality swings within its 8-frame groups, the 4:2:0 input path;
one test each); the 1080p check; a small viewing test. VMAF joins at H5, on
whole frames.
**Done when** the scene model's recipe is fixed from recorded jobs and its
curves on all 10 stretches (LPIPS, DISTS, PSNR on V, flicker, model size per
variant and payback) sit beside SVT-AV1's and the unadapted DCVC-UF's.

#### G6. The background in the first working PointStream (plan, 2026-10-12)

What H5 needs from G: for each of evaluation set v2's 34 windows (1080p,
4:2:0, 50 or 59.94 frames/s, 4 s), the background's bitstream, its decoded
frames and decode time, at several rates, so the foreground session can
composite and B2's harness can score the whole frame. G5c worked on 10
stretches at 960×540 and 10 frames/s; three gaps lie between. Two routes
go in (user, 2026-10-12): the scene model (DCVC-UF adapted, recipe L) and
the venue enhancer after SVT-AV1 (G5d). Cheapest first:
1. *Interface, agreed with the foreground session before any run.* Input:
   the window's 1080p frames and a per-frame foreground mask F (the pixels
   the foreground stream delivers; the encoder fills F with ProPainter and
   codes the filled frames). Output per rate point: bytes, decoded 1080p
   frames, decode milliseconds per frame; once per venue: the model, its
   shipped size and payback. Composite at the client: foreground pixels
   over the decoded background on F. Scoring: B2's harness (weighted PSNR
   and LPIPS on dataset masks) plus every metric as BD-rate against
   SVT-AV1 and DCVC-UF on the whole frame. Open: the mask source at the
   encoder (dataset masks for the first run, predicted masks after).
2. *Resolution.* Does a scene model trained at 960×540 help at 1080p?
   Two options, one smoke each on the pilots' windows: code the background
   at 1080p with the model as trained (patches are resolution-free), or
   code it at 540p and upsample at the client (the viewer barely looks at
   it; PRESLEY). Rate at equal quality on V decides.
3. *Frame rate.* The windows are 5–6× denser in time than the stretches'
   10 frames/s, so motion per frame is smaller and DCVC-UF's temporal
   prediction does more of the work. Training sequences at the window's
   native rate (footage extracted at native rate, shorter segments) against
   the 10 frames/s model, on the pilots' windows.
4. *Scene footage for every window.* Step 3's pipeline (SAM 3.1 and EPIC
   detections, which cover exactly the 34 videos, then ProPainter) on the
   window's own video outside the window and a guard, and the kitchen's
   other recordings for the venue-only variants. About 34 footage jobs.
5. *Both routes on all 34 windows,* against SVT-AV1 and DCVC-UF coding
   the same filled background; published for the foreground session's
   compositor, with rate allocation between foreground and background
   taken as the concave hull over their rate combinations, per metric.
Budget: steps 2–3 about 4 GPU-hours; step 4 about 25; step 5 about 40
(34 windows × two routes). Steps 2–3 run first and decide the settings
of 4–5; each stage stays under 45 minutes per job.
*Steps 2–3 done (2026-10-12, [experiments](docs/experiments.md), G6):*
SVT-AV1 codes the background at 1080p; the scene model at 960×540,
upsampled at the client; at the windows' native rate it beats SVT-AV1
on LPIPS_V, DISTS_V and KID_V on both pilots (−91% and −45% on
LPIPS_V). The frame-rate rule missed by 1 point and the user treated it
as passed: native-rate and 1080p training are deferred to the final
runs as ablations.
*Interface, agreed with the foreground session (2026-10-12):* the
background codes the whole 1080p frame with F filled, so its decoded
frames are valid everywhere. F, first run: B2's dataset foreground (the
union of B1b's `visor_dense_sam_fill` instances), filled exactly (no
dilation; the foreground blends linearly over an 8 px ring outside F, so
true background must survive there). The mask is granted in the first
run, stated, and reported as its own line (its cost, 72–170 kbps for a
raw outline per H3b2, is an open H5 item). Rates: about 10 kbps to
2 Mbps, at least five log-spaced points per route (the allocation is the
concave hull over foreground and background combinations). Published per
window: decoded frames (raw 4:2:0 or FFV1; DCVC-UF decodes differ across
GPU classes) and a manifest (window id and frame range, F's mask-set
sha256, and per rate point bytes, bitstream sha256, per-frame decoded
sha256s, decode ms per frame, GPU class, job id, `git_head`).
**Done when** both routes' background streams for all 34 windows are
published with the interface above and the foreground session has
composited them.

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

#### H2. Hand-pose estimators: HaMeR against WiLoR (done 2026-10-10)

Result: `experiments/visor/h2.py` from recorded jobs (HInt `20261009T220226Z-958cda4e`,
HOT3D `…220759Z-be2af4a6`, VISOR `…215322Z-188ed69c` + `…220823Z-b309d9a2`,
coding `20261010T054156Z-d9821c8e`;
[outcome](docs/experiments.md#2026-10-09--h2-hand-pose-estimators-hamer-against-wilor-and-pose-coding)).
WiLoR is chosen, 3 contests to 2 (HInt VISOR frames, HOT3D PA-MPJPE, VISOR
IoU against HInt New Days and HOT3D acceleration); HaMeR orients the whole
hand better (7.6° against 14.7° on HOT3D), which H3 should check on its
renders. The pose stream for H3: Gaussian smoothing over 33 ms, 15 Hz with
linear interpolation, the full 45-value pose at 8° steps, previous-frame
prediction: 1.93 kbps on VISOR (H1's coding 15.2), 83 ms of lookahead,
within 5% of the uncoded estimate's error against HOT3D's motion capture;
it passes H1's rate screen (10% of SVT-AV1's hand bits at CRF 62). At zero
latency the best is 7.7 kbps.

H2b (2026-10-10,
[outcome](docs/experiments.md#2026-10-10--h2b-a-hybrid-of-the-two-estimators)):
a hybrid of WiLoR's fingers and HaMeR's orientation, chosen on H2's HOT3D
clips, was tested on new clips, HInt and VISOR under a rule fixed
beforehand. Its 3D gain replicated, but it is worse than both models in the
image (HInt PCK 35.6 against 43.7, VISOR IoU 0.739 against 0.768): the
parts do not compose. WiLoR stays.


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

H3a (2026-10-10, [outcome](docs/experiments.md#2026-10-10--h3a-texture-transfer-oracle-for-hand-rendering)):
the texture-transfer oracle passes. On VISOR's hand pixels, SVT-AV1
references (CRF 48) sent on 26% of hand-frames and warped by the true motion
reach LPIPS 0.091 at 41.9 kbps, against SVT-AV1's 0.116 at the same rate,
on all 34 items; at equal quality that is a 65% saving on hand pixels, 32%
if a reference costs twice an inter-coded frame. The oracle's rate is 95%
references (texture transfer with refresh); it does not bound a generator
that renders from one reference and the pose (about 2 kbps), which needs
its own test. Next (user, 2026-10-10): H3b, a single-reference generator.
Audit existing pose-conditioned hand generators that take a reference
image, run the best on WiLoR's coded poses with one reference per track,
and score it with full-reference LPIPS against SVT-AV1 and H3a's curve.
Texture transfer (a MANO texture with references sent as an inter-coded
stream) stays as the fallback. Generic hands judged without a reference
were not chosen.

H3a2 (2026-10-11, [outcome](docs/experiments.md#2026-10-10--h3a2-the-texture-transfer-oracle-with-a-measured-reference-stream)):
with each operating point's references sent as their own SVT-AV1 stream
and measured, the oracle fails on all 12 points (LPIPS +0.002 to +0.029 at
equal rate, no equal-quality saving). Sparse references cost 1.4–2.1× an
inter-coded frame and decode worse than the same frames at full rate. So
texture transfer with refresh does not beat coded pixels, and the
textured-mesh fallback is not built. H3b (the generator at about the pose
stream's rate) is the remaining route.

H3b's oracle (to review with the user). A generator from one reference and
the pose is billed at about the pose stream. The oracle grants it a perfect
generator, so what remains is what the pose cannot carry:
- the coded pose's placement error (estimation and coding);
- what MANO does not describe: forearm and sleeve extent, held objects in
  front of the fingers, motion blur.
Built on VISOR as perfect appearance placed by WiLoR's and HaMeR's coded
poses: the true hand pixels moved by the mesh correspondence from the
uncoded to the coded pose, against the true frame. This also confirms the
estimator on renders. The uncoded estimate stands in for the true pose,
which VISOR lacks; HOT3D's motion capture checks that proxy.

Identity (user, 2026-10-10): a hand that looks different from the source
(skin tone, a missing watch) is acceptable as long as it stays consistent
across the video. Full-reference LPIPS charges for such differences, so
H3b's judge must change. Proposal: LPIPS beside measures that ignore
identity:
- pose fidelity (WiLoR re-run on the render against the source's pose);
- realism (FID/KID of hand crops against real VISOR hands);
- temporal consistency (flow-warped difference between consecutive
  rendered frames).
With these the generator may also use no reference at all, a generic hand
kept constant per track.

Agreed (user, 2026-10-11): the identity-free judges above, and H3b's audit
of pose-conditioned hand generators, with downloading the chosen weights
into `Models`. DISTS is reported beside LPIPS and PSNR, as in G5c. Order:
1. H3b oracle: perfect appearance at WiLoR's and HaMeR's coded poses on
   VISOR. Only the pose's placement error and what MANO does not describe
   are charged; scored on all judges at the pose stream's rate.
2. H3b generator: audit, smoke, then WiLoR's coded poses with one reference
   per track (and a generic hand), scored the same way.
3. H3c, a perceptually tuned coder of the foreground pixels: G5b's method
   (DCVC-UF with β·LPIPS added to its loss, which taught the background
   decoder to rebuild texture instead of blur and transferred across
   kitchens) fine-tuned on hands and forearms. It is a foreground
   candidate in its own right and the fair baseline: a generator at
   about 2 kbps must beat a perceptually tuned codec, not only SVT-AV1.
   It reuses G5b's trainer without changing the background session's
   files.
4. H3d, a content-aware super-resolution of the hand pixels (NAS, Yeo et
   al., OSDI 2018; raised by the user, 2026-10-11): SVT-AV1 at a quarter or
   an eighth of the resolution (H3b2: 4.7 and 1.9 kbps on the hands) and a
   small SR network trained on the video's own hands, its bits charged
   (amortized over the video, as PLAN I's payback time). Oracle first: an
   SR overfit to the evaluation window itself is its upper bound; it must
   beat H3c's curve at equal LPIPS to be built.

H3b oracle (2026-10-11, [outcome](docs/experiments.md#2026-10-11--h3b-oracle-perfect-appearance-at-the-coded-pose)):
fails on the 4 pilot items. Perfect appearance at WiLoR's coded pose, with
the forearm granted, reaches LPIPS 0.211 at 1.93 kbps against SVT-AV1's
0.102 at 35 kbps (CRF 62; +0.109, CI +0.062 to +0.156), and is worse on
DISTS, PSNR, flicker, pose fidelity and KID too; HaMeR is worse still, and
even the uncoded pose fails (0.168). MANO's silhouette misses or overshoots
about a tenth of the hand, and that alone costs more than SVT-AV1's
coarsest pixels. So the generator is not built at the pose stream's rate,
and H3c is next. H3b2 (user, 2026-10-11: judge at equal quality,
[outcome](docs/experiments.md#2026-10-11--h3b2-svt-av1-down-to-the-pose-streams-rate-equal-quality)):
SVT-AV1 downscaled reaches the pose stream's rate (1.9 kbps at an eighth of
the resolution, LPIPS 0.29) and needs about 5 kbps for the oracle's 0.21.
The oracle's equal-quality saving is 48% (CI 6% to 81%) with the forearm
granted, none drawing the hand alone: below the 50% bar, so the route does
not reopen. Sending the hand's outline instead would cost 72–170 kbps per
frame without prediction. If a generator is revisited (agreed, user,
2026-10-11): one appearance reference per hand per video, not per track;
then bridging pose gaps of ≤ 10 frames (83% of breaks) and boxes from the
masks rather than the detector.

H3c (2026-10-12, [outcome](docs/experiments.md#2026-10-11--h3c-a-perceptually-tuned-dcvc-uf-for-the-hand-pixels)):
tuning does not pay, but DCVC-UF itself does. Off the shelf, DCVC-UF HT-L
(at full, ½ and ¼ resolution, rate split onto the hand region by its own
entropy model) needs about 63% less rate than SVT-AV1 for the same LPIPS on
the 4 pilot items' hand pixels (CI −68% to −59%; PSNR agrees, DISTS on 2 of
4). Fine-tuned on the test windows with β·LPIPS it gains only 14% more; on
other participants' hands it is 151% worse than the published model, and
tuning without LPIPS starves the background instead of coding the hands
better. So H3d's bar, and any generator's, is the lower envelope of
SVT-AV1 and DCVC-UF off the shelf on R, per item. H3d is next.

H3c2 (2026-10-12, [outcome](docs/experiments.md#2026-10-12--h3c2-pose-fidelity-at-rate-pixels-against-keypoints);
user, 2026-10-12: judge the hand by the pose it carries, identity free, as
coding for machines does, while keeping a video people can watch; a
domain-specific mode beside the pixel route, not instead of it, since tennis
may not want it). With a perfect generator granted, H2's coded keypoints
(1.93 kbps) carry WiLoR's pose for 69% less rate than the best pixel coder
needs for the same pose error (CI 47% to 90%; 39% to 98% per item). Unclear
by the rule; the user opened the path (2026-10-12). Order now:
1. H3e, a generative-hand oracle (before H3d): a pretrained pose-conditioned
   hand generator, delivered with the decoder and not billed, renders each
   hand from the coded keypoints (no per-video reference; identity free,
   constant per track). Audit first (methods and weights, downloaded into
   `Models`), then judged on the pilot items by WiLoR's pose error on the
   composite, KID and flicker on hand crops, against the pixel envelope
   (SVT-AV1 and DCVC-UF off the shelf) at equal pose error. Built further
   only if it keeps a saving above 50% with the forearm, held objects and
   the boundary with the background charged as they are rendered. As
   generative face video coding does with facial keypoints. Variants
   (user, 2026-10-12): the hand alone generated, its forearm and held
   objects coded as pixels (DCVC-UF off the shelf), the generated hand
   pasted over them, both streams billed; and the forearm generated too.
   The generator's own silhouette places the hand, so no outline is sent
   (outlines cost 72–170 kbps, H3b2).
   Outcome (2026-10-12, [outcome](docs/experiments.md#2026-10-12--h3e-a-pretrained-pose-conditioned-generator-renders-the-hand-from-the-coded-keypoints)):
   **fails**. JointControlVideo (Wan2.1-I2V-14B with a hand-joint module,
   ECCV 2026) renders each pilot window from its first frame and H2's coded
   joints; pasted over DCVC-UF-coded surroundings, it needs more rate than the
   pixel coders for the same pose error on three of four items (mean −297%,
   CI −679% to −21%; +29% on P02_02 only). The keypoints are accurate, but
   the composite's pose error is 1–4 points above theirs (the generated
   hands land 3–11 points from their input), and without the camera's
   motion its scene drifts from the real one. The generative mode is not
   built now. What would change it, each with its own oracle: conditioning on
   the decoded pixels around the hand every frame (the proto-hand candidate
   below, or hand inpainting into the coded frames), or on camera motion.
2. H3d, as agreed, against H3c's bar (now the per-item concave hull of
   SVT-AV1 and DCVC-UF off the shelf on R, by BD-rate; project convention).
   Oracle (2026-10-12, [outcome](docs/experiments.md#2026-10-12--h3d-oracle-a-content-aware-super-resolution-of-the-hand-pixels-nas)):
   **passes**. G5d's enhancer (0.6 M parameters) fitted on the window
   itself, its bits free, over DCVC-UF's ¼ and ½ decodes needs 77% less rate
   than the bar for the same LPIPS_R (CI −89% to −66%, 4 of 4 items; DISTS
   and mostly PSNR agree; flicker unchanged). Over SVT-AV1 it is unclear
   (−19%) and synthesizes texture. Next, the component: fitted outside the
   window (memorization) and paid for. At 1.2 MB it costs more than it
   saves within each pilot video (1.7–3.5 minutes), so network size, and
   sharing one network across videos, are its axes.
   Component (2026-10-13, [outcome](docs/experiments.md#2026-10-12--h3d-component-per-video-per-person-and-generic-super-resolution-of-the-hand-pixels)):
   **G stands: H3d's coder is DCVC-UF off the shelf at ¼ and ½ with a
   generic super-resolution network** (594 k parameters, fitted on six
   recordings of five other participants, sent with the decoder): −22% rate at equal
   LPIPS on the hands against the bar (CI −34% to −10%, 4 of 4 items),
   DISTS agreeing, PSNR worse (+11% to +35%). Per video (NAS, billed) fails
   at every size on these short videos; per person (the hybrid) is unclear:
   better than G where the person looks as usual, but it recolours P02_02's
   gloves toward skin. Fitting outside the window keeps about half of the
   oracle's gain.
3. Candidate after H3e (user, 2026-10-12): DCVC-UF conditioned on a
   proto-hand. The decoder also receives a render of the coded keypoints
   (a skeleton or MANO image), so its bitstream carries only what the
   pose does not explain; billed as keypoints plus that residual. DCVC-UF
   has no such input, so this is a new input path and training, not a
   fine-tune. Cheapest oracle: the render as DCVC-UF's reference frame.
   A latent shaped like the keypoints alone, decoded into a hand, is the
   generator of H3e, not a different design.

Deferred to the optimization stage (user, 2026-10-12): the keypoints'
rate. H2 kept the pose essentially uncoded-accurate; coarser quantization
and a pose prior (PCA 6–12) traded a few percent of error for large
savings. A BD-rate on pose error over H2's combinations picks the point.
A cheaper point can only raise H3c2's saving, so deferring it cannot
change H3e's decision. A reference-based mode (the original hands' look)
stays out of scope unless an application needs it: H3a2 found texture
transfer loses once the reference is billed, and pixel codecs then serve
pixel fidelity.

From an appearance reference and the per-frame pose (H2), render the hand and
forearm at the client: a textured MANO mesh, pose-conditioned generation, or
a mesh with a learned residual; the forearm from the mask's extent or a
simple arm model. Scored on VISOR foreground quality against coding the same
pixels with SVT-AV1 at equal rate, with HOT3D's oracle poses as the upper
bound, and on render time. The pose input is H2's coded stream (1.93 kbps);
the renderer is also run on HaMeR's poses (both models' parameters are saved
for all 34 VISOR items): the two are close in the image (HOT3D 2D error 17.3
against 17.4 px, VISOR IoU 0.761 against 0.767) while HaMeR orients the hand
better in 3D, so render quality confirms the estimator (user, 2026-10-10).
**Done when** a hand renderer beats coded pixels at some rate, or the session
shows why not, and the estimator is confirmed on renders.

#### H4. Handled objects (rescoped after H1, user 2026-10-08)

Grip pose (item 1) fails (H4a, 2026-10-13, job `20261013T064014Z-04a44b1c`,
[outcome](docs/experiments.md#2026-10-13--h4a-grip-pose-on-hot3d-is-a-held-object-still-in-the-hands-frame)):
on HOT3D's motion capture, a reference carried by the wrist holds 49% of
an object's moving frames at 2 px with a median life of 0.05 s (rule 70%,
0.5 s). The grip explains most of the translation but half the rotation
(13 mm and 10° left within a grasp, against 82 mm and 21° in the world):
objects turn in the fingers continuously. Consequence: handled objects get
no motion for free from the hand stream; a pose representation would need
its own per-frame object tracker (no labels on VISOR), so the 3D-model
candidate loses its cheapest route. Grip pose may still serve as a prior
(a tracker's start, a codec's motion search). Hands (H3) and the
background (G) are unaffected.

A bank of reference views (item 2) fails too (H4b, 2026-10-13, pilot job
`20261013T074546Z-821ee185`, [outcome](docs/experiments.md#2026-10-13--h4b-a-bank-of-reference-views-for-handled-objects-oracle-visor)):
an unlimited bank, choosing views with hindsight, holds 56% of handled-object
frames at 30 dB and sends as many new views as one re-registering reference
(1.00×), one every 0.03 s; older views are never better than the latest.
Outcome of H4: **handled objects stay coded pixels** (item 3), coded with
the rest of the foreground. The later candidates (a per-object 3D model,
generative completion) have no support from H4a or H4b and wait for a
working pipeline.

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

H4c (agreed, user 2026-10-13): handled objects as pixels, H3d's recipe.
DCVC-UF off the shelf codes any pixels; only G is domain-trained (on hand
crops). Open question (user): hands are alike, objects are not, so one
enhancer for both may serve neither. An enhancer restores local texture
from a decoded image rather than generating objects, and G5d's enhancer
already covers whole kitchens, so variety may cost less than for a
generator; the pilot measures it. Arms, each scored on hand and on
handled-object pixels separately against the pixel envelope at equal
quality: G as it is; G refitted on object crops; G refitted on hands and
objects (one network, same size). Data as H3d's component (other
participants' recordings, SAM 3.1 masks). Fallback if all fail:
generative completion of unseen sides.

#### H5. First working PointStream

G's background and H's foreground in one bitstream with every byte counted:
references once, then camera pose, hand pose and object motion per frame,
plus the fallback pixels. Scored with B2's harness on evaluation set v2
(weighted PSNR on dataset masks, with and without B1b's fill) against B2's
curves, and timed end to end, which picks B2b's equal-time SVT-AV1 preset.
**Done when** PointStream's curves and encode and decode times on all 34
items come from recorded jobs, beside B2's.

#### K. Where keypoints win (analysis, user 2026-10-13)

Keypoints are PointStream's premise (after GenStream). On egocentric video
they have lost every time they were asked to replace pixels: as appearance
(H3b, perfect appearance at the coded pose: 48% saving at equal LPIPS,
under the 50% bar), as a real generator (H3e: worse on 3 of 4 items), and
for objects as motion of any kind (H1, H4a, H4b: references die in 1–4
frames). They win as information: with a perfect generator, H2's coded
keypoints carry the hand's pose for 69% less rate than pixels at equal
pose error (H3c2). The failures have causes the work identified: a pixel
bar that keeps falling (DCVC-UF −63% against SVT-AV1 on the hands, G −22%
more), rendering judged by the pixels around the subject (MANO's outline
misses a tenth of the hand), and egocentric video's worst case (moving
camera, large hands, objects showing new sides, 2–3 minute videos that no
per-video model pays back). Keypoints should win where those causes are
absent: very low rates; judges of what keypoints carry (pose, trajectory);
small fast structured objects; long videos with constant identity; a still
camera. Broadcast tennis has all five, so it is measured now, not left to
phase 2's end (user, 2026-10-13). Cheap first, each with its rule written
before its run:
- K1. *Very low rates.* H3b2's low-rate win (perfect appearance at the
  coded pose: LPIPS 0.21 at 1.93 kbps, against SVT-AV1's 0.29 at the same
  rate, forearm granted) was against SVT-AV1 only; DCVC-UF at ¼ already
  reaches 0.3–3 kbps on R. Check first from recorded data (H3b, H3e, H3d's
  G curves on the same items and region): does any keypoint point lie
  above DCVC-UF + G's curve at its rate? If so, the paper shows the
  crossover (RD curves of each mode, with the rate where keypoints stop
  winning); if not, the low-rate claim is dropped for egocentric hands.
  The regime itself (where pixel coders collapse) is defined per domain by
  the judge (e.g. a detector's recall on the decoded ball, K5).
- K2. *Judges of what keypoints carry (coding for machines).* H3c2 judged
  hands by the pose re-estimated on the decoded video; the mask or pose
  computed on the original against on the decoded foreground is the same
  idea. A short study of MPEG's Video Coding for Machines (VCM) and
  feature coding (FCM) first: how they judge (task accuracy on the decoded
  content against rate, BD-rate on the task metric, their anchor codec and
  test conditions), and what PointStream should adopt, so that a keypoint
  mode is judged on a recognized axis beside human-viewing metrics.
  Literature only; no runs.
- K3. *Keypoint-conditioned pixel coding* (H3's proto-hand candidate). The
  decoder receives the keypoints and the pixels code only what they do not
  explain; in principle never worse than pixels alone, but it needs a new
  input path and training. Cheap oracle, inference only: give DCVC-UF as
  its reference the previous decoded frame warped by the keypoints' motion
  (H3b's mesh correspondence from t−1's coded pose to t's) instead of the
  previous decoded frame, and measure the rate on R at equal quality, with
  the keypoints billed. DCVC-UF estimates and codes its own motion, so the
  gain is bounded by its motion bits; if even this oracle saves nothing,
  the trained version is not pursued.
- K4. *Tennis audit*, H1's method on broadcast tennis (the archived
  `tennis_games` matches at 1080p; TrackNet's dense ball labels): per part
  (players, racket, ball, background), its share of pixels and of
  SVT-AV1's bits (H1's bit accounting), how often the camera is still, a
  body-pose estimator's fit (audit first, e.g. ultralytics' pose models)
  and its parameter rate against the bits on those pixels. Decides whether
  players pass H1's screen and what the ball costs.
- K5. *Ball oracle.* The true ball drawn from its coded position over
  SVT-AV1 and DCVC-UF decodes: ball-region quality and a ball detector's
  recall on the decoded frames, against the pixel coders at equal rate and
  equal quality. Decides the ball mode.
- K6. *Player oracle* (after K4): H3b for bodies, perfect appearance at
  the coded body pose with what the pose cannot describe charged; whether
  the outline problem shrinks at broadcast scale. A per-player appearance
  model is amortized over a match (hours), the condition H3d found
  decisive. The racket in a fixed grip re-tests H4a's grip pose.
Order: H5 continues on egocentric (a working pipeline with what is
chosen: G's background, DCVC-UF + G on hands, H4c on objects), since its
bitstream, compositor and harness carry over to tennis. K4–K5 and K2 run
beside it in their own sessions; K1 and K3 are cheap and egocentric.

### After a working pipeline

Optimisation, training and extensions, ordered by what G and H show they need.
VVC (VVenC, `/opt/local/bin/vvencapp` on the hosts) joins the baselines here
(user, 2026-10-08: SVT-AV1 is enough until then).

#### H3d follow-ups (deferred, user 2026-10-13)

H3d adopted DCVC-UF with a generic super-resolution network (G) for the
hand pixels; optimizing it waits for a full pipeline. Deferred, each with
its own oracle when taken up:
1. *Person network with a fallback* (best of both). K beat G where the
   person looked as usual (−43%, −55%) but recoloured P02_02's gloves toward
   skin. A per-person network that defers to G on content unlike its
   footage (by a novelty test on the decoded hand, or by sending a one-bit
   choice per segment, billed) would keep K's gain without its failures.
2. *Longer videos for the per-video network.* Fitted outside the window,
   V kept about half of the oracle's gain (−37% free of bits) from 1,152
   frames of 1.7–3.5-minute videos, and its bits sank it. Longer videos give
   more of the person's own footage (expected to retain more of the oracle's
   gain) and more seconds to amortize the network over. Measure both on
   long recordings, e.g. Egocentric-10K (archived on disk,
   `Datasets/archive/pre-reset-2026-10-05/`).
3. *G at scale:* more footage and recordings, a longer schedule, network
   size; then G on all 34 items.

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
submission date. Wherever a model is delivered beforehand and not billed
(G5b's scene model), report beside the rate the playback time after which its
bytes are repaid (user, 2026-10-10; G5b's pilot: 2.8–14 h for 241 MB).
Results to state when writing (user, 2026-10-13), if they still hold in
the final experiments: a per-video network (NAS) does not pay for itself on
short egocentric videos (H3d: 1.7–3.5 minutes; even 86 KB costs more than
it saves at the hands' rates), so networks are better delivered once; and
the generic network's gain on the hands is perceptual (LPIPS and DISTS
better, PSNR worse, a slight colour shift, no invented texture), as G5d's
enhancer on the background.
Negative results to report (user, 2026-10-13), with their numbers:
parametric motion fails for egocentric objects (H1's homographies, H4a's
grip pose on HOT3D motion capture, H4b's bank of views), and keypoints
lose to pixels on egocentric hands at equal quality (H3b, H3e) while
carrying the pose for less rate (H3c2). Possibly two papers, in order, if
the final results hold (user, 2026-10-13): one where keypoints beat the
standard codecs (tennis), one on the learned pixel coders that beat the
keypoints on egocentric video. Order of writing (user, 2026-10-13): a
definitive pipeline; a first draft complete but for final touches and the
figures' final experiments; the final experiments (run once, after the
figure list is fixed); final touches before submission. The framing waits
for the pipeline and K4–K6.
