# Plan

Each workstream is scoped to one session. Items that do not depend on each
other run in parallel sessions; each item names what it waits for (user,
2026-10-16). This file is the source of truth: a session card only starts
an item already written here, and nothing exists only as a card. Work runs
in two phases
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

## Where PointStream stands (2026-10-20)

- **Egocentric pipeline (H5):** runs end to end on the 4 pilot items
  (`src/pointstream/`, `experiments/visor/h5.py`). Against off-the-shelf
  anchors with F granted: weighted LPIPS −29% (G_HO in both foreground
  slots). Against fair anchors (H5p 3): fails with venue models billed;
  at equal adaptation the venue-adapted whole-frame DCVC-UF beats it on
  LPIPS (+16%, 1 of 4) while PointStream wins DISTS (−17%) and PSNR
  (−11%). Billing the mask F costs most of the margin (H5m). **Open, and
  the critical path: H5q (does the split win at equal adaptation?) and
  H5p 4–5 (F carried by the stream; visual check).**
- **Background (G):** a scene model adapted to the venue, sent beforehand
  and unbilled (payback hours reported), plus a per-video enhancer route
  (G6).
- **Foreground (H):** hands and handled objects are coded pixels, DCVC-UF
  at ¼ or ½ plus the generic enhancer G_HO. Keypoints lose on egocentric
  video (H1, H3, H4, K1, K3).
- **Tennis (K):** the ball mode wins (K5); players wait on K4b (far-player
  pose, keypoint rate, identification); court-view shots (K8) wait on
  labels. Next: K8, K4b, K5b, then K6; K7 (HiEve) later.
- **Paper (I):** notes on what to state, the negative results, the order
  of writing; the 34-item final runs wait for the figure list.

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

Done (2026-10-07). Narrative: [history](docs/history.md#b1-visor-adapter-and-evaluation-set-done-2026-10-07).

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

Done (2026-10-08). Narrative: [history](docs/history.md#b1b-filling-visors-missing-hands-and-objects-done-2026-10-08).

### B2. Baseline rate-distortion on VISOR (done 2026-10-07)

Done (2026-10-07). Narrative: [history](docs/history.md#b2-baseline-rate-distortion-on-visor-done-2026-10-07).

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

Done (2026-10-08). Narrative: [history](docs/history.md#g1-camera-motion-and-coverage-audit-next).

#### G2. Background evaluation protocol

Superseded by G5–G6 (the background became a scene-adapted neural model). Narrative: [history](docs/history.md#g2-background-evaluation-protocol).

#### G3. Panorama background

Superseded: G1 ruled out a rotation-only panorama on VISOR. Narrative: [history](docs/history.md#g3-panorama-background).

#### G4. Clean background frames

Superseded by G6's fill (ProPainter). Narrative: [history](docs/history.md#g4-clean-background-frames).

#### G1d. Depth-aware warps on VISOR (done 2026-10-09)

Done (2026-10-09). Narrative: [history](docs/history.md#g1d-depth-aware-warps-on-visor-done-2026-10-09).

#### G1e. Reference refresh (done)

Done (2026-10-09). Narrative: [history](docs/history.md#g1e-reference-refresh-done).

#### G5. Neural background model

Done (2026-10-09). Narrative: [history](docs/history.md#g5-neural-background-model).

#### G5b. Scene-adapted neural background (user, 2026-10-10)

Done (2026-10-10). Narrative: [history](docs/history.md#g5b-scene-adapted-neural-background-user-2026-10-10).

#### G5c. Scene model at scale (agreed with the user, 2026-10-10)

Done (2026-10-12): the recipe and model size G6 uses. Narrative: [history](docs/history.md#g5c-scene-model-at-scale-agreed-with-the-user-2026-10-10).

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
*Pilot done (2026-10-12, [experiments](docs/experiments.md), G6 steps
4–5 pilot):* the per-window pipeline runs end to end on the pilots'
windows and publishes both routes at the agreed interface. The enhancer
after SVT-AV1 beats it on LPIPS_V, DISTS_V and KID_V on both (LPIPS_V
better at every rate | −83%); the window's own footage makes the same
scene model as the stretch (±3% LPIPS_V at matched rate). *Deferred
(user, 2026-10-12):* the other 32 windows run once, with the final
experiments, after the paper's draft fixes the figures they feed.
*Item 1 done (2026-10-13, [experiments](docs/experiments.md), G6 item
1):* with F filled exactly, the scene model is trained on footage masked
the same way (`footage --mask exact`; LPIPS_V −11% | −12% against G1's
footage) and codes at 1080p in SVT-AV1's range, 540p below it (LPIPS_V
against SVT-AV1 −97% | −33%; DISTS_V and KID_V still worse on P06_03).

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

Done (2026-10-08). Narrative: [history](docs/history.md#h1-foreground-motion-and-representation-audit-done-2026-10-08).

#### H2. Hand-pose estimators: HaMeR against WiLoR (done 2026-10-10)

Done (2026-10-10): WiLoR; the pose stream at 1.93 kbps. Narrative: [history](docs/history.md#h2-hand-pose-estimators-hamer-against-wilor-done-2026-10-10).

#### H3. Hand and arm rendering

Done (2026-10-13): hands are coded pixels (DCVC-UF at ¼/½ plus an enhancer, since H4c G_HO); keypoint rendering loses (H3b, H3e). Narrative: [history](docs/history.md#h3-hand-and-arm-rendering).

#### H4. Handled objects (rescoped after H1, user 2026-10-08)

Done (2026-10-16): handled objects are coded pixels with the hands (H4a, H4b fail; H4c adopts G_HO). Narrative: [history](docs/history.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08).

#### H5. First working PointStream

Handled objects and hands (decided by H4c, 2026-10-16): DCVC-UF HT-L off
the shelf at ¼ or ½ resolution plus G_HO, one generic enhancer for both,
applied on the foreground boxes and sent with the decoder (networks:
`h4c-2026-10-16` jobs `20261016T152552Z-c3d9ed14` (¼) and
`20261016T153801Z-03f2a819` (½), one per QP). It replaces G on the hands.

G's background and H's foreground in one bitstream with every byte counted:
references once, then camera pose, hand pose and object motion per frame,
plus the fallback pixels. Scored with B2's harness on evaluation set v2
(weighted PSNR on dataset masks, with and without B1b's fill) against B2's
curves, and timed end to end, which picks B2b's equal-time SVT-AV1 preset.
**Done when** PointStream's curves and encode and decode times on all 34
items come from recorded jobs, beside B2's.
*Pilot done (2026-10-17, [experiments](docs/experiments.md), H5):* the
pipeline runs end to end on B2's 4 pilot items: G6's two background routes,
one DCVC-UF foreground stream of F (¼, ½, full; G on the hands), the
compositor and a container with every byte counted
(`src/pointstream/`, `experiments/visor/h5.py`). With F granted, rule (a)
passes: weighted LPIPS −15% against the pooled envelope of B2's preset-4
SVT-AV1 and DCVC-UF (CI −25 to −5, 3 of 4 items); frame LPIPS −38% and
DISTS −48% (4 of 4); whole-frame PSNR loses (+70%) and weighted PSNR
splits. The neural background carries the gain; the foreground stream
is worse per bit than whole-frame DCVC-UF where F is large. Billing F
at its lossless cost (0.5–2.9 Mbps) flips every comparison. Encoder
163–166 s per window (ProPainter 77%), decoder 95–133 s (this
pipeline's CPU stages): **the equal-time SVT-AV1 preset is 1** (B2b's
re-encode, ladder and ROI run can go). Before the 34-item run (about 200
GPU-hours, final-runs-once): B2b at preset 1, a mask coder, and the
foreground's per-bit cost on large F.
*H5m (2026-10-17):* F's class map coded at 1/s and billed (16–116 kbps at
s = 8–16) does not keep the weighted-LPIPS margin (hull over s −4%, 2 of 4
items; rule −10%, 3 of 4); frame LPIPS −21% and DISTS −28% survive on 4 of
4. Next for F: re-encode the foreground with K and the fill from F̂ (the
measured streams were coded for exact F), then F carried by the
foreground stream itself. *G_HO (H4c) in both foreground slots
(2026-10-20):* weighted LPIPS −29% (CI −38 to −19, 4 of 4), DISTS −23%
(4 of 4), with F granted.

*H5p, pilot-scale checks before any 34-item run (user, 2026-10-20: the
34-item run waits for the paper's figures; plan these instead).* On the
4 pilot items, each with its rule written before its run, about 10
GPU-hours in all:
Order agreed with the user (2026-10-20): F's rate test moved up, before
the fair anchors.
1. *Venue models' payback (CPU, minutes).* **Done**: the scene model
   ships as a 50 MB 4-bit change. Billed per video, rule (a) fails (−9%;
   the curve is the enhancer route alone); billed per kitchen it passes,
   −25% on 4 of 4. The scene model repays in 0.4–3.3 h of stream, each
   enhancer in 10–324 s. The routes stand as per-kitchen models, pending
   a cross-video check (a model fitted on the kitchen's other recordings,
   scored on the pilot window; GPU).
2. *F as a video (CPU oracle).* **Done, fails**: SVT-AV1 (F's signed
   distance or its step in Y, the class in U) never matches lzma at s = 8
   for less (1.4–2.4× dearer at equal accuracy). It halves the rate only
   for a slightly worse F̂, and even then, re-billed optimistically,
   weighted LPIPS reaches −8% (3 of 4), short of rule (a). lzma stays;
   F carried in the foreground stream becomes the main route.
3. *Fair anchors.* **Done, fails.** Against SVT-AV1 at preset 1 and
   DCVC-UF off the shelf, both with a 720/540/360p ladder, plus a
   whole-frame DCVC-UF adapted to the venue the same way, PointStream with
   its models billed per kitchen needs 74% more rate on weighted LPIPS (0
   of 4).
   - The ladder reaches 8–80 kbps, where the billed 50 MB scene model
     loses. Break-even is 10–30 h of stream per scene model, against a
     kitchen's 0.65–3 h on disk.
   - At equal adaptation, neither model billed, the venue-adapted
     whole-frame codec beats PointStream on LPIPS (weighted +16%, frame
     +15%). PointStream wins on frame DISTS (−17%, 4 of 4) and weighted
     PSNR (−11%).
   - The background's gain is mostly adaptation, not the split.
   - Decided (user, 2026-10-20): venue models are sent beforehand and not
     billed, so the venue-adapted anchor is unbilled too. The paper
     reports the hours of stream a venue model needs to repay itself
     (10–30 h here). Smaller venue models are a later optimization, once
     there is a win.
4. *F, GPU.* Re-encode the foreground with K from F̂ (lzma at s = 8 and
   16) with G_HO in the slots (H5m had G), and rescore; then an oracle for
   F carried by the foreground stream itself (no MASK segment).
5. *Visual check (CPU).* Side-by-side crops of PointStream and the
   anchors at matched rates, for the user's review.
Scores stay LPIPS, DISTS and PSNR (a model trained on a metric may be
judged on it, user 2026-10-20).

*H5q, the split's win at equal adaptation (user, 2026-10-20: "as long as
there's a way to win, we should take it").*
- The target: PointStream against the venue-adapted whole-frame DCVC-UF,
  neither model billed. Rule (a) on weighted LPIPS (mean −10% or better, 3
  of 4 items below zero), and against the pooled fair envelope too. Today
  it reads +16% (1 of 4). DISTS and PSNR are reported beside it.
- The levers only the split has, cheapest test first; each one gets its
  rule written before its run:
  1. *A venue-adapted foreground coder.* PointStream adapts only its
     background; the anchor adapts the whole frame, hands included. Use
     recipe L's fine-tune of DCVC-UF HT-L on the own-video footage's
     foreground input (F and its ring, the rest push-pull padded, at ¼
     and ½), with G_HO after it. Oracle first: fit on the scored window
     itself, the memorization upper bound. If even that does not turn the
     LPIPS row negative, the lever is dropped.
  2. *A denser allocation grid.* The hull over (foreground, background)
     pairs already splits the rate between F and the background. Add
     points only where the hull is sparse near the anchor's curve.
  3. *F carried in the foreground stream* (PLAN H5p 4). This removes the
     MASK segment, which only matters once F is billed.
- Venue models are sent beforehand and unbilled, on both sides. Payback
  hours are reported. Smaller models come later (user, 2026-10-20).

#### K. Where keypoints win (analysis, user 2026-10-13)

Analysis and finished items: [history](docs/history.md#k-where-keypoints-win-analysis-user-2026-10-13).
In short: keypoints lost on egocentric hands and objects every time they
replaced pixels, and win where pixels collapse on small, fast, structured
objects (the tennis ball).
- K1. Done (2026-10-16): no low-rate win for egocentric hands against
  DCVC-UF + G; the claim is dropped (user, 2026-10-17).
- K2. Done (2026-10-13): [docs/vcm.md](docs/vcm.md), the machine-judge protocol.
- K3. Done, closed (2026-10-17): keypoint-conditioned coding does not pay
  for hands at any keypoint rate (K3b: K3's gain was all appearance).
- K4. Done (2026-10-13): players fail H1's screen on the far player (fit
  0.875, keypoints 22% of SVT-AV1's player bits at CRF 62); K4b next.
- K5. Done (2026-10-13): **a ball mode is built** (−57% / −38% BD-rate on
  ball-region PSNR over DCVC-UF / SVT-AV1, for 0.23 kbps); K5b next.
- K6. *Player oracle* (after K4; gated off by K4 until the far player is
  fitted, see K4): H3b for bodies, perfect appearance at
  the coded body pose with what the pose cannot describe charged; whether
  the outline problem shrinks at broadcast scale. A per-player appearance
  model is amortized over a match (hours), the condition H3d found
  decisive. The racket in a fixed grip re-tests H4a's grip pose. The
  question it answers is the total: keypoints plus appearance against the
  pixel coders' envelope (SVT-AV1, DCVC-UF), swept over the rendering
  resolution, so K4's keypoint share (22% of SVT-AV1's player bits at
  CRF 62) is judged there, not by H1's 10% screen.
- K7. *HiEve audit* (user, 2026-10-14: open a domain if it serves
  PointStream; [docs/vcm.md](docs/vcm.md#datasets-vcm-and-fcm-evaluate-on)).
  HiEve (crowded events, surveillance-style footage; manual 14-joint poses, tracks
  and actions; CC BY-NC-SA 4.0) holds the five videos MPEG's FCM conditions
  use for tracking, and their labels are public. K4's audit on them: people
  per frame, their share of pixels and of SVT-AV1's bits, the camera's
  stillness, a body-pose estimator's fit against HiEve's poses, and the
  keypoint rate against the bits on those people. Decides whether the body
  mode (K6) is also judged there, by MOTA with JDE and pose AP against
  VTM and HM through CompressAI-Vision, on MPEG's own videos. CPU first;
  the download (registration may be needed) is the first step.
Follow-ups from K4–K5 (user, 2026-10-14), in the order they gate each
other. Each has its rule written before its run, as everywhere.
- K8. *Court-view shots.* Split a match at its cuts (ffmpeg scene score)
  and send to PointStream only the shots that show the court; close-ups,
  crowd shots, replays and graphics go to the pixel coder. The old
  pipeline (`9cf3012`, `scene_classification.py`) called low-motion
  scenes points, which let a close-up and an outro card into K4's
  windows. Here a shot is a court view when its court lines register to
  the court model (the `Tennis_courts` reference) with a homography; the
  same homography gives the court outline that K4b uses to find the
  players and the per-frame camera registration a panning broadcast
  camera needs (K4: still on 35% of a match). Shared by foreground and
  background: G's per-shot background needs the registration, H/K the
  court outline and the shot list. Judged on the archived matches' scene
  clips, reviewed by eye, and on the whole match's shots.
- K4b. *Players before K6* (re-reads K4's (b) and the K6 gate).
  1. Identification. Candidate rules, content-blind: motion in body
     heights (K4's amendment), horizontal motion with the clip's end
     discounted, a racket mask touching the person on most frames,
     inside the court outline (K8), keypoint motion, and combinations.
     Chosen on players marked by hand on windows K4 did not measure (a
     review page where the user marks the two players on a few dozen
     frames; RacketVision has no player labels), then applied once to
     every window as amendment 2, with both earlier definitions' numbers
     kept.
  2. Pose on the far player. Top-down on each SAM player crop, enlarged:
     a cheap sweep of the crop size (three sizes, native to about 4×)
     measuring fit and latency, then a top-down model (DWPose, weights in
     `Models/DWPose`) if the sweep falls short, super-resolution of the
     crop last; fine-tuning only if all fail (no tennis pose labels;
     SportsPose or pseudo-labels from fitted near players). Players under
     10% of the frame height are fitted on 0–50% today, 15% and taller on
     97–100%.
  3. Keypoint rate against keypoint error: quantization 1, 2, 4, 8 px,
     constant-velocity prediction, a lower keypoint frame rate with
     interpolation; raw keypoints saved this time (K4 kept histograms
     only). Which error is acceptable is K6's to judge on the rendering.
- K5b. *Ball mode.* (1) The renderer's oracle: from the coded position,
  velocity and a per-window colour and size sent once, a sharp disk and
  an elongated blur, composited over the decodes and billed; scored as K5,
  and shown side by side with the source at equal rate for the user's
  preference (a sharp ball that does not look like broadcast video may be
  preferred). (2) An encoder-side ball tracker trained on TrackNet's and
  RacketVision's ball labels (PLAN's ball stage); it is also the judge,
  since an off-the-shelf detector finds 46% of the source's balls and
  SAM 3.1's text prompt 56%.
  Outcomes (2026-10-21, [entries](docs/experiments.md)): (1) no renderer
  adopted: a per-window appearance keeps 10–20% of the pixel oracle's
  gain (BD about 0% against the −25% bar); the diagnostic points to
  per-frame position and appearance. The user's rendering preference
  (disk or streak) is pending. (2) RacketVision's BallTrack adopted off
  the shelf as tracker and judge (0.918 on K5's TrackNet games, never in
  its training; 0.881 on RacketVision test; 0.022 false detections per
  frame); no training needed. Through SVT-AV1 at CRF 62 every tracker
  finds under 3% of the balls.
- K5b 1b. *Per-frame ball*: K5b 1's oracle with the
  encoder refining each position by fit (±4 px, billed through the same
  residual code) and sending the appearance per frame (setting and
  colour predicted from the previous frame, billed); same decodes, judges
  and rule; positions from BallTrack as well as the labels, so the ball
  mode is measured end to end. Judges (user, 2026-10-21): BallTrack
  detection and crop DISTS. Outcome (2026-10-21, [entry](docs/experiments.md)):
  not adopted, on rate: the drawn ball restores detection (0.01 → 0.90 at
  DCVC-UF's 13 kbps; 5–20× less rate than the pixel coders for 80% of the
  source's balls) and passes crop DISTS (streak −27% to −42%), but its
  ~1 kbps payload fails the frame-PSNR guard (+1.7% to +2.8% against
  +1%). Next: a rate-aware ball payload, or a restated guard (user).
  Rendering preference (disk or streak): pending.
Datasets: still-camera sets (TrackNet, RacketVision, OpenTTGames) are
fine where a decision needs labels, with the scope stated in the paper;
broadcast matches answer what a real broadcast costs.
Order: H5 continues on egocentric (a working pipeline with what is
chosen: G's background, DCVC-UF + G_HO on hands and objects, H4c), since its
bitstream, compositor and harness carry over to tennis. K4–K5 and K2 run
beside it in their own sessions; K1 and K3 are cheap and egocentric; K7
follows K4's method once K4 has run.

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
An observation for the tennis paper (user, 2026-10-21; K5b 2): pixel
coders erase the ball at low rate, measured with a valid ball detector.
RacketVision's BallTrack finds 0.938 of TrackNet's balls on games 8–10
(0.918 on K5's games, none in its training); through SVT-AV1 preset 8 at
CRF 62 (median 84 kbps, frame PSNR 33–34 dB) it finds 0.026, and every
other tracker of WASB's zoo under 0.03. K5's YOLO26x figure (0.02) came
from a detector that found only 46% of the source's balls. DCVC-UF at
equal rate is still to be measured with BallTrack (K5b 1b).
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
A point to argue (user, 2026-10-13), scoped to the evidence when written:
conventional codecs are one-size-fits-all, rigid in what they code and tuned
to pixel fidelity rather than to what people see; a learned codec's
strength is less its margin over AV1 than how much further it goes when
domain knowledge directs it. Evidence so far: DCVC-UF off the shelf is
−63% against SVT-AV1 on the hands (H3c), and directed by PointStream
(region rate split by its own entropy model, coding at ¼ and ½, a generic
enhancer) a further −22% (H3d); the scene-adapted background model (G5b,
G5c) is the same pattern. Limits to state with it: retraining DCVC-UF
itself on the domain gained little and did not transfer (H3c: 14% on the
test windows, 151% worse on other participants), so the direction works
through what surrounds the codec (scale, regions, enhancers, scene
adaptation) more than through retraining it; and SVT-AV1 with the same
enhancer was unclear (H3d oracle: −19%, synthesized texture), which
supports, weakly, that the learned codec composes better with learned
parts.
Checked (2026-10-16, [analysis](docs/experiments.md#2026-10-16--does-the-learned-codec-compose-better-with-a-learned-enhancer-analysis-of-h3ds-oracle)):
over its own base, the same enhancer gains about as much on SVT-AV1 as on
DCVC-UF (LPIPS −73% against −76%; PSNR −35% against −12%), so "the
learned codec composes better" is not supported. The argument to make
instead: DCVC-UF is the better base (−63%, H3c), and learned parts
directed by domain knowledge add large gains on top of a codec. A G
fitted on SVT-AV1 decodes would only refine this and is not planned.
