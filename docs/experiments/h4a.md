# Experiments: H4a

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-13 — H4a: grip pose on HOT3D (is a held object still in the hand's frame?)
- Question ([PLAN H4](../../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08),
  item 1). H1 found that no 2D warp of one reference view carries a handled
  object for more than 0.02 s (median), and that handled objects are 27–29%
  of the foreground's SVT-AV1 bits. Grip pose proposes a 3D motion model
  instead: while an object is held, its pose is the hand's pose (already in
  the stream) times one fixed transform G, resent only when the grip
  changes. This entry measures, on motion capture, whether that model holds
  and how often G must be resent. Bits are not the question: any 6DoF
  pose costs at most ~1 kbps however it is sent (6 values per frame), against
  41.3 kbps of SVT-AV1 on handled objects at CRF 62 (H1). The question is
  whether an object's motion can be *derived from the hand's*, so that the
  encoder needs an object pose only at the few frames where G changes
  rather than a 6DoF tracker on every frame (VISOR has no object models
  and no object-pose labels).
- Data: HOT3D-Clips `train_aria` (`Datasets/HOT3D`, 1,516 clips of 150
  frames at 30 fps; Aria only, as H2), 12 clips per participant (all 9, so
  108 clips), content-blind (lowest sha256 of
  "pointstream-h4:hot3d:<clip>"). Per frame: `objects.json`
  (`T_world_from_object` per object), `hands.json` (UmeTrack
  `T_world_from_wrist` per hand, motion capture), `cameras.json` (the RGB
  fisheye `214-1`, 1408²). Object geometry: the decimated meshes in
  `object_models_eval` (their vertices; no texture is used). Code
  `experiments/hot3d/h4a.py` (to be written after this entry).
- Method.
  1. Error of a predicted pose P̂ against the true pose P of an object at a
     frame: the mean over the mesh's vertices of their distance in the
     image `214-1` when projected with P̂ and with P (the toolkit's
     fisheye camera; px), and the same distance in 3D (mm) beside it.
  2. Predictors, each starting from a *refresh* (a frame at which the true
     pose is sent, exactly; the oracle's simplification): *world* (the
     object stays where it was in the world: P̂(t) = P(r)); *camera* (it
     stays where it was in the camera, H1's copy in 3D); *grip-left* and
     *grip-right* (P̂(t) = W(t) · G, G = W(r)⁻¹ · P(r), W the hand's
     motion-capture wrist pose). A predictor fails at the first frame whose
     error exceeds τ, or where its hand is missing; a refresh is sent there.
  3. Coders, replayed over every object of every clip: *world only*,
     *grip only* (the nearer hand at the refresh), and *hybrid* (at each
     refresh the predictor that lasts longest; an oracle choice by
     look-ahead, a 2-bit mode). A frame is *held* when it is not a refresh;
     a reference's *life* is the time from one refresh to the next.
  4. Scored frames: an object's *moving* frames, where its mean vertex
     displacement per frame (central difference over ±2 frames) exceeds
     0.5 mm, dilated by 5 frames. Objects at rest are carried by *world*
     anyway; the moving frames are where the hand must explain the motion.
     Lives are cut at the end of a moving segment and of the clip; cut lives
     count at their observed length (conservative), and their share is
     reported.
  5. Reported beside: the noise floor (*world*'s error on objects that never
     move, frame to frame), the grip's drift within a life (rotation and
     translation of W(t)⁻¹ P(t) against G), how often G changes (refreshes
     per second of moving time, per object), τ ∈ {1, 2, 4} px, and the rate
     of refreshes (6 values at 16 bits plus the mode) against SVT-AV1 on
     handled objects.
- Decision rule (fixed before any run; τ = 2 px, means over clips of
  per-clip values, pooled over each clip's moving frames). Grip pose is
  worth building on when the *hybrid* coder, on moving frames, (a) holds
  ≥ 70% of them and (b) has median reference life ≥ 0.5 s (H1's thresholds
  for a rigid part), and (c) *grip* is what explains them: the hybrid's
  median life is at least twice *world only*'s on the same frames. Rate
  ((refreshes × 98 bits) per second) must be ≤ 10% of SVT-AV1's 41.3 kbps
  on handled objects (H1's (c)); it is reported but not expected to bind.
  Pass: grip pose is a viable motion model for held objects, and the next
  step is an appearance oracle (the object's scanned model rendered at the
  true pose on HOT3D, against the pixel coders at equal quality), since a
  pose alone renders nothing. Fail: grip pose is dropped and H4 moves to
  item 2 (a bank of reference views on VISOR). Unclear (passes at 4 px but
  not 2 px, or (a)–(b) pass without (c)): reported, and the user decides.
- Hypothesis: rigid objects in HOT3D are held in a stable grip. Within a
  grasp G drifts by a few mm and a few degrees; refreshes come at grasp,
  release and regrasp, about once per second of motion. The hybrid holds
  ≥ 85% of moving frames with median life ≥ 1 s at 2 px, and *world only*
  lives a few frames.
- Competing explanations. (1) Objects move in the fingers, not with the
  palm (turning a mug, pouring, in-hand rotation), so G drifts past 2 px
  within a few frames; the drift measure separates a stable grip with
  finger slip from no grip at all. (2) The hand's and the object's motion
  capture each have their own noise, so G jitters even in a fixed grip; the
  noise floor and τ = 4 px test whether 2 px is above it. (3) HOT3D's lab
  tasks with rigid objects favour grip pose against VISOR's kitchens (food,
  cloth, deformable packaging); a pass is an upper bound for VISOR, which
  this entry does not test.
- Budget: CPU only (`"device": "cpu"`, 16 threads); reading a clip's poses
  means streaming its ~100 MB tar from NFS. Smoke: 2 clips, ≤ 600 s. No
  pilot: nothing is fitted, and τ is swept in the same pass. Full: 108
  clips, ≤ 45 min. Ceiling 2 CPU-hours.
- Jobs: `20261013T064014Z-04a44b1c` (code `b003be8`, environment
  `pointstream-20261009T214002Z`; CPU only, 16 threads on gpu1; inputs
  `pointstream-data/hot3d/h4a-2026-10-13/inputs-26fb572756ed.json`, spec
  `h4a.json` beside it). Smoke 2 clips (3 s), full 108 clips of 9
  participants (15 s of compute; 8.5 min with staging). Validators passed on
  the smoke and, locally, on the full stage's `published.tar`
  (`939b43ef…48bb3`; 8 checks: every clip published, eval meshes within
  2 mm of `models_info` (the keyboard's decimated mesh is 1.03 mm off, so
  the bound set before the run at 1 mm was raised to 2 mm after the dev
  run), true poses projecting inside the image, resting objects within 1 px,
  every scored frame held or refreshed once). Report (`h4a report`):
  `pointstream-data/hot3d/h4a-2026-10-13/report-b003be8/h4a-report.json`
  (`cb3b135a…463d`).
- Outcome (means over the 108 clips, all with moving frames; 189 moving
  objects, 15,558 scored object-frames). Noise floor: resting objects hold
  within 0.08 px (median; 95th percentile 0.55 px), so 2 px is well above
  the motion capture's noise. On moving frames:

  | coder | τ | held | median life | frame-weighted life | refresh rate |
  |---|---|---|---|---|---|
  | world | 2 px | 18.4% | 0.035 s | 0.068 s | 2.3 kbps |
  | grip | 2 px | 37.8% | 0.040 s | 0.103 s | 1.8 kbps |
  | hybrid | 2 px | 49.4% | 0.052 s | 0.129 s | 1.4 kbps |
  | hybrid | 4 px | 69.0% | 0.098 s | 0.246 s | 0.9 kbps |

  The hybrid's held frames come from the grip on 53% (left 22%, right 30%)
  and from the world on 46%. Grip spread over a segment's moving frames
  (201 segments, quartiles): G turns by 4.0 / 10.1 / 22.6° and shifts by
  6.7 / 13.1 / 30.4 mm from its middle frame (median over frames), against
  the world pose's 4.0 / 21.0 / 53.9° and 19 / 82 / 155 mm. Per-clip held
  share of the hybrid at 2 px: 28% / 50% / 70% (10th, 50th, 90th
  percentile).
- Decision (by the rule; `h4a-report.json`): **grip pose fails.** At 2 px
  the hybrid holds 49% of moving frames (rule 70%) with a median life of
  0.05 s (rule 0.5 s), 1.5× *world only*'s (rule 2×); at 4 px it holds
  69% with 0.10 s. Rate passes (1.4 kbps against 4.1) and does not bind.
  By the rule, grip pose is dropped and H4 moves to item 2, a bank of
  reference views on VISOR.
- Reading. The hypothesis is refuted: a held object is not still in the
  wrist's frame. The grip explains most of the object's translation
  (median shift 13 mm against 82 mm in the world) but only half of its
  rotation (10° against 21°); these are medians of per-segment spreads,
  not a variance split. What remains, mostly rotation (10° turns a 10 cm
  object's edge by tens of pixels at arm's length), is far above what a
  rendered object can absorb: a reference carried by the wrist dies in a
  tenth of a second, no longer than H1's 2D warps on VISOR. Competing
  explanation (1) holds: objects move in the fingers (regrasps, in-hand
  turning, two-handed handling such as the keyboard in the dev clip) and
  the grip drifts continuously, not in a few discrete changes. (2) does
  not hold: the noise floor is 0.08 px. (3) makes this a stronger
  negative: HOT3D's rigid objects in a lab are the favourable case, and
  the wrist is the motion-capture truth, not an estimate.
- Scope. The hand frame is the wrist (UmeTrack's); a frame fitted to the
  fingertips in contact, or a pose predicted from the whole hand's
  articulation, was not tested and could hold longer. The test is
  geometric only (vertex reprojection), so it bounds any renderer that
  needs the object's pose to within 2–4 px. Clips are 5 s long, which
  cuts 4% of lives; the decision does not rest on long lives.
