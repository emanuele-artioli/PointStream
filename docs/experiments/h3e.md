# Experiments: H3e

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-12 — H3e: a pretrained pose-conditioned generator renders the hand from the coded keypoints
- Question (user, 2026-10-12, after H3c2). H3c2 granted a perfect
  generator: the keypoints (1.93 kbps) carry WiLoR's pose for 69% less rate
  than the pixel envelope. Does a real, pretrained generator delivered with
  the decoder (billed nothing) keep a saving above 50% once its own pose
  error, the forearm, held objects and the boundary with the background are
  charged as they are rendered? Identity may change (constant per track);
  pose error decides (user, 2026-10-12).
- Audit (2026-10-12; methods, code and weights; no hand image generator was
  in `Models`). Criteria: conditioned on keypoints or MANO; egocentric
  hands with objects; temporal consistency; released weights and licence;
  fits a 48 GB card.

  | method | conditioning | domain | video | weights, licence | verdict |
  |---|---|---|---|---|---|
  | [JointControlVideo](https://github.com/ZhangCYG/JointControlVideo) (ECCV 2026) | first frame, 42 camera-frame 3D hand joints and intrinsics, text | egocentric hand–object (Ego4D) | yes: Wan2.1-I2V-14B-480P with LoRA and an occlusion-aware joint module, 81 frames | released, Apache-2.0 (Wan2.1 Apache-2.0) | **chosen** |
  | [FoundHand](https://github.com/arthurchen0518/FoundHand) (CVPR 2025) | reference image with its keypoints and mask, target 2D keypoints | hands, 10M images incl. Ego4D and **EPIC VISOR** | per frame, autoregressive (no video training) | released (notebooks only), GPL-3.0 code, CC BY-NC data | second: 256² crops, trained on our test source |
  | [HandRefiner](https://github.com/wenquanlu/HandRefiner) | inpainting with a mesh depth ControlNet (SD 1.5) | generated images | no | released, MIT | per-frame flicker; no temporal model |
  | HOGAN / [HOIG](https://github.com/play-with-HOI-generation/HOIG) (NeurIPS 2022) | hand and object poses | HO3D, DexYCB | no | code; needs object meshes | no object poses in our stream |
  | ManiVideo, RHanDS, HanDiffuser, Generated Reality | MANO/mesh or skeleton | various | some | no released weights found | not usable |

  JointControlVideo fits best: its input is what H2 codes (MANO joints in
  the camera frame, with the camera), it is trained on egocentric hand–object
  video that does not include EPIC-KITCHENS, and it models time. Its joint
  order is not documented (its dataset is gated); the order is settled on a
  development window below, not on the pilot items.
- Oracle. Per pilot item (B2's four), one generation over the 240-frame
  window: the source's frame 0 as the first frame (granted uncoded), and
  H2's coded 100 ms WiLoR joints (each track with its median shape, decoded
  as H3b) at window frames 0, 3, …, 240 (81 frames, about 17–20 fps; frame
  240 repeats 239), in camera coordinates with WiLoR's pinhole camera,
  resized and centre-cropped to 832×480 as the model's loader does; left
  hand in joints 0–20, right in 21–41, absent hands zero. A fixed prompt
  ("First-person view of hands handling objects in a kitchen."), sent with
  the decoder; cfg 5, seed 0, the released sampler. The generated frames are
  upscaled back to the window's resolution (Lanczos); they stand for window
  frames 3k, and frames in between are not generated (interpolation granted).
  Two compositions, inside R only (source outside R, as H3c scores):
  - **V1, hand generated** (user's variant 1): the generated pixels inside
    each coded hand's MANO silhouette, dilated by 10% of its box and
    feathered, over DCVC-UF off the shelf decoded at each of H3c's nine
    points (scales 1, ½, ¼ × QP 0, 21, 42). Rate on R = 1.93 kbps + frame 0's
    bits on R at that point + frames 1–239's bits on R outside the pasted
    masks (H3c's split by DCVC-UF's own entropy model, by area). The
    decoder has the coded pose, so the mask costs nothing.
  - **V2, hand and forearm generated** (user's variant 2): the generated
    pixels in all of R (dilated as V1), over the same decodes. Rate =
    1.93 kbps + frame 0's bits on R. R is the dataset's outline, which the
    decoder does not have (H3b2: 72–170 kbps): V2 is a bound, not a design.
- Why it bounds the generator route at this operating point. It grants the
  uncoded first frame while billing it at the coarse point, drops the hand's
  bits while DCVC-UF keeps the hand as context for its other cells, grants
  frame interpolation, and V2 grants the outline. It does not bound a
  generator fine-tuned on EPIC-KITCHENS or one at higher resolution: if it
  fails because the 480p frame loses the fingers (generated hands under
  about 100 px), FoundHand at 256² crops is the next oracle.
- Judges, as H3c2, on window frames that are both generated and judged
  there (t = 15, 30, …, 225): WiLoR re-run with H2's box on the composite;
  pose error = the mean joint distance to WiLoR on the source over the box,
  mean over the item's hands. The pixel envelope (SVT-AV1 and DCVC-UF off
  the shelf, Pareto on rate and pose error) is recomputed on the same
  frames from H3c2's saved joints (no new pixel coding). Reported beside:
  the generator's own error against the coded joints (does it follow its
  input), KID on hand crops and flicker (mean absolute frame difference
  inside R against the source's, over generated frames), for the composite
  and for DCVC-UF at the nearest rate; a visual check of every item.
- Decision rule (`h3e` in `DECISION`, fixed before the run). Per item and
  variant, the saving at equal pose error = 1 − rate / (the pixel
  envelope's rate for its pose error, linear in log rate along the front),
  maximized over V1's nine points (the encoder can choose). A pose error
  below every pixel point counts at the envelope's highest rate (a lower
  bound on the saving); one above every pixel point at its lowest rate.
  V1 decides. **Pass** (build the route: next its real coded first frame
  and the decoder's own forearm mask) if V1's mean saving's 95% bootstrap
  interval (by item) lies above 50%. **Fail** (no generator now; H3d next)
  if V1's mean saving is at or below 50%. Otherwise unclear, to the user.
  If only V2 clears 50%, that goes to the user too: it would need a
  decoder-side forearm mask, which is untested.
- Joint order and settings, before the pilot items (development window: an
  H2 VISOR item outside B2's four). One generation each with the joints in
  OpenPose order (as `mano_np` emits) and in MANO order (16 kinematic joints,
  then the tips); the order whose generated frames give WiLoR the lower
  error against the coded joints is used. The number of sampling steps is
  the released 50 if a window fits in 40 minutes on an RTX 6000 Ada,
  otherwise the most that fit, not below 30, recorded here before the
  oracle.
- Revision (2026-10-12, after the prepare smoke `20261012T102513Z-caafd73e`
  failed its depth check, before any generation). H2's joints are in WiLoR's
  camera (focal 5000/256 of the frame, 37,500 px at 1920: near-orthographic,
  hands about 20 m away). JointControlVideo's module reads inverse depth
  (clamped at 0.2 m) and orders occlusions by it, as in its metric Ego4D
  training data. The generator gets a stated camera instead: a 90° horizontal
  field of view (960 px at 1920), each hand's root at the depth that camera
  implies (0.026 × WiLoR's, about 0.5 m), MANO's metric depth offsets between
  joints, and every joint back-projected through its coded 2D position, so the
  projection the generator sees is exactly H2's coded keypoints.
- Revision (2026-10-12, project convention adopted on main, before any
  generation). Comparisons are BD-rates against an achievable anchor
  envelope. The pixel envelope (SVT-AV1 and DCVC-UF at three scales, several
  configurations) is their upper concave hull in quality (lower convex hull in
  pose error over rate), not a Pareto staircase, which understates a mixed
  anchor (G5d). V1's statistic per item is its BD-rate on pose error against
  that hull (PCHIP of log rate over the pose range both span, at least 0.1% of
  the box); where they do not overlap, the item is reported as better or worse
  at every rate, and its best point's rate at equal pose error stands in.
  V2 (one point) keeps the rate at equal pose error on the hull. The
  thresholds are unchanged: V1 passes with the mean saving (−BD-rate) above
  50% across its 95% interval and fails at or below 50% mean.
- Smoke (exclusive, Ada; `20261012T123958Z-7b531e66`, code `be04b54`, validator
  passed): 9 frames at 4 steps, then 81 frames at 10 steps on P22_107. 32.8 s
  per step at 81 frames, so the released 50 steps (≈ 27 min per window) fit
  the 40 minutes: **50 steps**, one job per generation. Peak 39.7 GB
  reserved; later generations run shared at 42,000 MiB. The T5 tokenizer's
  warning about a "Mistral regex" does not apply: the prompt's ids equal
  sentencepiece's own encoding and decode back to the prompt. Earlier
  attempts: `20261012T114547Z-8cecd105` (inputs read from the wrong level of
  the prepare archive) and `20261012T122341Z-9a1a31a1` (stopped as contended
  by another user's process). Not evidence.
- Joint order (development window P22_107, 30 judged hands; generations
  `20261012T132045Z-f518e2e7` OpenPose and `20261012T132054Z-829ed143` MANO,
  code `1b1ba9e`; judges `20261012T140150Z-f713de00` and
  `20261012T135958Z-806a82d3`, code `5204ab3`, validators passed). WiLoR on the
  generated frames against the coded joints: **OpenPose order 3.67%** of the
  box, MANO order 5.60%; OpenPose better on 27 of 30 hands (difference −1.9
  points, 95% CI −2.7 to −1.3). The pilot items use OpenPose order (as
  `mano_np` emits). For scale, on this window the coded keypoints themselves
  are 1.78% from WiLoR on the source, so the generator adds about 2 points.
  Development numbers, not evidence for the rule.
- Hypothesis. The joint module follows its input closely (the paper reports
  joint adherence), so the composite's pose error is the keypoints' (0.8–2.3%
  of the box) plus about half a percent, and V1 saves 50–70% (frame 0 and
  the forearm cost most of its rate).
- Competing explanations. At 480p the generated hands are 60–150 px tall
  and WiLoR misreads upscaled fingers, so the error grows past the pixel
  coders' at a few kbps; or the 81-frame clip drifts from the first frame
  (identity or objects change), which pose error does not see but KID,
  flicker and the visual check do; or frame 0's bits alone exceed the
  savings, since every window pays one.
- Budget. CPU: the Wan2.1 backbone and JointControlVideo weights (about
  80 GB) into `Models`, one new environment. GPU, RTX 6000 Ada: smoke ≤ 600 s
  (few frames and steps; exclusive, the model's first run on the class);
  joint-order check 2 jobs; oracle 4 jobs, one per item, each ≤ 45 minutes;
  about 5 GPU-hours.
- Jobs (Ada; validators passed). Generations (OpenPose order, 50 steps, code
  `9a67692`): P01_107 `20261012T141546Z-18c30e4a`, P02_02
  `20261012T141556Z-5b185f3a`, P03_10 `20261012T141605Z-ff439104`, P09_106
  `20261012T141614Z-326ad02f`. Judges (nine DCVC-UF points each): P01_107
  `20261012T145802Z-69fab945`, P02_02 `20261012T145812Z-5e63a0e4`, P09_106
  `20261012T145830Z-93a84904` (code `9a67692`), P03_10 `20261012T152219Z-20d57d2d`
  (code `e5e212c`, which changes only the spec writer). P03_10's first judge
  `20261012T145821Z-f99e456a` stopped when DCVC-UF's two decodes of one stream
  disagreed (scale ½, QP 42, on a GPU shared with another user's process); the
  resubmission decoded deterministically, as in H3c. Not evidence. Report (code
  `8b17aa5`, from the judges' and H3c2's archives on the server):
  `pointstream-data/visor/h3e-2026-10-12/report/h3e-report.json`
  (`49af833b24604d67…`).
- Outcome: **fails**. V1 against the pixel hull, at equal pose error:

  | item | judged hands | V1 | V1's pose error | keypoints' | generator vs its input | V2 |
  |---|---|---|---|---|---|---|
  | P01_107 | 30 | −171% (worse at every rate; flat) | 1.77% from 2.06 kbps | 0.77% | 3.0% | −175% |
  | P02_02 | 16 | **+29%** (BD-rate) | 5.2–5.5% | 1.67% | 4.9% | +76% |
  | P03_10 | 26 | −197% (worse at every rate; flat) | 2.0% from 1.95 kbps | 0.97% | 2.7% | −190% |
  | P09_106 | 9 | −848% (BD-rate) | 5.5–6.5% | 4.5% | 10.7% | −610% |

  V1's mean saving −297% (95% CI −679% to −21%), at or below 50%: **fail**.
  V2 (granted the outline) −225% (CI −501% to +9%). The pose error of V1 is
  flat across its nine points: it is the generator's, whatever the forearm
  and objects cost. The pixel hull reaches it at 0.3–1 kbps on P01_107 and
  P03_10. Flicker (luma, against the source's own change, between generated
  frames): V1 5–23, DCVC-UF 3–7 on the same frames. KID on hand crops, pooled:
  V1 0.008–0.009 (interval to 0.03), DCVC-UF 0.0 at QP 42 full scale to 0.031
  at QP 0 ¼ scale: realism is not where it loses.
- Visual check (the judges' crops at one judged frame per item, and the
  generated videos). The generator follows the joints but not the camera: it
  receives the first frame and the hand joints, not the head's motion, so the
  generated scene drifts from the real one (a toaster comes into view on
  P03_10; on P09_106 the frame shows a towel and a bowl that are not there).
  Pasted inside the coded silhouette, its hands then sit in a different scene
  from the coded pixels around them: smeared boundaries (P09_106), a forearm
  with a tattoo the source does not have (P01_107), which WiLoR misreads.
  P02_02, the one positive item, has a nearly still camera and fast gloved
  hands that the pixel coders blur.
- Reading. H3c2's margin (keypoints 69% cheaper with a perfect generator)
  does not survive a real one: its hands land 2.7–10.7 points (of the box)
  from the joints it is given, so the composite's pose error is 1.0–3.7
  points above the keypoints' own (0.8–4.5), and DCVC-UF reaches that
  fidelity below 1 kbps on three of four items. The competing explanation that held is the
  drift, not the 480p resolution (the generated hands are large enough for
  WiLoR on P01_107 and P03_10, where the error is still 3 points). What would
  change next, and needs its own oracle: a generator conditioned on the
  decoded pixels around the hand each frame, rather than on a first frame
  alone (the proto-hand-conditioned DCVC-UF candidate in PLAN, or hand
  inpainting into the coded frames), or one conditioned on camera motion
  (Generated Reality conditions on head pose). Not built now: H3d is next.
