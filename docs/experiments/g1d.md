# Experiments: G1d

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-09 — G1d: depth-aware warps on VISOR's background
- Question: G1 found that no VISOR stretch or window holds rotation-only
  (residual energy about 39% parallax and 32% independent motion on the
  stretches). How much of that residual does a warp that knows depth remove?
  Does egocentric video get a 3D background (a G3 variant), or go straight to
  the neural route (G5)?
- Clips (fixed content-blind before any run). All 34 windows of evaluation
  set v2, every frame, with G1's foreground (dense masks with B1b's fill). The
  10 stretches with the smallest sha256("pointstream-g1d:visor-long:<video>"):
  P26_02, P03_10, P06_10, P06_03, P37_102, P06_05, P02_09, P04_13, P09_106 and
  P06_106. Each is G1's 120 s at 10 frames/s with G1's foreground: the
  published SAM 3.1 `sam_text` masks plus the detector boxes, dilated as in
  G1. Frames are G1's decoded analysis frames. One CPU job decodes them once
  into a lossless archive (FFV1) with the masks. The job checks that each
  frame's foreground share equals the one G1 recorded and that the windows'
  released JPEGs still match the decoded frames.
- Pairs. G1's registration is re-run with G1's recorded lens. Every
  `direct` frame t is warped from its G1 reference keyframe k by every
  method, so all methods are compared on the same pairs. The job checks that
  the references and G1's rotation residual (G1's own measure) reproduce.
  Depth methods use EPIC Fields' published COLMAP calibration of
  EPIC-KITCHENS (P28_101: fx 239.6, fy 243.1 at 456×256, so 86.7° horizontal
  field of view with their mean; distortion negligible). A per-clip
  self-calibration is reported beside it as a diagnostic: the grid of focal
  length and k1 that minimises the calibrated essential matrix's Sampson
  error.
- Methods, in order:
  `rot` (G1's rotation and zoom, reproduced);
  (a) `h1` (one homography on undistorted pixels) and `planes2/3/4`. These
  carry a label map on the keyframe, sent once. K planes come from
  sequential MAGSAC (1 analysis px, 2 px at 1080p) on dense correspondences
  from k to its first companion: DIS flow seeded by a homography, sampled
  every 6 px on textured background. Each pixel takes the plane that best
  predicts its flow, then a 31 px mode filter. Each frame sends K
  homographies, which the encoder fits to the dense flow from k to t, per
  label, falling back to `h1` below 30 samples. The renderer forward-splats
  keyframe pixels, and the larger displacement wins a collision, because
  nearer surfaces move more;
  (b) `epi`: the dense flow from t to k, projected onto the pair's epipolar
  lines (fundamental matrix from the pair's SIFT matches). It removes the
  motion a rigid scene can explain, but it inherits every error of the
  per-pixel flow, so it is *not* a ceiling: the dev check below found `tri`
  above it. It is not a representation.
  `tri`, the triangulated upper bound: the keyframe's depth comes from dense
  flow to two companions. They are the frames nearest 0.5 s and 1.0 s after k
  (or before, at a clip's end; within 0.3–1.5 s, same segment, at least 30
  homography inliers). Each companion is posed by the essential matrix, and
  each pixel is triangulated (angle ≥ 0.5°, reprojection ≤ 1.5 analysis px).
  The primary is the companion with the most pose inliers. A second joins
  only if, scale-matched, it agrees on their shared pixels (median |log
  ratio| ≤ 0.05); where both measure, the larger angle wins. Background holes
  are filled by local plane fits of inverse depth (windows 31–241 px, at
  least 15% measured). Inverse depth is coded in 8 bits. Each frame's pose
  comes from PnP (RANSAC, then LM) on the keyframe's 3D points at the pair's
  matches. The renderer forward-splats the keyframe's background into t with
  a z-buffer, fills cracks of at most 3×3, then warps colour backwards.
  `tri_raw` renders only the triangulated pixels, so its holes show how much
  of `tri` rests on the fill. A frame t is excluded only when it is a
  companion of its own reference, the circular case;
  (c) `da3`, Depth Anything 3 depth of the keyframe, monocular. It uses the
  same PnP and renderer, runs only if the rule below reaches (c), and is
  audited into the environment first;
  (d) `kf`, depth-augmented keyframes as the representation: the better of
  `tri` and `da3`, rendered from the reference and the nearest other
  keyframe, merged by z-buffer. 3DGS is tried only if (d) leaves holes (below).
- Measures, per method and frame, are G1's `camera.residual` on the rendered
  region of the background: p90 residual flow on textured background at
  1080p (≤ 2 px), shares within 1/2/4 px, PSNR raw / after gain / after
  flow, and the attribution (exposure, parallax by the pair's epipolar test,
  independent motion, remainder). Before the flow is measured, each warp's
  map is extrapolated into its invalid pixels (`depth.complete`), so that
  holes inside the warped image do not drag the flow at their edges; the
  scored region is unchanged. Beside them: the hole share (the background
  both views see, `h1`'s region, that the method does not render), the share
  of > 2 px pixels within 32 px (1080p) of the foreground, the rate in bytes
  (`rot` 4 float32 per frame; `h1` 8; planes 8K per frame plus the PNG label
  map per keyframe; `tri`/`da3` 6 float32 per frame plus the PNG of the
  8-bit inverse depth per keyframe, its 16-bit size beside it; `kf` also the
  keyframe's colour as JPEG q90, which every keyframe representation pays),
  and the render time per frame (one CPU thread, warp construction only). A
  frame is *explained* when its p90 is ≤ 2 px **and** its hole share is
  ≤ 10%. Per clip, shares are over G1's measured frames, and a frame a
  method cannot render counts as not explained.
- Decision rule (`g1d.DECISION`, fixed before any fleet run). A method
  brings a group under 2 px *on a meaningful share* when its median clip
  explains at least 50% of frames. Windows and stretches are judged
  separately; clips with ≥ 90% (G1's clip bar) are reported beside. (1) If
  neither `epi` nor `tri` has a meaningful share in either group, no
  static-scene warp at hand brings VISOR under 2 px. Then (c) and (d) are
  not run, and egocentric goes to G5. (2) Otherwise (c) and (d) run. The
  chosen representation is the realizable method (`planes*`, `tri`,
  `tri_raw`, `da3`, `kf`) with the fewest bytes that has a meaningful share
  in both groups: egocentric video then gets that background in G3. If none
  qualifies, G5 leads, and the gap between the best measure (`epi` or `tri`)
  and the best realizable method is recorded. (3) 3DGS is tried only if the
  chosen method's median hole share exceeds 5%. In every case, the
  composition of the residual left by the best methods answers what the
  remaining residual is made of.
- Hypothesis: planes help little (median window explained ≤ 20% at K = 4).
  Depth reaches a meaningful share on the windows but not on the stretches,
  where references are older and other people, water and lighting change.
  So no realizable representation qualifies on both groups, and G5 leads for
  egocentric video. What remains is mostly independent motion near the
  hands and a photometric remainder.
- Competing explanations: (1) a measurement floor: DIS flow between views
  far apart in time or lighting reports misalignment where the geometry is
  right. Tested by the residual against the time to the reference, by the
  PSNR after flow, and against OpenTTGames' static floor (G1: 1.15 px p90).
  (2) Rolling shutter: no single pose fits, so the residual grows with
  rotation speed (Spearman). (3) Mask leaks: residual next to the foreground
  (the near-foreground share). (4) Calibration: `epi` and the planes do not
  need the focal length; `tri` does, and the self-calibration's error at
  EPIC Fields' lens shows how well that lens fits each clip.
- Budget: preparation CPU job (stage the 34 videos, 73 GB, plus the masks;
  decode 34 windows and 10 stretches) ≤ 2 h at 16 threads. Evaluation CPU
  jobs: smoke ≤ 600 s (2 windows and 1 stretch, 40 frames each), pilot on
  the same clips in full ≤ 1 h to measure seconds per frame, full ≤ 3 h at
  48 threads. DA3, if reached: one GPU job ≤ 1 GPU-hour on Ada or A6000,
  including its first-run kernel check. Ceiling 2 GPU-hours, 8 CPU-job-hours,
  14 h wall.
- Before any fleet run (dev checks on gpu6 with the packed environment, the
  pilot clips capped at 40 frames; not evidence). The preparation round trip
  is lossless, and its foreground equals G1's on every frame. Five changes
  followed. (i) The first lens was the per-clip self-calibration. It drifted
  to the grid's edge (37° and pincushion k1 on two of three clips), because
  near-rotational pairs let a long focal length fit the essential matrix
  too; EPIC Fields' calibration replaced it. (ii) Holes inside a splatted
  image are black, and DIS flow dragged at their edges: `tri` measured
  12.5 px on a pair whose own correspondences it reproduced within 1.1 px
  (p90). Hence `depth.complete`. G1 never met this, because its warps leave
  holes only at the border. (iii) Telea inpainting of inverse depth put
  8–15 px (p50) of error into the 16% of textured pixels it filled. Local
  plane fits brought them to 2.7–4.9 px, and `tri` on that pair from 18 to
  4.5 px. A second companion with 129 pose inliers had also overridden good
  pixels, hence the agreement test. (iv) Sparse SIFT (about 500 matches)
  supports two planes at most, so `planes3/4` equalled `planes2`; dense
  fitting separates them (P02_12 window: 7.7, 4.4 and 3.6 px p90 for K = 2,
  3 and 4, against 13.5 for `h1`). (v) `epi` was meant as the ceiling, but
  `tri` explained all 37 frames of P03_120 where `epi` explained 57%, so the
  rule's step 1 now asks of both. Reproduction of G1: references and rotation
  p90 matched on every frame (tolerance 0.05 px).
- Preparation `20261009T075810Z-295529d4` (gpu6, CPU, `499998c`; smoke 70 s,
  full 312 s; validator passed on smoke and full output, 8 checks): 44
  clips, 7.0 GB of FFV1 (`published.tar` `5a9c47db…5d62`). The round trip
  is lossless, the foreground equals G1's on every frame, and the window
  JPEGs match. Hosts gpu1, gpu3, gpu5 and gpu6: gpu2 refused SSH (no worker
  heartbeat), and gpu4 is left out because it is locked for measurements.
- Pilot `20261009T080919Z-2aa14966` (gpu1, CPU, `0a5bb1f`; smoke gate
  passed, 10 checks; full 623 s at 32 threads, about 950 frames; validator
  passed on the full output). G1's references and rotation p90 reproduce on
  every frame. Explained share (p90 ≤ 2 px, holes ≤ 10%):

  | Clip | rot | h1 | planes2 / 3 / 4 | epi | tri | tri_raw (ignoring holes) |
  |---|---:|---:|---|---:|---:|---:|
  | window P02_12 | 0.07 | 0.17 | 0.22 / 0.22 / 0.23 | 0.35 | 0.15 | 0.20 |
  | window P03_120 | 0.11 | 0.14 | 0.19 / 0.20 / 0.20 | 0.70 | 0.20 | 0.70 |
  | stretch P26_02 | 0.01 | 0.03 | 0.05 / 0.04 / 0.05 | 0.08 | 0.02 | 0.10 |

  Over a whole window, every residual grows with the time to the reference
  (Spearman 0.76–0.93; the windows' references are their first frame).
  Depth triangulated 0.5–1 s after the keyframe does not hold seconds
  later, so `tri` falls far below its 40-frame dev check. The full run of
  the other 41 clips is sized from this (about 18,500 frames, about 1 h at
  48 threads): `20261009T082847Z-36c4a84c`, same revision, so the pilot's
  clips stand.
- Full run `20261009T082847Z-36c4a84c` (gpu6, CPU, `0a5bb1f`; smoke gate
  passed, 10 checks; full 2,137 s at 48 threads, no contention; validator
  passed on the full output). G1's references and rotation p90 reproduce on
  every frame of all 44 clips. Report:
  `pointstream-data/background/g1d-2026-10-09/report-b009870/`
  (`g1d-report.json` `3cf1d238…07e7`, from the pilot's and this job's
  `g1d.json`). Medians over clips (explained: p90 ≤ 2 px and holes ≤ 10%;
  energy: share of the raw squared error, mean over frames):

  | Group | Method | explained (median clip) | clips ≥ 50% | p90 px | holes | bytes/frame | render s | parallax / independent / remainder / exposure |
  |---|---|---:|---:|---:|---:|---:|---:|---|
  | windows (34) | rot | 0.06 | 0 | 13.9 | 0.01 | 16 | 0.11 | 0.42 / 0.28 / 0.24 / 0.06 |
  | windows | h1 | 0.08 | 1 | 8.2 | 0.01 | 32 | 0.09 | 0.36 / 0.27 / 0.31 / 0.06 |
  | windows | planes2 / 3 / 4 | 0.14 / 0.14 / 0.16 | 1 / 3 / 3 | 5.7 / 4.8 / 4.4 | 0.02–0.03 | 343 / 451 / 535 | 0.44–0.46 | (K = 4) 0.28 / 0.24 / 0.42 / 0.06 |
  | windows | epi | 0.34 | 8 | 2.7 | 0.03 | – | 0.40 | 0.12 / 0.29 / 0.51 / 0.07 |
  | windows | tri | 0.09 | 2 | 5.5 | 0.03 | 4,042 | 0.49 | 0.28 / 0.28 / 0.37 / 0.07 |
  | windows | tri_raw | 0.00 (holes) | 1 | 2.7 | 0.35 | 4,042 | 0.33 | 0.25 / 0.14 / 0.53 / 0.08 |
  | stretches (10) | rot | 0.005 | 0 | 28.9 | 0.01 | 16 | 0.11 | 0.43 / 0.31 / 0.17 / 0.09 |
  | stretches | h1 | 0.016 | 0 | 22.3 | 0.01 | 32 | 0.09 | 0.36 / 0.34 / 0.22 / 0.09 |
  | stretches | planes2 / 3 / 4 | 0.04 / 0.04 / 0.04 | 0 | 16.9 / 14.4 / 13.2 | 0.02–0.04 | 888 / 1,224 / 1,465 | 0.51–0.53 | (K = 4) 0.28 / 0.33 / 0.30 / 0.09 |
  | stretches | epi | 0.09 | 0 | 6.6 | 0.02 | – | 0.40 | 0.11 / 0.37 / 0.42 / 0.10 |
  | stretches | tri | 0.02 | 0 | 14.8 | 0.04 | 12,406 | 0.57 | 0.29 / 0.33 / 0.29 / 0.09 |
  | stretches | tri_raw | 0.00 (holes) | 0 | 6.1 | 0.42 | 12,389 | 0.35 | 0.31 / 0.20 / 0.38 / 0.10 |

  Per clip, `epi` explains 2–100% of a window's frames (8 of 34 at least
  half, one all) and 5–27% of a stretch's. `tri` explains at most 68% of a
  window and 12% of a stretch. Bytes per frame count the per-frame
  parameters plus each used keyframe's map, spread over the frames that use
  it; stretches have a keyframe every few frames, hence their larger rates.
  Render times are one CPU thread in numpy, warp construction only.
- Outcome by the rule: **step 1**. Neither `epi` (median clip 0.34 on the
  windows, 0.09 on the stretches) nor `tri` (0.09, 0.02) reaches a
  meaningful share in either group. No depth-aware warp at hand brings
  VISOR's background under 2 px on a meaningful share of frames, so (c)
  (Depth Anything 3) and (d) (depth-augmented keyframes) were not run, and
  3DGS does not arise. **Egocentric video goes to G5**, the neural
  background model. Code for (c) and (d) (`experiments/background/da3.py`,
  methods `da3`, `da3_tri` and `kf`) is in the repository but has never run.
  DA3 was not audited into the environment and no weight was downloaded.
  The plan for that audit is a pure-Python overlay (DA3 at `3d835ec`,
  `addict`, `omegaconf`) with import stubs for `pycolmap` and `evo`, which
  monocular inference never calls.
- What depth removes and what remains. Depth does what it should to the
  geometric part. A warp that follows each pixel's epipolar line (`epi`)
  cuts parallax from 42% to 12% of the error on the windows (43% to 11% on
  the stretches), and the median p90 from 13.9 to 2.7 px (28.9 to 6.6 px).
  Four planes reach 4.4 px (13.2), so planes take a third to half of the way
  at about 535 bytes per frame. A representation built once from a keyframe
  does not hold, though. Depth triangulated 0.5–1 s after the keyframe
  (`tri`) is right where it was measured (`tri_raw`: p90 2.7 px on the
  windows), but that is 58–65% of the background, and the plane-filled rest
  pulls it back to 5.5 px. Every residual grows with the time to the
  reference (Spearman about 0.55–0.69 on the windows, whose reference is
  their first frame, and 0.36–0.42 on the stretches), and barely with the
  rotation speed (0.11–0.28), so rolling shutter is not the cause. What
  remains after `epi`, the most generous static-scene warp, is not geometry
  a better depth could fix. On the windows (stretches) it is 29% (37%)
  independent motion, 51% (42%) remainder (lighting, blur, noise,
  disocclusion), 7% (10%) exposure, and 12% (11%) parallax. Mask leaks are
  not it: only 4–7% of the misaligned pixels lie within 32 px of the
  foreground. PSNR after flow stays at 31.9 dB (30.2 on the stretches).
  Competing explanation (1), a measurement floor at wide time gaps, is not
  excluded: the residual's growth with the time gap fits both a scene and
  appearance that change and a flow that degrades. OpenTTGames' static floor
  (1.15 px p90) shows the measure can reach the bar on a static scene.
  Calibration (4): the self-calibration ranges 41–96° per clip against EPIC
  Fields' 86.7°. It is a diagnostic, but `tri`'s focal length is the one
  untested assumption. `epi` does not depend on it, and the decision rests
  on `epi` failing too.
- Hypothesis: partly right. Planes helped little (window median 0.16 at
  K = 4, inside the ≤ 0.20 predicted). Depth did *not* reach a meaningful
  share on the windows (predicted it would): the windows' reference is their
  first frame, up to 4.8 s back. The remainder is larger than predicted, and
  independent motion near the hands is not where it sits.
- Budget: CPU only, 0 GPU-hours (ceiling 2). CPU jobs: preparation 312 s,
  pilot 623 s, full 2,137 s, so about 50 min of CPU-job wall (ceiling 8
  CPU-job-hours). Wall time from the first submission to the report: about
  1.5 h (ceiling 14 h).
- After review (2026-10-09). (1) *Scope of the verdict.* The rule judged G1's
  pairs, where a frame keeps one reference for up to 4.8 s on the windows
  and seconds on the stretches. It rules out a background sent once and
  reused for seconds *on VISOR*. Racket sports are outside it: G1 found
  their fixed cameras hold a panorama sent once (OpenTTGames 6 of 7 clips,
  first frame already 90% of every later background). It says nothing about
  references kept fresh. Binned by
  the age of the reference (`g1d ages`, from the two runs' published
  results, no new computation; `ages-fc4ccdf/g1d-ages.json` `ad72bbe5…f6ef`):

  | Windows: reference age | frames | rot | planes4 | epi | tri |
  |---|---:|---:|---:|---:|---:|
  | < 0.1 s | 1,186 | 37% | 49% (p90 1.5 px) | 70% (1.3 px) | 33% |
  | 0.1–0.25 s | 922 | 10% | 35% | 43% | 19% |
  | 0.25–0.5 s | 883 | 4% | 29% | 37% | 16% |
  | 0.5–1 s | 1,203 | 2% | 20% | 33% | 9% |
  | 1–2 s | 1,601 | 1% | 8% | 24% | 6% |
  | 2–5 s | 1,742 | 2% | 11% | 25% | 10% |

  On the stretches, the best bin with many frames (0.1–0.25 s, 2,166 frames)
  reaches 18% (`epi`) and 10% (`planes4`). A bin's share also depends on
  which clips have frames at that age, so this is indicative. It moves the
  question to how often a reference must be refreshed and what the refreshes
  cost, which is G1e. G5 is not decided yet. (2) *The oracle was not a bound.*
  `epi` was meant to bound every static-scene warp, but `tri` beat it on one
  pilot clip, because `epi` inherits the per-pixel flow's errors. The
  protocol now requires an oracle to dominate what it gates (PR #180). The
  verdict still stands for G1's pairs, since `tri` and the planes failed
  too. (3) *A mask confound on the stretches.* Their foreground is G1's
  `sam_text` tier, which recalls about half of the dense masks' hand pixels
  (G1 tier check). Unmasked hands then count as background motion. That may
  be why the stretches stay far below the windows even with fresh
  references, and G1e checks it.
