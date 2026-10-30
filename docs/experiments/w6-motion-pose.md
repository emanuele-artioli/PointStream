# Experiments: W6

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md), J's experience
judges ([j.md](j.md)).

### 2026-10-29 — W6: are PointStream's players stable and in the right pose? (rule)
- Question. The visual check (2026-10-29) showed sharp generated players
  but a wrong pose and a missing racket at Al frame 120, and the hybrid
  exists only on every third frame. Are the players stable over time at
  the full frame rate, and is their pose as faithful as the anchor's?
- Step 1, full-rate hybrid. Generate the missing 160 frames per player on
  Al with the frozen generator, seeds and steps of the real-base recipe;
  first regenerate 6 of the saved ticks and require them to match the
  saved crops (max abs diff ≤ 1 level, else report the difference and
  continue with the rerun as the reference for all 240 frames), receiver
  masks for all frames, composite on the paid scale 4 base. Then Med the
  same way if the Al stage fits the budget.
- Step 2, J's judges on all 240 frames, unchanged (`j_players.py`, user's
  readings of 2026-10-22): pose — YOLO11x-pose on the method's frames and
  the source, mean joint distance as a share of the box; time —
  `h3b.flicker` against the anchor, and residual energy against 1.25 × the
  source's. Anchor: plain SVT-AV1 at the hybrid's rate (W1's half
  resolution encode at the bracketing CRFs), and DCVC-UF where W3b gives
  a point at that rate. Racket: reported beside, as the share of frames
  in which the source racket region (from the source mask) overlaps the
  generated player.
- Decision rule (J's, per window): pose passes if the method's error ≤
  max(the anchor's, 1.5 × the coded keypoints' own error against the
  source); time passes if `h3b.flicker` ≤ the anchor's OR residual energy
  ≤ 1.25 × the source's. Report each check separately; a fail narrows the
  claim (e.g. "sharper players, less faithful pose") rather than ending
  it.
- Hypothesis: pose passes against SVT (SVT's players are blurred, so the
  estimator fails on them); time fails on residual energy (J found
  generated players change 1.3–6× as much as the source) but may pass
  against SVT's flicker.
- Competing explanation: the generator's pose errors exceed SVT's blur
  errors, and its frame-to-frame changes exceed SVT's flicker.
- Budget: ≤1.5 GPU-hours (generation ≈820 GPU-s per window on A6000 plus
  masks and judges); every stage ≤45 min, saving each finished frame.
- Jobs: (none yet)

### 2026-10-29 — W6 implementation choices (written before any W6 job ran; the rule above is unchanged)
- Code: `experiments/tennis/k6b_generate.py` (`--keep-all`), `w6.py` (receiver
  masks, judges, racket measure, clips, report), `w6_specs.py`, tests
  `tests/test_w6.py`. Decisions the rule left open, made before any score:
  1. **Regenerate the whole 240-frame run of each player, not 160 frames.**
     The released long-video sampler renders 24-frame contexts overlapping by
     4, so the saved ticks (every third frame) belong to one 240-frame
     sequence and frames cannot be rendered alone. The rerun therefore
     reproduces all 80 saved ticks per player (not 6) in the same pass;
     cost is the original generation job's (`20261026T075716Z-f0549307`,
     1,264 s for both players on Ada), not 820 s. If the 80 ticks match the
     saved crops within 1 level, the rerun is the 240-frame set; if not, the
     difference is recorded and the rerun is still used for all 240 frames.
  2. **Ada, not A6000.** The model–GPU table has Moore-AnimateAnyone on
     RTX 6000 Ada only ("A6000 not run") and the saved crops came from Ada
     (gpu5/gpu6). A6000 would be a first run of the model on that class and a
     weaker reproducibility test. One job per player, shared-GPU class
     (outputs do not depend on wall time).
  3. **Masks.** `k6b_receiver_sam`'s isolated worker, every-frame fresh
     prompts, once per player (the earlier job ran it twice for a reset
     check; the 80 saved ticks are compared with its saved masks instead).
  4. **Anchor.** W1's plain half-resolution SVT-AV1 (policy `s2_o0`) at CRF 65
     (24.64 kbps) and 66 (22.28 kbps), which bracket the hybrid's 22.737962
     kbps; every per-frame quantity is combined linearly in log rate, never
     extrapolated. DCVC-UF: no W3b point exists at this rate; not used.
  5. **Pose judge.** `j_players.Judge` (YOLO11x-pose, K4b.2's crop path), on
     the source frames and on each output frame, fitted to the player's
     source region mask from the transport's exact matte (body plus touching
     racket, as the paid S/R masks); error = mean distance over the keypoints
     confident in the source's re-run over the region's height at that frame
     (K6 used its pose fit's box; the denominator is the same for every
     variant, so the ratios and the pass test are unaffected); a player not
     fitted counts as 1 (`j.MISSED`). Coded-keypoint floor from the
     transport's decoded keypoints, 1.5×. Per window means over the 240
     frames and both players.
  6. **Time judge.** `visor.j.residual_energies` over all 239 consecutive
     pairs, on the union of the source player regions, flow from the source
     (as H3e); outputs are whole frames (hybrid: paid base plus pasted
     players; anchor: the decoded stream upscaled by the paid cubic), judged
     on the source region. Pass by J's OR reading. A moving-block bootstrap
     (24-frame blocks, 2,000 resamples) gives 95% intervals for
     method−anchor beside the means (memory: significance, not threshold).
  7. **Racket share.** The transport holds one region mask per player (body
     plus touching racket), no separate racket mask. Racket region :=
     source-region pixels farther than 10% of the player's height from the
     dilated convex hull of the coded keypoints, counted only where ≥100 px.
     Reported: share of (player, frame) cells with a racket region in which
     the receiver's predicted mask overlaps it at all, covers ≥50% of it,
     and the mean coverage. Reported beside; it decides nothing.
  8. Clips: 25 fps, x264 crf 12, three 960×540 panels (source | plain SVT
     CRF 66 | hybrid), and a 2× zoom of the near player (480×270 windows,
     smoothed centre).
- Budget check: generation ≈ 1,300 s + masks ≈ 400–900 s + judge ≈ 600 s ≈
  0.7 GPU-h before smokes; Med only if Al finishes within 0.75 GPU-h.
- Jobs: generation, one per player, submitted 2026-10-29 19:28 UTC from commit
  `2b2f6eed` (Ada, shared): inbox entries `20261029T192854Z-225a66d0` and
  `20261029T192855Z-09c2200b` (snapshot transfer still in flight when
  recorded; a third entry `20261029T193013Z-1f570343` with a similar source
  archive appeared at 19:30 and may be a client retry or another session's).
  Inspect with `ps-fleet status JOB_ID`; not to be replayed.

### 2026-10-29 — W6 deviation: A6000 on gpu3 (before any W6 job ran)
- The two submissions above (`20261029T192854Z-225a66d0`,
  `20261029T192855Z-09c2200b`) never published: the snapshot extraction on
  gpu2 timed out after 300 s on both (inbox holds only `source.tar`, no spec
  or ready); they are abandoned and not replayed. `20261029T193013Z-1f570343`
  is not from this session's submit commands (both of which failed as above);
  not touched. The Ada workers gpu5 and gpu6 have no fresh heartbeat (`ps-fleet
  workers status`, 19:50 UTC), so the Ada plan (decision 2 above) cannot run.
- Generation therefore runs on gpu3's RTX A6000, as the brief said, with
  decision 1 unchanged. This is Moore-AnimateAnyone's first run on the A6000
  class: the generator's built-in probe records kernels (flash attention,
  CUTLASS) and the result goes into the model–GPU table. Expected
  consequence: the 80 saved ticks may not match the Ada crops within 1 level;
  the rule is then to record the difference and use the rerun for all 240
  frames (it is). Memory declared 16,000 MiB (peak 12.2 GiB on Ada).
- Jobs, inspected on gpu3's inbox at 20:00 UTC (spec/ready present = published).
  Unpublished (only `source.tar`; never ran, not replayed):
  `20261029T192854Z-225a66d0`, `20261029T192855Z-09c2200b`,
  `20261029T193013Z-1f570343` (not ours), `20261029T194501Z-5bee16ea` (Ada
  attempt abandoned and the A6000 attempt both timed out extracting on gpu2,
  load 30). **Published**: generation, player `…person_000000000001`, A6000
  gpu3, commit `5585edff`: `20261029T195226Z-26f4c52d` (the client's publish
  call timed out at 120 s but it landed; state pending).
- `20261029T195226Z-26f4c52d` stayed pending for 3 h: another user holds 25.5
  GB of gpu3 (48 GB) and shared admission needed 16,000 + 8,192 MiB free
  (23.6 GB free). It never started (no GPU time) and was cancelled by me
  (own job). Resubmitted with 13,500 MiB declared (peak 12.2 GiB) and a
  3,072 MiB margin. `20261029T224503Z-6d849ba1` (client timed out; only a
  `source` dir) is unpublished and abandoned.
- Jobs (resubmitted): see below.
- Jobs (A6000, gpu3, shared; as of 2026-10-29 23:55 UTC): player `...0001`:
  `20261029T232858Z-1e077edd` (published, pending; 13,500 MiB + 3,072 margin, commit
  `ddff2f89`/`5585edff` code). Player `...0006`: `20261029T224503Z-6d849ba1` was
  published late with the old 16,000 + 8,192 MiB declaration and stayed pending; cancelled
  by me (never started), to be resubmitted with the new declaration. Unpublished and
  abandoned source-only entries: `20261029T225219Z-bf7f10e2`, `20261029T225231Z-909e690c`,
  `20261029T233317Z-a9932064`, `20261029T234020Z-fc00eba0`, `20261029T234654Z-4f27910e`,
  `20261029T234831Z-962535a8` (not replayed). Cancelled earlier: `20261029T195226Z-26f4c52d`.

### 2026-10-30 — W6 outcome so far: generation not completed; stopped at the budget gate
- No W6 score exists. Job `20261029T232858Z-1e077edd` (player `...0001`, gpu3 RTX A6000, shared, code `5585edff`): smoke PASSED (validator all true: A6000 class, keep-all, peak within 13,500 MiB, weights loaded, kernels on GPU, 8 frames in 36.8 s while another user held 25.5 GB and the GPU; timings contaminated, infrastructure only), then FAILED at the full-stage gate: "Remaining budget cannot support the saved stage estimate" (1,725 s remaining of the declared 3,000 s; staging the 12 inputs took the rest). The full stage never started; no frame was generated. Charged conservatively 1,275 s (0.35 GPU-h) of the 1.5 GPU-h, none of it compute beyond the 8-frame smoke. This is the first A6000 run of Moore-AnimateAnyone: it passes the smoke checks; the model-GPU table row and the 80-tick reproduction against the Ada crops remain open.
- Other jobs: `20261030T002644Z-291d68b3` (player `...0006`) unpublished (source only), abandoned. `20261029T224503Z-6d849ba1` and `20261029T195226Z-26f4c52d` cancelled before starting. No masks or judge job was submitted.
- Why stopped. The remaining budget (about 1.15 GPU-h) is below the plan: two generation jobs at about 1,300 s of staging plus 700-1,100 s of compute each, then masks and the judge, each with its own staging. Decision for the coordinator: authorize more budget (about 2.5 GPU-h total, of which staging is about half), or accept a plan with fewer stages. `w6_specs.py` now declares full 1,500 s and budget 4,000 s for generation; masks and judge specs are written but untested on the fleet.

### 2026-10-30 — W6 plan change after the stopped attempt (written before the next submission)
- Coordinator decisions (2026-10-30): the W6 cap is raised to 2.5 GPU-h in total (about 1,275 s already charged); masks and judge are merged into one job after generation so the inputs are staged once more, not twice (`w6 post`; `w6_specs.py` `post`); the SAM weights sha256 was read (read-only) and replaces the placeholder (`0567debe...cb6`); A6000 on gpu3 is acceptable for generation and is recorded as this model's first A6000 run (AGENTS.md first-run rule: the job's probe records the device, kernels and attention family, and the model-GPU table gets a row from the full job). The reproduction check against the 80 saved ticks stays. Branch merged with origin/main `5143dbd0` (snapshots no longer extract file by file on gpu2).
- Generation is now one job for both players (staging once), full stage 2,600 s, budget 4,400 s, 13,500 MiB with a 3,072 MiB margin; the generator saves each finished unit, so a stage that runs out keeps the first player. The decision rule, anchors and judges are unchanged. Masks and judge in one job: stage 2,600 s, budget 4,200 s; its smoke runs both players for 16 frames (masks limited to those frames).
- Jobs: (next entries)
- Jobs (2026-10-30): generation, both players in one job, A6000 gpu3, code `63c3bccb`: **`20261030T075853Z-f5e70b8e`** (published on the 5th submit; the publish call hashes several GB of inputs on gpu2 and times out at 120 s, warming the cache for the next try). Unpublished source-only entries from the same spec, not replayed: `20261030T072115Z-05dfef11`, `20261030T073555Z-1dcd0bf4`, `20261030T074316Z-2a5c3def`, `20261030T074739Z-cfc21842`, `20261030T075036Z-b9fd06bd`, `20261030T075457Z-bd849a9b`.
- 2026-10-30 08:25 UTC: `20261030T075853Z-f5e70b8e` stayed pending: other users hold 38.8 GB of gpu3 (13,500 + 3,072 MiB needed, 10 GB free). The generation spec now admits gpu3, gpu5 and gpu6 (A6000 or Ada; `--gpu-class RTX`, the device and class are recorded in the output). The pending job is cancelled once its replacement is published; the replacement's ID follows below.
- 2026-10-30 08:45 UTC: stopped, no generation has run. `20261030T075853Z-f5e70b8e` is still pending (gpu3: 38.9 GB held by another user, 10 GB free; it needs 16.6 GB). Its replacement (gpu3/gpu5/gpu6 spec, generated into the scratchpad as `specs5/generate.json`) could not be published: five submits timed out in the publish call (unpublished: `20261030T082338Z-f57a2954`, `…082748Z-387e2089`, `…083150Z-c58be6d3`, `…083533Z-a947b5ce`, `…083949Z-beef39c6`), then the client refused with "worker gpu5 has no fresh heartbeat". Nothing was cancelled this round. GPU-seconds charged in total: about 1,275 (the 2026-10-29 attempt); nothing since. No W6 score exists. Needed: gpu3 memory freed or the fleet publish path fixed (the publish RPC hashes the inputs on gpu2 within 120 s), then submit `specs5/generate.json` (regenerate with `python -m experiments.tennis.w6_specs OUT generate`) and `post` after it.
- 2026-10-30 12:10 UTC: publish fixed for this item without a fleet code change. The publish call hashed 8.2 GB of staged inputs on NFS (the 3.4 GB denoising UNet alone took 429 s to read) inside its 120 s SSH limit; five more submits timed out (unpublished, source only: `20261030T113413Z-fbd95a85`, `…113738Z-4cfdb20e`, `…114008Z-ac290210`, `…114254Z-c7468bd3`, `…114559Z-c2367711`). Fix: the specs set `verify_staged_inputs` (commit `6f10fef4`), and gpu2's verified local cache was filled once with the release's own `staging.stage_input` outside the call (all 11 inputs hashed on copy). The next submit published in 124 s: **`20261030T120630Z-9190e75a`** (code `6f10fef4` merged with main, gpu3/gpu5/gpu6, A6000 or Ada).
- Mistake, recorded: `20261030T075853Z-f5e70b8e` had been admitted on gpu3 at 11:44 UTC (status checked last at about 11:30 showed pending) and was cancelled at 12:08 while in its full stage, without re-reading its status first. Cost about 1,450 GPU-s. Its `full/partial.tar` keeps player `…0006` complete (240 frames, 863 s, npz sha256 `4dca7d70…d3bbe4`); the replacement regenerates it with the same seed, so the two are a cross-job reproducibility check. Charged so far about 2,725 GPU-s of the 9,000 s cap; the replacement may use up to 4,400 s.
- 2026-10-30 12:48 UTC: generation **`20261030T120630Z-9190e75a` COMPLETE** (gpu3, RTX A6000, peak 12.21 GiB, 38 min). Local validation of the full stage (validator code from the job's own source, on gpu2): 16/16 checks pass (both units, 240 frames each, 784×512, hashes, not flat, all generated, kernels on CUDA, no missing keys). Cross-job reproducibility: unit `…0006` is bit-identical to the cancelled job's saved copy (240 frames, max abs diff 0). Archive `full/published.tar` sha256 `031ea5d7b744a0c52f9ee769fb38d5b9d30340f567ca042c6079d4f9240f354c`.
- Post (masks + judges): first submit `20261030T130055Z-06aacb72` refused before publishing (unpublished, source only): the pose weights were declared under `Models`, outside the fleet data root. Fixed to the identical-hash copy `tennis/k4-2026-10-13/inputs/yolo/yolo11x-pose.pt` (commit after `4eff2518`). Published: **`20261030T130413Z-325dde3b`**.
- Post `20261030T130413Z-325dde3b` FAILED in its smoke (gpu6, 639 s after submit): the SAM worker's directory `scratch/isolated/<unit>` was created without its parent (`FileNotFoundError`). Code bug, no GPU result. Fix: create the parent and clear an unfinished attempt before each worker; also a typed `fillConvexPoly` call so mypy passes. Resubmitted below.
- Resubmitted post: **`20261030T132001Z-e3d70926`** (code `40f390a1`).
- Post `20261030T132001Z-e3d70926` FAILED in its smoke (gpu6): the mask-reproduction check indexed saved ticks beyond the 16 judged smoke frames. Fix: match on the frame numbers the masks file records. Resubmitted below.
- Resubmitted post: **`20261030T132449Z-75957f16`** (code `c3b90760`).

### 2026-10-30 — W6 outcome: pose passes, time fails (narrowed: sharp, correctly posed players that flicker more than SVT's)
- Jobs: generation `20261030T120630Z-9190e75a` (gpu3, RTX A6000, full 1,908 s, code `6f10fef4` + main); masks and judges
  `20261030T132449Z-75957f16` (gpu5, RTX 6000 Ada, smoke 115 s + full 410 s, ready.json code `c3b90760`; failed
  attempts `…130413Z-325dde3b` and `…132001Z-e3d70926` were code bugs in the smoke, fixed above). Validator re-run
  locally on the full stage from the job's own source: 18/18 checks pass. Judges archive sha256 `4d71f6cc…c353f1`.
  GPU time for W6 in total about 5,500 s (1.5 GPU-h) of the 2.5 GPU-h cap.
- Reproduction (step 1): the full-rate regeneration does not match the saved 80 ticks within one level (max abs
  diff 165 and 80 levels per player, mean 0.09 and 0.09, 7.5% of pixels differ, 0 frames identical); the saved
  receiver masks differ on every tick too. The long-video sampler's 24-frame contexts now cover all 240 frames, so
  each tick is denoised in a different context; within one recipe generation is deterministic (unit `…0006`
  bit-identical across two jobs). By the rule, the rerun is the reference for all 240 frames.
- Pose (YOLO11x-pose error as a share of the box, 240 frames, both players): hybrid **0.063**, SVT at the hybrid's
  rate 0.901 (interpolated between CRF 65/66), coded keypoints 0.026. Limit max(0.901, 1.5 × 0.026) = 0.901:
  **pass**; block-bootstrap 95% interval of hybrid − anchor [−0.90, −0.80]. Caveat: the detector misses SVT's
  players on 78–88% of player-frames (hybrid 1.0%), so the anchor's error is mostly failed detections. The hybrid's
  pose error is 2.4× the coded keypoints' own, i.e. the generator adds pose error beyond the keypoints.
- Time (239 frame pairs): `h3b.flicker` hybrid **28.6** vs SVT 18.6 (interval of the difference [+9.3, +11.4]);
  residual energy 23.1 vs limit 16.6 (1.25 × source 13.3; interval [+5.9, +7.5]). **Both fail.**
- Racket (beside): the generated player overlaps the source racket region in 40% of 270 player-frames with a
  racket, covering 3% of it on average (written pixels 4%): the racket is essentially absent, as the visual check
  showed.
- Narrowed claim, stated before looking further: at equal rate PointStream's players are sharp and correctly posed
  where SVT's are blurred beyond detection, but at the full frame rate they change more from frame to frame than
  SVT's players do (and than the source does), and the racket is not drawn. Next test: split the flicker between
  its two sources on the saved outputs, without new generation: the receiver masks (fresh per frame, no
  propagation) and the generated pixels (score the hybrid with the source matte, and with temporally smoothed
  masks). The larger share decides whether mask propagation or a temporal generator is the fix.
- Effect on W7: W6 did not pass its rule, so W7 keeps the frozen W5 recipe (players on every third frame).
- Clips: `full/clips/w6-side-by-side-source-svt-hybrid.mp4` and `w6-zoom2x-near-player-source-svt-hybrid.mp4` in
  the post job's stage directory.

### 2026-10-30 — W6b: where does the flicker come from? (rule)
- Question. W6's full-rate hybrid flickers more than SVT on 99.6% of frame pairs (28.6 vs 18.6), at a constant
  level over the 240 frames (quarters 28.7 / 27.7 / 29.7 / 28.5; no drift). Is the excess caused by the receiver
  masks (SAM fresh on every frame, no propagation) or by the generated pixels?
- Variants, on Al's 240 frames from W6's saved inputs (generation `20261030T120630Z-9190e75a`, masks and judges
  `20261030T132449Z-75957f16`), flicker and residual energy computed exactly as W6 (`h3b.flicker`, DIS flow on the
  source, the same region union, the same composite and base):
  - H: the hybrid. Control: W6's per-frame values reproduced to 1e-6.
  - G+S: generated crops, the source's player matte (from the source masks W6 uses for its region).
  - S+R: source pixels, the receiver masks.
  - G+Rs: generated crops, receiver masks smoothed over t−1, t, t+1 (majority vote in box-normalised coordinates).
  - B: the paid base alone, no players.
  The source matte and source pixels are diagnostics only, never a coded variant.
- Decision rule. Excess E(V) = flicker(V) − 18.6 (W6's SVT anchor at the hybrid's rate). Mask share
  = (E(H) − E(G+S)) / E(H); content share = (E(H) − E(S+R)) / E(H). The larger share, if it leads by ≥ 0.1,
  names the fix: masks → mask propagation across frames (G+Rs is its first, cheap estimate); content → temporal
  consistency in the generator (a video model or temporal smoothing of the generated crops). Both < 0.3 or within
  0.1 of each other: the flicker lives where generated content meets the base; next test, a feathered composite.
  Residual energy is reported beside with the same shares.
- Hypothesis: the masks dominate; SAM run fresh per frame moves the outline by a few pixels each frame.
- Competing explanation: the generated texture changes from frame to frame (W6 residual energy 1.7× the source's)
  and dominates whatever the outline does.
- Budget: CPU only (≤ 2 CPU-hours on one server, scratch in `/local/users/emanuele`), no GPU job.
