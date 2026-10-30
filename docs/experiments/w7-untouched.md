# Experiments: W7

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[W1](w1-roi-pareto.md), [W5](w5-med-pareto.md), the
[fresh-evaluation card](../cards/k6b-fresh-evaluation-plan.md) and PLAN's
foreground-first claim.

### 2026-10-30 — W7: the foreground-first claim on untouched footage (rule)
- Question. W1 (Al) and W5 (Med) found that no SVT or DCVC-UF point reaches
  PointStream's (whole-frame, player) quality pair at equal total rate, and
  that PointStream's player region is far better. Both windows were used
  for development many times. Does the player win hold on footage that no
  experiment has looked at, and on a match the tennis base never trained on?
- What "seen" means here. The tennis base (decoder weights, rank-64 LoRA on
  AnimateAnyone `original`) trained on 36 windows of five matches. Each
  match adapter (sent in the bitstream and billed) trained on other clips
  of its own match. A per-match adapter fitted by the encoder on the same
  match is the codec's normal encode step, not overlap, as long as no
  evaluated frame is in its fit. The base must not have seen the evaluated
  footage for the clean tier.
- Tier 1, seen matches, untouched clips. Alcaraz–Hurkacz and
  Medvedev–Struff clips with at least 240 native frames that are outside
  every old manifest window, every step-5 fit clip and the twelve
  2026-10-25 confirmation windows (k6b-players.md, about 3 and 7 clips).
  Three windows per match, chosen mechanically before any quality is
  computed: every admissible Al clip in ID order up to three; Med clips by a
  seeded draw (seed 20261030) from the admissible list; in each clip the
  first 240-frame interval the frozen registration rule of the 2026-10-25
  windows admits. Clips that fail registration are listed, not replaced by
  hand. Existing Al and Med adapters and reference banks.
- Tier 2, new match. `nadal_kyrgios` (archive, 2,310 s), after a lineage
  check that no base, adapter or reference input contains its frames (W8
  shares this check). Adaptation split: the first half of the broadcast
  (adapter fit with the frozen step-5 recipe, reference bank from the same
  half, all billed). Evaluation: three windows from the second half by the
  same mechanical rule, seed 20261030. Its court marks were used in K8/K8b
  design; registration exposure is disclosed, generator exposure is none.
- Pipeline: the frozen W5 recipe (scale-4 paid SVT base, KPS2 keypoints,
  4-bit adapter, 12-image reference bank, receiver pose-prompted SAM masks,
  generated players on every third frame), unless W6's full-rate generation
  has passed its own rule before W7's first GPU job; the choice is recorded
  here before that job, never after seeing W7 quality.
- Anchors per window: plain SVT-AV1 at half and quarter resolution, SVT
  half with +20 QP outside R (the strongest player policy in W1/W5), and
  DCVC-UF, each with points chosen on rate alone to bracket the hybrid's
  rate. Judges: LPIPS and DISTS (ImageNet networks; no tennis frames seen),
  whole frame and fixed player region R, same controls as W1.
- Decision rule (primary, per tier). Per window, the player gap
  Δ = (best anchor's player-region score at the hybrid's rate) − (hybrid's),
  log-rate interpolation between bracketing points only. The player win
  holds on a tier if the mean Δ over its windows has a 95% t-interval above
  zero on LPIPS and on DISTS, with windows of one match reported as a
  cluster beside the pooled interval. Secondary, reported for every window:
  W1's dominance outcome (A or B) per anchor family and the whole-frame
  difference at equal rate.
- Narrowing, written before results. Tier 1 holds, Tier 2 fails: the claim
  narrows to matches the decoder's base has seen other footage of (a
  tournament- or league-specific decoder); next test: per-match adaptation
  strength on the new match (adapter rank, fit length) within the billed
  rate. Both fail: the claim narrows to the development windows and the
  paper reports a negative generalization result; next test: which stage
  fails (registration, pose, generator) from the per-window breakdown.
  Tier 1 fails on one match only: report the match, narrow to the other,
  next test: what distinguishes them (player size, motion, registration).
- Hypothesis: the player gap holds on both tiers; it shrinks on Tier 2
  because the base never saw those players or that court.
- Competing explanation: the development windows are easy cases (large,
  well-lit players, stable registration) chosen and tuned over many runs,
  and the gap collapses on new windows.
- Budget. CPU stage (inventory, lineage, selection, registration, base and
  SVT encodes): CPU only. GPU stage, staged: one Tier 1 window end to end
  first (≤1 GPU-hour, also the smoke), then the other five (≤4 GPU-hours),
  then Tier 2 including the adapter fit (≤3 GPU-hours). Every stage ≤45 min,
  checkpointed per window.
- Jobs: (none yet)

### 2026-10-30 — W7 CPU stage: Tier 1 selection, overlap screen, Tier 2 lineage, pipeline map (no GPU, no quality)
- Code: `experiments/tennis/w7_select.py` (`select`, `audit`), tests `tests/test_w7_select.py`. Run on gpu1 with scratch
  in `/local/users/emanuele/pointstream/w7` (CPU only, ffmpeg `/opt/local/bin`). Saved results (small copies):
  `docs/experiments/w7-specs/selection.json` (sha256 `e8803edafc2466e89318d78d24971dfd9ae45f1ef7e57601ef7190f5fc041951`) and
  `overlap-audit.json` (`b6ecd6df8029bc419b63dc174284eb41b24873bcf13d244552ed147ba80892cc`).
- Inventory. The archive has 21 Al and 41 Med clips (K8b report `8230bf1a…fdc3314`; every file's sha256 and decoded
  frame count re-checked against it and ffprobe). Excluded: the clips of the 57 old manifest windows (scope.json
  `ac640ae8…`, 49 clips), step-5 fit clips, the five unused-manifest Med clips, the twelve 2026-10-25 windows' clips.
  With at least 240 frames and not excluded: **Al 028 (492 frames), 034 (619), 043 (246); Med 000 (1914), 004 (355),
  019 (458), 024 (435), 027 (675), 031 (305), 042 (462)**. (Matches the 2026-10-24 count: 9/13 with ≥240 minus the 6/6 confirmation clips.)
- Selection. Al: the three, in ID order. Med: `numpy.random.default_rng(20261030).choice(sorted(admissible), size=3,
  replace=False)` = **000, 024, 019** (numpy 1.26.4). Registration: the frozen rule of the 2026-10-25 windows (K8b `fits` and a
  finite nonsingular homography on at least 228 of 240 frames, `k6b_confirm.gate`), applied from the K8b per-frame rows
  to every start (stride 1); the earliest admitted start is taken. Reading chosen: "first interval" = earliest start s over all
  starts, not only scene-aligned ones. All six start at frame 0 with 240/240 valid; none failed registration.

  | Clip | Start | Frames in clip | fps | Source sha256 |
  |---|---:|---:|---:|---|
  | alcaraz_hurkacz/028 | 0 | 492 | 25 | `c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934` |
  | alcaraz_hurkacz/034 | 0 | 619 | 25 | `7db0c143cd416ad17b4bda4eaa191d51da197eccc1143e9eb402f8bb2a7f0ee6` |
  | alcaraz_hurkacz/043 | 0 | 246 | 25 | `847cb8272134bd31a5447ff3cce02b28a906fae44559b3e0a58661d8171b6c16` |
  | medvedev_struff/000 | 0 | 1914 | 50 | `3bf7ddaae0db5c48b1072fccb15f9f162e41fa5999886b9a5bde58dc0495c630` |
  | medvedev_struff/024 | 0 | 435 | 50 | `4d602291ebea0fcb9698d86d3f93ce4c3076a13005e33079a82975b8b64605f2` |
  | medvedev_struff/019 | 0 | 458 | 50 | `1c13479eb3b6f8d175eb6f5020c629896e6b01f719014ec995a06b4c6c5a3e9c` |
- Overlap screen (the 2026-10-24 code `tools/datasets/k6b_recovery.py`, same descriptors and decoder): each window against the
  49 old clips (descriptors reused from the 2026-10-24 collections) and the twelve confirmation clips decoded whole, 67 clips and
  43,814 frames in all. 0 exact native yuv420p SHA256 matches, 0 frames at 64×36 RMSE ≤3. Closest: Med 019 vs 033 (RMSE 3.82; viewed
  at 640×360: same fixed camera, different score and positions, not the same frame), Med 024 vs 033 (4.04), Med 000 vs 006 (5.60), Al 034 vs 036 (5.42).
  As on 2026-10-24 this screen does not rule out altered replays or original-source overlap; same-broadcast, clip-local only.
  Consequence worth stating: Med 019 and 024 sit nearest to the development clip 033 (the same court and camera as all Med footage).
- Tier 2 lineage (job specs, ready.json heads, tar listings). Base job `20261023T194934Z-b0dcbb19` trains on eight prepared
  fit-data tars (36 windows: Al 3 (002, 006, 011), Med 2 (006, 008), djokovic_zverev 11, rublev_auger-aliassime 13, djokovic_federer 7), the AnimateAnyone `original` weights and nothing else; excluded windows are
  the eight K6 windows. Adapter fits `20261023T221650Z-3d7cbe2f` (Al: fit data from the Al 002/006/011 job) and
  `20261023T204452Z-d5c90446` (Med: Med 006/008) start from the base's published.tar (sha256 `075b001a3ae9…`, equal to the file hashed on
  the server). The prepare jobs (`…093741Z-1ce1d61b`, `…093749Z-1f0371dd`, `…094453Z-6f165e95`, `…094307Z-a06b224a`, `…192031Z-c32a29a2`,
  `…192023Z-6e446ad0`, `…192013Z-a58b341c`, `…192005Z-64582992`) list only windows of the five scene-clip matches; `k6-2026-10-21/oracle/inputs.tar`
  (172 members) contains none either. "nadal" occurs in no spec of the base, adapters or prepare jobs and in no tar member name.
  **Verdict: no nadal_kyrgios frame is in any generator training input of the base or the Al/Med adapters.** Not established:
  pretraining of AnimateAnyone's `original` weights, SAM 3.1, YOLO11x-pose and the judges (unknown, as for every tier); K4's camera
  measure (every 0.5 s) and K8's court marks (82 shots) used nadal frames for design, not for any generator input.
- nadal_kyrgios source: `archive/pre-reset-2026-10-05/tennis_games/raw_1080p/nadal_kyrgios.mp4` (hard link of
  `tennis/k8-2026-10-16/nadal_kyrgios.mp4`), sha256 `676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66`, H.264 1920×1080
  50 fps, 2310.10 s, 115,505 frames, 1,048,512,568 bytes. It is a whole broadcast, not scene clips. A K8 v0 court-shot list exists
  (`k8-2026-10-16/court-shots-v0/k8-court-shots-v0.json`: 41 predicted court shots, 449 s of about 1,300 s of court time; K8 not adopted). The
  K8b method has not been run on it.
- Not run (CPU plan for Tier 2, to be decided before the GPU stage): scene scan (ffmpeg scene score ≥0.25, `k8b.DECISION`) over the whole file,
  shots ≥240 frames as the "clip" list, K8b registration (`k8b.register_chunk`, 200-frame chunks) over the second half
  (frames 57,753–115,504), clips cut with the recovered `split_scenes.py` logic (commit `db11965e`, re-encode) or by `trim` as K8b
  decodes. Cost: K8b registered 38,142 frames in 485–564 s per part at 16 processes (about 9 frames/s per 16 processes), so the
  second half is about 6,300 s at 16 processes, about 26 min at 64, plus a 3–5 min scene scan; roughly 28 CPU-hours.
  Open choices for the lead: what the seeded draw ranges over on nadal (proposal: the shots ≥240 frames of the second half, as for
  Med) and whether a window may contain a cut (proposal: shots only, no cut inside).
- Pipeline for one window (Al031@129, Med014@194) and what a new window needs: recorded in the report of this task; summary in `docs/experiments/w7-specs/pipeline-map.md`.
- Jobs: none (CPU on gpu1 outside the fleet, attended).
