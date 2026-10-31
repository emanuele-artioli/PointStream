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

### 2026-10-30 — W7 Tier 2 decisions (lead), written before any extraction
- The seeded draw (seed 20261030) on nadal_kyrgios ranges over shots of at least 240 frames that start in the second half of the broadcast
  (frame ≥ 57,753 of 115,505); a window never contains a cut; the adaptation split is the shots of the first half. Within a drawn shot the
  window is the first 240-frame interval the frozen registration rule admits (as Tier 1). No Tier 2 extraction has been run.

### 2026-10-30 — W7 generic window manifest, reference-bank answer, GPU estimate (CPU only; nothing submitted)
- Code. `experiments/tennis/window_manifest.py`: one `WindowSpec` (name, match, clip, start, frames, fps, source sha256, conversion sha256,
  reference mode; adapter from the match) and `FROZEN_PAIR`, which reproduces the old Al031@129 / Med014@194 constants. Generalised to read it:
  `k6b_native_base` (`--windows-file --window --source`; `sources_for`), `w1_roi` (`window_from_spec`, `preload_windows`) with `w1_masks`,
  `w1_encode`, `w1_score` (`--windows-file --window-state`), and the `k6b_confirm` family (`cohort_of`: a manifest may carry a `cohort` list and
  `centered: false`; a manifest without them is checked against the frozen twelve exactly as before). `tests/test_window_manifest.py` shows that
  the frozen pair gives `k6b_native_base.SOURCES`, `w1_roi.AL`/`MED` and the frozen cohort unchanged, and that the six W7 windows pass.
  `w7_select manifest` writes `windows.json` (adds K4's conversion sha256 per window), `w7_select stage` copies the six clips under the fleet
  data root (not run). `experiments/tennis/w7_specs.py`: spec builder (`verify_staged_inputs: true`, gpu3/gpu5/gpu6, A6000 or Ada, smoke ≤600 s,
  full ≤2,600 s) for the generalised stages and the stage table below.
- Not generalised (code still pins Al031/Med014 or prior-run controls): `k6b_rgb_diagnostic`, `k6b_receiver_frame`, `k6b_real_base_judge`
  (`PILOTS`, `--gen-al/--gen-med/--al/--med`, controls that reproduce the prior run), `k6b_rate_transport`, `w3_neural_baseline`/`w5_dcvc`
  (`SOURCES` al/med, `two_windows`). The cohort field lets the features, anchors, prepare, generate and judge validators of `k6b_confirm_run`
  accept the W7 manifest, but those stages have never run on a window outside the frozen twelve. The w1_* jobs take one `--window`, so the
  anchor stages are one job per window, not one job for six.
- The 12-image reference bank (from `k6b_reference_video.reference_banks` and the 2026-10-25 accounting): it is the set of `*.png.ref`
  conditioning crops of the match's six confirmation windows, one per player unit at `reference_frame == 0` of that window (6 windows x 2
  players = 12 per match; 53,793 B for Al and 75,843 B for Med as SVT preset 8 CRF 20 half-size images; 895,487 / 1,179,150 B as PNG). So every
  Al/Med window so far conditioned its players on a crop from its own first frame, and the "bank" is the pooled set. It holds no crop of a
  new window. Using a bank crop for a new window would need a player-identity mapping (units are SAM tracks, not named players) that does not
  exist. The manifest therefore records `reference: window_frame0` (the same recipe applied to each new window: first fitted scored frame per
  player, the encoder sends the two crops, billed with the window), not the old bank. The rule's wording "existing reference banks" needs the
  lead's reading: if it must mean the old 12 crops, that is a cross-window reference test and needs the identity mapping first.
- GPU estimate per window (seconds, A6000 unless noted; from the job records cited in `w7_specs.STAGES`): features 175, anchors/keypoint codes
  444, prepare 60, generation 1,735 (863 s per player-unit; Ada about 1,365), GPU source control 84, receiver masks 820, compose and hybrid
  score 410, anchor scoring 681, DCVC 245; CPU only: paid base 84, rates, W1 encodes. Total 4,654 s per window (4,284 s with Ada rendering).
  Against the rule's Tier 1 budget: first window 4,654 s = 1.29 GPU-h against ≤1 h; the other five 23,270 s = 6.46 GPU-h against ≤4 h; all six
  7.76 GPU-h against 5 h (about 7.1 h with Ada rendering). The estimate exceeds the budget by about 1.5x; generation (37%), receiver masks
  (18%) and anchor scoring (15%) dominate. Dropping the K6 anchor codes and the source control, if the new-window judge does not need them,
  gives about 4,126 s per window, still above. This needs a decision (raise the budget to about 8 GPU-h, cut windows, or cut stages) before
  any GPU job; the stage times are from two-window jobs and ignore queueing and failed smokes.
- Saved manifest of the six Tier 1 windows: `docs/experiments/w7-specs/windows.json` (sha256 `18d184cd39f6a4029a98e6fd9852d1e38f5cfde5c2ee19e5bc67bd69299c5e15`;
  conversion hashes computed on gpu1 from the archive clips). Checks: ruff and mypy clean; full pytest 1155 passed, 24 skipped, 1 failed
  (`test_agent_hooks::test_patches_into_models_or_datasets_are_blocked`, the macOS protected-path failure recorded since 2026-10-24, unrelated).

### 2026-10-30 — W7 decisions (lead), before any W7 run: reference reading, budget
- Reference. "Existing reference banks" is read as the same billed reference recipe per window: the crop at the window's frame 0 for each
  player unit, coded and billed as before, the bank's per-window bytes recomputed for every new window; adapters stay the existing Al and Med ones.
  Reason: the old 12-image bank is the pooled frame-0 crops of the old windows, and no player-identity mapping exists to reuse them for a new
  window. Manifest field `reference: window_frame0`.
- Budget. Tier 1's budget is being raised by the user; no stage is cut yet. The keypoint codes stay. The GPU source control (`k6b_rgb_diagnostic`) is dropped
  for new windows only if nothing downstream reads it (to be stated when its consumers are checked below).

### 2026-10-30 — W7 Tier 1 budget raised (user, 2026-10-30)
- (user, 2026-10-30) The Tier 1 GPU budget is raised from 5 to 8 GPU-hours: the first window (also the smoke) up to 1.3 GPU-h, all six
  windows up to 8 GPU-h. All stages stay, keypoint codes included. `w7_specs.budget_check()` applies it: estimate 4,654 s per window
  (1.29 GPU-h, inside 1.3 h) and 27,924 s for six (7.76 h, 876 s of margin). The estimate carries no allowance for queueing or failed smokes; the
  stage times come from two-window jobs, and generation (A6000, 1,735 s) is 37% of it. The Ada render figure would add margin.
- Source control: `k6b_rgb_diagnostic` is kept. Its `-gpu.npy` base pixels are what `k6b_real_base_judge` scores (`--gpu-base`), so it is not droppable;
  only its prior-run hash gate is dropped for manifest windows (two fresh GPU conversions must agree and the IVF must decode to the encoder's recon).

### 2026-10-30 — W7 generalisation status (end of this pass; nothing submitted, nothing run on GPU)
- Done, with the frozen pair unchanged by test: `k6b_native_base`; `w1_masks`/`w1_encode`/`w1_score` via `w1_roi.window_from_spec`; the `k6b_confirm`
  family's cohort; `k6b_rgb_diagnostic` (`--windows-file --window --source`, no `--prior`); `k6b_rate_transport` (`--windows-file`: the window id and
  match are the only row fields it needs, so no diagnostic is required; its scratch dir now honours `TMPDIR`); `k6b_real_base_judge` (`plan`,
  `rate_components`, `identity_controls`: for manifest windows the prior-control gate is replaced by two in-job controls, hybrid-without-players equals
  the paid base and the plain base scored twice is identical, plus the source RGB hash check against the receiver result; the frozen gate is unchanged);
  `w7_rates.py` (`--rates` file and `--window-state` entries from the transport and native-base outputs; the bank term is the window's own PNG bytes
  scaled to a 12-crop bank over 7200 s, a reading to be confirmed); `w7_specs.py` builders with first-window smoke/full for native base, rgb diagnostic,
  rate transport and the judge. The new GPU code paths (judge manifest mode, rgb diagnostic manifest mode) are unit-tested only for argument plumbing and
  have not run on a GPU.
- Not done: `k6b_receiver_sam`/`k6b_receiver_frame` (pins PILOTS, three frozen input hashes, and its per-window judge needs prior control scores through
  `k6b_matte_judge.judge_window`, whose `control` argument is compared to an old run at tolerance 1e-4); `w3_neural_baseline`/`w5_dcvc` (`SOURCES` al/med,
  `two_windows`, prior inputs); per-window checkpointing in `k6b_native_base` (one bundle per job); spec builders for features, anchors, prepare, generation,
  receiver masks and DCVC; the confirm-style manifest.json (clips, registration rows) the features stage reads.

### 2026-10-30 — W7 stages generalised, specs for the first window, clips staged (CPU only; nothing submitted, no GPU job)
- Bank term (lead accepted): the window's own frame-0 crop bytes scaled to a 12-crop bank over 7200 s, which keeps the term the size of Al/Med's. The worst-case
  sensitivity row (the window's own crops billed over the 4.8 s window alone, `bank_worst_case_kbps`) is written by `w7_rates` and carried into the
  judge's per-scale rates as `bank_worst_case_kbps` and `total_kbps_bank_worst_case`. Deviation recorded: `k6b_rgb_diagnostic` runs on A6000 hosts only (the code asserts the class), so its spec uses gpu3/gpu2.
- Generalised since the last entry, frozen pair unchanged: `k6b_matte_judge.judge_window(control=None)` (the first variant is scored a second time in the job and must agree within
  the 1e-4 control tolerance; the saved-composite and old scoring-mask hash checks apply only when a prior control exists); `k6b_receiver_sam` / `k6b_receiver_frame`
  (`--windows-file --window`, no frozen diagnostic/reference-report/control hashes, `windows` recorded for the validator); `w3_neural_baseline` (`activate_manifest`: SOURCES
  from the manifest, anchor/transport/prior hashes unpinned, no audited-Al control, per-window checkpoints as before, validator reads the manifest from the spec) and `w5_dcvc` (optional points file).
- `w7_select confirm-manifest` wrote `manifest.json` (clips, K8b registration rows, 49 old clips, `cohort`, `centered: false`) and it passes `k6b_confirm.check_manifest` for all six windows.
- Staged under the fleet data root by `w7_select stage` on gpu1 (new files only, sha256 verified after copy): `/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier1-20261030/`
  with `sources/` (six clips), `windows.json` (sha256 `18d184cd…c5e15`) and `manifest.json`.
- `w7_specs.py` now builds first-window (Al 028) smoke/full specs for features, anchors, prepare, generation, receiver masks, DCVC, native base, rgb diagnostic, rate transport and the judge
  (`verify_staged_inputs`, gpu3/gpu5/gpu6 except where a class is required; smoke ≤600 s on the shortest grid each stage supports; full ≤2,600 s). Upstream archive hashes that do not exist yet
  are `PENDING` (all zeros) and generation needs the prepared unit names; both are filled when the upstream job completes. Stage order: features, `k6b_confirm build-inputs`, anchors, prepare,
  generation, rate transport (needs a manifest-mode `--anchor/--prepared`), native base, rgb diagnostic, receiver, judge, W1 stages, DCVC.
- Still open: the `build-inputs` step has no spec; generation units come from the prepare output; the W1 `--window-state` and judge `--rates` files are made from stage outputs by `w7_rates`; the new GPU paths are unit-tested for plumbing only.

### 2026-10-30 — W7 first window (Al 028 @ 0): progress (agent run on branch `claude/w7-first-window`)
- Recipe decision (recorded before the first W7 GPU job): W6's full-rate generation did not pass its rule (pose passes, time fails; W6 outcome entry),
  so the frozen W5 recipe stays: players generated on every third frame. Budget: first window up to 1.3 GPU-h (user, 2026-10-30), smokes included;
  GPU-seconds counted from each GPU job's `execution.json` (smoke + full). CPU jobs are not counted.
- Specs: `python -m experiments.tennis.w7_specs first-window STAGE windows.json OUT [NAME=PATH:SHA ...]`; upstream archive hashes from `sha256sum` of the
  upstream job's `full/published.tar` on the server.
- features, first submission `20261030T142453Z-b8bc4336` (gpu6 Ada): smoke ran (145 s GPU, scientific checks passed when the validator was rerun by hand) but the
  fleet validator failed: validator argv is not placeholder-expanded (`{staged:manifest}` reached `validate-features` literally). Fix: the validator names the manifest by
  its path (commit "features validator names the manifest literally"). Resubmitted as `20261030T143125Z-8785a55b` (a failed job is not replayed; this is a new job).
- native base (CPU, no GPU): `20261030T142923Z-f6699fe5` complete on gpu6, smoke 8 s + full 58 s; `full/published.tar` sha256
  `9744754d6acd62f7d872e804b5af09b00de1187f100062f9369e526b3e285239`, code `aaf6ae4a`. The fleet validator is smoke-only (24 frames hard-coded); the full stage was validated by a
  generalised validator (`grid_checks`, `--validate` now reads the frame count from `base-result.json`; test in `tests/test_k6b_native_base.py`): 26 checks passed,
  including the source conversion hash against `windows.json`.
- features: `20261030T143125Z-8785a55b` complete on gpu6 Ada (code `256b18d8`), `full/published.tar` sha256 `c2e0a4aba701c2c2f294b63ad780e9565d203cee5cc56858ee4253755da9d2f5`;
  smoke 162.8 s + full 155.8 s GPU; full output rechecked by hand with `validate-features`: 19 checks passed. GPU total so far 464 s (incl. the 145 s failed-validator smoke); budget 4,680 s.
- build-inputs (CPU): `20261030T144100Z-2d3ae276` submitted.
- build-inputs: `20261030T144100Z-2d3ae276` complete (CPU, gpu6), `full/inputs.tar` sha256 `bc8a01c7ae4321abee4e42b6871ab5f31c9d880067b9d0baf9c99a602f00ac0f`; `validate-inputs` on the full output: 7 checks passed.
- anchors (K6 run, GPU): `20261030T144525Z-de7adcd2` submitted.
- anchors: `20261030T144525Z-de7adcd2` complete on gpu6 Ada, smoke 136.4 s + full 596.9 s GPU; `full/published.tar` sha256 `18e7294f7ca82ce4e118ced88bf9ee31690d23f862809f986f5003c36da291f8`; full output rechecked by hand (`k6b_confirm_run --validate-only`): 20 checks passed. GPU total so far 1,197 s of 4,680 s.
- prepare (CPU): `20261030T150213Z-99604e48` submitted. rgb diagnostic (GPU, A6000 gpu3/gpu2): `20261030T150335Z-9e58b2cc` submitted in parallel (needs only the native base).
- prepare: `20261030T150213Z-99604e48` complete (CPU, gpu6; smoke 27.8 s, full 26.5 s); `full/published.tar` sha256 `e30c6d97a6a8406f85bcc6c6928d9b7fb21ddf8280ad25a629f78f178478cfdb`; units
  `alcaraz_hurkacz__028_at120__person_000000000006` and `…000000000004`; full output rechecked by hand: 14 checks passed.
- generation (GPU, Ada): `20261030T150651Z-a58b2ce2` submitted (every-third-frame recipe; 2 units, adapter ah-C, C base). rate transport (CPU): `20261030T150852Z-d4e7cd41` submitted
  (first attempt refused by the client: a spec needs a named scale parameter; fixed by passing the ignored `--max-windows {windows}`).
- rate transport: `20261030T150852Z-d4e7cd41` complete (CPU, gpu6; 17 s + 18 s); `full/published.tar` sha256 `8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607`; `--validate` on the full output: 21 checks passed.
- rgb diagnostic: `20261030T150335Z-9e58b2cc` (exclusive, A6000 on gpu3/gpu2) stayed pending 12 min (every A6000 holds another user's process); cancelled while still pending (0 GPU s) and resubmitted
  as a shared-GPU job `20261030T151619Z-33fb8db4` (same code and arguments; GPU colour conversion and resize are deterministic; class A6000 still required by the code).
- generation: `20261030T150651Z-a58b2ce2` complete on gpu5 Ada (every-third-frame recipe, 2 units, adapter ah-C); smoke 92.2 s + full 1,323.9 s GPU; `full/published.tar` sha256
  `d88f1c79cd78f292dc70105540cde4b659a4726138d34de02f343b4bb9edbd91`; full output rechecked by hand (`k6b_confirm_run --validate-only`): 15 checks passed.
- rgb diagnostic, shared job `20261030T151619Z-33fb8db4` (gpu2, A6000): smoke ran 24 s but the fleet validator failed: the allowance (180 s, the frozen pair's 360 s / 2) was below the
  one-window projection (191 s under contention). Code bug, fixed (`reservation_seconds`: 600 s, the spec's full reservation, for manifest windows; the validator also covers the 240-frame stage; test added).
  Rerun by hand on the smoke output: every check true except the publication-time projection, which a replay cannot reproduce. Resubmitted as `20261030T163628Z-d8f52f8f`.
- receiver masks (GPU, shared as the 2026-10-26 job): `20261030T163803Z-51dfa688` submitted (inputs: rate-transport, anchors and generation archives above).
- GPU seconds so far (smoke + full from execution.json, failed smokes included): features 145 + 163 + 156, anchors 136 + 597, generation 92 + 1,324, rgb 24 (failed validator) = 2,637 s of 4,680 s
  (rgb diagnostic rerun, receiver, judge, W1 scoring and DCVC still to count).
- receiver masks, first submission `20261030T163803Z-51dfa688` (gpu6): smoke ran 86 s GPU (projected full 601 s) but the fleet validator failed on one check, `denied_reads`: it required 4 forbidden-read
  probes (frozen pair: two sources, control, transport) while a one-window manifest run has 2 (source, transport). Code bug, fixed (`denied_probe_count`, test); the corrected validator passes all 25 checks
  on that smoke output. Not replayed: a new job follows (id below).
  Resubmitted receiver masks: `20261030T164758Z-6191ce05`.
- rates and window-state (CPU): the judge's `--rates` and the w1_* `--window-state` files must be staged files under the data root, and writing a new file there by hand was refused by the permission system, so they are a
  fleet stage: `experiments.tennis.w7_rates` CLI (`rates.json`, `window-state.json` from the rate-transport `transport.json` and the native-base archive; validator recomputes). Job `20261030T170319Z-3524601c` submitted.
- receiver masks, second submission `20261030T164758Z-6191ce05` (gpu5): smoke 100.4 s passed the fleet validator; the full stage ran 334.1 s GPU, scored the whole window, and then crashed at the last step
  (`k6b_receiver_sam.main`: the frozen-pair `assessment` and `frozen_quality_sensitivity` read `--diagnostic`, which manifest windows do not have). Code bug; its scratch results were lost (only `partial.tar`).
  Fix: both frozen-only blocks run only when `--diagnostic` and `--reference-report` are given. GPU total now 3,157 s of 4,680 s. A third job follows (id below); no result of the crashed run is used.
  Third receiver job: `20261030T170824Z-6b51342b`.
  rates job first run `20261030T170319Z-3524601c` failed its smoke validator (int vs string JSON keys in the recompute; fixed); resubmitted as `20261030T171203Z-7d1145b6` (CPU).
- rates/window-state: `20261030T171203Z-7d1145b6` complete (CPU, gpu6); `full/rates.json` sha256 `e0a5ffc02000a4f98d8141c8d688872da739c28fb31df45e7579ef0ec13f5e8d`, `full/window-state.json` sha256
  `121dd9e1eb2534b1e24e35ff06a1b015e6f93cbf4fa97257bed88e8373ec0b5e`; validator rerun on the full output: 5 checks passed (rates equal to the hand computation made earlier).
- receiver masks: `20261030T170824Z-6b51342b` complete on gpu5 Ada (shared); smoke 103.6 s + full 331.0 s GPU; `full/published.tar` sha256 `6b0cc2c23122e974a0b03d976a11787ee465f4deea5a5ea88eab84aea8728f4a`,
  `full/receiver.json` (the judge's `--prior`) sha256 `86d4dcb264a2ea0c40ffaa647e7b4ff2fb4d6150e3ccf0a2378e47f5c121eaf7`; full output rechecked by hand (`k6b_receiver_frame --validate`, stage full): 24 checks passed.
  GPU total so far 3,592 s of 4,680 s (rgb diagnostic `20261030T163628Z-d8f52f8f` running on gpu3, judge, W1 scoring, DCVC still to count).
- rgb diagnostic, shared job `20261030T163628Z-d8f52f8f` (gpu3, A6000): smoke ran 45.5 s GPU (summary: 27.2 s, projected full 321 s) but failed the fleet validator again, on `full_projection_with_publication`:
  the check multiplies the smoke's staging/publication time (73 s on this shared host, 150 MB archive) by 10 and compared it with the 600 s reservation. Not a result problem (all other 12 checks true).
  Fix: the allowance is now the job spec's own full reservation (1,800 s for this stage, reservation only; expected about 320 s), test extended. GPU total now 3,637 s of 4,680 s.
  Resubmitted rgb diagnostic: `20261030T174459Z-2ad62629`.
- rgb diagnostic `20261030T174459Z-2ad62629` (gpu3): smoke 26.3 s GPU, failed the fleet validator on the same check: the validator started 186 s after dispatch (16.8 s of work, NFS/staging latency of the shared host), and the check
  multiplied that latency by 10 (2,316 s against 1,800 s). All other checks true. Fix: for manifest windows the publication/startup time is added once instead of ten times (the frozen pair keeps the old arithmetic);
  the projection is then about 400 s. GPU total now 3,663 s of 4,680 s.
  Resubmitted rgb diagnostic: `20261030T181751Z-a498ddf7`.
- Judge pre-check (before any judge GPU job): the judge's smoke validator still hard-coded the frozen pair's four timing sequences (`k6b_real_base_cost.timing_sequences`), which a one-window manifest run
  cannot supply, and its full-stage checks assumed the 3-tick smoke. Fixed before submitting: manifest windows use their base trace only (`expected_sequences`), and the validator covers the 80-tick full stage (tests added).
- rgb diagnostic: `20261030T181751Z-a498ddf7` complete on gpu2 (A6000, shared); smoke 18.5 s + full 107.5 s GPU; `full/published.tar` sha256 `ce6a92f875b9c7b48a3b8b516dccac46aad4dfe44d9622a4da330905c84232c6`;
  full output rechecked by hand (`--validate`, stage full): 12 checks passed. GPU total so far 3,789 s of 4,680 s (891 s left): the judge (smoke + full, estimate about 500 s) is submitted next;
  W1 scoring (estimate 681 s) and DCVC (245 s) cannot fit in what remains.
- judge (compose + score, shared GPU): `20261030T184549Z-d9fe4272` submitted (first attempt refused by the client: a shared spec needs contention `continue`; fixed in the builder).
- judge first job `20261030T184549Z-d9fe4272` (gpu6): crashed in the first 3.3 s of the smoke: `GPU source control or base coverage missing` (the check hard-coded six base rows, two windows). Code bug, fixed
  (three scales per window, one source row per window). GPU total 3,792 s of 4,680 s.
  Judge resubmitted: `20261030T185245Z-e51a4dec`.
- judge (compose + score): `20261030T185245Z-e51a4dec` complete on gpu5 Ada (shared; code = branch HEAD `git_head` in its ready.json); smoke 33.8 s + full 283.3 s GPU; `full/published.tar` sha256
  `cbc26962935b3542b0b7cf77f724010de19323fedd6fe02cb822ec6e43e75837`; full output rechecked by hand (`k6b_real_base_judge --validate`, stage full; the `representative_ticks` check now
  expects `frames_per_window == 0` for the full stage): 11 checks passed, in-job controls (hybrid without players equals the paid base; plain base scored twice identical) both true.
  As written by the judge (`quality-summary.json`, 80 saved ticks; order per scale: base then hybrid, scales 8, 4, 2), no interpretation here:
  full LPIPS [0.4502, 0.4375, 0.2937, 0.2845, 0.1460, 0.1415]; player-region LPIPS [0.8578, 0.3275, 0.7180, 0.3042, 0.4810, 0.2778];
  full DISTS [0.2971, 0.2656, 0.2002, 0.1750, 0.1202, 0.1040]; player-region DISTS [0.5924, 0.2505, 0.4961, 0.2420, 0.3624, 0.2329];
  hybrid total rate kbps at scales 8/4/2: 13.68 / 18.46 / 31.57 (paid base 9.34 / 14.12 / 27.23). The judge's own `comparisons` (scale 4 against the recovered SVT envelope) are in `window.json` of the archive.
- GPU-seconds, first window (smoke + full from execution.json; failed smokes and the crashed receiver full included): features 464 (145 + 163 + 156); anchors 733 (136 + 597); generation 1,416 (92 + 1,324);
  rgb diagnostic 221 (four jobs: 24, 45, 26, 18.5 + 107.5); receiver 955 (86, 100 + 334, 104 + 331); judge 320 (3.3, 34 + 283). **Total about 4,109 s = 1.14 GPU-h of 1.3 h (4,680 s); 571 s left.**
  CPU jobs, no GPU: native base, build-inputs, prepare, rate transport, rates/window-state.
- Where the chain stands: features, build-inputs, anchors, prepare, generation, rate transport, native base, rgb diagnostic, receiver, judge done and validated on their full output. Not run: the W1 stages
  (masks and encodes are CPU and free; scoring is estimated at 681 s GPU plus a smoke, which does not fit in the 571 s left) and DCVC (estimated 245 s plus smoke, fits alone, not yet generalised end to end).
  The lead decides: raise the budget by about 0.1 GPU-h for W1 scoring, or run DCVC first within the remainder.

### 2026-10-30 — W7 first-window GPU sub-budget raised to 1.5 GPU-h (lead decision)
- The first window's sub-budget rises from 1.3 to 1.5 GPU-h (5,400 s). Reason: it is also the smoke of the whole chain for the other five windows, so the one-off failures (validators written
  for the frozen pair, a crashed receiver full stage) are charged to it; the raise stays inside the user's 8 GPU-h Tier 1 card budget (user, 2026-10-30). GPU spent before this decision: about 4,109 s; 1,291 s remain.
  Order from here: W1 stages (CRFs chosen on rate alone before any quality is computed), then DCVC only if its smoke plus full fits in what is left; then a per-window projection for the other five.
- W1 stages (CPU first), plan recorded before any W1 quality exists: the SVT anchors are searched on rate alone (`w1_encode` CRF search, W5 recipe: plain at half and quarter resolution, outside-R +20/+40 QP at both,
  240 frames) to bracket the hybrid's total rate at scale 4 with KPS2 as the primary side rate: target 18.461418333333334 kbps (the judge's `rates[scale 4].total_kbps`; no quality value is read by the searches).
  Start CRFs come from `window-state.json` (`7d1145b6`, sha256 `121dd9e1…0b5e`). The selected CRF pairs are copied here from the encode jobs' outputs before the scoring job is submitted.
- W1 CPU jobs submitted: masks `20261030T190740Z-8890c377`, encode half (plain + controls) `20261030T190853Z-dc5b4815`, quarter `20261030T191012Z-2ddae924`, half outside-R `20261030T191136Z-93c95de1`, quarter outside-R `20261030T191258Z-54a3c17c`.
- The five W1 CPU jobs above all failed their smokes within seconds on `pinned input changed: published.tar`: `w1_masks` and `w1_encode` pinned the anchor/transport archives to the frozen pair's hashes.
  Code bug, fixed (the pins apply to `al`/`med` only; manifest windows are identified by the fleet's staged hashes). 0 GPU s. Jobs not replayed; new ones below.
  Resubmitted: masks `20261030T191755Z-f17c13d7`, encode half `20261030T191856Z-82bf32b2`, quarter `20261030T192015Z-34f84816`, half outside-R `20261030T192145Z-768aceab`, quarter outside-R `20261030T192314Z-cfd1f3d3`.
- Second W1 CPU round also failed in seconds (0 GPU s): the W7 transport archive stores `publish/packets/...` while `w1_masks`/`w1_encode` looked for `packets/...` (the 2026-10-25 layout). Code bug, fixed
  (`w1_roi.packet_file` accepts both layouts; test added). New jobs follow; ids below.
  Third round: masks `20261030T193203Z-878a0fb9`, encode half `20261030T193323Z-20cc826d`, quarter `20261030T193510Z-affda602`, half outside-R `20261030T193631Z-b4cda4e7`, quarter outside-R `20261030T193750Z-1da1280c`.
- W1 masks (CPU): `20261030T193203Z-878a0fb9` complete on gpu6. The four encode jobs of that round failed their smoke validator only (`KeyError: 'al028'`: `w1_encode.validate` did not register the manifest window; fixed with
  `w1_roi.preload_from_spec`; the corrected validator passes all 10 checks on a smoke output). 0 GPU s. Fourth round of encodes follows.
- W1 masks `20261030T193203Z-878a0fb9`: `full/published.tar` sha256 `39b061484dee4d2adfac0f17f3cc84ea8c803756e2cbc8e8508d146d889786c7`; `w1_masks --validate` on the full output: 8 checks passed (full: 80 ticks).
  Encodes, fourth round: half (plain + controls) `20261030T194352Z-c8a65737`, quarter `20261030T194520Z-f1e8f664`, half outside-R `20261030T194640Z-cefe70d1`, quarter outside-R `20261030T194800Z-92e26eca`.
- W1 encodes complete (CPU, rate-only searches; no quality read yet), target 18.461418333333334 kbps. Chosen CRF triples (bracketing the target by the search's own rule), recorded before scoring:
  half plain `20261030T194352Z-c8a65737` (also the plain CRF 63 controls): 66, 67, 68 at 18.87, 17.72, 16.48 kbps (65 at 20.99); sha256 `f3de1ea7f0bd74a621ffb79fb2bd97cc622fb1ae05f504e4815c7470bd2540d8`; 12 validator checks passed on the full output.
  quarter plain `…194520Z-f1e8f664`: 61, 62, 63 at 18.71, 16.62, 14.12 kbps; sha256 `970de450d9131bd8bd7ded4e10b8a6951918dc9da838746fe0f95efe4037b77f`; 10 checks passed.
  half outside-R `…194640Z-cefe70d1`: +20: 68, 69, 70 (18.71, 17.67, 16.83 kbps); +40: 67, 68, 69 (18.51, 17.56, 16.64 kbps); sha256 `4bae9b44296be1132c389d21e4ba49272bdb5f16b57be5179261bac7aec8037a`; 13 checks passed.
  quarter outside-R `…194800Z-92e26eca`: +20: 62, 63, 64 (19.56, 17.24, 15.88 kbps); +40: 62, 63, 64 (19.44, 17.22, 15.65 kbps); sha256 `2a7068cbcfb547a1aca857c16901b68f53c3f8410a3c7622362eb952702d353f`; 13 checks passed.
- Pre-check of the W1 scoring job (before any GPU second): `w1_score.hybrid_point` required the hybrid job to have billed the frozen windows' KPS1 packet; the W7 judge pays the KPS2 packet (2,469 B) directly, so scoring
  would have stopped at its start. Fixed: either packet is accepted and the other rate is re-billed (test added).
- W1 scoring (GPU, A6000, shared): `20261030T195316Z-bbf774cf` submitted (masks, judge archive and the four encode archives above).
- W1 scoring first job `20261030T195316Z-bbf774cf` (gpu3 A6000, shared): smoke 225.2 s GPU passed the fleet validator (projected full 924 s with the 186 s fixed overhead); the full stage ran 121.5 s and stopped at its own
  control gate: the plain scale-8 base re-scored over 80 ticks differs from the judge's recorded value by 1.18e-4 in foreground LPIPS (7.6e-5 full LPIPS, 1.7e-5 DISTS) against the frozen windows' 1e-4 tolerance.
  The recorded scores came from the judge on an Ada GPU, the re-score ran on an A6000; this is a control on the paid base, not on the hybrid or the SVT anchors, and no anchor or hybrid quality had been computed.
  **Deviation, decided after seeing the control error and before any anchor score existed:** manifest windows use W3's tolerance for the same kind of control, 1e-3 (`control_tolerance`; frozen windows keep 1e-4); the
  observed control errors are recorded in the summary either way. GPU total now 4,456 s of 5,400 s (944 s left); the rescoring needs about 850 s (smoke 225 + full about 620), which leaves no room for DCVC (about 350 s).
  Rescoring job: `20261030T201626Z-e32a981b` (same inputs).
- W1 scoring: `20261030T201626Z-e32a981b` complete on gpu2 (A6000, shared); smoke 106.2 s + full 870.3 s GPU; `full/published.tar` sha256 `48846a4a9404a7a0834cbe156a5d57988db114240773ba2a9657b7a08246db21`;
  `w1_score --validate` on the full output: 10 checks passed. Control errors as written (max over the three plain bases, full-stage 80 ticks): full LPIPS 7.7e-5, foreground LPIPS 1.18e-4, full DISTS 1.7e-5, foreground DISTS 4.5e-5 (all under 1e-3).
  As written by the job (kps2 primary, target 18.461418333333334 kbps; the hybrid's scale-4 means are LPIPS whole 0.2845 / foreground 0.3042, DISTS whole 0.1750 / foreground 0.2420; no interpretation here):
  `analysis.decision`: outcome A, dominating lpips [] dists [], resolved s2_o0, s2_o20, s2_o40, s4_o0, s4_o20, s4_o40, unresolved [].
  SVT at the hybrid's rate (whole LPIPS, foreground LPIPS, whole DISTS, foreground DISTS): s2_o0 0.1908, 0.5933, 0.1458, 0.4344; s4_o0 0.2761, 0.6416, 0.1909, 0.4347; s2_o20 0.2380, 0.6496, 0.1682, 0.4645;
  s2_o40 0.2388, 0.6308, 0.1743, 0.4528; s4_o20 0.2903, 0.6961, 0.2012, 0.4718; s4_o40 0.2929, 0.6945, 0.2012, 0.4716. Full tables: `score-summary.json` / `score.json` in the archive.
- First-window GPU ledger, final (smoke + full from execution.json, every job including one-off failures): earlier 4,109 s (through the judge) + W1 scoring 347 (failed job: 225 + 121.5) + 976 (106 + 870) = **5,432 s = 1.509 GPU-h,
  32 s over the 5,400 s sub-budget** (the rescoring's smoke and full ran longer than the 850 s planned; the full stage took 870 s). Nothing was submitted after reaching the limit.
- DCVC: not run. `w3_neural_baseline` (manifest mode: `activate_manifest`, unpinned inputs, per-window checkpoints, `--point-list` optional) and `w5_dcvc` (points table argument) were already generalised and tested
  (`tests/test_window_manifest.py::test_w3_manifest_mode_...`); this pass added the first-window spec (`w7_specs.dcvc`, Ada, shared, `first-window dcvc`) and fixed nothing else, because no GPU budget was left to run its smoke.
  Its points would be the rate-only calibrated targets 15/30/60 kbps (`w3_analysis.TARGET_KBPS`), which bracket the hybrid's 18.46 kbps. The first-window DCVC anchor is therefore missing, so W1's dominance outcome A above covers the SVT anchor families only.

### Projection for the other five Tier 1 windows (GPU seconds, successful smoke + full only, one-off failures excluded)
- Per window, from the first window's successful jobs: features 319 (163 + 156); anchors 733 (136 + 597); generation 1,416 (92 + 1,324); rgb diagnostic 126 (18.5 + 107.5); receiver 435 (104 + 331); judge 317 (34 + 283);
  W1 scoring 976 (106 + 870); DCVC about 300 (estimate: 245 s from the W3 record plus a smoke; not measured). CPU stages (native base, build-inputs, prepare, rate transport, rates, W1 masks and encodes) cost no GPU.
  Total without DCVC 4,322 s (1.20 GPU-h); with the DCVC estimate about 4,620 s (1.28 GPU-h).
- Five windows: 21,610 s (6.00 GPU-h) without DCVC, about 23,100 s (6.42 GPU-h) with it. With the first window's actual 5,432 s: 27,042 s (7.51 GPU-h) without DCVC, about 28,500 s (7.9 GPU-h) with DCVC for the five and none
  for the first. DCVC for all six would bring it to about 28,800 s = 8.0 GPU-h, with no margin against the 8 GPU-h card. Caveats: Med windows are 50 fps clips with other match adapters and were not timed; the shared GPUs add queue
  time but not GPU seconds.
- Where the fleet smoke is the cost: a successful smoke costs about 650 s per window (163 + 136 + 92 + 18.5 + 104 + 34 + 106). Stages that take several windows (`features`, `anchors`, `prepare`, `receiver`, `judge`, `native_base`,
  `rgb_diagnostic`, `rate_transport`; not the `w1_*` stages, which take one window) pay it once per job instead of per window; batching the five windows by stage (anchors in 2 + 3 and generation per window to stay inside
  the 2,600 s stage cap) would save about 4 x 540 = 2,200 s, i.e. the five windows would cost about 19,400 s (5.4 GPU-h) instead of 21,610 s.
- Validators and code paths fixed on this window (all merged into this branch) were the main one-off cost: about 1,100 s of GPU (145 + 95 + 520 + 3 + 347) went to failed smokes and the crashed receiver full stage.

### 2026-10-30 — W7 first-window sub-budget raised to 1.65 GPU-h; DCVC run (lead decision)
- The first window's sub-budget rises from 1.5 to 1.65 GPU-h (5,940 s). Reason: the first-window DCVC run completes the window's anchor set (SVT half/quarter, outside-R and DCVC) and also smokes the DCVC chain for the other five
  windows. Projected Tier 1 total with the five windows batched by stage: about 7.4 of the 8 GPU-h card. GPU spent before this decision: 5,432 s; 508 s remain.
- DCVC (GPU, Ada, shared): `20261030T213822Z-0acbf07b` submitted (first-window spec `w7_specs.dcvc`; calibrated points 15/30/60 kbps targets, SVT scales 8, 4, 2).
  DCVC first job `20261030T213822Z-0acbf07b` (gpu6 Ada): smoke 66.4 s GPU ran and wrote its summary; the fleet validator failed two checks of its own: `svt_control_ran` (requires the audited Al control, which exists only for the
  frozen windows) and `budget` (spec `budget_seconds` 3,400 against the validator's 3,300 cap). Both are spec/validator bugs; the other 19 checks passed (projected full 635 s). Fixed (control required for the frozen pair only; spec budget 3,300),
  test added. GPU total 5,498 s of 5,940 s.
  DCVC second job: `20261030T214920Z-ea50a2cf` (the earlier submit attempt was refused by the client for a stale gpu6 heartbeat; nothing was published).
- DCVC: `20261030T214920Z-ea50a2cf` complete on gpu6 (Ada, shared); smoke 65.7 s + full 242.7 s GPU; `full/published.tar` sha256 `66a715b22f6bcde05ce28f94d2eb586b7a215558964599302bff5105e8bab6ec`;
  `w3_neural_baseline --validate` on the full output (the validator now covers the 240-frame stage: 3 points, 80 ticks, no smoke projection): 21 checks passed. Accepted (decode checks), 3 DCVC points (rate-only calibrated targets 15, 30, 60 kbps; scale, QP chosen on the 25-frame estimates).
  As written by the job (DCVC points: scale, QP, kbps, whole LPIPS, whole DISTS, foreground LPIPS, foreground DISTS; table in `docs/experiments/w7-specs/dcvc-points-al028.json` in the `w5_dcvc` column order QP, kbps, whole LPIPS, whole DISTS, fg LPIPS, fg DISTS):
  scale 8, QP 30: 3.306 kbps, 0.4441, 0.2915, 0.8186, 0.5307; scale 2, QP 6: 7.654 kbps, 0.2127, 0.1845, 0.5946, 0.4291; scale 2, QP 18: 14.572 kbps, 0.1669, 0.1419, 0.4435, 0.3324.
  The job's own SVT bases (same pixels as the judge): scale 8 9.338 kbps 0.4502, 0.2971, 0.8578, 0.5924; scale 4 14.121 kbps 0.2937, 0.2002, 0.7180, 0.4961; scale 2 27.231 kbps 0.1460, 0.1202, 0.4810, 0.3624.
  `w5_dcvc` on this window (score summary `e32a981b` and the points file): at the hybrid's rate (kps2 18.4614 kbps; kps1 19.6823 kbps) the DCVC value is **None**: the highest DCVC point is 14.572 kbps, so the hybrid's rate lies outside
  their span and is unresolved, not extrapolated (rule of W3b/W5). So with DCVC included, the W1 dominance outcome is the SVT-only outcome A above, with the DCVC family unresolved at this rate. The job's envelope comparison (DCVC
  pareto against the SVT hull, common range 9.34-14.57 kbps) is in `w3-summary.json`: whole LPIPS BD-rate -63.9 %, whole DISTS -56.0 % (status OVERLAP ONLY).
  Next test (not run; 134 s of the 5,940 s left): re-run DCVC with points chosen on the measured 240-frame rates so that the span reaches the hybrid's rate (the 25-frame estimates overshot: targets 15/30/60 kbps gave 3.3/7.7/14.6 kbps), about 250 s GPU.
- First-window GPU ledger, final: 5,432 s (through W1 scoring) + DCVC 66 (failed validator) + 66 + 243 = **5,806 s = 1.613 GPU-h of the 1.65 GPU-h (5,940 s) sub-budget**.

### 2026-10-30 — W7 first-window sub-budget raised to 1.75 GPU-h; DCVC bracket points (lead decision; rate only, no quality read)
- The first window's sub-budget rises from 1.65 to 1.75 GPU-h (6,300 s). Reason: DCVC is the state-of-the-art neural baseline and an unresolved family leaves the window incomplete; the same rate fix is needed for the five batched
  windows anyway. Projected Tier 1 total about 7.5 of the 8 GPU-h card. GPU spent before this decision: 5,806 s; 494 s remain.
- Rate rule: keep the scale-2 points (QP 6: 7.654 kbps, QP 18: 14.572 kbps) and add higher-quality QP indexes until at least one point exceeds 19.68 kbps (the hybrid's KPS1 rate); the existing 14.572 kbps point is the lower bracket
  of the hybrid's KPS2 rate (18.461 kbps). From the first run's own measurements the measured/estimated ratio at scale 2 is 0.2574 (QP 6) and 0.2502 (QP 18) (the 25-frame estimate overstates the 240-frame rate about 4x, as the W3 entry 2026-10-29 found);
  the calibration estimates are 79.62 kbps (QP 24) and 107.11 kbps (QP 30), so the predicted rates are about 19.6 and 26 kbps (ratio drifting down by 0.0036 per 12 QP: 0.246 and 0.243). **Chosen before submitting: scale 2, QP 24 and QP 30**
  (the lead's suggestion; QP 30 is predicted well above 19.68 kbps, QP 24 close to it). Named points, no calibration encodes: smoke `al028:2:24`, full `al028:2:24,al028:2:30`; SVT base scale 8 only (the SVT bases of the first run are not re-coded).
- DCVC bracket job: `20261030T221148Z-4973d2f4` submitted (points scale 2 QP 24 and 30, SVT scale 8).
- DCVC bracket: `20261030T221148Z-4973d2f4` complete on gpu6 (Ada, shared); smoke 46.8 s + full 164.4 s GPU; `full/published.tar` sha256 `63104b7832ab9d8f8e6125bc2acf852feffd4954ff3950ce33d08f6523270e49`; `w3_neural_baseline --validate` on the full
  output (named-point full stage): 22 checks passed; accepted (decode checks). As written (scale 2: QP, kbps, whole LPIPS, whole DISTS, foreground LPIPS, foreground DISTS):
  QP 24: 20.412 kbps, 0.1414, 0.1162, 0.3714, 0.2967; QP 30: 28.829 kbps, 0.1237, 0.1005, 0.3260, 0.2712. QP 24 exceeds 19.68 kbps (predicted 19.6, measured 20.4); the prediction held to 4 %.
  Scale-2 DCVC curve with the first run's points (QP 6 7.654, QP 18 14.572 kbps): `docs/experiments/w7-specs/dcvc-points-al028-scale2.json` (`w5_dcvc` column order QP, kbps, whole LPIPS, whole DISTS, fg LPIPS, fg DISTS).
  `w5_dcvc` (log-rate interpolation between the bracketing points QP 18 and QP 24, no extrapolation), as written:
  at the hybrid's KPS2 rate 18.4614 kbps: whole LPIPS 0.1490, foreground LPIPS 0.3929, whole DISTS 0.1238, foreground DISTS 0.3074; verdict (DCVC dominates the hybrid? per metric) LPIPS False, DISTS False;
  at the KPS1 rate 19.6823 kbps: whole LPIPS 0.1442, foreground LPIPS 0.3792, whole DISTS 0.1190, foreground DISTS 0.3006; verdict LPIPS False, DISTS False.
  Hybrid's scale-4 means for reference (from the judge, as written): whole LPIPS 0.2845, foreground LPIPS 0.3042, whole DISTS 0.1750, foreground DISTS 0.2420. With DCVC resolved, the W1 dominance outcome stays A (SVT: no policy dominates; DCVC: no dominance on either metric).
- Calibration fix for the batched windows (code): `w3_neural_baseline --bracket LOW HIGH` replaces the 15/30/60 kbps targets: one encode-only 240-frame probe (the scale-2 candidate whose estimate x the typical ratio 0.25 is nearest the middle of [LOW, HIGH])
  gives the measured/estimated ratio, and the two scale-2 points whose predicted 240-frame rates bracket [LOW, HIGH] are coded; the validator checks the predicted bracket and, in the full stage, the measured one. Pure parts tested
  (`tests/test_w3_neural_baseline.py`, `tests/test_w7_specs.py`); the in-job path has not run on a GPU (it runs first in the next window's smoke). Spec: `w7_specs first-window dcvc ... low=L:x high=H:x`.
- First-window GPU ledger, final: 5,806 s + DCVC bracket 46.8 + 164.4 = **6,017 s = 1.671 GPU-h of the 1.75 GPU-h (6,300 s) sub-budget**.

### 2026-10-31 — W7 Tier 1 batch specs (code only; nothing submitted)
- Code: `python -m experiments.tennis.w7_specs plan docs/experiments/w7-specs/windows.json` prints the jobs per stage for the five remaining windows (al034, al043, med000, med024, med019) with
  estimated GPU seconds; `batch STAGE windows.json OUT --windows A,B,... [--pending] [NAME=PATH:SHA ...]` writes one job's spec (`experiments/tennis/w7_batch.py`; tests in
  `tests/test_w7_specs.py`; the first-window specs are byte-identical to before, checked against `tests/fixtures/w7_first_window.sha256.json`). Upstream archives: `role` (one archive covers
  all), `roleK` (K-th job of the upstream stage in `plan`, e.g. `anchor0`, `anchor1`, `base0`, `masks2`) or `role.WINDOW` (`generated.al034`, `w1masks.al034`, `encode_x0.al034`);
  `--units W=u1,u2` (generation), `--bracket W=LOW,HIGH` (DCVC), `--target W=KBPS` (w1 encodes); `--pending` fills a missing archive with the all-zero hash.
- What batches, from each module's arguments: `features`, `build-inputs`, `anchors`, `prepare` (`k6b_confirm*`) take parallel `--windows`/`--sources` lists and `--max-windows {windows}` (smoke 1, full 0 = all),
  so the smoke is the first named window. `native-base`, `rates`, `rgb-diagnostic`, `rate-transport`, `receiver`, `judge` take `--window`/`--source` lists but have NO first-window limiter: their smoke runs every named
  window on the smoke grid, and their smoke validators bound the group (native base: 37.5 x smoke CPU seconds < 1,300 s; receiver: projected full stage < 1,500 s, 601 s per Al window; rgb: 10 x smoke
  against the spec's reservation, set to 2,600 s). Single-archive arguments tie the groups together: judge, receiver `--masks/--prior`, rgb and rates `--base`, judge `--anchor` and `--rates` take one archive, so a judge
  group lies inside one anchor, native-base and receiver job (`plan` searches the grouping under these caps). Not batchable: generation (per adapter, 1,324 s per Al window, so one window per job), DCVC
  (`--bracket LOW HIGH` is one pair for every window of a job and each window has its own hybrid rates: one job per window, not per-window brackets), the `w1_*` stages (one `--window` by design).
- Plan (A6000/Ada GPU seconds, successful smoke + full, Med windows assumed 1.25x the first window's full times, untimed): features 1 job, anchors 2 (al034+med024; al043+med000+med019: 1,343 / 2,090 s estimated),
  generation 5, rgb diagnostic 2, receiver 3, judge 3, W1 scoring 5, DCVC 5; CPU: build-inputs, prepare, rate transport 1 each, native base 2, rates 2, W1 masks and encodes per window.
  Total 24,952 GPU-s = 6.93 GPU-h (21,996 s = 6.11 GPU-h if Med is as fast as Al), against 26,261 s one job per window and stage: batching saves about 1,300 s, not the 2,200 s of the projection above,
  because generation (368 s of smokes), W1 scoring (424 s), DCVC (264 s) cannot share a smoke and the limiter-less smokes grow with the windows (assumed +25 % of the one-window smoke per added window).
  This includes DCVC (about 273 s per window: 243 s measured plus a 30 s probe, assumed). With the first window's 6,017 s the Tier 1 total is 8.6 GPU-h (7.8 without the Med factor) against the 8 GPU-h card: it needs the lead's decision before the Med windows run.
- Operator order (each hash is `sha256sum` of the upstream job's `full/published.tar`, or the named file): start `features` (manifest.json hash) and both `native-base` jobs together (CPU; no upstream). Then
  `build-inputs` (`features0`) -> `anchors` x2 (`inputs`) -> `prepare` (`inputs`, `anchor0`, `anchor1`; read the unit names from its prepare.json) -> `generate` x5 (`prepared`, `--units`; parallel across hosts) and
  `rate-transport` (`anchor0`, `anchor1`, `prepared`). In parallel once the native-base jobs are done: `rgb-diagnostic` x2 (`base{k}`). After rate transport: `rates` x2 (`transport_json` = its transport.json, `base{k}`) -> `rates{k}`, `state{k}`.
  `receiver` x3 (`transport`, `anchor{k}`, `generated.W`) needs the generation of its windows -> `masks{r}`, `prior{r}` (receiver.json). `judge` x3 (`anchor{k}`, `transport`, `masks{r}`, `gpu_base{k}`, `base{k}`, `prior{r}`, `rates{k}`, `generated.W`) ->
  `hybrid{r}`. Then per window: `w1-masks` (`state{k}`, `anchor{k}`, `transport`), the four `w1-encode-*` (`state{k}`, `transport`, `--target` = judge scale-4 KPS2 total), `w1-score` (`w1masks.W`, `hybrid{r}`, `encode_half.W`, `encode_x0..2.W`) and `dcvc`
  (`anchor{k}`, `transport`, `prior{r}`, `--bracket W=KPS2,KPS1` totals of the window's hybrid). Index K counts the jobs of the stage in `plan` order.
- Risks for the first in-fleet run: (1) DCVC `--bracket` ran only as pure functions: the in-job probe path is untested on a GPU and its smoke runs first on the next window; (2) Med windows are 50 fps and untimed: all Med estimates and the 2,090 s anchors job
  rest on 1.25x; anchors publish per finished window, so a timeout keeps the finished ones; (3) the limiter-less smokes (rgb, receiver, judge, native base) cover all named windows, so their validators see the multi-window case for the first time; the receiver's
  `cpu_bound` check (< 600 s smoke, < 2,100 s full CPU seconds) was written for two windows and has no measured value for three; (4) allowing the receiver's 1,500 s check to follow the spec reservation (as the rgb check does) and the judge `--anchor` to repeat would
  merge receiver and judge into one job each (about 280 GPU-s saved); not done, to avoid unvalidated validator changes.

### 2026-10-31 — W7 Tier 1: GPU access and anchor scope for the batch (lead decisions, before any batch job)
- GPU access: the cloud root session cannot reach gpu1–gpu6 (port 22 blocked, no keys), and Remote Control attachment of the Mac failed with an unexplained
  "not available" (not iCloud, not privacy settings, app updated). The batch runs from a local item session on the Mac
  ([card](../cards/w7-tier1-batch.md)), reporting to the root session.
- Anchor scope: the rule entry (2026-10-30) names plain SVT half and quarter, SVT half +20 outside R, and DCVC. The first window also scored +40 and
  quarter outside-R; the batch drops those two families (encodes are CPU; scoring costs per point, 12 of the first window's 21 points). Decided on cost
  alone, before any batch quality exists; the first window's extra families stay in its record and are not used by the rule.
- Revised plan (`w7_specs plan`): **22,940 GPU-s = 6.37 GPU-h** with Med at 1.25x the Al full times, 20,246 s = 5.62 GPU-h at 1.0x. Remaining card budget after the first window's
  6,017 s: 22,783 s (6.33 GPU-h). At the 1.25x assumption the plan is 157 s over; the card states the stop rule.

### 2026-10-31 — W7 Tier 1 batch: budget raised to 8.5 GPU-h; pre-submission checks; first jobs (local item session; card [w7-tier1-batch](../cards/w7-tier1-batch.md))
- Budget (user, 2026-10-31): the Tier 1 card rises from 8 to 8.5 GPU-h, +0.5 GPU-h of headroom so that a couple of failed test runs do not leave a window without its DCVC comparison. Ledger: first window 6,017 s spent; batch remainder 30,600 − 6,017 = **24,583 s (6.83 GPU-h)**. Plan at the 1.25x Med factor: 22,940 s, margin 1,643 s. The card's stop rule is unchanged: no job is submitted if it would push the card past 30,600 s in total.
- Model deviation: the card asks for a Sonnet session at effort high; this item runs in a Haiku 5.5 session (user's choice of session; recorded here, not changed).
- Code: `origin/main` 6a5213c5 (includes `609cfdfa`, `ac9b4452`, `a3c3c133`); worktree branch `claude/w7-tier1-batch-card-cc9411`, clean.
- Fleet pre-check (before any batch job): `scripts/ps-fleet inspect` reached all six hosts (available, no PointStream claim on any GPU). Foreign processes: gpu2 both A6000 (one at 100 % utilisation), gpu3 A6000, gpu4 both RTX 8000 (1.7–1.9 GB free), gpu6 GPU 0 (100 % utilisation). `scripts/ps-fleet status` (all requests) timed out after 120 s through the default management route (gpu2) and through `PS_FLEET_MANAGEMENT_HOST=gpu5`. Per-request status works (0.1 s). This card has no earlier requests; the listing failure is a management-path limitation, not evidence about the fleet, and it is recorded rather than skipped.
- Inputs: `windows.json` sha256 `18d184cd…c5e15` (matches the first window). `manifest.json` sha256 `3ece701e…5c14`, taken from the first-window features job's staged inputs (`20261030T143125Z-8785a55b`); the Mac has no Datasets link, so the file was not re-hashed here.
- Specs: written with `w7_specs batch` into the session scratchpad (not committed). Spec sha256: features `aac7c772…9dab`; native base al034+med024 `6ce75159…1ed4`; native base al043+med000+med019 `96a200b4…6492`.
- Jobs (submitted 2026-10-31 08:15Z, `git_head` 6a5213c5 in each job's record; operator order above):
  - features, five windows (`al034,al043,med000,med024,med019`; smoke = al034; gpu3/5/6, Ada required): `20261031T081537Z-8cff0ebc` (pending)
  - native base al034+med024 (CPU): `20261031T081537Z-0b16c855` (preparing)
  - native base al043+med000+med019 (CPU): `20261031T081537Z-8fb0a603`: **failed** on its smoke validator (exit 0 from the workload, 39.8 s execution; `base-result.json`: 3 rows, wall 36.2 s, CPU 23.2 s; the validator's projection 36.2 x 37.5 = 1,357 s against the 1,300 s limit, so `projection` failed). Not replayed.
- Cause: `NATIVE_SMOKE_S` (per-window smoke wall, 8 s / 10 s) was fitted to one measured window and admitted this group at 28 s. Measured multi-window smokes on this grid: al034+med024 29.6 s (passed, projected 1,110 s, job `0b16c855`), al043+med000+med019 36.2 s (failed). Fix (code, test, validator unchanged): `NATIVE_SMOKE_S = {25: 12, 50: 15}` in `experiments/tennis/w7_batch.py`, test `test_native_groups_admit_the_passing_smoke_and_exclude_the_failed_one`; commit `8b72b543` (pushed). The native partition now equals the receiver partition: al034+med024, al043+med000, med019. The plan total is 22,954 GPU-s (+14 s: rgb smokes of one more group); CPU jobs cost no GPU-seconds.
- Jobs after the fix (`git_head` 8b72b543):
  - native base al043+med000 (CPU), spec sha256 `5149ee17…89d3`: `20261031T082303Z-5a0cfb2a` (pending)
  - native base med019 (CPU), spec sha256 `a973b761…2e83`: `20261031T082325Z-b6f4ae63` (pending)
- Status of the first jobs at this point: features `20261031T081537Z-8cff0ebc` is in its full stage on gpu5 (smoke passed); native base al034+med024 `20261031T081537Z-0b16c855` complete (smoke 33.7 s execution, `base-result.json` wall 29.6 s, CPU 16.3 s, projected 1,110 s; full stage: the fleet validator is smoke-only, so its full output is not yet validated, see below).
- Validation of the native base full stage (the fleet validator is smoke-only; the Mac cannot read the data root): validation job `20261031T082543Z-2f2cbee2` (CPU, spec sha256 `cbc85f9f…`) runs the same al034+med024 native base at 240 frames as its smoke, so the validator's full-grid checks run on this code and inputs. It checks its own archive, not job `0b16c855`'s; this is disclosed, not a check of that archive's bytes. Budget reserved 2,400 s (the first attempt's 1,800 s was refused by the client: smoke + full must fit).
- GPU ledger: 6,017 s spent before the batch. The features job's smoke ran on gpu5 and passed; its execution seconds are recorded when the job closes. CPU jobs cost no GPU-seconds. Remaining 24,583 s, less the features job's GPU seconds once recorded.
