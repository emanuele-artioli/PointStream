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
