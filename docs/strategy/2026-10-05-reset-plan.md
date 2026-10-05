# Reset plan: archive the old codebase, restart from documentation (5 October 2026)

Purpose: hand the reset to the next session. Part A is the mechanics. Part B is
the knowledge the new documentation must carry. It is written as facts and
rules, without project history. Part C is the component map, high level below
segmentation.

## A. Mechanics

1. **Preserve before removing.** Merge `claude/pointstream-segmentation-module-3e59bf`,
   then tag `main` as `archive/pre-reset-2026-10-05` and push the tag. History,
   branches and worktrees stay; nothing is force-pushed, and
   `scripts/cleanup_merged_worktrees.sh` is not run. Do the same in the paper
   repository (`../67a9ea6275d3d9785ce57026/`).
2. **One reset commit on `main`.** Delete everything except the carry-over set
   below, add the new docs, and get the remaining tests green. Use a normal
   commit, not an orphan branch, so `git log`, blame and the archive tag stay
   one `git checkout` away.
3. **Carry-over set.** Code is cheap; verified infrastructure is not.
   - `src/segmentation/` and `tests/segmentation/` (cut its three imports into
     the old tree: `src.contracts.paths`, `src.contracts.observation`,
     `src.components.detection.weights`. Inline the few functions it uses).
   - Fleet tooling: `scripts/ps-fleet`, `experiments/jobs/` (move to a `tools/`
     or `fleet/` package) and their tests from `AGENTS.md`.
   - `pyproject.toml`, `environment.yaml`, `pytest.ini`, `.pre-commit-config.yaml`,
     trimmed to what remains.
4. **Data.** Move `Datasets/tennis_games`, `Datasets/Egocentric-10K` and
   `Datasets/pointstream-demo` to an archive directory under `Datasets` with a
   manifest (move, read-only, no deletion). Old job directories and manifests
   stay where they are; they are evidence of the archived tree.
5. **New docs replace** `AGENTS.md`, `PLAN.md`, `README.md` and `docs/`: an
   `AGENTS.md` (rules), `docs/components.md` (Part C), `docs/resources.md`
   (paths, links, environments), `docs/pitfalls.md` (Part B rules) and
   `docs/experiments.md` (protocol plus a running registry of decisions).
   Use the dataset and workstream content of
   [2026-10-05-labelled-data-direction.md](2026-10-05-labelled-data-direction.md),
   minus its history.

## B. Knowledge to carry (resources and pitfalls)

**Machines and storage**
- GPU hosts gpu1–gpu6. Fleet workers are verified on gpu3 (RTX A6000) and gpu5
  (RTX 6000 Ada, 2 GPUs, ~48 GiB each); others need `ps-fleet doctor` and
  `workers start` first. gpu4 shows ~46 GiB used while idle.
- Roots under `/home/itec/emanuele`: `Datasets` (raw datasets), `Models`
  (checkpoints by family: `YOLO/`, `SAM/`, `DWPose/`, …), and
  `Datasets/pointstream-data` (`PS_DATA_ROOT` on the fleet: jobs, manifests,
  outputs). Raw datasets resolve through `PS_DATASETS_ROOT` (default
  `~/Datasets`); do not use `PS_DATA_ROOT` for them.
- Home is NFS: per-file operations are slow. Snapshot only the needed paths
  (`--snapshot-path`; a full-repo archive was 475 MB and timed out extracting),
  allow minutes for `git status` on large checkouts, and hash big files once
  per run.
- The Mac has no project environment and its ffmpeg lacks `libx265`. Run
  ffmpeg-dependent tests on the fleet or in a pinned environment.

**Fleet discipline (`docs/workflow/long-jobs.md` carries over)**
- One module entrypoint, whole-argument scale placeholders, smoke ≤600 s,
  a validator writing substantive checks, inputs identified by a sha256
  manifest under the data root, deadline and budget. Smoke then promotes to
  full automatically in the same job.
- An uncertain submission is inspected, never replayed. Cancel only your own
  jobs.

**Models and environments**
- SAM 3.1 Object Multiplex runs only in conda env `pointstream-sam31`
  (Python 3.12, torch 2.10), from Meta's source at `~/.cache/sam3-meta`
  pinned to `2345a4ad109ac29c569da749c91d84f10dc08c40`. The checkpoint is in the
  HF cache (`models--facebook--sam3.1/.../sam3.1_multiplex.pt`, sha256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`), not in
  `Models/SAM`. Call it from the main env through a worker subprocess.
- SAM 3.1 behaviour: each text prompt resets the tracker, so use one session
  per class. A prompt that finds nothing makes propagation raise "No points
  are provided", which means empty, not a failure. The pinned build needs a
  `start_session` compatibility patch, plus an SDPA fallback on pre-Ampere
  GPUs. Measured: 0.35–0.83 s/frame (1080p–4K, two classes), 16–37 GiB peak,
  25–100 s load. Long clips need windows (~300 frames).
- The `sam3.pt` (SAM 3, image) checkpoint is a different model; always check
  which model actually runs before trusting a "SAM 3.1" label.
- YOLOE-26: only `yoloe-26n-seg.pt` and `yoloe-26x-seg.pt` are in
  `Models/YOLO`. Its text encoder is `mobileclip2_b.ts`; bind it locally or
  Ultralytics downloads. Text prompts "hand"/"arm" produce nothing on
  egocentric footage. Agreement with SAM 3.1 was J 0.15–0.31; do not use
  plain text-prompted YOLOE as a SAM substitute. ByteTrack loses hands; use
  mask-IoU association.
- Ultralytics auto-downloads when a weight path is missing or a dangling
  symlink. Always pass resolved absolute paths and fail if missing.
- ONNX Runtime GPU for DWPose has fallen back silently to CPU (missing
  `libcudnn.so.9`); assert the provider.
- VVC through the FFmpeg wrapper (preset `faster`) can exit 0 with an empty
  bitstream; check output size or use `vvencapp`.

**Evaluation and evidence rules**
- No result from synthetic stand-ins for a model (drawn masks, constant
  tables) is evidence. Every number names the job, code revision, inputs and
  GPU.
- Foreground/background regions for scoring come from dataset labels, never
  from the system under test. Every mask carries a provenance tier (`human`,
  `model_aided`, `sam_from_label_prompt`, `sam_text`).
- Charge every transmitted byte (masks, poses, appearance, metadata, model
  updates). Report decoder time.
- Masks travel losslessly (COCO-RLE); never round-trip them through lossy
  video.
- Anchors: VVC and SVT-AV1 at matched presets, plus a neural codec anchor.
  A claimable point is no more bytes at no lower quality on the agreed metric.

**Process**
- Rewriting code is cheap; GPU time and training are the constraint. Every
  experiment states its decision rule, hypothesis, competing explanation and
  budget first. Then a correctness smoke on minutes of compute, a bounded
  pilot over the one axis that matters, and only then the scaled run. Stop
  early when the decision is made.
- Rank experiments by value to the paper first and the demo second.

## C. Components (high level below segmentation)

1. **Data:** labelled datasets (OpenTTGames, RacketVision, TrackNet, VISOR,
   EgoHOS, HOT3D) behind one adapter interface with native classes, labelled-frame
   flags and provenance tiers. One domain per dataset.
2. **Segmentation** (carried over): SAM 3.1 reference, YOLOE candidates,
   label-prompted SAM for gaps, ball from point prompts, handled-object
   proposers. Output: lossless per-instance masks.
3. **Background:** remove the foreground (video inpainting; DiffuEraser was
   the working filler), then encode cheaply. The representation depends on the
   camera: near-static broadcast cameras allow a plate plus warps;
   egocentric cameras move, so expect a conventional low-rate stream.
4. **Foreground:** per-object representation (appearance reference, pose or
   keypoints, mask), decoded generatively or as coded crops. The ball is
   parametric (trajectory + blur). The racket and handled object are part of
   the problem, not decoration.
5. **Reconstruction:** composite foreground over background at the decoder.
6. **Bitstream and rate accounting:** one ledger of every byte.
7. **Evaluation:** anchors, label-region weighted metrics plus perceptual
   ones, timing, and later the ablations a TOMM paper needs.
8. **Demo:** second priority; built from the same components.
9. **Infrastructure:** fleet, provenance, manifests.
