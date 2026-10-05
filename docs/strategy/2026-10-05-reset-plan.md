# Reset plan: archive the old codebase, restart from documentation (5 October 2026)

Purpose: hand the reset to the next session. Part A is the mechanics. Part B is
what the new documentation must say, written as facts and rules without
project history. Part C is the component map, high level below segmentation.
Part D is the order of work.

## A. Mechanics

1. **Preserve before removing.** Merge `claude/pointstream-segmentation-module-3e59bf`,
   then tag `main` as `archive/pre-reset-2026-10-05` and push the tag. History,
   branches and worktrees stay; nothing is force-pushed, and
   `scripts/cleanup_merged_worktrees.sh` is not run. Do the same in the paper
   repository (`../67a9ea6275d3d9785ce57026/`).
2. **One reset commit on `main`.** Delete everything except the carry-over set,
   add the new docs, and get the remaining tests green. Use a normal commit,
   not an orphan branch, so the archive stays one `git checkout` away.
3. **Carry-over set.**
   - `src/segmentation/` and `tests/segmentation/`. Cut its three imports into
     the old tree (`src.contracts.paths`, `src.contracts.observation`,
     `src.components.detection.weights`) by inlining the few functions it uses.
   - Fleet tooling: `scripts/ps-fleet`, `experiments/jobs/` including host-local
     staging (`staging.py`) and packed environments (`environment.py`), moved to a
     `tools/` or `fleet/` package, with their tests. Keep
     `docs/workflow/long-jobs.md` (with its host-local staging section) and the
     NFS measurements from `docs/areas/infrastructure.md` as `docs/fleet.md`.
   - `pyproject.toml`, `pytest.ini`, `.pre-commit-config.yaml`, trimmed.
     `environment.yaml` is replaced by the output of the environment audit
     (Part D, step 4).
4. **Storage links.** `Models` → `/home/itec/emanuele/Models` and `Datasets` →
   `/home/itec/emanuele/Datasets` as symlinks in the repository root, listed in
   `.gitignore`. Code resolves every weight and dataset through them. The
   previous checkout avoided in-repo data symlinks because editors and tools
   that walk the tree crawled the half-million files behind them. So the same
   commit excludes both links from editor watchers/search, pytest collection,
   ruff and mypy.
5. **Data.** Move `Datasets/tennis_games`, `Datasets/Egocentric-10K` and
   `Datasets/pointstream-demo` to an archive directory under `Datasets` with a
   manifest (move, read-only, no deletion). Old job directories and manifests
   stay where they are.
6. **New docs replace** `AGENTS.md`, `PLAN.md`, `README.md` and `docs/`:
   - `AGENTS.md`: the rules in Part B;
   - `docs/components.md`: Part C;
   - `docs/resources.md`: datasets, models, links;
   - `docs/experiments.md`: protocol plus a running registry of decisions.

   Use the dataset and workstream content of
   [2026-10-05-labelled-data-direction.md](2026-10-05-labelled-data-direction.md),
   minus its history.

## B. What the new documentation says

**AGENTS.md rules**
- *Storage:* every weight lives in `/home/itec/emanuele/Models` (by family:
  `YOLO/`, `SAM/`, …) and every dataset in `/home/itec/emanuele/Datasets`,
  reached through the `Models` and `Datasets` links in the repository. Code
  never downloads into the checkout. A missing weight is an error naming the
  expected path. Experiment outputs go under `Datasets`, never in the checkout.
- *Environments:* before adding a component, audit the repositories and models
  PointStream will use and their pinned dependency versions. Use one
  environment if they are compatible. Otherwise find the exact conflicts and
  resolve them with the fewest, cleanly separated environments, each called
  through a narrow worker interface. Record each environment's lock file and
  the reason it exists.
- *Fleet:* remote GPU work goes through `scripts/ps-fleet`: one module
  entrypoint, whole-argument scale placeholders, a smoke of at most 600 s, a
  validator with substantive checks, sha256-identified inputs, a deadline and
  a budget. Uncertain submissions are inspected, never replayed; cancel only
  your own jobs. GPU hosts are gpu1–gpu6.
- *Shared home is slow NFS:* ~200 ms per small-file create, 11–17 ms per open,
  minutes for a cold `import torch`. Never do per-file work on it. In fleet
  jobs, write intermediates to `PS_SCRATCH_DIR`, pass large or many-file inputs
  as SHA256-identified archives in `staged_inputs`, and run from a packed
  `environment` (see `docs/fleet.md`). Outside the fleet, keep caches and
  scratch in `/tmp` or `/dev/shm`. Plan work around this: datasets of many small
  files are stored and moved as archives.
- *First run of a model on a GPU class:* GPUs differ in compute capability
  (Ada 8.9, A6000 8.6, RTX 8000 7.5, GV100 7.0) and libraries fall back silently
  (ONNX Runtime to CPU without cuDNN 9; attention kernels without flash support
  before 8.0). The first time a model runs on a GPU class, a smoke asserts the
  intended device, execution provider and kernels, and the result is recorded in
  the model–GPU table in `docs/fleet.md`. Later runs read the table instead of
  re-checking, and pick `gpu_models` from it.
- *Experiments:* GPU time and training are the constraint; code is cheap.
  Before a run, write its decision rule, hypothesis, competing explanation and
  budget. Then run a correctness smoke in minutes, a bounded pilot on the one
  axis that matters, and only then the scaled run. Stop when the decision is
  made. Rank by value to the paper first and the demo second.
- *Evidence:* a number is evidence only if it names its job, code revision,
  inputs and GPU, and comes from the real component, not a stand-in.
- *Baselines:* SVT-AV1 as the conventional codec and a state-of-the-art neural
  video codec. VVC only if a correct invocation turns out to be needed.

**docs/resources.md (models; facts, not warnings)**
- SAM 3.1 Object Multiplex: Meta's source pinned to
  `2345a4ad109ac29c569da749c91d84f10dc08c40`; checkpoint
  `sam3.1_multiplex.pt`, sha256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` (move
  it into `Models/SAM`).
- YOLOE-26:
  - Only the `n` and `x` segmentation weights are in `Models/YOLO`, with the
    `mobileclip2_b.ts` text encoder, which must be bound locally.
  - ByteTrack loses fast hands; mask-IoU association works.

## C. Components (high level below segmentation)

1. **Data:** labelled datasets (OpenTTGames, RacketVision, TrackNet, VISOR,
   EgoHOS, HOT3D) behind one adapter interface with native classes,
   labelled-frame flags and a provenance field per mask.
2. **Segmentation** (carried over): SAM 3.1, YOLOE candidates, label-prompted
   SAM for classes a dataset lacks, the ball segmented with SAM 3.1 from point prompts, 
   and proposers for the handled object. Output: lossless per-instance masks.
3. **Background:** remove the foreground (video inpainting if necessary), then encode
   (more details will be provided in dedicated session).
4. **Foreground:** per-object representation (appearance reference, pose or
   keypoints, mask), decoded generatively or as coded crops (more details will be provided in dedicated session).
5. **Reconstruction:** composite foreground (with transparency) over background at the decoder.
6. **Bitstream and rate accounting.**
7. **Evaluation:** AV1 and neural-codec baselines, region-weighted and
   perceptual metrics, timing, and later the ablations a TOMM paper needs.
8. **Demo:** second priority, built from the same components.
9. **Infrastructure:** fleet, provenance, manifests, environments.

## D. Order of work

1. Merge this branch; tag `archive/pre-reset-2026-10-05` in both repositories.
2. Reset commit: new docs, carry-over code, storage links with tool
   exclusions.
3. Download the adopted datasets; archive the three old ones. This can start
   right after step 1 because it only touches `Datasets`.
4. Environment audit for the first wave: SAM 3.1, YOLOE/Ultralytics, the
   dataset tools (e.g. VISOR and HOT3D loaders), SVT-AV1 and the chosen
   neural codec. It decides the environment set and replaces
   `environment.yaml`. Each later component starts with its own audit
   increment.
5. Ground-truth adapters and label-based evaluation in `src/segmentation`.
6. In parallel: label-prompted SAM 3.1 (ball points, players from racket
   keypoints, arms), and the segmentation benchmark on labels.
7. Ball and handled-object proposers, with cross-dataset tests.
8. Training-data export, then background and foreground encoding sessions.
9. Paper rescoping alongside, with a fresh paper repository state.

The NFS guidance and host-local staging come from PR #161 (merged into this
branch).
