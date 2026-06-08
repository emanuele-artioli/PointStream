# Pointstream Project - Upcoming Tasks

## Phase 1: Configuration & Documentation
- [ ] **Update Documentation:**
  - **Action:** Sync the `default.yaml` configuration file and `README.md` to reflect all recent architectural changes (e.g., the new unified residual compression CLI arguments, codec parameters, dropped debug flags).
- [ ] **Configuration Architecture Overhaul:**
  - **Action:** Refactor the triple-duplication of settings (YAML + CLI + Env Vars). Introduce a strongly-typed Pydantic or Dataclass `Config` object as the single source of truth.
  - **Scope:** YAML files will be the definitive configuration source. CLI arguments will be restricted to runtime overrides (`--input`, `--config`, `--debug`). Environment variables will be eliminated for application tuning and restricted only to system-level paths/secrets.
  - **Parameter Cleanup:** 
    - Eliminate literal redundancies in `default.yaml` (e.g., duplicate `debug` flags). 
    - Remove redundant conceptual parameters such as `skip-eval` (handled implicitly by `evaluation-mode`).
    - Remove `actor-extractor` and allow granular mocking (e.g., `detector: mock`).
    - Unify model naming to use exact filenames (e.g., `yolo26n-seg.pt`) instead of hardcoded aliases (`yolo26`).
    - Relocate `enable-shifted-ball` from Debug to Optimizations.
    - Unify `debug` artifact generation to trigger automatically via `--log-level debug`.
    - Drop `dry-run` as it is confirmed to be unused in CI/CD and testing workflows.

## Phase 2: Bug Fixes & Technical Debt
- [ ] **FutureWarnings and Deprecations in GenAI Backend:**
  - **Symptom:** Logs show `FutureWarning: LoRACompatibleConv is deprecated`, `torch.load with weights_only=False`, `UNet3DConditionModel in_channels deprecation`, and `scale is deprecated` in diffusers models.
  - **Strategy:**
    1. Switch to the `PEFT` backend for loading LoRA weights instead of `LoRACompatibleConv`.
    2. In `src/decoder/animate_anyone_runtime.py`, add `weights_only=True` to all `torch.load()` calls for safety.
    3. Update `animate_anyone/pipelines/pipeline_pose2vid_long.py` to access `in_channels` via `self.denoising_unet.config.in_channels`.
    4. Refactor `AnimateAnyone` pipeline code to pass `scale` via `cross_attention_kwargs` instead of as a direct parameter to silence the deprecation warning.
- [ ] **Decoder Evaluation Metrics Failure (VMAF/SSIM/PSNR):**
  - **Symptom:** `ffmpeg vmaf failed: ... Error opening input file .../decoded/0001 ... Is a directory`.
  - **Strategy:** `src/decoder/mock_renderer.py` returns `frame_output_dir` (a directory of PNGs) as the `output_uri` when `debug_enabled` is False. However, `src/experiment_evaluation.py` expects a video file. To fix this, either:
    1. Update `experiment_evaluation.py` so that if `decoded_path.is_dir()`, it evaluates the image sequence directly using `ffmpeg -framerate <fps> -i <dir>/frame_%06d.png`.
    2. Update `mock_renderer.py` to always generate the final decoded `.mp4` video regardless of the debug flag.

## Phase 3: Model Additions & Experiments
- [ ] **Transport Layer Alternatives:**
  - **Action:** Analyze the current transport interface. Propose and implement 1-2 network-based alternatives (e.g., `grpc`, `websockets`, or `zmq`) that are better suited for live streaming than the local disk.
- [ ] **ControlNet Evaluation:**
  - **Action:** Create an evaluation script/test harness to run an experiment using `ControlNet`. Generate metrics to compare its performance/results against `animate-anyone`.
- [ ] **Pix2Pix Integration:**
  - **Action:** Add a `pix2pix` pipeline as an alternative to `animate-anyone` and `ControlNet`. Ensure the architecture supports training/inferring on a specific player and their skeleton frames.