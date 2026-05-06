# Pointstream Project - Upcoming Tasks

## Phase 1: Configuration & Documentation
- [ ] **Expose Codec Parameters:**
  - **Action:** Expose FFMPEG/codec quality settings (specifically `crf` and `preset`) in the CLI arguments so we can easily tune bandwidth vs. quality without hardcoding.
- [ ] **Update Documentation:**
  - **Action:** Sync the `default.yaml` configuration file and `README.md` to reflect all recent architectural changes (e.g., the new unified residual compression CLI arguments, codec parameters, dropped debug flags).

## Phase 2: Model Additions & Experiments
- [ ] **Transport Layer Alternatives:**
  - **Action:** Analyze the current transport interface. Propose and implement 1-2 network-based alternatives (e.g., `grpc`, `websockets`, or `zmq`) that are better suited for live streaming than the local disk.
- [ ] **ControlNet Evaluation:**
  - **Action:** Create an evaluation script/test harness to run an experiment using `ControlNet`. Generate metrics to compare its performance/results against `animate-anyone`.
- [ ] **Pix2Pix Integration:**
  - **Action:** Add a `pix2pix` pipeline as an alternative to `animate-anyone` and `ControlNet`. Ensure the architecture supports training/inferring on a specific player and their skeleton frames.