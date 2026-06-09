# Pointstream Project - Upcoming Tasks

## Model Additions & Experiments
- [ ] **Transport Layer Alternatives:**
  - **Action:** Analyze the current transport interface. Propose and implement 1-2 network-based alternatives (e.g., `grpc`, `websockets`, or `zmq`) that are better suited for live streaming than the local disk.
- [ ] **Generative Model Alternative:**
  - **Action:** Add `pix2pix` and `ControlNet` pipelines as alternatives to `animate-anyone` for training/inferring on a specific player and their skeleton frames.
  - **Detailed Plan:**
    1. **Mock-First Prototyping:**
       - Implement `MockControlNetPipeline` and `MockPix2PixPipeline` returning deterministic dummy PyTorch tensors matching expected visual outputs to validate pipeline plumbing before loading heavy weights.
    2. **Architecture & Integration:**
       - Create the model wrappers within `src/decoder/` (e.g., `src/decoder/controlnet_engine.py`, `src/decoder/pix2pix_engine.py`).
       - Ensure all tensor manipulations include explicit shape hint comments (e.g., `# Shape: [Batch, Channels, Height, Width]`).
       - Update the `SynthesisEngine` so that both the server and the client can symmetrically execute these new pipelines to predict hallucinated frames and calculate generative residuals.
    3. **Weight Management:**
       - Search for the native `pix2pix` and `ControlNet` weights within `/home/itec/emanuele/Models` and symlink them directly to `assets/weights/`.
    4. **Evaluation & Ablation:**
       - Benchmark the new pipelines explicitly against the "Whole-Frame Residual Baseline" using the standard test sequence (`/home/itec/emanuele/pointstream/assets/real_tennis.mp4`).
       - Measure and compare their specific impact on reducing residual payload size against the `animate-anyone` approach.