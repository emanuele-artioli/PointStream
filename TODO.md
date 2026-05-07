# Pointstream Project - Actionable Implementation Plan

Based on the recent evaluations and architectural discussions, here is the structured plan to tackle the upcoming tasks, prioritized from bug fixes to new model integrations and architectural research.

## Phase 1: Bug Fixes & Visual Fidelity

### 1. Fix Actor Bounding Box Reshaping (Squeezing Bug)
**Problem:** Actors pasted onto the background appear squeezed thin, and this distortion worsens from the first to the last frame.
**Action Plan:**
*   Investigate the `_resolve_target_bbox` and cropping/warping logic in `src/decoder/genai_compositor.py`.
*   Ensure that aspect ratios are properly preserved when mapping the `[512, 512]` generated actor crops back into the original resolution bounding boxes.
*   Check if bounding box coordinates from the extractor (e.g., YOLO) are somehow drifting or if the interpolation is shrinking them over time.

---

## Phase 2: Performance vs. Quality Trade-offs ("Fast Base + Heavy Residual")
*Goal: We need a high-quality (but slow) alternative like Animate-Anyone, and a low-quality (but fast) alternative. The idea is that a fast, poor generation can simply be corrected by a heavier traditional residual payload.*

### 2. ControlNet Completion & Evaluation
**Problem:** We need concrete metrics to understand the speed-vs-quality trade-off of `ControlNet` compared to `Animate-Anyone`.
**Action Plan:**
*   Ensure `BaselineControlNetStrategy` is fully implemented and correctly hooked up to the `process_sequence` architecture.
*   Create an evaluation script/test harness (`scripts/evaluate_models.py`) to run identical chunks through both models.
*   **Metrics to capture:** Inference Latency (seconds per chunk), Residual Size (bytes), and overall PSNR.

### 3. Pix2Pix Pipeline Integration
**Problem:** We need an even faster generative baseline. A Pix2Pix GAN could be extremely fast compared to diffusion models.
**Action Plan:**
*   Implement a new `Pix2PixStrategy` inheriting from `BaseGenAIStrategy`.
*   Design the pipeline to support training/inferring on a specific actor (e.g., feeding it a specific tennis player's skeleton frames).
*   Add documentation on how to train the Pix2Pix model and where to place the weights.

---

## Phase 3: Architectural Research & Refinements

### 4. Sparse Batch Generation + Interpolation (Animate-Anyone Speedup)
**Problem:** Generating every frame with Animate-Anyone is slow. Can we pass a single batch of *sparse* keyframes (e.g., frames 0, 10, 20, 30) to speed up inference, then interpolate the missing frames?
**Action Plan:**
*   **Experiment:** Test how `Animate-Anyone`'s temporal attention handles non-sequential pose frames. Does it break the temporal coherence, or does it generate the sparse frames correctly?
*   **Implementation:** If successful, create a "Sparse Sequence" mode in `animate_anyone_runtime.py` that skips intermediate frames in the batch, generates the keyframes, and uses linear interpolation for the gaps. 

### 5. Enforce Strict Semantic Neural Codec Boundaries (The "Approach 3" Trap)
**Problem:** Generating actors on the server and sending those generated sprites over the network defeats the purpose of the research. At that point, traditional H.264 encoding of the original actors would be cheaper.
**Action Plan:**
*   Ensure the pipeline *strictly* adheres to the neural codec philosophy: The server **must not** send generated pixel data. 
*   The payload must only consist of: 
    1.  The base panorama.
    2.  The reference crops & lightweight skeleton coordinates (DWPose).
    3.  The traditional compressed residual (H.264/HEVC) that fixes the client's hallucinations.