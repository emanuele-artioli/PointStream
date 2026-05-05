# Pointstream Project - Upcoming Tasks

## Phase 1: Pipeline & Performance Optimizations
- [x] **GenAI Execution Optimization:** 
  - **Issue:** The GenAI step is currently a massive bottleneck because it runs on every frame for every player. 
  - **Action:** Add a CLI flag to run the GenAI model **only** on the subset of skeleton frames we actually receive from the network. After running GenAI on those keyframes, interpolate the missing player frames locally, bypassing the expensive GenAI model for interpolated frames.
- [ ] **Ball Extraction Refactor:**
  - **Issue:** `ball-det-model: null` handles both detection and ball segmentation, duplicating detection effort if we are already detecting actors.
  - **Action:** If the `ball-extractor` is set to segmentation, pass both the actor and ball class IDs to the detector simultaneously. Run detection once, then branch off the ball for separate segmentation/processing. *(Note: If this creates too much tight coupling/complexity in the pipeline, fall back to running detection twice).*

## Phase 2: Configuration & Documentation
- [ ] **Expose Codec Parameters:**
  - **Action:** Expose FFMPEG/codec quality settings (specifically `crf` and `preset`) in the CLI arguments so we can easily tune bandwidth vs. quality without hardcoding.
- [ ] **Update Documentation:**
  - **Action:** Sync the `default.yaml` configuration file and `README.md` to reflect all recent architectural changes (e.g., the new unified residual compression CLI arguments, codec parameters, dropped debug flags).

## Phase 3: Metrics & Telemetry Cleanup
- [ ] **Granular Pipeline Profiling:**
  - **Issue:** `encode_chunk_sec` and `decode_chunk_sec` are too generic.
  - **Action:** Instrument the pipeline to record step-by-step timings. Break down the encoding/decoding metrics into specific stages: detection time, segmentation time, GenAI inference time, residual calculation time, etc.
- [ ] **Deduplicate Evaluation Summary:**
  - **Issue:** Many fields in the evaluation summary are exact copies of the fields in the mail body of the `run_summary`.
  - **Action:** Clean up the reporting logic to remove these repetitions.
- [x] **Remove Irrelevant Metrics:**
  - **Action:** Delete `decoded_vs_reference_percent`. The size ratio between the original video and decoded frames is irrelevant because decoded frames are synthesized/uncompressed on the client side and don't impact network bandwidth.
- [ ] **Investigate Transport Latency:**
  - **Issue:** `transport_send_sec` and `transport_receive_sec` are taking ~2 seconds, which makes no sense for a server and client running on the same machine.
  - **Action:** Profile the `transport: disk` implementation. Find out where this 2-second bottleneck is coming from (e.g., blocking I/O, heavy file writes, inefficient serialization) and fix or annotate it.

## Phase 4: Model Additions & Experiments
- [ ] **Transport Layer Alternatives:**
  - **Action:** Analyze the current transport interface. Propose and implement 1-2 network-based alternatives (e.g., `grpc`, `websockets`, or `zmq`) that are better suited for live streaming than the local disk.
- [ ] **ControlNet Evaluation:**
  - **Action:** Create an evaluation script/test harness to run an experiment using `ControlNet`. Generate metrics to compare its performance/results against `animate-anyone`.
- [ ] **Pix2Pix Integration:**
  - **Action:** Add a `pix2pix` pipeline as an alternative to `animate-anyone` and `ControlNet`. Ensure the architecture supports training/inferring on a specific player and their skeleton frames.