### Phase 1: CLI and Configuration Cleanup
1. **Unify Debug Flags:** I ran an experiment with `debug=false` in the default config but still generated debug artifacts. This might be tied to `disable-debug-keyframes` (which was also false). 
   * **Action:** Remove the `disable-debug-keyframes` argument entirely. Consolidate all debug logic so there is only **one** master `debug` CLI argument. Wire any downstream components that relied on `disable-debug-keyframes` directly to this primary `debug` flag.
2. **Specify YOLO Models:** The CLI arguments currently allow `detector`, `pose-estimator`, and `segmenter` to be set to "yolo". 
   * **Action:** Update the configuration and validation logic to be explicit about the model version (e.g., `yolo26`, `yoloe`).
3. **Clarify `importance-mapper`:** The CLI argument `importance-mapper` defaults to `binary`, but can also be `uniform`. 
   * **Action:** Review the code handling this argument. Add comments/docstrings explaining exactly what `uniform` does. If it lacks implementation, stub it out with comments or implement the intended behavior.
4. **Clarify `panorama-warp-batch-size`:** 
   * **Action:** Analyze the codebase to determine exactly what `panorama-warp-batch-size` (currently `null`) does. Establish and hardcode/document a sensible default value for it based on the surrounding tensor/memory constraints.

### Phase 2: Pipeline & Performance Optimizations
1. **Ball Extraction Refactor:** Currently, `ball-det-model: null` handles both detection and ball segmentation. We need to align this with actor segmentation models.
   * **Action:** If `ball-extractor` is set to segmentation, attempt to pass both the actor and ball class IDs to the detector simultaneously. This allows us to run the detection model only once to save time, before segmenting and processing the ball separately. 
   * *Constraint:* Evaluate the code complexity of this shared-detection approach. If modifying the Pointstream pipeline to support this creates too much convolution or tight coupling, fallback to running the detection step twice (once for actors, once for the ball).
2. **GenAI Step Optimization:** The GenAI step is currently a massive bottleneck because it runs on every frame for every player. However, the client only requires a subset of skeleton keypoints because it interpolates the rest.
   * **Action:** Add a new CLI flag. When enabled, instead of interpolating skeletons first and running GenAI on *all* frames, the pipeline should run the GenAI model **only** on the received subset of skeleton frames. Afterwards, interpolate the missing player frames. 

### Phase 3: Residual Calculation Updates
1. **Information Thresholding:** We need to add a block-dropping step to the residual calculations to save bandwidth/processing. Currently, we calculate residuals and downsample blocks not belonging to players. 
   * **Action:** Introduce logic to drop blocks that contain an amount of information lower than a specific threshold (e.g., using entropy or a similar variance metric). 
   * **Behavior:** If a block's information is below the threshold, drop it entirely and replace it with a grey block (matching the base residual color to signal "no residuals here"). If the block clears the information threshold *but* belongs to the background, apply the downsampling factor.
   * **CLI Arguments:** Control this behavior using three arguments:
     1. `background threshold downsampling factor` (already exists)
     2. `block size` (need to add)
     3. `block information threshold` (need to add)

### Phase 4: Model Additions & Experiments
1. **Transport Alternatives:** The CLI currently has `transport: disk`. 
   * **Action:** Analyze the transport layer interface. Propose and briefly outline the implementation for 1-2 logical network-based alternatives (e.g., `grpc`, `websockets`, `zmq`) that would make sense for streaming this data.
2. **ControlNet Evaluation:** 
   * **Action:** Create a script, config, or test harness to run an experiment using `ControlNet`. We need to compare its output metrics/results directly against `animate-anyone`.
3. **Pix2Pix Integration:** 
   * **Action:** Add a `pix2pix` model pipeline as a direct alternative to `animate-anyone` and `ControlNet`. Ensure it is configured to be trained/inferred on a specific player and their corresponding skeleton frames.

Please acknowledge these requirements and let me know if you need to inspect specific files (like `config.py`, `cli.py`, or the pipeline orchestrator) before beginning Phase 1.