# Scene Classification Research Notes: Heuristic & Triage Iterations

This document summarizes our iterative pipeline for robustly extracting and classifying tennis video scenes. It covers the evolution of our extraction pipeline, feature engineering to balance clustering, our initial struggle with fragmented scene boundaries caused by high-speed camera pans, the failures of Vision-Language Models (VLMs) for semantic classification, and the ultimate success of adaptive statistical thresholding and pairwise visual invariants. These notes are formatted for inclusion in the PointStream project paper.

## Phase 1: High-Performance GPU Extraction Pipeline
We initially started with separate `process_dataset_2.py` and `run_batch.py` scripts, which were slow and bottle-necked by CPU decoding. 
To scale to large 4K datasets, we unified the scripts into `dataset_processing.py` and implemented a massive hardware-accelerated pipeline:
- **Parallel Hardware Acceleration:** We leveraged 96 CPU cores and 2x 48GB VRAM GPUs (`cuda:0`, `cuda:1`) using `torch.multiprocessing`, intelligently mapping parallel workers across available GPUs.
- **Chunked Processing:** To prevent memory overflow and allow safe state-saving, we processed the massive 4K videos in 1-minute temporal chunks.
- **FFmpeg Decoding:** We replaced slow frame-by-frame Python decoding with hardware-accelerated `ffmpeg -hwaccel cuda` batch extraction, radically improving the speed of the optical flow metric generation.

## Phase 2: Feature Engineering & Clustering Fairness
To properly classify *Points* vs *Interludes*, we needed to balance motion metrics (`avg_score`, `std_score`, `max_score`) against temporal `duration`.
- **Logarithmic Scaling & Whitening:** We found that linear `duration` overshadowed small motion deltas. We applied log scaling to normalize the distributions of both time and motion metrics. We utilized statistical *whitening* to ensure equal variance across all dimensions, preventing any single metric from unfairly skewing the clustering space.
- **PCA (Principal Component Analysis):** We applied PCA to decorrelate the motion features before feeding them into the Gaussian Mixture Model (GMM), ensuring that highly correlated motion metrics didn't outvote `duration`.
- **Outlier Detection:** We appended the L2 distance from the cluster center to every generated frame filename, allowing visual inspection of boundary cases and outliers to refine our heuristics.

## Phase 3: Scene Boundary Detection & Temporal Fusion
Before semantic classification, our optical-flow-based boundary detector (`scipy.signal.find_peaks`) suffered from extreme fragmentation. We replaced fixed global thresholds with a dynamic peak prominence threshold, but fast camera motion still plagued the system.

### Attempt 1: Median Filtering & Hard Duration Limits
Initially, we attempted to smooth the raw optical flow scores using a median filter, and subsequently enforced a hard `0.5s` minimum duration limit to force fragmented scenes to fuse.
> [!WARNING]
> **Failure Mode:** The median filter destructively erased true, rapid hard-cuts. Furthermore, blindly enforcing a `0.5s` minimum duration overrode legitimate short cuts (e.g., rapid cinematic cuts in highlight reels) without actually understanding the visual context.

### Attempt 2: Visual Invariance Filtering via Pairwise HSV Histograms
We realized that optical flow cannot natively distinguish between a **hard cut** and a **high-speed camera pan**, as both generate massive frame-to-frame pixel deltas. However, a fast pan preserves *global color distribution* (the ratio of red clay to green stands), while a true hard cut instantly scrambles it.

We implemented a visual validation layer immediately after `find_peaks`:
1. Iterate over every proposed cut. If it borders a scene shorter than `1.0s`, it is flagged as suspect.
2. Extract the frame exactly `0.05s` *before* the cut, and `0.05s` *after* the cut.
3. Compute the 2D HSV Histogram (Hue + Saturation, ignoring Value/Brightness for shadow invariance) for both frames at a low resolution (`320x180`).
4. Calculate the Correlation between the two histograms.

> [!TIP]
> **Why this works:** The temporal proximity of `0.05s` (just ~0.1 seconds apart) is the key. During a fast pan, the color distribution barely drifts in 0.1 seconds, yielding a correlation of `>0.96`. On a true hard cut, the correlation instantly plummets to `~0.001`. 
> 
> By setting a similarity threshold of `0.85`, the system naturally cascaded: if a fast pan generated 10 fragmented cuts in a row, the algorithm evaluated each internal boundary, proved they were all continuous, and seamlessly dropped the cuts, merging 10+ sub-second fragments back into a single massive 10-second shot.

*(Note: We encountered a silent C++ ABI mismatch between `opencv-python==4.11` (NumPy 2.x) and `moviepy` (NumPy 1.x) that caused `cv2.resize` to fail on valid frame buffers. Pinning `opencv-python==4.8.0.76` and `numpy==1.26.4` was strictly required for the C-level buffer protocol to bridge the packages).*

## Phase 4: Semantic Scene Classification (Heuristics vs VLMs)

Our baseline system utilized a Gaussian Mixture Model (GMM) clustering over the normalized motion metrics to distinguish **Points** (`cluster_point`) from **Closeups & Replays** (`cluster_interlude`).

### Attempt 1: CLIP Zero-Shot Classification for Resolving Closeups
We attempted to use `ViT-B/32` CLIP to classify ambiguous scenes using three prompts:
1. `"A wide shot of a tennis court during a match"`
2. `"A close up photo of a tennis player or coach"`
3. `"A crowd of spectators, audience, or a scoreboard graphic"`

> [!WARNING]
> **Failure Mode:** CLIP consistently and confidently misclassified closeups of players as "wide shots of a tennis court". 
> 
> Testing on a pure closeup frame, CLIP assigned a `0.64` softmax probability to the wide shot prompt, and a `0.005` probability to the closeup prompt. Because tennis closeups natively include the green/blue court geometry and net in the background, zero-shot CLIP lacked the spatial awareness to prioritize the foreground subject over the background environment.

### Attempt 2: Prompt Engineering & Framing-Centric Captions
We hypothesized that the textual embedding for "tennis court" was overpowering the image embedding. We tested three alternative sets of captions, ranging from framing-specific (`"A full tennis court viewed from above"` vs `"A portrait of a person's face"`) to minimalist (`"Tennis court"` vs `"Person's face"` vs `"Crowd or text"`).

> [!WARNING]
> **Failure Mode:** Removing the word "tennis" from the closeup prompt caused the model to latch onto the background crowd instead. For example, in the minimalist set, the "Crowd or text" prompt won with a `0.50` probability (since audiences are visible out-of-focus behind the player). The fundamental multimodal composition of a tennis closeup (Player + Court + Crowd) scrambles zero-shot holistic embeddings.

### Attempt 3: 1D K-Means on Log Motion Distribution
Realizing VLMs were unsuited for this specific distinction, we returned to motion heuristics. We attempted to apply K-Means (K=2) on the logarithmic `avg_score` of all clusters to automatically find a dynamic threshold separating the low-motion points from the high-motion closeups.

> [!WARNING]
> **Failure Mode:** This worked flawlessly for bimodal/trimodal videos containing a mix of points and interludes (e.g., separating `0.0008` points from `0.0035` closeups). However, it failed catastrophically on unimodal videos containing *only* points (e.g., `djokovic_zverev`), arbitrarily bisecting a uniform point distribution into two artificial classes.

---

## Phase 5: The Final Solution: Point-Anchored Distribution Thresholding

We determined that the GMM successfully separates points from closeups internally, but hardcoded multiplier thresholds (e.g., `2.5x point_mean`) failed to generalize across videos shot with differing baseline camera shakes.

Our final, robust solution relies 100% on modeling the internal statistical distribution of the core point cluster.

**The Algorithm:**
1. **Identify the Core `point_cluster`**: Find the cluster with the lowest mean `avg_score`, filtering out tiny outlier clusters (requiring the cluster to hold at least `5%` of total scenes).
2. **Calculate Internal Spread**: Extract both the `mean` and the absolute `max` motion (`avg_score`) of the scenes *inside* that specific `point_cluster`.
3. **Set a Dynamic Interlude Threshold**:
   ```python
   # Buffer 1.5x above the maximum motion the GMM allowed inside the point cluster
   dynamic_threshold = max(point_max * 1.5, point_mean * 2.0)
   ```
4. **Map the Remaining Clusters**: Any cluster whose mean `avg_score` exceeds the `dynamic_threshold` is hard-mapped to `cluster_interlude`. Otherwise, it is mapped to `cluster_point`.
5. **Prune the Garbage**: We discard the bottom 10% of scenes (based on individual GMM cluster confidence) by tagging them as `cluster_other`.

> [!TIP]
> **Why this works:** It mathematically anchors the separation boundary to the specific noise floor of the given video. If a video is shot with a jittery camera, the point cluster's internal `max` naturally expands, pushing the interlude threshold higher to prevent false positives. This allowed us to cleanly remove the massive `clip` and `torch` overhead from the pipeline while achieving near-perfect separation across 4K datasets.

---

## Phase 6: Pixel-Perfect Racket Skeleton Tracking

Alongside the player's DWPose skeleton, rendering an accurate 5-point skeleton for the tennis racket proved remarkably challenging due to motion blur, diagonal orientations, and the disparity between the segmentation mask and pose estimation network.

### Attempt 1: Axis-Aligned Bounding Box (AABB) Ray Intersections
Initially, we used YOLO's standard bounding box for the racket. We drew a vector from the player's wrist through the center of the bounding box to find the "tip", and used orthogonal ray intersections to define the head width.
> [!WARNING]
> **Failure Mode:** A tennis racket held diagonally creates a massive, square Axis-Aligned Bounding Box (AABB). When the heuristic rays intersected this square, it drastically inflated the calculated width of the racket head, drawing skeletons that looked like large squares rather than thin rackets.

### Attempt 2: Interpolation & Hand Majority Voting
We frequently observed the skeleton glitching out in two ways:
1. **Zero-Origin Snapping:** When motion blur caused YOLO to drop a racket detection for 1-2 frames, the code defaulted to drawing the racket to the `(0, 0)` origin in the top-left corner.
2. **Wrist Flipping:** If the player held the racket with two hands, the heuristic would rapidly flip between attaching the racket to the left and right wrist on a frame-by-frame basis, violently twisting the skeleton.

**Solutions:** 
- We implemented linear interpolation across the raw `metadata.json` payloads over dropped frames, ensuring the bounding box and mask data gracefully persisted through motion-blurred swings. 
- We implemented a **Majority Hand Voting** pass per track. The pipeline counts which wrist holds the racket the most across the entire sub-clip, and then rigidly forces that dominant hand for all frames in the track.

### Phase 3: Convex Hull Extreme Points
To solve the diagonal bounding box inflation, we completely bypassed bounding boxes. Inside the extraction pipeline, we grabbed the original high-resolution YOLO segmentation mask and extracted its `cv2.convexHull()`.
1. We calculated the two points on the hull furthest apart from one another. Given the elongated shape of a racket, these perfectly mapped to the **Tip** and the **Handle Base**.
2. We calculated the orthogonal normal vector to that axis, and found the two mask points furthest along the normal. These mapped perfectly to the **Head Left** and **Head Right**.

> [!TIP]
> By extracting these 4 precise points and saving them to the metadata payload, the drawing heuristic could trace the skeleton along the outermost true pixels of the mask, eliminating all diagonal inflation artifacts natively.

### Phase 4: Wrist Anchoring via Translation Offset
While the Convex Hull provided a pixel-perfect shape, a new issue arose: the skeleton appeared to float disconnected from the player's hand. 

> [!WARNING]
> **Failure Mode:** YOLO's segmentation mask (which defines the racket) and DWPose's keypoints (which define the wrist) are generated by two completely separate neural networks. During high motion, they frequently drift out of visual sync. Connecting the racket points strictly to the mask left the racket detached from the wrist.

**The Final Fix:** 
We instructed the heuristic to calculate a 2D offset vector between the mask's true `handle_base` and the DWPose `wrist` coordinate. We then mathematically translated all 5 of the racket points by that exact offset. 

This final iteration elegantly married the pixel-perfect structural shape of the segmentation mask with the strict anatomical anchor of the player's wrist, resulting in a cohesive, jitter-free skeleton!
