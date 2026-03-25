# **Project Pointstream: Architectural Blueprint**

## **1\. The Core Concept**

Pointstream is an Object-Centric Semantic Neural Codec. Instead of transmitting compressed pixel residuals (like H.264, HEVC, or AV1), it transmits semantic understanding, structural keypoints, camera poses, and highly quantized generative residuals. The receiver then leverages generative AI and neural rendering to reconstruct the video. The goal is to decouple semantic motion from pixel data to achieve extreme bitrate reduction, trading bandwidth for client-side compute.

## **2\. Domain Strategy: Starting with Tennis**

To validate this architecture, we constrain our initial domain to **Tennis**. Tennis provides an ideal proving ground: the camera is largely static, the background is known, there are a limited number of actors (players, ball kids), and occlusions are minimal.

* **Exchanges vs. Interludes:** Within tennis, Pointstream operates dynamically. The server classifies scenes into "Exchanges" (active gameplay where the semantic codec excels) and "Interludes" (crowd shots, replays). Interludes fall back to traditional codecs, while Exchanges use the Pointstream pipeline.  
* **Future Expansion:** Once validated on tennis, the architecture scales modularly to more complex sports (e.g., soccer) and eventually to general-purpose video streaming.

## **3\. System Pipeline: Server-Side Analysis (The "Encoder")**

The server deconstructs a VideoChunk (a pre-split temporal segment of gameplay, allowing for look-ahead interpolation) into semantic components.

1. **Scene Classification:** Routes the video stream to either the Pointstream pipeline or a traditional fallback codec.  
2. **Background Modeling:** Stitches a single panorama from early frames and extracts inverted camera intrinsics/extrinsics.  
   * *Transmits:* 1 Panorama Image (once per scene) \+ Sparse Camera Poses.  
3. **Actor Segmentation & Tracking:** Isolates humans and extracts motion (e.g., via YOLO-to-Student distilled models and DWPose).  
   * *Transmits:* Appearance embedding \+ **Non-Uniform Structural Keypoints** (Event-Driven Sparsity: data is only sent when motion deviates from linear interpolation).  
4. **Object Tracking (e.g., Rackets):** Tracks non-human rigid bodies using sparse point-tracking models (TAPIR/CoTracker).  
   * *Transmits:* Non-Uniform point trajectories.  
5. **Ball Tracking (The Heuristic):** Utilizes background subtraction against the stitched panorama to find the high-velocity trace of the ball.  
   * *Transmits:* Parametric coordinates (x, y, velocity vector) instead of pixels.  
6. **The Generative Residuals:** The server uses a shared, standalone SynthesisEngine to generate the exact predicted client frame. It compares this to the ground truth and isolates the difference.  
   * *Transmits:* Highly compressed, inter-frame residual video streams (HEVC/neural compression) to correct structural AI hallucinations.

## **4\. System Pipeline: Client-Side Synthesis (The "Decoder")**

The client uses the sparse semantic data to hallucinate the video back into existence via the exact same standalone SynthesisEngine used by the server.

1. **Background Rendering:** Warps the base panorama using the interpolated camera poses.  
2. **Actor & Object Generation:** Reconstructs moving entities via 2D diffusion models (e.g., *Animate Anyone*) conditioned on poses, or via 3D Neural Rendering (3D Gaussian Splatting) for ablation testing.  
3. **Ball Synthesis:** Parametrically draws the ball using trajectory data, applying physics and motion blur.  
4. **Compositing & Residual Correction:** The client decodes the generative residual stream into an independent video buffer. Using a single-pass GPU tensor addition, this buffer is overlaid onto the hallucinated video to correct anomalies.  
5. **Temporal & Spatial Upsampling:** Lightweight Frame Interpolation smooths sparse generation, and Super Resolution upscales the final composite.

## **5\. Systems & Hardware Optimization Strategy**

* **Multi-GPU Pipelining:** The server pipeline is split across available GPUs based on VRAM availability.  
* **Shared Memory & VRAM:** To eliminate PCIe latency, frame data and tracking coordinates are retained entirely within VRAM or passed via shared memory arrays.  
* **Compute Split:** CPU-bound tasks (FFmpeg decoding, event-driven keyframe logic) are separated from GPU-bound tasks (YOLO, GenAI, Tensor math) to maximize parallel throughput.

## **6\. Research Angles & Use Cases ("The Spins")**

* **The Downstream Bottleneck (Low-Bandwidth Streaming):** Delivering sports to rural areas by transmitting semantic data instead of heavy pixel payloads.  
* **The Upstream Bottleneck (Asymmetric Edge Uploads):** Content creators with powerful modern smartphones can run the "Server-Side Analysis" locally, uploading kilobytes of Pointstream data over weak connections.  
* **The Storage Bottleneck (Cloud Deduplication):** Social media platforms storing thousands of user videos of the same event by storing one "Scene Representation" and rendering unique user perspectives on demand.

## **7\. Implementation & Agent Guidelines**

The strict software architecture, data contracts, DAG orchestration rules, and network abstraction required to implement this blueprint are governed entirely by the pointstream.instructions.md file in the .github/instructions folder. All automated coding agents must adhere to those constraints to ensure a modular, scalable, and hardware-accelerated pipeline.