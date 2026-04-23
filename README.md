# **Pointstream**

Pointstream is an object-centric semantic neural codec pipeline. Instead of transmitting compressed pixel residuals (like H.264, HEVC, or AV1), it transmits semantic understanding, structural keypoints, camera poses, and highly quantized generative residuals. The receiver then leverages generative AI and neural rendering to reconstruct the video. The goal is to decouple semantic motion from pixel data to achieve extreme bitrate reduction, trading bandwidth for client-side compute.

## **Domain Strategy**

To validate this architecture, we constrain our initial domain to **Tennis**. Tennis provides an ideal proving ground: the camera is largely static, the background is known, there are a limited number of actors, and occlusions are minimal.  
Once validated on tennis, the architecture is designed to scale modularly to more complex sports and eventually to general-purpose video streaming.

## **System Pipeline Architecture**

### **1\. Server-Side Analysis (The "Encoder")**

The server deconstructs a VideoChunk into semantic components:

* **Scene Classification (TODO):** Routes active gameplay ("Exchanges") to the Pointstream pipeline, while falling back to traditional codecs for crowd shots ("Interludes").  
* **Background Modeling:** Stitches a single panorama and extracts inverted camera intrinsics/extrinsics.  
* **Actor Tracking:** Isolates humans and extracts motion via models like YOLO and DWPose. Dedicated tracking for non-human rigid bodies (e.g., rackets, balls) is currently a **TODO**.  
* **Event-Driven Sparsity:** Instead of standard fixed-framerate data, Pointstream transmits Non-Uniform Structural Keypoints, only sending data when motion deviates significantly from linear interpolation.  
* **Generative Residuals:** The server uses a shared *SynthesisEngine* to generate the exact predicted client frame, compares it to ground truth, and transmits a highly compressed inter-frame residual stream to correct AI hallucinations and missing unmodeled features.

### **2\. Client-Side Synthesis (The "Decoder")**

The client uses the sparse semantic data to hallucinate the video back into existence via the exact same standalone *SynthesisEngine*:

* **Background Rendering:** Warps the base panorama using interpolated camera poses.  
* **Entity Generation:** Reconstructs moving entities via 2D diffusion models (e.g., Animate Anyone) conditioned on poses.  
* **Compositing & Residual Correction:** Decodes the generative residual stream and applies a single-pass GPU tensor addition to correct structural anomalies.  
* **Upsampling (TODO):** Lightweight Frame Interpolation and Super Resolution to smooth and upscale the final composite.

## **System Prerequisites**

Pointstream expects system-level FFmpeg tools to be available before running tests or pipeline commands:

* ffmpeg  
* ffprobe

On Ubuntu/Debian:  
sudo apt-get update  
sudo apt-get install \-y ffmpeg

If you want to force non-default executable paths (for example /opt/local/bin), set:  
export FFMPEG\_BIN=/opt/local/bin/ffmpeg  
export FFPROBE\_BIN=/opt/local/bin/ffprobe

## **Environment Setup (CUDA-aware)**

We use conda via environment.yaml strictly as a "GPU Environment Bootstrapper" to fetch heavy CUDA drivers and PyTorch binaries that standard pip struggles with. All other Python packages (Diffusers, Transformers, Animate-Anyone, testing tools, etc.) are strictly managed by pyproject.toml.  
conda env create \-f environment.yaml  
conda activate pointstream  
pip install \-e .

## **Model Weights**

PointStream expects AI model weights to be placed in the assets/weights/ directory.  
Required YOLO actor weights:

* yolo26n.pt  
* yolo26n-seg.pt  
* yolo26n-pose.pt

Optional backend-ablation weights:

* yoloe-26n-seg.pt (YOLOE detector and segmenter)  
* mobileclip2\_b.ts (required by YOLOE text prompts)  
* sam3.pt (used by \--segmenter sam3)  
* sam2\_b.pt (used by \--segmenter sam)

### **Animate-Anyone Weights**

PointStream runs Animate-Anyone from the installed package and these local configuration assets.

Expected structure:

```text
assets/animate-anyone/
  configs/prompts/
    pointstream_original.yaml
    pointstream_finetuned_tennis.yaml
  profiles/
    original/            # canonical model store
    finetuned_tennis/    # canonical model store
```

Each profile must contain the AnimateAnyone layout:

```text
stable-diffusion-v1-5/
sd-vae-ft-mse/
image_encoder/
denoising_unet.pth
reference_unet.pth
pose_guider.pth
motion_module.pth
```

When `--animate-anyone-model-dir` is not provided, PointStream defaults to:

1. `assets/animate-anyone/profiles/<variant>`
2. `~/Models/AnimateAnyone/profiles/<variant>`

You can still override explicitly with:

* `--animate-anyone-model-dir /absolute/path/to/profile`

## **Run the Pipeline**

Run with a custom input video:  
python \-m src.main \--input /path/to/input.mp4

The run always writes runtime artifacts under a timestamped directory with:

* metadata.msgpack: for semantic metadata/events  
* panorama.jpg: encoded sidecar image for background re-warping  
* residual.mp4: encoded signed residual stream (H.265 / libx265)

metadata.msgpack: intentionally stores panorama\_uri and omits raw panorama\_image pixels to keep metadata size bounded. DiskTransport always writes panorama as an encoded sidecar image (never raw pixel arrays in metadata), using a pluggable encoder strategy.

### **Useful CLI options & Ablations**

* \--num-frames N: process only the first N frames  
* \--no-summary-file: print summary only (do not write run\_summary.json)  
* \--execution-pool inline|tagged with \--cpu-workers and \--gpu-workers  
* \--actor-extractor real|mock  
* \--detector yolo26|yoloe and \--detector-caption "tennis player"  
* \--pose-estimator yolo|dwpose  
* \--segmenter yolo|yoloe|sam3|sam|none  
* \--ball-extractor difference|mock  
* \--gpu-dtype fp16|fp32|bf16|fp8\_e4m3fn|fp8\_e5m2

GenAI backend switches:

* \--enable-genai or \--disable-genai  
* \--genai-backend controlnet|animate-anyone  
* \--animate-anyone-model-variant original|finetuned\_tennis  
* \--animate-anyone-model-dir /path/to/model/profile  
* \--compositing-mask-mode alpha-heuristic|metadata-source-mask|postgen-seg-client

### **Baseline & Component Ablation Strategy**

Pointstream evaluates the effectiveness of its specialized semantic extractors iteratively using a **Whole-Frame Residual Baseline**.  
Currently, imperfections—such as missing rackets, ball trajectories, or minor background movement—are natively caught and corrected by calculating the residuals of the *entire* hallucinated frame against the original video.  
**The Ablation Workflow:**

1. Generate the hallucinated scene (background panorama \+ segmented players overlay).  
2. Compute the residual of the *entire* video. This baseline ensures a visually perfect reconstruction at the cost of higher bitrate.  
3. Iteratively enable dedicated semantic components (e.g., the ball extractor, or future racket extractors).  
4. Measure how much the introduction of these components shrinks the required residual stream, directly revealing the bitrate savings each component yields.

**Ablation Example (Fast Mock):**  
python \-m src.main \\  
  \--input /path/to/input.mp4 \\  
  \--actor-extractor mock \\  
  \--ball-extractor mock \\  
  \--execution-pool inline \\  
  \--importance-mapper uniform

## **Development & Testing**

Run the full unit test suite:  
python \-m unittest discover \-s tests \-p "test\_\*.py"

Run specific tests:

* python \-m unittest discover \-s tests \-p "test\_end\_to\_end\_mock.py"  
* python \-m unittest discover \-s tests \-p "test\_background.py"

Check coverage:  
python scripts/check\_coverage\_gate.py

Lint, Type Check, and Pre-commit:  
ruff check src tests scripts  
mypy \--config-file pyproject.toml  
pre-commit install  
pre-commit run \--all-files

## **Docker Containers**

**CPU Image:**  
docker build \-f Dockerfile.cpu \-t pointstream:cpu .  
docker run \--rm pointstream:cpu

**GPU Image** (requires NVIDIA Container Toolkit):  
docker build \-f Dockerfile.gpu \-t pointstream:gpu .  
docker run \--gpus all \--rm pointstream:gpu

## **Continuous Integration & Release**

* **CI Pipeline:** Located in .github/workflows/ci.yml. Triggers on pushes/PRs. Runs linting, typechecking, and tests (coverage run \-m pytest).  
* **Release Flow:** Located in .github/workflows/release.yml. Triggered by pushing a v\* tag. Builds source/wheel distributions, creates a GitHub release, and pushes a GPU Docker image to GHCR.

## **Research Angles & Use Cases**

* **The Downstream Bottleneck (Low-Bandwidth Streaming):** Delivering sports to rural areas by transmitting semantic data instead of heavy pixel payloads.  
* **The Upstream Bottleneck (Asymmetric Edge Uploads):** Content creators with powerful modern smartphones can run the "Server-Side Analysis" locally, uploading kilobytes of Pointstream data over weak connections.  
* **The Storage Bottleneck (Cloud Deduplication):** Social media platforms storing thousands of user videos of the same event by storing one "Scene Representation" and rendering unique user perspectives on demand.

## **TODO / Next Steps**

* **Scene Classification:** Implement early-stage classification to dynamically route Exchanges vs. Interludes.  
* **Rigid Body Extraction:** Implement dedicated semantic extractors for the ball and rackets to further shrink the required residual payload size over the baseline.  
* **Upsampling:** Implement temporal and spatial upsampling (Frame Interpolation and Super Resolution) on the client decoder.  
* **Animate-Anyone Setup:** Finalize and document the directory placement for Animate-Anyone weights.
