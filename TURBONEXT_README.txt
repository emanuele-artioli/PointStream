Here is the exhaustive design summary for the "Turbo-NeXt" Real-Time Pose Transfer Model. This document contains all necessary architectural details, data specifications, and optimization strategies required for an agent to generate and test the implementation without access to the original source papers.
Shutterstock
Explore
1. High-Level System Overview
The system is a hybrid generative pipeline designed to transfer a target pose sequence (skeleton) onto a reference identity (source image) at 24 FPS. It prioritizes low latency over perfect pixel fidelity by stripping down heavy components (like standard ControlNet) and utilizing distillation techniques (LCM).
Base Model: Stable Diffusion v1.5 (SD1.5).
Resolution: 512x512 (Native SD1.5 resolution).
Inference Steps: 4 Steps (via LCM-LoRA).
Latency Strategy: StreamDiffusion (FIFO Buffer) + Cached Identity + TinyVAE.

2. Detailed Model Architecture
The model consists of four distinct neural modules interacting during the forward pass.
A. The Main Backbone (SD1.5 UNet + AnimateDiff)
Structure: Standard SD1.5 UNet2DConditionModel.
Temporal Injection: Injects AnimateDiff v3 Motion Modules into the UNet. These are Transformer blocks inserted after the spatial self-attention and cross-attention layers in every block of the UNet.
Role: Handles the denoising and temporal consistency.
B. The Identity Injector (ReferenceNet)
Structure: A copy of the SD1.5 UNet (excluding Motion Modules).
Mechanism:
Takes the Reference Image (encoded to latents) as input.
"Write" Mode: During the forward pass, it intercepts the output of Self-Attention layers.
"Read" Mode: These feature maps are passed to the Main Backbone. The Main Backbone replaces its own Self-Attention with a Spatial-Attention operation that attends to these ReferenceNet features.
Optimization: Since the reference image does not change during a video generation session, this network is run ONCE at initialization. Its feature maps are cached and reused for every subsequent frame.
C. The Pose Guider (ControlNeXt Style)
Rationale: Replaces the heavy "ControlNet" (which is a full UNet copy) with a lightweight encoder.
Architecture:
Input: Pose Image (3 channels, 512x512).
Layers: A simple Strided Convolution Stack.
Conv2d(3, 32, k=4, s=2, p=1) -> SiLU (256x256)
Conv2d(32, 64, k=4, s=2, p=1) -> SiLU (128x128)
Conv2d(64, 128, k=4, s=2, p=1) -> SiLU (64x64)
Conv2d(128, 320, k=3, s=1, p=1) (Matches UNet input channels)
Injection: A Zero-Convolution layer (initialized to 0) connects this output directly to the input latents or the first block of the Main UNet.
D. The Decoder (TinyVAE / TAESD)
Structure: A distilled, lightweight version of the AutoencoderKL.
Role: Decodes the final latents (64x64) into pixel space (512x512).
Performance: ~10ms decode time (vs ~500ms for standard VAE).

3. Data Preparation & Specification
The training data has been pre-processed rigorously to reduce model "cognitive load."
Resolution: 512x512 (Square).
Cropping: Square crop around the person. Padded with black if the aspect ratio is non-square.
Background: Pure Black (RGB 0,0,0). Used segmentation masks to remove the original background.
Pose Representation (Input):
Source: YOLO26-Pose.
Format: RGB Image (Black background).
Skeleton Coloring: COCO "Rainbow" format.
Crucial: Left limbs and Right limbs have distinct colors (e.g., Left Arm=Green, Right Arm=Red) to resolve depth ambiguities.
Face: 5 distinct dots (Eyes, Nose, Ears). No lines connecting face keypoints.
Directory Structure:
dataset/video_id/scene_id/images/00001.jpg (Target)
dataset/video_id/scene_id/poses/00001.jpg (Condition)

4. Training Protocol
Base Weights: Initialize with runwayml/stable-diffusion-v1-5.
Motion Weights: Initialize Motion Modules with guoyww/animatediff-motion-adapter-v1-5-2.
Trainable Parameters:
Pose Guider: Fully trainable.
Reference Attention Layers: Trainable (to learn how to map identity to the new pose).
Motion Modules: Optional fine-tuning (recommended if domain is specific, e.g., only tennis).
Frozen: Main UNet ResNets, VAE, Text Encoder.
Loss Function: MSE Loss on the noise residual.
Scheduler (Training): DDPMScheduler (standard 1000 steps).

5. Real-Time Inference Optimization (The "Turbo" Logic)
The inference code must differ structurally from the training code to achieve speed.
LCM-LoRA Fusion:
Load latent-consistency/lcm-lora-sdv1-5.
Fuse these LoRA weights into the Main UNet.
Switch Scheduler to LCMScheduler.
Set Inference Steps to 4.
StreamDiffusion (FIFO Buffer):
Maintain a buffer of 16 Latent Frames (Batch Size = 1, Channels=4, Frames=16, H=64, W=64).
Step 1: Shift buffer left (discard oldest frame).
Step 2: Encode incoming Pose Frame -> Pose Guider -> Append to buffer.
Step 3: Run UNet on the buffer.
Step 4: Decode ONLY the last frame using TinyVAE.
Result: The model always sees 15 frames of context but only does the heavy decoding work for the 1 new frame.
Compilation:
Use torch.compile(unet, mode="reduce-overhead") (if on Linux/standard PyTorch).
Alternatively, export the UNet to TensorRT (TRT) for NVIDIA GPUs.