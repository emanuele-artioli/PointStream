# GenAI Temporal Consistency and ControlNet Exploration

This document summarizes our research and architectural evolution regarding the integration of generative AI (GenAI) into the PointStream video coding pipeline. The primary focus was exploring conditioning strategies, fine-tuning ControlNets, and mitigating temporal artifact accumulation (flickering and "deep dream" noise loops) during autoregressive video synthesis.

## Phase 1: Zero-Shot Evaluation and Custom Fine-Tuning
Initially, we explored various ControlNet conditionings (Canny edges, DWPose skeletons, YOLO semantic segmentations, and IP-Adapter references) to synthesize tennis players from highly compressed, sparse metadata streams.
- We modified `process_dataset.py` to seamlessly extract Canny edge maps and BLIP captions from our tennis video datasets.
- We developed and executed a robust HuggingFace Diffusers training script (`train_controlnet.py`) to fine-tune `canny-controlnet`, `seg-controlnet`, `pose-controlnet`, and `ip-adapter-controlnet` over our custom dataset. 
- While single-frame generations improved, we observed that frame-by-frame synthesis resulted in massive pixel variance across time, drastically inflating the AV1 residual payload.

## Phase 2: Combating Residual Variance via Temporal Warping
To optimize the neural codec's efficiency, we introduced temporal consistency mechanisms rather than generating every frame independently from scratch.
1. **Optical Flow Warping**: We calculated the dense optical flow between successive condition maps (e.g., between Frame $N-1$ and Frame $N$ segmentation masks) and warped the previously synthesized actor crop to the current frame's position.
2. **Adaptive Denoising Strength**: We scaled the ControlNet denoising strength linearly against the optical flow magnitude, preventing high-motion frames from breaking the generation while keeping low-motion frames strictly tied to the temporal cache.
3. **Keyframe Resets**: To break artifact feedback loops, we instituted a forced generation from scratch every 8 frames.
- **Bug Fixes**: Addressed double-offsetting bugs caused by applying temporal warping to pre-cropped actor bounds, and ensured debug artifact pipelines fully supported the temporal state hooks.

## Phase 3: Condition Refinement and Cross-Frame Attention
Despite temporal warping, we encountered severe "deep dream" artifact accumulation, particularly with Canny and Segmentation conditions, where internal textures (like clothing folds) compounded into noisy blobs.

We implemented aggressive refinement strategies:
1. **Decoder-Side Reference Masking**: The initial `init_image` contained background tennis court pixels which leaked into the ControlNet condition. We applied the YOLO segmenter directly on the decoder side to perfectly isolate the player over a solid black background, ensuring the model only learned from the actor's pure texture.
2. **Cross-Frame Attention Injection**: We engineered a `ReferenceAttentionProcessor` to cache the Key (K) and Value (V) tensors from the original Reference Frame (Frame 0) and forcibly inject them into the UNet's Self-Attention layers for all subsequent frames. This mathematically forced the model to pull textures directly from the original reference latent.
- **Critical Fix**: An issue where the synthesized output degraded into a flat, grainy, black-and-white blur was traced to a double-scaling bug in the custom attention processor. PyTorch's native `scaled_dot_product_attention` already scaled queries by $1/\sqrt{d_{head}}$, and applying it manually a second time flattened the attention distribution. Removing this bug fully stabilized the temporal generations.

## Phase 4: Future Directions and Multi-Conditioning
Through our exploration, we identified the fundamental limitation of relying on single condition maps (e.g., Canny edge maps providing structure but no texture, or Segmentations providing silhouettes but no internal geometry). 

To continue exploring robust GenAI coding in future iterations, we have proposed two major architectures:

1. **Test-Time Optimization (Guidance via Energy Optimization)**
   Instead of updating IP-Adapters based on generated images, we proposed extracting conditionings (Canny, Seg, IP) from the *synthesized* image during the denoising steps. By comparing these extracted features against the target metadata conditions, we can compute an error loss and perform gradient-based latent guidance to course-correct the generation on the fly.
2. **Multi-ControlNet Training Architecture (The Ultimate Goal)**
   Instead of using separate ControlNets, we propose training a unified Multi-ControlNet model. This architecture would fuse multiple conditions—Canny Keypoints, YOLO Segmentations, DWPose, and Reference Image Prompts (IP-Adapter/subsampled colors)—into a single conditioning stream. This multi-modal approach provides strict geometric boundaries (Seg), rich internal structure (Canny/Pose), and accurate photometric data (Reference), theoretically neutralizing artifact accumulation and minimizing the need for inference-time corrections.
