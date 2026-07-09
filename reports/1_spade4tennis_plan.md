# Spade4Tennis: Reference-SPADE Player+Racket Synthesis

## Problem Statement

The current Pix2Pix baseline produces **blurry** outputs because:
1. The L1 loss (λ=100) dominates the GAN loss, encouraging the model to generate "safe" averaged textures
2. Reference image features are **concatenated at input** and squeezed through the U-Net bottleneck, destroying high-frequency identity details by the time they reach the decoder
3. The single PatchGAN discriminator only enforces realism at one spatial scale
4. Rackets are not synthesized at all—they rely entirely on residual transmission

---

## Architecture Candidates Evaluated

| Architecture | Params | VRAM (train, per GPU) | Training Time (2×A6000) | Inference (512×512) | Quality Ceiling |
|---|---|---|---|---|---|
| **Current Pix2Pix UNet** | 57M | ~39 GB (batch=64) | ✅ 9 hours | ~5ms | Low (blurry) |
| **Pix2PixHD Global** | ~50M G + 30M D | ~35 GB (batch=32) | ✅ ~25 hours | ~4ms | Medium-High |
| **Pix2PixHD Full (G+L)** | ~190M | ~42 GB (batch=16) | ⚠️ ~61 hours | ~12ms | High |
| **SPADE/GauGAN** | ~100M | ~40 GB (batch=16) | ⚠️ ~53 hours | ~8ms | High |
| **SD 1.5 + ControlNet LoRA** | 1.2B (860M frozen) | ~24 GB (batch=2) | ❌ ~123 hours | ~800ms (20 steps) | Very High |
| **SD Turbo + adapter** | 860M | ~20 GB (batch=2) | ❌ ~80 hours | ~40ms (1 step) | High |

> [!IMPORTANT]
> **SD-based approaches are ruled out** for training within our 2-day budget. At batch=2 per GPU, ControlNet fine-tuning would take ~5 days. SD Turbo adapter training would take ~3.5 days. The pretrained models are locally available (`/home/itec/emanuele/Models/stable-diffusion-v1-5/`, `sd_turbo.safetensors`, `control_v11p_sd15_openpose/`), but training time is the bottleneck, not VRAM.

---

## Recommended Architecture: SPADE-Conditioned GAN

The key insight is that we don't need diffusion. What we need is to **fix the three root causes of blur** in our current GAN:

### Why SPADE Solves Blur

In the current architecture, the reference image is concatenated to the skeleton at the **input** (6-channel). By the time it passes through 8 downsampling layers to the 1×1 bottleneck, all texture details are destroyed. The decoder must reconstruct them from a 512-dimensional vector—which it can't.

**SPADE** (Spatially-Adaptive Denormalization) works differently: the reference image is encoded by a **separate lightweight encoder**, and its features are injected into **every normalization layer** of the decoder via learned affine transforms. This means:
- The generator skeleton-encoder can focus purely on spatial structure
- High-frequency reference textures (skin tone, clothing wrinkles, shoe color) are injected **directly where they're needed** in the decoder
- The bottleneck doesn't need to carry reference information at all

### Loss Function Stack

| Loss | Weight | Purpose |
|---|---|---|
| Hinge GAN loss | 1.0 | Adversarial sharpness (more stable than BCE) |
| VGG-19 perceptual loss | 10.0 | Match high-level textures, not pixel values |
| Feature matching loss | 10.0 | Match discriminator's intermediate features |
| L1 pixel loss | 10.0 | Structural alignment (reduced from 100→10) |

> [!NOTE]
> The dramatic reduction of L1 weight from 100→10 is critical. At λ=100, the L1 loss was ~80% of the total generator loss (G_loss≈8.0, of which L1 contributed ~7.0). This forced the model to minimize pixel error at the expense of sharpness. At λ=10, the perceptual and adversarial losses gain proportional influence.

### Multi-Scale Discriminator

Three PatchGAN discriminators operating at:
- **D1**: Original resolution (512×512) → enforces micro-texture sharpness
- **D2**: 2× downsampled (256×256) → enforces medium-scale coherence  
- **D3**: 4× downsampled (128×128) → enforces global structure consistency

All use spectral normalization for stable training. Combined into a single forward pass for DDP compatibility (same technique as our current implementation).

---

## Racket Synthesis Strategy

Currently, rackets are detected by YOLO as separate COCO objects (class 38) but are **never passed through the GenAI pipeline**. They rely entirely on residual transmission, which is expensive.

## Racket Keypoint System & Rendering

We will define a **Racket Heuristic Skeleton** that uses the player's wrist and the racket's segmentation:
1. **Handle Base**: The racket handle is anchored at the dominant wrist (which we know from DWPose). We identify the dominant wrist by checking which wrist is closest to the racket bounding box.
2. **Racket Shape**: Given the racket segmented image, we need to devise a way to represent the racket shape as a set of keypoints.

### BBox Merging
We will perfectly merge the bounding boxes: `[min(p_x1, r_x1), min(p_y1, r_y1), max(p_x2, r_x2), max(p_y2, r_y2)]`. We will add a small 5-10% padding to ensure the racket isn't cut off by the border.

## Progressive Multi-Tier Training

You asked about the best way to get both a fast Lite model and a high-quality Full model. Knowledge distillation is one option, but Pix2PixHD's architecture provides a much cleaner solution: **Progressive Growing**.

The Pix2PixHD architecture is natively split into two networks:
1. **Global Generator (Lite)**: A fast, lightweight ResNet-9 that operates at 512x512.
2. **Local Enhancer (Full)**: A second network attached to the Global Generator that refines details.

This perfectly matches your request. We will:
1. Train the **Lite model (Global Generator)** from scratch. This gives us our fast inference model (~3ms).
2. Validate it.
3. Once validated, we freeze or joint-train the Lite model and attach the **Local Enhancer** to create the **Full model**. The enhancer learns to add high-frequency details on top of the Lite model's output.

### Tier 1: Lite Model (Fast Inference)
| Property | Value |
|---|---|
| Generator | ResNet-9 with SPADE reference injection (~15M params) |
| Discriminator | 2-scale PatchGAN with spectral norm (~6M params) |
| Losses | Hinge GAN + VGG perceptual + L1 (λ=10) |
| Batch size | 64 per GPU |
| Training | ~12 hours |
| Inference | ~3ms per forward pass |

### Tier 2: Full Model (Maximum Quality)
| Property | Value |
|---|---|
| Generator | Global Generator + Local Enhancer with SPADE (~55M params) |
| Discriminator | 3-scale PatchGAN with spectral norm + feature matching (~15M params) |
| Losses | Hinge GAN + VGG perceptual + Feature matching + L1 (λ=10) |
| Batch size | 32 per GPU |
| Training | ~24 hours (starting from pretrained Lite model) |
| Inference | ~12ms per forward pass |

---

## VGG Pre-trained Model Management

We will download the standard PyTorch `vgg19` weights directly to `/home/itec/emanuele/Models/vgg19/` via a download script, and then create a symlink to `assets/weights/vgg19-bn.pth`. We will configure `torchvision` to load from this local path to avoid hidden cache directories.

---

## Proposed Changes

### Dataset Pipeline

#### [NEW] [extract_player_racket_crops.py](file:///home/itec/emanuele/pointstream/scripts/extract_player_racket_crops.py)
- Modified version of the existing extraction script
- Expands player bounding boxes to include associated racket regions
- Adds racket endpoint to skeleton rendering
- Outputs to `assets/dataset/pix2pix_v2/`

---

### Training Script

#### [NEW] [train_spade4tennis.py](file:///home/itec/emanuele/pointstream/scripts/train_spade4tennis.py)
- `--model-size lite|full` flag for tier selection
- SPADE-conditioned generator (ResNet-9 or UNet depending on tier)
- Multi-scale discriminator (1 or 3 scales depending on tier)
- VGG-19 perceptual loss + feature matching loss
- Hinge GAN loss (replaces BCE)
- Spectral normalization on all discriminator layers
- DDP-compatible combined forward passes
- Saves to `assets/weights/spade4tennis_generator.pt`

---

### Inference Engine

#### [NEW] [spade4tennis_engine.py](file:///home/itec/emanuele/pointstream/src/decoder/spade4tennis_engine.py)
- `Spade4TennisStrategy(BaseGenAIStrategy)` implementation
- Loads SPADE generator from `spade4tennis_generator.pt`
- Handles reference image preprocessing via SPADE encoder (not concatenation)
- Supports both lite and full model architectures (auto-detected from checkpoint)

#### [MODIFY] [genai_compositor.py](file:///home/itec/emanuele/pointstream/src/decoder/genai_compositor.py)
- Add `spade4tennis` backend option to `_build_strategy()`

#### [MODIFY] [default.yaml](file:///home/itec/emanuele/pointstream/config/default.yaml)
- Add `spade4tennis` as a valid `genai-backend` option (keep `pix2pix` as baseline)

---

## Verification Plan

### Automated Tests
- Unit test for SPADE normalization layer: verify output shape matches input shape
- Unit test for multi-scale discriminator: verify 3 separate gradient-capable outputs
- Unit test for VGG perceptual loss: verify feature extraction produces correct shapes
- Smoke test: run 2 epochs on a 100-image subset to verify training loop stability
```bash
conda run -n pointstream python scripts/train_spade4tennis.py --model-size lite --epochs 2 --subset djokovic_front
```

### Manual Verification
- Compare epoch sample images between v1 (baseline) and v2 (SPADE) at epoch 50, 100, 200
- Run full Pointstream pipeline with `--genai-backend spade4tennis` and compare PSNR/visual quality
- Check inference time regression via `run_summary.json` timings
