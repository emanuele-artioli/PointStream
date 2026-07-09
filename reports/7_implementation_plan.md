# PointStream Implementation Plan (ACM TOMM)

This document is the comprehensive source of truth for the PointStream project. It fuses our latest internal research, the feedback from the ACM MM reviewers, and our strategic priorities into a complete guide for our remaining development and paper writing.

## 1. Core Paradigm: Residual-Guaranteed Encoding & Ablation Strategy
*This is the defining architectural philosophy of PointStream and must be explicitly referenced throughout the paper.*

*   **The Residual Guarantee**: PointStream is a hybrid codec that guarantees performance on par with traditional codecs by encoding a residual video. After all transformations (panorama warping, generative synthesis) are applied on the server, a residual video is calculated against the original frames. Because these steps are deterministically replicated on the client side, adding the residual perfectly restores the original video.
*   **The Optimization Goal**: The goal of PointStream is to generate metadata that results in a generative reconstruction *close enough* to the original that the resulting residual is tiny. The sum of (Metadata Size + Residual Size) must be smaller than a standard full-frame encoding.
*   **Handling Edge Cases**: This natively addresses reviewer concerns about **dynamic shadows** (R3) or generative imperfections. Shadows are naturally absorbed and corrected by the residual video, eliminating the need for a bespoke semantic shadow module.
*   **Ablation Testing Framework**: This paradigm gives us a rigorous mathematical framework for ablation tests (crucial for a journal paper). If we disable every PointStream component, the residual video is an exact negative of the original video (theoretically the same size). As we enable each component (e.g., racket tracking, panorama warping), it generates some metadata. We can mathematically prove a component's worth by showing that it reduces the residual video size by *more* than the size of the metadata it introduced. 

---

## 2. Immediate Engineering & Evaluation Tasks
*Before finalizing the paper, we must complete these technical components.*

### A. Generative Architecture Evaluation & Trade-offs
We have not yet selected a final generative architecture. We must finish building and evaluating our candidates to identify trade-offs in temporal consistency, visual quality, and inference latency.
*   **ControlNets**: Complete the ongoing work on ControlNets to ensure decent, stable results. Include evaluations of our temporal mechanisms (Optical Flow Warping, Adaptive Denoising, Cross-Frame Attention).
*   **Animate Anyone**: Thoroughly evaluate the `Moore-AnimateAnyone` integration.
*   **SPADE/GAN**: Compare the above diffusion-based methods against our SPADE implementation to finalize the architecture for the paper.

### B. Background Panorama Stitching
Transition from static backgrounds to stitching a full background panorama to allow for moderate camera movement, completing the implementation before drafting its section.

### C. Detection and Segmentation Model Selection
Evaluate modern vision models to balance quality against real-time constraints:
*   **SAM3** (high quality, computationally heavy) vs. **YOLOv26** (real-time SOTA) vs. **RF-DETR** (comparable to YOLO).

### D. Codec Benchmarks (Focused Scope)
*   **Traditional Codecs**: Benchmark exclusively against **AV1** (addresses R2, R5).
*   **Learned Codecs**: Identify and benchmark against a well-documented, modern neural codec (e.g., **HNeRV** or a modern DVC equivalent like **DCVC**) to satisfy reviewers without struggling with obsolete models.

### E. Component Ablation Experiments
Leverage the "Ablation Testing Framework" (described in Section 1) to validate our heuristics. Since we lack a validated ground-truth dataset, we will measure success purely by bitrate reduction:
*   Show how **Convex Hull Racket Tracking** and **Wrist Translation Offsets** reduce the residual size compared to naive bounding boxes.
*   Show how **Dynamic Thresholding** (scene classification) minimizes wasted bitrate on complex interlude scenes.

---

## 3. Paper Drafting Strategy (System Design Sections)
*The `main.tex` file will be expanded using this comprehensive structure, incorporating both our internal research progress and the reviewer responses.*

### 3.1 Scene Classification and Statistical Triage
*   **Invariance Filtering**: Explain the shift from fixed optical flow thresholds to Pairwise HSV Histogram correlation to distinguish true hard cuts from high-speed camera pans (answers R2 regarding dynamic backgrounds).
*   **Statistical Thresholding**: Detail the point-anchored distribution logic (1.5x max motion of the core point cluster) to handle complex scenes without relying on fragile Vision-Language Models.

### 3.2 Semantic Extraction & Racket Heuristics
*   **Model Selection**: Briefly justify the computational choice between YOLO, SAM3, and RF-DETR to maintain real-time constraints (answers R3).
*   **Pixel-Perfect Racket Tracking**: Detail the evolution from naive Axis-Aligned Bounding Boxes to extracting extreme points via the `cv2.convexHull()` of the segmentation mask.
*   **Wrist Anchoring**: Explain the 2D offset translation used to anchor the racket to the DWPose wrist, ensuring sync during high motion.

### 3.3 Unified Dataset and Pipeline Architecture
*   **Shared Library Pattern**: Discuss the unification of the extraction pipeline and GenAI backend.
*   **Universal Dataset Format**: Explain the shift to a unified standard (segmented, transparent RGBA PNG sequences paired with DWPose skeletons) that natively supports training multiple generative architectures.

### 3.4 Generative Reconstruction Engine
*   **Architecture Trade-offs**: Detail the exploration of SPADE, Animate Anyone, and ControlNets. Explicitly explain the trade-offs (training budgets, latency) that led to the final configuration (answers R1, R5).
*   **Temporal Coherence**: Document the strategies developed to combat artifact accumulation (Optical Flow Warping, Cross-Frame Attention, Reference Masking) to prove video continuity (answers R3, R4).

### 3.5 Residual Encoding & Edge Cases
*   **The Residual Guarantee**: Formally define the residual encoding philosophy in the methodology.
*   **Dynamic Shadows**: Use the residual concept to explicitly explain how dynamic shadows cast by moving humans are natively handled without additional metadata (answers R3).

---

## 4. Next Steps / Future Work
*These items are strategically delayed so we can focus strictly on tennis performance and core evaluations.*

*   **Extended Benchmarks**: Benchmarking against VVC and additional neural codecs.
*   **Domain Expansion**: Processing datasets from a second domain (e.g., basketball, pedestrian video).
*   **Subjective Evaluation**: Conducting a Mean Opinion Score (MOS) user study.
*   **Demo Video**: Rendering a side-by-side demo video showcasing temporal coherence.
*   **Multi-ControlNet Architecture**: Exploring a unified model fusing Canny, YOLO, DWPose, and IP-Adapter prompts.
