# ACM TOMM Submission: Review Synthesis & Action Plan

This document synthesizes the feedback from 5 reviewers for the ACM MM rejection, cross-referenced with internal research reports (the `reports/` folder) to identify which criticisms have already been addressed in the codebase.

## 1. Action Matrix

| Theme / Weakness | Reviewers | Current Status (Based on Reports) | Action Required | Effort |
| :--- | :--- | :--- | :--- | :--- |
| **Generative Model Choice**<br>cGAN is outdated; reviewers requested Diffusion models. | R1, R5 | 🟢 **Tackled**<br>Reports 1, 4 & 5 confirm extensive work integrating `Moore-AnimateAnyone` and fine-tuning `ControlNets`. SPADE was chosen for speed, but Diffusion was thoroughly evaluated. | Write a dedicated section comparing SPADE/GAN vs Diffusion/Animate Anyone, citing the temporal consistency and training time bottlenecks you discovered. | Low |
| **Temporal Coherence & Sync**<br>No proof of video continuity or background sync. | R3, R4 | 🟢 **Tackled**<br>Report 5 details optical flow warping, adaptive denoising, and cross-frame attention to solve "deep dream" loops. | Document these temporal consistency mechanisms (Optical Flow Warping, Cross-Frame Attention) explicitly in the paper. Generate a demo video. | Medium |
| **Dynamic Backgrounds & Cuts**<br>Handling camera cuts and dynamic spectators is unaddressed. | R2 | 🟢 **Tackled**<br>Report 2 details visual invariance filtering (HSV histograms) and point-anchored distribution to handle camera cuts vs. fast pans. | Integrate the Scene Classification heuristics into the methodology section to prove robustness to cuts. | Low |
| **Missing Baselines**<br>Omission of AV1/VVC and recent learned codecs (DVC, NeRV, SVC-LC). | R2, R5 | 🔴 **Not Tackled** | Run PointStream vs. AV1, VVC, and ideally 1-2 learned codecs. Add new tables to the Results section. | High |
| **Lack of Generalizability**<br>Tennis is too narrow. Need other sports/scenes. | R2, R3, R4, R5 | 🔴 **Not Tackled** | Prepare a small dataset for a second domain (e.g., basketball, urban pedestrian) and demonstrate the pipeline. | High |
| **Subjective Evaluation**<br>VMAF/LPIPS insufficient; MOS requested. | R2, R4 | 🔴 **Not Tackled** | Conduct a Mean Opinion Score (MOS) user study to validate perceptual quality. | Medium |
| **Temporal Segmentation**<br>Per-frame YOLO ignores temporal context (suggested SAM2). | R3 | 🟡 **Partially Tackled**<br>Report 2 shows improvements via Convex Hull and wrist anchoring, but no SAM2. | Address SAM2 vs YOLO computationally in the paper (or implement a quick SAM2 test). | Medium |
| **Shadow Handling**<br>No discussion on how shadows cast by moving humans are processed. | R3 | 🔴 **Not Tackled** | Add discussion/implementation on handling dynamic shadows on the client side. | Low |

---

## 2. Execution Checklist

### Phase 1: Experiments & Code
- [ ] **Codec Benchmarks:** Generate metrics for AV1 and VVC baselines on the existing tennis dataset.
- [ ] **Learned Codec Baselines:** Benchmark against at least one recent semantic codec (e.g., DVC or NeRV).
- [ ] **Domain Expansion:** Process a short dataset from a second domain (e.g., basketball or a generic pedestrian video) through the pipeline to prove generalizability.
- [ ] **Subjective Study (MOS):** Set up a small-scale survey comparing HEVC/AV1 outputs against PointStream outputs and collect MOS scores.
- [ ] **Demo Video:** Render a side-by-side demo video showcasing PointStream's temporal coherence and compression visual quality (requested by R3).
- [ ] **(Optional) SAM2 Test:** Evaluate if replacing YOLO with SAM2 is feasible or if it breaks the real-time constraints.

### Phase 2: Paper Writing & Revision
- [ ] **Generative Model Justification:** Draft a section detailing the exploration of Animate Anyone and ControlNet (from Report 4 & 5). Explain *why* you chose the final architecture (SPADE/GAN) based on training budgets and temporal stability.
- [ ] **Temporal Coherence Section:** Explicitly detail the temporal warping, cross-frame attention, and reference masking strategies used to maintain consistency.
- [ ] **Background & Cut Handling:** Add the scene classification and HSV histogram logic to prove the system handles sudden camera cuts gracefully.
- [ ] **Shadow Handling:** Add a paragraph addressing the shadow limitation (either acknowledging it as future work or explaining how the current bounding box captures it).
- [ ] **Update Results:** Add the new AV1/VVC and MOS tables.
- [ ] **Format Update:** Ensure the paper format strictly adheres to the ACM TOMM guidelines (Reviewer 3 noted formatting issues).
