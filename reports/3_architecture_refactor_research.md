# PointStream Architecture Unification: Refactoring for Scalable Generation

This document summarizes our iterative refactoring of the PointStream codebase. It covers the evolution from fragmented, specialized scripts (for dataset generation vs. runtime inference) into a unified, shared-library architecture. These notes are formatted for inclusion in the PointStream project paper.

## Phase 1: The Fragmentation Problem
Historically, our dataset processing pipeline (`process_dataset.py`) and our GenAI backend (`main.py`, `train_spade4tennis.py`, `train_pix2pix.py`) evolved independently.
- **Duplicate Logic:** Mathematical functions for intersection-over-union (IoU), bounding box area calculations, and DWPose translation were duplicated across encoder models and dataset scripts.
- **Divergent Rendering:** The runtime `genai_compositor.py` and the dataset builder each maintained their own copy of `_render_pose_with_racket`, making heuristic updates (like wrist anchoring) a nightmare to keep synchronized.
- **Redundant Dataset Workflows:** We had standalone scripts like `patch_dataset.py` and `build_pix2pix_dataset.py` acting as band-aids to filter or merge dataset components after extraction, creating brittle, multi-step workflows.

## Phase 2: Building the Unified `src/shared/` Library
To solve this, we instituted a strict "Shared Library" pattern. We purged all inline geometric, rendering, and architectural logic from the individual execution scripts and centralized them into `src/shared/`:
- **`geometry.py`**: Central source for `get_iou`, bounding box area operations, and global motion extraction.
- **`player_extraction.py`**: Houses the core YOLO tracking logic and standardizes the DWPose conversion (`coco17_to_dwpose18`), ensuring both dataset builder and runtime encoder interpret skeletons identically.
- **`racket_heuristic.py`**: Unifies all racket handling, containing `render_pose_with_racket`, `get_dominant_wrist`, and `interpolate_racket_track`. Any fix to the racket drawing now instantly applies to both training dataset rendering and live generative inference.
- **`spade4tennis_arch.py`**: Standardizes the SPADEResNet9Generator and discriminator components. The generative inference pipeline (`spade4tennis_engine.py`) now imports the exact same PyTorch classes as the training script (`train_spade4tennis.py`), guaranteeing perfect weight compatibility.
- **`tennis_dataset.py`**: Unified PyTorch Dataset class (`TennisSkeletonDataset`) dynamically handling both Pix2Pix (color + skeleton) and SPADE (color + skeleton + reference) paradigms, rendering old dataset classes obsolete.

## Phase 3: Dataset Pipeline Streamlining
Instead of generating temporary outputs and relying on a manual `to_delete.txt` cleanup step or subsequent scripts, we heavily modified `process_dataset.py`:
- **Dynamic Worker Scaling:** Replaced hardcoded strings with `torch.cuda.device_count()`, dynamically spawning multiprocessing pools based on available hardware.
- **Integrated Training Pairs Stage:** We introduced a native `--build-training-pairs` stage. The script now internally validates completed tracks, applies the unified tennis dataset formatting, and generates the final paired `{skeleton, colour}` crops in a single pass.

> [!TIP]
> **Why this works:** By folding the dataset pair construction directly into the core extraction pipeline, we eliminate intermediate I/O bottlenecks and ensure that the exact same validation logic is applied universally. There is no longer a risk of the dataset format drifting out of sync with the model's expectations.

## Phase 4: Runtime Environment Hardening
During the refactor, we encountered and solved runtime dependency paths:
- **Monkey-Patch Isolation:** Previous iterations dirtily patched core libraries (`ultralytics` internals, `torch.from_numpy`) directly inside the execution code. We extracted these into a dedicated `_compat_patches.py` module, imported cleanly at the very top of `process_dataset.py`.
- **Strict Import Structure:** We aggressively enforced strict `from src.shared...` absolute imports, allowing the codebase to be executed smoothly from the repository root, completely eliminating `ModuleNotFoundError` crashes during GPU multiprocessing.

The culmination of these architectural shifts is a rock-solid PointStream repository that gracefully handles distributed dataset generation and complex GenAI inference using a single, unified source of truth.
