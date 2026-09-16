# PointStream: 10x Egocentric Video Compression Demo for Figure.ai

This demo showcases **10x compression on real-world egocentric manufacturing video** ([builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K)) designed to demonstrate value for robotic teleoperation and fleet data logging to Figure.ai ([sam.baker@figure.ai](mailto:sam.baker@figure.ai), [max.berman@figure.ai](mailto:max.berman@figure.ai)).

All code and experiment outputs are self-contained within this `demo/` directory, adhering strictly to PointStream project isolation rules.

---

## Architecture Overview

PointStream decouples complex egocentric videos into two independent, low-bitrate representations:
1. **Hand Keypoint Stream ($\approx 7.1–10.0\text{ kbps}$)**: Extracted using MediaPipe Hands (21 3D landmarks per hand), quantized to 47 bytes per hand at 30 fps.
2. **Amortized Background Stream ($\approx 231–262\text{ kbps}$)**: Clean downscaled background stream encoded with SVT-AV1 preset 7.
3. **Client-Side Neural Synthesis (<3 ms)**: A lightweight conditional UNet generator restores sharp hand anatomy and object contacts from the keypoints and initial appearance anchor.

### Why PointStream Wins Over AV1 at 14x–18x Compression
- **No Finger Smearing**: At 14x–18x compression (<300 kbps for 1080p), conventional block-based codecs (AV1 / H.265) suffer severe motion blur and contour degradation across moving hands. PointStream transmits explicit joint telemetry, preserving articulated finger poses.
- **Up to 1.8x Lower Joint Tracking Error**: Downstream tracking achieves **58.8–69.6 px MPJPE** on PointStream vs **106–116 px on AV1 540p**, and beats AV1 on detection rate on complex assembly (52.4% vs 48.8%, capturing **90.4%** of the uncompressed reference ceiling).
- **Real-Time Teleop Feasible**: Total pipeline latency is **19.3 ms (parallel)** and **32.9 ms (serial)** on NVIDIA RTX 6000 Ada, comfortably inside the 50 ms teleoperation budget. Direct telemetry provides 8.4 kbps joint coordinates to robot control policies without re-running vision models on degraded frames.

---

## Directory Structure

```
demo/
├── README.md                      # This file
├── configs/
│   └── egocentric_demo.yaml       # Central configuration parameters
├── data/
│   ├── download_sample.py         # Downloads Egocentric-10K tar shard to /home/itec/emanuele/Datasets
│   └── extract_and_curate.py      # Extracts clips and selects longest continuous sequences from worker 001
├── pipeline/
│   ├── hand_keypoints.py          # MediaPipe 21-joint pose extraction, wire format & skeleton rendering
│   ├── foreground_segmenter.py    # Hand bounding box extraction, letterboxing, and background infill
│   ├── background_codec.py        # Background downscale/infill and SVT-AV1 encode/decode
│   └── keypoint_compressor.py     # Bitpacking 21 landmarks into 47 bytes per hand
├── models/
│   ├── unet_generator.py          # Fast conditional UNet (derived from PointStream Pix2Pix)
│   ├── dataset.py                 # Hand crop pairing and appearance anchor caching
│   └── train_overfit.py           # Overfits the generator on the curated worker clips
├── evaluation/
│   ├── encode_av1_ladder.py       # SVT-AV1 rate ladder encoding (including deblocked ablation)
│   ├── evaluate_quality.py        # PSNR, SSIM, LPIPS, DISTS, and Hand-ROI metrics
│   ├── evaluate_robotics_teleop.py# Downstream task utility: Hand Detection Rate & MPJPE error
│   └── latency_profiler.py        # Per-frame encode and decode latency breakdown
├── experiments/
│   ├── run_comparison.py          # End-to-end benchmark comparing PointStream vs AV1
│   └── plot_rd_curves.py          # Generates Pareto Rate-Distortion and Teleop utility curves
└── pitch/
    ├── figure_ai_pitch_draft.md   # Email pitch draft
    ├── side_by_side_video.py      # Stitches 3-panel video (Reference vs AV1 vs PointStream)
    ├── interactive_demo.html      # Inspector page (source; do not copy by hand)
    ├── interactive_report.html    # Static report page (source)
    └── publish_site.py            # Builds the public folder from JSON + HTML + media
```

### Where numbers and the website live

There are **not** two competing result dumps. They have different jobs:

| Path | What it is |
| --- | --- |
| `demo/outputs/results/comparison_results.json` | **Source of truth for numbers.** The rest of `results/` is local scratch (decoded clips, etc.) and is gitignored. |
| `demo/pitch/*.html` | **Source of truth for the demo pages.** Edit here. |
| `demo/outputs/pitch/*.mp4` and `keypoints_*.json` | **Source of truth for media** (too heavy to rebuild on GitHub Actions). |
| `demo/outputs/pitch/*.html` and `*.png` | **Generated** by `publish_site.py`. Do not edit. Gitignored. |

```bash
PYTHONPATH=. python demo/pitch/publish_site.py
```

A push to `main` that touches any of those inputs deploys to [emanueleartioli.com/pointstream](https://emanueleartioli.com/pointstream/).

---

## Step-by-Step Execution

### 1. Data Download & Curation
```bash
# Download sample shards from Hugging Face to /home/itec/emanuele/Datasets/Egocentric-10K/
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/data/download_sample.py
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/data/download_sample.py --filename factory_001/workers/worker_001/factory001_worker001_part01.tar

# Extract and curate the 3 longest clips from worker 001
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/data/extract_and_curate.py --top-k 3
```

### 2. Model Overfitting
```bash
# Rapidly overfit the conditional UNet on the curated clips (~1-2 minutes on GPU)
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/models/train_overfit.py --frames 300 --epochs 30 --device cuda:1
```

### 3. Run Benchmark Comparison
```bash
# Run PointStream vs AV1 comparison ladder across all 3 clips
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/experiments/run_comparison.py --frames 300 --device cuda:1
```

### 4. Generate Plots & Side-by-Side Video
```bash
# Plot Rate-Distortion, Teleoperation MPJPE, and Latency curves
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/experiments/plot_rd_curves.py

# Create 3-panel split video with real-time HUD and zoomed hand inset
PYTHONPATH=. /home/itec/emanuele/.conda/envs/pointstream/bin/python demo/pitch/side_by_side_video.py
```

### 5. Review Deliverables for Figure.ai
- **Benchmark JSON (numbers)**: `demo/outputs/results/comparison_results.json`
- **Demo pages (markup)**: `demo/pitch/interactive_demo.html`, `demo/pitch/interactive_report.html`
- **Stacked videos (media)**: `demo/outputs/pitch/side_by_side_demo_*.mp4`
- **Live site**: [emanueleartioli.com/pointstream](https://emanueleartioli.com/pointstream/) — assembled by `demo/pitch/publish_site.py` on each `main` push
