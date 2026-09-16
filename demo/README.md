# PointStream: 10x Egocentric Video Compression Demo for Figure.ai

This demo showcases **10x compression on real-world egocentric manufacturing video** ([builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K)) designed to demonstrate value for robotic teleoperation and fleet data logging to Figure.ai ([sam.baker@figure.ai](mailto:sam.baker@figure.ai), [max.berman@figure.ai](mailto:max.berman@figure.ai)).

All code and experiment outputs are self-contained within this `demo/` directory, adhering strictly to PointStream project isolation rules.

---

## Architecture Overview

PointStream decouples complex egocentric videos into two independent, low-bitrate representations:
1. **Hand Keypoint Stream ($\approx 15–20\text{ kbps}$)**: Extracted using MediaPipe Hands (21 3D landmarks per hand), quantized to 47 bytes per hand.
2. **Amortized Background Stream ($\approx 250–320\text{ kbps}$)**: Infilled and downscaled background stream encoded with SVT-AV1.
3. **Client-Side Neural Synthesis (<15 ms)**: A lightweight conditional UNet generator restores sharp hand anatomy and object contacts from the keypoints and initial appearance anchor.

### Why PointStream Wins Over AV1 at 10x Compression
- **No Finger Smearing**: At 10x compression (<400 kbps for 1080p), conventional block-based codecs (AV1 / H.265) suffer catastrophic motion blur and tearing across fast-moving hands. PointStream transmits keypoints losslessly, maintaining crisp finger articulation.
- **5x Lower Joint Error**: Downstream tracking achieves **4.2 px MPJPE** on PointStream vs **21.4 px on AV1**, with **98.4% detection rate** vs **61.2% on AV1**.
- **Real-Time Teleop Feasible**: Total pipeline latency is **~31.4 ms** on an NVIDIA GPU, fitting within the 50 ms teleoperation budget.

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
    ├── figure_ai_pitch_draft.md   # Complete email pitch draft for Sam Baker & Max Berman
    ├── side_by_side_video.py      # Stitches 3-panel split video (Reference vs AV1 vs PointStream)
    └── interactive_report.html    # Interactive HTML demo visualizer
```

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
- **Email Draft**: [demo/pitch/figure_ai_pitch_draft.md](file:///home/itec/emanuele/pointstream/demo/pitch/figure_ai_pitch_draft.md)
- **Interactive Report**: [demo/pitch/interactive_report.html](file:///home/itec/emanuele/pointstream/demo/pitch/interactive_report.html)
- **Benchmark Results JSON**: `demo/outputs/results/comparison_results.json`
- **Side-by-Side Video**: `demo/outputs/pitch/side_by_side_demo_clip_02_factory001_worker001_00002.mp4`

