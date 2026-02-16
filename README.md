# PointStream

Efficient tennis video streaming via keypoint-driven animation. Instead of transmitting full video frames, the server segments players, extracts whole-body pose keypoints (DWPose), and sends only the keypoints plus a single reference frame per player. The client reconstructs the player movement using AnimateAnyone.

## Architecture

```
Server (pointstream env)                        Client (animate-anyone env)
┌──────────────────────┐   keypoints CSV   ┌──────────────────────────┐
│ SAM3 segmentation    │──────────────────→│ Skeleton reconstruction  │
│ DWPose extraction    │   + ref images    │ AnimateAnyone inference  │
└──────────────────────┘                   └──────────────────────────┘
```

## Pipeline Steps

| Step | Script | Environment | Description |
|------|--------|-------------|-------------|
| Server | `pointstream_server.py` | pointstream | SAM3 segmentation + DWPose keypoint extraction |
| Client | `pointstream_client.py` | animate-anyone | Skeleton drawing + AnimateAnyone pose-to-video |
| A1 | `A1_segment_with_sam.py` | pointstream | SAM3 player segmentation (called by server) |
| A2 | `A2_extract_poses_from_crops.py` | pointstream | Legacy YOLO pose extraction (deprecated) |
| B1 | `B1_video_panorama.py` | pointstream | Video panorama stitching |

`pointstream.py` orchestrates both server and client from a single entry point.

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- `ffmpeg` / `ffprobe` on PATH
- [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) installed at `../Moore-AnimateAnyone`

### Server environment (pointstream)

```bash
conda env create -f environment.yaml
conda activate pointstream
```

The server needs the DWPose ONNX models from AnimateAnyone:
- `../Moore-AnimateAnyone/pretrained_weights/DWPose/yolox_l.onnx`
- `../Moore-AnimateAnyone/pretrained_weights/DWPose/dw-ll_ucoco_384.onnx`

### Client environment (animate-anyone)

The client runs in the existing animate-anyone conda environment. It must not be modified — AnimateAnyone's dependencies (torch 2.0.1, xformers 0.0.22) are version-locked.

```bash
conda activate animate-anyone
```

### Download models

| Model | Default path | Source |
|-------|-------------|--------|
| SAM3 | `../models/sam3.pt` | [Ultralytics SAM3](https://docs.ultralytics.com/models/sam/) |
| DWPose detection | `../Moore-AnimateAnyone/pretrained_weights/DWPose/yolox_l.onnx` | Bundled with AnimateAnyone |
| DWPose pose | `../Moore-AnimateAnyone/pretrained_weights/DWPose/dw-ll_ucoco_384.onnx` | Bundled with AnimateAnyone |

## Usage

### End-to-end streaming pipeline

```bash
# Step 1: Server — segment players and extract DWPose keypoints
conda activate pointstream
python pointstream_server.py --video_path /path/to/tennis_match.mp4

# Step 2: Client — reconstruct and animate
conda activate animate-anyone
cd /home/itec/emanuele/Moore-AnimateAnyone
python /path/to/pointstream/pointstream_client.py \
    --experiment_dir /path/to/experiments/YYYYMMDD_HHMMSS_sam_seg
```

### Via the unified orchestrator

```bash
# Server mode
python pointstream.py --mode server --video_path /path/to/video.mp4

# Client mode
python pointstream.py --mode client --experiment_dir /path/to/experiment

# Client: reconstruct skeletons only (no inference)
python pointstream.py --mode client --experiment_dir /path/to/experiment --skeletons_only
```

### Skip SAM3 (run DWPose on existing crops)

```bash
python pointstream_server.py --experiment_dir experiments/20260209_100146_sam_seg
```

### Process an entire dataset

```bash
python process_tennis_datasets.py --folder /path/to/dataset --skip-existing
```

### Evaluate experiments (config + timings + PSNR)

```bash
conda activate pointstream
python evaluate_experiments.py \
    --experiments_root /home/itec/emanuele/pointstream/experiments
```

Optional arguments:
- `--output_csv /path/to/evaluation_summary.csv`
- `--output_json /path/to/evaluation_summary.json`
- `--max_frames N` (limit PSNR frames per player for faster evaluation)

The evaluator collects, per experiment:
- Config parameters from `evaluation_server.json` and `evaluation_client.json`
- Sub-task timing (`sam_segmentation_sec`, `dwpose_extraction_sec`,
    `skeleton_reconstruction_sec`, `inference_sec`, totals)
- PSNR quality by player and weighted experiment mean (`output_player_*.mp4`
    vs `masked_crops/id*/`)

`evaluation_server.json` and `evaluation_client.json` are now written by
`pointstream_server.py` and `pointstream_client.py` at the end of each run.

## Output Structure

```
experiments/YYYYMMDD_HHMMSS_sam_seg/
├── tracking_metadata.csv      # SAM3 bounding boxes per frame (includes `video_fps` for new experiments)
├── dwpose_keypoints.csv       # 134 DWPose keypoints per frame per player
├── merged_metadata*.csv       # (new) merged tracking + pose metadata; filename includes `w{detect_w}_h{detect_h}_fps_{fps}` when available (detect size & fps not stored as columns)
├── masked_crops/
│   ├── id0/                   # Player 1 crops (512×512, masked + padded)
│   └── id1/                   # Player 2 crops
├── reference/
│   ├── id0.png                # First frame of player 1 (for client)
│   └── id1.png                # First frame of player 2
└── debug_skeletons/           # (created by client for visualisation)
    ├── id0/
    └── id1/
```

### Keypoints CSV format

| Column | Type | Description |
|--------|------|-------------|
| frame_index | int | Frame number |
| player_id | int | Player tracking ID (0 or 1) |
| keypoints | JSON | 134 [x,y] pairs in pixel coordinates (body + hands + face) |
| scores | JSON | 134 confidence scores |
| detect_width | int | Crop width used for detection |
| detect_height | int | Crop height used for detection |
