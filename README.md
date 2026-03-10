# PointStream

Tennis video-to-animation pipeline. Segments players from match footage, extracts pose skeletons, and prepares data for few-shot video-to-video synthesis training.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| A1 | `A1_segment_with_sam.py` | Segment players using SAM3 (text or bbox prompt) |
| A2 | `A2_extract_poses_from_crops.py` | Extract YOLO pose skeletons from crops |
| A3–A6 | (via `pointstream.py`) | Dataset preparation, LMDB build, training, inference |
| B1 | `B1_video_panorama.py` | Video panorama stitching and reconstruction |

`process_tennis_datasets.py` orchestrates A1 + crop encoding + DWpose extraction across an entire dataset folder.

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- `ffmpeg` / `ffprobe` on PATH (installed via conda below)

### Create the environment

```bash
conda env create -f environment.yaml
conda activate pointstream
```

This installs Python 3.10, FFmpeg, PyTorch with CUDA, and all pip dependencies in one step.

### Download models

The pipeline requires pretrained model weights. Place them in a `models/` directory (or adjust the default paths in the scripts):

| Model | Default path | Source |
|-------|-------------|--------|
| SAM3 | `../models/sam3.pt` | [Ultralytics SAM3](https://docs.ultralytics.com/models/sam/) |
| YOLO Pose | `../models/yolo11l-pose.pt` | [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/) |
| YOLOv8n (fallback) | `../yolov8n.pt` | [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) |

### Export ReID model to TensorRT

Use `utils.export_reid_model_to_tensorrt()` to export a YOLO classification model for BoT-SORT ReID. The function writes the `.engine` file in the same folder as the input model.

```bash
conda activate pointstream
python -c "from utils import export_reid_model_to_tensorrt; print(export_reid_model_to_tensorrt('/home/itec/emanuele/Models/YOLO/yolo26n-cls.pt', half=True, dynamic=True, batch=32, imgsz=640, overwrite=True))"
```

The tracker config at `/home/itec/emanuele/Models/YOLO/trackers/botsort.yaml` should point `model:` to the exported engine path (for example `/home/itec/emanuele/Models/YOLO/yolo26n-cls.engine`).

## Usage

### Full extraction pipeline (A1 + A2)

```bash
python pointstream.py --mode extract --dataset_dir /path/to/videos
```

### Process an entire tennis dataset

```bash
python process_tennis_datasets.py --folder /path/to/dataset --skip-existing
```

### Single-step examples

```bash
# Segment a single video
python A1_segment_with_sam.py --video_path video.mp4 --model_path /path/to/sam3.pt

# Extract poses from an experiment's crops
python A2_extract_poses_from_crops.py --experiment_dir experiments/20260209_100146_sam_seg

# Create a video panorama
python B1_video_panorama.py --video_path video.mp4
```

`utils.stitch_panorama()` uses a minimal ORB-based pairwise homography pipeline with identity fallback when feature matching is insufficient. All input frames should have the same resolution.

It now returns `(panorama_image, panorama_data)` instead of only the panorama image. `panorama_data` contains frame size, canvas size, translation matrix, and homographies needed to reverse the process. Use `utils.animate_panorama(panorama_image, panorama_data)` to reconstruct the background-frame sequence from the stitched panorama.

### Per-object segmentation from detections

`pointstream.py` also supports detection-first object masking. It saves bbox crops in `bbox_crops/`, runs segmentation on each crop (including single-frame list inputs), saves masks in `segmentation_masks/`, saves masked crops in `segmented_crops/`, and writes the background frame only after masking all objects in that frame.

After panorama creation, `pointstream.py` loops over each `person_*` track folder and runs a three-step DWPose flow:
1. `utils.extract_dwpose_keypoints(...)` returns one ndarray per frame with columns `[x, y]` (low-confidence keypoints are set to `-1`).
2. `utils.save_dwpose_keypoints_csv(...)` saves those keypoints to `dwpose_keypoints/*.csv`, then `utils.load_dwpose_keypoints_csv(...)` reads them back.
3. `utils.convert_dwpose_keypoints_to_skeleton_frames(..., frame_size=(H, W))` renders skeleton PNGs into `dwpose_frames/`.

The pipeline then encodes one AV1 (`libsvtav1`) video per person into `dwpose_videos/` using `utils.encode_video_libsvtav1(..., crf=25)`.

### Other pipeline modes

```bash
# Prepare dataset for training (A3 + A4)
python pointstream.py --mode prepare

# Train fs_vid2vid model (A5)
python pointstream.py --mode train

# Run inference (A6)
python pointstream.py --mode inference --checkpoint /path/to/checkpoint.pt
```

## Output Structure

Each extraction run creates a timestamped experiment folder under `experiments/`:

```
experiments/
└── 20260209_100146_sam_seg/
    ├── tracking_metadata.csv    # Bounding boxes and transform info per frame
    ├── pose_metadata.csv        # Keypoint coordinates per frame
    ├── dwpose_frames/
    │   ├── person_1/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── person_2/
    ├── dwpose_keypoints/
    │   ├── person_1_keypoints.csv
    │   └── person_2_keypoints.csv
    ├── dwpose_videos/
    │   ├── person_1_dwpose.mp4  # DWPose skeleton video for one tracked person
    │   └── person_2_dwpose.mp4
    ├── masked_crops/
    │   ├── id0/                 # Player 1 crops (512×512, masked + padded)
    │   └── id1/                 # Player 2 crops
    └── skeletons/
        ├── id0/                 # Player 1 skeleton images
        └── id1/                 # Player 2 skeleton images
```
