# PointStream - Streamlined Pipeline

A simplified, single-file implementation of the PointStream video processing pipeline.

## Overview

The PointStream pipeline processes videos through the following steps:

1. **Splitter**: Splits video into scenes using scene change detection
2. **Segmenter**: Uses YOLO to detect and segment objects in each scene
3. **Stitcher**: Creates panoramas from scene frames with object inpainting
4. **Classifier**: Classifies objects as human, animal, or other using semantic similarity
5. **Keypointer**: Extracts keypoints using MMPose (human/animal) or Canny edge detection (other)

## Files

- `pointstream.py` - Main streamlined pipeline with all components
- `test_pointstream.py` - Test script to verify pipeline functionality
- `requirements_streamlined.txt` - Minimal dependencies for the streamlined version

## Installation

```bash
# Install dependencies
pip install -r requirements_streamlined.txt

# Download YOLO segmentation model (will happen automatically on first run)
# Model will be downloaded to ~/.ultralytics/
```

## Usage

### Command Line

```bash
# Basic usage
python pointstream.py input_video.mp4

# With output directory to save results
python pointstream.py input_video.mp4 --output-dir results/

# With custom thresholds
python pointstream.py input_video.mp4 --scene-threshold 40.0 --confidence 0.3
```

### Python API

```python
from pointstream import process_video_pipeline

# Process a video
results = process_video_pipeline('input_video.mp4', output_dir='results/')

print(f"Processed {results['total_scenes']} scenes")
print(f"Found {results['total_objects']} objects")
print(f"Extracted keypoints for {results['total_objects_with_keypoints']} objects")
```

### Individual Components

```python
from pointstream import (
    split_video_into_scenes,
    segment_objects_in_scene,
    stitch_scene_panorama,
    classify_objects,
    extract_keypoints
)

# Split video into scenes
scenes = split_video_into_scenes('video.mp4')

# Process each scene
for scene in scenes:
    # Segment objects
    segmentation = segment_objects_in_scene(scene)
    
    # Create panorama  
    panorama = stitch_scene_panorama(scene, segmentation)
    
    # Classify objects
    classification = classify_objects(segmentation)
    
    # Extract keypoints
    keypoints = extract_keypoints(classification)
```

## Testing

```bash
# Run tests (requires room.mp4 in PointStream directory)
python test_pointstream.py
```

## Output Structure

When using an output directory, the pipeline creates:

```
output_dir/
├── scene_0001/
│   ├── panorama.jpg           # Scene panorama
│   ├── metadata.json          # Scene metadata and object info
│   └── objects/               # Cropped object images
│       ├── scene_1_frame_0_obj_0.jpg
│       └── ...
├── scene_0002/
└── ...
```

## Key Features

- **Minimal Dependencies**: Only essential libraries required
- **Single File**: All components in one file for easy deployment
- **Configurable**: Adjustable thresholds and parameters
- **Complete Pipeline**: All original functionality preserved
- **Clean API**: Simple function interfaces for each component
- **Error Handling**: Graceful fallbacks when components fail

## Component Details

### Splitter
- Uses PySceneDetect ContentDetector
- Configurable scene change threshold
- Returns scenes with extracted frames

### Segmenter  
- Uses YOLOv8 segmentation model
- Object tracking across frames
- Returns objects with bounding boxes, masks, and cropped images

### Stitcher
- OpenCV-based panorama creation
- Object inpainting (black masking)
- Fallback to first frame if stitching fails

### Classifier
- Sentence transformer embeddings for semantic similarity
- Human/animal/other classification
- Configurable similarity thresholds

### Keypointer
- MMPose for human and animal keypoints
- Canny edge detection for other objects
- Normalized keypoint coordinates [0,1]

## Performance

The streamlined pipeline maintains the core functionality while being much simpler to deploy and understand. Processing time depends on:

- Video length and resolution
- Number of objects detected
- Hardware capabilities (GPU recommended for YOLO and MMPose)

## Limitations

- Simplified stitching (no advanced homography computation)
- Basic inpainting (black masking instead of diffusion)
- No duplicate filtering or advanced tracking features
- Limited configuration options compared to full server

This streamlined version focuses on the essential pipeline functionality while maintaining clean, readable code.