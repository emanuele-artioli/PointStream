# Object Segmentation and Background Inpainting

This module extends the PointStream video processing pipeline with YOLO-based object segmentation and background inpainting capabilities. It processes frames yielded by the video scene splitter to detect, track, and extract objects while creating clean background images.

## Features

- **YOLO Segmentation**: Uses state-of-the-art YOLO models for object detection and instance segmentation
- **Flexible Object Selection**: Configurable strategies for selecting the most important objects (by confidence, size, or position)
- **Background Inpainting**: Automatically removes objects from backgrounds using OpenCV inpainting algorithms
- **Real-time Processing**: Maintains the generator pattern for streaming scenarios
- **Separate Timing**: Tracks processing time separately from file I/O operations
- **Configurable Pipeline**: All parameters configurable through config.ini

## Architecture

```
Video Scene Splitter → Object Segmentation Inpainter → Next Pipeline Stage
      (frames)              (objects + backgrounds)         (processed data)
```

The pipeline follows these steps:
1. **Scene Detection**: Video is split into scenes with frame extraction
2. **Object Detection**: YOLO processes representative frames to detect objects
3. **Object Selection**: Selects top N objects based on configured strategy
4. **Mask Creation**: Creates individual and combined masks for detected objects
5. **Object Extraction**: Extracts object images with transparency
6. **Background Inpainting**: Removes objects from background using inpainting
7. **Data Yielding**: Yields processed data for downstream stages

## Quick Start

### Basic Usage (No File Saving)
```bash
# Process video with object segmentation
python object_segmentation_inpainter.py input_video.mp4

# Use with pipeline demo
python pipeline_demo.py input_video.mp4
```

### With File Saving
```bash
# Save all intermediate files
python pipeline_demo.py input_video.mp4 --output-dir ./output --enable-saving
```

### Custom Configuration
```bash
# Use different YOLO model and batch size
python pipeline_demo.py input_video.mp4 --yolo-model yolov8s-seg.pt --batch-size 200
```

## Configuration

All parameters are configured in `config.ini`. Key sections:

### [segmentation]
```ini
yolo_model = yolov8n-seg.pt    # YOLO model size
confidence_threshold = 0.25     # Detection confidence threshold
max_objects_per_scene = 3      # Max objects to track per scene
device = auto                  # Processing device (auto/cpu/cuda)
```

### [object_tracking]
```ini
strategy = confidence          # Selection strategy
min_object_area = 500         # Minimum object size
exclude_classes = []          # Classes to ignore
```

### [inpainting]
```ini
method = telea                # Inpainting algorithm
radius = 3                    # Inpainting radius
dilate_mask = true           # Expand masks before inpainting
```

## YOLO Models

Available YOLO segmentation models (speed vs accuracy tradeoff):

- **yolov8n-seg.pt**: Nano (fastest, least accurate)
- **yolov8s-seg.pt**: Small (balanced)
- **yolov8m-seg.pt**: Medium (good accuracy)
- **yolov8l-seg.pt**: Large (high accuracy)
- **yolov8x-seg.pt**: Extra Large (highest accuracy, slowest)

## Object Selection Strategies

### Confidence (default)
Selects objects with highest detection confidence scores.

### Size
Selects largest objects by bounding box area.

### Center
Selects objects closest to the image center.

## Output Data Structure

Each processed scene yields a dictionary containing:

```python
{
    'scene_number': int,                    # Scene identifier
    'objects': [                           # List of detected objects
        {
            'class_name': str,             # Object class (e.g., 'person', 'car')
            'confidence': float,           # Detection confidence (0-1)
            'bbox': [x1, y1, x2, y2],     # Bounding box coordinates
            'center': (x, y),              # Object center point
            'size': float,                 # Bounding box area
            'mask_area': int               # Mask pixel count
        }
    ],
    'object_images': [np.ndarray],         # Extracted object images (RGBA)
    'combined_mask': np.ndarray,           # Binary mask of all objects
    'inpainted_background': np.ndarray,    # Background with objects removed
    'processing_time': float,              # Processing time (seconds)
    'original_scene_data': dict           # Original scene data from splitter
}
```

## Performance Considerations

### Real-time Processing
The system is designed for real-time streaming:
- Processing time is measured separately from file I/O
- GPU acceleration available with CUDA
- Batch processing for efficient memory usage

### Model Selection
Choose YOLO model based on requirements:
- **Real-time**: yolov8n-seg.pt or yolov8s-seg.pt
- **Accuracy**: yolov8m-seg.pt or larger
- **GPU Memory**: Smaller models for limited VRAM

### Performance Monitoring
The system reports:
- Processing FPS (scenes/second)
- Real-time factor (processing speed vs video speed)
- Memory usage per scene
- Object detection accuracy

## Integration Examples

### Standalone Object Detection
```python
from object_segmentation_inpainter import ObjectSegmentationInpainter

processor = ObjectSegmentationInpainter(
    model_name='yolov8s-seg.pt',
    enable_saving=False
)

# Process scene data from video splitter
for scene_data in scene_generator:
    for processed_data in processor.process_scene_generator([scene_data]):
        objects = processed_data['objects']
        background = processed_data['inpainted_background']
        # Use objects and background for next processing stage
```

### Full Pipeline Integration
```python
from video_scene_splitter import VideoSceneSplitter
from object_segmentation_inpainter import ObjectSegmentationInpainter

# Initialize pipeline stages
splitter = VideoSceneSplitter('input.mp4', enable_encoding=False)
processor = ObjectSegmentationInpainter(enable_saving=False)

# Process video through pipeline
scene_generator = splitter.process_video_realtime_generator()
for processed_data in processor.process_scene_generator(scene_generator):
    # Forward to next pipeline stage
    next_stage.process(processed_data)
```

## File Output Structure

When `enable_saving=True`, files are organized as:

```
output_dir/
├── objects/
│   ├── scene_0001_obj_1_person_0.85.png
│   ├── scene_0001_obj_2_car_0.72.png
│   └── ...
├── backgrounds/
│   ├── scene_0001_background.png
│   ├── scene_0002_background.png
│   └── ...
├── masks/
│   ├── scene_0001_mask.png
│   ├── scene_0002_mask.png
│   └── ...
└── scene_XXXX_metadata.json
```

## Troubleshooting

### Common Issues

**YOLO model not found**
```bash
# Models are downloaded automatically on first use
# Ensure internet connection for initial download
```

**CUDA out of memory**
```bash
# Use smaller model or reduce batch size
python pipeline_demo.py input.mp4 --yolo-model yolov8n-seg.pt --batch-size 50
```

**Slow processing**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if GPU issues
# Set device = cpu in config.ini
```

### Performance Optimization

1. **Use appropriate YOLO model**: Start with yolov8n-seg.pt for speed
2. **Enable GPU**: Set `device = cuda` in config if available
3. **Adjust confidence threshold**: Lower values detect more objects but slower
4. **Reduce max objects**: Fewer objects = faster processing
5. **Optimize batch size**: Balance memory usage and processing speed

## Development

### Adding New Object Selection Strategies

```python
def _select_important_objects(self, results, strategy='custom'):
    # Add custom strategy logic here
    if strategy == 'custom':
        # Implement custom object ranking
        pass
```

### Custom Inpainting Methods

```python
def _inpaint_background(self, image, mask):
    if self.inpaint_method == 'custom':
        # Implement custom inpainting algorithm
        pass
```

### Pipeline Extensions

The yielded data structure makes it easy to add new processing stages:

```python
def next_pipeline_stage(processed_data):
    objects = processed_data['objects']
    background = processed_data['inpainted_background']
    
    # Add your processing logic here
    # e.g., object tracking, scene understanding, compression
    
    return enhanced_data
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Core dependencies:
- ultralytics (YOLO)
- torch/torchvision (deep learning)
- opencv-python (computer vision)
- numpy (numerical computing)

## License

Part of the PointStream project. See main project license for details.
