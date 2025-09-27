# PointStream - Server-Client Video Reconstruction Pipeline

PointStream is a advanced video processing system that separates video analysis (server) from video reconstruction (client) using generative models. The server extracts metadata, panoramas, and keypoints from videos, while the client reconstructs videos from this metadata using trained generative models at much lower bitrates.

## Architecture Overview

```
📁 PointStream/
├── 🔥 server/           # Server-side processing (video → metadata)
│   ├── server.py        # Main server script
│   ├── config.ini       # Server configuration
│   ├── environment.yml  # Server environment
│   └── scripts/         # Server processing components
│       ├── segmenter.py     # Object segmentation
│       ├── stitcher.py      # Panorama stitching
│       ├── keypointer.py    # Keypoint extraction
│       ├── semantic_classifier.py  # Object classification
│       ├── duplicate_filter.py     # Duplicate filtering
│       ├── saver.py         # Data saving
│       └── splitter.py      # Video scene splitting
│
├── 🔧 client/           # Client-side processing (metadata → video)
│   ├── client.py        # Main client script
│   ├── config.ini       # Client configuration
│   ├── environment.yml  # Client environment
│   ├── requirements.txt # Additional dependencies
│   ├── models/          # Generative model definitions
│   │   ├── human_cgan.py     # Human generation model
│   │   ├── animal_cgan.py    # Animal generation model
│   │   └── other_cgan.py     # Other objects model
│   └── scripts/         # Client processing components
│       ├── background_reconstructor.py  # Background reconstruction
│       ├── object_generator.py          # Object generation with cGAN
│       ├── frame_composer.py            # Frame composition
│       ├── video_assembler.py           # Video assembly
│       ├── quality_assessor.py          # Quality assessment
│       └── model_trainer.py             # Generative model training
│
├── 🛠️ utils/            # Shared utilities
│   ├── config.py        # Configuration management
│   ├── decorators.py    # Performance tracking
│   └── error_handling.py # Error handling
│
├── 📊 models/           # Trained model storage
├── 🎬 main.py           # Unified pipeline entry point
└── 📋 README.md         # This file
```

## Workflow

### Phase 1: Server Processing (Video → Metadata)
1. **Scene Detection**: Split video into semantic scenes
2. **Object Segmentation**: Detect and segment objects (humans, animals, others)
3. **Panorama Stitching**: Create background panoramas with object masks
4. **Keypoint Extraction**: Extract pose keypoints for humans/animals, edge features for objects
5. **Metadata Export**: Save panoramas, homographies, keypoints, and object data

### Phase 2: Client Processing (Metadata → Video)
1. **Model Training**: Train category-specific cGAN models (human, animal, other)
2. **Background Reconstruction**: Recreate frame backgrounds from panoramas + homographies
3. **Object Generation**: Generate objects using keypoints + reference images via cGAN
4. **Frame Composition**: Overlay generated objects on reconstructed backgrounds
5. **Video Assembly**: Assemble frames into final video

### Phase 3: Quality Assessment
- Compare reconstructed videos with originals using SSIM, PSNR, LPIPS, and optionally VMAF

## Installation

### Server Environment
```bash
# Create and activate server environment
conda env create -f server/environment.yml
conda activate pointstream

# Install additional server dependencies
pip install -r server/requirements.txt  # If exists
```

### Client Environment
```bash
# Create and activate client environment
conda env create -f client/environment.yml
conda activate pointstream-client

# Install additional client dependencies
pip install -r client/requirements.txt
```

### Shared Installation
```bash
# If you want to use a single environment (not recommended for production)
conda env create -f server/environment.yml
conda activate pointstream
pip install -r client/requirements.txt
```

## Usage

### Full Pipeline (Recommended)
```bash
# Run complete pipeline: server → client → quality assessment
python main.py input_video.mp4 --full-pipeline

# With custom configurations
python main.py input_video.mp4 --full-pipeline \
    --server-config server/config.ini \
    --client-config client/config.ini
```

### Server Only
```bash
# Process video to extract metadata
python main.py input_video.mp4 --server-only --metadata-dir ./metadata

# Or run server directly
cd server
python server.py ../input_video.mp4 --output-dir ../metadata
```

### Client Only
```bash
# Reconstruct video from existing metadata
python main.py --client-only \
    --metadata-dir ./metadata \
    --output-dir ./reconstructed

# Skip model training (use existing models)
python main.py --client-only \
    --metadata-dir ./metadata \
    --output-dir ./reconstructed \
    --no-training

# Or run client directly
cd client
python client.py ../metadata --output-dir ../reconstructed
```

### Quality Assessment Only
```bash
# Assess quality against reconstructed videos
python main.py input_video.mp4 --assess-quality ./reconstructed
```

## Configuration

### Server Configuration (`server/config.ini`)
- Scene detection parameters
- Object segmentation settings (YOLO model, confidence thresholds)
- Panorama stitching options
- Keypoint extraction settings
- Output formatting

### Client Configuration (`client/config.ini`)
- Generative model settings (cGAN parameters)
- Training configuration (learning rates, epochs, batch sizes)
- Video reconstruction quality
- Background reconstruction parameters
- Quality assessment metrics

## Key Features

### Server-Side
- **Multi-Algorithm Scene Detection**: Content, adaptive, histogram, and hash-based
- **Advanced Object Segmentation**: YOLOv11 with semantic classification
- **Intelligent Panorama Stitching**: With object masking and background reconstruction
- **Multi-Modal Keypoint Extraction**: Human/animal poses + edge/corner features
- **Performance Optimization**: Parallel processing, caching, and GPU acceleration

### Client-Side
- **Category-Specific Generative Models**: Separate cGAN models for humans, animals, and objects
- **Temporal Consistency**: Frame smoothing and motion compensation
- **Quality Assessment**: Multiple metrics (SSIM, PSNR, LPIPS, VMAF)
- **Advanced Training**: Perceptual loss, spectral normalization, attention mechanisms
- **Flexible Pipeline**: Modular components for easy customization

### Generative Models
- **Human cGAN**: Uses COCO pose format (17 keypoints) with U-Net generator
- **Animal cGAN**: Uses animal pose format (20 keypoints) with enhanced stability
- **Other cGAN**: Uses edge/corner features with self-attention for complex objects

## Output Structure

### Server Output (Metadata)
```
metadata_dir/
├── scene_XXXX_metadata.json    # Scene metadata
├── panoramas/
│   └── scene_XXXX_panorama.jpg # Background panoramas
└── objects/
    └── scene_XXXX/
        ├── object_track_X_frame_XXXX.png  # Object images
        └── ... 
```

### Client Output (Reconstructed Videos)
```
output_dir/
├── scene_XXXX_reconstructed.mp4  # Reconstructed videos
└── quality_report.json           # Quality assessment (if enabled)
```

## Performance Considerations

### Server Performance
- GPU acceleration for segmentation and keypoint extraction
- Parallel scene processing
- Efficient video I/O with frame caching
- Optimized homography computation

### Client Performance
- GPU training for generative models
- Batch processing for object generation
- Temporal smoothing for consistent output
- Efficient video encoding with FFmpeg

### Memory Management
- Streaming video processing for large files
- Automatic cleanup of temporary files
- Configurable batch sizes
- Memory-mapped file operations

## Quality Metrics

- **SSIM**: Structural similarity (higher is better, max=1.0)
- **PSNR**: Peak signal-to-noise ratio (higher is better, dB)
- **LPIPS**: Learned perceptual image patch similarity (lower is better)
- **VMAF**: Video multi-method assessment fusion (higher is better, max=100)
- **MSE/MAE**: Basic pixel difference metrics

## Dependencies

### Core Dependencies
- PyTorch 2.6+ with CUDA support
- OpenCV for computer vision
- FFmpeg for video processing
- scikit-image for quality metrics
- NumPy, SciPy for numerical computation

### Optional Dependencies
- LPIPS for perceptual quality assessment
- Weights & Biases for experiment tracking
- TensorBoard for training monitoring
- VMAF for advanced video quality assessment

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch sizes in configurations
2. **FFmpeg not found**: Install FFmpeg or update PATH
3. **Model training slow**: Ensure GPU is available and drivers are updated
4. **Poor reconstruction quality**: Increase training epochs or adjust loss weights

### Performance Tips
1. Use SSD storage for better I/O performance
2. Ensure adequate GPU memory (8GB+ recommended)
3. Adjust worker processes based on CPU cores
4. Use appropriate batch sizes for your hardware

## Research & Development

This pipeline is designed for research in:
- Video compression using generative models
- Semantic video analysis and reconstruction
- Human/animal pose-guided generation
- Multi-modal content representation
- Quality assessment of generated content

## Citation

If you use PointStream in your research, please cite:
```bibtex
@software{pointstream2024,
  title={PointStream: Server-Client Video Reconstruction with Generative Models},
  author={[Author Names]},
  year={2024},
  url={https://github.com/[username]/PointStream}
}
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]

## TODO

We need an optimizer script, that takes as input a folder of videos, and a number of tests per video, and runs the pipeline on each video, modifying the values of config at each test, to find the best parameter config that results in 30 fps of processing speed, at the highest quality. We should train a model for this: at each iteration, it will try a new parameter config, obtained by taking each config value in the file, and calculating a random value between its square and square root, then run the pipeline, check how long it takes, and what quality it results in. It then saves this parameter config, video name and scene processed, and the results it achieves in a file, and ...

We should also train a model to predict the mask based on the pose image, so that we can remove the area around the object and have a correctly transparent background instead of the black border that we have in the rectangular object images. This actually requires saving from the server side, the masks of the segmented objects, and putting them into the training cache, so that we can use them as target for this model. This should also help with occlusions: if we miss certain keypoints from the image, that means they were not visible, and therefore there is likely an object occluding them. Therefore, the mask should have transparency in correspondence to those keypoints, so that the occluding object is shown. And also, How do you suggest enhancing the appearance vector so that it can capture more than the first frame, but still be useful for the first frame? I was thinking, maybe we leave it as representative only of the first frame, but then we train two different models: one takes the appearance vector, and the pose image, and learns to recreate the frame object. Another takes the frame object, and the pose, and learns to warp the object to the new pose. But then I guess we might as well send the first frame as appearance, and do away with the first model. What are your suggestions for this problem?

We should also make the background transitions much smoother. I can see from the generated panorama that the algorithm can find pretty good frames to stitch. So I would expect it to then have smooth transitions between each pair of frames, based on their homography matrices, but this does not happen, even with smoothing. What I think we should have for smoothing, is that on the server side, we take the first frame and the last, try to create a panorama, if we fail we go down recursively, and unless the scene is too complex, we can find couples of images that can be stitched. Then, we take the first couple of images, calculate the homography, warp the second image to fit the first, and then take another couple images, warp the second to fit the first (of this second group), and then take these two intermediate panoramas, and calculate a third homography to warp the second intermediate panorama to fit the first intermediate panorama. Is that what we do? If not, what else do we do?
