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
