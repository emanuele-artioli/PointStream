# Video Scene Splitter

This directory contains scripts for automatically splitting videos into scenes using PySceneDetect and encoding them with AV1 compression.

## Scripts

### 1. `video_scene_splitter.py` (Main Script)
The full-featured script that processes videos in batches and handles large files efficiently.

**Features:**
- Processes videos in batches of 100 frames (configurable)
- Detects scene cuts using PySceneDetect
- Enforces maximum scene length of 1000 frames (configurable)
- Outputs scenes in AV1 format using FFmpeg
- Merges adjacent scenes automatically
- Creates detailed logs and scene lists

**Usage:**
```bash
# Basic usage
python video_scene_splitter.py input_video.mp4

# Specify output directory
python video_scene_splitter.py input_video.mp4 ./output_scenes

# Custom batch size and max frames
python video_scene_splitter.py input_video.mp4 --batch-size 200 --max-frames 2000

# Create a scene list file
python video_scene_splitter.py input_video.mp4 --create-list
```

### 2. `simple_scene_splitter.py` (Simplified Version)
A simpler version that processes the entire video at once. Good for testing and smaller videos.

**Usage:**
```bash
# Basic usage
python simple_scene_splitter.py input_video.mp4

# Custom settings
python simple_scene_splitter.py input_video.mp4 --output-dir ./scenes --max-frames 500 --threshold 25.0
```

### 3. `check_dependencies.py` (Dependency Checker)
Verifies that all required dependencies are installed and configured correctly.

**Usage:**
```bash
python check_dependencies.py
```

## Setup

1. **Activate the pointstream environment:**
   ```bash
   conda activate pointstream
   ```

2. **Install additional dependencies if needed:**
   ```bash
   pip install scenedetect[opencv]
   ```

3. **Verify dependencies:**
   ```bash
   python check_dependencies.py
   ```

4. **Make scripts executable (optional):**
   ```bash
   chmod +x *.py
   ```

## Output Format

The scripts generate:
- Individual scene files: `scene_0001.mp4`, `scene_0002.mp4`, etc.
- AV1 encoding with libsvtav1 encoder
- Opus audio compression
- Preserves original video timing and quality settings

## Configuration Options

### Scene Detection
- `--threshold`: Scene detection sensitivity (default: 30.0)
  - Lower values = more sensitive (more scenes)
  - Higher values = less sensitive (fewer scenes)

### Processing
- `--batch-size`: Frames per batch (default: 100)
- `--max-frames`: Maximum frames per scene (default: 1000)

### Output
- `--output-dir`: Custom output directory
- `--create-list`: Generate a text file listing all scenes

## Examples

### Example 1: Basic Scene Splitting
```bash
python video_scene_splitter.py movie.mp4
```
Output: `movie_scenes/scene_0001.mp4`, `movie_scenes/scene_0002.mp4`, etc.

### Example 2: High-Sensitivity Detection
```bash
python simple_scene_splitter.py movie.mp4 --threshold 15.0
```
This will detect more scene changes (more sensitive).

### Example 3: Large Video Processing
```bash
python video_scene_splitter.py large_video.mp4 --batch-size 500 --max-frames 5000
```
Processes larger chunks and allows longer scenes.

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**
   - Ensure FFmpeg is installed and in your PATH
   - The pointstream environment includes FFmpeg

2. **"Module not found" errors**
   - Activate the pointstream environment: `conda activate pointstream`
   - Run the dependency checker: `python check_dependencies.py`

3. **AV1 encoding errors**
   - Some FFmpeg builds don't include AV1 support
   - The script will attempt to use available encoders as fallback

4. **Memory issues with large videos**
   - Reduce batch size: `--batch-size 50`
   - Use the simple splitter for smaller videos

### Performance Tips

- **For large videos:** Use the main script with appropriate batch sizes
- **For quick testing:** Use the simple splitter
- **For high quality:** Increase AV1 CRF value (lower numbers = better quality)
- **For speed:** Increase AV1 preset value (higher numbers = faster encoding)

## Technical Details

### Scene Detection Algorithm
- Uses PySceneDetect's ContentDetector
- Analyzes frame-to-frame content changes
- Threshold determines sensitivity to changes

### Video Encoding
- **Video Codec:** AV1 (libsvtav1)
- **Audio Codec:** Opus
- **Quality:** CRF 30 (configurable in code)
- **Speed:** Preset 6 (configurable in code)

### Memory Management
- Batch processing prevents memory overflow
- Releases video resources after each batch
- Suitable for videos of any size
