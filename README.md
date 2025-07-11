# PointStream: A Generalized Semantic Streaming Framework

This project contains the complete implementation for the PointStream framework,
including the segmenter, extractor, client, and evaluation pipelines.

## Environment Setup

This project uses Conda to manage its dependencies. To create the environment, run:

```bash
conda env create -f environment.yml
conda activate pointstream
```

## Project Structure

- `pointstream/`: Main source code directory.
  - `core/`: Core data structures (Scene, DetectedObject, etc.).
  - `pipelines/`: Modules for each processing step (detection, pose, etc.).
  - `utils/`: Helper functions for video, etc.
- `scripts/`: Main entry points for running the different processes.
- `tests/`: Unit and integration tests.
- `data/`: (Gitignored) For storing input videos.
- `output/`: (Gitignored) For storing processed scenes and bitstreams.
- `environment.yml`: The Conda environment definition file.

## Usage

1. **Segment Video:**
   `python scripts/segmenter.py --input-video data/my_video.mp4 --output-dir output/segmented_video`

2. **Extract Bitstream (for static scenes):**
   `python scripts/extractor.py --input-dir output/segmented_video`

3. **Run Client:**
   `python scripts/client.py --input-bitstream output/segmented_video/scene_001_pointstream.bin`