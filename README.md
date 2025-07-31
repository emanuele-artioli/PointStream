# Pointstream: A Content-Aware Neural Video Codec

Pointstream is a research project implementing a semantic video codec. Instead of compressing raw pixels, it analyzes the video content to separate the background from foreground objects. It then transmits a highly compressed, structured representation of the scene, which is reconstructed on the client-side using generative AI models.

This approach aims to achieve significantly lower bitrates for videos with simple camera motion compared to traditional codecs like AV1.

## Core Concepts

- **Content-Awareness:** The codec differentiates between static backgrounds, moving objects, and camera motion.
- **Disentanglement:** It separates an object's **appearance** from its **motion**, encoding them independently.
- **Hybrid Encoding:** Scenes with complex motion are gracefully handled by a traditional codec (AV1), while simpler scenes leverage the neural codec.

## Project Structure

The project is organized into a modular pipeline:

- `pointstream/pipeline/`: Contains the core server-side processing stages.
- `pointstream/models/`: Wrappers for interacting with AI models (YOLO, SAM, etc.).
- `pointstream/utils/`: Generic utility functions for video and data encoding.
- `pointstream/client/`: The client-side reconstruction logic.
- `scripts/`: Standalone scripts, such as for training student models with `autodistill`.
- `tests/`: Unit and integration tests for each module.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/pointstream.git](https://github.com/your-username/pointstream.git)
    cd pointstream
    ```

2.  **Create the Conda environment:**
    This project uses Conda to manage dependencies. Ensure you have Anaconda or Miniconda installed.
    ```bash
    conda env create -f environment.yml
    conda activate pointstream
    ```

3.  **Set up API Keys:**
    The project uses the Google Gemini API for some analysis tasks. Set your API key as an environment variable.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

To run the main processing pipeline on a video:

```bash
python pointstream/main.py --input-video /path/to/your/video.mp4
```

To run the tests:

```bash
pytest
```

## Research & Ablation Studies

This repository is designed to support the research outlined in the paper. Detailed plans for experiments and ablation studies can be found in `ABLATION_STUDIES.md`.

