# **Pointstream: A Content-Aware Neural Video Codec**

Pointstream is a research project and prototype implementation of a **semantic video codec**. Unlike traditional codecs that compress raw pixels, Pointstream analyzes video content to separate the static background from dynamic foreground objects. It then transmits a highly compressed, structured representation of the scene—consisting of appearance and motion data—which is reconstructed on the client-side using generative AI models.

This approach aims to achieve significantly lower bitrates for videos with simple camera motion compared to traditional codecs like AV1, forming the core thesis of the associated research paper.

## **Core Concepts**

* **Content-Awareness:** The codec differentiates between static backgrounds, moving objects, and camera motion, applying the optimal strategy for each.
* **Disentanglement:** The system separates an object's visual **appearance** (what it looks like, sent once) from its **motion** (how it moves, sent as a continuous, low-data stream of keypoints).
* **Hybrid Encoding:** The pipeline intelligently analyzes camera motion. **Simple scenes** (static, pans, zooms) are processed by the neural codec, while **complex scenes** are gracefully handed off to a traditional codec to ensure quality.

## **Project Structure**

The project is organized with a clean separation between the core Python package, executable scripts, and generated artifacts.

```
PointStream/
├── .gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Development guidelines  
├── LICENSE                    # MIT license
├── Makefile                   # Development commands
├── README.md
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Pip dependencies (fallback)
├── environment.yml           # Conda env for main pipeline
├── environment-training.yml  # Conda env for training
|
├── .github/                  # GitHub Actions CI/CD
│   └── workflows/
│       └── ci.yml
|
├── artifacts/                # All generated files (ignored by git)
│   ├── training/            # Labeled datasets and trained models
│   └── pipeline_output/     # JSON results and images from server
|
├── data/                    # Your raw, unlabeled images for training
|
├── pointstream/             # The core, importable Python package
│   ├── __init__.py         # Package initialization with version info
│   ├── cli.py              # Command line interface
│   ├── client/             # Client-side reconstruction logic
│   ├── config.py           # Centralized configuration
│   ├── models/             # Wrappers for AI models
│   ├── pipeline/           # The four stages of server pipeline
│   ├── scripts/            # Entry point scripts (moved from root)
│   └── utils/              # Reusable helper functions
|
├── tests/                  # Unit and integration tests
|
├── run_server.py          # Legacy entry point (use CLI instead)
├── run_client.py          # Legacy entry point (use CLI instead)
└── train.py               # Legacy entry point (use CLI instead)
```

## **Setup**

The project supports multiple installation methods. Choose the one that best fits your needs.

### **Quick Setup with Make (Recommended)**

If you have `make` installed, you can use our automated setup:

```bash
# Create conda environment and install dependencies
make conda-env
conda activate pointstream
make install-dev
make install-mmlab
```

### **Manual Setup**

#### **1. Main Environment Setup (`pointstream`)**

This environment is for running the main server pipeline and all tests.

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate pointstream
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e ".[dev,mmlab]"
   ```

4. **Install MMLab Dependencies:**
   ```bash
   mim install mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.2 mmpretrain
   pip uninstall mmcv -y
   FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir
   ```

#### **Alternative: pip-only Installation**

If you prefer not to use conda:

```bash
pip install -r requirements.txt
# Then follow the MMLab installation steps above
```

#### **Production Installation**

For production use, you can install the package directly:

```bash
pip install -e .
# or for development:
pip install -e ".[dev]"
```

### **2. Training Environment Setup (`pointstream-training`)**

This isolated environment is used **only** for running the `train.py` script.

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment-training.yml
   # or use: make conda-env-training
   ```

2. **Activate the environment:**
   ```bash
   conda activate pointstream-training
   ``` 

### **CUDA Compatibility & Troubleshooting**

PointStream requires CUDA-enabled PyTorch for optimal performance. Here's how to ensure compatibility:

#### **CUDA Version Requirements**
- **Minimum CUDA**: 12.1
- **Recommended CUDA**: 12.4+
- **PyTorch version**: 2.6.0+ with CUDA 12.4 support

#### **Check Your Setup**
```bash
# Check NVIDIA driver and CUDA support
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

#### **Fix CUDA Compatibility Issues**
If you encounter CUDA errors like "no kernel image is available for execution", update PyTorch:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### **Environment Variables**
For debugging CUDA issues, set:
```bash
export CUDA_LAUNCH_BLOCKING=1  # For detailed CUDA error messages
export MMPOSE_FORCE_CPU=1      # Force CPU for MMPose if needed
```

## **Usage**

> **Note:** The project now provides modern CLI commands (`pointstream-server`, `pointstream-client`, `pointstream-train`) and convenient Make targets. The original script files (`run_server.py`, etc.) are still available for backward compatibility but using the CLI commands is recommended.
> 
> **Important:** All usage commands require activating the appropriate conda environment first (`conda activate pointstream` or `conda activate pointstream-training`).

### **Command Line Interface**

After installation, you can use the convenient CLI commands:

```bash
# Run server pipeline
pointstream-server --input-video /path/to/video.mp4

# Run client reconstruction  
pointstream-client --input-json /path/to/results.json

# Train a model
pointstream-train --data_path /path/to/data
```

### **Using Make Commands**

For even more convenience, use the provided Makefile:

```bash
# Run server (pass video path)
make server VIDEO=/path/to/video.mp4

# Run client (pass JSON path)  
make client JSON=/path/to/results.json

# Train model (pass data path)
make train DATA=/path/to/data
```

### **1. Train a Custom Student Model**

(Requires the `pointstream-training` environment)

To train a new, specialized model (e.g., for sports), modify the `ontology` in `pointstream/scripts/train.py` to include your desired classes and prompts.

```bash
# Activate the training environment
conda activate pointstream-training

# Using the new CLI command (recommended)
pointstream-train --data_path ./data --epochs 200 --max_images 1000

# Or using Make
make train DATA=./data

# Or using the legacy script directly
python pointstream/scripts/train.py --data_path ./data --epochs 200 --max_images 1000
```

After training, copy the resulting `best.pt` file from `artifacts/training/train/weights/` to `pointstream/models/weights/yolo_sports_student.pt`. Then, add this new model to the `MODEL_REGISTRY` in `pointstream/config.py`.

### **2. Run the Server Pipeline**

(Requires the `pointstream` environment)

```bash
# Activate the main environment
conda activate pointstream

# Using the new CLI command (recommended)
pointstream-server --input-video /path/to/your/video.mp4

# Using Make
make server VIDEO=/path/to/your/video.mp4

# Or using custom-trained sports model
pointstream-server --input-video /path/to/your/video.mp4 --content-type sports

# Legacy method (still works)
python run_server.py --input-video /path/to/your/video.mp4
```


This will generate a `_final_results.json` file and all associated image artifacts in the `artifacts/pipeline_output/` directory.

### **3. Run the Client Reconstruction**

(Requires the `pointstream` environment)

```bash
# Activate the main environment
conda activate pointstream

# Using the new CLI command (recommended)
pointstream-client --input-json artifacts/pipeline_output/YOUR_VIDEO_final_results.json

# Using Make
make client JSON=artifacts/pipeline_output/YOUR_VIDEO_final_results.json

# Legacy method (still works)
python run_client.py --input-json artifacts/pipeline_output/YOUR_VIDEO_final_results.json
```

This will generate a `_reconstructed.mp4` video (or a folder of scene videos) in your project root.

## **Current Status & Placeholders**

The end-to-end pipeline is functional, but certain advanced components are currently implemented as simplified placeholders to allow for stable development and testing. These are primary areas for future research and implementation.

* **Rigid Object Keypoint Extraction:** In `stage_04_foreground.py`, the `_extract_feature_keypoints` function is a placeholder that currently returns only the four corners of an object's bounding box.
* **Background Mosaicking:** In `stage_03_background.py`, for scenes with `SIMPLE` camera motion, the background is currently represented by the first frame of the scene.
* **Client-Side Generative Model:** The `Reconstructor` in `pointstream/client/` uses a simple affine warp to demonstrate the reconstruction principle. This is a placeholder for the final, trained generative model.

## **Ablation Studies & Research Directions**

This project is designed to facilitate several key research experiments:

* **Student Model Architecture:** Compare the performance of different YOLO architectures when trained on the same auto-labeled dataset.
* **Specialized vs. Generalist Models:** Evaluate the trade-offs between using a single, general-purpose model versus a suite of specialized models.
* **Motion Representation for Rigid Objects:** Compare classical feature tracking against a custom, jointly trained keypoint extractor and object reconstructor.
* **Appearance Representation:** Investigate whether a learned appearance vector can achieve a better quality-to-bitrate ratio than transmitting a raw image patch.
* **End-to-End Codec Comparison:** Generate Rate-Distortion (RD) curves for the complete Pointstream system and compare them against traditional codecs like AV1.

## **Development**

### **Development Setup**

```bash
# Complete development setup
make dev-setup

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

### **Code Quality**

The project uses several tools to maintain code quality:

```bash
# Format code
make format

# Run linting
make lint  

# Type checking
make type-check

# Run tests
make test

# Run all CI checks
make ci-check
```

### **Pre-commit Hooks**

Pre-commit hooks are configured to run automatically before each commit:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- General file checks

### **Project Structure**

```
PointStream/
├── .gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT license
├── Makefile                   # Development commands
├── README.md
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda env for main pipeline
├── environment-training.yml  # Conda env for training
|
├── artifacts/                # All generated files (ignored by git)
│   ├── training/            # Labeled datasets and trained models
│   └── pipeline_output/     # JSON results and images from server
|
├── data/                    # Your raw, unlabeled images for training
|
├── pointstream/             # The core, importable Python package
│   ├── cli.py              # Command line interface
│   ├── client/             # Client-side reconstruction logic
│   ├── config.py           # Centralized configuration
│   ├── models/             # Wrappers for AI models
│   ├── pipeline/           # The four stages of server pipeline
│   ├── scripts/            # Entry point scripts
│   └── utils/              # Reusable helper functions
|
├── tests/                  # Unit and integration tests
|
├── run_server.py          # Legacy entry point (use CLI instead)
├── run_client.py          # Legacy entry point (use CLI instead)
└── train.py               # Legacy entry point (use CLI instead)
```

## **Quick Reference**

### **Common Commands**
```bash
# Setup (one time)
make dev-setup

# Development workflow
make format        # Format code
make lint         # Run linting
make test         # Run tests  
make ci-check     # Run all checks

# Usage
pointstream-server --input-video video.mp4
pointstream-client --input-json results.json
pointstream-train --data_path ./data

# Or with Make
make server VIDEO=video.mp4
make client JSON=results.json
make train DATA=./data
```

### **File Locations**
- **Generated artifacts:** `artifacts/pipeline_output/`
- **Training data:** `data/`
- **Model weights:** `pointstream/models/weights/`
- **Configuration:** `pointstream/config.py`
- **Tests:** `tests/`

### **Key Files Added in This Version**
- `pyproject.toml` - Modern Python packaging configuration
- `requirements.txt` - Pip dependencies for non-conda users
- `Makefile` - Convenient development commands
- `CONTRIBUTING.md` - Development guidelines
- `CHANGELOG.md` - Version history
- `.pre-commit-config.yaml` - Code quality automation
- `.github/workflows/ci.yml` - Continuous integration

