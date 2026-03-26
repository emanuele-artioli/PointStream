# Pointstream (Scaffold v1)

This directory contains the **initial robust scaffold** for Pointstream, an object-centric semantic neural codec pipeline.

This version is intentionally **mock-first**:
- It implements architecture, data contracts, orchestration, and transport.
- It does **not** load real AI models yet.
- All extractors/renderers return deterministic dummy tensors with correct shape conventions.

## System Prerequisites

Pointstream expects system-level FFmpeg tools to be available before running tests or pipeline commands:
- `ffmpeg`
- `ffprobe`

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

If you want to force non-default executable paths (for example `/opt/local/bin`), set:

```bash
export FFMPEG_BIN=/opt/local/bin/ffmpeg
export FFPROBE_BIN=/opt/local/bin/ffprobe
```

## Project Layout

```text
pointstream/
  .github/workflows/
    ci.yml
    release.yml
  .pre-commit-config.yaml
  Dockerfile.cpu
  Dockerfile.gpu
  assets/
    weights/
  old/                      # Legacy implementation kept untouched
  pyproject.toml
  scripts/
    download_weights.py
    run_mock_pipeline.py
  requirements-dev.txt
  src/
    __init__.py
    main.py
    shared/
      interfaces.py
      schemas.py
      synthesis_engine.py
      tags.py
    encoder/
      dag.py
      execution_pool.py
      mock_extractors.py
      orchestrator.py
      residual.py
    decoder/
      mock_renderer.py
    transport/
      disk.py
  tests/
    test_background.py
    test_dag.py
    test_decoder.py
    test_download_weights.py
    test_end_to_end_mock.py
    test_encoder_pipeline.py
    test_execution_pool.py
    test_integration_main.py
    test_schemas.py
    test_tags.py
    test_transport.py
```

## Environment Setup (CUDA-aware)

Use conda for explicit CUDA-compatible PyTorch.

Video decode is strict FFmpeg (`ffmpeg` + `ffprobe`) and stream-oriented by default (frame generator over pipe).
The project does not vendor FFmpeg via pip/conda packages.

```bash
cd /home/itec/emanuele/pointstream
conda env create -f environment.yaml
conda activate pointstream
```

For development tooling only (lint/type/pre-commit):

```bash
cd /home/itec/emanuele/pointstream
pip install -r requirements-dev.txt
```

## Python Packaging

The project is configured with a `pyproject.toml` (setuptools backend) so it can be installed as a package.

```bash
cd /home/itec/emanuele/pointstream
pip install -e .
```

## Run Mock Pipeline

```bash
cd /home/itec/emanuele/pointstream
python -m src.main
```

or

```bash
cd /home/itec/emanuele/pointstream
python scripts/run_mock_pipeline.py
```

The run writes a local payload bundle under:

```text
.pointstream/chunk_<chunk_id>/
```

with:
- `metadata.msgpack` for semantic metadata/events
- `residual.mp4` placeholder bytes for residual stream handoff

## Run Unit Tests

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_*.py"
```

Run only the end-to-end mock path (generates `assets/test_video.mp4`):

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_end_to_end_mock.py"
```

Run real background stitching test (requires `assets/real_tennis.mp4`):

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_background.py"
```

Run with a local coverage gate (same threshold as CI):

```bash
cd /home/itec/emanuele/pointstream
coverage run -m unittest discover -s tests -p "test_*.py"
coverage report --fail-under=80
coverage xml
```

## Lint and Type Check

```bash
cd /home/itec/emanuele/pointstream
ruff check src tests scripts
mypy --config-file pyproject.toml
```

## Pre-commit Hooks

```bash
cd /home/itec/emanuele/pointstream
pre-commit install
pre-commit run --all-files
```

## Docker (CPU)

Build and run the CPU-only container:

```bash
cd /home/itec/emanuele/pointstream
docker build -f Dockerfile.cpu -t pointstream:cpu .
docker run --rm pointstream:cpu
```

Run tests inside the container:

```bash
docker run --rm pointstream:cpu python -m unittest discover -s tests -p "test_*.py"
```

## Docker (GPU)

Build and run the CUDA-enabled container (requires NVIDIA Container Toolkit):

```bash
cd /home/itec/emanuele/pointstream
docker build -f Dockerfile.gpu -t pointstream:gpu .
docker run --gpus all --rm pointstream:gpu
```

## Continuous Integration

- GitHub Actions workflow: .github/workflows/ci.yml
- Triggers: pushes to main/master/march26version and all pull requests
- Runtime: Ubuntu + Python 3.10 + CPU PyTorch
- Jobs:
  - `lint`: `ruff check src tests scripts`
  - `typecheck`: `mypy --config-file pyproject.toml`
  - `tests`: `coverage run -m unittest discover -s tests -p "test_*.py"` + `coverage report --fail-under=80`

## Release Flow

- GitHub Actions workflow: `.github/workflows/release.yml`
- Trigger: push a tag matching `v*` (for example `v0.1.0`)
- Actions performed:
  - build source and wheel distributions with `python -m build`
  - upload `dist/*` artifacts
  - create GitHub release with generated notes and attached artifacts
  - build and push GPU Docker image to GHCR as:
    - `ghcr.io/<owner>/<repo>/pointstream-gpu:<tag>`
    - `ghcr.io/<owner>/<repo>/pointstream-gpu:latest`

Pull and run the published GPU image:

```bash
docker pull ghcr.io/<owner>/<repo>/pointstream-gpu:<tag>
docker run --gpus all --rm ghcr.io/<owner>/<repo>/pointstream-gpu:<tag>
```

## CPU/GPU Execution Pool Stub

- `src/encoder/execution_pool.py` provides:
  - `InlineExecutionPool`: synchronous local execution (default)
  - `TaggedMultiprocessPool`: torch.multiprocessing-ready stub with CPU/GPU queue separation
  - `make_shared_cpu_tensor(...)`: shared-memory tensor allocation helper for process handoff
- `DAGOrchestrator` can now run through an injected execution pool while preserving per-node CPU/GPU tags.
- `run_mock_pipeline(...)` supports execution-pool injection for integration testing of tagged dispatch.

## Notes

- `scripts/download_weights.py` validates expected weight files in `assets/weights/`.
- For missing custom weights, the script raises a clear `FileNotFoundError` with next actions.
- Real model integrations should replace only the mock class internals while preserving interfaces and schemas.