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
      residual_calculator.py
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

The current server driver is NVIDIA 535.x (`nvidia-smi` reports CUDA 12.2), so Pointstream pins CUDA 12.1 PyTorch builds for compatibility.

Video decode is strict FFmpeg (`ffmpeg` + `ffprobe`) and stream-oriented by default (frame generator over pipe).
The project does not vendor FFmpeg via pip/conda packages.

```bash
cd /home/itec/emanuele/pointstream
conda env create -f environment.yaml
conda activate pointstream
```

If `pointstream` already exists and previously resolved to CPU-only Torch, repair it in-place with:

```bash
cd /home/itec/emanuele/pointstream
conda activate pointstream
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
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

## Model Weights (Local Links)

Canonical model root on this machine:
- `/home/itec/emanuele/Models`

PointStream expects YOLO actor weights under `assets/weights/`:
- `yolo26n.pt`
- `yolo26n-seg.pt`
- `yolo26n-pose.pt`

If `assets/weights` is missing, recreate links from your shared model store:

```bash
cd /home/itec/emanuele/pointstream
mkdir -p assets/weights
ln -sfn /home/itec/emanuele/Models/yolo26n.pt assets/weights/yolo26n.pt
ln -sfn /home/itec/emanuele/Models/yolo26n-seg.pt assets/weights/yolo26n-seg.pt
ln -sfn /home/itec/emanuele/Models/yolo26n-pose.pt assets/weights/yolo26n-pose.pt
```

AnimateAnyone backend model variants are resolved from the Moore repo:
- `<repo>/Models/original`
- `<repo>/Models/finetuned_tennis`

Those two Moore paths are expected to be symlink views into the canonical model store:
- `/home/itec/emanuele/Models/AnimateAnyone/profiles/original`
- `/home/itec/emanuele/Models/AnimateAnyone/profiles/finetuned_tennis`

Recommended canonical layout:

```text
/home/itec/emanuele/Models/
  AnimateAnyone/
    profiles/
      original/
        stable-diffusion-v1-5/
        sd-vae-ft-mse/
        image_encoder/
        denoising_unet.pth
        reference_unet.pth
        pose_guider.pth
        motion_module.pth
      finetuned_tennis/
        stable-diffusion-v1-5/
        sd-vae-ft-mse/
        image_encoder/
        denoising_unet.pth
        reference_unet.pth
        pose_guider.pth
        motion_module.pth
  yolo26n.pt
  yolo26n-seg.pt
  yolo26n-pose.pt
```

Each variant folder must share the same structure:
- `stable-diffusion-v1-5/`
- `sd-vae-ft-mse/`
- `image_encoder/`
- `denoising_unet.pth`
- `reference_unet.pth`
- `pose_guider.pth`
- `motion_module.pth`

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
- `residual.mp4` encoded signed residual stream (H.265 / `libx265`)

Run with a custom input video and output folder:

```bash
cd /home/itec/emanuele/pointstream
python -m src.main --input /path/to/input.mp4 --output-dir /path/to/output_dir
```

Useful CLI options:
- `--num-frames N`: process only the first N frames
- `--summary-json /path/to/summary.json`: write summary JSON to a custom path
- `--no-summary-file`: print summary only (do not write `run_summary.json`)
- `--chunk-id custom_id`: choose chunk folder name (`chunk_<id>`)
- `--decoder-output-dir /path/to/decoded`: write decoded reconstructions to a custom directory

Module swap arguments (for ablations and performance tuning):
- `--execution-pool inline|tagged` with `--cpu-workers` and `--gpu-workers`
- `--actor-extractor real|mock`
- `--pose-estimator yolo|dwpose` (real actor extractor path)
- `--segmenter yolo|none` (real actor extractor path)
- `--payload-pose-delta-threshold <float>`
- `--ball-extractor difference|mock`
- `--ball-difference-threshold <float>` and `--ball-min-blob-area <int>`
- `--reference-jpeg-quality <1..100>` and `--reference-padding-ratio <float>`
- `--importance-mapper binary|uniform`

GenAI backend switches:
- `--enable-genai` or `--disable-genai`
- `--genai-backend controlnet|animate-anyone`
- `--animate-anyone-repo-dir /path/to/Moore-AnimateAnyone`
- `--animate-anyone-model-variant original|finetuned_tennis`
- `--animate-anyone-model-dir /path/to/model/profile`
- `--animate-anyone-window <int>`
- `--animate-anyone-transparent-threshold <int>`

Ablation examples:

```bash
# Fast mock ablation: no heavy detection/ball extraction
cd /home/itec/emanuele/pointstream
python -m src.main \
  --input /path/to/input.mp4 \
  --output-dir /tmp/ps_ablation_mock \
  --actor-extractor mock \
  --ball-extractor mock \
  --execution-pool inline \
  --importance-mapper uniform
```

```bash
# Real extraction with dwpose + no segmenter + tagged pool
cd /home/itec/emanuele/pointstream
python -m src.main \
  --input /path/to/input.mp4 \
  --output-dir /tmp/ps_ablation_dwpose \
  --actor-extractor real \
  --pose-estimator dwpose \
  --segmenter none \
  --execution-pool tagged \
  --cpu-workers 2 \
  --gpu-workers 1
```

## Run Unit Tests

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_*.py"
```

Run only the end-to-end mock path (generates `assets/test_runs/<timestamp>/test_video.mp4`):

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_end_to_end_mock.py"
```

Run real background stitching test (requires `assets/real_tennis.mp4`):

```bash
cd /home/itec/emanuele/pointstream
python -m unittest discover -s tests -p "test_background.py"
```

Run the coverage gate with a local safety buffer:

```bash
cd /home/itec/emanuele/pointstream
python scripts/check_coverage_gate.py
```

Default thresholds used by the script:
- local development: 85%
- CI: 80%

Override explicitly when needed:

```bash
cd /home/itec/emanuele/pointstream
POINTSTREAM_COVERAGE_THRESHOLD=87 python scripts/check_coverage_gate.py
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
docker run --rm pointstream:cpu python -m pytest
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
  - `tests`: `coverage run -m pytest` + `coverage report --fail-under=80`

## Test Tiers

- Default pytest run is fast by design and excludes integration/slow tests.
- Fast run: `pytest`
- Integration run (real YOLO weights): `pytest -m integration`
- Full run: `pytest -m "integration or slow or not (integration or slow)"`
- Integration tests load YOLO detect/seg/pose models once per session via `tests/conftest.py` fixtures.
- Unit plumbing tests use `MockActorExtractor` so they do not run real model inference on dummy videos.
- Test session startup creates a run-scoped artifact folder at `assets/test_runs/<timestamp>/` and exports `POINTSTREAM_DEBUG_ARTIFACT_DIR` for all test-time debug writers.

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
- Generated MP4 artifacts are encoded through `src/encoder/video_io.py::encode_video_frames_ffmpeg(...)` with explicit codec/pixel-format settings for player compatibility.
- `ActorExtractor` now runs a component-based pipeline (`detector -> heuristic -> segmenter -> pose -> payload encoder`) implemented in `src/encoder/actor_components.py`.
- Keyframe debug skeleton renders are written to `assets/test_runs/<timestamp>/debug_actors/` during tests; outside tests they default to `assets/debug_actors/`.
- `SynthesisEngine` (`src/shared/synthesis_engine.py`) now performs deterministic decode synthesis from payload metadata with strict seeding (`torch.manual_seed`, CUDA seed sync, deterministic algorithms enabled).
- Decoder-side panorama re-warping is GPU-native through `kornia.geometry.transform.warp_perspective` (inverse homography per frame), and mock actor compositing draws dense DWPose skeletons over reconstructed backgrounds.
- `tests/test_decoder.py` writes `assets/test_runs/<timestamp>/mock_reconstruction.mp4` so reconstruction smoothness and interpolation quality can be visually inspected without loading any heavy GenAI model.
- `ResidualCalculator` (`src/encoder/residual_calculator.py`) computes weighted server residuals using a pluggable saliency strategy (`BaseImportanceMapper`).
- Baseline saliency strategy is `BinaryActorImportanceMapper`, which converts player/racket segmentation masks from per-frame `FrameState` into continuous `[H, W]` importance tensors in `[0.0, 1.0]`.
- Residual encoding uses signed offsets with a neutral center: `encoded = clamp(((original - predicted) * importance) + 128, 0, 255)`.
- Transport now copies the encoded residual stream into `.pointstream/chunk_<id>/residual.mp4` and stores that chunk-local path in metadata.
- Residual stream encoding uses FFmpeg H.265 (`libx265`, `-crf 28`) for the transport payload.
- Client compositing decodes residual, shifts by `-128`, then applies `final = clamp(predicted + decoded_diff, 0, 255)` in `src/decoder/compositor.py`.
- `tests/test_residual.py` writes `assets/test_runs/<timestamp>/debug_residual.mp4` and `assets/test_runs/<timestamp>/debug_final_reconstruction.mp4` from a 10-frame real clip for offline verification of signed residuals and end-to-end reconstruction fidelity.
- `BallExtractor` (`src/encoder/ball_extractor.py`) computes per-frame parametric tennis-ball states (`x, y, vx, vy, visible`) by GPU-native panorama re-warp subtraction, actor-mask suppression, and largest-blob tracking.
- `SynthesisEngine` (`src/shared/synthesis_engine.py`) now reconstructs ball motion from parametric payloads and draws a velocity-aware motion-blurred ball trail during deterministic client synthesis.
- `tests/test_ball_tracking.py` validates both extractor-side ball trajectory recovery and full encode -> transport -> decode reconstruction with ball visibility in `debug_final_reconstruction_ball.mp4`.
- V2 baseline conditioning metadata now includes chunk-level `actor_references` (compressed per-track JPEG crops) in `EncodedChunkPayload`.
- `ReferenceExtractor` (`src/encoder/reference_extractor.py`) selects a best per-player frame (largest bbox, center-aware tie-break), applies 10% crop padding, and encodes references as JPEG bytes (`cv2.imencode`, quality 75).
- `DiskTransport` now writes metadata with python-mode pydantic dumps so binary JPEG payloads roundtrip losslessly through `.msgpack`.
- `DecoderRenderer` (`src/decoder/mock_renderer.py`) decodes actor reference JPEGs into an internal actor-state cache and passes them into `GenAICompositor`.
- `GenAICompositor` (`src/decoder/genai_compositor.py`) is the V2 interface stub for Animate Anyone integration; current mock behavior draws a filled actor box and pastes the reference crop to prove server -> transport -> client conditioning works.
- `tests/test_genai_baseline.py` validates extraction, transport serialization/deserialization, and reconstruction output with reference-conditioned mock compositing (`debug_final_reconstruction.mp4`).
- Heavy GenAI inference is feature-gated by `POINTSTREAM_ENABLE_GENAI`:
  - `0` or unset: `SynthesisEngine` exposes a lightweight `MockCompositor` (fast CI-safe default).
  - `1`: `SynthesisEngine` exposes `DiffusersCompositor`.
- GenAI backend strategy is selected with `POINTSTREAM_GENAI_BACKEND`:
  - `controlnet` (default when enabled): `BaselineControlNetStrategy` (`StableDiffusionControlNetImg2ImgPipeline`, OpenPose control).
  - `animate-anyone`: `AnimateAnyoneStrategy` (PointStream-owned runtime, requires `POINTSTREAM_ANIMATE_ANYONE_REPO_DIR`).
- AnimateAnyone runtime entrypoint lives in `src/decoder/animate_anyone_runtime.py` (integration code is maintained in PointStream, not in the external Moore repo).
- AnimateAnyone runtime model selection:
  - `POINTSTREAM_ANIMATE_ANYONE_MODEL_VARIANT=finetuned_tennis` (default) or `original`.
  - `POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR=<absolute-or-repo-relative-path>` overrides variant selection.
- AnimateAnyone output tuning:
  - `POINTSTREAM_ANIMATE_ANYONE_WIDTH` / `POINTSTREAM_ANIMATE_ANYONE_HEIGHT` (defaults: 512x784)
  - `POINTSTREAM_ANIMATE_ANYONE_STEPS` (default: 30)
  - `POINTSTREAM_ANIMATE_ANYONE_CFG` (default: 3.5)
  - `POINTSTREAM_ANIMATE_ANYONE_WINDOW` (default: 16, temporal pose window length for AnimateAnyone conditioning)
  - `POINTSTREAM_ANIMATE_ANYONE_TRANSPARENT_THRESHOLD` (default: 8, black-background alpha extraction)
- `tests/test_genai_node.py` is fully skipped unless `POINTSTREAM_ENABLE_GENAI=1`; when enabled it runs a 2-frame compositor smoke test and writes `assets/debug_genai_composite.mp4`.
- YOLO actor components load weights from local files first (`assets/weights/` or explicit path); implicit online weight download is disabled by default.
- Set `POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD=1` only if you intentionally want Ultralytics to fetch missing weights.
- Fail-fast policy: model initialization failures, missing source videos, and inference/runtime errors in detector/pose stages now raise explicit exceptions instead of silently injecting synthetic tracks/poses/black frames.
- Graceful degradation policy: noisy-data handling remains in place for track recovery/interpolation, segmenter per-frame bypass, homography identity fallback, and FFmpeg metadata tolerance.