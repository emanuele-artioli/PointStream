# PointStream

An object-centric semantic video codec. Rather than transmitting compressed pixel
residuals like H.264, HEVC or AV1, PointStream transmits semantic understanding —
each salient object's appearance and motion, plus a background model and an
optional corrective residual — and reconstructs frames generatively on the
client. The trade is bandwidth for client-side compute.

The initial domain is **tennis**, chosen because the camera is largely static,
the background is knowable, actors are few and occlusions are mild. The
architecture is not tennis-specific: the task domain is a configuration.

> **This README is deliberately minimal.** The system is mid-rewrite, and a
> README describing an architecture that is being replaced is worse than a short
> one. See `PLAN.md` for what the system is and where it stands, and the paper's
> System Design section for why it is built this way. **TODO:** write this
> properly once the rewrite lands and the pipeline is stable.

## Where things are

| You want | Look at |
|---|---|
| What the system is, status, what is next | `PLAN.md` |
| Rules for working here (agents read this automatically) | `AGENTS.md` |
| The spec for one workstream | `plans/` |
| What a component must satisfy | `src/contracts/` |
| The manuscript | `67a9ea6275d3d9785ce57026/` — a separate git repo |

## Prerequisites

System FFmpeg tools, with `libvmaf` enabled for VMAF evaluation:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

To force non-default executable paths:

```bash
export FFMPEG_BIN=/opt/local/bin/ffmpeg
export FFPROBE_BIN=/opt/local/bin/ffprobe
```

Encoders used by the codec ladder: `libx264` and `libvvenc` through ffmpeg,
`kvazaar` and `SvtAv1EncApp` as standalone binaries. Region-of-interest control
is only reachable through the standalone binaries, and requires SVT-AV1 1.8 or
newer.

## Environment

`environment.yaml` is a GPU bootstrapper only — it fetches the heavy CUDA and
PyTorch binaries that pip struggles with. Every other Python package is managed
by `pyproject.toml`.

```bash
conda env create -f environment.yaml
conda activate pointstream
pip install -e .
```

## Model weights

Weights live in `assets/weights/`, mostly as symlinks to a shared model store.
Required for the default configuration: `yolo26n.pt`, `yolo26n-seg.pt`,
`yolo26n-pose.pt`. Optional for backend comparisons: `sam3.pt`,
`yoloe-26n-seg.pt` with `mobileclip2_b.ts`.

If a symlink dangles, ultralytics silently downloads a replacement into the
working directory rather than failing — so check that every weight a config names
actually resolves.

## Development

```bash
conda run -n pointstream python -m pytest tests/ -q
conda run -n pointstream ruff check src tests scripts experiments
conda run -n pointstream mypy --config-file pyproject.toml
conda run -n pointstream python -m src.contracts.layers   # import direction
```

Pre-commit hooks: `pre-commit install`, then `pre-commit run --all-files`.

## Containers and CI

CPU and GPU images build from `Dockerfile.cpu` and `Dockerfile.gpu`; the GPU
image needs the NVIDIA Container Toolkit. CI runs lint, typecheck and tests on
every push and pull request; `release.yml` builds distributions on a `v*` tag.
