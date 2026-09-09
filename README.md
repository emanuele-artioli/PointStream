# PointStream

PointStream is an object-centric semantic video codec where every component is a config choice. The encoder transmits each salient object's appearance and motion plus a reusable background model and an optional corrective residual; the client reconstructs video frames generatively or through composited references, with an explicit coded fallback when semantic models fail.

> [!NOTE]
> **Research Status**: PointStream is research software under active development. A confirmed rate–distortion win over conventional codecs (AV1 / VVC) is not yet established at the current evidence revision. The primary configuration search keeps generative synthesis off and evaluates semantic decomposition and background amortization.

---

## Supported Setup

- **Operating System**: Linux (tested on Ubuntu 22.04 LTS).
- **Compute**: NVIDIA GPU with CUDA support for neural perception and metric models.
- **Python**: 3.10.
- **System Tools**: FFmpeg (with `libvmaf`, `libsvtav1`, and `libaom` support), `vvencapp` (for VVC intra background encoding).

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/emanuele-artioli/PointStream.git
cd PointStream
conda env create -f environment.yaml
conda activate pointstream
```

Dependencies and package standards are defined in `pyproject.toml` and `environment.yaml`.

---

## Quick Start (Synthetic Test)

To verify that the configuration lattice, runner pipeline, and metric interfaces work without requiring large video datasets, run the synthetic tier end-to-end suite:

```bash
python -m pytest tests/runner/test_tier_end_to_end.py -q
```

This test constructs a small synthetic clip, drives each shipped tier configuration (`fast`, `balanced`, `quality`) through `src.runner.run`, verifies that disabled stages execute zero calls, and checks output score emission. If `ffmpeg` lacks `libvmaf`, VMAF-specific assertions are cleanly skipped.

---

## Real-Data Workflow

PointStream processes video scenes defined by input manifests:

1. **Configure External Data Root**:
   PointStream datasets live outside the git tree to maintain index performance. Specify your data directory via [docs/setup.md](docs/setup.md):
   ```bash
   echo "/path/to/pointstream-data" > .ps-data-root
   ```

2. **Execute Tier Configurations**:
   Run the tier benchmark runner on a prepared scene:
   ```bash
   python -m experiments.tier.run --tiers fast balanced quality --frames 8 --out outputs/report.json
   ```
   *Note*: The tier runner evaluates against reference clips defined in the data manifest. If external source video is not present, use the synthetic test suite above.

---

## Output Locations

- **Run Artifacts & Metrics**: Written to `outputs/` (or `$PS_DATA_ROOT/outputs/`). Results include full-frame and object-scoped PSNR, SSIM, VMAF, payload byte ledgers, and timing profiles.
- **Logs**: Execution logs are output to stdout or saved alongside run JSON manifests.

---

## Repository Structure

```text
├── src/
│   ├── contracts/          # Machine-checkable interfaces and configuration schemas
│   ├── components/         # Background, appearance, motion, residual, and generation modules
│   └── pipeline/           # Reconstruction, codec, and quality evaluation engines
├── configs/                # Shipped tier definitions (fast.yaml, balanced.yaml, quality.yaml)
├── experiments/            # Benchmark scripts, ladders, and probe harnesses
├── tests/                  # Unit tests and end-to-end pipeline verification
└── docs/                   # Architecture, area documents, roadmap, and history
    ├── setup.md            # Data root and environment configuration
    ├── roadmap.md          # Submission gates A through E
    ├── areas/              # Current state and next actions by functional area
    ├── history/            # Pull request index, findings/retraction log, retired docs
    └── workflow/           # Agent session dispatch and closeout workflows
```

For overnight runs, use [script-based monitoring and bounded codec pilots](docs/workflow/long-jobs.md).
