# PointStream

PointStream is an object-centric semantic video codec where every component is a config choice. The encoder transmits each salient object's appearance and motion plus a reusable background model and an optional corrective residual; the client reconstructs video frames generatively or through composited references, with an explicit coded fallback when semantic models fail.

> [!NOTE]
> **Research Status**: PointStream is research software under active development. A confirmed rate–distortion win over conventional codecs (AV1 / VVC) is not yet established at the current evidence revision. The primary configuration search keeps generative synthesis off and evaluates semantic decomposition and background amortization.

---

## Supported Setup

- **Coordinator**: Work from the Mac checkout with Python and SSH access to `gpu1`–`gpu6`. Local CUDA is not required.
- **Compute**: The dispatch tool selects an available Linux GPU server for each CUDA job. The servers use PointStream's pinned Python environment and native codec tools.
- **Data**: Datasets and experiment outputs stay on the shared external data root, never inside this checkout.

---

## Installation

Clone the repository on the coordinator. The fleet dispatcher uses Python's standard library; project tests use the dependencies in `pyproject.toml`. CUDA, PyTorch, FFmpeg, and codec binaries are provided by the pinned environment on the GPU servers.

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

Read [the setup guide](docs/setup.md), check existing jobs and inspect the complete compatible fleet pool. Submit an immutable job specification with a representative smoke, workload-specific validator, saved full estimate, budget and absolute deadline:

```bash
scripts/ps-fleet status
scripts/ps-fleet inspect
scripts/ps-fleet submit /absolute/external/data/job.json
scripts/ps-fleet status JOB_ID
scripts/ps-fleet events JOB_ID
```

[Long jobs](docs/workflow/long-jobs.md) defines the specification and recovery procedure. Submission freezes the selected revision and explicit local changes. Smoke validation and identity/budget checks gate automatic full execution. Connection failures are per host; uncertain jobs are never replayed.

---

## Output Locations

- **Run Artifacts & Metrics**: Shared requests, snapshots and results are stored under `$PS_DATA_ROOT/jobs/fleet/inbox/<job-id>/`; datasets and older experiment outputs remain under the external data root.
- **Logs**: Each remote run keeps `command.log`, `status.json`, monitor state, the dispatch manifest, and outputs alongside the run record. A compact manifest is also stored on the coordinator.

---

## Repository Structure

```text
├── src/
│   ├── contracts/          # Machine-checkable interfaces and configuration schemas
│   ├── components/         # Background, appearance, motion, residual, and generation modules
│   └── pipeline/           # Reconstruction, codec, and quality evaluation engines
├── config/                 # Shipped tier definitions (tier_fast.yaml, tier_balanced.yaml, tier_quality.yaml)
├── experiments/            # Benchmark scripts, ladders, and probe harnesses
├── tests/                  # Unit tests and end-to-end pipeline verification
└── docs/                   # Architecture, area documents, roadmap, and history
    ├── setup.md            # Local coordination, remote fleet, and data setup
    ├── roadmap.md          # Submission gates A through E
    ├── research/           # Qualified current evidence index and historical preparation
    ├── research-recovery/  # Source-backed historical dossier and provenance
    ├── areas/              # Retained historical notes by functional area
    ├── history/            # Pull request index, findings/retraction log, retired docs
    └── workflow/           # Experiment procedures and long-job dispatch
```

For overnight runs, use [script-based monitoring and bounded codec pilots](docs/workflow/long-jobs.md).

Storage uses the three roots `pointstream`, `Datasets`, and `Models`; see [storage layout and migration](docs/workflow/storage-layout.md) for path configuration, model repositories/checkpoints, compatibility, and the pending attended cutover.

Start scientific review at the [research index](docs/research/README.md), which separates current qualification from historical records.
