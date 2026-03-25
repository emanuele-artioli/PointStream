# Pointstream (Scaffold v1)

This directory contains the **initial robust scaffold** for Pointstream, an object-centric semantic neural codec pipeline.

This version is intentionally **mock-first**:
- It implements architecture, data contracts, orchestration, and transport.
- It does **not** load real AI models yet.
- All extractors/renderers return deterministic dummy tensors with correct shape conventions.

## Project Layout

```text
pointstream/
  assets/
    weights/
  old/                      # Legacy implementation kept untouched
  scripts/
    download_weights.py
    run_mock_pipeline.py
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
      mock_extractors.py
      orchestrator.py
      residual.py
    decoder/
      mock_renderer.py
    transport/
      disk.py
```

## Environment Setup (CUDA-aware)

Use conda for explicit CUDA-compatible PyTorch.

```bash
cd /home/itec/emanuele/pointstream
conda env create -f environment.yaml
conda activate pointstream
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

## Notes

- `scripts/download_weights.py` validates expected weight files in `assets/weights/`.
- For missing custom weights, the script raises a clear `FileNotFoundError` with next actions.
- Real model integrations should replace only the mock class internals while preserving interfaces and schemas.