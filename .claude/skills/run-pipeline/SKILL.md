---
name: run-pipeline
description: Run the POINTSTREAM encode/decode pipeline via src/main.py, configure ablations through config YAMLs, and read back outputs. Use when the user wants to run the pipeline on a video, set up an ablation/experiment config, or check what a previous run produced.
---

# Running the POINTSTREAM pipeline

## Invocation

Only two CLI flags exist — **everything else is a config key**:

```
cd /home/itec/emanuele/pointstream   # required: `from src....` absolute imports
conda run -n pointstream python src/main.py \
    --input assets/real_tennis.mp4 --config config/default.yaml
```

Real end-to-end runs must pass `--input` explicitly (no `--input` = mock
source = tests nothing). Standard eval video: `assets/real_tennis.mp4`.

## Config cheat sheet (`config/default.yaml`)

Copy `config/default.yaml` for ablations; it documents every key. The ones
that matter most:

- `num-frames`: **default is 10** (smoke-test cap). A real experiment needs
  `num-frames: null`.
- Actor extraction: `detector` (any name containing `yolo`/`yoloe`, e.g.
  `yolo26n.pt` | `yoloe26n.pt` open-vocab), `pose-estimator` (name containing
  `yolo` for `yolo26n-pose.pt`, or `dwpose` — no hyphen — for
  `DwposeEstimator`), `segmenter` (name containing `yolo`, `yoloe`, or `sam`
  for `yolo26n-seg.pt` | `yoloe26n-seg.pt` | `sam3.pt`),
  `target-class-caption` for the open-vocab variants. **There is no
  config-string `mock` mode** — `src/main.py`'s builders only match these
  substrings; Mock classes are wired in only by unit tests via direct
  dependency injection.
- `ball-extractor`: `"segmentation"` selects `SegmentationBallExtractor`;
  any other value (including the default `difference`) falls through to
  the difference-based `BallExtractor` — `cascade`/`mock` in the config
  comment are aspirational, not implemented.
- `genai-backend`: `canny-controlnet` | `seg-controlnet` |
  `caption-controlnet` | `ip-adapter-controlnet` | `animate-anyone` |
  `pix2pix` | `null` (disables GenAI — much faster, residual-only).
- Codec: `ffmpeg-codec` (`libsvtav1`, `libx265`, …), `codec-crf`,
  `codec-preset`.
- Execution: `execution-pool` (`inline` | `tagged`), `gpu-dtype`
  (`fp16` default).
- Evaluation: `evaluation-mode` (`none` | `[psnr]` | `[psnr, ssim, vmaf]`),
  `evaluation-max-frames`.
- `seed`: keep deterministic — the Residual Guarantee depends on
  server/client synthesis agreeing bit-for-bit.

Fast smoke config (real backends, no mock mode exists): default
detector/pose-estimator/segmenter, `execution-pool: inline`,
`genai-backend: null`, `num-frames: 3`. ~3 min end-to-end on this host.

## Workflow

1. Confirm the input video exists (`ls assets/`) and weights are symlinked
   (`ls -l assets/weights/` — populate via `scripts/download_weights.py` or
   symlinks from `/home/itec/emanuele/Models`).
2. Write/copy the config for the run; keep it under `config/`.
3. **Smoke first:** small `num-frames` (e.g. `3`) + `genai-backend: null`
   to prove the plumbing before burning GPU time on a full GenAI run.
4. Run for real. GenAI runs cost minutes per 10 frames (a 10-frame
   canny-controlnet run measured ~6.5 min total) — use `run_in_background`
   or the `pipeline-runner` agent for anything full-length.
5. Read back `outputs/<YYYYMMDD_HHMMSS_micros>/`:
   - `run_summary.json` — the result: `evaluation.sizes_bytes` (source,
     metadata, residual, panorama, `transport_total`,
     `transport_to_source_ratio`), `timings_sec` per pipeline stage, and
     PSNR stats if evaluation ran.
   - `chunk_0001/` — `metadata.msgpack`, `panorama.jpg`, `residual.mp4`,
     `actor_references/`.
   - `decoded/0001.mp4` — the client-side reconstruction (watch this, or
     rather: save frames; host is headless).
   - `debug/` — intermediate artifacts.
6. Analyze against the Whole-Frame Residual Baseline per the Residual
   Guarantee (CLAUDE.md): did the component under test shrink
   `residual + metadata` vs the run without it? A run without its
   comparison target is not a result.

For a baseline-vs-variants ablation, don't run configs by hand — write a
matrix spec in `config/benchmarks/` and use the harness:
`python -m scripts.benchmark_matrix run config/benchmarks/<spec>.yaml`.
It materializes one config per variant, runs them sequentially (resumable —
completed variants are skipped on rerun), symlinks the run dirs under
`outputs/benchmarks/<name>_<ts>/runs/`, and emits the pays-for-itself table
(`report.md` + `report.json`). See `config/benchmarks/example_panorama_quality.yaml`.

## Gotchas

- Run from the repo root — `src/` uses `from src....` absolute imports.
- `psnr_mean: null` with note "no valid frame pairs" means evaluation found
  nothing to compare (e.g. decode produced no matching frames) — treat as a
  failure to investigate, not as "no metrics requested".
- Post-hoc evaluation of an existing run:
  `conda run -n pointstream python scripts/evaluate_experiments.py outputs/<ts>/`
  (writes `evaluation_summary.json` next to `run_summary.json`).
- opencv 4.8 / numpy 1.26.4 are ABI-coupled — don't bump either alone.
- Never `rm` the whole `outputs/` or `assets/` tree (guard-rm hook blocks
  it); discard one run by deleting its specific `outputs/<ts>/` dir.
