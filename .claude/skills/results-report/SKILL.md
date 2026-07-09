---
name: results-report
description: Summarize or compare POINTSTREAM run results under outputs/ (payload size accounting, residual vs metadata trade-off, PSNR/SSIM/VMAF, stage timings). Use when the user wants a table, comparison, or analysis of pipeline runs rather than raw JSON.
---

# Summarizing POINTSTREAM results

## Where results live

Each run is `outputs/<YYYYMMDD_HHMMSS_micros>/run_summary.json`. Key blocks:

- Config echo: `source_uri`, `num_frames`, `genai_backend`,
  `compositing_mask_mode`, `residual_mode`, `transport_backend`.
- `evaluation.sizes_bytes` — **the headline numbers**: `source`, `metadata`,
  `actor_reference`, `residual`, `panorama`, `transport_total`,
  `transport_to_source_ratio`, `transport_savings_percent`.
- `evaluation.timings_sec` — nested per-stage timings (`encode_chunk` →
  actor_bundle/residual/panorama/ball, `decode` → synthesis/genai/composite,
  `transport_send/receive`, `quality_evaluation`).
- `psnr_mean`/`psnr_std`/`psnr_num_frames` (plus SSIM/VMAF when
  `evaluation-mode` requested them). `psnr_mean: null` + "no valid frame
  pairs" = broken evaluation, flag it — don't silently omit the run.
- `scripts/evaluate_experiments.py outputs/<ts>/` re-evaluates post hoc into
  `evaluation_summary.json`.

## Methodology — the Residual Guarantee

The claim under test is always
`metadata + residual < full-frame encoding at equal quality`:

- **Compare against the Whole-Frame Residual Baseline** (same input, same
  codec settings, component under test disabled). A component is justified
  iff it reduces `residual` bytes by more than the `metadata` bytes it adds.
- Compare like with like: same `source_uri`, same `num_frames`, same
  `ffmpeg-codec`/`codec-crf`/`codec-preset`, same seed. A 10-frame smoke run
  is not comparable to a full run.
- `transport_total` (metadata + references + residual + panorama) is the
  bitrate-side number; quality-side is PSNR/SSIM/VMAF of `decoded/` vs
  source. Report both — savings at unknown quality is not a result.
- Beware amortization: panorama and actor-reference bytes are per-chunk
  setup costs; on 10-frame runs they dominate and understate the method.

## Workflow

1. Enumerate candidate runs: `ls outputs/` and check each
   `run_summary.json`'s config echo to find matched pairs (use
   python/jq — don't eyeball 100-line JSONs).
2. Build a comparison table: run id (timestamp), backend/config delta,
   metadata bytes, residual bytes, transport_total, savings %, PSNR, wall
   time.
3. State the verdict in Residual-Guarantee terms (component pays for
   itself or doesn't), noting num_frames and any null metrics.
4. Plots/media go to disk under the scratchpad or `outputs/debug/` — the
   host is headless.
5. If the finding matters beyond this conversation, fold it into
   `reports/` with `/update-reports`.
