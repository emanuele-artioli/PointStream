---
name: results-report
description: Summarize or compare POINTSTREAM run results under outputs/ (payload size accounting, residual vs metadata trade-off, PSNR/SSIM/VMAF, stage timings). Use when the user wants a table, comparison, or analysis of pipeline runs rather than raw JSON.
---

Read `/home/itec/emanuele/.agent-rules/skills/results-report/SKILL.md` and follow it.

POINTSTREAM specifics:

- Citability field: `invariant_failures` in each run's
  `outputs/<YYYYMMDD_HHMMSS_micros>/run_summary.json`. Non-empty = not
  citable (mock fallback, incomplete evaluation, accounting that doesn't fit,
  transported payload larger than source). No key = never evaluated —
  backfill with `python -m src.shared.invariants outputs/` (as of
  2026-07-22 this flagged 27 of 52 runs, nearly all null `psnr_mean`).
- Headline numbers live in `evaluation.sizes_bytes`: `source`, `metadata`,
  `actor_reference`, `residual`, `panorama`, `transport_total`,
  `transport_to_source_ratio`, `transport_savings_percent`; timings in
  `evaluation.timings_sec`; quality in `psnr_mean`/`psnr_std`/
  `psnr_num_frames` (+ SSIM/VMAF when requested).
- The claim under test is always the **Residual Guarantee**:
  `metadata + residual < full-frame encoding at equal quality`. Compare
  against the Whole-Frame Residual Baseline (same input/codec, component
  disabled); a component pays for itself iff it cuts `residual` bytes by more
  than the `metadata` bytes it adds.
- Compare like with like: same `source_uri`, `num_frames`, codec
  settings/seed — a `num-frames: 10` smoke run (the config default) is not
  comparable to a full run.
- VMAF floor-saturates at 0.00 on 512×512 actor crops — use LPIPS/DISTS
  there, keep VMAF/FVD for full frames. Ranking metric is residual bytes;
  everything else is diagnostic.
- Benchmark harness runs: `outputs/benchmarks/<name>_<ts>/manifest.json` +
  `report.md`/`report.json`, regenerate via
  `python -m scripts.benchmark_matrix report <dir>` (also works ad hoc on
  `<baseline_run_dir> <variant_run_dir>...`).
- `scripts/evaluate_experiments.py outputs/<ts>/` re-evaluates post hoc.
- Fold findings that matter into the paper via `/update-paper`.
