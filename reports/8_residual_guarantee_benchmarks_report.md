# Residual-Guarantee Benchmarks — Report
*Status: Active | Last updated: 2026-07-10 | Code: scripts/benchmark_matrix.py, src/shared/synthesis_engine.py*

## Scope

This report owns the Residual-Guarantee ablation effort: the
`scripts/benchmark_matrix.py` harness (baseline-vs-variants matrices,
pays-for-itself verdicts) and the findings its runs produce. Component
ablation results (racket heuristics, dynamic thresholding, panorama
settings, GenAI backends) accrue here as dated entries.

## Current state (TL;DR)

- The harness works end-to-end: matrix spec in `config/benchmarks/` →
  materialized per-variant configs → sequential runs (resumable) →
  `report.md`/`report.json` under `outputs/benchmarks/<name>_<ts>/`.
- **Its first smoke run exposed a likely Residual-Guarantee violation:**
  server-side residuals are computed against a synthesis that uses the raw
  in-memory panorama, while the client decodes the transmitted JPEG
  panorama — so the residual does not perfectly correct the client
  reconstruction. Unresolved; see 2026-07-10 entry.
- `evaluation-mode: [psnr]` (the `config/default.yaml` setting) yields
  `psnr_mean: null` / "no valid frame pairs" on real runs. Unresolved.
- No component ablations have been run yet — racket heuristics and dynamic
  thresholding remain owed ([7](7_implementation_plan.md) §2E).

## Findings log

### 2026-07-10 — Harness built; first matrix run
**Problem/Question:** Every pending ablation decision reduces to the same
comparison (does the component cut residual bytes by more than the semantic
bytes it adds vs the Whole-Frame Residual Baseline?), previously assembled
by hand from individual `run_summary.json` files.
**Diagnosis/Evidence:** `scripts/benchmark_matrix.py` (22 unit tests in
`tests/test_benchmark_matrix.py`) + example spec
`config/benchmarks/example_panorama_quality.yaml`. Verified with a real
2-variant matrix (3 frames, `genai-backend: null`,
`assets/real_tennis.mp4`):
`outputs/benchmarks/example-panorama-quality_20260710_001315/` — baseline
`panorama-jpeg-quality: 50` vs variant `90`; q90 added +528,077 semantic
bytes, saved 0 residual bytes → DOES NOT PAY. 3-frame smoke numbers, not a
swept result.
**Resolution:** Harness merged; usage documented in README, CLAUDE.md, the
`run-pipeline`/`results-report` skills, and both sibling rule files.
**Paper impact:** Enables the §2E ablation tables; no paper numbers yet.

### 2026-07-10 — Panorama JPEG quality does not affect the residual (suspected symmetry violation)
**Problem/Question:** In the matrix above, changing
`panorama-jpeg-quality` 50→90 changed the panorama sidecar
(`chunk_0001/panorama.jpg` sha256 `c9959bb9…` vs `95d3d3b0…`) but the
residuals are byte-identical
(`chunk_0001/residual.mp4` sha256 `f96b573f…` in both
`outputs/20260709_221319_778070/` and `outputs/20260709_221714_389957/`),
while the decoded outputs differ
(`decoded/0001.mp4` sha256 `bcaa9f70…` vs `7a8ea7fe…`).
**Diagnosis/Evidence:** Identical residuals mean the server-side synthesis
that the residual is computed against never saw the JPEG-encoded panorama.
`SynthesisEngine._resolve_panorama_image`
(`src/shared/synthesis_engine.py:89`) prefers raw
`payload.panorama.panorama_image` pixels and only falls back to
`panorama_uri`; on the encoder the in-memory payload carries raw pixels,
whereas the client after transport reconstructs from the JPEG. Differing
decoded outputs confirm the client-side path uses the JPEG. Therefore
`original = synthesis + residual` holds only for the *server's* synthesis,
not the client's — the core symmetry rule (CLAUDE.md "Symmetric synthesis")
appears violated, with reconstruction error scaling with panorama
compression. Single 3-frame observation; not yet traced through
`DiskTransport` to confirm exactly where `panorama_image` is stripped.
**Resolution:** Open. Candidate fix: encoder must compute the residual
against a synthesis that consumes the *codec-decoded* panorama (encode →
decode JPEG before synthesis), mirroring what the client will see.
**Paper impact:** Directly touches the paper's central Residual-Guarantee
claim (7_implementation_plan.md §1); must be resolved before any published
ablation numbers.

### 2026-07-10 — `evaluation-mode: [psnr]` yields null PSNR on real runs
**Problem/Question:** Both matrix runs (and the earlier
`outputs/20260708_203613_869715/` GenAI run) report `psnr_mean: null` with
note "no valid frame pairs" when `config/default.yaml`'s
`evaluation-mode: [psnr]` is active.
**Diagnosis/Evidence:** A config that instead uses the dataclass default
(`[psnr, ssim, vmaf]`, e.g. the 3-frame smoke run
`outputs/20260709_150929_285546/`-era runs) computes SSIM (0.818) and VMAF
(30.94) but PSNR is still null. So PSNR pairing is broken independent of
the mode list. Not yet diagnosed (frame pairing logic in
`src/experiment_evaluation.py`).
**Resolution:** Open.
**Paper impact:** PSNR is a headline metric for the codec comparison
tables; blocked until fixed.

## Open questions & next steps

1. Trace and fix the panorama symmetry violation (encoder residual must be
   computed against the codec-decoded panorama). Re-run the example matrix:
   after the fix, q50 vs q90 *should* produce different residuals.
2. Diagnose null PSNR ("no valid frame pairs") in
   `src/experiment_evaluation.py`.
3. First real ablation once (1) lands: racket heuristics vs naive bboxes
   ([7](7_implementation_plan.md) §2E), full-length
   (`num-frames: null`) matrix.
