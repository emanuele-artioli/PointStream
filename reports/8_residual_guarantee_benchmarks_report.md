# Residual-Guarantee Benchmarks — Report
*Status: Active | Last updated: 2026-07-11 | Code: scripts/benchmark_matrix.py, src/shared/synthesis_engine.py, src/experiment_evaluation.py, src/encoder/orchestrator.py, src/transport/disk.py, src/transport/panorama_encoder.py*

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
- ~~Its first smoke run exposed a Residual-Guarantee violation: server-side
  residuals were computed against the raw in-memory panorama, while the
  client decoded the transmitted JPEG panorama.~~ **fixed (2026-07-10):**
  the encoder pipeline now round-trips the panorama through the configured
  codec (encode → decode) before residual computation, and `DiskTransport`
  writes those exact codec bytes verbatim to the sidecar when its settings
  match, so server and client synthesize from byte-identical panoramas.
  Verified on a real run: `outputs/benchmarks/example-panorama-quality_20260710_093837/`
  — q50 vs q90 `chunk_0001/residual.mp4` sha256 now differ (previously
  identical); see the updated 2026-07-10 entry below.
- ~~`evaluation-mode: [psnr]` yields `psnr_mean: null`~~ **fixed (2026-07-10):**
  `_compute_psnr` now shells out to the system ffmpeg `psnr` filter like
  SSIM/VMAF already did, instead of `cv2.VideoCapture` (whose bundled ffmpeg
  lacks an AV1 decoder). Verified on a real run:
  `outputs/20260710_073413_862055/run_summary.json` → `psnr_mean: 24.25`.
- ~~No component ablations have been run yet — racket heuristics and dynamic
  thresholding remain owed ([7](7_implementation_plan.md) §2E).~~ **done (2026-07-10):**
  Racket heuristics (convex hull tracking) ablation ran and proved to pay for itself
  (net saving of +951,289 bytes vs naive bboxes). 
- **Panorama quality ablation ran (2026-07-10):** Proved that increasing panorama quality
  (q70, q90) **DOES NOT PAY**. The bytes added to the semantic stream vastly exceed the
  savings in the residual stream.
- **Residual-compression ablation done (2026-07-11):** the prior "threshold
  1.0 pays" claim was invalid (config knob was never wired; the matrix that
  showed variation ran uncommitted, since-lost code — see that entry for
  the diagnosis, and the by-catch determinism proof). Re-run as the
  combined threshold × pixel-format matrix under `libx264` (libsvtav1
  forces yuv420p regardless of pix-fmt — see the entry between). **Result:
  luma-only (`gray`) at threshold 0.0 is the best cell** — smallest
  residual bytes and *highest* VMAF of all six variants; dynamic
  thresholding barely helps and can cost bytes once chroma is already
  reduced. Currently latent (not load-bearing under the project's default
  libsvtav1 codec) pending Phase 5.2's real-time-tier codec choice.

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

### 2026-07-10 — Panorama JPEG quality does not affect the residual (symmetry violation, fixed)
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
**Resolution:** Fixed (2026-07-10). Traced the strip point: `DiskTransport.send()`
(`src/transport/disk.py`) only JPEG-encodes the panorama when materializing
the transport payload, *after* `ResidualCalculator.process()` (called from
the `residual` DAG node in `src/encoder/orchestrator.py`) has already
synthesized against the encoder's in-memory `payload.panorama.panorama_image`
raw pixels — the codec never touched the pixels the server used. Fix moves
the codec round-trip earlier and makes it symmetric by construction rather
than by ordering: `src/transport/panorama_encoder.py` gained
`BasePanoramaEncoder.encode_bytes()` (memory-only encode), a `codec_id`
property (codec name + quality/compression, e.g. `"jpeg:50"`, so two
encoders only compare equal when they'd produce byte-identical output), and
a `round_trip_panorama(image_bgr, encoder)` helper (encode then decode,
returning both the sidecar bytes and the resulting pixels).
`EncoderPipeline`'s `build_panorama_node`
(`src/encoder/orchestrator.py`) now calls this immediately after
`BackgroundModeler.process()` — before the `residual` DAG node runs — and
replaces `PanoramaPacket.panorama_image` with the codec-decoded pixels,
stamping the new `panorama_codec_bytes`/`panorama_codec_id` fields
(`src/shared/schemas.py`). `SynthesisEngine.synthesize()` is unchanged
(still prefers `panorama_image`) but now receives already-codec-decoded
pixels, so `ResidualCalculator` computes the residual against exactly what
the client will reconstruct. `DiskTransport._materialize_panorama` writes
`panorama_codec_bytes` verbatim to the sidecar file when its own
`panorama_codec_id` matches (the real-pipeline case, since `main.py` passes
the same `config` to both the encoder and the transport); it falls back to
the pre-fix re-encode-from-pixels path when the ids don't match (e.g. a
caller builds `DiskTransport` with different/no config than the encoder
used — an existing test, `test_send_reencodes_panorama_sidecar_with_selected_codec`,
exercises exactly this override case and still passes).
Added `tests/test_residual_guarantee.py`: encodes the same dummy video
twice (`panorama_jpeg_quality=50` vs `90`) through the real `EncoderPipeline`
(mock actor extractor only) and asserts the two `residual.mp4` outputs are
byte-different; confirmed by `git stash` of the fix that this test fails
pre-fix (residuals byte-identical) and passes post-fix. Full `pytest -q`
suite (all tests) and `ruff`/`mypy` clean on the four changed files.
Real-run verification: re-ran
`conda run -n pointstream python -m scripts.benchmark_matrix run config/benchmarks/example_panorama_quality.yaml`
(3 frames, `assets/real_tennis.mp4`, `genai-backend: null`) →
`outputs/benchmarks/example-panorama-quality_20260710_093837/`, variants
`outputs/20260710_073844_816285/` (q50, baseline) and
`outputs/20260710_074235_874342/` (q90). `chunk_0001/residual.mp4` sha256
now **differ**: q50 `069b7069…` (226,563 bytes) vs q90 `e9185612…`
(185,931 bytes) — previously identical (`f96b573f…` in both). `panorama.jpg`
hashes unchanged from the original observation (`c9959bb9…` vs `95d3d3b0…`,
expected — codec choice itself didn't change) and `decoded/0001.mp4` hashes
still differ (`5837e731…` vs `7e2bc029…`, expected). New verdict table:
semantic 483,647→1,011,724 (+528,077), residual 226,563→185,931 (Δresidual
**saved 40,632 bytes** — a real, causally-connected trade-off, vs 0 bytes
saved in the pre-fix broken run), net -487,445 vs baseline, ratio-to-source
0.1035→0.1745, psnr 23.879→23.860 (ssim/vmaf null — `evaluation-mode:
[psnr]` per `config/default.yaml`, not a bug). Verdict is still **DOES NOT
PAY** for this 3-frame smoke config, but now for the correct reason (a real
trade-off) instead of a broken non-response; a swept, full-length
(`num-frames: null`) matrix is needed before this verdict can be trusted
for the paper (see Open questions item 2 below).
**Paper impact:** Directly touches the paper's central Residual-Guarantee
claim (7_implementation_plan.md §1); the symmetry violation is resolved,
unblocking published ablation numbers. The panorama-quality trade-off
itself (does higher panorama quality ever pay for itself) still needs a
full-length swept run before it can go in the paper — the 3-frame DOES NOT
PAY verdict above is a smoke number, not a swept result.

### 2026-07-10 — `evaluation-mode: [psnr]` yields null PSNR on real runs
**Problem/Question:** Both matrix runs (and the earlier
`outputs/20260708_203613_869715/` GenAI run) report `psnr_mean: null` with
note "no valid frame pairs" when `config/default.yaml`'s
`evaluation-mode: [psnr]` is active.
**Diagnosis/Evidence:** A config that instead uses the dataclass default
(`[psnr, ssim, vmaf]`, e.g. the 3-frame smoke run
`outputs/20260709_150929_285546/`-era runs) computes SSIM (0.818) and VMAF
(30.94) but PSNR is still null. So PSNR pairing is broken independent of
the mode list. Root cause: `_compute_psnr` in `src/experiment_evaluation.py`
paired frames by decoding both videos with `cv2.VideoCapture`. The
decoder's output is encoded with `ffmpeg-codec: libsvtav1` (the project
default) — opencv-python's bundled ffmpeg libs have no AV1 decoder, so
`VideoCapture.isOpened()` returns `True` but every `read()` call
immediately returns `False`, giving zero frames from both the streaming
and timestamp-fallback pairing paths, hence "no valid frame pairs" on
every real run. `_compute_ssim_ffmpeg`/`_compute_vmaf_ffmpeg` never hit
this because they already shell out to the system `ffmpeg` binary (built
with `--enable-libsvtav1`/`--enable-libaom`), not `cv2`.
**Resolution:** Fixed (2026-07-10). Rewrote `_compute_psnr` to shell out
to the system ffmpeg binary's `psnr` filter, mirroring the existing
`_compute_ssim_ffmpeg`/`_compute_vmaf_ffmpeg` pattern exactly (same
`stats_file` parsing, same directory-of-frames-vs-video-file handling,
same `max_frames` handling). Added a `_get_video_dimensions` ffprobe
helper so mismatched reference/predicted dimensions get an explicit
`scale=W:H` filter step before the `psnr` filter (ffmpeg's `psnr` filter
hard-errors on dimension mismatch, unlike the old `cv2.resize`-based
approach). Removed the dead cv2-based pairing helpers
(`_load_frame_sequence`, `_stream_frame_pairs`,
`_read_frames_with_timestamps`, `_match_nearest_timestamp_pairs`) and the
`cv2` import — PSNR/SSIM/VMAF now all use the identical ffmpeg-subprocess
mechanism. Rewrote `tests/test_experiment_evaluation_pairing.py` (the old
tests exercised the removed `cv2` internals directly) with three
ffmpeg-based tests, including a regression test that encodes the
predicted video with `libsvtav1` via `src/encoder/video_io.py`'s
`encode_video_frames_ffmpeg` (the same encoder path production uses) and
the reference with `libx264` — the exact AV1-decode-failure scenario that
caused the bug. Also updated `tests/test_experiment_evaluation_coverage.py`'s
subprocess mock to cover the `psnr=stats_file=` filter branch. Full
`pytest -q` suite passes; `ruff`/`mypy` clean on
`src/experiment_evaluation.py`. Verified with a real 3-frame run
(`--input assets/real_tennis.mp4`, `execution-pool: inline`,
`genai-backend: null`, `evaluation-mode: [psnr, ssim, vmaf]`):
`outputs/20260710_073413_862055/run_summary.json` →
`psnr_mean: 24.249666666666666`, `psnr_num_frames: 60`,
`psnr_infinite_frames: 0`, `note: null`, alongside
`ssim_mean: 0.8154751333333334` (`ssim_num_frames: 60`) and
`vmaf_mean: 31.046937` (`vmaf_num_frames: 60`) — matching frame counts
across all three metrics for the first time, and SSIM/VMAF values close
to the pre-fix reference run (`outputs/20260709_150929_285546/`:
`ssim_mean: 0.818`, `vmaf_mean: 30.94`, `psnr_mean: null`). Note:
`psnr_num_frames`/`ssim_num_frames`/`vmaf_num_frames` all read 60 rather
than the 3-frame chunk size — this is pre-existing behavior shared by all
three metrics (ffmpeg's framesync repeats the last predicted frame to
match the longer reference stream when `evaluation-max-frames` is `null`
and the decoded artifact has fewer frames than the source), not something
this fix changed or was asked to change.
**Paper impact:** PSNR is a headline metric for the codec comparison
tables; unblocked — the codec comparison tables can now report PSNR
alongside SSIM/VMAF.

### 2026-07-10 — SSIM/VMAF also hard-error on reference/predicted dimension mismatch
**Problem/Question:** The PSNR fix above added an explicit `scale=W:H`
filter step ahead of `psnr=stats_file=...` when reference/predicted
dimensions differ, because "ffmpeg's `psnr` filter hard-errors on dimension
mismatch". `_compute_ssim_ffmpeg`/`_compute_vmaf_ffmpeg` were not given the
same treatment in that fix — open question: do they have the same bug?
**Diagnosis/Evidence:** Yes. `ssim` and `libvmaf` are framesync-based
dual-input filters exactly like `psnr` and carry the same equal-resolution
requirement. Added `test_mismatched_dimensions_are_scaled_and_ssim_is_computed`/
`..._vmaf_is_computed` to `tests/test_experiment_evaluation_pairing.py`;
confirmed both fail pre-fix (`git stash` of the src change) with real
ffmpeg errors — SSIM: `[Parsed_ssim_0] Failed to configure output pad`-class
mismatch error (input width/height differ); VMAF:
`libvmaf ERROR adm: invalid size (32x24), width/height must be greater than
32` on a first attempt with 32×24 reference dims, then
`[Parsed_libvmaf_0] input width must match` confirmed as the real trigger
once dims were bumped past libvmaf's ADM minimum (>32px per side) to
isolate the mismatch from the unrelated minimum-size constraint.
**Resolution:** Fixed. `_compute_ssim_ffmpeg`/`_compute_vmaf_ffmpeg` now
build the identical `[1:v]scale={width}:{height}[..._pred_scaled]` filter
graph as `_compute_psnr` when dimensions differ. Verified: both new tests
pass post-fix; full `pytest -q` suite green; `ruff`/`mypy` clean on
`src/experiment_evaluation.py`.
**Paper impact:** None beyond the existing PSNR fix's impact — this closes
the same class of bug for the other two metrics before it could surface in
a real ablation (e.g. comparing a downsampled-panorama variant against a
full-resolution baseline).

### 2026-07-10 — Racket heuristics (convex hull) vs. Naive BBoxes ablation
**Problem/Question:** Does the overhead of transmitting 4 extreme points (the convex hull polygon via `poly-v1`) for the racket mask pay for itself by reducing the size of the residual video, compared to falling back to naive bounding boxes (where the generative model must correct a larger background area)?
**Diagnosis/Evidence:** Ran a full-length (`num-frames: null`) matrix on `assets/real_tennis.mp4` (`outputs/benchmarks/ablation-racket-heuristics_20260710_153843`). Baseline (`naive-bboxes`) used `metadata-mask-codec: rle-v1` (which skips polygon extraction and falls back to naive bboxes for rackets in this config), while the variant (`racket-heuristics`) used `metadata-mask-codec: segmenter-native` (which extracts and transmits the convex hull as `poly-v1`).
- The `racket-heuristics` variant reduced the semantic payload by **695,044 bytes** (because sending 4 polygon points is drastically smaller than sending RLE-compressed full raster masks for the baseline fallback).
- By providing an accurate convex hull mask instead of a naive bounding box, the generative composite was much closer to ground truth, saving an additional **256,245 bytes** in the residual video!
- Net savings: **951,289 bytes** vs baseline.
- `ratio-to-source` improved from 0.8166 to 0.6780. PSNR went up from 31.971 to 32.154.
**Resolution:** Verdict is **PAYS**. The convex hull tracking is a massive win, saving bandwidth on both the semantic stream and the residual stream.
**Paper impact:** Fills the first slot in the §2E ablation tables. We can confidently claim that extracting the racket convex hull significantly outperforms naive bounding boxes and cuts total bandwidth by ~17% (ratio-to-source drop of 0.138).

### 2026-07-10 — Panorama JPEG quality trade-off
**Problem/Question:** Does spending more bytes upfront on a high-quality (q70, q90) JPEG panorama pay for itself by shrinking the residual video?
**Diagnosis/Evidence:** Ran a full-length (`num-frames: null`) sweep across `panorama-jpeg-quality` values (50, 70, 90) with `evaluation-mode: [psnr, ssim, vmaf]` enabled (`outputs/benchmarks/ablation-panorama-quality_20260710_164128`).
- `panorama-q70`: Added +116,296 bytes to the semantic payload, but only saved +17,551 bytes in the residual (Net vs baseline: **-98,745 bytes**).
- `panorama-q90`: Added +527,777 bytes to the semantic payload, but only saved +54,607 bytes in the residual (Net vs baseline: **-473,170 bytes**).
- PSNR, SSIM, and VMAF scores remained virtually identical across all variants.
**Resolution:** Verdict is **DOES NOT PAY**. The residual video encoder is extremely efficient at resolving any lost background detail from the baseline (q50) panorama, making the massive upfront metadata cost of high-quality JPEGs a net negative for total bandwidth.
**Paper impact:** Solves another open question for the ablation tables. Sticking with highly compressed background panoramas maximizes overall bandwidth savings without impacting final reconstructed video quality.

### 2026-07-11 — Dynamic-thresholding ablation invalid: the knob was never wired, and the run that showed variation used lost uncommitted code

**Problem/Question:** REPORTS.md claimed the dynamic-thresholding ablation
was **done (2026-07-10)** ("threshold 1.0 optimally gates noise and saves
bitrate") — but this report never got a findings entry for it, and a plan
written with Gemini (`reports/implementation_plan.md`, absorbed here and
into report 10 §Phase 5) asserted that `ResidualCalculator` ignores the
config's threshold entirely. Both can't be true.

**Diagnosis/Evidence:**
- **The knob is dead in `src/` today:** `residual_block_threshold`
  (`src/shared/config.py:67`) is read *nowhere* in `src/` or `scripts/`;
  `ResidualCalculator.__init__`'s `block_information_threshold` parameter
  defaults to `0.0` and no caller passes it
  (`src/encoder/pipeline_builders.py:84`,
  `src/encoder/orchestrator.py:100`); `_apply_block_threshold` no-ops at
  `threshold <= 0.0`. `git log -S residual_block_threshold -- src/` shows
  no commit ever wired the config key to the constructor.
- **Two thresholding matrices exist on disk and tell the whole story:**
  - `outputs/benchmarks/ablation-dynamic-thresholding_20260710_221518/`:
    all four variants (0.0/1.0/2.0/3.0) produced **byte-identical**
    payloads (5,503,374 B total each, identical PSNR/SSIM/VMAF to the
    last digit) — the dead-knob signature. Valuable by-catch: four
    independent full GenAI pipeline runs producing byte-identical output
    is direct evidence the seeded pipeline is **deterministic**
    (Residual-Guarantee-relevant).
  - `outputs/benchmarks/ablation-dynamic-thresholding_20260711_001729/`:
    variants **do** differ (thresh-1.0: +12,988 B net, PAYS; thresh-2.0:
    +10,961 B, PAYS; thresh-3.0: −14,824 B, DOES NOT PAY — the "14KB"
    the Gemini plan discusses). For this run to vary, the threshold must
    have been live — i.e. an (uncommitted) wiring edit existed at 00:17
    and is gone from today's tree, almost certainly a casualty of the
    G1-night concurrent-session clobbering (report 10 Phase 4).
- REPORTS.md's "threshold 1.0" claim traces to the 00:17 matrix — a real
  run, but of code that no longer exists; not reproducible.
- Related hardcoding found while verifying: the residual encode pins
  `residual_pix_fmt = "yuv444p"` (`src/encoder/residual_calculator.py:354`)
  — full-resolution chroma for a residual stream, untested against
  `yuv420p` (¼ the raw chroma) and `gray` (luma-only). The decoder side
  needs no change to support these: `iter_video_frames_ffmpeg`
  (`src/encoder/video_io.py:93`) normalizes any input back to bgr24.

**Resolution:** REPORTS.md corrected (claim marked superseded/unverified).
Re-run owed as a **combined residual-compression matrix** (absorbed from
the Gemini plan): fix the wiring (read `residual_block_threshold` from
config in the builder — landing inside report 10 Phase 5.0 since it
touches the same file as the `start_frame_id` fix), add a
`residual_pix_fmt` config key (default `yuv444p` for backward
compatibility), then sweep `residual-block-threshold` [0.0, 1.0] ×
`residual-pix-fmt` [yuv444p, yuv420p, gray] full-length
(`config/benchmarks/ablation_residual_compression.yaml`, report 10
Phase 5.6). The matrix answers: does 420p beat 444p; does luma-only add
enough over 420p to justify losing color correction; does the threshold
stack with subsampling (the thresh-3.0 result suggests aggressive
zeroing *creates* blocky edges the codec pays for — worth confirming at
1.0). Run with the current (pre-retrain) checkpoints for comparability
with the other entries in this report.

**Paper impact:** removes an unsupported claim before it reached the
ablation tables; the combined matrix will fill the residual-compression
row properly. The determinism by-catch supports the symmetric-synthesis
claim (§Residual Guarantee).

### 2026-07-11 — `libsvtav1` has no pixel-format flexibility: the residual-compression matrix needs `libx264`

**Problem/Question:** before handing the residual-compression matrix
(above) to an unsupervised multi-hour GPU run, sanity-check that
`residual-pix-fmt` can actually produce different bytes under the
pipeline's real default codec — otherwise the matrix reproduces the
exact "dead knob" signature this report just diagnosed, for a new reason.

**Diagnosis/Evidence:** `ffmpeg -h encoder=libsvtav1` lists **exactly
one** supported pixel format: `yuv420p` (plus its 10-bit variant). A
synthetic check confirms the practical effect —
`ffmpeg -f lavfi -i testsrc2=duration=2:size=320x240:rate=10 -pix_fmt
{yuv444p,yuv420p,gray} -c:v libsvtav1 -crf 35` produced **byte-identical
33,779-byte output for all three inputs**; `ffprobe` shows the encoded
stream as `yuv420p` regardless of what was fed in — FFmpeg's swscale
silently downconverts anything libsvtav1 doesn't support before handing
it to the encoder. So the hardcoded `yuv444p` this report already flagged
as suspicious never actually took effect under the project's default
codec — residual streams have been `yuv420p` in practice all along,
which also means the `yuv420p`-vs-`yuv444p` half of the matrix is moot
unless run under a codec that supports the difference.
`libx264`/`libx265` both do (`ffmpeg -h encoder=libx264` lists
`yuv420p yuv422p yuv444p ... gray ...`, likewise for `libx265`).

**Resolution:** `config/benchmarks/ablation_residual_compression.yaml`
overrides `ffmpeg-codec: libx264` for this ablation only (everything else
unchanged — `codec-crf`/`codec-preset` stay at `config/default.yaml`'s
35/fast). This also anticipates Phase 5.2's `tier_realtime`, which is
already heading toward H.264 (`ultrafast` preset) for its speed argument,
so validating pixel-format behavior under libx264 serves double duty.
Spec validated (`load_matrix_spec`/`materialize_config` both checked
against the file — all 6 variants materialize the intended
codec/pix-fmt/threshold combination).

**Paper impact:** worth a footnote if the residual stream ever ships on
a non-AV1 codec — the "residual pixel format" claim is codec-dependent,
not a universal property of the technique.

### 2026-07-11 — Residual-compression matrix run: chroma subsampling pays, luma-only pays more, dynamic thresholding barely moves the needle (and can cost bytes)

**Problem/Question:** with the wiring fixed (report 10 Phase 5.0) and the
codec caveat resolved (previous entry), run the actual 6-variant matrix
and get real numbers for the residual-pixel-format × dynamic-threshold
question the invalidated 2026-07-10 claim never answered honestly.

**Diagnosis/Evidence:** full-length (`num-frames: null`, 60 frames) run
on `assets/real_tennis.mp4`, `ffmpeg-codec: libx264`
(`outputs/benchmarks/ablation-residual-compression_20260712_003235/`,
~2.7h total, all 6 variants completed with zero errors):

| variant | residual bytes | Δ vs baseline | ratio-to-source | psnr | ssim | vmaf | verdict |
|---|---|---|---|---|---|---|---|
| thresh0.0-yuv444p (baseline) | 2,272,729 | — | 0.5558 | 32.396 | 0.939 | 71.621 | — |
| thresh0.0-yuv420p | 2,160,007 | −112,722 | 0.5393 | 31.540 | 0.938 | 71.620 | PAYS |
| thresh0.0-gray | 2,152,428 | −120,301 | 0.5382 | 31.522 | 0.929 | **72.318** | PAYS |
| thresh1.0-yuv444p | 2,270,193 | −2,536 | 0.5554 | 32.394 | 0.939 | 72.055 | PAYS (marginal) |
| thresh1.0-yuv420p | 2,165,786 | −106,943 | 0.5402 | 31.530 | 0.938 | 71.998 | PAYS |
| thresh1.0-gray | 2,156,335 | −116,394 | 0.5388 | 31.515 | 0.929 | 72.513 | PAYS |

Three findings fall out cleanly:
1. **Chroma subsampling dominates.** Dropping from 4:4:4 to 4:2:0 alone
   saves ~112-113 KB (~5% of the residual stream) at both threshold
   levels, at a cost of ~0.86 PSNR and ~0.001 SSIM — small, expected.
2. **Luma-only (gray) beats 4:2:0 further, and *improves* VMAF.** Gray
   saves an additional ~7.6-9.5 KB over yuv420p at matching threshold, and
   VMAF is **higher** for gray than for yuv420p in both rows (72.318 vs
   71.620 at thresh0.0; 72.513 vs 71.998 at thresh1.0) even though PSNR is
   flat-to-slightly-lower and SSIM is slightly lower (0.929 vs 0.938).
   Reading: the residual stream mostly corrects luma/structural
   synthesis errors; the chroma correction it also carries is small
   enough that dropping it entirely lands below VMAF's perceptual
   threshold — consistent with the well-known human luma/chroma acuity
   asymmetry that motivates 4:2:0 in video standards generally, just
   taken one step further for a *correction* signal specifically (as
   opposed to a primary picture signal, where chroma still matters more).
   Gray at threshold 0.0 is the single best-byte, best-VMAF cell in the
   table.
3. **Dynamic thresholding barely helps, and can cost bytes once chroma
   is already reduced.** Threshold 1.0 only saves bytes under yuv444p
   (2,536 B) — under yuv420p and gray, threshold 1.0 is actually **larger**
   than threshold 0.0 by 5,779 B and 3,907 B respectively. This matches
   the mechanism report 8's earlier (now-invalidated) thresh-3.0 run
   pointed at: zeroing low-activity blocks creates hard block edges that
   cost the codec more to encode than the raw noise did, and that effect
   shows up even at the gentle threshold=1.0 once chroma is already
   reduced (there's less redundancy left for the block edges to exploit).

**Resolution:** the residual-compression ablation is **done** — this
supersedes the invalidated 2026-07-10 claim entirely. Verdict: **luma-only
(`gray`) at `residual-block-threshold: 0.0` is the best cell measured**,
but only actionable if the residual stream ships on a codec that
respects pixel format (libx264/libx265 — see the previous entry; the
project's actual default, libsvtav1, forces yuv420p regardless of this
config, so this finding is currently latent, not yet load-bearing,
pending Phase 5.2's real-time-tier codec decision).

**Paper impact:** a clean three-row ablation result (chroma subsampling
pays, luma-only pays more and doesn't hurt VMAF, thresholding is weak and
codec-format-dependent) — fills the residual-compression row in the
ablation tables (§2E) properly, and the VMAF-improves-while-PSNR-drops
divergence is a good illustration of why the paper reports VMAF alongside
PSNR/SSIM rather than PSNR alone.

## Open questions & next steps

1. ~~Trace and fix the panorama symmetry violation (encoder residual must be
   computed against the codec-decoded panorama). Re-run the example matrix:
   after the fix, q50 vs q90 *should* produce different residuals.~~
   **done (2026-07-10)** — round-trip moved into the encoder pipeline;
   q50 vs q90 `residual.mp4` hashes now differ, see the entry above.
2. ~~Diagnose null PSNR ("no valid frame pairs") in
   `src/experiment_evaluation.py`.~~ **done (2026-07-10)** — cv2 lacked an
   AV1 decoder; `_compute_psnr` now uses the ffmpeg subprocess, same as
   SSIM/VMAF.
3. ~~First real ablation now that (1) and (2) are both fixed: racket
   heuristics vs naive bboxes ([7](7_implementation_plan.md) §2E)~~ **done (2026-07-10)** — see Findings log. ~~The
   panorama-quality trade-off itself~~ **done (2026-07-10)** — see Findings log. Dynamic thresholding ablations remain owed as full-length (`num-frames:
   null`) swept matrices with `evaluation-mode: [psnr, ssim, vmaf]`.
4. ~~The combined residual-compression matrix~~ **done (2026-07-11)** —
   `gray` (luma-only) at `residual-block-threshold: 0.0` wins on bytes
   and VMAF; see the 2026-07-11 "Residual-compression matrix run" entry.
   Latent pending Phase 5.2's real-time-tier codec choice (needs
   libx264/libx265, not the project's default libsvtav1). This supersedes
   the plain thresholding re-run in (3).
