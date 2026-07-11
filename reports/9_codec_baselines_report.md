# Codec Baselines — Report
*Status: Active | Last updated: 2026-07-11 | Code: scripts/codec_baseline_sweep.py, scripts/hnerv_baseline.py*

## Scope

This report owns the reviewer-critical "missing baselines" gap (R2, R5;
[6_action_matrix.md](6_action_matrix.md)): what conventional codecs achieve
on the tennis dataset with no semantic decomposition at all, so PointStream's
transport_total can be placed on a rate-distortion map against real anchors.
`scripts/codec_baseline_sweep.py` encodes the raw source video directly with
AV1 (`libsvtav1`) and HEVC (`libx265`) across a CRF ladder and reports
bytes/PSNR/SSIM/VMAF per point, optionally alongside a PointStream
`run_summary.json` for a side-by-side table.

This is a different comparison than
[8_residual_guarantee_benchmarks_report.md](8_residual_guarantee_benchmarks_report.md)'s
Whole-Frame Residual Baseline, which still pays for panorama + actor
references — the codec-baseline sweep here has zero semantic metadata, the
true floor/ceiling anchor a reviewer would expect.

## Current state (TL;DR)

- Tool built (2026-07-10): reuses `src.encoder.video_io.encode_video_frames_ffmpeg`
  (the exact FFmpeg wrapper the pipeline uses for its own residual stream)
  and `src.experiment_evaluation.evaluate_run_summary` (the same PSNR/SSIM/VMAF
  code path the pipeline evaluation uses). Supports a per-codec preset
  *ladder* (`--preset CODEC=p1,p2,...`, cross-producted with `--crf`), added
  specifically to sweep multiple encoder-effort tiers in one run.
- ~~Preset caveat: defaulting both codecs to `preset: fast` made HEVC
  dominate AV1~~ **fixed (2026-07-10):** default changed to `"slow"`.
- **Full-length sweep complete (2026-07-10, all 60 frames of
  `assets/real_tennis.mp4`, both `slow` and `veryslow`):** AV1 (`libsvtav1`)
  shows a real, consistent efficiency edge over HEVC (`libx265`) at matched
  VMAF — 5–32% fewer bytes depending on operating point and preset tier, the
  gap widening at lower bitrates and at the slower preset. See the
  "full-length sweep" entry below for the exact interpolated numbers. This
  resolves the earlier `fast`-preset domination and the inconclusive
  30-frame `slow` smoke test — both were artifacts of preset mismatch and
  small sample size, not a real absence of AV1's expected edge.
- Not yet compared side-by-side against a real PointStream `run_summary.json`
  via `--pointstream-run` — that's the next step to actually answer "does
  PointStream's semantic decomposition beat these curves."
- ~~One learned codec (HNeRV or DCVC) baseline remains entirely open (R2,
  R5).~~ **Closed (2026-07-11):** a from-scratch HNeRV implementation
  (`src/shared/hnerv_arch.py`, `scripts/hnerv_baseline.py`) trained and
  scored on the real 60-frame `assets/real_tennis.mp4` clip — see the dated
  entry below. **Honest result: it loses.** Every AV1/HEVC operating point
  in the full-length sweep above beats HNeRV on bytes *and* quality
  simultaneously except HEVC CRF18 (which is bigger but far higher quality).
  The reviewer-critical gap was "no learned-codec comparison exists" — one
  now exists, with real numbers, whichever way they land.

## Findings log

### 2026-07-10 — Harness built; smoke-verified on first 30 frames
**Problem/Question:** No conventional-codec anchor existed anywhere in the
repo — `scripts/benchmark_matrix.py` only compares PointStream pipeline
variants against each other (the Whole-Frame Residual Baseline), never
against a plain AV1/HEVC encode of the source video. Reviewers R2 and R5
explicitly flagged this omission.

**Diagnosis/Evidence:** Built `scripts/codec_baseline_sweep.py` and ran it
on `assets/real_tennis.mp4`, first 30 frames only (`--max-frames 30`, a
smoke test, not the real ablation), sweeping AV1 CRF {20,30,40,50} and HEVC
CRF {18,23,28,33} at `preset: fast` (matching `config/default.yaml`'s
`codec-preset`). Full table in
`outputs/codec_baselines/20260710_164414/report.md`:

| codec | crf | bytes | psnr | ssim | vmaf |
|---|---|---|---|---|---|
| AV1 | 20 | 1,354,539 | 39.325 | 0.992 | 96.813 |
| AV1 | 30 | 809,450 | 39.202 | 0.990 | 95.831 |
| AV1 | 40 | 468,338 | 38.906 | 0.988 | 94.281 |
| AV1 | 50 | 288,728 | 38.521 | 0.985 | 92.275 |
| HEVC | 18 | 2,688,034 | 39.294 | 0.993 | 97.649 |
| HEVC | 23 | 1,255,114 | 39.138 | 0.991 | 96.995 |
| HEVC | 28 | 642,773 | 38.854 | 0.988 | 95.644 |
| HEVC | 33 | 342,508 | 38.343 | 0.983 | 93.018 |

Bytes decrease monotonically with CRF for both codecs, but HEVC
**dominates** AV1 across this ladder rather than the reverse: HEVC CRF23
(1,255,114 bytes, 96.995 VMAF) is both smaller and higher-quality than AV1
CRF20 (1,354,539 bytes, 96.813 VMAF) — same pattern holds at the other
points too. This contradicts AV1's usual efficiency edge over HEVC and is
most likely a preset-parity artifact, not a real reversal of codec
efficiency: both codecs were run at `preset: fast`, but that name maps to
very different effort levels — SVT-AV1's `"fast"` resolves to numeric
preset **8** on its 0–13 (slowest→fastest) scale, a fairly speed-oriented
setting, while AV1 encoders are documented to only realize their
theoretical efficiency edge over HEVC at slower, more exhaustive presets.
`config/default.yaml`'s `codec-preset: fast` was tuned for PointStream's own
AV1 residual stream, not for a apples-to-apples cross-codec RD comparison —
reusing it verbatim for both codecs here was the mistake, not the tool's
plumbing. **Before the full-length sweep, revisit `DEFAULT_PRESET` (or sweep
presets explicitly) so the CRF ladder reflects each codec's realistic
best-effort operating point, not just a name collision.**

**Resolution:** Tool works end-to-end (`run outputs/codec_baselines/<ts>/`:
per-CRF `.mp4` files, `report.md`, `report.json`, `sweep.csv`). Tests added
(`tests/test_codec_baseline_sweep.py`, 9 cases covering CLI parsing, sweep
wiring via injected encode/evaluate functions, and report rendering);
`ruff`/`mypy` clean; full suite + coverage gate still pass at 85%
(215 passed, 1 skipped). This was a 30-frame smoke test to validate the
tool, not the real experiment — still open.

**Paper impact:** Feeds the Results section table reviewers requested
(R2, R5). No paper claim yet — numbers above are a tooling smoke test on 1
second of footage, not representative of the full dataset. **Superseded
2026-07-10:** these numbers used the buggy `preset: fast` default; see the
entry below for the corrected re-run.

### 2026-07-10 — Preset fix verified: domination resolved, but AV1 still not clearly ahead

**Problem/Question:** Does changing `DEFAULT_PRESET` from `"fast"` to
`"slow"` fix the HEVC-dominates-AV1 shape from the first smoke test?

**Diagnosis/Evidence:** Re-ran the identical 30-frame smoke test
(`assets/real_tennis.mp4`, `--max-frames 30`, same CRF ladders) at
`preset: slow`. Full table in
`outputs/codec_baselines/20260710_222256/report.md`:

| codec | crf | bytes | psnr | ssim | vmaf |
|---|---|---|---|---|---|
| AV1 | 20 | 1,322,189 | 39.354 | 0.992 | 96.784 |
| AV1 | 30 | 802,851 | 39.245 | 0.991 | 95.955 |
| AV1 | 40 | 460,629 | 38.994 | 0.989 | 94.649 |
| AV1 | 50 | 283,144 | 38.666 | 0.986 | 92.762 |
| HEVC | 18 | 3,293,337 | 39.325 | 0.994 | 97.736 |
| HEVC | 23 | 1,491,434 | 39.175 | 0.992 | 97.109 |
| HEVC | 28 | 729,847 | 38.910 | 0.989 | 95.774 |
| HEVC | 33 | 364,762 | 38.413 | 0.983 | 93.211 |

The strict domination is gone: e.g. AV1 CRF20 (1,322,189 bytes, 96.784 VMAF)
now has *fewer* bytes than HEVC CRF23 (1,491,434 bytes) but also slightly
*lower* VMAF (96.784 vs 97.109) — a genuine tradeoff point, not one codec
beating the other on both axes. AV1 CRF30 vs HEVC CRF28 is similarly mixed
(HEVC smaller, HEVC slightly lower VMAF). This is the expected RD-curve
shape — no domination in either direction — but it is **not** the ~20-30%
AV1-over-HEVC efficiency gap often quoted in the literature. Most likely
explanation: `libsvtav1` (the FFmpeg AV1 encoder used here, matching
PointStream's own `ffmpeg-codec` default) is known to trail the reference
`libaom-av1` encoder in compression efficiency at equivalent presets;
papers claiming AV1's full efficiency edge over HEVC in the SVT-AV1
implementation specifically typically need slower presets still
(`veryslow`/preset 4 or below) to show it clearly.

**Resolution:** Preset fixed to `"slow"` in `scripts/codec_baseline_sweep.py`
(`DEFAULT_PRESET`). Tests/ruff/mypy re-verified clean. The result is now
plausible enough to build the full-length sweep on, but the *lack* of a
decisive AV1 win at `slow` should be reported honestly in the paper table
rather than assumed away — it may be worth also sweeping `veryslow` for the
real experiment to see if the expected gap opens up, and noting `libsvtav1`
(not `libaom-av1`) as the specific AV1 implementation used.

**Paper impact:** This finding directly informs how the codec-baseline table
should be framed in the Results section — as "PointStream vs. a modern,
well-configured AV1/HEVC encoder," with the specific preset and encoder
implementation stated, rather than an unqualified "vs. AV1."

### 2026-07-10 — Full-length sweep: AV1's expected edge over HEVC confirmed at matched VMAF

**Problem/Question:** Does the full 60-frame video (not a 30-frame smoke
sample) at `slow` and `veryslow` presets show AV1's usual efficiency edge
over HEVC, resolving the inconclusive 30-frame `slow` result above? Required
extending `scripts/codec_baseline_sweep.py` to sweep multiple presets in one
run (`--preset CODEC=p1,p2,...`, cross-producted with `--crf`; previously
only one preset per codec per invocation).

**Diagnosis/Evidence:** Ran the full video (all 60 frames, no
`--max-frames`), both codecs, full CRF ladders, `preset` in `{slow,
veryslow}`. Full table in `outputs/codec_baselines/20260710_224721/report.md`:

| codec | crf | preset | bytes | vmaf |
|---|---|---|---|---|
| AV1 | 20 | slow | 2,562,545 | 97.487 |
| AV1 | 30 | slow | 1,481,893 | 96.661 |
| AV1 | 40 | slow | 808,460 | 95.331 |
| AV1 | 50 | slow | 453,237 | 93.094 |
| AV1 | 20 | veryslow | 2,839,857 | 97.582 |
| AV1 | 30 | veryslow | 1,541,760 | 96.862 |
| AV1 | 40 | veryslow | 797,439 | 95.483 |
| AV1 | 50 | veryslow | 449,612 | 93.466 |
| HEVC | 18 | slow | 6,562,943 | 98.254 |
| HEVC | 23 | slow | 2,916,807 | 97.679 |
| HEVC | 28 | slow | 1,379,563 | 96.303 |
| HEVC | 33 | slow | 679,055 | 93.670 |
| HEVC | 18 | veryslow | 6,813,860 | 98.259 |
| HEVC | 23 | veryslow | 3,020,543 | 97.734 |
| HEVC | 28 | veryslow | 1,425,972 | 96.419 |
| HEVC | 33 | veryslow | 696,529 | 93.761 |

Raw adjacent-CRF pairs still look mixed at a glance (e.g. AV1 CRF30/veryslow
is *larger* than HEVC CRF28/veryslow but at higher VMAF) because CRF values
don't land at exactly matched quality across codecs. Linearly interpolating
each HEVC curve to the exact VMAF of each AV1 point gives a cleaner,
apples-to-apples read (bytes HEVC would need for identical VMAF):

| preset | AV1 VMAF | AV1 bytes | HEVC bytes (interpolated to same VMAF) | AV1 saves |
|---|---|---|---|---|
| slow | 97.487 | 2,562,545 | 2,702,308 | 5.2% |
| slow | 96.661 | 1,481,893 | 1,779,515 | 16.7% |
| slow | 95.331 | 808,460 | 1,120,963 | 27.9% |
| veryslow | 97.582 | 2,839,857 | 2,836,228 | −0.1% (parity) |
| veryslow | 96.862 | 1,541,760 | 1,963,154 | 21.5% |
| veryslow | 95.483 | 797,439 | 1,169,103 | 31.8% |

(AV1 CRF50 at both presets falls below HEVC's tested VMAF range — HEVC's
lowest-quality point, CRF33, is already higher quality than AV1 CRF50 — so
no HEVC interpolation target exists there; not a data problem, just a range
mismatch in the CRF ladders as originally chosen.)

AV1 shows a real, consistent efficiency edge over HEVC once compared at
matched quality: roughly tied at the highest-quality tier tested, widening
to 17–32% fewer bytes at lower bitrates, and slightly larger at `veryslow`
than `slow`. This is the expected shape and resolves both earlier findings —
the `fast`-preset domination was a preset-parity bug, and the inconclusive
30-frame `slow` result was largely small-sample noise from testing half the
video.

One anomaly worth recording rather than hiding: at CRF20 and CRF30, AV1's
`veryslow` output is *larger* than its own `slow` output at the same CRF
(2,839,857 vs 2,562,545 bytes at CRF20), though VMAF is marginally higher.
Slower presets don't guarantee monotonically smaller files at fixed CRF —
CRF targets a perceptual-quality level, not a byte budget, and the encoder's
internal rate allocation can shift with search depth, especially in this
near-lossless region on a short (60-frame) clip. Not investigated further;
noted as a caveat for the paper if this exact CRF tier is used.

**Resolution:** `scripts/codec_baseline_sweep.py` now supports multi-preset
sweeps (`_parse_preset_overrides`, cross-product in `run_sweep`); tests
extended (`test_parse_preset_overrides`,
`test_run_sweep_cross_products_multiple_presets_with_crf`); ruff/mypy clean.
Full-length sweep data now exists and is trustworthy enough to build the
paper table on.

**Paper impact:** This is the AV1/HEVC anchor curve the Results section
needs (R2, R5). Still missing the actual PointStream data point(s) to
overlay — that's the next step, not yet done.

### 2026-07-11 — HNeRV learned-codec baseline: implemented, trained for real, and it loses

**Problem/Question:** Close the reviewer-critical "no learned-codec
baseline exists" gap (R2, R5; `reports/6_action_matrix.md`: "Benchmark
against at least one recent semantic codec (e.g., DVC or NeRV)"). This
report's TL;DR had flagged it as "entirely open" since 2026-07-10.

**Diagnosis/Evidence:** No pretrained HNeRV weights or reference
implementation existed anywhere under the shared `/home/itec/emanuele/Models`
cache (searched first, per CLAUDE.md's weights convention), so this required
a from-scratch implementation rather than a fetch-and-run. Built
`src/shared/hnerv_arch.py` — a simplified HNeRV (arXiv 2304.02633):
NeRVBlock (conv + PixelShuffle + GELU) decoder stack, a lightweight
strided-conv content-adaptive encoder standing in for the paper's ConvNeXt
encoder (discarded after training — only decoder weights + per-frame
embeddings are ever "transmitted," matching how HNeRV's own compression
numbers are reported), int8 embedding quantization + fp16 decoder weights,
gzip-compressed checkpoint serialization. `scripts/hnerv_baseline.py` wraps
it in the same shape as `scripts/codec_baseline_sweep.py`: train (per-video
overfit) → serialize → decode (decoder-only forward pass reloaded from
disk) → score via `src.experiment_evaluation.evaluate_run_summary`, the
identical PSNR/SSIM/VMAF code path the AV1/HEVC sweep uses, so the numbers
land on the same axes. Mock-first per CLAUDE.md: `tests/test_hnerv_arch.py`
+ `tests/test_hnerv_baseline.py` (fast, shapes/quantization round
trip/checkpoint round trip/CLI/report rendering) and
`tests/test_hnerv_baseline_integration.py` (integration+slow, proves the
full train→serialize→decode→score loop on a tiny synthetic clip) all pass;
ruff/mypy clean.

**Real run:** trained on all 60 frames of `assets/real_tennis.mp4` at a
reduced 640×360 training resolution (embedding grid 16×9×64, decoder
strides `[5, 4, 2]`, 3,101,795 decoder params, 15,000 epochs — a genuine
per-video overfit, not a smoke test), ~3,316 s (~55 min) wall-clock GPU
training, 0.44 s decode. `evaluate_run_summary` auto-scales the decoded
frames back to source resolution before scoring, the same mechanism the
AV1/HEVC sweep relies on for any resolution mismatch, so this stays
apples-to-apples. Artifacts: `outputs/hnerv_baseline/20260711_150415/`
(`report.md`, `report.json`, `history.json`, `progress.jsonl`,
`decoded_frames/`).

| codec | bytes | ratio-to-source | psnr | ssim | vmaf |
|---|---|---|---|---|---|
| HNeRV | 5,939,951 | 0.8656 | 32.502 | 0.918 | 79.079 |

**Compared directly against the full-length AV1/HEVC sweep above** (same
source video, same `evaluate_run_summary` scoring, so this is a real
apples-to-apples read, not an approximation): **HNeRV is strictly dominated
by every AV1 operating point tested.** AV1 CRF50/slow is 453,237 bytes —
13× *smaller* than HNeRV's 5.94 MB checkpoint — yet reaches PSNR
38.644/VMAF 93.094, far above HNeRV's PSNR 32.502/VMAF 79.079. The only
HEVC points bigger than HNeRV are CRF18 (slow 6.56 MB, veryslow 6.81 MB),
and those still reach PSNR ≈39.34–39.35/VMAF ≈98.25–98.26 — again far
higher quality at a comparable size. Every other HEVC point (CRF23/28/33,
both presets) is both smaller and higher-quality than HNeRV.

**Resolution:** Code landed on `main` as `7047207` (built on the spawned
worktree `worktree-agent-aa2b3e1a2a0fbcb8e`). Note on process: the agent
that built and trained this was cut off by a tooling outage before it
could fold the (already-complete) real run into this report — the
background training process kept running and finished on its own with the
numbers above; only the report-folding step was left undone. Re-verified
independently before landing: ruff, mypy, the full fast suite, and
`tests/test_hnerv_baseline_integration.py` all re-run from a clean state
and passed, not just trusted from the subagent's own claim.

**Paper impact:** Closes the R2/R5 "no learned-codec baseline" gap for the
Results section — the honest headline is that this specific from-scratch
HNeRV configuration (640×360 training resolution, 15,000-epoch
single-video overfit, ~3.1M decoder params) does not beat a well-tuned
AV1/HEVC anchor on this clip, on either bytes or quality. That is a
legitimate, reportable result: the ask was "does at least one recent
learned/semantic codec baseline exist for comparison," not "does it win."
Higher training resolution, more capacity, or more epochs could plausibly
close some of this gap and would be a natural follow-up if the paper wants
a stronger learned-codec showing, but is not required to close this gap.

## Open questions & next steps

1. Run a real PointStream pipeline pass (`num-frames: null`) at a
   `codec-preset` matching one of the tiers tested here (`slow` or
   `veryslow`, since that's what the anchor curve now uses) and pass its
   `outputs/<timestamp>/` dir via `--pointstream-run` to get the direct
   side-by-side table (PointStream `transport_total` vs the AV1/HEVC curve).
   This is the step that actually answers the paper's core claim.
2. ~~Learned-codec baseline (HNeRV or DCVC) — not started; heavier lift (new
   dependency, likely GPU).~~ **Done (2026-07-11)** — HNeRV implemented and
   trained for real; see the dated entry above. It loses to AV1/HEVC on this
   clip as configured; a higher-capacity/higher-resolution re-run is a
   possible follow-up, not a blocker.
3. Consider a VVC anchor too — currently deferred post-core per
   [7_implementation_plan.md](7_implementation_plan.md) §4, but cheap to add
   to this same script if `libx266`/`vvenc` becomes available via FFmpeg on
   this host.
