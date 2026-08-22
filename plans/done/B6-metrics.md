# B6 — Quality measurement

**Owns exclusively:** `src/components/metrics/**` and its tests.
**Implements:** `src/contracts/metrics.py` — the registry and tiers are already
there; this stream builds the computation.

## What to build

Metric backends, tiered so development stays fast and expensive metrics appear
only where they earn their place:

| Tier | Metric | When |
|---|---|---|
| Fast | **PSNR** | always on; the development and test default |
| Traditional | **VMAF**, SSIM | headline quality tables |
| Perceptual | **LPIPS** | generated content, where PSNR misleads |
| Temporal | **FVMD** | temporal-coherence claims |

**FVMD, not FVD.** Fréchet Video Motion Distance fits the temporal-coherence
question better. The existing FVD wiring is prior art to read, not to keep.

**LPIPS exists but is wired only into checkpoint evaluation**, never into
pipeline evaluation — a gap that has been mis-reported as closed twice. Wiring it
into the pipeline path is this stream's most load-bearing task.

**Rate-distortion comparison.** Implement **BD-rate** (Bjontegaard delta rate)
and BD-PSNR/BD-VMAF over a pair of rate-distortion curves, plus the overlap
range the integral is defined on. This is the currency every component ablation
and every codec comparison is settled in (`PLAN.md` §5), so it belongs here
rather than in an experiment script. Two configurations never land at the same
bitrate or the same quality, so a point-to-point comparison compares nothing;
the only exception is dominance, where one arm is better on both axes, and a
helper that detects that case is worth having.

## Traps specific to this stream

**Quality is measured in every configuration, without exception.** Not an
optional evaluation step — an architectural requirement. There is no
configuration where correctness can be assumed: the residual always carries some
coarseness, and generative inference is statistical, so encoder-side and
client-side generation are not guaranteed to match. A config that disables all
metrics is rejected, and PSNR is added back and *reported as enforced* rather
than silently.

**Distinguish deterministic from generative comparison.** Deterministic stages —
panorama warping, residual arithmetic — get bit-identity checks. Generative
stages get closeness measurement. Asserting bit-identity on a sampler will fail
for reasons that have nothing to do with correctness.

## Done when

- Every tier computes on the pipeline path, not only on checkpoints.
- A run with no metrics is impossible.
- Ranking code never special-cases direction — higher-better and lower-better are
  declared.
- BD-rate is computed from curves, reports its quality-overlap range, and refuses
  to return a number when the curves barely overlap.
- `ruff`, `mypy`, tests pass; import direction clean.

---

## Delivered — 2026-08-22 — and this stream closed the standing blocker

Landed in `src/components/metrics/`. **All five metrics compute real numbers**,
verified by scoring identical against degraded frames rather than by reading the
code:

| Metric | identical | degraded |
|---|---|---|
| PSNR | `inf` | 22.53 dB |
| SSIM | 1.0 | 0.9885 |
| VMAF | 97.43 | 28.93 |
| LPIPS | 0.0 | 0.00108 |
| FVMD | correctly refuses T=1 — it is a temporal metric | — |

VMAF runs through real libvmaf. This closes the blocker that had stood since
July, when no configuration produced a quality number at all.

**BD-rate landed** in `bd_rate.py` with the pieces `PLAN.md` §5 requires:
`compare_rd_curves`, `RDCurve`, `OperatingPoint`, `MIN_OVERLAP_FRACTION`,
`MIN_POINTS`, and an explicit `InsufficientOverlapError` — so a comparison over
a sliver of shared quality raises rather than quietly returning a number.

**Note on the VMAF reading:** 97.4 rather than 100.0 on identical 64x64 synthetic
frames is expected — the VMAF model is trained on 1080p content and is not exact
at tiny synthetic resolutions. It is not evidence of a wiring fault. Re-check on
real frames at real resolution before treating any deviation as a bug.

**Outstanding:** LPIPS is wired into the metrics registry, but the claim that
matters — LPIPS on the *pipeline* evaluation path — cannot be verified until
Phase C exists. That gap has been mis-reported as closed twice before; it is not
closed yet.
