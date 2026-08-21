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
- `ruff`, `mypy`, tests pass; import direction clean.
