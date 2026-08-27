# BP24 — findings that outlive this brief

Written 2026-08-27 while binding real encoders into the codec stage. These are
not the BP24 result; they are things measured on the way that change how other
work should be read.

## 1. The codec presets are not equal effort, so cross-codec numbers are unfair

`_PRESETS` in `src/components/codec/measure.py` mirrors
`experiments/headroom/ladder.py`: `avc: veryfast`, `hevc: ultrafast`,
`av1: 10`, `vvc: faster`. Those were chosen so BP21's 4K ladder would finish,
which is a legitimate reason — but they are **not comparable effort across
encoders**.

Measured on four real 960×540 frames, QPs 28/38/48, over a 13 dB overlap:

**HEVC over AVC = −4.2% BD-rate.** The literature expects **30–50%**.

x264 `veryfast` against kvazaar `ultrafast` is not a fair fight, and the
direction of the bias is predictable: it **understates the newer codec**, every
time.

**What this does and does not touch.**

- **Safe: BP21's concentration result.** It compares an original clip against a
  plate-filled clip *on the same codec at the same preset*, so the preset
  cancels. The 15–19× concentration stands.
- **Safe: BP21's retracted VVC gap.** It was already withdrawn as an
  operating-point confound; this is a second, independent reason the same
  comparison could not have been trusted.
- **Not safe: any future codec-vs-codec claim** built on these presets,
  including the P0 item 2 ladder. That is the one this brief hands forward.

**What "fair" would have to mean.** Not settled, and it needs deciding before
P0 item 2 quotes a cross-codec number. The options, cheapest first:

1. **Report the preset with every number and never compare across codecs.**
   Costs nothing, and makes within-codec results (concentration, residual
   coarseness, the ablation lattice) fully usable. It does not give the paper a
   "PointStream vs HEVC" line.
2. **Match wall-clock encode time per rung**, choosing each encoder's preset to
   hit a common time budget. Defensible for a systems paper, and it makes the
   real-time argument concrete. Needs a calibration pass per codec per
   resolution.
3. **Use each encoder's own default/`medium` tier.** Cheap and conventional,
   but "medium" is not a shared unit either — it is only less obviously unfair
   than `ultrafast` vs `veryfast`.
4. **Reference software at fixed CTC-style configurations** (HM, VTM). This is
   what the codec literature actually does and is the only genuinely comparable
   option. Far slower, and a reviewer will expect it if the paper claims a
   codec-vs-codec BD-rate at all.

**Recommendation:** (1) immediately, so nothing currently measured is quoted
unfairly, and (2) or (4) decided explicitly before P0 item 2 publishes a
cross-codec figure. Whichever is chosen, **state the presets beside the number**.

## 2. `bd_rate`'s overlap guard is relative, so it cannot catch a flat curve

`src/components/metrics/bd_rate.py` refuses to return a number when curves
"barely overlap", which is the right instinct. But the guard is
`overlap_fraction` — a *proportion* of the quality range.

A degenerate curve defeats it. Encoding synthetic gradient-plus-noise gave a
quality span of **0.5 dB across QP 32→46** (the encoder discards incompressible
noise at every QP, so PSNR saturates). Both curves were flat, so they overlapped
almost completely: `overlap_fraction = 1.0`, and `compare_rd_curves` returned a
confident-looking BD-rate of −0.88 over a range where nothing was resolved.

**A relative guard cannot see an absolutely tiny span.** It needs an absolute
floor as well — a curve spanning under roughly 3–6 dB should refuse, whatever
its overlap fraction. `InsufficientOverlapError` already exists and is the right
place to raise from.

This did not corrupt any published number: it was caught on a synthetic anchor
during BP24. But it is exactly the shape of failure this project has had twice
before — an instrument returning a confident number over a range where it
resolves nothing.

## 3. A synthetic anchor can hide a working instrument as easily as a broken one

The flat curve above looked like broken code. It was not: the same
`coded_curve` on four real frames produced textbook curves — 42.80 → 35.02 →
28.05 dB, 14.75 dB span, monotone in rate and quality.

**The contrast is what established the instrument**, not either run alone. Noise
is incompressible, so a noise-based anchor cannot exercise a rate-quality sweep
at all; it saturates and reports nonsense in both directions. When calibrating
anything rate-related, the anchor has to be compressible the way real content
is.
