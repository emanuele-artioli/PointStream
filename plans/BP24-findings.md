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

**What "fair" would have to mean — and why it mostly dissolves.**

The decisive point is that **PointStream's goal is not a codec comparison.** The
claim the paper needs is *PointStream against a codec*, not *codec against
codec*. That reframing removes most of the problem:

> **Pair every arm on one codec at one preset.** For codec X, measure
> (a) X coding the source, and (b) PointStream using X for its coded components
> — same encoder, same preset, same rungs. BD-rate between those two curves *is*
> the PointStream gain, and **the preset cancels**, exactly as it cancels in
> BP21's concentration result.

So option (1) — report the preset, never compare across codecs — is not a
stopgap for the real goal. It is **sufficient and correct** for it. The paper
reports a gain per codec: "PointStream improves BD-rate over AVC by X%, over
HEVC by Y%", each measured with that codec on both arms.

**The one thing that stays off-limits:** ranking those gains against each other,
or saying "PointStream beats VVC but not AV1". Comparing the *magnitudes*
across codecs re-imports the preset unfairness through the back door. State each
gain beside its preset; do not order them.

Cross-codec fairness only becomes necessary if the paper wants a direct
codec-vs-codec line, which it does not need. That decision is parked in
`DEFERRED.md` D-CODEC-PRESETS rather than settled here.

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

## 4. Counting coded bytes while reconstructing from the pre-codec array

**The trap that cost the most time this session, and it is invisible in a test.**

A rate is only a rate-distortion point if the quality was measured on **what the
codec returned**. It is very easy to write code that encodes a payload, records
`len(bitstream)`, and then reconstructs from the array it still has in memory.
Every test passes. The byte count is real. The quality is real. They belong to
**different operating points**, and the resulting RD point is fiction.

BP24 did this once: the first plate commit re-encoded an *already-decoded* plate
to get a size, while reconstruction used those same decoded pixels. Two separate
errors in one place — the size was a second, easier compression of a cleaned-up
image, and nothing tied it to the quality.

**The fix that generalises:** make the API return both halves together.
`coded_roundtrip` returns `(coded_bytes, decoded_frames)` rather than a size, so
a caller cannot take the rate without taking the reconstruction that goes with
it. Shape the interface so the mistake is hard, rather than documenting it.

**Where this still needs applying:** anywhere a component reports a byte count.
Appearance and motion payloads report measured sizes today; whether those are
transmitted costs or array sizes has not been checked with the same care.

## 5. A luma-only encode silently discards colour, and still returns a number

`coded_size` converts to luma before encoding. That is right for a still plate
scored on luma PSNR, and **wrong for anything carrying per-channel information**.
A residual is a correction in R, G and B; encode it through the luma path and two
thirds of it vanishes while the byte count still looks plausible.

Caught by checking per-channel correlation between the original and decoded
residual: **R 0.950, G 0.961, B 0.902** through the colour-preserving path. A
luma-only route would have shown a healthy R and destroyed B, and no byte count
would have hinted at it.

**Rule:** when a payload has channels that mean different things, verify each
channel survives the round-trip, not just the total size.

## 6. The runner's background stage was a stub, and half of it still is

Before BP24, `background()` in `src/runner/stages.py` was `plate=source[0]` with
an identity warp. It never called `BackgroundStrategy`, never ran a sidecar, and
`background.method` / `background.codec` / `background.jpeg_quality` reached
**nothing**. The component layer was correct and careful the whole time —
`transmit()` even documents that the plate handed forward must be the decoded
one — the runner simply was not using it.

`make_background` now binds the configured strategy, so the plate is transmitted
and the view carries the real payload length and the decoded pixels.

**Still a stub, and this is the honest limit of BP24's background work:** the
plate itself is the **first source frame**, not a stitched panorama. So
`background.method` currently selects a *transmission strategy* over a one-frame
plate. Panorama stitching (`build_plate` exists in
`src/components/background/plate.py`) is not wired into the runner. Any result
quoting a background saving must say which of these it measured.

## 7. Two measurements that are the easy case, not the typical one

Both headline ratios from BP24 were taken on favourable material and must be
re-measured before anything quotes them:

- **The residual at 3667×** (9,331,200 B → 2,545 B) was a near-static residual
  against a static plate, only **2.5% non-zero**. High motion will be far worse.
- **Round-trip error was mean 0.094 but max 100.** AV1 at CRF 35 smooths isolated
  large corrections — exactly the pixels a residual exists to fix. Expect the
  residual's rate-quality curve to be steeper than the plate's.

Neither is a reason to distrust the machinery; both are reasons to run the ladder
over real clips at several rungs rather than trusting a single operating point.
