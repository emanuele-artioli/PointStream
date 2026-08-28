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

---

## Added 2026-08-28, while running the ladder

## 8. `RunResult.frames` stopped being the clip the pipeline delivers

Finding §4 says the fix that generalises is to make the API return both halves
together. `coded_roundtrip` does that. The **runner** does not.

`src/runner/run.py` builds `frames = apply_residual(client.frames,
residual.payload)` — the client's reconstruction plus the residual **as the
residual stage produced it**. Since BP24, `make_codec` round-trips that payload
through `residual.codec` and rebuilds from what came back, and *that* clip is
what reaches transport and what `sizes` is the cost of. So `RunResult.frames`
and `RunResult.delivered_quality` describe two different operating points.

No published number is wrong: BP23's table reads `delivered_quality`, which was
always scored on the delivered clip. But the *array* was a trap, and it is the
one a rate ladder reaches for, because `frames` is the obvious attribute and the
delivered array was reachable only through `chunks[i].bag[ART_DELIVERED]`.

**Why it would not have looked like a mistake.** The two clips differ by exactly
the residual's coding loss. A ladder sweeping the residual's rung is sweeping
precisely that difference — so the fictional curve would have been smooth,
monotone, and better than the real one at every rung.

`RunResult.delivered_frames` now exists and the ladder uses it. The general
lesson is narrower than §4's and worth stating separately: **it is not enough
for the component that codes to return both halves. Every layer that carries
the result forward has to keep them together**, and the runner is a layer.

## 9. `actor_reference` is a wire cost on all three backends — but only one is coded

Driven per backend rather than argued from the code
(`outputs/bp24-ladder/appearance-cost.json`, `python -m
experiments.tier.appearance_cost`):

| backend | payload | declared == buffer | moves with quality | verdict |
|---|---|---|---|---|
| `compressed-image` | 4,746 B JPEG | yes | 1,448 / 2,020 / 7,732 B at q20/60/95 | **coded** |
| `image-embedding` | 204 B float16 | yes | n/a — no knob | **packed** |
| `diffusion-latent` | 3,072 B float16 | yes | n/a — no knob | **packed** |

The JPEG payload also decodes back to the crop (MAE 2.83), so those bytes carry
the appearance rather than merely being counted.

All three are **wire costs** — the buffer is what would be transmitted — so
`actor_reference` no longer withholds the ratio. Two things to keep straight:

- **Packed is not coded.** A float16 latent has had no entropy coder applied.
  It is an honest transmitted size and an *over*-count of what a coded one
  would be, which is the safe direction for a compression claim, but a table
  putting a JPEG appearance beside a latent one is comparing a bitstream
  against an array.
- `DiffusionLatent.measured_bytes` was documented as "size after any further
  entropy coding, if applied" while `latent.py` fills it with the length of the
  raw pack. The docstring was the wrong half of that pair and has been fixed.

## 10. QP is the only rate control the whole codec roster accepts

`src/contracts/codecs.py`: `avc` declares CRF, QP, BITRATE and LOSSLESS; `av1`
declares CRF, QP and BITRATE; **`hevc` and `vvc` declare only QP and BITRATE**.
The tier configs all name `rate-control: crf`, which works because they all name
`av1`.

So a ladder spanning the roster has to sweep QP, and a config that swaps
`residual.codec` to `hevc` without also changing `rate_control` will be refused
by `EncodeRequest.validate` rather than silently reinterpreted. Worth knowing
before the ablation lattice tries to vary the codec axis.

## 11. The clip BP24 measured on is the most static of the eight cached

Measured over the first 8 frames of every cached BP21 window
(`outputs/bp24-ladder/motion-survey.json`), as mean absolute difference in grey
levels:

| clip | consecutive frames | vs. the first frame |
|---|---|---|
| `alcaraz_highlights/scene_000` | **0.33** | 0.69 |
| `alcaraz_highlights/scene_010` | 0.47 | 1.31 |
| `sinner_alcaraz/scene_001` | 0.60 | 1.31 |
| `federer_djokovic/scene_001` | 0.77 | 1.59 |
| `djokovic_federer/scene_003` | 0.84 | 2.10 |
| `alcaraz_perricard/scene_002` | 3.17 | 6.23 |
| `djokovic_zverev/scene_002` | 5.91 | 12.35 |
| `federer_djokovic/scene_003` | **7.70** | 13.61 |

`alcaraz_highlights/scene_000` — BP23's clip, and the one every BP24 ratio was
measured on — is the **most static window available**, by a factor of 23 against
the most dynamic one. §7 called both headline ratios "the easy case" on the
evidence of a 2.5%-non-zero residual; this is the same conclusion measured on
the input rather than inferred from the output.

The second column is the one that matters for this architecture, because the
plate the runner transmits is the *first source frame* (§6). A clip whose last
frame differs from its first by 13.6 grey levels has no useful plate at all.
