# BP24 — the paired ladder

What this session set out to do (`plans/prompts/next-session.md`): the
`WireCost` honesty pass, settle `actor_reference`, and run `PLAN.md` §7 P0
items 2 and 3 as paired BD-rate curves with their bounds written first.

Read `plans/BP24-findings.md` before quoting any number here. Findings 8-14 were
added by this session, and §14 corrects §12.

---

## 1. The `WireCost` honesty pass — done

`exact` meant "follows from declared parameters and a declared quantization,
rather than from a model of the encoder". That was unambiguous only while
nothing in the project ran an encoder. Once BP24 started coding real payloads it
sat `True` on top of a `basis` describing an in-memory array, and *two* separate
mechanisms were each deciding whether a byte count was a bitstream: the flag,
and a hardcoded `raw_parts` list in `ledger_from_bag`.

**One meaning now**, written into `src/contracts/objectstream.py`:

> `exact=True` — this many bytes are transmitted. Either a **measured
> bitstream** an encoder returned, or a **packed payload at a declared
> quantization** sent verbatim because no further coding step is configured for
> it. `exact=False` — the number is a **stand-in** for a cost nobody has
> measured.

The test to apply is *would these bytes travel, as they are?* A dense int16
residual whose blocks the gate zeroed would not: a codec is supposed to run on
it next. A float16 embedding would; there is no next step.

Both residual paths in `src/pipeline/residual/signal.py` are now `exact=False`
with a basis that says `pre-codec, not a bitstream`. The absent path stays
`exact=True` at zero bytes — sending nothing is a measurement, and marking it a
stand-in would fire the guard on the one corner with nothing to hide.

`__add__` already conjoined the flag, which is the property that matters: a sum
containing a stand-in stays a stand-in, so an uncoded part cannot be laundered
into a total that calls itself a rate.

## 2. `actor_reference` — settled with evidence, and it clears

BP24 listed it in `raw_parts` unconditionally, for a stated reason: appearance
reported a measured size and nobody had shown it was a transmitted one. The
handoff was explicit that clearing it without evidence would let the ledger
silently regain a raw part.

Driven per backend rather than argued from the code — `python -m
experiments.tier.appearance_cost`, results in
`outputs/bp24-ladder/appearance-cost.json`:

| backend | payload | declared == buffer | size vs. quality knob | verdict |
|---|---|---|---|---|
| `compressed-image` | 4,746 B JPEG | yes | 1,448 / 2,020 / 7,732 B at q20 / q60 / q95 | **coded** |
| `image-embedding` | 204 B float16 | yes | no knob | **packed** |
| `diffusion-latent` | 3,072 B float16 | yes | no knob | **packed** |

The JPEG payload also decodes back to the crop at MAE 2.83, so those bytes carry
the appearance rather than merely being counted. All three tier configs use
`compressed-image`.

All three are wire costs, so `actor_reference` clears — but **from a flag the
appearance stage sets, not a rule in the ledger**. `_encoded_cost` reads the
descriptor's own `WireCost`, and `_actor_bytes_exact` defaults to *false* when a
payload does not state `exact`, so a backend added later withholds the ratio
until someone checks it. That is the property the handoff asked for.

**Packed is not coded**, and the distinction is recorded rather than collapsed.
A float16 latent has had no entropy coder applied. It is an honest transmitted
size and an over-count of a coded one — the safe direction for a compression
claim — but a table putting a JPEG appearance beside a latent one is comparing a
bitstream against an array.

## 3. Three defects found on the way, all fixed

**`RunResult.frames` was not the delivered clip** (findings §8). The runner
builds `frames` from the residual *as the residual stage produced it*; since
BP24, `make_codec` round-trips that payload through `residual.codec` and the
rebuilt clip is what transport delivers and what `sizes` costs. Pairing a coded
rate with `frames` is findings §4 — and the two clips differ by exactly the
residual's coding loss, so a ladder sweeping that rung would have produced a
smooth, monotone, entirely fictional curve. `RunResult.delivered_frames` now
exists; the ladder uses it. No published number was affected: BP23's table reads
`delivered_quality`, which was always scored on the delivered clip.

**`bd_rate`'s overlap guard could not see a flat curve** (findings §2, carried
forward). The guard was a *proportion* of the shorter curve's span, and two flat
curves overlap perfectly. `MIN_QUALITY_SPAN_DB = 3.0` and `DegenerateCurveError`
close it; the new error subclasses `InsufficientOverlapError` so callers already
declining on a bad overlap decline here too rather than crashing. A non-PSNR
metric must state its own floor — the function refuses rather than applying a dB
number to LPIPS. It earned its place on the first real run, refusing the curve
the next defect produced.

**The decode step re-encoded** (findings §14). `_decode_command` named no
`-c:v`, so ffmpeg picked the muxer's default encoder — rawvideo for a `.y4m`,
which is why `coded_curve` was always fine, and **libx264 at its own default CRF
for a `.mkv`**, which is what `coded_roundtrip` writes. Every frame that function
returned had been through the rung's codec and then through x264. Measured on an
av1 anchor over QP 15 to 55: the rate fell tenfold, from 851,572 to 85,995
bytes, and the Y-PSNR moved 2.77 dB, flat at the fine end.

This one reaches past the ladder. `_coded_residual` round-trips the residual
through `coded_roundtrip`, so PointStream's **delivered pixels** carried an extra
x264 pass on their correction term — a corrupted pipeline output, not only a
corrupted number — and BP24's residual round-trip correlations were measured
through it. `coded_roundtrip` was built on findings §4's principle, returning
cost and reconstruction together so a caller could not separate them, and it did
that faithfully while returning a reconstruction from the wrong operating point.
Returning both halves guarantees they come from the same call; it does not
guarantee they come from the same codec. What catches that is the ordinary
check — a coarser rung must come back visibly worse — now a required-behaviour
test.

## 4. The ladder

Design, per `plans/BP24-findings.md` §1: for codec X, measure X coding the
source and PointStream using X for its coded components, **same preset, same
rate control, same pixel format**, and take BD-rate between the two curves. Both
arms are built from one `EncodeRequest` per rung, so "same preset" is enforced
by construction rather than by two config files agreeing.

Two things this deliberately does not do:

- **It does not rank the per-codec gains against each other.** The presets are
  not equal effort across codecs (findings §1), so ordering the magnitudes would
  be measuring the presets.
- **It does not compare single-QP totals.** A QP is a knob, not a quality level.

Bounds written before the first encode:
`outputs/bp24-ladder/bounds-before-run.json`. The alarms in that file are also
evaluated in code (`check_bounds`) and written into every result, because a
bound that only exists in a file next to the result is the one that gets skipped
when the number is exciting.

Three axes are available, and the default changed once the first run showed
which of them moves PointStream's rate:

- `--sweep payload` (default) — plate quality and residual rate together.
  **P0 item 2.**
- `--sweep coarseness` — `coarseness_ladder()`, whose rungs bundle the codec
  rate with the block gate's size and threshold and the background downscale.
  **P0 item 3.**
- `--sweep qp` — the residual's rate alone, the plate held fixed. Kept because
  it is the axis that showed the plate dominates, not because it produces a
  usable curve on this content.

The clip axis is measured, not assumed
(`outputs/bp24-ladder/motion-survey.json`, findings §11): every BP24 ratio was
taken on `alcaraz_highlights/scene_000`, which is the **most static** of the
eight cached windows — 0.33 grey levels between consecutive frames against 7.70
for `federer_djokovic/scene_003`.

**BD-rate is taken on BT.601 Y-PSNR**, the conventional axis, with RGB-PSNR
recorded per rung so the chroma cost stays visible. Note the correction in
findings §14: the first reason given for that choice — that RGB-PSNR is capped
by the 4:2:0 round-trip — was wrong. The cap was the re-encoding decode. The
choice of axis stands on convention, not on that argument.

**A note on the rung, which is a finding in its own right** (findings §13). The
first paired run swept `residual.rate` alone and moved PointStream's total by 6%,
because the plate was 93% of the payload and does not move with that knob. The
default sweep now moves `background.jpeg_quality` and the residual's rate
together. Two consequences: P0 item 3 as written sweeps ~2% of the payload and is
close to unanswerable in isolation, and the plate-is-the-first-frame stub is not
a limitation of the background work but **the single largest lever on the rate**.

### Results

#### P0 item 2 — PointStream against av1, paired, low motion

`outputs/bp24-ladder/av1-payload-lowmotion.json`.
`alcaraz_highlights/scene_000`, 8 frames, 3840x2160, source 199,065,600 B,
inter-frame MAD 0.33 (the most static of the eight cached windows). av1,
preset 10, QP, yuv420p — **on both arms**.

| arm | rung | coded bytes | Y-PSNR | RGB-PSNR |
|---|---|---:|---:|---:|
| av1 on source | QP 55 | 85,995 | 39.45 | — |
| av1 on source | QP 45 | 156,710 | 41.65 | — |
| av1 on source | QP 35 | 289,198 | 42.97 | — |
| av1 on source | QP 25 | 526,104 | 43.72 | — |
| av1 on source | QP 15 | 851,572 | 44.02 | — |
| PointStream via av1 | jpeg30 / qp55 | 318,077 | 39.21 | 36.96 |
| PointStream via av1 | jpeg50 / qp46 | 390,889 | 41.45 | 39.20 |
| PointStream via av1 | jpeg75 / qp38 | 525,462 | 43.59 | 41.66 |
| PointStream via av1 | jpeg90 / qp28 | 808,573 | 45.39 | 43.42 |
| PointStream via av1 | jpeg98 / qp18 | 1,548,393 | 46.55 | 44.56 |

> **BD-rate +116.8%** over 39.45–44.02 dB, overlap fraction 1.00, BD-quality
> −0.49 dB. **PointStream costs 2.17x the rate of av1 alone at equal quality.**

No alarms fired. Both arms are monotone in rate and in quality; every
PointStream rung reported `is_rate: true` with an empty `raw_parts`; no rung was
excluded and nothing failed. The overlap is 4.57 dB, comfortably past the new
3 dB floor.

**Inside the bounds, and in the direction they predicted.** The pre-run bounds
put the plausible range at [−60%, +1500%] and said in as many words that
PointStream was expected to lose, because generation is off and the fixed plate
cost does not amortise over eight frames. The revised bound written after the
smoke run, [−85%, +400%], also holds. +116.8% is a finding, not an alarm.

**Where the bytes go**, at every rung:

| rung | plate | residual | appearance |
|---|---:|---:|---:|
| jpeg30 / qp55 | 89% | 3% | 7% |
| jpeg50 / qp46 | 89% | 5% | 6% |
| jpeg75 / qp38 | 88% | 7% | 4% |
| jpeg90 / qp28 | 88% | 9% | 3% |
| jpeg98 / qp18 | 91% | 7% | 1% |

The plate is 88–91% of the payload at every operating point. PointStream is
losing to av1 by sending one still image expensively, not by sending a residual
expensively.

**What this is not.** It is not a statement about PointStream's architecture in
general. Generation is off in every tier config, so the arm measured here is
*plate plus pasted crops plus a corrective residual*, with no generative
decoder. The plate is a single source frame, not a stitched panorama. And eight
frames is the least favourable amortisation a fixed plate cost can get. All
three push in the same direction, and the honest reading is: **as configured
today, on this content, PointStream is a more expensive way to send a video than
the codec it is built on.**

#### P0 item 3 — the residual-coarseness curve, low motion

`outputs/bp24-ladder/av1-coarseness-lowmotion.json`. Same clip, same anchor
(the five anchor rungs reproduced to the byte and to two decimal places, which
is a reproducibility check worth having).

| rung | coded bytes | Y-PSNR | plate | residual |
|---|---:|---:|---:|---:|
| absent *(control)* | 487,643 | 35.37 | 463,334 | 0 |
| coarse | 491,977 | 40.77 | 463,334 | 4,334 |
| medium | 508,413 | 43.33 | 463,334 | 20,770 |
| fine | 812,971 | 46.11 | 463,334 | 325,328 |
| lossless *(excluded)* | 398,618,843 | ∞ | 463,334 | 398,155,509 |

> **BD-rate +161.5%** over 39.45–44.02 dB, overlap fraction 1.00.

**The residual is the cheap part, and it is very good value.** Against the
unaided control:

| from absent to | extra bytes | extra rate | quality gained |
|---|---:|---:|---:|
| coarse | +4,334 | +0.9% | **+5.40 dB** |
| medium | +20,770 | +4.3% | **+7.96 dB** |
| fine | +325,328 | +66.7% | +10.74 dB |

A residual costing under one percent of the payload buys five and a half dB.
That is the clearest positive result in this work, and it sharpens rather than
softens the headline: PointStream's rate problem is **entirely the plate**, and
the component the architecture argues hardest for is the one earning its bytes.

**The control ran in the same session, as it must.** The unaided corner — plate
plus pasted crops, no residual — is 487,643 B at 35.37 dB, against av1's
85,995 B at 39.45 dB. The plate alone is already 5.7x the rate for 4 dB less
quality, before the residual is asked for anything.

**Three alarms fired, and two of them are the guard working.**

- `lossless` was **excluded from the curve**: `_coded_residual` returns `None`
  for the lossless variant, so its residual stays a dense int16 array — 398 MB,
  twice the source — and the ledger correctly withheld the ratio. That rung is
  the ceiling calibration `coarseness_ladder()` says it is, not an operating
  point, and it never reached the fit.
- The same rung tripped the "residual.codec did not run" alarm, which is exactly
  what happened.
- The third was a **false positive**: `absent` also reports an infinite
  pre-codec-versus-delivered gap, because there is no residual to code. The
  check now skips a rung that transmits no residual. The stored alarm text in
  this file and in the two later axes predates that fix.

#### The motion axis — the findings §7 re-measure

`outputs/bp24-ladder/av1-payload-highmotion.json`. `federer_djokovic/scene_003`,
8 frames, 3840x2160, **inter-frame MAD 7.70** against 0.33 for the clip above —
the most dynamic of the eight cached windows against the most static.

| arm | rung | coded bytes | Y-PSNR |
|---|---|---:|---:|
| av1 on source | QP 55 | 129,393 | 38.03 |
| av1 on source | QP 45 | 242,897 | 40.23 |
| av1 on source | QP 35 | 472,874 | 41.86 |
| av1 on source | QP 25 | 892,919 | 42.83 |
| av1 on source | QP 15 | 1,458,945 | 43.36 |
| PointStream via av1 | jpeg30 / qp55 | 666,124 | 29.80 |
| PointStream via av1 | jpeg50 / qp46 | 1,065,429 | 30.64 |
| PointStream via av1 | jpeg75 / qp38 | 1,667,807 | 30.88 |
| PointStream via av1 | jpeg90 / qp28 | 2,852,672 | 30.98 |
| PointStream via av1 | jpeg98 / qp18 | 4,711,054 | 31.00 |

> **No BD-rate.** The curves do not overlap at all: PointStream's *best* rung is
> 31.00 dB and av1's *worst* is 38.03 dB. `compare_rd_curves` refused, which is
> the correct answer and a stronger statement than a number would have been.

**PointStream saturates at 31 dB and stops.** Seven times the rate — 666 KB to
4.7 MB — buys **1.20 dB**. At 4.7 MB, which is 36x av1's cheapest rung, it is
still 7 dB below what av1 delivers for 129 KB. The quality is not limited by the
rate; it is limited by the plate, which is the first source frame of a scene
that has moved 7.7 grey levels per frame away from it. Findings §7 predicted
"far worse" on high motion. This is worse than far worse: the comparison stops
being a comparison.

**One alarm fired, and its bound was derived on the wrong clip.**

> pointstream at 4 delivered 29.80 dB, below the unaided reconstruction's
> neighbourhood.

That floor was 30 dB, taken from BP23's unaided reconstruction of **34.88 dB** —
measured on the *static* clip. On a clip 23x more dynamic the unaided
reconstruction is naturally far worse, so the floor was carried across a
condition it was never derived under.

**Closed by measuring the control on this clip**
(`outputs/bp24-ladder/av1-coarseness-highmotion.json`):

| rung | coded bytes | Y-PSNR | gain over unaided |
|---|---:|---:|---:|
| absent *(control)* | 554,215 | **18.36** | — |
| coarse | 688,553 | 26.26 | **+7.90 dB** |
| medium | 1,246,979 | 30.69 | +12.33 dB |
| fine | 2,373,711 | 33.19 | **+14.83 dB** |
| lossless *(excluded)* | 398,685,415 | ∞ | — |

The unaided reconstruction here is **18.36 dB**, not 34.88. So the residual is
not damaging anything — it is adding up to 14.8 dB, more than twice what it adds
on the static clip. The alarm was correct to fire on a rung below its floor and
the floor was the thing that was wrong; recording *why* it was wrong is the
point, because a bound that fires for the wrong reason is worth as much as one
that fires for the right one provided the reason is written down.

Two further readings from that table:

- **The coarseness knobs matter far more on high motion than the plate's
  quality does.** The payload sweep held the tier's medium coarseness and topped
  out at 31.00 dB for 4.7 MB; the coarseness sweep, which drops the block gate
  and the background downscale, reaches 33.19 dB for 2.4 MB — better quality for
  half the rate. The rung that moves this clip is the residual's *resolution*,
  not the plate's quality.
- **It still does not overlap the anchor.** 33.19 dB against av1's worst of
  38.03. `compare_rd_curves` refused again.

#### Remaining axes

<!-- RESULTS -->

## 5. What is still open

- The plate is still the first source frame, not a stitched panorama
  (findings §6), so `background.method` selects a transmission strategy over one
  frame. Any background saving quoted from this work must say so.
- Generation is off in all three tier configs, so this measures
  PointStream-as-codec, not PointStream with a generative decoder.
- The lossless coarseness rung cannot be a rate point: `_coded_residual` returns
  `None` for the lossless variant, so its residual stays an array size and the
  ledger correctly withholds the ratio. It is a ceiling calibration, which is
  what `coarseness_ladder()` already says it is.
