# BP24 — the paired ladder

What this session set out to do (`plans/prompts/next-session.md`): the
`WireCost` honesty pass, settle `actor_reference`, and run `PLAN.md` §7 P0
items 2 and 3 as paired BD-rate curves with their bounds written first.

Read `plans/BP24-findings.md` before quoting any number here. Findings 8-11 were
added by this session.

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

## 3. Two defects found on the way, both fixed

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
number to LPIPS.

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

Two axes are swept:

- `--sweep qp` — the candidate arm sweeps the codec's rate. **P0 item 2.**
- `--sweep coarseness` — it sweeps `coarseness_ladder()`, whose rungs bundle the
  codec rate with the block gate's size and threshold and the background
  downscale. **P0 item 3.**

The clip axis is measured, not assumed
(`outputs/bp24-ladder/motion-survey.json`, findings §11): every BP24 ratio was
taken on `alcaraz_highlights/scene_000`, which is the **most static** of the
eight cached windows — 0.33 grey levels between consecutive frames against 7.70
for `federer_djokovic/scene_003`.

### Results

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
