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

## 12. RGB-PSNR cannot be the quality axis against a 4:2:0 codec

The first full ladder run was stopped after three anchor points, because those
three points said the axis was wrong. Coding the source with av1, preset 10,
QP, `yuv420p`, on eight 4K frames
(`outputs/bp24-ladder/rgb-axis-saturation.json`):

| QP | coded bytes | RGB-PSNR |
|---:|---:|---:|
| 15 | 851,572 | 40.72 dB |
| 25 | 526,104 | 40.63 dB |
| 35 | 289,198 | 40.26 dB |

**A 2.9x change in rate moved the quality by 0.46 dB.** The arm is capped by the
RGB → 4:2:0 → RGB chroma round-trip, not by the quantizer, so the curve carries
almost no information about the encoder. This is the degenerate shape §2
describes, and the new absolute-span guard refuses it — which is the guard
paying for itself on the first real run after it was written.

**The worse half is that the cap is asymmetric between the arms.** PointStream
delivers a JPEG plate with source crops pasted over it, so most of its pixels
never make the 4:2:0 round-trip at all. Its delivered RGB-PSNR on the same clip
was 42.0-42.6 dB — *above a ceiling the anchor cannot reach at any QP*. The two
curves therefore had no overlapping quality range whatsoever, and any BD-rate
taken across them would have been measuring the colour format rather than the
coding.

BD-rate is now taken on **BT.601 Y-PSNR**, which is what the BD-rate literature
reports anyway, so this is the conventional axis rather than a convenient one.
RGB-PSNR is recorded per rung so the chroma cost stays visible instead of being
quietly dropped.

**The general shape, which is not about colour:** an arm whose quality is capped
by something other than the knob being swept produces a flat curve, and a flat
curve against a non-flat one produces either a refusal or a fiction. Before
sweeping a knob, check that the arm's quality is actually limited by *that* knob
over the range being swept.

## 13. Sweeping the residual's rung does not move PointStream's rate

The first paired run, on four 4K frames of the static clip: over QP 30 to 46
PointStream's transmitted total moved **526,079 → 495,739 B, a span of 6%**,
and its quality moved 0.55 dB. The reason is in the ledger's own breakdown:

| part | bytes at QP 46 | share |
|---|---:|---:|
| panorama (the plate) | 463,334 | 93% |
| actor_reference | 22,542 | 5% |
| residual | 8,871 | 2% |
| metadata | 992 | <1% |

**The plate is the payload.** On static content the residual is single-digit
percent of what PointStream sends, so the residual's rung — the knob `BP24`
built the rate axis around, and the knob `PLAN.md` §7 P0 item 3 names — cannot
produce a rate-distortion curve on its own. Neither can the
residual-coarseness ladder, which moves the same term.

The axis that moves PointStream's rate is **`background.jpeg_quality`**, and a
rung has to move it together with the residual's rate. That is what the shipped
tiers already do (`fast` is jpeg 50 with a coarse residual; `quality` is jpeg 95
with a fine one) — the tier ladder was the right shape all along and nobody had
said why.

Two consequences worth carrying forward:

- **P0 item 3 as written is close to unanswerable in isolation.** A
  residual-coarseness curve on this content sweeps 2% of the payload. The
  honest version of the question is what the residual buys *at a fixed plate
  quality*, which is a quality-per-byte statement, not an RD curve.
- **The panorama stub is now the load-bearing one.** §6 recorded that the plate
  is the first source frame rather than a stitched panorama, filed as a limit on
  the background work. It is more than that: the plate is 93% of the rate, so
  the single largest lever on PointStream's compression is the one component
  still standing in for itself.

## 14. The decode step re-encoded, and it capped every quality it touched

`_decode_command` in `src/components/codec/command.py` named no `-c:v`. ffmpeg
then picks the muxer's default encoder. To a `.y4m` that is rawvideo and
harmless — which is why `coded_curve` was always fine. To a `.mkv` it is
**libx264 at its own default CRF**, so `coded_roundtrip` handed back frames that
had been through the rung's codec *and then* through x264.

Measured on an av1 anchor, 8 real 4K frames, preset 10, QP sweep, Y-PSNR:

| QP | coded bytes | Y-PSNR |
|---:|---:|---:|
| 15 | 851,572 | 41.71 dB |
| 25 | 526,104 | 41.67 dB |
| 35 | 289,198 | 41.43 dB |
| 45 | 156,710 | 40.65 dB |
| 55 | 85,995 | 38.94 dB |

**A tenfold fall in rate moved the quality by 2.77 dB**, and the fine end is
flat: 15 to 25 moved 0.04 dB. The rung reached the encoder — the byte counts
prove that — but the quality never reached the measurement, because a second
encoder was standing between them holding the output near its own ceiling.

This is why finding §12 was wrong about the cause. The RGB numbers in §12 are
real and the reasoning about chroma was plausible, but the ceiling was **not**
4:2:0: switching to Y-PSNR barely moved it (40.72 → 41.71 dB at QP 15), which is
what showed the cap had to be somewhere else. Two hypotheses, one right, and the
thing that separated them was changing the metric and watching the ceiling stay
put. §12's conclusion — take BD-rate on Y-PSNR — stands anyway, because it is
the conventional axis; its stated *reason* was the wrong one.

**What this contaminated.** Everything downstream of `coded_roundtrip`, which
BP24 introduced specifically so a rate and its quality could not be separated:

- Every anchor rung this ladder measured before the fix.
- **The runner's residual.** `_coded_residual` round-trips the residual payload
  through `coded_roundtrip`, so PointStream's *delivered pixels* carried an
  extra x264 pass on their correction term. That is a corrupted pipeline output,
  not just a corrupted measurement.
- BP24's reported residual round-trip correlations (R 0.950 / G 0.961 /
  B 0.902) were measured through it and are pessimistic by an unknown amount.

**The lesson, which is not about ffmpeg.** `coded_roundtrip` was built on
finding §4's principle — return the cost and the reconstruction together so a
caller cannot take one without the other. It did that faithfully and still
returned a reconstruction from the wrong operating point, because the defect was
*inside* the function rather than in how callers used it. Returning both halves
together guarantees they belong to the same call. It does not guarantee they
belong to the same *codec*. The check that would have caught it is the ordinary
one: a coarser rung must come back visibly worse, and that is now a required
behaviour test in `tests/components/test_codec_encode.py`.

## 15. An out-of-range rung is caught only by the encoder refusing it

The roster ladder swept QP 15 to 55. kvazaar takes QP 0 to 51 and refused:

```
Input error: --qp parameter out of range [0..51]
Failed to open encoder.
```

`EncodeRequest.validate` did not catch it. `CodecCapabilities` in
`src/contracts/codecs.py` declares pixel formats and rate-control modes but no
**rate range**, so nothing between the config and the binary knows that a rung
is outside the codec's vocabulary.

The run degraded honestly — the anchor rung was recorded as a failure, the
PointStream rung at the same value could not code its residual so the ledger
withheld the ratio and the rung was excluded from the fit, and the monotonicity
check flagged it twice on the way. **But that honesty is kvazaar's, not ours.**
An encoder that clamped an out-of-range QP to its maximum instead of refusing
would have produced a rung that looks like a rung, at a quantizer nobody asked
for, and the only symptom would have been two rungs landing suspiciously close
together. x264 and vvenc both accepted QP 55 here and produced genuinely
distinct points (24,141 B at 28.18 dB and 8,939 B at 28.10 dB), so nothing was
clamped this time.

Worth adding a declared range to `CodecCapabilities` before the ablation lattice
sweeps the codec axis, where a silently clamped rung would be one row in a table
nobody re-derives.

---

## Added 2026-08-29, probing the plate

## 16. JPEG is the wrong codec for a 4K plate, by a factor of three to four

The plate is 88-91% of PointStream's payload (§13) and it is coded as a JPEG.
Measured on the same still — `alcaraz_highlights/scene_000` frame 0, 4K —
against modern intra coding (`outputs/bp24-ladder/plate-probe.json`,
`python -m experiments.tier.plate_probe`):

| route | knob | bytes | PSNR |
|---|---|---:|---:|
| jpeg | q10 | 199,933 | 31.03 |
| jpeg | q30 | 283,431 | 37.98 |
| jpeg | q50 | 345,558 | 40.04 |
| jpeg | q75 | 461,771 | 42.79 |
| jpeg | q90 | 709,794 | 45.45 |
| **av1** | qp55 | **79,726** | **38.25** |
| av1 | qp45 | 143,925 | 40.83 |
| av1 | qp35 | 253,346 | 42.55 |
| av1 | qp25 | 425,296 | 43.50 |
| **vvc** | qp35 | **68,477** | **38.38** |
| vvc | qp25 | 179,527 | 41.79 |

Read at matched fidelity rather than at matched knob:

- **~38 dB**: JPEG 283,431 B; av1 **79,726 B** (3.6x smaller, 0.3 dB better);
  vvc **68,477 B** (4.1x smaller, 0.4 dB better).
- **~40 dB**: JPEG 345,558 B; av1 143,925 B (2.4x smaller, 0.8 dB better).
- **~42.8 dB**: JPEG 461,771 B; av1 253,346 B (1.8x smaller, 0.24 dB worse).

The gap is widest exactly where the plate wants to operate — cheap. This is a
**factor of two to four on 88-91% of the payload**, for no architectural change:
the plate stays a single still, transmitted once, decoded once.

**It is not even new code.** `src/components/background/sidecar.py` already
offers `roi-video`, a single-frame libx264 encode with `addroi` bit steering, as
a value of `background.codec`. Nothing has ever measured it against `jpeg`,
because `background.codec` reached nothing at all until BP24 wired
`make_background` (§6). An av1 or vvc intra sidecar would be a third backend on
the same interface.

**Why it went unnoticed for so long:** `background.jpeg_quality` is a knob on the
*codec that was already chosen*, so every sweep of it — including BP24's payload
sweep — explored quality within JPEG and never questioned JPEG. A config axis
that offers `{jpeg, png, roi-video}` and is only ever set to `jpeg` is
indistinguishable from a hardcoded constant until somebody drives the others.

## 17. Scenes from one match do not share a background

Tested because it looked like free amortisation: if the camera returns to the
same court view, one plate could serve several scenes, and — unlike a codec,
which must start a fresh intra frame at every cut — PointStream could carry it
across. First frame against first frame, same match
(`outputs/bp24-ladder/plate-probe.json`):

| match | pair | PSNR | mean abs diff |
|---|---|---:|---:|
| `alcaraz_highlights` | scene_000 vs scene_010 | 13.75 dB | 39.39 |
| `federer_djokovic` | scene_001 vs scene_003 | 15.10 dB | 22.97 |

**This first measurement asked the wrong question**, and the correction is the
useful part. It tested whether two plates are *identical*. Nobody proposed
sending one unchanged; the proposal was to send a **residual against the
previous plate**, and a 13.75 dB gap says nothing about what that residual costs
to code — a large but smooth difference can be very cheap. Two follow-ups
settled it properly.

**All four scenes are points.** The dataset's own `scene_metadata.json` labels
them `cluster_point` with confidence 1.000, 1.000, 0.957 and 0.886, so this is
the proposal measured on exactly the content it was proposed for — points of a
match, not replays or interludes.

**Delta coding is dominated on both axes**
(`outputs/bp24-ladder/plate-delta.json`). Coding plate B fresh against coding
`B − A` biased into uint8, same encoder, same QP, quality scored on
`A + decoded delta`:

| pair | QP | fresh | delta | ratio |
|---|---:|---|---|---:|
| alcaraz 000→010 | 35 | 259,211 B @ 42.48 dB | 441,172 B @ 29.76 dB | 1.70x |
| alcaraz 000→010 | 45 | 151,822 B @ 40.81 dB | 253,226 B @ 29.54 dB | 1.67x |
| federer 001→003 | 35 | 292,980 B @ 42.38 dB | 452,786 B @ 29.13 dB | 1.55x |
| federer 001→003 | 45 | 174,615 B @ 40.30 dB | 260,837 B @ 28.89 dB | 1.49x |

**More bytes and 13 dB less quality**, so no rate ladder is needed — the fresh
arm dominates. The reason is mechanical: a difference of this size is not smooth,
it is full of edges, and an edge-dense image is *harder* to code than the
photograph it came from. Delta coding pays only when the reference is close.

**And the gap is content, not camera geometry**
(`outputs/bp24-ladder/plate-register.json`). SIFT found 534 and 1,203 good
matches and RANSAC fitted a homography covering 89% and 97% of the frame — so
the geometry is recoverable — and warping one plate onto the other moved PSNR
only from 13.75 to 14.60 dB, and from 15.10 to 20.01 dB. What is left after the
camera motion is removed is crowd, shadow, scoreboard and player position, and
no warp reaches those.

**The door is closed, now on the right evidence.** One plate cannot serve two
points of a match on this content — not by reuse, not by delta coding, and not
after registration. The idea was sound in principle and the content does not
support it; what makes it worth recording is that the first measurement said the
right thing for the wrong reason, and a reader who checked only that one would
have believed a conclusion that had not been tested.

## 18. Coding the next plate as a P-frame saves 31–53%. §17 measured the wrong mechanism

§17 subtracted two plates pixel by pixel and coded the difference, found it cost
*more* than coding the plate fresh, and closed the door. **The mechanism was
wrong.** Pixel subtraction destroys the spatial correlation a transform coder
depends on; a video codec's inter prediction does block-wise motion search,
which is exactly what a camera that panned between two plates needs. The right
question was never "how big is `B − A`?" but "how big is **B as a P-frame whose
reference is A**?"

Measured (`outputs/bp24-ladder/plate-interframe.json`,
`python -m experiments.tier.plate_interframe`). Frames `[A, B]` encoded as a
two-frame video, per-frame sizes from `ffprobe`, against coding B alone all-intra
at the same encoder and CRF:

| arm | encoder | CRF | frame types | P-frame | fresh intra | ratio |
|---|---|---:|---|---:|---:|---:|
| **control**: consecutive frames, one scene | libx265 | 38 | IP | 5,321 | 160,430 | **0.033** |
| **control** | libaom-av1 | 28 | IP | 11,601 | 758,199 | **0.015** |
| alcaraz 000→010 | libaom-av1 | 28 | IP | 517,226 | 754,741 | **0.685** |
| alcaraz 000→010 | libaom-av1 | 38 | IP | 329,183 | 490,943 | **0.671** |
| alcaraz 000→010 | libx265 | 28 | **II** | 441,593 | 441,591 | 1.000 |
| federer 001→003 | libaom-av1 | 28 | IP | 489,989 | 962,693 | **0.509** |
| federer 001→003 | libaom-av1 | 38 | IP | 294,997 | 627,435 | **0.470** |
| federer 001→003 | libx265 | 38 | IP | 171,388 | 195,687 | 0.876 |

**The control is what makes the arms readable**: two consecutive frames of one
scene cost 1.2–3.3% as a P-frame, so the harness really is measuring inter
prediction.

**Between points of a match, av1 saves 31% to 53%.** The same plates that
subtraction made *more* expensive become substantially cheaper once the codec is
allowed to use motion vectors. §17's conclusion — that one plate cannot help
code another — is **wrong and is retracted**; what is true is the narrower
statement that *pixel subtraction* cannot.

**Two things worth noticing in the table.** libx265 chose to code the second
alcaraz plate as an I-frame (`types=II`) — its own rate-distortion decision said
intra was better, and it was right for that encoder — while av1 found inter
worth 31%. So the saving is codec-dependent and must be measured per codec
rather than assumed. And the saving is *larger* on the pair whose plates are
further apart in PSNR (federer, 15.10 dB, saves more than alcaraz at 13.75 dB),
which is a reminder that PSNR distance does not predict coding distance.

**The reframing that makes this ordinary rather than exotic: the sequence of
per-scene plates is itself a video**, at roughly one frame per point. Coding it
as one needs no new technology — not CMAF, whose fragments are deliberately
independently decodable and therefore the opposite of what is wanted here. It
needs a long GOP, which is what every video stream already is.
`BackgroundConfig.method` already declares `panorama-delta` as a strategy and
nothing implements it; this is what it should be.

**The fairness question this raises, and it is not settled.** Amortising a plate
across scenes must not be given to PointStream alone: an anchor encoding the
same footage can also predict across a scene join. The asymmetry PointStream
might have is that its plates are composited backgrounds, plausibly more similar
to each other than two arbitrary frames at a cut are — but "plausibly" is not a
measurement, and the paired-arm discipline says the anchor gets the same
material.
