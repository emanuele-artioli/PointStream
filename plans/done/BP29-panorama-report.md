# BP29 stream D — the panorama the runner never called

**What changed:** `build_plate` has existed since the rewrite and
`src/runner/stages.py` never called it. `background.method` therefore chose a
*transmission strategy* over the **first source frame**, and a panorama's whole
argument — amortising one background across the clip — had never been available
(`plans/BP24-findings.md` §6 named this as the standing stub). `make_background`
now stitches the chunk.

**What it is worth, in one line:** on a clip whose camera barely moves the trade
is neutral, as predicted. On a moving clip the residual falls to **0.22x** and
delivered Y-PSNR rises **4.9–6.2 dB**, and the plate does **not** get bigger —
which is the opposite of what the brief and the bounds expected, and is the part
that needed the most checking.

Everything here is **one clip per motion regime, one chunk of eight frames, two
tier configs**. It is not a corpus and it cannot support a claim about content
in general. Eight frames is also the *least* favourable amortisation a fixed
plate cost can get, so a panorama is being judged on its worst case.

Bounds written before the first encode: `outputs/bp29-panorama/bounds-before-run.json`.
Numbers: `outputs/bp29-panorama/report.json`. Instrument checks:
`stitcher-probe.json`, `registration-probe.json`, `alignment-probe.json`,
`estimator-diagnosis.json`, `motion-model-comparison.json`, and the scripts that
produced them under `probes/`.

---

## 1. The instrument was wrong first, and the static clip is what proved it

Before any arm was run, the stitcher was driven on both clips and asked whether
it moves what it should and leaves alone what it should not.

**On `alcaraz_highlights/scene_000` it leaves everything alone.** Frame centres
move 0.00–0.02 px over eight frames, the canvas grows 3841x2161 against
3840x2160, and warping changes the frame-to-frame error by less than 0.01 grey
levels. A deliberately wrong 40 px shift scores 12.65 against the estimate's
0.92, so the measurement can tell aligned from misaligned. The brief's alarm —
*if the static clip shows a large change, the stitcher is moving something it
should not* — **did not fire, and it was capable of firing.**

**On `federer_djokovic/scene_003` it moved things too little.** The check that
caught it was cheap and worth repeating elsewhere: sweep a pure translation and
see whether anything beats the fitted model. A flat 40 px shift aligned the last
frame at MAD **6.72**; the fitted homography managed **8.37**. The median tracked
point had moved **37.7 px** while the fit moved the frame centre **21.2 px** — so
roughly half the pan was being spent on a spurious ~0.2%-per-frame zoom. Frame 6
also came back at 29.5 px and frame 7 at 21.2 px, non-monotone on a steady pan.

The cause is that a tennis broadcast is *not* a plane — court, players and
stands sit at different depths — so no homography fits every tracked point, and
a loose RANSAC threshold lets the leftovers be absorbed as scale. Candidates,
scored by mean background alignment error over frames 1–7 with the players
excluded (`motion-model-comparison.json`):

| motion model | dynamic clip | static clip |
|---|---:|---:|
| identity (no registration) | 12.32 | 0.411 |
| homography, RANSAC 3.0 (what shipped) | 4.73 | 0.413 |
| **homography, RANSAC 1.0, 5000 iters** | **3.24** | 0.411 |
| affine, 6 DOF | 6.37 | 0.412 |
| similarity, 4 DOF | 4.08 | 0.411 |

So the fault was the **threshold, not the model class**: neither lower-DOF model
beats the tightened homography. On the static clip every candidate ties with
identity, which is the control saying this is fitting real camera motion and not
fitting noise harder. `src/components/background/plate.py` now uses
`RANSAC_REPROJ_PX = 1.0`, with that table in the source next to the constant.

`tests/components/test_plate_registration.py` pins the property that would have
caught this from the start: a *known* synthetic pan must come back as that pan,
to within a pixel. A fit that recovers half a translation is stable,
monotone-looking and wrong, and only ground truth catches it.

---

## 2. What was wired

`make_background(ctx, *, span=None, register=True)` in `src/runner/stages.py`
now calls `build_plate` over the whole chunk and hands the artifact's
homographies to the `BackgroundModelView`, so reconstruction warps the plate per
frame instead of pasting one frame. `background.method` reaches `build_plate`;
`none` still reaches nothing, because `none` sends nothing.

Two keywords exist for measurement, both defaulting to the shipped behaviour:

- `span=1` reproduces the pre-BP29 plate **exactly** — `build_plate` over one
  frame is an identity warp and a median of one sample — so the control is the
  same code path rather than a second implementation;
- `register=False` composites the span with the homographies forced to
  identity, which separates what camera-motion compensation buys from what a
  temporal median buys.

Player masks the runner already holds now reach the median, so a player who
stands still for part of the chunk is not transmitted as scenery. Masks are
skipped at `span=1`: with one sample there is no other frame to fill a masked
pixel from, so masking would crudely inpaint rather than reveal.

**One gap, stated rather than assumed:** the homographies travel with the
artifact but are not in the ledger. At eight frames they are 8x9 float64 = 576 B
against a plate of roughly half a megabyte — under 0.15% of the payload.

**A second gap, confirming Stream A:** `BackgroundConfig` forwards only `codec`
and `jpeg_quality`. `png_compression`, `roi_crf` and `roi_preset` keep their
constructor defaults on every runner path, so those knobs are unreachable from a
config file today.

---

## 3. The trade

Two tier configs, one chunk of eight 4K frames each, three arms per block.
`sizes.is_rate` is true throughout — every part is a coded bitstream, so
`transport_total` is a rate and the ratios below are rate ratios.

### Static clip — `alcaraz_highlights/scene_000` (inter-frame MAD 0.33)

| tier | arm | plate B | residual B | total B | delivered Y |
|---|---|---:|---:|---:|---:|
| fast | keyframe (control) | 345,947 | 2,280 | 357,732 | 39.16 dB |
| fast | **panorama** | 342,384 | 2,668 | 354,557 | **39.97 dB** |
| fast | median only, no registration | 341,295 | 2,653 | 353,453 | 40.04 dB |
| balanced | keyframe (control) | 463,334 | 20,311 | 507,954 | 43.24 dB |
| balanced | **panorama** | 457,290 | 15,541 | 497,140 | **43.79 dB** |
| balanced | median only, no registration | 456,154 | 15,503 | 495,966 | 43.83 dB |

Plate **0.99x**, total **0.98–0.99x**, delivered Y **+0.55 to +0.82 dB**. Neutral,
which is what the bounds predicted and what a camera that does not move should
give.

The `fast` residual got **worse** — 2,280 → 2,668 B, a ratio of 1.17. It is
inside the bound and it is 0.1% of that arm's payload: at `tier_fast` the
residual is 0.6% of the total, so a 17% swing in it moves nothing, and the total
still fell while quality rose 0.82 dB. It is a rounding-scale effect on the
smallest part of the ledger, not a cost.

### Moving clip — `federer_djokovic/scene_003` (inter-frame MAD 7.70)

| tier | arm | plate B | residual B | total B | delivered Y |
|---|---|---:|---:|---:|---:|
| fast | keyframe (control) | 393,879 | 60,810 | 465,370 | 25.61 dB |
| fast | **panorama** | 391,798 | **13,388** | 415,867 | **31.79 dB** |
| fast | median only, no registration | 328,591 | 52,023 | 391,295 | 28.08 dB |
| balanced | keyframe (control) | 527,273 | 451,835 | 1,006,050 | 30.53 dB |
| balanced | **panorama** | 524,216 | **103,764** | 654,922 | **35.43 dB** |
| balanced | median only, no registration | 447,708 | 339,366 | 814,016 | 33.39 dB |

Plate **0.99x**, residual **0.22x**, total **0.89x** (`fast`) and **0.65x**
(`balanced`), delivered Y **+6.18 dB** and **+4.90 dB**.

**The brief's framing was wrong and should be replaced.** It expected the plate
to get *larger* and the residual smaller, with the finding being whether the
trade pays. There is no trade: the plate did not grow, and the entire gain is
the residual. §4 is why.

---

## 4. Why a plate covering more area does not cost more

The canvas *does* grow on the moving clip — **3891x2171 against 3840x2160**, an
area ratio of **1.0184** — and the plate still codes to 0.6% *fewer* bytes than
one frame. "More coverage for free" is exactly the kind of claim that needs
attributing rather than enjoying, so the same span was encoded four ways
(`plate_bytes_decomposition` in `report.json`), all with the arm's own sidecar:

| | static, jpeg q50 | static, jpeg q75 | moving, jpeg q50 | moving, jpeg q75 |
|---|---:|---:|---:|---:|
| keyframe (one source frame) | 345,947 | 463,334 | 393,879 | 527,273 |
| median, no registration, frame-sized | 341,295 | 456,154 | 328,591 | 447,708 |
| panorama, cropped back to frame size | 340,672 | 455,348 | 382,505 | 511,241 |
| panorama, whole canvas (transmitted) | 342,384 | 457,290 | 391,798 | 524,216 |

On the moving clip at q75: the **extra coverage costs 12,975 B (+2.5%)** and the
**temporal median saves 16,032 B (+3.0%)** by averaging out sensor noise,
compression dither and the masked players. The two nearly cancel, and the 0.6%
net saving is the remainder. Nothing is free; two effects of similar size point
in opposite directions.

The same table also kills a tempting misreading. The **unregistered** median is
by far the cheapest plate on the moving clip — 447,708 B against the panorama's
524,216 — and it is much the worst plate: 33.39 dB against 35.43, with a
residual over three times larger. It is cheap because a median of eight
unregistered frames of a panning camera is *ghosted*, and blur codes well. **A
cheaper plate is not a better plate**, and byte counts alone cannot tell the two
apart.

---

## 5. The control that says registration is doing the work

The panorama does two separable things: it compensates camera motion, and it
averages away whatever differs between frames. `register=False` keeps only the
second, so the win can be attributed instead of assumed.

| clip / tier | residual, median only | residual, registered | Y, median only | Y, registered |
|---|---:|---:|---:|---:|
| static / fast | 2,653 | 2,668 | +0.88 dB | +0.82 dB |
| static / balanced | 15,503 | 15,541 | +0.59 dB | +0.55 dB |
| moving / fast | 52,023 (0.86x) | **13,388 (0.22x)** | +2.47 dB | **+6.18 dB** |
| moving / balanced | 339,366 (0.75x) | **103,764 (0.23x)** | +2.86 dB | **+4.90 dB** |

(Y columns are deltas against that block's keyframe control.)

**On the static clip registration is worth −0.07 and −0.04 dB** — nothing, very
slightly negative because warping resamples. That is the correct null: a camera
that does not move has no motion to compensate, and a knob that "helped" here
would mean the plate was being improved by something other than what it claims.

**On the moving clip registration is worth +3.71 dB and +2.04 dB on top of the
median**, and it cuts the residual to 0.26x and 0.31x of the median-only arm.
The median alone recovers about 40% of the quality gain and 14–25% of the
residual saving; the homographies are the rest.

---

## 6. Bounds audit

Written before the first encode. Three failed; all three failed in the same
direction and for the same reason.

| bound | predicted | measured | verdict |
|---|---|---|---|
| canvas, static | 1.00–1.15 | 1.0007 | held |
| canvas, moving | 1.05–2.50 | **1.0184** | **failed, below** |
| plate bytes, static | 0.85–1.15 | 0.990, 0.987 | held |
| plate bytes, moving | 1.10–2.60, alarm < 1.0 | **0.995, 0.994** | **alarm fired** |
| residual, static | 0.60–1.20, alarm > 1.5 | 1.170, 0.765 | held |
| residual, moving | 0.25–0.90, alarm > 1.05 | 0.220, 0.230 | just below the low edge |
| total payload | 0.90–2.40, alarm < 0.7 on moving | **0.651** | **alarm fired** |
| delivered Y, moving | +0.5 to +8.0 dB | +6.18, +4.90 | held |
| delivered Y, static | −0.5 to +1.5 dB | +0.82, +0.55 | held |
| ledger cross-check | exact | exact, 12/12 | held |

**Why the canvas bound was wrong, which is the root of all three failures.** It
was derived from frame-difference MAD — 13.61 grey levels against the first
frame — treated as a proxy for how far the camera travelled. It is not one. On
4K broadcast content with dense edges, a pan of 21–38 px already produces 13.6
grey levels of difference. The camera moved about **1% of the frame width over
eight frames**, so the union of eight quadrilaterals is 1.8% larger, not 5–150%
larger. The bound reasoned about *content change* where the canvas depends on
*geometric displacement*, and those are only loosely related. This is the same
class of mistake as the BP24 bound derived in the wrong units.

Once the canvas bound is corrected, the plate-bytes alarm follows from it, and
the total-payload alarm follows from both plus the residual falling 4.4x.

**Closing the two alarms rather than explaining them away.** The plate bytes
were re-derived independently for all twelve arms and match the runner's ledger
exactly (`ledger-cross-check.json`). The residual is a real av1 bitstream and
`sizes.is_rate` is true, so nothing raw is hiding in the total. Delivered quality
*rose* by 4.9 dB on the arm whose total fell most, so the total is not falling
because something was dropped. And the effect survives its own null: with
registration off, the residual saving mostly disappears.

**One effect I had not modelled at all** and should have: the temporal median
denoises. That is worth 3.0% of the plate on its own, and it is why the plate
came in slightly *under* one frame rather than slightly over.

---

## 7. Provenance, and four things a reader should not conclude

The clips read were `outputs/bp21-headroom/clips/{alcaraz_highlights/scene_000,
federer_djokovic/scene_003}/window` with masks from
`assets/dataset/<video>/segmentations/<scene>`, both recorded per block in
`report.json`. A **different** video spelled `djokovic_federer` exists in both
trees (inter-frame MAD 0.84) and is not the clip measured here; the directory
actually opened is in the record, so the label is never the evidence.

The `span=1` control's plate is the first source frame **byte for byte**
(`degenerate_control.plate_equals_first_source_frame: true` in every block), and
its `alcaraz_highlights/scene_000` jpeg q75 encode is **463,334 B** — identical
to Stream A's independently measured baseline for that frame. The control is the
pre-BP29 behaviour, not an approximation of it.

**Wall clock is not comparable between arm 0 and the rest.** The first arm of
each process took 843.0 s against 82.8 s for the next arm on the same clip and
tier. The other three keyframe controls took 175.3, 64.8 and 169.3 s against
179.5, 76.5 and 179.8 s for their panorama pairs — the control is level with or
slightly *faster* than its treatment everywhere except the first arm. The 843 s
is this process's one-off import and model-load cost (`torch` alone is ~124 s on
this filesystem), not a property of the arm.

Do not conclude:

1. **That this generalises.** One clip per motion regime, one chunk, eight
   frames, one still-image codec per tier. It is not a corpus.
2. **That eight frames is the operating point that matters.** It is the *least*
   favourable amortisation a fixed plate cost can get. A longer span should help
   the panorama and also grow the canvas, and neither has been measured.
3. **That this closes the ladder's gap.** These are single operating points, not
   paired curves. Nothing here is a BD-rate, and the paired ladder is
   deliberately not re-run in this stream.
4. **That the plate codec question is settled by these numbers.** The sidecar
   route ceilings at about 44.2 dB RGB from a DC shift plus range handling plus
   4:2:0 (Stream A, corroborated by Stream B across three encoders within
   0.22 dB). The `balanced` arms here sit at 41.3–41.8 dB RGB on the static
   clip, close enough to that ceiling that plate-codec comparisons at this
   quality would be reading the ceiling as much as the codec.

## 8. What this leaves open

- **The span is the untested axis.** Everything here is eight frames. The
  panorama's argument gets stronger with a longer span and the canvas grows with
  it; `make_background(span=...)` already takes the knob.
- **The homographies are not in the ledger** (576 B at eight frames, under 0.15%
  of the payload).
- **`background.codec`'s other knobs are unreachable.** `BackgroundConfig`
  forwards only `codec` and `jpeg_quality`; `png_compression`, `roi_crf` and
  `roi_preset` keep constructor defaults on every runner path. Confirms Stream A.
- **The runner's background is still single-chunk.** `panorama-delta` needs the
  previous decoded plate across chunks, and `make_background` holds no state —
  deliberately, since BP30 owns that axis.
