# BP29 §1.1 — the plate's codec knob: what it costs, and what it buys

Wave 8, Stream A. Branch `wave8/plate-codec-sweep`. Module
`experiments/tier/plate_codec_sweep.py`; results under
`outputs/bp29-plate-codec/` (`plate-codec-sweep.json`,
`plate-codec-diagnostics.json`, `plate-codec-ceiling.json`, and the run logs).

Bounds were written before the first encode:
`outputs/bp29-plate-codec/bounds-before-run.json`.

**The plate here is the clip's first frame**, because that is what
`make_background` transmits today — a single source frame, not a stitched
panorama (`plans/done/BP24-findings.md` §6). `alcaraz_highlights/scene_000`,
frame 38, 3840x2160.

**No BD-rate, and no paired ladder.** Four streams were moving the plate at
once; this is what the plate costs under each codec, and nothing more.

## In one paragraph

All three codecs run, and all three were proved to run rather than assumed to.
**Which is cheapest depends entirely on the operating point**: below about 40 dB
`roi-video` (libx264 intra) beats `jpeg` — 1.21x at 38 dB, 1.58x at the crf 30
point config actually pins it to, and about 3x once the sidecar's ROI steering
is switched off — while above 40 dB `jpeg` wins, by 1.59x at 42.8 dB Y. **At the
rung the BP24 ladder uses, `jpeg` is the cheaper codec**, and `roi-video` cannot
reach that fidelity at any bitrate: the route saturates at 44.2 dB RGB /
44.0 dB Y, 4.61 dB of that lost to range handling and 3.54 dB to 4:2:0, neither
of which is the codec. `png` is 7.1x the jpeg75 plate and is not a candidate.
The end-to-end arms — total payload and delivered Y-PSNR — did not run: all four
failed on a repo-wide weights-path fault being fixed elsewhere (§7).

---

## 1. All three codecs reach the encoder

The failure this stream existed to catch is a config value that names a codec
and silently gets another one. It did not happen, and that is a measurement
rather than an assumption:

- every payload was identified from **its own first bytes** — JPEG SOI, PNG
  signature, MP4 `ftyp` — and each matched the codec that was asked for;
- every `roi-video` payload was **`ffprobe`d**: `codec_name: h264` on all
  fourteen of them (seven sweep rungs, seven control-grid points), at `yuv420p`,
  one frame;
- PNG decoded **bit-identical** to the plate at every compression level, which
  is what lossless means;
- each codec's byte count moves monotonically with its own knob, so the knob
  reaches the encoder.

ffmpeg and ffprobe resolved by path: `/opt/local/bin/ffmpeg`, `n7.1.1-56`.

## 2. The plate sweep

RGB is one PSNR over the whole plate's MSE; Y is the same on BT.601 luma, which
is the axis the BP24 ladder's BD-rate is taken on. Both are reported because
they disagree about which codec wins, and by how much.

| codec | knob | bytes | RGB-PSNR | Y-PSNR |
|---|---|---:|---:|---:|
| jpeg | q10 | 199,446 | 31.49 | 34.39 |
| jpeg | q30 | 283,483 | 37.67 | 40.10 |
| jpeg | q50 | 345,947 | 40.04 | 42.64 |
| jpeg | **q75** | **463,334** | **42.96** | **45.44** |
| jpeg | q90 | 713,320 | 45.47 | 48.41 |
| jpeg | q95 | 1,026,154 | 46.83 | 50.03 |
| png | z0 | 24,925,484 | ∞ | ∞ |
| png | **z3** | **3,272,798** | ∞ | ∞ |
| png | z6 | 2,928,924 | ∞ | ∞ |
| png | z9 | 2,810,300 | ∞ | ∞ |
| roi-video | crf12 | 593,242 | 42.56 | 42.99 |
| roi-video | crf18 | 389,340 | 40.77 | 41.64 |
| roi-video | crf23 | 277,261 | 38.79 | 39.84 |
| roi-video | crf28 | 183,191 | 36.41 | 37.38 |
| roi-video | **crf30** | **153,202** | **35.42** | **36.33** |
| roi-video | crf35 | 98,625 | 32.71 | 33.34 |
| roi-video | crf40 | 62,181 | 29.85 | 30.26 |

Bold rows are the operating points a config actually selects (§4). `jpeg:75`
reproduced the BP24 ladder's reference rung to the byte — 463,334 B — which is
what makes everything else here comparable to that ladder.

## 3. "Which is cheapest" has no single answer: the ranking inverts at ~40 dB

Read at matched **fidelity**, never at matched knob (log-linear interpolation on
each codec's own curve; no extrapolation):

| target | jpeg | roi-video | who wins |
|---|---:|---:|---|
| 38.0 dB RGB | 291,446 | 241,614 | roi-video, **1.21x** |
| 40.0 dB RGB | 344,727 | 341,304 | tie (1.01x) |
| 42.8 dB RGB | 455,946 | *out of range* | jpeg — roi-video cannot reach it |
| 38.0 dB Y | 249,128 | 203,267 | roi-video, **1.23x** |
| 40.0 dB Y | 281,764 | 285,927 | tie (0.99x) |
| 42.8 dB Y | 351,883 | 559,644 | jpeg, **1.59x** |

**The crossover sits at about 40 dB on both axes.** Below it x264 intra is
cheaper; above it JPEG is, and past ~44 dB x264 as configured cannot go at all
(§5).

**At the operating point the runner actually uses**, the two were also measured
directly rather than interpolated: `roi-video` crf30 costs **153,202 B at
35.42 dB** RGB, and a JPEG matched to that fidelity by bisection — q19 — costs
**241,909 B at 35.28 dB**. So **roi-video is 1.58x cheaper at 0.14 dB better
quality** there, and §5 shows that most of the rest of x264's advantage is being
given away by the sidecar's ROI steering.

**At the operating point the ladder uses**, JPEG wins. The BP24 payload ladder's
rungs are jpeg 30/50/75/90/98, i.e. plate fidelities of 37.7, 40.0, 43.0, 45.5
and ~47.5 dB RGB. Four of those five sit at or above the crossover, and the
reference rung (jpeg75, 42.96 dB RGB / 45.44 dB Y) is above the roi-video
route's ceiling entirely. **Swapping the plate to `roi-video` would not have
made the ladder cheaper at its reference rung; it would have made it worse and
capped its quality.** The lever from `plans/done/BP24-findings.md` §16 — a factor of
2 to 4 — is an **av1/vvc-intra** result, and this stream does not reproduce it
with x264: the shipped `roi-video` route is not a stand-in for that lever.

**png is never a candidate**, as expected: 3,272,798 B is 7.1x the jpeg75 plate
and 3.8x what av1 charged for the *entire eight-frame clip* at QP 15
(851,572 B). It stays in the sweep as a running-check on the axis.

## 4. The config axis selects one fixed operating point per codec

`BackgroundConfig` carries `method`, `codec` and `jpeg_quality` — nothing else —
and `src/components/background/strategy.py:bind` forwards only `codec`,
`jpeg_quality` and `domain`. So `png_compression`, `roi_crf` and `roi_preset`
keep their constructor defaults on **every** runner path:

| codec in config | what actually runs | plate fidelity |
|---|---|---:|
| `jpeg` | quality from `background.jpeg_quality` | tunable, 31-47 dB |
| `png` | compression level 3 | lossless |
| `roi-video` | libx264, **crf 30**, `veryfast`, 4 `addroi` regions | 35.4 dB RGB |

This is a real constraint on the sweep's reach and on the axis itself. A config
can ask for `roi-video`, but it cannot ask for `roi-video at the fidelity the
ladder operates at` — that rung is not expressible. Anyone comparing plate
codecs end to end is therefore comparing a tunable JPEG against two fixed
points, which is a different experiment from comparing codecs.

## 5. What the `roi-video` route gives away before the codec runs

Controls, all libx264, all confirmed `h264` by ffprobe
(`plate-codec-diagnostics.json`):

| crf | preset | addroi | bytes | RGB | Y |
|---:|---|---|---:|---:|---:|
| 0 | veryfast | off | 1,315,933 | **44.22** | **43.98** |
| 12 | veryfast | off | 424,613 | 42.34 | 42.82 |
| 12 | slow | off | 420,540 | 42.56 | 42.91 |
| 12 | veryfast | **on** | 593,242 | 42.56 | 42.99 |
| 18 | slow | off | 240,390 | 40.81 | 41.51 |
| 30 | veryfast | off | 81,690 | 34.78 | 35.69 |
| 30 | slow | off | 82,533 | 35.07 | 35.82 |

**The `addroi` steering is expensive.** At crf 12, ROI-on costs 593,242 B for
42.56 dB; ROI-off at `slow` reaches the *same* 42.56 dB for 420,540 B — **41%
more bytes for nothing measurable on the whole plate**. At the shipped crf 30 it
is worse: 153,202 B against 82,533 B, **1.86x the bytes for +0.35 dB**. The
regions are a fixed fractional heuristic for tennis broadcasts, not detections,
so on this plate they are steering bits at guesses.

**The preset is not the explanation.** `veryfast` to `slow` is worth about
0.2 dB at the same bytes — real, small, and not what separates x264 from JPEG at
high fidelity.

**Without the ROI steering, x264 intra's low-rate advantage is much larger than
the shipped sidecar shows.** At ~35 dB RGB, JPEG needs about 244,000 B
(interpolated between q10 and q30); x264 at crf 30 `slow` with no ROI costs
**82,533 B — about 3x cheaper**. The lever is real; the sidecar's fixed
configuration is spending most of it.

### The ceiling: this route cannot reach the fidelity the ladder operates at

`crf 0` is lossless coding, so anything it still loses happened in the colour
handling around the codec rather than in it. Three lossless round trips
(`plate-codec-ceiling.json`; bounds for this control were written before reading
it, in `outputs/bp29-plate-codec/bounds-before-ceiling-control.json`):

| variant | bytes | RGB | Y |
|---|---:|---:|---:|
| the sidecar's own chain (`format=yuv420p`) | 1,315,933 | 44.22 | 43.98 |
| full-range 4:2:0 | 1,412,082 | 48.83 | 49.07 |
| full-range 4:4:4 | 1,784,484 | 52.37 | 53.68 |

**The `roi-video` sidecar tops out at 44.2 dB RGB / 44.0 dB Y no matter what
bitrate it is given.** The ladder's reference rung asks for 42.96 dB RGB /
45.44 dB Y — so on the ladder's own axis the plate cannot be delivered by this
route at all, and the last two dB before the ceiling cost 1.3 MB.

Of the ~8 dB the route gives away, **4.61 dB is range handling** and a further
**3.54 dB is 4:2:0 chroma subsampling**. Neither is the codec: both are the
conversion the sidecar wraps it in. This carries directly to BP29 §1.2 — an
av1-intra or vvc-intra sidecar built on the same interface inherits the same
ceiling unless the colour handling is stated explicitly.

### What the range half actually is: a DC shift, not a quantiser

The bound written for the full-range 4:2:0 control said [44.0, 48.0] dB, reasoning
that recovering a 16-235 squeeze is worth 1-3 dB. It measured **48.83**, outside
the bound, so the mechanism was not understood and was chased down rather than
reported (`range_forensics` in `plate-codec-ceiling.json`).

- The coded stream carries **no colour tags at all**: `color_range: None`,
  `color_space: None`. Each half of the round trip therefore applies its own
  default, and nothing in the file says whether they agree.
- Fitting the decoded plate against the original, per channel, gives gains of
  **0.9997, 1.0014, 1.0005** and offsets of **-1.46, -1.66, -1.20** grey levels.
  It is not a gain error. It is a **constant darkening of about 1.4 levels**.
- Removing that gain and offset lifts the round trip from **44.22 dB to
  50.72 dB** — 6.5 dB recovered by subtracting a DC shift — which is most of the
  way to the full-range 4:4:4 control's 52.37 dB.
- Clipping is not the mechanism: 0.157% of decoded pixels sit at 0 or 255
  against 0.136% in the original.

**Why the bound was wrong.** It modelled the range effect as a *quantisation*:
255 levels squeezed into 219 and back, whose error sits around 57 dB and is
negligible here. The actual effect is a **bias**, and a 1.4-level DC shift alone
scores about 45 dB — dominant rather than negligible. The bound's number was a
consequence of modelling the wrong mechanism, so the correction is to the model,
not to the interval.

## 6. A bound fired on png, and the bound was the thing that was wrong

`bounds-before-run.json` gave png a window of **8,000,000-24,883,200 B**, on the
stated basis that "PNG on natural broadcast content typically gives 1.2-2.0x".
png z3 measured **3,272,798 B** — 7.6x, below the floor. That is an alarm, and
it is recorded here rather than edited away.

**The bound's two criteria disagreed with each other, and the byte window is the
one that was wrong.** The same file's running-check — "under 1,000,000 B, or
finite PSNR" — did not fire, correctly: png is running, and four independent
observations say so.

- **A second encoder agrees.** ffmpeg's own png encoder at compression level 3
  produced **4,633,389 B** on the same pixels — same order, different library.
- **A third one, months ago, agreed too.** The frame as originally extracted to
  disk is **4,259,511 B** (`frame_000038.png`).
- **The decode is bit-identical** at every level, and `z0` — compression
  disabled — lands at **24,925,484 B**, i.e. the raw 24,883,200 B plus container
  overhead. An encoder that returns raw when told not to compress is running.
- **The noise control.** Add ±2 grey levels of uniform noise to the same plate
  and re-encode: **12,067,448 B**, a 3.7x jump straight into the bound's window.
  PNG compresses this plate hard because of what is in it, not because it is
  skipping work.

**Why the basis was wrong.** 1.2-2.0x is the ratio for camera-original
photographs. This plate is a *decoded* 4K broadcast frame: its high frequencies
were quantised away by the delivery codec before PNG ever saw it. Measured on
the plate itself, the mean absolute horizontal gradient is **1.60 grey levels**
and **87.0%** of horizontally adjacent pixels differ by one level or less. The
noise control is the same statement from the other side — restore the high
frequencies and PNG's output moves back into the predicted window.

The lesson for the next bound: a compression-ratio prior has to be stated for
the *content class* being coded. "Broadcast video" and "a decoded frame of
broadcast video" are not the same class, and this project codes the second one
everywhere.

## 7. What is not measured here, and why

**The four end-to-end arms failed, all four, with the same error.** They were to
run `tier balanced` with the residual pinned to the BP24 ladder's reference rung
(av1, QP 38) and only `background.codec` moving:

```
jpeg:75          FAILED WeightResolutionError("Weight 'yolo26n-pose.pt' not found at
  /home/itec/emanuele/pointstream-w8-a/assets/weights/yolo26n-pose.pt. ...")
jpeg:19 (matched to roi crf30)  FAILED WeightResolutionError(... same ...)
png:3            FAILED WeightResolutionError(... same ...)
roi-video:crf30  FAILED WeightResolutionError(... same ...)
```

This is not this stream's lever failing. Since `assets/` and `outputs/` moved
out of the checkout, several call sites still resolve weights by joining
`"assets"` onto the repository root instead of going through
`src/contracts/paths.py`, so every weight-loading path fails inside a worktree.
The weight itself is present at
`/home/itec/emanuele/pointstream-data/assets/weights/yolo26n-pose.pt`. The fix
is being made centrally on `wave8/weights-path`; this stream did not touch it.

So **total payload and delivered Y-PSNR per codec are not measured yet**. The
harness for them is written and its cross-check is in place: each arm compares
the runner's `sizes.panorama` against the sidecar byte count measured here for
the same settings, and an inequality is an alarm — that is what proves the plate
the runner sent is the plate this module encoded. Quality would be scored on
`delivered_frames`, never `RunResult.frames`.

Also not measured, deliberately: no BD-rate and no paired ladder (that is one
run, once every stream's lever has landed); no av1/vvc intra sidecar (BP29 §1.2,
Stream B); one clip, one plate, 8 frames.

## 8. Provenance

- Bounds, before the first encode: `outputs/bp29-plate-codec/bounds-before-run.json`
- Bounds for the ceiling control, before reading it:
  `outputs/bp29-plate-codec/bounds-before-ceiling-control.json`
- Sweep and the failed arms: `outputs/bp29-plate-codec/plate-codec-sweep.json`, `sweep.log`
- x264 controls and the png investigation:
  `outputs/bp29-plate-codec/plate-codec-diagnostics.json`, `diagnose.log`
- Ceiling decomposition: `outputs/bp29-plate-codec/plate-codec-ceiling.json`, `ceiling.log`
- Re-run: `python -m experiments.tier.plate_codec_sweep --part {sidecar,diagnose,ceiling,end-to-end,all}`
- Tools by path: `/opt/local/bin/ffmpeg`, `/opt/local/bin/ffprobe`, both `n7.1.1-56`
