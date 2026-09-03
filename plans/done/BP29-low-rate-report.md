# BP29 §2 — is there a crossover at very low rate?

Wave 8, Stream C. Branch `wave8/low-rate`.

**The question.** The BP24 ladder stopped at QP 55. PointStream degrades to a
clean plate; a starved transform codec degrades to blocking, and the two are not
the same kind of bad. `presley`'s operating map records that *"the same video
flips sign along the QP ladder, and the same QP flips sign across videos"*
(`docs/PLAN_OPERATING_MAP.md`), so a crossover is a measured phenomenon in a
sibling project rather than a hope. This stream extends the **anchor** to the
bottom of av1's QP range and asks whether the curves cross.

**The answer, in three parts.**

1. **At low rate the curves do not cross, and the gap widens** — 3.89x at
   PointStream's cheapest quality against 2.35x integrated over the overlap.
   The reason is structural: av1 reaches 51,254 B while PointStream's floor is
   318,077 B, so only one arm is present down there (§5).
2. **av1 does not fall off a cliff.** Over the whole remaining eight steps of
   its QP range it loses 2.65 dB for a 40% rate cut. The premise the hypothesis
   rested on is not supported on this content (§3).
3. **An apparent crossover at HIGH rate turned out to be the instrument**, and
   it is the most transferable thing here: the anchor arm is capped at
   **44.44 dB** by a 4:2:0 round trip that PointStream's arm does not go
   through. Checked with a lossless control before being believed, and the bias
   runs *toward* PointStream, so it does not rescue any number (§6).

Bounds written before the first encode:
`outputs/bp29-low-rate/bounds-before-run.json`; second bounds file, written
before the high-rate check and after seeing the low-rate result:
`outputs/bp29-low-rate/bounds-before-highrate-check.json`. Raw results:
`av1-lowrate.json`, `ceiling-control.json`; logs `ladder.log`,
`highrate-check.log`, `ceiling-control.log` — all under
`outputs/bp29-low-rate/`.

---

## 1. What was run, and what was not touched

`experiments/tier/ladder.py` is **unmodified**; every extension is a `--rungs`
argument. The main run:

```
python -m experiments.tier.ladder --codecs av1 --frames 8 --tier balanced \
  --sweep payload --video alcaraz_highlights --scene scene_000 \
  --rungs 15 25 35 45 55 58 61 63 \
  --out outputs/bp29-low-rate/av1-lowrate.json
```

Two follow-ups, both prompted by what the first returned and both with their own
bounds written first:

- the same command with `--rungs 5 10 15`, to test whether the apparent
  high-rate crossing was a truncation artefact (§6). **Stopped after its three
  anchor rungs**: the PointStream arm it would have re-run had already
  reproduced byte-identically twice, and a third pass would have spent ~25
  minutes of a loaded shared host for no new information. Its anchor rungs are
  in `highrate-check.log`; no JSON was written, which is why the log is cited.
- `experiments/tier/ceiling_control.py`, a lossless-round-trip control isolating
  the colour path (§6). It runs no encoder, and is the one new file this stream
  adds.

Clip: `alcaraz_highlights/scene_000`, 8 frames, 3840x2160, source 199,065,600 B,
inter-frame MAD 0.33 — the same clip BP24 used, and the most static of the eight
cached windows. Quality is Y-PSNR, one pooled-MSE PSNR over the whole clip,
computed by the ladder for both arms.

Encoder resolved by path and version, not by name:
`/opt/local/bin/SvtAv1EncApp`, **SVT-AV1 v1.8.0 (release)**;
`/opt/local/bin/ffmpeg`, **n7.1.1-56-gc2184b65d2**.

The only outside work merged in is **PR #35** (`wave8/weights-path`), without
which the runner cannot load a model at all after the data move. Stream A's,
B's and D's levers are **not** in this measurement, on purpose: this is today's
plate against an extended anchor.

## 2. The instrument, checked before the result

Three checks, all passed, all cheap.

**The five known anchor rungs reproduce byte-for-byte.** BP24 measured them on
the same clip, tier, preset and rate control on 2026-08-28; this run repeated
them on 2026-08-30, after the data root moved and the weights path was
refactored.

| QP | BP24 bytes | this run | BP24 Y-PSNR | this run |
|---:|---:|---:|---:|---:|
| 15 | 851,572 | **851,572** | 44.016 | **44.02** |
| 25 | 526,104 | **526,104** | 43.715 | **43.72** |
| 35 | 289,198 | **289,198** | 42.968 | **42.97** |
| 45 | 156,710 | **156,710** | 41.646 | **41.65** |
| 55 | 85,995 | **85,995** | 39.448 | **39.45** |

**No second encoder.** `plans/BP24-findings.md` §14: a decode naming no `-c:v`
re-encodes through the muxer's default (libx264 for `.mkv`) and caps every
returned quality — the pre-fix table for these same five rungs read 41.71,
41.67, 41.43, 40.65, 38.94 dB, pinned near x264's own ceiling. This run returns
44.02 at QP 15, so the `ffv1` decode fix is live. The built decode argv was
inspected directly and names it:

```
ffmpeg -hide_banner -loglevel error -y -i a.ivf -pix_fmt yuv420p -c:v ffv1 decoded.mkv
```

At the extended end the flat-quality signature is absent: every step moves both
axes (§3).

**Nothing was clamped, and that is a positive control rather than an
inference.** `plans/BP24-findings.md` §15 warns that an out-of-range rung is
caught only by the encoder refusing it, and that an encoder which *clamped*
instead would produce "a rung that looks like a rung, at a quantizer nobody
asked for". `QP_BOUNDS` in `src/components/codec/encode.py` is consulted only on
the ROI path, so nothing between the config and the binary stops an out-of-range
QP on the anchor path. The binary itself was therefore driven directly:

| `--qp` | exit | bytes | encoder output |
|---:|---:|---:|---|
| 55 | 0 | 7,991 | `TPL is disabled for aq_mode 0` |
| 58 | 0 | 6,297 | `TPL is disabled for aq_mode 0` |
| 61 | 0 | 4,897 | `TPL is disabled for aq_mode 0` |
| 63 | 0 | 3,884 | `TPL is disabled for aq_mode 0` |
| 64 | **1** | — | `Svt[error]: Instance 1: QP must be [0 - 63]` |
| 70 | **1** | — | `Svt[error]: Instance 1: QP must be [0 - 63]` |

This build **refuses** an out-of-range QP rather than clamping it, so QP 63 is
genuinely the last legal rung and each of 58, 61, 63 reached the encoder as
asked. The one warning is the ladder's own deliberate CQP setting
(`--rc 0 --aq-mode 0`), identical at every rung, so it cannot differentiate
them. (Probe on a 640x360 synthetic clip; the point is the accept/refuse
boundary, not the byte counts.)

## 3. The extended anchor

Three new rungs. All three landed **inside** the bounds written before the run.

| QP | bytes | Y-PSNR | d bytes | d dB | bytes bound | dB bound |
|---:|---:|---:|---:|---:|---|---|
| 55 | 85,995 | 39.45 | — | — | *(known)* | *(known)* |
| 58 | **71,842** | **38.65** | -16.5% | -0.80 | [45,000, 82,000] ok | [36.8, 39.44] ok |
| 61 | **60,739** | **37.75** | -15.5% | -0.90 | [25,000, 72,000] ok | [34.5, 38.7] ok |
| 63 | **51,254** | **36.80** | -15.6% | -0.95 | [14,000, 66,000] ok | [32.5, 38.2] ok |

Strictly monotone in both axes at every step, and every step moves the rate by
~16% and the quality by ~0.9 dB — no tie, no flat stretch, no plateau.

**The load-bearing observation: av1 does not fall off a cliff.** The whole
hypothesis behind §2 was that below some rate the anchor's quality collapses
while PointStream's does not. Over the entire remaining eight QP steps of av1's
range, from QP 55 to its hard limit at 63, quality falls **2.65 dB** (39.45 →
36.80) while rate falls 40% (85,995 → 51,254 B). That is a graceful, nearly
linear degradation, not a collapse. There is no cliff for PointStream to be
better than.


## 4. The PointStream arm, and it reproduced too

The ladder's payload sweep, which is the cheapest PointStream configuration
reachable today (§6). All five rungs came back **byte-identical** to BP24, at
the same PSNRs:

| rung | bytes | Y-PSNR | plate | appearance | residual | metadata | vs BP24 |
|---|---:|---:|---:|---:|---:|---:|---|
| jpeg30/qp55 | **318,077** | **39.21** | 283,483 | 22,542 | 10,285 | 1,767 | +0.000% B, +0.000 dB |
| jpeg50/qp46 | 390,889 | 41.45 | 345,947 | 22,542 | 20,633 | 1,767 | +0.000% B, +0.000 dB |
| jpeg75/qp38 | 525,462 | 43.59 | 463,334 | 22,542 | 37,819 | 1,767 | +0.000% B, +0.000 dB |
| jpeg90/qp28 | 808,573 | 45.39 | 713,320 | 22,542 | 70,944 | 1,767 | +0.000% B, +0.000 dB |
| jpeg98/qp18 | 1,548,393 | 46.55 | 1,408,247 | 22,542 | 115,837 | 1,767 | +0.000% B, +0.000 dB |

Every rung reported `is_rate=true`; none was excluded; `failures` is empty and
the ladder's own `bound_alarms` list is empty.

**The 895.1 s first rung is the model-load cost, not a measurement artefact.**
It stands against 133.7, 136.8, 135.2 and 120.9 s for the other four. Three
things confirm the cause. BP24's independent run showed the same shape — 954.78 s
on this same first rung against 140-164 s for the rest. `AGENTS.md` documents
the mechanism and even the magnitude: on this NFS home directory "a codec-only
run that constructs a YOLO backend it never uses pays ~800 s for nothing — which
is exactly the gap between the ladder's first rung (950 s) and its later ones
(150 s)". And decisively, the rung's *output* is byte-identical to BP24's, so
whatever the wall clock did, it did not enter the measurement.

**The non-plate floor — the most reusable number here.** The plate is the only
part of the payload that moves with the rung. Appearance (22,542 B), metadata
(1,767 B) and the coarsest residual (10,285 B) do not, and together they are

> **34,594 B that PointStream transmits before it has paid for a plate at all.**

## 5. Do the curves cross? Not at low rate, and the gap widens

**No.** And the reason is structural rather than marginal.

| | rate range measured |
|---|---|
| anchor (av1 on source) | 51,254 - 851,572 B |
| PointStream | 318,077 - 1,548,393 B |

PointStream has **no operating point below 318,077 B**. The anchor's three new
rungs live at 51,254-71,842 B — a region PointStream cannot enter at all with
this plate. So at the low-rate end the curves cannot cross, because only one arm
is present there.

**What each arm delivers where the anchor bottoms out.** At its last legal rung,
QP 63, av1 delivers **36.80 dB for 51,254 B**. PointStream's cheapest
configuration delivers **39.21 dB for 318,077 B** — 2.41 dB better, for **6.21x
the bytes**. PointStream cannot trade that quality away for rate, because 89% of
its payload is a plate it must send whatever quality it is targeting.

**The gap widens as the rate falls**, which is the opposite of the hypothesis:

| comparison | anchor rate | PointStream rate | ratio |
|---|---:|---:|---:|
| at PointStream's cheapest quality (39.21 dB) | 81,753 B *(interpolated)* | 318,077 B | **3.89x** |
| at the anchor's cheapest rung (QP 63) | 51,254 B | >= 318,077 B | **>= 6.21x** |

**An honest limit on that second row.** PointStream's cheapest measured point
sits at 39.21 dB — *above* all three new anchor rungs (38.65, 37.75, 36.80 dB).
So at QP 58/61/63 there is no measured PointStream point to compare against;
the arm is at its floor and cannot descend. The ">= 6.21x" is therefore a
statement about a **floor**, not a measured comparison at that rate: PointStream
cannot get cheaper than 318,077 B, so whatever quality it were asked for down
there, it would still cost at least that. What was ruled out at 51,254 B is not
"PointStream measured worse" but "PointStream can reach this rate at all".

**BD-rate, with its range and its n.** The ladder reports **+134.6%** over
**39.21-44.02 dB** — PointStream costs 2.35x the anchor's rate at equal quality.
BP24 reported +116.8% over 39.45-44.02 dB. **The two are the same measurement on
byte-identical data**, and the difference is entirely the overlap's lower edge:
BP24's band started at the anchor's own minimum quality (39.45 dB), while here
the anchor extends *below* PointStream, so the band starts at PointStream's
minimum (39.21 dB) and integrates 0.24 dB further into the region where
PointStream is furthest behind. This is not a new lever's result and must not be
quoted as one. **n = 1 clip, 1 codec, 8 frames** — `presley`'s operating map
sets n>=6 videos before any significance claim, so no significance is claimed
here. The encodes themselves are deterministic (byte-identical across two runs
two days apart), so the *measurement* carries no sampling error worth a standard
error; the *generalisation* carries n=1 and must not be stated as a property of
the method.

## 6. An apparent crossover at HIGH rate — checked, and it is the instrument

**This is the part of the result that looked good, so it got the extra check.**

Over the measured rate overlap (318,077-851,572 B) the interpolated curves do
cross, near **543,419 B** where both arms sit at ~43.7 dB, and above it
PointStream reads **+1.47 dB above the anchor** at the top of the overlap. Taken
at face value that is "PointStream wins at high rate". It is not.

**The suspicion, written before the check** (`bounds-before-highrate-check.json`):
the anchor's curve was *truncated* at QP 15, the finest rung BP24 ever ran, and
it was already flattening — QP 25 -> 15 bought only 0.30 dB for 62% more bytes.
An arm that appears to win above the rate where the other arm's data simply
stops is a truncation artefact. Two hypotheses were written down with the
discriminator: does av1 at QP 5 climb past PointStream's 45.39/46.55 dB (H1,
truncation) or stall below ~44.6 dB no matter what it spends (H2, a ceiling)?

**The two extra encodes** (`outputs/bp29-low-rate/highrate-check.log`):

| QP | bytes | Y-PSNR | vs QP 15 |
|---:|---:|---:|---|
| 15 | 851,572 | 44.02 | *(reproduced byte-identically a third time)* |
| 10 | 1,218,919 | 44.16 | +43% bytes, **+0.14 dB** |
| 5 | 1,836,510 | 44.23 | +116% bytes, **+0.21 dB** |

Monotone in both axes, so the rungs are reaching the encoder. But **2.16x the
bytes buys 0.21 dB**. That fired the `hard_ceiling` alarm written into the
bounds file, which named its own prime suspect: the RGB -> yuv420p -> RGB round
trip inside `coded_roundtrip` feeding a luma PSNR.

**The control settles it.** The clip was pushed through `coded_roundtrip`'s
colour path with **no lossy codec at all** — rawvideo RGB -> pixel format ->
ffv1 (lossless) -> back to RGB — and scored with the ladder's own PSNR
(`experiments/tier/ceiling_control.py`, result in
`outputs/bp29-low-rate/ceiling-control.json`). Three anchors: identical, mild,
and the suspect.

| lossless round trip through | Y-PSNR | reading |
|---|---:|---|
| RGB planar (no chroma conversion) | **inf** | the harness is lossless; the null passes |
| yuv444p (8-bit YUV, no subsampling) | **53.69** | conversion rounding alone is negligible |
| **yuv420p** (chroma subsampled) | **44.44** | **the ceiling** |

**The anchor arm cannot score above 44.44 dB on this clip no matter how many
bytes it spends**, because 4:2:0 chroma subsampling throws away information
before the encoder is reached, and reconstructing RGB from subsampled chroma
perturbs the luma the PSNR is then computed on. av1 at QP 5 lands 0.21 dB under
that cap: **the codec is essentially transparent there and the measurement
cannot see it.**

**PointStream's arm is not on that interface.** Its delivered frames are
assembled in RGB and compared to the RGB source directly, so nothing forces its
whole frame through 4:2:0. That is why it reads 46.55 dB — above a ceiling the
anchor is structurally unable to cross. **At the high-quality end the two arms
are not measured through the same interface**, and the "crossover" is that
asymmetry, not a property of either system.

**Which way the bias runs, stated plainly.** The ceiling costs the *anchor*
quality it actually delivered, so at a given quality the anchor appears to need
more bytes than it does. **The defect flatters PointStream.** The +134.6% here
and BP24's +116.8% are therefore, if anything, *understatements* of
PointStream's loss — this finding does not rescue any number, it makes the
negative result slightly worse.

**The low-rate answer is untouched.** §5 rests on measured points between 36.80
and 39.45 dB, four to eight dB below the 44.44 dB cap, where nothing is near it.

**A bound that was wrong, and why** — recorded rather than edited away, per
`AGENTS.md`. The prediction written before this check was **H1, 47-52 dB at
QP 5**. It was wrong. The reasoning ("SVT-AV1 CQP at QP 5 on 4K should be close
to transparent") was actually *correct about the codec* — QP 5 is transparent.
The error was writing a bound about the **encoder's** behaviour when the binding
limit lives in the **measurement path**. A bound aimed at the wrong object
cannot fire correctly, and the thing that caught this was not the prediction but
the separately-written `hard_ceiling` alarm, which described a symptom (many
bytes, no dB) rather than a value. **Alarms phrased as symptoms survive being
aimed at the wrong object; alarms phrased as ranges do not.**

**What this is owed downstream.** This is a defect in the shared measurement
path, not in anything Stream C owns, and fixing it is out of scope here. It
should be recorded against `coded_roundtrip`: the high-quality end of every
paired ladder run so far is compressed against a 44.44 dB cap that applies to
one arm only. `plans/BP24-findings.md` SS12 reached for a 4:2:0 explanation and
SS14 corrected it to the second encoder; the second encoder was real and is
fixed, and **this is the 4:2:0 ceiling SS12 was groping for, still present and
now measured** — visible only once the anchor is pushed fine enough to hit it,
which is why extending a ladder *upward* turned out to be worth two encodes.

## 7. Which PointStream arm this is, and the gap that blocks a cheaper one

**Stated explicitly, because it bounds every claim above.** The PointStream arm
measured here is the `balanced` tier with `background.method: panorama-full` and
**`background.codec: jpeg`**, swept by the ladder's payload rungs, whose
coarsest is JPEG quality 30 with the residual at QP 55. That is **the cheapest
PointStream configuration this stream could actually build.**

It is not the cheapest imaginable, and the reason is a sequencing gap rather
than a measurement:

1. **Stream B's av1/vvc intra sidecars were not available to this arm.** They
   landed on PR #36 after this run and measure a vvc plate at **68,477 B at
   38.38 dB** and **26,502 B at 33.57 dB** — plates far cheaper than the
   283,483 B JPEG plate used here. This stream does not build on unmerged work,
   so those were not an option.
2. **Even with #36, the config cannot express the interesting rung.**
   `BackgroundConfig` (`src/contracts/config.py`) carries only `method`, `codec`
   and `jpeg_quality`. There is no field that reaches an intra sidecar's QP, so
   `background.codec: av1` arrives at the sidecar's default QP and nothing else.
   Reaching a 26,502 B plate needs a config knob that does not exist yet.

**So the honest statement of scope is narrow on purpose:** this stream shows
that low-rate PointStream loses *with today's JPEG plate*, and that no plate
codec alone closes the gap (§4's 34,594 B floor). It does **not** rule out
low-rate PointStream at the plate costs Stream B has since demonstrated. That
is the obvious next probe, and §8 says what it would have to beat.

**What the arithmetic says that probe will find**, so it can be checked rather
than rediscovered. Taking Stream B's plate costs and holding this arm's
non-plate floor fixed at 34,594 B:

| plate (Stream B, PR #36) | plate bytes | PointStream total | anchor at that rate | plate's own quality |
|---|---:|---:|---:|---:|
| vvc qp45 | 26,502 | 61,096 B | **37.78 dB** | 33.57 dB |
| av1 qp55 | 79,726 | 114,320 B | **40.49 dB** | 38.25 dB |
| vvc qp55 | 8,247 | 42,841 B | *below the anchor's measured range* | 27.39 dB |

The vvc-qp45 row is the interesting one: it would put PointStream at ~61,096 B,
almost exactly where the anchor sits at QP 61 (60,739 B, 37.75 dB). The
crossover question then becomes concrete and testable: **can PointStream deliver
more than 37.78 dB from a plate that is itself only 33.57 dB?** The plate covers
99.4% of the pixels, so the residual would have to lift the whole frame by more
than 4 dB — which BP24 did measure it doing (5.40 dB) but at 3.7x this rung's
residual budget and against a far better plate. Treat the table as the bracket
that frames the next run, not as a result: the residual's cost against a much
worse plate is exactly the term it does not predict.

## 8. What this establishes, and what it does not

**Establishes.**

- av1 does not collapse at the bottom of its range on this content: 2.65 dB lost
  over the eight QP steps from 55 to its hard limit at 63, for a 40% rate cut.
  The premise behind §2 — that a starved transform codec falls apart where a
  clean plate does not — is not supported here.
- PointStream cannot enter the rate region where that would matter. Its floor is
  318,077 B against an anchor that reaches 51,254 B, and 34,594 B of that floor
  is not plate at all.
- The gap widens toward low rate: 3.89x at PointStream's cheapest quality,
  against 2.35x integrated over the overlap.

**Does not establish.**

- Anything about a cheaper plate. See §7 — the arm that would test it needs
  PR #36 plus a config knob for sidecar QP.
- Anything general. n = 1 clip (the most static of eight), 1 codec, 8 frames,
  Y-PSNR only, generation off. `presley`'s bar for a significance claim is n>=6
  videos.
- Anything perceptual. Frame Y-PSNR is dominated by the background, which is
  99.4% of the pixels; PointStream's case has always been argued on the object
  region, and `plans/BP29-plate-rate.md` §3 is where that claim belongs — with
  its integrity conditions, which include declaring the metric *before* the run.
- Anything about the **high**-rate end of this ladder without reading §6 first.
