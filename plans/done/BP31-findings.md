# BP31 findings

Running notes for the paired-ladder-across-scenes work. Brief:
`plans/BP31-paired-ladder-across-scenes.md`. Prompt:
`plans/prompts/next-session-bp31.md`.

---

## 1. Levers (a) and (b) cannot both be on: `panorama-stream` bypasses the sidecar

**This retires the plan in the brief's §0 as written.** `plans/BP31-paired-ladder-across-scenes.md`
§0 and the prompt's step 1 both assume a cheap plate-codec sweep decides which
codec the ladder's PointStream arm uses, with the cross-scene stream on at the
same time — "(a), (b) and (c) together, then re-run the paired ladder". That
configuration is not expressible against the code as it stands.

`src/components/background/strategy.py`: `PanoramaStream` overrides `transmit`,
`decode_payload` and `reconstruct`, and **none of them touches `self._sidecar`**.
The base `BackgroundModel` sidecar path (the `build_sidecar` call and its
`encode`/`decode`) is dead code under this method. The plate goes through
`BackgroundStreamTransmitter` at `background.stream_codec` /
`background.stream_crf` instead — ffmpeg's `libaom-av1` / `libx265` / `libx264`
on a CRF ladder, which is a **different encoder family** from the
`SvtAv1EncApp` / `vvencapp` / `kvazaar` path `background.codec` selects.

### Measured, not asserted

A flag existing is not a feature working, so this was driven rather than read
off. Three plates, three values of `background.codec`, one model each; payload
lengths per scene (`outputs/bp31-ladder/probe-lever-exclusivity.json`):

| `background.method` | `codec: jpeg` | `codec: av1` | `codec: vvc` |
|---|---|---|---|
| `panorama-stream` | 227373, 139184, 139642 | **227373, 139184, 139642** | **227373, 139184, 139642** |
| `panorama-full` | 47617, 47594, 47517 | 70592, 70559, 70620 | 15417, 15315, 15578 |

Byte-identical across all three codecs under `panorama-stream`. **The
`panorama-full` row is the control** and it is the half that makes the null
readable: the same probe, on the same plates, separates the three codecs by a
factor of 4.6 when the method actually consults the sidecar. So the null is the
method ignoring the knob, not the probe being unable to see a difference.

### The reporting trap this leaves

`BackgroundModel.__init__` still runs `normalize_sidecar(config.background.codec)`
and stores it, and `PanoramaStream.transmit` puts that name into
`BackgroundArtifact.codec`. So a run configured `{method: panorama-stream,
codec: av1}` **reports `codec: av1`** in the artifact while its bytes came from
libaom at `stream_crf`. `codec_id` is honest — `PanoramaStream` overrides it to
report the stream spec — but `artifact.codec` is not, and it is the shorter name
to reach for. Anyone reading a ledger to check "did the intra sidecar arm run?"
gets a plausible yes.

### What this changes

- **The step-1 sweep decides nothing for a streamed arm.** The knob carrying
  88-91% of the payload under `panorama-stream` is `stream_codec` x
  `stream_crf`, not `background.codec`. That is what has to be swept before the
  ladder is spent, and it is a different sweep from the one the brief describes.
- **The still-sidecar sweep is still worth running**, but it prices the
  `panorama-full` arm — i.e. it helps answer whether streaming is the right
  method at the ladder's rung, rather than which codec the streamed arm uses.
- **Making (a) and (b) compose is a real change, not plumbing.** A keyframe is a
  still, so coding keyframes through the intra sidecar while P-frames go through
  the stream is the natural shape — but the transmitter owns the chain and its
  reconstructions, so the chain would have to decode a sidecar-coded keyframe
  bit-exactly. That is an architectural change to a component BP30 has tests
  around, and it is follow-up work rather than something to fold into this run.

### A second gap found on the way: `intra_qp` reaches nothing either

`BackgroundConfig` has no `intra_qp` field, and `strategy.bind` forwards only
`codec`, `jpeg_quality` and `domain` (plus the four stream knobs). So
`BackgroundModel.__init__` never passes an intra QP and `build_sidecar` falls
back to `DEFAULT_INTRA_QP = 45` on **every** runner path. `background.codec: av1`
is therefore a single fixed operating point, not an axis — the same limitation
`plans/BP29-plate-codec-report.md` §4 recorded for `roi-video`'s `crf`, now
confirmed for the intra sidecars as well. Lever (a) is not tunable end to end
today, whichever method is selected.

### Provenance

- Probe: `outputs/bp31-ladder/probe-lever-exclusivity.json`.
- Read against `src/components/background/strategy.py` at `68cf1c9`.

---

## 2. `panorama-stream` had never completed a multi-scene run

Found by running the scene ladder, not by reading. `make_background` passed
`artifact.mode` straight into `BackgroundModelView`, which accepts only
`full` / `delta` / `none`. `PanoramaStream.transmit` emits `full` for its
keyframe and **`stream` for every scene after it**, so:

```
chunk 0  mode=full    -> fine
chunk 1  mode=stream  -> ValueError: background mode must be 'full', 'delta' or 'none'
```

**Every existing test passed one chunk**, which is the only shape in which this
works. So the cross-scene amortisation BP30 measured, PR #41 wired, and this
whole brief is built on had **never completed a run through the runner** — and
nothing was red.

Fixed at the runner boundary: a `stream` scene decodes to a *whole* plate
(`decode_payload` returns the transmitter's own reconstruction, not a difference
image), so reconstruction must treat it as `full`. Only `delta` means "add me to
the previous plate". `artifact.mode` is untouched, so `SizesBytes.panorama`
keeps its marginal-cost meaning.

Guarded two ways, because the single-chunk blind spot is the whole story here: a
multi-chunk test, and one that greps the stage for the translation.

**The general shape, which is not about backgrounds.** A component whose entire
purpose is to carry state *between* units of work cannot be tested one unit at a
time. `panorama-stream` exists to make scene *n* cheaper given scenes 1..n-1;
every test that passed it a single scene was testing the one case where that
purpose is inactive.

## 3. Only av1 amortises across scenes, and the other two fail differently

`outputs/bp31-ladder/stream-codec-sweep-alcaraz_highlights.json`. Twelve
point-class scene frames of `alcaraz_highlights` at native 4K, each codec against
**its own** intra baseline. Bounds were written before the first encode.

| codec | crf30 | crf38 | crf45 | crf51 | frame types |
|---|---:|---:|---:|---:|---|
| av1 | 0.664 | **0.646** | 0.539 | 0.485 | `IPPPPPPPPPPP` |
| hevc | 1.042 | 1.036 | 1.001 | 0.932 | `IPPPPPPPPPPP` |
| avc | 0.998 | 0.995 | 0.991 | 0.986 | `IIIIIIIIIIII` |

**Do not rank these against each other** — the low-delay flag sets are per
encoder and are not equal effort (findings §1). Each column is that codec
against itself.

**The controls all passed** (av1 0.0074, hevc 0.0202, avc 0.0136), so no row is
disqualified for failing to predict at all. av1's control reproduces BP30's
published 0.0074 **to the digit**, which is the cheapest available check that
this harness measures what BP30's did.

**av1 at crf38 reads 0.646 against BP30's 0.492 ± 0.062 headline, and that is
consistent rather than an alarm.** BP30's headline pools five videos;
`alcaraz_highlights` was the *worst* of them, at 0.607 ± 0.015 over 16 scenes
(`PLAN.md` §2.22). Twelve scenes amortise the opening keyframe over fewer
successors than sixteen, so slightly above 0.607 on the same video is what the
arithmetic predicts. Inside the pre-written band [0.25, 0.75].

**hevc predicts and still loses.** Its frame types are `IPPPPPPPPPPP` — x265 is
doing inter coding — and the chain costs *more* than coding every plate fresh at
three of four rungs. This is not a broken flag; it extends findings §18's
codec-dependence note ("libx265 chose intra for one of the two pairs, av1 did
not") from a pair to a twelve-scene sequence.

### avc's `IIIIIIIIIIII` was x264 being right, not x264 being misconfigured

The avc row looked like a configuration artefact and was chased down rather than
reported, because its saving was **exactly 6,948 B at all four CRFs** —
rate-independent, which prediction never is — and 6,948 B over eleven joins is
container header overhead, not coding. The obvious reading was that x264's
scenecut detector fired at every join and the "chain" was twelve independent
intra frames.

`outputs/bp31-ladder/avc-scenecut-diagnostic.json` tests that by disabling it
(`-sc_threshold 0`), same plates, same baseline, both arms:

| crf | default | scenecut disabled |
|---:|---|---|
| 30 | 2.530, `IPPIIIIIIIIIIIIPII` | 2.501, `IPPPPPPPPPPPPPPPPP` |
| 38 | 2.464, same | 2.445, `IPPPP…` |
| 45 | 1.962, same | 1.939, `IPPPP…` |
| 51 | 1.361, same | 1.361, `IPPPP…` |

**The flag reaches the encoder** — the frame types change completely — **and the
cost does not move**, by under 1.2% at every rung. So a P-frame across one of
these scene joins costs what an I-frame costs, and x264's detector was making
the correct decision. avc's sweep row stands as a measurement rather than being
withdrawn.

**These absolute ratios are NOT comparable to the sweep table above.** This
diagnostic uses 18 cached plates and its own single-image intra baseline, not
the component's `encode_chain`, so it sits on a different axis. Only the
within-diagnostic comparison — default against scenecut-disabled, which share a
baseline — is being read here, and that is the only question it was built to
answer.

### What this settles for the ladder

`background.stream_codec` must be **av1**. That was the default, so nothing
changes in the config — but it was the default by inheritance from BP30 rather
than by measurement over a sequence, and the two alternatives are now priced:
one predicts and loses, the other correctly declines to predict at all.

## 4. Constraining the anchor to low delay is a 20-38% gift to PointStream

The brief's fairness condition asks for "the same low-delay constraint" on both
arms. Measured before using it, per `AGENTS.md` on flags that exist versus flags
that work — SvtAv1EncApp v1.8.0, `--pred-struct 1 --lookahead 0 --keyint 1000`:

- a 640x360 synthetic encode goes **66,485 -> 91,805 B, +38%**;
- the real two-scene 4K anchor goes **154,891 -> 186,437 B, +20%**, *and* loses
  quality, 38.23 -> 37.45 dB Y.

So the constrained anchor is both dearer and worse, and a ladder reporting only
that arm would hand PointStream a fifth of its rate and call it fairness. The
scene ladder therefore runs **both anchors at every rung** rather than taking a
flag, with the unconstrained one leading as the harder comparison and the
latency-matched one reported beside it. Which question each answers is in the
record.

The first version of the flag table was also simply wrong — it assumed ffmpeg
for every codec, and `src/components/codec/tools.py` sends `avc`/`vvc` to
ffmpeg, `hevc` to kvazaar and `av1` to `SvtAv1EncApp`. SvtAv1EncApp rejected the
ffmpeg-style argument outright ("single dash long tokens have been removed"),
which is the good failure. A codec whose low-delay vocabulary has not been
checked against its own binary now **refuses** rather than silently running
unconstrained, because a run that completes with the constraint absent is the
version of this that gets published.

## 5. The anchor really does predict across a scene join

The fairness condition is measured, not promised. At every rung the scene ladder
encodes the N scenes **jointly** (the arm) and **separately** (the control):

| arm | joint | separate | joint/separate |
|---|---:|---:|---:|
| anchor | 154,891 B | 181,578 B | **0.853** |
| anchor, low delay | 186,437 B | 212,108 B | **0.879** |

Both below 1.0, so the anchor given the concatenation genuinely predicts across
the join and takes a 12-15% discount for it. Had these come back at 1.0, the
anchor would have been effectively running per-scene and any PointStream gain
against it would have been an artefact of the rig. That check is an alarm in the
ladder rather than a note beside it.

Two scenes of `alcaraz_highlights`, one rung, av1 preset 10 — a path validation,
not a curve.

## 6. A bound fired, and the bound was the wrong instrument

`outputs/bp31-ladder/bounds-before-run.json` said the background's share of the
payload must fall out of the 88-91% band it occupied with a fresh plate per
scene, and that a share still at or above 88% means "the stream is not reaching
the ledger". It fired. The stream is reaching the ledger, and **the share cannot
tell**.

Two scenes of `alcaraz_highlights`, one rung, same everything but the method:

| arm | panorama | total | share | Y-PSNR |
|---|---:|---:|---:|---:|
| `panorama-stream` | 789,304 B | 862,585 B | 0.915 | 41.90 dB |
| `panorama-full` (control) | 568,954 B | 643,954 B | 0.884 | 39.30 dB |

**Why the bound was wrong, which is the part worth keeping.** Share is
`plate / (plate + residual + actor + metadata)`, and on this content the
non-plate parts are about 9% of the payload. That makes the share a *saturating*
function of the plate's cost: even halving the plate only moves it from 0.91 to
0.83, and at N=2 — where a keyframe plus one marginal scene is nearly two whole
plates anyway — it barely moves at all. The share was chosen because §2.20
reported the problem in those units, but "the number that stated the problem" and
"the number that detects the fix" are not the same number. The bound inherited
the units of the complaint.

**The right instrument is plate bytes against a fresh-plate control at matched
fidelity.** Which is also why the table above settles nothing on its own: the two
arms are 2.6 dB apart, so the stream is dearer *and* better and they are simply
at different operating points. Comparing at a matched *knob* rather than matched
fidelity is the error `plans/BP29-plate-codec-report.md` §3 was written about,
and reading "the stream costs 1.39x" off that table would be committing it.

So the comparison is being re-run as **two curves over the full payload ladder,
one per method**, which is the only form in which the question has an answer.

**And N=2 is the wrong N for it regardless.** Amortisation is what scene *n*
saves given scenes 1..n-1; at two scenes that is one keyframe plus one marginal
scene, the least favourable case the mechanism can be given. The sweep in §3
measures 0.646 at twelve scenes on this video. Only two scenes of
`alcaraz_highlights` have cached BP21 windows with player tracks, so the ladder
cannot yet be run at the N the mechanism needs — see the handoff note.

## 7. Where this leaves BP31, and what the next session needs

**The brief's plan cannot be run as written** (§1), and the run it describes had
a blocker nobody could have seen without trying it (§2). What is now true:

- `panorama-stream` completes a multi-scene run. It did not before this session.
- `background.stream_codec` should be **av1**, now by measurement over a
  twelve-scene sequence rather than by inheritance (§3).
- The paired ladder over N scenes exists, with the anchor on the same footage and
  the fairness condition measured rather than promised (§5).
- Both anchor arms run, so the low-delay constraint cannot quietly flatter
  PointStream (§4).

**The blocker on the number the paper needs is data, not code.** The ladder needs
each scene as a `TierClip` — source frames plus verified player tracks — and
those come from BP21's cached windows, which exist for **eight scene/video pairs
in total**, at most two per video:

| video | cached scenes |
|---|---|
| `alcaraz_highlights` | `scene_000`, `scene_010` |
| `federer_djokovic` | `scene_001`, `scene_003` |
| `alcaraz_perricard`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz` | one each |

So the ladder can run at N=2 on two videos and N=1 elsewhere, and N=2 is the
least favourable case amortisation can be given (§6). **Materialising more is
mechanical, not new work**: `experiments/headroom/real.load_scene_clip` writes
exactly the `window/frame_*.png` layout `load_tier_clip` reads, and the dataset
carries segmentations for ~10 point-class scenes on each of six videos.
`iter_point_scenes_spread()` enumerates them. That extraction is a long detached
job — 4K decode plus paste-back verification per scene — and it is the first
thing the next session should start, because everything else waits on it.

Target N: ten scenes on each of six videos clears `presley`'s n>=6 bar that
`plans/BP31-paired-ladder-across-scenes.md` §2 sets, which BP30 (five videos)
did not.

**Two things deliberately not done.**

- **Making levers (a) and (b) compose** (§1). A keyframe is a still, so coding
  keyframes through the intra sidecar while P-frames go through the stream is the
  natural shape — but the transmitter owns the chain and its reconstructions, so
  the chain must decode a sidecar-coded keyframe bit-exactly. That is an
  architectural change to a component with tests around it, and folding it into
  this run would have meant changing the thing being measured while measuring it.
- **`intra_qp` plumbing** (§1). Worth doing when (a) is next touched; useless
  before then, because the streamed arm never consults the sidecar.

**The reporting trap in §1 should be closed whatever happens next.**
`BackgroundArtifact.codec` reports a sidecar name a streamed run never used, so
a ledger reads as though the intra arm ran. It is a one-line honesty fix and it
is the kind of thing that silently backs a wrong claim.

## 8. The payload ladder froze the plate on the streamed arm

Both curves ran at N=2, and the streamed one is not an RD curve. Panorama bytes
per rung, `outputs/bp31-ladder/ladder-curve-panorama-{stream,full}.json`:

| rung | `panorama-stream` plate | `panorama-full` plate |
|---|---:|---:|
| q30/qp55 | 789,304 | 568,954 |
| q50/qp46 | **789,304** | 692,087 |
| q75/qp38 | **789,304** | 920,775 |
| q90/qp28 | **789,304** | 1,416,592 |
| q98/qp18 | **789,304** | 2,787,012 |

**Identical to the byte at all five rungs.** `PAYLOAD_RUNGS` pairs
`background.jpeg_quality` with the residual's rate, and §1 is that
`jpeg_quality` reaches nothing under `panorama-stream` — so the streamed ladder
swept the residual against a frozen background. `panorama-full`'s plate moves
4.9x across the same rungs, which is the control showing the table itself is fine.

This is `PAYLOAD_RUNGS`'s own docstring happening again, to the arm it was
written for: *"a rung has to move everything that trades rate for quality"*,
after a first ladder moved the payload 6% because the plate was 93% of it and
did not move with the residual's knob. The same trap, entered through a
different knob, one method later.

**It also explains both of §6's alarms.** The high-side one (share stuck at
91.5%) and the low-side one (share collapsing to 53.0% at the finest rung) are
the same fact: a fixed plate with a residual growing 25,269 -> 651,086 B under
it. The share was tracking the residual all along.

**And the streamed curve saturates** — 41.90 to 43.63 dB for 1.7x the rate,
while `panorama-full` reaches 45.88 dB — because the plate is pinned at av1
CRF 38 and no residual rung can buy back plate detail that was never sent. A
BD-rate taken against that would have measured the frozen knob.

**Fixed:** `STREAM_PAYLOAD_RUNGS` sweeps `stream_crf` (51/45/38/30/22) against
the same residual rates, so the two tables describe the same five operating
points through whichever knob the method actually reads. **Guarded:** the ladder
now raises an alarm when the plate is byte-identical across rungs, because this
failure produces a smooth, monotone, entirely plausible curve.

**No comparison is drawn between the two N=2 curves here.** The streamed one was
not measuring what it claimed, and the re-run with the corrected rungs is the
first version of it worth reading.

## 9. A BD-rate, at last — and the stream is worth 19 points of it

With the plate knob corrected (§8), both arms produce readable curves and
`experiments/tier/ladder_scenes_compare.py` integrates them. Two scenes of
`alcaraz_highlights`, 8 frames each, av1 preset 10 on both arms, anchor encoding
the concatenation:

| PointStream arm | BD-rate vs av1 anchor | overlap |
|---|---:|---|
| `panorama-full` — a fresh plate per scene | **+109.72%** | 39.3-43.9 dB |
| `panorama-stream` — amortised across scenes | **+90.97%** | — |
| `panorama-stream` with the frozen plate (§8) | *refused* | — |

Both inside the pre-written band of [-20%, +150%]
(`outputs/bp31-ladder/bounds-before-run.json`), so no alarm. The third row is
the guard from §8 doing its job on the run that produced it.

**The cross-scene stream is worth about 19 BD-rate points at N=2**, which is the
least favourable N amortisation can be given — one keyframe and one marginal
scene. §3 measures the underlying saving at 0.646 over twelve scenes on this
same video, so the ladder number should improve with N, and the next session's
first job is to make N large enough to say by how much.

**What this is not.** It is not comparable to `PLAN.md` §2.20's +116.8%: that was
one scene against a single-scene anchor, and this is two scenes against an
anchor encoding both. The right reading is the *within-this-run* comparison,
+109.72% against +90.97%, where everything except the background method is held
fixed. And two scenes of one video is a configuration measurement, not a claim —
`presley`'s bar is n>=6 videos.

**PointStream still loses by a wide margin.** +90.97% means it costs roughly
twice the anchor's rate at equal quality. Closing that is what the untried axes
in `plans/prompts/next-session-bp31.md` are for, and the content axis is the
first of them: this ran on `alcaraz_highlights`, which §2.20 chose as the most
*static* of eight clips — the friendliest case for the anchor and the worst for
an object-centric codec.

### The units trap, avoided by one reading

`BDComparison.bd_rate` is a **fraction**, not a percentage: `+1.168` is
`+116.8%`. The comparison module converts once, next to the band, with the
reason in a comment. `AGENTS.md` records a bound that fired against a correct
result because it had been derived in the wrong units, and a band written in
percent against a value returned as a fraction would have read `+0.91%` — a
spectacular and entirely fictional win, comfortably inside the band, with
nothing about it looking wrong.

## 10. Plate codecs as curves: the 3.6-4.1x lever is 1.45x where the ladder runs

**This supersedes the single-point probe** this session opened with, which
compared codecs at operating points they had not been asked to share — `vvc` came
back both cheaper *and* lower quality than `av1`, which is not a comparison of
anything. A codec is comparable to another only through curves read at matched
fidelity, and only with encode time in the same table. Harness:
`experiments/tier/plate_codec_curves.py`; bounds written first in
`outputs/bp31-ladder/bounds-before-codec-curves.json`; result in
`plate-codec-curves-alcaraz_highlights.json`. 25 points on one 4K panorama
plate, each codec swept over its own knob.

### Size, at matched Y-PSNR, interpolated on each codec's own curve

| target | jpeg | av1 intra | vvc intra |
|---|---:|---:|---:|
| 38 dB | 243,892 B | *below its range* | 46,101 B — **x0.189** |
| 40 dB | 277,512 B | 85,432 B — x0.308 | 71,003 B — x0.256 |
| 42 dB | 322,866 B | 151,542 B — x0.469 | 130,064 B — x0.403 |
| **43 dB** | **352,752 B** | **243,861 B — x0.691** | **243,617 B — x0.691** |
| 45 dB | 434,810 B | *above its range* | *above its range* |

No cell is extrapolated; a target outside a codec's measured span is reported as
such rather than fitted.

**The lever is strongly quality-dependent, and `PLAN.md` §2.21 quotes it at the
wrong end.** §2.21 claims "a factor of 3.6 to 4.1" for av1/vvc intra over JPEG on
88-91% of the payload. That figure is reproduced here — at **38 dB**, where vvc
is 5.3x cheaper. But the BP24 ladder's reference rung puts the plate near
**43 dB**, and there both modern codecs come in at **x0.691 — a 1.45x saving, not
3.6-4.1x**. The ratio falls monotonically as fidelity rises across the whole
measured band, which is the same crossover shape
`plans/BP29-plate-codec-report.md` §3 found between jpeg and x264 at ~40 dB.

So lever (a) is real and much smaller than the plan assumes: at the ladder's
operating point it takes about **31% off the plate**, hence roughly **27% off the
total payload** at an 88% background share — worth having, not the transformation
§2.21 implies.

**av1 and vvc are indistinguishable at 43 dB** (243,861 against 243,617 B, 0.1%
apart). vvc leads at every coarser target. Neither preset is a matched-effort
setting, so this is what *this plate* costs under each encoder as configured, not
a statement that one codec beats the other.

### Time, measured rather than interpolated — and only good to an order of magnitude

| codec | median encode over the curve | range | worst within-point spread |
|---|---:|---:|---:|
| jpeg | **0.018 s** | 0.016-0.023 | 0.005 |
| av1 | **12.12 s** | 9.62-15.53 | **12.96** |
| vvc | **9.94 s** | 7.42-12.96 | **17.40** |

**The within-point spread is larger than the range across the whole knob range**,
so on this shared host encode time is dominated by co-tenancy rather than by the
quantiser. Two consequences, both stated rather than smoothed:

- **Encode time is not interpolated against quality anywhere in this harness.**
  The first run did interpolate it, through an av1 point that measured 34.62 s
  where its neighbours took 9-12 s, and produced a headline "x1724 slower at
  43 dB" that was an artefact of one contended sample. Each point is now the
  median of three repeats, every sample is kept in the record, and the time
  column was removed from the matched-fidelity table.
- **av1 and vvc cannot be separated on time by this measurement.** 12.12 against
  9.94 s, with spreads of 13-17 s, supports no ordering. What it does support is
  the order of magnitude: both are **500-700x** jpeg's encode cost, and that gap
  is far larger than any noise here.

**Bytes and Y-PSNR are deterministic and reproduced exactly** across two full
runs (av1 qp20: 507,138 B at 43.71 dB both times; vvc qp48: 18,321 B at 32.63 dB
both times), and every repeat's payload is asserted byte-identical inside the
harness. So the size half of this table is solid and the time half is an
order-of-magnitude reading. They are reported with different confidence because
they were earned with different confidence.

### A bound fired and was wrong, again in its basis rather than its interval

The pre-written vvc encode-time floor of 5.0 s fired on three points at
2.9-4.6 s. Its stated basis was that "VVC intra at 4K is the slowest thing on
this roster" — and that is simply false here: `vvencapp` at `faster` is *quicker*
than `SvtAv1EncApp` at preset 10 on this plate. Four things say vvc ran: bytes
and Y-PSNR both monotone in its own knob over eight points, a decodable
bitstream, reproduction across two runs, and BP24's independent plate probe
measuring vvc intra at 68,477 B near 38 dB where this curve brackets it
(41,330 B at 37.46, 60,242 B at 39.32). The floor was corrected to 1.0 s with
the reason recorded in the module.

### What this does not decide

Nothing about `panorama-stream`, which never consults this sidecar (§1). This
prices the plate codec for the `panorama-full` arm only.

## 11. The third dimension: PointStream is ~20x the anchor's wall clock

`AGENTS.md` now requires every result to carry size, quality **and** speed, and
`PLAN.md` §5 item 1 had asked for it already. The wall clock was recorded per
rung all along and simply never reached the table;
`experiments/tier/ladder_scenes_compare.py` now prints it beside the BD-rate.

| arm | BD-rate vs av1 | wall clock over the curve | vs anchor |
|---|---:|---:|---:|
| `panorama-full` | +109.72% | 2,443 s | **x19.1** |
| `panorama-stream` | **+90.97%** | 2,686 s | **x19.7** |
| the anchor itself | — | 128-136 s | x1 |

**This makes the picture worse, and it is the honest picture.** PointStream at
its best configuration here costs roughly **twice the rate and twenty times the
encode time** of the codec it is built on. A table with two columns could not
say that, which is exactly the rule's point: cheaper-and-better-but-ten-times-
slower is a different result from cheaper-and-better-and-as-fast.

**What each number covers**, because they are not the same quantity. The
anchor's is one `coded_roundtrip` over the concatenated scenes — encode plus
decode of the source. PointStream's is a whole `run()`: every encode-side stage,
the residual's codec, and the client reconstruction. So this is *wall clock to
produce the delivered clip on this host*, not encoder against encoder. The
anchor's job really is smaller, and the ratio should be read with that in mind
rather than as a codec speed comparison.

**Confidence: an order of magnitude, not a measurement.** Single sample per rung
on a shared host, where §10 measured a within-point spread on repeated 4K
encodes *larger than the range across a whole knob sweep*. A 1.2x difference in
this column means nothing. x19 means something.

**Where it bites.** The cross-scene stream buys 19 BD-rate points (§9) and costs
about 10% more wall clock (2,686 s against 2,443 s) — a good trade on its own
terms. The x20 gap against the anchor is structural, not the stream's doing.

## 12. Span: the amortisation is real, and it collides with the stream at span 24

Run on the brief a parallel session sent (`plans/BP33-span-amortisation.md`),
whose bounds were **adopted verbatim** into
`outputs/bp31-ladder/bounds-before-span-run.json` — they were pre-registered
before that brief reached this session, and bounds written after hearing a
prediction are not bounds. Span is the only axis that moves; scene count is held
at N=2, the value §9's BD-rate was taken at.

**First: the axis needed no extraction at all.** The BP21 cache already holds 48
frames per clip and both scenes load at 48 with tracks intact and paste-back MAE
0.000. Every ladder in this project has run at 8.

### The amortisation is real, at the two spans that ran

| span | anchor B/frame | PointStream B/frame | ratio | plate | bg share |
|---:|---:|---:|---:|---:|---:|
| 8 | 28,359 | 61,556 | **2.17x** | 789,304 B | 0.801 |
| 16 | 16,558 | 33,328 | **2.01x** | 768,277 B | 0.720 |

Both arms get cheaper per frame, PointStream slightly faster, and the background
share falls 0.80 → 0.72. **The plate does not grow — it shrank 2.7%**, which is a
temporal median over more samples converging.

**That is a bound excursion, downward, and it is the band that was wrong.**
BP33's static-clip band is [1.00x, 1.60x]; the plate measured **0.9734x** at 16
frames against 8, below the floor. The band is written for 48-vs-8 so it does not
strictly apply at this span, but the sign is already the other way and the reason
generalises: the band is **one-sided upward**, because it modelled the only
span-dependent effect on the plate as canvas growth. A second effect runs the
other way — a median over 16 samples is cleaner than one over 8, so the plate
gets *easier to code* as the span lengthens. On a near-static camera that term
wins. The interval is not what needs revising here; the model behind it is, and
the corrected form is two-sided with the noise term named.

The asymptote is worth stating because it is the argument's actual shape: at
span 16 the plate is 768,277 B over 32 frames = 24,009 B/frame of PointStream's
33,328, so the non-plate cost is ~9,319 B/frame. As span grows the plate's
per-frame share tends to zero and the ratio tends toward non-plate over anchor.
That is why the axis matters — but it is an extrapolation until a long span
actually runs, and the next section is why one did not.

### Span 24 and beyond: the stream refuses, and the reason is structural

```
span 24  FAILED  scene 1 is (2172, 3881, 3), the stream is (2161, 3841, 3);
                 inter prediction needs a fixed frame size
```

`BackgroundStreamTransmitter.push` (`src/components/background/stream.py:456`)
requires every plate in a chain to have identical shape. `build_plate` sizes the
canvas from the homographies over the span, so a scene's canvas grows with *its
own* camera motion — and the two scenes do not move alike:

| span | scene_000 (static) | scene_010 (pans) | stream |
|---:|---|---|---|
| 8 | 2161x3841, x1.0007 | 2161x3841, x1.0007 | ok |
| 16 | 2161x3841, x1.0007 | 2161x3841, x1.0007 | ok |
| 24 | 2161x3841, x1.0007 | **2172x3881, x1.0163** | refuses |
| 32 | 2161x3841, x1.0007 | **2189x3919, x1.0343** | refuses |
| 48 | 2161x3841, x1.0007 | **2190x3932, x1.0382** | refuses |

**The divergence is asymmetric, and that is the whole mechanism.** The static
scene's canvas never grows; the panning scene's does. At short spans neither has
moved enough to matter and the shapes coincide by accident. Past ~24 frames the
panning scene has swept far enough that its canvas is genuinely bigger, and the
chain cannot hold two frame sizes.

**So BP33's growth worry is unfounded and its blocker is somewhere else.** The
canvas grows only x1.034 at span 32 against a `MAX_CANVAS_SCALE` of 4 — the plate
is not running away. What breaks is not size but *size agreement between scenes*,
which no bound in the brief covers because nobody had run the two levers together.

**What it would take.** A canvas common to the whole run — every scene's plate
padded to the union of the run's canvases — so the chain sees one frame size.
That is a change to `stream.py` and `plate.py`, i.e. a component change with
BP30's tests around it, and it is the prerequisite for measuring span past 16
under `panorama-stream`. Until then span and the cross-scene stream are
combinable only up to 16 frames.

**Two clean ways to get the span number sooner**, both avoiding the component
change: run the span ladder under `panorama-full`, where each scene codes its own
plate and no shape agreement is needed; or run `panorama-stream` at N=1, where
there is no second scene to disagree with. Neither answers the combined question,
and the combined question is the one that matters.

