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
