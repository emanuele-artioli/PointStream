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
