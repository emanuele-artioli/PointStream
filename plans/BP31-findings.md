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
