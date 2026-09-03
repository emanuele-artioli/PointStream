# BP40 — The background component: an honest ledger, a tunable lever, and levers that compose

**Three findings the BP31 session deliberately left alone**, because folding them
into a run would have meant changing the thing being measured while measuring it.
They are recorded in `plans/done/BP31-findings.md` §§1 and 7 and they do not go away.

**Owns:** `src/components/background/**`, `tests/components/test_background*.py`.
**Status 2026-09-03:** PR #45 and canonical-canvas integration are merged.
This brief is a deferred audit of remaining knobs, not a merge blocker; compare
each historical diagnosis with current code before implementing it. BP49 is next.

**Read first:** `AGENTS.md` (a flag existing is not a feature working) ·
`plans/done/BP31-findings.md` §§1, 7 · `plans/done/BP29-plate-codec-report.md` §4 ·
`plans/done/BP30-background-stream.md` · `PLAN.md` §2.21.

---

## 1. The reporting trap — fix this first, it is one line and it backs wrong claims

`BackgroundModel.__init__` runs `normalize_sidecar(config.background.codec)` and
stores it; `PanoramaStream.transmit` copies that name into
`BackgroundArtifact.codec`. Under `panorama-stream` the sidecar is never
consulted — the plate goes through `BackgroundStreamTransmitter` at
`stream_codec`/`stream_crf` — so **a run configured `{method: panorama-stream,
codec: av1}` reports `codec: av1` in its artifact while its bytes came from
libaom at a CRF.**

`codec_id` is honest, because `PanoramaStream` overrides it to report the stream
spec. `artifact.codec` is not, and it is the shorter name to reach for. Anybody
reading a ledger to check "did the intra sidecar arm run?" gets a plausible yes.

**Fix:** the artifact reports what produced the bytes, or reports nothing. Add a
test that drives both methods and asserts the reported codec matches the encoder
that ran — the same shape of test as `plans/done/BP31-findings.md` §1's probe, which
found this by driving three codec values and getting byte-identical payloads.

## 2. `intra_qp` reaches nothing, so lever (a) is a point and not an axis

`BackgroundConfig` has no `intra_qp` field, and `strategy.bind` forwards only
`codec`, `jpeg_quality` and `domain` plus the four stream knobs. So
`BackgroundModel.__init__` never passes an intra QP and `build_sidecar` falls
back to `DEFAULT_INTRA_QP = 45` on **every** runner path.

`background.codec: av1` is therefore a single fixed operating point. The same
limitation is recorded for `roi-video`'s `crf` in
`plans/done/BP29-plate-codec-report.md` §4. `PLAN.md` §2.21 calls the plate codec
"lever (a)" and it is not currently a lever in the sense of something you can
sweep.

**Fix:** plumb it, and add the invariant that every knob a config accepts changes
the bytes — driven, not read. `plans/done/BP31-findings.md` §1's probe table is the
template: a null result (three codecs, identical payloads) is only readable
beside a control where the same probe *does* separate them.

## 3. Levers (a) and (b) cannot compose, and that is architectural

`panorama-stream` overrides `transmit`, `decode_payload` and `reconstruct`, and
none of them touches `self._sidecar`. So the still-image codec and the
cross-scene stream are **mutually exclusive as implemented**, and the two largest
plate levers in `PLAN.md` §7 P0 item 8 cannot be on at the same time.

The natural shape is that **a keyframe is a still**: code the stream's keyframes
through the intra sidecar while P-frames go through the stream. The obstacle is
real — the transmitter owns the chain and its reconstructions, so the chain must
decode a sidecar-coded keyframe bit-exactly, and `BP30` has tests around that
component.

**Do not start this until `plans/BP32-rate-budget.md` has priced it.** BP31 §10
puts the intra sidecar at **x0.691** against jpeg at the ladder's rung — a 1.45x
saving on the plate, not the 3.6–4.1x `PLAN.md` §2.21 quotes — and the stream is
worth 0.646 over twelve scenes. Composing them is worth doing if the product is
worth the architectural change and not if it is not, and that is a number BP32
produces for free.

## 3b. A run-wide canvas — found 2026-09-02, and it gates the span axis

`plans/done/BP31-findings.md` §12 hit this running the span sweep. Spans 24, 32 and 48
refuse under `panorama-stream`:

```
scene 1 is (2172, 3881, 3), the stream is (2161, 3841, 3);
inter prediction needs a fixed frame size
```

`BackgroundStreamTransmitter.push` (`src/components/background/stream.py:456`)
requires one frame size across a chain, and `build_plate` sizes each scene's
canvas from *that scene's own* homographies. A static scene's canvas never grows;
a panning one's does. Below span ~24 the shapes coincide by accident; past it
they genuinely differ.

**It is a size-*agreement* problem, not a size problem**, and it exists only when
span and the cross-scene stream are on together — which is why no bound in
`plans/BP33-span-amortisation.md` covered it.

**The fix:** a canvas common to the whole run — pad every scene's plate to the
union of the run's canvases, so the chain sees one frame size.

**Bounds before building it.** Canvas growth is now measured, not guessed:
x1.0007 on a static scene at every span, **x1.038 at span 48** on a panning one,
against a `MAX_CANVAS_SCALE` of 4.

- **Padding to the union costs [1%, 8%] of plate area** on this content. Above
  that, the union is being computed over scenes that do not belong in one chain.
- **The padded region must not cost proportionally.** It is flat fill, which any
  intra coder handles almost for free, so **plate bytes should rise by less than
  half the area increase**. If bytes track area one-for-one, the padding is not
  flat and something is writing content into it.
- **Every existing single-canvas run reproduces to the byte.** This changes a
  component with BP30's tests around it; a published number that moves is a
  regression.

**Do the cheap thing first.** `BP33` §6 shows span amortises the plate toward
irrelevance, so this fix buys 4% of a term that is shrinking. **Run the span
points under `panorama-full` first** — no component change needed — and decide
from those numbers whether the combined question still justifies the change.

## 4. The bound this component keeps breaking

`plans/done/BP31-findings.md` §6 is worth re-reading before writing any bound here:
the background's *share* of the payload is a saturating function of the plate's
cost and **cannot detect a fix**, because the non-plate payload is ~9% on this
content. The share was chosen because `PLAN.md` §2.20 stated the problem in those
units, and *"the number that stated the problem and the number that detects the
fix are not the same number."*

**Bound plate bytes against a fresh-plate control at matched fidelity**, never
share, and never at a matched knob.

## Done when

- `BackgroundArtifact.codec` cannot name an encoder that did not run, and a test
  drives both methods to prove it.
- `intra_qp` reaches `build_sidecar` from config, with a driven test that moving
  it moves the bytes.
- The (a)/(b) composition is either implemented with its keyframe decode proven
  bit-exact, or declined in writing with BP32's number as the reason.
- The run-wide canvas is either implemented — with every existing single-canvas
  run reproducing to the byte — or declined with the `panorama-full` span numbers
  as the reason.
