# BP40 — The background component: an honest ledger, a tunable lever, and levers that compose

**Three findings the BP31 session deliberately left alone**, because folding them
into a run would have meant changing the thing being measured while measuring it.
They are recorded in `plans/BP31-findings.md` §§1 and 7 and they do not go away.

**Owns:** `src/components/background/**`, `tests/components/test_background*.py`.
**Blocked on PR #45 merging** — file ownership, not results.

**Read first:** `AGENTS.md` (a flag existing is not a feature working) ·
`plans/BP31-findings.md` §§1, 7 · `plans/BP29-plate-codec-report.md` §4 ·
`plans/BP30-background-stream.md` · `PLAN.md` §2.21.

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
that ran — the same shape of test as `plans/BP31-findings.md` §1's probe, which
found this by driving three codec values and getting byte-identical payloads.

## 2. `intra_qp` reaches nothing, so lever (a) is a point and not an axis

`BackgroundConfig` has no `intra_qp` field, and `strategy.bind` forwards only
`codec`, `jpeg_quality` and `domain` plus the four stream knobs. So
`BackgroundModel.__init__` never passes an intra QP and `build_sidecar` falls
back to `DEFAULT_INTRA_QP = 45` on **every** runner path.

`background.codec: av1` is therefore a single fixed operating point. The same
limitation is recorded for `roi-video`'s `crf` in
`plans/BP29-plate-codec-report.md` §4. `PLAN.md` §2.21 calls the plate codec
"lever (a)" and it is not currently a lever in the sense of something you can
sweep.

**Fix:** plumb it, and add the invariant that every knob a config accepts changes
the bytes — driven, not read. `plans/BP31-findings.md` §1's probe table is the
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

## 4. The bound this component keeps breaking

`plans/BP31-findings.md` §6 is worth re-reading before writing any bound here:
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
