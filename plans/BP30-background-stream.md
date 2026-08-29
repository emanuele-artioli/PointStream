# B'30 — The background as a stream, not a still per scene

**Why this exists.** The plate is 88–91% of PointStream's payload
(`plans/BP24-ladder-report.md`), and every scene currently pays for it from
scratch. `plans/BP24-findings.md` §18 measured that coding the next plate as a
**P-frame** against a previous one saves **31–53%** with av1 — where pixel
subtraction had cost *more*, which is why §17 wrongly closed this door.

This brief is the design for doing it properly, because the idea has four
independent parts that have to fit together and each of them can be got wrong
quietly.

**Read first:** `plans/BP24-findings.md` §§13, 16, 17, 18 ·
`plans/BP29-plate-rate.md` §3b · `src/components/background/` ·
`src/contracts/config.py` (`BackgroundConfig`).

---

## 1. The question of independence, which decides everything else

**Can each scene's payload be sent on its own, or must scenes be encoded into
one file?** If the latter, PointStream cannot claim a live scenario — the
encoder would have to see the whole match before emitting the first payload,
and that is not a codec, it is an offline archiver.

**It can be causal, and this is not a special trick — it is how every live
encoder already works.**

A P-frame references the **reconstructed** reference, not the original. Both
sides must hold the identical reconstruction, and both can, without any
knowledge of the future:

1. **Scene 1.** Encoder codes plate₁ as an I-frame, then *decodes its own
   output* — the closed loop every video encoder already runs — and keeps
   `platê₁`. Payload 1 is that I-frame's bytes. The client decodes it and now
   also holds `platê₁`. **Both sides hold the same reconstruction.**
2. **Scene 2.** The encoder codes plate₂ as a P-frame against `platê₁`. Payload
   2 is *only* that P-frame's bytes. The client decodes it against the
   `platê₁` it already has.
3. And so on. Nothing at step *n* depends on anything after *n*.

**The constraint that makes it true, and it is easy to lose.** Causality holds
only if the encoder is forbidden from looking ahead:

- **No B-frames.** A B-frame references a *future* picture by construction.
- **No lookahead / no multi-pass.** Both let frame *n*'s decisions depend on
  frame *n+1*.
- Concretely: low-delay P configuration. `-bf 0`, and the encoder's own lookahead
  set to zero (`x265-params bframes=0:rc-lookahead=0`, `libaom-av1` with
  `-usage realtime -lag-in-frames 0`, SVT-AV1 `--pred-struct 1 --lookahead 0`).

**Findings §18's numbers were measured without that constraint**, on two-frame
encodes where there was no future frame to look at anyway. At two frames it
makes no difference; over twenty scenes it will. **Re-measure under low-delay
before quoting a saving as achievable live** — that is the first alarm in §6.

**How the bytes actually get split.** Keep one encoder process alive and feed it
plates as scenes arrive; its output is a sequence of packets, one per frame, and
an elementary stream (Annex-B for AVC/HEVC, OBU for AV1, IVF) delimits them
without a container. Ship packet *n* as payload *n*. This is ordinary low-latency
streaming; the "single video file" is a storage convenience, not a requirement.

**CMAF is the wrong tool** and worth naming so nobody reaches for it: CMAF
fragments are designed to be *independently decodable*, each starting at an IDR,
precisely so a player can switch representations. That is the opposite of what
is wanted here.

**For measurement, a growing re-encode is equivalent and much simpler.** Encode
plates 1..n as one low-delay sequence and read per-frame sizes with `ffprobe`;
under low-delay P the bytes for frame *n* are identical to what a live encoder
would have emitted, because the encoder had no future information either way.
Build the streaming path only if a live demo needs it.

## 2. Panoramas have different resolutions. Two ways out

A stitched panorama's canvas depends on how far the camera travelled, so two
scenes give two different sizes, and inter prediction needs a fixed frame size.

**Option A — normalise the canvas.** Fix one canvas per *video* (or per camera
setup), sized to the union of the scenes' panoramas, and place each panorama in
it with the homography that registers it. Every plate is then the same size and
the sequence is a video. Costs: a registration pass, and undefined border
regions which must be filled consistently on both sides (fill with the previous
reconstruction, not with grey — grey is an edge, and edges cost bits).

**Option B — skip the panorama; code the background as a video.** This is what
the architecture already implies and what §3 makes attractive: rather than
compositing a still per scene, transmit background *frames* through a codec and
let inter prediction do the amortisation, including across scene boundaries.
Frame size is the video's, so the problem does not arise.

**Recommendation: B first.** It needs no registration, it removes the resolution
problem entirely, and it tests the same underlying claim. A is the more
ambitious version and should follow only if B shows the amortisation is real.

## 3. Which frame to reference — the ablation

With Option B the encoder chooses what the new scene predicts *from*. That
choice is an axis, and the cheap and expensive ends differ a lot in cost:

| mode | how the reference is chosen | cost to encoder |
|---|---|---|
| `first` | the previous scene's **first** background frame | free |
| `last` | the previous scene's **last** background frame | free |
| `best-scored` | search previous frames, pick the most similar | a similarity pass |
| `periodic-i` | force an I-frame every *k* scenes regardless | free |

Two things make this a real ablation rather than a formality:

- **PSNR distance does not predict coding distance.** Findings §18 measured the
  *further apart* pair (federer, 15.10 dB) saving **more** (47–53%) than the
  closer one (alcaraz, 13.75 dB, 31–33%). So a similarity search that ranks by
  PSNR may not pick the reference that codes best. **If `best-scored` is
  implemented, score it by trial encode or by a proxy validated against trial
  encodes — not by PSNR alone, on the assumption that similar means cheap.**
- **`periodic-i` is not a baseline, it is a requirement in disguise.** A chain of
  P-frames means losing payload *n* breaks every payload after it, and gives no
  random access — a client joining mid-match cannot start. A real system sends a
  periodic I-frame, and that costs rate. **The ablation must price it**, or the
  measured saving belongs to a system nobody could deploy.

## 4. Where this goes in the codebase

`BackgroundConfig.method` already declares **`panorama-delta`** and nothing
implements it (`src/components/background/`). That is the slot. The work
divides cleanly:

1. **A stateful background transmitter.** Today `make_background` in
   `src/runner/stages.py` codes one plate per run with no memory. This needs to
   carry the previous *reconstruction* across scenes — and it must be the
   reconstruction, never the original, or encoder and client drift. The paper
   already commits to exactly this discipline for the residual
   (`sections/system_design.tex`: the residual is computed against the
   codec-decoded background, not the raw one); the same rule now applies one
   level up.
2. **Reference-selection strategy** as a named component on the existing
   registry pattern, with the four modes above as backends.
3. **Ledger.** `SizesBytes.panorama` becomes the *marginal* cost for scenes
   after the first. The first scene's I-frame is not free and must not be
   dropped from any total that spans scenes.
4. **Config.** `background.reference-mode`, `background.keyframe-interval`.

## 5. Fairness, which has to be designed in and not bolted on

**The anchor gets the same footage.** A codec encoding a multi-scene sequence
can also predict across a scene join. If PointStream is allowed to amortise its
background across scenes, the anchor must be run on the same concatenated
material, under the same low-delay constraint, or the comparison is rigged.

PointStream's *possible* asymmetry is that composited backgrounds are more
similar to one another than two arbitrary frames at a cut are. That is a
hypothesis, not a result, and this brief does not get to assume it. The paired
arms are: PointStream over N scenes, and codec X over the same N scenes, both
low-delay, both with the same keyframe interval.

## 6. Bounds — write to `outputs/bp30-background/bounds-before-run.json` first

- **The low-delay re-measure must not be much worse than §18's 31–53%.** If
  forbidding B-frames and lookahead collapses the saving to under ~10%, the
  saving was non-causal and the whole idea is unavailable live. **This is the
  first thing to check and the cheapest.**
- **The control must hold.** Two consecutive frames of one scene coded as I then
  P must land at a few percent (§18 measured 1.2–3.3%). If not, the harness is
  not measuring inter prediction.
- **A periodic I-frame every *k* scenes costs roughly `1/k` of a fresh plate.**
  With savings of ~40% on the P scenes, the break-even against all-intra is
  around *k* = 2, so any *k* ≥ 4 should still pay. If it does not, the
  arithmetic is wrong somewhere.
- **`best-scored` must beat `first` by enough to pay for the search.** If it
  wins by under a few percent, `first` is the honest recommendation and the
  search is complexity for nothing.
- **The paired BD-rate must be reported over N scenes on both arms.** A number
  that amortises PointStream's background over 20 scenes against an anchor given
  one scene is not a result; it is the rigged comparison §5 exists to prevent.

## Done when

The background is transmitted as a low-delay stream across scenes, each scene's
payload independent of every future scene; the four reference modes are ablated
with the keyframe interval priced in; and the paired BD-rate over N scenes is
reported against an anchor given the same N scenes under the same constraint —
or the report says precisely which of those failed and why.
