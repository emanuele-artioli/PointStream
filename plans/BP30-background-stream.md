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

**Findings §18's numbers were measured without that constraint. They have since
been re-measured under it, and they survive** (`plans/BP24-findings.md` §19):
av1's 0.671 and 0.470 ratios are unchanged **to the byte** with
`-lag-in-frames 0`, because `-usage realtime` was already lookahead-free. So the
31–53% is causal and the scheme is achievable live.

Two things that re-measure settled beyond the gate itself. **x265 is not av1
here** — it saves 12% on one pair and loses 6% on the other, so the saving is a
property of av1's inter tools rather than of inter coding in general. And
**strict causality wants zero rate-control lookahead, which x265 cannot usefully
provide**: for one plate per scene, a lookahead of N frames is a delay of N
*scenes*. av1 being genuinely lookahead-free is what makes this deployable.

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
| `best-scored` | search previous frames, pick the structurally closest | an edge pass |
| `periodic-i` | force an I-frame every *k* scenes regardless | free |

**Score by structure, not by pixel distance — Canny is the right shape of
proxy.** Findings §18 measured the *further apart* pair (federer, 15.10 dB)
saving **more** (47–53%) than the closer one (alcaraz, 13.75 dB, 31–33%), so
PSNR distance does not predict coding distance. That is not a curiosity; it is
the same mechanism that made pixel subtraction fail in §17. What a codec spends
bits on is **residual structure after motion compensation** — edges that do not
line up — and not the mean squared difference of two images.

So the reference to prefer is the one whose *edge map* matches, and a Canny
edge map compared by IoU (or a Chamfer distance, which degrades more gracefully
when edges are near but not coincident) is a cheap stand-in for exactly that. It
also costs almost nothing: one Canny pass per candidate frame, no encoding.

**Validate the proxy before trusting it**, on the project's usual terms: on a
handful of candidate pairs, run the real trial encode *and* the Canny score, and
check they rank the same way. A proxy that has never been checked against the
thing it proxies is how a search ends up confidently picking the wrong
reference. If Canny does not track the trial encodes, say so and fall back to
`first` — which costs nothing and, per §18, is already worth 31–53%.

**`periodic-i` is an ablation, not a constraint.** A pure P-chain has no random
access and no loss resilience: a client joining mid-match cannot start, and
losing payload *n* breaks every payload after it. For a **research paper** that
is acceptable — the system is not being deployed, and assuming a reliable
channel is a normal scoping decision. So the keyframe interval is swept as an
axis rather than imposed as a floor.

**But the paper has to say so.** A rate that assumes no keyframes and no losses
is a rate under a stated assumption, and a reviewer will ask. The ablation
exists to make that conversation quantitative: reporting the cost of *k* = 2, 4,
8 and never lets the robustness discussion cite a number instead of a
hand-wave.

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

- ~~The low-delay re-measure must not be much worse than §18's 31–53%.~~
  **CLEARED 2026-08-29** (findings §19): unchanged to the byte for av1. The
  bound is kept rather than deleted because it fired usefully on the way — the
  first attempt misconfigured x265 and the control caught it at ratio 1.075.
  Keep the control in every future run of this.
- **The control must hold.** Two consecutive frames of one scene coded as I then
  P must land at a few percent (§18 measured 1.2–3.3%). If not, the harness is
  not measuring inter prediction.
- **A periodic I-frame every *k* scenes costs roughly `1/k` of a fresh plate.**
  With savings of ~40% on the P scenes, the break-even against all-intra is
  around *k* = 2, so any *k* ≥ 4 should still pay. If it does not, the
  arithmetic is wrong somewhere. Reported as an axis, not imposed.
- **`best-scored` must beat `first` by enough to pay for the search**, and its
  Canny score must rank candidates the same way trial encodes do. If the ranking
  disagrees, the proxy is wrong and the honest report says `first`; if it agrees
  but wins by under a few percent, `first` is still the recommendation and the
  search is complexity for nothing.
- **The paired BD-rate must be reported over N scenes on both arms.** A number
  that amortises PointStream's background over 20 scenes against an anchor given
  one scene is not a result; it is the rigged comparison §5 exists to prevent.

## Done when

The background is transmitted as a low-delay stream across scenes, each scene's
payload independent of every future scene; the four reference modes are ablated,
with the keyframe interval swept as an axis and its cost reported so the paper's
robustness paragraph can cite a number; and the paired BD-rate over N scenes is
reported against an anchor given the same N scenes under the same constraint —
or the report says precisely which of those failed and why.
