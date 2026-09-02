# BP43 — How to encode the background, and how small it can get

**The plate is 88–91% of the payload.** Every lever tried so far has changed *how
it is coded* or *how often it is paid for*. **Nobody has tried making it smaller
in pixels, and nobody has tried not sending it at all.** Those are the two
untested directions and one of them is structural.

**Owns:** `src/components/background/sidecar.py`, `src/contracts/config.py`
(`BackgroundConfig` only), `outputs/bp43-background/**`.
**Blocked on PR #45 merging** for the component files; the resolution sweep in §2
can be prototyped standalone before then.

**Read first:** `AGENTS.md` · `plans/BP29-panorama-report.md` §§3, 4 ·
`plans/BP29-plate-rate.md` · `plans/BP32-rate-budget.md` · `PLAN.md` §7 P0 item 8
and P2 item 15.

**Every arm reports size, quality *and* speed.** A plate that is cheaper and
sharper but takes 40x longer to build is a different answer from one that is not.

---

## 1. What has actually been tested — the honest scoreboard

Four ways of representing the background have been driven end to end. Two more
are named in the code and have never been measured.

| Representation | Status | What it measured |
|---|---|---|
| **Send frame 0** (`span=1`, the keyframe control) | ✅ tested | 345,947 B static / 393,879 B moving, jpeg q50, 4K |
| **Temporal median, no registration** — one frame averaging the span | ✅ tested | plate **0.99x** static, **0.84x** moving; but **−3.7 dB** on the moving clip |
| **Registered panorama + per-frame homographies** (the shipped default) | ✅ tested | plate 0.99x, **residual 0.22x**, delivered **+4.9 to +6.2 dB** on the moving clip |
| **Plate coded as a video across scenes** (`panorama-stream`, P-frames) | ✅ tested | **0.646** over twelve scenes, av1 crf38 |
| **`roi-video` sidecar** — the plate coded as a video *within* a scene | ❌ **never measured** | registered in `sidecar.py`; `roi_crf` and `roi_preset` are unreachable from config |
| **A smaller plate** — fewer pixels, upsampled at the client | ❌ **never possible** | `BackgroundConfig` has no resolution field at all |

So of the three schemes worth naming, **two are done and one is half done**:
"average the frames" and "panorama plus warp parameters" are both implemented and
compared against each other, and "encode it as a video" exists only across scenes,
never within one.

**The most useful thing in that table is the second row.** The unregistered
median gives a **16% smaller plate on the moving clip** and costs **3.7 dB**.
Registration is therefore already a priced rate-quality trade, and it is the only
one in the background component that has both numbers.

## 1b. How much this brief is worth, after the span run

**Read `plans/BP33-span-amortisation.md` §6 before spending anything here.** Two
measurements shrink the prize:

- **Span drives the plate's per-frame cost toward zero.** It is 80% of the
  payload at span 8, 72% at 16, and falling. Every lever in this brief acts on a
  term that a free flag is already shrinking.
- **The plate codec lever is smaller than `PLAN.md` §2.21 assumed.** At the
  ladder's operating point (~43 dB) av1 and vvc intra are **x0.691** against
  jpeg, not the 3.6–4.1x quoted from a single-point comparison at 38 dB
  (`plans/BP31-findings.md` §10, which corrected §2.21 in place). So a resolution
  lever multiplies against a base already ~31% smaller than the plan assumed.

**This does not make the brief pointless** — the plate is a real cost at any
finite span, and a 4x pixel reduction on it is still the largest single
compression available on that term. It does mean **this is no longer the item
that decides the paper**, and it should be sized accordingly: one sweep, not a
campaign, and after the `panorama-full` span points have confirmed or falsified
the marginal-cost picture.

## 2. Lever A — make the plate smaller in pixels

**This is the direct answer to "reduce the background size", and it has never
been available.** `BackgroundConfig` carries `method`, `codec`, `jpeg_quality`,
`reference_mode`, `keyframe_interval`, `stream_codec`, `stream_crf` — and no
resolution. Every plate ever transmitted has been at full source resolution.

The case for it is that the background is the low-frequency part of the picture
by construction: crowd, court, sky, stands. It is also the part the client
resamples anyway — every frame is produced by *warping* the plate through a
homography, which is already an interpolation. Sending it at half resolution is
a 4x pixel reduction on 88–91% of the payload, and the warp absorbs the
upsampling that the client has to do regardless.

`PLAN.md` §7 P2 item 15 is "JPEG quality versus downscaling", promoted into P0
item 8 in August and never run. This is that item.

**What to run.** Plate scale ∈ {1.0, 0.75, 0.5, 0.35, 0.25}, crossed with two
quality rungs, on one static and one moving clip. Report the plate bytes,
delivered Y-PSNR *and* VMAF, the residual's response, and the wall clock of both
the encoder's plate build and the client's warp.

**The measurement this must not get wrong.** Downscaling the plate makes the
residual work harder, so **plate bytes alone will look like a triumph and mean
nothing** — the same error `plans/BP29-plate-codec-report.md` §3 was written
about, at a different knob. Read **total payload at matched delivered quality**,
or read the whole RD curve. The plate-bytes column is diagnostic, not the result.

**Bounds, two-sided, before the first encode:**

- **Plate bytes at scale 0.5 land in [0.20x, 0.45x] of full scale.** Below 0.20x
  means the encoder is spending almost nothing on a nearly-empty image and the
  quality column will say so; above 0.45x means jpeg/av1 was already discarding
  the high frequencies the downscale removes, and the lever is smaller than the
  pixel count suggests — which would itself be the finding.
- **Delivered Y-PSNR falls by [0.5, 6.0] dB at scale 0.5** before the residual
  compensates. Outside that, check the upsample is happening where you think.
- **Total payload at matched quality improves at some scale between 1.0 and
  0.35.** If it never improves, downscaling is not a lever on this content and
  that closes P2 item 15 with a real answer.

## 3. Lever B — send a coarse plate and let the residual refine it

**A retraction first, because the original version of this section was wrong and
the reason is worth keeping.** It proposed not sending the plate at all: *"the
client already receives decoded frames, and the plate is a median composite of
registered frames, so the client can build it too."*

**There are no such frames.** In a PointStream scene the client's frames *are*
the reconstruction — warp the plate, composite the foreground, add the residual.
Building the plate from them requires the plate. The proposal was circular, and
a version of it that were not circular would mean transmitting frames
conventionally, which is the thing this codec exists not to do.

**The two things that do reach the client independently of the plate**, and what
each is worth:

- **Scenes routed to the conventional codec** by scene classification. Using one
  of those as plate content for a neighbouring PointStream scene is already
  measured and dead: `plans/BP24-findings.md` §17 puts two point-class plates
  from the same match **13.75–15.10 dB** apart, and coding one against the other
  costs **1.49–1.70x** the bytes at **13 dB lower** quality — dominated on both
  axes, so no ladder is needed. The version of that idea that works is coding the
  next plate as a **P-frame**, which is `panorama-stream` (§18, 0.646 over twelve
  scenes) and is already the shipped path.
- **The residual.** This is the one that survives, and it is a much smaller claim
  than the retracted one.

### What survives: the plate as a long-term reference

The residual is the only channel carrying true background detail that the plate
did not already have. So a closed loop is available and is not circular:

1. transmit a **coarse** plate — cheap;
2. each frame's residual corrects what the coarse plate got wrong, which injects
   real information;
3. accumulate those corrected reconstructions into an **updated** plate;
4. later frames warp from the better plate and need a smaller residual.

This is a long-term reference picture, updated from the decoded output — the same
mechanism a conventional codec uses, applied to the plate. The information comes
from the residual, not from nowhere, so there is no free lunch: it is a **rate
allocation** question. Is *coarse plate + larger early residual* cheaper than
*fine plate + small residual throughout*, at matched delivered quality?

**Which means it is not a separate lever — it is the coarse end of §2.** The
resolution and quality sweep in Lever A already samples "cheap plate, residual
does more work". **Run Lever A first.** Only if its coarse end is competitive is
there anything for a refinement loop to improve, and the sweep will say so for
free.

**If Lever A says the coarse end is competitive**, bound the loop before building
it: the refined plate should reach within **[0.5, 3.0] dB** of a
transmitted-at-full-quality plate by frame **[8, 24]**, and total payload at
matched quality should improve by **[5%, 30%]** over the best fixed-plate rung.
No improvement means the residual is not carrying enough background information
to refine anything, which would itself close the question.

**The decoder-drift trap, which is what makes this real work rather than
plumbing.** Encoder and client must build the *same* refined plate or they
diverge, and the divergence compounds every frame. The encoder therefore has to
run the client's loop on the client's decoded output, not on source frames —
which is the architecture the paper already describes for the residual
("the server runs the same reconstruction the client will"), and which
`PLAN.md` §3's *quality is always measured* section says must be **verified by
measurement, not asserted by construction**. Any implementation needs a
bit-identity check between the two plates at every frame, and that check is the
first thing to write.

## 4. Lever C — the `roi-video` sidecar, and why it is last

`SIDECAR_ROI_VIDEO` is registered and has never been driven; `roi_crf` and
`roi_preset` cannot be set from a config file (`plans/BP29-panorama-report.md`
§2). Coding a *single still plate* as a one-frame video is unlikely to beat the
av1/vvc intra sidecars, which `plans/BP31-findings.md` §10 already prices at
**x0.691** against jpeg at the ladder's rung.

Its real use is different: **a plate that updates during a scene.** The crowd
moves, the scoreboard changes, the shadow travels. Today the plate is one static
image for the whole scene and the residual pays for everything that changed.
Coding the plate as a short video — a few frames across the scene, inter-coded —
would let it track those changes at inter-frame prices.

That is a genuine fourth representation and it sits between "one averaged frame"
and "a video of the whole background". Price it only after levers A and B, and
only if `plans/BP32-rate-budget.md`'s ledger shows the residual carrying enough
background change to be worth chasing.

## 5. Ordering, and why

1. **Lever A's resolution sweep** — one sweep, the direct answer to "how small
   can the background get", and it also decides whether Lever B has anything to
   work with.
2. **Lever B**, only if Lever A's coarse end is competitive. It is a rate
   allocation question, not a way to stop paying for the plate.
3. **Lever C** — only if the ledger says the residual is paying for background
   change.

## Done when

- The resolution axis exists in `BackgroundConfig`, is driven (not read off), and
  has an RD curve at matched quality with encode and decode time beside it.
- Lever B is either taken up with a bounded plan and an encoder/client
  bit-identity check, or declined with Lever A's coarse-end numbers as the
  reason.
- The scoreboard in §1 has no "never measured" rows left, or each remaining one
  has a recorded reason.
