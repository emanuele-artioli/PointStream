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

## 3. Lever B — do not send the plate at all

**The structural idea, and it has never been written down anywhere in this
project.** The client already receives decoded frames. The plate is built by
median-compositing registered frames. **So the client can build it too.**

The paper already argues exactly this shape for the residual: *"the server runs
the same reconstruction the client will, so it can measure and correct what the
generative model gets wrong."* A client-side plate is the same principle applied
one component over. The encoder knows precisely what the client will have, so it
can build the identical plate from the identical inputs and code a residual
against it.

The consequence is the largest single number in the ledger: **the plate stops
being payload.** What is transmitted instead is the homographies (measured at
576 B for eight frames — under 0.15% of the payload) plus whatever frames the
client needs to bootstrap the composite.

**The obvious objection, and it is real:** during the bootstrap the client has
seen few frames and its plate is poor, so early frames get a large residual. That
makes it a *latency-versus-rate* trade rather than a free win, and it interacts
directly with `plans/BP33-span-amortisation.md` — a longer span is exactly what
makes a client-side composite good. It also has a low-delay cost that must be
reported, because the paper's anchor comparison is constrained on that axis.

**Bound it before building anything.** On a static clip, a client-side plate
built from the first *k* decoded frames should reach within **[0.5, 4.0] dB** of
the transmitted plate by **k ∈ [4, 16]**. Measure that convergence curve first —
it is a simulation over already-decoded frames, needs no new component, and it
decides whether the idea is worth implementing at all.

**Do this measurement before the implementation.** If the convergence is slow or
the plate never gets close, the idea dies for the cost of an afternoon.

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

1. **Lever B's convergence measurement** — an afternoon, no new component, and it
   is the only one that could remove the dominant cost entirely.
2. **Lever A's resolution sweep** — one sweep, and it is the direct answer to the
   question that prompted this brief.
3. **Lever C** — only if the ledger says the residual is paying for background
   change.

## Done when

- The resolution axis exists in `BackgroundConfig`, is driven (not read off), and
  has an RD curve at matched quality with encode and decode time beside it.
- The client-side-plate convergence curve exists and the idea is either adopted
  with a plan or declined with the number that declined it.
- The scoreboard in §1 has no "never measured" rows left, or each remaining one
  has a recorded reason.
