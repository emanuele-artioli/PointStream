# BP33 — Span: the amortisation axis nobody has swept

> **RAN 2026-09-02** (BP31 session, `plans/BP31-findings.md` §12). The mechanism
> is confirmed and **this brief's central expectation is wrong**. Read §6 below
> before acting on anything above it: span amortises both arms almost equally,
> so it narrows the gap by ~7% and then flattens. What it does do is expose the
> number the project's fate actually rests on.


**Every ladder in this project has run at eight frames per scene. The cache holds
forty-eight.** The plate is 88–91% of the payload and is paid once per scene
whatever the scene's length, so the frames-per-scene default is a direct divisor
on the dominant cost — and it has never been moved.

**Owns:** `outputs/bp33-span/**` and this brief. Until PR #45 merges it owns **no
code**: `experiments/tier/**` and `src/runner/stages.py` belong to the BP31
session. **The brief is the deliverable until then**, and it should reach that
session immediately, because it is about to commit an extraction campaign to a
frames-per-scene value.

**Read first:** `AGENTS.md` · `plans/BP32-rate-budget.md` ·
`plans/BP29-panorama-report.md` §4 and its closing "the span is the untested
axis" · `plans/BP24-ladder-report.md` (the payload tables) ·
`plans/BP31-findings.md` §§8, 9 · `plans/DEFERRED.md` D-PANORAMA-REOPEN axis 3.

---

## 0. Why this is not a detail

It is written down in three places already, and it has been read as a caveat
rather than as a lever every time:

- `plans/BP24-ladder-report.md`: *"eight frames is the least favourable
  amortisation a fixed plate cost can get"*, listed alongside two other caveats
  and then not acted on.
- `plans/BP29-panorama-report.md` §6: *"The span is the untested axis. Everything
  here is eight frames. The panorama's argument gets stronger with a longer span
  and the canvas grows with it; `make_background(span=...)` already takes the
  knob."*
- `plans/DEFERRED.md` D-PANORAMA-REOPEN, axis 3: *"A panorama's whole argument is
  amortisation, so measuring it on the shortest available clip understates it by
  construction."*

`experiments/tier/clip.py` has `n_frames: int = 8`; `experiments/tier/ladder.py`
and the new `ladder_scenes.py` both default `--frames 8`. The BP21 cache
(`outputs/bp21-headroom/clips/`) holds **48 frames per clip** — the same windows
`PLAN.md` §2.14 used to measure the headroom the system is being judged against.

**So the headroom was measured over 48 frames and the system is being scored over
8.** That is a like-for-unlike comparison sitting under every BD-rate this
project has published, and it is one flag away from being fixed.

## 1. The arithmetic, before any run

From `plans/BP31-findings.md` §9 — `alcaraz_highlights`, N=2 scenes, 8 frames
each, `panorama-stream`, av1 preset 10:

- plate 789,304 B; total 862,585 B; **non-plate 73,281 B over 16 frames =
  4,580 B/frame**;
- so the clip costs **53,912 B/frame**, of which 49,332 is amortised plate.

At 48 frames per scene the per-frame residual and homography cost should be
roughly unchanged, while the plate is paid the same number of times. The plate
does grow — the canvas follows the camera sweep — but `plans/BP29-panorama-report.md`
§4 measures that growth at **1.0184x frame area over 8 frames on the moving
clip** and **1.0007x on the static one**, against a `MAX_CANVAS_SCALE` of 4. Six
times the span on a pan-tilt broadcast camera is a materially larger canvas, but
not a six times larger one.

**The anchor improves too, and it must be given the same 48 frames.** Its gain is
smaller and comes from a different mechanism: one intra keyframe amortising over
six times as many inter frames. `outputs/bp24-ladder/av1-payload-lowmotion.json`
puts av1 at 85,995 B for 8 frames at QP 55; the keyframe is most of that.

Net expectation: **PointStream gains substantially more from span than the anchor
does**, and the current +90.97% BD-rate is quoted at the one length that
maximally disadvantages it.

## 2. Bounds — write to `outputs/bp33-span/bounds-before-run.json` first

Two-sided, and deliberately wide, because the quantity being bounded is exactly
the quantity the experiment exists to generalise past. Every one of these is a
prediction, made before the first encode, and recorded so that a good result
cannot be retrofitted as though it had been expected all along.

| quantity | band | an excursion means |
|---|---|---|
| **BD-rate at 48 frames, `panorama-stream`, N=2, av1** | **[−40%, +65%]** | below −40%: check the anchor was given all 48 frames jointly, per `BP31` §5's joint/separate control. Above +65%: the plate grew far more than the canvas measurement predicts, or quality decayed enough that the residual ate the saving. |
| plate bytes, 48 vs 8 frames, static clip | ~~[1.00x, 1.60x]~~ → **[0.85x, 1.60x]** | *revised 2026-09-02, see below* |
| plate bytes, 48 vs 8 frames, moving clip | ~~[1.05x, 2.50x]~~ → **[0.90x, 2.50x]** | above 2.5x: the canvas is running toward `MAX_CANVAS_SCALE` and the pan has left the plate's usefulness behind |
| anchor bytes per frame, 48 vs 8 frames | [0.45x, 0.90x] | outside: the anchor is being run per-chunk rather than over the whole span |
| **delivered Y-PSNR at frame 47 minus frame 0** | **[−4.0 dB, +0.5 dB]** | below −4 dB: the homography has drifted off the plate and the product is a reconstruction that visibly rots, which is a different product from a uniform one |

**Both plate bands were one-sided upward, and that was a modelling error, not a
tuning error.** Revised 2026-09-02 after the run measured **0.973x** at span 16
against span 8 — an **excursion below the floor**, not a pass. The original bands
modelled canvas growth as the only span-dependent effect on the plate, so they
could only run one way. A second effect runs the other way and won here: **a
temporal median over 16 samples is cleaner than one over 8.** More samples
average away sensor noise, compression dither and transient content, so the plate
gets *easier to code* as the span lengthens, and on a near-static camera that
term dominates the growth term entirely.

`plans/BP29-panorama-report.md` §4 had already measured that denoising term at
**16,032 B** against the extra-coverage term's 12,975 B on a moving clip at eight
frames — the two nearly cancelled there, and nobody carried the arithmetic
forward to what happens when the span grows. Both bands are now two-sided.

**The direction of the alarm matters here.** A BD-rate that comes back negative
is the result this project has been looking for, which is precisely why it gets
the extra check rather than the celebration: `AGENTS.md`'s asymmetry rule applies
at full force. Before reporting any win, confirm the joint/separate anchor
control from `plans/BP31-findings.md` §5 still reads below 1.0, and that
`delivered_frames` — not `RunResult.frames` — is what quality was measured on.

## 3. What to run

One sweep, one long-lived process, detached.

1. **Span ladder on the continuity clip.** `alcaraz_highlights/scene_000`, frames
   per scene ∈ {8, 16, 24, 32, 48}, both arms, av1 preset 10 on both, the anchor
   encoding the same span jointly. This is a curve, not two points, because the
   interesting question is whether the gain saturates and where.
2. **Repeat on the dynamic end of the motion range.** `federer_djokovic/scene_003`
   (inter-frame MAD 7.70 against `alcaraz_highlights`'s 0.33) is the clip where
   the canvas actually grows, so it is where the span argument is most likely to
   break. One clip per motion regime is not a corpus and this brief does not
   claim it is — it is deciding an operating point, not publishing a result.
3. **Quality against frame index**, per rung, for both arms. Not the clip mean.
   A mean hides monotone decay, and monotone decay is the failure mode a longer
   span is most likely to have.
4. **The plate's canvas and byte cost per span**, recorded, so the growth term in
   `plans/BP32-rate-budget.md` stops being an extrapolation.

## 4. What it decides

- **The frames-per-scene value for every ladder from here on**, including BP31's
  N-scene campaign. That campaign is the expensive one; this sweep is cheap and
  runs first.
- **Whether span belongs in `plans/FORK-bp31.md`'s branch A as part of the named
  winning regime**, and if so, whether "a longer static shot" is a content
  requirement the paper must state as a boundary rather than a free choice.
- Whether `PLAN.md` §7 P0 item 8's three plate levers should become four.

## 5. The trap this brief exists to avoid

**Do not sweep span and scene count at the same time.** They are both
amortisation axes on the same fixed cost, they will interact, and a two-axis
sweep run once will not separate them. Span first, at N=2, because span is the
cheaper axis to move and the cache already holds the frames; scene count second,
at the span this sweep chooses. `BP30` measured one lever on one video, drew two
conclusions, and both inverted at five videos — the lesson taken from that was
"use more videos", and the other half of it is "move one thing".

---

## 6. What the run found — and where this brief was wrong

`plans/BP31-findings.md` §12. Bounds adopted verbatim and pre-registered to
`outputs/bp31-ladder/bounds-before-span-run.json` before the brief arrived.

### Held

- **The axis needed no extraction at all.** Both cached scenes load at 48 frames
  with tracks intact, paste-back MAE 0.000. Every ladder in this project has run
  at 8 for no reason but a default.
- **The amortisation is real.** N=2, reference rung, `alcaraz_highlights`:

  | span | anchor B/frame | PointStream B/frame | ratio | plate | bg share |
  |---:|---:|---:|---:|---:|---:|
  | 8 | 28,359 | 61,556 | 2.17x | 789,304 B | 0.801 |
  | 16 | 16,558 | 33,328 | **2.01x** | 768,277 B | 0.720 |

- **The plate does not grow.** Canvas is x1.0007 at every span on the static
  scene and tops out at **x1.038 at span 48** on the panning one, against a
  `MAX_CANVAS_SCALE` of 4. Growth stops after span 32.

### Wrong, and it was this brief's central expectation

§1 said *"PointStream gains substantially more from span than the anchor does"*.
Measured over the same doubling, **PointStream improved 1.85x and the anchor
1.71x** — the ratio moved 2.17 → 2.01, about **7%**. Not substantially more.

The reason is structural and should have been obvious when the brief was
written: **the anchor has a fixed per-scene cost too.** Its intra keyframe is
exactly the same shape of cost as PointStream's plate, and span amortises both.
The brief modelled only one side of that.

The plate-growth bound was wrong in shape — [1.00x, 1.60x] is one-sided upward,
so the measured **0.973x** is an **excursion below the floor, not a pass**. The
band modelled canvas growth as the only span-dependent effect on the plate; a
temporal median over more samples is *cleaner*, so the plate gets easier to code
as span grows, and on a near-static camera that term wins. `BP29` §4 had already
priced that denoising at 16,032 B against 12,975 B of extra coverage at eight
frames — the two nearly cancelled, and nobody carried it forward. Both bands are
now two-sided in §2. Recorded per `AGENTS.md`: when a bound turns out wrong,
record why.

### What the run exposes instead, and it is the important part

Decompose both arms into a **fixed** cost and a **marginal** per-frame cost. The
anchor's marginal cost is the difference quotient between the two span points, so
it is **independent of how many keyframes the joint encode used** — 1, 2 or 4
give the same slope:

| | fixed (amortised by span) | **marginal, per frame** |
|---|---:|---:|
| av1 anchor | ~382,000 B (intra) | **4,757 B** |
| PointStream | 768,277 B (plate) | **9,319 B** (residual + crops + metadata) |

> **The ratio of marginals is 1.96x.**

Span drives both fixed terms toward zero per frame. What is left is the marginal
comparison, and **PointStream's per-frame payload is about twice what av1 spends
coding an entire inter frame** — a frame that includes the players PointStream is
sending separately.

**So span cannot close the gap, at any span.** Extrapolating the same fit:

| span | anchor B/f | PointStream B/f | ratio |
|---:|---:|---:|---:|
| 16 (measured) | 16,558 | 33,328 | 2.01x |
| 24 | 12,624 | 24,300–25,600 | 1.93–2.03x |
| 48 | 8,691 | 16,200–17,500 | 1.86–2.01x |
| → ∞ | 4,757 | 8,000–9,319 | **1.68–1.96x** |

**This retires "the plate is the whole problem".** That framing is a *span-8
artifact*: at 8 frames the plate is 80% of the payload, at 16 it is 72%, and it
keeps falling. Run at a span the cache already supports and the dominant cost is
**the residual and the crops**, which nothing in `PLAN.md` §7 P0 item 8 is about.

### The falsifier, and it is cheap

This is a two-point linear fit and it is stated as a **prediction**, not a
result. It fails if a third span point comes in materially below the table above
— which would mean the non-plate cost keeps falling rather than flattening (it
went 12,224 → 9,319 B/frame between spans 8 and 16, and two points cannot say
whether that continues).

**Run span 24, 32 and 48 under `panorama-full`.** No component change is needed
there — each scene codes its own plate, so the canvas-agreement blocker below
does not apply — and it tests this prediction directly. Do that before any
further plate work.

**And report the non-plate split** — residual against crops against metadata, per
frame. If the marginal cost is now the target, nobody can act on it as one
number.

## 7. The blocker: span and the cross-scene stream are not independent

Spans 24, 32 and 48 all refuse under `panorama-stream`:

```
scene 1 is (2172, 3881, 3), the stream is (2161, 3841, 3);
inter prediction needs a fixed frame size
```

`build_plate` sizes each scene's canvas from *that scene's own* homographies, and
the two scenes do not move alike — the static one never grows, the panning one
does. Below span ~24 the shapes coincide by accident; past it they genuinely
differ, and `BackgroundStreamTransmitter.push` requires one frame size across a
chain.

**This amends §5 of this brief.** "Sweep span before scene count, they are both
amortisation axes on the same fixed cost" was right about the confound and wrong
about independence: **span past 16 is simply unavailable under `panorama-stream`**
until the canvas is made run-wide. The fix — pad every scene's plate to the union
of the run's canvases — is `stream.py` plus `plate.py`, a component change with
BP30's tests around it, and it belongs to `plans/BP40-background-honesty.md`
rather than here.

Given the measured x1.038 ceiling, padding to the union costs roughly **4% of
plate area** — and by the decomposition above, 4% of a term that span is driving
toward irrelevance anyway. **Which is an argument for doing the `panorama-full`
span points first and deciding whether the combined question is still worth the
component change.**
