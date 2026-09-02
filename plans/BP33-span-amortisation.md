# BP33 — Span: the amortisation axis nobody has swept

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
| plate bytes, 48 vs 8 frames, static clip | [1.00x, 1.60x] | above: the median over a long span is not converging; check foreground exclusion |
| plate bytes, 48 vs 8 frames, moving clip | [1.05x, 2.50x] | above 2.5x: the canvas is running toward `MAX_CANVAS_SCALE` and the pan has left the plate's usefulness behind |
| anchor bytes per frame, 48 vs 8 frames | [0.45x, 0.90x] | outside: the anchor is being run per-chunk rather than over the whole span |
| **delivered Y-PSNR at frame 47 minus frame 0** | **[−4.0 dB, +0.5 dB]** | below −4 dB: the homography has drifted off the plate and the product is a reconstruction that visibly rots, which is a different product from a uniform one |

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
