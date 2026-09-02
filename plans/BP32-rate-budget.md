# BP32 — The rate budget: where the bits go against where the headroom said they could

**Cheap, mostly arithmetic over data already on disk, and the single item in
`plans/ROADMAP.md` most likely to redirect the project.** Do it before spending
another campaign.

**Owns:** `experiments/budget/**` (new), `outputs/bp32-budget/**`, and the
section of `PLAN.md` §5 that states the currency. **Does not own**
`src/runner/**`, `src/components/background/**`, `experiments/tier/**` — PR #45
holds those; report, do not edit.

**Read first:** `AGENTS.md` · `PLAN.md` §§2.6, 2.14, 2.20, 2.21 ·
`plans/BP24-ladder-report.md` (the payload tables) · `plans/BP31-findings.md`
§§3, 9, 10 · `plans/BP29-panorama-report.md` §4.

---

## 0. The observation

Two numbers in this project have never been put in the same table, and they do
not agree.

**What the motivation measured** (`PLAN.md` §2.14, real 4K, n=2 clips, 48-frame
windows). Removing the players and inpainting the plate saves **22.9% ± 3.0%**
of av1's bitrate at matched quality. Replacing the background with a transmitted
plate plus per-frame warps saves **34–69% of the background bitrate** — measured
for AVC, HEVC and VVC; **not reported for AV1**, because the PSNR overlap
between the arms was 0.46 and 0.20, below the 50% the BD-rate implementation
requires.

**What the system delivers** (`plans/BP24-ladder-report.md`, `BP31` §9, same
content family, 8-frame clips). BD-rate **+116.8%** against av1 with a
single-frame plate; **+109.72%** with a real panorama at N=2; **+90.97%** with
the cross-scene stream on.

Those are the same claim measured twice, once as headroom and once as a system,
and they are roughly **150 BD-rate points apart**. Nobody has written down where
the difference goes. Until somebody does, every choice about what to try next is
being made without a map — including BP31's, which is about to spend a
ten-scene, six-video extraction and ladder campaign.

## 1. The one methodological rule this brief turns on

**BD-rates do not add.** "Foreground saves 22.9% and background saves 45%, so the
ceiling is 58%" is not arithmetic, it is a category error: each figure is an
integral over a different rate-quality curve with a different overlap interval,
and the two arms were never run together. Any budget built that way will produce
a confident, plausible, wrong ceiling.

**Build the ledger in bytes at matched quality instead.** Pick one operating
point on the anchor curve — the natural one is av1 at the quality PointStream
actually delivers — and account for every byte both arms spend to reach it. Then
convert to a rate ratio once, at the end, and say what quality it is quoted at.
That is the same discipline `plans/BP29-plate-codec-report.md` §3 was written
about, applied one level up.

## 2. What to produce

A single table, `outputs/bp32-budget/ledger.json` plus a readable rendering, with
one row per term and every row citing the run it came from.

**The known starting values**, all from `alcaraz_highlights/scene_000`, 8 frames,
4K, av1 preset 10, so the whole ledger is on one clip before it is on six:

| quantity | bytes | source |
|---|---:|---|
| av1 anchor, whole clip, QP 55 | 85,995 | `outputs/bp24-ladder/av1-payload-lowmotion.json` |
| av1 anchor, whole clip, QP 15 | 851,572 | same |
| PointStream plate (jpeg, single source frame) | 463,334 | same |
| PointStream residual, coarse | 4,334 | `av1-coarseness-lowmotion.json` |
| PointStream unaided corner (plate + crops, no residual) | 487,643 | same |
| PointStream appearance/crops | ~24,300 (487,643 − 463,334) | derived |

**The fact that should be the headline of this brief:** the plate alone is
**5.4x the entire anchor clip** at the anchor's cheapest rung, and it is one
still image. Everything else in PointStream's payload — crops, residual,
metadata — is 24,309 B against the anchor's 85,995 B, i.e. **PointStream's
non-plate payload is already under a third of the anchor's total rate.** The
system is not losing because object-centric coding is expensive. It is losing
because it sends one very expensive picture.

### The terms to attribute, cheapest to measure first

1. **Span.** The plate is paid once per scene whatever the scene's length. The
   ladder runs 8 frames; BP21's cache holds 48. This is `BP33` and it is
   probably the largest single term — see that brief for the pre-registered
   bounds.
2. **Plate codec.** `BP31` §10 measures av1/vvc intra at **x0.691** against jpeg
   at 43 dB on a 4K panorama plate — a 1.45x saving, not the 3.6–4.1x
   `PLAN.md` §2.21 quotes from a single-point comparison at a fidelity the
   codecs had not been asked to share.
3. **Cross-scene amortisation.** `BP31` §3 measures **0.646** over twelve scenes
   of this video, av1 at crf38. Note it **cannot compose with term 2 today** —
   `panorama-stream` never consults the still-image sidecar (`BP31` §1) — so the
   ledger must show them as alternatives, not as a product, and say so.
4. **The panorama itself.** It does *not* shrink the plate (`BP29` §4: the plate
   codes to 0.6% *fewer* bytes while covering 1.8% more canvas). Its whole gain
   is on the residual — 0.22x, +4.9–6.2 dB delivered on a moving clip. So it
   belongs in the ledger as a **quality** term, not a rate term, and putting it
   in the rate column is a mistake this brief exists partly to prevent.
5. **Everything unattributed.** Whatever the four terms above do not explain is
   the number that matters. Name it, do not absorb it.

## 3. The sensitivity nobody has priced: the anchor is running at a speed preset

`src/components/codec/measure.py`: `PRESETS = {"avc": "veryfast", "hevc":
"ultrafast", "av1": "10", "vvc": "faster"}`. **SVT-AV1 preset 10 is a fast
preset** — the scale runs 0 (slowest, best) to 13, and every ladder in this
project has run the anchor at 10.

`plans/DEFERRED.md` D-CODEC-PRESETS reasons that this does not matter, because
the paper compares PointStream against a codec rather than codec against codec,
and *"pairing both arms on one codec at one preset makes the preset cancel"*.
**It does not cancel here, and the direction is unfavourable.** The anchor codes
100% of the pixels at preset 10. PointStream codes only its *residual* through
that path; its plate goes through jpeg or, under `panorama-stream`, through
libaom at a CRF — a different encoder with its own settings. Strengthening the
preset therefore improves the anchor across its whole payload and PointStream
across the ~9% of its payload that is not plate.

So **the +90.97% on record is measured against a handicapped anchor, and is a
lower bound on the gap at a quality preset.** A referee will ask why preset 10,
the honest answer is encode time, and the honest consequence has never been
measured.

**Price it here, because it is two encodes:** the anchor's rate at presets 10, 6
and 2 on one cached clip at one rung. Report the anchor's BD-rate against itself
across presets, and state what that does to every gap in the paper. Bound it at
**[10%, 40%]** rate reduction from preset 10 to preset 2 — outside that, check
the preset reached the encoder rather than being accepted and ignored.

If it lands at the top of that band, the paper must either move the anchor to a
quality preset and re-run every ladder, or state the preset beside every number
and name the handicap explicitly. Both are defensible; silence is not.

## 4. The gap to close on the way: av1's background headroom

`PLAN.md` §2.14 reports the background saving for AVC, HEVC and VVC and **not for
AV1** — and av1 is the anchor every ladder in this project uses. So the paper's
motivation section has a hole at exactly the codec its evaluation runs against,
and the budget above cannot be closed without it.

The recorded reason is insufficient PSNR overlap (0.46 and 0.20 against a 0.50
requirement), and the recorded fix is *"widen the QP sweep"*. Do that. It is four
encodes per clip on two cached clips.

**Bound it two-sided before running.** AVC 34.4%/58.6%, HEVC 38.3%/57.2%, VVC
59.9%/68.9%, and the section observes that background saving **improves** with
codec strength while foreground saving falls. av1 sits between HEVC and VVC in
strength, so **[35%, 70%]** is the band, with the two clips expected to differ by
about 20 points as every other codec's pair does. Outside that band is an alarm:
check the widened sweep did not change the integration interval underneath the
comparison, which is exactly the confound §2.14 flags for VVC's QP 47.

## 5. Bounds on the budget itself — write these before computing

Two-sided, because the quantity being bounded is the very thing the ledger exists
to explain.

- **The four terms explain between 40% and 95% of the 150-point gap.** Under 40%
  means the ledger is missing a term large enough to change the plan, and the
  next question is which — the likeliest candidates are that the headroom
  experiment's plate was built non-causally from the same frames it was scored
  on, and that its 48-frame window is doing the work `BP33` is about to isolate.
  Over 95% means the terms have been double-counted; check that (2) and (3) were
  not multiplied.
- **Span is the largest single term**, worth more than plate codec and
  cross-scene amortisation combined. If it is not, that is a genuine surprise and
  `BP33`'s bounds need revising before its run, not after.
- **The non-plate payload stays under 40% of the anchor's rate** at matched
  quality in every configuration. It is 28% today. If a configuration pushes it
  past 40%, the residual or the appearance channel has started to matter and the
  "the plate is the whole problem" framing has stopped being true — which would
  be a finding, and would move `PLAN.md` §7 P0 item 8.

## 6. What this brief must not do

- **Not run a ladder.** Every number it needs either exists or is four encodes.
  If it starts encoding a ladder it has become BP31 and it is colliding.
- **Not conclude the approach is dead.** A ledger says where bytes go; it does
  not say whether a regime exists. That is `plans/FORK-bp31.md`'s question.
- **Not quietly become an optimisation.** Attribute first. Fixing a term before
  it is measured is how the project ends up with a lever it cannot price.

## Done when

- `outputs/bp32-budget/ledger.json` accounts for the gap between §2.14's headroom
  and §2.20/BP31 §9's delivered BD-rate, term by term, in bytes at a stated
  matched quality, with each term citing its run.
- av1's background headroom is measured and `PLAN.md` §2.14's table has no hole
  at the anchor codec.
- The terms are **ranked by how many BD-rate points each is worth**, which is the
  artefact BP31 and `plans/FORK-bp31.md` both need.
- The result is told to the BP31 session **before** its extraction campaign
  commits to a frames-per-scene value.
