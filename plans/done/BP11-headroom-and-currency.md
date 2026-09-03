# B′11 — Measure the headroom, then change the currency

**This replaces `BP10` as the critical path.** `BP10` asked which engine uses
appearance. That question is real but it is being asked inside a 10 dB band with
2 dB noise, using a metric the subfield rejects for generative content, about a
region that is 1% of the frame. Answer the bigger question first.

**Read first:** `plans/done/RESEARCH-HISTORY.md` §2.5 and §2.6.

## Step 1 — How much is the player region actually worth? (do this first)

**The single most valuable number in the project, and it is cheap.**

Encode a probe clip's *source frames* with the codec ladder at several rate
points. Encode them again with the player bounding boxes flattened — blurred, or
filled with the background plate — simulating a generator that costs nothing and
is perfect. Difference the bitrates at matched quality.

That delta is **the entire headroom of this paper.** Everything PointStream can
possibly win on the object path is bounded by it.

**Bounds, written before running:**

- **≥ 25%** of the bitrate in ~2% of the pixels — the premise is strong; players
  are high-motion, high-detail, and expensive. Proceed with confidence.
- **10–25%** — a real but modest prize. The paper's claim narrows to a
  rate–quality trade rather than a large win, and the residual and background
  become the headline rather than the generator.
- **< 10%** — **the premise is weak and we must know now.** No generator, however
  good, can win much. The honest paper then leads with the framework, the
  lattice, and the background/residual result, and reports the object path as
  measured.

**Do not skip the matched-quality part.** A flattened region also changes
quality; comparing raw file sizes measures nothing (`plans/done/RESEARCH-HISTORY.md` §5).

## Step 2 — Change the evaluation currency to what the field uses

`plans/done/RESEARCH-HISTORY.md` §2.5: generative face video coding states plainly that because these
methods do not optimise pixel distortion, *"PSNR and SSIM are not suitable"*.
Sparse2Dense's headline 74.5% BD-rate is **DISTS**; LPIPS and FVD sit beside it.
Our own Evaluation section says the same thing and we gated on PSNR anyway.

- **Add DISTS** to `src/components/metrics/`. It is the field's default for this
  comparison and we do not have it. LPIPS is already wired.
- **BD-rate on DISTS and LPIPS** becomes the reported currency for any arm
  involving generation. PSNR stays as a reported number and as the always-on
  floor, never as the ranking key for a generative arm.
- **Re-rank the existing engines on LPIPS.** The probe already records it. The
  ranking may not survive the change of metric, and that is the point.

## Step 3 — Run the experiment the paper actually claims

`plans/done/RESEARCH-HISTORY.md` §7 P0 item 2 — PointStream against the codec ladder at matched rate,
frame level, with the residual — **has never been run**, in any configuration.
Every measurement so far has been component triage.

At the frame level ~98% of the pixels are background and court, which the
background model carries. A mediocre player reconstruction plus a cheap residual
may still produce a favourable rate–quality curve. That is the claim; it deserves
a measurement before the object path is declared fatal.

C1, C2 and C3 are merged, so the pipeline to run it now exists.

## Also fix, because it is cheap and real

**Stabilise the crop geometry.** §2.5: the tracking box swings 313–984 px wide
with aspect 0.30–1.14, costing ~4.5 dB at adjacent frames purely from
per-frame letterboxing. Use a constant crop size on a temporally smoothed centre,
or score in source coordinates. This does not rescue the scale on its own, but it
removes a defect that contaminates every object-level number.

## Traps

**Do not answer step 1 with a guess.** The whole strategy branches on it, and
both a large and a small number are useful — a small one redirects the paper
early, which is worth more in August than a large one is.

**Do not drop PSNR.** It stays always-on and always reported (`plans/done/RESEARCH-HISTORY.md` §3). The
change is that it stops being the *ranking key* for generative arms.

**A metric change is not a way to make a bad result look good, and must not read
as one.** The justification is that the subfield, our own paper text, and the
comparison we are positioned against all use perceptual metrics for this. Say so
explicitly wherever the currency is stated, and keep reporting PSNR beside it so
nobody has to take that on trust.

## Done when

- The headroom number exists with its bounds recorded and its verdict stated.
- DISTS is wired; BD-rate on DISTS/LPIPS is the generative currency.
- The engines are re-ranked on a perceptual metric.
- The codec-ladder comparison has been run at least once end to end.
