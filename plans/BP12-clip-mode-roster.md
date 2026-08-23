# B′12 — Re-run the roster in clip mode, on perceptual metrics

**The critical path. Supersedes `BP10` and takes priority over `BP11`.**

**Read first:** `PLAN.md` §2.7 — Animate-Anyone was being driven at T=1.
Clip mode is better. It is not a working engine.

## Why this is now the top item

Every Animate-Anyone number this project produced until 2026-08-23 came from
calling the single-frame path on a temporal model, and most of them were read
through a metric that was not LPIPS. Corrected (calibrated LPIPS on the player
bbox, offsets 1–8):

| Path | Object PSNR | LPIPS |
|---|---|---|
| frame-by-frame | 8.81 dB | 0.751 |
| **clip mode** | **11.57 dB** | **0.570** |
| static copy @ offset 1 | 17.00 dB | 0.239 |
| static copy @ offset 8 | 11.16 dB | 0.582 |
| unrelated image | — | 0.645 |

Clip mode is a real improvement over T=1. **AA at 0.570 still sits closer to an
unrelated image than to heavy blur (0.430)**, and only marginally ahead of a
static copy at offset 8. The withdrawn "best engine at LPIPS 0.067" claim was
the broken metric.

The architecture argument is unchanged and still worth testing, not a result:
ReferenceNet injects reference features **into the UNet's spatial
self-attention**, whereas ControlNet adds its condition **residually** and
relies on CLIP embeddings that *"lack fine-grained spatial details, causing
appearance drift under large deformations."* Our failed
reference-in-the-control-image retrain was fighting that boundary, which makes
that negative result **citable rather than embarrassing**.

## What to do

### 1. Re-run the whole roster in clip mode, at matched offsets

Every temporal engine gets `generate_sequence`; per-frame engines keep
`generate`. Same clips, same seed, same offsets. Report **LPIPS as the ranking
key and PSNR alongside** (`PLAN.md` §2.5: the subfield rejects PSNR for
generative coding, and our own Evaluation section says so).

Include the **static-copy floor at each offset** — §2.5 has it: 21.5 dB at
offset 1 falling to ~11 dB by offset 8.

### 2. Re-run the cross-appearance control in clip mode

The +0.93 dB measured frame-by-frame says nothing about a pathway that was
structurally disabled. Re-run it and judge against `BP10`'s bounds: **≥ +3 dB =
ReferenceNet works**, ≈ +0.9 = generic leakage, ≈ 0 = wiring fault.

### 3. Use LPIPS as the ranking key — it is already cheap

The published `lpips` package is **3.5 ms/frame** (`PLAN.md` §2.7). The 138 ms
CPU-VGG number was the fake backend; do not re-open it. Generation is
**4–6 s per frame**, so LPIPS is a rounding error. VMAF and DISTS stay for
final results only.

### 4. Then extend the roster

Once clip mode is the default, the SD-1.5 ReferenceNet family is the natural
place to look — same architecture as AA, open weights, no SVD licence problem:

| Candidate | Why | Note |
|---|---|---|
| **Champ** | SD-1.5, Reference UNet + guidance encoders + motion module, adds SMPL 3D guidance | `fudan-generative-vision/champ` |
| **MusePose** | SD-1.5, `reference_unet` + `pose_guider` + `motion_module` | `TMElyralab/MusePose` |
| **MagicAnimate / MagicPose** | Same appearance-encoder idea, older | fallback |

**Check the licence before integrating**, not after — that is what stranded
MOFA and StableAnimator.

## Traps

**Check the invocation before blaming the model.** This is the fourth time here
that a path existed, ran, passed its tests, and was not doing the job. Whenever
an engine underperforms, first confirm it is being called the way its
architecture intends.

**Do not read too much into the 11.57 dB.** n=32 frames, 4 clips, offsets 1–8 —
the easy regime, where the static-copy floor is 11.2–21.5 dB. What is
established is that clip mode transforms output quality, **not** that the coding
task is solved.

**Clip mode changes the cost model.** A clip must be generated as a unit, so
per-frame latency and VRAM both change. Record both; `subsec:eval-operating`
needs them.

**The held-out caveat still stands** (§2.8): AA's fine-tuning covered all seven
videos, so it is in-domain only whatever it scores.

## Done when

- The roster is re-ranked in clip mode with LPIPS as the key and PSNR beside it.
- The cross-appearance control is re-run in clip mode and judged against bounds.
- `PLAN.md` §6.2's roster is re-decided on that evidence.

---

## Delivered — 2026-08-23

Run on `gpu5`, seed 42, 12 probe clips, contiguous offsets 1–8. Roster on
`cuda:1`, cross-appearance on `cuda:0`. LPIPS bounds written into
`experiments/probe/bounds.py` before any number existed. Records in
`outputs/bp12-clip-roster/` (`report.json`, `summary.json`, per-engine and
per-control JSON), logs `outputs/bp12-clip-roster.log` and
`outputs/bp12-cross-appearance.log`. **Not citable**: 12 clips, all from the
five training-split videos.

Full numbers and their reading are in `PLAN.md` §2.10; `§6.2` is re-decided on
them. This section says what was done and what it cost.

### 1. Roster re-run in clip mode — done

`experiments/probe/` gained a `sequence=True` plan flag. Animate-Anyone is
driven once per clip through `generate_sequence`; a sequence plan whose backend
lacks that method now **raises** rather than falling back, and a backend
returning the wrong frame count raises rather than letting `zip` pair the wrong
frames. Per-frame engines keep `generate`, at the same offsets, so the arms are
matched.

LPIPS is the ranking key with PSNR beside it. An engine with no LPIPS is left
out of the ranking rather than quietly ranked on PSNR — pix2pix is 2nd on PSNR
and 7th on LPIPS, so those really are different orders.

Result: **every engine loses to the static-copy floor**, 2.5σ–10.6σ. The top
three (upscale-refine 0.5585, seg-controlnet 0.5595, animate-anyone 0.5692) are
not separable. AA's clip-mode number reproduces §2.7's 0.570 on 12 clips instead
of 4, on a different GPU.

### 2. Cross-appearance control re-run in clip mode — done, and it retired itself

Judged against BP10's bounds as instructed, and **those bounds are void**.

A prediction was written before the last two arms landed
(`outputs/bp12-cross-appearance-prediction.txt`), including its own failure
branch. The ControlNets came in *above* Animate-Anyone, which was the branch
that said investigate the harness first. Investigating meant driving the
copying baselines through the same code path:

| Arm | Δ LPIPS | share of a paste | Δ PSNR |
|---|---|---|---|
| **static-copy — no model at all** | **+0.285** | **100%** | **+4.45 dB** |
| upscale-refine — non-generative | +0.185 | 65% | +2.64 |
| seg-controlnet | +0.176 | 62% | +3.43 |
| pose-controlnet | +0.166 | 58% | +2.93 |
| animate-anyone (clip) | +0.107 | 37% | +3.46 |
| ip-adapter-controlnet | +0.055 | 19% | +0.25 |

**BP10's gate was "≥ +3 dB = ReferenceNet works". A pasted keyframe scores
+4.45 dB.** The test ranks how much of the reference survives into the output —
copying — not whether the right person was drawn.

The harness was checked and is sound: static-copy through the cross-appearance
path scores exactly the 0.285 computed independently from the two roster
baselines. The instrument agreed with itself; the interpretation was wrong.

Kept as a measure of **dependence on the reference**, always beside the arm's
own score against the floor. `judge_cross_appearance` refuses to classify
without a copying anchor; `report.py` re-judges stored records with the current
bound rather than the status they were written with.

### 3. LPIPS as the ranking key — done

3.5 ms/frame against ~1 s of generation, as the brief said. Scoped to the
bounding box of the letterboxed player mask, and the box travels with every
score. PSNR stays mask-scoped; the report states that the two columns are not
the same region.

### 4. Extend the roster — not done, and the case for it has changed

Champ / MusePose were to be added because ReferenceNet "is the direction if a
fix is needed". That rationale rested on the cross-appearance test showing AA's
pathway working, which the test cannot show. Adding two more SD-1.5
ReferenceNet models would produce two more arms that lose to a pasted keyframe.

**What is needed first is a measurement that can tell "the output moved" from
"the right person appeared"** — an identity metric (CSIM/ArcFace), which is what
the literature ranks these models on. Until that exists, extending the roster
buys rows, not answers.

### What this brief adds to the standing rules

**A control needs its own null.** The cross-appearance test *was* the control,
and it went four engines deep before anyone asked what an arm with no model
scores on it. When a control produces a ranking, run the degenerate arm through
it — the paste, the passthrough, the empty model — before reading the ranking.

### Also delivered, unasked

- `unrelated-image` is a permanent arm beside `static-copy`, and `drive_all`
  publishes **no ranking at all** if the two do not separate on the metric.
- `experiments/probe/report.py`: every comparison paired on clips, with n and a
  standard error, letting `compare_paired` decline the ones the sample cannot
  support. Only two adjacent pairs in the whole ranking are clear.
- `--donor-mode same-video`, so scene colour cannot be mistaken for identity.
  Not needed for this verdict once the paste baseline settled it; available.
- Suite 1039 → 1050, ruff / mypy / layers clean.

### Done when — status

- [x] Roster re-ranked in clip mode, LPIPS key, PSNR beside it.
- [x] Cross-appearance control re-run in clip mode and judged against BP10's
      bounds. **Judged, and the bounds failed the judging.**
- [x] `PLAN.md` §6.2 re-decided on that evidence. Quality flagship stays unset.
