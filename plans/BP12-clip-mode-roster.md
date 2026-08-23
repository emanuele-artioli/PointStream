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
