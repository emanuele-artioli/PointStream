# POINTSTREAM — the plan

*What the project is, where it stands, and what happens next. Read this plus
`AGENTS.md` plus the one brief in `plans/` for your workstream — that is the
whole context a session should need.*

---

## 1. What this is

An object-centric semantic video codec. Each salient object is transmitted as an
**appearance representation** established once, a **motion representation**
describing how that appearance evolves, and a **temporal policy** saying how
densely the motion is actually sent. A background model and an optional
corrective residual complete the payload. The client reconstructs frames
generatively.

The current cycle rewrites this into a platform where every component — codec,
detector, pose, segmenter, generator, appearance and motion representation,
background method, residual, transport, metric, and the task domain itself — is a
config choice, and the only code a new component needs is the wrapper that
satisfies its contract.

Target: **ACM TOMM, September 30.**

## 2. Status

| Phase | State | Next action |
|---|---|---|
| A — contracts and concepts | ✅ done | — |
| B — components | ✅ **done** | Merged-ready on `phase-b/integrate` (still unmerged to main) |
| **B′ — the engine roster** | BP12 ✅, BP25 ✅ | Re-ranked in clip mode on calibrated LPIPS (§2.10). IP-Adapter now has a trained appearance path (§2.17) and still loses to a paste. Quality flagship stays **unset**. |
| C — pipeline and runner | ✅ **done** | `C1`/`C2`/`C3` merged. A tier config runs end to end and is scored (§2.16, BP23). |
| D — experiments layer | 🟡 partly unblocked | `BP26` wired the six ablation axes (2026-08-26), so the lattice is now *measurable* but still un-run. Rate-based experiments still need a real encoder (`BP24`). |
| E — experiments and paper | ⬜ | Ordered by §7 |

**Code.** `src/contracts/` is complete and green. `src/components/` now covers all
sixteen axes: ~8.6k lines of source, ~3.3k of tests, 52 registered backends of
which 48 construct. 392 contract and component tests pass, plus 13 integration
tests that drive real tools. `ruff` and `python -m src.contracts.layers` are
clean. `mypy` is clean on `phase-d/cleanups` (was 66 in tests; D1 closed).

All three Phase-B gates pass: `av1` + `yuv444p` raises `CodecConstraintError`
rather than silently emitting yuv420p; every codec rung has a region arm or a
recorded reason it cannot; `python -m src.components` lists every backend on
every axis.

**Quality measurement works.** This was the standing blocker and it is closed at
the component level: VMAF runs through libvmaf (97.4 on identical frames, 28.9 on
degraded), PSNR, SSIM and LPIPS all return real numbers, and FVMD correctly
refuses a single frame. BD-rate with an overlap check is implemented in
`src/components/metrics/bd_rate.py`. What is *not* yet true is §7 P0 item 1 — a
tier config producing those numbers end to end — because there is no pipeline.

**Paper.** Introduction, Related Work, System Design, Future Work written.
Evaluation is a skeleton of `GOAL`/`HOLE` markers waiting for results.
Conclusion absent until there are results to conclude from.

### 2.1 Generation loaders — Wave 1 merged, numbers re-run on aligned pairs

The Phase-B socket is plugged. Tests that inject `_FakePipe` remain; the loaders
now load real weights when not injected. Registry entries for
`trajectory-controlnet` (alias `trajectory-render`) and `stable-animator` are
applied; the Animate-Anyone summary is 7 matches / 114 tracks, not a single
match. Driven 2026-08-22 on `cuda:0` (gpu6, both cards idle). Seed **42**.
These are triage numbers, not results.

**Comparison backbone**, re-run on the aligned probe set after the pose-offset
fix. Clip: `assets/probe_set/clips/alcaraz_perricard/scene_006/track_0196`
(one of the five that had 48 colour / 0 skeleton under filename reconstruction;
crop `frame_000641.png` pairs with skeleton `frame_000119.png`). One frame,
PSNR vs letterboxed appearance, 20 ControlNet steps, output ≠ input on every
engine. Bounds were written to `outputs/bp7-psnr-bounds.txt` before generate.
Log: `outputs/bp7-aligned-probe.log`.

| Engine | Checkpoint | PSNR vs letterboxed appearance | Notes |
|---|---|---|---|
| pose-controlnet | epoch **10** | 19.0 dB | `cuda:0`. Bounds 12–32, expect 14–28. Not identity. |
| seg-controlnet | epoch **7** | 20.1 dB | `cuda:0`. Same band. Mask is from the crop, not the skeleton. |
| ip-adapter-controlnet | epoch **10** + `h94/IP-Adapter` | 11.0 dB | `cuda:0`. Bounds worst 10 (txt2img) / best 28. Still on that floor; not identity. Not tuned. |
| pix2pix | `pix2pix_generator.pt` | 18.5 dB (CPU) | Bounds 10–30. |
| spade4tennis | `spade4tennis_lite_generator.pt` | 15.1 dB (CPU) | Bounds 10–30. Cheap blob; wired because it was cheap. |
| trajectory-render | pose-controlnet epoch **10** | 19.0 dB | `cuda:0`. Within 0.03 dB of pose; four synthetic sticks still barely move it. |

BP3's earlier numbers (pose 20.3, seg 21.7, ip-adapter 11.3, pix2pix 16.6,
spade 14.5, trajectory 20.5) were on
`alcaraz_highlights/.../track_0002`, which starts at source frame 0, so that
run never saw the offset. Deltas here are 0.3–1.9 dB on a *different clip*,
not a jump from aligning a mismatched pose. **BP3's "smeared but recognisable"
was not the filename-offset fault.** Alignment still had to land: five of
twelve probe clips had no skeleton at all, and any later ranking that used
those clips would have been measuring missing pose, not model quality.

**Quality flagship** (`phase-bp/bp4`): Animate-Anyone loads
`~/Models/AnimateAnyone/profiles/finetuned_tennis` (7 matches, 114 tracks — not
one). 3 DDIM steps melted (9.65 dB, below the 12 dB floor). 20 steps: 14.0 dB
in-set, 5.0 GiB, 32.5 s warm. StableAnimator is wrapped; SVD-XT not bundled
(§2.4). Sparse2Dense still has no public code.

### 2.2 The probe set — rebuilt and aligned

The inherited v1 set was unusable and is kept at `assets/probe_set.broken-v1/`.
Rebuilt as `pointstream.probe_set.v2` on `phase-bp/bp1`, merged, and now
**12/12 clips have matching colour and skeleton frame counts**. Same seed
(`20260712`), same 12 tracks, locked 5-train / 2-held-out split asserted by a
verifier that fails on the v1 snapshot.

**The lesson worth keeping, because it defeated two agents in a row:**
`assets/dataset` uses **two frame-naming conventions inside one track group** —
crop, `_canny`, `_pose_body` and `_pose_racket` carry global source frame ids,
while `_skeleton` is track-local from zero. 50 of 114 tracks (44%) carry the
offset; the rest align only because they start at frame 0. Two separate
measurements reached opposite conclusions because each read a different directory
and neither said which.

**Always resolve a frame by its position in the track, never by rebuilding a
filename.** `src/shared/tennis_dataset.py:95-110` has always done this correctly
and is the pattern to copy. Note its docstring describes the wrong convention —
do not "fix" the code to match the comment.

Still outstanding: `scripts/select_probe_set.py` writes v1 and must not be used
as the regenerator.

### 2.3 Why every engine scores badly: they were trained for the wrong task

Verified 2026-08-22 by driving the engines, not by reading the roster table.
This supersedes any reading of the Wave-2 numbers as "the models are weak".

**The probe measured self-reconstruction, not coding.**
`experiments/probe/run.py` passes `frame.appearance_rgb` as the *conditioning*
and the same `frame.appearance_rgb` as the *scoring reference*. Each engine was
asked: "here is an image and its pose — reproduce that image." That is not what
PointStream does, and it is in direct tension with the harness's own
`differs_from_input` check: the more an engine changes its input, the worse it
scores, while an unchanged output is failed outright.

**Re-run on the actual coding task** — appearance from frame 0, pose from frame
24, scored against frame 24, 12 clips, seed 42, 20 steps:

| Arm | Object PSNR | Frame PSNR |
|---|---|---|
| **static copy** (paste the keyframe, no model) | **11.82 dB** | 8.90 dB |
| seg-controlnet | 11.01 dB (**−0.81**) | 13.40 dB |
| pose-controlnet | 11.20 dB (**−0.62**) | 13.31 dB |

**Both generators lose to doing nothing.** The pose conditioning buys less than
zero on the object. They win on whole-frame only because they produce a
well-framed canvas.

**The cause is in the training script, and it is structural.**
`scripts/train_controlnet.py`'s `ControlNetDataset` yields
`{image_path, cond_path, prompt}` — **there is no appearance/reference image in
training at all.** These checkpoints were trained as:

> condition (pose / seg / canny) + the text prompt
> *"photorealistic tennis player, broadcast sports shot"* → image

So they synthesise *a* plausible tennis player from a pose. They have no
mechanism to reproduce *this* player, because appearance was never an input.
At inference the harness feeds appearance as an img2img init image — a path the
model was never trained on. A static copy beats them on the object precisely
because it at least has the right person in it.

**`ip-adapter-controlnet` is not an IP-Adapter checkpoint.** Line 82 of the
training script puts `"ip-adapter"` in the same branch as `"seg"` with
`cond_dir = None`. It was trained as a segmentation ControlNet. Its 7.9 dB — the
worst of the roster — is explained by that, not by the adapter being weak.

**What this changes.** This is not a tuning problem and no amount of parameter
search fixes it. The architecture requires appearance to be transmitted and
*used* by the generator; these checkpoints implement a codec that transmits pose
and a text prompt. `PLAN.md` §6.6's cost order — tune, then fine-tune, then
swap — starts at the wrong rung: tuning is ruled out on evidence.

Those options were driven on `phase-bp/bp8` (2026-08-23). Animate-Anyone 20 DDIM
letterbox: **8.96 dB**. Real IP-Adapter: **8.90 dB**. Pose-ref ControlNet, ten
epochs with a same-track reference painted under the skeleton: **11.18 dB**
(series 11.11–11.33; smoke `residual_delta=2.69` showed the reference entered
the net). All lose to this driver's static-copy floor (**11.47 dB** letterbox;
published shared-geometry paste **11.82 dB**). Success was ≥ 12.82 dB. Paths
under `pointstream-wt/bp8/outputs/bp8-coding-task*`. **Option C** — change what
the paper claims is transmitted — is the reported finding, not an escape.
`plans/BP9-probe-harness.md` made the static-copy floor a permanent arm
(`phase-bp/bp9`).

### 2.4 What the cross-appearance control actually shows

Wave 3 concluded that no engine "uses appearance", reading each engine's
object PSNR against the 11.47 dB static-copy floor. **That gate does not measure
what it was taken to measure**, and a control run on 2026-08-23 separates the two
questions.

**The test.** Hold the model, the pose and the metric fixed; vary *only* the
appearance. Generate twice — once with this clip's keyframe, once with another
clip's keyframe — and score both against the true target. A model that uses
appearance scores higher with the correct one. This is decisive where the floor
is not, because a static copy is *real pixels in the wrong pose* while a
generator is *synthetic pixels in the right pose*, and MSE structurally favours
the former.

| Engine | Δ correct − wrong | Δ LPIPS | Reading |
|---|---|---|---|
| `pose-controlnet` (no reference in training) | **+0.86 dB** | −0.004 | img2img init leakage |
| `pose-ref-controlnet` (retrained *with* reference) | **+0.98 dB** | −0.007 | **same as above; the retrain added nothing** |
| `ip-adapter-controlnet` | **+0.08 dB** | +0.009 | no appearance path at all |

n=12, per-clip sd ≈ 2.0 dB, so se ≈ 0.58. The two ControlNet figures are ~1.5σ —
suggestive, not solid. The IP-Adapter figure is solidly zero.

**What this changes.** Three sharper statements replace "no engine uses
appearance":

1. **The only appearance signal reaching any ControlNet is the img2img init
   image** (`strength=0.65`, so roughly a third of the init survives). That path
   is untrained and identical before and after the retrain.
2. **The retrain failed specifically, and the reason is architectural.** Painting
   the reference under the skeleton puts identity into the *control* image, and
   the control branch is trained to read structure. +0.12 dB over the
   un-retrained model is inside noise. **Do not repeat this recipe.**
3. **`ip-adapter-controlnet`'s dedicated appearance path is dead**, consistent
   with it being the mislabelled segmentation checkpoint.

**The static-copy floor is not a pass/fail gate for a generative arm on PSNR.**
It remains a useful published reference and it is what exposed the original
fault, but "below the floor" and "does not use appearance" are different claims
and only the cross-appearance delta settles the second. The paper already says
PSNR is the least informative metric for generatively reconstructed content;
a gate was nonetheless built on it.

**Still open, and it is the decisive one:** Animate-Anyone has not been run
through this test. It is the only remaining architecture with a *designed*
appearance pathway (ReferenceNet cross-attention rather than an init image).
Wave 3 fed it a valid reference and read 8.96 dB against the floor — which, per
the above, does not establish whether ReferenceNet is working.

### 2.5 Why 11 dB, and why the number was never going to work

Measured 2026-08-23. The 11 dB floor is **not** a model result and not really a
metric artefact either — it is what this task scores when framed as pixel
fidelity on a tracked crop.

**Static copy, keyframe 0 pasted at frame N, object-scoped PSNR:**

| Offset | Per-frame letterbox | Source coordinates |
|---|---|---|
| 1 | 17.00 dB | **21.53 dB** |
| 2 | 15.02 | 16.94 |
| 4 | 13.30 | 13.49 |
| 8 | 11.16 | 11.36 |
| 24 | 11.47 | 9.71 |

Three things follow, and the third is the one that matters.

**1. The tracking box is violently unstable.** Over one track: width 313–984 px,
aspect ratio **0.30 to 1.14**, adjacent frames differing by ~29 px in width and
~23 px in x. Each frame is letterboxed to 512² independently, so the player lands
at a different scale and position in every canvas. That costs ~4.5 dB at adjacent
frames (17.0 vs 21.5) and is a real, fixable defect.

**2. But fixing geometry does not rescue the scale.** Even in exact source
coordinates, two *adjacent* frames of the same player reach only 21.5 dB, and by
offset 8 everything converges to ~11 dB whichever way it is measured. At larger
offsets the letterbox is *better*, because re-centring accidentally compensates
for real motion. So the usable dynamic range is roughly **11–21 dB**, with a
per-clip sd of ~2 dB. Effects worth about 1 dB are being sought inside noise of
about 2 dB across a span of about 10 dB. **No engine ranking taken in that band
is trustworthy, in either direction.**

**3. The field does not evaluate this with PSNR, and says so explicitly.**
Generative face video coding states that because these methods do not optimise
pixel-level distortion, *"traditional measures like PSNR and SSIM are not
suitable"*, and uses **DISTS and LPIPS** instead. Sparse2Dense's headline
74.5% BD-rate is **DISTS**, with LPIPS and FVD alongside — not PSNR. Our own
Evaluation section already says PSNR is the least informative metric for
generatively reconstructed content. We then built the gate on it.

### 2.6 The players are 1% of the frame

Measured over 40 tracks: a player bounding box averages **88,415 px, or 1.07% of
a 4K frame**. Two players ≈ **2.1%**. So ~98% of every frame is background and
court, which the background model carries and the codec would otherwise spend
bits on.

This has a consequence the component triage kept hiding: **the paper's claim is a
frame-level rate–distortion claim, and it has never been run** (§7 P0 item 2).
Every measurement so far has been pixel fidelity of a 1%-of-frame crop, judged by
a metric the subfield rejects for exactly this content.

**The bounding question, and it is cheap to answer:** how many bits does a
conventional encoder actually spend on that 2%? Encode a clip normally, encode it
again with the player regions flattened, and difference the bitrates. That number
is the entire headroom of this paper. If the player region costs 30% of the
bitrate there is a real prize; if it costs 3%, the premise is weak and we should
know that before another engine is wired. **Run this before anything else.**

### 2.7 Two faults found together: T=1 invocation, and a fake LPIPS

**Both found 2026-08-23. The second invalidates more than the first fixes.**

#### The LPIPS was not LPIPS

`src/components/metrics/lpips.py` computed an **uncalibrated VGG-19-bn feature
MSE** under the name `lpips`. Its own docstring said so; it was registered,
reported and read as LPIPS anyway. Calibrated against published anchors:

| pair | old "lpips" | real LPIPS |
|---|---|---|
| identical | 0.000 | 0.000 |
| mild noise | 0.009 | 0.250 |
| heavy blur | 0.032 | 0.430 |
| **unrelated image** | **0.083** | **0.645** |

**An unrelated image scored 0.083 while a good reconstruction scored 0.085.** The
metric could not tell them apart. **Every LPIPS number produced before this date
is void**, including the ones in `§2.4`'s cross-appearance table.

Fixed by wrapping the published `lpips` package, which was already installed. It
is also **40× faster** — 3.5 ms/frame against 138 ms, because AlexNet replaces
VGG-19-bn. `tests/components/test_metrics_integration.py` now asserts the
anchors above rather than merely `> 0`, which is what let the defect through.

#### Animate-Anyone was driven at T=1

AA is ReferenceNet + pose guider + **motion module**. Every evaluation called
`generate()`, the single-frame path; `generate_sequence()` existed unused.
Offsets 1–8, 4 clips / 32 frames, seed 42, 20 steps:

| Path | Object PSNR | LPIPS (player bbox, calibrated) |
|---|---|---|
| frame-by-frame | 8.81 dB | 0.751 |
| **clip mode** | **11.57 dB** | **0.570** |
| delta | **+2.76 dB** | **−0.180** |

#### What this actually means — and it is not a success

Clip mode is genuinely better and every AA verdict taken frame-by-frame is void
(8.96 in Wave 3, 10.4 in Wave 2, 9.65 at 3 steps). **But AA in clip mode is still
not good**, and the corrected metric is what shows it:

| Arm | Object PSNR | LPIPS (player) |
|---|---|---|
| static copy @ offset 1 | 17.00 dB | **0.239** |
| static copy @ offset 8 | 11.16 dB | 0.582 |
| **AA clip mode, offsets 1–8** | 11.57 dB | **0.570** |
| *reference: heavy blur* | — | *0.430* |
| *reference: unrelated image* | — | *0.645* |

**AA at 0.570 sits closer to "unrelated image" than to "heavy blur",** and only
marginally ahead of a static copy. No engine on the roster produces a
perceptually good player. The earlier claim that AA was "best on perceptual
quality at LPIPS 0.067" was an artefact of the broken metric and is withdrawn.

#### The standing rule this earns

**Fifth occurrence of the same failure mode**: ten generators registered that
could not load weights; a verifier green while five clips had no pose; a roster
ranked on self-reconstruction; a video model evaluated one frame at a time; and a
metric that could not distinguish a match from an unrelated image. Every one
passed its tests.

Two rules follow, and they belong in every brief:

1. **Check the invocation before blaming the model.**
2. **Calibrate a metric against known anchors before trusting a ranking from
   it.** "Zero for identical, positive for different" is satisfied by a metric
   with no dynamic range. Assert the *far* end too.

### 2.8 Animate-Anyone has seen the held-out videos

Verified 2026-08-22 from `assets/dataset/pointstream_aa_meta.json`. The probe set
holds out `alcaraz_highlights` and `djokovic_zverev`. Animate-Anyone's
fine-tuning set contains **both**: 20 tracks from the first, 16 from the second,
out of 114 total across 7 videos.

**So for Animate-Anyone there is currently no held-out data at all**, and any
number it posts on the held-out split is an in-training number wearing the wrong
label. This bears directly on `subsec:eval-general`, which exists to separate
what fine-tuning buys from what a pretrained backbone already delivers.

Three options, and the choice is a decision for `BP5`, not an assumption:

1. **Re-split** so the held-out videos are ones AA never saw — cheapest, but AA
   saw 7 of the 7 videos we have, so this needs new source material.
2. **Report AA as in-domain only** and let a pretrained engine carry the
   held-out arm. Honest, and costs nothing.
3. **Retrain AA** on a proper split — explicitly out of scope (§7 P2 item 17).

Option 2 is the default unless someone argues otherwise. Whatever is chosen, the
paper says which, because an unlabelled in-training score is the kind of thing a
reviewer finds.

### 2.9 Known environment limits

- **SAM3 cannot load.** `torch.nn.attention` does not exist in torch 2.2.2. This
  blocks the SAM3 detector comparison (§7 P1 item 10) unless a second env is
  built. Both the detector and segmenter entries fail construction and say so.
- **RF-DETR is not installed.** It needs `transformers>=5.1`; this env pins
  4.46.3. Registered, honest about it, not usable.
- **MOFA-Video is a candidate, not an integration.** Its SVD weights are
  Stability-AI-licensed and not bundled, so construction refuses by design. §6.2
  says what replaces it for the trajectory arm.
- **StableAnimator is the same SVD class.** Wrapper on `phase-bp/bp4`. HF card
  `FrancisRing/StableAnimator` is Apache-2.0 (checked 2026-08-22); GitHub code
  is MIT; inference needs SVD-XT (Stability AI Community / SVD research licence)
  and InsightFace `antelopev2`, neither bundled. Cannot ship as a flagship until
  SVD is cleared. Live inference was not run: leftover VRAM ~11.6 GiB, VAE decode
  wants ~16 GiB.

### 2.10 The clip-mode roster, and the control that retired its own test

Run 2026-08-23 on `gpu5`, seed 42, 12 probe clips, offsets **1–8** contiguous,
`cuda:1` for the roster and `cuda:0` for the control. LPIPS bounds were written
into `experiments/probe/bounds.py` before a single number was generated.
`outputs/bp12-clip-roster/` (`report.json`, per-engine records), log
`outputs/bp12-clip-roster.log`. **Not citable**: 12 clips, all from the five
training-split videos.

**Both baselines ran before any engine, and the run refuses to rank at all if
they do not separate.**

| Anchor | Object LPIPS | Object PSNR |
|---|---|---|
| **static copy** — right player, wrong pose | **0.4505 ± 0.0220** | 13.51 ± 0.44 dB |
| **unrelated image** — wrong player | **0.7358 ± 0.0075** | 9.06 ± 0.13 dB |
| separation | **0.285** | 4.45 dB |

The floor reproduces §2.5 and §2.7 independently: 17.00 dB / 0.239 at offset 1,
11.16 dB / 0.582 at offset 8. The null arm is *flat* across offsets
(0.722 → 0.743) where the floor climbs, which is what identity-sensitivity looks
like — show the wrong person and distance in time stops mattering.

#### The roster

LPIPS is the ranking key, PSNR beside it. Scope differs between the two columns
and the report says so: PSNR over the letterboxed player **mask**, LPIPS over
the **bounding box** of that mask, because LPIPS is a patch metric and cannot
take a mask. Each column compares across arms; the two are not one measurement.

| Engine | Driven | Object LPIPS | Object PSNR | s/frame |
|---|---|---|---|---|
| upscale-refine | frame | **0.5585** | 12.73 dB | 0.00 |
| seg-controlnet | frame | 0.5595 | 12.19 | 0.94 |
| **animate-anyone** | **clip** | 0.5692 | 12.21 | 0.99 |
| pose-controlnet | frame | 0.6031 | 12.03 | 0.96 |
| trajectory-controlnet | frame | 0.6229 | 11.78 | 0.95 |
| ip-adapter-controlnet | frame | 0.7606 | 9.00 | 0.97 |
| pix2pix | frame | 0.7820 | 12.60 | 0.03 |
| spade4tennis | frame | 0.8342 | 9.89 | 0.03 |

`stable-animator` refused at generate and `mofa-video` at construct, both by
design (§2.9).

**Three things the table alone would not say, and the paired comparisons do.**

1. **Every engine loses to the static-copy floor**, at 2.5σ to 10.6σ. There is
   no exception and no near miss.
2. **The top three are not separable.** upscale-refine vs seg-controlnet is
   0.0σ; seg vs animate-anyone is 0.4σ. So "AA is the best engine" is not
   supportable — and neither is any generative engine beating a plain upscaler.
   Only two adjacent pairs in the whole ranking are clear.
3. **PSNR and LPIPS give different orders.** pix2pix is 2nd on PSNR and 7th on
   LPIPS. That is the concrete case for §2.5's choice of ranking key, measured
   rather than argued.

Three engines — ip-adapter-controlnet, pix2pix, spade4tennis — score **worse
than showing a completely different player**.

AA's clip-mode 0.5692 reproduces §2.7's 0.570, measured there on 4 clips and
here on 12, on a different GPU. Clip mode costs 6.2 GiB against 3.3 for a
ControlNet, at the same ~1 s/frame; `subsec:eval-operating` needs both.

#### The cross-appearance control, and why its own reading is withdrawn

Hold the model, pose, target and metric fixed; vary only which keyframe the
engine sees — its own, or a donor from a different video. **This was the test
BP10 and BP12 called decisive.** It is not, and the reason is worth more than
the result.

A prediction was written down before the last two arms finished
(`outputs/bp12-cross-appearance-prediction.txt`): the ControlNets should land at
0.03–0.09 LPIPS, well below Animate-Anyone's +0.107, because §2.4 says they have
no trained appearance path. It named the failure branch too — *a ControlNet near
+0.107 means investigate the harness before reporting anything about AA.*

They came in at **+0.166 and +0.176**, above AA. So the copying baselines were
driven through the identical code path:

| Arm | Δ LPIPS (wrong − right) | share of a paste | Δ PSNR |
|---|---|---|---|
| **static-copy — no model at all** | **+0.285** | **100%** | +4.45 dB |
| **upscale-refine — non-generative** | **+0.185** | 65% | +2.64 |
| seg-controlnet | +0.176 | 62% | +3.43 |
| pose-controlnet | +0.166 | 58% | +2.93 |
| animate-anyone (clip) | +0.107 | 37% | +3.46 |
| ip-adapter-controlnet | +0.055 | 19% | +0.25 |

n=12 clips, paired, every comparison in `report.json`.

**The arm with no network wins the test.** The delta measures how much of the
reference image survives into the output — copying — which a paste maximises by
construction. It cannot say whether a model renders the right *person*, because
the arm that renders nothing tops the scale.

On PSNR this kills BP10's threshold outright. BP10 set **"≥ +3 dB means
ReferenceNet works"**. A pasted keyframe scores **+4.45 dB**. That gate would
have certified a paste as having a working ReferenceNet.

**The harness is not the fault, and said so.** static-copy driven through the
cross-appearance path scores exactly the 0.285 computed independently from the
two roster baselines. The instrument agrees with itself; the reading was wrong.

**What survives.** The control is kept as a measure of *dependence on the
reference*, always reported beside the arm's own score against the floor, and
withdrawn as a test of "uses appearance". `judge_cross_appearance` now refuses
to classify without a copying anchor, and `report.py` re-judges stored records
with the current bound rather than the status they were written with — this
bound has been wrong once already.

**What is left standing about Animate-Anyone.** It is the only arm whose quality
sits in the top group while depending *least* on copying the reference: +0.107
against seg-controlnet's +0.176 at 2.5σ, with LPIPS 0.5692 against 0.5595 at
0.4σ. That is consistent with feature injection rather than pixel blending, and
it is **not** established by this test. Establishing it needs a measurement that
separates "the output moved" from "the right person appeared" — which the
literature does with an identity metric (CSIM/ArcFace), not with a distance to
the target frame. §7 should carry that as work, not as a settled result.

**And the standing negative is unchanged.** No engine on this roster
reconstructs a usable player. The best of them sits 0.108 LPIPS above a paste of
the keyframe, and a paste is not a codec.


### 2.11 The caption channel was trained and has never been switched on

Found 2026-08-23 while checking, not assuming, what appearance channels exist.

`scripts/train_controlnet.py` reads a **per-track BLIP caption** and falls back
to a generic prompt only when none exists. **114 caption files exist**, one per
track, and they carry appearance:

> *"a man in a purple shirt and blue shorts playing tennis, photorealistic
> tennis player, broadcast sports shot"*

**53 of 114 (46%) name a colour**; there are 57 distinct captions over 114
tracks. At inference `src/components/generation/controlnet.py:73` hardcodes the
*fallback* — `"photorealistic tennis player, broadcast sports shot"` — for every
frame of every clip, and `ConditioningBundle` has no caption field at all, so
there is no way to pass one.

**Every ControlNet number this project has measured, including §2.10's, was
taken with the text channel disabled.** §2.3 quoted the fallback prompt as
though it were the only prompt, which is how this stayed hidden through three
rounds of engine triage.

**Occurrence nine of the standing failure mode**: a pathway exists, is trained,
passes its tests, and is not driven. It is also the cheapest outstanding thing
that could move the roster, because it needs no training —
`plans/BP17-caption-channel.md`.

**Do not over-read it in advance.** A caption is a few tokens through CLIP text
encoding; it can say "purple shirt", not *this* player, and §2.7's own
literature note is that CLIP embeddings lack fine spatial detail. Half the
tracks name no colour. The expected result is a small effect or none — but every
roster number is currently labelled with a training condition that was never
true, and that is worth one probe run to correct.

**What this changes about the appearance story.** Three channels are registered;
their real status differs, and §6.2's roster should be read with this table:

| Channel | Registered as | Actual status |
|---|---|---|
| text / caption | `pose-controlnet` (alias `caption-controlnet`) | **trained, never driven** |
| keyframe / reference image | `pose-ref-controlnet` | trained, measured, **failed for a known architectural reason** (§2.4) |
| latent / image embedding | `ip-adapter-controlnet` (declares `appearance:image-embedding`) | **trained, uses appearance, still loses to a paste** (§2.17). The tennis directory named `ip-adapter-controlnet` remains the mislabelled segmentation ControlNet of §2.3 and is not loaded. |

Of three appearance pathways, one is switched off, one failed for a reason we
understand, and one now works as a *path* without working as an engine: it
moves when the reference is shuffled, and a paste still wins.


### 2.12 An instrument that can tell the right body from a moving output

Built and calibrated 2026-08-23 (`BP18`). `outputs/bp18-reid-calibration.txt`,
bounds in `outputs/bp18-reid-bounds.txt` written first.

§2.10 left a gap: the cross-appearance control cannot say whether a generator
renders the *right person*, because a paste tops it. Every proposal that follows
— retraining, a new dataset, a new architecture — would have been judged with a
yardstick that cannot answer the question. So the instrument came first.

**Not faces.** CSIM/ArcFace is the literature's answer and does not apply here:
we reconstruct bodies in motion, a player box averages ~88k px in a 4K frame,
and the face inside it is a few tens of pixels, often turned away. **DISTS is
not the answer either** — it is a distortion metric with LPIPS's structure, so a
paste tops it too; it stays for comparability with Sparse2Dense, which headlines
DISTS, and it is *not* registered on the metric axis today.

**What was built.** `reid`: OSNet x1_0 trained on MSMT17, embedding a full-body
crop; cosine similarity against the ground-truth target frame, pose-invariant by
design. Architecture vendored (MIT) rather than installed, because `torchreid`
risks moving torch and several pinned forks here cannot survive that; nothing
downloads at runtime. Beside it `palette`, an 8-bin RGB histogram intersection —
deliberately crude, as a check on the learned one.

**The decisive gate is ground truth, with nobody's judgement in it.**
`track_<id>_metadata.json` carries a `frame_id` per entry, so two tracks in one
scene whose frame ranges **overlap are two people on court at the same instant**
— necessarily different individuals. Scored at that shared frame, across all
seven videos:

| | n | `reid` | `palette` |
|---|---|---|---|
| same person *(one track, two frames)* | 106 | 0.8663 ± 0.0094 | 0.8637 ± 0.0072 |
| **different people** *(two tracks, one frame)* | 53 | **0.5315 ± 0.0171** | 0.4147 ± 0.0226 |
| **separation** | | **0.3348 ± 0.0195 (17.1σ)** | 0.4490 ± 0.0237 (19.0σ) |

Hand labels are secondary and no longer carry the gate; they remain only for the
two things derivation cannot do — marking **officials**, and same-player pairs
*across scenes*. Those anchors, on 27 hand-labelled tracks in three videos:

| anchor | n | `reid` | `palette` |
|---|---|---|---|
| identical | 27 | 1.0000 | 1.0000 |
| same track, offset 8 *(ground truth)* | 27 | 0.8506 ± 0.0185 | 0.8455 ± 0.0147 |
| same player, other track *(inferred)* | 42 | 0.7200 ± 0.0150 | 0.4961 ± 0.0300 |
| **different player, same match** | 52 | **0.5097 ± 0.0130** | 0.3644 ± 0.0173 |
| player vs official | 14 | 0.3943 ± 0.0242 | 0.5018 ± 0.0390 |
| different video | 60 | 0.3739 ± 0.0120 | 0.1937 ± 0.0066 |

**`reid` passes its gate at 17.1σ on ground-truth pairs**, and the
hand-labelled anchors agree (0.3410 ± 0.0226, 15.1σ) with a monotone ordering
across all six. Usable.

**Three bounds fired high, and the bounds were wrong.** They were written as
though cosine similarity had a natural zero for "unrelated". It does not: every
upright human in a tennis crop shares a large component with every other, so two
*different* people already score ~0.53. Quoting 0.51 against an imagined zero
overstates the distance about twofold.

**Recording that was not enough, so it is now a code path.** `IdentityScale`
carries the two measured anchors and prints a score between them — 0.5097 reads
as *"−7% of the way from a different person at 0.5315 to the same person at
0.8663"*, which is the honest sentence. `TENNIS_SCALE` holds the constants for
this dataset and backbone, dated, with an instruction to re-measure; an inverted
scale raises rather than returning a number. A lesson that lives only in prose
gets re-learned.

**The companion earned its place immediately**, by disagreeing:

- `palette` separates two players in one match *more* sharply than `reid`
  (0.481 vs 0.341). Kit colour really is most of the signal, and `reid` is
  partly a colour detector — worth knowing before anyone calls it "identity".
- But `palette` is **fooled by the official**: an umpire in a black tracksuit
  scores *higher* against a dark-shirted player (0.502) than two players score
  against each other (0.364), while `reid` puts the official lowest of all
  (0.394).
- And `palette` collapses across scenes for the same player (0.496 against
  `reid`'s 0.720) — a histogram has no invariance to lighting or scale.

So: **`reid` is the identity number, `palette` is the check, and where they
disagree, go and look.** An invariant asserts that disagreement, because if the
two ever collapse into one measurement the companion has stopped buying
anything.

**What this does not claim.** Cross-track "same player" labels are inferred from
kit, and kit is part of what both metrics read; the primary same-player anchor
is therefore same-track-different-frame, which is ground truth. Both players in
a match share a court, a broadcast and a lighting rig, all of which push their
embeddings together — so the decisive test is conservative rather than
flattering. And this settles nothing about any engine. It is an instrument; §7's
question is now answerable, not answered.


### 2.13 The headroom harness is real; its number is synthetic

`BP13` landed 2026-08-23 and did most of what it was asked. It also measured the
wrong thing, and the distinction matters because a project decision was taken on
the result.

**What is sound.** `experiments/headroom/` encodes, removes and differences at
matched quality; bounds were written before the encode; nulls were run (empty
mask 0% saving, duplicate-encode rate ratio 1.0); the tool was resolved by path
and version (`/opt/local/bin/ffmpeg n7.1.1-56-gc2184b65d2`). The paper paragraph
is scoped honestly and says "synthetic" in its first clause.

**What is not.** The clip is **96×128 and synthetic**
(`experiments/headroom/synthetic.py`); nothing under `experiments/headroom/`
reads `assets/`. So:

| | measured | real |
|---|---|---|
| player area | ~4.7% of pixels | **~2.3%** (from 4K bboxes; §2.6 puts one player at 1.07%) |
| codec | libx264 alone | the paper's ladder is **AVC/HEVC/AV1/VVC** (§7) |

Both push the same way. Getting player area wrong by 2× matters because the
quantity *is* what the players cost. And a stronger codec compresses the
near-static background better, so the players take a **larger** share of what
remains — making AVC the conservative rung and 12.2% possibly a floor.

**The numbers, so they are on the record and not re-derived:** FG plate removal
**12.2% ± 0.26 pp** (n=3 seeds), flat fill 3.6% ± 0.24, court-median 9.0%;
BG panorama plate + homographies **17.4×** smaller than JPEG-coding the
background over 24 frames — against a *still-image* baseline, which overstates
conventional cost relative to inter-predicted video.

**The alarm is real and survives.** Flat fill was written down as the arm that
*overstates* the prize. It **understated** it — 3.6% against plate's 12.2% —
because a grey hole in a green court is a high-contrast object and the encoder
spends bits on its edges. Court-median fill sat between at 9.0%. So the
bracketing assumption in `BP13` is void: **plate is both the honest
reconstruction and the cheaper one to code here.** Carry all three fills forward.

**The Wave-3 fork is WITHDRAWN.** It read "FG is in the 10–25% modest band;
background and residual carry the paper; generator work is an improvement
track". That is a project-steering decision taken from a 96×128 toy, and it does
not stand. Nothing about direction is settled until `BP20` reports.

**The material for the real measurement is on this host**, and the recorded
reason for skipping it — "no full-frame player masks" — was wrong.
`assets/raw_4k/` holds seven 3840×2160 matches, already AV1; and
`track_<id>_metadata.json` carries a per-frame **bbox in full-frame
coordinates**, which composites with the crop alpha into exactly the mask
thought to be missing. §2.12's identity work reads the same sidecars.
`plans/BP20-headroom-real-ladder.md` is the replacement.


### 2.14 The real headroom: the premise holds across the ladder

Measured 2026-08-23 on **real 4K** (`BP20`), replacing §2.13's synthetic number.
`outputs/bp20-headroom/`. Two broadcast scenes — `alcaraz_highlights/scene_000`
frames [38:86] and `federer_djokovic/scene_001` frames [93:141] — 3840×2160,
48 frames each, from `assets/raw_4k/`.

**The correctness gate that makes the rest meaningful.** Pasting each track crop
back into its sidecar `bbox` reproduces the source frame at **MAE 0.0**, under
the `extract_24_frame_id` convention. Native-fps and positional conventions both
failed it — §2.2 biting for the third time, and the reason this check is now
mandatory before any headroom byte is trusted. Nulls: empty mask saves 0.0,
duplicate-encode rate ratio 1.0.

#### The result the paper needs

| Codec | FG saving, plate inpaint (BD-rate, matched quality) |
|---|---|
| AVC | **0.244 ± 0.017** |
| HEVC | 0.234 ± 0.017 |
| AV1 | 0.229 ± 0.030 |
| **VVC** | **0.167 ± 0.015** |

Player area by **alpha silhouette** is **0.55%** and **1.02%** of frame. So:

> **A player pixel costs 15–47× what an average pixel costs.** The players are
> about 1% of the picture and about a quarter of the bitrate.

That concentration is the motivating example, measured rather than asserted.
n=2 clips, so it is a direction, not a final figure — the project's own bar is
n≥8.

#### VVC behaves differently; the rest of the ladder agrees

The prediction was **AV1 ≥ HEVC ≥ AVC** on FG saving. Read as a 4-point
ordering it fails, and this section first said the headroom "shrinks as the
codec strengthens". **That was over-reading n=2.** Paired on the same clips:

| pair | per clip | mean |
|---|---|---|
| AVC − HEVC | +0.009, +0.010 | +0.009 |
| AVC − AV1 | +0.001, **+0.029** | +0.015 |
| **AVC − VVC** | **+0.078, +0.075** | **+0.077** |

**AVC, HEVC and AV1 sit together** — the gaps between them are small and, for
AV1, inconsistent between the two clips (+0.001 on one, +0.029 on the other),
which is what noise looks like at this n. **VVC is a step down of ~0.077, and
that step is the one thing here that repeats cleanly on both clips.**

So the honest statement is not a trend. It is: *the premise holds across the
ladder, and VVC is the exception worth naming.*

**Two candidate explanations, and they are not separated yet.**

1. VVC codes the player region better, so removing it saves less. Plausible —
   modern tools pay off on hard moving detail — but a claim about codec
   generations cannot be made from one codec at n=2.
2. **VVC ran at a different operating point.** QP 32/40/**47** against
   32/40/48 for everything else, because `libvvenc` 1.11.0 writes an empty
   bitstream at 48 on some 4K fills. A different rate ladder changes the
   BD-rate integration interval. **This confound must be ruled out before the
   codec is blamed** — re-run VVC at a matched ladder, or integrate all codecs
   over the common interval.

Report every cell, flag VVC, claim no trend.

#### The background is worth 34–69%, not "orders of magnitude" and not nothing

This section first said the panorama half was "dead on real footage". **That was
wrong, and wrong by cherry-picking**: it quoted 1.39×, the single least
favourable cell in the table, and generalised from it. The table:

| Codec | alcaraz_highlights | federer_djokovic |
|---|---|---|
| AVC | 1.391× / **34.4%** | 1.830× / **58.6%** |
| HEVC | 1.459× / 38.3% | 1.879× / 57.2% |
| AV1 | 1.525× / *not reported* | 2.245× / *not reported* |
| VVC | 2.195× / **59.9%** | 2.487× / **68.9%** |

*(ratio of conventional to panorama rate; BD-rate saving at matched quality.
Homographies cost 1728 B per clip.)*

A transmitted plate plus per-frame warps saves **34–69% of the background
bitrate**. That is a real and reportable result. What it is *not* is the 17.4×
the synthetic JPEG-still comparison suggested — inter prediction already handles
a near-static background well, so the panorama competes with a strong baseline
rather than a trivial one. The pre-written [1.5, 12] band was derived from that
discredited synthetic number and is not a fair gate.

**And the background inverts the foreground's pattern**: BG saving *improves*
with codec strength (VVC best at 59.9% / 68.9%) while FG saving is lowest for
VVC. Both halves are worth showing.

**AV1's BG BD-rate is not reported** — PSNR overlap between the arms was 0.46
and 0.20, below the 50% the BD-rate implementation requires. That is a gap to
close by widening the QP sweep, not a result.

#### Bounds that were wrong, and why

- **Player-area band [0.015, 0.035]** was written from *bbox* area; the
  measurement correctly used the *alpha silhouette*, which is roughly half.
  Wrong by construction, not retconned.
- **FG bands** were carried over from the synthetic run and were too low; AVC and
  HEVC both landed above them.
- **Flat fill understates the prize on real 4K too** (0.12 against plate's 0.24),
  confirming §2.13's alarm on real content. "Flat is an upper bracket" stays
  void; plate is both the honest reconstruction and the cheaper one to code.

VVC ran at QP 32/40/**47**: `libvvenc` 1.11.0 writes an empty bitstream at 48 on
some 4K fills, and stepping down beats pretending the third curve point ran.

### 2.15 The caption channel is worth nothing measurable

`BP17`, run 2026-08-23 on the same 12 clips, seed 42, offsets 1–8 — everything
identical to §2.10 except the prompt. `outputs/bp17-caption/`.

**The control is exact.** Both no-model arms moved by **0.000 ± 0.000**:
static-copy 0.4505 and unrelated-image 0.7358, unchanged to four decimals. The
two runs are comparable.

| Arm | captions on − generic prompt | verdict |
|---|---|---|
| pose-controlnet | +0.020 ± 0.014 (1.5σ) | suggestive that captions are *worse*; not a result |
| seg-controlnet | +0.002 ± 0.011 (0.2σ) | inside noise |
| ip-adapter-controlnet | −0.002 ± 0.008 (0.2σ) | inside noise |
| trajectory-controlnet | +0.001 ± 0.019 (0.1σ) | inside noise |

Positive is worse; LPIPS is lower-better. **Switching on a channel the models
were trained with changes nothing**, and if anything mildly hurts. That is what
§2.11 predicted, and it is worth having: it retires the possibility that the
§2.10 roster was measured unfairly. The defect is still real and still fixed —
inference can now reach the channel — but the channel is not where the
appearance problem lives.

Checkpoint provenance was checked rather than assumed: captions landed on disk
2026-07-01, the trainer began reading them in `d1efbcf` (2026-07-06), and pose
epoch 10 (2026-07-07) and seg epoch 7 (2026-07-06 17:53) both post-date it. For
those two this really was switching a channel *back* on. `ip-adapter` loads a
stock OpenPose ControlNet, so for that arm it only switches SD's text encoder on.


---

### 2.16 The platform runs end to end — and it is not measuring rate

`BP23` (2026-08-26) drove all three tier configs plus two controls through the
runner on `alcaraz_highlights/scene_000` (8 frames, 3840x2160, players 0.573% of
pixels, paste-back MAE 0.0). **P0 item 1 is closed.**

| run | wall | delivered | residual bytes |
|---|---|---|---|
| all-off (control) | 11.0 s | bit-identical, PSNR inf | 0 |
| residual-absent (control) | 22.9 s | 34.88 dB | 0 |
| `tier_fast` | 29.1 s | 43.75 dB | 1,241,086 |
| `tier_balanced` | 131.6 s | 48.28 dB, SSIM 0.9970 | 2,523,202 |
| `tier_quality` | 299.6 s | 56.74 dB, SSIM 0.9999, VMAF 97.4986, LPIPS 0.0002 | 37,919,751 |

**These are not rate points.** The codec stage is an identity round-trip and no
encoder binary runs, so every byte count is pixel payload, not coded bytes, and
`transport_to_source_ratio` is not a compression ratio. The caveat is a field in
`outputs/bp23-tier/report.json`, not just prose. BD-rate needs a real bitstream
(`BP24`).

**Eight defects had to be fixed to get there, four of them silent wrong answers
rather than failures.** The one that mattered most: the size ledger read the
residual's dense array size, which does not shrink when the block gate zeroes a
block — without that fix the entire tier ladder would have been flat. Also: two
different PSNR conventions inside one ladder (47.63 vs 48.28 dB on identical
pixels), a residual-absent corner that delivered the source itself at PSNR inf,
and an encoder/client symmetry check comparing different pipeline points.

**A second confirmation of §2.6 from an independent direction.** The unaided
static plate scores 34.88 dB on the frame but **14.30 dB on the object** against
39.46 dB on the background — a 25 dB gap concentrated on 0.57% of pixels.

**27 of 32 config fields are inert** (`outputs/bp23-tier/inert-config-fields.json`,
driven one field at a time rather than read off the code). Only
`evaluation.metrics` and four `residual.*` knobs change a run. Generation knobs
are inert *in this corner* because generation is off — a statement about the
corner, not the knob — but detector, pose, appearance, motion and temporal names
currently reach nothing. `BP26` closes this; until then the ablation lattice is
not measurable.

**Metric calibration, and two findings that outlive this stream**
(`outputs/bp23-tier/metric-calibration.json`). All four metrics order correctly
at 4K. But **VMAF's ceiling on this content is 97.54, not 100**, and it **floors
at 0.00 for both severe blur and an unrelated clip** — nothing resolves below its
floor. And **LPIPS's ordering inverted at 960x540** while holding at 4K:
calibration anchors do not transfer across resolution. Given that two metrics
here were broken until 2026-08-23 (§2.7), both belong in the invariants (`BP27`).

**Reported, not patched:** `STAGE_CODEC.optional_inputs` in
`src/contracts/lattice.py` omits `generated-frames`, so a generation-on /
residual-off corner may order the codec first and cannot deliver a
reconstruction.


### 2.17 IP-Adapter uses appearance and still loses to a paste

`BP25`, 2026-08-26, GPU 0. Bounds in `outputs/bp25-ip-adapter/bounds-before-run.json`
were written before the first generation. Protocol matches §2.10: 20 steps,
12 clips, offsets 1–8, object-bbox LPIPS, seed 42. Extra check after a pleasing
item-level result: the same comparisons on **clip means** (n=12), in
`outputs/bp25-ip-adapter/clip-means.json`.

The 4-step stop-eval is a tripwire, not a ranking instrument. Same stock adapter
at 4 vs 20 steps separates at 3.5σ (n=12, offset 8), so 4 steps is not blind —
but at 4 steps the stock adapter is worse than an unrelated photo (3.8σ). Ranking
stays at 20 steps against real-image anchors.

| Arm | object LPIPS | `reid` on TENNIS_SCALE |
|---|---|---|
| static-copy | **0.4505 ± 0.0220** | 0.9135 ± 0.0087 (paste; same-person alarm fires, correctly) |
| unrelated image | 0.7358 ± 0.0075 | 0.4998 ± 0.0064 |
| stock IP-Adapter | 0.7586 ± 0.0092 | 0.5519 ± 0.0157 (+6% of span) |
| **epoch 1 (best)** | **0.6922 ± 0.0094** | 0.5647 ± 0.0147 (+10% of span) |
| epoch 2 | 0.6953 ± 0.0103 | 0.5589 ± 0.0149 |
| epoch 3 | 0.6947 ± 0.0101 | 0.5604 ± 0.0150 |
| epoch 1, shuffled appearance | 0.7662 ± 0.0085 | 0.4893 ± 0.0106 |

n=96 (12 clips × 8 offsets) unless noted. Stock reproduces §2.10's 0.7606.

**What holds at clip level (n=12), which is the extra check.** Epoch 1 beats stock
(−0.066 ± 0.012, 5.5σ). It uses appearance: own keyframe vs shuffled, −0.074 ±
0.020 LPIPS (3.8σ) and +0.075 ± 0.021 reid (3.6σ). It still loses to a paste
(+0.242 ± 0.059, 4.1σ). Vs unrelated is only 1.3σ on clip means — suggestive,
not a result.

Inside the pre-written band (LPIPS 0.50–0.78, reid 0.53–0.72). Not identity.
Not a working engine. The stop rule was right that epochs 2–3 were flat; the
4-step number it stopped on was not a ranking.

Uni-ControlNet remains last. P0 item 5 closes on this scoped result.


### 2.18 A byte count that is a rate

`BP24`, 2026-08-28. Until this landed, every byte count in the project was an
array size. `PLAN.md` §2.16 recorded the consequence: no compression ratio could
be quoted from any tier run.

**The boundary decision** (§3): the codec stage codes the **transmitted
payload** — plate, appearance, motion, residual, metadata — not the delivered
pixels. Encoding the delivered frames would measure PointStream's output
re-encoded, double-counting the reconstruction.

| component | raw | coded | measured on |
|---|---|---|---|
| background plate | 24,883,200 B | **342,694 B** | one real 4K frame, `jpeg:50`, plate PSNR 40.21 dB |
| residual | 9,331,200 B | **2,545 B** | 6 frames 960x540, av1 CRF 35, 2.5% non-zero |

**Both are the easy case.** A sparse residual against a static plate is the
friendliest possible input; re-measure on high motion before quoting either.

**A mixed ledger refuses to report a ratio.** `SizesBytes.raw_parts` names any
component still counted as an array size, and `transport_to_source_ratio` is
withheld entirely while that list is non-empty — including through `__add__`, so
summing chunks cannot launder an uncoded chunk into a total that claims to be a
rate.

**Read `plans/BP24-findings.md` before quoting any rate.** Eleven findings,
including the one that cost the most time: counting coded bytes while
reconstructing from the pre-codec array passes every test, uses two real
numbers, and yields a fictional rate-distortion point.

### 2.19 What `exact` means, and what `actor_reference` turned out to be

`BP24` continued, 2026-08-28. Report: `plans/BP24-ladder-report.md`.

**`WireCost.exact` has one meaning now.** It used to mean "follows from declared
parameters rather than from a model of the encoder", which was unambiguous only
while nothing here ran an encoder. Both residual paths sat at `exact=True` on
top of a basis describing an in-memory array, and two separate mechanisms were
each deciding whether a byte count was a bitstream. It now means **these bytes
are transmitted** — a measured bitstream, or a packed payload at a declared
quantization with no further coding step configured. Both pre-codec residual
paths are `exact=False`; the absent path stays an exact zero, because sending
nothing is a measurement.

**`actor_reference` is a wire cost, and it clears the ledger** — driven per
backend, not argued (`outputs/bp24-ladder/appearance-cost.json`).
`compressed-image` returns a real JPEG bitstream whose size moves with quality
(1,448 / 2,020 / 7,732 B at q20 / q60 / q95) and which decodes back to the crop
at MAE 2.83. `image-embedding` and `diffusion-latent` return a packed float16
buffer whose length equals the declared cost exactly: a wire cost, but **not a
coded one**. The flag comes off the payload the appearance stage produces, and
a payload that does not state `exact` still withholds the ratio, so a backend
added later cannot clear itself by default.

**Two defects found on the way.** `RunResult.frames` had stopped being the
delivered clip — it carries the residual as the residual stage produced it,
while `sizes` costs the coded one, and the two differ by exactly the axis a rate
ladder sweeps (findings §8). `delivered_frames` now exists. And `bd_rate`'s
overlap guard was a proportion, so two flat curves overlapped perfectly and
returned a confident number over 0.5 dB; an absolute 3 dB floor closes it.

**The clip every BP24 ratio was measured on is the most static of the eight
cached windows**, by 23x against the most dynamic
(`outputs/bp24-ladder/motion-survey.json`). §2.18's "both are the easy case" is
now measured on the input rather than inferred from the output.

### 2.20 The ladder ran, and PointStream loses to the codec it is built on

`BP24` concluded, 2026-08-28. `PLAN.md` §7 **P0 items 2 and 3 are closed.**
Report: `plans/BP24-ladder-report.md`. Bounds written before the first encode:
`outputs/bp24-ladder/bounds-before-run.json`.

**Paired arms, one codec on both, same preset** — the design
`plans/BP24-findings.md` §1 settled on, so the preset cancels. Y-PSNR, which is
the conventional BD-rate axis.

| codec | preset | BD-rate | overlap |
|---|---|---:|---|
| av1 | `10` | **+116.8%** | 39.45-44.02 dB |
| hevc | `ultrafast` | **+166.8%** | 38.72-43.47 dB |
| avc | `veryfast` | **+165.9%** | 35.76-43.87 dB |
| vvc | `faster` | **+378.1%** | 35.28-43.65 dB |

**Do not rank these against each other.** The presets are not equal effort, so
an ordering of the magnitudes would be measuring the presets. Each is a gain
against that codec at that preset, which is the claim shape the paper needs.

This is on `alcaraz_highlights/scene_000`, the **most static** of the eight
cached windows (inter-frame MAD 0.33 against 7.70 for the most dynamic) — the
friendliest content available. On the dynamic clip there is **no BD-rate at
all**: PointStream saturates at 31.0 dB, av1's cheapest rung is 38.0 dB, and the
curves do not overlap.

**The cause is the plate, not the residual.** The plate is 88-91% of the payload
at every rung of every sweep. The unaided corner — plate plus pasted crops, no
residual — is 487,643 B at 35.37 dB against av1's 85,995 B at 39.45 dB, so the
plate has already lost before the residual is asked for anything. The residual
is the opposite: 0.9% of the payload for 5.4 dB on static content, up to 14.8 dB
over unaided on dynamic content. **The plate is still the first source frame
rather than a stitched panorama, and that stub is now the single largest lever
on the project's rate.**

**Three defects were found and fixed on the way**, one of them a corrupted
pipeline output rather than a bad number: the decode step named no `-c:v`, so
ffmpeg re-encoded to Matroska with x264 at its default CRF, capping every
quality `coded_roundtrip` returned — including the residual the runner
delivers. `RunResult.frames` had stopped being the delivered clip. And
`bd_rate`'s overlap guard was relative and could not see a flat curve. All three
are in `plans/BP24-findings.md` §§8, 14 and 2.

**Scope, stated rather than implied.** Generation is off in every tier config,
so no generative decoder was measured. Eight frames is the least favourable
amortisation a fixed plate cost can get. Y-PSNR only; PointStream's case has
always been argued perceptually.

### 2.21 The plate is the lever, and it has three of them

2026-08-29, following §2.20. Briefs: `plans/BP29-plate-rate.md` (where
PointStream can win) and `plans/BP30-background-stream.md` (the background as a
stream). Findings: `plans/BP24-findings.md` §§16-18.

**(a) JPEG is the wrong codec for a 4K plate.** Same still, matched fidelity
near 38 dB: JPEG **283,431 B**, av1-intra **79,726 B**, vvc-intra **68,477 B** —
a factor of 3.6 to 4.1 on 88-91% of the payload, for no architectural change,
since a modern intra frame is what AVIF and HEIC already are. It is not even new
code for one route: `background.codec` already accepts `roi-video`, a
single-frame x264 encode, and nothing has ever measured it against `jpeg` —
because that axis reached nothing at all until BP24 wired `make_background`. A
config axis only ever set to one value is indistinguishable from a constant
until somebody drives the others.

**(b) The next plate need not be paid for in full.** Coding plate B as a
**P-frame** referencing plate A saves **31-53%** with av1 between points of a
match. This **retracts §17**, which subtracted two plates pixel by pixel, found
the difference cost *more*, and closed the door: subtraction destroys the
spatial correlation a transform coder depends on, while inter prediction does
block-wise motion search, which is what a panned camera needs. The control —
two consecutive frames of one scene — comes in at 1.2-3.3%, which is what makes
the arms readable.

**Each scene's payload stays independent of every future scene**, because a
P-frame references the *reconstruction*, which both sides hold without knowing
the future. That is how every live encoder works. It holds only under low-delay
P — no B-frames, no lookahead, no multi-pass — and **§18's numbers were measured
without that constraint**, so they must be re-measured before being quoted as
achievable live.

**Two cautions carried forward.** The saving is codec-dependent: libx265's own
rate-distortion decision chose intra where av1 found inter worth 31%. And
**PSNR distance does not predict coding distance** — the pair further apart in
PSNR saved *more* — so a reference search must rank by structure (Canny edge
overlap) rather than by pixel similarity, and the proxy must be validated
against trial encodes before it is trusted.

## 3. Architecture

Enough here that seven parallel sessions do not make conflicting decisions.
Everything deeper is in `src/contracts/`, which is authoritative, or in the
paper's System Design section, which explains why.

### Layers

```
  contracts/    protocols, schemas, capabilities, config — imports nothing heavy
      ▲
  components/   one package per axis, each with a registry table
      ▲
  pipeline/     stage DAG, encoder, decoder — never knows which backend was chosen
      ▲
  runner/       ONE run path: chunk loop, routing, accounting, evaluation
      ▲
  experiments/  matrices, sweeps, campaigns — consumes runner as a library
```

**Dependencies point inward, always**, enforced by
`python -m src.contracts.layers`. The arrangement being replaced had experiment
scripts shell out to the CLI and scrape stdout, which is what an unchecked
boundary decays into.

### What `src/shared/` is (BP22, 2026-08-26)

**(b) — `src/shared/` stays condemned.** It is not a layer. `src.contracts.layers`
already lists `src.shared` in `LEGACY_PACKAGES` (`src.decoder` was deleted this
wave); the diagram
above has no place for a junk drawer. Promoting it to a real layer would freeze
training helpers, a tennis dataset, skeleton drawing, old schemas, video IO,
and leftover eval metrics as architecture. Those do not share a contract.

Evidence on this tree (`7cf8e89`): `src/pipeline` and `src/runner` import
nothing from `src.shared` or `src.decoder`. The only rewrite-tree inbound was
`src.components.generation.animate_anyone_runtime` → `src.shared.dwpose_draw`
(moved to `src.components.generation.dwpose_draw`). Everything else that still
imports `src.shared` is a pre-rewrite script, legacy `src.transport`, or a
top-level `tests/test_*.py`. BP14's stop rule (`src/shared/training/`) is new
code that belongs under `src/experiments/` or a training helper in
`src/components/` — not a reason to invent a sixth layer.

**This wave does not move** `src/shared/tennis_dataset.py` or
`src/shared/training/**` / `scripts/train_controlnet.py` (Stream B is live on
the training path). They stay in condemned `src.shared` until that stream
lands. `src/shared/{schemas,interfaces,tags}.py` stay too: the only caller is
legacy `src.transport.disk`, which this stream does not own. `src.decoder` and
`scripts/eval_checkpoint.py` are gone — that script was the decoder's last
caller. Remaining pre-rewrite training scripts keep ``tennis_dataset``. The
rest of `src/shared/` that this stream could delete is gone.

### What the codec stage codes (BP24, 2026-08-26)

**Decision: the codec stage codes the transmitted payload, not the delivered
pixels.** C3 left this stage an identity round-trip and did not choose; this is
the choice, made before any encoder was bound.

PointStream does not transmit a pixel grid. It transmits a background plate
established once, per-object appearance, per-frame motion, an optional
corrective residual, and metadata; the client *reconstructs* frames from those.
Running an encoder over the delivered frames would measure "PointStream's output,
re-encoded" — a number that double-counts the reconstruction and corresponds to
nothing the system sends. So the encoder runs over **each transmitted component**,
and `byte_count` is the sum of real coded sizes.

**The contract.** The codec stage keeps returning the delivered `frames`
unchanged — reconstruction is not its job and quality must not move when only the
accounting changes. What changes is the accounting: it returns a per-component
breakdown of **coded** bytes alongside the total. The existing raw figures stay,
under names that say they are raw, so BP23's table remains comparable and the
change in meaning is visible rather than silent (`plans/BP24-encoder-boundary.md`
step 3).

| Component | Today | After BP24 |
|---|---|---|
| background plate | `nbytes` of the raw plate, and `_panorama_bytes` says so | intra encode via `background.codec` |
| residual | `ResidualResult.payload.byte_count` | coded stream via `residual.codec` |
| appearance | measured payload bytes | coded where the representation is an image; unchanged otherwise |
| motion + metadata | serialised bytes | unchanged — already the real transmitted size |
| **all-off corner** | `source.nbytes` (raw) | **the conventional codec baseline** |

**The all-off consequence is the useful one.** With a real encoder bound, the
all-off corner stops being "raw source bytes" and becomes exactly the arm P0
item 2 compares against: the source coded conventionally at a chosen rung. The
codec ladder's baseline therefore falls out of this change rather than needing a
separate harness.

**What this does not license.** Until every component on a path is coded, that
path's total is not a rate point, and `transport_to_source_ratio` is not a
compression ratio. A partially-coded total is more misleading than an obviously
raw one, so a path reports its ratio only when no component in it is still raw.

**Checked, not assumed: the residual is not entropy coded either.**
`ResidualResult.payload.byte_count` is `int(stored.nbytes)` / `int(encoded.nbytes)`
(`src/pipeline/residual/signal.py:245,281`) — the size of a quantised, block-gated
array, not a coded stream. BP23's fix made that array reflect the block gate, which
is why the coarseness ladder stopped being flat; it did not make it a rate. So the
residual needs a real encoder under `residual.codec`, not a relabelling. The
`WireCost` record already carries `exact: bool` and a `basis` string for exactly
this distinction, and every component's cost should set them honestly.

### The ablation lattice

**Every component is optional, and the residual absorbs whatever the disabled
stages would have handled.** Turn off detection but keep the background: metadata
shrinks, encode time drops, the residual grows to carry the players. Turn
everything off, residual included, and what remains is the source video.

Three things follow, and they are why the architecture is shaped this way:

- Component ablations share **one currency** — BD-rate against a common anchor,
  measured identically for every component (§5). Payload alone is not the
  currency: two corners never land at the same quality, so a byte count on its
  own compares nothing.
- Alternative encodings of the same thing are directly comparable.
- The whole-frame baseline is not a special mode; it is a corner of the lattice.

No stage may be structurally required except codec, transport and metrics.
Graceful degradation to the baseline codec is a property of the architecture, not
a routing special case.

### Quality is always measured

There is no configuration where correctness can be assumed. The residual always
carries some coarseness, and generative inference is statistical — encoder-side
and client-side generation are not guaranteed to produce identical pixels.

So **every run reports quality**. Symmetry between the two sides is a design goal
*verified by measurement*, not a guarantee asserted by construction.
Deterministic stages get bit-identity checks; generative stages get closeness
measurement.

### The pairing constraint

Appearance and motion representations are independent axes, but not every
combination is decodable — a generator declares what it accepts, and config
validation rejects a pair nothing can decode, naming what would work. This is
what stops the design sprawling into combinations nothing implements.

---

## 4. The component catalogue

Every row optional unless marked. "When off" says where the work goes instead —
which is what makes the lattice measurable. Specs live in `plans/`.

| # | Component | When off | Brief |
|---|---|---|---|
| 1 | Scene classification | one span; no semantic-vs-fallback routing | B2 |
| 2 | Subject detection | subjects land in the residual | B2 |
| 3 | Subject selection | every detection treated as salient | B2 |
| 4 | Tracking / identity | no appearance reuse across frames | B2 |
| 5 | Appearance representation | generator has no appearance cue | B3 |
| 6 | Motion representation | object static after appearance; motion to residual | B3 |
| 7 | Temporal policy | every frame fully processed | B5 |
| 8 | Pose estimation | motion must be trajectories or encoded video | B2 |
| 9 | Segmentation | compositing falls back to heuristic masks | B2 |
| 10 | Rigid objects | rackets and balls land in the residual | B4 |
| 11 | Background model | background lands in the residual | B4 |
| 12 | Generation | subjects land in the residual | B3 |
| 13 | Residual | nothing corrects generation error | C1 |
| 14 | Codec | **required** whenever anything is transmitted | B1 |
| 15 | Transport | **required** | B5 |
| 16 | Metrics | **never fully off** — PSNR always runs | B6 |

---

## 5. Evaluation

**The paper's Evaluation section is the living trace.** Each element below has a
`GOAL`/`HOLE` marker pair there; a result lands by clearing its `HOLE` and adding
a `CLAIM(id): src=` line naming a real `outputs/` path. A result that turns out
weak either moves to an appendix or is deleted before submission — it does not
accumulate in a side file.

### The currency is BD-rate, not a byte count

**Two configurations will not land at the same bitrate, and will not land at the
same quality.** Comparing them at one operating point each therefore compares
nothing: a corner that spends more bytes and scores better has told you only that
more bytes buy more quality. This is the same error as comparing a
region-of-interest arm against a baseline at matched QP.

So **every comparison sweeps a rate ladder and compares curves.** Each lattice
corner is run at several residual coarsenesses (and, where relevant, several
codec rate points), producing a rate–distortion curve. Configurations are then
compared by **Bjøntegaard delta rate (BD-rate)** against a common anchor — the
average bitrate difference at equal quality, integrated over the overlapping
quality range — with BD-PSNR/BD-VMAF reported alongside.

This changes what "pays for itself" means, and it is the definition to implement:

> A component earns its place if enabling it **improves BD-rate against the same
> anchor**. Not if it reduces payload at one operating point, and not if it looks
> better.

Consequences that must reach the code:

- **A single run is never a result.** The unit of evaluation is a swept curve.
  An experiment harness that produces one point per configuration is producing
  something uncomparable.
- **Curves must overlap in quality** for BD-rate to be defined. Report the
  overlap range; a comparison over a sliver of shared quality is weak evidence
  and should say so.
- **A point comparison is valid only under dominance** — one arm better on both
  axes. Where that holds, say so and use it; where it does not, BD-rate is the
  only honest answer.
- **With the residual absent there is no guarantee**, so the comparison against
  the baseline codec is not "did we reconstruct exactly" but "where does this
  configuration's RD curve sit relative to the anchor's". That is the question
  every residual-free corner has to answer.

In priority order:

1. **PointStream against the codec ladder** — AVC/HEVC/AV1/VVC, with and without
   region arms, **at matched rate**. Rate, quality *and encode time* on the same
   axes. The core claim. Verified never run: every sweep in
   `outputs/codec_baselines/` has a null PointStream point.
2. **The ablation lattice** — each component off in turn, each swept across a
   rate ladder, compared by BD-rate against the all-off corner.
3. **The residual-coarseness curve** — absent through fine, plus a lossless
   ceiling calibration.
4. **Object representation** — keypoints versus sparse trajectories versus
   per-object encoded video, on identical objects, identical appearance, backbone
   held fixed. No published work does this comparison.
5. **Generalization** — the general/DAVIS profile, and pretrained versus
   fine-tuned generation across both domains.
6. **Real-time versus compute-unbounded operating points**, no minimum resolution
   or framerate imposed; whatever is achieved is reported.
7. **Perceptual and temporal metrics** — LPIPS and FVMD where they earn their
   place, alongside always-on PSNR.

Three honesty constraints, recorded in the paper as `NOTE` markers: the
encode/decode speed reality gets reported rather than omitted; dataset labels are
model-generated rather than human ground truth and deep annotation covers a
subset of scenes; and baselines get region control wherever the encoder supports
it, or beating them proves nothing.

---

## 6. Build phases

**Phase B — components.** ✅ Done. Seven parallel workstreams, one brief each in
`plans/`; each brief now carries a *Delivered* section recording what landed.

### Phase B′ — the engine roster

The narrative comes first. **The paper decides which experiments matter, the
experiments decide which models we need, and only those models get built.** This
is what keeps September from filling up with runs nothing cites.

**When a component has no marker to serve, that is a question, not a verdict.**
Raise it rather than silently dropping it — the paper is a work in progress and
does not yet contain every `GOAL` we will want. A component with no home may be
telling us the evaluation is missing something.

**We do not need every generator to work.** We need two flagships and a small
set of alternatives that differ along axes the paper measures.

#### 6.1 What the paper asks of the generator

| Paper slot | What it demands |
|---|---|
| `subsec:eval-ladder` | The **best** engine we have, to put PointStream's strongest RD curve against the codec ladder |
| `subsec:eval-object` | Keypoints vs sparse trajectories vs encoded video, **backbone held fixed** |
| `subsec:eval-general` | Tennis *and* DAVIS, pretrained *and* fine-tuned |
| `subsec:eval-operating` | A compute-unbounded point *and* something fast enough to have a real-time point |
| `subsec:eval-metrics` | Temporal coherence worth measuring with FVMD |

#### 6.2 The roster, re-decided on the clip-mode run

**Everything the previous version of this section said is void.** It ranked
engines on Wave-2 self-reconstruction PSNR (seg 16.2, pose 15.9, trajectory
14.9, pix2pix 15.4, upscale 14.5, AA 10.4, SPADE 12.0) — a task that asks each
engine to reproduce an image it was handed (§2.3), scored by a metric the
subfield rejects for this content (§2.5), through an LPIPS that could not tell a
match from an unrelated image (§2.7). The numbers below are from
`outputs/bp12-clip-roster/` (§2.10): the coding task, calibrated LPIPS as the
key, PSNR beside it, both baselines in the same session, and every comparison
paired with a standard error.

**The decision this run forces, and it is uncomfortable.** No engine holds a
quality flagship, because **no engine beats pasting the keyframe**, and the top
three — upscale-refine, seg-controlnet, animate-anyone — are not separable from
each other at n=12. Naming a flagship among them would be naming noise.

| Role | Engine | Serves | Why, on this run |
|---|---|---|---|
| **Quality flagship** | **unset — honest negative** | `eval-ladder` | Every arm loses to the static-copy floor at 2.5σ–10.6σ. The ladder figure cannot yet show "the best PointStream can do" because nothing does better than not running a model. |
| **Comparison backbone** | ControlNet family (pose / seg / ip-adapter / trajectory) | `eval-object` | Unchanged and still the right choice: the only family where the backbone is fixed while the conditioning changes. Seg 0.5595, pose 0.6031, trajectory 0.6229, ip-adapter 0.7606 LPIPS. Seg vs pose is 1.4σ — report the ordering as unresolved, not as a result. |
| Temporal / FVMD | Animate-Anyone, **clip mode only** | `eval-metrics` | 0.5692 LPIPS, 12.21 dB, 6.2 GiB, ~1 s/frame over an 8-frame clip. The single-frame path is now refused by the harness. In-domain only (option 2, §2.8). |
| Speed rung | pix2pix | `eval-operating` | 0.03 s/frame at 0.3 GiB, and that is the whole case. Its quality is 7th of eight on LPIPS and *worse than showing an unrelated player* — report it as the speed corner, never as a quality point. |
| Floor | upscale-refine | all | 0.5585 LPIPS, the best of the eight, and it is not a generative model. That is the finding, not a footnote. |
| Null control | unrelated-image | all | 0.7358 LPIPS. New permanent arm; three engines score worse than it. |

**Do the two flagship roles collapse?** The question is moot while the quality
slot is empty. `eval-object` still needs the four conditionings, `eval-metrics`
still needs a temporal model, `eval-operating` still needs pix2pix.

**The trajectory arm still does not need MOFA-Video.** Trajectory-controlnet
0.6229 sits beside pose 0.6031 on the same epoch-10 OpenPose checkpoint at 1.4σ
— the control image is the only thing that changed, and the difference is not
resolved at this n. That is the `eval-object` experiment, and it currently has a
null result.

#### 6.2.1 Which existing engines survive, and why

Each keeps its place for a structural reason, **now backed by the clip-mode
run**. Every LPIPS figure is object-scoped on the player bbox, offsets 1–8,
n=12 clips, against a static-copy floor of 0.4505 and an unrelated-image null of
0.7358 measured in the same session.

| Engine | Verdict | On this evidence |
|---|---|---|
| **ControlNet family** | **Keep — comparison backbone. Not a flagship.** | Seg 0.5595 / pose 0.6031 / trajectory 0.6229 / ip-adapter 0.7606, epochs 7/10/10, ~3.3 GiB, ~0.95 s/frame. All lose to the floor. ip-adapter is worse than the null and remains the mislabelled seg checkpoint of §2.3. |
| **Animate-Anyone** | **Keep as the temporal incumbent. Clip mode is mandatory.** | 0.5692 LPIPS in clip mode, indistinguishable from seg (0.4σ). Depends least on copying the reference of any arm that reaches the top group (§2.10) — suggestive of feature injection, not established. In-domain only. |
| **upscale-refine** | **Keep, and promote it in the writing** | Best LPIPS on the roster (0.5585) while being non-generative. Generation currently buys **nothing** over it; the earlier "~1.7 dB" was self-reconstruction PSNR. |
| **pix2pix** | **Keep as the speed corner only** | 0.03 s/frame, 0.3 GiB. LPIPS 0.7820 — worse than an unrelated image, and 2nd on PSNR, which is the roster's clearest example of the two metrics disagreeing. |
| **SPADE4Tennis** | **Keep as a domain-specialisation control** | 0.8342 LPIPS, last of eight and worst of all. Domain-specific SPADE did not beat the fine-tuned general backbone. That *is* the comparison. |
| **MOFA-Video** | Stays dropped | Construction refuses (SVD licence). Trajectory-controlnet replaces it. |
| **StableAnimator** | Wrapped, not shipped | Constructs; generate refuses (SVD-XT not bundled). Not ranked. |

MTVCrafter is still a candidate *motion representation* (4D/SMPL tokens), not a
drop-in generator. Sparse2Dense still has no public code or weights.

**What this section cannot tell you, and BP12 assumed it could.** Whether any of
these engines has a working appearance pathway. The cross-appearance test was
supposed to answer it and cannot — a pasted keyframe tops that scale with no
network at all (§2.10). Answering it needs an identity metric, which is §7 work.


#### 6.3 What the 2026 literature says we should consider

Surveyed 2026-08-22. Two findings matter and both are actionable.

**Our engines are old.** Animate-Anyone is repeatedly outperformed in recent
comparisons, showing face and body distortion where newer work does not.
Candidates worth probing for the quality-flagship slot, in order of how cheaply
we could adopt them:

| Candidate | Why | Cost |
|---|---|---|
| **StableAnimator** | Best reported identity preservation (CSIM) and FVD in its class. Adapter Apache-2.0 on HF; inference needs SVD-XT (not bundled) — **same licence class as MOFA**, so "cheapest real upgrade" was wrong. Wrapped; live run blocked on VRAM and the pinned env. | high (licence) |
| **MTVCrafter** | SOTA on TikTok, +65% FID-VID over second best. Tokenises raw 4D motion rather than 2D pose images — *directly relevant to our motion-representation axis*. | medium |
| DisPose / Animate-X / StableAnimator++ | Incremental over the above | defer |

**The other new work is not a competitor system — it is a corner of our lattice.**
This was initially mis-read here, and the correction matters because it is the
stronger position. These systems all share one construction: code a reference
frame conventionally, send a compact per-frame motion signal, and synthesise the
rest. In this project's vocabulary that is an *appearance representation* plus a
*motion representation* plus a generator — with detection, selection, tracking,
rigid objects, background, residual and codec fallback all switched off.

| Work | As a lattice corner | What it does not have |
|---|---|---|
| **Sparse2Dense** (DCC 2026) | appearance = VVC-coded key-reference frame; motion = sparse **3D** keypoints; generator = keypoint-aware multi-task net | one subject; no background model, no non-person objects, no residual, no fallback |
| **T-GVC** | appearance = coded keyframe; motion = semantically weighted sparse trajectories; generator = training-free steered diffusion | whole-frame rather than per-object; one motion representation; no corrective channel |
| **ReGenVC** | appearance = neurally coded first frame; motion = per-frame pose keypoints | talking heads; single subject |

So the positioning is not "we compete with these". It is: **each of these
corresponds to a single configuration of the lattice this paper defines, and
PointStream is the framework in which such a configuration is one cell among
many** — with the components they lack, and with the representation comparison
none of them runs. `GVC-RT` is the exception and sits outside this framing; it
bears on `eval-operating`, where our measured speed is poor, and should be cited
plainly.

Say *corresponds to*, never *is a special case of ours*. These are independent
systems whose designs happen to land on corners we also define; claiming they are
instances of our framework would be both wrong and rude.

**Can we plug them in?** Architecturally yes — Sparse2Dense satisfies our
generator contract exactly. Practically, **no public code or weights were found**
for Sparse2Dense (verify once more before concluding), and T-GVC is an unreviewed
preprint. What *is* adoptable is the idea: **3D keypoints as a motion
representation**, against our current 2D COCO-17 stored as WholeBody-133. That is
a candidate arm for `eval-object` and costs a keypoint schema, not a new model.

**Bound before believing:** Sparse2Dense's 74.5% BD-rate against VVC is the
state of the art on an easier problem. A PointStream BD-rate substantially
better than that on harder content is an **alarm**, not a triumph — check the
anchor, the overlap range and the region scoping before believing it.

#### 6.4 Measure the part we generate, not the whole frame

**A frame-level score hides a broken object.** A reconstruction whose background
is perfect and whose player is mangled still posts a respectable frame PSNR,
because the player occupies a small fraction of the pixels. Every evaluation must
therefore be **scoped to the region the component under test produced**:

- object generation → scored on the **object crop / mask**, not the frame;
- background modelling → scored on the **background region**, with objects excluded;
- the whole frame → reported *as well*, never *instead*.

This is an architectural requirement, not an evaluation preference, and it is why
`BP2-region-metrics.md` exists. Today `src/components/metrics/` has no concept of
a region at all.

#### 6.5 Metric discipline: PSNR to develop, the rest to publish

**PSNR is the internal check.** It is fast, always comparable, and enough to
answer "did this engine run and produce something plausible". VMAF, SSIM, LPIPS
and FVMD are slow, and computing them during triage buys nothing.

- **Triage and development:** PSNR only, region-scoped.
- **Paper results:** the full set, once the roster is fixed and the runs are the
  ones we intend to cite.

A triage number is never citable and never gets a `CLAIM` line.

#### 6.6 Validate small before scaling

- **B′.1 — fix the probe set.** ✅ Rebuilt as v2 on `phase-bp/bp1`, pose
  alignment fixed on `phase-bp/integrate` (§2.3). The verifier fails on the
  v1 snapshot and on the unaligned v2 snapshot, and passes on the rematerialised
  tree. All 12 clips have 48 colour frames and 48 skeleton frames.
- **B′.2 — wire the loaders and probe.** Loaders wired on `phase-bp/bp3` /
  `phase-bp/bp4` (§2.1). The cross-engine probe is Wave 2 (`BP5`).
- **B′.3 — fix the roster in writing**, with the reason each engine holds its
  slot. **Only then** prepare the full dataset.

Improving numbers is a separate decision taken after B′.3, in this order of
cost: parameter tuning, then fine-tuning on our data, then swapping a model.
Tuning an engine that is about to be dropped is the most expensive way to waste
September.

**Phase C — pipeline and runner.**
- *C1 reconstruction and residual*: decompose `SynthesisEngine` (512 lines doing
  panorama resolution, pose unrolling, ball rendering, generative dispatch, CUDA
  determinism and OOM fallback) and `ResidualCalculator` (967 lines). Implement
  the residual coarseness spectrum including absent. Deliver bit-identity tests
  for deterministic stages.
- *C2 encoder pipeline*: DAG built from the enabled stage set, with the skips
  that make a reduced corner genuinely cheaper rather than nominally.
- *C3 runner*: one run path — single-chunk becomes the degenerate case of
  full-match. One accounting implementation instead of two. Quality evaluation
  mandatory on every path.

**Phase D — experiments layer.** Rebuilt to consume `runner/` as a library.
Lattice-driven matrices; the codec sweep actually *invoking* PointStream rather
than reading a pre-existing summary.

---

## 7. Delivery order

Nothing below P0 blocks a submission. **The order is set by the paper, not by
the code** — an item is here because a `GOAL` marker in the Evaluation section
depends on it, and it is prioritised by how much the paper loses without it.
When a result lands and changes what the narrative can claim, this order is
re-read rather than followed blindly.

**P0 — without these there is no paper**
1. ✅ **DONE 2026-08-26** — quality measurement working at all: a tier config
   producing real numbers. All three tiers plus two controls ran end to end on a
   real 4K clip and returned PSNR, SSIM, VMAF and LPIPS (§2.16, BP23).
2. ✅ **DONE 2026-08-28** — PointStream against the codec ladder, as paired
   curves, one codec on both arms at one preset. **PointStream loses on every
   codec**: BD-rate +116.8% (av1, preset 10), +166.8% (hevc, ultrafast),
   +165.9% (avc, veryfast), +378.1% (vvc, faster), on the most static of the
   eight cached clips. On the most dynamic clip there is no BD-rate: PointStream
   saturates at 31.0 dB while av1's cheapest rung is 38.0 dB, so the curves do
   not overlap. §2.20, `plans/BP24-ladder-report.md`. *Region arms are not in
   this ladder and remain open.*
3. ✅ **DONE 2026-08-28** — the residual-coarseness curve, and it is the good
   news. A residual costing 0.9% of the payload buys 5.4 dB on static content,
   and up to 14.8 dB over the unaided reconstruction on dynamic content. The
   rate problem is the plate, which is 88-91% of the payload at every rung.
   §2.20.
4. The core ablation lattice. *`BP26` (2026-08-26): detector, pose, segmenter,
   appearance, motion and temporal names now change a run. The lattice itself
   is still un-run (Phase D). Codec / fallback / `residual.codec` remain unwired
   (`BP24`). Note that the pose axis moved keypoints without moving PSNR, so a
   lattice quoting only PSNR will show a row of zeros for pose — see
   `plans/wave5-report.md`.*
5. ✅ **DONE 2026-08-26** — a working generative engine, or an honest scoped
   negative. No engine beats a pasted keyframe. IP-Adapter *uses* appearance
   (epoch 1 vs shuffled, 3.8σ on clip means) and beats the untrained adapter
   (5.5σ) but still loses to static copy (4.1σ). `reid` +10% of the
   same/different-person span — semantic match, not identity. §2.17, BP25.
   Roster and direction: `plans/ENGINE-ROSTER.md`.
6. Generalization on the general/DAVIS profile.
7. Evaluation and Conclusion sections; abstract reconciled with what was measured.

8. **The plate.** It is 88-91% of PointStream's payload at every rung (§2.20),
   which makes it the only lever large enough to close a +116.8% gap. **Three**
   levers, cheapest first (§2.21):
   *(a)* **change its codec** — JPEG costs 3.6-4.1x what av1/vvc intra costs at
   matched fidelity, and `background.codec` already accepts an unmeasured
   `roi-video` (`plans/BP29-plate-rate.md` §1);
   *(b)* **stop paying for it once per scene** — coding the next plate as a
   P-frame against the previous saves 31-53% with av1
   (`plans/BP30-background-stream.md`);
   *(c)* **stitch a real panorama**, which `build_plate` implements and the
   runner does not call.
   *Promoted from P2 items 15 and 18 on 2026-08-28, before (b) was known.*

**P1 — strongly strengthens**
8. Perceptual and temporal metrics. 9. Object-representation comparison — the most
novel item. 10. Detector comparison including SAM3. 11. Temporal-policy ablation.
12. Encode-time comparison against VVC.

**P2 — only if time remains**
13. Appearance-representation comparison. 14. Keypoint-schema richness.
15. ⬆ **promoted to P0 item 8** — JPEG quality versus downscaling, now part of
the plate work. 16. Open-vocabulary versus hand-written selection.
17. Animate-Anyone full retrain. 18. ⬆ **promoted to P0 item 8** —
background-layer ladder.
19. Football as a third domain.

**Out of scope, named as future work in the paper:** MOS user study;
variable-keypoint training-regime robustness; representations with no decoder.

---

## 8. Verification

Per stream before merge: `ruff` and `mypy` clean; the required-behaviour suite
passes; tests cover plausible misuse, not line coverage.

**Before any comparison is reported**, use
`src.components.metrics.comparison.compare_paired`. It pairs arms on the same
items, reports n and the standard error, and **refuses to name a winner the
sample cannot support**. A +0.98 dB effect over 12 clips with a per-clip sd of
2.0 dB is ~1.7σ, and was reported here as a finding because nothing in the path
computed that. Under ~1σ is inside noise; 1–2σ is suggestive; fewer than 8 items
is underpowered.

**The required-behaviour suite** replaces a percentage gate, because a percentage
gate is satisfiable by padding and this one is not. It asserts: **every metric is
calibrated against known anchors** — ordering from identical through mild to
severe, a mild perturbation ranked above an unrelated image, and an absolute
scale inside the published range; bit-identity for deterministic stages; every lattice corner produces a runnable pipeline; config
rejects unknown keys; codec constraint violations raise; an undecodable
appearance/motion pair is rejected; no layer imports outward; every registered
backend constructs; every domain profile round-trips; every weight a shipped
config names resolves; every run emits at least one quality metric.

The ~436 pre-rewrite tests are untouched and test modules Phase B and C delete.
They die with their modules; no separate culling is needed.

**The suite does not exist yet.** `tests/invariants/` is a three-test stub that
skips for want of a run summary. Most of the assertions above need Phase C, but
two were checkable at the end of B and were not written: *every registered
backend constructs* and *every weight a shipped config names resolves*. B′ owns
writing both, since B′ is where weights start mattering.

Phase gates:

- **After B — ✅ passed 2026-08-22.** `libsvtav1` + `yuv444p` raises
  `CodecConstraintError` rather than silently emitting yuv420p; every codec rung
  has a region arm or a recorded reason it cannot; `python -m src.components`
  lists every registered backend on every axis. Verified by driving each, not by
  reading the code.
- **After B′:** the probe-set verifier fails on the old set and passes on the
  rebuilt one; a perfect-background/destroyed-object reconstruction posts a good
  frame PSNR and a bad object-scoped PSNR; every engine on the §6.2 roster loads
  real weights and returns a frame that is not its input; the roster is fixed in
  writing with the reason each engine holds its slot.
- **After C — the decisive gate:** `config/tier_fast.yaml` and
  `config/tier_quality.yaml` each produce a run summary carrying **real PSNR,
  SSIM, VMAF and LPIPS numbers**; a residual-absent run completes and reports its
  measured quality drop; the all-off corner reduces to the source video. None of
  this works today in any configuration.
- **After D:** a codec sweep invokes PointStream itself and emits one
  rate-quality-time table containing every arm.
- **During §7:** every cited run has empty `invariant_failures`.
