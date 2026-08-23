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
| **B′ — the engine roster** | BP12 ✅ | Re-ranked in clip mode on calibrated LPIPS (§2.10). Quality flagship stays **unset**: every engine loses to a pasted keyframe at 2.5σ–10.6σ, and the top three are not separable. The cross-appearance test is withdrawn as a test of appearance use — a paste tops it. |
| C — pipeline and runner | ⬜ | `C1`/`C2`/`C3` landed unmerged |
| D — experiments layer | ⬜ | Blocked on C |
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
| latent / image embedding | `ip-adapter-controlnet` (declares `appearance:image-embedding`) | **declared, never trained** — the checkpoint is the mislabelled segmentation ControlNet of §2.3 |

Of three appearance pathways, one is switched off, one failed for a reason we
understand, and one was never built. That is a better description of where this
project stands than "the generators do not use appearance".


---

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
1. Quality measurement working at all — a tier config producing real numbers.
   *Half done: the metrics compute (§2), the tier config cannot run yet.*
2. PointStream against the codec ladder, including region arms.
3. The residual-coarseness curve.
4. The core ablation lattice.
5. A working generative engine, or an honest scoped negative result.
   *The negative landed 2026-08-23: no engine beat static copy on the coding
   task. Items 2–4 can proceed on residual/all-off corners; a quality flagship
   RD curve cannot.*
6. Generalization on the general/DAVIS profile.
7. Evaluation and Conclusion sections; abstract reconciled with what was measured.

**P1 — strongly strengthens**
8. Perceptual and temporal metrics. 9. Object-representation comparison — the most
novel item. 10. Detector comparison including SAM3. 11. Temporal-policy ablation.
12. Encode-time comparison against VVC.

**P2 — only if time remains**
13. Appearance-representation comparison. 14. Keypoint-schema richness.
15. JPEG quality versus downscaling. 16. Open-vocabulary versus hand-written
selection. 17. Animate-Anyone full retrain. 18. Background-layer ladder.
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
