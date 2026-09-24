# Foreground campaign: evidence and continuation — 23–24 September 2026

This is the single record for the foreground stage of the [development campaign](20260923-development-campaign.md). It consolidates the initial paste probe, the corrected separate-object retest, the bounded Pix2Pix fit, and the Animate Anyone/appearance/ROI continuation. The fixed backgrounds and their exact source anchors come from the [background campaign](20260923-background-campaign.md). Raw per-row ledgers and media stay outside the repository at `/home/itec/emanuele/pointstream-data/outputs/modular/`.

**Current verdict:** no tested 48-frame PointStream bitstream has weighted PSNR at least its own VVC QP 46 source anchor at no more bytes. The closest capped Alcaraz row in the latest continuation is 21.298 dB versus 25.366 dB. A temporally coded object-color arm clears the picture threshold but costs at least 193,195 B versus 65,149 B. A faster client on a losing picture or over-cap wire is not a codec win. These are development scenes; no held-out or 192-frame claim is inferred.

## Decision rule and fixed handoffs

Each clip is a separate bitstream, `T = B + F + M + R + H`: background, per-object appearance, motion, residual, and side data, each charged once. Score overall, foreground, background, and `0.7 × FG + 0.3 × BG` weighted PSNR with `score_regions` and the source anchor's `masks_48.npz`. **Only** weighted PSNR at least the source's at no more bytes is claimable. Then report whether the sender (including offline plate build), client (decode plus that arm's render), or both are faster than the source. Shared general-tennis weights are outside the bitstream; weights adapted to the particular clip are content-specific and must be charged. Oracles and 16-frame screens cannot be claim points.

| Handoff | Chosen QP 46 background | B | Source cap | Source overall / FG / BG / weighted (dB) | Source encode / decode (s) | FG mask fraction |
|---|---|---:|---:|---|---|---:|
| Perricard 002, large | cleaned video | 86,894 | 104,482 | 31.87 / 24.93 / 32.45 / **27.19** | 10.014 / 30.556 | .029924 |
| Alcaraz 000, medium | registered panorama | 24,648 | 65,149 | 33.508 / 21.734 / 33.843 / **25.366** | 8.459 / 29.581 | .005096 |
| Federer 007, small | registered panorama | 35,763 | 112,295 | 31.26 / 21.686 / 31.363 / **24.589** | 10.170 / 29.730 | .002888 |

Use the saved 48-frame cleaned/plate/homography caches in `background-arms/cache/` and RGB frames plus `masks_48.npz` in `bp46-long-scenes/clips/`. Do not rebuild plates, re-encode backgrounds beyond the three approved exact-byte QP 46 decodes, or run 192 frames. Reject an empty VVC output or changed background byte count before compositing. The native paths/versions used in the records below are `/opt/local/bin/ffmpeg` `n7.1.1-56-gc2184b65d2`, `/opt/local/bin/vvencapp` 1.11.0 when FFmpeg's VVC wrapper emitted an empty file, and `/opt/local/bin/SvtAv1EncApp` SVT-AV1 v1.8.0. The exact per-row encoder, decoder, color path, clocks, and JSON are in the external ledgers cited below. RGB/BGR conversion must reverse the **last** axis; AV1 intra crops need at least 64 pixels and dimensions divisible by 8.

## What the completed tests establish

| Test | Rate and weighted result | Main interpretation |
|---|---|---|
| Initial union-crop paste | Alcaraz best 58,930 B / 20.928 dB; Federer best 79,375 B / 19.229 dB | Diagnostic only: Alcaraz's two players shared one crop and motion; Federer's later second player had no crop. These cannot stand for a separate-object codec. |
| Corrected two-object bbox, affine, articulated | Alcaraz best 54,125 B / 21.155 dB; Federer best 94,746 B / 19.934 dB | All under their caps but miss their weighted anchors by 4.211 and 4.655 dB. Perricard's minimum two-crop residual-off wire is 110,242 B, already 5,760 B over cap. |
| Pix2Pix exact-video fit | 21.45 dB raw 256-pixel object PSNR with correct pose; 10.01 dB shuffled | A 94-target training-set capacity/conditioning diagnostic. It has no full-frame codec score and its roughly 208 MiB clip-specific generator cannot be free. |
| Existing tennis Animate Anyone, 16-frame two-object screen | Charged wire 34,956 B; FG 12.264 dB vs 16.301 dB articulated paste on the same frames; target-alpha oracle FG 10.193 dB | The current checkpoint uses pose/reference but generated player pixels are weak. Its own background contaminates the composite too, but a perfect output matte did **not** restore foreground quality. No 48-frame AA claim. |
| Alcaraz independent AV1 intra refreshes | Best capped 48-frame continuation: FG-only QP54 residual, 61,301 B / 21.298 dB; every-6 refresh 78,706 B / 22.167 dB | Independent refreshes are a measured control, **not** the intended predictive same-player transport. Every-6 is over cap and still below anchor quality. |
| Alcaraz temporal object color + alpha videos | 193,195–258,631 B / 29.126–31.029 dB | Current object pixels can exceed picture quality, but this implementation spends at least 128,046 B too much. The smallest alpha videos alone cost 83,417 B. |

The per-object bbox alpha IoUs were .353/.304 on Alcaraz and .313/.445 on Federer, with target recall .519/.423 and .401/.574. Thus the coded reference matte leaves many true player pixels uncovered. The first union probe had Alcaraz alpha IoU .142. These comparisons use source masks for *diagnosis* only. The decoder has no source mask unless it is encoded and charged.

For Federer, the separate-object best row has FG 17.143 and BG 26.447 dB against source FG 21.686 and BG 31.363. The foreground contributes `0.7 × 4.543 = 3.180 dB` and background `0.3 × 4.916 = 1.475 dB` to the 4.655 dB weighted gap. The earlier 26.38 dB BG figure is below the **31.36 dB BG anchor**; 24.59 dB is the source's weighted score, a different metric. Foreground causes about 68% of the corrected gap (72% in the initial union row).

The first paste probe logged foreground `residual_clip_fraction` .1088/.1177 for Alcaraz bbox/pose and .1273/.1248 for Federer. The separate-object values are .1254/.1292/.0898 for Alcaraz bbox/affine/articulated; .0902/.0662/.0716 for Federer; .1451/.1297/.0891 for Perricard. All exceed .05, so a finer residual quantizer is ruled out until the base picture improves. Foreground QP54/62 was tried before background QP54/62; keep residual-off, foreground-only, background-only, and both rows. Residual clocks in the first runs omit some construction/addition operations and are lower bounds. None of those slower residual rows is close enough in quality for that timing limit to affect the claim verdict.

## Next experiment: predictive appearance and explicit output masks

The next work is an **unmeasured plan**, not an extension of the tables below. Keep object identities separate: one initial reference per player, including Federer's entrant at frame 33. A later refresh represents that *same player's* evolution. Encode it predictively against the decoded initial reference and against the previous decoded refresh (P-frame style), with causal frame indices and all bytes charged. The prior 24/12/6-frame ladder coded independent AV1 intra crops; it does not answer this test. Compare sparse predictive refreshes at the same schedules and quality settings, plus residual-off single-reference controls. Motion-aligned reference prediction and delta coding are candidate predictors; measure the actual native wire, not a theoretical subtraction. An AV1 inter stream that emits intervening frames must charge every emitted frame. Keep each player's stream independent, including presence, crop geometry, and matte metadata.

Test output compositing and generation conditioning as two separate decisions. First, derive a plausible player-and-racket silhouette at the **decoder** from the transmitted joints and reference appearance; measure overlap, FG/BG/weighted PSNR and compute time. A wider learned human shape may cover limbs, clothing and racket better than warped first-frame alpha, but it must be deterministic from charged inputs. Then transmit a dense mask stream and use the **decoded** mask both to condition an eligible generator and to composite its player pixels, preventing generated background from replacing the court. Source-mask compositing remains an oracle, never a charged result. The AA target-alpha oracle (FG 10.193 dB) shows that output masking alone may fix background contamination yet cannot rescue poor generated player pixels.

Use per-object labels where identity matters. A 2-bit label has four values: background and up to three foreground labels (for example two players and a ball or racket). If ball and racket need separate identities in addition to both players, use more labels; do not silently merge objects. Evaluate ROI-local packed 1-bit per-object masks and 2-bit palette labels, polygon/contour quantization, temporal XOR or motion-predicted deltas with entropy coding, and a downsampled lossy mask reconstructed and sharpened at the client. Record actual side-data bytes `H`, object identity continuity, mask IoU/recall, contour/racket failures, and rate–distortion. The prior independent lossless ROI zlib silhouettes cost **62,977 B**; the two alpha AV1 videos cost at least **83,417 B**, both too much for Alcaraz. These are baselines for a better temporal mask representation, not a proof it will fit. [Ultralytics instance segmentation](https://docs.ultralytics.com/tasks/segment) exposes `masks.data` as binary `uint8` `(N,H,W)` tensors and `masks.xy` as polygon coordinates. Neither is inherently a compressed PointStream bitstream; audit actual local YOLO outputs and encode them explicitly. A COCO pose detector does not itself supply instance masks.

Prioritize a short **matched-wire** screen on the existing SPADE4Tennis checkpoint: same two Alcaraz objects, decoded appearance/motion, and either pose-derived or actually coded dense masks, with an output matte in both cases. Check its accepted conditioning interface before scoring; feeding a new representation into a model trained for another input is not a fair capacity verdict. Keep articulated paste and the existing AA screen as controls. If a dense-conditioned fast model improves FG substantially and reduces clipping, allocate bounded multi-video training on development matches, with frozen wire format and scene separation. Exact-video overfit may diagnose capacity but its adapted weights are rate-bearing and cannot be used as a general-tennis codec point. Recheck source anchors and complete 48-frame `T`, quality and timing before any paper number.

The [First Order Motion Model](https://arxiv.org/abs/2003.00196) motivates a later compact learned flow/occlusion decoder from a reference and charged motion. It could offer dense motion without diffusion latency; it is a design candidate, not a tested PointStream model. The installed tennis Animate Anyone checkpoint was a negative screen. Check whether a **checkpoint from the original authors** is actually obtainable and compatible before scheduling a separate short screen. The [authors' repository](https://github.com/HumanAIGC/AnimateAnyone) currently exposes documentation rather than an obvious released runnable checkpoint; do not label the locally used Moore-style reproduction as authors' weights or spend training time on an unavailable file. The [ControlNet OpenPose](https://huggingface.co/lllyasviel/sd-controlnet-openpose) and [normal-map](https://huggingface.co/lllyasviel/sd-controlnet-normal) cards use different conditioning signals and training sets. The visual impression that normal maps look better is a useful hypothesis, not a matched tennis comparison. Normals are a separate dense input whose inference/transmission cost must be counted; a binary segmentation model or retrained tennis model is the direct mask comparison. Test ControlNet only if it accepts the identical decoded object wire and a short screen justifies its client cost.

For every future candidate, keep `B + F + M + R + H` component bytes, source-matched overall/FG/BG/weighted PSNR, foreground `residual_clip_fraction`, exact encoder/decoder paths and versions, split encode/decode/render seconds and offline plate-build time. Coarsen FG residual before BG to fit; if residual-off two-object appearance still exceeds the cap, report that bound and stop the clip. Log every tested axis and failed point. Generation-on plus supplied crop and residual-off is not correctly represented by the current full runner; this remains a probe until a winning configuration is wired and retested there. Do not reopen backgrounds, motion packing, WebP plates, held-out confirmation, or 192 frames in this handoff.

## Detailed measured records

The following dated sections retain the actual protocol, all scored axes and provenance. The first union-crop tables are explicitly diagnostic; corrected separate-object tables govern the current codec boundary. Earlier proposed steps in the original part 2 plan have been replaced by their measured results and by the predictive/mask plan above.

## Initial union-crop probe, diagnostic only

### Method and wire

Run from main `b2a69c9177a01a7312aec66c727c56e3731734b8` with
`/home/itec/emanuele/.conda/envs/pointstream/bin/python`:

```sh
python experiments/modular/foreground_campaign.py perricard002
python experiments/modular/foreground_campaign.py alcaraz000
python experiments/modular/foreground_campaign.py federer007
```

Inputs were the three `*-n48.npz` cleaned/plate/homography caches under
`/home/itec/emanuele/pointstream-data/outputs/modular/background-arms/cache/`,
the `window_48` RGB frames, and their own `masks_48.npz` under
`/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/`.
The masks are the same as the source anchor's, and are used for scoring and
for forming encoder-side residual signals. The decoder does not receive a
target mask. It receives a losslessly packed, zlib-9 compressed alpha mask
for the first crop, and warps that alpha with the motion on each frame.

Each row charges `T = B + F + M + R + H` once. `B` includes the VVC background
payload and its existing 10 B video or 1,742 B panorama side data. `F` is
one AV1 intra QP 42 appearance payload. `M` is all 48 bbox placements at
8 B/frame; COCO-17 also sends all 48 poses at 102 B/frame, including the
source pose needed for the affine transform. `H` is the alpha-mask payload,
including its 8 B shape header. `R` is zero, one foreground or background
VVC residual payload, or the sum of two separately decoded payloads. There
are no container bytes in this probe; it is not a full-runner launch. The
keypoint arm also charges bboxes because it uses them when pose fitting falls
back. Poses were extracted with `yolo26n-pose.pt` on CPU, with no missing
frames in either completed clip. The residual is clipped to `[-128,127]`,
offset by 128, neutral outside its encoder-side region, then decoded and
added over the full frame; any chroma bleed counts in the PSNR.

The three allowed QP 46 background decodes were regenerated only from those
caches, with the same RGB → BT.601 4:2:0 → Y4M path as
`background_campaign.py`. The two panorama byte counts matched before any
composite. Crop input was BGR for the intra sidecar; decoded video was RGB
after the Y4M path. Channel reversal was on the last axis. The source anchors
and offline plate-build times are the saved background-campaign rows.

Native tools: VVC `/opt/local/bin/ffmpeg` version
`n7.1.1-56-gc2184b65d2`, preset `faster`, with the measured nonempty
`/opt/local/bin/vvencapp` 1.11.0 fallback; AV1 intra
`/opt/local/bin/SvtAv1EncApp` `SVT-AV1 v1.8.0`; decoding through
`/opt/local/bin/ffmpeg` n7.1.1. FFmpeg sometimes exited 0 with an empty VVC
file. Those attempts were rejected; their retry cost remains in the encode
clock. AV1 crops were padded to at least 64 pixels and multiples of 8.

The raw per-row JSON, including exact binary versions and clocks, is under
`/home/itec/emanuele/pointstream-data/outputs/modular/foreground-campaign/`.
The two completed files are `alcaraz000.json` and `federer007.json`; the
large-clip stop is `perricard002-stopped.json`.

### Large — Perricard scene 002

Mask fraction 0.029924. Source QP 46: **104,482 B**, overall 31.87 dB,
FG 24.93 dB, BG 32.45 dB, weighted **27.19 dB**; encode 10.014 s, decode
30.556 s. The chosen cleaned-video background was 86,894 B in the prior row
(86,884 B payload + 10 B side data), with 17,588 B of foreground headroom.
Its saved row used `/opt/local/bin/vvencapp` 1.11.0 after the FFmpeg wrapper
failed, and records 10.068 s encode, 29.714 s decode, and 122.423 s offline
plate build.

The regenerated QP 46 background was **87,186 B**, 292 B above the saved
86,894 B. FFmpeg's first attempt was empty; a retry emitted the changed
payload. Per the byte-stability gate, this clip **stopped before compositing**.
There is no appearance, motion, residual, clip fraction, or PointStream
quality row for Perricard. The previous background quality is not a foreground
result. The exact selected background was not reproduced, so no claim is made
for the large band.

### Medium — Alcaraz scene 000

Mask fraction **0.005096**. Source QP 46: **65,149 B**, overall **33.51**,
FG **21.73**, BG **33.84**, weighted **25.37 dB**; encode **8.459 s**,
decode **29.581 s**. Registered-panorama background reproduced at **24,648 B**
(22,906 B payload + 1,742 B side data), leaving 40,501 B before the
foreground. Its regenerated encode/decode/render times were
2.812/2.486/0.882 s; the offline plate build was 104.062 s.

| Arm | B | F | M | H | Residual-off total | `residual_clip_fraction` |
|---|---:|---:|---:|---:|---:|---:|
| Bbox | 24,648 | 19,301 | 384 | 2,050 | 46,383 | 0.1088 |
| COCO-17 | 24,648 | 19,301 | 5,280 | 2,050 | 51,279 | 0.1177 |

Both clip fractions exceed 0.05, so the tested residual QPs were 54 then
62, without buying QP 46. FG was coarsened before BG. The COCO-17 FG 54 +
BG 54 row is 232 B over the anchor; coarsening FG to 62 brings it to
64,299 B. All other rows below are under the source byte count. `E` is the
sum of the measured component encode clocks and motion preparation; `D` is
the sum of component decode clocks, before rendering. All PSNR columns are dB.

| Motion | Residual | R B | Total B | Overall | FG | BG | Weighted | E s | D s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bbox | off | 0 | 46,383 | 30.01 | 12.28 | 31.67 | 18.10 | 7.34 | 2.94 |
| bbox | FG 54 | 7,243 | 53,626 | 30.92 | 16.15 | 31.66 | 20.80 | 15.62 | 32.06 |
| bbox | FG 62 | 6,187 | 52,570 | 30.58 | 14.92 | 31.48 | 19.89 | 18.30 | 33.03 |
| bbox | BG 54 | 5,304 | 51,687 | 30.44 | 12.36 | 32.23 | 18.32 | 15.41 | 32.54 |
| bbox | BG 62 | 4,370 | 50,753 | 30.22 | 12.30 | 31.94 | 18.19 | 22.60 | 32.49 |
| bbox | FG 54 + BG 54 | 12,547 | 58,930 | 31.19 | 16.21 | 31.94 | **20.93** | 23.69 | 61.66 |
| bbox | FG 62 + BG 54 | 11,491 | 57,874 | 30.73 | 14.98 | 31.64 | 19.98 | 26.37 | 62.63 |
| bbox | FG 54 + BG 62 | 11,613 | 57,996 | 31.01 | 16.12 | 31.75 | 20.81 | 30.88 | 61.61 |
| bbox | FG 62 + BG 62 | 10,557 | 56,940 | 30.60 | 14.92 | 31.50 | 19.89 | 33.56 | 62.58 |
| COCO-17 | off | 0 | 51,279 | 29.31 | 12.40 | 30.73 | 17.90 | 27.66 | 2.94 |
| COCO-17 | FG 54 | 7,391 | 58,670 | 30.10 | 16.27 | 30.69 | 20.59 | 38.56 | 32.03 |
| COCO-17 | FG 62 | 6,309 | 57,588 | 29.89 | 15.24 | 30.60 | 19.85 | 35.78 | 32.71 |
| COCO-17 | BG 54 | 6,711 | 57,990 | 30.16 | 12.44 | 31.83 | 18.26 | 36.05 | 32.51 |
| COCO-17 | BG 62 | 5,344 | 56,623 | 29.82 | 12.40 | 31.37 | 18.09 | 36.41 | 32.68 |
| COCO-17 | FG 54 + BG 54 | 14,102 | 65,381 | 30.86 | 16.28 | 31.53 | 20.85 | 46.96 | 61.61 |
| COCO-17 | FG 62 + BG 54 | 13,020 | 64,299 | 30.72 | 15.26 | 31.56 | 20.15 | 44.18 | 62.28 |
| COCO-17 | FG 54 + BG 62 | 12,735 | 64,014 | 30.48 | 16.24 | 31.11 | 20.70 | 47.32 | 61.77 |
| COCO-17 | FG 62 + BG 62 | 11,653 | 62,932 | 30.35 | 15.23 | 31.12 | 20.00 | 44.54 | 62.45 |

The highest weighted score is bbox FG 54 + BG 54: **20.928 dB at 58,930 B**,
4.438 dB below the source despite being 6,219 B smaller. No claimable point.
The foreground peak in the table is 16.28 dB, far below the roughly 22.4 dB
needed on this court. For the best row, plate-inclusive sender is at least
127.752 s versus 8.459 s source encode; client decode plus measured render
is at least 67.059 s versus 29.581 s source decode. The residual-off bbox
client is 8.345 s, faster than the source, with much worse weighted quality.

### Small — Federer scene 007

Mask fraction **0.002888**. Source QP 46: **112,295 B**, overall **31.26**,
FG **21.69**, BG **31.36**, weighted **24.59 dB**; encode **10.170 s**,
decode **29.730 s**. Registered-panorama background reproduced at **35,763 B**
(34,021 B payload + 1,742 B side data), leaving 76,532 B before the
foreground. Its regenerated encode/decode/render times were
2.922/2.729/0.893 s; offline plate build was 118.982 s.

| Arm | B | F | M | H | Residual-off total | `residual_clip_fraction` |
|---|---:|---:|---:|---:|---:|---:|
| Bbox | 35,763 | 1,962 | 384 | 356 | 38,465 | 0.1273 |
| COCO-17 | 35,763 | 1,962 | 5,280 | 356 | 43,361 | 0.1248 |

Both clip fractions exceed 0.05. FG QPs 54 and 62 were tested before BG
QPs 54 and 62. Every row fits the source byte count.

| Motion | Residual | R B | Total B | Overall | FG | BG | Weighted | E s | D s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bbox | off | 0 | 38,465 | 22.90 | 13.40 | 22.99 | 16.28 | 4.18 | 2.82 |
| bbox | FG 54 | 6,673 | 45,138 | 22.94 | 16.17 | 22.98 | 18.21 | 11.91 | 32.34 |
| bbox | FG 62 | 5,817 | 44,282 | 22.91 | 14.95 | 22.97 | 17.36 | 11.86 | 32.57 |
| bbox | BG 54 | 30,347 | 68,812 | 26.25 | 13.41 | 26.49 | 17.34 | 13.38 | 32.40 |
| bbox | BG 62 | 15,295 | 53,760 | 25.06 | 13.27 | 25.23 | 16.86 | 13.42 | 32.35 |
| bbox | FG 54 + BG 54 | 37,020 | 75,485 | 26.23 | 16.15 | 26.32 | 19.20 | 21.11 | 61.92 |
| bbox | FG 62 + BG 54 | 36,164 | 74,629 | 26.12 | 14.95 | 26.24 | 18.33 | 21.06 | 62.15 |
| bbox | FG 54 + BG 62 | 21,968 | 60,433 | 24.96 | 16.00 | 25.03 | 18.71 | 21.15 | 61.88 |
| bbox | FG 62 + BG 62 | 21,112 | 59,577 | 24.89 | 14.81 | 24.98 | 17.86 | 21.10 | 62.11 |
| COCO-17 | off | 0 | 43,361 | 23.01 | 13.59 | 23.11 | 16.45 | 12.59 | 2.82 |
| COCO-17 | FG 54 | 6,606 | 49,967 | 23.05 | 16.16 | 23.10 | 18.24 | 23.54 | 32.34 |
| COCO-17 | FG 62 | 6,125 | 49,486 | 23.03 | 15.04 | 23.08 | 17.46 | 20.74 | 32.62 |
| COCO-17 | BG 54 | 29,408 | 72,769 | 26.29 | 13.64 | 26.53 | 17.51 | 22.41 | 32.54 |
| COCO-17 | BG 62 | 14,553 | 57,914 | 25.15 | 13.46 | 25.32 | 17.02 | 21.30 | 32.31 |
| COCO-17 | FG 54 + BG 54 | 36,014 | 79,375 | 26.29 | 16.16 | 26.38 | **19.23** | 33.35 | 62.06 |
| COCO-17 | FG 62 + BG 54 | 35,533 | 78,894 | 26.35 | 15.04 | 26.48 | 18.47 | 30.55 | 62.34 |
| COCO-17 | FG 54 + BG 62 | 21,159 | 64,520 | 25.09 | 15.93 | 25.16 | 18.70 | 32.24 | 61.83 |
| COCO-17 | FG 62 + BG 62 | 20,678 | 64,039 | 25.19 | 14.79 | 25.29 | 17.94 | 29.44 | 62.11 |

The highest weighted score is COCO-17 FG 54 + BG 54: **19.229 dB at
79,375 B**, 5.360 dB below the source despite being 32,920 B smaller. No
claimable point. FG peaks at 16.17 dB, far below the roughly 25.2 dB needed
on the panorama court. For the best row, plate-inclusive sender is at least
152.335 s versus 10.170 s source encode; client decode plus measured render
is at least 67.297 s versus 29.730 s source decode. The residual-off COCO-17
client is 8.052 s, faster than the source, with much worse weighted quality.

### Boundary and timing limits

The measured base's signed foreground error saturates on 10.88–12.73% of
masked pixels. The 0.05 rule therefore forbids a finer residual quantizer;
the next campaign must shrink the player error before residual coding. The
largest band's matched-court background would have been the easiest quality
hurdle, but its regenerated bytes moved and it has no valid composite.
These rows establish a negative boundary for one QP 42 crop, bbox or COCO-17
affine motion, and clipped QP 54/62 residuals on the two validated QP 46
backgrounds. They do not establish a result for a trained generator or a
different clip.

The tables report component encode and decode seconds separately. The
plate-inclusive sender and decode-plus-render client comparisons above are
**lower bounds** for residual rows: residual-signal construction and decoded
residual addition were not individually timed. They are already slower than
their source anchors, so this limitation cannot reverse the latency direction
of those rows. The residual-off client render includes panorama warp and
appearance paste. Latency is not used to turn a lower-quality row into a win.

### Post-run mask and warp diagnosis

A read-only check of the same `masks_48.npz` explains why one crop plus one
global affine transform starts far from the foreground target. Alcaraz frame
0 has two mask components of at least 100 pixels, with areas 34,441 and
12,253 pixels. Their joint padded crop spans 1,079,140 pixels, while only
46,694 pixels (4.33%) are foreground. One transform moves both players
together. Federer frame 0 has one main component; its padded crop spans
43,200 pixels, of which 10,645 (24.6%) are foreground.

Warping only the first-frame alpha through the transmitted bbox transform
gives mean alpha IoU with the target masks of 0.142 on Alcaraz and 0.244 on
Federer; mean target-mask recall is 0.171 and 0.350. Federer frame 47 alpha
IoU is 0.017. This diagnostic uses source masks only for comparison and does
not re-encode a background. It does not by itself assign all PSNR loss to
silhouette error, but it shows that much of the true player area is not
covered by the transmitted reference. COCO-17 changed the affine estimate,
not the one-transform representation; it did not materially close the
weighted-PSNR gap.

## Corrected separate-object retest

### Fixed inputs and decision

Use the same 48-frame RGB sources, `masks_48.npz`, cached plates and chosen
QP 46 backgrounds in the [background record](20260923-background-campaign.md).
Do not build a plate or run 192 frames. Regenerate only the three approved
decodes. Require exact background totals before a composite: Perricard
86,894 B, Alcaraz 24,648 B, Federer 35,763 B. Empty VVC output is failure;
stop that clip on a size mismatch. Record the actual encoder and decoder
paths, versions and elapsed time.

| Clip | Background | Source cap | Background bytes | Foreground target | Mask fraction |
|---|---|---:|---:|---:|---:|
| Perricard 002 | cleaned video | 104,482 | 86,894 | about 24.9 dB | 0.029924 |
| Alcaraz 000 | panorama | 65,149 | 24,648 | about 22.4 dB | 0.005096 |
| Federer 007 | panorama | 112,295 | 35,763 | about 25.2 dB | 0.002888 |

Each row is its own bitstream. Claimable means weighted PSNR,
`0.7 × foreground + 0.3 × background`, at least its own source anchor at
no more bytes. Report overall, foreground and background PSNR as well.
Only after a weighted tie or win at no more bytes compare sender time,
including the saved offline plate-build cost, and client decode plus render.

### Object and motion arms

Segment foreground into player objects; track those objects over 48 frames
with disjoint masks and a stable identity, including players first visible
after frame 0. Encode one AV1 intra QP 42 crop and one alpha per object from
its first visible frame. Charge each object's crop, alpha,
48 bbox placements and, in the pose arms, all transmitted COCO-17 joints.
The decoder must receive only the charged crop, alpha and motion. Source
masks may form encoder-side residuals and scores; they may not become an
uncharged decoder input.

Compare three reconstructions on each chosen background:

1. Each object's bbox placement.
2. One affine transform per object fitted from its transmitted joints, with
   bbox fallback when joints are unreliable.
3. Articulated or piecewise warping of each object from the same transmitted
   joints, with the same fallback. No extra motion bytes are free.

For each arm keep residual off; foreground only; background only; and both.
Log `residual_clip_fraction` on the foreground signal before training. If it
exceeds 0.05, do not buy a finer residual quantizer. Coarsen foreground
first, then background until `T = B + F + M + R + H` fits the source cap.
Keep each component's bytes once, actual encode/decode clocks, render clock,
and the score from `score_regions` on the anchor masks. If residual-off still
exceeds the source, reduce appearance to one crop per object; if that still
fails, stop and name the background choice as infeasible.

The run is an exploratory probe. A winning point must later be wired through
the full runner before a paper number is frozen; the current full runner
cannot correctly combine supplied crop, generation-on and residual-off.

The pre-encode tracking check found exactly two object tracks on each clip.
First visible frames are 0/0 for Alcaraz, 0/33 for Federer and 0/0 for
Perricard. The tracked union covers every qualifying Alcaraz and Federer
mask pixel; Perricard has 512 tiny pixels below the 100-pixel component
threshold. Each object's presence uses 6 B for 48 frames and bbox motion
uses 384 B, in addition to its crop and alpha.

A read-only silhouette check on the separate-object bbox path found mean
alpha IoU of 0.353/0.304 for Alcaraz's two tracks and 0.313/0.445 for
Federer's. Target-mask recall was 0.519/0.423 and 0.401/0.574. Separate
objects substantially improve the old Alcaraz union-crop IoU of 0.142,
but roughly half of the true player pixels still lie outside the pasted
alpha. This diagnoses a likely limit; the scored bitstream rows below
decide quality.

### Verification before native encodes

- **Behavior:** two Alcaraz players yield two separately coded appearances,
  alphas and motion streams; Federer's late entrant yields a second. All three motion
  arms use the same decoded wire values, and component totals sum exactly.
- **Plausible misuse:** reject empty or changed background payloads,
  nonfinite joints, object identity swaps and channel reversal on the width
  axis. Target masks cannot enter the decoder.
- **Deliberately untested:** no 192-frame amortization or held-out
  generalization claim is inferred from these 48-frame rows.

The exact command, paths and JSON ledger are below. Data stays under the
external `pointstream-data/outputs` tree.

### Launch record

Started 23 September 2026 at 21:17:44 UTC. The eight-hour CPU supervisor
reported `running` with PID 677681 and began Alcaraz's fixed QP 46
panorama regeneration. It runs the clips sequentially as three independent
bitstreams, Alcaraz then Federer then Perricard, each with a separate JSON
row ledger. An exact-byte failure stops only that clip and is recorded in
`batch-summary.json`. The supervisor stops the batch after eight hours and
requests one digest then. The runner uses
`/home/itec/emanuele/.conda/envs/pointstream/bin/python` and this command:

```sh
python -m experiments.jobs.monitor start \
  /home/itec/emanuele/pointstream-data/outputs/modular/object-foreground-campaign/job \
  --budget-hours 8 --thread 01a0ce73-4593-7810-97cf-4a3ca34e117c \
  --report-in-hours 8 --quiet-hours 8 --cpu-threads 8 \
  --claims-dir /home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/claims \
  --command python experiments/modular/run_object_foreground_batch.py
```

The recorded invocation expanded both `python` occurrences to the absolute
pinned interpreter, set `PYTHONNOUSERSITE=1` and the `/tmp` bytecode cache,
and supplied the absolute Codex executable for the deferred digest. The
actual argument vector, logs, status and outputs are under the job and
campaign directories above. The six focused tests and Ruff checks passed
before native launch.

The first saved row, while the batch is still running, is Alcaraz's
separate-object bbox path with both residuals off: **32,212 B** and weighted
**18.040 dB** against its **25.37 dB** source anchor; foreground
`residual_clip_fraction` is **0.1254**. Its two AV1 crops are 3,752 B and
1,829 B. This is a valid residual-off measurement, not a verdict on the
pending affine, articulated and residual rows.

The running process imported the runner before a timing-only correction was
made to its file. Its sender clock omits object-track splitting and
residual-signal construction; residual rows' client clock omits decoded
residual addition. Treat those clocks as lower bounds. Native decode seconds
and residual-off client render remain measured. If a claimable point emerges,
repeat its isolated timing before a speed sentence; do not re-encode the
background merely to tighten timing on a losing quality row.

### Completed results — 24 September

The CPU batch completed all three clips in 5,448 s. The three QP 46
backgrounds reproduced their saved byte counts, including Perricard's
86,894 B through direct `/opt/local/bin/vvencapp` 1.11.0. Panorama and
residual attempts used `/opt/local/bin/ffmpeg`
`n7.1.1-56-gc2184b65d2`, with the nonempty direct VVC fallback; AV1
intra used `/opt/local/bin/SvtAv1EncApp` `SVT-AV1 v1.8.0`. The full
encoder/decoder path, version and per-residual encode/decode clocks are in
the [external JSON ledgers](/home/itec/emanuele/pointstream-data/outputs/modular/object-foreground-campaign/batch-summary.json).
All scores use `score_regions` and each source anchor's `masks_48.npz`.
No row reaches the source weighted PSNR at no more bytes. A faster decode
or a smaller bitstream alone is not a claim.

Every completed medium/small warp tested residual off, foreground only,
background only and both. QPs 54 and 62 were encoded for each residual;
QP 54 gave the best weighted score in every class and arm. All measured
foreground clip fractions exceed 0.05, so QP 46 was excluded by the
predeclared saturation rule. The tables show the best under-cap row in each
class; the ledgers preserve all 27 measured rows per clip.

#### Medium — Alcaraz scene 000

Source: **65,149 B**, overall 33.51, FG 21.73, BG 33.84, weighted
**25.37 dB**; encode **8.459 s**, decode **29.581 s**. Foreground fraction
**0.005096**. The chosen panorama is **24,648 B**. Both object appearances
cost **5,581 B** (3,752 + 1,829); their alphas plus count cost **1,203 B**.
Bbox/presence motion is **780 B**; transmitting both COCO-17 sequences
raises motion to **10,572 B**. The foreground clip fractions are **0.1254**
bbox, **0.1292** global affine and **0.0898** articulated.

| Warp | Residual | QP | Total B | FG | BG | Weighted |
|---|---|---|---:|---:|---:|---:|
| bbox | off | — | 32,212 | 12.28 | 31.49 | 18.04 |
| bbox | FG | 54 | 39,752 | 15.98 | 31.44 | 20.62 |
| bbox | BG | 54 | 37,792 | 12.18 | 32.13 | 18.17 |
| bbox | both | 54/54 | 45,332 | 15.88 | 31.81 | 20.66 |
| global affine | off | — | 42,004 | 12.61 | 31.10 | 18.16 |
| global affine | FG | 54 | 49,322 | 16.16 | 31.12 | 20.64 |
| global affine | BG | 54 | 48,108 | 12.54 | 31.95 | 18.36 |
| global affine | both | 54/54 | 55,426 | 16.09 | 31.80 | 20.80 |
| articulated | off | — | 42,004 | 13.71 | 31.96 | 19.18 |
| articulated | FG | 54 | 49,111 | 16.51 | 31.95 | 21.14 |
| articulated | BG | 54 | 47,018 | 13.63 | 32.41 | 19.26 |
| articulated | both | 54/54 | **54,125** | **16.40** | **32.24** | **21.16** |

The best capped bitstream charges `B 24,648 + F 5,581 + M 10,572 +
R 12,121 + H 1,203 = 54,125 B`. Its overall PSNR is **31.42 dB** and
weighted PSNR is **4.21 dB below** the source. Encode is **at least
46.235 s**, decode **62.442 s**; with the 104.062 s offline plate build,
sender is **at least 150.297 s**, and decode plus render is **at least
106.275 s**, versus source 8.459/29.581 s. No claimable point.

#### Small — Federer scene 007

Source: **112,295 B**, overall 31.26, FG 21.69, BG 31.36, weighted
**24.59 dB**; encode **10.170 s**, decode **29.730 s**. Foreground fraction
**0.002888**. The chosen panorama is **35,763 B**. The first player's
crop is 1,962 B; the player entering at frame 33 adds an 8,173 B crop.
Thus `F = 10,135 B`, `H = 2,398 B`, bbox/presence `M = 780 B`, and
pose `M = 10,572 B`. Foreground clip fractions are **0.0902** bbox,
**0.0662** global affine and **0.0716** articulated.

| Warp | Residual | QP | Total B | FG | BG | Weighted |
|---|---|---|---:|---:|---:|---:|
| bbox | off | — | 49,076 | 14.68 | 23.12 | 17.21 |
| bbox | FG | 54 | 55,902 | 16.59 | 23.13 | 18.55 |
| bbox | BG | 54 | 78,625 | 14.47 | 26.54 | 18.09 |
| bbox | both | 54/54 | 85,451 | 16.42 | 26.47 | 19.44 |
| global affine | off | — | 58,868 | 15.85 | 23.11 | 18.03 |
| global affine | FG | 54 | 65,443 | 16.99 | 23.10 | 18.82 |
| global affine | BG | 54 | 88,480 | 15.65 | 26.54 | 18.92 |
| global affine | both | 54/54 | 95,055 | 16.84 | 26.41 | 19.71 |
| articulated | off | — | 58,868 | 15.95 | 23.14 | 18.11 |
| articulated | FG | 54 | 65,505 | 17.26 | 23.13 | 19.02 |
| articulated | BG | 54 | 88,109 | 15.88 | 26.56 | 19.08 |
| articulated | both | 54/54 | **94,746** | **17.14** | **26.45** | **19.93** |

The best capped bitstream charges `B 35,763 + F 10,135 + M 10,572 +
R 35,878 + H 2,398 = 94,746 B`. Its overall PSNR is **26.36 dB** and
weighted PSNR is **4.65 dB below** the source. Encode is **at least
29.855 s**, decode **61.646 s**; with the 118.982 s plate build, sender
is **at least 148.837 s**, and decode plus render is **at least 93.002 s**,
versus source 10.170/29.730 s. No claimable point.

#### Large — Perricard scene 002

Source: **104,482 B**, overall 31.87, FG 24.93, BG 32.45, weighted
**27.19 dB**; encode **10.014 s**, decode **30.556 s**. Foreground fraction
**0.029924**. The direct VVC background exactly matched **86,894 B**.
Two object appearances cost **18,925 B**, alphas/count **3,643 B**,
bbox/presence **780 B** and pose motion **10,572 B**. Even with both
residuals off, bbox totals **110,242 B** (5,760 B over source) at
weighted **17.36 dB**; global affine and articulated each total
**120,034 B** (15,552 B over), at weighted **17.54** and **18.65 dB**.
Foreground clip fractions are **0.1451**, **0.1297**, **0.0891**.
No residual was encoded on this clip. With a mandatory crop per player,
the selected matched-court background leaves too few bytes; this is an
infeasible background choice for the tested QP 42 two-crop appearance.
There is no under-cap PointStream row and no claimable point.

### Claim boundary

Separate-object coding cuts Alcaraz's residual-off bbox rate from the
earlier union crop's 46,383 B to 32,212 B, but the foreground picture
remains 12.28 dB. Articulated warping improves the residual-off foreground
to 13.71 dB on Alcaraz and 15.95 dB on Federer; the best capped residual
rows reach 16.40 and 17.14 dB. These are well short of the roughly
22.4 and 25.2 dB needed with the chosen panorama courts. The per-object
correction is necessary and saves rate, but it does not establish the
weighted quality claim on these backgrounds. Perricard's matched court
is rate-constrained before residuals. This is a 48-frame development
boundary only; the training diagnostic is recorded separately.

## Pix2Pix exact-video training diagnostic

### Eight-hour decision

Use one free GPU and one trainable family, pix2pix, for an exact-video
overfit diagnostic. This model has an existing direct training entry point,
checkpoint/resume support and pose/body conditioning. A fast sweep across
all available models would spend the eight-hour window on setup and
non-comparable minimum budgets. Spend the budget on a short smoke check,
then repeated training with hourly checkpoints and an eight-hour hard
wall-time cap. Record actual examples, updates and wall time. Keep CPU
threads bounded and do not train a second model in parallel.

The selected clip is Alcaraz–Perricard scene 002: its two dataset tracks
have the same source frame IDs 0–47 as the 48-frame handoff. Train on
frames 1–47 (94 object images); each object's frame 0 is its reference,
never its target. The one-batch GPU smoke check completed on the intended
data path with finite discriminator and generator losses. The run uses
`pose_body`, `reference_mode=first`, 256-pixel crops, batch size 2, seed 42,
and checkpoint interval 600 s. Its fixed first-reference path has no
target-copy shortcut.

The exact-video fit is an **optimization test**. It can show whether the
existing model and conditioning can reproduce its own training clip. It is
not a generalization result and cannot by itself support a pre-shared
general tennis model: weights fitted to the particular transmitted clip
would be content-specific data and would need to be counted as rate. Do not
call the fit a codec win. Compare deterministic, dropout-off inference on
the 48 frames with the separate-object warp, and inspect the model's
foreground PSNR, weighted PSNR, residual clip fraction and qualitative
failure frames. The source mask is used for scoring only. A future
development-video training run is required for the general model claim.

### Controls and stopping

- Use only the available development tennis scene, with a fixed seed, exact
  file manifest, model checkpoint and command saved in the job directory.
- Verify a batch and one backward pass before detaching the long job.
- Save checkpoints at least hourly and at termination; preserve the best
  and latest checkpoint if the trainer provides both.
- Stop at eight hours or earlier on nonfinite loss, empty data, checkpoint
  failure, or a GPU conflict. Do not start a second family automatically.
- Run deterministic inference with the intended decoder conditioning and
  a shuffled-pose control. A model that merely memorizes frames without
  using transmitted motion does not answer the codec question.
- The dataset's raw `_pose_body` raster and crop are diagnostic training
  inputs. Codec evaluation must redraw pose from the decoded COCO-17 wire
  and use the decoded AV1 crop. Scores on raw dataset inputs are labeled
  learnability only; do not substitute them for bitstream scores.
- Only a measured new foreground and weighted score on the fixed
  background can support an exploratory quality comparison. Report actual
  component bytes, including any content-specific adapter or checkpoint
  if one is transmitted.

The long-job supervisor stores ten-minute progress, command output,
status, checkpoints and a single digest eight hours after launch under
`/home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/`.
No periodic chat updates are requested while the user sleeps. Append the
actual command, GPU UUID, checkpoints, metrics, failures and stop time below.

### Launch record

Started 23 September 2026 at 21:10:43 UTC on free NVIDIA RTX A6000 GPU
`GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49`. Supervisor PID 673126;
the prelaunch probe reported 1 MiB of 49,140 MiB in use and 0% utilization.
Job directory:
`/home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/pix2pix-perricard002`.
The supervisor reported `running`, PID 673133. The first training steps had
finite losses. Python is
`/home/itec/emanuele/.conda/envs/pointstream/bin/python`; command:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python \
  -m experiments.jobs.monitor start \
  /home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/pix2pix-perricard002 \
  --budget-hours 8 --thread 01a0ce73-4593-7810-97cf-4a3ca34e117c \
  --report-in-hours 8 --quiet-hours 8 --cpu-threads 8 \
  --claim-gpu GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49 \
  --claims-dir /home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/claims \
  --command python scripts/train_pix2pix.py \
  --data-root /home/itec/emanuele/pointstream-data/assets/dataset \
  --condition pose_body --video-filter alcaraz_perricard \
  --scene-filter scene_002 --frame-start 1 --frame-count 47 \
  --reference-mode first --img-size 256 --batch-size 2 \
  --num-workers 0 --epochs 2000 --checkpoint-interval-sec 600 \
  --seed 42 --out-weights JOBDIR/generator.pt \
  --checkpoint-path JOBDIR/checkpoint.pt --sample-dir JOBDIR/samples
```

The recorded invocation used the absolute pinned Python path in both places,
the explicit Codex executable path for the deferred digest, and expanded
`JOBDIR` to the directory above. `status.json` and `command.log` preserve
the active job state and actual argument vector.

At 21:19 UTC, an atomic checkpoint and generator weight file existed at
655 MiB and 208 MiB respectively. Their size reinforces that a checkpoint
trained on this clip cannot be excluded from a content-specific bitstream.
The trainer prints step losses and writes checkpoints but does not call the
supervisor's explicit progress API; `progress.log` therefore remains a
heartbeat, not a training-progress measurement. Use `command.log`, the
checkpoint contents and the final evaluator for the morning verdict.

### Completed fit and exact-video diagnostic — 24 September

The job completed all **2,000 epochs** with exit code 0 at **01:45:31 UTC**.
Its last logged generator loss was **2.6665**; loss is an optimization trace,
not a reconstruction score. The saved generator is **208 MiB** and the
optimizer/checkpoint is **655 MiB**. The deterministic evaluation used all
94 targets, object frames 1–47 for each of the two scene 002 tracks, with
frame 0 as each fixed reference. A within-track pose shifted by about half
the sequence is the conditioning control. The scorer and its result are
[here](/home/itec/emanuele/pointstream-data/outputs/modular/overnight-training-20260923/pix2pix-perricard002/overfit-score.json).

| Raw dataset diagnostic | Correct pose | Shuffled pose |
|---|---:|---:|
| Object-pixel PSNR | **21.45 dB** | 10.01 dB |
| Whole 256-pixel crop PSNR | **26.64 dB** | 14.50 dB |

The large correct-versus-shuffled gap is evidence that this fit uses its
pose input, rather than simply ignoring it. It does **not** establish a
PointStream quality result: the evaluator used raw training crops and
pre-rendered pose rasters, not the decoded AV1 appearance and COCO-17 wire
poses. These crop/object-pixel scores cannot be subtracted from the
full-frame, source-coded Perricard foreground PSNR of 24.93 dB as though
they shared a metric or resolution. The exact-video checkpoint is
content-specific and would be a roughly 208 MiB rate charge if sent for
this clip, far beyond any of the three source anchors. Its purpose was
to establish learnability and conditioning use. A general pre-shared
tennis checkpoint trained without this test scene, followed by actual
wire-conditioned full-frame scoring, remains required for a codec claim.

The chosen one-family diagnostic therefore answered the narrow question:
Pix2Pix can fit these 94 object images enough to reach 21.45 dB on their
nonblack pixels, and the pose control degrades sharply. It has not yet
shown enough full-frame quality or a valid zero-rate model policy to
claim a PointStream operating point.

## Continuation: Federer clarification and fixed wire

### Federer correction

The best Federer row has BG 26.447 dB, FG 17.143 dB and weighted 19.934 dB. The source has BG **31.363** dB, FG **21.686** dB and weighted **24.589** dB. Comparing BG 26.447 to weighted 24.589 mixed different metrics. The BG deficit contributes `0.3 × (31.363 − 26.447) = 1.475 dB` of the weighted shortfall; the FG deficit contributes `0.7 × (21.686 − 17.143) = 3.180 dB`. Foreground accounts for about 68% of the 4.654 dB gap. The court is below its own anchor, while foreground is the larger problem. This campaign does not change Federer’s background or re-run that clip.

The quoted **26.38 BG / 16.16 FG** values came from the earlier union-crop
`COCO-17 + FG54 + BG54` row, whose weighted score was **19.23 dB**. Its
26.38 dB background should be compared with the source's **31.36 dB
background**, not the source's 24.59 dB weighted score. On that earlier row,
foreground explains about 72% of the weighted deficit. The separate-object
row above supersedes it for current foreground planning.

### Fixed wire and provenance

Use only the 48-frame Alcaraz `window_48`, its `masks_48.npz`, and the saved `alcaraz000-n48.npz` plate cache. Regenerate the selected QP 46 panorama decode once if necessary, using the background campaign’s RGB/YUV path and `/opt/local/bin/ffmpeg` n7.1.1 preset `faster`. Require **24,648 B**, including 1,742 B homography side data, and the saved encoder path before compositing. Save the decoded RGB pixels outside the repo for both probes. Never rebuild a plate or run 192 frames. Inputs are RGB; AV1 intra and the Animate Anyone runtime receive BGR where required, with swaps on the last axis only. AV1 QP 42 crops must be padded to at least 64 pixels and multiples of 8. Reject empty native output.

Each object gets its own appearance, first/reference alpha, presence, bbox and optional COCO-17 pose payload. Charge `T = B + F + M + R + H`, each byte once, including added crop indices, silhouettes and other side data. Source masks may select encoder references, construct residuals and score outputs; a decoder must see only transmitted fields. Source and PointStream use `score_regions` with the same masks. Sender includes offline plate build; client is decode plus render. Report exact encoder/decoder paths and versions, encode/decode/render seconds, and residual clip fraction. A faster decode at lower weighted quality is not a win.

## Continuation: executed screens and 48-frame arms

### Execution provenance

The approved Alcaraz QP 46 panorama was regenerated once from
`background-arms/cache/alcaraz000-n48.npz`, without rebuilding its plate.
The VVC payload is **22,906 B** and the homography side data is **1,742 B**:
**24,648 B** total, exactly the saved background row. The encoder is
`/opt/local/bin/ffmpeg`, `n7.1.1-56-gc2184b65d2`; encode took 2.817 s,
plate decode 2.560 s, and panorama render 0.913 s. The decoded RGB clip is
cached at `foreground-part2/alcaraz000-qp46-background-rgb.npy` outside the
repository and guarded by SHA-256 in its adjacent JSON. Both probes use this
same decode.

### Animate Anyone screen result

The installed `finetuned_tennis` checkpoint ran successfully at 20 steps on
two separately encoded players for frames 0–15. This is a **16-frame
development screen**, not a 48-frame claim point. It used AV1 QP 42 crops
through `/opt/local/bin/SvtAv1EncApp` v1.8.0, decoded COCO-17 pose wire,
and the saved QP 46 background. The wire charged B **24,648**, F **5,581**,
M **3,524**, R **0**, H **1,203** = **34,956 B**. The two crops were 3,752
and 1,829 B; the two poses were 1,632 B each. The checkpoint files are
3,438,373,605 B denoising UNet, 3,438,323,133 B reference UNet,
1,817,900,227 B motion module and 4,351,319 B pose guider. They are treated
as pre-shared for this diagnostic, though its training set includes the
development match.

| Same 16 frames | Overall | FG | BG | Weighted | Residual clip fraction |
|---|---:|---:|---:|---:|---:|
| Articulated paste, charged wire | 30.612 | 16.301 | 31.492 | 20.858 | 0.0565 |
| Animate Anyone, charged wire | 29.001 | 12.264 | 29.936 | 17.566 | 0.1059 |
| Animate Anyone with true target alpha, **oracle** | 29.712 | 10.193 | 31.771 | 16.666 | 0.2121 |

The generated result loses **4.037 dB FG** to articulated paste on the
same frames and wire. The target-alpha oracle is explicitly not decodable;
it shows the missing player pixels are a larger problem than the transmitted
matte alone. A four-frame control dropped from 12.674 dB FG with correct
conditioning to 10.889 dB with shifted pose and 8.903 dB with swapped
references, so the checkpoint is responding to both inputs. The first object
incurred 174.906 s cold model/render time, the warm second 4.594 s; total
model plus composite was 179.649 s and 16-frame client time was 183.331 s.
Foreground encode/decode were 14.895/0.210 s; the offline plate build was
104.062 s. These 16-frame timings are recorded separately from the 48-frame
source encode/decode of 8.459/29.581 s and do not establish a latency win.
The runtime used Torch 2.5.1, Diffusers 0.30.3 and Transformers 4.46.3.
Raw scores and saved generated pixels are under
`foreground-part2/animate-anyone/` outside the repo.

**Animate Anyone retraining decision:** do not start an exact-video overfit
or full multi-video retrain in this deadline campaign. This 20-step in-set
checkpoint is already well below the paste baseline, its generated pixels
still fail under target alpha, and the current PointStream trainer has no
Animate Anyone training entry point. Exact-video weights would also need to
be transmitted and charged; multi-video training would require new wiring
and enough development data to show a gain on the charged object wire. Keep
the checkpoint as a measured negative control and revisit it only after a
decoder/matte and training-path redesign or after a separate capacity probe
with a clear time budget.

### Alcaraz appearance, silhouette and residual result

The 48-frame source VVC QP 46 anchor is **65,149 B**, overall **33.508**,
FG **21.734**, BG **33.843** and weighted **25.366 dB**; sender is
**8.459 s** and client **29.581 s**. Every row below uses B = **24,648 B**
for the same decoded panorama and M = **780 B** for the two separate bbox
and presence timelines. Appearance is AV1 intra QP 42 through
`/opt/local/bin/SvtAv1EncApp` v1.8.0, decoded by `/opt/local/bin/ffmpeg`
`n7.1.1-56-gc2184b65d2`.

| Refresh | F | H | R | Total B | Overall | FG | BG | Weighted | Clip fraction | Sender s | Client s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First only | 5,581 | 1,207 | 0 | 32,216 | 29.718 | 12.277 | 31.486 | 18.040 | .1254 | 111.129 | 24.806 |
| Every 24 visible frames | 11,532 | 2,069 | 0 | 39,029 | 30.054 | 13.130 | 31.581 | 18.665 | .1006 | 112.867 | 24.801 |
| Every 12 visible frames | 23,824 | 4,447 | 0 | 53,699 | 30.404 | 14.197 | 31.658 | 19.435 | .0762 | 115.246 | 25.009 |
| Every 6 visible frames | 44,028 | 9,250 | 0 | **78,706** | 30.900 | 18.015 | 31.857 | 22.167 | .0493 | 119.842 | 25.625 |

The every-6 row exceeds the source by **13,557 B** and still trails weighted
quality by **3.199 dB**. Target-mask oracles are separate: true current
silhouette with the first warped crop scored FG **12.389 dB**, only 0.112 dB
above first-only; true current appearance through that silhouette exactly
reproduced the foreground, but sends no valid appearance wire. Losslessly
coding every object silhouette as an independent ROI zlib payload cost
**62,977 B** before appearance: even first-only would be **95,193 B**, so
no silhouette composite is a capped row.

The best capped residual-off reference schedule is every 12 frames. Its
foreground residual clip fraction is **.0762**, above the .05 stop for a
finer quantizer. The following QP 54/62 residuals are VVC `faster` through
the same ffmpeg n7.1.1 path. For every row B = **24,648**, F = **23,824**,
M = **780**, H = **4,447**; only R changes. The raw ledger includes all
individual encoder/decoder clocks and versions.

| Residual on every-12 | R | Total B | Overall | FG | BG | Weighted | Sender s | Client s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FG QP54 | 7,602 | **61,301** | 30.984 | 16.860 | 31.653 | **21.298** | 126.524 | 53.525 |
| FG QP62 | 6,369 | 60,068 | 30.779 | 15.808 | 31.578 | 20.539 | 126.785 | 54.035 |
| BG QP54 | 5,349 | 59,048 | 30.880 | 14.084 | 32.268 | 19.539 | 155.542 | 53.533 |
| BG QP62 | 4,401 | 58,100 | 30.591 | 14.130 | 31.895 | 19.460 | 155.738 | 53.698 |
| FG54 + BG54 | 12,951 | **66,650** | 31.409 | 16.727 | 32.142 | 21.352 | 166.820 | 82.122 |
| FG62 + BG54 | 11,718 | **65,417** | 31.175 | 15.765 | 32.040 | 20.648 | 167.081 | 82.633 |
| FG54 + BG62 | 12,003 | **65,702** | 31.091 | 16.767 | 31.776 | 21.269 | 167.015 | 82.287 |
| FG62 + BG62 | 10,770 | 64,469 | 30.865 | 15.783 | 31.673 | 20.550 | 167.276 | 82.797 |

The bold totals above the 65,149 B source cap fail the rate test. The best
under-cap row is FG QP54 only: **21.298 versus 25.366 dB weighted**, with
**3,848 B** slack. At its 31.653 dB background, foreground would need
**22.672 dB** to tie; it scores **16.860 dB**. No part 2 refresh,
silhouette or residual row is claimable. The residual-row client clocks in
this first native run omit the final correction-add operation, so they are
lower bounds; even those bounds exceed the source's 29.581 s. The runner now
times that operation for subsequent runs. Residual-off client clocks include
appearance decode and paste and are shorter than the source, but quality is
well below the decision threshold. Sender clocks include the 104.062 s
offline plate build and exceed the source throughout.

### Temporal object-video result

Stage C coded two independent 48-frame AV1 color videos and two independent
AV1 alpha videos. The first player's fixed canvas was 464×312, the second
240×328. Their bbox and presence streams cost M = **780 B**. The alpha
threshold after video decode kept the scored background at 32.333 dB on the
tested rows; the decoder received only those charged alpha videos. AV1 used preset
`10`, `/opt/local/bin/SvtAv1EncApp` v1.8.0, and decoded through
`/opt/local/bin/ffmpeg` n7.1.1; the exact color/YUV round trip, per-object
stream sizes and split clocks are in `roi-video/alcaraz000-roi-video.json`.
Every row has B = **24,648 B** and R = **0**.

| Color / alpha QP | F | H | Total B | Overall | FG | BG | Weighted | Clip fraction | Sender s | Client s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 / 32 | 132,737 | 100,466 | 258,631 | 32.318 | 30.470 | 32.333 | 31.029 | 0 | 115.654 | 6.411 |
| 42 / 42 | 132,737 | 83,417 | 241,582 | 32.318 | 30.470 | 32.333 | 31.029 | 0 | 115.727 | 6.288 |
| 50 / 32 | 84,350 | 100,466 | 210,244 | 32.287 | 27.751 | 32.333 | 29.126 | .0001 | 116.227 | 6.268 |
| 50 / 42 | 84,350 | 83,417 | **193,195** | 32.287 | 27.751 | 32.333 | 29.126 | .0001 | 116.299 | 6.268 |

All four rows beat the 25.366 dB source weighted score but exceed its
65,149 B by at least **128,046 B**. They are **not claimable**. The alpha
videos alone cost at least **83,417 B**, above the entire source bitstream;
coarsening this same independent video-matte format toward the cap is not a
credible near-term path. Client decode plus render is faster than the source,
but rate disqualifies every row and sender time remains much slower because
of the offline plate. The image quality result shows that sending current
object pixels can solve the distortion problem; the needed next invention is
a much cheaper motion/appearance or alpha representation.
