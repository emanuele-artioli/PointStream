# The engine roster — living scoreboard

**Update this file whenever an engine is scored.** It exists so that a session
can see, in one place, how good each engine has been made, what configuration
produced that, and which direction is worth more effort. Numbers that are
citable also live in `PLAN.md` §2; this file is the working view and may carry
in-progress arms that the paper must not.

**The task is the coding task**: appearance from a keyframe, condition from a
later frame, reconstruct the object. Metric is object-bbox LPIPS (lower better)
on the §2.10 protocol — 12 clips, calibrated LPIPS, seed 42. `reid` is on
`TENNIS_SCALE` (different person 0.5315, same person 0.8663).

**Every metric here was measured after 2026-08-23.** Anything scored before that
date is void (`PLAN.md` §2.7).

## Anchors — what any engine must be read against

| Anchor | Object LPIPS | Object PSNR | Meaning |
|---|---|---|---|
| identical | 0.000 | ∞ | instrument zero |
| **static copy (paste)** | **0.4505 ± 0.0220** | 13.51 ± 0.44 dB | **the bar. Right player, wrong pose, no model** |
| unrelated image | 0.7358 ± 0.0075 | 9.06 ± 0.13 dB | no-information floor |

Separation between the two live anchors is **0.285 LPIPS / 4.45 dB**. An engine
scoring near 0.74 is doing no better than handing over a photo of a different
player.

## The roster

| Engine | Driven | Object LPIPS | Object PSNR | s/frame | Trained? | Uses appearance? |
|---|---|---|---|---|---|---|
| **static copy** | — | **0.4505** | 13.51 dB | **0.00** | n/a | n/a (it *is* the appearance) |
| upscale-refine | frame | **0.5585** | 12.73 dB | **0.00** | no (not generative) | n/a |
| seg-controlnet | frame | 0.5595 | 12.19 dB | 0.94 | pose+caption only | **no** |
| animate-anyone | **clip** | 0.5692 | 12.21 dB | 0.99 | stock | not established |
| pose-controlnet | frame | 0.6031 | 12.03 dB | 0.96 | pose+caption only | **no** |
| trajectory-controlnet | frame | 0.6229 | 11.78 dB | 0.95 | pose+caption only | **no** |
| **ip-adapter, fine-tuned (ep 1)** | frame | **0.6922 ± 0.0094** | — | ~1.0 | **yes, BP25** | **YES — 3.8σ vs shuffled** |
| ip-adapter, fine-tuned (ep 2) | frame | 0.6953 ± 0.0103 | — | ~1.0 | yes | yes |
| ip-adapter, fine-tuned (ep 3) | frame | 0.6947 ± 0.0101 | — | ~1.0 | yes | yes |
| ip-adapter, stock | frame | 0.7586 ± 0.0092 | 9.00 dB | 0.97 | no | weakly |
| pix2pix | frame | 0.7820 | 12.60 dB | **0.03** | — | no |
| multi-controlnet | — | **never scored** | — | — | **never trained** | — |
| uni-controlnet | — | **never attempted** | — | — | no | — |

**Nothing beats the paste.** Best engine is `upscale-refine` at 0.5585, which is
not a generative model. The best *generative* engine is `seg-controlnet` at
0.5595, trained with no reference image at all.

## What BP25 actually changed

Fine-tuning IP-Adapter moved it **0.7606 → 0.6922**: from worst-but-one to
mid-pack, still behind four engines measured in wave 3. The LPIPS gain is not
the finding.

**The finding is that the appearance path is real.** Epoch 1 against its own
keyframe versus a shuffled one: **−0.074 LPIPS (3.8σ)** and **+0.075 reid
(3.6σ)** on clip means. This is the first evidence in this project that any
engine uses the reference. Previously the standing position was that none did.

Not established: better than an unrelated image. That is **1.3σ on clip means**
and must not be claimed, even though the item-level n=96 figure reads 3.3σ —
8 offsets inside one clip are not independent.

## The offset finding — where a generative engine could ever win

Computed 2026-08-26 from `outputs/bp25-ip-adapter/rows.json`, which carries
per-offset rows. Offset is distance from the keyframe, so it stands in for
deformation.

| offset | paste | ip-adapter ep1 | gap |
|---|---|---|---|
| 1 | 0.239 ± 0.053 | 0.682 ± 0.025 | +0.443 |
| 4 | 0.452 ± 0.060 | 0.664 ± 0.030 | +0.212 |
| 8 | 0.582 ± 0.043 | 0.720 ± 0.025 | +0.138 |

**The paste degrades about ten times faster than the model**: +0.0458 LPIPS per
offset against +0.0049. Linear extrapolation crosses at **offset ≈ 10.4**, just
past the measured range.

**Do not read that crossover as a win.** The paste at offset 8 is heading toward
0.74, which *is* the unrelated-image level, and the model sits flat at ~0.70.
A crossover there means both arms have degraded to roughly "photo of the wrong
player". The compression-relevant regime is the other end, where a paste scores
0.239 and nothing is close to it.

`BP28` tests this directly by extending the harness to offsets 12/16/24.

## Reading this table for direction

Two constraints narrow the search a lot:

1. **Speed.** BP25 established that the ranking only means anything at 20
   diffusion steps — at 4 steps the stock adapter cannot beat an unrelated
   photo. Twenty steps is ~1 s/frame against a 30 fps target. **The only two
   fast entries in the whole roster are the two non-generative ones**
   (`upscale-refine` 0.00, `pix2pix` 0.03).
2. **The task is reproduction, not synthesis.** The object is ~1% of a 4K frame.
   IP-Adapter conditions on a CLIP *image embedding* — semantic, not spatial —
   and its declared ceiling was always kit colour and build, never identity. It
   reached that ceiling and stopped.

The family that is both fast and identity-preserving *by construction* is
**warping**: take the keyframe crop, warp it to the new pose, refine the
residual. It inherits the paste's strength at low offset and attacks exactly what
the paste gets wrong at high offset. `animate-anyone`'s ReferenceNet is the
nearest thing already on the roster and **has never been retrained**
(`PLAN.md` §7 P2 item 17).

**The honest alternative, which the numbers currently favour:** the paper's
contribution is the object-centric decomposition plus a cheap appearance channel
(a paste) plus a residual — and the generative engine is not where the value is.
`BP24` is what makes that claim measurable, because it is a rate claim.

## Open arms, cheapest first

| Arm | Cost | Decides |
|---|---|---|
| extend offsets to 12/16/24 (`BP28`) | one GPU run, existing harness | whether keyframe interval is a lever |
| retrain animate-anyone / ReferenceNet | a training campaign | whether spatial appearance beats a paste |
| warping + residual refinement | new component | the only route that is plausibly real-time |
| train multi-controlnet | a training campaign | an unscored roster entry |
| uni-controlnet | a training campaign | last, by standing decision |
