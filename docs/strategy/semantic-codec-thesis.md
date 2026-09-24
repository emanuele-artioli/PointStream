# PointStream: semantic codec thesis and decision record

**Status:** working research strategy, 24 September 2026. This is a place to
revise the idea and record decisions. It does not replace the [active development
protocol](../workflow/session/evaluation-campaign/20260923-development-campaign.md),
its weighted-PSNR pass criterion, or the [paper area](../areas/paper.md).
Measured rows below are development evidence, not an assembled-codec win or a
submission claim.

## 1. The idea, from first principles

A conventional video codec predicts pictures from other decoded pictures and
codes the remaining error. Its motion prediction, transforms, and reference
pictures are powerful, but the decoder usually has no explicit representation
of *this player*, *this court*, or *this ball*. PointStream asks when a shared
semantic representation can replace repeated pixel updates: send reusable
appearance and scene information, then send only the changes needed to render
the next frame. The mental image of a familiar object is the intuition; a
counted, decoder-available model is the engineering requirement.

The lineage separates three questions:

| Work | Question and working answer | Boundary |
|---|---|---|
| [Presley](/home/itec/emanuele/presley/68e8b6bb11d0dd9e62a67aef) | Can a conventional codec spend fewer bits when player pixels are removed and the background is allowed less fidelity? Yes, on the measured headroom task. | Saved player bits must pay for putting the player back. |
| [GenStream](/home/itec/emanuele/genstream/682c320f388146ae7ee133b7) | If athlete appearance and a scene model are already shared, can pose and camera motion replace much of the pixel stream? The ice-skating demonstration motivates this route. | A pre-shared athlete and 3D scene are strong assumptions. |
| PointStream | Can a broadcast match bootstrap and reuse its own court/background and player appearances, then transmit motion, ball, refreshes, and necessary residuals? | The full wire, reconstruction quality, and amortization must beat strong video anchors under the same access and quality conditions. |

Tennis is a useful *candidate regime*: two usually separated players, a small
fast ball, a mostly planar court, a long-lived venue view, and many shots from
fixed-mount pan/tilt/zoom cameras. None is guaranteed. Cuts, moving camera
centers, occlusion, changing light, shadows, spectators, ball kids, clothing,
and broadcast graphics can invalidate reuse. A background may receive fewer
bits where viewers tolerate it, but court lines, the ball, and salient people
remain task-critical. "Background" therefore cannot be one quality class.

## 2. What the present evidence says

| Observation | Scope and provenance | Implication |
|---|---|---|
| Removing players before conventional re-encoding saved mean BD-rate of 14.2% ± 2.6% for VVC and 15.4% ± 2.8% for AV1. | Eight 48-frame 4K scenes in the [evaluation record](../areas/evaluation.md#what-this-session-measured-and-what-it-is-not--22-september-2026); the figures concern a player-free encode. | This is an upper budget for replacing the removed player at comparable quality, not a PointStream win. The saving depends strongly on player size and motion. |
| On 48-frame Federer scene 007, the measured one-crop + COCO-17 control is 136,228 B: 129,452 B plate, 1,962 B appearance, 4,794 B raw pose, and 20 B metadata. VVC QP 46 is 112,295 B. | [Appearance-motion ledger](../areas/evaluation.md#appearance-motion-only-control). The control uses a classical warp, not a generator. | The plate is about 95% of this wire; even free pose would leave the arm above this anchor. Pose coding alone cannot decide this comparison. |
| A registered VVC-intra plate on the same Federer window uses 58,814 B plus 1,728 B homographies. A 107,005 B background residual at QP 46 reaches 29.56 dB background, below the anchor's 31.36 dB, and the full arm has no Pareto win. | [Background scorecard](../scorecards/02_background.md) and [warp-residual record](../areas/evaluation.md#warp-residual-probe--22-september-2026). | A single registered plate does not eliminate the changing-background error on this clip. This measurement covers the whole background mask, not a separate court-only mask. |
| With players removed, a court-matched inpainted video leaves 17,588 B on large-player Perricard 002 and 4,103 B on small panning Federer 007. Alcaraz 000's panorama leaves 40,501 B but its background is about 1.5 dB below the source. | [23 September background campaign](../workflow/session/evaluation-campaign/20260923-background-campaign.md), 48 frames, QP 46. | The most favorable foreground budget is clip-dependent. Perricard is the measured candidate for restoring a player at matched court quality; none of these rows includes a restored player. |
| The current measured assembled configuration and the registered-plate residual arm do not dominate VVC or AV1 at the current weighted-PSNR gate. | [Evaluation area](../areas/evaluation.md) and [paper area](../areas/paper.md). | No competitive PointStream result is established. The earlier 65.4% and 32.5% ladder wins came from a constant table and remain withdrawn. |

The original 48-frame plate cost does not prove or disprove match-long
amortization. Conversely, a long match cannot rescue a **per-frame** warp or
residual cost that is close to the baseline's whole-frame cost. A longer
window, with actual scene persistence and refreshes, must be measured.

## 3. What conventional codecs already do

The argument cannot be that AV1 or VVC forget the background every segment by
definition. Their reference pictures and block motion prediction already
reuse earlier image content. A [published VVC composite long-term reference
method](https://www.jdl.link/doc/2011/20191223_08803708.pdf) builds a clean
non-displayed picture from selected background blocks and predicts later frames
from it. The paper describes an encoder-side technique, including block
selection and no-output reference signaling. [AV1 alternate reference
frames](https://aomedia.googlesource.com/aom/+/386cb69c9773d60f15631a8e61d81bb47f67fe91/doc/dev_guide/av1_encoder.dox)
are also encoded references that are often not displayed. These mechanisms are
related but are **not evidence that our present encoder commands create a
match-long composite plate**, nor that AV1's alternate reference is the same
algorithm. Check the actual builds, flags, emitted streams, and reference
lifetime before describing either measured anchor that way.

Fair comparison means giving conventional anchors the same match duration,
random-access intervals, segment boundaries, lookahead, and decoder state as
PointStream. If the use case requires each point to decode independently,
charge PointStream's plate and appearance at each reset too. If the client can
retain them across points, test continuous anchors that can retain references
across those points. Separately test a composite-reference VVC variant if one
is available and reproducible. Its costs, including the non-displayed picture,
belong in its bitstream.

Two ideas from composite references are useful to test inside PointStream:
refresh only stale plate blocks, and select clean background blocks with a
foreground mask or a measured selection cost. Both are proposals, not current
capabilities of the PointStream plate.

## 4. Candidate representation: three scene layers and an explicit ball

1. **Players and rackets.** Send one initial appearance per player, then
   temporally predicted, quantized pose and position. Represent the racket
   beyond body joints (for example hand, throat, and tip), or provide a small
   local correction where the generator fails. A model trained only on body
   pose cannot be assumed to reconstruct racket geometry. Refresh appearance
   when identity, kit, lighting, camera scale, or pose coverage requires it.
2. **Court geometry.** Build a court plate from original pixels, with player
   masks used for fitting and compositing. Estimate camera mapping on court
   lines and texture, assess the *court-only* warp error, and code residuals or
   refreshes when needed. A homography maps points on one plane between camera
   views, including zoom under a pinhole model; it does not make stands,
   moving people, occluded content, or resampling error invertible. Test plane
   versus [cylindrical warping](https://docs.opencv.org/4.x/d3/dd6/classcv_1_1detail_1_1CylindricalWarper.html)
   for wider pans, and split or refresh sprites when one map fails. The earlier
   [multi-sprite MPEG-4 work](https://www.dirk-farin.net/projects/multisprites/index.html)
   is direct prior art for this choice. A sharp line is a requirement, not a
   license to synthesize a different court.
3. **Stands and other background.** Start with a coarsely coded or softened
   plate, warped with the camera when adequate. Add low-rate temporal updates
   for conspicuous motion or lighting changes. This layer should be judged for
   visible distraction and temporal stability; the court should not inherit
   its error budget.
4. **Ball.** Send an explicit trajectory and visibility state; use a small
   local correction when detection, motion blur, or occlusion defeats that
   representation. The ball's small pixel area does not imply low importance.

The split is a *hypothesis*: the 107 kB Federer residual may lie mostly in the
off-plane background, or court alignment itself may remain expensive. A
court-mask error ledger resolves that question before implementing a more
complex warper. A multi-layer result must count all masks, maps, refreshes,
appearance, pose, ball, residuals, and headers on the physical wire.

## 5. Generator choice and limits

The [OpenPose ControlNet checkpoint](https://huggingface.co/lllyasviel/sd-controlnet-openpose)
is a pose-conditioned *image* model; independent frame generation gives no
temporal guarantee. The [Animate Anyone work](https://github.com/HumanAIGC/AnimateAnyone)
motivated the tennis fine-tune because it conditions character animation on a
reference image and pose over time. Local model and training results must
determine whether it actually restores identity, limb timing, racket shape,
and frame-to-frame consistency at the available rate.

[MTTF](https://arxiv.org/abs/2410.10171)
is a candidate comparison or borrowed motion representation. Its authors
report gains over VVC on talking-face and moving-body data at extreme rates;
that result does not establish a gain on 4K broadcast tennis, a court-line
constraint, or the PointStream wire. Its [implementation](https://github.com/xyzysz/Extreme-Human-Video-Compression-with-MTTF)
is available. Inspect its reference-picture and model costs before matching an
operating point. No MTTF tennis run is recorded here.

## 6. The claim to seek, and the test that could support it

**Conditional target:** In fixed-mount pan/tilt/zoom tennis with a persistent
court and players, at an ultra-low rate and over a duration long enough to
amortize shared state, a court-preserving, pose-driven object codec delivers
better *measured* viewer-relevant quality than AV1 and VVC at no more bytes.
This is a research target, not a present result. If it holds only for a subset
of camera motion, player scale, or duration, state those boundaries in the
headline claim. Do not replace the active weighted-PSNR gate silently; a new
perceptual claim needs an explicit protocol decision and fresh evidence.

The proposed evaluation should answer these questions in order:

1. **Where is the rate?** Compare 48-frame, point-length, and match-length
   windows. Plot cumulative transmitted bytes and the break-even time. Charge
   initial court, crowd, and player appearances once only while the decoder
   truly retains them; charge every update and reset. Use exact encoder and
   decoder binaries, versions, presets, and lookahead.
2. **Does the court lock?** Report court-mask distortion and line displacement
   after reconstruction, including pan and zoom. Measure residual bytes and
   temporal jitter on that mask. If the court residual remains large, stop the
   plate claim or narrow its camera regime.
3. **Can the player return inside the saved budget?** On Perricard 002 first,
   compare reconstructed player fidelity and identity with the anchor at the
   17,588 B foreground budget. Include racket and ball correctness, temporal
   consistency, and the full wire. Alcaraz and Federer test the boundary.
4. **Is the ultra-low-rate gain visible?** Reconstruct players at a resolution
   the generator supports, composite them into the transmitted court, and
   compare at matched full-wire rates and a common display resolution. Report
   player-region LPIPS alongside controlled
   human preference, temporal stability, court geometry, ball visibility, and
   conventional PSNR/VMAF. LPIPS alone cannot certify identity or correct
   geometry. Include resolution-adaptive AV1/VVC anchors and more than one
   source match before making a general claim.
5. **Is it practical?** Report sender and client runtime, memory, startup
   delay, model availability, and behavior at camera cuts. Shared neural model
   weights may be pre-installed only under a stated distribution assumption;
   content-specific fine-tuning or assets must be charged or justified.

The 23 September 48-frame background campaign is exploratory. Its projected
192-frame rows are arithmetic projections, not a measured match-length result.
The active protocol currently skips held-out confirmation; a publishable
broader claim needs a deliberate decision about independent-source evidence.

## 7. Open decisions and update rules

| Decision | Current status | Evidence needed to change it |
|---|---|---|
| Is a single court map accurate enough on pans and zooms? | Unknown; whole-background Federer warp is insufficient. | Court-only warp error, line displacement, and residual bytes on at least still and panning shots. |
| Does match-long reuse beat an equally persistent anchor? | Untested. | Continuous match-duration bitstreams for both, including refreshes and random-access policy. |
| Can a generator restore player and racket within the available bytes? | Untested on the one-crop wire; classical warp loses badly. | Same-wire generated decode, region quality, temporal and identity checks. |
| Is ultra-low-rate perceptual superiority a viable paper claim? | Hypothesis. | Frozen operating band, metric controls, independent sources, anchor parity, and a full-wire win. |
| Should the paper retain weighted PSNR or adopt a new primary endpoint? | Current development gate remains weighted PSNR. | Explicit protocol revision before testing a new headline claim. |

When updating this document, date the decision, link the immutable run record
or paper source, state the clip/window/quality/rate scope, and distinguish
**measured**, **projected**, and **proposed**. Keep a failed arm and its claim
boundary visible. Update [PLAN.md](../../PLAN.md) or an area document only
when its operational summary or acceptance criteria actually change.
