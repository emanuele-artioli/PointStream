# 3. Motivating Example: A Broadcast Tennis Point

**Draft status.** This chapter motivates the experiment and defines the accounting question. The conventional headroom values below are diagnostics on altered coding targets, not PointStream gains. PointStream's tested tennis configurations have not yet produced a claimable 48-frame point in the current campaign ([project status](../../PLAN.md); [campaign record](08-evidence-ledger.md)).

## Scene structure

Consider a 48-frame, 24 fps crop from a 4K broadcast point. A pan–tilt–zoom camera follows the players. The court and stands occupy most pixels; two players drive much of the visible action, while the ball and racket are small, fast, and semantically important. A candidate codec might encode a reusable view of the court and venue, transmit player appearance and motion, carry object masks or labels for reliable composition, then add residual data wherever its synthesized frame differs from the source.

This construction creates an explicit rate budget. Let `B_anchor(q)` be the anchor's bits for the declared full-frame quality target `q`. Let `B_PS(q)` include the PointStream background or venue data, all object appearance and motion, masks and metadata, residual, and any updates due within the same delivery interval. The practical requirement is:

```text
B_PS(q) <= B_anchor(q)
```

at a quality level that is actually measured on the delivered reconstruction. For curve-based evaluation, this inequality is only an intuition at a matched point: the comparison must use the agreed rate–distortion curves, anchor, and common quality range. A single bitrate or quality sample does not establish a BD-rate result. The [method chapter](04-method.md) and [evaluation chapter](06-evaluation.md) must define the exact measure and rate-bearing components.

## What the headroom diagnostic establishes

The motivating study used eight 4K scenes, each 48 frames at 24 fps, from six tennis matches. It encoded source and altered-content arms across AVC, HEVC, AV1, and VVC ladders. The player silhouettes covered `1.11 ± 0.32%` of frame area. Removing those players by inpainting the background produced the following mean BD-rate changes:

| Codec | Foreground-removal BD-rate change (mean ± SE) |
|---|---:|
| AVC | `17.0 ± 3.1%` |
| HEVC | `18.3 ± 3.4%` |
| AV1 | `15.4 ± 2.8%` |
| VVC | `14.2 ± 2.6%` |

These values summarize a diagnostic of conventional coding efficiency after the players are removed from the coding target. They do not measure a transmitted player representation, player restoration quality, full-scene rate, or PointStream. Scene variation matters: the observed removal changes ranged from `1.1%` on `djokovic_zverev/scene_002` to `26.4%` on `alcaraz_highlights/scene_000` for AVC. The table's mean cannot be treated as a guaranteed budget for an arbitrary tennis point.

A second diagnostic compared a JPEG panorama plus `1728 B` of per-frame homography metadata with conventional inter-coded background video. It reported background BD-rate changes of `64.3 ± 8.4%` for AVC (`n=6`), `66.5 ± 7.3%` for HEVC (`n=7`), `78.0 ± 5.6%` for AV1 (`n=8`), and `76.1 ± 3.9%` for VVC (`n=8`). Since the sample counts differ and the comparison isolates the background target, these values support only that measured background diagnostic. They do not establish an end-to-end advantage. Modern codec reference tools must also be represented fairly in the anchor configuration; a panorama is not presumed unique to semantic coding.

The headroom manuscript scores each coding arm against its own source across rate ladders. In the foreground diagnostic, removing the players changes the target itself. That is useful for estimating where a coding opportunity may exist, but it cannot establish the quality of any reconstructed player or provide a rate-matched comparison to the original full scene. The source and provenance are recorded in the audited manuscript's [problem section](../../../67a9ea6275d3d9785ce57026/sections/problem.tex) and [headroom appendix](../../../67a9ea6275d3d9785ce57026/appendices/headroom_measurement.tex), revision `21d887c`. The cited JSON, `outputs/bp21-headroom/report.json`, is external to this checkout and was not independently reopened while preparing this chapter.

## Test the full accounting boundary

For a delivered point, write the PointStream total as:

```text
B_PS = B_scene/setup + B_background + Σ_i(B_appearance,i + B_motion,i + B_mask,i)
       + B_metadata + B_residual + B_asset/update
```

The terms must be defined so that each transmitted byte is charged once. A model already installed on all clients can be reported as a deployment assumption; if it must be distributed or updated for the event, its delivery bytes belong in the setup or update cost. Similarly, an event venue scan or initial background reference cannot silently disappear from the accounting when it is required by the decoder. Report setup bytes and recurring bytes separately, along with the delivery scope over which setup is shared.

For an illustrative constant-cost model, let `ΔB_setup` be PointStream setup bytes minus anchor setup bytes, and let `A` and `P` be recurring bytes per point at the same quality. When `ΔB_setup >= 0` and `S = A - P > 0`, the projected break-even is:

```text
N_break-even = ceil(ΔB_setup / S)
```

This is a planning calculation, not a measured result. Actual points vary and refresh assets, so the experiment measures cumulative setup plus recurring traffic for both methods over real consecutive points, while checking quality on each point. If recurring savings are nonpositive and PointStream has an extra setup cost, repetition cannot repay that cost under this model. A hypothetical three-hour match or repeated multiplication of one point is not evidence of amortization.

For a component-level conclusion, compare enabled and disabled configurations under [Chapter 6](06-evaluation.md). Source-video anchors, background-only configurations, object appearance and motion, masks, and residual expose which terms buy measured reconstruction quality. Curves require adequate rate samples and reported common quality support; historical pointwise gates remain scoped to their original campaigns.

## A stronger motivating example for the revised paper

Pair two tennis windows: a relatively static view with large visible players, and a panning view with small players, racket swings, and occlusions. Compare a conventional anchor, MTTF, a qualified general-video generative codec, and PointStream at matched complete rates. Show whole frames, identical player/ball/racket crops, temporal trajectories, actual bytes, and client cost. This directly tests whether plausible generated video preserves the action people follow, and whether explicit object geometry helps. Include an arm without racket/ball geometry and a simple pasted-reference control.

This figure is proposed, not available evidence. If PointStream preserves an independently measured ball/racket trajectory or motion event at competitive perceptual quality and lower rate or client cost, it supplies a more relevant motivation than altered-target headroom alone. If it fails, retain the failure and avoid a content-novelty claim unsupported by a mechanism. E01, E03, E04, and E06 prepare this comparison before E08 measures full curves.

## Evidence boundary and next chapters

The latest project status says there is no confirmed rate–distortion win, and the current campaign records no claimable tested 48-frame point. This motivates measuring the full budget and reporting failure modes rather than extending the headroom percentages into a PointStream result ([active plan](../../PLAN.md), [campaign record](08-evidence-ledger.md)).

The next chapters should specify the [related work](02-related-work.md), [method and byte contract](04-method.md), [implementation](05-implementation.md), [evaluation](06-evaluation.md), [experiment plan](07-experiment-plan.md), and [evidence ledger](08-evidence-ledger.md). Each candidate contribution remains a hypothesis until a complete run provides the corresponding source, command, configuration, quality result, and wire accounting.
