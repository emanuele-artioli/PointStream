# Second-match temporal comparison: fixed before scoring

Use exactly the first 240 native 25-fps frames of authenticated Alcaraz–Hurkacz /028 (source SHA `c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934`), with the twelve preselected ending ticks 10, 30, …, 230 and their immediate predecessor ticks. Compare actual ordinary AV1 20.3608 kbps and accepted DCVC-UF 20.4117 kbps at identical displayed time/geometry. Once the separately running genuine joint receiver exists and passes its full-frame identity gate, append its 20.4075 kbps cached output on the identical 12 pairs; do not generate substitute player frames, interpolate frames, or move timestamps. State cold cost and cache/model assumptions with any joint comparison.

Reproduce the Nadal source-flow method in `joint-trajectory-20261122/temporal.py`: convert the 1920×1080 source to the same 960×544 working grid, calculate DIS source flow from each ending frame back to its immediate predecessor, warp each previous display with that flow, and calculate luma change energy and absolute source-referenced residual error. All codec frames first follow their actual display conversion to 1920×1080, then area downsampling to 960×544. Use the valid in-bounds warp region for whole-picture values and intersect it with each 32-pixel-bounded source-derived player mask from the fixed Al028 source-only review. Those masks are **unreviewed source-model proposals**, not human labels; their crop border is 32 source pixels and only proposal pixels within it enter the mask, matching the prior regional convention. No ball truth or ball-crop metric is inferred from these proposals. Report each arm's 12-pair whole frame and each separately available player support, the mask-valid pixel denominators and individual pair rows; reject empty masks rather than assume presence. Source self-error must be zero. The two control results are frozen before the joint output and must reproduce exactly when joint is added. Hash and record source, native decodes, source-review and each full output; separately preserve real 25-fps timing, frame count and luma/flow implementation versions. A better or worse outcome changes only the second-match temporal assessment, not the still-frame scores.

CPU only: at most 600 process CPU seconds per stage. A real first two-pair smoke must pass identities, source self-error, mask/tick checks and a 1.5× projection including setup; stop if it cannot finish within the cap. Frozen controls can proceed while joint generation is still running. Outcome-opened, one physical match and 12 correlated pairs; no population uncertainty or reader study claim.

The two-pair smoke used 0.93 process CPU seconds and reproduced exactly as the first two rows of the complete stage. The complete source/ordinary/neural stage used 4.28 CPU seconds and was independently repeated with exact equality of every source, ordinary and neural pair/region row. Its [numeric receipt](controls.json) includes all 12 native ending ticks, 36 region pixel denominators per arm, source/codec hashes and review SHA `508617ef01170813202eec89a154b0b61890756427a143ac2f2b0e429b00f419`. Mean absolute source-flow residual error in luma levels is:

| Display | Whole picture | Model-proposal track 0 | Model-proposal track 1 |
|---|---:|---:|---:|
| Source | 0 | 0 | 0 |
| Ordinary AV1, 20.3608 kbps | 0.3083 | 16.8502 | 17.2460 |
| DCVC-UF, 20.4117 kbps | 0.4843 | 14.9512 | 15.4638 |

Mean absolute source change energies on the same supports are 0.2688 / 16.5964 / 16.6597, ordinary AV1 0.1227 / 8.3256 / 3.4905, and DCVC-UF 0.3990 / 9.2618 / 8.5869. Thus in this sample neural has lower regional residual error while ordinary has lower whole-frame error. These are source-model proposal supports, 12 correlated adjacent pairs, and no human temporal preference result. The joint arm remains pending the independently authenticated receiver.

The genuine common-budget joint sequence passed its all-240-frame RGB digest (`43164345ecc74ea5c2dbb5644e1c0aa2a75bef32c4ce5c344dbab4997f856421`) and the scoring replay reproduced every frozen source/ordinary/neural pair-region row exactly. Its [complete numeric receipt](joint-result.json) adds:

| Display | Whole residual error | Track 0 residual error | Track 1 residual error | Track 0 change energy | Track 1 change energy |
|---|---:|---:|---:|---:|---:|
| Source | 0 | 0 | 0 | 16.596 | 16.660 |
| Ordinary AV1, 20.3608 kbps | 0.308 | 16.850 | 17.246 | 8.326 | 3.490 |
| DCVC-UF, 20.4117 kbps | 0.484 | 14.951 | 15.464 | 9.262 | 8.587 |
| PointStream cached, 20.4075 kbps | 0.341 | 25.289 | 25.436 | 16.127 | 20.088 |

The baselines have lower source-flow residual error on both source-model player supports in this sampled shot. This is a 12-pair temporal diagnostic, not a whole-rally temporal average or human rating. PointStream's rate assumes the previously paid appearance cache; its 157,517-byte initialization is separately charged in a cold start. No ball trajectory truth is inferred. The native source is 25 fps, so each measured pair is 40 ms apart.
