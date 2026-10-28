# Single-stream search: aligned restoration, native diagnosis, latent audit

User approved this batch on2026-10-28 after its concrete recommendation.
Goal: find an improvement to foreground/background/whole-frame quality without
duplicated base/reference/residual streams; preserve honest negative results.
Branch codex/single-stream-search. New ceiling3600GPU-s/3600CPU-s including
failed stages, validators and overhead. Old studies remain closed.

1. Restoration cap1800GPU-s. Reuse immutable native anchor43/53/63 streams from
   completed memory pilot2933ca5bcb82f52b832e20c21db7737acf72103d6905b1d816baacc992fa6554.
   Same96frames/32ticks on familiar Al at25fps, original source RGB controls.
   Decoder-only DIS optical flow at native960x544; align previous/next decoded
   frames; forward/backward consistency<=1.5pixels, luma difference<=12,
   reject outside pixels. Blend strengths0.15/0.35/0.60, frozen before outcomes.
   No source pixels/masks at restoration input; reference masks score only.
   Complete original IVF+8-byte filter header paid. One-frame lookahead, no
   real-time claim. Compare both whole metrics and both regions, report temporal
   luma residual distortion beside fidelity (no full experience-track claim).
   Smoke24-frame prefix/3ticks; full96/32. Cost projection35% safety,
   fixed input decode/import once. No warm-stability veto.
2. Native diagnosis cap600GPU-s, no quality curve or old-stream decoding.
   Re-encode failing QP21/960x544/nativeYUV, HT-L in each stage. Compare legacy
   custom CUDA stream with explicitly selected default stream. Hypothesis:
   extension kernels and producer operations may use inconsistent streams.
   Not a proven cause; shared-GPU contention is recorded, not manipulated.
   Smoke4frames, full24frames; each variant independent encode/decode process,
   two decode passes plus another fresh decoder. Full promotion requires the
   default-stream variant pass byte counts, two-pass determinism, independent
   decoder equality, first-frame/encoder equality, memory and budget checks.
   Legacy failure is a recorded diagnostic outcome, not missing evidence.
   Keep existing production default until validation; no retry for failure.
3. Conditional/latent feasibility CPU cap1200seconds (inside3600CPU-s).
   Audit exact installed DCVC sources and pin primary public source revisions
   for DCVC-DT/DCMVC. Check a differentiable path, native entropy round-trip,
   checkpoint/runtime compatibility and update delivery requirements. No
   pretrained checkpoint or environment download, no training by proxy.
   Pin file hashes and relevant source excerpts; distinguish missing API from
   proof of impossibility. Remaining CPU allowance is analysis/preservation.
4. Latent refinement cap1200GPU-s ONLY if task2 and task3 both establish a real
   compatible frozen-decoder differentiation/entropy path. Three bounded
   optimization settings; actual native stream bytes and decoder identity.
   If either prerequisite fails, record BLOCKED at that gate and spend0.
   No transplant of DCVC-DC code into UF without validation, no weaker anchor
   substituted to manufacture a win. Broad neural integration remains future work.

Promotion: representative real-data smoke, substantive byte/device/finite-metric
checks and absolute resource projection inside full allowance. No oracle
assumption that temporal fusion improves perceptual metrics: the real cheap
filter is the initial test, gating learned restoration investment. Its failure
does not bound a learned aligned model. Stop once finite decision is reached.
Report PCHIP BD-rate only on overlapping quality; no overlap is unresolved.
Exploratory SVT win needs both whole distances robust by0.001 at bracketed rate,
with regional regressions disclosed, plus matched enhanced-baseline controls.
No complete baseline/TOMM claim without fair expanded anchors and confirmation.

Deployment: algorithm/profile can be installed; pay actual filter-header bytes.
Any future generic model must transfer across content. No free target poses,
masks, caches or future frames beyond the declared one-frame lookahead.
Native waits<=45min, attended unless completion wakeup verified; no recurring
model monitor. Preserve IDs before waiting; inspect uncertain requests, never
resubmit. All source data and completed outputs remain read-only.

Execution clarification after exact native API review: initial diagnosis's
legacy4-frame arm passed, but the default-stream alternative is incompatible
with CUDA graph capture and failed BEFORE encoding. It is not a quality result
or a determinism verdict. Charge that request250GPU-s conservatively and use
at most350GPU-s to complete the unaffected original24-frame legacy reproduction
with a fresh4-frame correctness smoke. Same source/QP/geometry/checkpoints,
no new alternative, no further request after this completion. This finishes
the user-approved diagnosis inside its original600GPU-s, not a retry of a
failed scientific configuration or extra research allocation. The unsupported
default mode must never become the production default.

Primary literature: https://arxiv.org/abs/2605.13476 (DCVC-DT),
https://github.com/SunnyMass/DCVC-DT, https://github.com/Austin4USTC/DCMVC,
https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_Plug-and-Play_Versatile_Compressed_Video_Enhancement_CVPR_2025_paper.html.
