# Persistent prediction: representation, coding and deployment

User approved writing this direction and carrying out the recommended search
and tests on 2026-10-28. Branch: codex/persistent-prediction-search.

Goal: find a technically useful route to foreground AND background savings,
then test whole-frame delivery against fair SVT-AV1 and a reproducible neural
anchor. Familiar Al development footage cannot establish untouched confirmation.

Three axes, to remain available to future sessions:

1. Representation: synchronized decoded appearance memory for foreground;
   persistent background references plus short-term decoded context.
2. Coding: model outputs are predictions, not mandatory replacements. Pay for
   corrections; ultimately investigate conditional feature coding rather than
   assuming ordinary video coding of shifted pixel residuals is efficient.
3. Deployment: installed generic decoder, persistent venue cache, assets learned
   from already decoded video, compact model deltas, decoder-only adaptation.
   Report installation size, cold/warm rates, finite amortization, refresh and
   random-access costs. No free target-dependent masks, poses or references.
   Multicast savings do not divide an individual receiver's download cost.

Literature motivating hypotheses (not evidence of gains on PointStream):
- Context modulation, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.html
- Conditional masked feature residual coding: https://pmc.ncbi.nlm.nih.gov/articles/PMC12299318/
- Foreground/background libraries: https://doi.org/10.1016/j.jvcir.2019.03.009
- Codec-aware enhancement, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_Plug-and-Play_Versatile_Compressed_Video_Enhancement_CVPR_2025_paper.html
- Instance-adaptive video compression (compressed decoder updates): https://arxiv.org/abs/2111.10302
- Efficient adapters, ACCV 2024: https://openaccess.thecvf.com/content/ACCV2024/html/Oh_Parameter-Efficient_Instance-Adaptive_Neural_Video_Compression_ACCV_2024_paper.html

Budget: a NEW upper ceiling of 3600 GPU-seconds and 3600 CPU-seconds,
including every request, failed gate, validation and overhead. Not an extension
of either historical closed study. One initial GPU request at most 3000 seconds
(smoke450, full2100, validator100, overhead350); 600 GPU-seconds remain reserve,
not automatic retry authority. No training or model download in the first pilot.

Initial finite pilot: one native Al source prefix, 96 frames at25fps, score
every third frame (32 ticks). Smoke24frames/3ticks. Three residual CRFs43/53/63;
three native SVT anchor CRFs43/53/63 at the same960x544 coded geometry; all scored
at1080p. Source remains the complete frozen240-frame original for identity.
One scale4 decoded baseCRF63, one paid first-frame referenceCRF35. Compare
block-selected current/previous decoded base + warped long-term reference,
with and without player-aligned previous decoded-base patches. Pay the complete
base, reference, per-frame homographies, player boxes, selector and residual IVF.
The receiver sees no original masks or pixels. Memory is decoded history, not
the source. Fixed32x32 luma blocks; chroma uses the same even boundaries.

This small native pixel-residual implementation is a measured feasibility
pilot, not an upper bound on conditional neural coding. A failed residual
recipe cannot close the broader research direction. Zero correction-payload
prices on its measured reconstructions are finite rate relaxations only.
Generic restoration compatibility is audited separately; original PnP-VCVE
checkpoints expect H.264 metadata, so AV1 gains cannot be assumed.

Promotion: correctness requires repeated native decodes, wire-only replay,
source/weight/tool hashes, finite metrics, original-source RGB controls and
memory limits. Smoke must project within the absolute full allowance with35%
safety; timing variation is secondary. Smoke metrics are not accepted results.
Report foreground, background and whole-frame LPIPS/DISTS, source-mask
provenance, complete native bytes and BD-rate only on overlapping quality.
No overlap means unresolved coverage, not a manufactured win. An exploratory
SVT win requires improvement robust to +/-0.001 in both whole-frame distances
at a bracketed total rate, without a robust regional regression; require new
confirmation and a reproducible neural comparator before a paper claim.

Stop: decision reached, hard cap, failed correctness/cost gate, missing input
or unsupported model path. Inspect uncertain requests; never replay them.
No further training or larger experiment follows automatically from a pass.
Delegates: gpt-6-luna/max, precise launch-and-report instructions only.
