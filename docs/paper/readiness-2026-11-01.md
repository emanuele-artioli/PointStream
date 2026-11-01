# Manuscript readiness, 2026-11-01

The decoder witness is **one gate, not the complete submission checklist**.
The full [task audit](task-audit-2026-11-01.md) and
[claim-to-evidence matrix](evidence-matrix-2026-11-01.md) distinguish completed
studies, remaining scientific gaps, optional improvements and deferred roadmap
items. The author-approved empirical framing can report positive and negative
findings; it does not require every historical PLAN item to succeed. It also
does not turn unmeasured comparisons into completed evidence.

## Completed in this phase

- Updated from main including merged PR #317, and delegated three independent
  Terra 6.1 medium tasks: full task/evidence audit, negative-evidence manuscript,
  and W8 payload/oracle helpers. Requested model/effort are known; actual runtime
  telemetry and the other Codex tier-validation gaps remain unverified.
- Added an online appendix with bounded hand-mesh and object-motion/view-bank
  negatives and egocentric equal-adaptation results. The LPIPS loss remains
  beside DISTS/PSNR gains; post hoc selections and failed decodes are disclosed.
  A cost-stopped smoothing pilot and absent racket masks are not presented as
  quality negatives. See [negative-evidence audit](negative-evidence-2026-11-01.md).
- Overleaf main `7a7381e31526c304af16da79dfd96941c0d4a6e9` includes that appendix, the
  hand-transfer/reference-bank negatives and the passing conditional W8
  saved-quality screen described below.
  PDF `/private/tmp/pointstream-w7-manuscript/build/main.pdf`: **25 pages,
  body plus references 1–22, Online Appendix 23–25**. This meets the user's and
  TOMM guide's allowance of **23 + 5 additional appendix pages**. The empirical
  checker passes; modified results and appendix pages were visually reviewed;
  no overfull/undefined-reference warning remains. Venue-specific review format
  still needs final confirmation against the live filing instructions.
- Assembled a real, self-delimiting W8 candidate and independently verified its
  byte identities, rate accounting, exact original IVF identity and saved-anchor
  arithmetic. The local checkpoint diagnostic matches all 480 tensors byte for
  byte, including the added tensor directory and sign information. A frozen
  implementation reproduction at `8f6d0ece` matches both complete wire objects.

## Current evidence and resources

The [W8 rate screen](../experiments/w8-inputs/exclusive-20261101/rate-screen.json)
now **passes all eight cells** after independently audited exclusive request
`20261101T081828Z-d70bc44f`. The fixed 2,289,836-byte candidate costs 3,816.4 kbps
cold for 4.8 s; 46.3151 kbps assumes exact named reference/match objects reused
for 7,200 s, with reuse quality unproved. The additional 49.6733 kbps ROI-SVT
point closes the original 45.8233 kbps gap. All four families bracket the rate,
without extrapolation, and saved hybrid player quality favors both metrics.
This optimistic one-window saved-quality screen is not fresh output quality,
complete receiver qualification or a change to the W7 population result.

All three scoring jobs are terminal, process groups/claims released, waits
ended and completion events acknowledged. Inclusive scoring charge is 224 GPU-s
and ≤304 CPU-s of the approved 450/450 internal ceilings, inside original 3,600/3,600.
The initial two cost stops and initial audit's positional-name failure remain
recorded; reviewed auditor `142b6cc5` resolved argument binding without changing
scores, identities or thresholds. The full audit passed all 12 checks on 80 ticks;
maximum original-control difference 5.2273e-5 <0.001.

Original W8 cumulative after a separately bounded 15 CPU-s runtime preflight:
**224/3,600 GPU-s, ≤1,169/3,600 CPU-s**. MAIN preparation runtime and network namespace
feasibility were verified on gpu5. Exact packet geometry on the Mac previously
failed at ~4.55e-13 despite matching pose pixels; no tolerance waiver. Linux
Landlock, packet-derived inputs and independent full validation are not yet
qualified. The corrected complete-packet CPU oracle is under review; the older
transport-based draft is disabled. No CPU input-oracle fleet job or receiver job exists yet.
Unused generator/SAM/native installed-model qualifications remain a separate
pre-GPU gate, with unchanged original budgets. No training, encoding or model
changes follow from the scoring repair.

## What remains for submission

1. **Claim and evidence:** retain W7's inconclusive unseen-match interval at
   two hours, original Med-family omissions, correlated-window uncertainty,
   whole-frame losses, and W9's conditional measured-duration range. Neither
   the modeled rate nor the assembled bytes establish a complete codec.
2. **Fresh reconstruction:** only the authorized witness's specified inputs,
   payload, installation/cache contract, strict equality or prospective numerical
   gates, and final offline output validation can establish its narrow claim.
   Full-rate, sustained-reference and live claims require separate evidence.
3. **Other current holes:** matched ablations, a qualified generative peer,
   historical benchmark/source recertification, independently labelled all-frame
   egocentric tasks, and clean operational timing must be resolved for claims
   that rely on them or remain explicit limitations. The evidence matrix names
   the exact artifacts and gaps; this is not a claim that all are mandatory
   experiments for an honestly narrowed empirical study.
4. **Final publication review:** reconcile abstract/conclusions and all active
   markers with the chosen contribution, audit references/rights/provenance,
   verify format and final PDF, and obtain explicit authority to submit.

W6b smoothing, W9 live adaptation, W10 new generators/larger training, W11 racket
tracking/rendering, historical broad final sweeps and tooling migration remain
deferred or separately gated. Existing closed budgets do not authorize reopening
them. The paper remains **a qualified working draft, not submission-ready**.
