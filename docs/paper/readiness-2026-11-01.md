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
- Overleaf main `e366440` includes that appendix and the unresolved W8 accounting
  screen. PDF `/private/tmp/pointstream-w7-manuscript/build/main.pdf`: **25 pages,
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

## Scientific stop and resources

The [W8 outcome](../experiments/w8-wire-audit.md) is **unresolved**, not a quality
loss. The fixed 2,289,836-byte candidate costs 3,816.4 kbps cold for 4.8 s. Its
conditional rate with exact named reference/match objects reused for 7,200 s is
46.3151 kbps; one required ROI-SVT family ends at 45.8233 kbps. No extrapolation,
new fleet request or GPU work followed. The other three saved-quality family
comparisons are positive on both metrics. Reuse quality is unproved.

W8: **0 / 3,600 GPU-s; at most 850 / 3,600 CPU-s** under a deliberately
conservative envelope. No experimental reservation remains. A separate packet
input diagnostic fails exact geometry on the Mac at about 4.55e-13, despite
matching pose-map pixels. The original packed runtime and Linux read isolation
remain untested. A paid judged-frame selection was added because registration
and generated-output sampling differ. None establishes fresh output quality.

The preserved unsubmitted CPU qualification/scoring drafts cannot be submitted
under the stopped card. An existing upper ROI-SVT bitstream is now independently
verified at 49.6733 kbps, but is unscored. The pending prospective amendment asks
to score that one point and the original raw controls with at most 300 GPU-s and
300 CPU-s inside the existing ceilings; only a passing coverage screen could
reopen the original input/isolation/receiver gates. No training, encoding or
model changes follow. No amendment is approved while the question is pending.

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
