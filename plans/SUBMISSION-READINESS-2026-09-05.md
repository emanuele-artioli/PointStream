# PointStream submission readiness — 5 September 2026

## Verdict

PointStream is not ready to submit as a winning codec paper. The implementation
is ready for the decisive experiment, and the manuscript is within its provisional
page budget, but the central result does not yet exist: no verified PointStream
rate--distortion curve beats both AV1 and VVC.

The project is at the end of engineering preflight and the start of the decisive
evidence search. The hard submission deadline is 30 September.

## Established evidence

- BP52 found no win from background CRF alone. BP53 found no win from half-scale
  background transport on its one diagnostic pair.
- BP56 verified prefix-stable higher-effort libaom coding. On one development
  pair, `good/cpu-used=4/CRF63` strictly dominated the realtime CRF51 PointStream
  control: 377,360 versus 474,369 bytes and VMAF 79.339 versus 77.417. Metric
  controls, byte ledgers and recovery checks passed. This is an internal n=1
  candidate-selection result, not an AV1/VVC win.
- BP57 acquired two fresh provisional tennis sources within authority. Seven
  sampled intervals are visually verified continuous two-player shots.
- Generation has no demonstrated gain over the pasted-reference control and is
  correctly parked while background cost dominates.
- The paper builds reproducibly and was last measured at 22 pages through
  references plus 5 appendix pages, within the project budget.

## Submission blockers

1. **Gate A:** no negative BD-rate or low-rate-boundary dominance against both
   slowest-preset AV1 and VVC under one frozen source/config identity.
2. **Long-context evidence:** no citable 48/96/192/384-frame curve shows enough
   background-cost amortisation to cross either anchor. BP56 full attempts took
   66--72 minutes because scoring dominated, so longer execution needs an
   explicit revised budget and verified progress/checkpoint policy.
3. **Coherent PointStream curve:** current diagnostics isolate background axes;
   a publication curve must move all rate-bearing channels coherently or justify
   the fixed foreground terms.
4. **Timing:** runner wall includes reconstruction and scoring. Semantic encoder,
   receiver decode/reconstruction and evaluation time are not separated for all
   points.
5. **Gate B:** BP57 is two provisional videos, not accepted annotations or six
   independent matches. At least four more independent sources are needed if the
   six-video rule remains.
6. **Gate C:** no frozen-winner core component ablation curves exist.
7. **Gate D:** DCVC-RT and an independent second domain remain unrun; neither
   starts before Gate B.
8. **Hybrid routing:** fallback/accounting work, but an automatic mixed
   eligible/ineligible publication sequence is not validated.
9. **Paper claims:** the abstract, final curves, conclusion and contributions
   remain intentionally incomplete; diagnostic negatives cannot carry them.

## Deadline decisions

### By 10 September: Gate-A candidate or thesis switch

Use development footage for an openly reported search. Seed it with BP56
`good/cpu-used=4/CRF63`; test long eligible contexts and coherent payload
allocation against new same-frame slowest-preset AV1/VVC curves. Include both
continuous-context and segmented controls when the claim depends on boundary
reuse. Pass only with negative BD-rate over meaningful VMAF overlap or strict
low-rate-boundary dominance against both anchors.

If full-frame VMAF has no crossover by 10 September, activate the already
declared fallback thesis before measuring it: salient-object quality at matched
ultra-low rate, with a fixed background-quality floor and full-frame quality
reported alongside it. State that it followed a negative full-frame search.

### 11–14 September: freeze and confirm

Freeze eligibility, rates, duration and metrics before reading confirmation
results. Validate BP57 candidates and reach at least six independent videos, or
explicitly revise that requirement with statistical justification. Report
per-video curves, failures, n, paired differences and standard errors.

### 15–20 September: explain and freeze

Run only the core ablations needed to explain the winner. Add DCVC-RT and one
second-domain check only after confirmation. Close all measurement alarms and
freeze evidence by 20 September; allow correctness repairs, not a new design
campaign, afterward.

## Immediate next session

1. Write one bounded Gate-A brief freezing hashes, context mode, durations,
   coherent PointStream settings, anchor commands, bounds/nulls, timing fields,
   stop conditions and budget.
2. Decide whether the timing split can land without changing coded output;
   instrument before the publication sweep.
3. Write a separate BP57 extraction/annotation/eligibility brief. Ask before
   GPU annotation, more acquisition or experiment execution.
