# Experiments: prospective submission-ladder inference policy

### 2026-11-15 — user-requested policy amendment, before new outcomes

The user requested replacing the arbitrary 10% usefulness gate with inference
that preserves smaller effects. This note records recommendations for the next
protocol revision and cohort. It does not preregister opened outcomes, change
the frozen historical protocol, authorize compute, or alter saved evidence.
No new outcomes were generated for this review.

Remove the 10% floor from future scientific benefit and development-promotion
rules. Report three separate things: the estimated rate saving, its uncertainty
and inferential assumptions, and its practical value alongside complexity,
latency, implementation cost and failure conditions. A smaller effect can be
valuable; a small p-value alone establishes neither magnitude nor usefulness.
An imprecise estimate is unresolved, not evidence of no benefit. Resource-based
stopping remains a budget decision rather than a scientific usefulness cutoff.

## What the existing evidence supports

The two-domain static-patch pilot uses one exposed 64-frame development source
per domain. Conventional ROI saves 8.53% weighted DISTS in tennis and 5.39% in
egocentric video; the fixed additive semantic patch costs 17.10% and 18.47%.
The latter also loses with free placement/framing. Those are measured,
source-specific effects under the registered operating points. One source per
domain provides no estimate of between-source sampling uncertainty. Frames,
rate settings, crops and resampling the same source do not supply independent
sources. Neither the ROI savings nor the patch losses establish domain-wide
effects. The historical 10% gate remains part of the historical record; it is
not evidence that the smaller ROI savings are scientifically useless.

The tennis ball confirmation has three physical matches, not 282 independent
frames or 32 independent visible-point crops. Its saved -26.28% aggregate is
the backtransform of the equal-source mean log rate ratio, as implemented in
`experiments/tennis/ball_confirmation_analysis.py`; it is not an arithmetic
average of percentage BD-rates. The saved percentile interval is explicitly
descriptive. Resampling three observed matches cannot manufacture population
coverage or reliably characterize tails of between-match heterogeneity.

For illustration only, with three nonzero independent source effects, all in
one direction, an exact sign test has one-sided p=1/8 and two-sided p=1/4
under its equal-sign-probability null. A paired sign-flip test has only eight
assignments and the same minimum p-values for a conventional symmetric mean
statistic, subject to its additional exchangeability/symmetry assumptions.
Neither is an originally registered test or a new formal analysis of this
cohort. Six independent pairs are needed even to permit a two-sided exact
sign-flip p below .05 without multiplicity correction; six is a discreteness
bound, not an adequately powered cohort recommendation. Parametric inference
with three sources would depend heavily on unverified distribution assumptions.
Retain the narrow, observed three-match finding and its full-frame costs; do
not promote it to robust population significance or whole-foreground benefit.

## Prospective source-paired design

1. Freeze candidate code, settings, masks/reference tiers, endpoints, source
   eligibility and rate/quality support before opening new codec outcomes.
   Existing opened windows remain development or historical confirmation;
   changing the method requires a new confirmation cohort. Record prior
   exposure and encoder training overlap separately from outcome blindness.
2. Sample independent physical matches for tennis and participant groups for
   egocentric video. Multiple windows from one match or participant stay in
   the same cluster. Mechanically choose windows within sources and aggregate
   them by a rule frozen before outcomes, then give independent sources equal
   weight. Keep domains separate and state the sampled population.
3. Encode each method and each anchor on the same sources. Compare complete
   paid packets with effective conventional ROI and a matched neural anchor;
   a claim about a separate enhancement should also face the relevant matched
   enhancement/super-resolution control. Preserve all registered sources,
   frames and failures. No common-quality interval means no BD-rate for that
   comparison; report the registered ordinal result and support coverage, not
   an extrapolated number or silent source exclusion.
4. Use per-source integrated log rate differences on frozen common-quality
   support as paired effects; backtransform their equal-source mean for the
   headline percentage. Predefine the primary metric and comparison for each
   claimed level. Report regional, full-frame, temporal and task outcomes
   alongside it. Compare enhancements directly with ROI rather than subtracting
   two independently averaged percentages. Constructed anchor envelopes remain
   labelled constructed unless physical switching is demonstrated.
5. Specify the inferential target and assumptions before analysis. A source
   mean interval and test address mean benefit; an exact sign test addresses
   source-direction probability/median under its null, not mean magnitude.
   Source bootstrap intervals require enough independent clusters and a
   justified sampling design. Sign flipping is not automatically exact merely
   because both deterministic codecs were applied to the same source: its
   null exchangeability or symmetry must be justified. Swapping codec labels
   must respect the chosen log-rate estimand and common support.
6. A conventional proposed confirmatory family uses two-sided alpha .05,
   intervals, and Holm correction for the two domain-primary claims if either
   may be selected for success. Freeze any level hierarchy and additional
   confirmatory comparisons in advance; report other endpoints as secondary
   with their multiplicity status. Alpha controls a stated error rate and is
   not a usefulness threshold. Preserve small estimates and inconclusive
   results, regardless of significance labels.
7. Before allocation, simulate prospective power and interval precision at
   several effect sizes, including effects below 10%, using source-level
   heterogeneity and missing-support scenarios. Existing development sources
   can inform conservative variance scenarios, but n=1/domain or n=3 cannot
   establish a reliable single variance estimate. Choose and freeze cohort
   size against a declared precision/power target, annotation effort and
   compute budget. Report detectable-effect curves if the budget cannot reach
   the desired precision. Do not choose N from post hoc observed power or add
   sources until a desired p-value appears. Any sequential design needs its
   stopping/error-control rule frozen first.

## Actual ROI and additive-patch construction

`experiments/codec_ladder/__main__.py:201` writes a source-mask-driven ROI map;
the qualified SVT convention is zero offset inside touched 64-pixel blocks and
+20 outside. It encodes one full-frame AV1 stream, reallocating bits through
native quantization. There is no separately coded foreground texture stream
or patch paste at the receiver.

In the static additive arm (`__main__.py:218`), one 256-by-256 rectangle is
selected for all frames by summed foreground occupancy, or by summed native
luma reconstruction error for the competing selector. Its source pixels are
encoded separately at fixed CRF28. The base is the ordinary full-frame native
encode at CRF35/50/65 and still encodes the rectangle. The PLP1 packet pays
both streams and metadata. `experiments/codec_ladder/transport.py:42` replaces
the decoded base rectangle with the decoded patch planes. “Additive” describes
additional transmitted representation/bits, not addition of pixel values or
a residual codec. The proposed blacked-base experiment changes this duplicate
coding assumption; the completed pilot does not test that proposal. The later
tracked 32-pixel PTP1 ball patch is a separate construction and claim.

## Sources and provenance

Historical evidence: `submission-ladder-protocol-20261114.json`,
`submission-ladder-20261114.md` (two-domain pilot entries), and
`submission-ladder-ball-confirmation-analysis-20261115.json`. These remain
unchanged. The next released protocol/card must make the prospective amendment
explicit before new outcomes; this note alone does not release a run.

Statistical primary sources consulted:

- [ASA statement on p-values](https://doi.org/10.1080/00031305.2016.1154108):
  significance, effect size and importance are distinct.
- [NIST paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm):
  inference operates on naturally paired differences and its assumptions.
- [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html):
  paired sample-label permutations enumerate 2^n assignments; validity still
  depends on the null permutation model.
