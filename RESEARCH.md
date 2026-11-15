# PointStream: current research direction

Updated November 15, 2026. Start here for the scientific goal and evidence;
[PLAN](PLAN.md) gives the next decisions. Dated experiment records retain their
original protocols, failures and permissions. A closed allocation or failed
fixed recipe does not close every version of the research question.

## Main hypothesis

At low tennis bitrates, a native base plus reconstructed players and separately
coded equipment/ball can improve foreground rate–quality beyond ordinary coding
and conventional ROI allocation. Test **players, rackets and ball together**;
report foreground/background/full-frame quality, temporal behavior, object
failures and all transmitted costs. Background quality may be traded for
foreground quality. A full-frame win or real-time operation is a stronger,
separate claim.

The W7/W9 player evidence and paid ball evidence make this a credible lead
contribution. They are not yet a measured combined-codec win. Tennis is now
the critical path. Preserve egocentric positives/negatives and report its own
supported level; completing its dense annotation is not a prerequisite to
testing the tennis hypothesis.

## Evidence we should build on

| Route | What survives audit | Boundary |
|---|---|---|
| Generative players | W7 al028 at corrected modeled19.24kbps: player LPIPS/DISTS0.304/0.242 versus interpolated best DCVC0.384/0.303. Later native-IVF envelopes include both SVT and DCVC. | Tier1's six-window positive is conditional on available anchors: med000 DCVC is missing, the Medvedev subgroup is inconclusive, and six windows are two matches. Whole-frame quality loses; costs assume reference/adaptation reuse. Unseen-match evidence remains inconclusive under its original two-metric rule. |
| Tracked ball pixels | Three-source confirmation: equal-source crop-DISTS BD-rate−26.28%; complete ball packets and fresh sender/control evidence. | Full-DISTS+17.39% and full-LPIPS+24.64% costs; sparse visible point crops, encoder overlap and three-source uncertainty. Not racket or whole-foreground evidence. |
| Rackets | Need an explicit fidelity endpoint and independently aligned labels. | Correct rackets cannot be inferred from sharp players. Prior source-mask proxy was poor and is not annotated racket recall. |
| Static additive256px patch | Actual two-domain screen loses weighted DISTS17.10%/18.47%, even with free placement. Conventional ROI saves8.53%/5.39% on the development sources. | Stops that fixed design; not a rejection of generative players or all patch designs. One source/domain cannot establish population inference. |
| Complete decoder | Native/tracked-pixel endpoints and208-frame source-denied filesystem reconstruction pass. | That proof does not cover learned-player generation. Fresh source-independent generative reconstruction and a combined packet remain unfinished. |
| Temporal quality | Full-rate player pose has a positive diagnostic; source of excess flicker was traced to generated pixels. | Tested full-rate generation and two smoothing profiles fail the historical temporal rule. A slowed/sampled website preview cannot resolve this. |

Primary sources: [player native-IVF correction](docs/experiments/w7-anchor-rate-repair.md),
[reference billing and duration](docs/experiments/w9-adapter-break-even.md),
[Chapter14 identity](docs/experiments/submission-ladder-chapter14-demo-audit-20261115.md),
[ball confirmation](docs/experiments/submission-ladder-ball-confirmation-analysis-20261115.json),
[model history](docs/experiments/submission-player-model-history-audit-20261115.md),
[combined gaps](docs/experiments/submission-combined-foreground-gap-audit-20261115.md).

## Model and adaptation choices

Do not restart the entire model ladder. Generic AnimateAnyone, own-window
overfit, other-shot match adaptation and a common tennis base have already
been tested under different reference/matte/accounting conditions. Generic
weights won some development cases, but have not established the desired
general end-to-end result. The stronger saved recipe uses a common tennis
LoRA plus a small match adapter and pose-guider changes. Per-player fitting,
newer generators and live first-seconds adaptation are unrun extensions.

Use the strongest existing frozen recipe to establish whether the complete
construction works. Include generic and tennis-base-only alternatives only
where a small declared comparison answers the adaptation question; do not
retrain every alternative or select a new model before solving receiver,
temporal and cost limits of the measured positive.

The48.9MB common tennis LoRA was prospectively designed as a shipped decoder
resource. It can be declared installed and frozen, with native/neural decoder
resources treated consistently; report footprint and cold initialization
separately. This does not erase its training overlap. A model fitted or chosen
specifically for the test session, match adapters, guider updates, appearance
banks, refreshes and metadata remain content-specific. Keep historical
base-charged deployment sensitivities beside the installed-resource profile.

A causal warm-up system is a possible later scope: encode the initial frames
conventionally while adaptation uses only arrived data, then switch after
fitting, transmission and loading finish. Measure all warm-up frames/bytes,
training delay, updates and lookahead. W7's disjoint fit/evaluation clips do
not already demonstrate that system. No new training is authorized by this brief.

## Boundaries that keep the result defensible

- Compare against effective SVT-AV1 ROI and DCVC-UF under the same sources,
  cadence, color conversion and paid-state contract. Give a conventional
  anchor the same generic enhancement when that tests the competing explanation.
- Preserve every frame, unknown/absent object, decode failure and full-frame
  cost. Use conventional fallback where semantic reconstruction is uncertain;
  measure its bytes and failure frequency. Claim racket improvement only when
  an explicit racket endpoint supports it.
- Use70/30 foreground/background DISTS as the declared weighted objective,
  LPIPS corroboration, and separate player/racket/ball/temporal reports.
  Historical objectives stay separately labeled. Do not add player and ball
  BD-rates or infer the combined rate–quality curve from separate experiments.
- No10% usefulness floor for future studies. Report source-level effects,
  confidence intervals and justified tests, with practical value and novelty
  separate. Windows/frames from one physical match are not independent matches.
  Existing opened cohorts cannot become confirmation for a changed candidate.
- Show representative randomized, rate-labeled human pilots before crucial
  perceptual decisions. User development judgments do not establish population
  preference; source-only reference annotation must remain blind to codec outputs.

## What not to repeat

Do not repeat the stopped exact-mask/static-patch/ideal-background grids,
fixed smoothing weights, unqualified runtime retries or full model sweeps just
to obtain a favorable sign. Reuse valid archived qualifications and scores.
An oracle gates only constructions it demonstrably dominates; source-pixel
pasting is not a universal LPIPS/DISTS bound. A failed whole-frame objective
does not erase a scoped foreground trade-off.

## Audit and submission state

The [merged-PR index](docs/experiments/submission-merged-pr-index-20261115.json)
covers423 merged PRs through428: all title/body metadata indexed, with targeted
scientific documentation/correction review. This was not revalidation of every
historical diff or experiment. Older apparent wins, including pre-reset GateB,
must follow their later invalidations; see the [documentation audit](docs/paper/submission-documentation-audit-2026-11-15.md).

The current paper is V4,18body/reference+5appendix pages. Expand supported
methods, player positives, controls and negatives toward the verified journal
allowance; the smaller draft is not a desired final length. Research cutoff is
December19; submission preparation target December26,2026. Author/rights and
current venue-format checks remain. No submission is authorized. The previous
automatic compute window expired; new stages need concrete released budgets.
