# PointStream: current research direction

Updated November 17, 2026. Start here for the scientific goal and evidence;
[PLAN](PLAN.md) gives the next decisions. Dated experiment records retain their
original protocols, failures and permissions. A closed allocation or failed
fixed recipe does not close every version of the research question.

While campus access is unavailable, prioritize manuscript integration and
analysis of saved results under the [outage writing plan](docs/paper/outage-writing-plan-2026-11-17.md).
The latest V4 source/PDF are local. This changes the work order, preserves the
stronger player-and-ball hypothesis and releases no cloud spending or new runs.

## Main hypothesis

At low tennis bitrates, a native base plus reconstructed players and a paid
ball channel can improve **player-and-ball** rate–quality beyond ordinary coding
and conventional ROI allocation. The prospective primary foreground is visible
player body/clothing plus ball; rackets are a separately reported optional
extension. Excluded equipment remains in complementary/full-frame scoring and
visual review. Do not imply all-equipment fidelity from this narrower claim.
Report foreground/background/full-frame quality, temporal behavior, failures
and all transmitted costs. A complete decoder still reconstructs every frame;
it need not improve every object. Full-frame superiority and real time are
stronger, separate claims.

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
| Rackets | Optional extension after the player-and-ball result; existing5-landmark labels can guide an explicit equipment test. | Racket improvement is unestablished, not a gate for the narrower primary claim. Prior source-mask proxy is not annotated racket recall; baseline weakness does not establish harmless omission. |
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

Historical model comparisons are prioritization leads, not authoritative gates
against generic models. The [actual-code re-audit](docs/experiments/foreground-scope-and-requalification-20261115.md)
found unsupported-rate, profile-coverage and fractional-FPS defects and reviewed
conditioning assumptions. A hash-verified saved-report check finds no endpoint
label changes across72 generic/adaptation cells; that bug does not explain the
reported generic50%. Profile/coverage/resource joins and upstream preprocessing
still need validation. Audit positive and negative recipes before a consequential
decision; do not invent a corrected win share or invalidate all W7 quality.

The [November16 implementation](docs/experiments/player-requalification-20261116.md)
now authenticates the selected W8 source, conditioning and checkpoint bytes,
joins12 saved adaptation receipts, and checks actual processor tensors on CPU.
It fixes a pre-CLIP reference resize mismatch for future fitting; existing weights
and results remain unchanged. The subsequently approved **three-profile replay
passed**: both players,24native ticks/profile, exact common conditioning, and
bit-exact within-job smoke/full repeats. Conservative charges total1,203GPU/
3,003CPUseconds; all jobs/claims are closed. The source-aligned blind human
pilot passed exact reference/source pixel checks and native/slowed viewing.
The [user review](docs/experiments/player-replay-20261116/human-review-import.json)
is now imported: Player 1 uncertain (common tennis and match adapted similar),
Player 2 prefers common tennis; stock has reported over-sharpening and a
torso-facing hallucination. **Common tennis is the provisional next candidate**;
this is one author's development judgment on one match, not population preference
or equivalence of the adapted profiles. The short pilot establishes neither
longer temporal behavior, complete source-denied learned decoding nor a combined
codec win. W8's later
isolated1/3step execution also passed; do not restart from its older CUDA failure.

The stock failure is conditional on PointStream's paid body-only pose maps.
The retained upstream DWPose drawer also renders dense face/hand cues and uses
a different detector/visibility convention. Exact packet/tensor parity does
not validate those cues against the source. Actual Player2 head/left-right
accuracy and the cause of the failure remain unresolved; do not conclude that
stock weights are intrinsically inferior. Fresh-pilot LPIPS/DISTS have not
been computed, so metric disagreement is also unestablished.

The [whole-run motion pilot](docs/experiments/player-motion-20261116.md) now extends this check:
shared tennis, one continuous 240-frame call per player, with a real 44-frame
smoke and a cost gate. The approved November17 Player2 smoke passes independent
correctness checks, but its1,704.68 s full forecast exceeded the1,500 s cap;
neither full run started. The [closed release](docs/experiments/player-motion-20261117/release.json)
charges328GPU/1,828CPUseconds. A [prepared amendment](docs/experiments/player-motion-amendment-20261117.md)
raises the explicit full cap to1,800 s/request cap2,400 s, with conditional
sequential admission inside the same70-GPU-minute/2-CPU-hour envelope. The user
explicitly approved this amendment; the [amended live ledger](docs/experiments/player-motion-amendment-20261117/release.json)
counts the prior charge. Amended Player2 job `20261117T094919Z-5a0cc722`
completed44/240 frames and passed independent integrity validation. Its closure
left2,531GPU/3,281CPUseconds, covering the conditional Player1 reservation;
Player1 job `20261117T102252Z-ccb04da2` also completed44/240 frames with
independent integrity PASS. Total conservative charges, including the original
stop, are3,318GPU/6,318CPUseconds within4,200/7,200; both jobs and claims are closed.
The user's
[full-run Player2 review](docs/experiments/player-motion-amendment-20261117/player-2-human-review.json)
is **acceptable** for body/clothing, facing, pose and stability; the
[Player1 reply](docs/experiments/player-motion-amendment-20261117/player-1-human-review.json)
is also **acceptable**. This is author development judgment, not matched-codec
preference or population evidence. This still tests motion on
the same opened source; the Al036 development
source still needs source/frame/PTS and conditioning authentication before a
separate source-spread test. Preserve stock failures and historical controls;
do not repeat the three-profile grid without a new hypothesis. Per-player
fitting, newer generators and live adaptation remain conditional extensions.

The48.9MB common tennis LoRA was prospectively designed as a shipped decoder
resource. It can be declared installed and frozen, with native/neural decoder
resources treated consistently; report footprint and cold initialization
separately. This does not erase its training overlap. A model fitted or chosen
specifically for the test session, match adapters, guider updates, appearance
banks, refreshes and metadata remain content-specific. Keep historical
base-charged deployment sensitivities beside the installed-resource profile.

The [common-player packet contract](docs/experiments/common-player-packet-20261117.md)
now passes CPU tests and exact retained-input checks for both players across
all240ticks. It separates coded480x272 from the1920x1080 coordinate canvas,
charges ongoing KPS2 geometry, and supports native-only empty foreground.
This is not yet a source-denied learned receiver. Its saved lossless references
dominate the391,981B carrier (653.30kbps over4.8s): next resolve cheaper
reference coding or demonstrate actual reuse/refreshes before another expensive
matched-quality study. Do not infer a codec win from exact conditioning alone.
The [approved reference pilot](docs/experiments/player-reference-pilot-20261117.md)
has prepared qualities70/40 at65,120/41,075REF1bytes. Generated quality is
unknown; the pilot uses matched24-tick controls and requires both players to
be acceptable before uniform promotion. The user approved30GPU-minutes/
45CPU-minutes, with quality70 first and quality40 conditional on accepted full
publication and remaining budget. The [release ledger](docs/experiments/player-reference-pilot-20261117/release.json)
records the authority. The user requested local work while campus access is
unavailable: no network retry or GPU job is scheduled. Inspect the unknown
input publication and perform fresh fleet admission only after access returns;
the same budget does not need another approval. The GPU producer remains frozen
at55ba0629 independently of subsequent local packet/viewer changes.
The [local JPEG packet audit](docs/experiments/player-reference-pilot-20261117/packet-audit.json)
now reproduces both prepared reference tensors and all240conditioning ticks
from paid packets. The65,120/41,075B references yield91,283/67,238B total
carriers. These short-window totals include framing and ongoing geometry;
they do not demonstrate generated fidelity or amortized low-rate performance.

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
  measure its bytes and failure frequency. Racket quality is visible in full-frame
  and any independently labelled equipment report; it does not gate the
  explicitly player-and-ball claim.
- Use70/30 player-and-ball/complement DISTS as the new declared weighted
  objective, LPIPS corroboration and separate player/ball/temporal reports.
  Racket-inclusive reports are secondary and require the corresponding labels.
  Historical objectives stay separately labeled. Do not add player and ball
  BD-rates or infer the combined rate–quality curve from separate experiments.
- No10% usefulness floor for future studies. Report source-level effects,
  confidence intervals and justified tests, with practical value and novelty
  separate. Windows/frames from one physical match are not independent matches.
  Existing opened cohorts cannot become confirmation for a changed candidate.
- Show representative randomized, rate-labeled human pilots before crucial
  perceptual decisions. User development judgments do not establish population
  preference; source-only reference annotation must remain blind to codec outputs.
- Report human-observed structural failures, including wrong torso orientation,
  beside perceptual scores. Sharpness and aggregate metrics alone do not establish
  correct anatomy or action. Preserve uncertain judgments rather than counting
  them as wins, losses or ties.

## What not to repeat

Do not repeat stopped grids, smoothing weights, runtime retries or full model
sweeps merely to obtain a favorable sign. A material correctness or conditioning
defect does justify a new declared test; old rejection labels do not block it.
Reuse archived qualifications and scores only after checking their applicability.
An oracle gates only constructions it demonstrably dominates; source-pixel
pasting is not a universal LPIPS/DISTS bound. A failed whole-frame objective
does not erase a scoped foreground trade-off.

## Audit and submission state

The [v4 scope/requalification policy](docs/experiments/submission-tennis-protocol-20261115-v4.json)
supersedes v3's equipment-inclusive primary target before new outcomes. At most
one additional-sport pilot is conditional on usable footage, the common pipeline
and a bounded budget: skating with an actual paid native background first if
prior sources are recoverable; volleyball is a credible alternative. Soccer
adds a higher actor/occlusion/compute burden. No assumed client3D scene or new
background-model research is on this pilot path. Keep an informative egocentric
comparison in the sports-led journal if coherent; a separate egocentric paper
needs its own contribution, not simply a large backlog of work.

The [merged-PR index](docs/experiments/submission-merged-pr-index-20261115.json)
covers423 merged PRs through428: all title/body metadata indexed, with targeted
scientific documentation/correction review. This was not revalidation of every
historical diff or experiment. Older apparent wins, including pre-reset GateB,
must follow their later invalidations; see the [documentation audit](docs/paper/submission-documentation-audit-2026-11-15.md).

The current paper is V4,18body/reference+5appendix pages. Expand supported
methods, player positives, controls and negatives toward the verified journal
allowance; the smaller draft is not a desired final length. Research cutoff is
December19; submission preparation target December26,2026. Author/rights and
current venue-format checks remain; the [November17 public-access recheck](docs/paper/tomm-public-access-recheck-2026-11-17.json)
retains the legacy23+5 planning allowance but cannot verify the live portal.
No submission is authorized. The previous
automatic compute window expired; new stages need concrete released budgets.
