# Plan

## Current direction — November 17, 2026

Start with [the research brief](RESEARCH.md). The lead hypothesis is a complete
low-bitrate tennis codec combining generated players and a paid ball channel,
beating effective ROI/native/neural controls on player-and-ball rate–quality.
Rackets are an optional, separately reported extension, not a prerequisite for
this explicitly scoped primary benefit. Full-frame quality costs remain explicit. Existing
player and ball positives justify this priority; a combined win is not yet measured.
Egocentric results remain part of the evidence, at their own supported level;
the192-frame egocentric annotation task does not block the tennis path.

The current user direction supersedes older priority/threshold/closure prose.
Historical scientific contracts, outcomes, budgets and failures are preserved
in the [lossless previous PLAN](PLAN-history-2026-11-15.md). Current stages must
still satisfy the [scientific protocol](docs/experiments.md), [prospective v2
inference policy](docs/experiments/submission-ladder-protocol-20261115-v2.json)
and [fleet rules](docs/fleet.md). The [v4 scope and requalification amendment](docs/experiments/submission-tennis-protocol-20261115-v4.json)
narrows the prospective primary target and requires contemporary recipe validation;
stage inputs and budgets still need freezing/release. No10% scientific usefulness floor applies to
future studies; effects, source-level uncertainty, practical value and novelty
are distinct. Opened cohorts cannot become confirmation for a changed method.

## Completed audit and evidence

- [423 merged PRs indexed](docs/experiments/submission-merged-pr-index-20261115.json),
  with targeted scientific correction/history review, not revalidation of every
  diff or raw experiment. The [model audit](docs/experiments/submission-player-model-history-audit-20261115.md)
  distinguishes generic, sport, match and own-window fits, their controls and
  failures. The subsequent [source-code re-audit](docs/experiments/foreground-scope-and-requalification-20261115.md)
  found comparison/coverage/rational-FPS defects. The saved72-cell rate-support
  check preserves old labels; profile/mask/checkpoint joins and upstream
  conditioning remain to be validated before an old share gates a new model.
- [W7/W9 and later native-IVF correction](docs/experiments/w7-anchor-rate-repair.md):
  Tier1 player gains survive available ROI/native/neural comparisons
  within their modeled-rate scope, conditional on missing med000 DCVC; the
  Medvedev subgroup and unseen-match tier remain inconclusive. Whole-frame losses, temporal failures,
  match/source exposure, reference reuse and incomplete operational receiver
  remain. The website's18.46kbps bill predates corrected19.24kbps modeling.
- [Paid ball confirmation](docs/experiments/submission-ladder-ball-confirmation-analysis-20261115.json):
  narrow crop-DISTS benefit across three sources, with full-frame costs,
  sparse labels and disclosed encoder exposure. Do not add its BD-rate to W7.
- [Combined foreground audit](docs/experiments/submission-combined-foreground-gap-audit-20261115.md):
  common source/cadence/player-and-ball references/receiver/cost contract remains
  necessary. Racket evidence is required only for a racket-inclusive claim;
  equipment harms remain visible in full-frame metrics and human review.
- Native/tracked-pixel endpoints and source-denied replay are implemented;
  the complete learned-player receiver is a separate unresolved qualification.
  [Fixed patch negatives](docs/experiments/submission-ladder-20261114.md) remain
  scoped negatives, not a rejection of generative players.

## Next work, in decision order

**Stage A implementation:** [November16 requalification](docs/experiments/player-requalification-20261116.md)
authenticates the selected source/conditioning/checkpoints, checks CPU processor
tensors and corrects future-fit CLIP resizing. The approved three-profile GPU
replay now passes all source/output/cost checks and within-job exact repeats,
charged1,203GPU/3,003CPUseconds. The [source-aligned user review](docs/experiments/player-replay-20261116/human-review-import.json)
is imported: Player1 uncertain; Player2 prefers common tennis and reports a
stock torso-facing hallucination. Common tennis is the provisional development
lead. The [next motion pilot](docs/experiments/player-motion-20261116.md) is
approved and stopped at its original cost gate: the real Player2 44-frame
smoke validates, but forecast1,704.68 s exceeds the1,500 s full cap. No full run
or Player1 request started; conservative charge328GPU/1,828CPUseconds. The
[amendment](docs/experiments/player-motion-amendment-20261117.md) was explicitly
approved for a repeated smoke and larger per-request cap inside the unchanged
70GPUminutes/2CPUhours overall envelope; [live cumulative accounting](docs/experiments/player-motion-amendment-20261117/release.json)
includes the original charge. Player2 job `20261117T094919Z-5a0cc722` completed
and passed independent44/240-frame integrity checks. Its actual closure left
2,531GPU/3,281CPUseconds, enough to admit Player1 job
`20261117T102252Z-ccb04da2` with its full2,400GPU/3,150CPU reservation. Player1
has now completed and passed independent44/240-frame integrity checks. Both
jobs/claims are closed; total conservative3,318GPU/6,318CPUseconds includes
the original stop and both full externalCPU reservations, within4,200/7,200.
The user's [full-run Player2 review](docs/experiments/player-motion-amendment-20261117/player-2-human-review.json)
is acceptable; Player1 review remains pending. Integrity and one-author
acceptability establish neither matched-codec preference nor population benefit.
The24frame pilot and
author review establish neither a whole-codec nor confirmation result. No training ran.

| Stage | Deliverable | Admission/stop decision |
|---|---|---|
| A — code/input requalification and receiver contract | Audit saved rate-support/profile/checkpoint/mask joins, upstream preprocessing and source geometry. Freeze one development source, cadence, every-frame output/fallback, complete packet/state and cost forecast. Requalify the three existing generic/tennis-only/match-adapted recipes on a small paired subset before selecting a model. Reduced-cadence profiles need same-cadence anchors; sampled scores do not certify every-frame behavior. | No automatic acceptance of old positives or negatives; no broad model search/new fitting. Reuse valid W8 and W7/W6 proofs only in their scope. If a fresh witness cannot fit a costed stage, revise the minimal contract or report the blocker; do not replay failed jobs. |
| B — representative human pilot | Source-only reviewed player-and-ball references through a qualified dedicated editor; separate randomized output viewing of representative successes/failures at matched or explicitly stated rates. Optional equipment audit on a fixed labelled subset. Reuse compatible saved full-rate outputs with explicit trust/temporal qualifications. | Freeze sample/presentation before outcomes. Human judgments are development evidence. Temporal failures and racket harms remain visible, while racket improvement does not gate the narrower claim. |
| C — one combined development screen | Native base+players+ball on the same sources, three rates; player-only/ball-only ablations; effective ROI and DCVC controls; all-frame components, temporal/failure metrics and actual paid bytes. Racket mode is an optional separate case. A saved-output finite-family screen is labelled as such. | Correctness smoke and complete measured lifecycle first. Reuse exact identities only. No arbitrary effect-size floor or unsupported BD-rate. Human review precedes a consequential quality decision. |
| D — minimal complete decoder | Independently regenerate one frozen source-denied bundle, including reference bank, adapter/guider, receiver masks, ball modes and fallback; an equipment mode is included only if claimed. The witness covers actual generation, not just CPU conditioning. | Reuse qualified runtime/model/GPU facts. One bounded operational pilot unlocks actual fresh output; avoid a week of qualification without that decision. Full native-cadence behavior remains explicit. |
| E — confirmation | Freeze the strongest complete candidate, reference policy, primary contrast/metrics, source sampling and power/precision/budget plan; confirm on new physical matches. | Aggregate windows within match. Include small-effect scenarios and uncertainty, all sources and unsupported intervals. No retuning against W7 or opened ball confirmation. |
| F — manuscript/artifact | Restore supported player positives, methods, controls, informative negatives and temporal/racket limitations; tables from records; author/rights/live venue review. | Expand toward the verified venue allowance. Do not claim a combined win, racket benefit, live adaptation or real time unless actually established. Filing needs separate instruction. |

Stages A/B can prepare independently. A small learned-runtime witness from D
may be needed before C, but a full receiver integration must not obscure the
scientific screening decision. Every compute stage needs a concrete released
CPU/GPU budget, correctness smoke, validator, deadline and <=45-minute recovery
bound. Proposed stage descriptions are not dispatch authority or runtime forecasts.

## Model and scope choices

The generic, common-tennis and adapted short recipes are requalified and viewed.
Prioritize common tennis for longer motion, without claiming a final method
selection from two short player crops. The user found no clear need for paid
match state in this pilot; stock's facing error is a retained diagnostic,
not a universal rejection or permission for a new parameter sweep. Source-spread
and complete paid codec comparisons still precede the confirmation freeze. The common sport
model is a declared installed resource, with fair baseline resource accounting,
footprint and cold initialization reported. Match adapters, guider changes,
references and refreshes remain content-specific and paid. Preserve training
overlap and the historical base-charged sensitivity. No model is reclassified
as generic merely to erase a session cost.

Preserve the stock conditioning qualification: retained body-only COCO/OpenPose
maps omit dense face/hand cues drawn by the upstream DWPose path. Transport
parity is not physical cue correctness, and the torso-facing cause is unresolved.
A code audit finds no explicit reference RGB swap. The fresh24-frame outputs
remain unscored by LPIPS/DISTS; a CPU scorer would need explicit judge-resource
staging, a bounded cost pilot and source-aligned crop/mask semantics. No metric
failure or inherent stock-weight failure is inferred from the human review.

New fitting, per-player fitting, newer backbones and first-N-second adaptation
are conditional extensions. A causal warm-up study must encode all initial
frames conventionally, use only arrived data, charge update/compute/delay and
switch after the receiver can load the update. It is not already demonstrated
by a disjoint first-half/second-half study. Do not train before the existing
receiver/cost/temporal evidence identifies a concrete limiting assumption.

## Schedule, authority and continuation

Racket synthesis is a possible small appendix pilot after the primary result:
try paid pixels or coded reference geometry before a new generator. The
[additional-domain screen](docs/experiments/additional-sport-domain-screen-20261115.md)
allows read-only source preparation now and at most one separately costed
inference pilot after common-pipeline correctness. Prefer existing usable
GenStream skating RGB with an actual coded background; volleyball/SportsMOT is
an alternative, soccer a larger actor/occlusion stress test. No assumed3D scene,
new background reconstruction, extra model training or bulk dataset download.
The [scope decision](docs/experiments/foreground-scope-and-requalification-20261115.md)
keeps an egocentric comparison if coherent; a separate paper is a later author
decision requiring its own defensible contribution, not a second project now.

### Current dispatch ledger

| item | card / evidence | branch | model / review | status |
|---|---|---|---|---|
| Tennis contribution and history audit | [research brief](RESEARCH.md); [combined gaps](docs/experiments/submission-combined-foreground-gap-audit-20261115.md) | `codex/tennis-contribution-audit-20261115` | primary plus scoped reviewers; actual tier telemetry unverified | complete: PR429 merged with successful CI;423mergedPR metadata indexed and historical plan preserved |
| Foreground scope and actual-code re-audit | [scope/repair record](docs/experiments/foreground-scope-and-requalification-20261115.md) | `codex/foreground-scope-and-evidence-audit-20261115` | primary plus scoped source reviewers; actual tier telemetry unverified | complete: PR430 merged28377fb5 with successfulCI; player-and-ball primary, racket optional, strict support/coverage/rational-FPS fixes |
| Player requalification implementation | [exact inputs, code and release](docs/experiments/player-requalification-20261116.md); [PR431](https://github.com/emanuele-artioli/PointStream/pull/431) | `codex/player-requalification-20261116` | primary plus scoped builders/reviewer; actual tier telemetry unverified | CPU source/receipt/processor checks complete; future-fit correction and paired worker/specs implemented; configuredRuff/Mypy668/full2650passed39skipped; PR431 merged with successfulCI; subsequent replay recorded below |
| Approved three-profile player replay and human pilot | [actual jobs and audits](docs/experiments/player-replay-20261116/release.json); [human review](docs/experiments/player-replay-20261116/human-review-import.json); [PR432](https://github.com/emanuele-artioli/PointStream/pull/432) | `codex/player-replay-release-20261116` | primary plus scoped audit/preview builders; actual tier telemetry unverified | all three complete/PASS;1,203GPU/3,003CPUseconds; PR432 merged with successfulCI. Human review imported: Player1 uncertain, Player2 common tennis preferred; stock torso-facing failure. Common tennis provisional, no codec win/new training |
| Human-informed shared-tennis motion pilot | [closed original release](docs/experiments/player-motion-20261117/release.json); [approved amendment and cumulative ledger](docs/experiments/player-motion-amendment-20261117/release.json) | `codex/player-motion-amended-run-20261117` | primary plus scoped runner/independent audit; actual tier telemetry unverified | both amended44/240-frame requests complete and independently integrity-accepted; no active claims. Conservative3,318GPU/6,318CPU includes original stop and both externalCPU caps; within original4,200/7,200. Player2 author review acceptable; Player1 review pending |
| Common-player packet contract | [unreleased receiver design](docs/experiments/player-common-receiver-design-20261117.md) | `codex/common-player-packet-20261117` | primary plus scoped builder; actual tier telemetry unverified | local CPU implementation/review in an isolated checkout: explicit common-model and native-only modes, paid references/geometry and every-tick coverage. No GPU/model/scoring release; actual source-denied generation/masks and retained-input equivalence remain gates |
| Combined receiver/development pilot | stagesA–D above | — | unassigned | unreleased: exact inputs, source-independent contract and complete cost forecast precede budget approval |

### Dates and authority

Research cutoff: **December19,2026**. Submission preparation target:
**December26**, subject to actual author approval/rights and current venue
rules. November15–20 focuses on A/B and a costed minimal witness; November21–27
on the bounded combined screen and candidate choice; November28–December11 on
confirmation preparation/execution; December12–19 closes the strongest justified
result and reproducibility. Use December20–26 for final paper/artifact review.
If a stage slips or fails, narrow the supported claim instead of extending the
schedule for a favorable sign. Preserve the empirical fallback paper.

The automatic twelve-hour compute authority expired **November15,12:07:34CET
/11:07:34UTC**. Its scientific jobs are closed; unused allocations
are not reopened. No new GPU/training/paid hosting or filing is released by
the earlier audit. The separately approved bounded replay is now complete;
unused GPU/CPU allocations were not transferred to another experiment. The
human-review import itself authorized no computation. The November17 motion
approval stopped at its cost gate; its unused envelope is not automatically
reopened. The specific amendment was separately approved; no further retry or
scope expansion is released. Its live ledger must count the original charge
before every sequential admission. Local
implementation, read-only audits and repository checks may continue. Check
exact experiment IDs/status before acting; never replay unknown
submissions. [Original authority ledger](docs/experiments/submission-ladder-autonomy-20261114.json).

A new session reads RESEARCH.md, this file and only the linked evidence needed
for its decision. The detailed [November15 continuation](docs/submission-next-actions-2026-11-15.md)
remains available; this plan supersedes its earlier ordering where different.
[Current paper readiness](docs/paper/submission-ladder-readiness-2026-11-15.md)
does not mean submission approval. PRESLEY stays read-only.

## Historical section compatibility

The previous PLAN is preserved byte-for-byte as PLAN-history-2026-11-15.md;
SHA256: 016e105e52d2c5e69dc4757d9a7975d49705b9559436757ec9c7602624ab1169. Older links into PLAN still land below and point to their
original dated section. These are navigation stubs, not current dispatches.

<details><summary>Open historical section links</summary>

<a id="plan"></a>
[Historical: Plan](PLAN-history-2026-11-15.md#plan)

<a id="current-priorities--user-revision-november-15-2026"></a>
[Historical: Current priorities — user revision, November 15, 2026](PLAN-history-2026-11-15.md#current-priorities--user-revision-november-15-2026)

<a id="submission-contribution-ladder--authorized-implementation-november-14-2026"></a>
[Historical: Submission contribution ladder — authorized implementation, November 14, 2026](PLAN-history-2026-11-15.md#submission-contribution-ladder--authorized-implementation-november-14-2026)

<a id="where-pointstream-stands-2026-11-14-utc"></a>
[Historical: Where PointStream stands (2026-11-14, UTC)](PLAN-history-2026-11-15.md#where-pointstream-stands-2026-11-14-utc)

<a id="current-research-decision-november-14-utc"></a>
[Historical: Current research decision (November 14, UTC)](PLAN-history-2026-11-15.md#current-research-decision-november-14-utc)

<a id="completed-1648-evidence-and-earlier-scheduling-snapshots--history"></a>
[Historical: Completed 16/48 evidence and earlier scheduling snapshots — history](PLAN-history-2026-11-15.md#completed-1648-evidence-and-earlier-scheduling-snapshots--history)

<a id="historical-af6-phase--balances-waits-and-watch-superseded-above"></a>
[Historical: Historical af6 phase — balances, waits and watch superseded above](PLAN-history-2026-11-15.md#historical-af6-phase--balances-waits-and-watch-superseded-above)

<a id="records-of-2026-10-28"></a>
[Historical: Records of 2026-10-28](PLAN-history-2026-11-15.md#records-of-2026-10-28)

<a id="resume-here-after-the-root-chat-is-archived"></a>
[Historical: Resume here after the root chat is archived](PLAN-history-2026-11-15.md#resume-here-after-the-root-chat-is-archived)

<a id="historical-findings-and-completed-components"></a>
[Historical: Historical findings and completed components](PLAN-history-2026-11-15.md#historical-findings-and-completed-components)

<a id="dispatch-ledger"></a>
[Historical: Dispatch ledger](PLAN-history-2026-11-15.md#dispatch-ledger)

<a id="done"></a>
[Historical: Done](PLAN-history-2026-11-15.md#done)

<a id="phase-1-egocentric"></a>
[Historical: Phase 1: egocentric](PLAN-history-2026-11-15.md#phase-1-egocentric)

<a id="b1-visor-adapter-and-evaluation-set-done-2026-10-07"></a>
[Historical: B1. VISOR adapter and evaluation set (done 2026-10-07)](PLAN-history-2026-11-15.md#b1-visor-adapter-and-evaluation-set-done-2026-10-07)

<a id="b2b-baseline-follow-ups-agreed-2026-10-07-after-b2"></a>
[Historical: B2b. Baseline follow-ups (agreed 2026-10-07, after B2)](PLAN-history-2026-11-15.md#b2b-baseline-follow-ups-agreed-2026-10-07-after-b2)

<a id="b1b-filling-visors-missing-hands-and-objects-done-2026-10-08"></a>
[Historical: B1b. Filling VISOR's missing hands and objects (done 2026-10-08)](PLAN-history-2026-11-15.md#b1b-filling-visors-missing-hands-and-objects-done-2026-10-08)

<a id="b2-baseline-rate-distortion-on-visor-done-2026-10-07"></a>
[Historical: B2. Baseline rate-distortion on VISOR (done 2026-10-07)](PLAN-history-2026-11-15.md#b2-baseline-rate-distortion-on-visor-done-2026-10-07)

<a id="g-background-user-2026-10-08"></a>
[Historical: G. Background (user, 2026-10-08)](PLAN-history-2026-11-15.md#g-background-user-2026-10-08)

<a id="g1-camera-motion-and-coverage-audit-next"></a>
[Historical: G1. Camera motion and coverage audit (next)](PLAN-history-2026-11-15.md#g1-camera-motion-and-coverage-audit-next)

<a id="g2-background-evaluation-protocol"></a>
[Historical: G2. Background evaluation protocol](PLAN-history-2026-11-15.md#g2-background-evaluation-protocol)

<a id="g3-panorama-background"></a>
[Historical: G3. Panorama background](PLAN-history-2026-11-15.md#g3-panorama-background)

<a id="g4-clean-background-frames"></a>
[Historical: G4. Clean background frames](PLAN-history-2026-11-15.md#g4-clean-background-frames)

<a id="g1d-depth-aware-warps-on-visor-done-2026-10-09"></a>
[Historical: G1d. Depth-aware warps on VISOR (done 2026-10-09)](PLAN-history-2026-11-15.md#g1d-depth-aware-warps-on-visor-done-2026-10-09)

<a id="g1e-reference-refresh-done"></a>
[Historical: G1e. Reference refresh (done)](PLAN-history-2026-11-15.md#g1e-reference-refresh-done)

<a id="g5-neural-background-model"></a>
[Historical: G5. Neural background model](PLAN-history-2026-11-15.md#g5-neural-background-model)

<a id="g5b-scene-adapted-neural-background-user-2026-10-10"></a>
[Historical: G5b. Scene-adapted neural background (user, 2026-10-10)](PLAN-history-2026-11-15.md#g5b-scene-adapted-neural-background-user-2026-10-10)

<a id="g5c-scene-model-at-scale-agreed-with-the-user-2026-10-10"></a>
[Historical: G5c. Scene model at scale (agreed with the user, 2026-10-10)](PLAN-history-2026-11-15.md#g5c-scene-model-at-scale-agreed-with-the-user-2026-10-10)

<a id="g6-the-background-in-the-first-working-pointstream-plan-2026-10-12"></a>
[Historical: G6. The background in the first working PointStream (plan, 2026-10-12)](PLAN-history-2026-11-15.md#g6-the-background-in-the-first-working-pointstream-plan-2026-10-12)

<a id="h-foreground-user-2026-10-08"></a>
[Historical: H. Foreground (user, 2026-10-08)](PLAN-history-2026-11-15.md#h-foreground-user-2026-10-08)

<a id="h1-foreground-motion-and-representation-audit-done-2026-10-08"></a>
[Historical: H1. Foreground motion and representation audit (done 2026-10-08)](PLAN-history-2026-11-15.md#h1-foreground-motion-and-representation-audit-done-2026-10-08)

<a id="h2-hand-pose-estimators-hamer-against-wilor-done-2026-10-10"></a>
[Historical: H2. Hand-pose estimators: HaMeR against WiLoR (done 2026-10-10)](PLAN-history-2026-11-15.md#h2-hand-pose-estimators-hamer-against-wilor-done-2026-10-10)

<a id="h3-hand-and-arm-rendering"></a>
[Historical: H3. Hand and arm rendering](PLAN-history-2026-11-15.md#h3-hand-and-arm-rendering)

<a id="h4-handled-objects-rescoped-after-h1-user-2026-10-08"></a>
[Historical: H4. Handled objects (rescoped after H1, user 2026-10-08)](PLAN-history-2026-11-15.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08)

<a id="h5-first-working-pointstream"></a>
[Historical: H5. First working PointStream](PLAN-history-2026-11-15.md#h5-first-working-pointstream)

<a id="k-where-keypoints-win-analysis-user-2026-10-13"></a>
[Historical: K. Where keypoints win (analysis, user 2026-10-13)](PLAN-history-2026-11-15.md#k-where-keypoints-win-analysis-user-2026-10-13)

<a id="after-a-working-pipeline"></a>
[Historical: After a working pipeline](PLAN-history-2026-11-15.md#after-a-working-pipeline)

<a id="h3d-follow-ups-deferred-user-2026-10-13"></a>
[Historical: H3d follow-ups (deferred, user 2026-10-13)](PLAN-history-2026-11-15.md#h3d-follow-ups-deferred-user-2026-10-13)

<a id="d1-segmentation-benchmark-on-visor-and-egohos"></a>
[Historical: D1. Segmentation benchmark on VISOR and EgoHOS](PLAN-history-2026-11-15.md#d1-segmentation-benchmark-on-visor-and-egohos)

<a id="e1-handled-object-proposer"></a>
[Historical: E1. Handled-object proposer](PLAN-history-2026-11-15.md#e1-handled-object-proposer)

<a id="f1-training-data-export"></a>
[Historical: F1. Training-data export](PLAN-history-2026-11-15.md#f1-training-data-export)

<a id="phase-2-racket-sports"></a>
[Historical: Phase 2: racket sports](PLAN-history-2026-11-15.md#phase-2-racket-sports)

<a id="c-sam-31-from-label-prompts"></a>
[Historical: C. SAM 3.1 from label prompts](PLAN-history-2026-11-15.md#c-sam-31-from-label-prompts)

<a id="players-stage"></a>
[Historical: Players stage](PLAN-history-2026-11-15.md#players-stage)

<a id="ball-stage"></a>
[Historical: Ball stage](PLAN-history-2026-11-15.md#ball-stage)

<a id="d2-segmentation-benchmark-on-racket-sports"></a>
[Historical: D2. Segmentation benchmark on racket sports](PLAN-history-2026-11-15.md#d2-segmentation-benchmark-on-racket-sports)

<a id="throughout"></a>
[Historical: Throughout](PLAN-history-2026-11-15.md#throughout)

<a id="w-paths-to-a-baseline-win-root-2026-10-29"></a>
[Historical: W. Paths to a baseline win (root, 2026-10-29)](PLAN-history-2026-11-15.md#w-paths-to-a-baseline-win-root-2026-10-29)

<a id="j-experience-track-re-judging-user-2026-10-22"></a>
[Historical: J. Experience-track re-judging (user, 2026-10-22)](PLAN-history-2026-11-15.md#j-experience-track-re-judging-user-2026-10-22)

<a id="i-paper"></a>
[Historical: I. Paper](PLAN-history-2026-11-15.md#i-paper)

<a id="after-the-paper"></a>
[Historical: After the paper](PLAN-history-2026-11-15.md#after-the-paper)

<a id="agent-tooling-to-coding-agent-config-user-2026-10-29"></a>
[Historical: Agent tooling to coding-agent-config (user, 2026-10-29)](PLAN-history-2026-11-15.md#agent-tooling-to-coding-agent-config-user-2026-10-29)

</details>
