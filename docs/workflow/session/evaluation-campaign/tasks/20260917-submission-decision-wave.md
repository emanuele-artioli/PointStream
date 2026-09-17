# Submission decision wave — 17 September 2026

**Purpose.** Find or rule out a credible, scoped development candidate before
the 20 September evidence/claim/schedule checkpoint. This is one bounded
decision wave, not a promise of a win, confirmation release, or date change.

## Read and report path

Read [AGENTS.md](../../../../../AGENTS.md), [PLAN.md](../../../../../PLAN.md),
this card, and the assigned area only. Report to the coordinator with the
input/output paths below, status (`pass`, `fail`, `inconclusive`, or `blocked`),
bytes, quality, speed, bounds and controls. The coordinator integrates shared
contracts and decides scientific scope.

Historical briefs `20260915-next-stage.md`, `20260916-probe-review.md`, and
`20260917-strategy-roadblock.md` are **HISTORICAL**. They preserve evidence but
do not authorize launches. Do not redo merged #127/#128/#129/#132 repairs.

## Current facts and boundaries

| Topic | Current fact | Decision consequence |
|---|---|---|
| E06 | Lowest physical total is 17,581 B, inherited 20.68 dB/0.672 SSIM/18.18 VMAF; saved VVC is 21,288 B/24.6 dB and AV1 19,116 B/27.4 dB. | Rate saving alone is not a win; no BD-rate. |
| Headroom | Exact-parent pixels and `n_client=2` pass. Prior residual 72–78 kB buys ~3.4 dB. | Remaining VVC budget is 3,707 B incl. overhead; probe cheap BG+residual as a sum. |
| Perfect FG | QP47 21.18 dB; QP32 22.82 dB; FG area about .4%. Artifact: labelled location `/var/tmp/emanuele-codex/visualizations/2026/09/16/01a0a923-4c5e-71c3-8993-5c68f2a76bb4/pointstream-headroom-20260917.json`. | Diagnostic is limited to its masks/predictor/scorer; no universal limit. |
| E04B | Uncompressed ghost MAD 5.135 with removal ON is observed pipeline error; compression adds. Mask exclusion changes 2.5–4.9%; Telea zero-hole path unexercised. | Keep original alarms and post-hoc disposition separate; no geometric-limit claim. |
| E05 | Paste 33.94 dB, ~1 MB, 1.30 s; generator 31.98 dB, 6.67 MB, 7.03 s on 16 native 4K frames. Fitted weights are 217,736,406 B. | Reject this checkpoint; preserve family, do not salvage or grid-train. |
| GenStream | PR #1 is open and green (head `2e7f7e4b3522308fb52690a17d2d9c4330e8b578`, CI `35270622033`); CPU receiver supports reference/pose affine only and has no neural benchmark. | Optional reusable infrastructure, never the research gate or reason to defer neural anchor. |
| Completed repairs | PR #132 merged as `5482370`; PRs #127/#128/#129 are also merged. | Do not reassign repairs as current work. |
| Data | Three 1080p reservations need exposure/replay/PTS audit; development scene 007 is not confirmation. Ego candidates exist but are not frozen confirmation data. | Audit before scores; select development windows from source properties, not score. |

## Phase 0 — release now, in parallel

### A. Background/headroom diagnostic

Inputs are the saved E06 floor arms in $PS_DATA_ROOT/outputs/evaluation-20260914/e06/probe-20260917-floor-arms/,
the labelled perfect-FG artifact location above, plus material locators
$PS_DATA_ROOT/outputs/evaluation-20260914/e03b/run-20260916-federer007/prepared_rgb.npy,
$PS_DATA_ROOT/outputs/evaluation-20260914/e04a/run-20260916-federer007/probe_report.json,
and $PS_DATA_ROOT/outputs/evaluation-20260914/e04b/run-20260916-paired-removal-r2/e04b_derived_audit_report.json.
Resolve arrays only through those manifests/reports; do not infer paths.
Write a new record under $PS_DATA_ROOT/outputs/evaluation-20260917/submission-decision-wave/a-headroom/run-UTC-slug/.

**Owner area:** evaluation + codec. **Inputs:** saved E06 arrays and existing
artifact ledgers only. **Output:** one record separating calibrated regional
metrics for perfect-FG, perfect-BG, lossless plate, coded plate, construction,
registration, and compression error; at most one costed cheap-BG+residual
system-probe card.

Where saved arrays cannot causally separate construction, registration, and
compression, report their combined observed error as unidentified; do not infer
that separation from this diagnostic.

Before reading a new score, record plausible two-sided quality/byte/time bounds
and rationale. Check source identity, then use at most 30 CPU minutes for
profiling/scoring. No predictor change or native encode. Stop this named E06
configuration if it has no plausible headroom under its 3,707-B VVC budget;
do not apply that local budget to changed backgrounds, domains, or ROI objectives.

### B. Executable neural foreground preparation

Inputs are manifests/candidate_inventory.json, the candidate's recorded training
reference, receiver contract, and paste/warp controls. The rejected E05
checkpoint/report under $PS_DATA_ROOT/outputs/evaluation-20260914/e05/ is a control, not a
candidate to salvage. Write a new record under
$PS_DATA_ROOT/outputs/evaluation-20260917/submission-decision-wave/b-foreground/run-UTC-slug/.

**Owner area:** generation. **Inputs:** existing checkpoint, training records,
receiver contract, paste and warp controls. **Output:** checkpoint SHA,
training/reference provenance, receiver path `compact vector → skeleton renderer
→ model`, and one bounded compact-inference card.

Do not transmit raw RGB pose data. Compare same-container paste, genuine warp,
and generation; declare shared versus fitted model bytes and all deployment
costs. Existing neural inference is allowed; new training grids are not. Cap
readiness at 2 active hours and accepted dev inference at 15 GPU minutes. If the
checkpoint/provenance is invalid or unavailable, return `blocked`, not invented
performance. GenStream work may only replace a missing capability.
A blocked foreground path does not stop the valid no-generation diagnostic.

### C. Neural anchor and score-free confirmation audit — priority

The neural-anchor inputs are manifests/evaluation_20260916_neural_anchor_readiness.json
and its prepared E03B stack. Write its record under
$PS_DATA_ROOT/outputs/evaluation-20260917/submission-decision-wave/c-neural-anchor/run-UTC-slug/.

These are independent deliverables and may proceed together when resources
permit. **Neural-anchor output:** exact weight SHA, config, binary/version,
actual encode, independent standalone decode, decoded-frame hashes, bytes, and
timing; otherwise the smallest specific blocker. The existing cap is 2 active
hours plus one single-QP, <=15-minute/<=0.5-GPU-hour run. HEAD-only HTTP 403 is
not proof every download route fails. The upstream combined runner reopens
source during decode and assumes 30 fps for 12-fps input, so use a standalone
wrapper and independently verify achieved bitrate/frame count. No environment
forcing, broad ladder, or silent anchor substitution. A recent published
alternative follows only after this named blocker and uses the same protocol.

**Audit output:** a score-free source report for the three reservations and ego
candidates: source and frame counts, exposure/replay/PTS status, content-based
occupancy/object-motion/camera-motion/occlusion selection for two development
windows per domain, and training-provenance cross-check. Do not infer holdout
status from filenames or claim AssemblyHands ground truth. No quality scores,
codec scores, acquisition, or confirmation release.

Audit inputs are manifests/evaluation_20260916_coordinator_confirmation_reservation.json
and /home/itec/emanuele/Datasets/Egocentric-10K/curated_v2_new/candidates.json.
Write the score-free record under
$PS_DATA_ROOT/outputs/evaluation-20260917/submission-decision-wave/c-source-audit/run-UTC-slug/.

### D. ROI, metric, and domain preparation

Inputs are saved Presley controls and source-audit selections. Write a new
record under $PS_DATA_ROOT/outputs/evaluation-20260917/submission-decision-wave/d-roi-metric-domain/run-UTC-slug/.

**Owner area:** evaluation + codec + data. Reuse
`src/presley/encode_utils.py` helpers `scores_to_qp_offsets`,
`encode_with_roi_kvazaar`, and `encode_with_roi_svtav1`; do not build a blank ROI library. Use
fixed QP/CRF where VBR can ignore ROI. Return actual ROI/null effect, final
QP/CRF, binary version, timing, standalone frame count, coverage, calibrated
whole/FG/BG metrics, source masks, and a fixed objective. Native ROI maps are
encoder-only; client masks are charged. x265 AQ is not semantic ROI.
This phase prepares/reuses a saved ROI effect only. If absent, make a separately
costed card for synthetic ROI on/off verification; it does not authorize a
candidate encode.

## Conditional development decision screen

Release only after A–D return usable identity, calibrated metric/source policy,
and each comparator fits its card. Compare at most two development windows per
domain and two rate targets:

1. feasible complete PointStream;
2. uniform AV1 and VVC;
3. verified conventional ROI;
4. simple paste and genuine warp; and
5. published neural anchor.

Use <=6 new PointStream settings/domain and <=30 minutes/domain including
decode/scoring. Native anchors are separately costed. Presley alone, charged
GenStream-style path, and fusion are diagnostic ablations; fusion must beat its
components. If a comparator cannot fit, return a cost card without dropping it.
Every comparator is required for competitive promotion. An incomplete set is a
labelled partial diagnostic, never a competitive result.

All arms use the same source/display/timebase and accounting policy, matching
pixels/frame count/aspect/fps, no uncharged lookahead/masks, physical native
files with each format's overhead, standalone decoded pixels, and
`T=B+F+M+R+H+declared deployment`. Report whole-frame context plus calibrated
FG/BG task metrics. Detection coverage/PCK use all denominators; never only
conditional MPJPE. Never black-mask whole-frame LPIPS/VMAF.

**Promote** only for matched achieved rate or an overlapping-quality advantage
greater than the predeclared meaningful effect and source uncertainty, with a
BG floor (for ROI/task objective), ledger/standalone pass, runtime budget, and
honest encoder stratum. A one-scene point is never a submission win. **Stop** on
no headroom/budget for the named configuration, instrument failure, or
cap-consuming repeats; distinguish a failed configuration from an inconclusive
uncertainty result. Fine-tune one hypothesis only if its next test could alter
selection. A blocked external neural anchor remains a visible submission/SOTA
blocker and never justifies endless readiness work.

## Release calendar and integration

| Date/state | Coordinator decision | Release condition |
|---|---|---|
| Now | Phase 0 diagnostics/preparation | Bounded cards above; no confirmation scoring. |
| After A–D | Development screen | Comparable complete arms and fixed measurement/source policy. |
| 20 Sep | Record claim/domain and neural-winner decision | Credible replicated development candidate, or explicit no-go/claim-date recommendation. |
| After freeze | Confirmation, essential ablations, domain bounds, reproducibility | Frozen source/metric/rate/procedure/adaptation/deployment policy. |
| 30 Sep | Submission target | Accepted evidence; no acceptance/win guarantee. |

Original submission contract remains tennis-primary, neural-foreground winner,
and a second domain. The development screen prioritizes ego while retaining
tennis; that development exception does not silently waive the contract. A
semantic-task pivot needs its own fixed metrics/weights and confirmation; it
cannot inherit tennis confirmation.

The paper stays untouched in this documentation task. It has a verified
28-page budget, so accepted panels replace material; do not add prose before
evidence. At any deadline blocker, record the no-go and its smallest evidence
path rather than submitting a losing headline.
