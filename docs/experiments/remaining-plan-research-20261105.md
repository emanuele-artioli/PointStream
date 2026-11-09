# Experiments: Remaining PLAN research decisions

### 2026-11-05 — Research-first audit and dispositions

This is a decision inventory, not authority for a job or a change to the
shared budget. It reads the current PLAN status, the November 5 continuation,
the closure scan and the linked experiment records; historical status embedded
later in PLAN is superseded where those records say so. The user's current
instruction is to finish worthwhile research before manuscript changes. The
30-clip ball scale and source-hidden decoder are owned in parallel. Training
and the legacy 200-GPU-hour study are outside scope. No new computation was run
for this audit. Costs below are ceilings or historical charges where explicitly
identified, not forecasts for an unprepared new job.

## Decisions with potential submission value

| Item | Evidence and remaining question | Cheapest next gate, cost and stop |
|---|---|---|
| Ball retrospective 30 clips | Actual 16/48 controls and 188-frame six-point result are accepted; source-only cross-sport judge failed. The 30 clips were used in tracker selection and include 197 absent and six unknown frames, so the intended result is retrospective. [K5d](k5d-ball-benchmark.md), jobs `d8b5c76b`, `cc450baf`; [continuation](../continuation-2026-11-05.md). | Approved first tranche 2.5 GPU-h/3 CPU-h is a ceiling, not an established forecast. Real-native normal case passed but interruption API failed; raw pidfd syscall self-capability is repaired, with the full four-case proof still required. Full controls are under review before exact-source freeze, then 16-frame parity and 292/586 confirmations precede a complete 30-clip forecast with reserves. Stop on correctness/cost failure. Do not label the 188-frame result corpus evidence. |
| W8 fresh receiver | Real packet and folder mechanics are proved; no fresh generated reconstruction or model-quality claim. The clock-policy repair passed a real one-step reconstruction and kernel/source guards, then failed the combined cost gate; a fresh longer paired attempt was contended before a completed step. [W8](w8-wire-audit.md), `df001de2`; [continuation](../continuation-2026-11-05.md). | Two latest phases are closed: one-step cost gate failed (`8eeacee2`), then paired pilot `64ef0bb9` stopped on foreign GPU work before sealing a step. No quality claim or 1→3 slope. Await user preference for an exclusive Ada window or bounded queues before any new source/host/deadline release. A successful future paired cost pilot supplies a slope, not 30-step quality. A separately admitted fresh witness follows only if its complete forecast fits. |
| W7 primary strict-IVF coverage | Corrected accepted `d09b0543`, independently audited `5a435128`, covers 75 roles/71 distinct streams and six familiar-match windows; player advantage is conditional, whole-frame anchor error lower, Med subgroup inconclusive, med000 DCVC missing. Strict half-out brackets remain missing for med000/019; MKV sensitivity brackets them. [Repair](w7-anchor-rate-repair.md). | No additional encodes authorized by the accepted repair record. If root decides the strict-family distinction changes a paper decision, first derive two exact target rates/CRFs from the frozen report and forecast encoding, complete-byte proofs, scoring controls and audit. The historical card allocation of 0.5 GPU-h/0.75 CPU-h is not a new release or a cost estimate for that proposal. Otherwise retain the explicit limitation. User dropped med000 DCVC; do not repair it. |
| GVC peer curve | `20261105T121902Z-4020b8ee`, source `9d56c5e1`, independently audited fixed-QP3 16-frame native point: crop LPIPS improves against interpolated anchor but whole-frame PSNR worsens; no curve or broad ranking. [Report](ego-peer-inputs/20261104/gvcrt-4020b8ee-independent-report-20261105.md). | QP0 `78e54aff` completed with an independently frozen validator and root mean/byte/source audit: 175.815 native /219.270 adapter-wire kb/s. Crop LPIPS improves on this clip while full-frame PSNR worsens; full DISTS advantage at native rate reverses at adapter wire rate. QP9 `2f65a26e` was contended before producing a point. Phase closed at300 GPU/1,670 CPU seconds including proof and failed transfer. Planned three-point sensitivity remains incomplete; no BD-rate/corpus claim. Hold retry pending the same exclusive Ada resource decision as W8. |
| W11 racket | No independent pixel-accurate racket mask linked to W6 frames; body-hull coverage is not racket recall. [Annotation audit](w11-racket.md). | Read-only inventory for eligible independently annotated masks with source/frame identity; stop if none. Only then define a source-pixel paste oracle at W1's operating point and measure both player and full-frame gains before a tracker/renderer. Cost unknown until truth and scoring inputs are identified; no proxy is an adequate substitute. |
| B2b equal-time, ROI and low-rate ladders | Scoring, timing, range and complexity work exists, but its explicit full completion requires actual pipeline timing and appropriate ROI/low-rate encodes. [B2b](b2b.md), [PLAN closure scan](../paper/plan-closure-scan-2026-11-05.md). | First determine whether a retained claim actually depends on this VISOR full-pipeline comparison, then recover real end-to-end timing and existing coverage; only if needed specify a small offset/resolution pilot and complete forecast. Source inspection of SVT ROI/lambda is cheap hypothesis work, not evidence for causality or a substitute for the encoder experiment. Cost unknown; no full 34-item release implied. |

## Closed, scoped, or deferred — do not repeat to clear a row

| PLAN area | Disposition and next condition |
|---|---|
| G1–G5c, H1/H2/H4, K1–K4 and completed K4b/K5/K6 paths | Completed or superseded in [PLAN](../../PLAN.md#phase-1-egocentric) and its linked history. A later positive ball result does not reverse earlier failed renderer/guard outcomes. |
| G6/H5 full 34-window endpoint | Incomplete: short and pilot results do not certify a 34-window pipeline. A substantive end-to-end run requires its own fixed decision rule, scored input inventory, upper bound and staged cost; it is not admitted by the current bounded program. Do not imply completion from 2/4-window results. |
| H5q paid-mask route | [Paid-mask oracle](ego-peer-feasibility-20261104.md) fails the four-item fixed-pixel pooled rule even at 2 kb/s; one small-foreground window remains a narrow positive. A different independent small-foreground window with real paid masks and exact accounting is a new conditional hypothesis, not a rerun that erases the negative; no cost forecast or release yet. |
| W6 temporal | Accepted 239-difference smoothing analysis improves descriptive one-window flicker and perceptual measures, but original temporal rule against SVT still fails. [W6](w6-motion-pose.md), `30f6379a`. No fresh full decoding or population generalization follows. A new temporal method needs its own upper bound and rule. |
| W1–W5, W7 Tier 2, W9 | W2 and W4 negative; W1/W3/W5 are assessed with their recorded anchor limits. Corrected W7 Tier 2 fails the original joint confidence rule at two hours; W9 duration model has bounded coverage and is not global break-even. [W9](w9-adapter-break-even.md). W9 live adaptation requires new fitting, so is outside the no-training scope; generation speed also blocks live operation. |
| W10 larger base/new backbone | New backbone and larger base remain untested, but overfit/fine-tune oracles and training are outside current authorization. A zero-shot-only screen would not settle that intended lever; do not spend solely to mark it done. |
| J experience-track and identity warp | All four tested routes remain closed or not re-judged because outputs were not retained; J2 smoothing fails. Identity warp is a new diagnostic for estimator versus compositing, with no existing paper endpoint requiring it; specify a decision and budget if revived. [J](j.md). |
| Phase 2 cross-sport and segmentation | OpenTT source judge `a4067c30` found 0/85 visible; cross-sport coding was not admitted. D1, E1, F1, phase-2 C/players/D2 require benchmark, cross-test or provenance-filtered data export, not merely a segmentation CLI. These are post-pipeline/extension tasks; E1 involves training. [Closure scan](../paper/plan-closure-scan-2026-11-05.md). Reopen only with an independent judge that passes source eligibility and a concrete submission claim. |
| H3d follow-ups and G at scale | Optimisation/longer-video adaptation require oracle, suitable recordings and often training; deferred after a working pipeline, not covered by this program. |

A negative or incomplete gate narrows only the tested configuration. No
research point above is marked complete because its code exists, its smoke
passed, or a related pilot succeeded. Source-footage permissions, author
review and filing authority are human prerequisites, not experiments.

## Root disposition — W7 strict-family sensitivity, November5

Root re-inspected the accepted `ivf-mkv-corrected-tier1-report-20261105.json` at the two-hour primary scenario: every six-window SVT LPIPS/DISTS comparison is an interpolated bracket; five DCVC comparisons are bracketed and med000 DCVC remains the explicit user-dropped absence. Only the historical-family sensitivity has unresolved `s2_o20` cells for med000 and med019. Thus the remaining two strict-family cells do not prevent the retained pooled achievable-envelope primary decision. No further encoding is worth spending solely to erase those sensitivity omissions. Retain them beside the MKV sensitivity and the missing DCVC case, without claiming all original families are complete. This closes the *current research decision* to pursue additional SVT points, not the missing measurements themselves. No new scoring, rates, thresholds or claim promotion.

## Root disposition — W11 and B2b, November5

The parallel read-only audit found no newly usable independent, W6-frame-linked racket pixel or occlusion truth in the bounded repository inventory and documented prior manifest search. Root re-read W11's exact SHA/window and annotation audit: the RacketVision filename/hash search does not rule out re-encoded overlap, its native labels are boxes/keypoints, and W6's body-hull proxy can omit detached racket pixels. **Defer the paste oracle for a missing prerequisite**; this is neither a failed oracle nor evidence that no labels exist anywhere. Do not create labels or substitute proxy coverage for recall. Reopen with an independently annotated pixel mask and exact source/frame alignment. See [W11](w11-racket.md), [W6](w6-motion-pose.md), and the recorded dataset inventory; no fresh corpus-wide visual search or experiment was run.

Root also re-read B2b/H5/H5p: actual four-window pilot timing and the preset-1/low-resolution ladder exist, including H5p's **+74% billed pooled weighted-LPIPS failure**. Preserve those scoped results and the four-item preset-4 ROI pilot. They do not certify the original 34-item endpoint, complete paid receiver or production equal-time claim; F/model training was not timed. The saved manuscript's pilot-only comparison and explicit end-to-end-speed disclaimer do not depend on that missing completion. **Defer the 34-item ROI/ladder/full-pipeline completion outside this finite no-training programme**, rather than run it merely to close a checkbox or imply no ladder was encoded. Reopen only for a concrete retained claim requiring that endpoint, with new complete timing and a scoped costed card. See [B2b](b2b.md), [H5](h5.md), and [H5p](h5p.md). No manuscript was changed.

## Root continuation — November9 bounded queues and admission scope

The user prefers bounded fleet queues rather than an exclusive reservation.
The first Ball GPU parity queue expired without host/GPU assignment; a fresh
all-host observation on November9 at about17:52UTC found an idle gpu2 A6000,
so root conditionally released one new exact-source parity request after
remaining readiness, identity and cost checks. This is no automatic replay or
292/586/full30 release. The first-tranche remaining reservation is8,900GPU /
8,870CPU seconds after a conservative50CPU shared readiness charge; the
programme total is18,382GPU /37,730CPU seconds before any later leaf charges.

The new W8 witness paths now use stat-only denial-target identities. Historical
protected SHA values are references, not current byte verification. Root's
fixed-input admission scope is a trusted fleet/controller and cooperative
immutable public inputs: every concrete declared input/environment/source path
is compared by metadata against the five protected targets before publication.
The independent review found no actual path/reference-digest overlap; its
pre-staging concern was a hypothetical alias or replacement, which the workload
guards alone do not cover. The admission check does not establish protection
against hostile replacement after review. No shared fleet security deployment
is required for this fixed scientific workflow; actual Landlock denial and
namespace/native proof remain mandatory. No new W8 job or remote upload has
been released yet. Software checks on b5661eaa passed2,001tests/39skips, Ruff
and mypy, but are superseded for final-source admission by subsequent patches.
The paper and PRESLEY remain untouched.
