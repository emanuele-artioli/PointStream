# Experiments: submission contribution ladder

## Decision and current state

The user requested implementation of the six-week plan after a read-only audit of code `f2704daf` and manuscript `690cad0`. This is a new research programme, not completion or reopening of the [closed 24 GPU-hour/28 CPU-hour programme](research-disposition-20261113.md). Code and protocol implementation may proceed; no new scientific GPU allocation, training, paid hosting or submission is authorized. The first costed proposal is in the [stage 1 card](../cards/submission-ladder-stage1-2026-11-14.md).

**Goal:** a complete, independently decoding codec evaluated on tennis and egocentric hand-object video. **Research cutoff:** December 19, 2026. **Submission preparation target:** December 26, 2026, subject to author review and source permissions. Dates use America/Sao_Paulo. These are planning targets; no scheduled automation or experiment submission is created by this record.

The [prospective protocol](submission-ladder-protocol-20261114.json) is frozen before new scientific outcomes. Record a new protocol revision and the reason for any amendment; do not retroactively change the contract for completed results. Implementation is underway. No new contribution level has been scientifically established by this programme, and implementation tests are not rate–quality evidence.

## Contribution ladder and decisions

| Level | Required result |
|---|---|
| 1: narrow benefit | Predefined object/condition/objective benefit against the simplest relevant competitors, with complete transmitted costs. |
| 2: foreground priority | Better rate–quality under fixed 0.7 foreground / 0.3 background regional DISTS; regional LPIPS corroborates, and both components and full-frame quality remain visible. |
| 3: full frame | Standard full-frame metric improvement across independent footage, with temporal and object-level failures reported. |
| 4: real time | Retained quality benefit plus measured end-to-end encode/decode throughput, latency, buffering and resources under a declared streaming configuration. |

Evaluate both domains, but establish levels separately. A tennis success is not egocentric evidence. A working fallback codec establishes reconstruction, not a coding win. Use −10% BD-rate as the development promotion target for regional and weighted objectives; smaller effects remain descriptive. Confirmation additionally requires consistent source-level direction and uncertainty supporting the stated claim. Claim neither an automatic statistical pass nor a higher level from a development threshold alone.

Use three spread rate points initially. Add points only for inadequate common-quality overlap or an unclear curve, recording the reason before seeing their outcomes. Anchors are SVT-AV1, DCVC-UF and relevant conventional ROI allocation; include native base plus pixel patches, semantic versus motion/error selection, and ball disk/streak/coded-pixel controls as applicable. Preserve the corrected, effective SVT ROI convention rather than reviving inert negative-offset maps.

## Evaluation and data contract

- Reconstruct every frame from the packet and declared installed generic resources alone. Charge native streams, geometry, timing, placements, masks, references, refreshes, mode signaling and content-specific adaptation. Report cold resources separately from steady state. No content-specific state is silently installed.
- Weighted regional DISTS is a custom experimental objective, not a calibrated human preference score. When one region is empty, give all weight to the region that exists. Reject missing/nonfinite required scores; do not silently drop frames. Preserve historical aggregation unchanged and label new-protocol results separately.
- Report foreground, background, full-frame DISTS/LPIPS and pixel fidelity separately, plus all-frame temporal and applicable task outcomes. Visible/absent task denominators remain explicit; no discrete-task BD-rate. A source-qualification failure remains beside the judge response.
- Compute BD-rate only on supported common-quality intervals and disclose minimum support and interpolation. Audit saved-point sensitivity to linear-rate hull and Pareto/PCHIP references; call interpolated pooled envelopes constructed references unless physical switching is demonstrated.
- Freeze development and confirmation selections before outcome inspection, mechanically choose windows and audit exposure at match/source-video level. Existing TrackNet test results and VISOR v2 baseline exposure prevent describing their reused footage as untouched. New prefixes do not establish new-match generalization. Group uncertainty by source, never by correlated frames.
- Freeze the candidate and score contract before opening confirmation outcomes. Preserve every registered item, including failed decoding, absence, empty foreground and bad quality; no repeated tuning on the same confirmation set.
- Check independent annotation availability and exact source/frame joins first. VISOR human sparse annotations, dense interpolations and model-filled masks remain separate provenance tiers. Tennis source-model masks are not independent ground truth.
- If eligible tennis labels are unavailable, use SAM 3.1 proposals with human source-only review, boundary correction, missing-object marking and explicit uncertainty. Begin with a timed small batch. Freeze the source-spread confirmation sample before codec outcomes; include a subset annotated without the initial overlay to assess proposal bias. Preserve original proposals, corrections, uncertain/rejected cases and hashes. Call accepted references human-reviewed, model-assisted annotations; unreviewed propagation remains separate. Instance identity review alone does not validate silhouettes or rackets.

## Implementation and scientific order

1. Establish stable encode/decode/inspect interfaces and a versioned, fully billed packet. Reuse existing native codec workers and packet components without rewriting historical formats. Decode only packet plus declared resources; test fresh source-denied reconstruction and fallback.
2. Compare native coding and working native ROI allocation with native base plus rectangular coded object patches. Initially avoid exact-mask overhead. Compare semantic placement with motion/error placement and retain ball disk/coded-pixel controls. Count duplicate coding and all placement/refresh costs.
3. Before elaborate patch transport, screen an explicit finite family of real coded outputs with free placement metadata. Include the actual candidate among tested policies: source-pixel replacement is not a universal LPIPS/DISTS upper bound. A failed screen closes that construction; success admits paid implementation, not a final claim.
4. Complete source-denied decoding in both domains, freeze candidates and confirm on separate source groups. One machine-readable record per run retains source/provenance, exact configuration, resources, native/payload bytes, frame identities, metrics and timing.
5. Reopen generative synthesis only for a concrete changed hypothesis with feasible fresh decoding and complete paid-state accounting. Training/new models require separate authority and budget. Better synthesis alone cannot repair the tested player background; mask compression alone cannot repair the paid fixed-pixel egocentric construction.

Keep three reproducibility levels explicit: arithmetic from scores; rescoring saved reconstructions; and fresh re-encoding/independent decoding. Generate manuscript tables from the records and state which levels the artifact actually supports.

## Schedule and stopping

| Period, 2026 | Output / decision |
|---|---|
| Nov 14–20 | Consolidate evidence, verify pinned environment, audit source/annotation exposure, freeze metrics, prepare review pilot and receiver contract; submit first costed proposal. |
| Nov 21–27 | Bounded two-domain design screens and simplest controls; at most one added representation per domain; minimal complete decode/fallback. |
| Nov 28–Dec 4 | Establish narrow benefit, pursue foreground priority, measure paid costs and all-frame behavior; freeze confirmation and finish review. |
| Dec 5–11 | Confirm strongest lower-level result; attempt full-frame advance only with development headroom; prepare figures continuously. |
| Dec 12–19 | Independent confirmation and reproduction; real-time work only after lower levels are secure and measured cost fits. Freeze research Dec 19. |
| Dec 20–26 | Rewrite around actual achievements, author review, artifacts, permissions and live venue checks. |

Stage budgets cover preparation, staging, execution, encoding/decoding, scoring, validation, publication and failure allowance. Use existing subscriptions/GPUs, valid frozen qualifications and the fleet adapter. Run representative smokes, bounded pilots and scale confirmation; stages end within 45 minutes. Persist task IDs and budgets before canonical waits. No recurring LLM polling or automatic retry. Do not spend another week on qualification absent a specific unlocked scientific decision.

At cutoff publish the highest supported level per domain and retain all failures. Do not extend to seek a favorable sign. If simple competitors explain all gains, report the empirical result and reassess venue fit. Submission also requires full configured checks in the pinned environment, complete receiver verification, author approval, corresponding-author designation, footage permissions and live TOMM instructions. Filing needs a separate explicit instruction; PRESLEY remains read-only.

## Historical constraints retained

- [Ball control](k5d-inputs/controls-3ec159d8-independent-audit-20261105.json): disk and streak have equal 45/48 WASB and 47/48 BallTrack hits; useful placement does not identify a streak-specific mechanism.
- [Prior disposition](research-disposition-20261113.md): the primary neural/conventional ball result is 48 development frames; ROI benefits coexist with full-frame penalties; conventional follow-ups retain false detections and source failures.
- [Paid-mask oracle](ego-peer-feasibility-20261104.md): actual fixed-pixel paid shape costs fail pooled weighted DISTS (+27.92%); the small-foreground exception remains scoped.
- [Ideal player](../k6b-ideal-player-full-report-2026-10-28.md) and [paid clean background](../k6b-paid-clean-background-report-2026-10-28.md): the tested background/rate construction cannot be rescued by improving appearance alone. These finite negatives do not rule out every representation.

### 2026-11-14 — implementation checkpoint

This entry records design and implementation authority, not experiment completion. No fleet job, new source selection, new annotation, native encode or rate–quality result is asserted here. Record new job IDs, exact revisions, inputs, hardware, selected patch identities, budgets and results in a subsequent dated entry before waiting. Previous ledgers and evidence remain read-only.

The [first source/exposure inventory](submission-ladder-source-inventory-20261114.json) audits tracked metadata only, with actual SHA256 identities of cited repository records. It retains all 30 prior-exposed TrackNet label identities and 34 prior-used VISOR windows/source-video hashes, and records RacketVision frame joins and new-video eligibility as unresolved. It selects no new clips, authenticates no remote data, and creates no labels. Existing VISOR exclusions are not an eligible holdout list. The human-reviewed SAM path remains preparation until actual source-only decisions are imported.

### 2026-11-14 — implementation checkpoint

The initial native fallback profile and `python -m src.pointstream {encode,decode,inspect}`
interface are implemented; [endpoint documentation](../codec-endpoint.md) states
its constant-cadence, BT.709 and decoded-window limits. It is not yet a semantic
foreground/background endpoint. ROI/patch/ball integration awaits its design gate.
The source-only annotation tool supports model-overlay and blind review,
polygon replacement, failure retention and review-bound persistent drafts.
No actual annotation batch has been selected or reviewed; browser visual QA
was attempted but the browser tool timed out. Generated JavaScript behavior is
tested independently, including draft restoration and storage failure.

The authenticated [saved-scalar sensitivity](submission-ladder-anchor-sensitivity-20261114.json)
reproduces the primary historical comparison and adds Pareto/PCHIP and literal
linear-rate chord integration. Across the three geometries, ROI LPIPS BD-rate
is −76.97% to −74.63%; full LPIPS is +0.32% to +1.22%. The signs survive, while
magnitudes depend on construction. This is arithmetic on accepted saved scores,
not a fresh metric, native decoder or GPU experiment.

A fresh temporary environment installed the configured project dependencies:
Python3.12.7, NumPy1.26.4, OpenCV4.10.0.84, SciPy1.17.1, Ruff0.11.2,
Mypy1.15.0 and Pytest9.0.2. Configured Ruff and Mypy pass (608 source files).
The full local suite returned 2407 passed,42 skipped,one failure: sandbox denial
of `ps` in the existing process-suspension test. That exact test passes with
process-inspection permission. No tracked baseline errors were patched to hide
an environment mismatch.

The final endpoint's native integration test is skipped locally because
Homebrew ffprobe/ffmpeg cannot load libx265.215.dylib. An earlier, less strict
native roundtrip passed SVT4.0.1/dav1d1.5.4 before native-probe validation was
added; it does not qualify the current endpoint, pinned SVT4.2.0, or a scientific
receiver. No global installation was repaired and no replacement provider was
used. OS-enforced source denial and frozen dynamic-library/runtime identity
remain required before promotion.

The stage1 scientific allocation remains proposed/unreleased, with no fleet
jobs, GPU usage or new training. Source/exposure inventory, actual annotation
pilot and real production-path cost measurements remain open. Prepare their
exact source manifests before asking for that stage's release. Manuscript690cad0
and all completed raw evidence remain unchanged.

### 2026-11-14 — twelve-hour autonomous research release

The user explicitly instructed repair of FFmpeg, actual footage selection, annotation review preparation and ROI/patch experiments, and authorized autonomous budget approval for twelve hours. The [authority and allocation ledger](submission-ladder-autonomy-20261114.json) expires November15 at11:07:34UTC (08:07:34America/Sao_Paulo). Within this window root may release bounded costed stages without requesting approval again. This supersedes the earlier unreleased-stage language for newly recorded allocations; all scientific/device/input gates and the45-minute stage bound remain. Human annotation decisions still require the human reviewer; an agent cannot approve reference labels on their behalf. No recurring monitoring is created.

FFmpeg8.1 and ffprobe8.1 now start after restoring access to the already installed x265 ABI215 library, preserving the ABI216 default. All twelve native endpoint tests pass locally, including the previously skipped native roundtrip. This qualifies the tested local correctness path, not SVT4.2.0 production/GPU scoring or an empirical codec win.

Tennis input preparation submitted once as `20261114T232159Z-4846011d`, frozen11c35c9e/spec572fbcc/sourcec281ff7c,2400CPU/0GPU ceiling, deadlineNovember15 00:30UTC. First16nativeframes/masks and publication/cost gate precede64; this is preparation, not codec quality evidence. Inspect this ID on any uncertainty; never replay it.

First tennis smoke4846011d stopped before native preparation because the selected source archive omitted `tools`, an imported dependency. Canonicalstatus isFAILED; full did not run. Retain full2400CPU conservative charge and error log; repair source closure by including reviewedtools in a new request. VISOR eb4865c0 was physically verifiedUNPUBLISHED: only events/source.tar, no spec/ready, so it cannot run. Retain its receipt/artifacts and2400CPU conservative preparation allowance. Its original source path failed the external-root precheck; locate an existing eligible copy or create a fresh authenticated one before any replacement. Neither failure is a codec result.

Corrected tennis preparation submitted once as `20261114T232911Z-6e73f222`, same frozen11c35c9e and exact scientific inputs/spec, source91c2d668 now includes tools closure. Ceiling2400CPU/0GPU; predecessor failure remains charged. Independent pre-score review also strengthened primary anchors to literal nativeIVF cost and prohibited prefix-quality/full-window-rate promotion. Shared-wrapper curves remain explicitly secondary.

Corrected tennis preparation6e73f222 completed smoke/full and physical validator:64frames, native raw199065600bytes SHAb50beb54, source masksSHA0d478396, archive199260160bytes SHAee2a6150. Measured full preparation/publication2.19s passes600s; full2400CPU conservative charge retained. This is accepted source/mask preparation only. VISOR replacement4223e78c uses the new independent read-only data-root source copy, original contentSHAunchanged, source91c2d668, specabe10357,2400CPU/0GPU ceiling. Its predecessor remains verifiedunpublished and preserved.

The native/patch pilot uses literal nativeIVF anchor bytes, complete candidatepackets, complete64-frame coverage for promotion, and a secondary shared-wrapper sensitivity. Hash-bound preparedarchives now stage on hostlocaldisk directly. Scoring records source-self metric controls, CUDA network/device identities, memory, allframevalues and63 temporal pairs. The new annotation pilot is [ready for humanreview](submission-ladder-annotation-pilot-20261114.md):12sourceframes/5matches/3blind, no new labels yet. Root expands the pilotCPUceiling21600→36000s under12hauthority to retain failures and budget full native/receiver/scorer lifecycles; GPUceiling7200s unchanged. No previous failed charge is refunded.

Both source preparations are accepted64-frame inputs. VISOR4223e78c full preparation/publication5.67s passes600s, rawSHA15cc7ab1, maskSHA2cd5d387, archiveSHA766265d3. Its complete2400CPU charge is retained. Actual tennis native producer `20261114T234255Z-4e9c77aa` now uses frozenc68baf30/specb9024bfe/sourcef72e2c18,4800CPU/0GPU ceiling, deadlineNovember15 01:00UTC. It runs the first16 correctness/cost gate before64; there is no quality/scientificwin yet.

VISOR native producer `20261114T234622Z-874978b6` submitted once on disjointCPUhostpoolgpu2/gpu5 (tennis usesgpu1/gpu3/gpu6), frozen c68baf30/spec7b077cdb/sourcef72e2c18,4800CPU/0GPU ceiling, deadline01:00UTC. Canonicalwaitmax2400s, receipt `/private/tmp/pointstream-ladder-visor-native-wait-20261114.json`; no native model-resume wakeup guarantee is assumed. Both producer specifications enforce16→64 correctness/cost gates and freshpacket replay.

The two first nativeproducer requests4e9c77aa/874978b6 were verifiedPENDINGwithnohost/stage, then ROOTcancelled and preserved. Host-level `required_commands` incorrectly included dav1d, which exists in the packed runtime but not workerPATH. No nativeworkload or GPUran. Return each never-spent4800CPUhold; administrativepreparation remains covered by readiness, no earlierchargedfailure is refunded. Corrected submissions require hostffmpeg only; productionqualify_tools still enforcesSVT4.2.0/dav1d1.5.x and exact runtime/resourcehashes. This is an admission repair, not changed scientificsettings.

Host-admission corrected producer requests: tennis `20261115T000304Z-56cba37f`/spec5315b7af and VISOR `20261115T000301Z-2c3ec7a1`/spec4123b368, same sourcef72e2c18/frozenselectedcode, HEAD e8fc9ba1 documentationcheckpoint. Each holds4800CPU/0GPU,deadline01:30UTC. Original pendingrequests remaincancelled/ACKed and their never-spentworkloadholds returned. Inspect these newIDs oninterrupt; no replay.

The actual16frame producer commands for56cba37f/2c3ec7a1 passed internalpacketchecks/cost but fleetauto-publication replacedtheir thinarchives. Bothsmokevalidators failed and fullneverran. Keep4800CPUeachcharged; no qualitynegative follows. Theba1b1340 privateworkspace/lifecycle regression fixes onlypublication. Correctedtennis `20261115T001337Z-3ea86d31` isphysicallyaccepted64frames/12rows/28thinmembers withallROIpointsactive, sourceb50beb54, archive2,641,920B SHA62dfc7c5; commandscopedCPU69.62/wall56.64 (notwholecoldlifecycle/speed). NewVISORproducer `20261115T001900Z-c260a64e` and tennisCUDA scorer `20261115T002959Z-92d68496` remaininspectable; hold4800CPU0GPU and2430CPU2430GPU respectively. GPU2managementSSHrefused; supportedroutegpu6works andallworkersfresh. No replay/restart. Onecanonicalbatchwait coversc260a64e/92d68496, max2400s.

Baselineboundary frozen before quality inspection: this fastpreset8/fullresolution/singleROIoffset20 pilot is a finite mechanism screen. An apparentgain requiresstrongerconventional controls (preset4 full/halfresolution envelopes and ROIoutside40 where supportrequires), plusqualifiedDCVC-UF, before any submissionheadline or distinct semanticmechanismclaim. Do not mistake a developmentthreshold for baselinecompleteness.

### 2026-11-15 — accepted tennis quality and frozen next test

Full64-frame tennis scoring `20261115T002959Z-92d68496` is accepted against producer `20261115T001337Z-3ea86d31`, ba1b1340/source377a3499, exact sourceb50beb54 and reference SAM masks0d478396. Score archive389120B SHAb51d9991, producer2641920B SHA62dfc7c5. Shared gpu5 Ada UUID `GPU-7754164a-33cd-265c-3696-c20966d228fe`, exact packed32e68872 and authenticated LPIPS/DISTS resources; source-self controls, full receiver hashes, all64 frame arrays, means, region handling and63 temporal pairs passed. This is quality/correctness evidence, not speed or population confirmation. Retain2430CPU/2430GPU ceiling charge; command-scoped wall170.12s/CPU169.38s excludes lifecycle overhead. [Saved scalar scores](submission-ladder-tennis-scores-20261115.json) and [recomputable analysis](submission-ladder-tennis-analysis-20261115.json) preserve all outcomes.

Primary hull/PCHIP, actual nativeIVF anchor cost: ROI weightedDISTS **−8.53%**, weightedLPIPS **−7.19%**; semantic patch **+17.10%/+27.94%** versus native and **+28.53%/+37.61%** versus ROI. Error patch loses **+54.05%/+65.23%** versus native. Every arm misses the predeclared development promotion test. Semantic placement improves against error placement (−17.78% DISTS/−16.08% LPIPS), a narrow selection finding; both remain worse than native. Even free placement/framing for identical semantic pixels loses16.76% weightedDISTS. This closes only this static256×256/fixedCRF28 construction on this exposed development window, not all possible object coding. The next test, fixed before examining further quality, is the already preregistered unchanged egocentric64-frame construction. Do not retune on tennis. Stronger conventional/neural anchors remain prerequisites for any future positive submission claim.

VISOR producer `20261115T001900Z-c260a64e` accepted64frames/12rows and actual ROI changes, exact source15cc7ab1/masks2cd5d387; published6594560B SHAeeea2925, ba1b1340/source377a3499. Retain4800CPU charge. Command-scoped41.59wall/55.98CPU; no speed inference. Its unmodified quality stage receives2430CPU/2430GPU under the existing36000CPU/7200GPU ceiling. All43RacketVision test rallies were previously used for tracker selection, so newly joined native clips are development conditions, never untouched selection holdouts; original-source deduplication remains unknown.

Independent scalar/implementation review reproduced the saved analysis and all138 score checks. It found no blocking issue for the scoped negative decision. Semantic-vs-error comparison is objective alignment (foreground occupancy versus unweighted full-frame luma error), not an isolated semantic-representation advantage. Linear-rate sensitivity still loses7.88% for semanticpatch while primary log-rate PCHIP loses17.10%; ROI full-frame DISTS changes sign across geometries. No general full-frame or experience benefit is claimed. The next VISOR scorer is `20261115T005151Z-a8933c78`, source377a3499/specf93738b3,2430CPU/2430GPU,deadline02:00UTC. Its shared-device contention event is preserved/acknowledged under the predeclared continue policy; no replay or speed claim.

A second metric audit found that the pilot’s `full_lpips` means the **mean full-resolution LPIPS-Alex spatial map**, not mathematically identical to the official `spatial=False` scalar (nonuniform bilinear averaging weights). Existing records remain unchanged and explicitly retain that definition; weighted regional decisions are unaffected. Add a separately named standard full-frame scalar before conventional full-frame LPIPS claims. Full-mask DISTS agrees mathematically with standard DISTS, modulo reduction order. This metric correction must not be presented as a new winning result or silently rewrite historical scores. Configured Ruff and mypy pass; full pinned-environment tests now **2441 passed,37 skipped**, one existing multiprocessing fork/thread deprecation warning.

### 2026-11-15 — egocentric result and stopping decision

VISOR score `20261115T005151Z-a8933c78` accepted, source377a3499/ba1b1340/specf93738b3, exact source15cc7ab1/masks2cd5d387/producerc260a64e. Score389120B SHA4c97b5c5; sharedgpu6 Ada UUID `GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553`. All138 independent scalar checks pass, all64 frames and63 temporal pairs retained; self-controls0LPIPS/5.96e-8DISTS. [Scores](submission-ladder-visor-scores-20261115.json) and [analysis](submission-ladder-visor-analysis-20261115.json) preserve arrays and geometry sensitivities. Retain2430CPU/2430GPU conservative charge; scoped288.81wall/288.16CPU is contention-contaminated cost, not speed.

WeightedDISTS/weightedLPIPS versus literalnativeIVF: ROI **−5.39%/−4.52%**, semanticpatch **+18.47%/+23.55%**, errorpatch **+101.77%/+114.95%**. Semanticpatch versusROI **+25.56%/+29.44%**; samepixels freeframing still costs18.24% weightedDISTS. No arm meets10% development promotion. Close this frozen additive rectangle construction on both tested development windows. Do not retune size/CRF/selection on them or spend a confirmation cohort on an unpromising candidate. This is not a universal lower bound for jointly coded, dynamic or other representations. Ordinary ROI gives descriptive regional benefits, not a PointStream novelty claim.

The mechanism allocation closes at conservative33660CPU/4860GPU seconds charged,2340CPU/2340GPU never released; failures stay charged. This is ceiling accounting, not measured fleet occupancy. Under the continuing12h authority, a separate3600CPU/3600GPU standard-metric contract stage is released: same accepted packets/receivers, new explicit scalarLPIPS only, frozen legacy weighted results checked for invariance,450s smoke→900s full/60s validator within1530s perdomain. This fixes measurement naming before paper use; it cannot rescue the failed construction. Next scientific priority is the existing compact-ball representation's missing matched-objective conventionalROI control, subject to exact source and full native-framing audit, rather than another player generator or broad qualification stack.

### 2026-11-15 — matched-ball competing explanation released

The [new ballROI card](../cards/submission-ladder-ball-roi-2026-11-15.md) freezes threeSVTpoints35/48/62, preset4, corrected+40outside/0inside source-label64crop blocks on existing48frames. This supersedes only the intake’s proposed48/55/62/+20 settings before any ROI outcome. Allocation7200CPU/3600GPU is new under12hauthority; oldprogramme budgets remainclosed. Actual goldYUV/native62/freshdecode parity precedesGPUscoring; no newtracker qualification/training or dataset acquisition. It tests a missing simple competing explanation, not a new object architecture. The original nativeIVF price audit passes: accepted690/e431 already used90949/63879/34146B; no correction or sign reversal to+1.2158968% fullLPIPS follows. The [pricing receipt](submission-ladder-ball-price-basis-20261115.json) retains the labeled608B-framing counterfactual separately.

The explicit metrics-v2 implementation passes configured Ruff/mypy (619files) and the full pinned suite **2444passed/37skipped**, one existing fork/thread deprecation warning. It adds official scalar `full_lpips_standard`, same weights/preprocessing, and restores the spatial setting even on failure. V2 validation rejects missing/invalid standard arrays or definitions while retaining explicitly identified historical profiles. Frozen v1 regional comparisons/decisions reproduce unchanged; no v2 GPUmeasurement is accepted yet. New ballROI files are a separate unfinished workstream and are excluded from the standard-metric source checkpoint until reviewed.

Accepted41ee’s exact native CPU contract was re-read: cpu_threads1/single_cpu_campaign=true/index0, actualsmoke/fullaffinity[0], PS_CPU_ALLOWANCE1. SVT `--lp6` is an algorithm parallelism level, **not six allocated threads**. The new ballROI CPUstage adopts the same affinity and1530CPU ceiling (450smoke+900full+60validator+120lifecycle); unused native allowance moves to explicit preparation/failure reserve, total7200CPU unchanged. Matching source/command without matching this affinity would not establish old bit-exact control. GPUmetric correction uses separate affinities2/3.

V2 standard-metric jobs submitted once: tennis `20261115T010800Z-1cfef16e`/spec00af063f and VISOR `20261115T010834Z-13806d69`/spec82cf1597. Frozen70340ffc/source9dc45f1f; each1530CPU/1530GPU, deadline02:30UTC. Same accepted physical packets/source/metricweights, independent legacy-array parity required. State and singlecanonicalbatchwait preserved at `/private/tmp/pointstream-standard-metrics-v2-runner-state.json`; inspect these IDs oninterrupt, neverreplay. New unfinishedballROIcode is excluded from this committedselectedsource.

### 2026-11-15 — standard metric contract accepted; regional decisions unchanged

V2 jobs1cfef16e/13806d69 both accepted at64frames, source9dc45f1f/70340ffc. All150 scalar/device/control checks pass per domain, with all11580 legacy frame/mean/temporal values exactly equal to preservedv1 (maxdifference0). Tennis scorearchiveSHA847616b2 on gpu5/UUID7754164a; VISOR SHA2fd2c14f on gpu6/UUIDef96fc4f. [Tennisv2 scores](submission-ladder-tennis-standard-v2-scores-20261115.json), [analysis](submission-ladder-tennis-standard-v2-analysis-20261115.json), [VISORv2 scores](submission-ladder-visor-standard-v2-scores-20261115.json) and [analysis](submission-ladder-visor-standard-v2-analysis-20261115.json) retain new scalar values separately. Maximum perframe standard-vs-map LPIPS difference6.93e-7/1.28e-6; no weighted outcome changes.

OrdinaryROI standard fullLPIPS **costs+1.15% tennis/+0.64% VISOR** while supplying the previously reported scoped regional savings. Paidsemanticpatch standardfullLPIPS costs58.44%/27.76%. These are objective-specific trade-offs, not broad fidelity wins. Stagecloses conservative3060CPU/3060GPU charged,540eachneverreleased; oldfailuresremaincharged. Actualcommandscoped172.39wall/171.63CPU tennis,263.63wall/263.02CPU VISOR excludeslifecycle; sharedcontentionprecludesspeedclaims. Both terminalevents ACKafterROOTacceptance. V1 evidence is independently archived122members/455813120B SHA853f587b withverifiedreadback0444/0555; modesare notOS-enforcedimmutability.

Independent CPU and scientific review admits only the bounded real ballROI smoke: original48 JPEG/YUV/native62/decode gates, actualaffinity[0], corrected+40ROI, completeIVF prices, newprivatepublicationlifecycle and freshreplay. GPUmode is conditional on nativeacceptance and all48historicalmetriccontrolparity. ConfiguredRuff/mypy621files and fullpinnedsuite **2455passed/37skipped**, existingfork/threadwarning, pass. Separate scalar source-self and all-networkCUDA checks were added during finalreview; focusedchecks remain required before the frozen source launch.

BallROI nativeCPU job submittedonce `20261115T012257Z-5055d064`, frozen142b51e3/source87d314b3/spec64eaa7f1,1530CPU/0GPU,deadline02:30UTC. Actualgpu6 singlecoreaffinity0 matchesold41ee. Oldlegacyjob ismissingfromthisclientcatalog but its sharedstatewasindependently readCOMPLETE; immutableartifactidentity andallgoldsource/nativechecks remainmandatory. No oldjob replay/re-registration. Singlecanonicalwait/handle at `/private/tmp/pointstream-ball-roi-native-runner-state.json`; inspectnewID afterinterrupt. GPUscoring remainsconditional.

BallROI CPU5055d064 ROOTacceptedfull48, archive348160B SHA588a928e/6literalhash-boundmembers, stageSHA7480df6b. Exact48sourceYUVa40b172b/native62cd79e397/decode652302be andall21internalfreshchecks pass; ROI62 actualpixelsdiffer. Fullentrypointbody17.2435wall/17.1853CPU excludesimports/lifecycle; outerexecutioncumulativeCPUvectors are not summed or treated as new measured stageCPU. CompleteIVF bytesCRF35/48/62=134971/76309/28658, not qualityevidence. Conservative1530CPU charged andterminaleventACKed.

NewconditionalballROI score `20261115T012841Z-7ce10c5b` submittedonce, same142b51e3/source87d314b3/speced576b37,2430CPU/2430GPU,deadline03:00UTC. First16 scores still freshlydecode48 and mustreproduceall48historicalnative62metrics beforeROIvalues; no relaxation/replay on failure. SharedqualifiedAda, oneCPUaffinity1, no speed claim. Singlecanonicalwait and jobstate preserved; currentjobIDmustbeinspectedafterinterrupt.

### 2026-11-15 — ball benefit survives the first matched ROI competitor

BallROI scorer7ce10c5b ROOTaccepted: full48/all21checks, exactoldnative62RGBSSE andallperceptualmetricarrays (maxdifference0), selfcontrols8.39e-12LPIPS/5.96e-8DISTS/0standardLPIPS; Adaef96/gpu6, peakreserved1799356416B. Scorearchive153600B SHAb05d359d/memberb213a4ed. [Rawscalars](submission-ladder-ball-roi-scores-20261115.json) and [SVT-onlyanalysis](submission-ladder-ball-roi-analysis-20261115.json) retain allpoints, actualcompleteIVF/PSBL prices andgeometrysensitivities.

Paidstreak versus privilegedGTROI: cropLPIPS **−32.29%**, cropDISTS **−16.48%**, usingcommon-qualityhull/PCHIP. FullspatialLPIPS **+3.80%** is interpolation-sensitive (literal-linear −11.11%); no generalfullframebenefit. This is48allvisibleexposedframes, onefixed+40ROIcontrol, notconfirmedgeneralization/absence/task/newcodec-levelclaim. The originalpooledneural/nativeheadline mustberecomputedwiththe newROIpoints andmatchedintervals beforepaperuse; do not substituteSVT-only32.29% for the oldpooled76.97% result.

Thisstagecloses3960CPU/2430GPU conservativecharged,3240CPU/1170GPU neverreleased. Body37.565wall/37.408CPU excludesstartup/lifecycle; outerchildrenCPUvectors arecumulative andnotsummed. Sharedcontention meansnospeedclaim. Newdistinctnexttest is the [actualcoded-pixel finiteoracle](../cards/submission-ladder-ball-pixel-oracle-2026-11-15.md), preregisteredbeforeitsoutcomes: samequantizedsendertrajectory,32pixelregionpadded64tile, onepatchCRF20/preset4, threeexistingbasepoints, freeplacement/framinggrant. It is not a universalLPIPS/DISTSbound and does notbuildpaidtransportyet. New4500CPU/3000GPUallocation undercontinuing12hauthority; oldbudgetsclosed.

### 2026-11-15 — strengthened pooled ball reference and honest intervals

The [new pooled analysis](submission-ladder-ball-roi-pooled-analysis-20261115.json) authenticates originalprimaryscalarSHA071dd85e andnewrawROIscoreSHAb213a4ed, exactlyjoinsallfiveSVTmetric/physical-price triplets, then adds3measuredROIpoints to6oldneural/nativeanchors. Same6paidcandidatepoints. PrimaryPCHIP savings now **−60.95%cropLPIPS/−36.23%cropDISTS**; original76.97%/44.44% referenced aweakeranchor anddifferentoverlap. On the sharedoriginalqualityinterval, newreference savings are **−71.07%/−39.24%**. Do notattribute thewholechange tobitallocation: qualitysupportextends. Whole-mapLPIPS remains+1.11%,DISTS+3.28%,RGBPSNR+1.11% ratecosts. All3geometrysensitivitiesretained; PCHIPslopescanchange whenhullknotsareadded, so thefit isnotaproof ofphysicallydominating switching. No newmodel/encode/scoring occurredinthispostprocessing.

The [ballarchive](submission-ladder-ball-roi-evidence-archive-20261115.json) nowpreserves ACTUALoldproducer1.3278GB andscorerpayloads plusentire2.564GB sourceZIP andnewmatchedROIoutputs:4135086080B/81members, SHA72b24b4a, verifiedallhashesbeforecopyandafterreadback,47.577s/0GPU,0444files/0555directory. Historicalproof-only5c11wasnotmistakenforpixels. Standardv2supplementalarchive0de0e3d6 has42members/16363520B, readbackaccepted. Fulloriginalinputsandoperationalcode references remainavailableforre-encoding; modesarenotOSimmutability.

Papertablesnowcomefrom `python -m experiments.codec_ladder.report --record ANALYSIS SCORES ... --out NEW_DIR`: rawscorehashverification, independentanalysisrecomputation, duplicate-sourceprotection, officialscalarfullLPIPScolumnandexplicitunsupportedcells. CSV/TeX/provenance aregeneratedwithoutGPU; nohandcopiedheadlinenumbers. TeXsourceisnotyetamanuscriptrenderorvenueformatverification.

PixeloracleCPU/GPUimplementationpassedindependentgeometry/source-lineage review and6focusedtests; all-networkCUDA/device-count/memory<=8000MiB gates and8192margin were reconciled beforeGPUadmission. SecondarypatchIVF-framing relaxation was frozen beforefirstencode, alongsideprimaryfullbase+patchIVFprice. ConfiguredRuff/mypy627files andfullpinnedsuite **2465passed/37skipped**, oneexistingfork/threadwarning, pass. The neworacle isstillunexecuted: CPU/gold/nativecostgate precedesGPUcontrol, and paidtransportrequirescompetitivenesswithcompactstreak inadditiontonativegain.

PixeloracleCPU submittedonce `20261115T020136Z-f5dd76c2`, frozen09b03b4b/source3d815ed4/spec09576da3,1530CPU/0GPU,deadline03:00UTC. Currentgpu6fresh/headroom93 and priorCPU/ROIscoregatescomplete. Representative16→48 withnativegold/sourcebeforeactual64tileencoding; noGPUadmissionfromcodechecks. State/singleboundedwait `/private/tmp/pointstream-ball-pixel-native-runner-state.json`; inspectthisIDoninterruptneverreplay.

### 2026-11-15 — coded pixels challenge the compact-rendering claim

Actualpixeloracle26806765 ROOTaccepted17checks/full48 andnative62allmetricarrays exact(maxdifference0), Adaef96/gpu6, peakreserved1799356416B, source3d815ed4/09b03b4b/spec9a4ba5c6. Originalarchive153600B SHAc9da533b/member00e2885b. [Unchanged rawscorebytes](submission-ladder-ball-pixel-oracle-scores-20261115.json) and [analysis](submission-ladder-ball-pixel-oracle-analysis-20261115.json) retainunsupportedfields andprice-relaxations. Source-specificplacementisfromactualserializedencoderPSBL, notGT.

Free-placementactualcodedpixels save **48.29%cropDISTS versusnative,31.53%versuspaidstreak**. CropLPIPS hasnoqualityoverlap: worstpixel0.01513985849 isbetterthanbeststreak0.01568543191/bestnative0.03069898277, andrate-supportoverlap exists(native40640–90949B/streak40640–91272B). This isstrictqualityrangeseparation, **notLPIPSBD-rate**. Preserve programmednative_gate=null; ROOTadmits paidtestunderpredeclaredDISTS+improvingLPIPScorroboration usingthe documented strongerordinaldirection, withoutmanufacturingaBDfigure. FullframeLPIPS/DISTS costs10.17%/12.42%vsnative remainvisible. RemovingpatchIVFframingstillpreservesthesamepixels anddoesnotcreateaqualityoverlap.

Thiscloses3960CPU/2430GPUconservativecharged,540CPU/570GPUneverreleased. Actualscorebody38.6644wall/38.4958CPU excludeslifecycle;sharedtimingnorealtimecredit. The [newpaidPTP1card](../cards/submission-ladder-paid-tracked-pixel-2026-11-15.md) releases7500CPU/0GPU foractualcosts, samepixelreuseonlyafterallframehashparity, egocentricnativefallbackandboundedfilesystemdenial. A successfulfreegeometrycontrol isnotacompletecodec. The compactrenderer’sdistinctperceptualadvantagehasnot survivedthiscontrol; source-independentpaidversionandtask/temporalbehaviorremainseparatequestions.

### 2026-11-15 — confirmation sources reserved; annotation truth remains a gate

[Source-onlyconfirmationintake](submission-ladder-confirmation-intake-20261115.md) reservesP35_105175–238/P12_101217–280/P11_103229–292: threeinstalled1080p50fullBT709sourcesfromparticipantsoutsideprior34. AllhadpriorB1bmetadata-onlysurvey; HIntTRAIN/VALsourceoverlapandanonymousTESTuncertaintyaredisclosed,candidate-specific. No confirmationcodec/modelqualityoutcomeopened.

P35has26frameswithnodenserecord; P11hasmanual/denseobjectomissionsatnative229/279; P12’sintermediatevisibilityflagsarenotproofobjectsareactuallyvisible. Do nottreatmissingclassesasemptyforeground. Fixedwindowsremainreserved,notshiftedtoeasierones. New18-source-framepilot(officialmanualproposalsatanchors,denseelsewhere,unknownsblind) receives2400CPU/0GPU undercontinuingauthority; humanmarking/approvalstillpending. No datasetreplacement/downloadneeded. Correctallrequiredreferencesandfreezefinalcandidatebeforeconfirmationmetrics.

### 2026-11-15 — paid profile frozen for bounded CPU qualification

PTP1 adds a shared public raw-YUV encode / packet-only RGB24 decode / inspect path with paid per-frame placement and both complete IVF streams. Cached genuine automatic PSBL positions are explicit encoder inputs, not hidden receiver state or a fresh tracker timing claim. All-absent placement omits the patch. The accepted pixel oracle admits the paid test by crop-DISTS and strictly improving LPIPS range; its literal unsupported LPIPS-BD gate remains null. Full 48-frame byte/frame parity is required before any saved score reuse. Egocentric fallback tests function only.

Configured Ruff and Mypy (636 files) pass. Full pinned tests pass: 2,505 passed, 37 skipped, one existing fork/thread deprecation warning. An earlier sandbox-only failure denied the existing fleet test's `ps` call; the permissioned rerun passes. PR418 exposed a separate report portability issue: verify SHA256-bound scores exactly, normalize JSON arrays, and allow only negligible float interpolation roundoff while decisions/structure stay exact. The corrected table test passes; CI must confirm Linux portability.

The independently reviewed reused Landlock adapter and bounded one-to-four-case bridge are implemented, with exact protected-file identities and archive-proof bindings. Host qualification remains pending; fresh process separation alone is not called source denial. Tennis and EGO CPU specs each cap 1,530 seconds, zero GPU, deadline 05:00 UTC; no job has yet been submitted at this freeze.
