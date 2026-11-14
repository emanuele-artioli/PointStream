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
