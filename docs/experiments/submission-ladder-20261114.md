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
