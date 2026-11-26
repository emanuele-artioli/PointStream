# PointStream submission handoff

> **Active TOMM 2-week plan (2026-11-24):** See [docs/tomm-submission-plan-20261124.md](tomm-submission-plan-20261124.md) for the updated validation-first priorities, pre-flight V1–V4 audit findings, and two-track schedule superseding the legacy London 18-rally queue below.

Checkpoint from November 24, 2026 (Sydney). Start with this file, [PLAN.md](../PLAN.md), and [AGENTS.md](../AGENTS.md). The plan owns the priority list. Experiment cards own run specifications and evidence.

## November 26 review package

The active Overleaf revision is `976ec6bb2fcdeb711cf094e57ae896a6e66f74d5`.
The anonymous and named builds have 21 body/reference pages and five online
appendix pages. Rebuild with the paper repository's `tools/build_pdf.sh`.
For a portable anonymous source archive, use
`python3 tools/export_review_sources.py build/PointStream-review-source.zip`.
A clean ZIP rebuild matched the full anonymous PDF's extracted text. The
export contains only active sources and figures. It omits the two archived
broadcast plates and comparison video. Coauthors should review the named
copy before anyone files with ACM.

The [validation ledger](experiments/tomm-validation-20261124/README.md#verified-submission-evidence)
records complete Halle replay, corrected Shanghai pose inference timing, and
the controlled synthesis and reference experiments. The new evidence branch
retains the historical paths from main and carries the saved comparisons and
workers from the earlier submission branch. The paper scopes the automated
tennis ball study, annotation-conditioned cross-sport coding, and estimated
route counts separately. The current evidence package does not require new
GPU work. A continuous cold-to-warm broadcast evaluation remains a follow-up.
The [paid-pose control accounting](experiments/pose-sidecar-accounting-20261126/)
identifies two Shanghai video-plus-pose candidates under the warm byte ceiling.
An actual packet replay and machine comparison are still required.

## November 25 validation

The active manuscript is based on local revision `30afdee`. It compiled
on [Overleaf](https://www.overleaf.com/project/67a9ea6275d3d9785ce57026).
A download from the live project matched all 20 changed files and confirmed
the archival files were preserved. Both local PDFs use 21 body/reference
pages and five appendix pages, and the journal evidence checks passed.
The reviewer PDF and active-source archive omit the two broadcast crop
plates and the optional video. The quantitative results remain in the
manuscript. The RacketVision dataset card declares MIT and its paper
describes using YouTube broadcasts. The inspected sources supplied no
broadcaster permission for the specific imagery.

[PR #457](https://github.com/emanuele-artioli/PointStream/pull/457) and
[PR #458](https://github.com/emanuele-artioli/PointStream/pull/458) passed
Linux CI and merged at `6763d3a` and `a8408a2`.

- **Halle.** The follow-up fleet preflight failed host lookup on all six
  hosts. The [Halle recovery](experiments/halle-foreground-20261123/README.md)
  lists confirmed prior work and records the near-role stop rule. Reconcile
  the remaining GPU budget and reopen the far archive before launching.
- **Timing.** The previous exclusive spec covered one role and 48 frames
  and has expired. The [matched timing recovery](experiments/tomm-validation-20261124/v5-matched-240-recovery.md)
  records prior shared-load measurements, the bounded design and its
  required implementation. Segmented speed is distinct from uninterrupted
  native generation or complete receiver latency.
- **Gap and reference results.** Five-frame Sydney gaps met the coordinate
  gate for both roles. Ten did for the far role only. Run a paid native
  rendered comparison for five missing ticks against uninterrupted and
  held poses. Hold conditioning and crop support fixed. For Shanghai and
  Sydney AVIF Q90, the measured savings concern reference images. Compare
  identically conditioned lossless and Q90 24-tick renders before scaling
  that fidelity claim. Halle's replacement image bytes remain unmeasured.
- **Submission.** Selecting or initiating the ACM submission is the
  author's decision. Retain the no-broadcast package until the specific
  plates and comparison video have documented clearance.

## November 23–25 historical revision notes

| Repository | Verified revision and access |
|---|---|
| PointStream code and results | Sydney qualification is merged. The integrated overnight evidence and code are on `codex/match-specific-operating-ranges-20261123` at evidence commit `c403b6b4` in [PR #456](https://github.com/emanuele-artioli/PointStream/pull/456). Start from a checkout that contains it. |
| Manuscript | Overleaf project `67a9ea6275d3d9785ce57026`, revision `471d0eafa62ab81a9abb619e724038a3f5c28d42`. Read its own instructions. The shared draft contains both joint rate ladders, the twelve-match metric suite and the London coverage check. Evidence checks passed. The author PDF uses 23 body/reference pages plus five appendix pages, and the anonymous PDF uses 22 plus five. The paper is still a draft. |
| Data and results | Follow the paths and hashes in the experiment cards. The full clean rasters, native receiver witnesses, model environments and source footage are campus assets. The Markdown, checked-in receipts and manuscript are available with repository access alone. |

[GitHub PR and checks](https://github.com/emanuele-artioli/PointStream/pull/456) · [Overleaf manuscript](https://www.overleaf.com/project/67a9ea6275d3d9785ce57026)

Use the platform's existing GitHub, Overleaf and campus access. On a registered machine, reuse an existing execution record if there is one. Otherwise start from the revisions above and carry only the new scoped changes. The originating Mac checkout has unrelated uncommitted clean-court and DCVC work. Keep that checkout intact and leave those edits out of the handoff.

## What has been evaluated

| Evidence | Coverage | Use in the paper |
|---|---|---|
| Complete joint player-and-ball reconstructions with foreground scores | Four rally excerpts from four matches. Nadal–Kyrgios and Alcaraz–Hurkacz informed development. Shanghai (Sinner–Shelton) and Sydney (Safiullin–Shapovalov) are external cases, one excerpt each. | Two development shots support the matched low-rate comparison. Shanghai and Sydney support the two external rate ladders and their separate operating ranges. Extra rates reuse the same footage and player generations. |
| Ball-only matched control | Twelve matches and 402 independently marked visible instants | Broad ball enhancement evidence. Report the perceptual, pixel and whole-picture endpoints separately. |
| Expansion | Halle has a paid sender and one generated role. London has six ball/routing points and a short player feasibility candidate. | Partial results. The 18-rally joint cohort remains open. |

More matches are worth the cost for breadth. First use the cheapest completed or nearly completed shots with compatible inputs, actual receiver bytes and matched controls. Prioritize distinct events, then different prespecified points within each event. A qualified generated-player bank can serve several rate settings. Preserve conventional fallbacks in the denominator. See [development evidence](experiments/reviewed-foreground-20261122/README.md) and the [cohort design](experiments/broad-tennis-18-rallies-20261123/README.md).

## Historical execution queue

Use [PLAN.md](../PLAN.md) as the queue. **Halle is the first compute priority** because the sender and far-player archive exist and a complete third external example would add new footage to the central result. Inspect the far receipt and frozen near-role spec before dispatch. Recover the source-matched 640-wide AV1 and neural controls from saved evidence where possible. Then qualify the short native London player path and expand the prespecified cohort. A higher-rate Shanghai neural control is conditional on extending the stated comparator claim. Finish with the manuscript and replication audit. The previous overnight compute authorization has ended.

Check recorded fleet IDs with `scripts/ps-fleet status JOB_ID` before acting. For an existing registered agent task, capture `agentctl handoff TASK` using its recorded ID. Start the receiving harness from the pinned repository and manuscript revisions, with the same input hashes and a new scoped checkout. Follow [fleet rules](fleet.md) for new authorization, preflights and bounded wait periods.

## Redraw and prepare the replication release

| Figure or table | Versioned source for redrawing |
|---|---|
| External combined, players, and ball rate ladders | [Figure code, input receipts, command, and physical results](experiments/joint-operating-range-20261123/README.md#reproduce-the-figures). Run the command from the repository root with SciPy and Matplotlib, using a fresh directory. |
| Matched ball metric suite | [Twelve per-match score files, aggregator, and summary](experiments/ball-metric-suite-20261123/README.md#completed-twelve-match-comparison). The last paragraph gives the aggregation command. |
| Development foreground | [Figure inputs, replay receipts, and plotting command](experiments/reviewed-foreground-20261122/README.md). Use the committed aggregate for redrawing. |

Redrawing from committed measurements requires neither campus access nor GPU execution. Recomputing the measurements requires the archived sources, models, environments and native streams identified by their checksums and run receipts. For a replication release, export the plot-ready data, scripts, metric definitions, source and split identifiers, actual per-component byte counts, cache assumptions, environment versions and hashes of the native inputs and outputs. Audit sample images, labels, packet media, dataset terms and machine-local paths before making a public export. Describe any restricted inputs and how eligible researchers can obtain them. The internal repository is the current evidence archive. A public release has not been prepared.

The revision and stage in each receipt determine what the result can support. Use the figure's input list as the active evidence list. Keep legacy outputs with their original code, data fingerprints and job receipts. Mark evidence superseded for a specific figure when its replacement is verified. Preserve old paths and files, including failed or incomplete comparisons that explain a gap. A new run gets a new output path.

## Results to carry forward

- **Sydney.** The source-denied native replay reproduced all 240 archived frame hashes. Five wider physical replays reused generated players. Cached PointStream led the measured foreground DISTS comparisons through 189.076 kbps; DCVC-UF led at 191.730 kbps. First use of the highest shown PointStream rung cost 487.976 kbps for the four-second rally. This is a one-rally bracket. Last wider replay `20261123T010601Z-9342c742` has a completed receipt. [Native replay, scores and rates](experiments/sydney-joint-foreground-20261122/higher-rate.md)
- **Shanghai.** Two complete cached joint displays improve foreground DISTS at 13.54 and 17.78 kbps on the frozen sample. Three wider receiver replays and stronger mid-rate comparators extend the measured bank. Check the joint rate analysis for actual scores and remaining gaps. [Matched comparison and scope](experiments/shanghai-joint-low-anchors-20261122/README.md)
- **Broader tennis.** The twelve-match ball study and the London source review are reusable. London ball-only delivery, routing and accepted source proposals cannot stand in for an 18-rally joint claim or independent silhouette IoU. [Multi-match ball evidence](experiments/cohort-expansion-20261122/three-family-all-verified.md) · [London scoring and denominators](experiments/broad-tennis-18-rallies-20261123/source-marks.md)
- **Paper.** Lead with the foreground benefit at measured starved rates when appearance is cached. Put cold costs, fallback, background and temporal comparisons next to the corresponding wins. Keep direct ball coordinates and learned-player reconstruction time as separate measured endpoints. [Metric cross-check](experiments/joint-metric-crosscheck-20261123/README.md)
