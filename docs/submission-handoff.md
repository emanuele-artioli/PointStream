# PointStream submission handoff

Checkpoint from November 23, 2026 (Sydney). Start with this file, [PLAN.md](../PLAN.md), and [AGENTS.md](../AGENTS.md). The plan owns the priority list. Experiment cards own run specifications and evidence.

## Resume from these revisions

| Repository | Verified revision and access |
|---|---|
| PointStream code and results | Sydney qualification is merged in GitHub main at `c52c9efabff6c4d209a356fe2cc8c14aa8be5085`. PR #454 passed CI. Use a fresh scoped branch from current main with this commit and the handoff present. |
| Manuscript | Overleaf project `67a9ea6275d3d9785ce57026`, revision `91562ebd18749661b301cdfd510ebf0b288e43d7`. Read its own instructions. The project already contains the Sydney revision and figures. The author and anonymous PDFs were built. The paper is still a draft. |
| Data and results | Follow the paths and hashes in the experiment cards. The full clean rasters, native receiver witnesses, model environments and source footage are campus assets. The Markdown, checked-in receipts and manuscript are available with repository access alone. |

[GitHub PR and checks](https://github.com/emanuele-artioli/PointStream/pull/454) · [Overleaf manuscript](https://www.overleaf.com/project/67a9ea6275d3d9785ce57026)

Use the platform's existing GitHub, Overleaf and campus access. On a registered machine, reuse an existing execution record if there is one. Otherwise start from the revisions above and carry only the new scoped changes. The originating Mac checkout has unrelated uncommitted clean-court and DCVC work. Keep that checkout intact and leave those edits out of the handoff. Prior Sydney time and resource limits have expired.

## First work session

1. **Audit Shanghai's archived inputs and controls before spending compute.** Read the [Shanghai comparison](experiments/shanghai-joint-low-anchors-20261122/comparison-final-neural.json), the [original sender receipt](experiments/test-crop-joint-20261122/match102-sender.json), and the [execution checkpoint](experiments/test-crop-joint-20261122/match102-execution-checkpoint.json). Identify the exact source, 240-frame receiver witness, paid player and ball payloads, clean-base identity, cold initialization and existing AV1, ROI and DCVC-UF controls. On campus, query any recorded active or uncertain job with `scripts/ps-fleet status JOB_ID` before scheduling. If campus is inaccessible, continue the receipt and comparator audit locally.
2. **Qualify one original native Shanghai replay.** Use the [Sydney parity procedure](experiments/sydney-joint-foreground-20261122/higher-rate.md) as a template. The existing wider-base opt-in was scoped to the 60-fps Sydney recipe. Shanghai needs its own pinned 29.97-fps profile and exact original packet and display checks on the native decoder before widening. Reuse the existing player witness and foreground bytes. The earlier Shanghai CRF60 sender packet has no qualified matched display. Its previous submission and failed Mac replay are recorded in the checkpoint.
3. **Fill the smallest rate gap that could change the paper figure.** Once parity passes, register a bounded broader-base CPU pilot near Shanghai's measured ordinary-AV1 bracket, 118.46–146.81 kbps against the existing bank. Recover compatible stored neural controls before encoding a missing one. Plot actual delivered bitrate, common support, separate player, ball and background scores, and cold versus cached costs. Report an observed bracket separately for each rally and metric. If parity or comparator availability fails, record the exact missing asset or check and leave the broad crossover claim open. Use the [fleet run protocol](fleet.md) for any dispatch and record each new job in its card.

After Shanghai, qualify variable-length joint replay on the shortest London point and expand the [six-match study](experiments/broad-tennis-18-rallies-20261123/README.md) with reused streams, counted fallback and a second independently held-out event. Then update the paper figures and verify the claim-to-result mapping. Keep the existing VISOR and table-tennis pilots in the appendix.

## Results to carry forward

- **Sydney.** The source-denied native replay reproduced all 240 archived frame hashes. Five wider physical replays reused generated players. Cached PointStream led the measured foreground DISTS comparisons through 189.076 kbps; DCVC-UF led at 191.730 kbps. First use of the highest shown PointStream rung cost 487.976 kbps for the four-second rally. This is a one-rally bracket. Last wider replay `20261123T010601Z-9342c742` has a completed receipt. [Native replay, scores and rates](experiments/sydney-joint-foreground-20261122/higher-rate.md)
- **Shanghai.** Two complete cached joint displays improve foreground DISTS at 13.54 and 17.78 kbps on the frozen sample. The higher-rate Shanghai joint curve is still open. [Matched comparison and scope](experiments/shanghai-joint-low-anchors-20261122/README.md)
- **Broader tennis.** The twelve-match ball study and the London source review are reusable. London ball-only delivery, routing and accepted source proposals cannot stand in for an 18-rally joint claim or independent silhouette IoU. [Multi-match ball evidence](experiments/cohort-expansion-20261122/three-family-all-verified.md) · [London scoring and denominators](experiments/broad-tennis-18-rallies-20261123/source-marks.md)
- **Paper.** Lead with the foreground benefit at measured starved rates when appearance is cached. Put cold costs, fallback, background and temporal comparisons next to the corresponding wins. Keep direct ball coordinates and learned-player reconstruction time as separate measured endpoints. [Metric cross-check](experiments/joint-metric-crosscheck-20261123/README.md)
