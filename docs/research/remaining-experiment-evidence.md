# Remaining experiment evidence, 1 October 2026

This integration branch combines the three explicitly authorized Sol 6.1 low-effort studies without changing the active Cursor checkout. Each study preserves its own measurement contract. No complete PointStream advantage is established.

## Results and contracts

- [Receiver qualification](receiver-evidence.md): 12 prepared E06 packages replay in fresh processes; four compact packages reproduce original decoded pixels while reducing physical package bytes. An RLE floor costs 17,581 B at pooled Y-PSNR 20.680 dB. Prepared RGB-source/cache identity is verified; native source extraction, matched native anchors and held-out exposure are not newly qualified. Final evidence branch `22429ff`.
- [Guided foreground transport](guided-transport-replay.md): three 16-frame 4K packages reproduce retained decoded pixels and source hashes. Background, generation, pose and correction are disabled. These foreground-only packages cannot be promoted to complete-video results. Final branch `22429ff`; fresh decoder `4748cbd`.
- [Task denominator audit](task-denominator-audit.md): 45 interventions on three retained model-reference caches, with nine full comparisons unchanged. The scorer now joins frame IDs and includes missing predictions in the supplied reference denominator. Controlled suffix removal exposes legacy denominator inflation; source joins, independent truth and exposure remain unqualified. Final branch `800dbaa`; frozen worker `7228db3`.
- [Consecutive fixed-region reuse](consecutive-region-reuse.md): 32 full native AV1 controls on two 1,798-frame intervals, plus 80 descriptive hold-prefix ledgers. Fresh manifests and independent container/probe checks pass. Cheap hold/refresh policies retain substantially lower original-region quality than continuous coding. A small favorable native reset scheduling observation is scoped to one fixed region; there is no complete-codec or matched-quality hold/refresh gain. Frozen worker `eb47a04`, independent summary/audit `4f8b099`.

Raw JSONs, exact receipt archives, pose inputs and all media remain outside Git. Compact evidence retains report hashes, exact code revisions, source identities and accounting boundaries. Infrastructure and synthetic scorer fixtures are not paper measurements. Server studies use CPU only, bounded threads, low priority and durable monitored execution. The post-run native audit's local rather than shared claim registry is disclosed in its study record; no cross-project exclusion claim is made for that audit. Completed own claims were verified released.

## Manuscript and validation

The coherent 22-page working paper (19 pages body/references, 3 appendix pages) is committed at Overleaf revision `215a976` on `codex/paper-remaining-evidence`, published to the existing project with a non-force push. It retains explicit missing-data markers and conditional paths to future qualified AV1 or generative-peer results. The original recovery dossier remains pinned at `21931f6`; these new measurements are separately identified.

Focused integration validation: 80 tests pass across receiver replay, denominator/scorer, consecutive-region and claim/fleet/monitor behavior. Manuscript checks pass for 11 TeX inputs, 23 citations and 56 labels, including arithmetic, receiver identities, physical ledgers, original replay/support bounds and task denominators. All 22 PDF pages were rendered and inspected; new tables and changed pages were checked at full-page scale. No overfull boxes or unresolved references/citations. Submission readiness is not certified.

Suggested review comparison: https://github.com/emanuele-artioli/PointStream/compare/main...codex/remaining-experiment-evidence . No pull request is opened.
