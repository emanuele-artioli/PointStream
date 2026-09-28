# PointStream current plan

Updated 28 September 2026. Primary domain: broadcast tennis. Target submission: ACM TOMM, 30 September 2026. The deadline does not relax evidence requirements.

The authoritative research plan is now [docs/research/07-experiment-plan.md](docs/research/07-experiment-plan.md), governed by [the prospective evaluation protocol](docs/research/06-evaluation.md). Start with the [documentation index](docs/README.md).

The next stage is bounded qualification: intake and verify saved artifacts/source exposure (E00), qualify metrics (E01), close dataset/client conditioning gaps (E02), smoke selected official baselines (E03), and verify PointStream's real generation-on receiver path (E04). These precede broad sweeps, training, and confirmation. No experiment was launched by the documentation rebuild.

MTTF, GLC-video, and GVC-RT are the first replication candidates; S²VC is a heavier optional arm. Keep AV1/VVC and one qualified DCVC variant as anchors. Availability in a public repository is not a successful local replication. [Related work](docs/research/02-related-work.md) records source links, reported baselines/results, and replication caveats.

The audited record contains no confirmed complete-codec win. Historical weighted-PSNR gates, exposed sources, failed controls, and retractions remain in [the evidence ledger](docs/research/08-evidence-ledger.md). Older instructions to skip confirmation or keep generation off do not govern this prospective plan; the generation-off runner remains a useful control.

[Setup](docs/setup.md), [experiment design](docs/workflow/experiment-design.md), and [long jobs](docs/workflow/long-jobs.md) remain operational references. All retired plans, areas and scorecards can be recovered exactly from [the retirement map](docs/history/documentation-rebuild-20260928.tsv). Legacy `BP*` section references in script comments describe historical provenance; they are not current execution instructions.
