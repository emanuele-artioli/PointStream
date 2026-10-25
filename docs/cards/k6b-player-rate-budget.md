# Card: CPU-only player side-information rate-budget audit

Goal: decide whether the completed frozen 12-window player-rendering advantage can survive deliverable reference, target-matte and real keypoint transport. Use the frozen rule in the new CPU-only entry of `docs/experiments/k6b-players.md`; do not alter the prior diagnostic or its eight-GPU-hour ledger.

Scope: saved inputs/outputs and CPU code/rate measurements only; 0 GPU-hours, at most 2 measurement CPU core-hours / 3 h wall, four CPU threads, commands at most 30 min. No fitting, generation, GPU jobs, model/cohort changes, new chats or recurring monitors. Preserve all originals; new output paths only.

Stop: the cost/headroom decision is made; the stated CPU/wall budget expires; an essential prerequisite or user judgment remains unresolved. A positive configuration-specific gate permits only a proposed next smoke/pilot with a separate rule/budget, not its GPU execution.

Branch: `codex/k6b-player-confirmation-plan` (the completed diagnostic's PR #236 remains open). Report back to the requesting/planning chat `01a16cfc-1a88-7013-bd79-8489fe7c1eb2`.

Preserve source/replay uncertainty, two familiar match clusters, the granted-side-information limitation and historical step-5 failure. Lossy or decoder-derived side information cannot inherit old quality as a measured result.

State: rule frozen before new measurements; exact start/deadline and stage budgets are in the experiment entry.

Completed CPU-only fleet job: `20261025T082944Z-29f021a5`; source `dfc06164`. Inspect this ID before acting; no GPU job exists for this audit.

Transport job is terminal, with 219 full checks and independent archive decode passed. No audit jobs are active; final accounting and reporting are complete.

## Completed outcome

The tested lossless route fails the frozen 18/24 fixed-output bar. Actual keypoints plus a 7200 s PNG bank, even with free target mattes, retain 17/24. Adding the tested exact matte packets leaves 0 known wins (22 bracketed losses, two unresolved equal-rate cells); all 24 have a real cheaper/no-worse anchor witness. Exact matte alone, with all other side information free, is already 0/24. This is a configuration-specific bound, not a lower bound on all possible coders.

All 5718 native pose-map/crop pairs and 24 PNG references reproduce their saved inputs exactly. Approximate mattes and JPEG references change conditioning: their original-quality counts are counterfactual, never new measured wins. No complete deliverable route passes, so no GPU smoke/pilot is proposed. Source independence, familiar-match scope, granted-side-information caveats, historical step-5 failure and exploratory 115 px are unchanged.

CPU job `20261025T082944Z-29f021a5` complete; no active jobs. Recorded CPU 896.09 s; conservative native four-core bound 2501.68 s, within the 7200 s CPU cap and wall deadline. **0 new GPU-hours**; the old 6.995428306/8 h run remains closed.

Evidence `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-rate-budget-20261025T091041Z/cpu-rate-audit-evidence.tar.gz`, SHA256 `80d20f1838d3ae2dc2d39662201b64b6b313d57785b2cbc3cc5a3089d5c31376`. Complete measurements, per-cell residual margins, duration/base sensitivities and limitations are in the appended experiment outcome. Source analysis `b129ce04`; 11 focused tests and configured ruff/mypy pass. Stop condition reached; further work requires a new separately scoped question and budget.
