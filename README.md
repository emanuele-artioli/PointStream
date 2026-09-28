# PointStream

PointStream is an object-centric semantic video codec. It represents video using background data, object appearance and motion, optional neural synthesis, and corrective residuals. **Broadcast tennis is the primary research setting; the egocentric demo is secondary.**

No confirmed end-to-end rate–quality advantage is established by the audited evidence. The current research compares PointStream with reproducible generative codecs as well as AV1/VVC, testing perceptual quality, preservation of tennis action, complete transmitted rate, and client computation. Candidate contributions remain hypotheses.

## Research documentation

Start with the [documentation index](docs/README.md). Its paper-shaped chapters cover the problem, related work and replication priorities, motivation, method, implementation, evaluation, experiment cards, and historical evidence. [PLAN.md](PLAN.md) routes to the current next steps; it is not a second campaign log.

The manuscript lives in the sibling repository `../67a9ea6275d3d9785ce57026/`, under its own instructions. Documentation drafts do not certify manuscript claims or experimental results.

## Working with the code

Use the Mac checkout for development and coordination. Follow [setup](docs/setup.md) for the pinned environment, external data root, native codec tools, and fleet admission rules. Dispatch CUDA through `python -m experiments.jobs.fleet`; inspect all six hosts before choosing one. Every extended experiment requires a passing bounded smoke through the same entry point. [Long-job operation](docs/workflow/long-jobs.md) describes detached supervision and provenance.

Dependencies are specified in `pyproject.toml` and `environment.yaml`. In an environment with project dependencies, a synthetic integration check is:

```bash
python -m pytest -q tests/runner/test_tier_end_to_end.py
```

Synthetic tests establish software behavior, not research performance. The [experiment plan](docs/research/07-experiment-plan.md) lists focused tests and qualification gates before model comparisons.

## Repository map

| Path | Purpose |
|---|---|
| `src/contracts/` | Configuration, observation, schema and component contracts |
| `src/components/` | Background, appearance, perception, motion and generation implementations |
| `src/runner/` | Stage execution, persisted client payload, accounting and delivered-frame scoring |
| `src/pipeline/` | Reconstruction, residual transport and metrics |
| `experiments/` | Dataset/evaluation protocols, probes, native ladders and fleet dispatch |
| `manifests/` | Versioned source selections, review and provenance records |
| `tests/` | Focused contracts and integration checks |
| `demo/` | Secondary demonstration and development tools |
| `docs/research/` | Current research narrative, evaluation protocol and experiment plan |

Keep datasets, weights and new run outputs outside the code tree. Do not introduce `assets/` or `outputs/` symlinks. Preserve unrelated work and use the repository [AGENTS.md](AGENTS.md) for branch, dispatch, review and reproducibility requirements.
