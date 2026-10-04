# 10. Preparation audit

Checked 29 September 2026 on the Mac coordinator. Documentation baseline:
`6d06767fccd35b494a21e3e6007a5fd58b8ee83d`; implementation remains the audited
`086ae9035bc0ff7b552c05eb04904224f9abb32a` except unrelated demo work excluded
from this task. This is a preparation record, not paper evidence.

## Objective coverage

| Requested item | Evidence and disposition |
|---|---|
| Recover the earlier discussion | Chapter 8 identifies the original thread and all three inspected pages; preserves tennis/PRESLEY/GenStream rationale and rejected JE claims. |
| Inspect project and paper | Chapters 4–5 map implementation; Chapter 8 records exact local manuscript and predecessor revisions, conflicts and limits. Remote Overleaf equivalence is not assumed. |
| Scoped branch and documentation replacement | Branch `codex/paper-documentation-20260928`; commits `a1113f1` and `6d06767` pushed. Fifty-six obsolete documents retired with Git blob recovery records. Operational guides and working code preserved. |
| Introduction and contributions | Chapter 1 states the problem, lineage and testable conventional/generative, component and reuse hypotheses. No unmeasured contribution is asserted as achieved. |
| Related work, reproducibility, anchors and results | Chapter 2 links primary papers/artifacts and distinguishes author results from local replication. Chapter 9 refines the shortlist using inspected codec entry points. |
| Motivating example | Chapter 3 preserves the altered-target headroom diagnostic with its limits and proposes a matched-rate tennis action-fidelity figure. |
| Method and alternatives | Chapter 4 explains representation, optional generation and residual, accounting, rigid-object semantics and failure boundaries; Chapter 8 preserves failed approaches. |
| Low-level implementation | Chapter 5 supplies source links, signed-residual equations, execution pseudocode, observation/wire contracts and test mappings. |
| Evaluation setup and existing results | Chapter 6 defines prospective sources, metrics, physical rate, access parity, ablations, resources and uncertainty. Chapter 8 reports historical results/retractions; there are no newly replicated generative comparisons. |
| Tests and experiments needed for the paper | E00–E12 specify hypotheses, controls, prerequisites, outputs, budgets and stop decisions. Chapter 9 identifies baseline-specific code gaps rather than pretending public inference is a complete codec. |
| Verification | Local Markdown links, referenced test paths, recovery blobs and whitespace checked. The contract run and its exclusions are recorded below. |

## Local software checks

The first three focused command groups in Chapter 7 were run together with the
then-listed calibration file under the default pytest configuration. Of 130
executed cases, 127 passed and three failed. The calibration file's invariant
cases were deselected by `pytest.ini`; listing its path did not execute them.

The default `/opt/homebrew/bin/ffmpeg` resolves to Homebrew 8.1 but cannot load
`libx265.215.dylib`. Its older 7.1_4 installation also fails, on
`librubberband.3.dylib`. Two failures were native residual transport/accounting;
the third was generated-output sensitivity after residual fallback.

An existing alternative worked without installation or global environment edits:

- Executable: `/Users/manu/miniconda3/envs/media_organizer/bin/ffmpeg`
- Version: FFmpeg 7.1.1
- SHA256: `aad96aab978868777cc5b5c0882df60a0567cbf5ef1fb8c4ce9e974051493933`

All three failed cases passed on rerun in 3.02 seconds with that explicit
executable. This isolates the observed failures to the native-tool environment
for these cases; it does not certify every codec or remote toolchain.

```bash
FFMPEG_BIN=/Users/manu/miniconda3/envs/media_organizer/bin/ffmpeg python -m pytest --override-ini='addopts=' -q tests/runner/test_residual_transport.py::test_coded_roundtrip_with_native_codec tests/runner/test_residual_transport.py::test_all_byte_ledger_reconciliation tests/runner/test_generation_appearance.py::test_generator_perturbation_changes_hashes_quality_and_residual
```

Explicit collection with `--override-ini='addopts=' --collect-only` found 22
metric-calibration cases. Collection is not execution. Chapter 7 now provides
an explicit calibration command; E01 must inspect skips and actual metric
controls before promoting a benchmark. No CUDA work, real-checkpoint baseline
inference, training, source-annotation study, or research sweep ran here.

## Remaining execution boundary

The documentation and test plan are prepared. Full scientific execution still
requires E00 artifact/source intake; E01 metric and annotation qualification;
E02 accepted dataset and decoded conditions; E03 real competitor qualification;
and E04 real PointStream receiver verification. Broad training and confirmation
remain dependent stages. These are substantive tasks, not boxes closed by this
audit. Completing the documentation must not be described as reproducing a
baseline or establishing a successful journal result.
