# Data Area

**Evidence Revision**: Reconciled through PR #63 (`5ca87b2ee4`) and PR #72.
**Owned Scope**: Data acquisition, manifest definitions, eligibility criteria, sequence splits, domain definitions.

---

## 1. Current State

PointStream processes video sequences that exhibit salient foreground objects moving against a coherent, reconstructible background:
- **Primary Diagnostic Domain**: 4K broadcast tennis footage. Features camera pans, static court backgrounds, high player motion, and ball trajectories.
- **Data Isolation**: All raw video, intermediate frames, and large feature files live strictly outside the tracked git repository under `.ps-data-root` (see [docs/setup.md](../setup.md)). YouTube-derived source media is not redistributed.
- **Sequence Lengths**:
  - Initial tests: 8 frames (quick unit checks).
  - Gate A diagnostic: 48 frames (~2 seconds at 25 fps).
  - Extended amortization sequences: 96 and 192 frames (identified in #63 for Gate A competitive evaluation).

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| External data root | #32 (`d436b02`), #35 (`420c3bec4a`) | Isolated `.ps-data-root` resolver; no symlinks in repo. |
| Acquisition audit | #56 (`09727a4f16`), #57 (`77f30ecbe0`) | Validated diagnostic corpus manifests and integrity hashes. |
| Domain candidates | #60 (`f6f4f72cd9`) | Screened secondary domain candidates for Gate D. |
| Long sequences | #63 (`5ca87b2ee4`) | Characterized 96- and 192-frame tennis sequences for background amortization. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `DATA-ACT-01` | Ready | None | #63 | **Long sequence manifest packaging**: Package manifests and frame bounding boxes for 96- and 192-frame tennis scenes. Acceptance: Verified clips available to `load_tier_clip` without missing frame annotations. |
| `DATA-ACT-02` | Proposed | None | #60 | **Secondary domain eligibility screening**: Screen candidates (surveillance, conferencing) for camera stability and foreground isolation. Acceptance: Documented eligibility report and rights-cleared sample manifest for Gate D. |
| `DATA-ACT-03` | Previously deferred (BP46) | `DATA-ACT-01` | `plans/BP46-long-tennis-scenes.md` | **Camera pan boundary verification**: Verify background plate stitching across camera sweeps. Acceptance: Zero canvas seam artifacts on extreme pan sequences. |

## 4. Confirmation Protocol

The seven existing sources are development data, not a clean seven-video training partition: `manifests/bp46_long_tennis_scenes.json` records prior model fitting, sweeps, calibration, and panorama work, and has no accepted confirmation videos. The six-match target concerns new independent matches; six scenes from one match do not count as six sources.

Distinguish three operations:

- **Shared model training and codec selection:** use development data. Reserve validation data for hyperparameters and rate/quality choices; never choose the winner using final test scores.
- **Per-video encoding:** fitting a background or a video-specific model on the very frames being compressed is legitimate when it is part of the fixed encoding algorithm. Count transmitted parameters, references, metadata and adaptation time; a standalone decoder must not read source frames or untransmitted fitted state. Offline use of future frames must be declared and matched in comparisons. This is the setting used by [NeRV](https://proceedings.nips.cc/paper/2021/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf).
- **Generalization evaluation:** hold out at the level named by the claim. New matches require match-level separation. Disjoint scenes within every source can support a narrower “new scenes from known broadcasts” claim, but shared courts, players, lighting, and replay footage make them dependent. Use contiguous scene blocks, temporal gaps, replay/duplicate exclusion, separate validation and final test blocks, and source-level uncertainty. Do not retrospectively call already-inspected scenes untouched. Group separation follows the rationale in [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Recommended path: use all seven already-exposed sources for Gate A development and any shared-model fitting; preserve the fresh match candidates for final confirmation. Audit their actual prior use before accepting them. Optional within-source holdouts are supplementary, with the narrower claim explicit. If fresh independent sources are unavailable, record the deficit rather than silently weakening Gate B. A revised gate needs an explicit protocol decision plus matching manifest/verifier changes before scoring.

Grouped outer cross-validation can use each source for training in other folds and testing once, with tuning confined to inner folds. It costs multiple fits and cannot erase prior manual design exposure to these seven videos; here it is a robustness analysis, not fresh confirmation.

Next action `DATA-ACT-04` (Complete): audited candidate exposure and match identity, reserved source IDs and scene/time bounds in `manifests/gate_b_confirmation.json`, verified SHA256 integrity hashes, and launched Gate B confirmation under frozen procedure.
