# Data Area

**Evidence Revision**: Reconciled through PR #63 (`d26a27e`) and PR #72.
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
| External data root | #32 (`d436b02`), #35 (`6c4fa10`) | Isolated `.ps-data-root` resolver; no symlinks in repo. |
| Acquisition audit | #56 (`14fb05a`), #57 (`82c9bf3`) | Validated diagnostic corpus manifests and integrity hashes. |
| Domain candidates | #60 (`e14ea9d`) | Screened secondary domain candidates for Gate D. |
| Long sequences | #63 (`d26a27e`) | Characterized 96- and 192-frame tennis sequences for background amortization. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `DATA-ACT-01` | Ready | None | #63 | **Long sequence manifest packaging**: Package manifests and frame bounding boxes for 96- and 192-frame tennis scenes. Acceptance: Verified clips available to `load_tier_clip` without missing frame annotations. |
| `DATA-ACT-02` | Proposed | None | #60 | **Secondary domain eligibility screening**: Screen candidates (surveillance, conferencing) for camera stability and foreground isolation. Acceptance: Documented eligibility report and rights-cleared sample manifest for Gate D. |
| `DATA-ACT-03` | Previously deferred (BP46) | `DATA-ACT-01` | `plans/BP46-long-tennis-scenes.md` | **Camera pan boundary verification**: Verify background plate stitching across camera sweeps. Acceptance: Zero canvas seam artifacts on extreme pan sequences. |
