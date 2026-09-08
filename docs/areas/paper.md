# Paper Area

**Evidence Revision**: Reconciled through paper commit `87d9e52` and code PR #72 (`bc09184`).
**Owned Scope**: Code-side evidence delivery (`outputs/`), publication tables/plots, coordination with manuscript repository (`67a9ea6275d3d9785ce57026/`).

---

## 1. Current State

The manuscript is maintained in a dedicated git repository at `67a9ea6275d3d9785ce57026/` with its own rules and marker conventions.

### Venue & Page Budget
- **Target Venue**: ACM TOMM (Transactions on Multimedia Computing, Communications, and Applications).
- **Submission Deadline**: **30 September 2026** (hard deadline).
- **Page Limits**:
  - Main text + references: **23 pages maximum**.
  - Appendices: **5 pages maximum**.
  - Total allowed: **28 pages**.
- **Manuscript Measurement (September 2026)**:
  - Current build: **30 pages**.
  - Main text: 21 pages (2 pages remaining for result figures and tables).
  - Appendices: 9 pages (**4 pages over budget**).
  - Action: Trim historical survey and background derivations in `appendices/` down to 5 pages.

### Headline Claim Governance
- A submission must demonstrate where an object-centric semantic video codec wins over conventional baselines.
- The headline claim must land in the regime where PointStream strictly wins; if no winning regime is established, that is surfaced early rather than papering over a loss.
- Secondary findings (such as generative engines failing to beat pasted references on PSNR) are documented transparently as empirical contributions.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Paper repository isolation | Initial setup | Manuscript separated into `67a9ea6275d3d9785ce57026/`. |
| Structural page audit | `SUBMISSION-READINESS` | Measured 30 pages total; established appendix reduction requirement. |
| Negative result disclosure | Paper Section 5 | Documented generative synthesis rate–distortion deficit relative to pasted crops. |
| Cross-repo permalinks | PR #73 | Historical code citations migrated to immutable commit permalinks. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `PAPER-ACT-01` | Ready | `CODEC-ACT-01` | #70, #71 | **Background plate multi-codec appendix table**: Benchmark table comparing `libaom-av1`, `SVT-AV1`, and `VVC libvvenc` across QPs on the 4K canvas. Acceptance: LaTeX table and caption formatted for ACM TOMM appendix. |
| `PAPER-ACT-02` | Ready | `CODEC-ACT-02` | #71, #72 | **Foreground crop codec appendix table**: Benchmark table comparing JPEG, WebP, and AVIF across quality levels for actor crops. Acceptance: Empirical size and crop PSNR table for appendix. |
| `PAPER-ACT-03` | Ready | None | Manuscript budget | **Appendix budget reduction**: Condense `appendices/` in paper repository from 9 pages to ≤5 pages. Acceptance: Clean LaTeX compilation under 28 total pages. |
| `PAPER-ACT-04` | Blocked | Gate A passed | Gate A | **Result figures integration**: Render rate–distortion curves and amortization plots into the 2 available main text pages. Acceptance: Vector figures embedded with exact run provenance cited in comments. |
