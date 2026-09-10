# Paper Area

**Evidence Revision**: Paper `55e4bc4`; code Gate A/B audit through PR #85, 2026-09-09.
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
- **Latest validation (2026-09-09)**: paper commit `55e4bc4` builds successfully with the Gate A/B audit caveat. The previous 30-page/9-page-appendix status was stale; the current complete PDF has 27 pages. Recheck body and appendix boundaries before allocating final-result space.

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
| `PAPER-ACT-03` | Ready | None | Manuscript budget | **Page-budget recheck**: Measure current body and appendix boundaries before final result integration; trim only if needed. The old 9-page appendix diagnosis is stale. Acceptance: body/references ≤23 pages and appendices ≤5 pages. |
| `PAPER-ACT-04` | Blocked | Gate A passed | Gate A | **Result figures integration**: Render validated rate–distortion curves and amortization plots after rechecking available space. Acceptance: Vector figures embedded with exact run provenance cited in comments. |

Gate A/B audit: `sections/evaluation.tex` now explicitly retains AV1/VVC, distinguishes native and planned resolution-adaptive comparisons, and states that the pilot pass claims do not establish competitive confirmation. No result HOLE was cleared. Next: `EVAL-ACT-06` integrity repair, then `EVAL-ACT-07` matched-rate/quality development search; integrate final results only after a valid frozen confirmation.

PR #88 audit (2026-09-10): no new result is ready for manuscript promotion. Its matrix controls and model ranking require reruns, and its codec comparisons remain uncertified. Keep existing result HOLEs; use the [submission dispatch](../workflow/session/submission-search.md) to obtain source-free, fully accounted and independently confirmed evidence before new quantitative claims.
