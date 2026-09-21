# Paper Area

## Modular rate ladder victory integration and visual strip — 21 September 2026

Paper commit `fc65fb8` (pushed to Overleaf `origin/main`).
1. **Section 4.2 (`subsec:eval-ladder`) Updated**: Obsolete 8-frame failure text replaced with verified modular rate ladder victory numbers (Presley plate $B=5.8\text{ kB}$, steered cropped actor residual $R=4.1\text{ kB}$, Rung C1 winning against VVC QP47 by 65.4% at 192f and 32.5% at 48f).
2. **Table & Strip Integrated**: Table `tab:modular-rate-ladder` and Figure `fig:modular-strip` (`figures/modular_rate_ladder_strip.png`) embedded.
3. **Markers Cleared & Pinned**: Cleared `% HOLE(subsec:eval-ladder)`; pinned `% CLAIM(subsec:eval-ladder): src=outputs/modular/rate_ladder/results.json commit=7966827 date=2026-09-21`.
4. **Strict Budget Adherence**: Compiled PDF strictly adheres to the 28-page ACM TOMM limit (**exactly 28 pages**: 23 pages body/references + 5 pages appendix).

## Pilot return audit — 16 September 2026

Paper commit `18d4778` records E05 missing-control/deployment evidence,
E04B open alarms and the scoped E06 stop in status markers. Rendered TeX is
unchanged; no result HOLE cleared and no alarmed metric promoted. Overleaf main is verified at `18d4778` after the explicitly authorized push.
The current PDF remains 28 pages; final result panels must replace material.
Follow [the reuse assignments](../workflow/session/evaluation-campaign/tasks/20260916-return-audit-and-reuse.md).


## Coordinator setup preparation — 16 September 2026

Paper baseline `55e4bc4` renders to 27 pages. Separate paper branch
`codex/campaign-setup-20260916` adds receiver reference-policy requirements, data
separation, full-wire accounting, common-grid scoring and immutable reproduction
records. Result HOLEs remain open; no new quantitative result is promoted.
Updated paper renders successfully to 28 pages: body/references end on page 23,
appendices occupy pages 24–28. No page slack remains. Approximate prose counts
put method near 40% and evaluation near 30%; final result panels must replace
material rather than simply extend it. Existing overfull-box warnings remain.
Paper commit `7a0476e` is pushed to Overleaf main (parent `55e4bc4`). Neural-anchor readiness and source reservation are prepared
in the evaluation/data areas; execution remains unreleased.


## Current campaign — 14 September 2026

E08 follows the [campaign](../workflow/session/evaluation-campaign/plan.md): setup,
anchors, backgrounds/removal, foreground/training, residual, assembled codec and
domain limits. Setup/structure and accepted component prose may proceed early;
final advantage claims require confirmed evidence. Once a neural foreground
model clears its baselines, prioritize writing over optional model improvements.
Code-side figures use one eligible-result reader; manuscript remains a separate
repo. The September 20 evidence freeze is provisional, submission September 30.

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
