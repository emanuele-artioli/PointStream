# Plan index

Start with `../PLAN.md`. `ROADMAP.md` controls priorities; read only the assigned
brief. Archived reports describe what happened, not what to run next.
The current dispatch entry point is `DISPATCH-GATE-A-48FRAME-RUN.md`.

## Active

| Brief | Current purpose | Harness |
|---|---|---|
| `GATE-A-LONG-CONTEXT-2026-09-05.md` | Frozen Gate-A search brief: bounds, ladder, identity, timeout and stop rules | Authority / Codex |
| `DISPATCH-GATE-A-48FRAME-RUN.md` | Dispatch prompt and run report for the bounded 48-frame native run | Dispatched session |
| `PAPER-NEXT.md` | Scope, provenance, page budget and final evidence | Antigravity; Codex review |
| `SUBMISSION-READINESS-2026-09-05.md` | Current evidence, blockers and deadline decisions | Authority / Codex |
| `BP46-long-tennis-scenes.md` | Candidate long eligible scenes and manifest rules (D1) | Antigravity / Cursor |

## Next gates (parked until Gate A/B win)

- `BP41-ablation-lattice.md`: core component ablation matrix after freezing the regime (Gate C).
- `BP36-second-domain.md`: independent public sequence/domain evaluation (Gate D).
- `BP34-operating-point.md`: profile and optimize speed of the frozen winner (Gate D).
- `BP37-required-behaviour.md`: remaining behaviour-suite invariants audit.

## History and archived work (`plans/done/`)

Completed reports, superseded briefs, and historical diagnostics live in `plans/done/`:
- `BP56-background-encoder-effort.md`, `BP56-background-effort-report.md`: completed; PR #63 merged.
- `BP57-confirmation-acquisition-pilot.md`, `BP57-acquisition-report.md`: completed; PR #64 merged.
- `BP55-timing-boundaries.md`: implemented in PR #65/#66 (`encoder_seconds`, `client_seconds`, `evaluation_seconds`).
- `BP45-ultra-low-rate-search.md`: superseded by `GATE-A-LONG-CONTEXT-2026-09-05.md`.
- `BP38-paper-infrastructure.md`: superseded by `PAPER-NEXT.md`.
- `BP39-all-off-corner.md`: implemented in PR #45 (conventional fallback control).
- `BP32-rate-budget.md`, `BP33-span-amortisation.md`: historical payload and span diagnostics.
- `BP40-background-honesty.md`, `BP43-background-representation.md`: background reporting and representation diagnostics.
- Earlier briefs `BP49`, `BP51`, `BP52`, `BP53`, `BP54` and wave reports.

`done/` contains completed reports and explicitly superseded or parked briefs.
It does not certify that every task in each document is finished. The old long
PLAN is `done/RESEARCH-HISTORY.md`. Validity warnings and retractions continue to apply.

`SESSION-REPORT.md` defines dispatch/report requirements. `TERMINOLOGY.md`
defines plain names. `DEFERRED.md` records additional deferred work and closed
findings. `ENGINE-ROSTER.md` records model/checkpoint status.
