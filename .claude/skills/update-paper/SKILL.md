---
name: update-paper
description: Fold new POINTSTREAM findings (run results, diagnoses, retractions, dead ends) into the paper (67a9ea6275d3d9785ce57026/main.tex), guided by its GOAL/HOLE/CLAIM markers, and into RESEARCH_LOG.md. Use after experiments complete and results are committed/tested, or when a conclusion changes. Replaces the retired update-reports workflow.
---

# Folding findings into the paper

The paper is the primary living document. Its comment markers
(`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):` — full spec in the paper repo's
`CLAUDE.md`) record each element's goal, missing data, and provenance.
`RESEARCH_LOG.md` (paper repo) is the secondary store for knowledge that
cannot live in the manuscript.

## Procedure

1. **Locate.** Run the discovery grep in `67a9ea6275d3d9785ce57026/`:
   ```
   grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex
   ```
   Find the anchors this finding touches; read the surrounding text and the
   file's `STATUS` header. Note that System Design, Evaluation, and Conclusion
   are currently `HOLE`s for *entire missing sections* — a finding may be the
   first data to land in one.
2. **Classify the impact:**
   - **Fills a `HOLE`** → proceed to write (steps 3–4).
   - **Contradicts a standing claim** → find its `CLAIM` provenance line;
     decide rewrite vs reframe-as-ablation vs cut. Never leave a stale number
     standing. Add the old version to `RESEARCH_LOG.md`'s Superseded registry.
   - **Not paper-relevant** (an infra bug, a dead end, a rule) → one entry in
     `RESEARCH_LOG.md` (Bug registry, Dead-end registry, Hard rules, or the
     append-only Log), stop.
3. **Gate the claim** before wording anything — the hard rules in
   `RESEARCH_LOG.md`, in short:
   - A component is justified **only** by the Residual Guarantee: Δresidual >
     Δmetadata against the Whole-Frame Residual Baseline.
   - Scope negative results to what was run. "Conclusively"/"definitively" is
     banned on single-clip, single-architecture experiments.
   - Label the config (`num-frames`, backend, codec/preset); a 10-frame smoke
     number is labeled as one. Say whether it's a single run or a sweep.
   - No generative quality claim unless the model was trained without
     `alcaraz_highlights` and `djokovic_zverev`.
   - Verify the knob you ablated is actually wired — grep its consumer, not
     just the config schema.
4. **Edit via the `paper-editor` agent** for substantive text (it knows the
   layout and the claim rules). Hand it *verified* numbers and the exact
   `outputs/` paths — check them against `run_summary.json` yourself first;
   past reports mis-summarized their own numbers. **Clear the `HOLE` and
   write/update the `CLAIM(id): src=<outputs paths> date=` line in the same
   edit** — a HOLE may never be cleared without its data landing in the text.
5. **Cross-update:** if a referee item advanced, update
   `67a9ea6275d3d9785ce57026/reviewers_comments.md` (Status + a concrete
   Resolution naming the section/markers; Done only when text or experiment is
   actually in place). If a Standing-results entry just landed in the text,
   remove it from that queue in `RESEARCH_LOG.md`.
6. **Verify:** no local TeX on this host — check balanced braces/environments
   in the edited file, then push and let Overleaf compile.
7. **Commit** the paper repo naming the anchor ids and the `outputs/` paths.
   Commit the code repo separately (it is a different git repo).

## Guardrails

- No number without an `outputs/<timestamp>/` path (or an
  `outputs/_superseded/` path, explicitly flagged as superseded — check the
  Superseded registry before citing anything older than a week).
- Invalidated runs get `mv`'d to `outputs/_superseded/<ts>_<reason>/`, never
  `rm`'d.
- Never edit `main_old.tex` or `backup/` — historical reference only. Do not
  copy a result across from `main_old.tex` without re-deriving it.
- Markers are comments; they are invisible to reviewers and stay that way.
- Camera-ready sweep: before final submission the discovery grep must return
  only `CLAIM` lines.
