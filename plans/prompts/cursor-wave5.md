# Prompt for Cursor — wave 5, streams C, D, E

Three independent streams. **Run them in separate worktrees**, not in one
checkout — two agents in one checkout share one HEAD, and this project has
already lost work that way. Paste the relevant block below the line.

Set each up as:

```bash
git worktree add -b wave5/<name> /home/itec/emanuele/pointstream-w5-<x> origin/main
cd /home/itec/emanuele/pointstream-w5-<x>
rm -rf assets outputs && mkdir assets
for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -s /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -s /home/itec/emanuele/pointstream/outputs outputs
```

---

## Common to all three streams

You are working on PointStream, an object-centric semantic video codec targeting
ACM TOMM on **30 September** (35 days out).

**Read first, in order:** `/home/itec/emanuele/.agent-rules/AGENTS.md`,
`/home/itec/emanuele/.agent-rules/harness/cursor.md`, this worktree's
`AGENTS.md`, `plans/WAVE-2026-08-26.md`, then **only your own brief**. Do not
read the whole plan tree — it does not fit, and a session that needs all of it is
scoped too broadly.

**Rules that code cannot enforce:**
- **Bound before believing.** Write a plausible best and worst case with its
  reasoning *before* reading any measured number. A result outside that range is
  an alarm to investigate, not a finding to report.
- **A flag existing is not a feature working.** Drive the option and measure that
  the output changed the way the option claims.
- **Never add a test to raise a coverage number.** A test that exists only to
  execute lines makes the gate lie.
- **Report what happened.** If a run failed, say so with the output. If a step was
  skipped, say that.

**Environment:** `conda run -n pointstream --no-capture-output <cmd>`; imports are
absolute from the repo root. Before you finish: `ruff check`,
`mypy --config-file pyproject.toml`, `python -m src.contracts.layers`, and the
tests for what you touched.

**Two host quirks that cost wave 4 real time:**
- `conda run` **swallows pytest's summary line**. Use
  `python -m pytest -p no:warnings --junit-xml=<file> -q` and read the counts from
  the XML. A piped exit code is **not** evidence a suite passed.
- Anything over ~10 minutes runs **detached in the background**, never in a
  foreground poll loop. Confirm a process is actually dead with `ps` before
  relaunching it.

**Never `git add -A`** in a worktree — it commits a spurious
`D assets/weights/.gitkeep`. Add explicit paths. Write results only under your
brief's own `outputs/` subdirectory; `outputs/` is shared with every worktree.

Open a PR when green, and **confirm CI is actually green before saying it is** —
wave 4 merged a red lint because nobody re-checked.

---

## Stream C — `plans/BP22-test-boundary.md`

Finish the cull BP15 started. It removed 213 of 433 pre-rewrite tests and
stopped; **220 remain in 32 top-level `tests/test_*.py` files**.

**Your first deliverable is a decision, not a deletion.** BP15's premise —
"only three modules are still imported by new code" — is now **twelve** inbound
edges, and BP14 built `src/shared/training/stop.py` *inside* the tree BP15 is
deleting. "Delete `src/shared`" and "keep the stop rule" cannot both be true.
Pick (a) `src/shared/` becomes a real layer with a contract, or (b) it stays
condemned and `training/` moves out first. Write the choice into `PLAN.md` §3
before touching a file.

**Hard constraints:** do **not** touch `src/shared/tennis_dataset.py` or the
training path — another stream is live on `scripts/train_controlnet.py` this
wave. Keep `src/pipeline` and `src/runner` importing nothing from `src/shared`
or `src/decoder`; two other streams depend on that staying true.

**Port, don't drop:** if a pre-rewrite test covers behaviour the new tree has and
does not test, port it and say which.

---

## Stream D — `plans/BP26-config-plumbing.md`

**Unblocks P0 item 4, the ablation lattice.** BP23 drove all 32 config fields one
at a time and **27 changed nothing**. An ablation on detector, pose, appearance,
motion or temporal policy would currently produce a table of identical numbers.

Start by reading `outputs/bp23-tier/inert-config-fields.json`. It scopes its own
claim honestly — generation knobs are inert *in that corner* because generation
was off, which is a statement about the corner, not the knob. **Your first job is
to separate genuinely-unwired fields from corner-inert ones, with evidence,
before fixing anything.**

Then wire the axes the lattice needs — detector, pose, segmenter, appearance,
motion, temporal policy. `src/components/` already registers backends on all
sixteen axes and 48 of 52 construct: this is **binding, not building**. A field
is done when changing it changes an output and you have both numbers.

**You own `src/runner/routing.py` and the stage factories — but NOT the codec
stage.** Another stream owns `make_codec` and `STAGE_CODEC` in
`src/runner/stages.py` this wave. If you collide there, they win and you rebase.

The all-off control must stay bit-identical, and every disabled stage's call
count must stay 0.

---

## Stream E — `plans/BP27-metric-invariants.md`

Small and self-contained. Two metrics in this project were broken until
2026-08-23 and **every engine ranking before that date is void**. BP23 found two
more instrument limits that currently live only in a JSON file:

- **VMAF's ceiling on this content is 97.54, not 100**, and it **floors at 0.00
  for both severe blur and an unrelated clip** — nothing resolves below its floor.
- **LPIPS's ordering inverted at 960×540** and held at 4K. Anchors do not transfer
  across resolution.

Pin both as invariants beside `tests/invariants/test_metric_calibration.py`.
**Assert the absolute scale, not just the ordering** — an ordering-only check
would have passed both of the metrics that were broken here. Put each metric's
usable range in its docstring, so nobody can quote a number without its scale.

Three or four real properties beat twenty assertions. Say what you deliberately
did not cover. If an invariant fails on arrival, that is a finding — report it,
do not tune the threshold until it passes.
