# B'26 — Make the config axes reach the run

**Unblocks `PLAN.md` §7 P0 item 4**, the core ablation lattice. Right now an
ablation on detector, pose, appearance, motion or temporal policy would produce
a table of identical numbers, because none of those names reach anything.

**Owns:** `src/runner/routing.py`, `src/runner/stages.py`, `config/**`,
`tests/runner/**`. **Read first:** `AGENTS.md`, `PLAN.md` §2.16 and §4,
`outputs/bp23-tier/inert-config-fields.json`, `plans/done/BP23-first-tier-run.md`.

**Does not own** `src/shared/**` or `src/decoder/**` (BP22 is deleting those) or
the codec stage (BP24 owns it). Coordinate through the wave plan, not by editing.

## The evidence

BP23 drove all 32 config fields **one at a time** and recorded which changed a
run. **27 changed nothing.** Only `evaluation.metrics` and four `residual.*`
knobs move anything today.

Read `inert-config-fields.json` before planning: it scopes its own claim
honestly. Generation knobs are inert *in that corner* because generation was
off — that is a statement about the corner, not the knob. Your job is to
separate the two cases:

- **genuinely unwired** — the name reaches no code at all; and
- **inert in that corner** — wired, but the stage was disabled by that config.

Report the split before fixing anything. The count that matters is the first
group.

## What to do

1. **Classify all 27.** For each: unwired, or corner-inert. Evidence per field,
   driven not read.
2. **Wire the axes the ablation lattice needs first** — detector, pose,
   segmenter, appearance, motion, temporal policy. `src/components/` already
   registers backends on all sixteen axes and 48 of 52 construct; this is
   binding, not building.
3. **Prove each one moves a run.** A field is done when changing it changes an
   output and you have the two numbers to show it.
4. **Delete or mark any field that should not exist.** A config key that reaches
   nothing and has no owner is a lie in the schema; better removed than left.
5. **Add required-behaviour tests**: for each newly-wired axis, changing the name
   changes the result. Not one test per field for coverage — one property per
   axis.

## Bounds — write to `outputs/bp26-config/bounds-before-run.json` first

- **Swapping a detector or pose backend should change the reconstruction
  measurably but not catastrophically.** State a plausible band per axis before
  measuring. A swap that changes nothing means it is still unwired; a swap that
  destroys quality means it is wired wrong.
- **The all-off control must stay bit-identical.** If wiring an axis perturbs the
  all-off corner, the stage is running when it is supposed to be disabled.

## Traps

- **A flag existing is not a feature working** — the rule this whole brief exists
  to enforce. Drive every option and measure.
- **Do not widen into Phase D.** Wire the axes; do not run the lattice.
- Every disabled stage's call count must stay 0, as BP23 established.

## Done when

- All 27 fields are classified with evidence, and the ablation axes are wired.
- Each wired axis has a test proving the name changes the result.
- The report states which fields were removed and why.
- `PLAN.md` §7 P0 item 4's blocker line is updated or removed.
