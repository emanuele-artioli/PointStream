# POINTSTREAM — state of the work

*Written 2026-08-21, at the boundary between Phase A and Phase B. Read this
first if you are picking the project up cold.*

---

## What this project is

An object-centric semantic video codec. The encoder transmits semantic
understanding — each salient object's appearance and motion, a background model —
plus an optional corrective residual; the client reconstructs frames with
generative models. Tennis is the current domain. The target is an ACM TOMM
resubmission on **September 30**.

The current cycle is a **full rewrite into a configurable research platform**:
every axis — codec, detector, pose, segmenter, generator, appearance and motion
representation, background method, residual, transport, metric, and the task
domain itself — becomes a config choice, and the only code a new component needs
is the wrapper that satisfies the agreed interface.

## The three documents, and what each is for

| Document | Job | Where |
|---|---|---|
| **The blueprint** | What we are doing next: architecture, component specs, phases, priorities. | `~/.claude/plans/i-have-been-away-binary-steele.md` |
| **`src/contracts/`** | The machine-checkable truth. If code and prose disagree, this wins, because CI runs it. | this repo |
| **The paper** | The conceptual record and the long-term trace of what is done and what is left. | `67a9ea6275d3d9785ce57026/` (a **separate git repo**, Overleaf sync) |

Keeping them separate is deliberate and was learned the hard way: the blueprint
had absorbed related work, literature review and measured results, which made it
unusable by the agents it exists to drive. Results go to the paper. Secondary
findings go to **appendices** — one `.tex` file each under
`appendices/`, `\input{}` from `main.tex`. That is the release valve: a finding
worth keeping but not load-bearing gets a sentence in the main text and its
substance in an appendix, with no budget on how many.

**Claim discipline, non-negotiable.** No unmeasured quantitative claim appears in
the paper. Every designed-but-unproven mechanism carries a `NOTE()` or `HOLE()`
marker; a `CLAIM(id): src=` line appears only when a real `outputs/` path backs
it. Right now exactly one `CLAIM` line exists in the whole manuscript.

## Where things stand

### Done

**Phase A — `src/contracts/`, complete and green.** Errors, registry, capability
vocabulary, layers and their import check, keypoint schemas, codec ladder,
conditioning, metrics, object stream, domain profiles, stage lattice, strict
config parsing, and the run configuration. 190 tests, ruff and mypy clean.

The property worth preserving: **the package imports nothing heavy.** No torch,
no cv2, no ffmpeg. A whole configuration validates on a machine with none of the
backends installed. Several design decisions exist to keep that true — tensors
are described by a structural `ArrayLike` protocol rather than imported, and
registry targets are `"module:attr"` strings resolved lazily.

**The paper's concept sections.** Related Work (five new subsections), System
Design (seven), Future Work, and the first appendix. Reviewer themes 3 and 8
closed on text; six others advanced.

**The paper purge.** Unsupported abstract claim removed with a `HOLE` naming the
experiment that would restore it; the rejected ACM MM submission archived behind
a README; `RESEARCH_LOG.md` split into Standing and History so retracted numbers
cannot be read as current.

### Next — Phase B, seven parallel component streams

Each owns one axis package exclusively. See the blueprint §7 for the table and
§4 for per-axis specs. B3 (generation) is the largest; start it first.

Two things Phase A left wired but unused, deliberately:

- **Per-axis registry modules are pre-created empty**, so parallel streams fill
  their own file without contending on a shared table.
- **`config.validate_backends`** is the third validation pass. It takes
  registries as arguments — so `contracts` keeps importing nothing heavy — and
  checks that every named backend exists and that the chosen appearance/motion
  pairing has a generator able to decode it. It is a no-op until registries are
  populated. Wiring it is each stream's job.

## Gotchas discovered the hard way

**Encoders lie by omission.** SVT-AV1 accepts `-pix_fmt yuv444p`, returns
success, and emits yuv420p. Every residual encode since that knob was added
silently requested a format it never got. `contracts/codecs.py` now declares what
each encoder honours and rejects the rest — extend that table rather than
trusting a flag.

**Two builds of the same encoder is a trap.** The conda environment carried
`svt-av1 1.4.1`, shadowing the system 1.8.0 on `PATH`. Only 1.8 has
`--roi-map-file` at all, so testing the wrong one reads as "region control does
not work" for reasons unrelated to region control. The conda package has been
removed; ffmpeg here comes from the system, not conda.

**Matched QP is not matched rate.** Our own first ROI measurement compared two
arms at the same QP, so the region arm also spent more bytes — more bits buying
more quality is not a result. `codecs.assert_matched_rate_control` encodes the
rule; the outstanding work is a matched-*bitrate* comparison. A prior project
lost a published table to the harsher version of this.

**Bound before believing.** State a plausible best and worst case *before*
reading a result. When our ROI bound fired an alarm, the bound was wrong — it
had been derived in QP units when AV1's offsets are q_index (≈4 per QP step) —
not the measurement. Record the alarm and the reason either way.

**Weight symlinks go stale.** Seven links under `assets/weights/` were dangling,
including every YOLO26 default the config names, because the models moved into a
`Models/YOLO/` subdirectory. Ultralytics silently auto-downloads replacements
into the repo root when this happens. Repaired; a check that every weight a
config names actually resolves belongs in the behaviour suite.

## Tests

**Do not add tests to raise a coverage number.** The old suite has ~2,100 lines
in files literally named `*_coverage*`, and the percentage gate that motivated
them is being replaced by a **required-behaviour suite**: a named list of
properties that must hold, which cannot be satisfied by padding.

The ~436 old tests are untouched and test modules Phase B and C delete. They die
with their modules; no separate culling exercise is needed.

## Working environment

- `conda run -n pointstream --no-capture-output <cmd>`; imports are absolute
  from the repo root (`from src.contracts... import ...`).
- GPUs are shared and assumed available; more than one server can be used.
- Long jobs run detached with hourly checkpointing.
- `outputs/` and `assets/` are gitignored. Cite paths; never paste contents into
  the paper.
