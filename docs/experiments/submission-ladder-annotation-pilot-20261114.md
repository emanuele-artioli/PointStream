# Experiments: source-only annotation boundary pilot

### 2026-11-14 — real 12-frame pilot prepared; human review pending

**Decision:** existing source-only K4b artifacts make a small, real SAM boundary
review possible without another inference run. This pilot tests annotation
workflow and boundary/racket completeness on exposed development footage. It
provides no untouched-confirmation, codec-performance or new human-label claim.

The [machine receipt](submission-ladder-annotation-pilot-20261114.json) freezes
12 source frames across five broadcasts, selected by source-group round robin
with SHA256-ranked IDs, with three blinded frames. Selection uses frame identity,
not mask quality, codec outcomes or returned boundary decisions. The groups are
Djokovic–Zverev, Alcaraz–Hurkacz, Medvedev–Struff, Djokovic–Federer and
Rublev–Auger-Aliassime. Source-video, conversion, JPEG, mask and proposal hashes
are retained in that receipt and the local package.

## Actual sources and boundaries

The original 92,293,120-byte K4b publication is authenticated by SHA256
`2dc9acb6541224f9f72af8ceb9c781308c3e62d19fcb819515736e4b7da261ee`.
The old source-only player-instance marks are authenticated by SHA256
`05f5130526254316c078003a3a66398c97aa33027ac56519e0f267429eba50ba`.
Those historical selections seed proposals only; they never become new
boundary annotations. Mask bytes and original review JPEGs come from that
publication, with no model execution, source-video decode or codec output.
Source JPEGs are the historical quality92 renders of the frozen converted
YUV frames, at native1920×1080 geometry; they are not pixel-identical native
video inputs. The frozen selection binds their exact original video/frame join.

The annotation target is the union of the actual players and their visible
rackets, excluding officials/crowd. The ball has separate labeled-point/task
endpoints. Existing person/player SAM prompts can miss rackets or player parts;
reviewers must flag and correct those, rather than approve identity alone.
Polygon corrections replace the entire union. Three blind frames hide the
initial mask and require a full human redraw or an explicit empty decision.
Rejected, missing and uncertain cases remain in the imported record.

## Review handoff

Durable local package:
`/Users/manu/Documents/PointStream-review/annotation-pilot-2026-11-14`.
Open `review/index.html` locally, or serve its `review/` directory with the
existing loopback server. The verified URL is `http://127.0.0.1:17691/`.
The source-only package is local and has not been published or redistributed.

At this checkpoint a task-owned server uses PID14388, wrapper session1944,
with a hard expiry of **2026-11-15 00:12:11 UTC** (**November14,21:12:11
America/Sao_Paulo**). It stops only its own child. The offline files and browser
drafts survive expiration. `/private/tmp/pointstream-annotation-pilot-20261114/server-state.json`
retains the exact command, start and expiry; no periodic model call is scheduled.

The review digest is
`d7aa8d744a938fc71c077c5305fffaf3aa3c9cf239470be1621983b062a6c788`.
Firefox visual QA verified a blind frame, navigation, an actual two-player
proposal and the annotation controls. No decisions, reviewer identity, polygon
or approval were supplied by the agent. Browser draft contains navigation only.
IAB initialization did not provide usable navigation; native Firefox succeeded.
The browser's elapsed time is dwell time, including QA/navigation, not measured
human annotation effort. Marking-time estimates remain unknown until user review.

Download decisions when all12 frames have been assessed, then import into a
fresh output directory:

```bash
python -m tools.datasets.annotation_review import \
  --review /Users/manu/Documents/PointStream-review/annotation-pilot-2026-11-14/review \
  --decisions /absolute/path/annotation-decisions.json --out /absolute/new/annotations
```

Only actually approved frames acquire reviewed-reference provenance; the
unreviewed propagated frames and historical labels retain their old tiers.

## Preparation and verification

`python -m tools.datasets.annotation_pilot` reproduces the export from the
three frozen archive/marks/selection inputs, authenticates every input and
selected mask, and refuses overwrite. A premature attempt while the archive
transfer was incomplete was rejected by the input hash guard before creating
an output; the original transfer completed without replay or segmentation.
The accepted export used1.933385 process CPU seconds and0.775191 wall seconds,
peak RSS184,516,608 bytes on macOS; no GPU/native encode/model inference.
These scope only the exporter, not whole administrative CPU accounting.
Three focused exporter tests and configured Ruff pass. Parent owns the final
configured Mypy/full-suite gate and aggregate readiness-stage accounting.
