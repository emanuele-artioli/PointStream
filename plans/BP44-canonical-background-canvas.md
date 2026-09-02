# BP44 — Canonical background canvas for long compatible scenes

**Roadmap ID:** B1  
**Preferred harness:** Cursor for implementation; Codex/Claude reviews geometry
and the measured control.  
**Outcome:** predictive coding of background reference images must accept long
compatible scenes whose independently estimated panorama bounds differ.

## Read

`AGENTS.md`, `plans/ROADMAP.md` §§4 and 8,
`plans/TERMINOLOGY.md`, `plans/BP31-findings.md` §12,
`src/components/background/{plate,stream,strategy,types}.py`, and existing
background stream tests. Do not read the full plan tree.

## Owns

- `src/contracts/config.py` only for the smallest required context/canvas fields
- `src/components/background/plate.py`
- `src/components/background/stream.py`
- `src/components/background/strategy.py`
- focused tests under `tests/components/background/` and
  `tests/runner/test_background_stream_stage.py`
- a small diagnostic under `experiments/tier/`

If another file is required, report it before expanding scope.

## Design

The unit of reuse is a **background context**, normally one camera/view and
venue. Do not predict backgrounds across unrelated cameras merely because their
images have equal dimensions.

Predictive background-sequence coding treats the first scene's decoded
background reference image as an intra-coded reference. Later scene backgrounds
may be inter-coded against the previous decoded background, so the background
codec transmits its own prediction difference instead of another full
background. This is not the same as sending only PointStream's per-frame
correction signal for the first frames of the next scene: each scene still has a
decoded background, object data and any configured correction signal.

The canvas failure is a consequence of putting those scene backgrounds into one
predictive video sequence: video frames in the sequence need common dimensions.
It is not caused by keyframes themselves. The geometry also needs a common
origin, because equal-size images with different local coordinate systems would
decode successfully but reconstruct the players and background in the wrong
places.

Implement the offline path:

1. collect each scene's homography bounds without encoding;
2. transform them into a shared origin and take their union;
3. allocate one even-sized canonical canvas;
4. render each scene background at the correct offset;
5. update the per-frame reconstruction transforms for that offset;
6. use a deterministic fill outside each scene's valid region;
7. predictively encode those equal-sized images;
8. reset the sequence at a background-context boundary.

Keep independent background coding working. Do not implement causal canvas growth
in this brief.

## Required tests

- two unequal local panoramas produce equal encoded dimensions;
- source-frame reconstruction remains aligned after the origin shift;
- sender and receiver reconstruct byte-identical background images;
- static and panning scenes share a context without failure;
- unrelated contexts force a new independently coded background;
- padding is included in coded bytes;
- changing predictive coding changes bytes against independent coding;
- segmented and continuous AV1/VVC controls use the same reset boundaries as
  the PointStream configuration they compare against;
- prior 8/16-frame behavior remains valid.

## Diagnostic measurement

Use the two BP31 scenes at 24, 32 and 48 frames. Before the first encode, write
two-sided bounds for:

- reconstruction error against the independent-background path;
- canonical canvas area against the largest local canvas;
- predictive-background bytes against independent-background bytes;
- first-to-last-frame quality change.

Report bytes for padding itself only as a diagnostic; the encoded bitstream size
is the rate.

## Completion report

Follow `plans/SESSION-REPORT.md`. Include a diagram or coordinate example,
tests, before/after canvas dimensions, the three duration controls, all alarms,
commit and PR. State explicitly that this mode is offline/buffered.
