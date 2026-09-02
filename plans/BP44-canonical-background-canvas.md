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

## Delivered

**Outcome:** complete for the offline/buffered canonical canvas. Predictive
coding now accepts the two BP31 scenes at 24, 32 and 48 frames. This mode
sees every scene in a background context before any background is encoded; it
is not a live or causal path.

**This mode is offline/buffered.**

### Coordinate example (span 24, `alcaraz_highlights`)

```
scene_000 (static) frame-0 is the shared origin
  local canvas 2161 x 3841   (unchanged from BP31)

scene_010 (panning) registered into scene_000 coordinates
  local canvas 2172 x 3881
  alignment ≈ identity (same camera)

union origin  ≈ (-2.23, -7.34)
canonical     = even_up(union) = 2180 x 3884

per-frame map = T(-origin) @ H_align @ H_local
pad outside the scene's valid region = 128
```

A later context id (`replay` vs `court`) resets the stream: the next plate is
an independently coded keyframe, possibly on a different canvas.

### Scope expansions (reported)

Needed for a real `run()` to see all scenes before the first encode:

- `src/runner/stages.py` — `model.stitch` / `prepare_context` prepass
- `src/runner/run.py` — pass sliced chunks on `StageContext.source_chunks`
- `config/default.yaml` — `context-id`, `canvas`, and the stream fields the
  schema already had

Not wired in this PR: `experiments/tier/ladder_scenes.py` still uses joint vs
separate concatenation for the anchor. `context_reset_indices` /
`segmented_reset_indices` in `stream.py` are the reset boundaries the
continuous and segmented AV1/VVC controls must share; they are tested, not
yet called from the ladder.

### Tests

`conda run -n pointstream --no-capture-output python -m pytest` on the
background plate, stream, strategy, config, panorama runner, and new
`tests/components/background/test_canonical_canvas.py` files: 70 passed.
`ruff check` on the touched files: clean. `mypy` on the touched files: clean.
`python -m src.contracts.layers`: OK.

### Diagnostic (`outputs/bp44-canonical-canvas/`)

Bounds written first to `bounds-before-run.json`. ffmpeg n7.1.1
`/opt/local/bin/ffmpeg`. Video `alcaraz_highlights` scenes `scene_000`,
`scene_010`. Encode time is wall clock of plate build plus AV1 stream, one
sample on a shared host.

| frames | local (static, pan) | canonical | area vs max local | independent B | predictive B | ratio | recon MAE | last−first dB | encode s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 2161×3841, 2172×3881 | 2180×3884 | 1.004 | 894974 | 744592 | 0.832 | 0.40 | **−3.86** | 151 |
| 32 | 2161×3841, 2189×3919 | 2196×3922 | 1.004 | 918100 | 745276 | 0.812 | 0.43 | **−7.04** | 165 |
| 48 | 2161×3841, 2190×3932 | 2196×3934 | 1.003 | 918851 | 753934 | 0.821 | 0.35 | **−6.22** | 266 |

Local shapes reproduce BP31 §12 exactly. The stream no longer refuses them.

PNG padding-byte delta is a diagnostic only (and can be negative on the static
scene, because mid-grey pad compresses harder than nearest-finite warp
margins). The AV1 bitstream size is the rate.

### Alarms

Three of four pre-registered bands held at every duration: reconstruction MAE
vs independent (0.35–0.43, band 0–3), canvas area (1.003–1.004, band 1.00–1.25),
predictive/independent bytes (0.81–0.83, band 0.20–1.10).

**`last_minus_first_psnr_dB` fired at all three durations.** The bound's basis
was "a drop worse than 3 dB suggests the homography walked off the canvas."
That attribution is wrong. The static scene is stable or slightly *up* (33.6 →
34.7 dB at 48 frames). The panning scene drops 31 → 17 dB, and reconstruction
MAE against the independent local plate on that scene is 0.19–0.23 — the two
paths agree, so the drop is the panning plate versus source, not the origin
shift. The band stays as a late-frame check; its cause is revised to "plate vs
source on a pan," and a split against the independent path is the check that
would catch a canvas walk-off.

### Config

`background.canvas`: `independent` (default, local canvases) or `canonical`
(offline union). `background.context_id`: scenes that share it share a canvas
and may be predicted across; a change forces a new keyframe.

### Next

Wire `context_reset_indices` into the paired ladder's continuous AV1/VVC
control (M2 / E1). Then run the long-scene search at 48/96/192/384 with
`canvas: canonical`. Do not call this configuration live.
