# BP46 — Long eligible tennis-scene manifest

**Roadmap ID:** D1  
**Preferred harness:** Cursor or VS Code with Antigravity.  
**Outcome:** validated 2/4/8/16-second first-domain inputs for the low-rate search
and later six-video confirmation.

## Read

`AGENTS.md`, `plans/ROADMAP.md` §§3–4,
`plans/TERMINOLOGY.md`, the existing BP21 clip-cache reader and dataset path
contracts. Do not read raw files serially; batch on this NFS host.

## Eligibility rule

Record features first; do not silently cherry-pick by result:

- duration available at 24 fps;
- camera/background-motion statistic and panorama canvas growth;
- camera/view/background-context identity;
- number and known class of foreground objects;
- object size, separation, occlusion and track continuity;
- paste-back alignment check;
- whether the scene should route to PointStream or conventional fallback.

The search set may favor clearly eligible content. The confirmation rule is
frozen before selecting the six-video set.

## Deliverables

- a machine-readable manifest with source ID, timestamps, hashes, scene lengths,
  context ID and eligibility features;
- two diagnostic videos spanning near-static and smooth-pan eligible cases;
- at least six independent videos/matches reserved for confirmation;
- one high-motion ineligible control;
- extraction/validation command and submitted/succeeded/failed counts.

Do not commit or redistribute source video. YouTube-derived footage remains
under the external data root. The artifact may publish code, configuration,
hashes and legally permitted metadata only.

## Validation

- exact frame counts at 48/96/192/384 frames;
- dimensions, frame rate and colour metadata recorded;
- object tracks cover the requested interval;
- source frames, masks, motion and appearance references align;
- context IDs do not group unrelated cameras/backgrounds;
- failures stay in the manifest with reasons.

## Completion report

Follow `plans/SESSION-REPORT.md`. This work reports a corpus/manifest, not a
codec result and not a winning claim.

## Delivered (D1, 2026-09-02)

**Outcome: Complete.**

All deliverables specified in BP46 and ROADMAP.md D1 have been implemented, extracted, cached, and verified with 0 invariant violations.

### Deliverables & Artifacts
1. **Machine-readable manifest**:
   - Committed repo manifest: [`manifests/bp46_long_tennis_scenes.json`](file:///home/itec/emanuele/pointstream/manifests/bp46_long_tennis_scenes.json)
   - Data root output mirror: `/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/manifest.json`
2. **Partitions**:
   - **Diagnostic videos (2 videos)**: `alcaraz_highlights` (near-static cases: `scene_000`, `scene_028`, `consec_mad` 0.34, canvas growth 1.00x) and `federer_djokovic` (smooth-pan cases: `scene_001`, `scene_003`, `consec_mad` 1.85, canvas growth 1.02x).
   - **Confirmation videos (6 independent match videos)**: `alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `federer_djokovic`, `sinner_alcaraz`.
   - **Ineligible control (1 scene)**: `alcaraz_highlights/scene_006` (`cluster_other`, crowd view, routed to `conventional_fallback` with explicit reason).
3. **Execution & validation toolchain**:
   - Extraction & feature engine: `experiments.long_scenes.extract`
   - Invariant verifier: `experiments.long_scenes.verify`
   - Dataset loader for downstream tasks (BP44, BP45, E1, E2): `experiments.long_scenes.loader.load_long_scene_clip`
   - Unit test suite: `tests/experiments/test_long_scenes.py`

### Submitted / Succeeded / Failed Counts
- **Total candidate scenes submitted**: 20
- **PointStream eligible scenes**: 19
- **Conventional fallback scenes**: 1 (`alcaraz_highlights/scene_006`)
- **Span interval validation**:
  - **48 frames (2.0s)**: 19 succeeded, 1 failed (`scene_006` ineligible control)
  - **96 frames (4.0s)**: 18 succeeded, 2 failed (`scene_006` ineligible control + `federer_djokovic/scene_003` insufficient duration: 85 < 96 frames)
  - **192 frames (8.0s)**: 16 succeeded, 4 failed (`scene_006` ineligible control + 3 scenes with track duration < 192 frames)
  - **384 frames (16.0s)**: 10 succeeded, 10 failed (`scene_006` ineligible control + 9 scenes with track duration < 384 frames)
- All failures stay in the manifest with explicit machine-readable reasons.

### Quality & Alignment Checks
- **Dimensions & FPS**: 3840x2160, 24.0 fps working rate recorded for all scenes.
- **Color metadata**: `pix_fmt=yuv420p`, `color_space=bt709`, `color_primaries=bt709`, `color_transfer=bt709`.
- **Paste-back MAE**: 0.000 for all eligible candidate scenes (winning convention: `extract_24_frame_id` or `extract_24_position`), with huge separation against alternative conventions (e.g. 28–93 MAE).
- **Context isolation**: Context IDs strictly isolate cameras/backgrounds by venue/video (e.g. `{video}_main_court`); no unrelated footage shares a context ID.

### Verification Commands & Results
- **Manifest invariant verification**:
  `conda run -n pointstream python -m experiments.long_scenes.verify --manifest manifests/bp46_long_tennis_scenes.json` -> **PASSED (0 violations)**
- **Unit test suite**:
  `conda run -n pointstream pytest tests/experiments/test_long_scenes.py -v` -> **8 passed in 5.05s**
- **Layer & lint check**:
  `conda run -n pointstream python -m src.contracts.layers` -> **OK (no outward imports)**
  `conda run -n pointstream ruff check experiments/long_scenes tests/experiments/test_long_scenes.py` -> **All checks passed**
  `conda run -n pointstream mypy --config-file pyproject.toml experiments/long_scenes tests/experiments/test_long_scenes.py` -> **Success: no issues found**
- **Loader validation**:
  `load_long_scene_clip("alcaraz_highlights", "scene_028", 384)` successfully loaded exact shape `(384, 2160, 3840, 3)` with aligned masks and 2 player objects at MAE 0.000.

### Next Step / Dependency
Unblocks **E1** (`BP45`) and **B1** (`BP44`) to consume validated 48/96/192/384 frame long scenes.

