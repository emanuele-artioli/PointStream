# BP46 — Long eligible tennis-scene manifest

**Current state (3 September):** diagnostic inputs are available; independent
confirmation is incomplete. PR #52 contains the working extraction validation.
An incomplete later deletion was archived at
`archive/bp46-incomplete-extraction-2026-09-03`, not merged. The next execution
gate is BP49, not an unrestricted E1 batch. Do not count this whole brief done.

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

### Delivered (D1, 2026-09-02)

**Outcome: Diagnostic inputs ready; confirmation corpus incomplete.**

All diagnostic deliverables specified in BP46 and ROADMAP.md D1 have been implemented, extracted, cached, and verified with 0 diagnostic violations. The diagnostic search (E1) is fully unblocked. The confirmation corpus is honestly reported as incomplete due to host dataset constraints, which does not block the diagnostic search.

### Deliverables & Artifacts
1. **Machine-readable manifest**:
   - Committed repo manifest: [`manifests/bp46_long_tennis_scenes.json`](file:///home/itec/emanuele/pointstream/manifests/bp46_long_tennis_scenes.json)
   - Data root output mirror: `/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/manifest.json`
2. **Partitions & Strict Split Isolation**:
   - **Diagnostic videos (2 videos, 16 scenes)**:
     - `alcaraz_highlights` (near-static: `scene_000`, `scene_028`, canvas growth 1.00x; smooth-pan: `scene_010`, `scene_018`, `scene_026`, canvas growth 1.04x–1.08x; control: `scene_006`).
     - `federer_djokovic` (smooth-pan: `scene_001`, `scene_003`, `scene_005`, `scene_007`, `scene_009`, `scene_011`, `scene_013`, `scene_015`, `scene_017`, `scene_019`).
     - *Audit of previous use*: `federer_djokovic` was used throughout prior development and exploration (BP20, BP21, BP24, BP29, BP30, BP31, BP33) for tuning plate registration, panorama spans, and headroom ladders. Because it is contaminated by hyperparameter tuning, it belongs strictly in the diagnostic partition, satisfying the 2-video diagnostic requirement.
   - **Confirmation videos (5 candidate tournament matches on host, 37 candidate scenes)**:
     - `alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz`.
   - **Strict Isolation Invariant**: `set(diagnostic_videos).isdisjoint(set(confirmation_videos))` is strictly `True`. Every confirmation scene has `role: "confirmation"`, and no confirmation video appears in diagnostic or control partitions.
   - **Ineligible control (1 scene)**: `alcaraz_highlights/scene_006` (`cluster_other`, crowd view, routed to `conventional_fallback` with explicit reasons).
3. **True Per-Interval Validation Engine**:
   - Active simultaneous player window search: identifies exact `[start_frame:end_frame]` window where both players are simultaneously present and tracked (>= 85% coverage).
   - Computes interval-specific motion MAD and homography canvas growth from the window's exact downscaled frames.
   - Computes interval-specific frame SHA256 hashes (`first`, `mid`, `last`) of the exact extracted window PNGs.
   - **Interval-specific paste-back MAE**: measured on each interval's own track pairs and frames via `measure_interval_paste_back` (not copied from base window).
   - Ineligible intervals retain explicit failure reasons (e.g. `simultaneous player overlap 0 < 96 frames` or `insufficient_duration`).
4. **Execution & Validation Toolchain**:
   - Extraction & feature engine: `experiments.long_scenes.extract`
   - Invariant verifier: `experiments.long_scenes.verify` (reports `"diagnostic inputs ready; confirmation corpus incomplete"`)
   - Dataset loader: `experiments.long_scenes.loader.load_long_scene_clip` with `allow_ineligible=True`, returning `is_eligible`, `route`, and `failure_reasons` to support planned fallback-control testing on `alcaraz_highlights/scene_006`.
   - Unit test suite: `tests/experiments/test_long_scenes.py`

### Submitted / Succeeded / Failed Counts
- **Total candidate scenes submitted**: 53
- **PointStream eligible scenes**: 44
- **Conventional fallback scenes**: 9 (including `alcaraz_highlights/scene_006` control, and `alcaraz_ruud` where annotations only track one player at a time with 0 simultaneous overlap)
- **Succeeded by span**:
  - **48 frames (2.0s)**: 44 succeeded, 9 failed
  - **96 frames (4.0s)**: 37 succeeded, 16 failed
  - **192 frames (8.0s)**: 18 succeeded, 35 failed
  - **384 frames (16.0s)**: 8 succeeded, 45 failed
- All failures stay in the manifest with explicit machine-readable reasons.

### Confirmation Deficits (Honestly Reported)
1. Match count: host provides 5 candidate tournament matches (`alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz`), whereas the plan specifies 6 independent confirmation matches.
2. `alcaraz_ruud`: annotated tracks only track 1 player at a time (0 simultaneous overlap), failing 2-player eligibility across all spans.
3. 384-frame coverage: the report lists 3 confirmation matches (`djokovic_zverev`, `sinner_alcaraz`, and `djokovic_federer`); counts must be taken from the current manifest. `alcaraz_perricard` track length is 372 < 384 frames.
*Missing confirmation footage need not block the diagnostic search.*

### Verification Commands & Results
- **Manifest invariant verification**:
  `conda run -n pointstream python -m experiments.long_scenes.verify` -> **PASSED: diagnostic inputs ready; confirmation corpus incomplete**
- **Unit test suite**:
  `conda run -n pointstream pytest tests/experiments/test_long_scenes.py -v` -> **12 passed in 9.58s**
- **Layer & lint check**:
  `conda run -n pointstream python -m src.contracts.layers` -> **OK (no outward imports)**
  `conda run -n pointstream ruff check experiments/long_scenes tests/experiments/test_long_scenes.py` -> **All checks passed**
  `MYPY_CACHE_DIR=/tmp/mypy-pointstream conda run -n pointstream mypy --config-file pyproject.toml experiments/long_scenes tests/experiments/test_long_scenes.py` -> **Success: no issues found in 6 source files**
- **Loader validation**:
  `load_long_scene_clip("alcaraz_highlights", "scene_006", 48, allow_ineligible=True)` loads fallback control with `is_eligible=False` and `route="conventional_fallback"`. Verified valid clips load with 2 player objects and MAE 0.000.

### Next Step / Dependency
Diagnostic inputs are fully ready. Unblocks **E1** (`BP45`) and **B1** (`BP44`) long-scene search. Confirmation corpus can be expanded when additional raw match footage is ingested.
