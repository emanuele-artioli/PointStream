# B4 — Background model and rigid objects

**Owns exclusively:** `src/components/{background,rigid}/**` and their tests.
**Implements:** the background and rigid-object protocols in `src/contracts/`;
camera-motion validity lives in `src/contracts/domain.py`.

## What to build

**Background**, as two independent axes rather than the one knob that currently
conflates them:

- *Transmission strategy* — `panorama-full`, `panorama-delta`, `none`.
- *Sidecar codec* — jpeg, png, roi-video.

Splitting them is what makes `{strategy: panorama-delta, codec: roi-video}`
expressible; today it is not.

**Rigid objects**, per class: racket (convex hull plus wrist anchoring), ball
(difference-based or segmentation-based). Both are optional lattice rows — turn
them off and those objects land in the residual, which is exactly how their value
gets measured.

## Traps specific to this stream

**A panorama under a free-moving camera is not merely worse, it is meaningless.**
Parallax means no single homography relates the frames, so the plate cannot be
internally consistent. `DomainProfile.assert_background_valid` already rejects
this at config validation — do not work around it. DAVIS clips are handheld, so
the general domain runs without a panorama.

**`panorama-delta` needs multi-scene runs.** With one chunk there is only ever
one panorama sent, so delta is byte-identical to full — the correct result for
that harness, not a bug. Its payoff needs a full-match run.

**A rigid object has no skeleton.** Keypoints are not a motion representation for
a racket or a ball; they carry sparse trajectories or encoded video. The contract
enforces this, and an explicit override that violates it is rejected.

## Done when

- Strategy and sidecar codec are independently selectable.
- Both rigid strategies are switchable off, and the payload change is measurable.
- Panorama validity is checked against the domain's camera assumption.
- `ruff`, `mypy`, tests pass; import direction clean.
