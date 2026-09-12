# Antigravity evidence audit — 2026-09-12

Reviewed main `4fa7cd3` (PRs #93, #95–#98). PR #98 and its stated CI run
`34601076209` are merged/successful; no open PRs existed at audit start. This is
an audit of saved observations and code, not a new benchmark or model ranking.
Original artifacts stay immutable. The evaluation handoff owns next actions.

## What was useful

The component ledger arithmetic and the three-representation prototype are useful
development tools. Mask-aware fill preserves visible background pixels, and the
prototype includes still, panorama and video, quality and stage timings. Compact
mask transport and positional pose loading repair real issues. The background
probe's reported execution fits the bounded pilot, rather than another overnight
search. Keep those implementations and outputs.

The move back to the older Wave 2 generator matrix did not complete the planned
background production comparison. `CODEC-ACT-07` remains open. A successful
matrix invocation is not a substitute for the user's representation/removal
rate–distortion–computation evaluation.

## Findings that change interpretation

1. **The background saving is not established at matched final quality.**
   `scripts/background_probe.py::charge_side_data` returns arithmetic byte counts;
   no geometry packet is written/read, and rendering uses original float64
   homographies despite charging float32. Encoded image/video payload lengths are
   measured; the complete package is a proposed format, not verified transport.
   Comparing its QP47 background-only point with the older whole-codec background
   allocation does not prove a 94% matched-quality saving. The comparison also
   changes coding settings/presets and quality scope. Keep panorama as a candidate,
   not the selected winner. Its visible-background PSNR plateau can still leave
   expensive correction. A camera-warped single-frame control would distinguish
   registration from the additional temporal observations/canvas coverage.

2. **The prototype is not yet a reusable evidence pipeline.**
   Its cache key is video/scene/frame-count, without input/mask/code identity;
   the report lacks exact source-frame/mask hashes and a complete code identity.
   The all-ones canvas validity mask cannot distinguish observed pixels from
   nearest-filled holes in `build_plate`; “zero uncovered holes” is not proof all
   background was observed. Preprocessing is recorded separately and must be
   included in end-to-end costs. Visible SSIM is global masked RGB SSIM, distinct
   from the full-frame SSIM path. The still/panorama expectations in bounds prose
   were missed even though the broader machine bands passed; record that as a
   hypothesis update, not “all expectations confirmed.” Do not reuse caches or
   publish component plots before fixing the paths needed for that claim.

3. **Byte diagnosis: arithmetic useful, canned conclusions unsafe.**
   `scripts/byte_diagnosis.py` computes the component sums and local interpolation,
   but `hypothesis_evaluation`, promotion and runtime ranges/ratios are hardcoded.
   Alarm reconciliation labels are also fixed and do not verify all anchor alarms.
   Do not carry its fixed “below 50 kB at any quality” rule into another run.
   H=0 means no remainder outside named buckets, not zero physical envelope
   overhead (which can be inside metadata). Its conversion of 11,028 bytes/frame
   to 88.2 kbps omits frame rate; at 24 fps the background allocation is about
   2.12 Mbps. Those are arithmetic corrections, not new codec measurements.
   At the saved lowest Federer rung, even setting background bytes to zero leaves
   more bytes than the saved local AV1/VVC estimates: background work is relevant,
   but cannot alone be assumed sufficient. Retain source identity/evidence flags,
   interpolation sensitivity and missing spatial attribution in every reuse.

4. **The generator diagnostic exercises sparse placements.**
   Exact saved hash comparison finds gen-on/residual-off differs from paste only
   at frame index 0 on each 16-frame scene. The shuffled pose corner also differs
   from ordinary generation only at frame 0. Identity records show two placements
   both at frame 0; `load_long_scene_clip` creates one ObjectRequest at first
   appearance per track. This is consistent with sparse placement testing, not
   evidence of full-sequence pose-driven synthesis. Whole-frame null scores are
   almost unchanged. Hash inequality proves some pixel changed, not useful pose
   adherence. No effect-size threshold or repeated same-condition control was
   established here. Fixed residual QP is not matched final fidelity. Retain the
   paste control and current generation-off policy, but retract broad “strictly
   superior model” and resource-frontier interpretations. Do not discard pix2pix
   or infer anything about diffusion from this probe.

5. **Validity repairs remain incomplete at integration boundaries.**
   - `run_matrix` hashes `clip.objects` before `_augment_objects_with_pose` loads
     skeletons. Saved identity `config.conditioning` is `[null, null]` for both
     scenes. A changed skeleton file can therefore escape reuse identity. Hash
     the actual augmented inputs before lookup; include every effective config
     field rather than a manually selected subset. Avoid `--reuse-results` now.
   - `resolve_client_generator` hashes the explicitly requested checkpoint path
     when checking a registry result. A factory returning a backend pointing to B
     while A was requested is accepted. `_from_registry` also retries without
     checkpoint on TypeError. Verify the backend's actual loaded weights or reject
     unsupported checkpoint selection. This does not show these saved pix2pix
     runs used wrong weights; it disproves the generic fail-closed guarantee.
   - `assess_generator_comparison` inspects `metrics`, while actual rows use
     `scores`, and does not require scores/timing keys or the full declared corner
     set. Removing scores and timing from all otherwise valid saved rows still
     returns valid. Require schema-complete finite results and a meaningful null.
   - Pose loading falls back from absent positional lookup to a global-numbered
     skeleton if that file exists. Require an explicit indexing convention so a
     missing positional file cannot silently select an unrelated pose.

## Verification and limits

Before reading scores, carried forward stored bounds and used the user-reported
ranges as audit expectations (not independent preregistration). All four report
hashes below match the supplied or newly located immutable files. The original
metric calibrations were not rerun; no new rate–quality advantage is certified.

The selected existing tests for byte diagnosis, background probe (excluding native
integration), diagnostic matrix and generation identity passed (26 selected cases).
Run with the pinned Python, `import sqlite3` before pytest, and host-local caches.
Two extra read-only boundary checks reproduced findings above: an A-request/B-factory
result was accepted; removing saved scores/timing still yielded valid. No GPU
inference or new codec encodes were run. No manuscript was changed.

Reproduce the checkpoint check by creating two different temporary weight files,
constructing `GeneratorRef`s for A and B, making metadata with
`identity_from_ref(ref_A, seed=42, params={})`, patching
`src.runner.generation_identity._from_registry` to return ref_B, and calling
`resolve_client_generator(meta, checkpoint=A, require_identity=True)`. It returns
ref_B on the reviewed revision. This tests the factory boundary, not real weights.

For the validity check, deep-copy the Federer report's matrix, remove `scores`,
`metrics` and `timing` from every row, then call `assess_generator_comparison`
with the report's checkpoint/backend. The reviewed revision returns valid.

Artifacts relative to `/home/itec/emanuele/pointstream-data/outputs/development-recovery/`:

| Artifact | SHA-256 |
|---|---|
| `wave2-byte-diagnosis.json` | `74120308fef2e036d66a68b587cd42b6cc1a57d9b3b19ade21dd32d26c1c1aa9` |
| `wave2-background-probe/probe_report.json` | `c12b490085cb197ddcb26eabbd36e4ded846446418fe7d15d47dacda7851f3b7` |
| `diagnostic-pix2pix-alcaraz.json` | `ccfaa34b8ceca48409839c3baefec20e91f87a0425aea6da2b7429c37ed2fa50` |
| `diagnostic-pix2pix-federer.json` | `407148416bf1455ccfb68cc025a2ff31899689ebd0ffc722fd4f0dddc085af25` |

The diagnostic identities refer to `accfa15` plus a dirty-diff SHA, not directly
to final `4fa7cd3`. Preserve/reconstruct that exact diff before reproducibility
claims; a digest without the diff is not sufficient. Final CI checks the committed
code, not necessarily every detail of the dirty tree that generated the reports.
