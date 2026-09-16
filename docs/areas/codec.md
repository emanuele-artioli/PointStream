# Codec Area

## Pilot return audit — 16 September 2026

#119 is merged as `aefdeb2`. The E04B source is committed locally at `d30e59f`
but not merged to shared main. R2 has two unresolved ghost-MAD alarms and dirty
run-code provenance; two directories contain ON streams and retry/copy history
needs reconciliation. Missing boundary MAD defaults are not measured zero errors.
The error-floor explanation is a hypothesis. Close alarms/provenance by reuse;
no further candidate encodes or second-camera probe is released.
Follow [the reuse assignments](../workflow/session/evaluation-campaign/tasks/20260916-return-audit-and-reuse.md).


## Historical conditional release — 16 September 2026

#119 at `745271f` has green CI but is not accepted as calibrated evidence:
main continues after calibration failure and always labels derived calibration
verified, while its output directory allows overwrite. The revised proposal also
mixes four OFF-arm entries inconsistent with the canonical saved report.
The current brief releases those fixes, then conditionally exactly two scene007
panorama removal-ON points at QP47/32 within 30 CPU minutes. Reuse original OFF
streams. This is a same-camera diagnostic, not second-camera evidence or proof
of eliminated ghosting.
Follow [the current launch gates](../workflow/session/evaluation-campaign/tasks/20260916-bounded-pilot-release.md).


## Historical review — 16 September 2026

#113 merged as `c476840`. E04A's six removal-OFF bitstream hashes match the saved
report, whose prepared RGB identity agrees with E03B. Treat this as a reusable
background component diagnostic. The general low-rate panorama recommendation
and claim of eliminated ghosting are not accepted from one scene.

### Evidence completion and common-scope comparison (CODEC-ACT-07)

Completed evidence repair and common-scope rescoring under strict zero-new-candidate-encode budget:
1. **Scorer calibration**: Enforces registered numerical bounds, identity checks (`PSNR=inf`, `SSIM in [0.999, 1.0]`), null controls (safe `NaN` on empty mask without warnings, unrelated structured anchor floor), and whole-frame windowed orderings across blur, noise, and unrelated content (`scorer_calibration.json`). All checks hold (0 alarms, `valid=true`).
2. **Standalone client decoding**: Decoded all 6 saved E04A bitstreams strictly from bitstream bytes and packed side data (`.bin`) with zero geometry inputs available (`standalone_decode_report.json`). Bit-identical parity (`max diff = 0`) verified across all 6 settings. Rejection of truncated and extra video frames without silent padding/clipping verified and covered by unit tests in `tests/test_background_probe_standalone.py`.
3. **Common-scope comparison**: Rescored saved E03B conventional video decodes (VVC and AV1 at QP 47 and 63) across matched visible, object, boundary, and full-frame windowed scopes (`comparison_table.md`, `e03b_rescored.json`). Distinctly separates background-only rate scope from whole-codec conventional scope, and windowed whole-frame SSIM from global masked SSIM.
4. **Timing strata, host provenance & lookahead**: Replaced hardcoded metadata with verified provenance. Candidate encode host: `gpu5` for E04A, `gpu6` for E03B. Measured foreground compositing stratum (0.0463 s for 48 frames, 0.97 ms/frame). Unmeasured lookahead for conventional presets and cleaned video is explicitly labeled as `unmeasured` missing evidence rather than assumed as 0 or 16.
5. **One-scene conditional observation**: Replaced broad Pareto claims with conditioned observations. On Federer scene 007, registered panorama provides camera motion compensation (ghost MAD 6.70–9.16 vs 20.33–28.55 on still frame 0, visible PSNR +4.0 to +5.6 dB), but still frame 0 remains a valid ultralow-rate point (3,983 B vs 4,554 B at QP 47). Ghost-region error remains nonzero across all arms.
6. **Reduced E04B probe proposal**: Replaces Cartesian sweep with minimal discriminative probe resolving the named uncertainty (*causal impact of foreground removal on registered panorama plate quality and ghosting*). Reuses all 6 existing scene 007 removal-OFF points and executes only 2 missing candidate arms (`registered_panorama` removal=ON at QP 47 and 32, ~15s CPU, ~50 KB storage; `second_camera_proposal.json`), bounded by explicit promote/stop rules.


## Historical assignment — 15 September 2026

E04A / CODEC-ACT-07 is the independent Antigravity
[removal-OFF background assignment](../workflow/session/evaluation-campaign/tasks/20260915-next-stage.md).
Compare still/panorama/video at two QPs on the E03B-aligned 48-frame 360p/12 fps
development window, at most six settings/30 minutes after bounded preparation.
Reuse compatible artifacts and retain old removal-ON labels. Keep generation
and residual OFF; verify no removal/fill calls and reconcile actual transport.
#109 is not a prerequisite. No new measurement or winner is certified by dispatch.

## Current campaign — 14 September 2026

CODEC-ACT-07 continues as E04 in the [campaign](../workflow/session/evaluation-campaign/plan.md).
PR #101 repairs probe transport/precision, but saved pre-repair artifacts do not
automatically inherit those guarantees. The known three-mode probe uses temporal
foreground removal plus Telea fill. Check older background runs before scheduling
missing removal-OFF and paired removal-ON comparisons. No winner is certified.

**Evidence Revision**: Implementations through PR #85; gate interpretations audited 2026-09-09.
**Owned Scope**: `src/components/background/`, `src/components/appearance/`, `src/components/motion/`, `src/components/residual/`, `src/pipeline/codec/`.

---

## 1. Current State

PointStream structures video coding into distinct semantic components:
1. **Background**: A reusable high-resolution canvas / plate transmitted once or periodically.
2. **Appearance**: Sparse foreground actor crops transmitted as reference keyframes.
3. **Motion**: Trajectories, 2D bounding boxes, and pose/keypoints driving actor deformation.
4. **Residual**: Optional corrective high-frequency residual signal for high-fidelity reconstruction.
5. **Fallback**: Explicit coded conventional fallback when scene dynamics violate semantic assumptions.

### Empirical Benchmarks and Bottlenecks
In the Gate A 48-frame native run (#69), PointStream payload was dominated by two elements:
- **Background Plate Bitfloor**: `src/components/background/stream.py` hardcoded `StreamCodec` to `libaom-av1`. The archived #71/#72 probe reported 268,949 bytes at CRF 63 on one canvas/build. This is a configuration-specific observation, not proof of a universal AV1 floor; reproduce the command and reconstruction quality before attributing the cause.
  - Benchmarked alternatives on the same 4K canvas:
    - `libaom-av1 (CRF 63)`: 268,949 bytes (262.6 KB)
    - `SVT-AV1 (QP 63)`: 41,162 bytes (40.2 KB)
    - `VVC libvvenc slower (QP 63)`: 5,347 bytes (5.2 KB)
    - `VVC libvvenc slower (QP 55)`: 14,313 bytes (about 14.0 KiB; archived #72 value)
- **Actor Appearance Crops**: The pre-#75 implementation used baseline JPEG (`CompressedImageAppearance` in `src/components/appearance/compressed.py`), spending 4–25 KB per crop with ringing artifacts below quality 40. WebP encoding was implemented in #75 and used by #83/#85; standalone matched-quality crop benchmarking remains needed. The proposed 35–50% reduction is unverified at matched quality on this corpus; it is not an established result.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Sidecar encoders | #36 (`09c92880c4`) | Standalone AV1 and VVC intra sidecar encoders implemented. |
| Background canvas | #45 (`ecebd9b97d`), #50 (`bb5d17445f`) | Canonical offline background stitched representation established. |
| Fallback mechanism | #52 (`68a03dc542`) | Explicit conventional video fallback path added for ineligible segments. |
| Native 48-frame run | #69 (`648325b`) | Measured whole-system rate/quality; identified background & appearance byte floor. |
| Competitive regime | #70–#72 | Proposed VVC/SVT background sidecars and WebP/AVIF appearance crops. |
| VVC background streaming | #75, #80 | Implemented VVC low-delay background streaming (`StreamCodec.VVC`) with periodic intra refresh (`-period 1`) for causal prefix stability. Resolves `CODEC-ACT-01`. |
| WebP appearance crops | #75 | Implemented OpenCV WebP foreground actor reference compression (`AppearanceFormat.WEBP`). Resolves `CODEC-ACT-02`. |
| Gate A Rate Ladder | `outputs/gate-a-vvc-webp-n96-run2/report.json` | Completed development sweep; competitive pass superseded by 2026-09-09 evaluation audit. |
| Gate B Procedure Freeze | `src/contracts/frozen_procedure.py` | Locked down C0–C3 rate ladder configuration, VVC low-delay background streaming with intra refresh (`-period 1`), WebP appearance crops, zero generation/residual lattice settings, and metric thresholds/bounds for confirmation. |
| Gate B pilot | `outputs/gate-b-confirmation/report.json` | Execution completed, but competitive confirmation and fair speed claims are not established; see evaluation audit. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `CODEC-ACT-01` | Complete | None | #70, #71, #75, #80 | **Connect VVC intra / SVT-AV1 to background stream**: Implemented in `src/components/background/stream.py` with causal prefix intra-refresh stability. |
| `CODEC-ACT-02` | Complete | None | #71, #72, #75 | **Add WebP/AVIF appearance crops**: Implemented WebP encoding in `src/components/appearance/compressed.py` with matched-quality savings and client decode round trip. |
| `CODEC-ACT-03` | Ready | `CODEC-ACT-01`, `02` | #72 | **Payload ledger simplification**: Streamline byte allocation tracking across components. Acceptance: All transmitted bytes reconcile to the serialized payload, including background, appearance, motion, residual, fallback, headers, and container/metadata overhead; do not hide overhead inside a component saving. |
| `CODEC-ACT-04` | Previously deferred (D5) | None | `plans/DEFERRED.md` | **Coded fallback verification**: Verify behavior when semantic tracking fails. Acceptance: Clean switch to conventional intra/inter coding without crash or pipeline desynchronization. |
| `CODEC-ACT-05` | Ready | Coordinate EVAL-ACT-06 | 2026-09-09 code audit | **Residual fidelity and transport**: Gate A/B explicitly disabled residuals; shipped tiers enable them. Serialized client residuals remain unsupported; lossy uint8 offset clips large signed differences. Preserve the actual coded stream, validate full-range correction and independent output scoring, and measure high-fidelity residual-on curves. See [worker brief](../workflow/session/repair-residual-transport.md). |

PR #88 recovery repairs now complete the current CODEC-ACT-05 implementation: actual residual streams/full-range mapping, client-originated delivered scoring, serialized generation conditioning/reference data and complete envelope reconciliation are covered by integration tests and a native residual smoke test. This validates transport plumbing, not high-fidelity rate–distortion performance; run the residual-on ladder through the repaired client before claiming a codec result. See [evaluation audit](evaluation.md#pr-88-audit--2026-09-10) and [next dispatch](../workflow/session/submission-search.md).

## Current background decision — 2026-09-12

`CODEC-ACT-06` delivered a useful component prototype, not verified package
transport or an established 94% matched-quality saving. Registered panorama is a
promising candidate showing positive geometry compensation (+6–7 dB over unwarped still),
not a selected winner: the 94% byte reduction compared a QP 47 background-only point
against the legacy whole-codec background allocation under mismatched settings/presets
and quality scopes. Furthermore, the visible-background PSNR plateaus at ~26–27 dB,
leaving residual correction demand to be evaluated in production integration (`CODEC-ACT-07`).
The prototype code has been hardened with float32 homography casting matching charged
transport precision, binary side data round-trip serialization, and complete input/code
identity hashing. `CODEC-ACT-07` remains open: test the user's still / panorama / cleaned-video
and separate fill axes under one bounded card, then measure full-codec headroom before
selecting a production representation. See [audit](../history/antigravity-audit-2026-09-12.md)
and [handoff](../workflow/session/evaluation-handoff.md).

## Background experiment priority & findings — 2026-09-11 (Calibrated 2026-09-12)

`CODEC-ACT-06` — **Calibrated Prototype** (PR #96, `0bdf0ef`, updated under Lane 3 audit fixes).
Evaluated three background representations on the 48-frame Federer sequence (`federer_djokovic/scene_007`, 48 frames @ 4K 24 fps) using a common foreground-removed frame stack (visible background preserved bit-identically, player mask filled via temporal composite plate, 0 uncovered holes):

| Representation | QP | Payload (B) | Side Data (B) | Total (B) | PSNR-Y Vis (dB) | SSIM Vis | Enc Time (s) | Dec Time (s) |
|---|---|---|---|---|---|---|---|---|
| still_frame0 | 47 | 31,804 | 10 | 31,814 | 20.06 | 0.8158 | 2.05 | 4.18 |
| registered_panorama | 47 | 30,626 | 1,742 | 32,368 | 26.22 | 0.9541 | 2.25 | 8.31 |
| cleaned_video | 47 | 74,178 | 10 | 74,188 | 31.18 | 0.9824 | 6.97 | 7.00 |
| still_frame0 | 32 | 126,184 | 10 | 126,194 | 20.02 | 0.8173 | 2.43 | 4.22 |
| registered_panorama | 32 | 118,157 | 1,742 | 119,899 | 26.97 | 0.9643 | 2.61 | 8.02 |
| cleaned_video | 32 | 364,212 | 10 | 364,222 | 38.98 | 0.9967 | 9.31 | 7.07 |

- **Verdict: Hypothesis SUPPORTED with calibration.** Still frame 0 is fundamentally limited by uncompensated camera motion (~40 px pan), pinning visible PSNR at ~20 dB and SSIM at 0.817 regardless of rate.
- **Registered panorama provides positive geometry compensation**: Camera homographies gain +6.2 dB (QP 47) and +7.0 dB (QP 32), boosting SSIM to 0.954–0.964. However, it is a promising candidate, not an established winner: the 94% byte reduction compared against legacy whole-codec allocations under mismatched settings and was not at matched final quality. The visible PSNR plateau at ~26–27 dB leaves residual correction to be evaluated in full production integration (`CODEC-ACT-07`).
- **Cleaned video** reaches higher quality (+12 dB over panorama at QP 32), but incurs a 3.0x byte multiplier and 3.6x encode time.
- **Production path**: Investigate panorama and video under full-codec rate–distortion–computation integration in `CODEC-ACT-07`.
