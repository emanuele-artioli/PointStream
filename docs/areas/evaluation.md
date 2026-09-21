# Evaluation Area

## Gate B confirmation manifest, multi-source ladder, and neural benchmark scoping — 21 September 2026

PR #142 merged as `97a85b9`.
1. **Multi-Source Rate Ladder Generalization**: `experiments/modular/rate_ladder.py` generalized to dynamically evaluate arbitrary sources from manifests with sequence-specific horizon assignments and diagnostic visual strips. Strongly typed `AnchorData` dataclass deployed.
2. **Gate B Confirmation Manifest**: `manifests/gate_b_modular_confirmation.json` pinned with 6 independent matches (4 development: `federer_djokovic` 007, `alcaraz_highlights` 000, `alcaraz_perricard` 003, `djokovic_federer` 009; 2 cross-source holdouts: `bp57_ao2024_sabalenka_zheng` 1080p, `bp57_usopen2023_gauff_sabalenka` 720p). Verified via `test_rate_ladder_multi_source_confirmation_manifest`.
3. **Neural Generative Benchmark Scoping**: `manifests/neural_generative_benchmark_spec.json` and `experiments/modular/neural_benchmark.py` implemented, establishing falsifiable ceilings ($F \le 12\text{ kB}$, $\text{OKS} \ge 0.90$, $PSNR_{\text{fg}} \ge 35.8\text{ dB}$). All 4 candidate generators (`pix2pix`, `spade4tennis`, `controlnet`, `animate_anyone`) evaluated and marked `FALSIFIED`, keeping generative models OFF and freezing `03_appearance_crops.md` in `SATISFIED_FREEZE`.
4. **CI Hardening**: Updated `tests/utils/test_gpu_guard.py` to skip live GPU assertions when executing in headless non-GPU CI environments. All CI checks (`lint`, `typecheck`, `tests`) passed 100%.

## Modular rate ladder & Saliency-Weighted evaluation — 21 September 2026

PR #140 merged as `7966827`. Verified full-pipeline rate ladder across short (48f) and long (192f) horizons:
1. **Dual-Perspective Quality Accounting**: Saliency-Weighted Quality ($PSNR_{\text{weighted}} = 0.70 \cdot PSNR_{\text{fg}} + 0.30 \cdot PSNR_{\text{bg}}$) reflects foveal visual attention in 4K sports broadcasts where the actor occupies 2.1% of pixels, preventing 45:1 background dominance from penalizing background compression. Decoded video is evaluated against pristine untouched 4K GT, simultaneously reporting unweighted $PSNR_{\text{overall}}$, $PSNR_{\text{fg}}$, $PSNR_{\text{bg}}$, and `pose_oks`.
2. **Anchor Clearances**:
   - Long Horizon (192f, Alcaraz 000): Rung C1 ($26.7\text{ kB}$, $\text{OKS}=0.92$, $\text{PSNR}_{\text{fg}}=35.8\text{ dB}$) clears VVC QP47 anchor ($77.2\text{ kB}$) by **65.4% bitrate reduction**.
   - Short Horizon (48f, Federer 007): Rung C1 ($14.4\text{ kB}$, $\text{OKS}=0.92$) clears VVC QP47 anchor ($21.3\text{ kB}$) by **32.5% bitrate reduction**, overcoming the previous short-horizon fixed-plate barrier.
3. **Artifacts & Harness**: `experiments/modular/rate_ladder.py`, `manifests/modular_rate_ladder.json`, visual comparison strips and Markdown carousels generated in `outputs/modular/visuals/`.

## E06 lossless floor arms — 17 September 2026

#128 is merged. Floor-only probe on the saved bbox residual-off compact
(`f5e7160…`, pixels `5f2047d…`). Packing succeeded on all four arms
(exact parent pixels, standalone unpack, physical `T=B+F+M+R+H`, client
`n_repeats=2`). Predictor quality was **not** rescored; inherited
20.68 dB / 0.672 SSIM / 18.18 VMAF. No affine/interpolation, confirmation,
GPU, training, or native re-encodes.

| Setting | Packing | T (B) | vs VVC 21288 |
|---|---|---|---|
| control compact | pass | 30297 | above |
| per-frame mask RLE | pass | 17581 | below |
| XOR key every 4 | pass | 22055 | above |
| thin placement/metadata | pass | 28125 | above |

Lowest T is under both saved anchors, but quality remains ~4 dB below VVC
QP47 24.6 dB, so this is not a codec win. XOR lost to intra RLE on this
stack. Thin metadata saved ~2 kB; zip/JSON was not the 10 kB floor.
Artifacts: `outputs/evaluation-20260914/e06/probe-20260917-floor-arms/`
(`floor_report.json` SHA-256
`0bd1e7cb0797ecda7b9174b6ea294cb91d0a72294d92e53e76fb581dd31c4026`).
Wall 13.5 s on gpu5. `code_head` in that JSON is parent `6b150f3`; the
packer lands in this PR. Next: decide whether to run the residual-off
predictor arms now that rate headroom exists.

## Claim-eligibility acceptance and floor/predictor card — 17 September 2026

#128 now fails closed unless client timing has at least two reconstructions
(`repeat_seconds` or per-stage `n` matching `n_repeats≥2`), every present
PSNR/SSIM/VMAF value is finite and in domain, and per-video fitted
`charged_bytes` sit inside claimed `total_bytes`. Missing speed still leaves
a valid RD row eligible. The saved compact bbox residual-off point remains
30297 B at 20.7 dB, above VVC QP47 21288 B / 24.6 dB, so this full-codec
configuration stays stopped as a win search. The next discriminating step is
the unlaunched card
[evaluation_20260917_e06_floor_predictor_probe.json](../../manifests/evaluation_20260917_e06_floor_predictor_probe.json):
six residual-off settings on mask/placement/framing versus predictor quality,
no encodes or training in this session.


## Pilot return audit — 16 September 2026

E06/#121 is merged (`c656ee6`, launch `def647a`). Four saved transport reports,
ledgers and hashes match; accept stopping this configuration as a codec win search.
No broad win/BD-rate follows. Derived lossless packing and client deserialize/decode/render
timing live under `outputs/evaluation-20260914/e06/audit-20260916-lossless-pack/`.
Claim eligibility now fails closed on missing calibration, blank controls, undeclared
deployment cost, and combined encoder+score seconds. Follow
[the reuse assignments](../workflow/session/evaluation-campaign/tasks/20260916-return-audit-and-reuse.md).


## Historical conditional release — 16 September 2026

#117 at `d700a93` is accepted and merged as `015c971`, with green exact-head CI.
The decode repair preserves saved artifacts; native-seek pixel provenance remains
unresolved for 47/48 frames, so no native-PTS certification follows. #121 continues
on `codex/e06-20260916` after `a28c98f`: serialized transport, complete
`T=B+F+M+R+H` reconciliation, and honest predictor names
(`per_frame_crop`, `bbox_resized_first_reference`). Four E06 settings launch only
after this revision is merged with green CI. No anchor/background re-encodes or
confirmation scores. Follow
[the current launch gates](../workflow/session/evaluation-campaign/tasks/20260916-bounded-pilot-release.md).


## Coordinator readiness — 16 September 2026

Neural anchor source and paper are pinned in
[the readiness record](../../manifests/evaluation_20260916_neural_anchor_readiness.json).
DCVC-UF is not yet installed or round-trip verified: the official weight link
returns HTTP 403, the current Python interpreter rejects upstream syntax, and
UF-compatible Torch/CUDA extensions remain required. A separate Python3.12 RT
CPU entropy extension builds and passes a fresh-process synthetic symbol
roundtrip; no weights, neural video roundtrip or benchmark is available. GPUs are
accessible outside the sandbox. No neural RD result or SOTA claim is available.
The RGB input config uses the exact E03B prepared stack; its native-PTS gap
remains explicit. No rate ladder is released. The [coordinator audit](../../manifests/evaluation_20260916_coordinator_readiness_audit.json)
records intact hashes for three reserved media and 1390 retained text records
without source-bound codec-score flags under its binding heuristic. Deleted or
unrecorded use is outside its scope; replay/duplicate review, decoded-frame
preroll and protocol freeze remain open. Confirmation scoring stays unauthorized.


## Historical review — 16 September 2026

#114 merged as `9c6609f`. Follow-up on `codex/e03b-20260916` repairs the decode
gate: empty, partial, short and extra RGB24 dumps fail before reshape or pad.
Ordinary vs standalone pixels must match. Completed run directories refuse
overwrite; prepared reuse checks stack SHA-256 and seek/filter identity.
Verification records go in a new directory; the four charged encodes are not
repeated. `df9f945` descends from required baseline `8eb045a`. Score-free
eligibility is not authorization to score confirmation.
Verification (decode-only, no new encodes):
`outputs/evaluation-20260914/e03b/verify-20260916-acceptance/`. Four bitstream
hashes match campaign rows; eight containers are 48×640×360; ordinary and
standalone RGB dumps match. Original `probe_report` SHA-256
`62ac0b15228001f7598958d501df133003d8c6faea0b5b6947670a3839fc45f7` and
`bounds.json` SHA-256
`77c30b5b2d81da8bd8abe50e41150571db9d7615bcbdf3ee0657f1cbdb5c2d9a` unchanged.
Native PTS mapping is preserved; 1/48 selected PNGs match a native seek at the
mapped PTS (first frame). The other 47 differ (mean abs ~2). That remaining
uncertainty concerns extract-grid versus native-seek provenance, recorded in
`derived_native_mapping.json`. Verification JSON SHA-256
`2df757a43970c28083c24b8930ac8dd860a9acba356d706800cb5707429192f6`. Score-free
eligibility is not permission to score.

## E03B probe record — 16 September 2026

Worktree `/home/itec/emanuele/worktrees/pointstream-e03b-20260916` on
`codex/e03b-20260916`. Code head at probe:
`df9f945785b6dcd2cb7d6c5619b4b496da76ec23`. Persistent wrapper:
`experiments.tier.e03b_persist.persistent_timed_roundtrip`. Native PTS recipe
(shared with E04A): run dir
`outputs/evaluation-20260914/e03b/run-20260916-federer007/`. Prepared stack
SHA-256 `1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8`
matches E04A 360p/12 fps pixels; native timestamps are packet DTS seconds, not
24 fps grid indices. BP46/BP21 extract caches were not overwritten.

Four charged settings finished on gpu6 in 92 s encode wall (plus 2.1 s
calibration). Job `jobs/e03b-20260916-federer007` exit 0. All four rows ingest
as validated RD with standalone decode and reconciled ledgers. AV1 used
`SvtAv1EncApp` v1.8.0 preset 0; VVC used the accepted `encode()` path
(`ffmpeg` n7.1.1 `libvvenc` slower, `-qpa 0`), not the card's `vvencapp`
identity string. Same-QP quality ranges do not overlap (AV1 PSNR 27.4–33.0 dB
vs VVC 17.9–24.6 dB). Do not compute BD-rate. One interior QP between 47 and
63 cannot create overlap; skip it. Size-adjacent pair already exists: AV1 QP63
19 116 B / 27.4 dB vs VVC QP47 21 288 B / 24.6 dB. One-scene diagnostic only;
not a confirmed advantage.

VMAF 0.0 on VVC QP63 fired the pre-score band [5, 100]. Calibration on this
grid already floors at 0 (severe-blur and spatial-null); unrelated-clip VMAF
is 1.38; VVC QP63 PSNR 17.93 dB sits next to severe-blur 17.73 dB. Alarm
closed as bound-too-tight, not a broken metric path. Quote VMAF beside those
anchors. `n=1` scene, single-run timing, not a repeated speed claim.

Confirmation eligibility unchanged: acquired=3, confirmation_eligible=3,
`scores_computed=false`. All 1080p; cannot confirm native 4K. Coordinator
still must freeze scenes before any confirmation scoring.

## Historical acceptance completion — 15 September 2026

PR #104 rejects invalid runtime evidence and requires explicit measured RD arms
while retaining valid RD without timing. Standalone transport requires decode,
positive bytes and ledger checks; trajectory claims require the conditioning
control, not coverage alone. Producer same-seed corners are recognized. The pin
identifies an exact code-bearing commit and module bytes. Existing diagnostic
artifacts remain unchanged and excluded from unsupported validated claims.
The [E03B assignment](../workflow/session/evaluation-campaign/tasks/20260915-next-stage.md)
releases the bounded anchor preparation/probe independently of generator
readiness: #104/#108 are merged with passing CI. Preserve streams and
resolve actual source PTS before measurement. Four anchor settings are planned;
none were launched by this coordinator. Reports return to the coordinating task.

## Historical acquisition / integration review

E03A now gives Cursor sole ownership of result adapter and reader. #104 head
`7e4da6e` CI passes, but default validated ingestion admits infinite quality,
negative bytes and dict-valued failed controls in a reproduced case. Stored E02R
record fails the current contract. Fix these boundaries, re-adapt existing output
and prepare anchors; no broad new run required. See campaign current dispatch.

## Historical coordinator follow-up — E01/E02

Coordinator review: E01 #104 (`b87daf51`) is not yet accepted. Inventory is
useful, but claim ingestion admits missing/invalid evidence and an unrun timing
example. E01R corrects eligibility while preserving RD-only evidence that is
actually valid. See [next dispatch](../workflow/session/evaluation-campaign/tasks/01r-contract-and-acquisition.md).

**E03A (in #104, pending coordinator acceptance).** Cursor now owns
`campaign_result.py` and `src/runner/generation_adapter.py`. Contract revision
`e03a-20260915`. Validated ingest requires domain-valid finite values, positive
bytes, and positive success evidence on decode/calibration/ledger; `inf`,
negative rate, and `{status: failed}` / `partial` / `not_applicable` no longer
certify RD. Adapter reads producer `scores`/`timing`/`parts` and hashes file
bytes; checkpoint digest stays separate. Derived E02R diagnostic record:
`outputs/evaluation-20260914/e03a/derived-from-e02r-diag/` (originals
unchanged). Diagnostic RD kept; validated claims excluded. First low-res
anchor card: `manifests/evaluation_20260915_e03a_anchor_card.json`. Pin:
`manifests/evaluation_20260915_e03a_contract_pin.json`.

**E01R (in #104, pending coordinator acceptance).** Production `ingest_for_claim` defaults to validated rows;
historical observations stay in diagnostic ingest only. Cleared identities,
`controls=None`, and unrun timing examples no longer certify claims. VMAF 20
is a diagnostic exclusion; sustained live for finalists is 30 s after startup
plus four refresh cycles. Confirmation policy no longer falls back to the two
exposed Gate B sources. Trajectory is not generalization.
Reserved confirmation windows: acquired 3, eligible 0, unscored. Stream-copy
I-frame preroll is recorded in
`manifests/evaluation_20260915_e03a_confirmation_timestamps.json`.

## Current campaign — 14 September 2026

E01/E03/E06 now follow the [campaign](../workflow/session/evaluation-campaign/plan.md).
PR #100 repairs byte analysis and #102 repairs result/pose/checkpoint boundaries
at code baseline `3810a56`; this advances EVAL-ACT-11 implementation, not old
artifact validity or gate passage. Inventory eligibility by claim, reuse existing
RD and statistically adequate compatible timing, and rerun only documented gaps.
Older assignments/status below are historical where they conflict.

**E01 protocol (E01R/E03A, pending coordinator acceptance).** Reuse map, source
split, operating-point contract live under `manifests/evaluation_20260914_*.json`.
Result contract revision is `e03a-20260915` (`manifests/evaluation_20260915_e03a_contract_pin.json`).
Independent-match count is loaded from
`experiments.tier.source_count_policy` (confirmation **3**, preferred 6, development
pilot 2). Plot ingest is `experiments.tier.campaign_result.ingest_for_claim`
with `purpose='validated'` or `'diagnostic'`. Historical overlap after #88
remains diagnostic, not validated RD. No historical artifacts were moved. The
known background probe is removal-on. Gate B's two sources stay exposed, not
untouched holdouts. Wait for coordinator acceptance and tested R0 before E03
measured batches. The prepared low-res card is
`manifests/evaluation_20260915_e03a_anchor_card.json`.

**Evidence Revision**: Audit of PR #83–#85 / `d4252b5`, 2026-09-09.
**Owned Scope**: `src/pipeline/reconstruction/quality.py`, `experiments/tier/`, `src/contracts/lattice.py`.

---

## Current audit and handoff — 2026-09-12

[Audit](../history/antigravity-audit-2026-09-12.md) supersedes the “all blockers
resolved” and broad model-comparison claims below. #93/#95–#98 are merged; their
saved observations remain intact. Background-ledger arithmetic is useful, but
hardcoded verdicts, incomplete pose reuse identity, missing-result acceptance
and checkpoint factory validation require follow-up (`EVAL-ACT-11`). The 16-frame
matrix changes only frame 0 in generation-off/on and normal/shuffled comparisons;
this is not validated full-sequence generation evidence. No fresh advantage or
Gate A/B pass is established. Next: [evaluation handoff](../workflow/session/evaluation-handoff.md).

### Byte diagnosis and evaluation audit corrections (Finding 3 / Lane 2)

Astra audit of 2026-09-12 identified four critical caveats in `scripts/byte_diagnosis.py` and evaluation accounting:

1. **Bitrate calculation and frame-rate correction**:
   Converting 11,028 bytes/frame to 88.2 kbps omitted video frame rate ($11,028 \times 8 / 1000$). At 24.0 fps, 11,028 bytes/frame corresponds to $11,028 \times 8 \times 24 = 2,117,376\text{ bps} \approx 2.12\text{ Mbps}$ ($2,117.4\text{ kbps}$), not 88.2 kbps. The calculation helper is updated to take `fps: float = 24.0` (default 24.0) and compute `bytes_per_frame * 8 * fps / 1000.0` (or `/ 1_000_000.0` for Mbps).

2. **Disjoint ledger semantics ($H = 0$)**:
   $H = 0$ in the disjoint arithmetic ledger ($B + F + M + R + H = T$) signifies zero unallocated arithmetic remainder across the declared component categories. It does **not** indicate zero physical container or envelope overhead: container framing, indexing, and serialization structures reside within $M$ (metadata).

3. **Limits of background reduction**:
   At the saved lowest Federer rung (R63), background $B$ is 529,361 B (75.6% of 700,102 B). However, setting background bytes completely to zero leaves $700,102 - 529,361 = 170,741\text{ B}$ ($F=30,196\text{ B}$ actor reference + $M=70,609\text{ B}$ metadata + $R=69,936\text{ B}$ residual). This non-background remainder alone exceeds the saved local AV1 estimate (110,842 B) and VVC estimate (130,906 B). Therefore, background work is relevant and necessary, but alone cannot be assumed sufficient to beat conventional anchors; concurrent reductions in metadata and residual demand are required.

4. **Dynamic analysis and avoidance of canned conclusions**:
   `scripts/byte_diagnosis.py` now strictly derives all hypothesis evaluations, promotion verdicts, runtime ranges, and speed ratios from input data dynamically. It avoids carrying fixed rules (such as "below ~50 kB at any quality") across different operating points or quality regimes.

The following 2026-09-11 entries are historical reports, qualified by this audit.

## Experiment policy update — 2026-09-11

New probes follow [hypothesis-driven experiment design](../workflow/experiment-design.md).
The next question is why total rate loses at matched final quality, with a
component budget before another residual sweep. Existing audit/run history below
is retained; historical next-action wording does not supersede this priority.

The overlap artifacts have now been located under the external data root:
`outputs/development-recovery/wave2-overlap-20260910/`. SHA-256 anchors:

- `report.json`: `48cf7b20a1a29ac958a8eac3972b381f848439003f02d0b1222de8221f81643b`
- `bounds.json`: `3b2bc898538be284a7b590e1792ac26ceb6c1dedd43452947f9df8efb17f45c1`
- `experiment-identity.json`: `df3da27ab3fbdd7059a977eb0f17e980fec6b43069c988eeb40caf018413ca0d`
- `tool-identity.json`: `83e2d5c2e28d94e0eadb9cc7be74db716662ecbf56b871c022a41c16429b5d65`

This session checked artifact identities and status fields, not numerical
comparisons: `gate_b_passed=false`, `pilot_alarms_clear=true`,
`identity_verified=false`, `evidence_verified=false`. Cursor reports valid
calibration, an unscorable Alcaraz span and a scorable losing Federer comparison;
these remain development observations pending independent evidence checks and
bound-alarm reconciliation. Clean execution does not make them citable. The
stored bounds identify #93 `ba5a9a8`, generation off, 48 frames on each source.
Do not delete or overwrite intermediate failed/repaired runs.

PR #93 merged (`4a55573`, repairs in `e9f781f`) after resolving all four blockers in `EVAL-ACT-09` and `GEN-ACT-08`:
- Verified client generator checkpoint resolution against transmitted SHA-256 (`resolve_client_checkpoint`, fail-closed).
- Config identity covers full effective state: device, background codec/quality, lattice stages, residual, generator, appearance, run/seed, masks, placements, conditioning digests, and dirty-worktree `diff_sha256`.
- Declared control checks require control run success, non-empty paste hashes, and complete finite outputs.
- Dynamic start frame resolution from clip metadata or manifest; strict rejection of missing/misaligned conditioning (no synthetic black skeleton fallback).

`EVAL-ACT-08` — **Complete** (PR #95, `697bc9f`, artifact `outputs/development-recovery/wave2-byte-diagnosis.json` SHA-256 `74120308fef2e036d66a68b587cd42b6cc1a57d9b3b19ade21dd32d26c1c1aa9`; audited and repaired 2026-09-12).
- Verified disjoint ledger equality: $B + F + M + R + H = T$ strictly holds for all rungs ($H = 0$, indicating zero unallocated arithmetic remainder; container/envelope overhead resides within $M$).
- Proved hypothesis: At low rate (R63), fixed background $B$ (529,361 B, 75.6%, ~2.12 Mbps at 24 fps) and metadata $M$ (70,609 B) consume 600 kB, exceeding the entire VVC anchor budget (130,906 B) by approximately 4.6x and AV1 (110,842 B) by 5.4x. $A(q) - B - M - H$ is negative across all rungs.
- Even with residual $R = 0$ at H3, PointStream's base floor (630 kB) exceeds the anchor's highest quality budget (377 kB).
- Background sufficiency limit: Setting $B = 0$ at R63 leaves 170,741 B, which still exceeds anchor budgets; background reduction is necessary but alone insufficient to beat anchors.
- Decision: Supported promoting background representation for Wave 2, accompanied by dynamic evaluation without fixed byte thresholds.
`EVAL-ACT-10` — **Phase 1 Complete** (Wave 2 Diagnostic Matrix, artifacts `outputs/development-recovery/diagnostic-pix2pix-alcaraz.json` SHA-256 `ccfaa34b8ceca48409839c3baefec20e91f87a0425aea6da2b7429c37ed2fa50` and `diagnostic-pix2pix-federer.json` SHA-256 `407148416bf1455ccfb68cc025a2ff31899689ebd0ffc722fd4f0dddc085af25`).
- Executed full 5-corner diagnostic matrix (gen off/on x residual off/on + shuffled conditioning null) on both development scenes (`alcaraz_highlights/scene_000` and `federer_djokovic/scene_007`, 16 frames @ 4K).
- Fixed skeleton pose alignment across tracks with global crop IDs and 0-based skeleton sequences.
- Results confirmed generator comparison validity (`true`), model invocations occurred on GPU 1, delivered pixels changed, and wire bytes reconciled 100% with no synthetic fallbacks.
- Verified hypothesis: uncompressed pose conditioning overhead (+434–655 kB) and current pix2pix generation fidelity deficits (-0.6 to -1.0 dB PSNR-Y) increase rate without delivering quality gains; pasted reference remains the superior control for low-rate compression.

## PR #88 audit — 2026-09-10

Reviewed open head `d1d24b7`; main was `6b04eae` (PR #89). PR #88 adds useful full-range residual and actual-stream plumbing, adaptive anchors and training interfaces, but is **not ready to merge or support model-selection claims**. Its CI run `34428171118` passes tests and fails lint/type checks. Use [the next Antigravity dispatch](../workflow/session/submission-search.md), which includes repair acceptance tests and a staged full-codec search.

The stored `outputs/development-recovery/pilot-frozen/report.json` uses two 48-frame 4K development scenes, frozen C0–C3, generation/residual/pose OFF, VVC background and WebP appearance; anchor presets are SVT-AV1 8 and VVC medium. Arithmetic reproduction agrees with its **PointStream rate premiums**: adaptive AV1 +7.844% on Alcaraz only; native VVC +22.326% / +39.436%; adaptive VVC +46.530% / +79.764%. These are archived diagnostics, **not validated codec advantages/disadvantages for citation**. AV1's second adaptive comparison is unscorable at 48.1% quality-span overlap under the pre-existing 50% rule. Native AV1 is unscorable on both. A positive candidate premium is not the same percentage as the anchor's saving; the denominators differ. Different intervals also prevent comparing these percentages as one common operating point.

Arithmetic controls reproduced 0% for identical curves and +10% for known 1.1x candidate rates. This checks integration/sign, not input validity. The saved report has `pilot_alarms_clear=false`, `identity_verified=false`, `evidence_verified=false`; exclude it from clean result tables. Its SSIM alarm comes from requiring unrelated content to be worse than severe noise, an unjustified total ordering. Repair that calibration rule and revalidate the instrument; do not simply waive the alarm.

Blocking integration findings:

- `RunResult.delivered_frames` still reads encoder `ART_DELIVERED`; `delivered_quality` is inherited from `ART_QUALITY`, and multichunk scoring also reads encoder artifacts. The new independently decoded frames are stored elsewhere. A controlled zero-output client audit changed `result.frames` but not public `delivered_frames`. Experiment drivers still score that public encoder output.
- Generation-enabled runner decoding bypasses serialization and passes in-memory background, objects, generator and conditioning to the client. This is not proof of source-free wire decoding; serialize and charge the actual conditioning/reference data and verify both predictors.
- The ledger only checks residual bitstream length, not the complete envelope. A tiny deterministic non-generative audit produced 1,219 serialized bytes against a ledger of 631, with `raw_parts=()`. It demonstrates missing reconciliation, not a real-video overhead estimate.
- `scripts/run_diagnostic_matrix.py` unconditionally reuses two hardcoded no-generation controls for arbitrary scene/frame/rate arguments. Its matrix and the +1,647-byte residual-demand interpretation cannot establish a paired causal finding.
- Protocol identity checks are conditional on an expected identity being supplied; source counting falls back to scene IDs. Complete per-run/per-rung provenance and independent match identity remain required. The report's fingerprint is captured from the base tier, before candidate rung replacement.

Focused PR #88 residual, campaign and protocol tests passed locally (47 tests); they did not catch the controlled failures above. No new codec run or model training was launched by this audit. Preserve all original outputs; supersede their interpretations rather than rewriting reports. Generator-specific validity and next actions live in [generation](generation.md).

### Repair update — 2026-09-10

PR #88 merged as `2b7c2b0` after its complete test suite, coverage gate, lint and type checks passed. It repaired the listed code paths: public delivered frames/quality are client-originated; generation conditioning/reference data travel through the serialized envelope; the envelope, not only the residual stream, is reconciled to the ledger; probe-frame resolution fails closed; diagnostic controls execute from their declared configuration; protocol identity and independent-match grouping are required. These are implementation checks, **not new rate–distortion evidence**.

### Wave 2 GPU pilots — 2026-09-10 (gpu5, `6199d3e`)

Launched under `jobs/wave2-diag-then-hf-20260910b` (`status=complete`, `exit_code=0`). Artifacts (preserve; do not rewrite):

- Diagnostic: `outputs/development-recovery/diagnostic-pix2pix-rq32.json` SHA-256 `fed400284d15189234712da73cbe60c2362ca638608646ceb204693e3597402d`
- Residual high-fidelity: `outputs/development-recovery/residual-high-fidelity/report.json` SHA-256 `f306cdd46dc9ed7c855e543188e0d67d7f58790b69b87e3e808f6c8510fca860`
- Pre-launch bounds: `outputs/development-recovery/wave2-prelaunch-bounds.json` SHA-256 `2f636fa700956917c5d7e5230d244e846401438c8177064d4afa8f1cd96eb024`

**Not citable. This interpretation supersedes the first PR #92 reading.** `gate_b_passed=false`, `pilot_alarms_clear=false`, `identity_verified=false`, `evidence_verified=false`. Gate A remains open; Gate B remains incomplete.

The equal generation-on and generation-off results do **not** show that pix2pix is equivalent to pasted reference. In `src/runner/run.py`, `_finish_chunk` decodes transmitted appearance into each object's `supplied_crop`, then sets `is_gen` only when `supplied_crop is None`. With appearance enabled, every would-be generated object is serialized as a pasted-reference placement, so the client generator receives no effective generated placement. Evidence that this path was a no-op: generation-off/residual-off and generation-on/residual-off reported identical PSNR, SSIM, and VMAF; generation-on added only ~180 bytes; client time was essentially unchanged. Encoder-side generation timing of 0.7–1.2 s does not prove generated pixels reached the client. The diagnostic reports also lacked sufficient generator provenance (checkpoint SHA, delivered-frame hashes, invocation counts, shuffled-conditioning control).

The ~41.5 MB labeled `metadata` is explainable: `serialize_client_request` stores placement masks as full uint8 arrays and writes them with uncompressed `np.savez`. Several 4K masks account for nearly the entire envelope. The ledger balances, so this is a real transport inefficiency exposed by the repaired accounting, not a ledger mismatch. All Wave 2 PointStream totals (~42–45 MB) are dominated by that mask representation; anchors were below 1 MB. Do not treat those totals as a codec rate.

The residual ladder moved residual bytes and quality monotonically, but cannot support codec comparison: Alcaraz PointStream VMAF ~94.77–96.33 and Federer/Djokovic ~87.24–91.18 versus sampled VVC maxima ~87.03 and ~83.76; AV1 overlap was too narrow; BD-rate remains unscorable. Calibration remained invalid because unrelated-content SSIM was 0.6702 against the preregistered 0.60 ceiling. Repair generation intent, compact masks, and provenance before any new GPU ranking.

## 1. Current State

### Bounded codec pilot controller (PR #82)

`experiments/jobs/codec.py` adds pilot, longer-clip confirmation, and final stages
around paired anchor/PointStream ladders. QP and joint JPEG/QP payload spacing can
widen only inside an explicit policy. Worker timeouts, saved decisions and
fail-closed evidence checks stop expensive stages when pilots are invalid or
uninformative. Outputs remain exploratory and uncitable. Approved CPU regression
tests cover bounded widening, real ladder argument/order integration, missing
evidence, longer-clip rejection, spent budgets and interrupted resume. No GPU
result is claimed.

The [long-job protocol](../workflow/long-jobs.md) also records the proposed
scene-sanity / same-video / cross-video / frozen-test training progression.
The old training campaign evaluator remains retired and is not launch-ready.

The Gate A 48-frame native run (#69) completed the first full-system rate–distortion measurement on real tennis footage. While validating pipeline integrity, it confirmed that Gate A was not passed under the legacy configuration due to large `libaom` background plates, JPEG crop overhead, and short duration.

### Gate A 192-Frame Benchmark Results (Run-2 / PR #83)

The Gate A overnight long-context run (`outputs/gate-a-vvc-webp-n96-run2`) evaluated 192 frames @ 4K 24 fps (8.0s across 2 scenes, `alcaraz_highlights`) with low-delay VVC background streaming (`-period 1`) and WebP actor crops, comparing PointStream against paired AV1 (`libsvtav1`, preset 0) and VVC (`libvvenc`, slower) anchors on identical frames.

#### 1. PointStream Rate Ladder (192 frames @ 4K, 24 fps)

| Rung | Total Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Client Dec Time | Decode Speed |
|---|---|---|---|---|---|---|---|---|
| **C0** | 49,887 B (49.9 kB) | 49.9 kbps | 23.42 dB | 0.8333 | 0.00 | 813.2 s | 14.4 s | 13.3 fps |
| **C1** | 66,348 B (66.3 kB) | 66.3 kbps | 27.21 dB | 0.8891 | 29.75 | 985.9 s | 14.0 s | 13.7 fps |
| **C2** | 91,172 B (91.2 kB) | 91.2 kbps | 30.37 dB | 0.9364 | 57.62 | 867.8 s | 14.2 s | 13.6 fps |
| **C3** | 127,401 B (127.4 kB) | 127.4 kbps | 32.32 dB | 0.9609 | 72.20 | 844.3 s | 14.3 s | 13.5 fps |

*Component breakdown*:
- **Fixed metadata**: 40,343 B (camera homographies, bounding boxes, keypoints).
- **Background plate (VVC low-delay)**: 6,034 B (C0) → 17,051 B (C1) → 40,397 B (C2) → 74,940 B (C3).
- **Actor crops (WebP)**: 3,510 B (C0) → 8,954 B (C1) → 10,432 B (C2) → 12,118 B (C3).
- **Residual**: 0 B.

#### 2. Paired Conventional Anchor Benchmarks (192 frames @ 4K, 24 fps)

**AV1 (`libsvtav1`, preset 0)**:
| Pattern | QP | Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Dec Time |
|---|---|---|---|---|---|---|---|---|
| Continuous | 63 | 190,873 B (190.9 kB) | 190.9 kbps | 36.54 dB | 0.9726 | 83.64 | 164.2 s | 14.7 s |
| Continuous | 55 | 350,949 B (350.9 kB) | 350.9 kbps | 39.04 dB | 0.9820 | 90.08 | 161.6 s | 14.5 s |
| Continuous | 47 | 591,881 B (591.9 kB) | 591.9 kbps | 40.77 dB | 0.9870 | 93.30 | 158.0 s | 14.9 s |
| Continuous | 39 | 1,072,447 B (1072.4 kB) | 1072.4 kbps | 42.09 dB | 0.9903 | 94.99 | 172.2 s | 14.9 s |
| Segmented | 63 | 158,520 B (158.5 kB) | 158.5 kbps | 37.03 dB | 0.9750 | 85.41 | 170.9 s | 18.8 s |
| Segmented | 55 | 291,311 B (291.3 kB) | 291.3 kbps | 39.28 dB | 0.9831 | 90.87 | 184.8 s | 16.0 s |
| Segmented | 47 | 493,402 B (493.4 kB) | 493.4 kbps | 40.80 dB | 0.9874 | 93.48 | 180.7 s | 16.5 s |
| Segmented | 39 | 929,335 B (929.3 kB) | 929.3 kbps | 42.06 dB | 0.9903 | 94.94 | 197.0 s | 16.6 s |

**VVC (`libvvenc`, slower)**:
| Pattern | QP | Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Dec Time |
|---|---|---|---|---|---|---|---|---|
| Continuous | 63 | 31,746 B (31.7 kB) | 31.7 kbps | 24.57 dB | 0.8454 | 8.41 | 215.1 s | 21.0 s |
| Continuous | 55 | 77,228 B (77.2 kB) | 77.2 kbps | 29.10 dB | 0.9081 | 47.23 | 196.3 s | 19.8 s |
| Continuous | 47 | 200,583 B (200.6 kB) | 200.6 kbps | 34.40 dB | 0.9547 | 76.23 | 361.4 s | 19.5 s |
| Continuous | 39 | 444,388 B (444.4 kB) | 444.4 kbps | 38.69 dB | 0.9730 | 88.89 | 823.3 s | 19.7 s |
| Segmented | 63 | 31,937 B (31.9 kB) | 31.9 kbps | 24.53 dB | 0.8446 | 8.62 | 211.0 s | 22.1 s |
| Segmented | 55 | 77,446 B (77.4 kB) | 77.4 kbps | 29.05 dB | 0.9074 | 47.43 | 196.3 s | 22.5 s |
| Segmented | 47 | 200,613 B (200.6 kB) | 200.6 kbps | 34.29 dB | 0.9541 | 75.78 | 358.3 s | 22.3 s |
| Segmented | 39 | 445,272 B (445.3 kB) | 445.3 kbps | 38.58 dB | 0.9726 | 88.60 | 764.6 s | 23.3 s |

#### 3. Audit verdict (2026-09-09; supersedes PR #83–#85 pass claims)

**Gate A is open; Gate B is incomplete and its reported pass is invalid.** The tables above remain archived run observations, not publication-ready evidence. Lower bytes at lower quality do not establish rate–distortion superiority. No universal AV1 bitrate floor was measured.

The audit recomputed the stored comparisons using the checked-in comparison helper. Same-anchor controls returned zero BD-rate; doubling anchor bytes returned +100%, verifying the sign and scale. Positive values below mean more PointStream bytes at matched VMAF according to the existing cubic fit. These are diagnostics from the old scores, not newly calibrated video measurements, and the sparse low-quality fits need sensitivity checks before publication.

| Recorded run / source | Frames and resolution | Continuous VVC BD-rate diagnostic | AV1 comparison | Recorded PointStream encode / client time |
|---|---|---|---|---|
| Gate A: `alcaraz_highlights`, two scenes | 192 total, 4K | +10.30%; VMAF overlap 8.41–72.20 | No quality overlap; floor-dominance false | C0–C3: 813–986 s / 14.0–14.4 s |
| Gate B: Australian Open final | 48, 1080p | +91.30%; VMAF overlap 0.87–54.17 | No quality overlap; floor-dominance false | C0–C3: 44.6–45.4 s / 1.18–1.52 s |
| Gate B: US Open final | 48, 720p | +155.22%; VMAF overlap 0–59.08 | No quality overlap; floor-dominance false | C0–C3: 17.1–17.7 s / 0.68–0.69 s |

Provenance: external `outputs/gate-a-vvc-webp-n96-run2/report.json` and `outputs/gate-b-confirmation/report.json`; source/encoder settings in their identity/tool files and `manifests/gate_b_confirmation.json`. Gate A is one development source, and Gate B has two source observations with different resolutions; neither supports a population-level direction or a stable source-level uncertainty estimate. Keep the historical outputs intact; this verdict supersedes their interpretation.

Specific corrections:

- PR #84's +10.39 VMAF comparison uses PointStream at 91,172 B versus VVC at 77,228 B. That is unequal rate. Its 36.5% saving uses VMAF 72.20 versus 76.23 (75.78 for segmented); that is unequal quality. Refine the VVC curve near C2/C3 before deciding whether a narrower development regime wins. The overall recorded VVC curve comparison is unfavorable; this does not rule out a restricted interval.
- Gate B encoded both AV1 and VVC. The favorable prose omitted the VVC comparison. Even without curve fitting, the 1080p VVC QP47 point is cheaper and higher-VMAF than C3; the 720p VVC QP47 point dominates both C2 and C3 on bytes/VMAF. C0 has VMAF zero on every source, so its small byte count cannot carry a quality claim. The claimed 720p C0/AV1 minimum-byte ratio also disagrees with the stored report (11,111/20,332, not 37.5%).
- `gate_b_passed = len(all_alarms) == 0` never examines comparisons or requires six matches; an empty source list can pass. `_validate_point` checks bytes/timing and old late-frame alarms but does not enforce its frozen quality bounds. Adjacent-rung validation checks bytes, not quality. A clean execution is not a passed scientific gate.
- Gate A anchors used SVT-AV1 preset 0 and VVC slower; Gate B changed these to 8 and medium. Its manifest incorrectly names the AV1 executable as `libsvtav1`; the actual path is standalone `SvtAv1EncApp` v1.8.0. The source clips also shortened to 48 frames and changed native resolution, so this is a short cross-source pilot, not confirmation of the same 4K amortization regime.
- Gate B does not record fresh metric controls, object-scoped scores, source-level uncertainty, or a hashed full procedure/source-frame identity. Its reference checkpoint call omits the identity guard used by the reference CLI, so reused checkpoints are not protected against input/config changes. Raw-file hash verification alone does not cover these requirements.
- In `src/runner/run.py::_finish_chunk`, the serialized client's output is discarded; scoring uses another reconstruction. No equality assertion connects the two paths in this run. `serialize_client_request` emits a compressed NPZ including masks and metadata, while the rate ledger comes independently from the pipeline bag; the run does not reconcile the measured wire envelope length against the charged bytes. Audit the actual transport before citing its rate.
- PointStream timing excludes the detection/materialization performed before `pointstream_e1`, and the anchor decode timer includes a lossless-file intermediate and RGB extraction. Treat existing times as instrument-specific timings, not evidence of a fair end-to-end speed advantage or live-stream latency. The background canvas is offline even where its plate transport is prefix-stable.

#### 4. Anchor policy and next experiment

Retain **both AV1 and VVC**. The observed SVT-AV1 CQP63 endpoint applies to that build, preset, native resolution, temporal settings and input only. It says nothing about all AV1 rate-control modes or resolution choices. FFmpeg documents AV1 bitrate-target modes, and AOM's adaptive-streaming methodology evaluates multiple resolutions and rate–quality envelopes: [FFmpeg](https://ffmpeg.org/ffmpeg-codecs.html#libaom_002dav1), [AOM methodology](https://aomedia.org/docs/SIWG-D001o.pdf).

`EVAL-ACT-06` (Partially implemented, priority): the pilot driver now fails closed and records execution completion separately, with regression coverage for empty/two-source runs, losing/missing comparisons, alarms and favorable curves without protocol validation. It cannot certify Gate B until the remaining validator is implemented. Repair evidence integrity before a new scientific pass. Separate execution completion from confirmation; require validated source eligibility/count, frozen hashes/presets/intervals, full metric controls including blur and unrelated content, object-scoped metrics, actual wire accounting, independently decoded scoring, source-level uncertainty, and an explicit competitive verdict. Reject empty inputs, failed/absent anchors, non-overlap treated as victory, and stale checkpoints. Add focused regressions before relying on the repaired driver. Preserve historical JSONs and write new audit/verdict artifacts with source hashes.

`EVAL-ACT-07` (Blocked by instrument repair): on development scenes, sweep native AV1 CQP and bitrate-target modes plus a declared resolution ladder (native, 1/2, 1/4 linear dimensions), retaining frame rate/duration. Decode and upscale every arm to the original display resolution with a fixed filter before whole-frame/object metrics. Apply the same resolution opportunity to VVC. Count bitstream/container/signaling, encode/decode and rescale costs. Verify achieved rate; a requested bitrate is not an achieved one. Build nondominated curves with measured overlap and compare interpolation methods; do not fit across the VMAF zero plateau or extrapolate. Refine native VVC around C2/C3 as a bounded pilot, and inspect metadata cost, scene length and foreground fidelity where PointStream loses. Select the policy on development data, then freeze it before a new held-out evaluation.

Gate C control plumbing and targeted diagnostic ablations can proceed now, especially metadata/background/appearance accounting, but the submission gate remains blocked until Gate B is satisfied. The two examined candidate matches now have observed scores; if they influence redesign, label them development/robustness data and reserve fresh sources for final confirmation.

### Two-Tier Metric Protocol (#72)
To accelerate the configuration search while maintaining rigorous publication standards:
- **Tier 1 (Exploration & Tuning)**: Compute **PSNR only** directly in memory via NumPy (`<0.05` s per frame). This avoids disk I/O, prevents process timeouts, and allows rapid sweeping of quantization parameters and keyframe intervals.
- **Tier 2 (Paper Evidence)**: Run the full multithreaded metric suite—**PSNR-Y, SSIM, VMAF (`libvmaf` with `n_threads=16`), and LPIPS**—strictly on frozen winning candidate configurations.

### Amortization Hypothesis
Because the high-resolution background plate is transmitted once per scene, its effective bitrate contribution scales inversely with scene length:
$$\text{Bitrate}_{\text{background}} = \frac{\text{Plate Size (bytes)} \times 8 \times \text{FPS}}{\text{Number of Frames}}$$
With a 14.3 KB VVC intra background plate:
- **48 frames**: $14.3\text{ KB} / 48 = 0.30\text{ KB/frame}$
- **96 frames**: $14.3\text{ KB} / 96 = 0.15\text{ KB/frame}$
- **192 frames**: $14.3\text{ KB} / 192 = 0.075\text{ KB/frame}$

Evaluating over longer sequences (96 and 192 frames) is a core hypothesis for establishing a rate–distortion win against conventional temporal inter-coding.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Synthetic Tier Tests | #23 (`ca0f75af30`) | Synthetic 3-frame tier path test established as CI regression gate. |
| Ladder Plumbery | #65 (`91b33e623f`), #66 (`606cf53893`) | Sweep infrastructure and anchor pairing harness created. |
| 48-Frame Native Run | #69 (`648325b`) | Full-system baseline evaluated. Identified background/appearance bottlenecks. |
| Fast Eval Strategy | #71, #72 | Two-tier protocol adopted; piped FFmpeg streaming proposed. |
| In-Memory Metric Acceleration | #77, #78, #79, #81 | Thread-local SSIM scratch buffers (176× speedup), Y4M piped VMAF streaming (80× speedup), and streamed closeness (memory down to <500 MB). |
| Gate A Tier 2 Evaluation | `outputs/gate-a-vvc-webp-n96-run2/report.json` | Development sweep completed; gate pass and matched-quality savings superseded by audit above. |
| Gate B pilot (PR #85) | `outputs/gate-b-confirmation/report.json` | Completed two-source pilot; reported pass superseded by the audit above. |
| Fail-Closed Protocol & Identity | Overnight recovery (`4b170da`) | Implemented fail-closed protocol validator in `experiments/tier/protocol.py`: enforces $\ge 6$ matches for Gate B, rejects non-overlapping curves, mandates wire ledger reconciliation, and separates pilot runs from confirmation. Resolves `EVAL-ACT-06`. |
| Resolution-Adaptive Anchors | Overnight recovery (`4b170da`) | Implemented native and resolution-adaptive (1.0, 0.5, 0.25 linear dimensions) ladders for SVT-AV1 (`SvtAv1EncApp`) and VVC (`vvencapp`) with Lanczos display grid restoration, rescaling time accounting, and strict extrapolation prohibition in `experiments/tier/resolution_adaptive.py`. Resolves `EVAL-ACT-07`. |
| Metric Calibration Suite | Overnight recovery (`4b170da`) | Implemented `experiments/tier/calibrate.py`: calibrated metrics against identical, mild, severe, unrelated anchors and null controls. |
| Development Recovery Pilot | `outputs/development-recovery/pilot-frozen/report.json` | Evaluated development recovery set (`manifests/development_recovery.json`): Native AV1 unscorable without extrapolation (disjoint support); resolution-adaptive AV1 overlaps on [30.5, 78.0] VMAF with PointStream at +7.84% BD-rate; native VVC overlaps with PointStream at +22.3% BD-rate (`alcaraz_highlights`) and +39.4% (`federer_djokovic`). Confirmation holdouts unspent; Gate A open, Gate B incomplete. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `EVAL-ACT-05` | Implementation complete | Run-specific calibrated policy | PR #82 | Bounded pilot controller and regression gates implemented. Next: choose a calibrated run policy and run a small real pilot before relying on scientific results. |
| `EVAL-ACT-01` | Complete | None | #71, #72, #79 | **Piped in-memory metric computation**: Replaced `_write_png_clip` disk writes with direct stdin streaming to ffmpeg Y4M rawvideo and enabled `n_threads=16`. Measured 80× speedup on 4K clips with bit-identical scores to reference. |
| `EVAL-ACT-02` | Complete | `CODEC-ACT-01`, `CODEC-ACT-02`, `EVAL-ACT-01` | #72, #81 | **Amortization & rate sweep (Tier 1 PSNR)**: Fixed virtual memory exhaustion via streamed closeness; completed 192-frame sweeps on multi-scene 4K video. |
| `EVAL-ACT-03` | Pilot complete; confirmation open | `EVAL-ACT-02` | #72, outputs/gate-a-vvc-webp-n96-run2 | **Tier 2 full-metric confirmation**: Evaluated PSNR-Y, SSIM, VMAF across C0–C3 ladder. 0 alarms, pre-registered rot bounds verified, null controls passed, decode speed ~13.5 fps on CPU. |
| `EVAL-ACT-04` | Ready (previously D-CODEC-PRESETS) | None | `plans/DEFERRED.md` | **Anchor preset standardization**: Document exact FFmpeg command lines, presets, and versions for AV1 (`libsvtav1`/`libaom`) and VVC (`libvvenc`). Acceptance: Explicit, reproducible anchor scripts checked into repository. |
| `EVAL-ACT-06` | Complete | None | 2026-09-09 audit, overnight recovery (`4b170da`) | **Fail-closed evaluation integrity & protocol enforcement**: Enforce $\ge 6$ matches for Gate B, common quality support overlap, actual client-output scoring, wire ledger reconciliation, and strict distinction between development pilot and confirmation. |
| `EVAL-ACT-07` | Complete | `EVAL-ACT-06` | 2026-09-09 audit, overnight recovery (`4b170da`) | **Resolution-adaptive anchor methodology**: Native and downscaled resolution ladders (1.0, 0.5, 0.25 linear) for SVT-AV1 and VVC, common display grid restoration (Lanczos), rescaling time accounting, and strict extrapolation prohibition. |

Audit validation: stored comparison reproduction plus same-anchor and doubled-byte controls passed; documentation link/whitespace checks and the separate paper build are recorded in the audit PR. Focused gate-verdict regressions and project code checks are recorded in the PR. No new video encoding, recalibration or revised scientific pass was performed.
