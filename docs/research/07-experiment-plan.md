# 7. Experiment preparation and execution plan

Status: documentation redesign, 28 September 2026. Tennis remains primary.
No jobs are authorized or launched by the presence of a command in this file.
The current task prepares the research plan; subsequent compute uses the Mac
fleet dispatcher and a same-entry-point bounded smoke before extended work.

## Readiness and decision order

| Gate | State at documentation audit | Evidence needed to close it |
|---|---|---|
| Code/source inventory | Audited at `086ae9035bc0ff7b552c05eb04904224f9abb32a`; source chat recovered | Reconcile exact payload/metric paths with chosen frozen experiment revision |
| Historical evidence | Source records found; external files not rehashed | E00 verifies original artifacts and exposure; no recertification from prose alone |
| Dataset | SAM3.1 development pilot documented; racket/joint views quarantined; no certified new training view | E02 closes lineage, review, eligible split and decoded-conditioning parity |
| Metrics | Core PSNR/SSIM/VMAF/LPIPS exist; DISTS and tennis task instruments not qualified | E01 produces calibrated metrics, region policies and frozen task tolerances |
| Generative competitors | Public artifacts inspected, no local inference replication | E03 verifies supported-domain and tennis smoke, complete stream and receiver isolation |
| PointStream generative assembly | Existing serializer/client reusable; actual selected generation-on path not qualified | E04 passes real-checkpoint source-free decode with full accounting |
| Confirmation | Three 1080p reservations, exposure/eligibility/freeze unfinished | E00/E10 confirm match-level separation and prospectively freeze procedure |
| Resources | Historical GPU state is stale | Fresh fleet inspect and launch recheck; exact environment/model/native tools recorded |

The immediately useful next work is E00 artifact intake and E01/E02 CPU contract
validation, then the smallest E02/E03/E04 GPU smoke. **Full training, broad
rate sweeps and confirmation are not ready yet.** Availability checks can be
performed independently; dependent scientific runs wait for their gates.

## Shared record and stop rules

Each card becomes a versioned run manifest under the external data root before
execution. Record: card ID; hypothesis and competing explanation; source hashes,
frame/PTS selection and exposure; exact command; code revision/patch digests;
model/checkpoint/native binary identities; environment; GPU UUID; resource
estimate; output location; seeds; complete byte policy; controls and metric
versions; success/failure/inconclusive rule; maximum configurations/time; and
what decision follows. Use existing protocol/identity/checkpoint utilities from
[implementation](05-implementation.md), rather than inventing another ledger.

Default diagnostic budget is at most six new configurations and 30 minutes
including scoring, after a 10-minute evidence triage. This is a planning cap,
not a runtime prediction. A supported-model installation can exceed it: give
installation its own bounded card and report blocker rather than silently
consuming the experiment budget. Reserve at least one third for controls and
diagnosis. Estimate memory from representative measurements plus fleet margin;
do not assume a published model fits merely because a GPU is idle.

Stop on wrong source, uncharged side information, missing/empty native output,
wrong frame count/PTS, failed metric nulls, source-dependent decode or contaminated
timing. Preserve failures. Inconclusive means no promotion; it does not become a
zero score for an unavailable competitor. Reuse compatible saved decodes for
rescoring before new encoding. Every extended run needs a passing representative
smoke through the **same** codec/preprocessing/training entry point.

## Experiment cards

| ID / question | Smallest useful experiment and controls | Reuse / implementation prerequisite | Artifacts and promotion/stop decision |
|---|---|---|---|
| **E00 Evidence and source intake** | Read-only locate/hash the headroom, background, foreground and SAM pilot records; reconcile decoded source frame IDs. Audit confirmation exposure without quality scoring. | [Evidence ledger](08-evidence-ledger.md), existing manifests and `experiments.tier.protocol`; no re-encoding | Intake table classifies verified/reported/missing/invalid, with external paths and digests. Freeze development roster. Missing artifacts remain historical reports, not new evidence. |
| **E01 Metric qualification** | Identity, color conversion, known blur/noise/shift, frame-order/cut controls; annotate a small fixed development set of court lines, ball/racket visibility and coordinates. | Existing metric registry/calibration/region tests; implement DISTS adapter and qualified temporal/task evaluators only if selected. Verify crop/mask policy on tiny players. | Metric test JSON, annotation instructions and overlays, numerical tolerances/version hashes. Freeze primary metrics and task thresholds before comparison. Stop if identity/degradation or geometry controls fail. |
| **E02 Shared dataset/client conditioning** | Reuse 3 scenes × 16 frames SAM pilot; verify masks and explicit missing observations; finalize quarantine, compare offline/runtime identical-policy observations, then encode→decode→render conditions. | `scripts/audit_dataset_pipeline.py`, `scripts/finalize_sam31_manifest.py`, observation/perception components; no stale crop pairing. Existing pilot lacks pose/racket transport parity. | Visual HTML/PNG audit, manifests/sample digests, validity/coverage/fallback counts, transform and decoded-motion error, real bytes, fresh-client result. Do not activate training until reviewer decisions and split checks pass. |
| **E03 External baseline qualification** | Per selected codec: one short supported-domain case at two released rates, then 16 frames each from contrasting tennis scenes. Compare official reconstruction and wrapper output. | MTTF,GLC-video,GVC-RT first; S²VC optional heavier arm; use official code/weights, pinned separate environments and licenses. No verified local adapters yet. | Install/runtime manifest, persisted complete stream, fresh-process decode, frame hashes, rate reconciliation and visual strip. A failed supported-domain smoke is integration failure; successful native but poor tennis is reported transfer limitation. |
| **E04 PointStream receiver integrity** | Generation off/on × residual off/on where compatible, with two players, changing aspect ratios, decoded appearance and masks, and missing racket observations. Use a real selected checkpoint; mocks only unit coverage. | Existing client serializer/accounting/residual transport; qualify generation-on + supplied reference + residual-off; verify segmentation does not get bypassed by placeholder masks and actors are grouped independently. | Fresh client with source directory unavailable; exact packet sizes and final-frame equality/tolerance; envelope/schema records. Stop any arm needing hidden source arrays or uncoded geometry. |
| **E05 Background cause and headroom** | Paired still, registered plate and coded background-video controls; removal off/on with cheapest valid fill; court vs off-plane regions; one static and one panning scene. | Existing background_campaign/registration/warp_residual probes; same-source final scoring, not own-source headroom alone. | B+maps+correction bytes; court-line/error overlays, non-court errors, sender/client cost. Advance only a representation with useful total budget; do not assume crowd caused prior residual. Cylinder/local warps only if diagnosis supports them. |
| **E06 Foreground and geometry** | Fixed background, same appearance policy: static/articulated paste, temporal coded object crops, ready small generator, ready animation/generative codec. Isolate player/racket/joint and ball transport separately. | E01–04; existing foreground campaigns and registry. Verify MTTF's native feature interface; do not replace it with skeletons. Decoder-derived mask and charged-mask controls. | Object/whole-frame rate–quality and geometry, M/F/R byte breakdown, failures and client resources. Stop dominated tested settings; preserve tradeoffs. Source-mask oracles and target-fitted models are explicitly noncompetitive controls. |
| **E07 Bounded training/adaptation** | Only if E06 diagnoses useful remaining headroom: short training smoke with distinct appearance/target frames, decoded conditioning, shuffled-motion and held-out validation controls. Extend only on meaningful validation progress. | E02 eligible data and compatible checkpoint, common crop/pose renderer, exact training budget per model. No target==reference shortcut. | Loss/control/validation plots, checkpoint identity, shared vs per-video cost, actual model bytes/time. Stop when control fails or gain cannot offset transport/residual cost. Training from scratch not a default deadline task. |
| **E08 Complete tennis curves and ablations** | Freeze 1–2 promising configurations; ≥4 valid rates where available vs AV1,VVC,qualified DCVC,GLC/GVC-RT and MTTF; S²VC if qualified. Paired one-axis ablations for representation, refresh, mask, pose, rigid objects, residual and fallback. | E01–06, E07 only when training required. Existing generation-off low_rate_sweep is a control runner, not this new generative benchmark. | Persisted bytes/decodes, full metrics/time/memory, per-scene curves, BD-rate overlap and failure tables. No overlap → report points; no advantage → document boundary, do not tune metrics after scoring. |
| **E09 Measured reuse/access** | Encode 48-, 96-, and 192-frame windows then actual consecutive eligible points; include real refresh/cut/lighting/view changes. Compare continuous-reference and independently seekable anchors. | E08 qualified configurations; long_scenes loaders/manifests; causal vs offline distinction fixed. | Cumulative bytes AND quality vs time, startup and refresh breakdown. Never multiply a single point's cost to label measured amortization. No positive recurring margin → reject that crossover hypothesis. |
| **E10 Frozen confirmation** | Apply selected procedure and budgets to largest prospectively verified untouched match set; no configuration/metric tuning. | E00 eligibility and E08/E09 frozen code/config/models/metrics/thresholds/rate ladder. Existing 3-match policy must match manifest/verifier. | All matches including failures, match-level uncertainty and scope. Exposed 1080p cannot prove unseen native 4K. If unavailable, explicitly restrict publication claims to development evidence. |
| **E11 Perception and systems** | Blinded randomized equal-rate pairs on qualified decodes; independent task-fidelity questions. Measure synchronized complete sender/client path on available named hardware with cold/warm states and repeated timing. | E01 questions/thresholds and E08 decodes; frozen user-study design/sample-size rationale before collecting responses. | Preference uncertainty, geometry errors, throughput, latency, peak memory, startup/lookahead and fallback frequency. No realtime claim from a fast isolated generator or compositor. |
| **E12 Domain boundary and paper integration** | Small declared secondary-domain test (egocentric optional) with unsupported court module disabled; audit every planned claim/table against eligible results. | E10 if claiming generalization; separate secondary-domain split. Sibling manuscript repository own AGENTS/markers. | State what transfers and fails. Integrate evidence with CLAIM provenance, repair contradictory text, clear HOLE only with data, verify page budget and compilation in paper repo. No success promised by completing a plan. |

E03 installs and E01 instrumentation can proceed independently of background
diagnosis. E07 is conditional, not a reason to delay useful no-training controls.
E10 cannot be replaced by more exposed development clips. Before any optional
new architecture, exhaust the costed simple controls and identify what measured
gap the architecture addresses.

## CPU verification commands

Run in the project environment, from repository root. These are focused
existing tests, not proof of GPU model quality. No need to run dispatcher tests
for a prose-only edit; run them when its behavior/environment is changed.

```bash
python -m pytest -q tests/contracts/test_observation.py tests/components/test_perception_geometry.py tests/components/test_perception_association.py tests/components/test_dwpose_estimator.py tests/experiments/test_audit_dataset_pipeline.py tests/experiments/test_finalize_sam31_manifest.py
python -m pytest -q tests/runner/test_rate_honesty.py tests/runner/test_accounting_information_content.py tests/runner/test_residual_transport.py tests/runner/test_mask_wire.py tests/runner/test_generation_identity_grouping.py tests/runner/test_generation_appearance.py
python -m pytest -q tests/components/test_bd_rate.py tests/components/test_metrics_integration.py tests/components/test_metrics_region.py tests/invariants/test_metric_calibration.py tests/experiments/test_evaluation_protocol.py
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py
```

Add behavior tests for missing source-free generative decoding, DISTS direction
and crop rules, annotation geometry/nulls, complete baseline-byte accounting,
schema/version compatibility and missing-object handling when implementing those
paths. Tests must challenge failures, not simply echo implementation constants.
Use repository-wide CI/type/layer checks for code changes, preserving unrelated
user work. List unavailable dependencies and skipped GPU cases explicitly.

## Bounded launch templates

Inspect before choosing hosts. A historical free GPU is not permission to reuse
its occupancy observation:

```bash
python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
```

After E00 revalidates the original pilot directory and review file, the following
bounded finalization updates quarantine annotations using the CPU. The existing
fleet command still claims an idle GPU to obtain its supervised execution and
provenance; this is not a CPU-only admission path or GPU inference. Run the
all-host inspection above immediately before choosing candidates. Both files are
tracked now, so the stale `--include-untracked` options from the old plan are
omitted. It does not make the training dataset active or rerun segmentation.

```bash
python -m experiments.jobs.fleet launch \
  --gpu-memory-mib 128 --cpu-threads 1 --budget-hours 0.05 \
  --require-path /home/itec/emanuele/pointstream-data/outputs/sam31-unification/pilot-v1-20260927-run-05/audit.json \
  --require-command /home/itec/emanuele/.conda/envs/pointstream/bin/python \
  -- /home/itec/emanuele/.conda/envs/pointstream/bin/python \
  scripts/finalize_sam31_manifest.py \
  --run-dir /home/itec/emanuele/pointstream-data/outputs/sam31-unification/pilot-v1-20260927-run-05 \
  --visual-review manifests/sam31_pilot_visual_review_v1.json
```

Preflight the shared perception input without model inference using
`scripts/audit_dataset_pipeline.py --inspect-only --pilot-manifest
manifests/sam31_pilot_v1.json --data-root <external-root>` plus the exact
SAM source/checkpoint/revision/hash and DWPose model options. Pin those paths in
the E02 manifest, and choose an isolated compatible environment; do not mutate
the shared pinned environment. `--output-dir` and `--preflight-output` must be
external. The real pilot uses this same entry point without `--inspect-only`.
`--dwpose-device cpu` is supported and must be reflected in timing; CUDA provider
fallback is not GPU-speed evidence. Do not use full-dataset flags until that
same path passes its representative smoke and visual review.

For a pure low-rate native-codec infrastructure smoke, the existing entry point
is `python -m experiments.tier.low_rate_smoke --codec av1 --qp 63 --preset 0
--out-dir <external-job-dir>`. This is not a generative model smoke or paper
evidence. For each external baseline, E03 must first record the **actual pinned
official CLI**, settings and input adapter; no fictitious unified generative
runner command is supplied here.

Fleet launch snapshots clean HEAD and claims a GPU UUID immediately before
execution. Pass selected dirty files only when intended, then record their
checksums. Jobs are detached; preserve returned job ID and use `fleet status
JOB_ID`/`fleet cancel JOB_ID`. No auto-migration/replay. See
[setup](../setup.md) and [long jobs](../workflow/long-jobs.md) for admission,
logs, external roots and contamination handling.

## Completion criteria for preparation

Documentation is complete when every research question maps to source evidence,
an available or explicitly missing implementation, a bounded test, and a
decision/output. Execution readiness is stronger: E00 source access, E01
instruments, E02 dataset/client contract, E03 competitor smokes and E04 actual
PointStream decode must pass for the selected comparison. Until then, report
“preparation ready for staged validation,” not “all experiments ready” or a
successful-journal outcome. The remaining technical work is explicit above.
