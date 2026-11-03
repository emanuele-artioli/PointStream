# K5c: complete the approved 48-frame pilot using accepted controls

Root operational amendment, 2026-11-03. This stays inside the user's existing
900 GPU-second /1,800 workload CPU-second approval for the one 48-frame pilot.
It authorizes no larger study, training, new model or quality-rule change.

The first request remains failed at its cost-promotion gate. Its actual
16-frame records passed independent CPU audit: native and ball packets,
registered labels, source qualification, model/tool identities and predictions.
The issue was the 420-second allocation to the 48-frame stage; the conservative
whole-command forecast was 138.1097049713 x3 x1.25 +60 =577.9113936424 seconds.
The original approval specified the total ceiling and target, not that smaller
internal allocation. After closing the request, a conservative 201 GPU-s and
1,101 CPU-s envelope leaves **699 GPU-s and 699 CPU-s**. Repeating an accepted
16-frame run would spend that remainder without adding an independent sample.

Root therefore explicitly revises operational allocation for the already
approved target, rather than silently treating the failed gate as passed:

- Use only the exact independently audited 16-frame archive, SHA256
  `f160ce69f66b1272236c9c6c98ea72c65912bc500754141d419ebb15a8e3227e`,
  from `20261103T132157Z-f665bece`. Revalidate its six actual native/ball records
  and bind manifest, exposure, model/checkpoint/native-tool identities and the
  first 16 source-image hashes before any new CUDA execution.
- Measure frames 0–47 of the same preselected development clip exactly once,
  using the same numerical path, six codec settings, one-CPU runtime, source
  qualification, physical byte ledger, independent image judge and PSNR guard.
  Numerical helper ASTs must match frozen source `862d56c2`; the new code only
  changes controller admission and terminal-stage bookkeeping.
- The fleet adapter represents this single registered **real-component pilot**
  in its `smoke` stage with 48 frames. Its `full` stage receives zero frames,
  performs no model/codec operation, and emits an explicit non-result marker.
  The zero-frame marker is never evidence. The actual 48-frame measurements
  require substantive validation and independent post-run audit; they are not
  an infrastructure stand-in or a statistical confirmation. The 16 and 48
  prefixes are nested, not independent samples.
- New campaign at most 570 seconds; real 48-frame command at most 535 seconds,
  validator at most 20 seconds, zero-frame completion at most 5 seconds. The
  measured 517.91-second conservative work forecast fits the command bound;
  its 577.91-second complete forecast fits the **660 GPU-s phase ceiling**.
  Reserve earlier source/admission and release time outside the campaign.
- New phase workload CPU at most **650 seconds**: 20 preparation, 600
  single-core lifecycle, 20 independent audit and 10 monitoring. Cumulative
  envelopes are then at most **861/900 GPU-s and 1,751/1,800 CPU-s**, leaving
  39 GPU-s and 49 CPU-s for uncertainty, not another experiment. The inherited
  single-core/supervisor checks remain cooperative operational safeguards,
  not adversarial containment; unknown administrative telemetry stays explicit.

The new mode reuses valid correctness evidence as the project protocol requires;
it does not relabel the first cost stop or waive source, device, frame, byte or
quality checks. It is covered by the already authorized 48-frame task and
unchanged total caps, so no additional compute allocation is requested. This
amendment is recorded before dispatch and preserves the first spec/results.

Inspect the exact closed request before any action, inspect all eligible hosts
and current jobs, preserve foreign work and select the complete healthy A6000/
Ada pool. Submit the new versioned spec exactly once; unknown submission is
inspected, never replayed. Persist its ID and budgets before one bounded fleet
wait. Stop on a failed identity/source/device/native/packet/frame check,
timeout or cap. No third attempt or larger confirmation is admitted. Outputs
and failures stay separate in Datasets and fleet scratch; old evidence and
PRESLEY remain untouched.
