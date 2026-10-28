# Experiments: single-stream search

### 2026-10-28 — Accepted parallel batch, frozen initial tests

- User instructed execution of the recommended tasks. New independent
  ceiling3600GPU-s/3600CPU-s: restoration1800, diagnosis600, refinement1200
  conditional on both native correctness and differentiation/entropy audit.
- [Card](../cards/single-stream-search.md) records hypotheses, competing
  explanations, scope, configuration, stage gates and stop rules before outcomes.
- Prior paid memory request20261028T181634Z-05c49090 COMPLETE; never replay.
  Its native SVT anchors are immutable inputs to restoration. Existing stored
  DCVC streams are not portable; diagnosis must re-encode then decode within
  its own request. No new model download or full training authorized.
- All six hosts inspected; doctor for compatiblegpu2/gpu3 ongoing. No new job
  submitted yet. Charged0GPU-s; CPU source audit/preflight ongoing.
- Subagents only gpt-6-luna/max with exact frozen launch/report instructions.

### 2026-10-28 — First two independent requests submitted

- CPU source audit20261028T190840Z-eb011d4b, pending at receipt,
  cap600CPU-s/0GPU. Receipt/private/tmp/single-code-audit-submission.json;
  specSHAd667a3cb48e6f0c368bf1b26e7c8eec82804d3b543acf04754337ee2fc132cfe.
- Native diagnosis20261028T190838Z-3dea4d74, pending at receipt,
  cap600GPU-s. Receipt/private/tmp/single-diagnosis-submission.json;
  specSHA923bc96d1f161dc71f6291edca625faf45717fdfad05a3f33131194e4abc22a6.
- Both use frozen sourceafe86f57620d85161aba5a8dc60e17c8e820fb4e,
  snapshotSHAd30a05a849236fae7c78c3a20cacbed98819035c2b206b66608899e7fae0753b,
  no selected patches/untracked files. All6hosts inspected, doctorPASS.
  CPU request uses all6compatible hosts; diagnosis A6000poolgpu2/gpu3.
- Diagnosis initial sandbox health query hit DNS denial BEFORE job_id creation
  (verified submit implementation). Corrected network permissions, then one
  request accepted. No uncertain replay, GPU execution or scientific retry at
  that preflight failure. Parent owns status and one bounded batch wait.
- Pinned upstream source input33fec4182805ebc5c5ab3cd90715abf70d855d71916837e65e55beca992c0f76,
 133120bytes; DCVC-DTe3deeda1cd99566498504231f6853fd554ec00e9,
 DCMVC4aa95ef4b0176b6076b1ba0396d2cd178258d506. Selected original files only,
 archived with per-file SHAs; no model download or source execution.
