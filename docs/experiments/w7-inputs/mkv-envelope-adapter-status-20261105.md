# Source-bound IVF and MKV envelope analysis readiness

The primary baseline uses complete original IVF byte lengths with the fixed preinstalled DAV1D, limited-BT.709 and cubic-upscale receiver profile. Complete delivered MKV is a separately identified transport sensitivity. This does not imply generic metadata interoperability, newly measured IVF RGB, or an archived historical RGB hash; the strict metadata failure remains failed. The four new scores are attached to original IVF bytes by independently verified native-YUV equality and the fixed scorer conversion, an inference rather than a fresh IVF neural replay.

The `w7_mkv_envelope` loader authenticates the approved historical and new-score reviews, nine old score archives, all three new/pilot publications, fixed role inventory and delivery map. It independently reads and hashes all 75 original IVF members from 18 SHA-verified encode archives, checks complete framing byte lengths, native-YUV identity and the fixed 75-role oracle, and validates all four deduplicated pairs. Both framings use the same 71-stream primary and 57-role historical-family sensitivity. It retains per-role native, score, GPU-class, IVF and MKV provenance.

The report uses the existing G5 achievable hull, W1 bracket-only family comparisons and dominance, W7 t intervals and missing-DCVC sensitivity, and SHA-pinned corrected W9 7,200/3,600/1,800-second bank rates. ±0.001 uniformly perturbs anchor errors as a deterministic stress, not a statistical bound. The hybrid has a single measured point per window, so no hybrid BD-rate is invented.

No actual outcome has been computed. The [CPU-only pilot](mkv-analysis-synthetic-cost-pilot-20261105.json) verified actual archive and IVF inputs, then ran ten complete paired IVF/MKV reports on synthetic scores: 1.620 seconds wall, 1.478 CPU, first paired report 0.942 seconds. W7's decision uses analytical t intervals; no bootstrap is required. The proposed and root-approved envelope is 60 CPU-seconds for the command and 60 seconds for external/audit, inclusive 120 CPU-seconds. An independent source review and outcome gate remain pending.

Final execution sets the exact 60-second process CPU cap and validates a SHA-pinned root gate **before reading score or transport artifacts**. The gate must bind the module SHA, accepted four-score audit SHA, `max_cpu_seconds: 60`, and the entire tracked Python source manifest plus numerical runtime identity printed in the pilot receipt. The manifest and runtime are rechecked before output. A source change invalidates the gate. The final report writes a fresh exclusive path and must not be run without a gate approved by root.

```sh
python -m experiments.tennis.w7_mkv_envelope --mode final \
  --historical-inbox /private/tmp/w7-original-53-audit \
  --pilot-publication /private/tmp/w7-shared-aa3070-audit/smoke.tar \
  --new-smoke-publication /private/tmp/w7-narrow-40ee-audit/smoke/published.tar \
  --new-full-publication /private/tmp/w7-narrow-40ee-audit/full/published.tar \
  --ivf-archive-dir /private/tmp/w7-fps-oracle-archives \
  --gate ROOT_APPROVED_GATE --gate-sha256 ROOT_APPROVED_GATE_SHA \
  --output /private/tmp/w7-ivf-mkv-final-NEW.json
```

This document does not create a gate or authorize a new GPU, paper or ledger action.
