# MKV envelope analysis readiness

The CPU-only `experiments.tennis.w7_mkv_envelope` loader authenticates both approved historical receipts and all nine old score archives; the accepted SHA-pinned 40ee/aa307 audit, two 40ee stage publications and prior aa307 publication; the fixed 75-role inventory and delivered map; and each full-timeline score member. It joins all 75 roles, verifies the four equal-IVF/YUV/score/byte-count duplicate pairs and preserves the two independent MKV hashes. The resulting 71-stream primary and 57-role planned-family sensitivity views retain source identity, per-row GPU class and exact native and billed byte provenance.

The report implements the existing G5 achievable hull, W1 bracket-only log-rate family comparisons, W1 family dominance, W7 pooled and per-match t intervals and missing-DCVC sensitivity. It uses the SHA-pinned corrected W9 billing artifact and its original all-assets formula for 7,200/3,600/1,800 seconds. The 3,600/1,800 rates agree with the existing W9 results within 7.2e-15 kbps. Whole and foreground LPIPS/DISTS gaps and all roles are retained. Three fixed stress cases add −0.001, 0 or +0.001 uniformly to anchor errors; this is deterministic sensitivity, not a statistical or per-row bound. The separate curve helper uses K5 PCHIP over real quality overlap and returns no BD number without overlap. The hybrid currently has only one measured point per window, so a hybrid BD-rate is not invented.

No actual quality outcome has been computed. The [CPU-only pilot](mkv-analysis-synthetic-cost-pilot-20261105.json) authenticated the actual inputs, then ran ten complete reports on synthetic scores only: 1.260 s total wall and 1.080 CPU; first report 0.936 s. W7's original decision uses analytical t intervals, so there is no 1,000-replicate bootstrap to forecast. The pilot is local warm-cache evidence, not a production performance guarantee. Proposed next allocation is **60 CPU-seconds for the final command plus 60 CPU-seconds external/audit = 120 inclusive**, with a 1.5× measured margin and the remaining amount as explicit reserve; root must review source and allocate it before producing outcomes.

The `--mode final` entrypoint requires an actual SHA-pinned root gate with `approved: true`, `analysis_source_sha256` matching this module, `new_score_audit_sha256` equal to the accepted audit and positive `max_cpu_seconds` at most 500. It writes only an exclusive new output path. Pilot invocation and corresponding final command (add `--gate PATH --gate-sha256 SHA` to the latter):

```sh
python -m experiments.tennis.w7_mkv_envelope --mode pilot \
  --historical-inbox /private/tmp/w7-original-53-audit \
  --pilot-publication /private/tmp/w7-shared-aa3070-audit/smoke.tar \
  --new-smoke-publication /private/tmp/w7-narrow-40ee-audit/smoke/published.tar \
  --new-full-publication /private/tmp/w7-narrow-40ee-audit/full/published.tar \
  --output /private/tmp/w7-mkv-analysis-cost-pilot-NEW.json
```

For final use replace `pilot` with `final`, add the gate arguments, and use a new final output path. The new source and its local input paths are subject to root review; no authorization or gate is created by this document. No paper, ledger or GPU action follows.
