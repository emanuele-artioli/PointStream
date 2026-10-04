# Fresh-session handoff — 4 October 2026

## Objective

Finish the nearest native VVC comparison against the fresh96-frame receiver, then strengthen AV1 quality overlap. A narrow qualified gain is acceptable; the empirical paper remains valuable without a win. Do not reconstruct the entire chat or reopen the survey.

## Verified evidence

- Fresh generation-disabled alpha receiver: source000 original4K96, frames38–133; **58,139 complete bytes / VMAF0.6.1 62.695164**. Payload-only decoding, costs and all-frame source/metric joins verified. Initial poses repeat and clipping remains; this is not action preservation or historical Gate A recertification.
- Same96 VVC QP50/QPA0: **66,955 bytes /68.285094**. Neither point dominates the other.
- Same96 AV1 CRF59/preset0/1080p Lanczos ladder: **101,542 bytes /88.612482**; quality support still needs extending.
- VVC QP52/QPA0 48-frame pilot: **27,084 bytes /59.528320**, independently verified. Do not compare it directly with candidate96. No new QP52f96 job or registration was created during the last resume.
- Historical Gate A PCHIP: approximately −13.49% in posthoc VMAF50–70 but +14.09% across full support against continuous VVC. Sparse historical encoder-side scores have receiver/accounting defects; preserve both findings.

## First actions

1. Inspect the existing QP52f48 job via canonical fleet status and preserved receipts, then audit all eligible hosts. The last direct GPU1 SSH query was refused; other hosts were not established unavailable. Committed inbox status code selects its first management host: inspect current code before assuming fallback exists. A necessary fallback fix should be small, tested and separate from paper evidence.
2. Use the supported admission path and full compatible host pool; let admission claim the node. CPU work needs equivalent isolation/claims without reserving an unnecessary GPU. Never replay an uncertain prior job. Fresh inspection determines availability.
3. Register/complete new QP52f96 using the immutable qualified worker and the same reference, decoding/scoring/accounting contract. Test only QP51 if needed through existing smoke/pilot/scale gates. Retain every registered result.
4. Then extend AV1 into overlapping quality using supported low-rate settings and resolution ladders, reconstructing to original4K. Only a surviving gain warrants allocation/packing optimization and frozen unseen-scene evaluation. GVC-RT/MTTF remain a later alternative; GLC rate stays estimated separately.

## Locations and revisions

Durable selected receipts, proofs and `continuation.json`:
`/Users/manu/.codex/visualizations/2026/09/30/01a0f3d8-8225-77c0-b10a-4a9fd2401c9b/packet-rate-quality/gate-a-local-20261002/`.

Existing remote job:
`gpu1:/home/itec/emanuele/pointstream-data/audits/gate-a-native-vvc-q52-qpa0-f48-job-20261004`.
Check shared evidence from reachable nodes with identity verification.

Scientific code: `codex/gate-a-local-confirmation` at `fab719c`; nativev2 worker SHA begins `6c6aefdf`. Evidence documentation: `codex/gate-a-receiver-pilot-evidence` at `7e51162`. `/private/tmp/pointstream-packet-rate-quality` is on the documentation branch: recover exact scientific files from immutable stages or `git show fab719c`, not current working files.

Published Overleaf main: `e3d52d3`, 23-page empirical working paper. Local `/private/tmp/pointstream-paper-packet-rate-quality`, branch `codex/paper-gate-a-local`. Follow its own AGENTS.md; build `bash tools/build_pdf.sh`. Recheck Overleaf before publication; no force push or PR opening.

## Efficiency and preservation

Preserve dirty Cursor work in `/Users/manu/Desktop/PointStream`, raw results and old jobs; use a scoped checkout. One bounded helper at most, Luna6/max default. Reuse valid receipts and protocols; do not repeat completed audits, expand grids or add approval requirements. Batch manuscript updates after substantive results. The prior host-specific stop is not a fleet-wide blocker, and speculative cautions from this chat are not new policy. Follow current user instructions/repository rules and continue independent authorized work where feasible.
