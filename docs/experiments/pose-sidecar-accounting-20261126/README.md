# Conventional video with a paid pose sidecar: accounting feasibility

**Decision registered before calculation (2026-11-26).** Test whether an already
measured physical ordinary-video stream plus the identical paid pose section can
fit the external Full-Obj warm byte ceiling. Hypothesis: direct pose access need
not require generating player pixels. Competing explanation: the pose and framing
bill leaves no affordable ordinary-video control. Count a feasible accounting
candidate only after matching source identity, native duration, scored bitstream
hash and byte receipts, and charging all specified envelope bytes. Budget: one
local CPU minute, no remote jobs, encoding, model inference or GPU computation.

This is an **accounting-only proposal**, not a newly constructed or independently
decoded control. It establishes a useful next control and prevents attributing
direct machine access specifically to player synthesis. It does not establish
equal visual quality, pose accuracy, machine recall or a measured parse time.

| Shot and saved video | Video | Existing pose section | PSW8 envelope | Proposed total | Warm Full-Obj ceiling |
|---|---:|---:|---:|---:|---:|
| Shanghai, 320x192 CRF 63.5 | 13,408 B | 2,301 B | 569 B | 16,278 B (16.262 kbps) | 17,794 B |
| Shanghai, 480x272 CRF 69.25 | 13,494 B | 2,301 B | 569 B | 16,364 B (16.348 kbps) | 17,794 B |
| Halle, saved 320-wide control | 16,782 B | 2,165 B | not needed for rejection | at least 18,947 B | 18,319 B |

Shanghai uses all 240 native frames at 30000/1001 fps (8.008 s). Both Shanghai
candidates have room below the ceiling; they are not optimized to exhaust it.
Halle's saved control exceeds the ceiling before any envelope bytes, so this
audit does not supply a matched Halle control. A lower-rate encode would be a
separate experiment.

The pose byte count is the independently coded `poses` section inside the
existing joint stream, **not** the much larger appearance-bearing `.pscp`
carrier. The proposal sends that unchanged section with ordinary IVF video in
the existing uncompressed PSW8 envelope. Its profile explicitly carries frame
count, exact cadence, coded and display dimensions, track order and coordinate
convention; the directory charges section lengths and SHA-256 checksums. No
appearance initialization, rendering model or ball payload is needed for this
specific video-plus-pose control. Every checksum has fixed 64-character width,
so uncompressed framing length is exact despite not possessing the raw bodies
locally. Size probes use zero-filled bodies only to check existing pack/unpack
accounting; those probes are never labeled physical experiment evidence.

The script verifies the saved source and video receipts, and exercises PSW8
pack/unpack accounting. The output identifies all input receipt hashes and the
original video producer jobs/archives. It records the parent joint hash for the
pose section; a physical replay still needs to extract and hash that section,
verify its decoded tracks against the existing direct-pose endpoint, combine it
with the archived IVF, and independently decode the resulting transport. Any
machine-score reuse must first establish exact decoded-track identity. Saved
video scores can be reused only after verifying unchanged decoded video pixels.

Reproduce from the repository root with Python's standard library:

```sh
python3 docs/experiments/pose-sidecar-accounting-20261126/account.py /tmp/pose-sidecar-accounting.json
```

The checked-in `result.json` is the calculation receipt. Original experimental
receipts and source data remain unchanged. No physical combined-packet hash or
new performance number is claimed.
