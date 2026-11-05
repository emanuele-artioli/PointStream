# Prospective GVC endpoint sensitivity — 2026-11-05

**Status: preparation only; no endpoint job or remote CPU proof released.** Test
whether the saved QP3 comparison is representative across a wider native
rate range on the same 16 visible frames of TrackNet game10/Clip3. Hypothesis:
the crop/full-frame trade-off persists at other settings. Competing
explanations include a QP-specific trade-off, rate outside the six-anchor
range, input conversion, exposure and the adapter itself. Report both base
QP0 and QP9, including failures and outside-range points, alongside immutable
QP3; no favorable-point selection, BD-rate, absent-frame or corpus claim.
A mixed result is a mixed result, not evidence of general dominance. If there is
no usable rate overlap, stop at descriptive points; do not extrapolate.

Pinned [upstream common model](https://github.com/semcomm/GVC-RT/blob/d0e32bfa3e8e282f9a77437c223c605858eea637/src/models/common_model.py)
sets ten base entries; [P model](https://github.com/semcomm/GVC-RT/blob/d0e32bfa3e8e282f9a77437c223c605858eea637/src/models/video_model_gvcrt.py)
adds two and shifts `[0,2,1]`. Its [test entrypoint](https://github.com/semcomm/GVC-RT/blob/d0e32bfa3e8e282f9a77437c223c605858eea637/test_video_gvcrt_1088.py)
constructs evenly spread base indices. This source range does not prove the
actual checkpoints or monotonic rates. The existing QP3 supervisor is extended
only through explicit `--qp 0` or `--qp 9`; default 3 stays unchanged. Each
spec must pass its QP to both run and validator, independent audit must pass the
same QP, and a full stage must match its validated smoke QP and pixel prefix.
Sender argv, native manifest, all packet placements and actual fresh decode
must agree. The existing bridge binds stream packets to placements; the strict
receiver denies source reads. No old artifact is changed.

Before GPU admission, run the new CPU-only `gvcrt_qp_coverage` on a fresh,
fully hashed pinned author source and I/P checkpoints, with CUDA explicitly
hidden, actual locked environment and entropy extension, no datasets or
model generation. The helper reuses bridge strict whole-checkpoint key/shape
loading and asserts finite complete I 10/P 12 quantizer tables and all three
shifts for bases 0, 3, 9. Check helper/source/bridge/weights/runtime/extension
hashes, fresh separate output, process/claim cleanup and CPU/wall receipt.
One proposed prequalification allocation is at most 100 CPU-s, inclusive of
staging, validation and administration; root must approve an exact enforceable
host/spec/deadline before any remote command. Failure or incomplete receipt
stops, not a GPU release. The local mocked table tests are not checkpoint proof.

Each endpoint first runs the identical four-frame actual native sender, fresh
source-hidden receiver and scorer. It includes I plus every P shift and must
pass strict source/weights/ABI, isolation, native-byte accounting, packet and
reconstruction parity, judge, device and model guards. Its existing 1.5×
projection scales the 4-frame measured first I, reset and non-reset costs to
16 and conservatively repeats staging, prep, probe, scoring, output,
validation and release; it must fit the 500-second full-stage and remaining
campaign gates. Use distinct fixed-QP specs and independent full-stage audits,
source and byte/metric/frame joins. Forecast both endpoints against the whole
remaining budget before promoting either full stage. A failure, contention,
missing cache, provenance mismatch or budget failure stops; preserve all results.

For context only, actual QP3 request `20261105T121902Z-4020b8ee` had
72.902-second smoke, 46.382-second full command and 131.018 seconds from
recorded supervisor start to terminal; actual full stage includes 82.471
seconds of charged staging/accounted overhead in its own timing record, so
these overlapping fields must not be added. Its 4→16 complete forecast was
265.705 seconds including 1.5× margin, not a measured endpoint forecast.
Indicative two-endpoint prior-wall scaling is 2 × 1.5 × 131.018 = 393.054
seconds of campaign wall, plus up to 100 CPU-s prequalification and explicit
independent audit, admission, staging/transfer, monitoring and teardown
allowances; this is not a reservation, QP-independent bound or sufficient
forecast. A prospective aggregate ceiling of **1,800 GPU-s / 2,200 CPU-s**
includes all endpoint failures, prequalification and audits; root must build
an inclusive stage/external accounting plan and reconcile it with the shared
ledger before release. Endpoint smoke costs and first-hand cache availability
may raise this forecast; no fitting-by-assumption.

Score original RGB and fixed registered ball crops with the identical six
physical anchor records, WASB and BallTrack, full LPIPS/DISTS/PSNR and crop
metrics. The anchors span 123.660–862.215 native kb/s; interpolate only inside
measured support. Report complete GVC native and manifest bytes separately,
with native anchor rates and their actual wire framing separately. Any ball-mode
comparison needs independently verified same-frame native-plus-BALL source,
settings, judge and complete-byte joins. Scorer exposure, 16 visible frames,
released 1920×1088 padded canvas, eager backend and adapter interoperability
limits remain as in the [QP3 audit](gvcrt-4020b8ee-independent-report-20261105.md).

## CPU proof concrete draft

`gvcrt-qp-cpu-proof-draft-20261105.json` is intentionally expired and is not
an admission. Its CPU-only single-core campaign has a 40-second smoke, 5-second
validator, 3-second full no-op and 55-second total campaign ceiling. The smoke
rehashes the four declared author export/checkpoint/entropy inputs, unpacks
only the source in fleet scratch, then runs the CUDA-hidden qualification with
a 35-second child timeout. The validator checks device and CUDA status, strict
I/P coverage, finite table shapes, shifts, process/usage receipt and helper,
bridge, export marker, interpreter and weights/extension identities. The full
stage is a no-op; it performs no additional evidence work. The five-second
validator and staging must fit inside the 55-second campaign. Missing warm
cache or a timeout means stop, not retry. The child exits on timeout and fleet
owns campaign process-group teardown; independently verify PID/PGID and claim
release. The new receipt is published atomically without replacing any existing
file. This is a checkpoint qualification only; it says nothing about native
encode, output quality or GPU timing.

Allocate at most 45 CPU-s outside campaign (including the already reserved
20 for metadata) for snapshot/transport, admission, bounded independent receipt
review, monitoring and teardown; 55 + 45 = 100 CPU-s inclusive. Record actual
bounded costs and stop before the next operation if its remaining allowance is
insufficient; the 55-second one-CPU campaign bound does not bound external
processes. Fresh exact snapshot SHA, source/marker/weights/runtime/extension
hashes, host and affinity, cache identity, independently reviewed draft,
deadline, source allowlist, teardown and complete accounting are required
before root releases even this proof. The gpu5 cache metadata check found all
five corresponding hash-key directories and separate `/local` storage; it did
not hash their contents. gpu6 had no CPU headroom at that check. Capacity and
checks are transient and must be repeated at admission. No CPU proof, GPU
endpoint or shared-budget release is implied by this draft.
