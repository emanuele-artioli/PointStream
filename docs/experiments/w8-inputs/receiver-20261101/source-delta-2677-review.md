# Source archive transport by bounded argument — review draft

The initial 512-byte block delta exceeded the original 96 KiB payload cap at 109,088 bytes, and was not sent. Root subsequently specified limits of 120,000 bytes for the *entire safely quoted remote command* and 200,000 bytes for local argv plus environment. The prepared exact command is 113,504 bytes; the measured local argv/environment is 117,617 bytes and will be checked again before launch. No chunked fallback is prepared.

The source pair remains the exact 5651 archive SHA `d3fbd915d12184e99d0e7e847a268190bd81e2067aeed1f25fbc79f0d8aec7b4` and 2677 archive SHA `fc3846bb0c788bbe55fa1b67d71f5ef6468b64b21559ad19f3b71036ed88f635`. A separate local verifier reconstructed the full 10,741,760-byte target byte-for-byte. The delta, independent receipt, exact command, CPU plan and identities are in `/private/tmp/w8-source-delta-5651-to-2677-20261101/`. The command passes compressed base64 in one argument and uses `stdin=DEVNULL`; the reviewed remote program reads and hashes the existing old local archive, bounds every delta reference and literal, creates only a new exclusive host-local directory/file, checks reconstructed and re-read SHA/size, and sets the new file to mode 0444. A failure leaves its new partial output for inspection; no old source is changed or automatically retried.

The locally prepared artifacts are:

- `/private/tmp/w8-source-delta-build-2677-v2.py`: deterministic 512-byte block builder.
- `/private/tmp/w8-source-delta-verify-2677.py`: independent local reconstruction verifier.
- `/private/tmp/w8-source-delta-remote-2677.py`: remote reconstruction/readback program, SHA `30b1b3a58250d2ee75dfca2315d96a7379f724d9f9900dffc864caec5e2eae21`.
- `/private/tmp/w8-source-delta-launcher-2677.py`: guarded SSH launcher, SHA `b1e359c3d63337c8749b158fb749618f9faf7bcce21219fbc7e93d7fa8c2444b`.
- `/private/tmp/w8-cpu-preflight-existing-source-2677.py`: separate conditional CPU preflight launcher, SHA `7adfc778ec9a3b481822856931f832130e215297c41124a05c74fde9c6e3ace3`.

The publication proposal reserves at most 12 CPU-s including preparation, one bounded launch and independent readback/audit, and zero GPU-s; its own readback is not an independent audit. Wall caps are 120 seconds remotely and 130 locally. Only after a successful independently audited publication may root separately review and reserve the unchanged 165 CPU-s preflight from the already-published 0444 source archive; that launcher sends no archive through stdin and extracts a fresh separate source tree. Closed W8 is currently ≤1,654 CPU-s / 224 GPU-s; reserving both prospective operations would bring its CPU upper bound to 1,831 of 3,600. Neither operation is authorized or executed by this draft, and source publication would not qualify models, runtimes, output quality or GPU work.
