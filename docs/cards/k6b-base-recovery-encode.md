# Native whole-frame base recovery encode — COMPLETE

Goal: produce verifiable actual whole-source SVT streams and decoded pixels for
Al031@129 and Med014@194, as inputs to a later saved-80 feasibility test. This
is an authorized CPU job, not a result or a delivered codec. No model inference,
quality scores, ball reconstruction or new generation. Independently search
only the exact paths identified by the inventory first; if matching streams
and pixel manifests turn up, verify them instead of encoding duplicates.

Source clips and native intervals: Al SHA256
054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76,
frames 9–248 at 25/1 fps; Med SHA256
a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be,
frames 74–313 at 50/1 fps. Convert using recorded ffmpeg 8.1.2 SHA256
7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2,
select inclusive frame indices, area scale 1920×1080, TV-range yuv420p,
passthrough timing, 240 frames. Compare full converted YUV SHA256 against
the two values in the inventory; halt on a mismatch.

Use pinned packed environment SHA256
32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661,
SVT-AV1 4.2.0 executable SHA256
4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4
and dav1d 1.5.4 SHA256
e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6.
Native per-plane downscaling in `experiments.tennis.k6.downscaled_source`
and its pinned dependencies, not an invented resize, at scales 8, 4, 2
yield 240×136, 480×272, 960×544 respectively. Encode each at CRF63,
preset4, 8-bit, range0, rc0, keyint -1, lp6, 240 frames and native fps.
Decode each via dav1d. Preserve six complete IVF streams, decoded yuv420p
outputs, source-conversion hash, per-frame decoded hashes, exact dimensions,
frame indices, FPS/timebase and each IVF timestamp, tool/environment hashes,
commands, payload bytes, 32-byte IVF and 12-byte/frame header bytes plus any
external framing. Save only to unique new job output, intermediates in scratch.
Do not overwrite data. File integrity checks on a fresh independent read and
decode are required. The old stream hashes are a reproducibility comparison,
not a requirement to call a newly encoded stream the old stream.

Validator: verify input hashes, exact converted YUV hash and expected sizes;
IVF signature, 240 complete frames/timestamps, no truncation, resolution,
correct total bytes and headers; independently decoded 240 frame sizes and
per-frame hashes; verify selected 80 ticks 0,3,...,237 against manifest, no
silent frame reorder, each output nonempty and finite, plus a simple source
luma sanity check that does not invoke a quality judge. Re-decode at least one
complete point independently; validate all six file hashes and metadata. A
mismatch blocks promotion. This is a whole-source base containing player and
ball; no target matte/Telea is transmitted to receiver.

Authorized budget: <=1800 aggregate CPU-s and <=45 min wall, 0 GPU; one-thread/small-thread native smoke <=300 wall seconds with measured
child CPU, first 24 frames on both sources, scale2 and exact commands, then
project the six full 240-frame encodes, all decoding/validation and overhead
under both caps before promotion. A short stream's encode hash need not match
a full stream. Original receipts show scale8/4/2 encode wall 0.77/1.65/5.90 s
Al and 0.77/1.59/4.15 s Med under other parallel load; these do not measure
CPU-seconds, conversion, output, decoding or current headroom, so are not a
validated quote. Reserve and revise only from measured smoke. Archive each
completed setting. Stop on invalid smoke or projection/cap failure.

This does not supply ball metadata or full-frame quality and makes no claim
about the old missing streams. A later separate quality/DCVC quote needs real
inputs, native decoder availability, scorer smoke, controls and measured cost;
there is no measured GPU-hour quote yet. Ball needs exact-window encoder-side
position/visibility/shape/color/appearance and packet/timing provenance,
including unknown frames and cost. Any new detection/capture needs its own
scope and authority. Prior player pass used granted Telea and is development.

## Authorized continuation — preflight

Planning explicitly authorized this stage, <=1800 measured aggregate CPU-s,
<=45 minutes wall, 0 GPU. The proposed smoke, inputs, settings and validator
remain unchanged. Do not submit until implementation, packed-native path,
independent decode checks and total CPU accounting—including fleet setup and
publication—are demonstrably enforceable. Any failure stops without promoting.

## Durable overnight stop — 2026-10-26

No job, smoke, worker implementation, executable job specification, stream,
decoded pixels or quality result exists for this continuation. The scoped
branch `codex/k6b-native-base-encode` starts from inventory commit `e517eb6b`;
prior inventory PR #248 records actual archive/ball searches. The native
anchor job `20261024T201059Z-29586984` was checked complete; it is historical,
not a submitted job for this continuation. Old allocations are closed.

The user's unattended-work request does not waive AGENTS.md: unattended
execution requires verified isolation including enabled file and MCP tools.
Planning reports its local scoped probe found filesystem isolation verified,
but daemonized process lifetime unverified and no enabled-MCP isolation proof;
the probe overall failed. Remote agent sessions are prohibited for this
project. Stop unattended dispatch; no timers, recurring monitors, new agents,
additional chats or fleet work. This is an operational stop, not a failed
scientific test or exhausted budget. Resume attended, or only after the actual
execution environment has independently verified both process lifetime and
file/MCP isolation.

Fleet `budget_seconds` controls wall time. Its cumulative child usage recorded
at workload return can measure child CPU, but the saved snapshot does not
establish an enforceable aggregate ceiling covering staging, supervisor,
publication and the later smoke validator. The card's 1800 CPU-s remains the
assigned cap; do not relabel it a workload-only cap or promise full accounting
without a verified mechanism. A revised workload-only cap or explicitly
bounded overhead requires separate authority. Do not modify fleet infrastructure
just to claim strict global accounting. Before any eligible attended run,
implement the retained-stream worker and independent validator; establish a
measured and enforceable budget approach, then conduct the fixed 24-frame
smoke and promote only when its projection fits. Native comparison converts
YUV to RGB with `src.codecs.quality.yuv420_to_rgb` (bilinear chroma, limited
range) and bicubic resizes RGB to 1920×1080 in `experiments.tennis.k6`; preserve
this path and verify CPU/native pixel identities rather than substitute an
unverified generic resize. Six streams must include all framing and complete
240-tick pixels; no data produced yet.

## Attended accounting amendment — 2026-10-27

Planning amended this card prospectively to <=1800 measured/enforced CPU-s
for the workload process tree (conversion, encodes, decodes, internal checks and
packaging), <=45 min fleet wall and 0 GPU. Fleet staging, supervision and
external validator CPU are separately measured where available or identified
as unmeasured; the cap is not an aggregate fleet CPU guarantee. Earlier strict
proposal and operational stop remain historical. Work is attended. The original
two-window, 24-frame scale-2 smoke and substantive gate still govern promotion;
no generation, scoring or ball inference. All earlier budgets remain closed.

### Attended implementation gate

The new worker restricts its workload and inherited children to one CPU,
with 250 s smoke and 1300 s full workload wall alarms; under the verified
inherited affinity this bounds their combined CPU by <=1550 s plus startup
and termination margin within the 1800 s workload cap. It also records
self+completed-child CPU; fleet staging and external validator remain separate.
The smoke projects 30 times the larger measured smoke CPU/wall plus 25%, and
requires <=1300 s, before the native manager may promote. It validates both
24-frame windows at scale2 with independent decode and a truncated IVF fixture.
Pinned paths/hashes, full framing, decoded all240 YUV and 80 reconstructed RGB
ticks/point are required. Six full decodes are independently repeated inside
the worker. Source judged-frame RGB hashes are retained as controls. New streams
are never labelled recovered. If any assumption, affinity, validation or
projection fails, the stage stops; no GPU/quality work follows.

Submitted CPU-only job `20261027T072713Z-2888cefe`, source `44ee278b`,
normalized specification SHA256 `50d151e64a3fc48ed3d10533099475dcffc1ead7a921470a7fdfc516619fcfc8`,
2026-10-27 08:10 UTC deadline. Inspect this same job, never replay. Full
promotion is gated by substantive smoke and <=1300 s projection. Outcome and
usage must be reported before deciding next steps.
Smoke gate PASSED: 18/18 checks, 18.224570 workload CPU-s, conservative
683.421375 s full projection below the 1300 s full guard. Full running; inspect
job `20261027T072713Z-2888cefe` rather than resubmit.

**COMPLETE:** `20261027T072713Z-2888cefe`, workload184.541090/1800 CPU-s;
independent archive audit separately37.743262 CPU-s, supervisor/staging CPU
unmeasured. All six 240-frame IVF/YUV and six 80-frame RGB arrays passed
independent per-frame hashes. The six new IVF byte hashes match historical
receipts. Immutable read-only archive and exact identities are recorded in the
[experiment result](../experiments/k6b-players.md#2026-10-27--native-base-recovery-complete).
No GPU work was run; the [quality proposal](k6b-real-base-quality-pilot.md)
requires new authorization and measured smoke. CPU workload ledger closed.
