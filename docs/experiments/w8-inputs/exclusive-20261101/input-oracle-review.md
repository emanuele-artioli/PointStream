# W8 input-oracle review, 2026-11-01

This is a read-only implementation review, not an oracle run, proof of Linux
isolation, new experiment evidence, a resource charge, or authority to submit a
job. The strict rate gate is unresolved at this review; make no decoder or GPU
submission unless that gate passes and the root reviews the next specification.
The earlier strict Mac geometry mismatch remains a failure; this review does not
change its tolerance or establish the behavior of the original Linux runtime.

## Fixed packet and draft mismatch

The fixed candidate is `/private/tmp/w8-bundle-oracle-20261101-v2/window.psw8`,
SHA-256 `eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4`,
2,289,836 bytes. Its PSW8 v2 envelope contains these sections:

| Section | Bytes |
| --- | ---: |
| `ivf` | 21,963 |
| `kps` | 1,913 |
| `judged` | 30 |
| `ref0` | 277,749 |
| `ref1` | 88,412 |
| `match` | 1,897,395 |

The existing `w8_input_job.py` and `input-oracle-draft.json` instead extract KPS
and references from the older transport archive and construct a judged bitmap
from a command-line recipe. That exercise cannot establish that the fixed
candidate was decoded, that its paid bitmap was used, or that its IVF and match
sections and model identities were honored. The packet profile fixes 240 native
frames, the 80 saved ticks `0:240:3`, 24/4 context and the original generation
recipe. All conditioning frames required by the original runs must be retained;
the 80 ticks are the saved output, not a substitute for the 240-frame context.

## Proposed implementation after the rate gate

1. Stage and independently SHA-256-pin the complete candidate and a reviewed
   expected profile. The receiver worker reads the packet itself and calls
   `w8_payload.unpack_window` with that independently pinned profile, rather
   than trusting an expectation taken only from the packet header. Verify
   section closure, boundaries and hashes, window/unit/reference/tick/box
   bindings and the exact 30-byte paid `judged` bitmap. Pass the actual KPS,
   reference and bitmap sections to the conditioning decoder. Preserve the
   packet and conditioning decoder code identities declared in the profile;
   put any needed orchestration in the input job and explicitly record all
   executed code identities.
2. Pin and verify the original prepared archive and the original source
   (`b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d`),
   score masks, relevant golden outputs and generation checkpoint separately.
   The parent may establish their identities and stage them for the independent
   validator; the receiver may not read them or derive its inputs from them.
   Verify the match section's quantized symbols, per-tensor scales and metadata,
   exact tensor names/shapes and reconstructed bytes against the original
   checkpoint and pinned guider in that validator. Preserve strict native
   geometry equality and record any failure rather than introducing tolerance.
3. The packet's model hashes are declarations, not proof of installation. Audit
   and hash the actual model/configuration files, original guider and tennis
   base, code, native decoder and versions, plus the packed environment locks.
   The declared generator environment is `bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e`;
   SAM and native environments are
   `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
   Use the original compatible packed Linux runtime for the geometry oracle;
   preserve separate generator, SAM and native interfaces. These identities do
   not themselves establish that the environments are accessible or compatible.
4. Start a fresh worker with a sanitized environment, closed inherited file
   descriptors and no enabled MCP/tool or network access. Apply Linux Landlock
   with a narrow read allowlist for the packet, fresh outputs and necessary
   reviewed code/runtime/model paths. The current job applies Landlock after
   Python, NumPy and OpenCV imports and tests only four paths: that is not proof
   that startup, inherited descriptors, network or other tools cannot expose
   forbidden data. Audit resolved paths and mount/symlink aliases and reject
   overbroad ancestors. Require successful allowed packet/model reads and denied
   opens of actual source, target masks, prepared/golden and prior result paths,
   including accessible aliases. Missing probes, unavailable Landlock or an
   unverified runtime fail closed; no unisolated fallback.
5. Seal only newly decoded output and a receipt identifying packet/profile,
   code, environment and relevant models, exact time/run/reference/seed/order
   bindings, all 240 conditioning frames, the exact 80 scored indices, isolation
   checks and resource usage. A fresh offline validator independently rehashes
   sealed outputs and its own pinned golden inputs, recomputes exact q4
   keypoints/visibility, geometry and pose maps, and checks the checkpoint
   contract. Preserve immutable source and golden artifacts. Failure retains
   identified scratch and logs without silent deletion or unbudgeted salvage.
   The fleet's ordinary validator is a smoke gate; explicitly run and seal an
   independent full-output validation, and recheck its identities before using
   the result. Native IVF conversion and full receiver output also require the
   separate gates in the card; this CPU oracle does not substitute for them.
6. Before dispatch, update the W8 ledger with actual prior charges and prove the
   remaining original 3,600 workload CPU-second ceiling covers an enforceable
   complete campaign, including worker imports, staging, parent and child work,
   validators and publication, plus separately bounded admission and monitor
   termination overhead. The draft's 650-second one-core campaign and
   per-process `RLIMIT_CPU` values do not alone prove an aggregate bound; they
   also need the fleet's inherited one-core and process-lifetime assumptions
   verified for this path. Account for cumulative parent/child snapshots without
   double counting, reserve any separately run audit, prohibit automatic replay,
   and stop if the measured pilot and remaining-stage projection do not fit.
   Current model/environment availability and remaining-budget fit are not yet
   established by this review.

## Focused tests before any dispatch

- Reject truncated, trailing, duplicate, changed-hash or profile-mismatched
  packet sections; reject changed bitmap bits and an 80-bit mask with wrong
  indices, and swaps of units, references, ticks or boxes.
- Reject missing or extra tensor names, changed shapes, symbols or scales,
  wrong installed guider/model/code/environment identities, and missing or
  altered source and golden inputs. Preserve the existing strict geometry-fail
  test and compare full conditioning coverage and exact saved ticks.
- Reject missing, aliased or actually readable forbidden probes; verify
  allowlisted packet/model reads, closed descriptors and disabled tool/network
  access. Missing Landlock or an unsupported platform must fail closed.
- Reject changed sealed output, receipt, manifest or validator inputs, and a
  failed full validator. Exercise failed worker and interrupted publication
  paths without destroying retained scratch.
- Exercise aggregate budget exhaustion, process termination and projection
  failure. After the rate gate and root review, run a bounded Linux isolation
  smoke against the actual pinned runtime and inputs; mocks or this Mac review
  do not prove Linux isolation or original-environment equivalence.
