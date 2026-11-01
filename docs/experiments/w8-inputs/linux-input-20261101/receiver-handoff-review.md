# W8 source-hidden receiver handoff review

This is a read-only design review, not a receiver run, proof of reproducibility,
or authorization for a GPU job. The input-equivalence gate and a remaining-budget
projection must pass before implementation is promoted or GPU work starts.

## Generator loader

The fixed packet carries the quantized match adapter and guider delta. In a fresh
isolated worker, unpack the paid match with the pinned generic guider, compare all
480 decoded tensor fingerprints against the independent CPU oracle, and serialize
that decoded rank-4 state to a new local scratch `.pt`. Preserve names, shapes,
dtypes and exact bytes, including signed zeros. Recheck the fingerprints in the
generator's JCV runtime; a serialization file hash is provenance, not a claim
that it equals the original FP16 checkpoint file hash.

The existing `k6b_fit.apply_fit` first merges the supplied tennis base, then
loads the match state. Its `quantize_adapters` call is conditional on nonzero
`requantize`. Use the unmodified `k6b_generate` path with direct `--fit match5`
pointing to the packet-decoded checkpoint and `--fit-base match5` to the pinned
tennis-base checkpoint, **without `--fit-bits`**; its default is zero. Passing
the original `--fit-bits match5 4` would quantize the already decoded adapter
and guider a second time. The worker must use only the paid packet-derived
state, explicitly installed pinned models/configuration and its own outputs; it
must not receive the original fit checkpoint or prepared/source/golden files.

The original W7 `k6b_confirm_validate.check_generated` requires the old FP16
fit-file SHA, render-time `bits == 4` and a billing record. It will reject this
semantically equivalent but newly serialized state. Do not change its historical
check or relabel the new file as the original. A W8-specific independent
validator must instead bind the paid packet and match hashes, 480 tensor
fingerprints, correct base and layer/key loading, model identities, billing,
device/kernel checks and exact output coverage. The original
`k6b_generate.command_validate` hardcodes `/tmp`; equivalent substantive
validation needs host-local scratch rather than that path.

## Conditioning and downstream interfaces

Construct a fresh minimal `publish/prepare.json` and `publish/units/*.npz`
archive only from the decoded packet's two units and conditioning arrays, or
transfer the already sealed packet-derived arrays through a narrow interface.
Never substitute the original prepared archive. Use `match5`, `--steps 30`,
guidance 2, `--from-t 0`, full runs and both units. Preserve the original
`SCHEDULER`, 512×784 inputs, 24-frame context and overlap 4. The native loop
generates all rows in each run, seeds each unit with the first seven SHA-256 hex
digits and each run with `unit_seed * 1009 + run_id`, and retains the 80 paid
judged ticks. The renderer's kernel probe and per-unit completion checkpoints
also need accounting. A shortened context is not a full-run equivalence test.

Transfer only new generated crops and payload-derived KPS/geometry into the
separately isolated MAIN-environment SAM stage. `k6b_receiver_sam.worker`, its
pose prompt and projection helpers, and independent-frame mode are useful
interfaces. The existing worker's broad filesystem allowlist and inherited
environment require W8-specific isolation and actual denied-read probes before
reuse. The original receiver spec points both `--weights` and `--text` to the
same pinned SAM asset; verify the actual asset and runtime identities. Preserve
the original independent-frame and fresh-reset behavior where required.

Decode the paid scale-4 IVF through the pinned native decoder and original
limited-range YUV420-to-RGB GPU conversion and bicubic resize path. Compose
new crops and predicted masks with decoded boxes and the payload's order using
`k6b_real_base_compose.compose`; missing player output falls back to decoded
base. Seal receiver output before an independent offline judge reads the
original source, masks or goldens. `k6b_carrier_judge` reads those original
assets and previous scores and is therefore not a source-hidden receiver.

Before a GPU smoke, independently hash actual installed generator, SAM, native
and quality-model files, code, configurations, packed environments and native
binary paths/versions against the frozen profile. Confirm the relevant
GPU-class/device/kernel qualification and fresh fleet admission, establish
negative filesystem/network/tool probes, and project the full run including
staging, all GPU companions, validation and publication within the original W8
ceilings. CPU input equivalence alone qualifies none of those installs or the
generated output. The saved historical timings are context, not a performance
guarantee.

The earlier local synthetic CPU arithmetic check ran with
`/Users/manu/miniconda3/envs/tigas/bin/python`, reported 2.2 seconds wall, and
did not capture CPU use or enforce one-core affinity. It was not a model run or
Linux isolation proof. The root's conservative ledger charge is 25 CPU-seconds
(ten logical cores times the wall rounded up to 2.3 seconds, plus two seconds
overhead); this is an accounting bound, not a measurement of its CPU use.
