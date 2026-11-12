# Ada 16/48 adapter: code-preparation proposal

This is not a fleet specification, witness, cost forecast, or release. The
reference is smoke A from `20261112T210150Z-2c8b49a0`, archive SHA256
`4bfff442081d988039abb308a3e0b8ed22632f5f598858d6eeaf4d0ef6824a1a`.
The reference establishes packet replay and calibrated measurements only; its
receiver did not establish filesystem denial.

The separate adapter provides strict 16-frame source/score/native comparisons,
producer and sealed-pixel scorer entrypoints, fixed namespace modes and a spec
builder. It authenticates retained checkpoint pixels, producer job, stage and
Ada GPU. The scorer does not decode archived DCVC. Every point receipt must
pass the independently built `verify_receipt` API with the producing context
and config hash. A SHA-bound root-approved actual denial witness and the
source-bound native CPU proof are mandatory; none is supplied by this code.

A prospective producer job has two stages: independently prepare all six
points for 16 frames and compare them to the Ada reference, then prepare all
six points on the entire 48-frame clip with isolated receiving. A prospective
scorer takes the sealed output of each corresponding producer stage, checks
producer linkage and actual receiver denial, and scores all 16 or 48 frames
with whole-clip temporal context; the 16-frame output must exactly match the
Ada reference. The 48-frame stage is not a cross-class parity claim.

Separate producer and scorer candidate ceilings are each 900 one-core campaign
seconds, 400 external CPU seconds and 200 additional GPU lifecycle seconds. A
possible fleet shape is 300 seconds for smoke, 600 seconds for full, 30 seconds
for validation, and namespace limits strictly below their stage limits. This
shape is only a ceiling proposal: setup, publication, validation and available
campaign time must be included, and there must be a prospective, complete
forecast and fail-closed promotion gate before an actual specification is
created. The 2c8b combined A/B measurements (126.574/112.491 wall seconds and
125.244/111.751 namespace CPU seconds) are not separate 48-frame production
or scoring measurements, and do not establish this forecast.

Before either job, root must review the combined source, source bridge and
exact-source native CPU accounting proof; authenticate the policy and real
SVT CPU denial oracle; and accept the separate same-job Ada denial witness.
The complete source/model/environment/CPU proof, producer publication and
witness assets must be bound by their real hashes. No later decode fallback,
188/292/586/full30 release, manuscript claim, or scientific acceptance follows
from these helpers.

The provisional `k5d_ada_witness.spec` constructor also defines a CPU-only SVT
oracle and a separate first GPU witness. Both need the exact-source `bridge`,
`native_cpu_receipt`, and `receiver_isolation_config` assets; each asset entry
has `local`, `remote`, and `sha256` fields. The CPU oracle additionally needs
`wire`, `resources`, and `protected` assets and expected decoded YUV and RGB
SHA256 arguments. The protected JSON maps each of `source_archive`,
`source_frame`, `ground_truth`, `encoder_state`, and `private_detector` to a
nonempty array of existing readable absolute file paths, with any required
aliases included. The worker derives CPU context from its actual dispatch.

The first GPU witness additionally needs `cpu_denial`, `fidelity_gate`, and
`ada_reference` assets. Its CPU-denial approval JSON includes `schema: 1`,
`approved: true`, `actual_filesystem_denial_passed: true`, `host: gpu6`,
`codec: svt-av1-ivf`, the CPU oracle `publication_sha256`,
`consumer_execution_identity`, `config_sha256`, `producer_context`, `receipt`,
and `actual_policy_sha256`. Its actual source-frame and encoder-state denial
targets are derived at replay time. The witness does not consume a previous GPU
witness; regular 16/48 jobs still do. Score specs additionally require root
producer16/48 attestation assets, each binding the producing job, stage, archive
SHA, device, identity, approved config/profile/inventory and denial result.

The CPU oracle candidate reserves 220 campaign seconds and 30 additional CPU
seconds within its 250-second envelope. The first GPU witness reserves a
500-second campaign, 400 external CPU seconds, and 200 GPU lifecycle seconds.
No actual runtime, source inventory, denial proof or release is supplied here.
