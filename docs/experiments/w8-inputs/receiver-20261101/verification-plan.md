# Receiver installation and runtime verification plan — not executed

`asset-manifest.json` is a catalog of recorded identities, not evidence that the
files are installed or accessible now. It separates paid packet bytes; installed
generic generator/SAM weights, common tennis base and two environments; and
validator-only original inputs and goldens. The original fit archive must never
be handed to the receiver. The SAM record identifies an installed git revision
and RECORD verification, not a proven source archive; no SAM archive path or
SHA was found in the inspected records. The installed AA tree's equivalence to
its recorded source archive is also unverified. The profile's code hashes bind
individual files, not the whole executing source tree; freeze and hash the W8
orchestrator, all executed helper code and source snapshot separately.

After the CPU input gate actually passes, first reconcile the live W8 ledger and
root must authorize exact internal reserves inside the unchanged 3,600/3,600
CPU/GPU ceilings. Suggested *upper bounds for a bounded feasibility pilot*, not
an allocation or proof of fit: at most 60 CPU-s and 120 wall-s for single-core
read-only file-hash/metadata inspection; at most 500 GPU-s for a substantive
whole-context device/kernel and isolation smoke; reserve all associated CPU,
staging, offline audit, monitor and cleanup separately. Abort before dispatch if
those and the remaining full path do not fit. Failed work consumes the same
caps. Do not introduce encoding, training, downloads or changed models.

After fleet queue/all-host preflight, select only an eligible GPU and verify the
local root is a separate filesystem with fleet-required free space. Use existing
packed environments, source and immutable staged paths; no `/tmp`, `/var/tmp`,
or NFS small-file scratch. A proposed read-only native/file identity check,
after expanding reviewed exact manifest paths to separate argv (no shell
interpolation of untrusted names), is:

```bash
# Run under an attended, root-reviewed supervisor with one-core affinity and
# hard aggregate 60 CPU-s / 120 wall-s, including its children and output.
# These examples only bound their individual command, not the whole supervisor.
timeout 120s bash -c 'ulimit -t 60; exec sha256sum -- "$@"' _ \
  '<STAGED_MODEL_OR_BINARY_1>' '<STAGED_MODEL_OR_BINARY_2>'
timeout 10s '<PACKED_MAIN_ENV>/bin/dav1d' --version
timeout 10s '<PACKED_MAIN_ENV>/bin/ffmpeg' -version
```

Hash every actual model/config and native binary listed, compare exact SHA256s,
record resolved path, file size, version, packed environment SHA, host, source
snapshot and error; a symlink escape or mismatch fails closed. Verify AA's
actual installed source revision/files against the archive and generator locks,
SAM's installed `direct_url.json` commit and every installed RECORD entry, and
model loader key coverage. Measure/cap any full-tree or large-file hash; a
cached SHA-named directory alone is insufficient. There is no reviewed executable
aggregate supervisor or complete source-tree verification tool yet; prepare and
review those before performing the checks, and stop on budget exhaustion.
Do not assume the recorded archive paths or native historical versions establish
current installed identity.

The historical original run used gpu6 RTX 6000 Ada. The model–GPU table records
Moore-AnimateAnyone match5 on Ada and A6000, torch 2.7.0+cu128, fp16, SDPA flash
and memory-efficient attention and CUTLASS/cuDNN convolution kernels; SAM has
separate Ada/A6000 records with CUDA, bf16 and profiled flash. These historical
records establish eligible classes, not the current job's device or kernel.
W8's frozen substantive smoke must record selected device, compute capability,
actual model/module paths and hashes, CUDA/provider, attention and convolution
kernel observations, SAM prompt/reset behavior, full sampler context, paid
packet/profile, 480 reconstructed tensor fingerprints, exact base and denied
source/mask/golden paths including aliases and tools. No unverified unisolated
fallback. Measure total GPU wall, CPU companions, process setup, staging and
publication; promote only if independent smoke validation passes and the
remaining full-stage projection plus full validation fits the remaining cap.

The isolated receiver decodes paid tensor state to new scratch, validates its
480 fingerprints in both runtimes, and uses unchanged `k6b_generate` direct
`--fit match5 <decoded-state> --fit-base match5 <pinned-base>` with **no
`--fit-bits`**, preserving rank 4, 30 steps, guidance 2, 24/4 context, seed,
full runs and original fallback/compositing order. The original W7 validator
expects an original FP16 fit-file hash and would wrongly reject this new
serialization; a reviewed W8-specific independent validator is required. It
must bind identities, decoding, actual model keys and full coverage, then seal
outputs before any offline source/target/golden access. CPU input equivalence or
a smoke alone cannot certify receiver quality. The original 80-tick per-frame
exact/numerical gates and 0.001 metric bound remain unchanged.
