# Ball receiver isolation API contract — November 12

Root design for the two independent builders. This is code preparation, not a
claim of isolation or a compute release. Actual runtime inventory values will
be collected read-only and authenticated before CPU oracle admission. Missing
values are a fail-closed runtime condition, not a reason to omit implementation.

## Shared interface

Module `experiments.tennis.k5d_isolated_receiver` exposes:

```python
def replay(wire_path, map_path, work, *, config_path, config_sha256,
           protected_paths, producer_context) -> dict: ...
def verify_receipt(receipt, *, expected_context, expected_config_sha256) -> dict: ...
```

`protected_paths` is a mapping from mandatory category names (`source_archive`,
`source_frame`, `ground_truth`, `encoder_state`, `private_detector`) to nonempty
lists of existing Paths. Aliases may be additional entries. The producer derives
actual paths from its staged inputs and actual encoder work; never accept absent
files as successful denial. It checks readable originals outside and records
only path identities/hashes/readability, never private content in diagnostics.

`producer_context` and `expected_context` contain actual `job_id`, `stage`,
`execution_identity` (source/snapshot/environment hashes) and `gpu` (UUID/name/
capability, or CPU marker for the CPU oracle). The verifier must check these
against the current actual dispatch or the declared immutable producing job,
not manufacture them from expected strings. It returns verified denial evidence
or raises. A legacy scope string, an unavailable verifier or boolean alone fails.

The encoder opt-in is `args.isolated_receiver=True`, together with
`args.receiver_isolation_config` and `args.receiver_isolation_config_sha256`.
These point to the staged authenticated config. Legacy default remains false.
The actual producer supplies protected targets internally. The scalar flags
and config are passed through new producer adapters. No numeric codec, fit,
renderer or scoring change. The actual worker gains an optional isolation-proof
argument, rechecks live denials after exec before decoding, and attaches the
verified evidence to its ordinary receipt. Preserve its exact ordinary fields
and numerical outputs. New adapters require this evidence and cannot accept
old packet-replay-only receipts.

## Authenticated configuration, schema 1

The config has `schema:1`, source/snapshot/environment identity, and per-host
profiles. A profile contains the packed runtime root and file manifest, approved
source-code relative file manifest, exact executable paths/hashes, exact system
library files/hashes, and typed system leaves for required NVIDIA/proc/sys paths.
Only exact code files from the reviewed decoder import closure are granted;
no blanket PointStream code root containing docs, fixtures or labels.

Runtime/code manifests map relative paths to SHA256 and size; resolve under the
actual trusted packed prefix/source root, reject traversal, surprising symlink
escapes and mismatch. System/device/proc leaves use the already reviewed W8
strict typed inventory machinery, not hashes of device contents. Directory
grants permit READ_DIR listing only; file reads/executions are exact. Never grant
home/datasets/cache parents or recursive source/label/private-state file access.
Validate protected targets/aliases cannot be reached by granted roots/files;
prevent cross-directory linking/refer/truncate/mutation escapes with ABI>=3
policy, and keep code/resources read-only. Fresh writable view contains only
approved packet/resource-map copies and generated output/proof/cache files.
No source images, labels, private detector features or encoder state are copied
into that view. Symlink/hardlink and `/proc/self/fd` controls are part of tests.

Start single-threaded before restriction, with clean whitelisted environment,
closed inherited descriptors, stdin `/dev/null`, and fresh exec. Use fixed argv,
not shell strings. Preserve CUDA UUID/thread controls and exact runtime/source
import paths. Native decoder subprocesses inherit Landlock. W8
`w8_receiver_native_bootstrap._landlock(files, dirs, execute, output)` ALREADY
handles EXECUTE bit; combine its exact file/exec grants with ABI>=3 strict
mutation handling from `w8_cuda_system_leaves`, rather than assuming the CUDA
probe's simpler restriction helper is sufficient. Do not preload private data
or leave unrestricted native threads before exec.

Actual receiver proof contains policy/config hashes, actual process PID/birth
before and after exec, ABI, descriptor/thread state, outside-positive and
inside-EACCES/EPERM results for each protected category and aliases, plus current
producer context. Actual decoder/grandchild execution and output hashes bind
the ordinary wire/native/YUV/RGB receipt to this proof. Standard OS semantics
establish inheritance; record actual child/command identities too. No mocked
or Python-only denial, ENOENT acceptance, legacy scope substitution or prior
W8 proof credited as Ball qualification.

## Implementation and validation

The denial builder owns this module, kernel bootstrap, optional encoder/worker
routing and meaningful focused tests. It supplies the pure metadata verifier
API above. It can implement and test using synthetic temporary inventories;
these are unit checks only. Root supplies actual host inventory and live paths
for the genuine SVT CPU oracle, then a fresh same-job/GPU DCVC witness.

The Ada adapter builder imports the verifier lazily and provides the real
producer/scorer entrypoints and fleet spec builders with required config and
accepted witness assets. It can use an interface stub only in unit fixtures;
operational paths fail closed until the integrated module and actual witness
exist. Authenticated marker/pixel provenance is mandatory and later native
DCVC fallback is disabled. Root merges both branches, reviews code/archives and
runs full checks once, then releases source-bound CPU and actual witness stages.

Do not stop at comparison helpers or a non-executable proposal: complete the
code paths and spec constructors while keeping all execution unreleased.

## Prospective private-proc amendment — before any oracle

Root CPU-only feasibility found nested user/PID/mount/IPC and private proc work
on gpu6 (receipt5611e6af). This is not a Landlock or decoder qualification; maps
was stat-checked only. Exact parent `/proc/self` grants do not cover native
child self paths. The candidate may therefore grant READ_FILE/READ_DIR only on
a VERIFIED freshly mounted PRIVATEproc containing only receiver descendants,
with distinct namespace identities, no encoder PID visibility, closed FDs and
actual outside worker/native birth observation. No broad host-global proc grant.
Stacked old/active proc mounts must be resolved by effective mount identity;
namespace PIDs must not be mistaken for host proc PIDs. Actual source/GT/encoder
file denials remain EACCES/EPERM, while process invisibility is separate proof.
A real pre-exec native-child observation barrier is required. CPU oracle must
exercise these conditions before any GPU witness; all limits remain unchanged.
