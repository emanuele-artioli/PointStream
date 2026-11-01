# CPU input oracle: reviewed candidate pending freeze

`score-spec-candidate.json` is a genuine normalized fleet CPU spec, deliberately
expired. It fixes `fixed-run` and the `fixed-validate` fleet smoke gate, nine
SHA-identified staged inputs, main preparation environment `32e68872…`, one
CPU, 80/100/60 s smoke/full/validator, and a 300 s total campaign. The smoke and
full each require both units and the full 240-frame conditioning context plus
all 80 paid ticks. Smoke is not receiver quality evidence. Root must review the
actual supervisor guard and a measured cost projection before full; if the
complete full work cannot fit the remaining campaign budget, stop.

The temporary selected CPU pool is gpu5 and gpu6 because both passed this
preflight's cached-tree and separate local-filesystem inventory. gpu5 alone was
then checked for the original main environment's NumPy 1.26.4/OpenCV 4.10.0,
and an isolated user/network namespace distinct from its parent. gpu6 has not
had that runtime or namespace probe. gpu2 inventory timed out, so its state is
unknown and it is excluded until freshly qualified. The separate namespace feasibility test did not inspect the interface list.
The worker checks its actual interfaces with `socket.if_nameindex()` and must
prove the `lo`-only condition, Landlock, closed descriptors, tool/network
denial and source-path isolation in smoke. The preflight does not prove those
properties or cache-to-tar identity. Actual fixed worker checks must pass without waivers.
The portable `runtime-preflight-summary.json` preserves the raw summary and
identities; additional original stdout/stderr and receipts remain in the local
preflight directory. Full 15 CPU-s was conservatively charged, not inferred
from elapsed time.

After that charge the conservative original W8 ledger is at most 1,169 CPU-s
and 224 GPU-s of the unchanged 3,600/3,600. This proposal reserves at most
300 campaign + 10 new preparation/publication + 20 monitoring/cleanup + 10
independent post-campaign audit = **340 CPU-s**, with no new GPU. Thus maximum
prior plus this reserve is 1,509 CPU-s / 224 GPU-s. Counts include failed work,
child processes and publication; reservations are not expenditure. All guards,
separate audits and operations must be enforceable inside these amounts. No
budget is transferred from GPU. Abort on uncertain status; no automatic replay.

The packet and profile paths in the candidate are proposed exclusive-create
locations only: neither publication nor immutability has been established.
Root must first review and pin packet/profile bytes, implementation, source and
worker releases; establish original and staged forbidden aliases actually exist;
qualify the selected runtime, disk, isolation and complete current host/queue
preflight; and publish the two inputs without overwriting. Then set a reviewed
future deadline, renormalize and record exact canonical spec/source hashes.
No submission is authorized by an expired template or preflight alone.

The worker may read the packet, profile and pinned original generic guider; the
prepared, fits, base, source, masks and golden-generated archives are
validator-only and must be denied through their actual original and staged
paths, including aliases. The original guider is not the match-specific adapter.
Independent validation must compare full tensor symbols, names/shapes/scales and
reconstructed values against the fit archive, as well as original geometry,
reference binding, exact paid bitmap and all required frames. Strict failure is
a stop, not a tolerance waiver. Seal and independently verify full outputs
within the separate 10 CPU-s audit allocation, with identity and no forbidden
reads checked before interpreting a result. A CPU oracle pass permits further
original gates to be evaluated, not an inference-quality claim. Before any later
GPU job, separately verify the unused installed generator, SAM and native model,
code, binaries and environments against the declared hashes; the CPU oracle
does not establish those installations.
