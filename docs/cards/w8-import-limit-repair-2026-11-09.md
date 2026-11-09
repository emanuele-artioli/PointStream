# W8 import-limit compatibility repair — November9

Status: prospective bounded repair; no replacement run is released here.

Repair only the CPU-limit compatibility defect that stopped the exactff6 CPU
proof `20261109T184450Z-edd3f290` before scratch or any proof leaf. Root locally
reproduced the imported legacy `w8_input_job` changing an unlimited limit to
`(120,650)` when the unrelated witness entrypoint's first argument was `run`;
the later request for hard3600 then fails. The live fleet worker and independent
system Python were unlimited. This is an import side effect, not evidence of
a host policy restriction or a failed decoder/scientific gate.

The repair keeps the legacy standalone CLI limits and removes their import
side effect. New witness parents and reachable launchers preserve an inherited
hard limit or fail before a child whose required cap cannot fit; they do not
raise system limits. Record startup and applied limits. A real subprocess test
with inheritedhard650 must retain enough headroom for a child160 while bounding
parentsoft100; insufficient limits must fail closed. Audit the full reachable
packet, namespace, strict MAIN/JCV, native and assessment call chain.

The historical preflight source closure includes the input helper. The witness
may therefore accept exactly the resource-only repaired input helper and the
already registered witness bootstrap only after their actual archive members
and SHA256s are joined to the root-pinned source and spec/gate. Missing or wrong
hashes, unreviewed member changes or legacy-profile overrides fail. No general
source exception is granted. Require fresh configured checks, a complete exact
archive review, new raw CPU proof and independent root acceptance. Preserve the
failed ff6 source and all raw records.

The prospective cumulative CPU-proof ceiling is **900CPU seconds /zeroGPU**,
wholly inside the existing4,550/4,550 staged witness programme reservation.
The failed proof has **120CPU seconds** charged, including20preparation and100
conservative failure/closeout. At most **780 futureCPU seconds** may be proposed
for ONE replacement:30preparation +600one-core campaign +100post-campaign to
canonical terminal +50independent audit. Parent currently has4,430CPU /4,550GPU
remaining; this is not an additional reservation or execution release. No
aggregate or workstream ceiling increases. Stop if a complete source/spec/cost
review cannot fit, another implementation defect remains, or isolation, actual
source/host identities, substantive checks, teardown or bounds fail. No automatic
retry; this is the first implementation repair of the new source proof.

Original witness goal and quality/access stops remain unchanged: both native
240-frame runs, registered seeds,30 DDIM steps/CFG2/context24-overlap4, all80 saved
ticks0:240:3, RGB/composite channels within1eight-bit level, bit-exact masks,
per-tick and mean player/full LPIPS and DISTS within0.001. All five protected
categories remain metadata-only and unavailable before full producer sealing.
No training, encoding, source, judge, reference, numerical, device/provider,
quality or permission-rule relaxation. CPU proof cannot release qualification,
full generation or scoring; each future stage still needs its own gate and
complete remaining-cost forecast.
