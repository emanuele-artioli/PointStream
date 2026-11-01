# W8 Linux exact-input oracle: asset and execution proposal

The independently audited exclusive scoring request `20261101T081828Z-d70bc44f`
passed all eight strict rate cells at the unchanged 46.315062222222224 kbps;
see `../exclusive-20261101/rate-screen.json` and `full-audit.json`. This opens
the original input gate; it does not establish decoder equivalence, generation,
or the wider codec claim. The prior Mac geometry mismatch is not waived.

`expected-profile.json` is copied from the previously frozen pre-scoring bundle
receipt, not generated from the candidate on a future worker; its SHA256 is
`bec98d67280e9f777a0d3cc110baa51ae2bf2c876aefe6b247d9691432f61107`.
`asset-manifest-proposed.json` separates actual packet/profile and runtime from
validator-only preparation, fit/checkpoint, original guider, source, masks and
golden data. It also records model identities declared by the original
nad000 generation job. Staging an archive for the parent/validator does not
permit receiver access; aliases, inherited descriptors and network/MCP paths
must also be blocked and probed. The checkpoint member identity, `525830aa…`,
and original guider, `1a8b7c1b…`, are from the prior local checkpoint receipt;
the prepared, fit, base, generation and guider files were separately recovered
and hashed in `/private/tmp/w8-input-recovery-20261101/verified.json`. The
complete PSW8 packet was independently rehashed locally as `eff1aca8…`.
The local packet/profile have not been published to an immutable remote path.
Use a new, exclusive-create path only after root review, never overwrite.

The original preparation spec, SHA256
`6aa451191e0f49fa7bd993b417e06a18e1ddf73dbb1ee831416d9d84cfbadb3c`,
uses the main packed environment `32e68872…` for the geometry being checked.
Its repository lock records NumPy **1.26.4** and OpenCV headless **4.10.0.84**.
The separate original generation JCV environment `bf738ce5…` has a lock
recording NumPy **2.5.4** and OpenCV headless **4.11.0.86**. These are lock
records, not proof of installed versions or tarball correspondence; verify the
main runtime before the input oracle. JCV remains a separate later generator
requirement. The profile also declares model, native decoder/ffmpeg and code
hashes; declarations require actual host verification.

`spec-proposal.json` is the earlier design record, **not a fleet specification**. The later normalized but expired `score-spec-candidate.json` and its execution plan are separate. It
contains null CLI/staging/phase fields and an expired deadline on purpose; the
builder's final CLI, hard workload-wide enforcement and host isolation remain
unresolved. Do not submit it. The later candidate reserves 300 s campaign plus 40 CPU-s for preparation, monitoring and independent audit; the main-runtime preflight added a conservative 15 CPU-s, bringing prior W8 charges to at most 1,169 CPU-s and 224 GPU-s. See `execution-plan.md` for exact allocations and prerequisites. There
is no transfer from GPU budget. Fresh queue and all eligible CPU-host preflight,
committed source identity, immutable packet/profile staging, real future
deadline and canonical normalized fleet spec are prerequisites.

The complete fixed PSW8 bundle and independently pinned profile must be decoded
by a fresh isolated worker, exercising all 240 conditioning frames and the
actual 30-byte paid judged bitmap with exactly ticks `0:240:3`. Verify section
closure, unit/reference bindings, exact q4 keypoints, visibility, time/run IDs,
reference RGB, strict geometry and pose maps. Separately compare every match
tensor's symbols, scales, names, shapes and reconstructed bytes against the
original checkpoint and original guider; reject missing or extra tensors. A
fresh offline validator rehashes sealed outputs and its own immutable golden
inputs. Verify allowed reads and denied actual source, target masks, prepared,
golden and prior-output paths and aliases; require closed inherited descriptors,
no network/tools and enforceable Landlock before reading protected inputs.
Failure or uncertainty stops; no tolerance change or unisolated fallback.
Retain failed scratch and charge all setup, smoke, full, validation, monitoring
and publication. A successful CPU input oracle is not a fresh receiver result.
