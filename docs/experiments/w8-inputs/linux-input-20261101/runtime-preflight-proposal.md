# Proposed read-only original preparation runtime preflight — root review required

No command below has been run. The purpose is to identify an existing cached
`32e68872…` original preparation environment and check its actual NumPy/OpenCV imports and
installed distribution metadata, without torch/model import, GPU use, download,
extraction, file creation or cache mutation. The repository's lock is evidence
of a declared build, not proof that an existing cache matches the packed tarball.
A cache directory named by a SHA is not itself a fresh verification of its tree.
Verify its original staging receipt and packed archive pin separately before
claiming a correspondence; fail closed if unavailable.

Root must first check current fleet jobs and the three protocol-compatible hosts gpu2, gpu5 and gpu6 under the
project protocol. Proposed probe source is `runtime-probe-proposal.py`; review it
and the exact transport and timeout implementation before running anything.
Use an attended local supervisor with a hard **120 s total wall** and **15 s
aggregate CPU** budget, inclusive of local supervisor, SSH children and all
remote processes. The local supervisor and each of five SSH children are limited to 1 CPU-s each.
Three inventory processes are limited to 1 CPU-s each, the qualification to
3 CPU-s, and the namespace probe to 1 CPU-s. The worst-case bound is 13 CPU-s;
root reserved 15 CPU-s including 2 s shell margin inside the original W8 cap.
Never automatically retry a timeout or unknown remote status. Charge all actual
preflight work to the original W8 CPU ceiling, not the proposed 300/40 oracle
reservation. Root's confirmed prior W8 upper bound is 1,154 CPU-s and 224 GPU-s,
leaving at least 2,446 CPU-s and 3,376 GPU-s before this preflight.

For each host gpu2, gpu5 and gpu6, pass the same reviewed script via stdin, with
noninteractive SSH, no remote agent and a **15 s** wall timeout. Conceptually
(the supervising program must enforce and record the limits above):

```text
ssh -o BatchMode=yes -o ConnectTimeout=5 gpuN \
  'env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /usr/bin/python3 -B - inventory' \
  < docs/experiments/w8-inputs/linux-input-20261101/runtime-probe-proposal.py
```

The probe sets affinity to one CPU and reports resolved filesystem device IDs,
free local bytes, and cache/interpreter existence only. A failure or timeout is
an unknown host, not evidence of an empty cache. Do not operate in a local root
on the root filesystem; apply the fleet's 50 GiB disk-free rule. After all three,
choose one reachable host with a separate local filesystem, enough free space,
and the exact named cached interpreter. With a **25 s** wall limit use:

```text
ssh -o BatchMode=yes -o ConnectTimeout=5 gpuN \
  'env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/local/users/emanuele/pointstream TMP=/local/users/emanuele/pointstream TEMP=/local/users/emanuele/pointstream /local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/python -B - qualify' \
  < docs/experiments/w8-inputs/linux-input-20261101/runtime-probe-proposal.py
```

The script checks the interpreter identity, pins its own affinity and CPU limit,
imports NumPy and OpenCV only, and returns version strings, actual module paths,
installed distribution versions and RECORD SHA256s. Preserve returned raw
records, host ID, errors, elapsed time and CPU accounting in a new receipt.
No package files are written; Python bytecode and user site are disabled. The
planned 6×15 + 25 seconds leave five seconds within the 120-second wall cap;
if time or CPU remaining is insufficient, stop rather than overrun. Conflicting
module/version/metadata or unexpected symlink paths fail qualification. An
uncached pack is a blocker, not permission to unpack or download.

On the chosen host, separately test the user and network namespace, within a
**10 s wall / 1 CPU-s** hard limit, inheriting the script's one-core affinity:

```text
ssh -o BatchMode=yes -o ConnectTimeout=5 gpuN \
  'env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /usr/bin/python3 -B - namespace' \
  < docs/experiments/w8-inputs/linux-input-20261101/runtime-probe-proposal.py
```

The same process records `/proc/self/ns/net`, then execs `/usr/bin/unshare
--user --map-root-user --net` and a tiny stdlib Python socket-creation probe.
It fails unless the child network namespace identity differs from the parent.
It does not connect or send traffic. Lack of `unshare`, denied user namespaces,
or inability to verify a distinct namespace is a blocker, not an isolation
pass; this feasibility test alone does not prove the final worker's isolation.

The proposed executable supervisor is `/private/tmp/w8-linux-runtime-preflight-20261101.py`, SHA256 `3e6a8887c0177b8a1363aadbe3f6e71e9431e22b3fa14af5e84356ae3c3fc7d3`; the proposed stdin probe SHA256 is `22c08c8cc937e0732d6293d443d89db2b667f3f24a1d4388a1653f85d76366f6`. Both await root review. Results go to the new exclusive `/private/tmp/w8-linux-runtime-preflight-20261101` directory.

The expected original preparation lock versions are NumPy 1.26.4 and OpenCV headless 4.10.0.84, based on preparation spec SHA256 `6aa451191e0f49fa7bd993b417e06a18e1ddf73dbb1ee831416d9d84cfbadb3c`. A reported mismatch stops qualification. The JCV generation runtime is not tested by this preflight.

Each remote command also has its own timeout at most 20 seconds (qualification local SSH timeout remains 25 seconds). An SSH spawn OSError is retained as a failure receipt; it is not retried.
