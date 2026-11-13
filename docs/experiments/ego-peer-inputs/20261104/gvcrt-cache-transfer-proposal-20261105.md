# Prospective shared snapshot cache qualification

**Not released or run.** The prior gpu5 `_send_snapshot` probe timed out after
15.011 seconds with 977,721 bytes of the 6,133,760-byte source tar present;
checksum was not reached. Its partial remains at the path in the
[negative receipt](gvcrt-route-gpu5-probe-20261105.json). The partial length
is uncompressed destination data, not a network-byte count. On the exact
archive, gzip-6 encodes that prefix to 262,153 bytes and the full tar to
1,571,902 bytes. Treating that prefix ratio as if it represented throughput,
15.011 × 1,571,902 / 262,153 = 90.01 seconds and a 1.5× allowance is 135.02
seconds, below the proposed 180-second upload limit. This rough model includes
fixed setup in the measured interval but does not prove actual network bytes,
constant throughput or checksum time. In particular, the source partial might
not reflect all bytes in flight. A 180-second limit can fail; do not promote
or retry on the basis of this estimate.

One new probe, only after root reviews this change and authorizes the concrete
frozen client and destination: verify fleet doctor/route-worker/shared path and
space, then use the unchanged `_send_snapshot` over gpu5 to a **new unique**
`jobs/fleet/checks/gvcrt-cache-<nonce>/source.tar`. The input is the exact
reviewed `cde9046414f4412a92652b11c0c75ac7a4965e7843badb65a1f449b3368ae7ff`
tar; check local bytes before the attempt. One 180-second whole-upload timeout
includes local compression, SSH and remote SHA256. Use `extract=False`; the
existing helper's SHA verification must pass. Save local monotonic wall and
self/children resource-usage deltas, source and resulting SHA, PID/group and
post-exit route-process inspection. Retain the archive on success for possible
read-only reuse, and retain any partial on failure. No model, job, new remote
protocol, another attempt or alternate host follows. The 180 CPU-second shared
phase is charged in full, including preflight, upload and audit; aggregate
remote CPU is unmeasured, so this is an accounting envelope, not a measured
aggregate. A timeout, changed source, failed hash, unhealthy worker, absent
space or indeterminate cleanup stops.

The opt-in `--snapshot-cache-archive` requires a separately verified upload
route host. The route checks fresh fleet doctor/shared-base visibility and
heartbeat. The client still constructs the exact locally selected archive and
hash; a fixed remote Python copier checks the existing shared-fleet cache is a
regular non-symlink file under that base and hashes it, opens the fresh job
`source.tar` exclusively, copies bytes rather than linking, rechecks the source
identity while copying and hashes the resulting destination. A mismatched or
truncated source, foreign path, existing destination or failed checksum stops
publication; any new partial remains. The cache is never changed or deleted.
This path requires source-archive mode; neither execution pool, scientific
specification nor the default streamed route changes. The opt-in CLI transfer
bound can be at most 450 seconds; the existing default remains 300 seconds.

Only a successful independent review of the actual cached file, new copy and
complete stage costs could justify separate endpoint admissions. The 450-second
limit is an operational ceiling, not a forecast that any larger snapshot fits.
