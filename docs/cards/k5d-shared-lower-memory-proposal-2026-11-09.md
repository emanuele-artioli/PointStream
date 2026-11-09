# Ball shared numerical parity: lower-memory candidate — November9

Status: prospective local engineering proposal only; no source upload, CPUproof
or GPU run is released here. ExistingDF6/v1 and its pending queue remain unchanged.

The fixed16GiB claim plus8GiB margin currently cannot be admitted beside foreign
work on gpu2. A separate explicit profile may test4GiB parent +2GiB serial DCVC
child PyTorch allocator limits,6GiB declared plus the unchanged8GiB admission
margin (14GiB required free). This changes only resource policy. It does not
prove that the smaller caps fit, and it is not a whole-device hard memory cap.

Static scope: the parent holds BallTrack, WASB, LPIPS-AlexNet and DISTS-VGG;
BallTrack/WASB retain the original batch16, and DISTS retains both full-frame
feature pyramids. Twelve direct DCVC encode/decode/replay calls are synchronous,
so parent plus one child matter. An old1,048.5MiB allocated encode peak is context
only; decoder, reserved, context, driver and custom allocations remain unknown.
The first complete16-frame exact-parity smoke is the feasibility gate, not a
claimed memory oracle or peak guarantee.

Preserve every original frame, source, encoder/judge, model, batch, dtype,
algorithm, codebook, QP, bitstream framing, rate, quality rule and numerical
comparison. All six native/wire/pixel, full metric and judge arrays must still
match the registered16reference exactly. Parent and every one of12DCVC calls
retain action/plan/hostPID/startticks/PGID/GPUUUID/budget receipts. No partial
promotion. Stop on OOM, nonzero exit, reactive monitor limit, missing receipts,
wrong device/provider/ownership, any numerical mismatch, deadline or budget.
Shared results remain unsuitable for timing, throughput, scale or cost evidence.

If the currentDF6 queue produces accepted parity, root decides whether this
alternative still adds value. If it expires without execution, a smaller
registered resource profile may be worthwhile before another unchanged queue.
Do not edit the running/pendingDF6 profile, lower its margin, or replay it.

Before execution: immutable new source/profile, exact pinned resource-only
H2source bridge with every other old blob unchanged, configured checks and
fresh actual-source native CPU/namespace proof. Then one separately released
GPU16pilot with a complete staging/lifecycle/audit forecast. A prospective
CPUproof<=300CPU/0GPU and GPU16pilot<=1,000GPU/1,200CPU may fit the existing
Ball workstream/first-tranche reservation; neither is extra budget or execution
authority. All failed/partial/gated outcomes remain beside the prior records.
No full30/292/586 release follows; those still need their separate complete
remaining forecasts and a failure reserve.
