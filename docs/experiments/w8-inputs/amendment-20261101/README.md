# W8 prospective anchor amendment: preparation only

`selection.json` is the exact W1 repair-selection manifest, SHA256
`4f5af8c90f5fd909abac2aed123917795f388b3961e31e18cb2e0f306aa7acb0`.
It identifies the single retained `nad000` `s2_o20` CRF-67 trial and the
three original CRF-63 controls. The trial is 29,804 bytes, 49.67333333333334
kbps over 4.8 seconds, stream SHA256
`182c7eee4a0ea2e98ef0a967ecfcdf8958898ad7ce035e54f0ea91b5109b6fa3`,
recorded decoded SHA256
`71b3ee98cc83349d6377394e5003ae2da872c0e4a0e3a56685eed3a81a7dfe9f`,
and recorded ROI SHA256
`f00e397789e9aa23b096e5cec18dfdb04e08663a44bb4978638cb52c1593ac98`.
Its original encoder job is `20261031T181055Z-e4dc2d25`, archive SHA256
`9736920b256b2b770c3b35e8784570a50df2720115fbb134970439adea4240f0`.
The stream and archive were verified read-only; decode and ROI identities are
recorded metadata, not new verification. Control identities and all remaining
staged input/environment pins are in the adjacent draft and metadata audit.

The approved amendment reserves at most 300 GPU-s and 300 workload CPU-s
inside W8's original 3,600/3,600 ceilings. Previous W8 charges are at most
850 CPU-s and zero GPU-s. No score job has been submitted. A real normalized
fleet spec cannot yet be frozen: it needs the reviewed workload-wide guard and
its interface, an immutable remotely readable selection path, allocation for
staging, validation and independent audit, a future deadline, a frozen code
revision and successful queue/host preflight. The adjacent draft is not a
submittable spec. No source archive publication is authorized by this preparation.

Use the same W1 scorer and all 80 original ticks, including all three original
controls. Smoke must establish exact identities and substantive control results;
measured projection and enforceable aggregate caps must cover every stage before
full promotion. Independently run `experiments.tennis.w9_full_audit` against the
full output, the frozen normalized spec and selection, original windows and
hybrid archive. It recomputes all raw control errors, requires each of four
metrics for each control within 0.001, and reruns the full validator. Its
resources must fit the amendment allocation. Record the new job, archive,
normalized spec and independent audit hashes before interpreting results.

`experiments.tennis.w8_amendment_merge.screen` loads the accepted original and
independently audited W9 repair points, verifies the separately audited new
archive, and adds only the selected trial. It evaluates the strict eight-cell
four-family LPIPS/DISTS rule at the actual fixed W8 rate
`46.315062222222224` kbps. It uses raw points and no cached old-target decision
or extrapolation. A pass alone permits the remaining original W8 gates to be
considered, subject to their unchanged constraints; fail, unresolved coverage,
identity/control failure, or inability to fit the allocation stops promotion.

`score-spec-candidate.json` is a schema-normalized **non-submittable** candidate.
Its deliberately expired `2000-01-01T00:00:00Z` deadline is a sentinel, not a
proposed job deadline; after preflight replace it with the reviewed job UTC plus
at most 45 minutes and regenerate normalization and its digest. The proposed
selection path in the spec has not been checked or published. The candidate
uses the W8 wrapper around the unchanged W1 scorer and validator, one CPU core,
40/160/20-second smoke/full/validator allowances and a 230-second campaign
budget. The reviewed provisional CPU allocation is campaign 230, monitoring
and cleanup 30, independently limited full audit 20, and preparation/final
publication 20 seconds. GPU reserve is 300 seconds. These are reservations,
not spent resources. Root must review the guard implementation and resulting
candidate before any publication or submission.

The candidate sets `defer_failed_publication: true`: failed scratch remains
intact and automatic salvage is deferred. The previously running worker release
predates this flag and would perform an unbounded 900-second salvage; before any
job, root must audit and refresh each selected worker to the committed guard-aware
release. Complete compatible-pool selection must also consider protocol and host
fitness, not GPU class alone. A passing actual smoke and its measured remaining
budget projection are prerequisites for full GPU scoring.
