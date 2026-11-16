# Experiments: player profile receipt joins

### 2026-11-16 — partial historical receipt audit

This is a **partial, receipt-to-receipt audit**, not a requalification of saved
player generation. No model was run and no scientific data was altered. The
[derived rows](player-profile-join-requalification-20261116.json) name the 12
scaled generation jobs and the fit jobs and exact checkpoint hashes they
report. They are declarations from retained stage JSON, not independent hashes
of archive or checkpoint bytes. The selected original generic and W7 jobs have
not been audited here. The prior 72 comparison support labels were not revisited.

Read-only retrieval on gpu1 used
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/<job>/full/`:
`k6b_generate.json`, `k6b_fit.json`, or `k6b.json`. Preparation job
`20261022T123413Z-eba7629b` records 16 units across eight windows. Both
step-4 judges `20261023T132947Z-fb8a8581` and
`20261023T132954Z-ec202dfb` request `match`; step-5 judges
`20261023T231003Z-4fe6e239` and `20261023T232338Z-94c7a048` request
`match5`. All 12 selected generators and four judges report preparation archive
SHA256 `f832b70720acb3704fa83a6b7484fd18f97ef53d004e19452a528f075d737b8c`.
The judges list generated archive hashes, but those hashes were not joined to
independently hashed archive bytes in this audit.

For each of the six `match` and six `match5` generation jobs, the recorded
checkpoint hash matches exactly one selected fit receipt, that receipt names the
same match as the generation units, the model merge records the same checkpoint
hash, and the recorded weight maps agree. The `match` outputs cite the fit's
8-bit checkpoints. The `match5` outputs cite the fit receipts' **unquantized**
checkpoints, with generation-time 4-bit billing; comparing these hashes to the
separate 8-bit checkpoint would falsely flag a mismatch. The six `match5`
generation merge records and the corresponding fit base records all name
`85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4`,
which also appears as the checkpoint hash in tennis-base fit job
`20261023T194934Z-b0dcbb19`. This checks receipt agreement, not numerical
merge equivalence, reference-image identity, adapter bytes, or source provenance.

The base fit receipt lists 36 training windows; step-5 match fits list their
selected windows. Different window IDs do not prove disjoint source footage or
intervals. Older step-4 fit receipts inspected here do not list selected windows;
their fit-data archive hash and preparation manifests must be joined before
making a frame-level exposure claim. The in-window oracle is explicitly trained
on its evaluated window. Same-broadcast training exposure remains material;
no clean unseen-match claim follows from these receipt joins.

`experiments.tennis.player_profile_audit` provides a CPU-only **normalized-facts**
validator for exact judged ticks per unit and profile, duplicate and missing
records, mask-check facts, declared archive and NPZ hashes, and match, adapter
and base attribution. It rejects absent mask evidence; it does not extract
facts, hash the archive itself, establish that caller-supplied facts came from
the named job, or prove pixel coverage. The CLI consumes a JSON facts object;
its field contract is visible in the focused test fixture. It has not been run
on a complete historical manifest. Passing it after a trustworthy extraction is
necessary, not sufficient, for historical attribution.

Still required: independently verify exact source, preparation, reference,
generation, fit and judge archive hashes and their member identities; extract
actual saved `t` arrays for every profile and judge tick; compare player boxes
with actual per-frame masks and written composite coverage; join training source
hashes and intervals, including W7; and audit original and in-window-oracle
profiles. The judge's counts and the generator's non-flatness checks alone cannot
establish these facts. An absent receipt or coverage fact is unresolved evidence,
not proof that the historical generation failed or silently substituted a crop.
