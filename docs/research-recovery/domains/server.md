# Server evidence recovery

Audit date: 2026-09-30. Read-only SSH from the Mac; no inference, training, source-tree changes, job intervention, or artifact mutation. Shared NFS was inspected through gpu1 once. Host-local storage was probed separately.

## Host and storage coverage

Accessible gpu1, gpu2, gpu3, gpu5 and gpu6 all identify `/home/itec/emanuele` as NFS `data3:/x/home/itec/emanuele`; these are aliases of one shared source, not five independent evidence copies. gpu4 reprobe was refused with “Server is locked for measurements”; its local storage remains inaccessible. Sandbox subprocess SSH initially failed DNS; authorized read-only escalated probes succeeded. A transient gpu6 connection closure was followed by successful inspection. `/usr/bin`-style `python` alias was absent on all hosts; retry with python3 succeeded.

`records/server/host-coverage.json` records access and negative searches. Bounded PointStream-name discovery under `/local`, `/mnt`, `/tmp`, `/var/tmp` found no matching paths on gpu1/gpu2 and 160 immediate host-local artifacts across gpu3 (50), gpu5 (95), gpu6 (15). This is name-based coverage, not proof that unnamed relevant files cannot exist. A depth-three name scan found 57 gpu3 and 129 gpu5 matching paths; those raw inventories remain in external scratch. Nonzero find exit status reflects inaccessible unrelated subtrees whose error text was suppressed; no whole-local-filesystem completeness claim is made.

## Host-local source recovery

`records/server/host-local-source-map.json` maps current temporary worktree payloads to HEAD/branch/dirty status. Administrative Git metadata may be shared while payload files are host-local. Existing shared `git worktree list` can therefore call paths prunable on one host even while their payload exists on another; confirm host before treating such a worktree as lost.

Five dirty gpu3 worktrees were preserved as read-only snapshots in external `workers/server/source-snapshots/`, with original paths, capture timestamps, HEAD and SHA256 in `records/server/dirty-source-snapshots.json`. All 18 records passed local hash verification and capture exit status 0. Selected untracked files were restricted to small source/manifests/tests; datasets/media/weights were excluded. `source-snapshot-dedup.json` identifies duplicate-content source files.

- `gpu3:/tmp/pointstream-motion-contract-20260926`: HEAD `3725388ebadb14f09c046657082d187b7735786d`, branch `codex/motion-contract-20260926`; complete tracked binary diff 104,762 bytes, 20 changed files, 1,959 additions/61 deletions. Existing code and docs patches also retained (93,758 and 11,004 bytes).
- `gpu3:/tmp/pointstream-r02-anchor-roundtrip`: same HEAD, `codex/r02-anchor-roundtrip`; tracked binary diff 16,545 bytes plus untracked anchor-roundtrip implementation/test and two journal manifests.
- `gpu3:/tmp/pointstream-c01-harness`: same HEAD, `codex/c01-mask-codec-harness`; tracked diff 4,634 bytes plus untracked pilot, mask screens and test.
- `gpu3:/tmp/pointstream-temporal-mask-wire`: same HEAD, `codex/temporal-mask-wire`; tracked diff 2,517 bytes.
- `gpu3:/tmp/pointstream-mask-screen`: same HEAD, `codex/yolo-mask-screen`; tracked diff empty, two untracked implementation/test files retained.

Clean host-local source references include gpu3 foreground-campaign (`b771b4bc…`), thesis-rebuild (`88f17b98…`), r01-landmark-metrics (`4e9bcba1…`), journal-plan (`ef87e1e0…`) and gpu5 wave1-genstream-source (`2e7f7e4b…`). These observations establish source recovery opportunities, not validated experiment evidence.

Selected gpu5 cleanup/handoff/worktree audit JSONs and review note were read and hashed; raw extracts are in `workers/server/gpu5-*`. gpu6 `/tmp/audit-pointstream-archive` contains historical repository instructions/tools; it is treated only as archival source evidence, not as current instructions. The high-value shared archive and output findings are consolidated below when their worker inventories complete.

The canonical shared dirty `/home/itec/emanuele/pointstream` checkout is separately preserved: complete tracked binary patch (~865 KiB) and 19 selected small untracked source/docs/tests/manifests, yielding 38 total source snapshot records across canonical and host-local captures. All hashes and capture exit statuses verified. Its HEAD is `3725388ebadb14f09c046657082d187b7735786d`; root separately records main as ahead 12 / behind 30. These snapshots preserve uncommitted states omitted by Git bundles. Root's dirty manuscript patch is indexed in `manuscript-dirty-snapshot.json` with exact gpu3 path, HEAD `2f96b25660d22482a81c8a69998bb597d28e4b19`, SHA256 and timestamp. That root artifact appends status text after the binary diff; separate the two before replay.

The canonical tracked patch is exactly 885,408 bytes and contains 104 `diff --git` file sections. It preserves source/deleted historical docs independently of committed Git history. Root compares recovered bare Git union to public GitHub refs: recovered tags/commits may be absent from current public refs; the server audit does not treat recovered historical labels such as “winning” as current scientific conclusions. In particular JE-05/07/10/11/12/16 claims require later acceptance/rejection context from the chat/evidence domains.

## Fleet source snapshots and archived refs

Corrected snapshot-root dispatch parsing finds 157 source snapshot directories, 153 dispatches with source HEAD/selected patch and snapshot checksums (13 distinct HEADs), and 150 run directories. The initial run-root-only dispatch parsing falsely produced missing provenance and was superseded. Of 157 snapshots, seven have no matching run directory; four have no dispatch/source provenance record: `20260927T201003Z-26f7ce24`, `20260927T212735Z-3e3f9daa`, `20260929T132042Z-72c58cb5`, `20260929T140215Z-a0bb64bc`. Three other dispatch-bearing snapshots lack a run directory. Status metadata of the 150 runs: 68 complete, 60 failed, 12 cancelled, seven contended, two budget exhausted, one running. Status labels establish recorded lifecycle only; “complete” does not establish paper evidence. The running job was untouched. Native encoder paths/versions, GPU UUID, Python version and selected source checksums are recoverable from snapshot-root dispatch JSONs; selected artifact pointers appear in the archive records.

Shared Sep 16 cleanup archives preserve 16 current `archive/20260916/*` tags: 11 planned/verified branch tips plus five stash commit refs not in the branch plan. Historical cleanup inventory lists main plus eight worktrees; current shared worktree metadata and host-local payloads must be reconciled before declaring disappearance. Relevant archive tips include evaluation E01/E02/R0/campaign, handoffs, interrupted checkout preservation, and PR104/108 acceptance fixes. These are recovery sources rather than evidence acceptance decisions.

## Record index and validation

- [Host access and negative-search coverage](../records/server/host-coverage.json)
- [Host-local artifact inventory](../records/server/host-local-inventory.jsonl)
- [Host-local source/worktree map](../records/server/host-local-source-map.json)
- [Preserved dirty source snapshot identities and hashes](../records/server/dirty-source-snapshots.json)
- [Content duplicate map](../records/server/source-snapshot-dedup.json)
- [Selected host-local provenance/source hashes](../records/server/host-local-selected-hashes.json)
- [Raw extract index](../records/server/host-local-extract-index.json)
- [Manuscript dirty patch identity](../records/server/manuscript-dirty-snapshot.json)
- [Search log](../records/server/search-log.json)
- [Validation](../records/server/validation.json)

Worker model settings were `gpt-6-luna`, reasoning `xhigh`, `fork_turns=none`, on nonoverlapping outputs and archives/provenance shards. Parent checked source snapshot hash/status integrity, mount alias identity, and fleet dispatch schema; that review corrected run-only dispatch extraction to snapshot-root parsing. No source changes were made, so source tests were not run. Existing result values were not recomputed and scientific eligibility remains pending experiment-domain assessment.

## Shared output families and experiment handoff

All 179 top-level output families have metadata/source-map records: 169 recursive traversals completed without pruning; ten completed with 112 explicitly listed media/cache directories pruned. No traversal errors were recorded. Visible totals are 122,960 regular files and 874,878,350,575 bytes outside pruned directories; these are lower bounds. The initial depth-four inventory covered all families with 19,544 metadata records. Symlinks were recorded and not followed. Selected content recovery is 242 small report/provenance paths totaling 1,217,351 bytes with SHA256.

Of 99 timestamp-shaped run directories, 47 lack `run_summary.json`; these are orphan/unindexed candidates, not confirmed completed experiments. Most July 9 candidates are tiny debug fragments. `gpu1:/home/itec/emanuele/pointstream-data/outputs/20260711_061829_768133` is a higher-priority recovery candidate: 77 files include anchor cache, scene cache and transport but no summary. `20260709_140832_616572` appears empty within bounded metadata.

Experiment lead should start with these explicitly mapped families:

- `gpu1:/home/itec/emanuele/pointstream-data/outputs/journal-20260924/je00` through `je16`. JE00 includes source revision `ef87e1e` and pending branch `b771b4b`; JE01/03/04 include `10d0ab9`. Inspected JE05/07/10/11/12/16 result documents contain no PointStream source revision field. Recover source from commit/history/dirty snapshots and protocol context before evaluating eligibility. `je10-winning-result` is a historical filename, not accepted scientific evidence.
- `gpu1:/home/itec/emanuele/pointstream-data/outputs/journal-20260925/{c01-gpu,mask-screen}`. Run-3 smoke provenance includes a runner SHA and A6000 device; match exact runner/manifests to preserved c01/r02 source snapshots.
- `gpu1:/home/itec/emanuele/pointstream-data/outputs/journal-20260926/sam31-racket-prompt-smoke16-run1/{result.json,run-manifest.json,status.json,sam31_smoke_runner.py}`. Provenance includes GPU UUID, checkpoint/repository revision/hash and a 16-frame development window. `pointstream_codec_run=false`; result explicitly says codec rate/quality/decode metrics were not measured. Two selected tracks appear in 12/16 frames (75% coverage). The durable runner SHA exactly matches the result’s recorded runner SHA despite the historical `/tmp` runner path.

- [All family identities, stage hints and visible counts](../records/server/outputs_family_inventory.tsv)
- [Actionable deduplicated output source map](../records/server/outputs_actionable_source_map.jsonl)
- [High-signal artifact paths](../records/server/outputs_candidate_artifacts.jsonl)
- [Orphan/unindexed timestamp run candidates](../records/server/outputs_orphan_runs.json)
- [Selected report source hashes](../records/server/outputs_selected_report_hashes.tsv)
- [Output coverage, pruning and negative searches](../records/server/outputs_coverage.json)
- [Output findings](../records/server/outputs_findings.md)

## Shared datasets, jobs, dependency links and legacy archive coverage

The archive shard supplies a depth-two metadata inventory for the scoped shared roots: assets, manifests, jobs, weights, third_party, canonical source and confirmation checkout, worktrees, shared tmp and Sep 16 cleanup archive. The inventory includes 38 top-level job family/script entries and direct children; dataset family/scene identities are explicit through depth two. Fifteen asset/weight/dependency symlinks were resolved in the bounded searched scope; all targets existed, with no broken links in that scope. Three PointStream-specific preserved local-file directories appear in cleanup metadata; their payloads were not deeply traversed. These sources can recover historical protocols, job entrypoints, weights/input identities and archived source associations without copying large payloads.

Two archive recursive attempts were stopped (~two minutes and ~75 seconds) and yielded zero-byte captures retained with explicit incomplete status. Recursive dataset/vendor totals and deeper payload paths remain unverified; no selected archive report/manifest file hashes were collected. This limitation does not apply to the separate output shard’s 242 selected source hashes or the 153 fleet dispatch recorded source identities. Shared manifests and legacy paths remain actionable pointers in the scoped raw metadata record. No searches of credentials, unrelated home history, or unrelated projects were performed.

- [Archive source map](../records/server/archives_SOURCE-MAP.md)
- [Corrected fleet provenance](../records/server/archives_fleet-provenance-gpu1.json)
- [Exact experiment command/input/source map](../records/server/archives_fleet-command-source-map-gpu1.json)
- [Shared job family inventory](../records/server/archives_jobs-families-gpu1.json)
- [Dataset family and scene identities](../records/server/archives_asset-identities-gpu1.json)
- [Scoped resolved symlink targets](../records/server/archives_symlink-targets-gpu1.json)
- [Current Git worktree metadata](../records/server/archives_git-worktrees-gpu1.txt)
- [Current Sep 16 archive commit refs](../records/server/archives_pointstream-archive-tags-gpu1.json)
- [Historical cleanup recovery records](../records/server/archives_cleanup-pointstream-records-gpu1.json)
- [Preserved local-file directory names](../records/server/archives_cleanup-local-file-names-gpu1.tsv)
- [Scoped shared root inventory](../records/server/archives_initial-lists-gpu1.tsv)
- [Incomplete archive recursion coverage](../records/server/archives_recursive-coverage-status-gpu1.json)
- [Archive extract file hashes](../records/server/archives_extract_index.json)

The large raw fleet metadata listing (90,858 depth-four rows, ~13 MiB) stays in external `workers/server/archives/fleet-structure-gpu1.tsv`; committed records retain lightweight selected identities and provenance instead.

Parent reconciliation corrected 131 source-map/TSV status labels from `depth4_only` to `complete`: the initial unrestricted raw family summaries lacked an explicit status key, but contained completed recursive counts, no pruning and zero errors. All 179 source-map statuses now agree with coverage (169 complete, ten pruned). Two full reports remain targeted content follow-ups: `bp21-headroom/report.json` (176,353 bytes) and `gate-a-vvc-webp-n96-run2/report.json` (231,232 bytes). They were inventoried but excluded from bounded selected small-report copying; bounds alone do not substitute for these reports. [Exact follow-up paths](../records/server/outputs_targeted_followup.json).

## Coordinator follow-up

The two large-summary copy gaps above were subsequently resolved: BP21 and native Gate A full reports were copied through the same NFS via GPU3 and their hashes checked. Their original shard disposition remains in the [follow-up record](../records/server/outputs_targeted_followup.json), alongside the completed coordinator action. [Independent arithmetic](../records/coordinator-arithmetic.json), [Gate A curve analysis](../records/gate-a-derived.json) and [seven declared source-file matches](../records/gate-a-source-file-check.json) retain separate limits. Source/data recertification is not implied.
