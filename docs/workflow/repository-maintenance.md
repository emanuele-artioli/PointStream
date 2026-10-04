# Repository consolidation — 4 October 2026

## What the recent sessions established

The paper chat (`01a0f3d8-8225-77c0-b10a-4a9fd2401c9b`) published an empirical
working manuscript, preserved receiver/anchor accounting, and wrote the
[scientific handoff](pointstream-handoff.md). The fresh96 alpha receiver uses
58,139 complete bytes at VMAF62.695164; the null uses52,831 at62.708. Static
foreground adds cost here without improving that metric. The48-frame QP52 pilot
cannot certify a comparison against the96-frame candidate. Same96 QP50 is
66,955 bytes at68.285094; neither point dominates. Finish the native96 bracket,
then strengthen AV1 overlap before claiming a local rate–quality gain. Repeated
initial poses and clipping still limit semantic conclusions.

The background chat (`01a10618-57f9-7701-a1a8-cd435197159c`) improved bounded
source/environment probes, immediate failure receipts, selected snapshots, and
budget reconciliation. Its latest inventory failed on the first DCVC Git diff
at30 seconds. This is a provenance-collection failure, not a failed model or
exhausted budget. B2 codec, B3 drift, B4 latent diagnostics and training remain
unrun. Diagnostic authority remains50 cumulative GPU minutes and60 preparation
minutes, including prior attempts:1480 GPU reservation seconds and2564.531
conservative preparation seconds charged. Do not reset those charges. Debugging
within remaining caps may continue; failed validation blocks promotion. Final
training requires its own configuration, pilot, split checks and authority.

## Merged software and evidence

Merged in order: #150 fleet inbox; #151 clip sampling; #152 qualified experiment
evidence; #154 dataset/model path compatibility; #155 audited foreground smoke
foundation; #153 background diagnostics; #156 concise fleet guidance and handoff.
Every PR passed full Linux lint, type, test and coverage checks before merging.
The combined first four trees also passed166 focused tests; foundation checks
passed160 locally with the optional Torch objective skipped; the background
suite passed230 with one Torch skip. Full CI exercised the Torch fixtures.

The smoke stack needed concrete repairs: equal-sized objective test tensors
retaining unequal alpha areas, inherited fleet type narrowing, missing `Any`,
typed packet fixtures and lint fixes. Scientific receipts and manifests were
not rewritten. The remaining maintenance change corrects FFmpeg's native
version option (`-version`), confirmed by the fleet probe regression test;
legacy `--version` returned8 on inspected hosts. No experiment, model training,
worker restart, remote primary update or storage cutover was launched.

## Cleanup and recovery

Recovery archive:
`/Users/manu/Datasets/audits/pointstream-maintenance-20261004T203008Z/`.
`pointstream-all-refs.bundle` is verified and preserves the pre-cleanup history;
`refs-before.txt` maps old branch names to exact commits. Per-checkout tarballs,
binary patches and `worktrees-before.json` preserve unfinished and non-regenerable
ignored files. Only reproducible caches, virtual environments and node modules
were excluded from those tarballs. Three redundant standalone clones were
atomically moved intact into `retired-checkouts`, including their Git metadata.
Machine-readable cleanup records and a supplementary bundle preserve subsequent
maintenance revisions. Keep this archive private; it can contain local configuration.

Eleven obsolete/merged worktrees were removed in the first audit. After Cursor
finished, its primary checkout was independently archived again at
`/Users/manu/Datasets/audits/pointstream-maintenance-20261004T211109Z/`.
Every loose file was compared with its tarball before cleanup. Identical copies
already exist in merged background code; differing background copies would
restore dependency stubs, remove bounded provenance/progress collection and
undo typed fleet recovery checks. The fleet timeout override would increase
300 seconds to900 without addressing the NFS failure, so it was discarded.

The untracked `gate_a_confirmation/anchors.py` and its five contract tests are
preserved in that archive. This earlier two-scene sweep prototype is superseded
by the registered single-arm `native_anchor.py` and `native_anchor_v2.py` in the
retained Gate-A branch. It lacks the newer original-window/per-frame resource
receipts and does not bind full promotion to the completed smoke path. It was
not promoted to a second production runner. Private `.cursor` policy is archived.
Saved outputs and non-regenerable ignored files remain intact.

The Desktop checkout now follows merged code. Cursor's merged foundation branch
and the redundant clean background continuation checkout/branch were retired.
The user also approved retiring all six archived legacy branch families:
foreground smokes, foreground campaign, GPU permissions, GPU workflow,
packet-rate quality, and research evidence recovery. Exact heads were checked
against the verified bundle; remote deletions used expected-head leases.
No blanket cleanup script or claim/process cancellation was used.

| Branch retained at the first audit (local and remote unless noted) | Initial reason |
| --- | --- |
| `main` | Canonical maintained software and default branch. |
| `codex/gate-a-local-confirmation` | Exact scientific execution revision; deleting or merging its broad experimental tree would obscure the distinction from the evidence revision. |
| `codex/gate-a-receiver-pilot-evidence` | Paused scientific checkout and complete96-frame evidence; requires focused scientific integration review. |
| `codex/paper-evidence-integration` | Large recovered dossier and manuscript integration contain unique provenance; qualify its claims and source records before merging. |
| `codex/paper-documentation-20260928` | Unique71-file documentation rewrite needs reconciliation with current paths, results and claims. |
| `cursor/cloud-agent-1790859677423-fmt6d` | Thirteen unique diagnostic scripts plus historical media/output snapshot. They lack current fleet admission/provenance review; preserve until useful diagnostics and external data are separated. |
| `cursor/rtm-sam-ladder-at-risk` (local only) | Contains unresolved clip identity warning plus unverified metric/cache and mask-selection changes. The visible historical-demo warning is integrated separately; keep original branch until source/media identities are audited. Do not push its unverified metric replacements as reviewed work. |

The only additional local PointStream worktree is
`/private/tmp/pointstream-packet-rate-quality` on the paused Gate-A evidence
branch. Keep it clean and pinned; create a separate integration branch when
reviewing its source. The standalone manuscript clone
`/private/tmp/pointstream-paper-packet-rate-quality` is a different repository,
not a PointStream Git worktree, and remains useful for the manuscript session.

Legacy GPU permission prose and unrestricted-launch workflow branches are
superseded by the inbox interface; the useful FFmpeg correction is ported
separately. The old foreground branch never launched a GPU job and its failed
connectivity receipt is archived. The PR143 head was independently confirmed
merged; its original commit is in the archive. The packet-rate branch is included
in the retained Gate-A evidence history. No novel research or checkpoint branch
is deleted merely because it is old or unmerged.

## Remote state and next work

A fresh fleet inspection reaches gpu1, gpu3, gpu5 and gpu6; gpu2 times out.
Availability is per host and temporary. This does not certify shared-inbox doctor
checks, model inputs, or a storage maintenance window. The earlier gpu1 refusal
is historical. Inspect existing jobs and receipts before further dispatch.
The current management RPC still defaults to its first host; bounded read-only
fallback and CPU-only admission remain implementation follow-ups.

The remote primary `/home/itec/emanuele/pointstream` has substantial unfinished
changes and must be preserved. Its shared Git metadata lists six older worktrees.
Two have explicit unfinished changes (mask-wire code and second-domain manifest).
Several clean heads are absent from the Mac's fetched history, and ownership
across unavailable hosts is not verified. Archive and reconcile their remote
history/process ownership before removing them; cleanliness alone is insufficient.
No remote worktree, dataset, model, checkpoint, frozen release or source tree was moved.

The bounded follow-up on gpu3 confirmed four old `/tmp` checkouts clean and
`temporal-mask-wire` dirty. Primary and confirmation status exceeded six seconds;
the previous observed dirty state remains unresolved. Server-only branch
retention is explicit:

| Server branch | Reason retained |
| --- | --- |
| `main` | Remote primary contains unfinished work; do not fast-forward/reset over it. |
| `feat/gate-b-confirmation-and-second-domain` | Paused confirmation work and changed second-domain manifest require provenance review. |
| `codex/foreground-campaign-20260924` | Server head b771b4b differs from the retired Mac/PR143 head; retain until remote history is archived and ownership reconciled. |
| `codex/journal-experiment-plan` | Clean checkout, but unique remote-only head ef87e1e is absent from the verified Mac archive; preserve research planning until archived/reviewed. |
| `codex/r01-landmark-metrics` | Clean checkout, but remote-only head4e9bcba has not been imported/reviewed or archived here. |
| `codex/temporal-mask-wire` | Uncommitted `src/runner/mask_wire.py` remains; preserve until its code and scientific use are reviewed. |
| `codex/rebuild-semantic-thesis` | Clean checkout, but remote-only head88f17b9 is outside the verified Mac archive; reconcile manuscript/provenance before retirement. |

These seven server worktree registrations share Git metadata. A clean `/tmp`
checkout does not establish inactivity on every host; gpu2 is unreachable.
None is silently classified as safe to delete based only on its age.


Next: audit the demo source/media identities and selectively integrate qualified
scientific branches; diagnose DCVC Git collection with a discriminating bounded check;
finish the96-frame VVC comparison and AV1 overlap; review/integrate qualified
scientific evidence; then perform the separate storage maintenance procedure in
[storage-layout.md](storage-layout.md). Keep final training gated. Neither recent
session established a complete codec win.

## Follow-up reconciliation — 5 October 2026 (Vienna)

All ten server branch heads were imported for read-only local content/ancestry
review and archived in a verified1,188,324,607-byte bundle at
`gpu3:/home/itec/emanuele/Datasets/audits/pointstream-server-reconciliation-20261004T212959Z`.
The archive includes the primary's binary diff and90 loose files. Its599
non-regenerable ignored entries are inventoried and remain in place; this is
not a license to reset or remove that active checkout. The broad confirmation
ignored-file inventory timed out; its dirty second-domain manifest was archived
and verified separately. Both shared checkouts remain unchanged.

The all-host inventory confirmed shared primary/confirmation inode identity
on gpu1/gpu3/gpu5 and physical gpu3-local `/tmp` worktrees. GPU4 remains locked;
gpu2/gpu6 connected but their bounded Git inventory timed out. Process checks
found an owned `MainThread` using the primary, and no owned cwd/argument
references to the five temporary checkouts before retirement. Exact saved
statuses, owner IDs, tar hashes and absent non-regenerable ignored files were
checked. The four clean temporary checkouts were removed with Git. The mask
prototype's exact dirty file was verified against its archive, restored, then
its checkout removed with Git; no recursive-force fallback was used.

The mask prototype defines a new temporal API but replaces `wire_declaration`
with `NotImplementedError`; it is unfinished design, not a codec implementation.
Its original file remains archived. The useful R01 landmark scorer and its
hand-computed controls are integrated separately, with nonfinite-score rejection.
Three additional branches were found: `feat/journal-je10-winning-codec-candidate`
is an ancestor of retained server main; `feat/semantic-generative-eval` has the
same maintained metric implementations except unused imports/type annotation;
`modular-evaluation-framework` is an older implementation and synthetic ceiling
record whose claims are qualified in the recovered dossier. Their whole source
histories remain archived; none is a scientific winner merely because of its name.

The research dossier, preparation chapters and qualified campaign receipts are
now indexed at [research/README.md](../research/README.md). Old documentary
states are labelled historical; current Gate-A pauses/limits and manuscript
revision remain separate. [Source map](reconciliation-source-map.json) pins every
selected original file/revision/hash and notes review changes. Original runner
experiments are not imported wholesale. Packet packing remains opt-in with
lossless pixel/native-byte parity tests, explicit lossy flags and allocation/
corrupt-header checks. Software tests are not new paper evidence.

The [demo audit](../research/demo-source-identity-audit.md) independently confirms
conflicting factory001/factory035 names and missing source-to-stream hashes.
All three retained reference videos have300 frames at1920x1080/30fps; that
metadata does not establish which source was intended. Future exports reject
positional/unhashed manifests, bind explicit IDs, verify sources before/after,
record identities and reserve fresh external outputs before model work. Old
media/results remain unchanged. README/setup and imported preparation commands
now use the canonical gated fleet interface rather than unrestricted launches.

Remaining worktree targets are the shared active primary and paused confirmation
on the server, the clean Mac primary, the paused Mac Gate-A evidence checkout,
and the independent manuscript clone. A new clean Claude checkout appeared during
the audit at `.claude/worktrees/unify-demo-pointstream-6365e8`, on
`claude/unify-demo-pointstream-6365e8` (observed head `4e81729`). Its four commits
add egocentric-domain, objectstream-wire and shared-checkout provenance work.
It is separate concurrent work, preserved for its own review rather than retired. Existing claims/jobs/processes were not
cancelled. No new compute campaign, model training, frozen-worker upgrade, live
storage cutover or manuscript edit occurred. The scientific/native comparisons
and background preparation budget remain as recorded in the handoff; they were
not resumed by this maintenance pass.


The selectively integrated documentation source histories are retained as merge
parents without restoring the old tree, deletions or launch instructions. Once
this reviewed integration is merged, both paper-documentation and paper-evidence
branches are eligible for ordinary merged-branch retirement; immutable source
citations remain reachable. The five legacy temporary server branches and three
older unoccupied server branches are eligible for archived retirement after
publication of the selected R01 functionality and provenance record. Retain server
main and confirmation, and the Mac Gate-A execution/evidence and two Cursor
source-audit branches. Exact head leases and verified recovery history remain
required for remote deletions. The final external cleanup manifest records actual
operations; these eligibility statements do not certify a still-pending merge.
