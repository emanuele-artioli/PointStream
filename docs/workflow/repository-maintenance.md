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

| Branch retained (local and remote unless noted) | Reason it is neither merged nor deleted |
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

Next: audit the demo source/media identities and selectively integrate qualified
scientific branches; diagnose DCVC Git collection with a discriminating bounded check;
finish the96-frame VVC comparison and AV1 overlap; review/integrate qualified
scientific evidence; then perform the separate storage maintenance procedure in
[storage-layout.md](storage-layout.md). Keep final training gated. Neither recent
session established a complete codec win.
