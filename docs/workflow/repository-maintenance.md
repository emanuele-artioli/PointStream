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

Eleven obsolete/merged worktrees were removed after verification, including the
old foreground checkout's uncommitted connectivity receipt. Remaining useful
checkouts are the dirty Desktop Cursor checkout, the updated background
continuation checkout, and the Gate-A evidence checkout. Temporary review
checkouts can be retired after their revisions and this record are published.
No blanket cleanup script, reset of user work, or claim/process cancellation was used.

Merged branches were deleted only after a positive ancestry check and exact
archive identity match. Remote deletions use expected-head leases to protect
concurrent changes. Checked-out branches and Cursor's shared foundation branch
are protected. Additional superseded branches may be retired only with their
recorded disposition and verified recovery history.

| Branch family retained | Disposition |
| --- | --- |
| `main` | Canonical software; local ref updated without switching Cursor. |
| `codex/demo-smoke-plans-ps-fleet` | Active dirty Cursor checkout; preserve both local and remote heads. |
| `codex/demo-background-smokes` | Useful clean continuation checkout, fast-forwarded to repaired PR head. No diagnostic was replayed. |
| `codex/gate-a-local-confirmation` | Preserve immutable scientific worker revision; it differs from the evidence branch. |
| `codex/gate-a-receiver-pilot-evidence` | Preserve paused scientific checkout and complete96 evidence. Needs a focused integration review; do not merge the broad experimental tree merely because its evidence is useful. |
| `codex/paper-evidence-integration` | Retains the recovered dossier and manuscript integration; parent recovery branch can be retired because its exact history is included. Review the large dossier before merging. |
| `codex/paper-documentation-20260928` | Unmerged71-file documentation rewrite; reconcile current paths and claims before integration. |
| `cursor/cloud-agent-1790859677423-fmt6d` | Unique282-file snapshot; keep as recovery state, do not merge wholesale. |
| `cursor/rtm-sam-ladder-at-risk` | Preserve unresolved source/clip warning until a consumer/input audit proves it superseded. |

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

Next: integrate main into Cursor's branch after preserving its selected dirty
changes; diagnose DCVC Git collection with a discriminating bounded check;
finish the96-frame VVC comparison and AV1 overlap; review/integrate qualified
scientific evidence; then perform the separate storage maintenance procedure in
[storage-layout.md](storage-layout.md). Keep final training gated. Neither recent
session established a complete codec win.
