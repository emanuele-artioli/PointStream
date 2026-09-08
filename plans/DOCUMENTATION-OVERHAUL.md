# Documentation overhaul — implementation brief

**Outcome of this session: a migration plan, not an applied overhaul.**
Audited code baseline: `bc09184d87` (PR #72), plus the user's five uncommitted
edits, preserved in [user-edits.json](documentation-audit/user-edits.json).
The JSON `patch` field contains the exact original Git diff, with whitespace preserved.
Paper baseline: `87d9e52`, a separate repository. Reconcile any later changes
before implementation. Dates describe observations; commits identify states.

**Dispatch:** VS Code with Antigravity, its strongest available reasoning setting
for the reconciliation pass; a middle effort for the subsequent mechanical edits.
The latest execution report names Gemini 3.8 Flash, but verify that exact model
and its effort options in the current selector rather than assuming availability.
Use Codex/GPT-6 Astra at high effort for unresolved scientific conflicts only;
do not ask it to reread all history. This is one sequential migration with a
cross-repo reference dependency; parallel agents skipped for the initial pass.

Read this brief and the inventory first. Read old documents by assigned area,
not all at once. Do not run the old experiment prompts while auditing them.

## 1. Decision: two entry points, one current-state index

Keep `README.md` for people installing and running PointStream. Keep `AGENTS.md`
for stable project purpose, rules and read/write routing. Keep `PLAN.md` as a
short index of current areas. **Do not move its changing status into AGENTS.**
That would load every area's state into every session and recreate the same
staleness with a more expensive entry point.

Use this final layout (create a supporting file only when content earns it):

```text
README.md                         purpose, prerequisites, install, first run
AGENTS.md                         purpose, project rules, read/write map
PLAN.md                           current area table and active assignments
CLAUDE.md                         existing adapter to AGENTS
.github/copilot-instructions.md   existing adapter to AGENTS

docs/setup.md                     portable data/config/tool setup and usage
docs/roadmap.md                   gates, dependencies, criteria; no daily schedule
docs/areas/codec.md               background, appearance, motion, residual, routing
docs/areas/evaluation.md          metrics, search, confirmation, ablations, baselines
docs/areas/data.md                acquisition, eligibility, splits, domains
docs/areas/generation.md          model status, controls, candidate next actions
docs/areas/infrastructure.md      environments, CI, jobs/recovery, profiling, cleanup
docs/areas/paper.md               code-side evidence delivery and paper-repo entry
docs/history/pull-requests.md     searchable PR titles/state/commits, grouped by area
docs/history/findings.md          short validity/retraction index, not copied reports
docs/history/retired-documents.tsv old path -> exact historical blob and owning area
docs/workflow/session/SKILL.md    route, dispatch, report and closeout workflow
```

No `docs/README.md`, permanent root handoff, global terminology sheet, second
roadmap, or separate deferred backlog. Retire `plans/` once its useful content
has destinations and references resolve. Retire this migration bundle last;
its PR and parent commit preserve the audit.

The paper repo keeps its own AGENTS, marker convention and source layout. A
link from the paper area makes it reachable; it is not a child source tree to
reorganize as part of the code-repo PR.

## 2. Audit findings that the migration must correct

1. PLAN/HANDOFF dispatch a native run already reported in **#69 / `648325b`**.
   Current state is a completed diagnostic with checkpoint alarms, not a passed
   Gate A or a pending initial launch. Do not convert the report's recommendation
   into automatic authority to run another experiment.
2. **#63 is BP57 acquisition; #64 is BP56 background effort.** `plans/README.md`
   reverses them. The attached PR index uses exact GitHub API titles and merge
   SHAs, including closed-unmerged #5. A title alone is a search key, not evidence
   that an implementation or a scientific claim is valid.
3. **#70–#72 are plans.** The competitive-regime document mixes measured plate
   diagnostics with predicted system quality/rate/speed and statements that a win
   is unlocked. Preserve the proposed VVC/SVT background work, appearance-codec
   sweep, fast evaluation and client profiling, but label unmeasured outcomes as
   hypotheses. A smaller plate at a different quality is not a system win or
   proof of a universal libaom bug. Preserve the user's edits to that plan.
4. The stream registry already has AV1/HEVC/AVC; only its AV1 entry selects
   libaom. Standalone AV1/VVC intra sidecars already landed in **#36**. The next
   codec task must inspect `sidecar.py`, `stream.py`, config, runner and independent
   client decoding to identify the missing connection. Do not dispatch another
   standalone VVC sidecar just because the latest plan says to add one.
5. `vmaf.py` still writes PNGs and does not pass `n_threads`. The refactor is
   proposed, not delivered. PSNR-only exploration must retain its exploratory
   label and cannot pass a VMAF gate. A changed implementation needs agreement
   checks against the old metric path on known inputs before its results are used.
6. `plans/done/` contains **93 files**, including parked BP19/BP28, void roster
   gates, retractions and long reports cited by code and the paper. Retirement
   is appropriate; assuming they are all completed or duplicated in PR bodies
   is not. #69's body, for example, omits the report's full timing/control ledger.
7. DEFERRED D1/D3 are closed; D5's requested no-branch architecture conflicts with
   the later explicit coded fallback design. D6 names pre-rewrite paths. The
   panorama item describes work that later landed. Resolve these against code
   and newer decisions; do not copy all of them as fresh TODOs.
8. README has no clean-machine first run and assumes a pre-existing host env.
   `environment.yaml` requests Torch 2.5.1, but the measured local env imports
   2.2.2+cu121. A successful local test cannot establish that the declared fresh
   install works. Do not silently change dependency pins during a docs migration.
9. The external-data fix is recorded in **#32**, including `d436b02`, and the
   weights-path follow-up in **#35**. Before this audit only the main worktree
   remained. `.vscode/settings.json` is absent despite AGENTS claiming it exists.
10. **Do not automate the existing cleanup script.** #68's
    `scripts/cleanup_merged_worktrees.sh` suppresses fetch/status/log failures,
    falls back from `git worktree remove` to `rm -rf`, and performs remote-ref
    pruning. These are incompatible with treating Git refusal as a stop. Record
    a bounded infrastructure repair task; this audit did not execute the script.
11. Code and paper contain **126 references** selected by the audit's path/name
    search. Some already point to pre-archive paths or removed PLAN sections.
    Fixing Markdown links alone will not make the documentation traceable.
12. The paper's AGENTS and its archived reviewer checklist disagree about the
    old review's authority. Keep the checklist historical; paper AGENTS says TOMM
    is a fresh submission, not a rebuttal. Preserve raw reviews as evidence.

## 3. Current performance and the environment decision

Remove the old numerical NFS/import narrative from project AGENTS. Keep the
preventive rules, and link setup/infrastructure for detail. Do not edit the shared
host-wide rules from this project session.

Observed on **gpu6**, code `bc09184`, 2026-09-08. These are short repeated
process timings on the existing shared host, with existing caches, no login
shell, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and explicit checkout
`PYTHONPATH`. No GPU workloads were launched. Bounds declared before observation:
ordinary Git/discovery 0–5 s; fresh Torch process 0–300 s. All samples completed;
no bound alarm fired. `perf_counter` passed a 0.1-second sleep anchor three times
(0.10014–0.10015 s); no-op processes are the startup controls.

| Operation | n | Mean ± standard error, seconds | Observed range, seconds |
|---|---:|---:|---:|
| `git status --short` | 5 | 0.057 ± 0.034 | 0.020–0.195 |
| `git ls-files` | 5 | 0.0051 ± 0.0001 | 0.0046–0.0053 |
| `rg --files src tests scripts experiments` | 5 | 0.0147 ± 0.0006 | 0.0134–0.0168 |
| System Python no-op | 5 | 0.0227 ± 0.0006 | 0.0215–0.0243 |
| PointStream Python no-op | 5 | 0.0455 ± 0.0113 | 0.0330–0.0905 |
| PointStream Python importing sqlite3 | 3 | 0.0440 ± 0.0074 | 0.0353–0.0587 |
| PointStream Python importing sqlite3 then Torch | 3 | 2.101 ± 0.620 | 1.350–3.331 |
| PointStream Python importing sqlite3 then `src.runner` | 3 | 0.529 ± 0.117 | 0.406–0.764 |

Runner imports did not load Torch. This demonstrates usable current Git,
discovery and import paths, **not** a causal speedup factor against August,
universal NFS health, GPU throughput, editor latency, or timings on other hosts.
The historical data-move counts are recorded history, not remeasured here.
Raw samples: [environment.json](documentation-audit/environment.json).
Reproduction: [measure_environment.py](documentation-audit/measure_environment.py).

Keep these practical constraints in AGENTS, without old timings:

- Data lives outside code; use the resolver and set each new worktree's marker
  or explicit environment override. The gitignored marker is not copied by Git.
- Do not bring assets/outputs back as symlinks. Open a single checkout in editors.
- Keep regenerable caches on host-local disk, namespaced by full checkout path.
- Read host rules for shared GPU use, detached/checkpointed jobs and package
  isolation; keep `sqlite3` before Torch. Do not duplicate the host rule essay.
- Resolve native encoder paths, versions and capabilities explicitly.
- Keep the existing lint, mypy, relevant-test and import-direction requirements.

Put `.ps-data-root` precedence and `paths.describe()` in `docs/setup.md`; both
README and AGENTS link it. Keep migration history and these measurements in the
infrastructure area or pinned history, not as permanent claims about host speed.

## 4. Exact entry-point contracts

### README: a stranger can start without the author's server

Order: short purpose/pipeline paragraph; research status; supported setup;
installation; smallest runnable example; real-data workflow; output locations;
repository map. No session routing, quotas, host home paths, submission calendar,
agent checks, paper-writing command, or list of old plans.

Include scene selection, compatible reusable background, sparse object
appearance/motion, receiver reconstruction, optional correction/generation and
coded fallback. State that it is research software and a confirmed AV1/VVC win
is not established at the indexed evidence revision. Avoid promising generative
reconstruction in configurations that paste reference crops.

Use `environment.yaml`/`pyproject.toml` as installation sources. Starting recipe:
clone the public repo, `cd PointStream`, `conda env create -f environment.yaml`,
`conda activate pointstream`. Document Linux/NVIDIA CUDA expectations and optional
native encoder/model requirements based on actual selected paths. Do not claim
Windows/macOS or CPU-only feature parity without a driven path. Do not install
into the existing research environment to validate the recipe.

For a data-free first check, derive the command from the existing synthetic tier
integration tests: `python -m pytest tests/runner/test_tier_end_to_end.py -q`.
Explain what it checks and any libvmaf-dependent skip. It is a synthetic path
check, not a real video demonstration. For actual use, inspect
`experiments/tier/run.py` (`--tiers`, `--frames`, `--video`, `--scene`, `--out`),
its loader and manifest contract, then document a concrete supported input
preparation/run sequence. There is no `src.runner.__main__`; do not invent
`python -m src.runner` or an arbitrary-MP4 CLI. If a public-data end-to-end recipe
cannot be exercised in scope, label that exact limitation and link the supported
manifest/API path; leave its verification as a named infrastructure action.
Do not present cached author-only data as bundled example data.

### AGENTS: purpose, read/write routing, completion behavior

Promote the PLAN objective and ROADMAP core pipeline here. Retain the priority:
find a named size–quality regime, always report time, confirm independently,
then optimize a frozen configuration. Keep the submission date as the current
project target; avoid fabricated daily deadlines. Research negatives, search
scope and all validity limits remain visible. If no regime wins, raise the
claim decision; documentation cannot guarantee a winning outcome.

Replace the read-only location table with:

| Need | Read | Write/update |
|---|---|---|
| Current assignment | PLAN and one linked area | That area's state/actions; PLAN only if its summary changes |
| Gate dependency | docs/roadmap.md | Roadmap only when dependency or pass criteria changes |
| Prior decision/failure | Area evidence links, history indexes, relevant PR discussion and pinned files | PR for session detail; findings index for validity changes |
| Component behavior | src/contracts and relevant code/tests | Code/contracts/tests in the owned scope |
| Human install/run behavior | docs/setup.md; README only when changing user instructions | README/setup alongside behavior change |
| Research evidence | Area protocol and immutable outputs | Run records; area verdict; PR provenance |
| Paper claim | Paper area's link, then paper AGENTS and section markers | Separate paper commit; evidence references back in paper area |
| Dispatch/report/closeout | docs/workflow/session/SKILL.md, relevant mode only | Prompt in response; PR report; durable area update |

Add this completion rule, in the project's plain voice:

> Before ending each response, check whether the requested issue is solved.
> If it is, record the result and validation in a PR, update the owning area's
> current state and next action, and consider whether this is a useful session
> boundary. Do not create a new report file for a completed session. If work
> remains, state the unresolved decision and produce a scoped continuation
> prompt when a handoff is useful. A prompt does not replace durable decisions
> or the area update. Avoid a PR for a question answered without repo changes.
> Keep one PR per independently reversible change; do not open a PR per reply.

Cleanup is an explicit check, not a blanket deletion instruction. Confirm merge
against fresh `origin/main`, clean status, unique commits/diff and the worktree's
session owner. Preserve unmerged work before retirement. Current host rules
require asking before removing a worktree that may host a paused session and
reserve remote-branch deletion for the user. Respect existing authorization;
do not ask again for a specific cleanup already approved. Never force removal
or use `rm -rf` to bypass Git. Do not make irreversible operations part of an
implicitly invoked skill. Do not run the #68 script in its current form.

### PLAN: only a dispatchable area index

Header: `State reconciled through PR #72 / bc09184; evidence anchors are per row.`
After migration, replace the header's revision with the actual audited baseline;
use the current PR number while it is open, then its merge SHA at the next
reconciliation. A file cannot contain its own final commit hash; avoid endless
self-update commits. Distinguish implementation SHA, experiment run SHA and
state-doc reconciliation revision.

One row per area: area link, current state, last verified code/evidence PR or
commit, next action ID, dependency/owner. Keep the whole index under roughly
100 lines, and avoid restating area details. Starting rows:

| Area | Reconciled state to carry forward | Evidence anchor / next action |
|---|---|---|
| Codec | Offline canonical canvas and explicit fallback exist; leaner background/appearance work proposed | #36/#45/#50/#52 implementation; #64 seed; #70–#72 proposals; inspect missing stream/client connection |
| Evaluation | 48-frame diagnostic completed, Gate A not passed, checkpoint alarms persist | #65/#66 plumbing; #69 result; #72 fast-eval proposal; repair/validate metric path before new evidence |
| Data | Diagnostic corpus exists; fresh sources provisional, six-source confirmation not established | #56/#57 audit, #60 shortlist, #63 acquisition; validate eligibility/independence |
| Generation | No confirmed gain over pasted-reference control; old roster is scoped historical evidence | #20/#27/#28 plus BP12/19/28 pinned files; reassess previously deferred actions with dependencies |
| Infrastructure | External data and clean worktree layout; current imports fast; recovery/cleanup/CI issues distinct | #32/#35 data, #53 recovery, #68 cleanup, #69 gap alarms, this audit; repair cleanup separately |
| Paper | Separate repo; no supported winning headline yet | paper `87d9e52`; verify page counts at paper revision, then update claims from admissible evidence |

For active assignments, record task ID, branch/worktree, base revision, owner,
allowed paths, dependencies and PR. This is a coordination aid, not a lock.
Use isolated worktrees, recheck shared files before merge, and update only your
own area/row. Do not leave completed tasks shown as assigned.

## 5. Area documents and roadmap

Each area starts with its evidence/reconciliation revisions, scope and owned
paths. Then: current state; a short evidence/decision table; next actions;
acceptance/dependencies; and only the operational context needed to continue.
A next-action row has stable ID, proposed/ready/active/blocked status, dependency,
acceptance condition, and source PR/file. Mark inherited backlog items
**previously deferred**, with the old reason and the condition for reconsidering.
That label is not permission to launch them. Closed findings are not TODOs.
A finished task becomes a concise state/evidence update; its narrative stays in
its PR or immutable output report, not a growing Delivered section.

Move ROADMAP sections as follows:

| Old section | Destination and treatment |
|---|---|
| Opening PR chronology | history index; current facts to owning areas; correct swapped PRs |
| 1, core pipeline | AGENTS and concise human version in README |
| 2, gates A–E | docs/roadmap.md; keep pass conditions and dependencies |
| 3, search/protocol | evaluation area; codec/data sections link their owned work |
| 4, canvas | codec area: implemented design vs still-unmeasured long-context evidence |
| 5, payload simplification | codec next actions, preserve order as strategy, not proof of benefit |
| 6, calendar | drop daily rows; preserve hard submission target and currently agreed evidence freeze as policy, with revision and ability to revise |
| 6.1/6.2, fallback/relaxations | evaluation area: explicit alternative hypotheses, parity and disclosure conditions; do not silently switch the headline metric |
| 7, workstreams | area actions + assignment table, not another S0–P4 backlog |
| 10, parked work | area next actions, each marked previously deferred with its dependency |

Keep Gate B's frozen criteria and independent-source requirement; Gate C's core
ablations; Gate D's learned baseline/second-domain/speed work. Reconcile the
newly proposed client profiling with the previous optimize-after-confirmation
rule: gathering a profile may be a bounded diagnostic; a speed campaign is a
separate priority decision. Do not silently promote it because #72 is newer.
Similarly preserve the difference between full-frame VMAF evidence and cheap
PSNR exploration, and between offline canvas preparation and real-time decoding.

DEFERRED disposition (recheck exact current behavior before closing):

| Item | Destination / decision |
|---|---|
| D1 typing | infrastructure history; closed, do not reopen old error count |
| D2 SAM3 | generation/data action, previously deferred; verify current Torch/API need, isolated env only |
| D3 AVC ROI no-op | findings index and paper ROI provenance; closed scoped finding, not work |
| D4 SVD dependencies | generation action, previously deferred; preserve license/runtime boundary; reverify before any new availability claim |
| D5 all-off shortcut | codec history/action only if a current gap remains; distinguish raw passthrough diagnostic from coded fallback; do not restore obsolete no-branch requirement |
| D6 test pollution | infrastructure action to reproduce current moved tests; do not claim they still fail |
| D7/BP14 | generation: delivered training stop rule vs training not run |
| D-CODEC-PRESETS | evaluation: record exact presets, no unsupported cross-codec effort equivalence |
| D-PANORAMA-REOPEN | codec/evaluation: split already-wired panorama/stream work from still-open regime search |

Do the same extraction for RESEARCH-HISTORY §§3–8, not just its result narrative:
retain architecture and pairing invariants, component catalog links, evaluation
rules, unresolved representation/generation actions and required-behavior audit.
Old §7 numbering is historical; give live tasks new area IDs with old IDs as
search aliases. BP34 -> infrastructure; BP36/BP41 -> evaluation (data supplies
inputs); BP37 -> infrastructure; BP46 -> data; PAPER-NEXT -> paper; ENGINE-ROSTER
-> generation. The [inventory](documentation-audit/inventory.tsv) covers every
tracked Markdown file, including all 93 old archive entries.

## 6. One small session skill, not a chain of mandatory ceremonies

Create the project-owned `docs/workflow/session/SKILL.md`, linked explicitly from
AGENTS so every harness can read it. Use `skill-creator` when implementing it.
Do not write global platform configs from this repo or claim automatic discovery
by every harness. An optional native adapter can be added in the owning platform
session; the canonical procedure must remain readable without it.

Frontmatter name: `pointstream-session`. Description: route a PointStream task,
prepare a scoped dispatch, or report/close a completed work session. It should
not trigger an expensive routing discussion for every routine follow-up.
Use three short modes in one file:

1. **Route/dispatch:** extract requested outcome and ambiguity; read PLAN and one
   area; resolve facts locally/through relevant PRs; ask only materially blocking
   questions while progressing independent work. Respect the user's chosen
   harness/model. Recommend a different route only if it helps. Correct factual
   mistakes visibly, preserve intent, list assumptions separately. Do not silently
   rewrite scientific aims or turn uncertainty into an instruction.
2. **Execute/report:** reuse SESSION-REPORT's useful fields, with applicability.
   A documentation edit needs changed files, links, checks and unresolved items;
   it does not need artificial metric fields. Research reports need sample counts,
   failures, versions/config/source identity, rate/quality/encode/decode times,
   uncertainty, calibration/nulls, alarm status, reproduction paths and licensed
   conclusion. Distinguish all entries failed from successful batch exit.
3. **Close/continue:** apply the AGENTS completion rule, update the owning area and
   PR, check CI, assess cleanup and context boundary. Use existing end-of-session
   or handoff skills when invoked and available; do not duplicate their platform
   mechanics or introduce a new approval rule. Default continuation is a prompt
   in the response with durable state linked, not root HANDOFF.md. A saved prompt
   is warranted only when another queued session needs it; own it by area and
   retire it when consumed.

Routing preferences are **the user's working preferences**, not a benchmark or
an immutable statement about model size:

| Harness | Preferred use | Tradeoff / restriction |
|---|---|---|
| Codex | Deep analysis, large context integration, experiments/claims, final paper synthesis | Strong reasoning; scarce token budget: send a bounded question with relevant evidence |
| Claude | Deep technical analysis and large integration tasks | Similar token-budget concern; do not assign paper prose under the user's writing preference |
| Cursor | Coding and tasks with independent parallel subtasks | Relatively generous limits; needs clear interfaces/ownership; spend reasoning on ambiguous contracts |
| VS Code + Antigravity | Bounded execution, bulk edits, visual/multimodal inspection, plots and paper mechanics | User reports fastest model/largest limits, less depth; unresolved scientific judgments go back to a deeper session |

Choose among the actual models available in the selected harness. Small/fast
with low effort for mechanical edits; middle model/medium effort for bounded
implementation; strongest model/high effort for hard design, scientific judgment
or integration. Increase effort for ambiguity, not just file count. Do not
invent an effort setting or quote an account's quota from memory. A routing
prompt recommends a selection; it does not claim to change the active model.
[Official Codex model guidance](https://learn.chatgpt.com/docs/models) is a
reference for current Codex options, not evidence for cross-harness rankings.

Dispatch output: objective; area/action ID; base and evidence revisions;
allowed paths/branch/worktree; exact inputs and outputs; corrected assumptions;
acceptance checks; dependencies/stop conditions; pre-run bounds/nulls when
relevant; intended PR report; and recommendation with a one-sentence reason.
Split complex independent tasks into dependency waves only when worthwhile,
with separate worktrees and immediately shared coordination docs. Otherwise say
that the task is sequential/small. Do not make delegation a default token tax.

Validate the skill on three cases before declaring it useful: a routine typo fix
(no routing ceremony), a disputed codec win (scientific escalation with evidence),
and a completed docs task (area/PR update and safe cleanup check). Do not launch
real experiments, delete worktrees or change global model settings for that test.

## 7. History retrieval and safe document retirement

Seed `docs/history/pull-requests.md` from the attached
[pull request index](documentation-audit/pull-requests.md). Keep exact number,
title, state and merge commit; area tags are many-to-many when useful. Add a
small section for meaningful direct commits with no PR. Do not infer a PR from
similar commit messages, and do not label a closed-unmerged PR as delivered.
No agent should read all PR bodies at session start: search the area index or
`gh pr list --state closed --search '<topic>'`, then read relevant
`gh pr view <n> --comments` and the referenced files/diff. Check later retractions
before adopting an old result. The index is a locator, not a substitute for
current area state or an offline copy of every discussion.

Deletion gate, one row per retired file:

1. Pin the final file **blob/revision**, not just the PR that created it.
   The inventory supplies a verified baseline Git blob, working-content hash
   and full-commit recovery URL for all 116 files. The five edited files also
   need their user-edited state preserved; the patch snapshot supplies it here.
2. Extract current requirements, unresolved tasks and operative validity warnings
   into the named area or findings index. Record the destination action/anchor.
   A report's presence in a PR diff is recoverability, not proof its body is a
   redundant copy of the PR description.
3. Preserve the lookup `old path -> owner + immutable URL + validity status` in
   retired-documents.tsv. Mark completed/superseded/previously-deferred/retracted
   separately after review. Default inventory actions are migration instructions,
   **not** completed semantic-equivalence checks.
4. Repair active links in docs, code comments/docstrings and generated-report
   references. Prefer a live area link for current instructions and a commit
   permalink for historical evidence. Never rewrite old output JSON or claim a
   historical run used the new path/protocol. Do not change executable behavior
   just to replace a reference string.
5. Check paper-side citations before deleting their target. Migrate source-only
   provenance references to code-repo commit permalinks in a separate paper commit;
   preserve the facts inside markers. If the paper edit is not included, keep
   the cited historical file at its current path with a history-only banner and
   a manifest row until the cross-repo dependency closes. Do not retire it early.
6. Delete with ordinary tracked-file edits in one documentation-retirement PR.
   Recover using `git show <full-base-sha>:<old-path>` or the permalink. No branch
   deletion, worktree purge, force push or garbage collection is needed to remove
   tracked obsolete docs. Verify at least one full report can actually be read
   back from the pinned commit.

The short findings index must retain at least: invalid early LPIPS/self-image
roster results; the BP10 paste-certifying gate; synthetic vs real headroom;
withdrawn VVC-gap and cross-plate-subtraction conclusions; BP43's circular client
background; BP53 identity/timing limitations; source contamination/provisional
status; AVC QP ROI no-op; and #69 checkpoint-gap/BD-rate usability limits.
Link full evidence instead of copying numeric tables into active rules.

## 8. Implementation sequence and review boundaries

**Preflight:** read current AGENTS, PLAN, this brief and user-edits.json; fetch
current main/PRs and inspect the live dirty diff. Preserve and build on user
changes, applying saved hunks only where they are still missing. A saved patch
is not permission to reverse later changes. Work in an isolated branch/worktree
and explicitly configure external data only if a driven example needs it.
The main checkout must retain its user's edits unchanged until intentionally
integrated. Do not run any archived dispatch or the cleanup helper.

**Pass A — current state:** create six area documents and the short PLAN from
verified code/PR state; move gates into one roadmap. Resolve the conflicts in §2,
including observed vs proposed improvements. Record unresolved scientific choices
as action IDs; do not guess answers during a docs migration.

**Pass B — entry points/workflow:** write the human README/setup and concise
AGENTS; implement the single session skill. Keep adapters as pointers. Use
ordinary terms after retiring TERMINOLOGY; preserve behavioral warnings such as
unwired knobs only when current code still supports them.

**Pass C — history:** populate PR/findings/retired-file indexes. Process the
inventory by area. All archive files receive a disposition; unresolved unique
content gets an explicit retained-history exception, never silent deletion.

**Pass D — incoming references:** consume
[external-references.tsv](documentation-audit/external-references.tsv). Recheck
line numbers at the new revision. Repair code references and coordinate a
separate paper-repo commit for source-only cross-references. Paper AGENTS,
sections/README, appendices/README, companion/README and raw reviews remain
reachable through the paper area; archived figure/reviewer docs remain historical.
The eight paper Markdown entries are listed in
[paper-documents.tsv](documentation-audit/paper-documents.tsv).
Keep the paper's CLAUDE symlink. Do not rewrite manuscript claims or citations
as part of a link repair.

**Pass E — validation and retirement:** apply §9; retire the approved files,
including old plans/README and root DATA/HANDOFF; open one code documentation PR
that contains the final readable structure and its retirements. The code PR
must not depend on an unmerged document a worker cannot see. Merge coordination
changes promptly once checks pass; finish dependent paper-reference work before
removing its cited targets. Keep any cleanup-script fix in a separate code PR
with behavior tests; it is independently reversible and outside the doc edit.

**Pass F — finish:** PR body records what moved, corrected state, retained
exceptions, exact checks, paper dependency and immutable restoration recipe.
Update area status and clear the migration assignment. Check CI before merging.
Retire this one-use planning bundle with its recovery link once it has served its
purpose. Assess session handoff/cleanup under the actual permissions; do not
remove another session's worktree merely because its PR has merged.

## 9. Acceptance checklist

- [ ] Every tracked code-repo document appears in the final live graph from
  README or AGENTS, or in the explicit retired/history manifest. PLAN and area
  links are real Markdown links, not only backticked filenames.
- [ ] A human can find install, synthetic first check, supported real-input path,
  outputs and optional requirements without reading agent plans. Run documented
  commands where feasible; record skips/failures and do not claim an untested
  fresh installation. No environment changes to the pinned research env.
- [ ] A fresh agent needs AGENTS, PLAN and one area, with no mandatory README,
  history dump, root handoff, terminology doc or duplicate roadmap.
- [ ] Every area names a verified evidence/code revision, concrete next action,
  acceptance criterion and dependencies; proposed work is not marked delivered.
- [ ] #69 completion/alarms and #70–#72 proposal status are represented correctly;
  BP56/BP57 PR identities are correct; closed PR #5 is not counted as implemented.
- [ ] Every old deferred/open item has an area action, recorded supersession or
  closed finding. BP19/BP28 and RESEARCH-HISTORY §§3–8 have explicit coverage.
- [ ] Retraction/contamination/runtime warnings survive as searchable current
  validity constraints; historical evidence is restorable by full commit+path.
- [ ] Scan all tracked Markdown links for targets and fragments, then scan textual
  old-path/PLAN-section references in code and paper. Classify immutable history
  references separately; there are no new broken live references.
- [ ] Paper cross-links are repaired in its own commit or cited files are retained
  pending that dependency. Do not clear paper HOLE markers with this migration.
- [ ] No source/config/output behavior changes in the docs PR. If comment/string
  reference edits touch Python, run lint and the appropriate static checks;
  preserve meaningful explanation, not merely the old filename.
- [ ] Run `git diff --check`, `ruff check`,
  `mypy --config-file pyproject.toml`, `python -m src.contracts.layers`, and the
  synthetic tier checks used in README. Use checkout-specific local caches.
  No new tests for prose; validate links, restoreability and the skill cases.
- [ ] Watch the PR's actual CI result with `gh`; use failed logs if it fails.
  Check main/user edits and job ownership before claiming the session is closed.

## Audit coverage and limitations

Structural/content-marker inventory: all **116 tracked Markdown files** in the
code repo. Deep content review: both entry points, every active root/plan document,
archive index, research-history architecture/evaluation/backlog/verification,
flagged validity passages, relevant code and GitHub PR state. Historical metrics
were treated as records, not rerun or newly certified. The per-file retirement
pass is deliberately assigned above rather than falsely claiming that every
archived paragraph equals its PR body. Paper documentation boundaries and
incoming references were inspected; the paper was not rebuilt or scientifically
reaudited. The audit introduces no codec changes, no experiment runs and no
installed skill. It leaves the user's five original edits untouched and supplies
a snapshot so another session can preserve them.
