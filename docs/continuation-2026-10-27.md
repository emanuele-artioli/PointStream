# PointStream continuation after root-chat archive — 2026-10-27

Resumed root update: CPU-only allocation proposal is now recorded in
[the cost revision](cards/k6b-real-base-cost-revision.md), with its state in
PLAN.md and the appended experiment entry. It is not accepted for execution;
the original terminal research state and580/3600 conservative ledger below
are unchanged. Canonical status timed out through gpu1; read-only shared-inbox
inspection through gpu2 then confirmed all three terminal states. gpu2/gpu3
heartbeat files were about16min old at11:59 UTC; current worker health remains
unverified. Do not replay any request.

## Goal and current decision

Seek a selective semantic-coding win that can support a paper on players and
ball, followed by extension beyond these development windows. A ball-only
contribution is insufficient (user). No complete, fully delivered PointStream
win over both SVT-AV1 and the required neural baseline is established.

The next task is CPU-only planning of a feasible continuation of the existing
real-background player study. Do not start a new generator, fitting, full240
rendering, ball integration or another baseline experiment as a substitute.

The governing card's goal is:

> Goal: test whether the frozen PR242 lossy-reference generation and PR247
> per-frame SAM players improve a REAL decoded whole-frame SVT base at equal
> TOTAL rate on the same Al031@129 and Med014@194 development windows.

Read `AGENTS.md`, `PLAN.md`, `docs/cards/k6b-real-base-quality-pilot.md`, and
only the relevant end sections of `docs/experiments/k6b-players.md` (check file
length first). The experiment's final section is titled "2026-10-27 — Real-base
quality smoke stopped at the cost gate". Earlier "running" entries record
history; this terminal section supersedes them. PR #252 merged its report and
worker; PR #253 merged storage safeguards. This continuation is maintained by
the final root handover PR linked in PLAN's ledger.

## What has been learned

- Egocentric split gains disappear once masks and fair adaptation are charged.
  Hand generators failed transfer at the keypoint rate. Tennis pixel splitting
  also failed its tested oracle/fully billed configuration.
- The tennis ball streak component improves ball detection at very low rate,
  with its documented whole-frame quality tradeoff; this does not establish
  the desired broad paper contribution.
- Earlier player fit variants failed their frozen six-window rules. A later
  12-window familiar-broadcast diagnostic reached 23/24 metric wins with
  references and exact mattes granted. Actual side-information accounting then
  removed the apparent complete win. These repeatedly used windows are now
  development data, not fresh confirmation.
- Video-compressed **unmasked full-RGB references** retained actual pilot
  generation quality (4/4 strict metric cells). Alpha/black references, dense
  carriers, and the tested paid coarse mattes failed. See PRs #242–243.
- Receiver silhouettes: YOLOE 1/4; SAM tracking 2/4; independent-frame
  pose-prompted SAM **4/4** strict, sensitivity +0.001 **3/4**, zero empty masks
  across 320 player-mask predictions. Independent receiver-read isolation and
  reset checks passed. This is the promising component (PR #247), with no
  transmitted matte bits, but its background was target-derived Telea and
  granted. It is not a delivered-background win.
- Native source RGB conversion is now reproduced exactly on A6000; CPU/GPU
  rounding differed by at most one 8-bit level. Do not relax source identity
  checks or reuse CPU arrays as if they were byte-identical GPU controls.

## Frozen scope and stop conditions

Two windows Al031@129 and Med014@194; **80 saved judged ticks/window out of240**.
Actual native whole-frame SVT base scales8/4/2; **one global scale4 central
operating point**, with scale8/2 completing the limited curve. Same exact
source, R/S evaluation masks, crop/reference order, conditioning and tick
membership. Original base has240 frames; missing160 generated ticks are not
silently interpolated or claimed delivered. No new generation, model, fitting,
reference, ball inference, DCVC reproduction or ROI comparison in this budget.

The card requires >=3/4 strict LPIPS/DISTS full-frame cells AND >=3/4 fixed
foreground cells at equal total rate, with frozen ±0.001 sensitivity. When
quality ranges overlap, additional per-metric BD-rate uses PCHIP and the
achievable anchor envelope, mean <=−10% and >=75% window wins. No extrapolated
BD-rate, no per-window quality-selected settings. The existing early-stop rule
and original controls remain unchanged; read the card and worker for detail.

Original source/granted-matte/predicted-mask controls precede new quality.
Stop on failed identities, controls, resource or decision gates, exhausted
budget, uncertain submission, or a required scope/budget decision. Every waited
stage is <=45 min. A failed projection is not permission to raise a reservation
and continue silently. The next plan must be prospective and separately
accepted; no new GPU job was authorized at this handover.

## Jobs, code, results and budget

Re-read IDs here and run canonical `scripts/ps-fleet status JOB_ID` before any
job action. Inspect an uncertain request; never replay it. The terminal quality
state was rechecked at closure. The two recovery states are also checked for
this handover and their result is recorded in the closure note below.

| Job | State/outcome | Code/budget |
|---|---|---|
| `20261027T072713Z-2888cefe` | COMPLETE, CPU native-base recovery, six240-frame streams and retained YUV | 184.54109/1800 workload CPU-s, no GPU; audit37.743262 CPU-s separate |
| `20261027T074823Z-135421ee` | COMPLETE, GPU conversion; 80/80 original source hashes per window; six80-frame base RGB sets independently repeated/verified | frozen diagnostic worker `146f06ee`; admitted282.821585s, conservatively370 GPU-s |
| `20261027T080644Z-9dba0803` | FAILED **cost projection only**; both window controls pass on the predeclared two smoke ticks/window; full never launched | worker `8bd01ff5703ebf5b1cf41feef4f564f948cf44b2`; admitted151.469499s, conservatively210 GPU-s |

**Current study total: 580/3600 GPU-s conservative (0.161111/1.0 GPU-hour),
434.291084 actually admitted; nominal3020s remain.** All setup, smoke, failure,
validation and publication costs stay inside the same cap. Previous study
budgets are closed. No accepted full quality score, win/loss, or BD-rate from
the latest real-base run exists.

The full worker projected **2227.523141s**, **2448.314646s including
publication**, exceeding its frozen1800s full-stage reservation. Other smoke
checks passed. This is a computational feasibility stop, not a quality failure.

Immutable evidence lives under
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/`:

- CPU recovery: `20261027T072713Z-2888cefe/full/published.tar`,
  SHA256 `00b13e89077a27a3e6ff23bf476e9795c2120c5b1f5252ddfdd6f533a8119a39`.
- GPU conversion: `20261027T074823Z-135421ee/full/published.tar`,
  SHA256 `5c4a0cfced5501e085f67d394f3a50ca7f92d2cb592f73fbd3ad661461cb691b`.
- Exact smoke validation is versioned at
  `docs/cards/k6b-real-base-quality-smoke-validation.json`,
  SHA256 `fd5509a633ae0a4dccdc944d66b4f66113f3c8a2900c463630e92ed3b7455085`.
  The experiment/card records input, spec, environment and patch identities.

## Next actions, in order

1. Reconcile current job/worker health with these terminal records. Preserve all
   existing evidence and inspect unknown jobs rather than duplicating them.
2. On CPU, decide whether a computational redesign or a revised bounded stage
   allocation can fit the unchanged full test, controls, new smoke, publication
   and overhead inside **3020 remaining seconds**. The prior2448s estimate alone
   is not a validated complete quote; shared-host timing is not a speed claim.
   Record the proposed revision and acceptance before execution. Do not tune
   cohort, central scale, metrics or quality thresholds to make it fit.
3. If no defensible plan fits, stop and explicitly decide whether more compute
   is worth buying with a new user-approved budget. If a revised plan is
   accepted and its smoke passes, finish the same real-base study and stop at
   its predeclared decision. No scientific quality conclusion is implied now.
4. Only if delivered-background feasibility survives: serialize and independently
   decode the missing adapter tensor/bootstrap directory, close all delivery
   accounting, add qualified DCVC and fair ROI/pixel-residual controls with
   separately approved budgets, then integrate the paid ball component on the
   same windows. Freeze the method before acquiring untouched evaluation data
   and auditing train/reference/judge lineage and original source/PTS/replays.

Rates in the current study are explicitly **optimistic floors**: native IVF,
KPS and actual reference/update payload bytes are known, but required adapter
serialization/bootstrap/framing is incomplete. A negative result can reject
the tested configuration even at that floor; a positive remains conditional.
Installed common tennis/SAM models must be declared and installation sensitivity
reported. Three SVT points do not establish superiority over DCVC or stronger
ROI/pixel alternatives. All12 windows remain development; the115px cutoff is
exploratory, and historical step5 failure is not erased.

## Storage, workflow and coordination

Use `/local/users/emanuele` for host-local runtime, environments and scratch;
never large `/tmp` or `/var/tmp` files. It avoids NFS small-file latency while
using a separate disk. Shared NFS remains durable source of truth, with packed
inputs/environments staged through the fleet.

Read `docs/storage-maintenance-2026-10-27.md` before storage or worker changes.
Inactive files on all six hosts were copied/checksum verified; originals and
old paths remain accessible through compatibility symlinks. Remaining real
root-disk trees have active process references. They must not be removed to
"finish cleanup". Relocation journals and hidden live originals are application
state, not all disposable cache. Fresh-shell defaults are installed/verified;
manager safeguard rollout remains blocked before any worker change by the
existing20s environment-probe gate on gpu2/gpu3 (bounded diagnostic: gpu2 valid
in38.84s; gpu3 >60s). Do not bypass the health gate or infer that reachable
hosts are unavailable. gpu5/gpu6 live public-directory switching was rejected
by automatic approval review; no retry bypass occurred. Finish migration only
when safely idle or with a specifically reviewed live-migration plan.

Existing study chat: `01a16e7a-e752-78f3-b372-c338d8cdbf46`, title
"Prepare tennis confirmation study", local checkout
`/Users/manu/.codex/worktrees/7e8e/PointStream`. It is idle, not archived by this
request. The user has authorized direct coordination with that study chat.
Do not start a turn there merely to read status. The user prefers paced active
sessions; no new chat was created as part of this closure.

The root chat being archived is `01a16cfc-1a88-7013-bd79-8489fe7c1eb2`.
No new research job, recurring LLM heartbeat, keepalive or pending overnight
execution was created. Prior overnight work did not run because unattended
file/process/MCP isolation was not verified. Attend future execution unless
that gate is actually satisfied. Cache lifetime is unknown; do not promise KV
retention. Preserve this record and existing stopped work when starting a new
root chat from current main.

## Closure verification

Canonical fleet status rechecked all three IDs at root-chat closure: CPU recovery
COMPLETE, GPU conversion COMPLETE, real-base quality FAILED at its smoke gate.
The full stage remains unlaunched. All scientific jobs listed here are terminal.
Storage PR #253 merged with Linux CI passing. The final documentation is in PR #254,
linked in PLAN.md; no future session needs the archived conversation to resume.
