# Continuation, 2026-11-01: W8 coverage repair stopped at cost gates

## Current decision

The user asked whether the larger saved bitstream meant PointStream costs more.
Root clarified that the saved 49.6733-kbps stream belongs to the **baseline**:
its score would bracket the fixed PointStream rate, not add bytes to PointStream.
For a standalone 4.8-s clip PointStream's assembled cold payload really is much
larger (3,816.4 kbps); 46.3151 kbps assumes exact match/reference reuse over
7,200 s, not demonstrated reuse quality. Whole-frame results favor the baseline;
the positive player-region claim is conditional.

The user conditionally approved the 300 GPU-s / 300 CPU-s internal W8 amendment;
root proceeded after the clarification. Two requests stopped before full scoring. Root
has now asked for one **exclusive-GPU** attempt with an internal inclusive cap
of **450 GPU-s and 450 CPU-s**, retaining the original W8 **3,600/3,600 totals**.
That question is **pending**, not approved. The concrete
[proposal](experiments/w8-inputs/amendment-20261101/exclusive-prospective-card.md)
and expired, non-runnable spec preserve the original point, controls and rule.
No new encoding, training, model/payload change or original-total increase.
Do not run a third request without authority and a reviewed source/policy update.

## Immutable scientific state

- Fixed candidate: 2,289,836 B, SHA256
  `eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4`;
  conditional 46.315062222222224 kbps. Original `s2_o20` upper quality point
  ends at 45.82333333333334 kbps. The full four-family screen is unresolved.
- Selected new baseline: nad000, scale 2, outside +20, CRF 67,
  29,804 B / 49.67333333333334 kbps, stream SHA256
  `182c7eee4a0ea2e98ef0a967ecfcdf8958898ad7ce035e54f0ea91b5109b6fa3`;
  original encoder archive `20261031T181055Z-e4dc2d25`, SHA256
  `9736920b256b2b770c3b35e8784570a50df2720115fbb134970439adea4240f0`.
  It is verified as bytes but has no accepted full 80-tick quality score.
- Original three CRF-63 controls, 80 full ticks and four per-tick metrics,
  all raw-control differences ≤0.001, exact source/native/environment identities,
  independent full-output audit, no extrapolation. Strict pass on both player
  metrics against all four families at the fixed candidate rate is required.
- The W7/W9 repaired nine-window results remain unchanged. Familiar-match Tier 1
  passes conditionally; unseen-match Tier 2 does not pass at two modeled hours;
  all whole frames lose. Joint duration sensitivity is conditional about
  133–198 min, not a global minimum. Original Med omissions remain.
- Exact input geometry on Mac failed at ~4.55e-13; source-hidden reconstruction,
  original packed input oracle and output reproduction remain unproved.
  A passing rate screen alone may only reopen those original gates.

## Jobs, outcomes and budgets

1. **`20261101T071706Z-afde26d5` — failed cost promotion, gpu6.**
   Source `a5fdb23536c646851a83b226789ee3e68ccc50a9`, normalized spec SHA
   `781dc0ba6fef4db759f7b1d24a755b452fcf312e44cc40ee668b965fde20c840`.
   Scorer exited zero in 15.9436 s; all twelve correctness checks passed, but
   original W1 projected full 170.7294 s exceeded the 160-s allowance.
   No full stage. Supervisor 32.1922 s; charge **33 GPU-s / 33 CPU-s**.
   Native wait handle 31915 ended; events acknowledged.
2. **`20261101T073429Z-04a50c10` — failed smoke time limit, gpu5.**
   Source `67291ddbb57efa11b8d957539e84832c5111f44a`, normalized spec SHA
   `9ae2ee29080ee5115a3abcadd471c019a515229ffd8a438f214ed8cecc0164f7`.
   Fresh smoke hit its 20-s limit; shared contention recorded. No full or
   accepted smoke quality point. Supervisor 33.6719 s; charge
   **34 GPU-s / 34 CPU-s**. Native wait 96739 ended; events acknowledged.

Both recorded supervisors/process groups are absent; relevant task claims empty,
verified read-only. No process was signaled. Failed host-local scratch is retained;
automatic salvage was explicitly deferred to prevent unbudgeted copying.
Never replay either request or delete their artifacts.

Closed amendment: **67 / 300 GPU-s; ≤87 / 300 CPU-s** including a conservative
20 CPU-s preparation/publication charge. No reservation remains. Cumulative W8:
**67 / 3,600 GPU-s; ≤937 / 3,600 CPU-s**. Remaining original headroom is
3,533 GPU-s and at least 2,663 CPU-s; it does not override the internal stop rule.

## Code, fleet and manuscript

PR #318 merged `538b1022`, PR #319 merged `1b282587`, PR #320 merged `02c6c10e`.
PR #321 (cost reallocation) passed CI on exact `67291ddb`, run 41285565801;
final result/continuation documentation is being published on
`codex/w8-cost-reallocation-2026-11-01`. Inspect current PR/head before claiming
the final merged state. Focused tests/Ruff pass; seven unrelated Mac mypy errors
remain. Earlier full pytest has the same thirteen Mac baseline failures.

Manager release `20261101T071313Z-74a9271d` runs commit `a5fdb235` on gpu2/5/6.
Verified bundle SHA256
`14339c1ad228814d2d7857a8f944639a27b78b907fb31f9da70d4d8b8e1431a9`.
It supports one-CPU campaign opt-in, monotonic dispatch clocks and deferred failed
publication. These are cooperative operational guards, not immutable security
containment. GPU3's old attention request `20261024T082408Z-3cb3f467` is unrelated,
unresolved and untouched. Other workers/supervisors were preserved.

The new concurrent status reader completed full snapshots (1,544 then 1,545
records); a read is over time, not proof all processes globally stopped.
Refresh queue/host preflight before any future dispatch. Packed environment
`32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`
and all three native binaries were checked; all scientific imports came from
clean frozen source. Source/input artifacts and diagnostics are referenced in
[W8](experiments/w8-wire-audit.md) and its amendment input directory.

Overleaf main remains **`e366440`**, local
`/private/tmp/pointstream-w7-manuscript`, tracking `overleaf/main`; original server
checkout is untouched. Reviewed PDF `build/main.pdf`: **22 pages body/references
and 3 online appendix pages**, within user's 23+5. Negative hand/object representation
results and mixed-metric ego adaptation are included; no cost stop is presented
as a quality negative. [Readiness](paper/readiness-2026-11-01.md) and the evidence
matrix distinguish other submission gaps from deferred PLAN research. The paper
is not submission-ready. No recurring timer/cache keepalive or active task-owned
experiment remains. Cache lifetime is unknown.
