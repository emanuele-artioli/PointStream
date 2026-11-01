> Latest operational closeout: CPU source staging and full-archive transport diagnostic are terminal; no matching remote diagnostic remains. Original cumulative≤1654CPU-s/224GPU-s; no live compute reservation. Exact-source delta recovery is being prepared, not executed. See latest W8 card for all historical charges and stop receipts.


# Continuation, 2026-11-01: exclusive W8 coverage repair completed

## Current decision

The user asked whether the larger saved bitstream meant PointStream costs more.
Root clarified that the saved 49.6733-kbps stream belongs to the **baseline**:
its score would bracket the fixed PointStream rate, not add bytes to PointStream.
For a standalone 4.8-s clip PointStream's assembled cold payload really is much
larger (3,816.4 kbps); 46.3151 kbps assumes exact match/reference reuse over
7,200 s, not demonstrated reuse quality. Whole-frame results favor the baseline;
the positive player-region claim is conditional.

The approved exclusive scoring request **`20261101T081828Z-d70bc44f`** is
complete on gpu6 Ada, independently full-audited, with all eight strict rate
cells passing at unchanged 46.315062222222224 kbps. This opens the original
CPU input/isolation gate; it does not prove fresh decoding or two-hour reuse.
Scorer source `c0f25e4e`, normalized spec
`e9d926b6f61685a288e90225c38bd1d84050c9187dfa25180d298ce317b03230`.
Full archive `f87e9cba68063fb115828dddbffb9b7fbaffa1e61003ac4292f4adfaab1cc41f`;
durable full audit and raw screen are in `docs/experiments/w8-inputs/exclusive-20261101/`.
All three scoring requests, waits and task claims are closed; never replay them.

The corrected Linux CPU input oracle **`20261101T093351Z-24914a4d`**
completed from frozen source `e8f8827b` and original MAIN environment `32e688`;
all four independent input, golden/exact-match, isolation and payload checks pass
for one profile. It covers the full 240-frame context and 80 judged ticks, not
fresh crops, masks, native RGB, composites, reuse or timing. The independent report
SHA256 is `8047b67694dd64171d2c0d13a9eaff53b18cd833b8fdc45ba054185c382848cc`;
[audit](experiments/w8-inputs/linux-input-20261101/bootstrap-independent-full-audit.json)
and [closeout](experiments/w8-inputs/linux-input-20261101/bootstrap-closeout.json)
record details. The job, process group and CPU claim are closed; wait `70480`
ended and event `b2ef4c7dacbf5f0d93f9d76e0cd37d7d` was delivered and acknowledged.
Do not replay it. Its conservative charge is 101 CPU-s (61 campaign plus 40
extras). Golden qualification closed with a conservative 150 CPU-s charge;
15 reference files are sealed read-only. The later bootstrap stopped at its cost
projection (60+5 CPU-s); the first native inventory failed on transient loader-cache
access (20 CPU-s). The corrected native inventory failed before decoder launch,
closed at ≤4 CPU-s. Source staging for utility `61563` stopped after identity-guarded
termination of only task-owned processes; handle is terminal 137 with SSH -9,
no source archive, asset/runtime/inference or quality result. Its closed charge is
30 CPU-s inclusive of diagnostics and stops, replacing the unused 165 reservation.
A separate tiny 28-byte SSH/EOF smoke passed and closed at 4 CPU-s. Thus W8 closed
cumulative is **≤1,654/3,600 CPU-s and 224/3,600 GPU-s** after the full-archive
transport diagnostic73336 timed out (4 CPU-s) and its bounded closeout (5 CPU-s)
found no matching process. No live reservation remains; exact-source delta recovery
is under review, not executed. No fresh GPU job or receiver qualification
exists; these stops are not quality negatives. See [W8](experiments/w8-wire-audit.md)
for immutable identities and closeout receipts; inspect existing work, never replay.

## Immutable scientific state

- Fixed candidate: 2,289,836 B, SHA256
  `eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4`;
  conditional 46.315062222222224 kbps. Original `s2_o20` upper quality point
  ended at 45.82333333333334 kbps; the accepted 49.67333333333334 kbps repair
  now brackets it. The single-window four-family screen passes both metrics.
- Selected new baseline: nad000, scale 2, outside +20, CRF 67,
  29,804 B / 49.67333333333334 kbps, stream SHA256
  `182c7eee4a0ea2e98ef0a967ecfcdf8958898ad7ce035e54f0ea91b5109b6fa3`;
  original encoder archive `20261031T181055Z-e4dc2d25`, SHA256
  `9736920b256b2b770c3b35e8784570a50df2720115fbb134970439adea4240f0`.
  It now has an independently audited full 80-tick quality score.
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

3. **`20261101T081828Z-d70bc44f` — complete, full-audited, gpu6 exclusive.**
   Source/spec above, full 105.5921 s, projected 173.8745 s <195 s allowance.
   All 12 full checks pass; maximum control discrepancy 5.227327e-5 <0.001.
   Audit code `142b6cc5` corrects frozen extra-argument positional binding;
   initial 0.129421 CPU-s audit failure is preserved, fresh successful audit
   used 0.140220 CPU-s, no score/spec/tolerance change. Native wait 67073 ended;
   terminal event `48e93fc725d7528a91c612b39a44ecee` acknowledged.

All recorded supervisors/process groups are absent; relevant task claims empty,
verified read-only. No process was signaled. Failed host-local scratch is retained;
automatic salvage was explicitly deferred to prevent unbudgeted copying.
Never replay either request or delete their artifacts.

The original 300/300 amendment closed after 67 GPU-s / ≤87 CPU-s. Subsequent
user-approved inclusive 450/450 scoring closeout is **224 GPU-s / ≤304 CPU-s**:
prior 67 + 157 for the request, plus 30 for monitoring + 10 for audit + 40 for all preparation (includes
prior 20). This is inside original 3,600/3,600, not extra budget. Original local
≤850 CPU-s + 304 for scoring + 15 for runtime preflight = **≤1,169 CPU-s**; 224 GPU-s at that historical checkpoint. Later CPU-oracle accounting is above.
No scoring reservation remains. No balance transfers to other cards.

## Code, fleet and manuscript

PR #318 merged `538b1022`, PR #319 merged `1b282587`, PR #320 merged `02c6c10e`.
PR #321 merged `b014067c`; exclusive-policy PR #322 merged `f5597299` after
CI passed on exact `c0f25e4e`. Current branch is
`codex/w8-receiver-witness-2026-11-01`, draft PR #325. CI passed for `5651`, run `41301707915`. The accepted input gate
is described above; fresh GPU generation remains unqualified. Frozen source
`5651a24b` full pytest had the same 13 failing node names as the saved clean-main
Mac run, with 1,407 passed and 32 skipped; its 42 receiver tests passed. Frozen
Ruff passed and mypy reported the seven known Mac baseline errors.

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

Overleaf main and the local manuscript are **`525f423bca542148645890063474e4a85723c6cf`**,
pushed. It links both the initial `21931f6` recovery dossier and subsequent
controls at `5a6f892b`; it retains the qualified claims and evidence boundaries.
The review format has 22 body/reference plus 3 appendix pages; an isolated
small-format check has 22+3 after bibliography corrections, without clipped paths. Live venue format instructions
remain unverified. The paper is not submission-ready.
Negative results and conditional saved-quality boundaries remain.
No recurring timer or cache keepalive remains; the bounded transport-only diagnostic is active as recorded above.
Cache lifetime is unknown.
