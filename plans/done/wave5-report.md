# Wave 5 — what landed, and what to check before merging

Written 2026-08-26 after Cursor (C/D/E) and Antigravity (F) reported. Claude's
own streams — **A (`BP24` encoder boundary)** and **B (`BP25` IP-Adapter
re-score)** — have **not started**.

Cursor's full report is appended to `plans/done/wave5-cursor-report.md`;
Antigravity's to `plans/done/wave5-antigravity-report.md`. This file records
only what an orchestrator needs, plus **three things neither report caught**.

## Status

| Stream | Brief | PR | CI (independently re-checked) |
|---|---|---|---|
| C — cull | `BP22` | [#25](https://github.com/emanuele-artioli/PointStream/pull/25) `549848b` | green |
| D — plumbing | `BP26` | [#26](https://github.com/emanuele-artioli/PointStream/pull/26) `ebb1204` | green |
| E — invariants | `BP27` | [#24](https://github.com/emanuele-artioli/PointStream/pull/24) `8a71fb2` | tests red — **pre-existing**, not a regression |
| F — paper | `P1` | paper repo `f32c1cf` | n/a |
| A — encoder | `BP24` | — | **not started** |
| B — re-score | `BP25` | — | **not started** |

**Headline results.** C took the pre-rewrite tests from **220 to 0**
(+158 / −12147 across 75 files) after deciding `src/shared/` **stays condemned**
(recorded at `plans/done/RESEARCH-HISTORY.md` §3), and ported ten test files rather than dropping them.
D classified all 27 inert fields — **25 genuinely unwired, 2 corner-inert** — and
wired the six ablation axes, each with both numbers. E pinned four metric
properties on **absolute scale, not ordering**, which is precisely the check that
would have caught the two metrics broken before 2026-08-23.

## Three things the reports did not catch

### 1. `main` is red, and has been since #22

Not a wave-5 regression, but it is the reason C, D and E all went red on their
first CI run. Lint and typecheck pass; the tests job fails on
`ffmpeg libvmaf failed` — the GitHub runner's ffmpeg has no `libvmaf`.
**Merging C or D turns `main` green**, because each adds a skip for the quality
rung when the filter is absent.

The skip is a defensible response to a runner limitation, but note what it buys
and what it costs: **VMAF is now never exercised in CI at all**, and VMAF is one
of the two metrics that were broken here until 2026-08-23. Worth revisiting with
a runner that has `libvmaf` rather than leaving it skipped forever.

### 2. E's invariants never run in CI

`tests/invariants/test_metric_calibration.py` carries
`pytestmark = pytest.mark.invariants`, and `pytest.ini`'s `addopts` deselects
`invariants` by default. CI runs `scripts/check_coverage_gate.py`, which calls
`coverage run -m pytest` and therefore inherits that deselection.

So the four properties BP27 pinned — the whole point of which was to stop these
findings being lost — **are not gating anything**. They pass locally and are
invisible to CI. Closing this needs a CI job that runs `-m invariants`, and that
job needs an ffmpeg **with** `libvmaf` or the VMAF properties will fail there for
the same reason as (1). **This is unfinished BP27 work, not a new brief.**

### 3. The paper and the invariants disagree about LPIPS

`sections/evaluation.tex:255` states LPIPS calibrates to
**0.000 / 0.250 / 0.430 / 0.645** at 4K. E's invariant pins
**0.000 / 0.0171 / 0.2982 / 0.5493** (`test_metric_calibration.py:243-245`).

Both may be internally right — they look like different anchor sets — but the
paper presents its numbers as *the* 4K calibration while the repo's invariants
assert different ones. **One of them will be wrong in print.** Reconcile before
submission, and say which anchor set each number came from.

The paper also says LPIPS "holds reliably at 4K" without carrying the more
consequential half of the finding: **its ordering inverts at 960×540**, so
anchors do not transfer across resolution. That omission is what would let a
future cross-resolution comparison go wrong.

## Merge order

`merge-tree` against `7cf8e89`: **C ∩ E and D ∩ E are empty; C ∩ D conflict** on
`plans/done/RESEARCH-HISTORY.md` and `tests/runner/test_tier_end_to_end.py`.

1. **E** (#24) — independent. Merge as-is; its red is pre-existing.
2. **C** (#25) — brings the `src/shared/` decision into `plans/done/RESEARCH-HISTORY.md` §3 and the
   libvmaf skip. Turns `main` green.
3. **D** (#26) — **rebase onto C first.** Keep both libvmaf skips (same intent)
   and D's new axis assertions; both `plans/done/RESEARCH-HISTORY.md` edits must survive — C's §3
   decision and D's §7 P0 item 4 rewrite.

Do not merge D before C without rebasing.

## One judgement call worth keeping visible

D's **pose** axis swap moved keypoints (L2 56.57 px) but left **PSNR identical
at 24.633**, below its own pre-written PSNR band of [0.05, 3.0]. D did not call
this an alarm, reasoning that "unwired" required identical keypoints *and* zero
PSNR delta, that keypoints demonstrably moved, and that a live residual can hide
pose error.

That reading is defensible and is recorded rather than buried — which is the
right handling. But the consequence is concrete: **an ablation table that quotes
only PSNR would show a row of zeros for the pose axis.** Whoever runs the lattice
needs a metric that is sensitive to pose, or that row will read as "pose does not
matter" when what it means is "PSNR cannot see this."

## Still blocked

- **P0 items 2 and 3** — no encoder binary runs. `BP24`, not started.
- **P0 item 5** — the IP-Adapter run remains `not_citable`. `BP25`, not started.
- **P0 item 4** — axes are wired now; the lattice itself is still un-run (Phase D),
  and codec / fallback / `residual.codec` stay unwired until `BP24`.
