# Handoff — PointStream wave 3 (BP21 / BP15 / BP19)

**Triggered by:** session length and a stalled Cursor host. Background
agent shells stopped returning; they are not GPU work. The user asked to
collect state and report back to Claude for the next steps. This session
is Cursor Grok 4.6, 2026-08-25 evening.

A pointer lives at `/home/itec/emanuele/pointstream-w3-c/HANDOFF.md` and
in `.agent-rules/var/precompact/f478a659-eb99-4069-a21c-b1f4af038fb7.md`.

## Summary

Wave 3 of PointStream: (A) widen the 4K headroom argument from n=2 to n=8
and settle the VVC QP confound; (B) cull dead decoder code; (C) make
IP-Adapter a real appearance path. Plan: `plans/BP21-headroom-widen.md`,
`plans/BP19-conditioning-architecture.md`. Do not edit the paper repo
`67a9ea6275d3d9785ce57026/`, `src/components/metrics/**`, `PLAN.md` §2, or
`plans/README.md`. Bound before believing. IdentityScale/TENNIS_SCALE live
in `src/components/metrics/reid.py` (do not edit).

## Current state (verified from git refs and files, 2026-08-25 ~19:25 local)

| Stream | Path | Branch | HEAD | Remote | PR |
|---|---|---|---|---|---|
| A BP21 | `/home/itec/emanuele/pointstream-w3-a` | `wave3/bp21-headroom` | `55e267b` | same SHA | [#18](https://github.com/emanuele-artioli/PointStream/pull/18) |
| B BP15 | `/home/itec/emanuele/pointstream-w3-b` | `wave3/bp15-cull` | `5905675` | — | [#19](https://github.com/emanuele-artioli/PointStream/pull/19) |
| C BP14/19 | `/home/itec/emanuele/pointstream-w3-c` | `wave3/bp14-bp19` | `c327feb` | same SHA | [#20](https://github.com/emanuele-artioli/PointStream/pull/20) |

Main checkout: `/home/itec/emanuele/pointstream`. `outputs/` and `assets/`
are gitignored; worktrees symlink into the main checkout. Do not commit them.
`conda` env: `pointstream`. `gh` as `emanuele-artioli`.

### A — BP21: done, with alarms; typecheck follow-up pushed

**Done and verified:** n=8 4K encode finished `2026-08-25T11:57:44Z`.
Report: `outputs/bp21-headroom/report.json`. Bounds written before the run:
`outputs/bp21-headroom/bounds-stream-a.json`. Write-up:
`plans/BP21-headroom-widen.md` § n=8.

Headline vs pre-written bands (do not cite as clean):

- AVC FG plate **0.170 ± 0.031** n=8 — **outside** [0.184, 0.304]
- HEVC 0.183 ± 0.034 — inside
- AV1 0.154 ± 0.028 — **outside** [0.169, 0.289]
- VVC 0.142 ± 0.026 — inside
- VVC gap, common QP: AVC−VVC **+0.028 ± 0.015** (n=8, 1.8σ, suggestive)
- Same gap, common PSNR (AVC/HEVC/VVC only): **+0.023 ± 0.017** (1.3σ)
- Sentence: *confound: the AVC−VVC FG gap did not survive a common QP set;
  it also did not survive a common PSNR interval.*

Two near-zero FG clips (paste-back MAE 0.0): `djokovic_zverev/scene_002`
0.011, `federer_djokovic/scene_003` 0.099. BP20's n=2 were the high-saving
clips; the ±0.06 band is too tight. Bound not retconned.

Common-PSNR window was empty because AV1 at QP 32/40/46 sits 5–10 dB above
the others. Fix: `common_quality_interval` raises on a disjoint range;
`_fill_common_interval` slices AVC/HEVC/VVC only. Refresh:
`python -u -m experiments.headroom.real_ladder --out outputs/bp21-headroom --summarize-only`

libvvenc 1.11.0 at `faster` wrote 0 frames at some clip×QP (including QP 32
on `djokovic_federer/scene_003` original). Fallback walks nearby QPs and
must not reuse another curve point (`qps=(32,32,46)`). That clip used QP
**31** in place of 32.

**CI on `840a6d7`:** lint green, tests green, **typecheck red** (run
32871918654). Local follow-up `55e267b` is on the branch *and* on
`origin/wave3/bp21-headroom` (mypy: `Sequence[Any]` instead of `list[Any]`;
typed test doubles). **Not verified:** whether CI on `55e267b` went green.
Check: `gh run list --branch wave3/bp21-headroom --limit 3`.

**Deliberately not done:** `PLAN.md` §2.14 not edited (central). Paper not
touched. Do not merge PRs unless asked. Do not rebase C onto B while both
PRs are open (would drag BP15 into #20).

### B — BP15: done

HEAD `5905675`, PR #19 mergeable. DecoderRenderer/compositor, synthesis_engine,
mask_codec, profiling, track_id, `benchmark_mask_codecs` culled. Coverage 77.
CI was green. Do not port or re-cull.

### C — BP14 done; multi-controlnet measured; IP-Adapter *wired*, not trained

BP14 `TaskStopRule` on coding-task LPIPS: `7e5103c`. Dataset honesty:
ip-adapter is not on the seg branch (`4aa7c94`). Loader bug for multi:
`c0e2744`. Multi-controlnet measurement (not citable): LPIPS object-bbox
0.579 ± 0.013 n=96; reid 0.628 ± 0.013 through TENNIS_SCALE. Two ControlNets
do not create an appearance path. Recorded in `plans/BP19-conditioning-architecture.md`.

**IP-Adapter training loop (`c327feb`, pushed):**
`--condition-type ip-adapter --include-reference` freezes stock OpenPose
ControlNet, attaches `h94/IP-Adapter` (`ip-adapter_sd15.bin` ~22M) on the
UNet, optimiser sees only adapter params (aborts if count not in 10–40M),
reference goes through CLIP vision not the control image, checkpoints write
`ip-adapter.bin`. Inference loads that file if present next to the ControlNet
dir. `--smoke-check-reference` is refused on this condition (pose-ref recipe).

Weights: `~/.cache/huggingface/hub/models--h94--IP-Adapter` (not under
`assets/weights/ip-adapter-controlnet` — that directory is a mislabelled
seg ControlNet).

**Bounds written before any train sample** (in the BP19 plan): after a
finished run, object-bbox LPIPS 0.50–0.78 (below 0.45 = paste alarm; above
0.74 = worse than unrelated). reid through TENNIS_SCALE 0.53–0.72; a
same-person 0.87 is an alarm. Expect semantic appearance, not identity.

**Not done:** a training run. A `--max-steps 1` smoke was launched
2026-08-25 ~16:30Z, produced **no logs**, and Cursor agent shells then
stopped returning. GPUs at 19:23 local showed **yiying `wan`**, ~6 GiB each,
0% util — not our process. Do not assume the smoke is still using a GPU;
check `nvidia-smi` process list. Killing leftover Cursor `tail`/`conda run`
shells in the IDE is safe.

Do not repeat pose-ref (reference painted into the control image). Uni-ControlNet
is last. Coding-task ControlNet retrain is after a real IP-Adapter result.

## What's running

Nothing of ours on the GPUs as of the user's `nvidia-smi` 2026-08-25 19:23.

The nine (or more) Cursor **background terminals** are leftover agent
commands (`gh run watch`, hung `conda run` smoke, hung `git commit`, hung
`ps`/`kill`). They are not encoding. Safe to close in the IDE. If a
`train_controlnet.py` python is still in `ps`, it had no GPU footprint;
killing it loses only an incomplete 1-step smoke under `/tmp/ip-adapter-smoke`.

Check:

```
nvidia-smi
ps -eo pid,etime,cmd | rg 'train_controlnet|real_ladder|recover.py' | rg -v rg
gh run list --branch wave3/bp21-headroom --limit 3
gh run list --branch wave3/bp14-bp19 --limit 3
```

## Open questions

1. **Cite BP21 how?** AVC/AV1 means are out-of-band because two clips are
   near zero. Keep the alarm and quote mean±SE + those clips, or retcon the
   ±0.06 band with an explicit reason? This session did not retcon.
2. **Start IP-Adapter training now?** GPUs had ~43 GiB free under yiying's
   idle 6 GiB jobs. Launch command is in the BP19 plan. First confirm CI on
   `c327feb` and that no stale smoke python is alive.
3. **Merge order:** A/B then C rebase after #19 lands. Do not merge unless
   asked.

## Next steps, in order

1. Close the stuck Cursor background terminals in the UI.
2. `gh run list --branch wave3/bp21-headroom --limit 3` — confirm typecheck
   on `55e267b`. If red, `gh run view <id> --log-failed`.
3. `gh run list --branch wave3/bp14-bp19 --limit 3` — confirm CI on `c327feb`.
4. If the user wants GPU work next: one-step smoke with logs to a file
   (`python -u ... >> outputs/ip-adapter-smoke.log 2>&1`), then detached
   train with BP14 stop:
   `scripts/train_controlnet.py --condition-type ip-adapter --include-reference --output-dir assets/weights/ip-adapter-trained`
5. Do not edit `PLAN.md` §2 until asked. Do not launch `--condition-type
   ip-adapter` without `--include-reference`.

## Landmarks

- A report: `outputs/bp21-headroom/report.json`
- A bounds: `outputs/bp21-headroom/bounds-stream-a.json`
- A write-up: `plans/BP21-headroom-widen.md`
- VVC fallback: `experiments/headroom/ladder.py` (`vvc_fallback_qps`)
- Common PSNR: `experiments/headroom/measure.py`, `experiments/headroom/real_ladder.py`
- IP-Adapter train: `scripts/train_controlnet.py` (`attach_ip_adapter`,
  `controlnet_cond_for_batch`)
- IP-Adapter load: `src/components/generation/controlnet.py`
- Tests: `conda run -n pointstream --no-capture-output python -m pytest tests/experiments/test_headroom.py tests/test_controlnet_dataset.py -q`
- Host rules: `/home/itec/emanuele/AGENTS.md`
- Skills: `results-report`, `end-of-session`, `evaluate-candidates` (9 open
  candidates under `.agent-rules/candidates/open`; 2 pending-verification
  for cursor)
