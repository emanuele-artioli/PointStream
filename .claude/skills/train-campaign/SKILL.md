---
name: train-campaign
description: Launch, monitor, resume, and diagnose POINTSTREAM generative training runs (train_controlnet/train_pix2pix/train_spade4tennis/train_campaign). Use for any multi-hour GPU training job, especially the G2 architecture campaign. Covers detached launch that survives SSH disconnects, resume after a drop, and the tripwires that stop a doomed run early.
---

# Running a training campaign without losing a night

The host is a shared GPU server reached over SSH, and **the connection drops at
least twice a day** (home ↔ office). A training run that dies with the shell
loses every epoch since its last checkpoint. The G2 campaign already burned a
night this way: SPADE OOM'd at launch, the harness silently pruned it, and the
"result" was an architecture verdict produced by a memory error.

Two rules follow, and they are not optional.

## Rule 1 — never launch attached

Always detach, so the job outlives the shell:

```
cd /home/itec/emanuele/pointstream
setsid nohup conda run -n pointstream --no-capture-output \
  python -m scripts.train_campaign --campaign-dir outputs/campaign/<name> \
  --data-root assets/probe_set/training_view \
  --manifest assets/probe_set/manifest.json \
  > outputs/campaign/<name>/launch.log 2>&1 < /dev/null &
```

`setsid` detaches from the controlling terminal, `< /dev/null` stops it blocking
on stdin, and `--no-capture-output` matters because `conda run` otherwise buffers
everything until exit — on a multi-hour job that means no progress at all, then a
truncated dump.

From a Claude session use `run_in_background: true` instead; the harness
re-invokes on exit and reports the output path. **Never** poll with a
hand-written `until ! pgrep …` loop — see the global CLAUDE.md for why that loop
can never terminate.

## Rule 2 — verify it can resume before it needs to

Before a long launch, confirm the resume path actually works: start the run,
kill it after one checkpoint, restart with the same command, and check it picks
up rather than starting from zero. An untested resume is not a resume.

- `train_pix2pix.py` / `train_spade4tennis.py`: `--resume` + `--checkpoint-path`,
  and they track cumulative epochs, so the campaign passes an absolute target.
- `train_controlnet.py`: saves `checkpoint-epoch-N/` per epoch; resume by
  passing the latest as `--controlnet-model-id`. It trains `--epochs` *more*
  epochs on top, so the campaign computes `rung_epochs = target - previous`.
- `train_campaign.py`: `campaign_state.json` is the resume mechanism — edit it
  to repair state, don't delete the campaign dir. A `state.json` beside it is a
  stale leftover from an aborted attempt; ignore it.

Checkpoints are written **per epoch**. When an epoch is ~40 min that is a
lot to lose — prefer a step-interval checkpoint for anything longer.

## Tripwires — stop a doomed run in minutes, not overnight

Check these before letting a run continue past its first probe. Each one has
already cost real time here.

| Symptom | Meaning | Action |
|---|---|---|
| Training exits nonzero (OOM, CUDA error) | **Not a quality result.** | Retry once at half batch; if it fails again, mark the rung incomplete and exit *without ranking* (this is `train_campaign`'s fixed behaviour; the old pruning lives behind `--prune-on-train-failure`) |
| Loss NaN / inf | Divergence | Kill. Lower LR; do not "wait and see" |
| Score below the **reference-copy floor** after a few epochs | The model is worse than pasting the reference crop | Kill and diagnose — it is not learning the task |
| Output variance collapses (near-constant frames) | Mode collapse / dead generator | Kill |
| VMAF exactly 0.00 on actor crops | Metric floor saturation, **not** a model signal | Ignore VMAF at crop scale; use LPIPS/DISTS |
| A variant scores ~0 while others are sane | Suspect the *measurement* first | Check it runs the decoder's path (`build_eval_strategy`), not a reimplementation |

That last row is the expensive lesson: `controlnet_pose` scored PSNR 9.76 for
days because eval ran text-to-image from noise while the decoder ran img2img
from the reference. Fixing only the measurement moved it to 16.06.

## Before any launch

1. `ruff check src tests scripts`, `mypy`, and the fast pytest suite.
2. **Commit.** Other sessions share this working directory and can clobber
   uncommitted edits; the run must validate committed code.
3. Confirm the held-out gate: `--data-root assets/probe_set/training_view`
   contains only the 5 training videos, and
   `verify_data_root_excludes_probe_set` guards the split. No generative quality
   claim is citable without this.
4. Pick the GPU deliberately — `nvidia-smi` first. The 48 GB card is **shared**;
   SPADE OOMs at batch 16 / 512 px, batch 4 is the documented fallback.
5. Pin `seed`, image size, and inference-step count so runs are comparable.

## Reading results

Rank by **residual bytes**, not by crop VMAF — see `RESEARCH_LOG.md`. Perceptual
metrics on crops (LPIPS/DISTS) are diagnostics that explain *why* a model wins;
PSNR/SSIM are corroborators; VMAF and FVD belong on the final full frame, not on
actor crops. Fold conclusions in with `/update-paper`.
