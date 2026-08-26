# Prompt for Cursor — BP25, IP-Adapter re-score

Paste everything below the line. One worktree, one stream.

```bash
git worktree add -b wave5/bp25-rescore /home/itec/emanuele/pointstream-w5-b origin/main
cd /home/itec/emanuele/pointstream-w5-b
mkdir -p assets && for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -sfn /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -sfn /home/itec/emanuele/pointstream/outputs outputs
```

---

You are running **BP25** on PointStream, an object-centric semantic video codec
targeting ACM TOMM on **30 September**.

**Read first, in order:** `/home/itec/emanuele/.agent-rules/AGENTS.md` —
especially *"control the instrument, then the result"* — then this worktree's
`AGENTS.md`, then `plans/BP25-ip-adapter-rescore.md`, which is your brief, then
`PLAN.md` §2.10 and §2.12.

**You own:** the *eval path* of `scripts/train_controlnet.py`,
`src/shared/training/task_eval.py`, `outputs/bp25-ip-adapter/**`,
`plans/BP19-conditioning-architecture.md`.
**You must not touch:** `src/runner/**`, `src/pipeline/**`,
`src/contracts/lattice.py`, `config/tier_*.yaml` — another stream (BP24) owns
those this wave and is live in them right now. If you need a change there,
**report it, do not make it.**

## The situation

An IP-Adapter training run finished 2026-08-25 and self-stopped at epoch 3. The
**stop rule worked exactly as designed** — that is not in question and is not
your subject. What is in question is the number it stopped on:

| | LPIPS |
|---|---|
| static-copy floor | 0.5269 |
| unrelated null | 0.7497 |
| stock untrained IP-Adapter | 0.7606 |
| best checkpoint (epoch 1) | 0.8112 |
| final (epoch 3) | 0.8281 |

Every value is **above the unrelated null**, which the pre-written bound calls an
alarm. Two defects explain why it cannot be read as a verdict:

1. **The eval generates at 4 diffusion steps** (`STOP_EVAL_STEPS = 4`, into
   `ControlNetGenerator(steps=4)`). Vanilla SD1.5 needs 20–50. Worse, the two
   anchors it is scored against — the static-copy floor and the unrelated null —
   are **real images that never pass through diffusion at all.** A barely
   denoised generation is being compared against undegraded photographs.
2. **Most evals scored stale weights.** 11 evals produced **5 distinct values**.
   `_run_task_eval` writes `checkpoint-epoch-N` only if that directory does not
   already exist, so every mid-epoch eval after the first re-scores frozen
   weights.

Both `bounds.json` and `stop_series.json` already carry `"not_citable": true`.

## Your task, in this order

1. **Calibrate the instrument before re-scoring anything.** Score a known-good
   generation at 4 steps and at 30 steps. Answer in writing: *can the 4-step eval
   distinguish a good generation from a bad one at all?* If it cannot, that is
   itself the finding — the tripwire is fit for stopping runs, never for ranking
   models.
2. **Fix the anchoring.** If the candidate goes through diffusion, the floor must
   too, or you are running a step-count comparison dressed as a model comparison.
   State which anchoring you chose and why.
3. **Re-score the three saved checkpoints** — `checkpoint-epoch-1/2/3` under
   `assets/weights/ip-adapter-trained/` — at 20–30 steps, **with n and standard
   error**. n=4 clips is too thin for a verdict; widen it.
4. **Fix the stale-checkpoint bug.**
5. **Only then** state whether IP-Adapter uses appearance.

## Bounds — already written, do not rewrite

From `plans/BP19-conditioning-architecture.md`: object-bbox LPIPS expected
**0.50–0.78** (pose 0.60, paste 0.45, unrelated 0.74); below 0.45 is an alarm for
paste-through, above 0.74 an alarm for worse-than-unrelated. `reid` through
`TENNIS_SCALE` expected **0.53–0.72**; a score at the same-person anchor (0.8663)
is an alarm.

**The number to beat to have done anything at all is 0.7606** — the stock
untrained adapter.

**The declared ceiling stands:** semantic appearance match — kit colour, roughly
right build — **not identity**, because CLIP image embeddings lack fine spatial
detail. That is a real result to report, not a disappointment. **An honest
negative closes `PLAN.md` §7 P0 item 5 just as well as a win.**

## Traps

- **Do not retrain first.** Three checkpoints are on disk; the cheap experiment
  comes before the expensive one.
- **If the re-score comes out good, add a check rather than stopping.** Beating
  the static-copy floor would be the first such result in this project's history.
- **Do not repeat pose-ref.** Uni-ControlNet remains last.
- GPUs are shared and were free at 2026-08-26 16:00. Check before claiming one,
  and say which you took.

## Host quirks that cost real time in wave 4

- **`conda run` swallows pytest's summary line.** Use
  `python -m pytest -p no:warnings --junit-xml=<file> -q` and read counts from
  the XML. A piped exit code is **not** evidence.
- **Long jobs run detached in the background**, never a foreground poll loop, and
  **confirm a process is dead with `ps` before relaunching it** — a notification
  is not proof, and a relaunch that begins with `rm -rf` destroys the evidence.
- **The local suite now takes ~18 minutes** because tier tests do real 4K VMAF
  work here. CI skips them (its ffmpeg has no libvmaf) and stays fast.
- **Confirm CI is actually green before saying it is.**

Never `git add -A` — it commits a spurious `D assets/weights/.gitkeep`. Add
explicit paths. Open a PR when green.

Report: what the calibration showed, which anchoring you chose, the re-scored
numbers with n and SE against each bound, which bounds fired, and whether P0
item 5 now closes.
