# G2 Training Campaign — Completion & Repair Plan
*Status: Active | Last updated: 2026-07-13 | Code: `scripts/train_campaign.py`, `scripts/eval_checkpoint.py`, `outputs/campaign/g2_overnight/`*

## Scope

The real G2 campaign (report 10 Phase 5.4 harness) was launched on
2026-07-12/13 (`python -m scripts.train_campaign --campaign-dir
outputs/campaign/g2_overnight --initial-epochs 20 --auto-continue
--train-batch-size 16`). This plan records its actual state as of
2026-07-13 midday, the three defects that will corrupt its ranking if
left alone, and the exact steps from here to a G2 quality claim.

## Campaign state as found (2026-07-13 ~12:00)

| variant | rung 0 status | evidence |
|---|---|---|
| `spade4tennis_lite` | **CRASHED at 00:24 — CUDA OOM** (batch 16 @ 512 px on the shared 48 GB GPU, alloc failed in the first SPADE blocks) | `logs/spade4tennis_lite_rung0.log` tail |
| `controlnet_pose` | trained 20/20 epochs (00:24→07:05); probe eval ran ≈07:05→10:47 (~3.7 h of SD1.5 sampling) | `logs/controlnet_pose_rung0.log` |
| `pix2pix` | training since 10:47 (16,272 images, 20 epochs, batch 16) | `logs/pix2pix_rung0.log`, live PID |

Housekeeping facts: the harness's real state file is
`campaign_state.json` (`state.json` alongside it is a stale leftover from
an earlier aborted attempt — ignore/delete it). The held-out-video gate
**is** respected: `--data-root assets/probe_set/training_view` contains
only the 5 training videos; `verify_data_root_excludes_probe_set` guards
the probe split.

## Defect 1 (must fix before rung 0's ranking is accepted): infra failure = silent elimination

`train_campaign.py` handles a nonzero training exit by *skipping that
variant's eval* (`WARNING: … skipping its eval this rung`). Ranking then
runs on the scored variants only, and `promote_survivors` keeps
`ceil(n_ranked/2)`. Concretely for this campaign: spade never gets
scored, `ranked` has 2 entries, **only 1 survivor is kept — so rung 0
will eliminate spade4tennis for an OOM and one of pix2pix/controlnet on
quality, leaving a single variant after one rung.** A variant knocked out
by a memory error is not a quality result; the campaign's conclusion
("architecture X won") would be untrustworthy for the paper.

**Fix (harness, small and testable):**
1. In `main()`, when a training subprocess exits nonzero, retry once with
   the batch size halved (append/override `--batch-size`); if it fails
   again, mark the rung **incomplete** and exit nonzero *without ranking
   or pruning* — never rank a rung in which an alive variant has no score
   for infra reasons. Add a unit test: fake variant whose train command
   fails → assert no state mutation and nonzero exit.
2. Keep the current behavior only behind an explicit
   `--prune-on-train-failure` flag (default off) for the case where a
   variant is genuinely broken and a human says so.

**Repair (this campaign, after pix2pix's rung-0 train+eval finishes):**
1. If the running invocation has already ranked/pruned by the time you
   get to it: restore `campaign_state.json`'s `alive` to all 3 variants
   and reset `rung` to 0 (the JSON is the resume mechanism; edit it,
   don't restart the campaign dir from scratch — completed checkpoints
   and eval JSONL rows stay valid).
2. Train spade4tennis rung 0 alone with the fixed harness (batch 4 —
   its OOM was at 16; SPADE's ref-encoder activations at 512 px are the
   heavy part) or directly via `scripts/train_spade4tennis.py` with the
   same cumulative-epoch target (20) and the same `--data-root`, then
   eval it through `scripts/eval_checkpoint.py` so its JSONL row is
   comparable.
3. Only then accept the rung 0 ranking (now over 3 scored variants) and
   let successive halving proceed.

## Defect 2: ControlNet probe-eval cost (~3.7 h/rung) will dominate the campaign

12 probe clips × 48 frames × full SD1.5 sampling ≈ 3.7 h per ControlNet
variant per rung — and rung budgets double. Report 10 already flagged
this ("consider a faster inference-step count during training-time
probing"). **Fix:** add an `--eval-steps` knob to
`eval_checkpoint.py`/`train_campaign.py`, use 10 steps for in-campaign
probing (identical across variants and rungs, so the ranking is fair),
and reserve full-step sampling for the final winner's quality check.
Record the step count in every JSONL row.

## Defect 3: batch size was chosen blind on a shared GPU

Batch 16 @ 512 px OOM'd spade at 00:24 while other users held ~14 GB.
**Rule going forward** (also in report 12's checklist): check
`nvidia-smi` free memory immediately before launching; default batch 4 +
gradient accumulation to an effective 16 unless ≥ 30 GB are free. The
campaign runs for hours — other users' allocations *will* change under it.

## Defect 4 (2026-07-13, found post-repair): a stale-state relaunch clobbered the real controlnet checkpoint and silently downgraded rung 0 to 2 epochs

**What happened:** after Defect 1's harness fix landed, the *original*
orchestrator process (already running with the pre-fix code loaded in
memory — a fix on disk cannot change an already-running process) was left
to finish or was restarted; either way, a second `train_campaign`
invocation was launched:

```
python -m scripts.train_campaign --campaign-dir outputs/campaign/g2_overnight \
  --data-root assets/probe_set/training_view --manifest assets/probe_set/manifest.json \
  --train-batch-size 16 --auto-continue --eval-steps 10 --train-timeout-sec 7200
```

**`--initial-epochs` is missing from this command** — it silently fell
back to the argparse default of **2**. `campaign_state.json`'s history now
shows rung 0 completed at `target_epochs: 2`, not the intended 20:

| variant | rung-0 result at 2 epochs |
|---|---|
| pix2pix | PSNR 21.02, VMAF 6.07 |
| spade4tennis_lite | PSNR 21.89, VMAF 3.00 |
| controlnet_pose | PSNR 9.76, VMAF 0.11 — pruned |

**None of these numbers mean anything.** VMAF in the single digits for
*every* variant is what 2-epoch, barely-started training looks like for
all three architectures — this is not a real quality comparison, and
controlnet_pose was not "atrociously bad relative to the others," the
whole rung was undertrained.

**Worse: the real 20-epoch ControlNet checkpoint was overwritten.**
`eval_checkpoint`'s controlnet path reads the *top-level* files in
`checkpoints/controlnet_pose/` (`config.json`,
`diffusion_pytorch_model.safetensors`), not a numbered `checkpoint-epoch-N`
subdirectory. Those top-level files' mtimes are 2026-07-13 15:43 — i.e.
they were rewritten by this second invocation's ~3-minute rung-0 pass
(`logs/controlnet_pose_rung0.log`: load at 15:40:18, save at 15:43:32 — far
too short to be 2 real epochs of SD1.5 ControlNet fine-tuning on 16k+
images, meaning `rung_epochs` for this call likely evaluated to zero or a
negative number from a state/`--initial-epochs` mismatch). The genuinely
completed 20-epoch fine-tune from the original overnight run is **not
lost** — it survives intact in
`checkpoints/controlnet_pose/checkpoint-epoch-20/` (`config.json` +
`diffusion_pytorch_model.safetensors`, dated 07:05, untouched) — but the
file the harness actually evaluates and would resume from is now the
corrupted near-empty retrain.

pix2pix and spade4tennis are not similarly recoverable: pix2pix's resume
path requires *both* `--resume` *and* an existing
`pix2pix_checkpoint.pt` (`scripts/train_pix2pix.py:145`) — no such file
exists in `checkpoints/pix2pix/` (only the final `pix2pix_generator.pt`,
which the 2-epoch rerun overwrote at 21:23 today), so pix2pix's real
in-flight 20-epoch progress from the original run is gone. spade4tennis's
generator file is similarly dated 15:35 today (from the harness's
Defect-1 retry, not the original run).

**Also compounding this:** rung 1 is currently running
(`--auto-continue`), training pix2pix/spade4tennis toward 4 cumulative
epochs *on top of* this corrupted 2-epoch foundation — every additional
hour it runs is spent extending a campaign whose rung 0 answer is
meaningless.

**Recovery steps:**
1. **Stop the running campaign** (`pid` from `ps aux | grep train_campaign`)
   before it advances further — do not let rung 1 (or beyond) compound this.
2. **Recover the real ControlNet checkpoint:** copy
   `checkpoints/controlnet_pose/checkpoint-epoch-20/{config.json,diffusion_pytorch_model.safetensors}`
   over the clobbered top-level files in `checkpoints/controlnet_pose/`.
3. **Accept pix2pix/spade4tennis's real progress is gone** — their
   original in-flight training was never checkpointed anywhere the harness
   can recover from; restarting them from scratch at a real epoch budget
   is the only option (their own training scripts are fast enough — hours,
   not the ControlNet's ~7h — that this is a minor loss compared to
   ControlNet's).
4. **Reset `campaign_state.json` cleanly**: `rung: 0`, `alive`: all three,
   `cumulative_epochs`: `{"pix2pix": 0, "spade4tennis_lite": 0,
   "controlnet_pose": 20}` (reflecting the just-recovered real checkpoint —
   NOT 0, or the next rung will treat it as needing a `--from-scratch`
   retrain), `history: []`.
5. **Relaunch with `--initial-epochs` set explicitly** (20, matching the
   original intent — never rely on the argparse default for a real
   campaign) alongside the Defect-1/2/3 fixes already in place
   (`--eval-steps 10`, a checked batch size). Since `cumulative_epochs`
   for controlnet_pose is already 20, rung 0's `target_epochs=20` will
   correctly make controlnet_pose a no-op resume for this rung (0 more
   epochs needed) while pix2pix/spade4tennis train their real 20 from
   scratch — reconcile in code if `rung_epochs` going to zero for a
   variant needs special-casing (skip the training subprocess entirely
   rather than invoking it with `--epochs 20 --resume` for no-op work).
6. **Verify before trusting any number this produces:** re-check
   `controlnet_pose_rung0.log`'s new timestamp span is consistent with 20
   real epochs (~6-7h, per the original run), not another 3-minute no-op.

**Root cause / standing rule:** a `train_campaign` relaunch command must
always carry the *same* `--initial-epochs` (and other budget-defining
flags) as the invocation whose state it's resuming — the harness has no
way to detect a mismatched relaunch and will silently treat "no flag
given" as "start a tiny smoke test," even against a state file that
implies a much larger campaign is in progress. Consider a follow-up
harness change: persist `--initial-epochs` into `campaign_state.json` on
first launch and error out on a relaunch that passes a different value
without an explicit `--force-rebudget` flag.

## From repaired rung 0 to the G2 claim (order matters)

1. **Rung 0 complete + accepted** (3 scored variants, defect-1 repair).
   Add `controlnet_fused` (report 12 Phase B) as a 4th variant when
   ready, honoring report 12's fairness rule (train to the current
   cumulative budget before it may be ranked).
2. **Rungs 1+ via successive halving** (`--auto-continue` only after the
   harness fixes land). Watch the probe JSONL curve per rung; stop
   doubling when the winner's curve plateaus (two consecutive rungs
   < 1 % composite improvement) — that plateau epoch count is the budget
   for the final training run.
3. **Final training of the winner** at the plateau budget on the full
   `training_view` (still excluding held-out videos), full-step final
   eval, weights saved under `assets/weights/` (symlink convention), and
   the config default (`genai-backend` or generator weights key) updated
   in the same commit.
4. **G2 headline runs** (report 10 Phase 4 protocol, unchanged by this
   plan): full-match runs on the held-out videos (`alcaraz_highlights`,
   `djokovic_zverev`) at `tier_balanced` (+ `tier_quality` if the winner
   is a GenAI backend), vs post-hoc AV1/HEVC anchors on PointStream's own
   scene spans, BD-rate via the residual-CRF sweep with the metadata
   floor shown, win/loss distribution over all scenes. Run via the
   `pipeline-runner` agent; multi-hour.
5. Fold results into report 10 (`/update-reports`), then the paper's
   Results section.

**HARD gate for step 4 (unchanged from report 10):** no G2 quality claim
may use generative weights whose training data included the held-out
videos. The campaign checkpoints satisfy this; any *older* checkpoint in
`assets/weights/` does not automatically — check its provenance before
letting a tier config load it.

## Self-verification checklist (each campaign intervention)

- [ ] `campaign_state.json` is the file you edited/read (not stale `state.json`).
- [ ] After any training subprocess: its log tail shows decreasing loss and a final "Saving … checkpoint" line; checkpoint file mtime is fresh.
- [ ] Every eval appended a JSONL row containing: variant, rung, cumulative epochs, eval step count, all metrics.
- [ ] Ranking only ever computed over variants scored *in that rung* with none missing for infra reasons.
- [ ] ruff + mypy + fast pytest green on harness edits; harness edits committed **before** the next multi-hour rung uses them.
- [ ] GPU free-memory check recorded in the session notes before each launch.
