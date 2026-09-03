# B'25 — Re-score IP-Adapter on an instrument that can rank models

**Closed `plans/done/RESEARCH-HISTORY.md` §7 P0 item 5 on 2026-08-26.** Honest scoped negative on a
working engine (still loses to a paste). The appearance path is real.

**Owns:** `scripts/train_controlnet.py` (eval path only),
`src/shared/training/task_eval.py`, `outputs/bp25-ip-adapter/**`,
`plans/done/BP19-conditioning-architecture.md`.

## Result

Calibration first, 12 clips at offset 8. 4-step vs 20-step on the same stock
IP-Adapter txt2img pipeline: +0.079 ± 0.023 LPIPS, 3.5σ. The tripwire is not
blind. At 4 steps the stock adapter is worse than an unrelated photo (3.8σ),
so that eval cannot rank models against real-image anchors. Ranking used 20
steps, the §2.10 protocol.

| Arm | object LPIPS (n=96) | `reid` |
|---|---|---|
| static-copy | 0.4505 ± 0.0220 | 0.9135 ± 0.0087 |
| unrelated | 0.7358 ± 0.0075 | 0.4998 ± 0.0064 |
| stock | 0.7586 ± 0.0092 | 0.5519 ± 0.0157 |
| epoch 1 (best) | **0.6922 ± 0.0094** | 0.5647 ± 0.0147 |
| epoch 1 shuffled | 0.7662 ± 0.0085 | 0.4893 ± 0.0106 |

Stock reproduces 0.7606. Extra check on clip means (n=12): epoch 1 beats stock
5.5σ, uses appearance vs shuffled 3.8σ, loses to paste 4.1σ. Vs unrelated is
1.3σ — not claimed.

Stale-checkpoint bug: mid-epoch evals now write
`checkpoint-epoch-{N}-step-{S}` and always save before scoring.

Uni-ControlNet remains last. Artifacts: `outputs/bp25-ip-adapter/`.
`plans/done/RESEARCH-HISTORY.md` §2.17.

## History — why the 2026-08-25 stop-eval was not a ranking

Training self-stopped at epoch 3, step 18000. The stop rule worked. The number
it stopped on did not: 4 diffusion steps scored against undegraded photographs,
and mid-epoch evals re-scored frozen weights. Epoch 1 then read 0.8112 LPIPS,
above the unrelated null. That run stays `not_citable`. The table above is the
ranking.
