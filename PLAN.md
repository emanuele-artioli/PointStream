# PointStream Area Index & Active Plan

Research thesis, evidence boundaries, and proposed next tests:
[semantic codec thesis](docs/strategy/semantic-codec-thesis.md). This working
note does not change the active evaluation gate below.

Current coordination: **23 September 2026**.
Start with the [23 September development campaign](docs/workflow/session/evaluation-campaign/20260923-development-campaign.md).
The September 17 roadblock, the 20 September evidence freeze, and the older
handoffs are historical. Submission **30 September**. Evidence freeze
**29 September**. Held-out confirmation is skipped. The decision metric is
weighted PSNR (`0.7` foreground + `0.3` background). A claimable point is at
least the anchor's weighted PSNR and no more bytes.

## Active measured follow-up

The previous modular victory was a constant table, not an encode. The current
work is a sequential 48-frame Federer measurement on real frames, with the
192-frame window held until the controls are correct.

1. **Background codec:** keep the current PointStream plate on WebP for now;
   record the full-resolution intra-codec replacement as deferred work. The
   old Presley contract is not a PointStream acceptance criterion.
2. **VVC failure mode:** reproduce the empty-file/exit-0 behavior, compare the
   FFmpeg `libvvenc` wrapper with the installed `vvencapp`, and make measured
   runs reject or recover from empty output while recording the actual binary.
3. **Appearance transport:** retain the per-frame crop as a control, then
   measure one initial appearance plus bbox motion and one initial appearance
   plus COCO-17 keypoint motion. Charge every motion byte. Do not claim a
   generative win until a generator reconstructs this arm and is scored on the
   same weighted ledger.
4. **Quality policy:** score PointStream and both native anchors with
   foreground/background PSNR and `0.7 FG + 0.3 BG` weighted PSNR. Rate-only
   wins remain insufficient when the quality arm is unusable.
5. **Registration diagnostic:** compare the unregistered plate control with a
   registered plate reconstructed through the transmitted camera maps. Record
   map bytes and the quality/rate tradeoff before changing the production
   default.
6. **Documentation:** update the area notes and scorecards from measured JSON
   only; keep all 65.4% / 32.5% / 66.7% claims withdrawn.

## Coordinator parallel readiness — 16 September 2026

Neural-anchor code/input are pinned in the evaluation area; weights and UF-compatible Torch/CUDA remain unresolved. RT CPU entropy preflight
passes; there is no neural video benchmark. The data area pins the
three-source score-free reservation, with exposure/PTS audit and execution
freeze still open. Paper setup `7a0476e` is pushed to Overleaf and builds to
23 body/reference pages plus five appendix pages. No E05, neural rate ladder
or confirmation scoring is released by this preparation. The latest reports
are reviewed below; only the evidence reuse tasks in the current brief are released.

## Areas

| Area | Current state | Next campaign work |
|---|---|---|
| [Evaluation](docs/areas/evaluation.md) | #128 compaction/client audit returned; tested configuration stopped | E06 reuse/claim audit; no native re-encodes |
| [Codec](docs/areas/codec.md) | #127 saved-data decomposition returned; narrow audit corrections open | E04 background coverage and paired removal |
| [Generation](docs/areas/generation.md) | #129 merged; blank completed, fitted checkpoint not promoted | E05 saved-checkpoint evidence; Stage2 unreleased |
| [Data](docs/areas/data.md) | Preserve exposure history; six fresh matches preferred but lower prospective count permitted | E01 split/protocol, E07 confirmation and second domain after tennis win |
| [Infrastructure](docs/areas/infrastructure.md) | Existing detached monitoring; verify host availability at each launch | Per-host CPU <=90% available; any free GPU; no cleanup helper |
| [Paper](docs/areas/paper.md) | Separate repo; no new competitive evidence certified by repairs | E08 setup/structure early, final claims after accepted results |

## Module Scorecards & Operational Headroom

| Module | Owner Lane | Verdict | Short Headroom (48f) | Long Headroom (192f) | Next Action |
|---|---|---|---|---|---|
| [01 Segmentation](docs/scorecards/01_segmentation.md) | Antigravity FG | SATISFIED_FREEZE | $\Delta F = -1.1\text{ kB}$ (10%) | $\Delta F \approx -4\text{ kB}$ | Freeze YOLO; SAM 3.1 below 15% threshold |
| [02 Background](docs/scorecards/02_background.md) | PointStream codec | ACTIVE_SEARCH | Registered VVC QP 40 plate 58,814 B; warp residual QP 46 is 107 kB at BG 29.56 dB | not remeasured | Matched-QP residual fell only ~29 kB; foreground still blocks a weighted win |
| [03 Appearance Crops](docs/scorecards/03_appearance_crops.md) | PointStream foreground | ACTIVE_SEARCH | 12 kB budget, 6 crops, FG 17.70 dB | not remeasured | Budget raised FG +3.4 dB and left 42 frames suppressed |
| [04 Motion & Metadata](docs/scorecards/04_motion_metadata.md) | Cursor / Antigravity | SATISFIED_FREEZE | $T=17.6\text{ kB} < \text{VVC}$; pred $\le +0.12\text{ dB}$ | -29.8 kB | Freeze E06 per-frame RLE wire packing |
| [05 Residuals](docs/scorecards/05_residuals.md) | PointStream codec | ACTIVE_SEARCH | Warp-error BG residual 107,005 B at QP 46; weighted 21.26 dB with the 12 kB crop budget | not remeasured | Next residual is the player error on a plate-inpainted anchor encode, not another still-plate QP |

## Current boundary — component strategy checkpoint

E06 compact transport and client timing support stopping this tested codec
configuration; #128 remains under narrow eligibility review. E05/#129 merged
`7b771f0` completes blank/repeat controls and supports no Stage2 promotion for
this fitted checkpoint; complete calibration/generalization remain unaccepted.
E04B/#127 supplies saved-data decomposition but reporting/provenance and empty-
mask/policy interpretation need narrow correction. Retain removal-OFF provisionally
on this scene; no Telea capability or global geometric bound is established.

The component strategy remains in force. Search coverage is still narrow and
September17 has no valid baseline-clearing generator: an explicit roadblock.
Next work is costed mechanism-driven cards and narrow PR acceptance corrections,
not another unbounded evidence-repair cycle or execution sweep.

| Owner | Next action |
|---|---|
| Cursor | #128 eligibility correction; costed physical-wire/predictor headroom card |
| Antigravity foreground | Preserve checkpoint stop; two costed baseline-clearing alternatives, no training |
| Antigravity background | #127 derived provenance, empty-mask and scoped policy correction; no encodes |
| Codex | Neural weights/whole-model readiness, confirmation provenance and paper integration |

Follow [the checkpoint and paste-ready task scopes](docs/workflow/session/evaluation-campaign/tasks/20260917-strategy-roadblock.md).
Reports return to Clean project state and dispatch,
`01a0a923-4c5e-71c3-8993-5c68f2a76bb4`. Preserve paused worker worktrees.
Confirmation scoring remains unauthorized. September20 is at risk; neural anchor,
valid neural baseline clearance, confirmed tennis win and second domain remain
submission requirements. No scope/date change or final procedure freeze.
