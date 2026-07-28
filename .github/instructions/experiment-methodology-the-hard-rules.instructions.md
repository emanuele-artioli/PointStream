---
applyTo: "src/**,scripts/**,config/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Experiment methodology — the hard rules' section. Copilot's cloud agent and code review
     read the whole of AGENTS.md; this copy is for Copilot Chat, which
     reads only .github/. -->

## Experiment methodology — the hard rules

The full set, with the evidence for each, is in `RESEARCH_LOG.md` (paper repo).
The ones that bite most often:

- **Symmetry is the guarantee.** Never fork `SynthesisEngine` behavior between
  encoder and decoder. The encoder computes residuals against the
  *codec-decoded* panorama, never the raw in-memory one — that asymmetry was a
  real bug that made panorama quality a silent no-op. Any new synthesis path
  gets a bit-identity check before results built on it are trusted.
- **Verify a knob is actually wired before ablating it.** Grep its *consumer*,
  not just the config schema. An unwired `residual_block_threshold` produced a
  clean, plausible, entirely fictional ablation table that stood for a day.
- **Infra failure is not a quality result.** Never rank or prune a training
  rung in which an alive variant has no score because it OOM'd or crashed.
- **Held-out gate:** no generative quality claim unless the model was trained
  without `alcaraz_highlights` and `djokovic_zverev`.
- **Scope negative results.** "Conclusively"/"definitively" is banned on
  single-clip, single-architecture experiments — that rule was written and
  violated within a day, and the claim had to be retracted.
- **Preset names are not comparable across codecs.** Compare at matched VMAF
  across a CRF ladder, and state the preset tier.
- **Invalidated runs get `mv`'d** to `outputs/_superseded/<ts>_<reason>/`,
  never `rm`'d.
- **Evaluation must run the decoder's own code path.** Symmetry applies to
  measurement, not just synthesis: `scripts/eval_checkpoint.py` builds
  strategies via `build_genai_strategy`, the same factory the compositor uses.
  A reimplemented inference path once scored ControlNet as text-to-image from
  noise while the decoder ran img2img from the reference crop — fixing only the
  measurement was worth **+6.3 dB PSNR**. If a variant scores near zero while
  others look sane, suspect the measurement before the model.
- **Metrics are scale-specific.** VMAF floor-saturates on 512×512 actor crops
  (it returned exactly 0.00) — use LPIPS/DISTS there, and keep VMAF/FVD for the
  final full frame. Ranking is by **residual bytes**; everything else is a
  diagnostic that explains why a model won or lost.
