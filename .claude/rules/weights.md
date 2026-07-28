---
paths:
  - "assets/weights/**"
  - "scripts/download_weights.py"
  - "src/decoder/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Weights' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Weights

Search `/home/itec/emanuele/Models` first and **symlink** into
`assets/weights/` (see existing symlinks there); `scripts/download_weights.py`
fetches what's missing. Never expose the absolute host path in README or any
user-facing doc — users are told to place weights in `assets/weights/`.
Naming trap: `assets/weights/custom-controlnet` is the fine-tuned **Canny**
checkpoint (there is no `canny-controlnet` path), and
`ip-adapter-controlnet` is architecturally a fourth `ControlNetModel`, not a
diffusers-native IP-Adapter.
