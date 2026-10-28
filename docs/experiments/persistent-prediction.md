# Experiments: persistent prediction

### 2026-10-28 — Accepted direction and bounded feasibility search

- Human requested plan persistence, then authorized the recommended search and
  tests. New ceilings3600GPU-s/3600CPU-s; all earlier studies remain closed.
- Hypothesis: persistent decoded memory plus explicitly coded corrections can
  exploit foreground/background repetition without expensive exact mattes or
  mandatory generative replacement. Competing explanation: ordinary native
  SVT already exploits this redundancy better; selector/geometry and correction
  streams erase any advantage. A second explanation is loss in low-resolution
  decoded history, so this finite pilot cannot bound better feature predictors.
- Scope, three-axis search, literature, configurations and stop rules are in
  [the accepted card](../cards/persistent-prediction.md). No GPU job submitted
  yet; charged0/3600GPU-s. CPU audit/preflight ongoing.
- Prior relevant findings: model discounts already tested on egocentric split
  did not recover LPIPS; tennis rendered-pixel wins granted dynamic mattes and
  background. Recent paid background LPIPS+230.28%; fixed rectangles oversized.
  The new pilot must not repeat these as hidden decoder assumptions.
- Native execution uses fleet, packed environment and host-local scratch;
  preserve prior publications and failed evidence read-only. No recurring timer.
