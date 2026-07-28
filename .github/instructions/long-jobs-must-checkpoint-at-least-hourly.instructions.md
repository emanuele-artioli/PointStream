---
applyTo: "scripts/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Long jobs must checkpoint at least hourly' section. Copilot's cloud agent and code review
     read the whole of AGENTS.md; this copy is for Copilot Chat, which
     reads only .github/. -->

## Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day, and a job can also be killed
by accident. The rule is therefore about *how much progress a kill can cost*,
not about how long a job may run:

- Any job expected to exceed an hour **checkpoints at least every 60 minutes
  of wall clock**, independent of its epoch or step cadence, and its resume
  path is verified *before* it is relied on. A training script that cannot
  checkpoint hourly must be cheap to restart from scratch (well under an
  hour) or be redesigned.
- Long-running scripts append a progress line (step, loss or metric,
  timestamp) to their log **at least every 10 minutes**, so a silent hang is
  visible in minutes and a progress watcher always has something fresh to
  match on.
- Launch detached (your agent's own background-launch mechanism, or
  `setsid nohup … < /dev/null &`) —
  never attached to a foreground shell that an SSH drop takes with it. See
  `/train-campaign` for the full launch/resume/tripwire workflow.
