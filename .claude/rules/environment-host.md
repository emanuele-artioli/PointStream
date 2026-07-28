---
paths:
  - "pyproject.toml"
  - "environment.yaml"
  - "setup.py"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Environment & host' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Environment & host

Shared remote Linux GPU server, **no root/sudo/apt**, headless — save media
and plots to disk, never `cv2.imshow()`/`plt.show()`. `pyproject.toml` is the
one and only source of truth for pip packages (add new deps there, then
`pip install -e .`); `environment.yaml` is strictly the CUDA/PyTorch
bootstrapper; never create a requirements.txt. Known pin: opencv 4.8 /
numpy 1.26.4 ABI coupling (recorded in the paper repo's `RESEARCH_LOG.md`).
`git push` works via stored credential helper; no `gh`/PRs needed. When
dependencies or structure change, update `pyproject.toml` and `README.md` in
the same pass.
