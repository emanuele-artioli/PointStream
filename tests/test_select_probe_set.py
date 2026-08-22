"""Pre-rewrite v1 probe-set tests. The helpers they imported were removed.

``scripts/select_probe_set.py`` is now a v2 wrapper around
``experiments.probe_set regenerate``. Coverage of that path lives in
``tests/components/test_probe_set.py``. This module is kept so an old path
does not silently collect assertions against deleted v1 helpers.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "v1 select_probe_set helpers were removed; see tests/components/test_probe_set.py",
    allow_module_level=True,
)
