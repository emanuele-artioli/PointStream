"""Apply the invariant checks to the real outputs/ tree.

Excluded from the default run (marker `invariants`) because it needs the
outputs directory, which is gitignored and only exists on the host that
produced it:

    pytest -m invariants

Violations are a normal research outcome — the failsafe is that they stay
recorded and visible, not that they never happen. What must not exist is a run
carrying *no* verdict, because a missing verdict reads as clean to every
consumer of these summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.invariants import check_run

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"

pytestmark = pytest.mark.invariants


def load_all():
    if not OUTPUTS.is_dir():
        pytest.skip("outputs/ not present on this host")
    loaded = []
    for path in sorted(OUTPUTS.glob("*/run_summary.json")):
        try:
            loaded.append((path.parent.name, json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError):
            continue  # a corrupt summary is re-run, not asserted on
    if not loaded:
        pytest.skip("outputs/ has no readable run_summary.json files")
    return loaded


def test_every_run_carries_a_verdict():
    """The one state the citation rule cannot act on is a missing verdict.

    Backfill with: python -m src.shared.invariants outputs/
    """
    missing = [name for name, s in load_all() if "invariant_failures" not in s]
    assert not missing, (
        f"{len(missing)} run(s) have no invariant verdict and would be treated as "
        f"citable by default; run `python -m src.shared.invariants outputs/`. "
        f"First few: {missing[:10]}"
    )


def test_stored_verdicts_match_the_current_checks():
    """A stale verdict is worse than none: it asserts a run is clean."""
    stale = []
    for name, summary in load_all():
        stored = summary.get("invariant_failures")
        if stored is None:
            continue  # covered by the test above
        if sorted(stored) != sorted(check_run(summary)):
            stale.append(name)
    assert not stale, (
        f"{len(stale)} run(s) carry a verdict that disagrees with the current checks; "
        f"re-run `python -m src.shared.invariants outputs/ --force`. "
        f"First few: {stale[:10]}"
    )


def test_flagged_runs_are_reported():
    """A standing report of what is currently uncitable, so it cannot creep up."""
    offenders = {name: f for name, s in load_all() if (f := check_run(s))}
    if offenders:
        print(f"\n{len(offenders)} run(s) are NOT citable:")
        for name, failures in offenders.items():
            print(f"  {name}: {'; '.join(failures)}")
