"""The accepted allocation changes a resource gate, never a quality decision."""
import math

import pytest

from experiments.tennis.k6b_real_base_judge import projection_checks


def test_saved_projection_fails_old_reservation_and_fits_approved_revision():
    worker = 2227.523141285288
    publication = (2448.3146460367134 - worker) / 54
    old, projected = projection_checks(worker, publication, 1800)
    revised, repeated = projection_checks(worker, publication, 2600)
    assert not any(old.values())
    assert all(revised.values())
    assert projected == repeated == pytest.approx(2448.3146460367134)


def test_publication_and_strict_boundary_remain_binding():
    checks, projected = projection_checks(2500, 2, 2600)
    assert projected == 2608
    assert checks['projection_under_full_reservation']
    assert not checks['projection_including_publication']
    assert not any(projection_checks(2600, 0, 2600)[0].values())


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1])
@pytest.mark.parametrize('position', range(3))
def test_invalid_cost_evidence_cannot_promote(value, position):
    values = [2200, 4, 2600]
    values[position] = value
    checks, _ = projection_checks(*values)
    assert not any(checks.values())
