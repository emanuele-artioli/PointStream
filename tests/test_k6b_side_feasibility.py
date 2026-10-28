import hashlib

import pytest

from experiments.tennis.k6b_side_feasibility import hull, packing, side_ceiling, verified


def test_inverse_uses_achievable_hull_not_dominated_middle_point():
    curve = hull([(10., -.5), (20., -.4), (30., -.1), (10., -.6)])
    assert curve == [(10., -.5), (30., -.1)]
    assert side_ceiling(curve, .3, 15., 0.) == pytest.approx(5.)
    assert side_ceiling(curve, .3, 15., .02) == pytest.approx(4.)


def test_no_extrapolation_and_negative_budget_preserved():
    curve = hull([(10., -.5), (30., -.1)])
    assert side_ceiling(curve, .09, 15., 0.) is None
    assert side_ceiling(curve, .51, 15., 0.) is None
    assert side_ceiling(curve, .5, 15., 0.) == -5.
    assert side_ceiling(curve, .1, 15., 0.) == 15.
    with pytest.raises(ValueError):
        side_ceiling(curve, .3, 15., -.001)


@pytest.mark.parametrize('points', [[], [(1., .1)], [(1., .1), (2., .1)],
                                   [(1., .1), (2., float('nan'))]])
def test_invalid_hull_rejected(points):
    with pytest.raises(ValueError):
        hull(points)


def test_packet_wrappers_pay_header_and_roundtrip():
    packet = bytes(range(256)) * 16
    rows = packing(packet, 8.)
    assert len(rows) == 3
    assert all(r['roundtrip'] and r['kbps'] == r['bytes'] / 1000 for r in rows)
    # The pilot must not select a changed packet or charge payload alone.
    import zlib
    assert rows[0]['bytes'] == len(zlib.compress(packet, 9)) + 8
    with pytest.raises(ValueError):
        packing(packet, 0.)


def test_frozen_evidence_identity_cannot_be_substituted(tmp_path):
    path = tmp_path / 'changed.tar'
    path.write_bytes(b'changed')
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verified(path, 'result')
