import pytest

from experiments.visor import ego_mask_budget_oracle as oracle


def shape():
    return {'item': 'x', 'frames': 240, 'fps': 50, 'scale': 8,
            'rows': [{'bytes': n, 'roundtrip': True, 'accuracy_identical': True} for n in (20, 10, 30)]}


def test_rebills_exact_byte_and_framing_costs_without_changing_scores():
    rows = [{'bytes': 1200, 'kbps': 2.0, 'breakdown': {'mask': 240}, 'scores': {'quality': 5}}]
    free = oracle.rebill(rows, shape(), 'free')[0]
    ideal = oracle.rebill(rows, shape(), 'ideal_2kbps')[0]
    actual = oracle.rebill(rows, shape(), 'shape')[0]
    assert free['kbps'] == pytest.approx(1.6)
    assert ideal['kbps'] == pytest.approx(3.6)
    assert actual['kbps'] == pytest.approx(1.6 + 12 * 8 / 4.8 / 1000)
    assert free['scores'] == ideal['scores'] == actual['scores'] == rows[0]['scores']


def test_rejects_unverified_shape_and_wrong_original_rate():
    s = shape()
    s['rows'][1]['accuracy_identical'] = False
    rows = [{'bytes': 1200, 'kbps': 2.0, 'breakdown': {'mask': 240}}]
    with pytest.raises(ValueError, match='exact-roundtrip'):
        oracle.rebill(rows, s, 'shape')
    rows[0]['kbps'] = 7
    with pytest.raises(ValueError, match='cadence'):
        oracle.rebill(rows, shape(), 'free')


def test_rejects_unauthenticated_archive(tmp_path):
    path = tmp_path / 'wrong.tar.gz'
    path.write_bytes(b'not the report')
    with pytest.raises(ValueError, match='SHA-256'):
        oracle.run(path)
