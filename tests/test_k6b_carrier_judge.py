"""Reject partial carrier evidence and unregistered representation substitutions."""
import io
import json
import tarfile

import numpy as np

from experiments.tennis.k4 import safe
from experiments.tennis.k6b_carrier_judge import validate
from experiments.tennis.k6b_foreground_video import PILOTS, SOURCE_SHA
from experiments.tennis.k6b_rate_transport import sha


def archive(tmp_path, *, judged=(0, 3), variants=('old-control', 'alpha'), expected=('old-control', 'alpha')):
    wid = PILOTS[0]
    arrays = {m: np.full((len(variants), 240), np.nan) for m in ('lpips', 'dists', 'outside_r_sse')}
    for a in arrays.values():
        a[:, list(judged)] = .2
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    blob = buffer.getvalue()
    row = {'id': wid, 'variants': list(variants), 'source_sha256': SOURCE_SHA[0],
           'judged': list(judged), 'expected_judged': list(judged), 'original_judged': [0, 3],
           'control_max_abs_difference': 0., 'files': {'scores.npz': sha(blob)},
           'means': {v: {m: .2 for m in arrays} for v in variants}}
    report = {'on_cuda': {'lpips': True, 'dists': True}, 'device': {'class_matches': True},
              'gpu_peak_gib': 7., 'memory_limit_mib': 14000, 'rows': [row], 'requested': [wid],
              'expected_variants': {wid: list(expected)}}
    path = tmp_path/'synthetic-validator-contract.tar'
    with tarfile.open(path, 'w') as t:
        for name, data in [('publish/carrier-judge.json', json.dumps(report).encode()),
                           ('publish/'+safe(wid)+'/scores.npz', blob)]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            t.addfile(member, io.BytesIO(data))
    return path


def test_full_gate_rejects_smoke_coverage(tmp_path):
    path = archive(tmp_path, judged=(0,))
    assert validate(path, False)['passed']
    assert not validate(path, True)['passed']


def test_gate_rejects_missing_or_substituted_representation(tmp_path):
    assert validate(archive(tmp_path), True)['passed']
    assert not validate(archive(tmp_path, variants=('old-control',)), True)['passed']
    assert not validate(archive(tmp_path, variants=('old-control', 'other')), True)['passed']


def test_saved_rgb_and_fresh_generator_tolerances_are_distinct():
    import pytest
    from experiments.tennis.k6b_matte_judge import control_error

    expected = np.array([.2])
    fresh = np.array([.200183225])
    with pytest.raises(ValueError):
        control_error(fresh, expected)
    assert control_error(fresh, expected, tolerance=1e-3) < 1e-3
    with pytest.raises(ValueError):
        control_error(np.array([.2011]), expected, tolerance=1e-3)
