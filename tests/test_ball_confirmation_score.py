import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.jobs import inbox
from experiments.tennis import ball_confirmation_score as score

FREEZE = Path('docs/experiments/submission-ladder-tennis-confirmation-source-freeze-20261115.json')


def test_official_sparse_labels_and_exact_scaling():
    data = FREEZE.read_bytes()
    assert score.common.digest(data) == score.SOURCE_MANIFEST
    manifest = json.loads(data)
    for source_id, counts in [('match209_000', (9, 3, 52)), ('match269_000', (12, 0, 52)),
                              ('match235_000', (11, 2, 141))]:
        labels = score.labels_for_source(manifest, source_id)
        assert tuple(sum(row[0] == k for row in labels) for k in (1, 0, -1)) == counts
    altered = copy.deepcopy(manifest)
    altered['sources'][0]['official_window_rows'][0]['GT720_X_exact'][0] += 1
    with pytest.raises(ValueError):
        score.labels_for_source(altered, 'match209_000')


def test_temporal_difference_includes_adjacent_pairs():
    x = np.array([[[[0, 0, 0]]], [[[10, 10, 10]]], [[[20, 20, 20]]]], dtype=np.uint8)
    y = x.copy()
    y[1] = 11
    rows = score.temporal_errors(x, y)
    assert [r['sse'] for r in rows] == [3, 3]
    assert [(r['from_frame'], r['to_frame']) for r in rows] == [(0, 1), (1, 2)]


def test_coverage_crop_null_and_full_prefix_rejection():
    n = 16
    labels = [(-1, 0., 0.)] * 64
    labels[2] = (1, 100., 100.)
    frames = []
    for i in range(n):
        crop = 0.1 if i == 2 else None
        frames.append({'frame': i, 'label_kind': labels[i][0], 'sse': 0, 'pixels': 1280 * 720 * 3,
                       'frame_psnr': 100., 'full_lpips_spatial_map': 0., 'full_lpips_standard': 0.,
                       'full_dists': 0., 'crop_lpips': crop, 'crop_dists': crop})
    temporal = [{'from_frame': i, 'to_frame': i + 1, 'sse': 0, 'pixels': 1280 * 720 * 3} for i in range(n - 1)]
    rows = [{'family': family, 'point': p, 'coded_bytes': 1, 'packet_sha256': 'b' * 64,
             'pooled_rgb_psnr': 100., 'full_lpips_standard': [0.] * n, 'frames': copy.deepcopy(frames),
             'decoded_rgb_frame_sha256': ['a' * 64] * n, 'temporal': temporal} for family, p in score.MATRIX]
    result = {'frames': n, 'available_receiver_frames': 64, 'full_frames': 64, 'promotable': False, 'rows': rows}
    assert all(score.validate_scores(result, labels, 'smoke', 64).values())
    assert not all(score.validate_scores(result, labels, 'full', 64).values())
    rows[0]['frames'][0]['crop_dists'] = 0.
    assert not all(score.validate_scores(result, labels, 'smoke', 64).values())
    rows[0]['frames'][0]['crop_dists'] = None
    rows.pop()
    assert not all(score.validate_scores(result, labels, 'smoke', 64).values())


def test_spec_is_schema_valid_and_uses_exact_full_input_gates():
    supplied = {f'{part}_{name}': (f'/home/itec/emanuele/Datasets/pointstream-data/test/{part}-{name}', 'a' * 64)
                for part in ('bases', 'controls') for name in ('archive', 'stage', 'validation', 'execution')}
    spec = inbox.validate_spec(score.spec('match235_000', '2026-11-16T12:00:00+00:00', supplied))
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 154}
    assert len(spec['staged_inputs']) == 11
    assert spec['full']['seconds'] == 1800 and spec['gpu_sharing']['margin_mib'] == 8192
    assert spec['budget_seconds'] == 2430
    with pytest.raises(ValueError):
        score.spec('match235_000', '2026-11-16T12:00:00+00:00', {})


def test_full_stage_acceptance_requires_bound_validation_and_exit(tmp_path, monkeypatch):
    import hashlib

    def write(name, value):
        path = tmp_path / name
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    archive, archive_hash = write('published.tar', b'bound archive')
    stage, stage_hash = write('stage.json', {'phase': 'bases', 'source_id': 'match209_000', 'frames': 64,
                                           'archive': {'sha256': archive_hash}})
    execution, execution_hash = write('execution.json', {'exit_code': 0, 'seconds': 100})
    dispatch = {'job_id': 'expected'}
    closure = {'schema': 'pointstream-ball-confirmation-accepted-full-v1', 'phase': 'bases',
               'source_id': 'match209_000', 'frames': 64, 'producer_job_id': '20261115T123456Z-1234abcd',
               'archive_sha256': archive_hash, 'stage_sha256': stage_hash, 'execution_sha256': execution_hash,
               'dispatch_canonical_sha256': score.canonical_sha(dispatch), 'canonical_complete': True,
               'raw_full_validation_sha256': 'f' * 64, 'passed': True, 'checks': {'native': True}}
    gate, gate_hash = write('accepted-full.json', closure)

    def extract(archive, binding, root):
        root.mkdir(exist_ok=True)
        (root / 'receipt.json').write_text(json.dumps({'dispatch': dispatch}))

    monkeypatch.setattr(score.confirmation, 'extract_tree', extract)
    monkeypatch.setattr(score.confirmation, 'validate_publication', lambda *a: {'all_frames': True})
    args = ('match209_000', tmp_path / 'out', archive, stage, gate, execution,
            archive_hash, stage_hash, gate_hash, execution_hash, 'bases')
    assert score.read_full_inputs(*args)['frames'] == 64
    for field, wrong in [('execution_sha256', '0' * 64), ('stage_sha256', '0' * 64),
                         ('source_id', 'match269_000'), ('frames', 16), ('canonical_complete', False),
                         ('dispatch_canonical_sha256', '0' * 64), ('producer_job_id', '../unsafe')]:
        bad_gate, bad_hash = write('accepted-full.json', {**closure, field: wrong})
        bad_args = (*args[:4], bad_gate, *args[5:8], bad_hash, *args[9:])
        with pytest.raises(ValueError):
            score.read_full_inputs(*bad_args)
    unbound, unbound_hash = write('accepted-full.json', {'passed': True, 'checks': {'native': True}})
    with pytest.raises(ValueError):
        score.read_full_inputs(*(*args[:4], unbound, *args[5:8], unbound_hash, *args[9:]))
