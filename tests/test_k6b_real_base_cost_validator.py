"""Production validator selects only the frozen resource model and native timers."""
import io
import json
import tarfile

import numpy as np
import pytest

from experiments.tennis import k6b_real_base_judge as judge


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    stage = tmp_path / 'smoke'
    stage.mkdir()
    ticks = [0, 3, 6]
    windows = ['alcaraz_hurkacz/031@129', 'medvedev_struff/014@194']
    rows = [{'id': wid, 'ticks': ticks, 'frame_seconds': {'0': 4, '3': 2, '6': 2},
             'rates': [{'total_kbps': 2, 'base_kbps': 1, 'bootstrap_bytes': 1}] * 3,
             'composite_hashes': {str(t): ['hash'] * 6 for t in ticks}} for wid in windows]
    calibration = [{'id': wid, 'ticks': ticks, 'scored_frame_seconds': {'0': 4, '3': 2, '6': 2},
                    'exact_control': True, 'prediction_control': True} for wid in windows]
    summary = {'rows': rows, 'calibration': calibration, 'seconds': 20, 'projected_full_seconds': 100000,
               'cost_model': 'separated-v1', 'frames_per_window': 3,
               'device': {'class_matches': True}, 'peak_gpu_gib': 10}
    spec = {'arguments': ['--cost-model', 'separated-v1'], 'full': {'seconds': 2000}, 'scale': {'judged': {'smoke': 3}}}
    files = {'quality-summary.json': summary, 'dispatch.json': {'started': 979},
             'execution.json': {'seconds': 42}, 'staging.json': {'publication_seconds': 2}}
    for name, value in files.items():
        (stage / name).write_text(json.dumps(value))
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    metrics = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists', 'full_psnr')
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        for wid in windows:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **{m: np.zeros((6, 3)) for m in metrics}, ticks=np.asarray(ticks))
            content = buffer.getvalue()
            member = tarfile.TarInfo('publish/' + wid.replace('/', '__') + '/scores.npz')
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    output = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(output))
    monkeypatch.setattr(judge.time, 'time', lambda: 1000)
    return stage, spec, summary, output


def update(path, key, value):
    content = json.loads(path.read_text())
    content[key] = value
    path.write_text(json.dumps(content))


def test_native_command_and_direct_publication_override_internal_timing(pilot):
    stage, _, _, output = pilot
    judge.validate()
    result = json.loads(output.read_text())
    assert result['passed']
    assert result['projection']['fixed_seconds'] == 10
    assert result['projected_with_publication'] == pytest.approx(960.3)
    assert result['projection_error'] is None


@pytest.mark.parametrize('fault', ['native_missing', 'publication_missing', 'unstable', 'duplicate_window', 'model_mismatch', 'scale_mismatch', 'publication_overflow', 'strict_boundary'])
def test_incomplete_changed_or_expensive_evidence_fails_closed(pilot, fault):
    stage, spec, summary, output = pilot
    if fault == 'native_missing':
        (stage / 'execution.json').unlink()
    elif fault == 'publication_missing':
        (stage / 'staging.json').write_text('{}')
    elif fault == 'unstable':
        summary['rows'][0]['frame_seconds']['6'] = 3
    elif fault == 'duplicate_window':
        summary['calibration'][1]['id'] = summary['calibration'][0]['id']
    elif fault == 'model_mismatch':
        summary['cost_model'] = 'legacy'
    elif fault == 'scale_mismatch':
        spec['scale']['judged']['smoke'] = 2
    elif fault == 'publication_overflow':
        update(stage / 'staging.json', 'publication_seconds', 500)
    else:
        spec['full']['seconds'] = 960.3
    (stage / 'quality-summary.json').write_text(json.dumps(summary))
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(SystemExit):
        judge.validate()
    assert json.loads(output.read_text())['passed'] is False


def test_legacy_request_retains_two_ticks_and_original_projection_without_native_timers(pilot):
    stage, spec, summary, output = pilot
    spec['arguments'] = []
    spec['scale']['judged']['smoke'] = 2
    summary.pop('cost_model')
    summary['frames_per_window'] = 2
    summary['projected_full_seconds'] = 900
    for collection in ('rows', 'calibration'):
        for row in summary[collection]:
            row['ticks'] = [0, 3]
            row.pop('frame_seconds', None)
            row.pop('scored_frame_seconds', None)
            if 'composite_hashes' in row:
                row['composite_hashes'].pop('6')
    (stage / 'quality-summary.json').write_text(json.dumps(summary))
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'execution.json').unlink()
    (stage / 'staging.json').unlink()
    metrics = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists', 'full_psnr')
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        for row in summary['rows']:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **{m: np.zeros((6, 2)) for m in metrics}, ticks=np.array([0, 3]))
            content = buffer.getvalue()
            member = tarfile.TarInfo('publish/' + row['id'].replace('/', '__') + '/scores.npz')
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    judge.validate()
    result = json.loads(output.read_text())
    assert result['passed']
    assert result['cost_model'] == 'legacy'
    assert result['projection'] is None
    assert result['projected_with_publication'] == 954
