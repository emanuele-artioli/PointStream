import copy
from pathlib import Path

import pytest

from experiments.tennis import w8_cuda_job as job


def original():
    return {'allow_paths': [job.OLD_SOURCE, job.OLD_OUTPUT, '/runtime', '/weight'],
            'denied_paths': {'source': ['/private/source'], 'prepared': ['/private/prepared'],
                             'fit': ['/private/fit'], 'golden': ['/private/golden'],
                             'masks': ['/private/masks'], 'aliases': [job.OLD_ALIASES + '/source']},
            'output': job.OLD_OUTPUT, 'parent_user_namespace': 'old',
            'parent_network_namespace': 'old', 'cpu_seconds': 120, 'worker_kind': 'cost_probe',
            'probe_steps': 1, 'files': {'weight': {'path': '/weight', 'sha256': 'x'}},
            'environment': {'TMPDIR': job.OLD_OUTPUT, 'CUDA_VISIBLE_DEVICES': 'old'}}


def test_normalization_preserves_grants_and_denials_without_weight_reads():
    historical = original()
    saved = copy.deepcopy(historical)
    result = job.normalize(historical, Path('/new/source'), Path('/new/output'),
                           Path('/new/aliases'), 'GPU-new', 'user:new', 'net:new')
    assert historical == saved
    assert result['allow_paths'] == ['/new/source', '/new/output', '/runtime', '/weight']
    assert result['denied_paths']['source'] == ['/private/source']
    assert result['denied_paths']['aliases'] == ['/new/aliases/source']
    assert result['files'] == {} and result['worker_kind'] == 'preflight'
    assert 'probe_steps' not in result
    assert job.diff(historical, result)['files']['before'] == historical['files']


def test_budget_admission_includes_arm_and_reserve():
    assert job.enough_budget(90, 150)
    assert not job.enough_budget(91, 150)
    assert not job.enough_budget(90, 151)


def test_isolated_command_is_fresh_namespace():
    cmd = job.worker_command('/runtime/python', '/script.py', 'isolated', Path('/cfg'), 'sha', Path('/src'))
    assert cmd[:4] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
    assert job.worker_command('/runtime/python', '/script.py', 'generic', Path('/cfg'), 'sha', Path('/src'))[0] == '/runtime/python'


def test_invalid_denials_are_rejected():
    with pytest.raises(ValueError, match='isolation'):
        job.check_isolation({'denials': {'/secret': 0}}, {'denied_paths': {'source': ['/secret']}})


def test_validator_rejects_changed_historical_policy(tmp_path, monkeypatch):
    import json

    historical = original()
    raw = json.dumps(historical).encode()
    (tmp_path / 'historical-config.json').write_bytes(raw)
    monkeypatch.setattr(job, 'OLD_CONFIG_SHA', job.digest(tmp_path / 'historical-config.json'))
    config = job.normalize(historical, Path('/new/source'), Path('/new/output'),
                           Path('/new/aliases'), 'GPU-new', 'user:new', 'net:new')
    config['denied_paths']['source'] = ['/wrong']
    job.write_json(tmp_path / 'worker-config.json', config)
    job.write_json(tmp_path / 'manifest.json', {'profile': job.PROFILE,
        'historical_config_sha256': job.OLD_CONFIG_SHA, 'source_sha256': job.SOURCE_SHA,
        'runtime_archive_sha256': job.RUNTIME_SHA, 'config_sha256': job.digest(tmp_path / 'worker-config.json'),
        'normalized_diff': job.diff(historical, config), 'gpu_uuid': 'GPU-new'})
    job.write_json(tmp_path / 'summary.json', {'arms': []})
    with pytest.raises(ValueError, match='normalized profile'):
        job.validate(tmp_path)


def test_cuda_failures_are_measurements_but_never_generation_pass(tmp_path, monkeypatch):
    import errno
    import json

    historical = original()
    (tmp_path / 'historical-config.json').write_text(json.dumps(historical))
    monkeypatch.setattr(job, 'OLD_CONFIG_SHA', job.digest(tmp_path / 'historical-config.json'))
    config = job.normalize(historical, Path('/new/source'), Path('/new/output'),
                           Path('/new/aliases'), 'GPU-new', 'user:new', 'net:new')
    job.write_json(tmp_path / 'worker-config.json', config)
    job.write_json(tmp_path / 'manifest.json', {'profile': job.PROFILE,
        'historical_config_sha256': job.OLD_CONFIG_SHA, 'source_sha256': job.SOURCE_SHA,
        'runtime_archive_sha256': job.RUNTIME_SHA, 'config_sha256': job.digest(tmp_path / 'worker-config.json'),
        'normalized_diff': job.diff(historical, config), 'gpu_uuid': 'GPU-new'})
    denial_paths = [p for paths in config['denied_paths'].values() for p in paths]
    isolation = {'denials': dict.fromkeys(denial_paths, errno.EACCES), 'landlock_abi': 1,
                 'network_interfaces': ['lo'], 'pre_sandbox_threads': 1, 'extra_fds': False,
                 'cpu_bootstrap': {'before_landlock_sha256': 'x', 'after_landlock_sha256': 'x',
                                   'cuda_initialized_before_isolation': False},
                 'namespaces': {'user': 'child-user', 'net': 'child-net'}}
    arms = []
    for name in ('generic', 'isolated'):
        result = {'condition': name, 'status': 'failed', 'phase': 'cuda_probe',
                  'error': {'type': 'RuntimeError', 'message': 'CUDA unavailable'}}
        if name == 'isolated':
            result['isolation'] = isolation
        job.write_json(tmp_path / f'{name}.stdout', result)
        (tmp_path / f'{name}.stderr').write_text('CUDA unavailable')
        arm = {'condition': name, 'admitted': True, 'status': 1, 'timed_out': False,
               'stdout_sha256': job.digest(tmp_path / f'{name}.stdout'),
               'stderr_sha256': job.digest(tmp_path / f'{name}.stderr')}
        job.write_json(tmp_path / f'{name}.exit.json', arm)
        arms.append(arm)
    job.write_json(tmp_path / 'summary.json', {'arms': arms})
    assert job.validate(tmp_path) == 0
    assert json.loads((tmp_path / 'validation.json').read_text()) == {
        'measurement_complete': True, 'both_cuda_passed': False, 'generation_gate_passed': False}
