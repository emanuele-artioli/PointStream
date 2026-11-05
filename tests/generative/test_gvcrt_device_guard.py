import copy
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.generative import gvcrt_device_guard as guard

UUID = 'GPU-12345678-1234-4234-8234-123456789abc'


def valid():
    return {'count': 1, 'uuid': UUID, 'visible': UUID, 'capability': [8, 9],
            'name': 'NVIDIA RTX 6000 Ada Generation', 'provider': 'torch_cuda',
            'model_first_parameters_on_cuda': True}


@pytest.mark.parametrize('key,value', [('uuid', ''), ('uuid', 'garbage'),
                                     ('uuid', 'GPU-00000000-0000-4000-8000-000000000000'),
                                     ('visible', 'GPU-00000000-0000-4000-8000-000000000000'),
                                     ('capability', [8, 6]), ('name', 'NVIDIA RTX A6000'), ('count', 2),
                                     ('provider', 'cpu'), ('model_first_parameters_on_cuda', False)])
def test_rejects_missing_foreign_spoofed_or_unqualified_device(key, value):
    data = valid()
    guard.check_device(data, UUID, scorer=True)
    bad = copy.deepcopy(data)
    bad[key] = value
    with pytest.raises(ValueError):
        guard.check_device(bad, UUID, scorer=True)


def test_observe_requires_initialized_actual_single_device():
    props = SimpleNamespace(name='NVIDIA RTX 6000 Ada Generation', uuid=UUID[4:], major=8, minor=9)
    cuda = SimpleNamespace(is_initialized=lambda: True, device_count=lambda: 1, get_device_properties=lambda _: props)
    assert guard.observe(SimpleNamespace(cuda=cuda), UUID, UUID)['uuid'] == UUID[4:]
    cuda.is_initialized = lambda: False
    with pytest.raises(ValueError, match='initialized'):
        guard.observe(SimpleNamespace(cuda=cuda), UUID, UUID)


def test_guarded_scorer_preserves_original_numeric_output(tmp_path, monkeypatch):
    from experiments.generative import gvcrt_peer_scorer as original
    monkeypatch.syspath_prepend(str(__import__('pathlib').Path(original.__file__).parent))
    from experiments.generative import gvcrt_guarded_scorer as wrapper
    props = SimpleNamespace(name='NVIDIA RTX 6000 Ada Generation', uuid=UUID[4:], major=8, minor=9)
    cuda = SimpleNamespace(is_available=lambda: True, init=lambda: None, is_initialized=lambda: True,
                           device_count=lambda: 1, get_device_properties=lambda _: props)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    payload = {'metrics': [{'frame': 0, 'full_lpips': .25}], 'wasb': {'rows': [1]}, 'balltrack': {'rows': [2]}}
    def score(args):
        args.output.write_text(json.dumps(payload))
    monkeypatch.setattr(wrapper.original, 'score', score)
    output = tmp_path / 'score.json'
    argv = ['wrapper', '--source', 's', '--decoded', 'd', '--labels', 'l', '--config', 'c',
            '--output', str(output), '--count', '4', '--expected-uuid', UUID]
    monkeypatch.setattr(sys, 'argv', argv)
    wrapper.main()
    result = json.loads(output.read_text())
    assert all(result[k] == v for k, v in payload.items())
    guard.check_device(result['device'], UUID, scorer=True)


def test_saved_device_proof_rebinds_dispatch_and_observations():
    from experiments.generative.gvcrt_peer_job_w8leaves import check_saved_device_proof

    dispatch = {'environment': {'gpu': {'uuid': UUID}}}
    result = {'gpu_uuid': UUID, 'gpu_identity': {'uuid': UUID, 'name': valid()['name']},
              'probe_cuda': valid()}
    score = {'device': valid()}
    check_saved_device_proof(dispatch, result, score)
    foreign = 'GPU-00000000-0000-4000-8000-000000000000'
    for obj, path, value in [
        (dispatch, ('environment', 'gpu', 'uuid'), foreign),
        (dispatch, ('environment', 'gpu', 'uuid'), ''),
        (result, ('gpu_uuid',), foreign),
        (result, ('gpu_identity', 'uuid'), foreign),
        (result, ('probe_cuda', 'uuid'), ''),
        (result, ('probe_cuda', 'count'), 2),
        (result, ('probe_cuda', 'capability'), [8, 6]),
        (score, ('device', 'uuid'), foreign),
        (score, ('device', 'count'), 2),
        (score, ('device', 'capability'), [8, 6]),
    ]:
        changed = copy.deepcopy(obj)
        leaf = changed
        for key in path[:-1]:
            leaf = leaf[key]
        leaf[path[-1]] = value
        args = [dispatch, result, score]
        args[[dispatch, result, score].index(obj)] = changed
        with pytest.raises((ValueError, KeyError, TypeError)):
            check_saved_device_proof(*args)
