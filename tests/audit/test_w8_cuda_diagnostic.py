import errno

import pytest

from experiments.audit.w8_cuda_diagnostic import policy_fingerprint, run_condition, validate_isolation


def sample():
    config = {'allow_paths': ['/runtime'], 'denied_paths': {'source': ['/protected'], 'aliases': ['/alias']},
              'parent_user_namespace': 'user:parent', 'parent_network_namespace': 'net:parent'}
    receipt = {'denials': {'/protected': errno.EACCES, '/alias': errno.EPERM}, 'landlock_abi': 4,
               'network_interfaces': ['lo'], 'pre_sandbox_threads': 1, 'extra_fds': False,
               'cpu_bootstrap': {'before_landlock_sha256': 'x', 'after_landlock_sha256': 'x',
                                 'cuda_initialized_before_isolation': False},
               'namespaces': {'user': 'user:child', 'net': 'net:child'}}
    return config, receipt


def test_policy_identity_and_missing_alias():
    config, receipt = sample()
    validate_isolation(receipt, config)
    original = policy_fingerprint(config)
    config['allow_paths'].append('/extra')
    assert policy_fingerprint(config) != original
    del receipt['denials']['/alias']
    with pytest.raises(ValueError):
        validate_isolation(receipt, config)


def test_isolated_arm_never_probes_after_bad_receipt():
    config, receipt = sample()
    receipt['cpu_bootstrap']['cuda_initialized_before_isolation'] = True
    calls = []
    result = run_condition('isolated', config, bootstrap=lambda _: receipt,
                           probe=lambda: calls.append('cuda'))
    assert result['status'] == 'failed' and not calls
    assert result['error']['type'] == 'ValueError'


def test_condition_order_and_error_preservation():
    config, receipt = sample()
    calls = []
    def isolated(_):
        calls.append('isolation')
        return receipt
    def failure():
        calls.append('cuda')
        raise RuntimeError('CUDA error 2')
    result = run_condition('isolated', config, bootstrap=isolated, probe=failure)
    assert calls == ['isolation', 'cuda']
    assert result['status'] == 'failed' and result['error']['message'] == 'CUDA error 2'
    assert 'user_seconds' in result['usage']
    calls.clear()
    assert run_condition('generic', config, bootstrap=isolated, probe=failure)['status'] == 'failed'
    assert calls == ['cuda']


def test_bad_cuda_receipt_fails_closed():
    config, _ = sample()
    result = run_condition('generic', config, bootstrap=lambda _: {}, probe=lambda: {'device_count': 1})
    assert result['status'] == 'failed'
    assert result['error']['type'] == 'ValueError'


def test_cuda_probe_rejects_non_cuda_output():
    from types import SimpleNamespace

    from experiments.audit.w8_cuda_diagnostic import cuda_probe

    class FakeTensor:
        device = SimpleNamespace(type='cpu')

        def __mul__(self, _):
            return self

        def __add__(self, _):
            return self

    cuda = SimpleNamespace(device_count=lambda: 1, get_device_properties=lambda _: SimpleNamespace(
        name='mock', major=8, minor=9, total_memory=1), synchronize=lambda _: None)
    torch = SimpleNamespace(cuda=cuda, device=lambda _: 'cuda:0', float32='float32',
                            arange=lambda *args, **kwargs: FakeTensor())
    with pytest.raises(RuntimeError, match='did not execute on CUDA'):
        cuda_probe(torch)
