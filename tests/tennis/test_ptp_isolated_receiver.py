"""CPU-only contract tests; real kernel qualification remains a Linux fleet stage."""
from copy import deepcopy
import errno
import inspect
from types import SimpleNamespace

import pytest

from experiments.tennis import ptp_isolated_receiver as ptp


def proof(patch=True):
    namespaces = {k: k + '-new' for k in ('user', 'pid', 'mnt', 'ipc')}
    observer = {'birth': '100', 'namespace_pids': [34, 1], 'namespaces': namespaces}
    roles = {r: [{'path': '/' + r, 'sha256': 'a' * 64, 'size': 1}] for r in ptp.ROLES}
    denied = {c: [] for c in ptp.kernel.CATEGORIES}
    groups = {'staged_source_archive': 'source_archive', 'source_yuv': 'encoder_state'}
    for role, rows in roles.items():
        denied[groups.get(role, role)].append({**rows[0], 'inode': len(denied) + 1, 'errno': errno.EACCES})
    actual_policy = {'denials': {k: [{a: b for a, b in row.items() if a != 'errno'} for row in rows] for k, rows in denied.items()}}
    calls = []
    for argv in (['/bin/dav1d', '--version'], ['/bin/dav1d', '-i', 'base.ivf'], *([['/bin/dav1d', '-i', 'patch.ivf']] if patch else [])):
        calls.append({'index': len(calls), 'argv': argv, 'argv_sha256': ptp.kernel._object_sha(argv), 'returncode': 0,
                      'restriction_scope': 'inherits restricted worker Landlock and namespaces'})
    details = {'context': {}, 'policy_sha256': 'b' * 64, 'worker_observer': observer, 'pid': 1, 'birth': '100',
               'namespace': {'namespaces': namespaces, 'parent_namespaces': {k: k + '-old' for k in namespaces},
                             'private_proc': True, 'initial_pids': [1]},
               'threads_before_lock': 1, 'extra_fds_before_lock': False, 'cuda_visible_devices': '',
               'landlock_abi': 3, 'handled_mask': ptp.kernel.leaves.handled_rights(3),
               'environment': {'PATH': '/nonexistent', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1',
                               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                               'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'},
               'runtime_archive_sha256': ptp.RUNTIME_SHA, 'network_isolation': False, 'mcp_isolation': False,
               'native_scope': 'inherited restrictions; no independent native birth observation',
               'actual_policy': actual_policy, 'actual_policy_sha256': ptp.kernel._object_sha(actual_policy),
               'denials': denied, 'protected_roles': roles, 'native_calls': calls, 'decoder_path': '/bin/dav1d',
               'profile_sha256': 'e' * 64, 'runtime_inventory_sha256': 'e' * 64, 'source_inventory_sha256': 'e' * 64}
    return {'schema': ptp.SCHEMA, 'scope': ptp.SCOPE, 'filesystem_proof': details,
            'decode': {'packet_sha256': 'c' * 64, 'decoded_frames': 2, 'rgb_frame_sha256': ['d' * 64] * 2,
                       'patch_bytes': 10 if patch else 0}}, observer


def verify(receipt, observer):
    ptp.verify_receipt(receipt, context={}, policy_sha='b' * 64, packet_sha='c' * 64, observer=observer)


@pytest.mark.parametrize('patch', [False, True])
def test_separate_native_calls_and_explicit_limited_scope(patch):
    receipt, observer = proof(patch)
    verify(receipt, observer)
    receipt['filesystem_proof']['native_calls'].pop()
    with pytest.raises(ValueError, match='invocation'):
        verify(receipt, observer)


@pytest.mark.parametrize('invalid', [errno.ENOENT, errno.EINVAL, None, 0])
def test_receipt_rejects_missing_or_inconclusive_denial(invalid):
    receipt, observer = proof()
    receipt['filesystem_proof']['denials']['source_frame'][0]['errno'] = invalid
    with pytest.raises(ValueError, match='denial'):
        verify(receipt, observer)


def test_receipt_rejects_role_relabeling_and_wrong_runtime():
    receipt, observer = proof()
    receipt['filesystem_proof']['protected_roles']['source_frame'][0]['path'] = '/different'
    with pytest.raises(ValueError, match='linked'):
        verify(receipt, observer)
    receipt, observer = proof()
    receipt['filesystem_proof']['runtime_archive_sha256'] = 'z' * 64
    with pytest.raises(ValueError, match='incomplete'):
        verify(receipt, observer)


def test_actual_hashes_and_granted_hardlink_or_recursive_output_fail(tmp_path):
    target = tmp_path / 'source'
    target.write_bytes(b'actual authenticated input')
    row = {'path': str(target), 'sha256': ptp.kernel.sha(target), 'size': target.stat().st_size}
    roles = {key: [deepcopy(row)] for key in ptp.ROLES}
    evidence = ptp.protected_evidence(roles)
    alias = tmp_path / 'alias'
    alias.hardlink_to(target)
    with pytest.raises(ValueError, match='inode'):
        ptp.reject_protected_grants(evidence, [alias], tmp_path / 'out')
    with pytest.raises(ValueError, match='recursively'):
        ptp.reject_protected_grants(evidence, [], tmp_path)
    roles['source_frame'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        ptp.protected_evidence(roles)


def test_recorded_decoder_gets_only_public_inputs_and_restores_runner(monkeypatch, tmp_path):
    observed = []
    def runner(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ptp.subprocess, 'run', runner)
    def decoder(packet, output, *, scratch, decoder):
        assert packet == tmp_path / 'packet' and output == tmp_path / 'rgb'
        assert scratch == tmp_path
        for command in ([decoder, '--version'], [decoder, '-i', 'base'], [decoder, '-i', 'patch']):
            ptp.subprocess.run(command)
        return {'ok': True}
    result, calls = ptp.recorded_decode(tmp_path / 'packet', tmp_path / 'rgb', tmp_path, '/bin/dav1d', decoder)
    assert result == {'ok': True} and len(calls) == 3 and len(observed) == 3
    assert ptp.subprocess.run is runner
    assert 'expected' not in inspect.signature(ptp.recorded_decode).parameters
    def bad(*args, **kwargs):
        ptp.subprocess.run(['/bin/undeclared'])
    with pytest.raises(ValueError, match='undeclared'):
        ptp.recorded_decode(tmp_path / 'packet', tmp_path / 'rgb', tmp_path, '/bin/dav1d', bad)
    assert ptp.subprocess.run is runner


def test_policy_amendment_keeps_generic_inventory_and_hashes_code(tmp_path):
    for name in ptp.DECODER_SOURCE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# decoder code\n')
    original = {'profiles': {'gpu6': {'source_files': {}, 'runtime_files': {'numpy.so': {'sha256': 'a', 'size': 1}},
                                       'executables': [{'path': '/bin/python'}]}}}
    frozen = deepcopy(original)
    amended = ptp.amend_policy(original, tmp_path, {'environment_sha256': ptp.RUNTIME_SHA})
    assert original == frozen
    assert amended['profiles']['gpu6']['runtime_files'] == original['profiles']['gpu6']['runtime_files']
    assert set(amended['profiles']['gpu6']['source_files']) == set(ptp.DECODER_SOURCE)
    with pytest.raises(ValueError, match='runtime'):
        ptp.amend_policy(original, tmp_path, {'environment_sha256': 'wrong'})


def test_receipt_rejects_inherited_credential_environment():
    receipt, observer = proof()
    receipt['filesystem_proof']['environment']['API_TOKEN'] = 'unexpected'
    with pytest.raises(ValueError, match='environment'):
        verify(receipt, observer)
