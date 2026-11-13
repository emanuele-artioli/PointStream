import copy
import hashlib
import stat

import pytest

from experiments.generative import gvcrt_peer_job_w8leaves as job
from experiments.generative import gvcrt_receiver_worker_w8leaves as worker
from experiments.tennis import w8_cuda_system_leaves as w8


def policy_fixture():
    c = {'runtime': '/safe/runtime', 'author': '/safe/author', 'bridge': '/safe/bridge',
         'entropy': '/safe/entropy', 'i_model': '/safe/i', 'p_model': '/safe/p',
         'wire': '/safe/wire', 'output': '/safe/output', 'device_paths': [],
         'parent_user': 'user-parent', 'parent_net': 'net-parent',
         'policy_profile': worker.POLICY,
         'loader_identity': {'device': 1, 'inode': 2, 'mode': stat.S_IFREG | 0o644, 'uid': 0},
         'denied_paths': {'source': ['/secret'], 'aliases': ['/alias']}}
    leaves = []
    for index, name in enumerate(worker.APPROVED_LEAVES):
        leaves.append({'name': name, 'resolved': name.replace('/proc/self/', '/proc/1234/'),
                       'device': 1, 'inode': 2 if index == 0 else 10 + index,
                       'mode': stat.S_IFREG | 0o644, 'uid': 65534})
    roots = ['/safe/runtime', '/safe/author', '/safe/bridge', '/safe/entropy',
             '/safe/i', '/safe/p', '/safe/wire', '/safe/output'] + [x['resolved'] for x in leaves]
    receipt = {'policy_profile': worker.POLICY, 'approved_leaves': leaves, 'pid': 1234,
               'landlock_abi': 3, 'handled_fs_rights': (1 << 15) - 1,
               'mutation_denials': dict.fromkeys(('write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'), 13),
               'output_write_probe': True,
               'parent_canary_post_sha256': dict.fromkeys(('write', 'truncate', 'remove', 'rename', 'link'), hashlib.sha256(b'canary').hexdigest()),
               'denials': {'/secret': 13, '/alias': 13}, 'namespaces': {'user': 'user-child', 'net': 'net-child'},
               'network_interfaces': ['lo'], 'pre_sandbox_threads': 1, 'extra_fds': False, 'cuda_preinitialized': False,
               'cpu_before_sha256': 'x', 'cpu_after_sha256': 'x', 'allow_roots': roots,
               'protected_identities': {'source': [{'path': '/secret', 'resolved': '/secret', 'device': 3, 'inode': 8}],
                                        'aliases': [{'path': '/alias', 'resolved': '/secret', 'device': 3, 'inode': 8}]}}
    return c, receipt


def test_policy_uses_exact_previously_qualified_leaf_set():
    assert worker.APPROVED_LEAVES == w8.APPROVED_LEAVES == job.APPROVED_LEAVES
    assert len(worker.APPROVED_LEAVES) == 12
    assert '/proc' not in worker.APPROVED_LEAVES and '/sys' not in worker.APPROVED_LEAVES


def test_receipt_requires_child_self_inode_and_strict_confinement():
    config, receipt = policy_fixture()
    job.check_isolation(receipt, config)
    bad = copy.deepcopy(receipt)
    bad['approved_leaves'][6]['resolved'] = '/proc/999/maps'
    with pytest.raises(ValueError, match='leaf identity'):
        job.check_isolation(bad, config)
    bad = copy.deepcopy(receipt)
    bad['mutation_denials'].pop('truncate')
    with pytest.raises(ValueError, match='mutation'):
        job.check_isolation(bad, config)
    bad = copy.deepcopy(receipt)
    bad['approved_leaves'][0]['inode'] += 1
    with pytest.raises(ValueError, match='loader'):
        job.check_isolation(bad, config)
    bad = copy.deepcopy(receipt)
    bad['policy_profile'] = 'old-policy'
    with pytest.raises(ValueError, match='profile'):
        job.check_isolation(bad, config)


def test_approved_leaf_identity_refuses_protected_hardlink(tmp_path, monkeypatch):
    leaf = tmp_path / 'leaf'
    leaf.write_bytes(b'not a source body')
    monkeypatch.setattr(worker, 'APPROVED_LEAVES', (str(leaf),))
    c = {'denied_paths': {'source': [str(leaf)]}, 'loader_identity':
         {'device': 1, 'inode': 2, 'mode': stat.S_IFREG | 0o644, 'uid': 0}}
    with pytest.raises(ValueError, match='aliases protected'):
        worker.approved_leaves(c)


def test_new_spec_preserves_single_point_budget_and_qualified_pool():
    from experiments.jobs.inbox import validate_spec
    spec = job.make_spec('2026-11-05T04:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 800
    assert spec['hosts'] == ['gpu5', 'gpu6'] and spec['gpu_models'] == ['RTX 6000 Ada']
    assert spec['scale']['frames'] == {'smoke': 4, 'full': 16}
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (200, 500, 30)
    assert spec['entrypoint'][-1].endswith('gvcrt_peer_job_w8leaves')
    staged = {item['name']: item['sha256'] for item in spec['staged_inputs']}
    assert staged['source'] == job.SOURCE_SHA and staged['entropy'] == job.ENTROPY_SHA
    assert staged['i_model'] == job.I_SHA and staged['p_model'] == job.P_SHA
    assert spec['environment']['sha256'] == job.RUNTIME_SHA
