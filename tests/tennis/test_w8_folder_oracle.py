import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.jobs.inbox import validate_spec
from experiments.tennis import w8_folder_oracle as oracle


def test_spec_is_cpu_only_bounded_and_uses_real_oracle():
    spec = oracle.make_spec('2026-11-05T07:05:00Z')
    validate_spec(spec, now=1793859600)
    assert spec['device'] == 'cpu' and spec['gpu_models'] == []
    assert spec['budget_seconds'] == 70 and spec['cpu_threads'] == 1
    assert spec['arguments'] == ['{phase}']
    assert spec['scale']['phase'] == {'smoke': 'run', 'full': 'marker'}
    assert spec['validator'][-1] == 'validate'


def test_validator_rejects_unproven_symlink_denial(tmp_path, monkeypatch):
    stage = tmp_path / 'smoke'
    stage.mkdir()
    root = Path(oracle.__file__).resolve().parents[2]
    names = ['experiments/tennis/w8_folder_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
             'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
             'experiments/tennis/k6b_generate.py']
    hashes = {n: oracle.sha(root / n) for n in names}
    with tarfile.open(tmp_path / 'source.tar', 'w') as tar:
        for n in names:
            tar.add(root / n, arcname=n)
    (stage / 'dispatch.json').write_text(json.dumps({'code': {'snapshot_sha256': oracle.sha(tmp_path / 'source.tar')}}))
    rows = {}
    for name in oracle.materialize.SIZES:
        body = oracle.expected_bytes(name)
        h = hashlib.sha256(body).hexdigest()
        rows[name] = {'source_sha256': h, 'copy_sha256': h, 'bytes': len(body)}
    result = {'scope': 'synthetic CPU-only strict policy; no models or real I/O benchmark',
              'profile': oracle.materialize.PROFILE, 'exit_code': 0, 'cuda_visible_devices': '', 'no_torch': True,
              'cleanup_complete': True, 'output_write_ok': True, 'generator_sha256': oracle.GENERATOR_SHA,
              'symlink_errno': 13, 'worker_cpu_seconds': 1, 'child_cpu_seconds': 1, 'parent_cpu_seconds': 1,
              'supervisor_wall_seconds': 1, 'denial_errnos': [13] * 10, 'landlock_abi': 3,
              'handled_mask': oracle.policy.handled_rights(3),
              'mutation_denials': dict.fromkeys(('write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'), 13),
              'namespaces': {'user': 'child', 'net': 'child'}, 'parent_ns': {'user': 'parent', 'net': 'parent'},
              'approved_leaves': [], 'loader': {}, 'materialization': {'files': rows, 'bytes': sum(r['bytes'] for r in rows.values())},
              'code_sha256': hashes}
    def publish():
        body = json.dumps(result).encode()
        with tarfile.open(stage / 'published.tar', 'w') as tar:
            info = tarfile.TarInfo('publish/folder-oracle.json')
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': oracle.sha(stage / 'published.tar')}}))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(oracle.resource, 'setrlimit', lambda *a: None)
    monkeypatch.setattr(oracle.policy, 'validate_leaf_receipt', lambda *a: None)
    publish()
    oracle.validate()
    assert json.loads((tmp_path / 'validation.json').read_text())['passed']
    result['symlink_errno'] = 0
    publish()
    with pytest.raises(ValueError, match='strict policy'):
        oracle.validate()
    assert not json.loads((tmp_path / 'validation.json').read_text())['passed']


def test_loader_receipt_uses_parent_root_identity_and_child_inode_check():
    # Host root can be unmapped in the child user namespace; inode/mode still match.
    loader = {'uid': 0, 'mode': 0o100644, 'device': 58, 'inode': 123}
    leaves = []
    for name in oracle.policy.APPROVED_LEAVES:
        resolved = name.replace('/proc/self/', '/proc/101/')
        leaves.append({'name': name, 'resolved': resolved, 'uid': 65534, 'mode': 0o100644,
                       'device': 58, 'inode': 123})
    oracle.policy.validate_leaf_receipt(leaves, loader)
    leaves[0]['inode'] = 124
    with pytest.raises(ValueError, match='loader cache'):
        oracle.policy.validate_leaf_receipt(leaves, loader)
    leaves[0]['inode'] = 123
    loader['uid'] = 65534
    with pytest.raises(ValueError, match='loader cache'):
        oracle.policy.validate_leaf_receipt(leaves, loader)


def test_worker_rejects_unbound_loader_config_before_bootstrap(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='config identity'):
        oracle.worker(config, '0' * 64)
