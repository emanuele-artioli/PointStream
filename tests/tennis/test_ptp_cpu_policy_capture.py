"""Explicit-path capture contracts; no remote or namespace execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.tennis import ptp_cpu_policy_capture as capture


def base_policy():
    def record(path):
        return {'path': path, 'sha256': 'a' * 64, 'size': 100}
    profile = {'runtime_root': capture.PACKED, 'runtime_prefix': capture.PACKED,
               'runtime_import_paths': [capture.PACKED + '/lib/python3.12'],
               'runtime_files': {'numpy/file.so': {'sha256': 'b' * 64, 'size': 123}},
               'source_root': '$FROZEN_SOURCE', 'source_files': {'old.py': {'sha256': 'c' * 64, 'size': 50}},
               'python': capture.PACKED_EXECUTABLES[0], 'unshare': record('/usr/bin/unshare'),
               'executables': [record(p) for p in (*capture.PACKED_EXECUTABLES, capture.LOADER)],
               'system_files': [record(p) for p in (*capture.OS_LIBRARIES, *capture.DROP_SYSTEM)],
               'system_leaves': [{'path': p, 'kind': 'character' if p.startswith('/dev/') else 'regular'}
                                 for p in (*capture.CPU_LEAVES, *capture.DROP_LEAVES)]}
    return {'schema': 1, 'execution_identity': {'old': 'frozen'}, 'profiles': {'gpu6': profile}}


def install_stubs(monkeypatch, base):
    data = json.dumps(base).encode()
    monkeypatch.setattr(capture, 'BASE_SHA', capture.digest(data))
    reads = []
    def identity(path):
        reads.append(path)
        return {'path': path, 'logical_path': path, 'resolved_path': path,
                'sha256': 'a' * 64 if path in capture.PACKED_EXECUTABLES else 'd' * 64, 'size': 100}
    monkeypatch.setattr(capture, 'file_identity', identity)
    monkeypatch.setattr(capture, 'check_leaf', lambda row: reads.append(row['path']))
    monkeypatch.setattr(capture, 'kernel_metadata', lambda: {'landlock_abi': 4, 'errno': 0, 'kernel_release': '6.8'})
    return data, reads


def test_strict_subset_preserves_entire_packed_runtime_and_recaptures_unshare(monkeypatch):
    base = base_policy()
    frozen = deepcopy(base)
    data, reads = install_stubs(monkeypatch, base)
    result = capture.capture(data, 'gpu1')
    assert base == frozen
    profile = result['profiles']['gpu1']
    original = base['profiles']['gpu6']
    for key in ('runtime_files', 'source_files', 'runtime_root', 'runtime_prefix', 'runtime_import_paths', 'python'):
        assert profile[key] == original[key]
    assert profile['unshare']['sha256'] == 'd' * 64 != original['unshare']['sha256']
    assert {r['logical_path'] for r in profile['system_files']} == set(capture.OS_LIBRARIES)
    assert {r['path'] for r in profile['system_leaves']} == set(capture.CPU_LEAVES)
    assert not (set(reads) & (set(capture.DROP_SYSTEM) | set(capture.DROP_LEAVES)))
    assert set(reads) == set(capture.OS_LIBRARIES) | set(capture.PACKED_EXECUTABLES) | set(capture.CPU_LEAVES) | {'/usr/bin/unshare'}


@pytest.mark.parametrize('change', ['new_system_path', 'new_leaf', 'new_executable', 'new_unshare'])
def test_unreviewed_roles_stop_before_capture_reads(monkeypatch, change):
    base = base_policy()
    profile = base['profiles']['gpu6']
    if change == 'new_system_path':
        profile['system_files'].append({'path': '/unreviewed/library'})
    elif change == 'new_leaf':
        profile['system_leaves'].append({'path': '/proc/new-resource'})
    elif change == 'new_executable':
        profile['executables'].append({'path': '/bin/sh'})
    else:
        profile['unshare']['path'] = '/unreviewed/unshare'
    data, reads = install_stubs(monkeypatch, base)
    with pytest.raises(ValueError, match='reviewed'):
        capture.capture(data, 'gpu1')
    assert reads == []


def test_wrong_host_base_or_packed_binary_cannot_be_promoted(monkeypatch):
    base = base_policy()
    data, _ = install_stubs(monkeypatch, base)
    with pytest.raises(ValueError, match='hash'):
        capture.capture(data + b'\n', 'gpu1')
    with pytest.raises(ValueError, match='host'):
        capture.capture(data, 'unreviewed-host')
    real = capture.file_identity
    def substituted(path):
        row = real(path)
        if path == capture.PACKED_EXECUTABLES[1]:
            row['sha256'] = 'e' * 64
        return row
    monkeypatch.setattr(capture, 'file_identity', substituted)
    with pytest.raises(ValueError, match='runtime replacement'):
        capture.capture(data, 'gpu1')


def test_explicit_logical_os_role_can_resolve_host_version_and_later_retarget_fails(monkeypatch, tmp_path):
    # The production resolver permits only OS roots; this test exercises the
    # postcapture comparison without reading any host libraries.
    logical = capture.LOADER
    captured_target = '/usr/lib/x86_64-linux-gnu/ld-2.31.so'
    row = {'path': captured_target, 'logical_path': logical, 'resolved_path': captured_target}
    config = {'profiles': {'gpu1': {'system_files': [row], 'executables': [],
                                    'unshare': {'path': '/usr/bin/unshare', 'logical_path': '/usr/bin/unshare',
                                                'resolved_path': '/usr/bin/unshare'}}}}
    targets = {logical: captured_target, '/usr/bin/unshare': '/usr/bin/unshare'}
    monkeypatch.setattr(Path, 'resolve', lambda self, strict=True: Path(targets[str(self)]))
    capture.verify_links(config, host='gpu1')
    targets[logical] = '/usr/lib/x86_64-linux-gnu/ld-new.so'
    with pytest.raises(ValueError, match='retargeted'):
        capture.verify_links(config, host='gpu1')


def test_merge_keeps_host_specific_libc_and_namespace_utility_identities(monkeypatch):
    data, _ = install_stubs(monkeypatch, base_policy())
    one, three = capture.capture(data, 'gpu1'), capture.capture(data, 'gpu3')
    three['profiles']['gpu3']['unshare']['sha256'] = 'e' * 64
    three['cpu_derivation']['namespace_utility'] = deepcopy(three['profiles']['gpu3']['unshare'])
    three['profiles']['gpu3']['system_files'][0]['sha256'] = 'f' * 64
    three['profiles']['gpu3']['executables'][-1]['sha256'] = 'f' * 64
    result = capture.merge([one, three], data)
    assert list(result['profiles']) == ['gpu1', 'gpu3']
    assert result['profiles']['gpu3']['unshare']['sha256'] == 'e' * 64
    changed = deepcopy(three)
    changed['profiles']['gpu3']['runtime_files']['new.so'] = {'sha256': 'e' * 64, 'size': 1}
    with pytest.raises(ValueError, match='inventory'):
        capture.merge([one, changed], data)


def test_no_landlock_stops_without_claiming_qualification(monkeypatch):
    data, _ = install_stubs(monkeypatch, base_policy())
    monkeypatch.setattr(capture, 'kernel_metadata', lambda: {'landlock_abi': -1, 'errno': 38})
    with pytest.raises(ValueError, match='Landlock'):
        capture.capture(data, 'gpu5')


def test_capture_follows_only_declared_logical_os_role_and_binds_actual_target(monkeypatch):
    import io
    import stat
    from types import SimpleNamespace
    payload = b'actual different host loader bytes'
    logical = capture.LOADER
    target = '/usr/lib/x86_64-linux-gnu/ld-2.31.so'
    monkeypatch.setattr(Path, 'resolve', lambda self, strict=True: Path(target))
    metadata = SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(payload), st_dev=1, st_ino=2, st_mtime_ns=3)
    monkeypatch.setattr(Path, 'stat', lambda self: metadata)
    monkeypatch.setattr(Path, 'open', lambda self, mode: io.BytesIO(payload))
    row = capture.file_identity(logical)
    assert row == {'path': target, 'logical_path': logical, 'resolved_path': target,
                   'size': len(payload), 'sha256': capture.digest(payload)}
    monkeypatch.setattr(Path, 'resolve', lambda self, strict=True: Path('/unreviewed/loader'))
    with pytest.raises(ValueError, match='outside'):
        capture.file_identity(logical)


@pytest.mark.parametrize('change', ['runtime', 'source', 'imports', 'GPU_library', 'GPU_leaf', 'shell',
                                   'loader_mismatch', 'other_OS_family', 'private_target', 'unshare', 'packed_hash', 'host'])
def test_pure_derivation_validator_rejects_resource_broadening(monkeypatch, change):
    data, _ = install_stubs(monkeypatch, base_policy())
    config = capture.capture(data, 'gpu1')
    profile = config['profiles']['gpu1']
    if change == 'runtime':
        profile['runtime_files']['model.pt'] = {'sha256': 'e' * 64, 'size': 100}
    elif change == 'source':
        profile['source_files']['encoder.py'] = {'sha256': 'e' * 64, 'size': 100}
    elif change == 'imports':
        profile['runtime_import_paths'].append('/home/private')
    elif change == 'GPU_library':
        profile['system_files'].append({'logical_path': capture.OS_ROOT + 'libcuda.so.535.183.01'})
    elif change == 'GPU_leaf':
        profile['system_leaves'].append({'path': '/dev/nvidia0', 'kind': 'character'})
    elif change == 'shell':
        profile['executables'].append({'logical_path': '/bin/sh'})
    elif change == 'loader_mismatch':
        profile['executables'][-1]['sha256'] = 'e' * 64
    elif change == 'other_OS_family':
        profile['system_files'][1]['path'] = capture.OS_ROOT + 'libcuda.so.535.183.01'
        profile['system_files'][1]['resolved_path'] = profile['system_files'][1]['path']
    elif change == 'private_target':
        profile['system_files'][1]['path'] = '/home/private/libc.so.6'
        profile['system_files'][1]['resolved_path'] = profile['system_files'][1]['path']
    elif change == 'unshare':
        profile['unshare']['path'] = '/usr/bin/bash'
        profile['unshare']['resolved_path'] = '/usr/bin/bash'
    elif change == 'packed_hash':
        profile['executables'][0]['sha256'] = 'e' * 64
    else:
        config['profiles']['unreviewed'] = config['profiles'].pop('gpu1')
    with pytest.raises(ValueError):
        capture.validate_derivation(config, data)


def test_pure_validation_allows_declared_host_versions_and_does_not_touch_files(monkeypatch):
    data, _ = install_stubs(monkeypatch, base_policy())
    one, three = capture.capture(data, 'gpu1'), capture.capture(data, 'gpu3')
    profile = three['profiles']['gpu3']
    loader = profile['system_files'][0]
    loader.update(path=capture.OS_ROOT + 'ld-2.31.so', resolved_path=capture.OS_ROOT + 'ld-2.31.so', sha256='e' * 64)
    profile['executables'][-1] = deepcopy(loader)
    profile['system_files'][1].update(path=capture.OS_ROOT + 'libc-2.31.so', resolved_path=capture.OS_ROOT + 'libc-2.31.so')
    profile['unshare']['sha256'] = 'f' * 64
    three['cpu_derivation']['namespace_utility'] = deepcopy(profile['unshare'])
    merged = capture.merge([one, three], data)
    monkeypatch.setattr(Path, 'resolve', lambda *a, **kw: pytest.fail('pure validator read filesystem'))
    monkeypatch.setattr(Path, 'stat', lambda *a, **kw: pytest.fail('pure validator read filesystem'))
    capture.validate_derivation(merged, data)
