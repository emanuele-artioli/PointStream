"""Unit tests only: these do not establish a Linux isolation witness."""
import errno
import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from experiments.tennis import k5d_isolated_receiver as isolated


def row(path: Path) -> dict[str, object]:
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': path.stat().st_size}


def test_inventory_rejects_escape_and_changed_bytes(tmp_path: Path) -> None:
    root = tmp_path / 'source'
    root.mkdir()
    item = root / 'worker.py'
    item.write_bytes(b'code')
    assert isolated._manifest(root, {'worker.py': row(item)}) == [item]
    with pytest.raises(ValueError):
        isolated._manifest(root, {'../other': row(item)})
    old = row(item)
    item.write_bytes(b'evil')
    with pytest.raises(ValueError):
        isolated._manifest(root, {'worker.py': old})
    alias = root / 'alias'
    alias.symlink_to(item)
    with pytest.raises(ValueError):
        isolated._manifest(root, {'alias': row(item)})


def test_protected_requires_real_positive_files(tmp_path: Path) -> None:
    target = tmp_path / 'source'
    target.write_bytes(b'private')
    paths = {category: [target] for category in isolated.CATEGORIES}
    evidence, flat = isolated._protected(paths)
    assert len(flat) == 5 and evidence['source_frame'][0]['sha256'] == row(target)['sha256']
    paths['ground_truth'] = [tmp_path / 'missing']
    with pytest.raises(ValueError):
        isolated._protected(paths)


def test_receipt_rejects_legacy_and_incomplete_proof() -> None:
    with pytest.raises(ValueError):
        isolated.verify_receipt({'scope': 'fresh packet-path replay; filesystem isolation NOT established'},
                                expected_context={}, expected_config_sha256='a' * 64)
    proof: dict[str, Any] = {'context': {}, 'config_sha256': 'a' * 64, 'landlock_abi': 3,
             'handled_mask': isolated.leaves.handled_rights(3), 'threads': 1, 'extra_fds': False,
             'pid': 1, 'postexec_pid': 1, 'birth': '2', 'postexec_birth': '2',
             'decoder': {'pid': 10, 'returncode': 0, 'started_monotonic_ns': 1,
                         'argv': ['/native'], 'executable_resolved': '/native',
                         'argv_sha256': isolated._object_sha(['/native'])},
             'profile_sha256': 'a' * 64, 'runtime_inventory_sha256': 'b' * 64,
             'actual_policy': {}, 'actual_policy_sha256': isolated._object_sha({}),
             'denials': {k: [{'errno': errno.EACCES, 'path': '/real', 'inode': 2, 'sha256': 'a' * 64}]
                         for k in isolated.CATEGORIES}}
    ns = {k: k + '-inner' for k in ('user', 'pid', 'mnt', 'ipc')}
    proof['namespaces'] = {'private_proc': True, 'initial_pids': [1], 'namespaces': ns,
                           'parent_namespaces': {k: k + '-outer' for k in ns},
                           'outer_pid_hidden': 20, 'mountinfo': '1 2 0:3 / /proc - proc proc rw'}
    proof['campaign_host'] = {'host_pid': 20, 'host_pgid': 40, 'campaign_pgid': 40, 'start_ticks': 11}
    proof['launcher_observer'] = {'pid': 25, 'host_ppid': 20, 'host_pgid': 40, 'birth': '1', 'namespace_pids': [25, 2]}
    proof['worker_observer'] = {'birth': '2', 'namespaces': ns, 'pid': 30, 'host_ppid': 25, 'host_pgid': 40,
                                'namespace_pids': [1], 'launcher_observer': proof['launcher_observer']}
    proof['nonce'] = 'a' * 32
    proof['decoder']['outer_observation'] = {'namespace_pids': [10], 'namespaces': ns, 'birth': '3', 'pid': 31, 'host_ppid': 30, 'host_pgid': 40, 'nonce': 'a' * 32}
    proof['decoder']['preexec'] = {'pid': 10, 'birth': '3', 'observer': proof['decoder']['outer_observation'],
                                    'argv_sha256': proof['decoder']['argv_sha256'], 'closed_extra_fds': True}
    proof['actual_policy'] = {'private_proc_read_only': True, 'namespaces': proof['namespaces'],
                              'execute_files': ['/native'], 'context': {}, 'pid': 1, 'birth': '2', 'handled_mask': proof['handled_mask'],
                              'campaign_host': proof['campaign_host'], 'launcher_observer': proof['launcher_observer'],
                              'denials': {k: [{x: y for x, y in row.items() if x != 'errno'} for row in rows]
                                          for k, rows in proof['denials'].items()}}
    proof['actual_policy_sha256'] = isolated._object_sha(proof['actual_policy'])
    receipt = {'scope': isolated.SCOPE, 'isolation': proof,
               **{k: 'b' * 64 for k in ('wire_sha256', 'resource_map_sha256', 'native_sha256',
                                        'decoded_yuv_sha256', 'rendered_rgb_sha256')}}
    assert isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64) is proof
    with pytest.raises(ValueError, match='CPU-only'):
        isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64, require_cpu_only=True)
    receipt['codec'] = 'svt-av1-ivf'
    proof.update(cpu_only=True, cuda_visible_devices='')
    proof['actual_policy']['cpu_only'] = True
    proof['actual_policy_sha256'] = isolated._object_sha(proof['actual_policy'])
    proof['decoder']['preexec']['cuda_visible_devices'] = ''
    assert isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64, require_cpu_only=True) is proof
    proof['decoder']['preexec']['cuda_visible_devices'] = 'GPU-leak'
    with pytest.raises(ValueError, match='CPU-only'):
        isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64, require_cpu_only=True)
    proof['decoder']['preexec']['cuda_visible_devices'] = ''
    proof['decoder']['outer_observation']['host_pgid'] = 41
    with pytest.raises(ValueError, match='native decoder observation'):
        isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64)
    proof['decoder']['outer_observation']['host_pgid'] = 40
    proof['decoder']['outer_observation']['namespace_pids'] = [11]
    with pytest.raises(ValueError):
        isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64)
    proof['decoder']['outer_observation']['namespace_pids'] = [10]
    proof['denials']['source_frame'][0]['errno'] = errno.ENOENT
    with pytest.raises(ValueError):
        isolated.verify_receipt(receipt, expected_context={}, expected_config_sha256='a' * 64)


def test_source_root_requires_frozen_token() -> None:
    assert isolated._source_root({'source_root': '$FROZEN_SOURCE'}) == Path(isolated.__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        isolated._source_root({'source_root': '/untrusted/source'})


def test_cpu_oracle_rejects_non_cpu_before_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='CPU producer'):
        isolated.cpu_oracle(tmp_path / 'missing', tmp_path / 'map', tmp_path / 'work',
                            wire_sha256='a' * 64, map_sha256='b' * 64,
                            expected_yuv_sha256='c' * 64, expected_rgb_sha256='d' * 64,
                            config_path=tmp_path / 'config', config_sha256='e' * 64,
                            protected_paths={}, producer_context={'gpu': {'uuid': 'GPU-test'}})


def test_effective_mount_handles_overmounted_proc() -> None:
    old = '5698 1 0:20 / /proc rw - proc proc rw'
    new = '5781 1 0:21 / /proc rw - proc proc rw'
    assert isolated._effective_proc_mount([old, new], 5781, (0, 21)) == new
    with pytest.raises(ValueError):
        isolated._effective_proc_mount([old, new], 5781, (0, 20))


def test_child_translation_uses_host_parent_and_namespace(monkeypatch: Any) -> None:
    original = Path.read_text
    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == '/proc/3392605/task/3392605/children':
            return '3392606 3392607'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    rows = {3392606: {'pid': 3392606, 'namespace_pids': [3392606, 2], 'namespaces': {'pid': 'pid:[4]'}},
            3392607: {'pid': 3392607, 'namespace_pids': [3392607, 3], 'namespaces': {'pid': 'pid:[4]'}}}
    monkeypatch.setattr(isolated, '_proc_identity', lambda pid: rows[pid])
    assert isolated._translate_child(2, 3392605, 'pid:[4]')['pid'] == 3392606
    with pytest.raises(ValueError):
        isolated._translate_child(2, 3392605, 'pid:[wrong]')


def test_native_wrapper_waits_for_matching_birth_and_nonce(tmp_path: Path, monkeypatch: Any) -> None:
    import json
    namespace = {'pid': 'pid:[inner]'}
    observer = {'nonce': 'secret', 'namespace_pids': [2], 'birth': '55', 'namespaces': namespace, 'host_pgid': 40}
    (tmp_path / 'native-observer.json').write_text(json.dumps(observer))
    payload = tmp_path / 'native-exec.json'
    payload.write_text(json.dumps({'argv': ['/native', '--decode'], 'env': {'PATH': '/none'}, 'nonce': 'secret', 'campaign_pgid': 40}))
    monkeypatch.setattr(isolated, '_birth', lambda: '55')
    monkeypatch.setattr(isolated, '_namespaces', lambda _: namespace)
    monkeypatch.setattr(isolated.os, 'getpid', lambda: 2)
    monkeypatch.setattr(isolated.os, 'listdir', lambda _: ['0', '1', '2'])
    called = []
    monkeypatch.setattr(isolated.os, 'execve', lambda *args: called.append(args))
    isolated._native_wrapper(payload)
    record = json.loads((tmp_path / 'native-preexec.json').read_text())
    assert record['observer'] == observer and record['pid'] == 2 and record['closed_extra_fds'] is True
    assert called == [('/native', ['/native', '--decode'], {'PATH': '/none'})]
    observer['nonce'] = 'wrong'
    (tmp_path / 'native-observer.json').write_text(json.dumps(observer))
    with pytest.raises(ValueError):
        isolated._native_wrapper(payload)


def test_import_roots_are_canonical_packed_dirs_and_read_dir_only(tmp_path: Path, monkeypatch: Any) -> None:
    import ctypes

    source = tmp_path / 'source'
    runtime = tmp_path / 'runtime'
    imports = runtime / 'lib' / 'python3.12'
    for directory in (source, imports):
        directory.mkdir(parents=True)
    monkeypatch.setattr(isolated, '_source_root', lambda _: source)
    profile = {'runtime_root': str(runtime), 'runtime_prefix': str(runtime), 'runtime_import_paths': [str(imports)]}
    roots = isolated._import_directories(profile)
    assert roots == [source, imports]
    protected = tmp_path / 'protected'
    protected.write_text('private')
    allowed = runtime / 'module.py'
    allowed.write_text('public')
    output = tmp_path / 'output'
    output.mkdir()
    opened: dict[int, Path] = {}
    grants: list[tuple[Path, int]] = []
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    class FakeLib:
        def syscall(self, number, *args):
            if number == 444:
                return 3
            if number == 445:
                rule = ctypes.cast(args[2], ctypes.POINTER(Rule)).contents
                grants.append((opened[rule.parent_fd], rule.allowed_access))
            return 0
        def prctl(self, *args): return 0
    def fake_open(path, flags):
        descriptor = len(opened) + 20
        opened[descriptor] = Path(path)
        return descriptor
    monkeypatch.setattr(isolated.ctypes, 'CDLL', lambda *a, **kw: FakeLib())
    monkeypatch.setattr(isolated.os, 'O_PATH', 0x200000, raising=False)
    monkeypatch.setattr(isolated.os, 'open', fake_open)
    monkeypatch.setattr(isolated.os, 'close', lambda _: None)
    monkeypatch.setattr(isolated.leaves, 'handled_rights', lambda _: 1)
    monkeypatch.setattr(isolated.leaves, 'allowed_rights', lambda *a: 2)
    isolated._restrict([allowed], [], [], output, roots)
    assert (source, 8) in grants and (imports, 8) in grants
    assert all(bits == 8 for path, bits in grants if path in (source, imports))
    assert all(path != protected for path, _ in grants)
    assert (allowed, 4) in grants
    outside = tmp_path / 'outside'
    outside.mkdir()
    with pytest.raises(ValueError, match='packed runtime'):
        isolated._import_directories({**profile, 'runtime_import_paths': [str(outside)]})


def test_launcher_cleanup_only_signals_owned_child_and_reaps() -> None:
    events: list[str] = []
    class Process:
        def poll(self):
            events.append('poll')
            return None
        def kill(self):
            events.append('kill-child')
        def wait(self, timeout):
            assert timeout == 10
            events.append('wait-child')
    isolated._stop_launcher(cast(Any, Process()))
    assert events == ['poll', 'kill-child', 'wait-child']


def test_launcher_cleanup_reaps_already_exited_child() -> None:
    events: list[str] = []
    class Process:
        def poll(self): return 1
        def kill(self): raise AssertionError('must not signal exited child')
        def wait(self, timeout): events.append('wait')
    isolated._stop_launcher(cast(Any, Process()))
    assert events == ['wait']


def test_receiver_inherits_group_and_monitor_keeps_foreign_processes(monkeypatch: Any) -> None:
    from types import SimpleNamespace
    from experiments.jobs import monitor

    calls: list[tuple] = []
    monkeypatch.setattr(isolated.subprocess, 'Popen', lambda command, **kwargs: calls.append((command, kwargs)))
    isolated._launch_receiver(['/unshare', '--kill-child=SIGKILL'], {'PATH': '/none'})
    command, kwargs = calls[0]
    assert command == ['/unshare', '--kill-child=SIGKILL']
    assert kwargs.get('start_new_session', False) is False and kwargs['close_fds'] is True
    monkeypatch.setattr(monitor.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout='GPU-test, 501\nGPU-test, 502\nGPU-test, 999\n'))
    groups = {501: 70, 502: 70, 999: 99}
    monkeypatch.setattr(monitor.os, 'getpgid', lambda pid: groups[pid])
    assert monitor.foreign_gpu_processes('GPU-test', 70) == [999]


def test_cpu_only_native_exec_refuses_visible_device_before_wait(tmp_path,monkeypatch):
    import json
    payload = tmp_path/'native-exec.json'
    payload.write_text(json.dumps({'cpu_only':True,'env':{'CUDA_VISIBLE_DEVICES':''}}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-parent')
    with pytest.raises(ValueError,match='visible CUDA'):
        isolated._native_wrapper(payload)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    payload.write_text(json.dumps({'cpu_only':True,'env':{'CUDA_VISIBLE_DEVICES':'GPU-leaked'}}))
    with pytest.raises(ValueError,match='visible CUDA'):
        isolated._native_wrapper(payload)
