"""Real qualification APIs with synthetic files; no Linux/codec/model execution."""
import argparse
import errno
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
from typing import Any

import pytest

from experiments.tennis import k5d_ada_scale_adapter as adapter, k5d_ada_witness as witness
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_isolated_receiver as isolated
from experiments.tennis import k5d_pgid_oracle as pgid, k5d_scoped_namespace as namespace
from experiments.tennis import k5d_source_oracle as source, k5d_svt_oracle as svt
from src.pointstream import ball, ball_stream

IDENTITY = {key: char * 64 for key, char in zip(
    ('source_sha256', 'snapshot_sha256', 'environment_sha256'), '123', strict=True)}


def save(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True))
    return {'local': str(path), 'remote': str(cf.DATA_ROOT / 'test-qualification' / path.name), 'sha256': source.sha(path)}


def receipt(context: dict, config_sha: str, profile: dict) -> dict:
    """Synthetic namespace evidence exercises the unchanged production verifier."""
    ns = {k: k + '-inner' for k in ('user', 'pid', 'mnt', 'ipc')}
    denials = {k: [{'errno': errno.EACCES, 'path': '/test-private', 'inode': 2, 'sha256': 'a' * 64}]
               for k in isolated.CATEGORIES}
    campaign = {'host_pid': 20, 'host_pgid': 40, 'campaign_pgid': 40, 'start_ticks': 11}
    launcher = {'pid': 25, 'host_ppid': 20, 'host_pgid': 40, 'birth': '1', 'namespace_pids': [25, 2]}
    observer = {'birth': '2', 'namespaces': ns, 'pid': 30, 'host_ppid': 25, 'host_pgid': 40,
                'namespace_pids': [1], 'launcher_observer': launcher}
    private = {'private_proc': True, 'initial_pids': [1], 'namespaces': ns,
               'parent_namespaces': {k: k + '-outer' for k in ns}, 'outer_pid_hidden': 20,
               'mountinfo': '1 2 0:3 / /proc - proc proc rw'}
    native = {'namespace_pids': [10], 'namespaces': ns, 'birth': '3', 'pid': 31,
              'host_ppid': 30, 'host_pgid': 40, 'nonce': 'a' * 32}
    proof: dict[str, Any] = {'context': context, 'config_sha256': config_sha, 'landlock_abi': 3,
        'handled_mask': isolated.leaves.handled_rights(3), 'threads': 1, 'extra_fds': False,
        'pid': 1, 'postexec_pid': 1, 'birth': '2', 'postexec_birth': '2', 'nonce': 'a' * 32,
        'profile_sha256': isolated._object_sha(profile),
        'runtime_inventory_sha256': isolated._object_sha(profile['runtime_files']),
        'denials': denials, 'namespaces': private, 'campaign_host': campaign,
        'launcher_observer': launcher, 'worker_observer': observer,
        'cpu_only': True, 'cuda_visible_devices': '',
        'decoder': {'pid': 10, 'returncode': 0, 'started_monotonic_ns': 1, 'argv': ['/native'],
            'executable_resolved': '/native', 'argv_sha256': isolated._object_sha(['/native']),
            'outer_observation': native, 'preexec': {'pid': 10, 'birth': '3', 'observer': native,
                'argv_sha256': isolated._object_sha(['/native']), 'closed_extra_fds': True,
                'cuda_visible_devices': ''}}}
    proof['actual_policy'] = {'cpu_only': True, 'private_proc_read_only': True, 'namespaces': private,
        'execute_files': ['/native'], 'context': context, 'pid': 1, 'birth': '2',
        'handled_mask': proof['handled_mask'], 'campaign_host': campaign, 'launcher_observer': launcher,
        'denials': {k: [{x: y for x, y in row.items() if x != 'errno'} for row in rows]
                   for k, rows in denials.items()}}
    proof['actual_policy_sha256'] = isolated._object_sha(proof['actual_policy'])
    return {'scope': isolated.SCOPE, 'codec': 'svt-av1-ivf', 'isolation': proof,
            **{k: 'b' * 64 for k in ('wire_sha256', 'resource_map_sha256', 'native_sha256',
                                     'decoded_yuv_sha256', 'rendered_rgb_sha256')}}


@pytest.fixture
def qualification(tmp_path: Path) -> tuple[dict[str, dict[str, str]], argparse.Namespace, dict]:
    profile: dict[str, Any] = {'runtime_files': {}}
    config = save(tmp_path / 'config.json', {'schema': 1, 'execution_identity': IDENTITY,
                                           'profiles': {'gpu6': profile}})
    bridge = save(tmp_path / 'bridge.json', {'approved': True, 'consumer_execution_identity': IDENTITY,
        'calibration_execution_identity': adapter.REFERENCE_IDENTITY,
        'consumer_snapshot_sha256': IDENTITY['snapshot_sha256']})
    native = save(tmp_path / 'native.json', {'schema': 1, 'audited': True, 'passed': True,
        'scope': 'native CPU PID containment and accounting', 'gpu_used': False,
        'consumer_execution_identity': IDENTITY, 'namespace_wrapper_sha256': source.sha(Path(namespace.__file__)),
        'unshare_sha256': '4' * 64, 'job_id': '20261113T000000Z-12345678',
        'all_owned_pids_absent': True, 'independent_cpu_match': True,
        'cases': ['normal', 'timeout', 'sigterm', 'sigkill']})
    context = {'job_id': '20261113T000000Z-12345678', 'stage': 'smoke',
               'execution_identity': IDENTITY, 'gpu': {'kind': 'cpu'}}
    received = receipt(context, config['sha256'], profile)
    denial = save(tmp_path / 'denial.json', {'schema': 1, 'approved': True,
        'actual_filesystem_denial_passed': True, 'consumer_execution_identity': IDENTITY,
        'config_sha256': config['sha256'], 'publication_sha256': '5' * 64, 'codec': 'svt-av1-ivf',
        'host': 'gpu6', 'producer_context': context, 'receipt': received,
        'actual_policy_sha256': received['isolation']['actual_policy_sha256']})
    assets = {'bridge': bridge, 'receiver_isolation_config': config, 'native_cpu_receipt': native,
              'cpu_denial': denial}
    args = argparse.Namespace(**{k: v['local'] for k, v in assets.items()},
                             **{k + '_sha256': v['sha256'] for k, v in assets.items()}, unshare_sha256='4' * 64)
    return assets, args, received


def test_source_admission_uses_real_cpu_receipt_verifier(qualification) -> None:
    _, args, _ = qualification
    svt.source_admission(args, IDENTITY)


@pytest.mark.parametrize('damage', ['missing_mode', 'visible_native', 'wrong_codec', 'wrong_context'])
def test_source_admission_rejects_legacy_and_corrupt_receipts(tmp_path, qualification, damage) -> None:
    _, args, _ = qualification
    record = json.loads(Path(args.cpu_denial).read_text())
    if damage == 'missing_mode':
        record['receipt']['isolation'].pop('cpu_only')
    elif damage == 'visible_native':
        record['receipt']['isolation']['decoder']['preexec']['cuda_visible_devices'] = 'GPU-leak'
    elif damage == 'wrong_codec':
        record['receipt']['codec'] = 'dcvc-uf'
    else:
        record['receipt']['isolation']['context'] = {}
    item = save(tmp_path / 'changed-denial.json', record)
    args.cpu_denial, args.cpu_denial_sha256 = item['local'], item['sha256']
    with pytest.raises(ValueError):
        svt.source_admission(args, IDENTITY)


@pytest.mark.parametrize('enabled', [False, True])
def test_real_constructor_dispatch_and_cpu_api_reach_receiver_mode(
        tmp_path, monkeypatch, qualification, enabled) -> None:
    assets, _, _ = qualification
    assets = {k: v for k, v in assets.items() if k != 'cpu_denial'}
    private = tmp_path / 'private'
    private.write_bytes(b'test private')
    assets['protected'] = save(tmp_path / 'protected.json', {k: [str(private)] for k in isolated.CATEGORIES})
    header = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 16, 12, 30, 1, 2, 0)
    native = header + b''.join(struct.pack('<IQ', 1, t) + b'x' for t in range(2))
    packet = ball_stream.Stream(16, 12, Fraction(30), 2, 'svt-av1-ivf', 'test', native,
                                ball.encode_frames([None, None], [None, None], 16, 12))
    wire = tmp_path / 'wire.psbl'
    wire.write_bytes(ball_stream.pack(packet))
    assets['wire'] = {'local': str(wire), 'remote': str(cf.DATA_ROOT / 'test-qualification' / wire.name), 'sha256': source.sha(wire)}
    assets['resources'] = save(tmp_path / 'resources.json', {})
    options = dict(assets=assets, deadline='2099-01-01T00:00:00+00:00', unshare_sha256='4' * 64,
                   identity=IDENTITY, expected_yuv_sha256='b' * 64, expected_rgb_sha256='b' * 64)
    base = json.loads(cf.BASE.read_text())
    legacy = pgid.spec(base, **options)
    spec = pgid.spec(base, cpu_only_receiver=enabled, **options)
    if not enabled:
        assert spec == legacy
    assert ('--cpu-only-receiver' in spec['arguments']) is enabled
    tail = spec['arguments'][spec['arguments'].index('{proof_mode}') + 1:]
    tail = [next((row['local'] for name, row in assets.items() if arg == '{staged:' + name + '}'), arg)
            for arg in tail]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    command = namespace.worker_command(['ada-denial-cpu', *tail])
    stage = tmp_path / 'stage'
    stage.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (stage / 'dispatch.json').write_text(json.dumps({'code': IDENTITY,
        'staged_environment': {'sha256': IDENTITY['environment_sha256']}}))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setattr(sys, 'argv', command[2:])
    calls = []
    previous = sys.getprofile()
    def trace(frame, event, arg):
        if event == 'call' and frame.f_code is isolated.replay.__code__:
            calls.append(frame.f_locals['cpu_only'])
        if previous:
            previous(frame, event, arg)
    # Stop at the OS boundary; all feature APIs above it are the real functions.
    monkeypatch.setattr(sys, 'platform', 'qualification-test-no-linux')
    sys.setprofile(trace)
    try:
        with pytest.raises(RuntimeError, match='Linux and one CPU'):
            witness.main()
    finally:
        sys.setprofile(previous)
    assert calls == [enabled]


def test_cpu_option_cannot_select_gpu_witness(monkeypatch) -> None:
    monkeypatch.setattr(sys, 'argv', ['witness', 'witness', '--cpu-only-receiver'])
    with pytest.raises(ValueError, match='CPU qualification'):
        witness.main()
    with pytest.raises(ValueError, match='explicit CPU-oracle'):
        witness.spec({}, action='witness', assets={}, deadline='', unshare_sha256='', identity={},
                     cpu_only_receiver=True)
