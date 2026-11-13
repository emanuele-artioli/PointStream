"""CPU-only direct oracle for the GVC receiver's actual Landlock helpers.

No Torch, CUDA, model, source body or protected file body is read. This is not
an execution of the receiver bootstrap, and does not admit any model work.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import resource
import socket
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from experiments.generative import gvcrt_peer_job_w8leaves as peer
from experiments.generative import gvcrt_receiver_worker_w8leaves as worker

OLD_CONFIG_SHA = 'cda68dac4dea488195093c1580b00360f80e61712ce4e9e8f6f06a27c9191442'
OLD_CONFIG = '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261105T003105Z-26df2a36/smoke/gvcrt-peer/probe-config.json'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
SCOPE = 'GVC actual policy helpers only; no torch, CUDA, model or receiver bootstrap'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')


def validate_config(old: dict[str, Any]) -> None:
    denied = old.get('denied_paths', {})
    if set(denied) != worker.DENIALS or sum(len(v) for v in denied.values()) != 52:
        raise ValueError('the actual 26df protected and alias paths must all be present')
    groups = [k for k in denied if k != 'aliases']
    if any(not isinstance(denied[k], list) or not denied[k] for k in denied):
        raise ValueError('empty protected category')
    ids = {k: {(Path(p).stat().st_dev, Path(p).stat().st_ino) for p in v} for k, v in denied.items()}
    if any(ids[a] & ids[b] for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('protected categories overlap')
    if set.union(*(ids[k] for k in groups)) != ids['aliases']:
        raise ValueError('actual aliases do not cover protected identities')
    if len({p for paths in denied.values() for p in paths}) != 52:
        raise ValueError('duplicate denial path')


def loader_identity() -> dict[str, int]:
    path = Path('/etc/ld.so.cache').resolve(strict=True)
    info = path.stat()
    if str(path) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('unsafe host-parent loader cache')
    return {'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}


def child(config_path: Path, expected_sha: str) -> None:
    if sha(config_path) != expected_sha:
        raise ValueError('config identity differs')
    c = json.loads(config_path.read_text())
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('Linux one-core child required')
    namespaces = {n: os.readlink('/proc/self/ns/' + n) for n in ('user', 'net')}
    if (namespaces['user'] == c['parent_user'] or namespaces['net'] == c['parent_net']
            or sorted(name for _, name in socket.if_nameindex()) != ['lo']):
        raise ValueError('fresh user and network namespaces required')
    if len(os.listdir('/proc/self/task')) != 1 or any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise ValueError('child must have one thread and no extra descriptors')
    resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('this oracle must not import Torch or expose a GPU')
    validate_config(c)
    roots = [Path(c[k]).resolve(strict=True) for k in ('runtime', 'author', 'bridge', 'entropy', 'i_model', 'p_model', 'wire', 'output')]
    roots.extend(Path(p).resolve(strict=True) for p in c['device_paths'])
    leaves, leaf_receipt = worker.approved_leaves(c)
    roots.extend(leaves)
    ids = worker.identities(c)
    protected = {(r['device'], r['inode']) for group in ids.values() for r in group}
    worker.reject_hidden_hardlinks([Path(c['runtime']), Path(c['author']), Path(c['wire'])], protected)
    worker.reject_shared_system_filesystem([Path(p).resolve(strict=True) for p in c['device_paths'] if Path(p).is_dir()], protected)
    if any(Path(r['resolved']) == root or Path(r['resolved']).is_relative_to(root) for group in ids.values() for r in group for root in roots):
        raise ValueError('an allowed path exposes a protected input')
    abi, mask = worker.restrict(roots, Path(c['output']).resolve(strict=True))
    mutations = worker.mutation_probes(Path(c['canary_dir']), Path(c['output']))
    denied = {}
    for paths in c['denied_paths'].values():
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise RuntimeError('denial inconclusive: ' + path) from exc
                denied[path] = exc.errno
            else:
                os.close(fd)
                raise RuntimeError('protected open succeeded without reading: ' + path)
    save(Path(c['output']) / 'isolation.json', {'scope': SCOPE, 'policy_profile': worker.POLICY,
         'pid': os.getpid(), 'landlock_abi': abi, 'handled_fs_rights': mask, 'denials': denied,
         'mutation_denials': mutations, 'output_write_probe': True, 'namespaces': namespaces,
         'network_interfaces': ['lo'], 'protected_identities': ids, 'approved_leaves': leaf_receipt,
         'allow_roots': [str(p) for p in roots], 'torch_imported': False, 'gpu_visible': False})


def verify_receipt(c: dict[str, Any], receipt: dict[str, Any], canary: dict[str, str]) -> None:
    expected = {p for paths in c['denied_paths'].values() for p in paths}
    mutations = receipt['mutation_denials']
    if (receipt.get('scope') != SCOPE or receipt.get('policy_profile') != worker.POLICY
            or set(receipt['denials']) != expected or len(expected) != 52
            or any(x not in (errno.EACCES, errno.EPERM) for x in receipt['denials'].values())
            or receipt.get('landlock_abi', 0) < 3 or receipt['handled_fs_rights'] != worker.handled_rights(receipt['landlock_abi'])
            or set(mutations) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
            or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in mutations.items())
            or receipt.get('output_write_probe') is not True or receipt.get('torch_imported') is not False
            or receipt.get('gpu_visible') is not False or receipt.get('network_interfaces') != ['lo']
            or receipt['namespaces']['user'] == c['parent_user'] or receipt['namespaces']['net'] == c['parent_net']
            or set(canary) != {'write', 'truncate', 'remove', 'rename', 'link'}
            or any(value != hashlib.sha256(b'canary').hexdigest() for value in canary.values())):
        raise ValueError('GVC CPU policy receipt incomplete')
    expected_roots = [str(Path(c[k]).resolve()) for k in ('runtime', 'author', 'bridge', 'entropy', 'i_model', 'p_model', 'wire', 'output')]
    expected_roots.extend(str(Path(p).resolve()) for p in c['device_paths'])
    leaves = receipt['approved_leaves']
    if [x['name'] for x in leaves] != list(worker.APPROVED_LEAVES):
        raise ValueError('qualified 12 system leaves differ')
    pid = receipt.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('child pid missing')
    if c['loader_identity']['uid'] != 0 or c['loader_identity']['mode'] & stat.S_IWOTH:
        raise ValueError('host-parent loader identity unsafe')
    for leaf in leaves:
        expected_path = leaf['name'].replace('/proc/self/', f'/proc/{pid}/')
        if leaf['resolved'] != expected_path or not stat.S_ISREG(leaf['mode']):
            raise ValueError('system leaf identity differs')
        expected_roots.append(leaf['resolved'])
        if leaf['name'] == '/etc/ld.so.cache' and any(leaf[k] != c['loader_identity'][k] for k in ('device', 'inode', 'mode')):
            raise ValueError('loader inode differs from parent')
    if receipt['allow_roots'] != expected_roots:
        raise ValueError('actual policy grant set differs')
    ids = receipt['protected_identities']
    if set(ids) != set(c['denied_paths']) or any([x['path'] for x in ids[k]] != v for k, v in c['denied_paths'].items()):
        raise ValueError('protected identities differ')
    alias_ids = {(x['device'], x['inode']) for x in ids['aliases']}
    if any((x['device'], x['inode']) not in alias_ids for k, group in ids.items() if k != 'aliases' for x in group):
        raise ValueError('protected alias identity missing')


def run(old_path: Path) -> None:
    if (os.environ.get('PS_STAGE') != 'smoke' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or len(os.sched_getaffinity(0)) != 1 or not socket.gethostname().startswith('gpu6')):
        raise ValueError('CPU smoke stage required')
    if sha(old_path) != OLD_CONFIG_SHA:
        raise ValueError('old configuration SHA differs')
    old = json.loads(old_path.read_text())
    validate_config(old)
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'gvc-policy-cpu'
    work.mkdir()
    out, canary = work / 'output', work / 'canary'
    out.mkdir()
    peer.make_canaries(canary)
    config = {**old, 'output': str(out), 'canary_dir': str(canary), 'device_paths': peer.device_paths(),
              'policy_profile': worker.POLICY, 'loader_identity': loader_identity(),
              'parent_user': os.readlink('/proc/self/ns/user'), 'parent_net': os.readlink('/proc/self/ns/net')}
    path = work / 'config.json'
    save(path, config)
    command = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', sys.executable, '-m',
               'experiments.generative.gvcrt_policy_cpu_oracle', 'child', '--config', str(path), '--sha256', sha(path)]
    started = time.monotonic()
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=45)
    if result.returncode != 0:
        raise RuntimeError('isolated policy child failed: ' + result.stderr.decode(errors='replace')[-2000:])
    receipt = json.loads((out / 'isolation.json').read_text())
    hashes = peer.check_canaries(canary)
    verify_receipt(config, receipt, hashes)
    publication = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    publication.mkdir()
    save(publication / 'cpu-policy.json', {'scope': SCOPE, 'passed': True, 'old_config_sha256': OLD_CONFIG_SHA,
         'config': config, 'isolation': receipt, 'parent_canary_sha256': hashes,
         'child_wall_seconds': time.monotonic() - started,
         'limitation': 'No Torch bootstrap, CUDA, native model or scoring was run.'})


def validate(stage: Path, target: Path) -> None:
    try:
        with tarfile.open(stage / 'published.tar') as t:
            members = [m for m in t if m.name == 'publish/cpu-policy.json' and m.isfile()]
            if len(members) != 1 or members[0].size > 500_000:
                raise ValueError('policy publication missing')
            stream = t.extractfile(members[0])
            if stream is None:
                raise ValueError('policy publication unreadable')
            result = json.load(stream)
        if result['scope'] != SCOPE or result['old_config_sha256'] != OLD_CONFIG_SHA or result['passed'] is not True:
            raise ValueError('policy publication identity differs')
        verify_receipt(result['config'], result['isolation'], result['parent_canary_sha256'])
        save(target, {'passed': True, 'scope': SCOPE, 'checks': {'actual_policy_receipt': True, '52_denials': True,
              '12_leaves': True, 'mutation_canaries': True, 'host_parent_loader': True}})
    except Exception as exc:
        save(target, {'passed': False, 'scope': SCOPE, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'child', 'validate', 'marker'))
    parser.add_argument('--old-config', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.mode == 'run':
        if args.old_config is None:
            parser.error('old config required')
        run(args.old_config)
    elif args.mode == 'child':
        if args.config is None or args.sha256 is None:
            parser.error('child config and hash required')
        child(args.config, args.sha256)
    elif args.mode == 'validate':
        validate(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['PS_VALIDATION_PATH']))
    else:
        out = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        out.mkdir()
        save(out / 'marker.json', {'scope': SCOPE, 'marker_only': True})


if __name__ == '__main__':
    main()
