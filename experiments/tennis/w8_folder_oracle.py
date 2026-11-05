"""CPU-only, synthetic strict-Landlock oracle for W8 public regular folders."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

from experiments.tennis import k6b_generate, w8_cuda_system_leaves as policy
from experiments.tennis import w8_public_folder_materialize as materialize
from experiments.tennis import w8_receiver_bootstrap as bootstrap

GENERATOR_SHA = '00248aa4fdd47e634f82b37a7a33c99f1794440d65cbfe9395255a4225735542'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def expected_bytes(name: str) -> bytes:
    return ('pointstream-synthetic-folder-oracle-v1:' + name).encode()


def parent_loader_identity(path: Path) -> dict[str, int]:
    if str(path.resolve(strict=True)) != '/etc/ld.so.cache':
        raise ValueError('loader cache path differs')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('host loader cache must be root-owned and non-world-writable')
    return {'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}


def worker(config_path: Path, config_sha256: str) -> None:
    if sha(config_path) != config_sha256:
        raise ValueError('oracle config identity mismatch')
    config = json.loads(config_path.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('GPU-hidden, no-Torch process required')
    if len(os.sched_getaffinity(0)) != 1 or len(os.listdir('/proc/self/task')) != 1:
        raise ValueError('single-thread and CPU affinity required')
    if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise ValueError('extra inherited descriptor')
    ns = {n: os.readlink('/proc/self/ns/' + n) for n in ('user', 'net')}
    if any(ns[k] == config['parent_ns'][k] for k in ns) or sorted(n for _, n in socket.if_nameindex()) != ['lo']:
        raise ValueError('fresh user and network namespace required')
    output = Path(config['output'])
    canary = Path(config['canary'])
    denied = [Path(p) for p in config['denied']]
    if len(denied) != 10 or any(p.resolve().is_relative_to(output) for p in denied):
        raise ValueError('synthetic denied paths invalid')
    loader_receipt = config['loader_identity']
    extra, leaves = policy.approved_leaves(denied, loader_receipt)
    paths = [output, Path(sys.prefix), *(Path(p) for p in policy.SYSTEM_GRANTS),
             *(Path(x['path']) for x in config['inputs'].values()), *extra]
    sys.addaudithook(bootstrap._deny_processes)
    abi, mask = policy.restrict(paths, output)
    mutations = policy.mutation_probes(canary, output)
    probes = bootstrap._check_denials(denied)
    try:
        (output / 'forbidden-symlink').symlink_to(next(iter(config['inputs'].values()))['path'])
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM):
            raise
        symlink_errno = exc.errno
    else:
        raise RuntimeError('MAKE_SYM unexpectedly allowed')
    folders = output / 'scratch' / 'folders'
    result = materialize.copy_declared(config['inputs'], folders)
    weights = {name: Path(item['path']) for name, item in config['inputs'].items()}
    for family in ('vae', 'image_encoder'):
        path = k6b_generate.folder(weights, family, folders.parent)
        for name, item in config['inputs'].items():
            if name.startswith(family + '/'):
                copy = path / name.split('/', 1)[1]
                if copy.is_symlink() or sha(copy) != item['sha256']:
                    raise ValueError('real loader folder did not reuse regular files')
    materialize.cleanup_completed(folders, result)
    if any(p.is_file() or p.is_symlink() for p in folders.rglob('*')):
        raise ValueError('synthetic copied data not cleaned')
    receipt = {'scope': 'synthetic CPU-only strict policy; no models or real I/O benchmark',
               'profile': materialize.PROFILE, 'landlock_abi': abi, 'handled_mask': mask,
               'denial_errnos': sorted(probes.values()), 'mutation_denials': mutations,
               'symlink_errno': symlink_errno, 'namespaces': ns, 'parent_ns': config['parent_ns'],
               'loader': loader_receipt, 'approved_leaves': leaves, 'materialization': result,
               'generator_sha256': GENERATOR_SHA, 'output_write_ok': True, 'no_torch': 'torch' not in sys.modules,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cleanup_complete': True}
    receipt['worker_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    (output / 'oracle.json').write_text(json.dumps(receipt, sort_keys=True))


def run() -> None:
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux CPU-only fleet required')
    resource.setrlimit(resource.RLIMIT_CPU, (20, 30))
    if sha(Path(k6b_generate.__file__)) != GENERATOR_SHA:
        raise ValueError('original folder helper differs')
    root = Path(os.environ['PS_SCRATCH_DIR']) / 'folder-oracle'
    root.mkdir()
    inputs, output, canary, protected = (root / n for n in ('inputs', 'output', 'canary', 'protected'))
    for p in (inputs, output, canary, protected):
        p.mkdir()
    declared: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(materialize.SIZES):
        p = inputs / str(i)
        p.write_bytes(expected_bytes(name))
        declared[name] = {'path': str(p), 'sha256': sha(p), 'size': p.stat().st_size}
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (canary / name).write_bytes(b'canary')
    denied: list[str] = []
    for i in range(5):
        p = protected / str(i)
        p.write_bytes(b'synthetic protected canary')
        a = protected / f'alias-{i}'
        a.symlink_to(p)
        denied.extend((str(p), str(a)))
    loader_receipt = parent_loader_identity(Path('/etc/ld.so.cache'))
    config = {'inputs': declared, 'output': str(output), 'canary': str(canary), 'denied': denied,
              'loader_identity': loader_receipt,
              'parent_ns': {n: os.readlink('/proc/self/ns/' + n) for n in ('user', 'net')}}
    cfg = root / 'config.json'
    cfg.write_text(json.dumps(config, sort_keys=True))
    cmd = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', sys.executable,
           '-m', 'experiments.tennis.w8_folder_oracle', 'worker', '--config', str(cfg), '--config-sha256', sha(cfg)]
    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(cmd, cwd=env['PYTHONPATH'], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=35, check=False, preexec_fn=limit)
    if proc.returncode:
        raise RuntimeError('synthetic isolated oracle failed: ' + proc.stderr.decode(errors='replace')[-4000:])
    result = json.loads((output / 'oracle.json').read_text())
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['supervisor_wall_seconds'] = time.monotonic() - started
    result['child_cpu_seconds'] = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    result['parent_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    result['code_sha256'] = {name: sha(Path(__file__).resolve().parents[2] / name) for name in
                             ('experiments/tennis/w8_folder_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
                              'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
                              'experiments/tennis/k6b_generate.py')}
    result['exit_code'] = proc.returncode
    pub = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    pub.mkdir()
    (pub / 'folder-oracle.json').write_text(json.dumps(result, sort_keys=True))


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    result_path = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('validator must be CPU-only')
        staging = json.loads((stage / 'staging.json').read_text())
        if sha(stage / 'published.tar') != staging['published']['sha256']:
            raise ValueError('published oracle archive differs')
        with tarfile.open(stage / 'published.tar') as archive:
            members = archive.getmembers()
            items = [m for m in members if m.isfile()]
            if (len(items) != 1 or items[0].name != 'publish/folder-oracle.json'
                    or any(not m.isfile() and not (m.isdir() and m.name.rstrip('/') == 'publish') for m in members)):
                raise ValueError('oracle archive membership differs')
            stream = archive.extractfile(items[0])
            if stream is None:
                raise ValueError('oracle receipt missing')
            result = json.load(stream)
        if (result['scope'] != 'synthetic CPU-only strict policy; no models or real I/O benchmark'
                or result['profile'] != materialize.PROFILE or result['exit_code'] != 0
                or result['cuda_visible_devices'] != '' or result['no_torch'] is not True
                or result['cleanup_complete'] is not True or result['output_write_ok'] is not True
                or result['generator_sha256'] != GENERATOR_SHA or result['symlink_errno'] not in (1, 13)
                or not 0 <= result['worker_cpu_seconds'] <= 30 or not 0 <= result['child_cpu_seconds'] <= 30 or not 0 <= result['parent_cpu_seconds'] <= 20
                or not 0 < result['supervisor_wall_seconds'] <= 35
                or len(result['denial_errnos']) != 10 or any(x not in (1, 13) for x in result['denial_errnos'])
                or result['landlock_abi'] < 3 or result['handled_mask'] != policy.handled_rights(result['landlock_abi'])
                or set(result['mutation_denials']) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
                or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in result['mutation_denials'].items())
                or any(result['namespaces'][k] == result['parent_ns'][k] for k in ('user', 'net'))):
            raise ValueError('strict policy or CPU-only oracle receipt differs')
        policy.validate_leaf_receipt(result['approved_leaves'], result['loader'])
        rows = result['materialization']['files']
        if set(rows) != set(materialize.SIZES):
            raise ValueError('four synthetic folders required')
        for name, row in rows.items():
            expected = expected_bytes(name)
            if row['source_sha256'] != hashlib.sha256(expected).hexdigest() or row['copy_sha256'] != row['source_sha256'] or row['bytes'] != len(expected):
                raise ValueError('synthetic bytes differ')
        if result['materialization']['bytes'] != sum(len(expected_bytes(name)) for name in rows):
            raise ValueError('synthetic byte accounting differs')
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        source = stage.parent / 'source.tar'
        if sha(source) != dispatch['code']['snapshot_sha256']:
            raise ValueError('frozen source archive differs')
        code_files = {'experiments/tennis/w8_folder_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
                      'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
                      'experiments/tennis/k6b_generate.py'}
        if set(result['code_sha256']) != code_files:
            raise ValueError('oracle code membership differs')
        with tarfile.open(source) as archive:
            for name, digest in result['code_sha256'].items():
                f = archive.extractfile(name)
                if f is None or hashlib.sha256(f.read()).hexdigest() != digest:
                    raise ValueError('oracle source identity differs')
        if result['code_sha256']['experiments/tennis/k6b_generate.py'] != GENERATOR_SHA:
            raise ValueError('real unchanged folder helper differs')
        receipt = {'passed': True, 'scope': result['scope'], 'checks': {'real_folder_helper': True,
                   'strict_policy_synthetic_only': True, 'no_gpu_or_model': True, 'source_identity': True}}
    except Exception as exc:
        result_path.write_text(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    result_path.write_text(json.dumps(receipt) + '\n')


def make_spec(deadline: str) -> dict[str, Any]:
    base = Path(__file__).resolve().parents[2] / 'docs/experiments/w8-inputs/receiver-20261101/w8-strict-packet-cpu-v3-spec-20261105.json'
    spec = json.loads(base.read_text())
    selected = [x for x in spec['staged_inputs'] if x['name'] == 'identity']
    if len(selected) != 1 or selected[0]['sha256'] != '0953c5284ee8dc462d4eb95b7ce49cb9bad68e4d612fce5d167706f954668ebe':
        raise ValueError('historical identity input differs')
    spec['staged_inputs'] = selected
    spec['inputs'] = [{'path': selected[0]['path'], 'sha256': selected[0]['sha256']}]
    spec.update(entrypoint=['-m', 'experiments.tennis.w8_folder_oracle'],
                arguments=['{phase}'], scale={'phase': {'smoke': 'run', 'full': 'marker'}},
                smoke={'seconds': 45, 'representative_basis': 'Synthetic four-file materialization and real unchanged folder helper under the qualified strict Landlock policy, without model or GPU.'},
                full={'seconds': 5}, validator=['{python}', '-m', 'experiments.tennis.w8_folder_oracle', 'validate'],
                validator_seconds=15, budget_seconds=70, stall_seconds=45, deadline=deadline)
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'validate', 'marker', 'make-spec'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--config-sha256')
    parser.add_argument('--deadline')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'make-spec':
        if args.deadline is None or args.output is None:
            parser.error('make-spec requires --deadline and --output')
        from experiments.jobs.inbox import validate_spec
        spec = validate_spec(make_spec(args.deadline))
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2)
            stream.write('\n')
    elif args.mode == 'worker':
        if args.config is None or args.config_sha256 is None:
            parser.error('worker requires --config and --config-sha256')
        worker(args.config, args.config_sha256)
    elif args.mode == 'run':
        run()
    elif args.mode == 'validate':
        validate()
    else:
        pub = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        pub.mkdir()
        (pub / 'marker.json').write_text(json.dumps({'scope': 'synthetic CPU-only oracle'}) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
