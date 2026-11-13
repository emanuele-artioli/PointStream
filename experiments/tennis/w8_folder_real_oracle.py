"""CPU-only, real public-file strict-Landlock W8 folder I/O oracle."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import socket
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

from experiments.tennis import k6b_generate, w8_cuda_system_leaves as policy
from experiments.tennis import w8_public_folder_materialize as materialize
from experiments.tennis import w8_receiver_bootstrap as bootstrap
from experiments.tennis.w8_receiver_preflight import PROTECTED

GENERATOR_SHA = '00248aa4fdd47e634f82b37a7a33c99f1794440d65cbfe9395255a4225735542'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


INPUT_NAMES = {'w0': 'vae/config.json', 'w1': 'vae/diffusion_pytorch_model.safetensors',
               'w2': 'image_encoder/config.json', 'w3': 'image_encoder/pytorch_model.bin'}
IDENTITY_SHA = '0953c5284ee8dc462d4eb95b7ce49cb9bad68e4d612fce5d167706f954668ebe'
SYNTHETIC_SCOPE = 'synthetic CPU-only strict policy; no models or real I/O benchmark'
PROOF_NAMES = ('proof_validation', 'proof_gate', 'proof_state', 'proof_ready', 'proof_published', 'proof_source')


def stage_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('device') != 'cpu' or spec.get('gpu_models') != [] or spec.get('gpu_memory_mib') != 0
            or spec.get('hosts') != ['gpu6'] or spec.get('cpu_threads') != 1
            or spec.get('single_cpu_campaign') is not True or spec.get('budget_seconds') != 180
            or spec.get('environment', {}).get('sha256') != '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
            or staging.get('environment', {}).get('sha256') != spec['environment']['sha256']
            or len(dispatch.get('cpu_affinity', [])) != 1):
        raise ValueError('one-core gpu6 CPU-only 180-second campaign required')
    job_id = spec.get('synthetic_reference_job_id', '')
    if not re.fullmatch(r'20[0-9]{6}T[0-9]{6}Z-[0-9a-f]{8}', job_id):
        raise ValueError('actual completed synthetic reference job required')
    proof_root = '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/' + job_id + '/'
    suffixes = {'proof_validation': 'validation.json', 'proof_gate': 'gate.json',
                'proof_state': 'state.json', 'proof_ready': 'ready.json',
                'proof_published': 'smoke/published.tar', 'proof_source': 'source.tar'}
    declared = {x['name']: x for x in spec['staged_inputs']}
    staged = {x['name']: x for x in staging['inputs']}
    wanted = set(INPUT_NAMES) | {'identity'} | set(PROOF_NAMES)
    if (len(declared) != len(spec['staged_inputs']) or len(staged) != len(staging['inputs'])
            or set(declared) != wanted or set(staged) != wanted):
        raise ValueError('exact public files and synthetic proof set required')
    reference_spec = json.loads((Path(__file__).resolve().parents[2] / 'docs/experiments/w8-inputs/receiver-20261101/w8-strict-host-cost-v2-spec-20261105.json').read_text())
    pinned_sources = {x['name']: (x['path'], x['sha256']) for x in reference_spec['staged_inputs'] if x['name'] in set(INPUT_NAMES) | {'identity'}}
    if (set(pinned_sources) != set(INPUT_NAMES) | {'identity'}
            or {k: (declared[k]['path'], declared[k]['sha256']) for k in pinned_sources} != pinned_sources):
        raise ValueError('original public input paths or hashes differ')
    if (len(spec['inputs']) != len(declared)
            or {(x['path'], x['sha256']) for x in spec['inputs']} != {(x['path'], x['sha256']) for x in declared.values()}):
        raise ValueError('fleet input declarations differ')
    root = Path(staging['local_root']).resolve(strict=True)
    paths = {}
    for name, item in staged.items():
        source = declared[name]
        expected = source['sha256']
        if (item['sha256'] != expected or item.get('source') != source['path']
                or source.get('extract', False) or item.get('extract', False)):
            raise ValueError('staged input identity differs: ' + name)
        path = Path(item['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('input is not staged on the local root')
        if name in INPUT_NAMES:
            key = INPUT_NAMES[name]
            if (expected != materialize.PINNED[key] or path.stat().st_size != materialize.SIZES[key]
                    or item.get('mode') != 'local-hit'):
                raise ValueError('public weight identity or size differs: ' + name)
        else:
            if sha(path) != expected:
                raise ValueError('receipt bytes differ: ' + name)
        if name in suffixes and source['path'] != proof_root + suffixes[name]:
            raise ValueError('synthetic proof is not from selected job')
        paths[name] = path
    if declared['identity']['sha256'] != IDENTITY_SHA:
        raise ValueError('historical identity receipt differs')
    validation = json.loads(paths['proof_validation'].read_text())
    gate = json.loads(paths['proof_gate'].read_text())
    state = json.loads(paths['proof_state'].read_text())
    ready = json.loads(paths['proof_ready'].read_text())
    checks = {'real_folder_helper', 'strict_policy_synthetic_only', 'no_gpu_or_model', 'source_identity'}
    if (validation.get('passed') is not True or validation.get('scope') != SYNTHETIC_SCOPE
            or set(validation.get('checks', {})) != checks or any(v is not True for v in validation['checks'].values())
            or gate.get('passed') is not True or gate.get('validation') != validation
            or state.get('status') != 'complete' or state.get('job_id') != job_id
            or gate.get('spec_sha256') != ready.get('spec_sha256')
            or gate.get('source_sha256') != ready.get('source_sha256')
            or declared['proof_source']['sha256'] != ready.get('snapshot_sha256')):
        raise ValueError('synthetic reference has not passed its bound terminal gate')
    with tarfile.open(paths['proof_published']) as archive:
        files = [x for x in archive.getmembers() if x.isfile()]
        if len(files) != 1 or files[0].name != 'publish/folder-oracle.json':
            raise ValueError('synthetic published evidence differs')
        stream = archive.extractfile(files[0])
        if stream is None:
            raise ValueError('synthetic published receipt missing')
        prior = json.load(stream)
    if (prior.get('scope') != SYNTHETIC_SCOPE or prior.get('profile') != materialize.PROFILE
            or prior.get('cuda_visible_devices') != '' or prior.get('no_torch') is not True
            or prior.get('cleanup_complete') is not True or prior.get('output_write_ok') is not True
            or prior.get('symlink_errno') not in (1, 13) or prior.get('landlock_abi', 0) < 3
            or prior.get('handled_mask') != policy.handled_rights(prior['landlock_abi'])
            or len(prior.get('denial_errnos', [])) != 10 or any(e not in (1, 13) for e in prior['denial_errnos'])
            or set(prior.get('mutation_denials', {})) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
            or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in prior['mutation_denials'].items())
            or any(prior['namespaces'][k] == prior['parent_ns'][k] for k in ('user', 'net'))):
        raise ValueError('synthetic prerequisite isolation receipt differs')
    policy.validate_leaf_receipt(prior['approved_leaves'], prior['loader'])
    rows = prior['materialization']['files']
    if set(rows) != set(materialize.SIZES):
        raise ValueError('synthetic prerequisite file set differs')
    if prior['materialization'].get('bytes') != sum(len(('pointstream-synthetic-folder-oracle-v1:' + name).encode()) for name in rows):
        raise ValueError('synthetic prerequisite total bytes differ')
    for name, row in rows.items():
        body = ('pointstream-synthetic-folder-oracle-v1:' + name).encode()
        digest = hashlib.sha256(body).hexdigest()
        if row['source_sha256'] != digest or row['copy_sha256'] != digest or row['bytes'] != len(body):
            raise ValueError('synthetic prerequisite bytes differ')
    expected_code = {'experiments/tennis/w8_folder_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
                     'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
                     'experiments/tennis/k6b_generate.py'}
    if (set(prior.get('code_sha256', {})) != expected_code
            or prior['code_sha256']['experiments/tennis/k6b_generate.py'] != GENERATOR_SHA):
        raise ValueError('synthetic prerequisite helper identities differ')
    with tarfile.open(paths['proof_source']) as archive:
        for name, digest in prior['code_sha256'].items():
            member = archive.getmember(name)
            stream = archive.extractfile(member) if member.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                raise ValueError('synthetic prerequisite source archive differs: ' + name)
    return paths, spec


def check_denied_without_read(paths: list[Path]) -> dict[str, int]:
    """Probe actual protected paths and aliases without ever reading their bodies."""
    denied = {}
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError('protected denial inconclusive') from exc
            denied[str(path)] = exc.errno
        else:
            os.close(fd)
            raise RuntimeError('protected input unexpectedly openable')
    return denied


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
        raise ValueError('protected denied paths invalid')
    loader_receipt = config['loader_identity']
    extra, leaves = policy.approved_leaves(denied, loader_receipt)
    paths = [output, Path(sys.prefix), *(Path(p) for p in policy.SYSTEM_GRANTS),
             *(Path(x['path']) for x in config['inputs'].values()), *extra]
    sys.addaudithook(bootstrap._deny_processes)
    abi, mask = policy.restrict(paths, output)
    mutations = policy.mutation_probes(canary, output)
    probes = check_denied_without_read(denied)
    try:
        (output / 'forbidden-symlink').symlink_to(next(iter(config['inputs'].values()))['path'])
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM):
            raise
        symlink_errno = exc.errno
    else:
        raise RuntimeError('MAKE_SYM unexpectedly allowed')
    folders = output / 'scratch' / 'folders'
    result = materialize.materialize_public({f'weight:{name}': {'path': item['path'], 'sha256': item['sha256']}
                                             for name, item in config['inputs'].items()}, folders, materialize.PINNED)
    weights = {name: Path(item['path']) for name, item in config['inputs'].items()}
    helper_start, helper_cpu = time.monotonic(), time.process_time()
    for family in ('vae', 'image_encoder'):
        path = k6b_generate.folder(weights, family, folders.parent)
        for name, item in config['inputs'].items():
            if name.startswith(family + '/'):
                copy = path / name.split('/', 1)[1]
                if copy.is_symlink() or sha(copy) != item['sha256']:
                    raise ValueError('real loader folder did not reuse regular files')
    result['helper_wall_seconds'] = time.monotonic() - helper_start
    result['helper_cpu_seconds'] = time.process_time() - helper_cpu
    cleanup_start, cleanup_cpu = time.monotonic(), time.process_time()
    materialize.cleanup_completed(folders, result)
    result['cleanup_wall_seconds'] = time.monotonic() - cleanup_start
    result['cleanup_cpu_seconds'] = time.process_time() - cleanup_cpu
    if any(p.is_file() or p.is_symlink() for p in folders.rglob('*')):
        raise ValueError('task-owned copied data not cleaned')
    receipt = {'scope': 'CPU-only strict policy public model-folder I/O; no model or GPU',
               'profile': materialize.PROFILE, 'landlock_abi': abi, 'handled_mask': mask,
               'denial_errnos': sorted(probes.values()), 'denied_paths': sorted(probes), 'mutation_denials': mutations,
               'symlink_errno': symlink_errno, 'namespaces': ns, 'parent_ns': config['parent_ns'],
               'loader': loader_receipt, 'approved_leaves': leaves, 'materialization': result,
               'generator_sha256': GENERATOR_SHA, 'output_write_ok': True, 'no_torch': 'torch' not in sys.modules,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cleanup_complete': True}
    receipt['worker_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    (output / 'oracle.json').write_text(json.dumps(receipt, sort_keys=True))


def remaining_gate(dispatch: dict[str, Any], phase_reserve: float, campaign_reserve: float) -> dict[str, float]:
    clock = dispatch['campaign_clock']
    now = time.monotonic()
    started, phase, end = clock['started'], clock['phase_started'], clock['budget_end']
    if not started <= phase <= now < end or end - started > 181:
        raise ValueError('campaign clock identity differs')
    phase_left, campaign_left = phase + 120 - now, end - now
    if phase_left < phase_reserve or campaign_left < campaign_reserve:
        raise ValueError('insufficient remaining time for copy, validation and release')
    return {'phase_seconds_remaining': phase_left, 'campaign_seconds_remaining': campaign_left,
            'phase_seconds_reserved': phase_reserve, 'campaign_seconds_reserved': campaign_reserve}


def run() -> None:
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux CPU-only fleet required')
    resource.setrlimit(resource.RLIMIT_CPU, (20, 90))
    if sha(Path(k6b_generate.__file__)) != GENERATOR_SHA:
        raise ValueError('original folder helper differs')
    stage = Path(os.environ['PS_STAGE_DIR'])
    staged, spec = stage_inputs(stage)
    local = Path('/local/users/emanuele/pointstream').resolve(strict=True)
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    datasets = Path('/home/itec/emanuele/Datasets').resolve(strict=True)
    if (not scratch.is_relative_to(local) or scratch.stat().st_dev != local.stat().st_dev
            or local.stat().st_dev == datasets.stat().st_dev):
        raise ValueError('scratch must be on the separate provisioned local filesystem')
    root = scratch / 'folder-real-oracle'
    root.mkdir()
    if shutil.disk_usage(root).free < 3 * 1024**3:
        raise ValueError('at least 3 GiB free scratch required')
    output, canary, aliases = (root / n for n in ('output', 'canary', 'aliases'))
    for path in (output, canary, aliases):
        path.mkdir()
    declared = {name: {'path': str(staged[key]), 'sha256': materialize.PINNED[name],
                       'size': materialize.SIZES[name]} for key, name in INPUT_NAMES.items()}
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (canary / name).write_bytes(b'canary')
    identity = json.loads(staged['identity'].read_text())['protected_inputs']
    denied: list[str] = []
    for group, (name, digest) in PROTECTED.items():
        path = Path(name)
        info = path.stat()
        if identity[group] != {'path': name, 'sha256': digest, 'device': info.st_dev, 'inode': info.st_ino}:
            raise ValueError('historical protected host identity differs: ' + group)
        alias = aliases / group
        alias.symlink_to(path)
        denied.extend((name, str(alias)))
    loader_receipt = parent_loader_identity(Path('/etc/ld.so.cache'))
    config = {'inputs': declared, 'output': str(output), 'canary': str(canary), 'denied': denied,
              'loader_identity': loader_receipt,
              'parent_ns': {n: os.readlink('/proc/self/ns/' + n) for n in ('user', 'net')}}
    cfg = root / 'config.json'
    cfg.write_text(json.dumps(config, sort_keys=True))
    cmd = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', sys.executable,
           '-m', 'experiments.tennis.w8_folder_real_oracle', 'worker', '--config', str(cfg), '--config-sha256', sha(cfg)]
    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    staging_cost = sum(float(x['seconds']) for x in staging['inputs']) + float(staging['environment']['seconds'])
    if not math.isfinite(staging_cost) or staging_cost < 0:
        raise ValueError('measured staging cost missing')
    restage_reserve = max(15.0, 1.5 * staging_cost)
    admission = remaining_gate(dispatch, 110, 140 + restage_reserve)
    admission['full_restage_reserve_seconds'] = restage_reserve
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(cmd, cwd=env['PYTHONPATH'], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=105, check=False, preexec_fn=limit)
    if proc.returncode:
        raise RuntimeError('real-file isolated oracle failed: ' + proc.stderr.decode(errors='replace')[-4000:])
    remaining_gate(dispatch, 5, 35 + restage_reserve)
    result = json.loads((output / 'oracle.json').read_text())
    result['admission'] = admission
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['supervisor_wall_seconds'] = time.monotonic() - started
    result['child_cpu_seconds'] = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    result['parent_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    result['code_sha256'] = {name: sha(Path(__file__).resolve().parents[2] / name) for name in
                             ('experiments/tennis/w8_folder_real_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
                              'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
                              'experiments/tennis/w8_receiver_preflight.py', 'experiments/tennis/k6b_generate.py')}
    result['exit_code'] = proc.returncode
    result['synthetic_reference_job_id'] = spec['synthetic_reference_job_id']
    result['source_file_sha256'] = {k: materialize.PINNED[v] for k, v in INPUT_NAMES.items()}
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
        paths, spec = stage_inputs(stage)
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
        if (result['scope'] != 'CPU-only strict policy public model-folder I/O; no model or GPU'
                or result['profile'] != materialize.PROFILE or result['exit_code'] != 0
                or result['cuda_visible_devices'] != '' or result['no_torch'] is not True
                or result['cleanup_complete'] is not True or result['output_write_ok'] is not True
                or result['generator_sha256'] != GENERATOR_SHA or result['symlink_errno'] not in (1, 13)
                or not 0 <= result['worker_cpu_seconds'] <= 90 or not 0 <= result['child_cpu_seconds'] <= 90 or not 0 <= result['parent_cpu_seconds'] <= 20
                or not 0 < result['supervisor_wall_seconds'] <= 105
                or len(result['denial_errnos']) != 10 or any(x not in (1, 13) for x in result['denial_errnos'])
                or result['landlock_abi'] < 3 or result['handled_mask'] != policy.handled_rights(result['landlock_abi'])
                or set(result['mutation_denials']) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
                or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in result['mutation_denials'].items())
                or any(result['namespaces'][k] == result['parent_ns'][k] for k in ('user', 'net'))):
            raise ValueError('strict policy or CPU-only oracle receipt differs')
        policy.validate_leaf_receipt(result['approved_leaves'], result['loader'])
        alias_root = Path(staging['scratch']) / 'folder-real-oracle' / 'aliases'
        expected_denials = {name for name, _ in PROTECTED.values()} | {str(alias_root / group) for group in PROTECTED}
        if set(result.get('denied_paths', [])) != expected_denials or len(result['denied_paths']) != 10:
            raise ValueError('actual protected path and alias probes differ')
        admission = result['admission']
        restage = admission.get('full_restage_reserve_seconds')
        observed = sum(float(x['seconds']) for x in staging['inputs']) + float(staging['environment']['seconds'])
        if (not isinstance(restage, (int, float)) or not math.isfinite(restage)
                or restage < max(15.0, 1.5 * observed) or admission.get('phase_seconds_reserved') != 110
                or admission.get('campaign_seconds_reserved') != 140 + restage
                or admission.get('phase_seconds_remaining', 0) < 110
                or admission.get('campaign_seconds_remaining', 0) < 140 + restage):
            raise ValueError('pre-copy readiness receipt differs')
        if (result.get('synthetic_reference_job_id') != spec['synthetic_reference_job_id']
                or result.get('source_file_sha256') != {k: materialize.PINNED[v] for k, v in INPUT_NAMES.items()}):
            raise ValueError('real input or synthetic reference provenance differs')
        rows = result['materialization']['files']
        if result['materialization'].get('profile') != materialize.PROFILE:
            raise ValueError('materialization profile differs')
        if set(rows) != set(materialize.SIZES):
            raise ValueError('four pinned files required')
        for name, row in rows.items():
            if (row['source_sha256'] != materialize.PINNED[name] or row['copy_sha256'] != materialize.PINNED[name]
                    or row['bytes'] != materialize.SIZES[name]):
                raise ValueError('pinned copied bytes differ')
            input_key = next(key for key, value in INPUT_NAMES.items() if value == name)
            st = paths[input_key].stat()
            if row.get('source_stat') != [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]:
                raise ValueError('public source identity changed since copy')
        timings = ('wall_seconds', 'cpu_seconds', 'helper_wall_seconds', 'helper_cpu_seconds', 'cleanup_wall_seconds', 'cleanup_cpu_seconds')
        if any(not isinstance(result['materialization'].get(k), (float, int)) or not math.isfinite(result['materialization'][k])
               or result['materialization'][k] < 0 for k in timings):
            raise ValueError('copy/hash/helper/cleanup timing missing')
        if result['materialization']['bytes'] != sum(materialize.SIZES.values()):
            raise ValueError('real byte accounting differs')
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        source = stage.parent / 'source.tar'
        if sha(source) != dispatch['code']['snapshot_sha256']:
            raise ValueError('frozen source archive differs')
        code_files = {'experiments/tennis/w8_folder_real_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
                      'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
                      'experiments/tennis/w8_receiver_preflight.py', 'experiments/tennis/k6b_generate.py'}
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
                   'strict_policy_real_public_only': True, 'no_gpu_or_model': True, 'source_identity': True}}
    except Exception as exc:
        result_path.write_text(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    result_path.write_text(json.dumps(receipt) + '\n')


def make_spec(deadline: str, reference: dict[str, Any]) -> dict[str, Any]:
    """Construct only after actual synthetic job identities have been independently reviewed."""
    if set(reference) != {'job_id', 'sha256'} or not re.fullmatch(r'20[0-9]{6}T[0-9]{6}Z-[0-9a-f]{8}', reference.get('job_id', '')):
        raise ValueError('actual synthetic job identity required')
    digests = reference.get('sha256')
    if not isinstance(digests, dict) or set(digests) != set(PROOF_NAMES) or any(
            not isinstance(x, str) or not re.fullmatch(r'[0-9a-f]{64}', x) or x == '0' * 64 for x in digests.values()):
        raise ValueError('all six actual synthetic artifact SHA256 identities required')
    repo = Path(__file__).resolve().parents[2]
    spec = json.loads((repo / 'docs/experiments/w8-inputs/receiver-20261101/w8-folder-synthetic-oracle-v2-spec-20261105.json').read_text())
    old = json.loads((repo / 'docs/experiments/w8-inputs/receiver-20261101/w8-strict-host-cost-v2-spec-20261105.json').read_text())
    selected = [x for x in old['staged_inputs'] if x['name'] in set(INPUT_NAMES) | {'identity'}]
    expected = {**{k: materialize.PINNED[v] for k, v in INPUT_NAMES.items()}, 'identity': IDENTITY_SHA}
    if {x['name']: x['sha256'] for x in selected} != expected:
        raise ValueError('pinned real public input identities differ')
    prefix = '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/' + reference['job_id'] + '/'
    suffix = {'proof_validation': 'validation.json', 'proof_gate': 'gate.json', 'proof_state': 'state.json',
              'proof_ready': 'ready.json', 'proof_published': 'smoke/published.tar', 'proof_source': 'source.tar'}
    selected += [{'name': name, 'path': prefix + suffix[name], 'sha256': digests[name], 'extract': False}
                 for name in PROOF_NAMES]
    spec['staged_inputs'] = selected
    spec['inputs'] = [{'path': x['path'], 'sha256': x['sha256']} for x in selected]
    arguments = ['{phase}']
    for item in selected:
        arguments.extend(['--staged-' + item['name'].replace('_', '-'), '{staged:' + item['name'] + '}'])
    spec.update(hosts=['gpu6'], entrypoint=['-m', 'experiments.tennis.w8_folder_real_oracle'],
                arguments=arguments, scale={'phase': {'smoke': 'run', 'full': 'marker'}},
                smoke={'seconds': 120, 'representative_basis': 'One CPU-only strict-policy copy, verification and cleanup of four pinned public model-folder files after a completed synthetic prerequisite.'},
                full={'seconds': 10}, validator=['{python}', '-m', 'experiments.tennis.w8_folder_real_oracle', 'validate'],
                validator_seconds=20, budget_seconds=180, stall_seconds=120,
                synthetic_reference_job_id=reference['job_id'], deadline=deadline)
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'validate', 'marker', 'make-spec'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--config-sha256')
    parser.add_argument('--deadline')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--synthetic-reference', type=Path)
    for name in (*INPUT_NAMES, 'identity', *PROOF_NAMES):
        parser.add_argument('--staged-' + name.replace('_', '-'))
    args = parser.parse_args()
    if args.mode == 'make-spec':
        if args.deadline is None or args.output is None or args.synthetic_reference is None:
            parser.error('make-spec requires --deadline, --output and actual --synthetic-reference')
        from experiments.jobs.inbox import validate_spec
        spec = validate_spec(make_spec(args.deadline, json.loads(args.synthetic_reference.read_text())))
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2)
            stream.write('\n')
    elif args.mode == 'worker':
        if args.config is None or args.config_sha256 is None:
            parser.error('worker requires --config and --config-sha256')
        worker(args.config, args.config_sha256)
    elif args.mode in ('run', 'marker'):
        stage = Path(os.environ['PS_STAGE_DIR'])
        items = json.loads((stage / 'staging.json').read_text())['inputs']
        staged = {x['name']: x['path'] for x in items}
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
                or any(getattr(args, 'staged_' + name) != staged.get(name) for name in (*INPUT_NAMES, 'identity', *PROOF_NAMES))
                or any(x.get('mode') != 'local-hit' for x in items if x['name'] in INPUT_NAMES)):
            raise ValueError('CPU visibility, fleet arguments or public cache-hit binding differs')
        if args.mode == 'run':
            run()
        else:
            pub = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
            pub.mkdir()
            (pub / 'marker.json').write_text(json.dumps({'scope': 'CPU-only public-file oracle'}) + '\n')
    elif args.mode == 'validate':
        validate()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
