"""CPU-only fresh strict W8 packet proof and cost pilot; never schedules GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

from experiments.tennis import w8_input_job as fixed, w8_receiver_generate as receiver
from experiments.tennis import w8_strict_cost_job as cost, w8_strict_source as source
from experiments.tennis import w8_cuda_system_leaves as policy
from experiments.tennis.w8_receiver_preflight import PROTECTED

PREFLIGHT_SHA = 'ab5f1b932918ae9956f6ef648e36e59c6a1fd9ef6c36a1722a1916b0da63a4c5'
IDENTITY_SHA = '0953c5284ee8dc462d4eb95b7ce49cb9bad68e4d612fce5d167706f954668ebe'
CHECKPOINT_SHA = '1bb95a11c25ac66774b887c93277b695779e21cc9803cc565408206934f0fdeb'
NAMES = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': cost.ORACLE_SHA,
         'preflight': PREFLIGHT_SHA, 'identity': IDENTITY_SHA, 'guider': fixed.GUIDER_SHA}


def bound_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    combined = spec.get('w8_combined_cpu_proof_profile') == 'w8-exact-source-cpu-proof-v1'
    expected_names = dict(NAMES)
    if combined:
        expected_names['native_inventory'] = 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b'
    if (spec.get('device') != 'cpu' or spec.get('gpu_models') != [] or spec.get('gpu_memory_mib') != 0
            or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('hosts') != ['gpu5'] or spec.get('budget_seconds', 0) != (600 if combined else 240)
            or spec.get('environment', {}).get('sha256') != cost.MAIN_ENV
            or staging['environment']['sha256'] != cost.MAIN_ENV or len(dispatch.get('cpu_affinity', [])) != 1):
        raise ValueError('CPU-only single-core MAIN environment contract differs')
    if combined and (spec.get('smoke', {}).get('seconds') != 530 or spec.get('full', {}).get('seconds') != 5
                     or spec.get('validator_seconds') != 50):
        raise ValueError('combined CPU proof stage limits differ')
    declared = {item['name']: item for item in spec['staged_inputs']}
    actual = {item['name']: item for item in staging['inputs']}
    if len(declared) != len(spec['staged_inputs']) or len(actual) != len(staging['inputs']) or set(declared) != set(expected_names) or set(actual) != set(expected_names):
        raise ValueError('exact public packet input set required')
    declared_paths = {(item['path'], item['sha256']) for item in spec['inputs']}
    if declared_paths != {(item['path'], item['sha256']) for item in declared.values()}:
        raise ValueError('declared input identities differ')
    root = Path(staging['local_root']).resolve(strict=True)
    paths = {}
    for name, expected in expected_names.items():
        item = actual[name]
        if (declared[name]['sha256'] != expected or item['sha256'] != expected
                or declared[name].get('extract', False) or item.get('extract', False)):
            raise ValueError('input binding differs: ' + name)
        path = Path(item['path']).resolve(strict=True)
        if combined:
            from experiments.tennis.w8_protected_metadata import reject_protected_alias
            reject_protected_alias(path)
        if not path.is_relative_to(root) or fixed.file_sha(path) != expected:
            raise ValueError('staged public input differs: ' + name)
        paths[name] = path
    return paths, spec, dispatch


def evidence_names() -> list[str]:
    return ['decoded.pt', 'decoded-manifest.json', 'prepare.json', 'cpu-only.json',
            *(f'{name}.npz' for name in fixed.UNITS), 'worker-config.json',
            'loader-identity.json', 'protected-identity.json', 'source-manifest.json', 'worker-exit.json', 'worker.stdout', 'worker.stderr']


def check_isolation(config: dict[str, Any], isolation: dict[str, Any], loader: dict[str, Any],
                    source_manifest: dict[str, Any], loader_sha: str) -> None:
    expected = {p for group in config['denied_paths'].values() for p in group}
    denials, mutation = isolation['denials'], isolation['mutation_denials']
    if (config['worker_kind'] != 'packet' or config['environment']['CUDA_VISIBLE_DEVICES'] != ''
            or len(expected) != 10 or set(denials) != expected or any(v not in (1, 13) for v in denials.values())
            or isolation['landlock_abi'] < 3 or isolation['handled_mask'] != policy.handled_rights(isolation['landlock_abi'])
            or isolation['policy'] != 'w8-approved-system-leaves-mutation-v1'
            or isolation['network_interfaces'] != ['lo'] or isolation['pre_sandbox_threads'] != 1
            or isolation['extra_fds'] is not False or isolation['output_write_ok'] is not True
            or isolation['cpu_bootstrap']['cuda_initialized_before_isolation'] is not False
            or isolation['cpu_bootstrap']['before_landlock_sha256'] != isolation['cpu_bootstrap']['after_landlock_sha256']
            or isolation['namespaces']['user'] == config['parent_user_namespace']
            or isolation['namespaces']['net'] == config['parent_network_namespace']
            or isolation['loader_identity_sha256'] != loader_sha):
        raise ValueError('strict packet isolation receipt incomplete')
    if (set(mutation) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
            or any(code not in ((1, 13, 18) if name in ('rename', 'link') else (1, 13)) for name, code in mutation.items())):
        raise ValueError('strict mutation probe incomplete')
    policy.validate_leaf_receipt(isolation['approved_leaves'], loader)
    origins = isolation.get('source_origins')
    if (not isinstance(origins, dict) or 'experiments.tennis.w8_receiver_generate' not in origins
            or any(not isinstance(path, str) or path not in source_manifest['files']
                   or path not in (name.replace('.', '/') + '.py', name.replace('.', '/') + '/__init__.py')
                   for name, path in origins.items())):
        raise ValueError('packet project source origins differ from manifest')


def reconcile_protected_identity(historical: dict[str, Any], fresh: dict[str, Any]) -> None:
    old, current = historical['protected_inputs'], fresh['protected_inputs']
    if set(old) != set(PROTECTED) or set(current) != set(PROTECTED):
        raise ValueError('protected receipt categories differ')
    for group, (name, digest) in PROTECTED.items():
        prior, now = old[group], current[group]
        if (set(prior) != {'path', 'sha256', 'device', 'inode'} or set(now) != set(prior)
                or prior['path'] != name or now['path'] != name or prior['sha256'] != digest
                or now['sha256'] != digest or prior['inode'] != now['inode']
                or type(prior['device']) is not int or type(now['device']) is not int):
            raise ValueError('protected historical/fresh identity mismatch: ' + group)
        st = Path(name).stat()
        if (now['device'], now['inode']) != (st.st_dev, st.st_ino):
            raise ValueError('protected host identity changed: ' + group)


def check_grants(config: dict[str, Any], runtime: dict[str, Any]) -> None:
    actual = config['allow_paths']
    expected = cost.strict_roots(Path(actual[0]), Path(config['output']), runtime, config['files'])
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError('packet read grants differ')


def worker(python: Path, root: Path, work: Path, config: dict[str, Any], loader: Path) -> dict[str, Any]:
    cfg = work / 'worker-config.json'
    cfg.write_text(json.dumps(config, sort_keys=True))
    canary = work / 'policy-canary'
    canary.mkdir()
    names = {'write', 'truncate', 'remove', 'rename', 'link'}
    for name in names:
        (canary / name).write_bytes(b'canary')
    command = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', str(python), '-m',
               'experiments.tennis.w8_strict_packet_worker', '--config', str(cfg), '--config-sha256', fixed.file_sha(cfg),
               '--loader', str(loader), '--loader-sha256', fixed.file_sha(loader), '--canary', str(canary), '--source-root', str(root)]
    env = {'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (100, 100))
    with (work / 'stdout').open('xb') as out, (work / 'stderr').open('xb') as err:
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, close_fds=True, preexec_fn=limit)
        status, timed_out = receiver.wait_owned_worker(proc, 130)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    canaries_ok = {p.name for p in canary.iterdir()} == names and all((canary / name).read_bytes() == b'canary' for name in names)
    record = {'argv': command, 'cwd': str(root), 'cuda_visible_devices': '', 'pid': proc.pid, 'status': status, 'timed_out': timed_out,
              'wall_seconds': time.monotonic() - start,
              'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
              'stdout_sha256': fixed.file_sha(work / 'stdout'), 'stderr_sha256': fixed.file_sha(work / 'stderr'),
              'config_sha256': fixed.file_sha(cfg), 'canaries_preserved': canaries_ok}
    (work / 'worker-exit.json').write_text(json.dumps(record, sort_keys=True))
    if status != 0 or timed_out or not canaries_ok:
        raise RuntimeError('strict packet worker failed; preserved task scratch')
    return record


def require_remaining(dispatch: dict[str, Any], phase_reserve: float, campaign_reserve: float) -> dict[str, float]:
    clock = dispatch['campaign_clock']
    now = time.monotonic()
    started, phase, end = clock['started'], clock['phase_started'], clock['budget_end']
    if not started <= phase <= now < end or end - started > 241:
        raise ValueError('campaign clock identity differs')
    phase_left = phase + 180 - now
    campaign_left = end - now
    if phase_left < phase_reserve or campaign_left < campaign_reserve:
        raise ValueError('insufficient time for packet, publication and independent validator')
    return {'phase_seconds_remaining': phase_left, 'campaign_seconds_remaining': campaign_left,
            'phase_seconds_reserved': phase_reserve, 'campaign_seconds_reserved': campaign_reserve}


def run(stage: Path) -> None:
    if stage.name != 'smoke' or sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single-core Linux CPU smoke required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_CLAIMED_GPU_UUID'):
        raise ValueError('CPU fleet must hide GPU and claim no GPU')
    resource.setrlimit(resource.RLIMIT_CPU, (40, 100))
    paths, _, dispatch = bound_inputs(stage)
    snapshot = Path(__file__).resolve().parents[2]
    preflight = json.loads(paths['preflight'].read_text())
    cost.check_preflight_bundle(preflight, paths['identity'])
    cost.check_preflight_sources(preflight, snapshot)
    identity = json.loads(paths['identity'].read_text())
    fingerprints = cost.read_oracle(paths['oracle'])
    rate = cost.audit_public_rate(paths['packet'], paths['profile'])
    scratch = fixed.checked_scratch()
    require_remaining(dispatch, 141, 190)
    protected_path = scratch / 'protected-identity.json'
    protected = policy.protected_identity({'denied_paths': {group: [name] for group, (name, _) in PROTECTED.items()}}, protected_path)
    reconcile_protected_identity(identity, json.loads(protected_path.read_text()))
    root = scratch / 'receiver-source'
    manifest = source.stage_source(snapshot, root)
    source.verify_source(root, manifest)
    loader = Path('/etc/ld.so.cache').resolve(strict=True)
    info = loader.stat()
    if str(loader) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('unsafe host loader cache')
    loader_path = scratch / 'loader-identity.json'
    loader_path.write_text(json.dumps({'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}, sort_keys=True))
    aliases = scratch / 'protected-aliases'
    aliases.mkdir()
    denied = {group: [name] for group, (name, _) in PROTECTED.items()}
    denied['aliases'] = []
    for group, (name, _) in PROTECTED.items():
        alias = aliases / group
        alias.symlink_to(Path(name).resolve(strict=True))
        denied['aliases'].append(str(alias))
    work = scratch / 'packet-worker'
    work.mkdir()
    output = work / 'output'
    output.mkdir()
    main = preflight['runtimes']['main']
    python = Path(main['python']).resolve(strict=True)
    if not python.is_relative_to((Path('/local/users/emanuele/pointstream/cache') / cost.MAIN_ENV / 'tree').resolve(strict=True)):
        raise ValueError('MAIN runtime identity mismatch')
    files = {key: cost._item(paths[name]) for key, name in (('bundle', 'packet'), ('profile', 'profile'), ('guider', 'guider'))}
    config = {'worker_kind': 'packet', 'unit': fixed.UNITS[0], 'cpu_seconds': 100, 'files': files, 'output': str(output),
              'allow_paths': cost.strict_roots(root, output, main, files), 'denied_paths': denied,
              'parent_user_namespace': os.readlink('/proc/self/ns/user'), 'parent_network_namespace': os.readlink('/proc/self/ns/net'),
              'protected_receipt': protected, 'tensor_fingerprints': fingerprints,
              'environment': {'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(output), 'TMP': str(output), 'TEMP': str(output),
                              'XDG_CACHE_HOME': str(output / '.cache'), 'TORCH_HOME': str(output / '.cache' / 'torch'),
                              'PS_GPU_MEMORY_LIMIT_MIB': '0'}}
    admission = require_remaining(dispatch, 141, 190)
    usage = worker(python, root, work, config, loader_path)
    require_remaining(dispatch, 10, 60)
    decoded = json.loads((output / 'decoded-manifest.json').read_text())
    if (decoded.get('packet_sha256') != fixed.WINDOW_SHA or decoded.get('profile_sha256') != fixed.PROFILE_SHA
            or decoded.get('guider_sha256') != fixed.GUIDER_SHA or decoded.get('match_tensors') != fingerprints
            or decoded.get('unit_sha256') != receiver.ORACLE_UNIT_SHA or decoded.get('checkpoint_sha256') != CHECKPOINT_SHA
            or fixed.file_sha(output / 'decoded.pt') != CHECKPOINT_SHA):
        raise ValueError('decoded packet does not match retained independent oracle')
    check_isolation(config, decoded['isolation'], json.loads(loader_path.read_text()), manifest, fixed.file_sha(loader_path))
    cpu = json.loads((output / 'cpu-only.json').read_text())
    if cpu.get('cuda_visible_devices') != '' or cpu.get('cuda_initialized_after_decode') is not False:
        raise ValueError('GPU-hidden CPU receipt missing')
    pub = scratch / 'publish'
    pub.mkdir()
    manifest_path = scratch / 'source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    members = {'decoded.pt': output / 'decoded.pt', 'decoded-manifest.json': output / 'decoded-manifest.json',
               'prepare.json': output / 'publish/prepare.json', 'cpu-only.json': output / 'cpu-only.json',
               'worker-config.json': work / 'worker-config.json', 'loader-identity.json': loader_path,
               'protected-identity.json': protected_path,
               'source-manifest.json': manifest_path, 'worker-exit.json': work / 'worker-exit.json',
               'worker.stdout': work / 'stdout', 'worker.stderr': work / 'stderr'}
    members.update({f'{name}.npz': output / 'publish/units' / f'{name}.npz' for name in fixed.UNITS})
    if list(members) != evidence_names() and set(members) != set(evidence_names()):
        raise ValueError('unexpected packet evidence member set')
    archive_path = pub / 'packet-evidence.tar'
    with tarfile.open(archive_path, 'x') as tar:
        for name in evidence_names():
            item = members[name]
            if item.is_symlink() or not item.is_file():
                raise ValueError('packet evidence missing')
            tar.add(item, arcname=name, recursive=False)
    summary = {'scope': 'strict CPU packet decoder cost/correctness only', 'quality_claim': False,
               'archive_sha256': fixed.file_sha(archive_path), 'archive_bytes': archive_path.stat().st_size,
               'members': {name: fixed.file_sha(members[name]) for name in evidence_names()},
               'inputs': {name: fixed.file_sha(path) for name, path in paths.items()}, 'public_rate': rate,
               'worker': usage, 'admission': admission, 'parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
               'code_identity': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256']}
    (pub / 'packet-summary.json').write_text(json.dumps(summary, sort_keys=True, allow_nan=False) + '\n')


def check_serialized(evidence: Path, packet: dict[str, Any], fingerprints: dict[str, Any]) -> None:
    if (packet.get('packet_sha256') != fixed.WINDOW_SHA or packet.get('profile_sha256') != fixed.PROFILE_SHA
            or packet.get('guider_sha256') != fixed.GUIDER_SHA or packet.get('match_tensors') != fingerprints
            or packet.get('unit_sha256') != receiver.ORACLE_UNIT_SHA or packet.get('checkpoint_sha256') != CHECKPOINT_SHA
            or any(fixed.file_sha(evidence / f'{name}.npz') != receiver.ORACLE_UNIT_SHA[name] for name in fixed.UNITS)
            or receiver.verify_serialized_match(evidence / 'decoded.pt', fingerprints) != CHECKPOINT_SHA):
        raise ValueError('packet bytes or independent 480-tensor serialization mismatch')


def validate(stage: Path, output: Path) -> int:
    """Independently verify published evidence; qualification only, never GPU admission."""
    try:
        if stage.name != 'smoke' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU-only smoke validator required')
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        paths, _, dispatch = bound_inputs(stage)
        fingerprints = cost.read_oracle(paths['oracle'])
        rate = cost.audit_public_rate(paths['packet'], paths['profile'])
        preflight = json.loads(paths['preflight'].read_text())
        cost.check_preflight_bundle(preflight, paths['identity'])
        cost.check_preflight_sources(preflight, Path(__file__).resolve().parents[2])
        outer_path = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if fixed.file_sha(outer_path) != staging['published']['sha256'] or outer_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('published packet archive identity or size differs')
        with tempfile.TemporaryDirectory(prefix='w8-packet-audit-', dir=os.environ['TMPDIR']) as folder:
            temporary = Path(folder)
            with tarfile.open(outer_path) as outer:
                entries = outer.getmembers()
                files = [entry for entry in entries if entry.isfile()]
                if (len(files) != 2 or {entry.name for entry in files} != {'publish/packet-summary.json', 'publish/packet-evidence.tar'}
                        or any(not entry.isfile() and not (entry.isdir() and entry.name.rstrip('/') == 'publish') for entry in entries)):
                    raise ValueError('unexpected published packet entries')
                summary_file = outer.extractfile('publish/packet-summary.json')
                evidence_file = outer.extractfile('publish/packet-evidence.tar')
                if summary_file is None or evidence_file is None:
                    raise ValueError('published packet evidence unreadable')
                summary = json.load(summary_file)
                evidence = temporary / 'packet-evidence.tar'
                with evidence.open('xb') as target:
                    shutil.copyfileobj(evidence_file, target, 1024 * 1024)
            if (summary.get('quality_claim') is not False or summary.get('scope') != 'strict CPU packet decoder cost/correctness only'
                    or summary.get('archive_sha256') != fixed.file_sha(evidence) or summary.get('archive_bytes') != evidence.stat().st_size
                    or set(summary.get('members', {})) != set(evidence_names()) or summary['public_rate'] != rate
                    or summary['inputs'] != {name: fixed.file_sha(path) for name, path in paths.items()}
                    or summary['code_identity'] != dispatch['code']['source_sha256']
                    or summary['snapshot_sha256'] != dispatch['code']['snapshot_sha256']):
                raise ValueError('packet summary identity differs')
            unpacked = temporary / 'evidence'
            unpacked.mkdir()
            with tarfile.open(evidence) as archive:
                members = archive.getmembers()
                if len(members) != len(evidence_names()) or {m.name for m in members} != set(evidence_names()) or any(not m.isfile() for m in members):
                    raise ValueError('packet evidence member set differs')
                for member in members:
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError('packet evidence member unreadable')
                    with (unpacked / member.name).open('xb') as target:
                        shutil.copyfileobj(stream, target, 1024 * 1024)
            if any(fixed.file_sha(unpacked / name) != digest for name, digest in summary['members'].items()):
                raise ValueError('packet evidence bytes differ')
            admission = summary['admission']
            if (admission.get('phase_seconds_reserved') != 141 or admission.get('campaign_seconds_reserved') != 190
                    or admission.get('phase_seconds_remaining', 0) < 141 or admission.get('campaign_seconds_remaining', 0) < 190):
                raise ValueError('packet campaign readiness receipt differs')
            config = json.loads((unpacked / 'worker-config.json').read_text())
            loader = json.loads((unpacked / 'loader-identity.json').read_text())
            manifest = json.loads((unpacked / 'source-manifest.json').read_text())
            packet = json.loads((unpacked / 'decoded-manifest.json').read_text())
            cpu = json.loads((unpacked / 'cpu-only.json').read_text())
            usage = json.loads((unpacked / 'worker-exit.json').read_text())
            if (usage != summary['worker'] or usage['status'] != 0 or usage['timed_out'] is not False
                    or not 0 < usage['wall_seconds'] <= 130 or not 0 <= usage['cpu_seconds'] <= 100
                    or usage['canaries_preserved'] is not True or usage['config_sha256'] != fixed.file_sha(unpacked / 'worker-config.json')
                    or usage['stdout_sha256'] != fixed.file_sha(unpacked / 'worker.stdout')
                    or usage['stderr_sha256'] != fixed.file_sha(unpacked / 'worker.stderr')
                    or usage['cuda_visible_devices'] != ''
                    or not 0 <= summary['parent_cpu_seconds'] <= 40):
                raise ValueError('packet process limits or canary receipt differ')
            if (usage['cwd'] != config['allow_paths'][0] or '--source-root' not in usage['argv']
                    or usage['argv'][usage['argv'].index('--source-root') + 1] != config['allow_paths'][0]
                    or usage['argv'][:4] != ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
                    or 'experiments.tennis.w8_strict_packet_worker' not in usage['argv']):
                raise ValueError('packet worker source-origin argv differs')
            main = preflight['runtimes']['main']
            check_grants(config, main)
            if ({name: item['sha256'] for name, item in config['files'].items()}
                    != {'bundle': NAMES['packet'], 'profile': NAMES['profile'], 'guider': NAMES['guider']}
                    or config['protected_receipt']['sha256'] != fixed.file_sha(unpacked / 'protected-identity.json')
                    or config['cpu_seconds'] != 100):
                raise ValueError('packet read grants or public files differ')
            reconcile_protected_identity(json.loads(paths['identity'].read_text()),
                                         json.loads((unpacked / 'protected-identity.json').read_text()))
            check_isolation(config, packet['isolation'], loader, manifest, fixed.file_sha(unpacked / 'loader-identity.json'))
            if (cpu.get('cuda_visible_devices') != '' or cpu.get('cuda_initialized_after_decode') is not False
                    or cpu.get('torch_version') != preflight['receipts']['main']['torch']
                    or cpu.get('pre_serialization', {}).get('state_tensors') != 480
                    or cpu.get('pre_serialization', {}).get('unit_arrays', 0) <= 0
                    or cpu.get('pre_serialization', {}).get('restored_tensors', 0) <= 0):
                raise ValueError('CPU-only runtime evidence differs')
            check_serialized(unpacked, packet, fingerprints)
            files = manifest.get('files')
            if not isinstance(files, dict) or not files or any(not name.endswith('.py') or not name.startswith(('experiments/', 'src/', 'tools/')) for name in files):
                raise ValueError('source manifest includes non-source paths')
            encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
            source_archive = stage.parent / 'source.tar'
            if hashlib.sha256(encoded).hexdigest() != manifest['manifest_sha256'] or fixed.file_sha(source_archive) != dispatch['code']['snapshot_sha256']:
                raise ValueError('source manifest or fleet archive identity differs')
            with tarfile.open(source_archive) as archive:
                for name, digest in files.items():
                    member = archive.getmember(name)
                    stream = archive.extractfile(member) if member.isfile() else None
                    if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise ValueError('source archive member differs: ' + name)
        receipt = {'passed': True, 'scope': 'CPU strict packet cost/correctness only; no GPU admission',
                   'checks': {'strict_denials_and_mutations': True, 'no_cuda_initialization': True,
                              'independent_480_tensor_serialization': True, 'two_unit_hashes': True,
                              'source_package_and_origins': True, 'public_packet_rate': True},
                   'packet_cpu_seconds': usage['cpu_seconds'], 'packet_wall_seconds': usage['wall_seconds'],
                   'parent_cpu_seconds': summary['parent_cpu_seconds']}
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    output.write_text(json.dumps(receipt, allow_nan=False) + '\n')
    return 0


def make_spec(deadline: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    base = json.loads((root / 'docs/experiments/w8-inputs/receiver-20261101/w8-strict-cost-spec-20261105.json').read_text())
    selected = []
    for item in base['staged_inputs']:
        name = 'guider' if item['name'] == 'w6' else item['name']
        if name in NAMES:
            selected.append({**item, 'name': name})
    if {item['name'] for item in selected} != set(NAMES) or any(item['sha256'] != NAMES[item['name']] for item in selected):
        raise ValueError('base public packet inputs differ')
    base['staged_inputs'] = selected
    base['inputs'] = [{'path': item['path'], 'sha256': item['sha256']} for item in selected]
    base.update(device='cpu', hosts=['gpu5'], gpu_models=[], gpu_memory_mib=0, cpu_threads=1,
                single_cpu_campaign=True, budget_seconds=240, validator_seconds=40,
                entrypoint=['-m', 'experiments.tennis.w8_strict_packet_job'],
                arguments=['packet-run', '--phase', '{phase}'], scale={'phase': {'smoke': 'smoke', 'full': 'full'}},
                smoke={'seconds': 180, 'representative_basis': 'Fresh source-hidden strict CPU packet decode and full independent oracle proof; full stage is only a completion marker.'},
                full={'seconds': 10}, validator=['{python}', '-m', 'experiments.tennis.w8_strict_packet_job', 'packet-validate'],
                required_commands=['unshare'], deadline=deadline, stall_seconds=180)
    base['environment'] = {'path': '/home/itec/emanuele/Datasets/pointstream-data/environments/pointstream-20261009T214002Z.tar.gz',
                           'sha256': cost.MAIN_ENV}
    base.pop('gpu_sharing', None)
    base.pop('receiver_cpu_budget_seconds', None)
    base.pop('receiver_external_cpu_reserve_seconds', None)
    base.pop('receiver_validator_cpu_seconds', None)
    return base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('packet-run', 'packet-validate', 'make-spec'))
    p.add_argument('--phase', choices=('smoke', 'full'))
    p.add_argument('--deadline')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'make-spec':
        if args.deadline is None or args.output is None:
            p.error('make-spec needs --deadline and --output')
        spec = make_spec(args.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.mode == 'packet-validate':
        return validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    if os.environ.get('PS_STAGE') != args.phase or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('phase or GPU visibility differs')
    if args.phase == 'smoke':
        run(stage)
    elif args.phase == 'full':
        pub = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        pub.mkdir()
        (pub / 'packet-only.json').write_text(json.dumps({'scope': 'CPU packet pilot complete; no GPU admission'}) + '\n')
    else:
        p.error('run needs --phase')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


def read_witness_packet_cpu_receipt(path: Path, expected_torch: str) -> dict[str, Any]:
    """Require the actual instrumented packet receipt, not a post-hoc inference."""
    cpu = json.loads(path.read_text())
    observed = cpu.get('pre_serialization', {})
    if (cpu.get('scope') != 'CPU packet decoder only' or cpu.get('cuda_visible_devices') != ''
            or cpu.get('cuda_initialized_after_decode') is not False or cpu.get('torch_version') != expected_torch
            or observed.get('state_tensors') != 480 or observed.get('restored_tensors') != 480
            or type(observed.get('unit_arrays')) is not int or observed['unit_arrays'] <= 0):
        raise ValueError('actual pre-serialization CPU packet receipt differs')
    return cpu


def run_proof(stage: Path) -> dict[str, Any]:
    """Run the fixed real packet path for the combined CPU proof, without publication."""
    from experiments.tennis import w8_strict_host_cost_job as host

    if (stage.name != 'smoke' or sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_CLAIMED_GPU_UUID')):
        raise ValueError('combined packet proof requires hidden CUDA and one-core Linux')
    paths, spec, dispatch = bound_inputs(stage)
    if spec.get('w8_combined_cpu_proof_profile') != 'w8-exact-source-cpu-proof-v1':
        raise ValueError('explicit combined proof profile required')
    bootstrap_sha = spec.get('witness_bootstrap_sha256')
    if (not isinstance(bootstrap_sha, str) or len(bootstrap_sha) != 64
            or any(c not in '0123456789abcdef' for c in bootstrap_sha)):
        raise ValueError('frozen witness bootstrap SHA required')
    from experiments.tennis.w8_witness_job import check_bootstrap_archive
    input_sha = spec.get('witness_input_job_sha256')
    if not isinstance(input_sha, str):
        raise ValueError('frozen witness input module SHA required')
    policy_sha = spec.get('witness_system_leaves_sha256')
    if not isinstance(policy_sha, str):
        raise ValueError('frozen witness system leaves SHA required')
    check_bootstrap_archive(stage, dispatch, bootstrap_sha, input_sha, policy_sha)
    snapshot = Path(__file__).resolve().parents[2]
    preflight = json.loads(paths['preflight'].read_text())
    host.check_preflight_bundle(preflight, paths['identity'], witness_metadata=True)
    host.check_preflight_sources(preflight, snapshot, witness_bootstrap_sha=bootstrap_sha,
                                 witness_input_sha=input_sha, witness_policy_sha=policy_sha)
    fingerprints = host.read_oracle(paths['oracle'])
    rate = host.audit_public_rate(paths['packet'], paths['profile'])
    scratch = fixed.checked_scratch()
    protected_path = scratch / 'protected-identity.json'
    from experiments.tennis import w8_protected_metadata
    protected = w8_protected_metadata.capture(paths['identity'], protected_path, expected_historical_sha256=IDENTITY_SHA)
    root = scratch / 'receiver-source'
    manifest = source.stage_source(snapshot, root)
    source.verify_source(root, manifest)
    manifest_path = scratch / 'source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    loader = Path('/etc/ld.so.cache').resolve(strict=True)
    info = loader.stat()
    if str(loader) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('unsafe host loader cache')
    loader_path = scratch / 'loader-identity.json'
    loader_path.write_text(json.dumps({'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}, sort_keys=True))
    aliases = scratch / 'protected-aliases'
    aliases.mkdir()
    denied = {group: [name] for group, (name, _) in PROTECTED.items()}
    denied['aliases'] = []
    for group, (name, _) in PROTECTED.items():
        alias = aliases / group
        alias.symlink_to(Path(name).resolve(strict=True))
        denied['aliases'].append(str(alias))
    policies = preflight['runtimes']
    for name, expected in (('main', host.MAIN_ENV), ('jcv', host.JCV_ENV)):
        root_path = Path('/local/users/emanuele/pointstream/cache') / expected / 'tree'
        if not Path(policies[name]['python']).resolve(strict=True).is_relative_to(root_path.resolve(strict=True)):
            raise ValueError('proof runtime differs from pinned cache')
    work = scratch / 'packet-worker'
    work.mkdir()
    output = work / 'output'
    output.mkdir()
    def config(kind: str, files: dict[str, dict[str, str]], out: Path, runtime: dict[str, Any],
               cpu: int, visible: str, unit: str = fixed.UNITS[0]) -> dict[str, Any]:
        return {'worker_kind': kind, 'unit': unit, 'cpu_seconds': cpu, 'files': files, 'output': str(out),
                'allow_paths': host.strict_roots(root, out, runtime, files), 'denied_paths': denied,
                'parent_user_namespace': os.readlink('/proc/self/ns/user'),
                'parent_network_namespace': os.readlink('/proc/self/ns/net'), 'protected_receipt': protected,
                'tensor_fingerprints': fingerprints,
                'environment': {'CUDA_VISIBLE_DEVICES': visible, 'TMPDIR': str(out), 'TMP': str(out),
                                'TEMP': str(out), 'XDG_CACHE_HOME': str(out / '.cache'),
                                'TORCH_HOME': str(out / '.cache' / 'torch'), 'PS_GPU_MEMORY_LIMIT_MIB': '0'}}
    files = {'bundle': host._item(paths['packet']), 'profile': host._item(paths['profile']),
             'guider': host._item(paths['guider'])}
    packet_config = config('packet', files, output, policies['main'], 100, '')
    usage = host._worker(Path(policies['main']['python']), packet_config, work, 100, 80, root, loader_path,
                         witness_mode='packet')
    decoded_path = output / 'decoded-manifest.json'
    decoded = json.loads(decoded_path.read_text())
    check_grants(packet_config, policies['main'])
    check_isolation(packet_config, decoded['isolation'], json.loads(loader_path.read_text()), manifest,
                    fixed.file_sha(loader_path))
    cpu_path = output / 'cpu-only.json'
    read_witness_packet_cpu_receipt(cpu_path, preflight['receipts']['main']['torch'])
    if (decoded.get('match_tensors') != fingerprints or decoded.get('unit_sha256') != receiver.ORACLE_UNIT_SHA
            or decoded.get('packet_sha256') != fixed.WINDOW_SHA or decoded.get('profile_sha256') != fixed.PROFILE_SHA
            or decoded.get('guider_sha256') != fixed.GUIDER_SHA or decoded.get('checkpoint_sha256') != CHECKPOINT_SHA
            or receiver.verify_serialized_match(output / 'decoded.pt', fingerprints) != CHECKPOINT_SHA
            or any(fixed.file_sha(output / 'publish' / 'units' / f'{u}.npz') != receiver.ORACLE_UNIT_SHA[u] for u in fixed.UNITS)
):
        raise ValueError('combined proof packet/oracle identity differs')
    receipt_paths = {'packet-config.json': work / 'worker-config.json', 'packet-exit.json': work / 'exit.json',
                     'packet-self-leaves.json': work / 'self-leaves.json',
                     'packet-namespace.json': work / 'namespace-receipt.json',
                     'packet-namespace.json.started': work / 'namespace-receipt.json.started',
                     'packet-decoded-manifest.json': decoded_path, 'packet-cpu-only.json': cpu_path,
                     'source-manifest.json': manifest_path, 'loader-identity.json': loader_path,
                     'protected-identity.json': protected_path}
    return {'scope': 'strict_packet_cpu_proof_only', 'paths': paths, 'scratch': scratch, 'root': root,
            'protected_path': protected_path, 'code_manifest': manifest, 'loader_receipt': loader_path,
            'preflight': preflight, 'policies': policies, 'config': config, 'fingerprints': fingerprints,
            'decoder_out': output, 'packet_usage': usage, 'public_rate': rate,
            'native_inventory': paths['native_inventory'], 'proof_receipts': receipt_paths}
