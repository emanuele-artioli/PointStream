"""Fleet entrypoint for source-hidden W8 1/3-step receiver cost probes.

These runs are qualification only; they cannot certify candidate quality.
A root-reviewed preflight bundle and fleet specification are prerequisites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_receiver_generate as receiver
from experiments.tennis.w8_receiver_preflight import ENV_SHAS, PROTECTED

ORACLE_SHA = '8047b67694dd64171d2c0d13a9eaff53b18cd833b8fdc45ba054185c382848cc'
JCV_ENV = ENV_SHAS['jcv']
MAIN_ENV = ENV_SHAS['main']
BASE_ARCHIVE_SHA = fixed.FIXED_INPUTS['base']
WEIGHT_NAMES = {
    'w0': 'vae/config.json', 'w1': 'vae/diffusion_pytorch_model.safetensors',
    'w2': 'image_encoder/config.json', 'w3': 'image_encoder/pytorch_model.bin',
    'w4': 'original/denoising_unet.pth', 'w5': 'original/reference_unet.pth',
    'w6': 'original/pose_guider.pth', 'w7': 'original/motion_module.pth',
}


def stage_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('device', 'gpu') != 'gpu' or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or len(dispatch.get('cpu_affinity', [])) != 1
            or spec.get('environment', {}).get('sha256') != JCV_ENV
            or spec.get('gpu_sharing', {'mode': 'exclusive'}) != {'mode': 'exclusive'}
            or spec.get('contention', {}).get('policy') != 'stop'
            or not 0 < spec.get('budget_seconds', 0) <= 500):
        raise ValueError('one-core JCV GPU campaign with at most 500 seconds required')
    wanted = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': ORACLE_SHA,
              'base': BASE_ARCHIVE_SHA, 'aa': receiver.AA_ARCHIVE_SHA,
              **{key: receiver.PINNED_WEIGHTS[name] for key, name in WEIGHT_NAMES.items()}}
    declared = {x['name']: x for x in spec['staged_inputs']}
    actual = {x['name']: x for x in staging['inputs']}
    if len(declared) != len(spec['staged_inputs']) or len(actual) != len(staging['inputs']):
        raise ValueError('duplicate staged inputs')
    if set(actual) != set(wanted) | {'preflight', 'identity'} or set(declared) != set(actual):
        raise ValueError('exact staged input set required')
    for extra in ('preflight', 'identity'):
        wanted[extra] = declared[extra]['sha256']
    paths = {}
    for name, expected in wanted.items():
        if (declared[name]['sha256'] != expected or actual[name]['sha256'] != expected
                or declared[name].get('extract', False) or actual[name].get('extract', False)):
            raise ValueError(f'staged identity mismatch: {name}')
        path = Path(actual[name]['path']).resolve(strict=True)
        if fixed.file_sha(path) != expected:
            raise ValueError(f'staged bytes mismatch: {name}')
        paths[name] = path
    return paths, spec


def read_oracle(path: Path) -> dict[str, Any]:
    if fixed.file_sha(path) != ORACLE_SHA:
        raise ValueError('independent oracle report identity mismatch')
    report = json.loads(path.read_text())
    result = report['scientific_result']
    if (report.get('passed') is not True or result.get('passed') is not True
            or result.get('bundle_sha256') != fixed.WINDOW_SHA
            or result.get('profile_sha256') != fixed.PROFILE_SHA
            or result.get('decoded_sha256') != receiver.ORACLE_UNIT_SHA):
        raise ValueError('independent oracle scientific result mismatch')
    fingerprints = result['match']['tensor_fingerprints']
    receiver.check_oracle_fingerprints(fingerprints)
    return fingerprints


def extract_base(archive_path: Path, out: Path) -> Path:
    if fixed.file_sha(archive_path) != BASE_ARCHIVE_SHA:
        raise ValueError('common base archive mismatch')
    with tarfile.open(archive_path) as archive:
        found = [entry for entry in archive if entry.name == 'publish/tennis-base@fit-100.pt']
        if len(found) != 1 or not found[0].isfile() or not 0 < found[0].size <= 60_000_000:
            raise ValueError('common base member missing or ambiguous')
        stream = archive.extractfile(found[0])
        if stream is None:
            raise ValueError('common base member unreadable')
        with out.open('xb') as target:
            shutil.copyfileobj(stream, target, 1024 * 1024)
    if fixed.file_sha(out) != receiver.BASE_SHA:
        raise ValueError('common base checkpoint mismatch')
    return out


def _item(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': fixed.file_sha(path)}


def _worker(python: Path, config: dict[str, Any], work: Path, cpu_seconds: int, wall_seconds: int) -> dict[str, Any]:
    config_path = work / 'worker-config.json'
    config_path.write_text(json.dumps(config, sort_keys=True))
    source = Path(__file__).resolve().parents[2]
    env = {'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    command = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', str(python), '-m',
               'experiments.tennis.w8_receiver_bootstrap', '--config', str(config_path),
               '--config-sha256', fixed.file_sha(config_path), '--cpu-seconds', str(cpu_seconds)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, 3600))
    with (work / 'stdout').open('xb') as stdout, (work / 'stderr').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                close_fds=True, env=env, preexec_fn=limit)
        (work / 'start.json').write_text(json.dumps({'pid': proc.pid, 'pgid': os.getpgrp(),
                                                     'started_monotonic': started}, sort_keys=True))
        status, timed_out = receiver.wait_owned_worker(proc, wall_seconds)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {'pid': proc.pid, 'pgid': os.getpgrp(), 'status': status, 'timed_out': timed_out,
               'wall_seconds': time.monotonic() - started,
               'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
               'config_sha256': fixed.file_sha(config_path)}
    (work / 'exit.json').write_text(json.dumps(receipt, sort_keys=True))
    if status or timed_out:
        raise RuntimeError(f'isolated worker failed; retained {work}')
    return receipt


def projection(one: dict[str, Any], three: dict[str, Any], remaining_gpu: float,
               downstream_reserve: float, extra_generation_overhead: float) -> dict[str, Any]:
    """Project two original 30-step units from measured 1/3-step sampling times."""
    if (one.get('scope') != 'qualification_only' or three.get('scope') != 'qualification_only'
            or one.get('steps') != 1 or three.get('steps') != 3 or one.get('unit') != three.get('unit')):
        raise ValueError('matching 1/3-step qualification receipts required')
    a, b = one['phase_costs'], three['phase_costs']
    one_time, three_time = a['sampling']['gpu_reserved_wall_seconds'], b['sampling']['gpu_reserved_wall_seconds']
    if (not all(math.isfinite(v) for v in (one_time, three_time, remaining_gpu, downstream_reserve, extra_generation_overhead))
            or not 0 < one_time < three_time or remaining_gpu <= 0 or downstream_reserve <= 0
            or extra_generation_overhead < 0):
        raise ValueError('nonpositive, nonmonotonic or invalid cost evidence')
    per_step = (three_time - one_time) / 2
    sampling_30 = one_time + 29 * per_step
    overhead = sum(max(a[k]['gpu_reserved_wall_seconds'], b[k]['gpu_reserved_wall_seconds'])
                   for k in ('model_load', 'original_probe', 'context_probe', 'publication'))
    predicted = 1.35 * 2 * (sampling_30 + overhead + extra_generation_overhead)
    return {'per_step_gpu_wall_seconds': per_step, 'two_unit_30_step_seconds_upper_projection': predicted,
            'downstream_reserve_seconds': downstream_reserve,
            'extra_generation_overhead_seconds_per_unit': extra_generation_overhead,
            'remaining_gpu_seconds': remaining_gpu,
            'fits': predicted + downstream_reserve <= remaining_gpu,
            'scope': 'measured projection only; no quality evidence or runtime guarantee'}


def retained_path(stage: Path) -> Path:
    staging = json.loads((stage / 'staging.json').read_text())
    local = Path(staging['local_root']).resolve(strict=True)
    host_root = Path('/local/users/emanuele').resolve(strict=True)
    if not local.is_relative_to(host_root) or stage.name not in ('smoke', 'full'):
        raise ValueError('host-local retained-output root and valid stage required')
    job = stage.parent.name
    if not job or job in ('.', '..') or '/' in job:
        raise ValueError('invalid fleet job identifier')
    return local / 'w8-cost-retained' / job / stage.name


def check_preflight_bundle(preflight: dict[str, Any], identity_path: Path) -> None:
    """Bind embedded MAIN/JCV receipts to the accepted preflight result."""
    from experiments.tennis.w8_receiver_preflight import EXPECTED_TORCH

    if (preflight.get('passed') is not True or preflight.get('identity_sha256') != fixed.file_sha(identity_path)
            or set(preflight.get('runtimes', {})) != {'main', 'jcv'}
            or set(preflight.get('receipts', {})) != {'main', 'jcv'}):
        raise ValueError('accepted preflight bundle and identity receipt required')
    identity = json.loads(identity_path.read_text())
    assets = identity['files']
    expected_assets = {'main_archive': MAIN_ENV, 'jcv_archive': JCV_ENV, 'aa_archive': receiver.AA_ARCHIVE_SHA,
                       **{f'weight_{name}': receiver.PINNED_WEIGHTS[weight] for name, weight in (
                           ('vae_config', WEIGHT_NAMES['w0']), ('vae_weights', WEIGHT_NAMES['w1']),
                           ('image_encoder_config', WEIGHT_NAMES['w2']), ('image_encoder_weights', WEIGHT_NAMES['w3']),
                           ('denoising_unet', WEIGHT_NAMES['w4']), ('reference_unet', WEIGHT_NAMES['w5']),
                           ('pose_guider', WEIGHT_NAMES['w6']), ('motion_module', WEIGHT_NAMES['w7']))}}
    if (any(assets.get(name, {}).get('sha256') != digest for name, digest in expected_assets.items())
            or identity.get('aa_installed', {}).get('verified_files', 0) <= 0
            or identity.get('sam_distribution', {}).get('record_hashed_sources', 0) <= 0):
        raise ValueError('preflight installed asset identity checks incomplete')
    result = preflight['result']
    if result['assets']['status'] != 0 or result['source_files'] <= 0:
        raise ValueError('preflight source or asset checks failed')
    source = Path(preflight['source_tree'])
    if not source.is_absolute():
        raise ValueError('preflight source tree must be absolute')
    for name in ('main', 'jcv'):
        receipt = preflight['receipts'][name]
        digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        if (digest != result['runtimes'][name]['receipt_sha256']
                or result['runtimes'][name]['usage']['status'] != 0
                or receipt['torch'] != EXPECTED_TORCH[name] or receipt['cuda_initialized'] is not False
                or receipt['visible_devices'] != ''
                or Path(receipt['executable']).resolve() != Path(preflight['runtimes'][name]['python']).resolve()):
            raise ValueError(f'preflight runtime receipt mismatch: {name}')
        isolation = receipt['isolation']
        if (isolation['landlock_abi'] < 1 or isolation['pre_sandbox_threads'] != 1
                or isolation['extra_fds'] is not False or isolation['network_interfaces'] != ['lo']
                or not isolation['denials'] or any(value not in (1, 13) for value in isolation['denials'].values())
                or isolation['cpu_bootstrap']['before_landlock_sha256']
                   != isolation['cpu_bootstrap']['after_landlock_sha256']):
            raise ValueError(f'preflight isolation receipt mismatch: {name}')
        if not receipt['source_modules']:
            raise ValueError('empty preflight source module closure')
        for module in receipt['source_modules'].values():
            try:
                relative = str(Path(module['path']).relative_to(source))
            except ValueError as exc:
                raise ValueError('preflight module outside frozen source') from exc
            if preflight['frozen_source_files'].get(relative) != module['sha256']:
                raise ValueError('preflight source closure mismatch')


def cost_run(stage: Path, steps: int, parent_cpu: int, packet_cpu: int, generator_cpu: int,
             worker_wall: int, preflight_path: Path, preflight_sha: str) -> None:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux one-core fleet admission required')
    if steps not in (1, 3) or type(steps) is not int:
        raise ValueError('campaign runs only 1-step smoke and 3-step full')
    if stage.name != {1: 'smoke', 3: 'full'}[steps]:
        raise ValueError('step count does not match fleet stage')
    if any(type(n) is not int or n <= 0 for n in (parent_cpu, packet_cpu, generator_cpu, worker_wall)):
        raise ValueError('positive CPU and wall limits required')
    if parent_cpu > 60 or packet_cpu > 60 or generator_cpu > 300 or worker_wall > 450:
        raise ValueError('worker limit exceeds reviewed ceiling')
    resource.setrlimit(resource.RLIMIT_CPU, (parent_cpu, 3600))
    paths, spec = stage_inputs(stage)
    if preflight_path.resolve(strict=True) != paths['preflight'] or fixed.file_sha(preflight_path) != preflight_sha:
        raise ValueError('CLI preflight path or SHA differs from staged input')
    external = spec['receiver_external_cpu_reserve_seconds']
    total_cpu = 2 * (parent_cpu + packet_cpu + generator_cpu) + 20 + external
    if (type(external) not in (int, float) or not 0 < external <= 100
            or not total_cpu <= spec['receiver_cpu_budget_seconds'] <= 2018):
        raise ValueError('campaign CPU reservations exceed remaining budget')
    fingerprints = read_oracle(paths['oracle'])
    preflight = json.loads(paths['preflight'].read_text())
    check_preflight_bundle(preflight, paths['identity'])
    identity = json.loads(paths['identity'].read_text())
    for group, (protected_path, protected_sha) in PROTECTED.items():
        item = identity['protected_inputs'][group]
        st = Path(protected_path).stat()
        if item != {'path': protected_path, 'sha256': protected_sha, 'device': st.st_dev, 'inode': st.st_ino}:
            raise ValueError(f'protected preflight identity changed: {group}')
    root = Path(__file__).resolve().parents[2]
    for name, source_sha in preflight['frozen_source_files'].items():
        source_file = (root / name).resolve(strict=True)
        if not source_file.is_relative_to(root) or fixed.file_sha(source_file) != source_sha:
            raise ValueError(f'preflight-qualified source changed: {name}')
    scratch = fixed.checked_scratch()
    aliases = scratch / 'protected-aliases'
    aliases.mkdir()
    denials = {group: [path] for group, (path, _) in PROTECTED.items()}
    denials['aliases'] = []
    for group, (protected_path, _) in PROTECTED.items():
        alias = aliases / group
        alias.symlink_to(Path(protected_path).resolve(strict=True))
        denials['aliases'].append(str(alias))
    parent_user = os.readlink('/proc/self/ns/user')
    parent_net = os.readlink('/proc/self/ns/net')
    base = extract_base(paths['base'], scratch / 'tennis-base.pt')
    decoder_work = scratch / 'packet-worker'
    decoder_work.mkdir()
    decoder_out = decoder_work / 'output'
    decoder_out.mkdir()
    model_work = scratch / 'cost-worker'
    model_work.mkdir()
    model_out = model_work / 'output'
    model_out.mkdir()
    cached_main = Path('/local/users/emanuele/pointstream/cache') / MAIN_ENV / 'tree'
    cached_jcv = Path('/local/users/emanuele/pointstream/cache') / JCV_ENV / 'tree'
    policies = preflight['runtimes']
    for name, root_path in (('main', cached_main), ('jcv', cached_jcv)):
        policy = policies[name]
        python = Path(policy['python']).resolve(strict=True)
        if not python.is_relative_to(root_path.resolve(strict=True)) or not isinstance(policy['allow_paths'], list):
            raise ValueError(f'preflight runtime identity mismatch: {name}')
    if Path(sys.executable).resolve() != Path(policies['jcv']['python']).resolve():
        raise ValueError('fleet worker must use preflight-qualified JCV runtime')
    receipt = {'path': str(paths['identity']), 'sha256': fixed.file_sha(paths['identity'])}
    def config(kind: str, files: dict[str, dict[str, str]], out: Path, policy: dict[str, Any],
               cpu: int, visible: str) -> dict[str, Any]:
        roots = [str(root), str(out), *policy['allow_paths'], *(entry['path'] for entry in files.values())]
        if any(not Path(p).is_absolute() for p in roots):
            raise ValueError('all worker paths must be absolute')
        return {'worker_kind': kind, 'unit': fixed.UNITS[0], 'cpu_seconds': cpu, 'files': files,
                'output': str(out), 'allow_paths': list(dict.fromkeys(roots)), 'denied_paths': denials,
                'parent_user_namespace': parent_user, 'parent_network_namespace': parent_net,
                'protected_receipt': receipt, 'tensor_fingerprints': fingerprints,
                'environment': {'CUDA_VISIBLE_DEVICES': visible, 'TMPDIR': str(out), 'TMP': str(out),
                                'TEMP': str(out), 'XDG_CACHE_HOME': str(out / '.cache'),
                                'TORCH_HOME': str(out / '.cache' / 'torch'),
                                'PS_GPU_MEMORY_LIMIT_MIB': str(spec['gpu_memory_mib'])}}
    packet_files = {'bundle': _item(paths['packet']), 'profile': _item(paths['profile']), 'guider': _item(paths['w6'])}
    packet_cfg = config('packet', packet_files, decoder_out, policies['main'], packet_cpu, '')
    packet_usage = _worker(Path(policies['main']['python']), packet_cfg, decoder_work, packet_cpu, worker_wall)
    decoder_manifest = decoder_out / 'decoded-manifest.json'
    generated = {'checkpoint': _item(decoder_out / 'decoded.pt'),
                 'decoder_manifest': _item(decoder_manifest),
                 'unit': _item(decoder_out / 'publish' / 'units' / f'{fixed.UNITS[0]}.npz'),
                 'base': _item(base), 'aa_archive': _item(paths['aa']),
                 **{f'weight:{name}': _item(paths[key]) for key, name in WEIGHT_NAMES.items()}}
    aa_root = (cached_jcv / 'opt' / 'Moore-AnimateAnyone' / 'animate_anyone').resolve(strict=True)
    for source_path, digest in identity['aa_installed']['files'].items():
        path = Path(source_path).resolve(strict=True)
        if not path.is_relative_to(aa_root) or path.suffix != '.py':
            continue
        parts = list(path.relative_to(aa_root).parts)
        if parts[-1] == '__init__.py':
            parts.pop()
        else:
            parts[-1] = Path(parts[-1]).stem
        name = '.'.join(['animate_anyone', *parts])
        generated[f'code:{name}'] = {'path': str(path), 'sha256': digest}
    if 'code:animate_anyone' not in generated:
        raise ValueError('preflight verified AA package source is missing')
    for name, digest in (('experiments.tennis.k6b_generate', receiver.GENERATOR_CODE_SHA),
                         ('experiments.tennis.k6b_fit', receiver.FIT_CODE_SHA)):
        current = root.joinpath(*name.split('.')).with_suffix('.py')
        if fixed.file_sha(current) != digest:
            raise ValueError(f'current generator source differs from frozen identity: {name}')
        generated[f'code:{name}'] = {'path': str(current), 'sha256': digest}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible:
        raise ValueError('fleet must supply an explicit GPU device')
    gpu_cfg = config('cost_probe', generated, model_out, policies['jcv'], generator_cpu, visible)
    gpu_cfg['probe_steps'] = steps
    gpu_usage = _worker(Path(policies['jcv']['python']), gpu_cfg, model_work, generator_cpu, worker_wall)
    expected = {'unit': fixed.UNITS[0], 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                'tensor_fingerprints': fingerprints, 'files': {k: v['sha256'] for k, v in generated.items()}}
    validation = receiver.validate_cost_probe(model_out, expected, steps)
    seal = json.loads((model_out / 'sealed.json').read_text())
    retained = retained_path(stage)
    retained.parent.mkdir(parents=True, exist_ok=True)
    retained.mkdir()  # exclusive; preserve partial evidence on failure
    for filename in (f'{fixed.UNITS[0]}@match5.npz', f'{fixed.UNITS[0]}@match5.json', 'sealed.json'):
        source = model_out / filename
        if source.is_symlink() or not source.is_file():
            raise ValueError('invalid sealed output member')
        os.link(source, retained / filename)
    for source, filename in ((decoder_out / 'decoded.pt', 'decoded.pt'),
                             (decoder_manifest, 'decoded-manifest.json'),
                             (decoder_out / 'publish' / 'units' / f'{fixed.UNITS[0]}.npz', 'packet-unit.npz')):
        if source.is_symlink() or not source.is_file():
            raise ValueError('invalid packet-derived retained input')
        os.link(source, retained / filename)
    publication = scratch / 'publish'
    publication.mkdir(exist_ok=False)
    summary = {'scope': 'qualification_only', 'steps': steps, 'unit': fixed.UNITS[0],
               'validation': validation, 'generation_seal': seal, 'packet_usage': packet_usage,
               'generation_usage': gpu_usage, 'inputs': {k: fixed.file_sha(v) for k, v in paths.items()},
               'preflight_sha256': fixed.file_sha(paths['preflight']),
               'actual_parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
               'parent_cpu_limit': parent_cpu, 'packet_cpu_limit': packet_cpu, 'generator_cpu_limit': generator_cpu,
               'retained_output': str(retained), 'quality_claim': False}
    (publication / 'cost-summary.json').write_text(json.dumps(summary, sort_keys=True))


def check_summary_resources(summary: dict[str, Any]) -> None:
    if summary.get('quality_claim') is not False or summary.get('scope') != 'qualification_only':
        raise ValueError('summary must be qualification only')
    parent = summary.get('actual_parent_cpu_seconds')
    cap = summary.get('parent_cpu_limit')
    if not isinstance(parent, (int, float)) or not isinstance(cap, int) or not 0 <= parent <= cap:
        raise ValueError('invalid parent CPU accounting')
    for usage_name, limit_name in (('packet_usage', 'packet_cpu_limit'),
                                   ('generation_usage', 'generator_cpu_limit')):
        usage, limit = summary.get(usage_name, {}), summary.get(limit_name)
        cpu, wall = usage.get('cpu_seconds'), usage.get('wall_seconds')
        if (not isinstance(limit, int) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu)
                or not isinstance(wall, (int, float)) or not math.isfinite(wall)
                or not 0 <= cpu <= limit or wall <= 0 or usage.get('status') != 0
                or usage.get('timed_out') is not False):
            raise ValueError(f'invalid worker accounting: {usage_name}')


def read_published_summary(stage: Path) -> dict[str, Any]:
    staging = json.loads((stage / 'staging.json').read_text())
    archive = stage / 'published.tar'
    if archive.stat().st_size > 2 * 1024 * 1024 or fixed.file_sha(archive) != staging['published']['sha256']:
        raise ValueError('published summary archive mismatch or oversized')
    with tarfile.open(archive) as tar:
        members = list(tar)
        entries = [entry for entry in members if entry.isfile()]
        if any(not e.isfile() and not (e.isdir() and e.name.rstrip('/') == 'publish') for e in members):
            raise ValueError('unexpected cost publication member')
        if len(entries) != 1 or entries[0].name != 'publish/cost-summary.json' or entries[0].size > 1024 * 1024:
            raise ValueError('unexpected cost publication contents')
        stream = tar.extractfile(entries[0])
        if stream is None:
            raise ValueError('unreadable cost summary')
        result = json.loads(stream.read())
    if not isinstance(result, dict):
        raise ValueError('invalid published cost summary')
    return result


def audit_retained_inputs(retained: Path, seal: dict[str, Any], fingerprints: dict[str, Any]) -> None:
    """Independently reload paid state and the selected packet-derived NPZ."""
    for name in ('decoded.pt', 'decoded-manifest.json', 'packet-unit.npz'):
        if (retained / name).is_symlink() or not (retained / name).is_file():
            raise ValueError('retained packet-derived input missing')
    checkpoint_sha = receiver.verify_serialized_match(retained / 'decoded.pt', fingerprints)
    manifest_path = retained / 'decoded-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (checkpoint_sha != seal['checkpoint_sha256'] or fixed.file_sha(manifest_path) != seal['decoder_manifest_sha256']
            or fixed.file_sha(retained / 'packet-unit.npz') != receiver.ORACLE_UNIT_SHA[fixed.UNITS[0]]
            or manifest.get('packet_sha256') != fixed.WINDOW_SHA or manifest.get('profile_sha256') != fixed.PROFILE_SHA
            or manifest.get('checkpoint_sha256') != checkpoint_sha or manifest.get('match_tensors') != fingerprints
            or manifest.get('unit_sha256') != receiver.ORACLE_UNIT_SHA):
        raise ValueError('retained packet-derived inputs do not bind the independent oracle')


def cost_validate(stage: Path, output: Path) -> int:
    """Independently re-open the retained smoke crops and the bounded publication."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (20, 3600))
        if stage.name != 'smoke':
            raise ValueError('fleet promotion validator applies only to smoke')
        paths, spec = stage_inputs(stage)
        summary = read_published_summary(stage)
        check_summary_resources(summary)
        if (summary.get('scope') != 'qualification_only' or summary.get('steps') != 1
                or summary.get('quality_claim') is not False):
            raise ValueError('smoke is not a 1-step cost-only probe')
        retained = Path(summary['retained_output']).resolve(strict=True)
        if retained != retained_path(stage).resolve(strict=True):
            raise ValueError('retained crop location differs from this stage scratch')
        seal_path = retained / 'sealed.json'
        seal = json.loads(seal_path.read_text())
        if seal != summary['generation_seal']:
            raise ValueError('published generation seal differs from retained output')
        fingerprints = read_oracle(paths['oracle'])
        audit_retained_inputs(retained, seal, fingerprints)
        expected = {'unit': fixed.UNITS[0], 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                    'tensor_fingerprints': fingerprints, 'files': seal['files']}
        validation = receiver.validate_cost_probe(retained, expected, 1)
        if validation != summary['validation']:
            raise ValueError('independent cost output validation differs')
        for key, name in WEIGHT_NAMES.items():
            if seal['files'][f'weight:{name}'] != fixed.file_sha(paths[key]):
                raise ValueError(f'cost output weight binding mismatch: {name}')
        if (summary['inputs'] != {k: fixed.file_sha(v) for k, v in paths.items()}
                or seal['files']['aa_archive'] != fixed.file_sha(paths['aa'])
                or summary['preflight_sha256'] != fixed.file_sha(paths['preflight'])):
            raise ValueError('cost output staged-input binding mismatch')
        clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
        now = time.monotonic()
        started, phase, end = clock['started'], clock['phase_started'], clock['budget_end']
        full_limit = spec['full']['seconds']
        measured = now - phase
        sample_wall = validation['phase_costs']['sampling']['gpu_reserved_wall_seconds']
        if not 0 < sample_wall <= measured:
            raise ValueError('smoke sampling duration is invalid')
        projected = 1.35 * (measured - sample_wall + 3 * sample_wall)
        stage_reserve = max(10.0, 1.25 * (phase - started))
        if (not started <= phase <= now < end or end - started > spec['budget_seconds'] + 1
                or projected > full_limit or end - now < projected + stage_reserve + 10):
            raise ValueError('remaining campaign cannot conservatively admit 3-step cost stage')
        parent_cap = summary['parent_cpu_limit']
        packet_cap = summary['packet_cpu_limit']
        generator_cap = summary['generator_cpu_limit']
        external = spec['receiver_external_cpu_reserve_seconds']
        total_cap = 2 * (parent_cap + packet_cap + generator_cap) + 20 + external
        if (not 0 < external <= 100 or total_cap > spec['receiver_cpu_budget_seconds']
                or spec['receiver_cpu_budget_seconds'] > 2018):
            raise ValueError('aggregate CPU reservation exceeds declared remaining budget')
        result = {'passed': True, 'checks': {'qualification_only': True, 'oracle_binding': True,
                  'independent_crop_validation': True, 'remaining_budget': True, 'cpu_reservation': True},
                  'scope': '1-step cost probe only, no quality claim', 'validation': validation,
                  'projection': {'measured_smoke_wall_seconds': measured, 'three_step_allowance': projected,
                                 'stage_reserve_seconds': stage_reserve, 'remaining_seconds': end - now,
                                 'cpu_reservation_seconds': total_cap}}
        output.write_text(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'cost_validation': False},
                                      'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))
        return 1


def _measured_stage_overhead(stage: Path, validation: dict[str, Any]) -> float:
    """Retain all observed non-model work in each future unit's reservation."""
    execution = json.loads((stage / 'execution.json').read_text())
    elapsed = execution.get('seconds')
    if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
            or execution.get('gpu_sharing') != {'mode': 'exclusive'}
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed <= 0):
        raise ValueError('invalid completed stage execution receipt')
    phases = validation['phase_costs']
    accounted = sum(phases[name]['gpu_reserved_wall_seconds'] for name in
                    ('model_load', 'original_probe', 'context_probe', 'sampling', 'publication'))
    if not math.isfinite(accounted) or accounted < 0 or accounted > elapsed + 0.01:
        raise ValueError('phase costs exceed full stage execution time')
    return max(0.0, elapsed - accounted)


def audit_pair(one_path: Path, three_path: Path, oracle_path: Path, remaining_gpu: float,
               downstream_reserve: float, extra_generation_overhead: float) -> dict[str, Any]:
    """Independently inspect retained crops for both bounded cost stages."""
    if one_path.parent.name != three_path.parent.name:
        raise ValueError('cost stages must belong to one fleet campaign')
    fingerprints = read_oracle(oracle_path)
    validations = []
    seals = []
    summaries = []
    for path, steps in ((one_path, 1), (three_path, 3)):
        if path.name != {1: 'smoke', 3: 'full'}[steps]:
            raise ValueError('cost audit stage mismatch')
        summary = read_published_summary(path)
        check_summary_resources(summary)
        summaries.append(summary)
        if summary.get('scope') != 'qualification_only' or summary.get('steps') != steps:
            raise ValueError('cost summary scope or step mismatch')
        output = Path(summary['retained_output']).resolve(strict=True)
        local = Path('/local/users/emanuele').resolve(strict=True)
        if not output.is_relative_to(local) or output != retained_path(path).resolve(strict=True):
            raise ValueError('cost output does not match host-local stage retention')
        seal = json.loads((output / 'sealed.json').read_text())
        if seal != summary['generation_seal']:
            raise ValueError('cost summary seal changed')
        audit_retained_inputs(output, seal, fingerprints)
        expected = {'unit': fixed.UNITS[0], 'packet_sha256': fixed.WINDOW_SHA,
                    'profile_sha256': fixed.PROFILE_SHA, 'tensor_fingerprints': fingerprints,
                    'files': seal['files']}
        result = receiver.validate_cost_probe(output, expected, steps)
        if result != summary['validation']:
            raise ValueError('independent cost validation mismatch')
        validations.append(result)
        seals.append(seal)
    if (summaries[0]['inputs'] != summaries[1]['inputs']
            or summaries[0]['preflight_sha256'] != summaries[1]['preflight_sha256']):
        raise ValueError('cost stages used different staged inputs or preflight')
    dynamic = {'checkpoint', 'decoder_manifest'}
    a = {k: v for k, v in seals[0]['files'].items() if k not in dynamic}
    b = {k: v for k, v in seals[1]['files'].items() if k not in dynamic}
    if a != b or seals[0]['device'] != seals[1]['device']:
        raise ValueError('cost stages used different installed inputs or GPU class')
    measured_overhead = max(_measured_stage_overhead(one_path, validations[0]),
                            _measured_stage_overhead(three_path, validations[1]))
    if extra_generation_overhead < measured_overhead:
        raise ValueError('per-unit setup reserve is below measured stage overhead')
    decision = projection(validations[0], validations[1], remaining_gpu, downstream_reserve,
                          extra_generation_overhead)
    decision['measured_stage_overhead_seconds_per_unit'] = measured_overhead
    return {'passed': True, 'scope': 'cost projection only, not quality or promotion authority',
            'one_step': validations[0], 'three_step': validations[1], 'projection': decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('cost-run')
    run.add_argument('--steps', type=int, required=True)
    run.add_argument('--parent-cpu', type=int, required=True)
    run.add_argument('--packet-cpu', type=int, required=True)
    run.add_argument('--generator-cpu', type=int, required=True)
    run.add_argument('--worker-wall', type=int, required=True)
    run.add_argument('--preflight', type=Path, required=True)
    run.add_argument('--preflight-sha256', required=True)
    sub.add_parser('cost-validate')
    audit = sub.add_parser('cost-audit')
    audit.add_argument('--one', type=Path, required=True)
    audit.add_argument('--three', type=Path, required=True)
    audit.add_argument('--oracle', type=Path, required=True)
    audit.add_argument('--remaining-gpu', type=float, required=True)
    audit.add_argument('--downstream-reserve', type=float, required=True)
    audit.add_argument('--extra-generation-overhead', type=float, required=True)
    audit.add_argument('--output', type=Path, required=True)
    audit.add_argument('--cpu-seconds', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'cost-audit':
        if not 1 <= args.cpu_seconds <= 20:
            raise ValueError('independent cost audit requires a CPU limit of at most 20 seconds')
        if sys.platform == 'linux' and len(os.sched_getaffinity(0)) != 1:
            raise ValueError('independent cost audit requires one CPU')
        resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds))
        result = audit_pair(args.one, args.three, args.oracle, args.remaining_gpu, args.downstream_reserve,
                            args.extra_generation_overhead)
        result['audit_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
        result['audit_cpu_limit'] = args.cpu_seconds
        with args.output.open('x') as stream:
            json.dump(result, stream, sort_keys=True)
        return 0
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.command == 'cost-validate':
        return cost_validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    cost_run(stage, args.steps, args.parent_cpu, args.packet_cpu, args.generator_cpu, args.worker_wall,
             args.preflight, args.preflight_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
