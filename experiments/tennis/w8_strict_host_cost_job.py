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
import stat
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_receiver_generate as receiver
from experiments.tennis import w8_strict_source as strict_source
from experiments.tennis import w8_strict_packet_job as packet_audit
from experiments.tennis import w8_public_folder_materialize as folder_profile
from experiments.tennis import w8_cuda_system_leaves as strict_policy
from experiments.tennis import w8_clock_profile
from experiments.tennis.w8_receiver_preflight import ENV_SHAS, PROTECTED

CLOCK_AUDIT_SHA = '9cf6eff76f0c0e40228d658bb439d92a261f1e87bb3ce9c546567f829759ae07'
CLOCK_AUDIT_PATH = 'docs/experiments/w8-inputs/receiver-20261101/w8-cuda-profiler-500ccc2f-independent-audit-20261105.json'
FOLDER_AUDIT_SHA = '75174fe559f3ecb4d73dbd1950eb74bc0eec2b47a470885c3b914f9f09ee600c'
FOLDER_AUDIT_PATH = 'docs/experiments/w8-inputs/receiver-20261101/w8-folder-real-io-b5eb8a21-independent-audit-20261105.json'
FOLDER_IO_WALL_FLOOR = 5.554623505100608
PACKET_AUDIT_SHA = 'c1923844722a088bbcc948c3d3d43535c31cf0440db98abbc152669287c8fc3a'
PACKET_AUDIT_PATH = 'docs/experiments/w8-inputs/receiver-20261101/w8-strict-packet-bd0b5915-independent-audit-20261105.json'
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


class StableDigests:
    """Reuse an actual digest only while the exact path's file identity is unchanged.

    The cache lives in one process and is never a substitute for the separate
    staged-input verification in another worker or validator process.
    """
    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}
        self.reads = 0
        self.bytes_read = 0
        self.cpu_seconds = 0.0

    @staticmethod
    def _stamp(info: os.stat_result) -> tuple[int, int, int, int, int]:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('digest input is not a regular file')
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def digest(self, path: Path) -> str:
        # Key the path, not its inode: distinct aliases are independently read.
        key = path.absolute()
        current = self._stamp(key.stat())
        cached = self._cache.get(key)
        if cached is not None and cached[0] == current:
            return cached[1]
        start = time.process_time()
        digest = hashlib.sha256()
        with key.open('rb') as stream:
            before = self._stamp(os.fstat(stream.fileno()))
            if before != current:
                raise ValueError('digest input changed before read')
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = self._stamp(os.fstat(stream.fileno()))
        if before != after or after != self._stamp(key.stat()):
            raise ValueError('digest input changed during read')
        result = digest.hexdigest()
        self._cache[key] = (after, result)
        self.reads += 1
        self.bytes_read += after[2]
        self.cpu_seconds += time.process_time() - start
        return result

    def receipt(self) -> dict[str, Any]:
        return {'whole_file_reads': self.reads, 'bytes_read': self.bytes_read,
                'hash_cpu_seconds': self.cpu_seconds, 'scope': 'this process only'}


def stage_inputs(stage: Path, digest_file: Callable[[Path], str] | None = None, *,
                 witness: bool = False) -> tuple[dict[str, Path], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    profile = spec.get('w8_paired_cost_profile')
    witness_profile = spec.get('w8_witness_profile')
    if witness and (profile is not None or witness_profile not in ('w8-context-pilot-v1', 'w8-context-unit1-v1',
            'w8-context-qualification-v1', 'w8-full-finish-v1', 'w8-full-unit0-v1', 'w8-full-unit1-v1')):
        raise ValueError('explicit separate context witness profile required')
    if witness and (spec.get('witness_cpu_proof_host') not in ('gpu5', 'gpu6')
                    or spec.get('hosts') != [spec['witness_cpu_proof_host']]
                    or (dispatch.get('environment') or {}).get('alias') != spec['witness_cpu_proof_host']
                    or 'RTX 6000 Ada' not in ((dispatch.get('environment') or {}).get('gpu') or {}).get('name', '')):
        raise ValueError('witness dispatch differs from CPU proof host or Ada model')
    if profile is not None and (profile != 'w8-paired-cost-800-v1'
            or spec.get('budget_seconds') != 800 or spec.get('smoke', {}).get('seconds') != 260
            or spec.get('full', {}).get('seconds') != 500 or spec.get('validator_seconds') != 60
            or spec.get('receiver_validator_cpu_seconds') != 60
            or spec.get('receiver_cpu_budget_seconds') != 1000
            or spec.get('receiver_external_cpu_reserve_seconds') != 100
            or spec.get('scale') != {'steps': {'smoke': 1, 'full': 3}}
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w8_strict_host_cost_job']
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w8_strict_host_cost_job', 'cost-validate']
            or spec.get('arguments') != ['cost-run', '--steps', '{steps}', '--parent-cpu', '30',
                 '--packet-cpu', '100', '--generator-cpu', '290', '--worker-wall', '300',
                 '--preflight', '{staged:preflight}', '--preflight-sha256',
                 'ab5f1b932918ae9956f6ef648e36e59c6a1fd9ef6c36a1722a1916b0da63a4c5',
                 '--materialize-profile', '--clock-profile']):
        raise ValueError('invalid paired cost profile')
    if (spec.get('device', 'gpu') != 'gpu' or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or len(dispatch.get('cpu_affinity', [])) != 1
            or spec.get('environment', {}).get('sha256') != JCV_ENV
            or spec.get('gpu_sharing', {'mode': 'exclusive'}) != {'mode': 'exclusive'}
            or spec.get('contention', {}).get('policy') != 'stop'
            or not 0 < spec.get('budget_seconds', 0) <= (2000 if witness and witness_profile == 'w8-full-finish-v1' else 1500 if witness and witness_profile in ('w8-full-unit0-v1', 'w8-full-unit1-v1')
                                                        else 900 if witness else 800 if profile is not None else 600)):
        raise ValueError('one-core JCV GPU campaign with a valid bounded budget required')
    wanted = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': ORACLE_SHA,
              'base': BASE_ARCHIVE_SHA, 'aa': receiver.AA_ARCHIVE_SHA,
              **{key: receiver.PINNED_WEIGHTS[name] for key, name in WEIGHT_NAMES.items()}}
    if witness:
        from experiments.tennis.w8_receiver_finish import SAM_SHA
        proof_sha = spec.get('witness_cpu_proof_sha256')
        if (not isinstance(proof_sha, str) or len(proof_sha) != 64
                or any(c not in '0123456789abcdef' for c in proof_sha)):
            raise ValueError('root-pinned CPU proof SHA required')
        wanted['cpu_proof'] = proof_sha
        if witness_profile == 'w8-context-pilot-v1':
            ledger_sha = spec.get('witness_ledger_sha256')
            if not isinstance(ledger_sha, str) or len(ledger_sha) != 64 or any(c not in '0123456789abcdef' for c in ledger_sha):
                raise ValueError('first-pilot pinned ledger SHA required')
            wanted['ledger'] = ledger_sha
            admission_sha = spec.get('witness_admission_sha256')
            if not isinstance(admission_sha, str) or len(admission_sha) != 64 or any(c not in '0123456789abcdef' for c in admission_sha):
                raise ValueError('first-pilot reviewed attempt SHA required')
            wanted['admission'] = admission_sha
        if witness_profile in ('w8-context-unit1-v1', 'w8-full-unit0-v1', 'w8-full-unit1-v1',
                               'w8-context-qualification-v1', 'w8-full-finish-v1'):
            for name, key in (('promotion', 'witness_promotion_sha256'), ('ledger', 'witness_ledger_sha256')):
                value = spec.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('root-pinned promotion and ledger SHA required')
                wanted[name] = value
        if witness_profile in ('w8-context-qualification-v1', 'w8-full-finish-v1'):
            for name in ('producer_audit', 'producer0', 'producer1'):
                value = spec.get('witness_' + name + '_sha256')
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('root-pinned producer audit and archives required')
                wanted[name] = value
            wanted.update(sam_weights=SAM_SHA,
                          native_inventory='cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b')
    declared = {x['name']: x for x in spec['staged_inputs']}
    actual = {x['name']: x for x in staging['inputs']}
    if len(declared) != len(spec['staged_inputs']) or len(actual) != len(staging['inputs']):
        raise ValueError('duplicate staged inputs')
    if set(actual) != set(wanted) | {'preflight', 'identity'} or set(declared) != set(actual):
        raise ValueError('exact staged input set required')
    for extra in ('preflight', 'identity'):
        wanted[extra] = declared[extra]['sha256']
    paths = {}
    digest_file = digest_file or fixed.file_sha
    for name, expected in wanted.items():
        if (declared[name]['sha256'] != expected or actual[name]['sha256'] != expected
                or declared[name].get('extract', False) or actual[name].get('extract', False)):
            raise ValueError(f'staged identity mismatch: {name}')
        path = Path(actual[name]['path']).resolve(strict=True)
        if witness:
            from experiments.tennis.w8_protected_metadata import reject_protected_alias
            reject_protected_alias(path)
        if digest_file(path) != expected:
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


def extract_base(archive_path: Path, out: Path, digest_file: Callable[[Path], str] | None = None) -> Path:
    digest_file = digest_file or fixed.file_sha
    if digest_file(archive_path) != BASE_ARCHIVE_SHA:
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
    if digest_file(out) != receiver.BASE_SHA:
        raise ValueError('common base checkpoint mismatch')
    return out


def _item(path: Path, digest_file: Callable[[Path], str] | None = None) -> dict[str, str]:
    return {'path': str(path), 'sha256': (digest_file or fixed.file_sha)(path)}


def _worker(python: Path, config: dict[str, Any], work: Path, cpu_seconds: int, wall_seconds: int, code_root: Path, loader: Path, *, materialize_profile: bool = False, clock_profile: bool = False, witness_mode: str | None = None) -> dict[str, Any]:
    model_modes = ('generate', 'context_qualification')
    finish_modes = ('context_rgb', 'context_sam', 'context_compose', 'rgb', 'sam', 'compose', 'isolation_proof')
    if (witness_mode not in (None, 'packet', *model_modes, *finish_modes)
            or (witness_mode is not None and config.get('worker_kind') != witness_mode)
            or (witness_mode not in (None, 'packet') and not clock_profile)
            or (witness_mode == 'packet' and (materialize_profile or clock_profile))
            or (witness_mode in model_modes and not materialize_profile)
            or (witness_mode in finish_modes and materialize_profile)):
        raise ValueError('witness worker mode or required safeguards differ')
    config_path = work / 'worker-config.json'
    config_path.write_text(json.dumps(config, sort_keys=True))
    source = code_root
    canary = work / 'policy-canary'
    canary.mkdir()
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (canary / name).write_bytes(b'canary')
    env = {'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    if witness_mode is not None:
        env.update({'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work), 'HOME': str(work),
                    'XDG_CACHE_HOME': str(work / '.cache'), 'TORCH_HOME': str(work / '.cache' / 'torch'),
                    'HF_HOME': str(work / '.cache' / 'huggingface'),
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        stage_dir = os.environ.get('PS_STAGE_DIR', '')
        if not Path(stage_dir).is_absolute() or Path(stage_dir).name not in ('smoke', 'full'):
            raise ValueError('witness requires the actual fleet stage directory')
        env['PS_STAGE_DIR'] = stage_dir
    module = ('experiments.tennis.w8_strict_receiver_worker' if witness_mode == 'packet'
              else 'experiments.tennis.w8_strict_packet_worker' if config['worker_kind'] == 'packet'
              else 'experiments.tennis.w8_strict_receiver_worker' if witness_mode in finish_modes
              else 'experiments.tennis.w8_strict_host_model_worker')
    if config['worker_kind'] in ('packet', 'isolation_proof'):
        env['CUDA_VISIBLE_DEVICES'] = ''
    prefix = ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
    outside_pidns = None
    if witness_mode is not None:
        if wall_seconds <= 10:
            raise ValueError('witness timeout must leave namespace cleanup reserve')
        outside_pidns = os.readlink('/proc/self/ns/pid')
        prefix.extend(('--pid', '--fork', '--kill-child=SIGKILL', str(python), '-m',
                       'experiments.tennis.w8_witness_namespace', '--outside', outside_pidns,
                       '--timeout', str(wall_seconds - 10), '--receipt', str(work / 'namespace-receipt.json'),
                       '--', module))
    else:
        prefix.extend((str(python), '-m', module))
    command = [*prefix, '--config', str(config_path),
               '--config-sha256', fixed.file_sha(config_path), '--loader', str(loader),
               '--loader-sha256', fixed.file_sha(loader), '--canary', str(canary),
               '--source-root', str(code_root)]
    if materialize_profile:
        if config['worker_kind'] != 'cost_probe' and witness_mode is None:
            raise ValueError('materialization applies only to model cost probes')
        command.append('--materialize-profile')
    if clock_profile:
        if (not materialize_profile and witness_mode not in finish_modes) or (config['worker_kind'] != 'cost_probe' and witness_mode is None):
            raise ValueError('clock profile applies only to materialized model cost probes')
        clock_path = work / 'clock-identity.json'
        clock_sha = w8_clock_profile.write_parent_receipt(clock_path)
        command.extend(('--clock-receipt', str(clock_path), '--clock-receipt-sha256', clock_sha))
    if witness_mode is not None:
        command.extend(('--witness-mode', witness_mode))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    parent_pid = os.getpid()
    owner_guard = None
    if witness_mode is not None:
        from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal
        owner_guard = parent_death_signal
    inherited_limit = fixed.ensure_cpu_hard_limit(cpu_seconds) if witness_mode is not None else None
    def limit() -> None:
        if owner_guard is not None:
            owner_guard(parent_pid)
        if witness_mode is not None:
            fixed.set_cpu_soft_limit(cpu_seconds, required_child=cpu_seconds)
        else:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, 3600))
    with (work / 'stdout').open('xb') as stdout, (work / 'stderr').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                close_fds=True, env=env, cwd=code_root, preexec_fn=limit)
        owned_pgid = os.getpgrp()
        (work / 'start.json').write_text(json.dumps({'pid': proc.pid, 'pgid': owned_pgid,
                                                     'started_monotonic': started}, sort_keys=True))
        status, timed_out = receiver.wait_owned_worker(proc, wall_seconds)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {'pid': proc.pid, 'pgid': owned_pgid, 'status': status, 'timed_out': timed_out,
               **({'inherited_cpu_limit': list(inherited_limit)} if inherited_limit is not None else {}),
               'wall_seconds': time.monotonic() - started,
               'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
               'config_sha256': fixed.file_sha(config_path)}
    (work / 'exit.json').write_text(json.dumps(receipt, sort_keys=True))
    if witness_mode is not None and (status != 0 or timed_out):
        from experiments.tennis import w8_witness_profile
        try:
            assert outside_pidns is not None
            w8_witness_profile.namespace_started_absence(work / 'namespace-receipt.json',
                expected_pgid=owned_pgid, expected_outside=outside_pidns, expected_target=module)
            teardown = {'owned_init_and_worker_absent': True,
                        'basis': 'PID namespace teardown; descendants not individually observed'}
        except (OSError, ValueError, KeyError, IndexError) as exc:
            teardown = {'owned_init_and_worker_absent': False, 'error': str(exc)}
        with (work / 'namespace-teardown.json').open('x') as output:
            json.dump(teardown, output, sort_keys=True)
    if witness_mode is not None and status == 0 and not timed_out:
        from experiments.tennis import w8_witness_profile
        assert outside_pidns is not None
        namespace_receipt = w8_witness_profile.namespace_completion(work / 'namespace-receipt.json',
            expected_pgid=owned_pgid, expected_outside=outside_pidns, expected_target=module)
        receipt['namespace_receipt_sha256'] = fixed.file_sha(work / 'namespace-receipt.json')
        receipt['namespace_children_cpu_seconds'] = namespace_receipt['children_cpu_seconds']
        with (work / 'namespace-validation.json').open('x') as output:
            json.dump({'receipt_sha256': receipt['namespace_receipt_sha256'],
                       'target': module, 'host_pgid': owned_pgid, 'outside': outside_pidns}, output, sort_keys=True)
    names = {'write', 'truncate', 'remove', 'rename', 'link'}
    if {x.name for x in canary.iterdir()} != names or any((canary / name).read_bytes() != b'canary' for name in names):
        raise RuntimeError('mutation canary changed')
    receipt['canaries_preserved'] = True
    if status or timed_out:
        raise RuntimeError(f'isolated worker failed; retained {work}')
    return receipt


def projection(one: dict[str, Any], three: dict[str, Any], remaining_gpu: float,
               downstream_reserve: float, extra_generation_overhead: float, *, safety_factor: float = 1.35) -> dict[str, Any]:
    """Project two original 30-step units from measured 1/3-step sampling times."""
    if (one.get('scope') != 'qualification_only' or three.get('scope') != 'qualification_only'
            or one.get('steps') != 1 or three.get('steps') != 3 or one.get('unit') != three.get('unit')):
        raise ValueError('matching 1/3-step qualification receipts required')
    a, b = one['phase_costs'], three['phase_costs']
    one_time, three_time = a['sampling']['gpu_reserved_wall_seconds'], b['sampling']['gpu_reserved_wall_seconds']
    if (not all(math.isfinite(v) for v in (one_time, three_time, remaining_gpu, downstream_reserve, extra_generation_overhead))
            or not 0 < one_time < three_time or remaining_gpu <= 0 or downstream_reserve <= 0
            or extra_generation_overhead < 0 or not math.isfinite(safety_factor) or safety_factor < 1.35):
        raise ValueError('nonpositive, nonmonotonic or invalid cost evidence')
    per_step = (three_time - one_time) / 2
    sampling_30 = one_time + 29 * per_step
    overhead = sum(max(a[k]['gpu_reserved_wall_seconds'], b[k]['gpu_reserved_wall_seconds'])
                   for k in ('model_load', 'original_probe', 'context_probe', 'publication'))
    predicted = safety_factor * 2 * (sampling_30 + overhead + extra_generation_overhead)
    return {'admission_authorized': False, 'per_step_gpu_wall_seconds': per_step, 'two_unit_30_step_seconds_upper_projection': predicted,
            'downstream_reserve_seconds': downstream_reserve,
            'extra_generation_overhead_seconds_per_unit': extra_generation_overhead,
            'remaining_gpu_seconds': remaining_gpu,
            'fits': predicted + downstream_reserve <= remaining_gpu,
            'scope': 'measured projection only; no quality evidence or runtime guarantee'}


def retained_path(stage: Path) -> Path:
    """New job-owned durable evidence directory; never reuse old experiment data."""
    if stage.name not in ('smoke', 'full') or not stage.is_absolute():
        raise ValueError('absolute fleet stage required')
    return stage / 'w8-retained'


def copy_new(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError('retained input must be a regular file')
    with source.open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out, 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    if fixed.file_sha(source) != fixed.file_sha(target):
        raise ValueError('durable retained copy differs')
    target.chmod(0o444)


def durable_member_names(materialized: bool = False) -> list[str]:
    names = ['source-manifest.json', 'loader-identity.json', 'packet-config.json', 'generator-config.json',
            f'{fixed.UNITS[0]}@match5.npz', f'{fixed.UNITS[0]}@match5.json', 'sealed.json',
            'decoded.pt', 'decoded-manifest.json', 'packet-unit.npz', 'packet-unit-other.npz', 'packet-cpu-only.json', 'protected-identity.json', 'gpu-identity.json']
    return names + (['materialization.json'] if materialized else [])


def seal_durable_archive(retained: Path, names: list[str], *, materialized: bool = False) -> dict[str, Any]:
    """Preserve an explicit fixed set of immutable files, with an exclusive archive."""
    if names != durable_member_names(materialized) or len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        raise ValueError('invalid durable archive member list')
    if sum((retained / name).stat().st_size for name in names) > 512 * 1024 * 1024:
        raise ValueError('durable artifact payload exceeds 512 MiB')
    archive = retained / 'receiver-artifacts.tar'
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w|') as tar:
            for name in names:
                member = retained / name
                if member.is_symlink() or not member.is_file():
                    raise ValueError('durable archive member missing')
                tar.add(member, arcname=name, recursive=False)
        raw.flush()
        os.fsync(raw.fileno())
    archive.chmod(0o444)
    return {'path': str(archive), 'sha256': fixed.file_sha(archive), 'bytes': archive.stat().st_size,
            'members': {name: fixed.file_sha(retained / name) for name in names}}


def verify_durable_archive(retained: Path, receipt: dict[str, Any], *, materialized: bool = False) -> None:
    archive = retained / 'receiver-artifacts.tar'
    if (receipt.get('path') != str(archive) or archive.is_symlink() or not archive.is_file()
            or archive.stat().st_size != receipt.get('bytes') or fixed.file_sha(archive) != receipt.get('sha256')):
        raise ValueError('durable archive identity mismatch')
    expected = receipt.get('members')
    if not isinstance(expected, dict) or set(expected) != set(durable_member_names(materialized)):
        raise ValueError('durable archive member identities missing')
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if len(entries) != len(expected) or {e.name for e in entries} != set(expected) or any(not e.isfile() for e in entries):
            raise ValueError('durable archive member set differs')
        for member in entries:
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('durable archive member unreadable')
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != expected[member.name] or fixed.file_sha(retained / member.name) != expected[member.name]:
                raise ValueError('durable archive member bytes differ')


def check_preflight_bundle(preflight: dict[str, Any], identity_path: Path, *, witness_metadata: bool = False) -> None:
    """Bind embedded MAIN/JCV receipts to the accepted preflight result."""
    from experiments.tennis.w8_receiver_preflight import EXPECTED_TORCH

    if witness_metadata:
        from experiments.tennis import w8_protected_metadata
        identity = w8_protected_metadata.read_historical(identity_path, preflight['identity_sha256'])
        actual_sha = preflight['identity_sha256']
    else:
        actual_sha = fixed.file_sha(identity_path)
        identity = json.loads(identity_path.read_text())
    if (preflight.get('passed') is not True or preflight.get('identity_sha256') != actual_sha
            or set(preflight.get('runtimes', {})) != {'main', 'jcv'}
            or set(preflight.get('receipts', {})) != {'main', 'jcv'}):
        raise ValueError('accepted preflight bundle and identity receipt required')
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


def check_preflight_sources(preflight: dict[str, Any], root: Path, *,
                            witness_bootstrap_sha: str | None = None,
                            witness_input_sha: str | None = None) -> None:
    """Recheck the exact preflight entrypoints and imported project closure.

    Unexecuted files in the same source archive may change in the separately
    frozen fleet revision; the fleet independently verifies that full snapshot.
    """
    source = Path(preflight['source_tree'])
    if not source.is_absolute():
        raise ValueError('preflight source tree must be absolute')
    required = {'experiments/tennis/w8_receiver_preflight.py',
                'experiments/tennis/w8_receiver_bootstrap.py'}
    for receipt in preflight['receipts'].values():
        for module in receipt['source_modules'].values():
            try:
                required.add(str(Path(module['path']).relative_to(source)))
            except ValueError as exc:
                raise ValueError('preflight source module outside frozen tree') from exc
    for name in required:
        expected = preflight['frozen_source_files'].get(name)
        if witness_bootstrap_sha is not None and name == 'experiments/tennis/w8_receiver_bootstrap.py':
            if (len(witness_bootstrap_sha) != 64 or any(c not in '0123456789abcdef' for c in witness_bootstrap_sha)):
                raise ValueError('invalid reviewed witness bootstrap identity')
            expected = witness_bootstrap_sha
        if witness_input_sha is not None and name == 'experiments/tennis/w8_input_job.py':
            if len(witness_input_sha) != 64 or any(c not in '0123456789abcdef' for c in witness_input_sha):
                raise ValueError('invalid reviewed witness input module identity')
            expected = witness_input_sha
        if not isinstance(expected, str):
            raise ValueError(f'preflight source identity missing: {name}')
        actual = (root / name).resolve(strict=True)
        if not actual.is_relative_to(root) or fixed.file_sha(actual) != expected:
            raise ValueError(f'preflight-qualified source changed: {name}')


def validator_cpu_limit(spec: dict[str, Any]) -> int:
    limit = spec.get('receiver_validator_cpu_seconds', 20)
    if type(limit) is not int or not 1 <= limit <= 60:
        raise ValueError('validator CPU limit must be between 1 and 60 seconds')
    return limit


def audit_public_rate(packet: Path, profile: Path) -> dict[str, Any]:
    """Reparse the paid bytes and compute the registered conditional rate."""
    from experiments.tennis.w8_payload import unpack_window
    if fixed.file_sha(packet) != fixed.WINDOW_SHA or fixed.file_sha(profile) != fixed.PROFILE_SHA:
        raise ValueError('public packet or profile identity mismatch')
    blob = packet.read_bytes()
    _, accounting = unpack_window(blob, json.loads(profile.read_text()))
    expected = {'frame': 23876, 'window': 2404, 'reference': 366161, 'match': 1897395}
    if accounting != expected or len(blob) != 2289836 or sum(accounting.values()) != len(blob):
        raise ValueError('public packet section accounting mismatch')
    conditional = (accounting['frame'] + accounting['window']) * 8 / 4.8 / 1000
    conditional += (accounting['reference'] + accounting['match']) * 8 / 7200 / 1000
    if not math.isclose(conditional, 46.315062222222224, rel_tol=0, abs_tol=1e-10):
        raise ValueError('public rate differs')
    return {'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
            'packet_bytes': len(blob), 'accounting': accounting, 'conditional_kbps': conditional,
            'assumption': 'exact reference and match objects reused for 7200 seconds'}


def strict_roots(root: Path, output: Path, runtime: dict[str, Any], files: dict[str, dict[str, str]]) -> list[str]:
    """Use only the authenticated runtime and the already-qualified portable system grants."""
    runtime_root = str(Path(runtime['python']).parent.parent)
    preflight = runtime['allow_paths']
    expected = {runtime_root, '/usr/lib', '/lib', '/lib64'}
    if not isinstance(preflight, list) or len(preflight) != len(expected) or set(preflight) != expected:
        raise ValueError('preflight runtime read grants differ from fixed profile')
    roots = [str(root), str(output), runtime_root, *strict_policy.SYSTEM_GRANTS,
             *(entry['path'] for entry in files.values())]
    if any(not Path(p).is_absolute() for p in roots):
        raise ValueError('all worker paths must be absolute')
    return list(dict.fromkeys(roots))


def require_gpu_uuid_api(torch: Any) -> dict[str, Any]:
    """Inspect the JCV CUDA-properties class without initializing CUDA."""
    properties = getattr(torch._C, '_CudaDeviceProperties', None)
    attribute = getattr(properties, 'uuid', None) if properties is not None else None
    if (not isinstance(attribute, property) or str(torch.__version__) != '2.7.0+cu128'
            or str(torch.version.cuda) != '12.8' or torch.cuda.is_initialized()):
        raise ValueError('JCV GPU UUID property unavailable or CUDA already initialized')
    return {'torch_version': str(torch.__version__), 'cuda_version': str(torch.version.cuda),
            'uuid_attribute_type': 'property', 'cuda_initialized': False}


def check_folder_receipt(receipt: dict[str, Any], files: dict[str, dict[str, str]]) -> None:
    rows = receipt.get('files')
    if (receipt.get('profile') != folder_profile.PROFILE or receipt.get('cleanup_complete') is not True
            or not isinstance(rows, dict) or set(rows) != set(folder_profile.SIZES)
            or receipt.get('bytes') != sum(folder_profile.SIZES.values())):
        raise ValueError('materialization receipt scope or cleanup differs')
    for name, size in folder_profile.SIZES.items():
        row, item = rows[name], files.get('weight:' + name, {})
        if (item.get('sha256') != folder_profile.PINNED[name] or row.get('source_sha256') != folder_profile.PINNED[name]
                or row.get('copy_sha256') != folder_profile.PINNED[name] or row.get('bytes') != size
                or type(row.get('copy_inode')) is not int or row['copy_inode'] <= 0):
            raise ValueError('materialized file identity differs: ' + name)
        st = Path(item['path']).stat()
        if row.get('source_stat') != [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]:
            raise ValueError('materialization source changed: ' + name)
    for key in ('wall_seconds', 'cpu_seconds', 'cleanup_wall_seconds', 'cleanup_cpu_seconds'):
        value = receipt.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('materialization timing missing: ' + key)


def materialized_three_step_forecast(measured_stage: float, sample_wall: float, worker_wall: float,
                                     setup_wall: float, folder_receipt: dict[str, Any]) -> dict[str, float]:
    """Conservatively project the entire measured stage plus three-step sampling."""
    values = (measured_stage, sample_wall, worker_wall, setup_wall)
    if (any(not math.isfinite(v) or v < 0 for v in values) or not 0 < sample_wall <= worker_wall <= measured_stage):
        raise ValueError('materialized full-stage timing differs')
    copied = folder_receipt['wall_seconds'] + folder_receipt['cleanup_wall_seconds']
    if not math.isfinite(copied) or not 0 <= copied <= worker_wall:
        raise ValueError('materialized I/O time exceeds whole worker')
    extra = max(0.0, FOLDER_IO_WALL_FLOOR - copied)
    return {'three_step_seconds': 1.5 * (measured_stage + 2 * sample_wall + extra),
            'stage_reserve_seconds': max(15.0, 1.5 * setup_wall), 'io_floor_extra_seconds': extra}


def check_gpu_binding(receipt: dict[str, Any], sealed: dict[str, Any], claimed: str, dispatch: dict[str, Any],
                      source_files: dict[str, str]) -> None:
    """Bind this stage's model/kernels to the fleet-selected physical Ada device."""
    wrapper_path = 'experiments/tennis/w8_strict_host_model_worker.py'
    if (wrapper_path not in source_files or receipt.get('wrapper') !=
            {'path': wrapper_path, 'sha256': source_files[wrapper_path]}):
        raise ValueError('model wrapper origin differs from authenticated source')
    gpu = dispatch.get('environment', {}).get('gpu') or {}
    expected = strict_policy.canonical_gpu_uuid(claimed)
    if (strict_policy.canonical_gpu_uuid(gpu.get('uuid', '')) != expected
            or strict_policy.canonical_gpu_uuid(receipt.get('device_uuid', '')) != expected
            or strict_policy.canonical_gpu_uuid(receipt.get('visible_devices', '')) != expected
            or receipt.get('cuda_device_count') != 1 or (receipt.get('major'), receipt.get('minor')) != (8, 9)
            or 'RTX 6000 Ada' not in receipt.get('name', '')
            or sealed.get('device') != {'name': receipt['name'], 'major': 8, 'minor': 9}):
        raise ValueError('model receipt is not bound to dispatched RTX 6000 Ada')
    receiver.check_execution_receipt(sealed)


def add_clock_source(generated: dict[str, dict[str, str]], root: Path, manifest: dict[str, Any]) -> None:
    """Bind the opt-in executing helper to the staged source manifest."""
    name = 'experiments.tennis.w8_clock_profile'
    relative = name.replace('.', '/') + '.py'
    expected = manifest['files'].get(relative)
    source = root / relative
    if not isinstance(expected, str) or fixed.file_sha(source) != expected:
        raise ValueError('staged clock helper source identity differs')
    generated[f'code:{name}'] = {'path': str(source), 'sha256': expected}


def check_clock_source_binding(config: dict[str, Any], origins: dict[str, str], files: dict[str, str]) -> None:
    """Join retained generator identity to the authenticated source manifest."""
    name = 'experiments.tennis.w8_clock_profile'
    relative = name.replace('.', '/') + '.py'
    entry = config['files'].get(f'code:{name}', {})
    if (origins.get(name) != relative or entry.get('path') != str(Path(config['allow_paths'][0]) / relative)
            or entry.get('sha256') != files.get(relative) or not isinstance(files.get(relative), str)):
        raise ValueError('clock helper generator config and source manifest differ')


def prepare_receiver_inputs(paths: dict[str, Path], spec: dict[str, Any], *, packet_cpu: int,
                            materialize_profile: bool, clock_profile: bool,
                            digests: StableDigests, witness: bool = False) -> dict[str, Any]:
    """Authenticate and decode the fixed packet under the existing strict policy."""
    rate = audit_public_rate(paths['packet'], paths['profile'])
    fingerprints = read_oracle(paths['oracle'])
    preflight = json.loads(paths['preflight'].read_text())
    if witness and not (isinstance(spec.get('witness_input_job_sha256'), str)
                        and len(spec['witness_input_job_sha256']) == 64
                        and all(c in '0123456789abcdef' for c in spec['witness_input_job_sha256'])):
        raise ValueError('frozen witness input module SHA required')
    check_preflight_bundle(preflight, paths['identity'], witness_metadata=witness)
    if witness:
        from experiments.tennis import w8_protected_metadata
        identity = w8_protected_metadata.read_historical(paths['identity'], preflight['identity_sha256'])
    else:
        identity = json.loads(paths['identity'].read_text())
    snapshot_root = Path(__file__).resolve().parents[2]
    if spec.get('packet_reference_audit_sha256') != PACKET_AUDIT_SHA or fixed.file_sha(snapshot_root / PACKET_AUDIT_PATH) != PACKET_AUDIT_SHA:
        raise ValueError('accepted CPU packet reference evidence differs')
    if materialize_profile and (spec.get('folder_reference_audit_sha256') != FOLDER_AUDIT_SHA
            or fixed.file_sha(snapshot_root / FOLDER_AUDIT_PATH) != FOLDER_AUDIT_SHA):
        raise ValueError('real folder I/O audit reference differs')
    check_preflight_sources(preflight, snapshot_root,
                            witness_bootstrap_sha=spec.get('witness_bootstrap_sha256') if witness else None,
                            witness_input_sha=spec.get('witness_input_job_sha256') if witness else None)
    scratch = fixed.checked_scratch()
    protected_path = scratch / 'protected-identity.json'
    if witness:
        from experiments.tennis import w8_protected_metadata
        fresh_receipt = w8_protected_metadata.capture(paths['identity'], protected_path, expected_historical_sha256=preflight['identity_sha256'])
    else:
        fresh_receipt = strict_policy.protected_identity({'denied_paths': {group: [path] for group, (path, _) in PROTECTED.items()}}, protected_path)
        packet_audit.reconcile_protected_identity(identity, json.loads(protected_path.read_text()))
    root = scratch / 'receiver-source'
    code_manifest = strict_source.stage_source(snapshot_root, root)
    strict_source.verify_source(root, code_manifest)
    loader = Path('/etc/ld.so.cache').resolve(strict=True)
    info = loader.stat()
    if str(loader) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('loader cache owner or permissions invalid')
    loader_receipt = scratch / 'loader-identity.json'
    loader_receipt.write_text(json.dumps({'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}, sort_keys=True))
    aliases = scratch / 'protected-aliases'
    aliases.mkdir()
    denials = {group: [path] for group, (path, _) in PROTECTED.items()}
    denials['aliases'] = []
    for group, (protected_name, _) in PROTECTED.items():
        alias = aliases / group
        alias.symlink_to(Path(protected_name).resolve(strict=True))
        denials['aliases'].append(str(alias))
    parent_user = os.readlink('/proc/self/ns/user')
    parent_net = os.readlink('/proc/self/ns/net')
    base = extract_base(paths['base'], scratch / 'tennis-base.pt', digests.digest)
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
    receipt = fresh_receipt
    def config(kind: str, files: dict[str, dict[str, str]], out: Path, policy: dict[str, Any],
               cpu: int, visible: str, unit: str = fixed.UNITS[0]) -> dict[str, Any]:
        roots = strict_roots(root, out, policy, files)
        return {'worker_kind': kind, 'unit': unit, 'cpu_seconds': cpu, 'files': files,
                'output': str(out), 'allow_paths': roots, 'denied_paths': denials,
                'parent_user_namespace': parent_user, 'parent_network_namespace': parent_net,
                'protected_receipt': receipt, 'tensor_fingerprints': fingerprints,
                'environment': {'CUDA_VISIBLE_DEVICES': visible, 'TMPDIR': str(out), 'TMP': str(out),
                                'TEMP': str(out), 'XDG_CACHE_HOME': str(out / '.cache'),
                                'TORCH_HOME': str(out / '.cache' / 'torch'),
                                'PS_GPU_MEMORY_LIMIT_MIB': str(spec['gpu_memory_mib'])}}
    packet_files = {'bundle': _item(paths['packet'], digests.digest),
                    'profile': _item(paths['profile'], digests.digest), 'guider': _item(paths['w6'], digests.digest)}
    packet_cfg = config('packet', packet_files, decoder_out, policies['main'], packet_cpu, '')
    packet_usage = _worker(Path(policies['main']['python']), packet_cfg, decoder_work, packet_cpu, 130, root,
                           loader_receipt, witness_mode='packet' if witness else None)
    decoder_manifest = decoder_out / 'decoded-manifest.json'
    decoded = json.loads(decoder_manifest.read_text())
    cpu_receipt = json.loads((decoder_out / 'cpu-only.json').read_text())
    packet_audit.check_grants(packet_cfg, policies['main'])
    packet_audit.check_isolation(packet_cfg, decoded['isolation'], json.loads(loader_receipt.read_text()),
                                 code_manifest, fixed.file_sha(loader_receipt))
    if (decoded.get('match_tensors') != fingerprints or decoded.get('unit_sha256') != receiver.ORACLE_UNIT_SHA
            or decoded.get('packet_sha256') != fixed.WINDOW_SHA or decoded.get('profile_sha256') != fixed.PROFILE_SHA
            or decoded.get('guider_sha256') != fixed.GUIDER_SHA
            or decoded.get('checkpoint_sha256') != packet_audit.CHECKPOINT_SHA
            or receiver.verify_serialized_match(decoder_out / 'decoded.pt', fingerprints) != packet_audit.CHECKPOINT_SHA
            or any(fixed.file_sha(decoder_out / 'publish' / 'units' / f'{name}.npz') != receiver.ORACLE_UNIT_SHA[name] for name in fixed.UNITS)
            or cpu_receipt.get('cuda_initialized_after_decode') is not False
            or cpu_receipt.get('cuda_visible_devices') != ''
            or cpu_receipt.get('torch_version') != preflight['receipts']['main']['torch']
            or cpu_receipt.get('pre_serialization', {}).get('state_tensors') != 480
            or cpu_receipt.get('pre_serialization', {}).get('restored_tensors') != 480
            or cpu_receipt.get('pre_serialization', {}).get('unit_arrays', 0) <= 0):
        raise ValueError('fresh packet CPU/oracle proof failed before model')
    generated = {'checkpoint': _item(decoder_out / 'decoded.pt', digests.digest),
                 'decoder_manifest': _item(decoder_manifest, digests.digest),
                 'unit': _item(decoder_out / 'publish' / 'units' / f'{fixed.UNITS[0]}.npz', digests.digest),
                 'base': _item(base, digests.digest), 'aa_archive': _item(paths['aa'], digests.digest),
                 **{f'weight:{name}': _item(paths[key], digests.digest) for key, name in WEIGHT_NAMES.items()}}
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
    if clock_profile:
        add_clock_source(generated, root, code_manifest)
    return {'paths': paths, 'rate': rate, 'fingerprints': fingerprints, 'preflight': preflight, 'identity': identity, 'snapshot_root': snapshot_root, 'scratch': scratch, 'protected_path': protected_path, 'fresh_receipt': fresh_receipt, 'root': root, 'code_manifest': code_manifest, 'loader_receipt': loader_receipt, 'decoder_work': decoder_work, 'decoder_out': decoder_out, 'model_work': model_work, 'model_out': model_out, 'cached_jcv': cached_jcv, 'policies': policies, 'config': config, 'packet_usage': packet_usage, 'decoder_manifest': decoder_manifest, 'generated': generated}


def cost_run(stage: Path, steps: int, parent_cpu: int, packet_cpu: int, generator_cpu: int,
             worker_wall: int, preflight_path: Path, preflight_sha: str, *, materialize_profile: bool = False,
             clock_profile: bool = False) -> None:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux one-core fleet admission required')
    if steps not in (1, 3) or type(steps) is not int:
        raise ValueError('campaign runs only 1-step smoke and 3-step full')
    if stage.name != {1: 'smoke', 3: 'full'}[steps]:
        raise ValueError('step count does not match fleet stage')
    if any(type(n) is not int or n <= 0 for n in (parent_cpu, packet_cpu, generator_cpu, worker_wall)):
        raise ValueError('positive CPU and wall limits required')
    if parent_cpu > 60 or packet_cpu > 100 or generator_cpu > 300 or worker_wall > 450:
        raise ValueError('worker limit exceeds reviewed ceiling')
    resource.setrlimit(resource.RLIMIT_CPU, (parent_cpu, 3600))
    import torch
    uuid_api = require_gpu_uuid_api(torch)
    digests = StableDigests()
    paths, spec = stage_inputs(stage, digests.digest)
    if spec.get('clock_policy_profile') != (w8_clock_profile.PROFILE if clock_profile else None) or (clock_profile and not materialize_profile):
        raise ValueError('declared clock policy and command differ')
    if clock_profile and (spec.get('clock_reference_audit_sha256') != CLOCK_AUDIT_SHA
            or fixed.file_sha(Path(__file__).resolve().parents[2] / CLOCK_AUDIT_PATH) != CLOCK_AUDIT_SHA):
        raise ValueError('synthetic CUDA clock audit identity differs')
    profile = spec.get('materialization_profile')
    if profile not in (None, folder_profile.PROFILE) or (profile == folder_profile.PROFILE) != materialize_profile:
        raise ValueError('declared materialization profile and command differ')
    if preflight_path.resolve(strict=True) != paths['preflight'] or fixed.file_sha(preflight_path) != preflight_sha:
        raise ValueError('CLI preflight path or SHA differs from staged input')
    external = spec['receiver_external_cpu_reserve_seconds']
    validator_cap = validator_cpu_limit(spec)
    total_cpu = 2 * (parent_cpu + packet_cpu + generator_cpu) + validator_cap + external
    if (type(external) not in (int, float) or not 0 < external <= 100
            or not total_cpu <= spec['receiver_cpu_budget_seconds'] <= 1000):
        raise ValueError('campaign CPU reservations exceed remaining budget')
    prepared = prepare_receiver_inputs(paths, spec, packet_cpu=packet_cpu,
        materialize_profile=materialize_profile, clock_profile=clock_profile, digests=digests)
    rate = prepared['rate']
    fingerprints = prepared['fingerprints']
    scratch = prepared['scratch']
    protected_path = prepared['protected_path']
    root = prepared['root']
    code_manifest = prepared['code_manifest']
    loader_receipt = prepared['loader_receipt']
    decoder_work = prepared['decoder_work']
    decoder_out = prepared['decoder_out']
    model_work = prepared['model_work']
    model_out = prepared['model_out']
    policies = prepared['policies']
    config = prepared['config']
    packet_usage = prepared['packet_usage']
    decoder_manifest = prepared['decoder_manifest']
    generated = prepared['generated']
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if not visible or visible != os.environ.get('PS_CLAIMED_GPU_UUID'):
        raise ValueError('fleet must supply its explicitly claimed GPU device')
    if strict_policy.canonical_gpu_uuid(visible) != strict_policy.canonical_gpu_uuid((dispatch['environment'].get('gpu') or {}).get('uuid', '')):
        raise ValueError('dispatch and claimed GPU differ')
    gpu_cfg = config('cost_probe', generated, model_out, policies['jcv'], generator_cpu, visible)
    gpu_cfg['probe_steps'] = steps
    gpu_usage = _worker(Path(policies['jcv']['python']), gpu_cfg, model_work, generator_cpu, worker_wall, root, loader_receipt, materialize_profile=materialize_profile, clock_profile=clock_profile)
    expected = {'unit': fixed.UNITS[0], 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                'tensor_fingerprints': fingerprints, 'files': {k: v['sha256'] for k, v in generated.items()}}
    validation = receiver.validate_cost_probe(model_out, expected, steps)
    device_receipt = json.loads((model_out / 'gpu-identity.json').read_text())
    sealed_device = json.loads((model_out / 'sealed.json').read_text())
    if (device_receipt.get('sealed_sha256') != fixed.file_sha(model_out / 'sealed.json')
            or device_receipt.get('config_sha256') != fixed.file_sha(model_work / 'worker-config.json')):
        raise ValueError('GPU identity is not bound to model receipt and configuration')
    check_gpu_binding(device_receipt, sealed_device, visible, dispatch, code_manifest['files'])
    if materialize_profile:
        folder_receipt = json.loads((model_out / 'materialization.json').read_text())
        check_folder_receipt(folder_receipt, gpu_cfg['files'])
        if folder_receipt['wall_seconds'] + folder_receipt['cleanup_wall_seconds'] > gpu_usage['wall_seconds']:
            raise ValueError('materialization timings exceed whole model worker')
        if (device_receipt.get('materialization_profile') != folder_profile.PROFILE
                or device_receipt.get('materialization_sha256') != fixed.file_sha(model_out / 'materialization.json')):
            raise ValueError('materialization receipt is not GPU-worker-bound')
    elif device_receipt.get('materialization_profile') is not None:
        raise ValueError('unexpected materialization profile')
    seal = json.loads((model_out / 'sealed.json').read_text())
    retained = retained_path(stage)
    retained.parent.mkdir(parents=True, exist_ok=True)
    retained.mkdir()  # exclusive; preserve partial evidence on failure
    source_manifest = scratch / 'source-manifest.json'
    source_manifest.write_text(json.dumps(code_manifest, sort_keys=True))
    for src, name in ((source_manifest, 'source-manifest.json'), (loader_receipt, 'loader-identity.json'),
                      (protected_path, 'protected-identity.json'),
                      (decoder_out / 'cpu-only.json', 'packet-cpu-only.json'),
                      (model_out / 'gpu-identity.json', 'gpu-identity.json'),
                      (decoder_work / 'worker-config.json', 'packet-config.json'),
                      (model_work / 'worker-config.json', 'generator-config.json')):
        copy_new(src, retained / name)
    for filename in (f'{fixed.UNITS[0]}@match5.npz', f'{fixed.UNITS[0]}@match5.json', 'sealed.json'):
        source = model_out / filename
        if source.is_symlink() or not source.is_file():
            raise ValueError('invalid sealed output member')
        copy_new(source, retained / filename)
    for source, filename in ((decoder_out / 'decoded.pt', 'decoded.pt'),
                             (decoder_manifest, 'decoded-manifest.json'),
                             (decoder_out / 'publish' / 'units' / f'{fixed.UNITS[0]}.npz', 'packet-unit.npz'),
                             (decoder_out / 'publish' / 'units' / f'{fixed.UNITS[1]}.npz', 'packet-unit-other.npz')):
        if source.is_symlink() or not source.is_file():
            raise ValueError('invalid packet-derived retained input')
        copy_new(source, retained / filename)
    if materialize_profile:
        copy_new(model_out / 'materialization.json', retained / 'materialization.json')
    durable_archive = seal_durable_archive(retained, durable_member_names(materialize_profile), materialized=materialize_profile)
    publication = scratch / 'publish'
    publication.mkdir(exist_ok=False)
    summary = {'scope': 'qualification_only', 'steps': steps, 'unit': fixed.UNITS[0],
               'validation': validation, 'generation_seal': seal, 'packet_usage': packet_usage, 'public_rate': rate,
               'durable_archive': durable_archive, 'materialization_profile': folder_profile.PROFILE if materialize_profile else None,
               'folder_reference_audit_sha256': FOLDER_AUDIT_SHA if materialize_profile else None,
               'source_manifest_sha256': fixed.file_sha(retained / 'source-manifest.json'),
               'loader_identity_sha256': fixed.file_sha(retained / 'loader-identity.json'),
               'protected_identity_sha256': fixed.file_sha(retained / 'protected-identity.json'),
               'packet_cpu_only_sha256': fixed.file_sha(retained / 'packet-cpu-only.json'),
               'gpu_identity_sha256': fixed.file_sha(retained / 'gpu-identity.json'),
               'packet_config_sha256': fixed.file_sha(retained / 'packet-config.json'),
               'generator_config_sha256': fixed.file_sha(retained / 'generator-config.json'),
               'generation_usage': gpu_usage, 'inputs': {k: digests.digest(v) for k, v in paths.items()},
               'hash_accounting': digests.receipt(), 'uuid_api_pre_model': uuid_api,
               'preflight_sha256': fixed.file_sha(paths['preflight']),
               'actual_parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
               'parent_cpu_limit': parent_cpu, 'packet_cpu_limit': packet_cpu, 'generator_cpu_limit': generator_cpu,
               'retained_output': str(retained), 'quality_claim': False}
    if clock_profile:
        summary['clock_policy_profile'] = w8_clock_profile.PROFILE
        summary['clock_reference_audit_sha256'] = CLOCK_AUDIT_SHA
    (publication / 'cost-summary.json').write_text(json.dumps(summary, sort_keys=True))


def check_summary_resources(summary: dict[str, Any]) -> None:
    if summary.get('uuid_api_pre_model') != {'torch_version': '2.7.0+cu128', 'cuda_version': '12.8',
                                              'uuid_attribute_type': 'property', 'cuda_initialized': False}:
        raise ValueError('pre-model GPU UUID property check missing')
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
                or usage.get('timed_out') is not False or usage.get('canaries_preserved') is not True):
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
            or fixed.file_sha(retained / 'packet-unit-other.npz') != receiver.ORACLE_UNIT_SHA[fixed.UNITS[1]]
            or manifest.get('packet_sha256') != fixed.WINDOW_SHA or manifest.get('profile_sha256') != fixed.PROFILE_SHA
            or manifest.get('checkpoint_sha256') != checkpoint_sha or manifest.get('match_tensors') != fingerprints
            or manifest.get('unit_sha256') != receiver.ORACLE_UNIT_SHA):
        raise ValueError('retained packet-derived inputs do not bind the independent oracle')


def audit_strict_policy(stage: Path, retained: Path, summary: dict[str, Any]) -> None:
    """Recheck preserved isolation and the complete code-only package against the fleet archive."""
    materialized = summary.get('materialization_profile') == folder_profile.PROFILE
    verify_durable_archive(retained, summary['durable_archive'], materialized=materialized)
    for file, key in (('source-manifest.json', 'source_manifest_sha256'), ('loader-identity.json', 'loader_identity_sha256'),
                      ('packet-config.json', 'packet_config_sha256'), ('generator-config.json', 'generator_config_sha256'),
                      ('protected-identity.json', 'protected_identity_sha256'), ('packet-cpu-only.json', 'packet_cpu_only_sha256'),
                      ('gpu-identity.json', 'gpu_identity_sha256')):
        if fixed.file_sha(retained / file) != summary[key]:
            raise ValueError('retained policy identity mismatch: ' + file)
    manifest = json.loads((retained / 'source-manifest.json').read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or any(not k.endswith('.py') or not k.startswith(('experiments/', 'src/', 'tools/')) for k in files):
        raise ValueError('strict code manifest contains non-source paths')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest.get('manifest_sha256'):
        raise ValueError('strict code manifest hash mismatch')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    archive = stage.parent / 'source.tar'
    if fixed.file_sha(archive) != dispatch['code']['snapshot_sha256']:
        raise ValueError('fleet source archive mismatch')
    with tarfile.open(archive) as bundle:
        for name, expected in files.items():
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError('source member is not regular')
            stream = bundle.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError('source member hash mismatch: ' + name)
        if materialized:
            member = bundle.getmember(FOLDER_AUDIT_PATH)
            stream = bundle.extractfile(member) if member.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != FOLDER_AUDIT_SHA:
                raise ValueError('frozen real folder I/O audit differs')
    loader = json.loads((retained / 'loader-identity.json').read_text())
    staged = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    preflight_path = Path(staged['preflight']['source'])
    if fixed.file_sha(preflight_path) != summary['preflight_sha256']:
        raise ValueError('preflight receipt identity differs')
    preflight = json.loads(preflight_path.read_text())
    identity_path = Path(staged['identity']['source'])
    if fixed.file_sha(identity_path) != summary['inputs']['identity']:
        raise ValueError('historical protected identity differs')
    packet_audit.reconcile_protected_identity(json.loads(identity_path.read_text()), json.loads((retained / 'protected-identity.json').read_text()))
    cpu_only = json.loads((retained / 'packet-cpu-only.json').read_text())
    if (cpu_only.get('cuda_visible_devices') != '' or cpu_only.get('cuda_initialized_after_decode') is not False
            or cpu_only.get('torch_version') != preflight['receipts']['main']['torch']
            or cpu_only.get('pre_serialization', {}).get('state_tensors') != 480
            or cpu_only.get('pre_serialization', {}).get('restored_tensors') != 480
            or cpu_only.get('pre_serialization', {}).get('unit_arrays', 0) <= 0):
        raise ValueError('packet CPU-only receipt incomplete')
    gpu_receipt = json.loads((retained / 'gpu-identity.json').read_text())
    sealed_gpu = json.loads((retained / 'sealed.json').read_text())
    generator_config = json.loads((retained / 'generator-config.json').read_text())
    if (gpu_receipt.get('sealed_sha256') != fixed.file_sha(retained / 'sealed.json')
            or gpu_receipt.get('config_sha256') != summary['generator_config_sha256']):
        raise ValueError('saved GPU identity binding differs')
    check_gpu_binding(gpu_receipt, sealed_gpu, generator_config['environment']['CUDA_VISIBLE_DEVICES'], dispatch, files)
    if materialized:
        folder_receipt = json.loads((retained / 'materialization.json').read_text())
        check_folder_receipt(folder_receipt, generator_config['files'])
        if (summary.get('folder_reference_audit_sha256') != FOLDER_AUDIT_SHA
                or gpu_receipt.get('materialization_profile') != folder_profile.PROFILE
                or gpu_receipt.get('materialization_sha256') != fixed.file_sha(retained / 'materialization.json')):
            raise ValueError('retained materialization binding differs')
    elif gpu_receipt.get('materialization_profile') is not None:
        raise ValueError('unexpected retained materialization profile')
    spec = json.loads((stage.parent / 'spec.json').read_text())
    clock_enabled = spec.get('clock_policy_profile') == w8_clock_profile.PROFILE
    if spec.get('clock_policy_profile') not in (None, w8_clock_profile.PROFILE):
        raise ValueError('unknown clock profile')
    if summary.get('clock_policy_profile') != (w8_clock_profile.PROFILE if clock_enabled else None):
        raise ValueError('clock profile summary differs')
    if clock_enabled:
        if (not materialized or spec.get('clock_reference_audit_sha256') != CLOCK_AUDIT_SHA
                or summary.get('clock_reference_audit_sha256') != CLOCK_AUDIT_SHA):
            raise ValueError('clock audit reference differs')
        with tarfile.open(archive) as bundle:
            entry = bundle.getmember(CLOCK_AUDIT_PATH)
            stream = bundle.extractfile(entry) if entry.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != CLOCK_AUDIT_SHA:
                raise ValueError('frozen synthetic CUDA clock audit differs')
    records = (('packet-config.json', json.loads((retained / 'decoded-manifest.json').read_text())['isolation']),
               ('generator-config.json', json.loads((retained / 'sealed.json').read_text())['isolation']))
    for filename, isolation in records:
        config = json.loads((retained / filename).read_text())
        name = 'main' if filename == 'packet-config.json' else 'jcv'
        expected_roots = strict_roots(Path(config['allow_paths'][0]), Path(config['output']), preflight['runtimes'][name], config['files'])
        if (len(config['allow_paths']) != len(expected_roots) or len(set(config['allow_paths'])) != len(expected_roots)
                or set(config['allow_paths']) != set(expected_roots)
                or config['protected_receipt']['sha256'] != summary['protected_identity_sha256']):
            raise ValueError('strict runtime and file grants or protected identity differ')
        denied = {p for paths in config['denied_paths'].values() for p in paths}
        if (len(denied) != 10 or set(isolation['denials']) != denied or any(v not in (1, 13) for v in isolation['denials'].values())
                or isolation['landlock_abi'] < 3 or isolation['handled_mask'] != strict_policy.handled_rights(isolation['landlock_abi'])
                or isolation['policy'] != 'w8-approved-system-leaves-mutation-v1'
                or isolation['loader_identity_sha256'] != summary['loader_identity_sha256']
                or isolation['output_write_ok'] is not True or isolation['network_interfaces'] != ['lo']
                or isolation['pre_sandbox_threads'] != 1 or isolation['extra_fds'] is not False
                or isolation['cpu_bootstrap']['before_landlock_sha256'] != isolation['cpu_bootstrap']['after_landlock_sha256']
                or isolation['cpu_bootstrap']['cuda_initialized_before_isolation'] is not False
                or isolation['namespaces']['user'] == config['parent_user_namespace']
                or isolation['namespaces']['net'] == config['parent_network_namespace']):
            raise ValueError('strict source-hidden isolation incomplete')
        origins = isolation.get('source_origins')
        if (not isinstance(origins, dict) or not origins
                or 'experiments.tennis.w8_receiver_generate' not in origins
                or any(not isinstance(value, str) or value not in files
                       or value not in (key.replace('.', '/') + '.py', key.replace('.', '/') + '/__init__.py')
                       for key, value in origins.items())):
            raise ValueError('pre-isolation project source origins differ from pinned source manifest')
        strict_policy.validate_leaf_receipt(isolation['approved_leaves'], loader)
        if filename == 'generator-config.json' and clock_enabled:
            check_clock_source_binding(config, origins, files)
            w8_clock_profile.validate_receipt(isolation.get('clock_profile'))
        elif isolation.get('clock_profile') is not None:
            raise ValueError('unexpected clock policy in worker')
        mutation = isolation['mutation_denials']
        if (set(mutation) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
                or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in mutation.items())):
            raise ValueError('strict mutation controls incomplete')


def cost_validate(stage: Path, output: Path) -> int:
    """Independently re-open the retained smoke crops and the bounded publication."""
    try:
        pre_spec = json.loads((stage.parent / 'spec.json').read_text())
        validator_cap = validator_cpu_limit(pre_spec)
        resource.setrlimit(resource.RLIMIT_CPU, (validator_cap, 3600))
        if stage.name != 'smoke':
            raise ValueError('fleet promotion validator applies only to smoke')
        digests = StableDigests()
        paths, spec = stage_inputs(stage, digests.digest)
        summary = read_published_summary(stage)
        if summary['public_rate'] != audit_public_rate(paths['packet'], paths['profile']):
            raise ValueError('public rate recheck differs')
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
        audit_strict_policy(stage, retained, summary)
        expected = {'unit': fixed.UNITS[0], 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                    'tensor_fingerprints': fingerprints, 'files': seal['files']}
        validation = receiver.validate_cost_probe(retained, expected, 1)
        if validation != summary['validation']:
            raise ValueError('independent cost output validation differs')
        for key, name in WEIGHT_NAMES.items():
            if seal['files'][f'weight:{name}'] != digests.digest(paths[key]):
                raise ValueError(f'cost output weight binding mismatch: {name}')
        if (summary['inputs'] != {k: digests.digest(v) for k, v in paths.items()}
                or seal['files']['aa_archive'] != digests.digest(paths['aa'])
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
        profile = spec.get('materialization_profile')
        materialized = profile == folder_profile.PROFILE
        if profile not in (None, folder_profile.PROFILE) or summary.get('materialization_profile') != profile:
            raise ValueError('materialization stage profile differs')
        if materialized:
            audit = Path(__file__).resolve().parents[2] / FOLDER_AUDIT_PATH
            if spec.get('folder_reference_audit_sha256') != FOLDER_AUDIT_SHA or fixed.file_sha(audit) != FOLDER_AUDIT_SHA:
                raise ValueError('real folder I/O audit reference differs')
            io = json.loads((retained / 'materialization.json').read_text())
            forecast = materialized_three_step_forecast(measured, sample_wall, summary['generation_usage']['wall_seconds'],
                                                        phase - started, io)
            io_extra = forecast['io_floor_extra_seconds']
            projected = forecast['three_step_seconds']
            stage_reserve = forecast['stage_reserve_seconds']
        else:
            io_extra = 0.0
            projected = 1.35 * (measured - sample_wall + 3 * sample_wall)
            stage_reserve = max(10.0, 1.25 * (phase - started))
        if (not started <= phase <= now < end or end - started > spec['budget_seconds'] + 1
                or projected > full_limit or end - now < projected + stage_reserve + 10):
            raise ValueError('remaining campaign cannot conservatively admit 3-step cost stage')
        parent_cap = summary['parent_cpu_limit']
        packet_cap = summary['packet_cpu_limit']
        generator_cap = summary['generator_cpu_limit']
        external = spec['receiver_external_cpu_reserve_seconds']
        if validator_cpu_limit(spec) != validator_cap:
            raise ValueError('validator CPU limit changed during validation')
        total_cap = 2 * (parent_cap + packet_cap + generator_cap) + validator_cap + external
        if (not 0 < external <= 100 or total_cap > spec['receiver_cpu_budget_seconds']
                or spec['receiver_cpu_budget_seconds'] > 2018):
            raise ValueError('aggregate CPU reservation exceeds declared remaining budget')
        result = {'passed': True, 'checks': {'qualification_only': True, 'oracle_binding': True,
                  'independent_crop_validation': True, 'remaining_budget': True, 'cpu_reservation': True},
                  'scope': '1-step cost probe only, no quality claim', 'validation': validation,
                  'hash_accounting': digests.receipt(),
                  'projection': {'measured_smoke_wall_seconds': measured, 'three_step_allowance': projected, 'io_floor_extra_seconds': io_extra,
                                 'stage_reserve_seconds': stage_reserve, 'remaining_seconds': end - now,
                                 'cpu_reservation_seconds': total_cap,
                                 'validator_cpu_limit_seconds': validator_cap}}
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
        staged = {x['name']: x for x in json.loads((path / 'staging.json').read_text())['inputs']}
        if summary['public_rate'] != audit_public_rate(Path(staged['packet']['source']), Path(staged['profile']['source'])):
            raise ValueError('public rate recheck differs')
        check_summary_resources(summary)
        summaries.append(summary)
        if summary.get('scope') != 'qualification_only' or summary.get('steps') != steps:
            raise ValueError('cost summary scope or step mismatch')
        output = Path(summary['retained_output']).resolve(strict=True)
        if output != retained_path(path).resolve(strict=True):
            raise ValueError('cost output does not match durable stage retention')
        seal = json.loads((output / 'sealed.json').read_text())
        if seal != summary['generation_seal']:
            raise ValueError('cost summary seal changed')
        audit_retained_inputs(output, seal, fingerprints)
        audit_strict_policy(path, output, summary)
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
    profile = summaries[0].get('materialization_profile')
    materialized = profile == folder_profile.PROFILE
    if profile not in (None, folder_profile.PROFILE) or summaries[1].get('materialization_profile') != profile:
        raise ValueError('cost stages used different materialization profiles')
    staging_reserve = 0.0
    validator_reserve = 0.0
    if materialized:
        for path in (one_path, three_path):
            staged = json.loads((path / 'staging.json').read_text())
            value = sum(float(x['seconds']) for x in staged['inputs']) + float(staged['environment']['seconds'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('measured staging receipt invalid')
            staging_reserve = max(staging_reserve, value)
        gate = json.loads((one_path.parent / 'gate.json').read_text())
        validator_reserve = float(gate['validator_seconds'])
        if gate.get('passed') is not True or not math.isfinite(validator_reserve) or validator_reserve < 0:
            raise ValueError('passed validator timing required')
    required_overhead = measured_overhead + staging_reserve + validator_reserve
    if extra_generation_overhead < required_overhead:
        raise ValueError('per-unit setup reserve is below measured stage, staging and validation overhead')
    decision = projection(validations[0], validations[1], remaining_gpu, downstream_reserve,
                          extra_generation_overhead, safety_factor=1.5 if materialized else 1.35)
    decision['measured_stage_overhead_seconds_per_unit'] = measured_overhead
    decision['measured_staging_seconds_per_unit'] = staging_reserve
    decision['measured_validator_seconds_per_unit'] = validator_reserve
    return {'passed': True, 'admission_authorized': False, 'provenance_scope': 'stage-artifact calculation; terminal job and authoritative ledger require independent review',
            'scope': 'advisory cost projection only, no quality or promotion authority',
            'one_step': validations[0], 'three_step': validations[1], 'projection': decision}


def make_materialized_spec(deadline: str) -> dict[str, Any]:
    spec = make_spec(deadline)
    spec['arguments'].append('--materialize-profile')
    spec['materialization_profile'] = folder_profile.PROFILE
    spec['folder_reference_audit_sha256'] = FOLDER_AUDIT_SHA
    spec['smoke']['representative_basis'] = ('One complete 240-frame unit with one actual generation step, '
        'strict source-denied receiver and SHA-bound regular-file materialization; whole-stage measured 1.5-margin admission precedes the three-step cost pilot.')
    return spec


def make_spec(deadline: str) -> dict[str, Any]:
    base_path = Path(__file__).resolve().parents[2] / 'docs/experiments/w8-inputs/receiver-20261101/fleet-cost-b581-proposed.json'
    spec = json.loads(base_path.read_text())
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_strict_host_cost_job']
    spec['arguments'] = ['cost-run', '--steps', '{steps}', '--parent-cpu', '30', '--packet-cpu', '100',
                         '--generator-cpu', '210', '--worker-wall', '230', '--preflight', '{staged:preflight}',
                         '--preflight-sha256', 'ab5f1b932918ae9956f6ef648e36e59c6a1fd9ef6c36a1722a1916b0da63a4c5']
    spec['scale'] = {'steps': {'smoke': 1, 'full': 3}}
    spec['smoke'] = {'seconds': 260, 'representative_basis': 'One real generation step across a complete 240-frame unit under the strict namespace/12-leaf policy; independent packet, source and cost validation before the 3-step pilot.'}
    spec['full'] = {'seconds': 300}
    spec['budget_seconds'] = 600
    spec['gpu_sharing'] = {'mode': 'exclusive'}
    spec['validator_seconds'] = 30
    spec['receiver_validator_cpu_seconds'] = 30
    spec['receiver_external_cpu_reserve_seconds'] = 100
    spec['receiver_cpu_budget_seconds'] = 900
    spec['packet_reference_audit_sha256'] = PACKET_AUDIT_SHA
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_strict_host_cost_job', 'cost-validate']
    spec['deadline'] = deadline
    return spec


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('make-spec')
    make.add_argument('--deadline', required=True)
    make.add_argument('--output', type=Path, required=True)
    make.add_argument('--materialize-profile', action='store_true')
    run = sub.add_parser('cost-run')
    run.add_argument('--steps', type=int, required=True)
    run.add_argument('--parent-cpu', type=int, required=True)
    run.add_argument('--packet-cpu', type=int, required=True)
    run.add_argument('--generator-cpu', type=int, required=True)
    run.add_argument('--worker-wall', type=int, required=True)
    run.add_argument('--preflight', type=Path, required=True)
    run.add_argument('--preflight-sha256', required=True)
    run.add_argument('--materialize-profile', action='store_true')
    run.add_argument('--clock-profile', action='store_true')
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
    if args.command == 'make-spec':
        spec = make_materialized_spec(args.deadline) if args.materialize_profile else make_spec(args.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0
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
             args.preflight, args.preflight_sha256, materialize_profile=args.materialize_profile,
             clock_profile=args.clock_profile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
