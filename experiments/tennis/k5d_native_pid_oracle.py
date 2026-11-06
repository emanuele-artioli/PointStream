"""CPU-only, non-scientific native PID/accounting oracle; no fleet admission.

The coordinator is separate from the K5d namespace init. It records owned PID
identities and native executable observation before interruption, compares the
normal namespace accounting to the coordinator's independent waited rusage,
and refuses incomplete proof. Never signals an unobserved or foreign process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tennis.k5d_pid_escalation_oracle import gone, valid_identity, write
from experiments.tennis.k5d_pidfd import Pidfd

CASES = ('normal', 'timeout', 'sigterm', 'sigkill')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def descendants(pid: int) -> list[dict[str, Any]]:
    """Read only descendants of the owned launcher, guarding PID reuse."""
    result: list[dict[str, Any]] = []
    pending = [pid]
    visited = set(pending)
    while pending:
        parent = pending.pop()
        try:
            children = (Path('/proc') / str(parent) / 'task' / str(parent) / 'children').read_text().split()
        except FileNotFoundError:
            continue
        for item in children:
            child = int(item)
            if child in visited:
                continue
            visited.add(child)
            try:
                base = Path('/proc') / str(child)
                text = (base / 'stat').read_text()
                fields = [line.split()[1:] for line in (base / 'status').read_text().splitlines()
                          if line.startswith('NSpid:')]
                pids = [int(n) for n in fields[0]]
                record = {'host_pid': child, 'namespace_pid': pids[-1], 'namespace_pids': pids,
                          'start_ticks': int(text[text.rfind(')') + 2:].split()[19]),
                          'host_pgid': int(text[text.rfind(')') + 2:].split()[2]),
                          'pid_namespace': os.readlink(base / 'ns/pid'),
                          'executable': os.readlink(base / 'exe')}
            except (FileNotFoundError, IndexError, ProcessLookupError):
                continue
            result.append(record)
            pending.append(child)
    return result


def observed_worker_group(group: dict[str, Any], members: list[dict[str, Any]], namespace: str) -> bool:
    return any(m['host_pid'] == group['host_pid'] and m['start_ticks'] == group['start_ticks']
               and m['namespace_pid'] == 2 and m['host_pgid'] == group['exec_host_pgid']
               and m['pid_namespace'] == namespace for m in members)


def one_case(root: Path, case: str, unshare: Path, native_args: list[str],
             expected_native: dict[str, str], pidfd: Pidfd) -> dict[str, Any]:
    proof = root / case
    proof.mkdir()
    outside = os.readlink('/proc/self/ns/pid')
    timeout = 20.0 if case == 'normal' else 12.0
    command = [sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'oracle-outer',
               '--unshare-sha', sha(unshare), '--timeout', str(timeout), '--', 'native-proof', *native_args,
               '--output', str(Path(os.environ['PS_SCRATCH_DIR']) / ('native-' + case))]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log = proof / 'command.log'
    with log.open('xb') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PS_K5D_PROOF_ROOT': str(proof),
                                     'PS_K5D_DETACH_NATIVE': '0' if case == 'normal' else '1'})
        try:
            owner = {'host_pid': proc.pid, 'start_ticks': int((Path('/proc') / str(proc.pid) / 'stat').read_text().split(') ')[1].split()[19])}
            write(proof / 'launcher.json', owner)
            observed: dict[tuple[int, int], dict[str, Any]] = {}
            active_native = None
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                for member in descendants(proc.pid):
                    observed[(member['host_pid'], member['start_ticks'])] = member
                    binary = Path(member['executable'])
                    if (binary.name in expected_native and (case == 'normal' or binary.name == 'SvtAv1EncApp')
                            and binary.resolve() == (Path(sys.prefix) / 'bin' / binary.name).resolve()):
                        active_native = member
                        break
                if active_native is not None:
                    break
                if proc.poll() is not None:
                    raise RuntimeError('native subprocess was not observed while alive')
                time.sleep(0.005)
            if active_native is None:
                proc.kill()
                proc.wait()
                raise TimeoutError('native subprocess observation deadline')
            init = json.loads((proof / 'init.json').read_text())['identity']
            group = json.loads((proof / 'worker-group.json').read_text())
            if (group['host_pgid'] != group['campaign_pgid']
                    or group['exec_host_pgid'] != (group['campaign_pgid'] if case == 'normal' else group['host_pid'])
                    or group['start_ticks'] <= 0
                    or (group['host_pid'], group['start_ticks']) not in observed
                    or observed[(group['host_pid'], group['start_ticks'])]['host_pgid'] != group['exec_host_pgid']):
                raise ValueError('CPU worker group proof differs')
            launcher_exe = Path(os.readlink(f'/proc/{proc.pid}/exe')).resolve(strict=True)
            launcher_stat = Path(f'/proc/{proc.pid}/stat').read_text()
            launcher_start = int(launcher_stat[launcher_stat.rfind(')') + 2:].split()[19])
            if launcher_exe != unshare or launcher_start != owner['start_ticks']:
                raise ValueError('outer launcher executable or identity differs')
            if (not valid_identity(init) or init['namespace_pid'] != 1 or init['pid_namespace'] == outside
                    or active_native['pid_namespace'] != init['pid_namespace'] or not valid_identity(active_native)):
                proc.kill()
                proc.wait()
                raise ValueError('namespace or native process identity differs')
            write(proof / 'before-interruption.json', {'case': case, 'launcher': owner, 'init': init,
                  'native': active_native, 'worker_group': group, 'observed_members': list(observed.values()), 'outside_namespace': outside,
                  'launcher_executable': str(launcher_exe), 'launcher_executable_sha256': sha(launcher_exe)})
            if case != 'normal':
                if gone(active_native):
                    raise ValueError('owned native process exited before interruption')
                fd = pidfd.open(active_native['host_pid'])
                try:
                    if gone(active_native):
                        raise ValueError('owned native process identity changed')
                    pidfd.send(fd, int(signal.SIGSTOP))
                finally:
                    os.close(fd)
            escalated = False
            if case == 'sigterm':
                proc.terminate()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    escalated = True
                    proc.kill()
            elif case == 'sigkill':
                proc.kill()
            # timeout case waits for the init's own watchdog; normal completes naturally.
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise TimeoutError('owned launcher exceeded outer deadline')
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    waited_cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    deadline = time.monotonic() + 2
    members = [init, *observed.values()]
    while not all(gone(member) for member in members):
        if time.monotonic() >= deadline:
            raise TimeoutError('recorded owned namespace member remains')
        time.sleep(0.02)
    lines = log.read_text().splitlines()
    records = [json.loads(line.removeprefix('K5D_NAMESPACE_CANDIDATE ')) for line in lines
               if line.startswith('K5D_NAMESPACE_CANDIDATE ')]
    if case == 'normal':
        if proc.returncode != 0 or len(records) != 1 or records[0]['reason'] != 'complete':
            raise ValueError('normal native workload failed')
        outputs = [json.loads(line.removeprefix('K5D_NATIVE_CPU ')) for line in lines
                   if line.startswith('K5D_NATIVE_CPU ')]
        if (len(outputs) != 1 or outputs[0]['frames'] != 2
                or outputs[0]['clip'] != 'tracknet/game9/Clip5'
                or outputs[0]['native']['temporal_units'] != 2 or outputs[0]['native']['decoded_frames'] != 2
                or any(len(outputs[0][name]) != 64 for name in ('source_rgb_sha256', 'yuv_sha256', 'decoded_sha256'))):
            raise ValueError('normal native output is incomplete')
        inner_cpu = records[0]['children_cpu_seconds'] + records[0]['init_cpu_seconds']
        # The external total also includes unshare; allow a small positive overhead.
        if not 0 <= waited_cpu - inner_cpu <= max(0.2, inner_cpu * 0.05):
            raise ValueError('outer and namespace CPU counters disagree')
    elif case == 'timeout':
        if (proc.returncode != 1 or len(records) != 1 or records[0].get('reason') != 'timeout'
                or records[0].get('complete_descendant_wait') is not True):
            raise ValueError('namespace watchdog timeout was not observed')
    elif case == 'sigterm':
        expected = -signal.SIGKILL if escalated else -signal.SIGTERM
        if proc.returncode != expected:
            raise ValueError('SIGTERM or its escalation did not determine launcher exit')
    elif case == 'sigkill' and proc.returncode != -signal.SIGKILL:
        raise ValueError('SIGKILL did not determine launcher exit')
    result = {'case': case, 'launcher': owner, 'init': init, 'native': active_native,
              'observed_members': list(observed.values()), 'worker_group': group, 'members_absent': True,
              'returncode': proc.returncode, 'escalated_to_sigkill': escalated, 'outer_waited_cpu_seconds': waited_cpu,
              'inner_records': records, 'native_output': outputs[0] if case == 'normal' else None,
              'native_commands_sha256': hashlib.sha256(json.dumps(
                  [outputs[0]['conversion_command'], outputs[0]['native']['encode_command'],
                   outputs[0]['native']['decode_command']], separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                  if case == 'normal' else None, 'log_sha256': sha(log)}
    write(proof / 'result.json', result)
    return result


def execute(args: argparse.Namespace) -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('CPU-only one-core oracle required')
    unshare = Path(args.unshare).resolve(strict=True)
    if sha(unshare) != args.unshare_sha:
        raise ValueError('unshare hash differs')
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if dispatch['environment']['gpu'] is not None:
        raise ValueError('GPU must be hidden')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.environ['PS_STAGE'] == 'full':
        publish = scratch / 'publish'
        publish.mkdir()
        write(publish / 'no-op.json', {'schema': 1, 'stage': 'full', 'scope': 'no repeated oracle or production work',
              'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
              'environment_sha256': dispatch['staged_environment']['sha256']})
        return
    archive = Path(args.archive).resolve(strict=True)
    if sha(archive) != args.archive_sha:
        raise ValueError('staged source hash differs')
    st = archive.stat()
    archive_identity = {'sha256': args.archive_sha,
                        'stat': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]}
    os.environ['PS_K5D_ARCHIVE_IDENTITY'] = json.dumps(archive_identity, sort_keys=True)
    publish = scratch / 'publish'
    publish.mkdir()
    root = publish / 'cases'
    root.mkdir()
    native_args = ['--archive', args.archive, '--archive-sha', args.archive_sha,
                   '--ffmpeg-sha', args.ffmpeg_sha, '--encoder-sha', args.encoder_sha,
                   '--decoder-sha', args.decoder_sha]
    expected_native = {'ffmpeg': args.ffmpeg_sha, 'SvtAv1EncApp': args.encoder_sha,
                       'dav1d': args.decoder_sha}
    for name, digest in expected_native.items():
        if sha(Path(sys.prefix) / 'bin' / name) != digest:
            raise ValueError('native executable hash differs: ' + name)
    pidfd = Pidfd()
    capability = pidfd.self_check()
    if 'torch' in sys.modules:
        raise ValueError('CPU oracle coordinator must not import Torch')
    from experiments.tennis import k5d_pidfd as pidfd_module
    capability.update(source_sha256=sha(Path(pidfd_module.__file__)), cuda_visible_devices='',
                      cpu_affinity=sorted(os.sched_getaffinity(0)), torch_imported=False)
    write(publish / 'pidfd-capability.json', capability)
    results = []
    for case in CASES:
        try:
            results.append(one_case(root, case, unshare, native_args, expected_native, pidfd))
        except Exception as exc:
            write(publish / f'failed-{case}.json', {'case': case, 'error': str(exc),
                  'scope': 'partial native oracle proof; do not retry automatically'})
            raise
    st = archive.stat()
    if archive_identity['stat'] != [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]:
        raise ValueError('staged archive identity changed during oracle')
    write(publish / 'oracle.json', {'schema': 1, 'archive_stat': archive_identity['stat'], 'stage': os.environ['PS_STAGE'],
          'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
          'environment_sha256': dispatch['staged_environment']['sha256'], 'unshare_sha256': args.unshare_sha,
          'archive_sha256': args.archive_sha, 'native_tools': expected_native, 'pidfd_capability': capability, 'cases': results,
          'scope': 'CPU native process containment and accounting; no GPU, quality or file-denial claim'})


def check_interruptions(cases: list[dict[str, Any]]) -> None:
    timeout, term, killed = cases[1:]
    timeout_records = timeout['inner_records']
    if (timeout['returncode'] != 1 or len(timeout_records) != 1
            or timeout_records[0].get('reason') != 'timeout'
            or timeout_records[0].get('complete_descendant_wait') is not True
            or type(term.get('escalated_to_sigkill')) is not bool
            or term['returncode'] != (-signal.SIGKILL if term['escalated_to_sigkill'] else -signal.SIGTERM)
            or killed['returncode'] != -signal.SIGKILL):
        raise ValueError('interruption cause differs from declared case')


def validate() -> None:
    import tarfile

    stage = Path(os.environ['PS_STAGE_DIR'])
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        archive = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if sha(archive) != staging['published']['sha256']:
            raise ValueError('publication hash differs')
        with tarfile.open(archive, 'r:') as bundle:
            members = bundle.getmembers()
            if any(not (m.isfile() or m.isdir()) or '..' in Path(m.name).parts or m.name.startswith('/')
                   for m in members) or sum(m.size for m in members) > 2_000_000:
                raise ValueError('invalid oracle publication')
            def read(name: str) -> dict[str, Any]:
                stream = bundle.extractfile(name)
                if stream is None:
                    raise ValueError('missing oracle artifact')
                return json.load(stream)
            record = read('publish/oracle.json')
            cases = [read(f'publish/cases/{case}/result.json') for case in CASES]
            before = [read(f'publish/cases/{case}/before-interruption.json') for case in CASES]
            capability = read('publish/pidfd-capability.json')
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        spec = json.loads((stage.parent / 'spec.json').read_text())
        arguments = spec['arguments']
        def option(name: str) -> str:
            if arguments.count(name) != 1:
                raise ValueError('spec option missing or ambiguous: ' + name)
            return arguments[arguments.index(name) + 1]
        tools = {'ffmpeg': option('--ffmpeg-sha'), 'SvtAv1EncApp': option('--encoder-sha'),
                 'dav1d': option('--decoder-sha')}
        inputs = [entry for entry in staging['inputs'] if entry['name'] == 'archive']
        if (len(inputs) != 1 or inputs[0]['sha256'] != option('--archive-sha')
                or record['archive_sha256'] != option('--archive-sha')
                or record['unshare_sha256'] != option('--unshare-sha') or record['native_tools'] != tools
                or record['pidfd_capability'] != capability
                or capability.get('source_sha256') != sha(Path(__file__).with_name('k5d_pidfd.py'))
                or capability.get('cuda_visible_devices') != '' or capability.get('torch_imported') is not False
                or not isinstance(capability.get('cpu_affinity'), list) or len(capability['cpu_affinity']) != 1
                or record['pidfd_capability'].get('backend') not in ('python', 'libc-exports', 'linux-x86_64-syscall')
                or record['pidfd_capability'].get('own_pid_signal_zero') is not True
                or not record['pidfd_capability'].get('libc_version')
                or not record['pidfd_capability'].get('python_version')
                or (capability.get('backend') == 'linux-x86_64-syscall'
                    and (capability.get('machine') != 'x86_64' or capability.get('abi_source') !=
                         'https://raw.githubusercontent.com/torvalds/linux/v6.8/arch/x86/entry/syscalls/syscall_64.tbl'))):
            raise ValueError('oracle assets differ from actual fleet specification')
        if (record['schema'] != 1 or record['stage'] != 'smoke' or record['cases'] != cases
                or record['source_sha256'] != dispatch['code']['source_sha256']
                or record['snapshot_sha256'] != dispatch['code']['snapshot_sha256']
                or record['environment_sha256'] != dispatch['staged_environment']['sha256']
                or dispatch['environment']['gpu'] is not None or [x['case'] for x in cases] != list(CASES)):
            raise ValueError('oracle execution identity differs')
        campaign_group = json.loads((stage.parent / 'run' / 'status.json').read_text())['process_group_id']
        for case, pre, result in zip(CASES, before, cases, strict=True):
            group = result['worker_group']
            observed_members = result['observed_members']
            if (group != pre['worker_group'] or group['host_pgid'] != campaign_group
                    or group['campaign_pgid'] != campaign_group
                    or group['exec_host_pgid'] != (campaign_group if case == 'normal' else group['host_pid'])
                    or not observed_worker_group(group, observed_members, result['init']['pid_namespace'])):
                raise ValueError('worker group identity or live procfs join differs')
            if (pre['case'] != case or pre['native'] != result['native'] or pre['init'] != result['init']
                    or result['members_absent'] is not True or not valid_identity(result['native'])
                    or not valid_identity(result['init']) or result['init']['namespace_pid'] != 1
                    or result['native']['pid_namespace'] != result['init']['pid_namespace']
                    or result['init']['pid_namespace'] == pre['outside_namespace']
                    or (result['returncode'] == 0) != (case == 'normal')
                    or isinstance(result['outer_waited_cpu_seconds'], bool)
                    or not isinstance(result['outer_waited_cpu_seconds'], (int, float))
                    or not math.isfinite(result['outer_waited_cpu_seconds']) or result['outer_waited_cpu_seconds'] <= 0
                    or pre['launcher_executable_sha256'] != option('--unshare-sha')
                    or (case != 'normal' and Path(result['native']['executable']).name != 'SvtAv1EncApp')):
                raise ValueError('native or teardown evidence differs')
        check_interruptions(cases)
        normal = cases[0]
        native = normal['native_output']
        cpu = normal['inner_records']
        if (not isinstance(native, dict) or native['frames'] != 2 or native['clip'] != 'tracknet/game9/Clip5'
                or native['source_sha256'] != record['archive_sha256'] or native['native']['decoded_frames'] != 2
                or native['native']['temporal_units'] != 2 or len(cpu) != 1 or cpu[0]['reason'] != 'complete'
                or cpu[0]['complete_descendant_wait'] is not True):
            raise ValueError('normal CPU/native proof differs')
        values = [cpu[0]['children_cpu_seconds'], cpu[0]['init_cpu_seconds']]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid namespace CPU counter')
        commands = [native['conversion_command'], native['native']['encode_command'], native['native']['decode_command']]
        if (normal['native_commands_sha256'] != hashlib.sha256(json.dumps(commands, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                or native['tools'] != {'ffmpeg': tools['ffmpeg'], 'encoder': tools['SvtAv1EncApp'], 'decoder': tools['dav1d']}
                or [Path(c[0]).name for c in commands] != ['ffmpeg', 'SvtAv1EncApp', 'dav1d']
                or native['native']['temporal_units'] != 2):
            raise ValueError('native command or executable binding differs')
        inner = sum(values)
        if not 0 <= normal['outer_waited_cpu_seconds'] - inner <= max(0.2, inner * 0.05):
            raise ValueError('normal CPU accounting differs')
        write(target, {'passed': True, 'checks': {'native_and_namespace': True, 'four_exit_modes': True,
              'owned_pids_absent': True, 'normal_cpu_comparison': True}})
    except Exception as exc:
        write(target, {'passed': False, 'checks': {'native_oracle': False}, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'validate'))
    parser.add_argument('--phase', choices=('smoke', 'full'))
    for name in ('archive', 'archive-sha', 'unshare', 'unshare-sha', 'ffmpeg-sha', 'encoder-sha', 'decoder-sha'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.action == 'validate':
        validate()
    else:
        if args.phase != os.environ.get('PS_STAGE'):
            parser.error('phase must match fleet stage')
        if any(getattr(args, name) is None for name in ('archive', 'archive_sha', 'unshare', 'unshare_sha',
                                                        'ffmpeg_sha', 'encoder_sha', 'decoder_sha')):
            parser.error('all pinned inputs and executable hashes are required')
        execute(args)


if __name__ == '__main__':
    main()
