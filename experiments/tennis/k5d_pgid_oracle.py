"""Prospective CPU-only real-receiver lifecycle oracle; no dispatch authority.

Each case is a separate task-owned surrogate campaign session. It exercises the
real scoped namespace and SVT receiver, not the fleet supervisor's watchdog.
An externally frozen cost proof is required before building a candidate spec.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_ada_witness as witness, k5d_checkpoint as cp
from experiments.tennis import k5d_isolated_receiver as isolated, k5d_source_oracle as oracle
from experiments.tennis.k5d_pidfd import Pidfd

CASES = ('normal', 'timeout', 'sigterm', 'sigkill', 'launcher_kill')


def spec(base: dict[str, Any], *, assets: dict[str, dict[str, str]], deadline: str,
         unshare_sha256: str, identity: dict[str, str], expected_yuv_sha256: str,
         expected_rgb_sha256: str) -> dict[str, Any]:
    """Prepare a candidate only; approval and a fresh source CPU proof are external."""
    from experiments.jobs.inbox import validate_spec

    result = witness.spec(base, action='cpu', assets=assets, deadline=deadline,
                          unshare_sha256=unshare_sha256, identity=identity,
                          expected_yuv_sha256=expected_yuv_sha256, expected_rgb_sha256=expected_rgb_sha256)
    result['scale']['proof_mode'] = {'smoke': 'ada-pgid-cpu', 'full': 'ada-pgid-cpu-noop'}
    result['scale']['namespace_timeout'] = {'smoke': 220, 'full': 4}
    result['smoke'] = {'seconds': 230, 'representative_basis':
                       'One real SVT decode and four synchronized lifecycle interruptions.'}
    result['budget_seconds'] = 265
    result['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_pgid_oracle', 'validate']
    validate_spec(result)
    return result


def _group(process: subprocess.Popen[Any]) -> int:
    if process.poll() is not None or os.getpgid(process.pid) != process.pid or process.pid == os.getpgrp():
        raise ValueError('surrogate group is absent or is the harness group')
    return process.pid


def _gone(identity: dict[str, Any]) -> bool:
    try:
        return isolated._proc_identity(identity['pid'])['birth'] != identity['birth']
    except (OSError, StopIteration, ValueError):
        return True


def _child(stage: Path, unshare: Path, unshare_sha: str, args: list[str]) -> None:
    if oracle.sha(unshare) != unshare_sha:
        raise ValueError('unshare identity differs')
    group = isolated._proc_identity(int(os.readlink('/proc/self')))['host_pgid']
    status = stage.parent / 'run' / 'status.json'
    status.parent.mkdir(parents=True, exist_ok=False)
    status.write_text(json.dumps({'process_group_id': group, 'scope': 'task-owned surrogate campaign'}))
    outside = os.readlink('/proc/self/ns/pid')
    command = [str(unshare), '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
               sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'inner', '--outside', outside,
               '--timeout', '160', '--', 'ada-denial-cpu', *args]
    os.execv(str(unshare), command)


def _wait_marker(path: Path, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass  # The production observer writes this short file in place.
        if process.poll() is not None:
            raise RuntimeError('receiver exited before observer barrier')
        time.sleep(.02)
    raise TimeoutError('receiver observer barrier not reached')


def _cleanup_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(_group(process), signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run(args: list[str]) -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('CPU-only one-core execution required')
    parsed = witness.parser().parse_args(['cpu', *args])
    if not parsed.unshare_sha256:
        raise ValueError('pinned unshare required')
    unshare_name = shutil.which('unshare')
    if not unshare_name:
        raise ValueError('unshare unavailable')
    unshare = Path(unshare_name).resolve(strict=True)
    if oracle.sha(unshare) != parsed.unshare_sha256:
        raise ValueError('unshare identity differs')
    original_stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    dispatch = json.loads((original_stage / 'dispatch.json').read_text())
    identity = cp.execution_identity()
    output = scratch / 'publish'
    output.mkdir()
    control = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               close_fds=True, start_new_session=True)
    harness_host = int(os.readlink('/proc/self'))
    control_identity = isolated._translate_child(control.pid, harness_host, os.readlink('/proc/self/ns/pid'))
    if control_identity['pid'] != control_identity['host_pgid']:
        raise ValueError('control must occupy its own process group')
    results = []
    try:
        for case in CASES:
            case_root = scratch / 'cases' / case
            stage = case_root / original_stage.parent.name / 'smoke'
            stage.mkdir(parents=True)
            (stage / 'dispatch.json').write_text(json.dumps(dispatch))
            case_scratch = case_root / 'scratch'
            case_scratch.mkdir()
            env = dict(os.environ, PS_STAGE_DIR=str(stage), PS_SCRATCH_DIR=str(case_scratch), PS_STAGE='smoke')
            env['CUDA_VISIBLE_DEVICES'] = ''
            cmd = [sys.executable, '-m', 'experiments.tennis.k5d_pgid_oracle', 'child',
                   str(stage), str(unshare), parsed.unshare_sha256, *args]
            log = case_root / 'command.log'
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            with log.open('wb') as stream:
                process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                try:
                    namespace_group = _group(process)
                    harness_host_pid = int(os.readlink('/proc/self'))
                    leader = isolated._translate_child(process.pid, harness_host_pid, os.readlink('/proc/self/ns/pid'))
                    group = leader['host_pgid']
                    if group != leader['pid']:
                        raise ValueError('surrogate leader is not its host group leader')
                    observed = _wait_marker(case_scratch / 'publish' / 'worker-observer.json', process, 160)
                    launcher = observed['launcher_observer']
                    if (launcher['host_pgid'] != group or observed['host_pgid'] != group
                            or observed['host_ppid'] != launcher['pid']):
                        raise ValueError('observed receiver lineage left surrogate group')
                    worker = isolated._proc_identity(observed['pid'])
                    live_launcher = isolated._proc_identity(launcher['pid'])
                    if worker['birth'] != observed['birth'] or live_launcher['birth'] != launcher['birth']:
                        raise ValueError('observer process birth changed')
                    native_path = case_scratch / 'publish' / 'native-observer.json'
                    native = json.loads(native_path.read_text()) if native_path.is_file() else None
                    if case != 'normal':
                        os.killpg(_group(process), signal.SIGSTOP)
                        if case == 'launcher_kill':
                            adapter = Pidfd()
                            harness_depth = len(isolated._proc_identity(harness_host_pid)['namespace_pids'])
                            launcher_pid = launcher['namespace_pids'][harness_depth - 1]
                            fd = adapter.open(launcher_pid)
                            try:
                                if isolated._proc_identity(launcher['pid'])['birth'] != launcher['birth']:
                                    raise ValueError('launcher birth changed before pidfd signal')
                                adapter.send(fd, signal.SIGKILL)
                            finally:
                                os.close(fd)
                            os.killpg(namespace_group, signal.SIGCONT)
                        elif case == 'sigterm':
                            os.killpg(_group(process), signal.SIGTERM)
                            os.killpg(_group(process), signal.SIGCONT)
                        else:
                            if case == 'timeout':
                                time.sleep(.2)  # external test watchdog, never production algorithm
                            os.killpg(_group(process), signal.SIGKILL)
                    try:
                        process.wait(timeout=160 if case == 'normal' else 10)
                    except subprocess.TimeoutExpired:
                        _cleanup_group(process)
                        raise TimeoutError('surrogate campaign did not terminate')
                except BaseException:
                    _cleanup_group(process)
                    raise
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            owned = (observed, launcher) + ((native,) if native else ())
            reap_deadline = time.monotonic() + 10
            while not all(_gone(x) for x in owned) and time.monotonic() < reap_deadline:
                time.sleep(.02)
            gone = all(_gone(x) for x in owned)
            if (not gone or control.poll() is not None or _gone(control_identity)
                    or control_identity['host_pgid'] == group):
                raise ValueError('owned process survives or unrelated control exited')
            normal = None
            if case == 'normal':
                if process.returncode != 0:
                    raise ValueError('real SVT normal case failed')
                receipt_path = case_scratch / 'publish' / 'receiver.json'
                receipt = json.loads(receipt_path.read_text())
                proof = isolated.verify_receipt(receipt, expected_context=witness.context(stage, identity, {'kind': 'cpu'}),
                                                expected_config_sha256=parsed.receiver_isolation_config_sha256)
                if (proof['campaign_host']['campaign_pgid'] != group
                        or proof['decoder']['outer_observation']['host_pgid'] != group
                        or receipt['decoded_yuv_sha256'] != parsed.expected_yuv_sha256
                        or receipt['rendered_rgb_sha256'] != parsed.expected_rgb_sha256
                        or oracle.sha(case_scratch / 'publish' / 'native.yuv') != parsed.expected_yuv_sha256
                        or oracle.sha(case_scratch / 'publish' / 'rendered.rgb') != parsed.expected_rgb_sha256):
                    raise ValueError('normal native or denial evidence differs')
                shutil.copyfile(receipt_path, output / 'normal-receiver.json')
                for name in ('native.yuv', 'rendered.rgb'):
                    shutil.copyfile(case_scratch / 'publish' / name, output / name)
                normal = {'receipt_sha256': oracle.sha(output / 'normal-receiver.json'),
                          'decoded_yuv_sha256': receipt['decoded_yuv_sha256'],
                          'rendered_rgb_sha256': receipt['rendered_rgb_sha256']}
            elif case in ('timeout', 'sigkill') and process.returncode != -signal.SIGKILL:
                raise ValueError('surrogate group kill did not terminate leader')
            elif case == 'sigterm' and process.returncode != -signal.SIGTERM:
                raise ValueError('surrogate group TERM did not terminate leader')
            elif case == 'launcher_kill' and process.returncode == 0:
                raise ValueError('launcher-only kill unexpectedly succeeded')
            log_copy = output / (case + '.log')
            shutil.copyfile(log, log_copy)
            results.append({'case': case, 'group': group, 'leader': leader, 'control': control_identity,
                            'launcher': launcher, 'worker': observed,
                            'native_at_barrier': native, 'all_observed_births_absent': gone,
                            'unrelated_control_alive': control.poll() is None, 'returncode': process.returncode,
                            'waited_cpu_seconds': (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
                            'log_sha256': oracle.sha(log_copy), 'normal': normal})
        (output / 'oracle.json').write_text(json.dumps({'schema': 1, 'execution_identity': identity,
            'cases': results, 'scope': 'task-owned surrogate campaigns; not the fleet watchdog or GPU monitor',
            'timeout_scope': 'harness-initiated surrogate group kill at synchronized barrier',
            'abort_native_scope': 'native child only where observed at barrier',
            'claims_scope': 'fleet claim removal requires independent post-job audit'}, indent=2, sort_keys=True) + '\n')
    finally:
        if control.stdin is not None:
            control.stdin.write(b'x')
            control.stdin.close()
        control.wait(timeout=10)


def validate() -> None:
    """Validate actual normal bytes and recorded lifecycle; claims are post-job only."""
    from experiments.tennis import k5d_cost_pilot as cost, k5d_ada_scale_worker as scale

    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        command = dispatch['command']
        if command.count('ada-pgid-cpu') != 1:
            raise ValueError('expected one lifecycle worker')
        args = witness.parser().parse_args(['cpu', *command[command.index('ada-pgid-cpu') + 1:]])
        identity = cp.execution_identity()
        scale.base_admission(args, identity)
        with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output:
            result = json.loads((output / 'oracle.json').read_text())
            cases = result['cases']
            if result['execution_identity'] != identity or [x['case'] for x in cases] != list(CASES):
                raise ValueError('oracle source or case set differs')
            for case in cases:
                leader, launcher, worker = case['leader'], case['launcher'], case['worker']
                native = case['native_at_barrier']
                if (leader['pid'] != case['group'] or leader['host_pgid'] != case['group']
                        or case['control']['host_pgid'] == case['group']
                        or case['control']['host_pgid'] != case['control']['pid']
                        or launcher['host_pgid'] != case['group'] or worker['host_pgid'] != case['group']
                        or worker['host_ppid'] != launcher['pid']
                        or not case['all_observed_births_absent'] or not case['unrelated_control_alive']
                        or not _gone(leader) or not _gone(worker) or not _gone(launcher)
                        or (native is not None and (not _gone(native) or native['host_pgid'] != case['group']
                                                    or native['host_ppid'] != worker['pid']))
                        or oracle.sha(output / (case['case'] + '.log')) != case['log_sha256']):
                    raise ValueError('lifecycle identity or cleanup differs')
            if (cases[0]['returncode'] != 0 or cases[1]['returncode'] != -signal.SIGKILL
                    or cases[2]['returncode'] != -signal.SIGTERM or cases[3]['returncode'] != -signal.SIGKILL
                    or cases[4]['returncode'] == 0):
                raise ValueError('case exit status differs')
            receipt_path = output / 'normal-receiver.json'
            receipt = json.loads(receipt_path.read_text())
            proof = isolated.verify_receipt(receipt, expected_context=witness.context(stage, identity, {'kind': 'cpu'}),
                                            expected_config_sha256=args.receiver_isolation_config_sha256)
            if (oracle.sha(receipt_path) != cases[0]['normal']['receipt_sha256']
                    or proof['campaign_host']['campaign_pgid'] != cases[0]['group']
                    or not _gone(proof['decoder']['outer_observation'])
                    or receipt['wire_sha256'] != args.wire_sha256 or receipt['resource_map_sha256'] != args.resources_sha256
                    or receipt['decoded_yuv_sha256'] != args.expected_yuv_sha256
                    or receipt['rendered_rgb_sha256'] != args.expected_rgb_sha256
                    or oracle.sha(output / 'native.yuv') != args.expected_yuv_sha256
                    or oracle.sha(output / 'rendered.rgb') != args.expected_rgb_sha256):
                raise ValueError('real native pixels or policy receipt differ')
        from experiments.tennis import k5d_ada_fresh as fresh
        wall, cpu_seconds = fresh.stage_receipt(stage)
        target.write_text(json.dumps({'passed': True, 'checks': {'surrogate_lifecycle': True,
            'real_svt_normal': True}, 'wall_seconds': wall, 'cpu_seconds': cpu_seconds,
            'limitation': 'fleet watchdog and post-job claims not tested'}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'child', 'validate', 'noop'))
    args, tail = parser.parse_known_args()
    if args.action == 'run':
        run(tail)
    elif args.action == 'child':
        if len(tail) < 4:
            raise ValueError('child parameters missing')
        _child(Path(tail[0]), Path(tail[1]), tail[2], tail[3:])
    elif args.action == 'validate':
        validate()
    else:
        output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        output.mkdir()
        (output / 'noop.json').write_text('{"scope":"no repeated lifecycle cases"}\n')


if __name__ == '__main__':
    main()
