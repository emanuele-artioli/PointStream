"""Candidate namespace primitives; not wired to fleet or production-admitted.

The caller must independently authenticate its fleet spec and executable. The
only permitted target is the existing K5d control-study prepare/score worker.
"""
from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TARGET = 'experiments.tennis.k5d_control_study'


def worker_command(arguments: list[str]) -> list[str]:
    """Construct, rather than accept, the executable and worker entrypoint."""
    if (not arguments or any(not isinstance(x, str) or not x for x in arguments)
            or '--accounted-child' in arguments):
        raise ValueError('invalid fixed worker arguments')
    if arguments[0] in ('prepare', 'score'):
        return [sys.executable, '-m', TARGET, '--accounted-child', *arguments]
    if arguments[0] == 'native-proof' and os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        return [sys.executable, '-m', 'experiments.tennis.k5d_native_cpu_workload', *arguments[1:]]
    raise ValueError('only fixed prepare/score or CPU native-proof workers are allowed')


def supervise(arguments: list[str], timeout: float, outside_namespace: str) -> dict[str, Any]:
    """Run as PID namespace init, reap all children and measure waited CPU.

    Must be inside an independently verified distinct PID namespace. On a
    deadline or signal, kill the entire owned namespace, including new sessions.
    The caller must authenticate the input and persist/validate the receipt.
    """
    if sys.platform != 'linux' or os.getpid() != 1:
        raise ValueError('namespace PID 1 required')
    lines = [line.split()[1:] for line in Path('/proc/self/status').read_text().splitlines()
             if line.startswith('NSpid:')]
    if (not outside_namespace.startswith('pid:[') or os.readlink('/proc/self/ns/pid') == outside_namespace
            or len(lines) != 1 or len(lines[0]) < 2 or lines[0][-1] != '1'):
        raise ValueError('distinct nested PID namespace required')
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout < 2700:
        raise ValueError('invalid namespace deadline')
    command = worker_command(arguments)
    interrupted: list[int] = []

    def stop(signum: int, _frame: Any) -> None:
        interrupted.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    from experiments.tennis.k5d_pid_escalation_oracle import process_identity, write

    proof_root = os.environ.get('PS_K5D_PROOF_ROOT')
    if proof_root:
        root = Path(proof_root).resolve(strict=True)
        scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
        if root == scratch or not root.is_relative_to(scratch):
            raise ValueError('proof root must be within scratch')
        write(root / 'init.json', {'identity': process_identity(), 'argv': command,
                                 'outside_namespace': outside_namespace})
    child = subprocess.Popen(command, start_new_session=True)
    status: int | None = None
    reaped = 0
    reason = ''
    while True:
        if not reason and (interrupted or time.monotonic() - start >= timeout):
            reason = 'signal' if interrupted else 'timeout'
            # PID namespace init is excluded from kill(-1); kernel sends only
            # to processes in its namespace. The wrapper must verify isolation.
            os.kill(-1, signal.SIGKILL)
        try:
            pid, raw = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid:
            reaped += 1
            if pid == child.pid:
                status = os.waitstatus_to_exitcode(raw)
                if status != 0 and not reason:
                    reason = 'worker_failed'
                    os.kill(-1, signal.SIGKILL)
        else:
            time.sleep(0.02)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    own = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return {'status': status, 'reason': reason or 'complete', 'reaped': reaped,
            'namespace_init': process_identity(), 'worker_namespace_pid': child.pid, 'argv': command,
            'children_cpu_seconds': cpu, 'init_cpu_seconds': own.ru_utime + own.ru_stime,
            'elapsed_seconds': time.monotonic() - start, 'complete_descendant_wait': True}


def outer(unshare_sha: str, timeout: float, arguments: list[str]) -> None:
    """Exec the selected unshare binary only for the exact fleet command."""
    import hashlib
    import json
    import shutil

    from experiments.jobs.inbox import phase_command

    stage_name = os.environ.get('PS_STAGE', '')
    if stage_name not in ('smoke', 'full'):
        raise ValueError('fleet stage required')
    stage = Path(os.environ['PS_STAGE_DIR'])
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    if (spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']
            or phase_command(spec, stage_name, staging['inputs'], sys.executable) != [sys.executable, '-m',
                'experiments.tennis.k5d_scoped_namespace', *sys.argv[1:]]):
        raise ValueError('launcher arguments differ from actual fleet stage')
    if not 0 < timeout < spec[stage_name]['seconds'] or timeout >= 2700:
        raise ValueError('namespace timeout exceeds stage')
    worker_command(arguments)
    path = shutil.which('unshare')
    if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest() != unshare_sha:
        raise ValueError('unshare binary identity differs')
    outside = os.readlink('/proc/self/ns/pid')
    cmd = [path, '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
           sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'inner',
           '--outside', outside, '--timeout', str(timeout), '--', *arguments]
    os.execv(path, cmd)


def oracle_outer(unshare_sha: str, timeout: float, arguments: list[str]) -> None:
    """Exec the same launcher for a CPU oracle bound to its parent fleet spec."""
    import hashlib
    import json
    import shutil

    from experiments.jobs.inbox import phase_command

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_STAGE') != 'smoke':
        raise ValueError('CPU smoke oracle profile required')
    stage = Path(os.environ['PS_STAGE_DIR'])
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staged = json.loads((stage / 'staging.json').read_text())['inputs']
    expected = phase_command(spec, 'smoke', staged, sys.executable)
    parent = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    parent_args = [item.decode() for item in parent if item]
    if (spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_native_pid_oracle']
            or parent_args != expected or expected[3] != 'run'):
        raise ValueError('oracle parent does not match fleet stage')
    def option(name: str) -> str:
        if expected.count(name) != 1:
            raise ValueError('oracle option differs')
        return expected[expected.index(name) + 1]
    pairs = ('--archive', '--archive-sha', '--ffmpeg-sha', '--encoder-sha', '--decoder-sha')
    if (unshare_sha != option('--unshare-sha') or len(arguments) != 13 or arguments[0] != 'native-proof'
            or any(arguments[1 + 2 * i] != name or arguments[2 + 2 * i] != option(name)
                   for i, name in enumerate(pairs)) or arguments[11] != '--output'):
        raise ValueError('oracle workload differs from fleet spec')
    output = Path(arguments[12]).resolve()
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if output.parent != scratch or not output.name.startswith('native-') or not 0 < timeout < 21:
        raise ValueError('oracle output or timeout differs')
    worker_command(arguments)
    path = shutil.which('unshare')
    if not path or Path(path).resolve() != Path(option('--unshare')).resolve() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != unshare_sha:
        raise ValueError('oracle unshare identity differs')
    outside = os.readlink('/proc/self/ns/pid')
    cmd = [path, '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
           sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'inner',
           '--outside', outside, '--timeout', str(timeout), '--', *arguments]
    os.execv(path, cmd)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    outer_parser = sub.add_parser('outer')
    outer_parser.add_argument('--unshare-sha', required=True)
    outer_parser.add_argument('--timeout', required=True, type=float)
    outer_parser.add_argument('arguments', nargs=argparse.REMAINDER)
    oracle_parser = sub.add_parser('oracle-outer')
    oracle_parser.add_argument('--unshare-sha', required=True)
    oracle_parser.add_argument('--timeout', required=True, type=float)
    oracle_parser.add_argument('arguments', nargs=argparse.REMAINDER)
    inner_parser = sub.add_parser('inner')
    inner_parser.add_argument('--outside', required=True)
    inner_parser.add_argument('--timeout', required=True, type=float)
    inner_parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.mode == 'outer':
        outer(args.unshare_sha, args.timeout, arguments)
    elif args.mode == 'oracle-outer':
        oracle_outer(args.unshare_sha, args.timeout, arguments)
    else:
        receipt = supervise(arguments, args.timeout, args.outside)
        print('K5D_NAMESPACE_CANDIDATE ' + json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        if receipt['reason'] != 'complete' or receipt['status'] != 0:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
