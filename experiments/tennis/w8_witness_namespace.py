"""Bound W8 witness leaves inside a reaping PID namespace, preserving fleet PGID."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any

TARGETS = frozenset({'experiments.tennis.w8_strict_host_model_worker',
                     'experiments.tennis.w8_strict_receiver_worker'})
PROOF_TARGET = 'experiments.tennis.w8_witness_namespace_proof'


def command(target: str, args: list[str]) -> list[str]:
    if target == PROOF_TARGET:
        scratch = Path(os.environ.get('PS_SCRATCH_DIR', '')).resolve(strict=True)
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1
                or len(args) != 4 or args[0] != 'worker' or args[1] not in ('normal', 'timeout', 'term', 'kill')
                or args[2] != '--output' or not Path(args[3]).resolve(strict=True).is_relative_to(scratch)
                or Path(args[3]).resolve(strict=True) == scratch):
            raise ValueError('CPU proof target requires fixed scratch-bound invocation')
    elif target not in TARGETS:
        raise ValueError('unapproved witness worker')
    return [sys.executable, '-m', target, *args]


def identity(status: Path) -> dict[str, int]:
    group = json.loads(status.read_text())['process_group_id']
    raw = Path('/proc/self/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    pid, pgid, start = int(raw[:raw.find(' ')]), int(fields[2]), int(fields[19])
    if type(group) is not int or group <= 0 or pgid != group or pid <= 0 or start <= 0:
        raise ValueError('witness worker differs from campaign process group')
    return {'host_pid': pid, 'host_pgid': pgid, 'start_ticks': start, 'campaign_pgid': group}


def supervise(outside: str, timeout: float, receipt: Path, target: str, args: list[str]) -> dict[str, Any]:
    if sys.platform != 'linux' or os.getpid() != 1:
        raise ValueError('Linux PID namespace init required')
    nspid = [line.split()[1:] for line in Path('/proc/self/status').read_text().splitlines()
             if line.startswith('NSpid:')]
    current = os.readlink('/proc/self/ns/pid')
    if (not outside.startswith('pid:[') or current == outside or len(nspid) != 1
            or len(nspid[0]) < 2 or nspid[0][-1] != '1'):
        raise ValueError('distinct nested PID namespace required')
    if not math.isfinite(timeout) or not 0 < timeout < 2700:
        raise ValueError('invalid witness timeout')
    if not receipt.is_absolute() or receipt.exists() or not receipt.parent.is_dir():
        raise ValueError('fresh absolute receipt path required')
    argv = command(target, args)
    status_path = Path(os.environ['PS_STAGE_DIR']).parent / 'run' / 'status.json'
    interrupted: list[int] = []
    def stop(signum: int, _frame: Any) -> None:
        interrupted.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    read_fd, write_fd = os.pipe()
    def before_exec() -> None:
        os.close(read_fd)
        os.write(write_fd, json.dumps(identity(status_path), sort_keys=True).encode())
        os.close(write_fd)
    try:
        child = subprocess.Popen(argv, preexec_fn=before_exec, pass_fds=(write_fd,))
    finally:
        os.close(write_fd)
    try:
        worker = json.loads(os.read(read_fd, 4096))
    finally:
        os.close(read_fd)
    child_status = None
    reason = ''
    reaped = 0
    while True:
        if not reason and (interrupted or time.monotonic() - start >= timeout):
            reason = 'signal' if interrupted else 'timeout'
            os.kill(-1, signal.SIGKILL)
        try:
            pid, raw = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid:
            reaped += 1
            if pid == child.pid:
                child_status = os.waitstatus_to_exitcode(raw)
                if child_status != 0 and not reason:
                    reason = 'worker_failed'
                    os.kill(-1, signal.SIGKILL)
        else:
            time.sleep(.02)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    own = resource.getrusage(resource.RUSAGE_SELF)
    result = {'schema': 1, 'reason': reason or 'complete', 'status': child_status,
              'target': target, 'outside_namespace': outside,
              'worker_host_identity': worker, 'worker_namespace_pid': child.pid,
              'namespace_init': {'host_pid': int(nspid[0][0]), 'namespace_pid': 1, 'pid_namespace': current},
              'reaped': reaped, 'complete_descendant_wait': True,
              'elapsed_seconds': time.monotonic() - start,
              'init_cpu_seconds': own.ru_utime + own.ru_stime,
              'children_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime}
    with receipt.open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def validate_receipt(path: Path, *, expected_pgid: int, expected_outside: str,
                     expected_target: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    worker = result.get('worker_host_identity', {})
    init = result.get('namespace_init', {})
    if (expected_target not in TARGETS or result.get('schema') != 1
            or result.get('target') != expected_target or result.get('outside_namespace') != expected_outside
            or result.get('reason') != 'complete' or result.get('status') != 0
            or result.get('complete_descendant_wait') is not True or result.get('worker_namespace_pid') != 2
            or init.get('namespace_pid') != 1 or init.get('pid_namespace') == expected_outside
            or not str(init.get('pid_namespace', '')).startswith('pid:[')
            or type(expected_pgid) is not int or expected_pgid <= 0
            or worker.get('host_pgid') != expected_pgid or worker.get('campaign_pgid') != expected_pgid
            or type(worker.get('host_pid')) is not int or worker['host_pid'] <= 0
            or type(worker.get('start_ticks')) is not int or worker['start_ticks'] <= 0
            or type(result.get('reaped')) is not int or result['reaped'] < 1
            or not isinstance(result.get('elapsed_seconds'), (int, float))
            or not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] <= 0
            or not isinstance(result.get('children_cpu_seconds'), (int, float))
            or not math.isfinite(result['children_cpu_seconds']) or result['children_cpu_seconds'] < 0):
        raise ValueError('witness namespace receipt differs')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outside', required=True)
    parser.add_argument('--timeout', required=True, type=float)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not rest:
        raise ValueError('fixed target required')
    result = supervise(args.outside, args.timeout, args.receipt, rest[0], rest[1:])
    if result['reason'] != 'complete' or result['status'] != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
