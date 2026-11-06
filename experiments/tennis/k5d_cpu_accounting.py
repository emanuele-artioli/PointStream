"""Linux subreaper accounting for a single K5d stage; not a cost forecast.

A fresh wrapper adopts orphaned descendants, waits until no child remains, and
records cumulative waited-child CPU plus wrapper CPU. A command timeout kills
its session, but escaped descendants and interruption cleanup are not proven.
Do not admit GPU work until isolation and teardown are independently qualified.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import resource
import signal
import subprocess
import sys
import time
from typing import Any


def usage(which: int) -> float:
    value = resource.getrusage(which)
    return value.ru_utime + value.ru_stime


def become_subreaper() -> None:
    if sys.platform != 'linux':
        raise RuntimeError('Linux child subreaper required for complete CPU accounting')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise RuntimeError('child subreaper could not be verified')


def wait_all(deadline: float) -> int:
    """Wait for adopted orphans, without exceeding the stage allowance."""
    adopted = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid:
                adopted += 1
            elif time.monotonic() >= deadline:
                raise TimeoutError('descendants remain after accounting deadline')
            else:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        except ChildProcessError:
            return adopted
        except OSError as exc:
            if exc.errno != errno.EINTR:
                raise


def measure(command: list[str], *, phase: str, clip: str, frames: int,
            stage: str, identity: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    if stage not in ('smoke', 'full') or phase not in ('prepare', 'score') or frames <= 0:
        raise ValueError('invalid accounting scope')
    if not 0 < timeout_seconds < 2700:
        raise ValueError('accounting deadline must be below 45 minutes')
    become_subreaper()
    deadline = time.monotonic() + timeout_seconds
    before = usage(resource.RUSAGE_CHILDREN)
    child = subprocess.Popen(command, start_new_session=True)
    try:
        exit_code = child.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise TimeoutError('accounted command exceeded its deadline') from exc
    adopted = wait_all(deadline)
    children = usage(resource.RUSAGE_CHILDREN) - before
    own = usage(resource.RUSAGE_SELF)
    if exit_code != 0 or children < 0 or own < 0:
        raise RuntimeError('worker failed or CPU accounting invalid')
    receipt = {'schema': 1, 'method': 'linux-subreaper-waited-rusage-v1', 'phase': phase,
               'stage': stage, 'clip': clip, 'frames': frames, 'execution_identity': identity,
               'child_exit_code': exit_code, 'wrapper_pid': os.getpid(), 'command_pid': child.pid,
               'adopted_children_waited': adopted,
               'children_cpu_seconds': children, 'wrapper_cpu_seconds': own,
               'complete_descendant_wait': True,
               'scope': 'stage command only; excludes fleet staging, publication and validation'}
    print('K5D_CPU_ACCOUNTING ' + json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return receipt


def read_receipt(log: str, *, phase: str, stage: str, clip: str, frames: int,
                 identity: dict[str, str]) -> float:
    import math

    prefix = 'K5D_CPU_ACCOUNTING '
    lines = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError('exactly one CPU accounting record required')
    record = json.loads(lines[0])
    if (record.get('schema') != 1 or record.get('method') not in ('linux-subreaper-waited-rusage-v1', 'linux-pidns-init-waited-rusage-v1')
            or record.get('phase') != phase or record.get('stage') != stage or record.get('clip') != clip
            or record.get('frames') != frames or record.get('execution_identity') != identity
            or record.get('child_exit_code') != 0 or record.get('complete_descendant_wait') is not True
            or type(record.get('wrapper_pid')) is not int or record['wrapper_pid'] <= 0
            or type(record.get('command_pid')) is not int or record['command_pid'] <= 0
            or record['wrapper_pid'] == record['command_pid']
            or type(record.get('adopted_children_waited')) is not int or record['adopted_children_waited'] < 0
            or record.get('scope') != 'stage command only; excludes fleet staging, publication and validation'):
        raise ValueError('CPU accounting provenance differs')
    if record['method'] == 'linux-pidns-init-waited-rusage-v1':
        namespace_prefix = 'K5D_NAMESPACE_CANDIDATE '
        namespaces = [line[len(namespace_prefix):] for line in log.splitlines() if line.startswith(namespace_prefix)]
        if len(namespaces) != 1:
            raise ValueError('exactly one namespace execution record required')
        namespace = json.loads(namespaces[0])
        init = namespace.get('namespace_init')
        if (not isinstance(init, dict) or type(init.get('namespace_pid')) is not int
                or init['namespace_pid'] != 1 or record['wrapper_pid'] != 1
                or type(namespace.get('status')) is not int or namespace['status'] != 0 or namespace.get('reason') != 'complete'
                or namespace.get('complete_descendant_wait') is not True
                or type(namespace.get('worker_namespace_pid')) is not int
                or namespace['worker_namespace_pid'] != record['command_pid']
                or type(namespace.get('reaped')) is not int or namespace['reaped'] < 1
                or namespace['reaped'] - 1 != record['adopted_children_waited']
                or isinstance(namespace.get('children_cpu_seconds'), bool) or isinstance(namespace.get('init_cpu_seconds'), bool)
                or namespace.get('children_cpu_seconds') != record['children_cpu_seconds']
                or namespace.get('init_cpu_seconds') != record['wrapper_cpu_seconds']):
            raise ValueError('namespace execution and CPU accounting differ')
    values = [record.get('children_cpu_seconds'), record.get('wrapper_cpu_seconds')]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid CPU accounting values')
    total = sum(values)
    if total <= 0:
        raise ValueError('zero CPU accounting')
    return total
