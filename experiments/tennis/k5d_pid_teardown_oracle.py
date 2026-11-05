"""Synthetic PID-namespace teardown oracle; no model, GPU or production release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

CASES = ('normal', 'timeout', 'sigterm', 'sigkill')
SCOPE = 'synthetic PID namespace only; no native, CUDA or production qualification'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, record: dict[str, Any]) -> None:
    with path.open('x') as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def process_identity() -> dict[str, Any]:
    status = Path('/proc/self/status').read_text()
    fields = [line.split()[1:] for line in status.splitlines() if line.startswith('NSpid:')]
    if len(fields) != 1 or len(fields[0]) < 2:
        raise ValueError('host and namespace PIDs unavailable')
    pids = [int(p) for p in fields[0]]
    pid = pids[0]
    stat = Path('/proc/self/stat').read_text()
    start = int(stat[stat.rfind(')') + 2:].split()[19])
    return {'host_pid': pid, 'namespace_pid': pids[-1], 'namespace_pids': pids,
            'pid_namespace': os.readlink('/proc/self/ns/pid'), 'start_ticks': start,
            'namespace_pgid': os.getpgrp(), 'namespace_sid': os.getsid(0),
            'observed_monotonic': time.monotonic()}


def namespace_init(root: Path) -> None:
    # PID 1 ignores default-action signals; install an explicit hard-expiry handler.
    signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
    signal.alarm(4)
    init = process_identity()
    if init['namespace_pid'] != 1:
        raise ValueError('namespace init is not PID 1')
    write(root / 'init.json', init)
    child = os.fork()
    if child == 0:
        signal.alarm(4)
        write(root / 'intermediate.json', process_identity())
        os.setsid()
        grandchild = os.fork()
        if grandchild == 0:
            signal.alarm(4)
            write(root / 'escaped.json', process_identity())
            time.sleep(3)
        os._exit(0)
    _, status = os.waitpid(child, 0)
    if status != 0:
        raise ValueError('synthetic first child failed')
    deadline = time.monotonic() + 2
    while not (root / 'escaped.json').is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('synthetic detached grandchild missing')
        time.sleep(0.02)
    write(root / 'ready.json', {'ready': True})
    # Hold both members alive until the outside controller has observed them.
    if (root / 'case').read_text() == 'normal':
        while not (root / 'go').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('normal case release absent')
            time.sleep(0.02)
    else:
        time.sleep(3)



def valid_identity(identity: dict[str, Any]) -> bool:
    pids = identity.get('namespace_pids')
    return (isinstance(pids, list) and len(pids) >= 2
            and all(type(pid) is int and pid > 0 for pid in pids)
            and pids[0] == identity.get('host_pid') and pids[-1] == identity.get('namespace_pid')
            and type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0
            and isinstance(identity.get('pid_namespace'), str) and identity['pid_namespace'].startswith('pid:['))

def gone(identity: dict[str, Any]) -> bool:
    path = Path('/proc') / str(identity['host_pid']) / 'stat'
    try:
        stat = path.read_text()
    except FileNotFoundError:
        return True
    # A reused PID is not the task-owned original. A zombie is not treated as reaped.
    return int(stat[stat.rfind(')') + 2:].split()[19]) != identity['start_ticks']


def one_case(root: Path, case: str, unshare: str) -> dict[str, Any]:
    case_dir = root / case
    case_dir.mkdir()
    (case_dir / 'case').write_text(case)
    argv = [unshare, '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
            sys.executable, '-m', 'experiments.tennis.k5d_pid_teardown_oracle', 'namespace-init',
            '--root', str(case_dir)]
    launcher = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while not (case_dir / 'ready.json').exists():
            if launcher.poll() is not None:
                raise RuntimeError('namespace launch failed: ' + (launcher.stderr.read() if launcher.stderr else ''))
            if time.monotonic() >= deadline:
                raise TimeoutError('namespace readiness deadline exceeded')
            time.sleep(0.02)
        init = json.loads((case_dir / 'init.json').read_text())
        escaped = json.loads((case_dir / 'escaped.json').read_text())
        intermediate = json.loads((case_dir / 'intermediate.json').read_text())
        outside_ns = os.readlink('/proc/self/ns/pid')
        if (not all(valid_identity(item) for item in (init, intermediate, escaped))
                or init['namespace_pid'] != 1 or init['pid_namespace'] == outside_ns
                or escaped['pid_namespace'] != init['pid_namespace'] or escaped['namespace_pid'] == 1
                or escaped['namespace_sid'] != escaped['namespace_pgid']
                or escaped['namespace_sid'] == init['namespace_sid']
                or len({init['host_pid'], escaped['host_pid'], intermediate['host_pid'], launcher.pid}) != 4
                or intermediate['pid_namespace'] != init['pid_namespace'] or not gone(intermediate)
                or gone(init) or gone(escaped)):
            raise ValueError('namespace membership proof failed')
        if case == 'normal':
            write(case_dir / 'go', {'go': True})
        elif case == 'timeout':
            try:
                launcher.wait(timeout=0.1)
                raise ValueError('timeout case exited unexpectedly')
            except subprocess.TimeoutExpired:
                launcher.kill()
        elif case == 'sigterm':
            launcher.terminate()
        elif case == 'sigkill':
            launcher.kill()
        exit_code = launcher.wait(timeout=3)
        end = time.monotonic() + 1
        while not (gone(init) and gone(escaped)):
            if time.monotonic() >= end:
                raise TimeoutError('namespace member remains after launcher exit')
            time.sleep(0.02)
        elapsed_since_escape = time.monotonic() - escaped['observed_monotonic']
        if not 0 < elapsed_since_escape < 1.5:
            raise ValueError('child absence could be due to its own expiry')
        if case == 'normal' and exit_code != 0:
            raise ValueError('normal namespace init failed')
        expected = {'normal': 0, 'timeout': -signal.SIGKILL, 'sigterm': -signal.SIGTERM, 'sigkill': -signal.SIGKILL}
        if exit_code != expected[case]:
            raise ValueError('launcher exit does not match requested signal')
        return {'case': case, 'launcher_host_pid': launcher.pid, 'launcher_exit_code': exit_code,
                'outside_pid_namespace': outside_ns, 'init': init, 'intermediate': intermediate, 'escaped': escaped,
                'members_absent': True, 'signals_sent_only_to_launcher': True,
                'elapsed_since_escape_seconds': elapsed_since_escape}
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=3)
        if launcher.stderr is not None:
            launcher.stderr.close()


def execute(expected_sha: str) -> None:
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Linux CPU-only no-Torch execution required')
    if sha(Path(__file__)) != expected_sha or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('source or single-core affinity differs')
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if dispatch['environment']['gpu'] is not None:
        raise ValueError('GPU must be hidden')
    unshare = subprocess.run(['which', 'unshare'], capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    helptext = subprocess.run([unshare, '--help'], capture_output=True, text=True, check=True, timeout=3).stdout
    if any(flag not in helptext for flag in ('--user', '--map-root-user', '--pid', '--fork', '--kill-child')):
        raise ValueError('util-linux unshare lacks required namespace flags')
    version = subprocess.run([unshare, '--version'], capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    root = scratch / 'pid-cases'
    root.mkdir()
    cases = [one_case(root, case, unshare) for case in CASES]
    output = scratch / 'publish'
    output.mkdir()
    write(output / 'oracle.json', {'schema': 1, 'scope': SCOPE, 'stage': os.environ['PS_STAGE'],
          'source_sha256': expected_sha, 'dispatch_source_sha256': dispatch['code']['source_sha256'],
          'dispatch_snapshot_sha256': dispatch['code']['snapshot_sha256'],
          'environment_sha256': dispatch['staged_environment']['sha256'], 'unshare_path': unshare,
          'unshare_version': version, 'unshare_sha256': sha(Path(unshare)), 'cases': cases})


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    output = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        staging = json.loads((stage / 'staging.json').read_text())
        archive_path = stage / 'published.tar'
        if sha(archive_path) != staging['published']['sha256']:
            raise ValueError('publication SHA differs')
        with tarfile.open(archive_path, 'r:') as tar:
            members = tar.getmembers()
            if (len(members) != 2 or {x.name for x in members} != {'publish', 'publish/oracle.json'}
                    or any(not (x.isfile() or x.isdir()) for x in members) or sum(x.size for x in members) > 100000):
                raise ValueError('unsafe or incomplete publication')
            stream = tar.extractfile('publish/oracle.json')
            if stream is None:
                raise ValueError('proof missing')
            record = json.load(stream)
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        cases = record['cases']
        if (record['schema'] != 1 or record['scope'] != SCOPE or record['stage'] != 'smoke'
                or record['source_sha256'] != sha(Path(__file__))
                or record['dispatch_source_sha256'] != dispatch['code']['source_sha256']
                or record['dispatch_snapshot_sha256'] != dispatch['code']['snapshot_sha256']
                or record['environment_sha256'] != dispatch['staged_environment']['sha256']
                or dispatch['environment']['gpu'] is not None or [x['case'] for x in cases] != list(CASES)
                or record['unshare_sha256'] != sha(Path(record['unshare_path']))
                or not record['unshare_version'].startswith('unshare from util-linux ')):
            raise ValueError('source, scope or case coverage differs')
        for case in cases:
            init, escaped, intermediate = case['init'], case['escaped'], case['intermediate']
            if (not all(valid_identity(item) for item in (init, intermediate, escaped))
                    or init['namespace_pid'] != 1 or init['pid_namespace'] == case['outside_pid_namespace']
                    or init['pid_namespace'] != escaped['pid_namespace'] or escaped['namespace_pid'] <= 1
                    or escaped['namespace_sid'] != escaped['namespace_pgid']
                    or escaped['namespace_sid'] == init['namespace_sid']
                    or len({init['host_pid'], escaped['host_pid'], intermediate['host_pid'], case['launcher_host_pid']}) != 4
                    or intermediate['pid_namespace'] != init['pid_namespace']
                    or case['members_absent'] is not True or case['signals_sent_only_to_launcher'] is not True
                    or not 0 < case['elapsed_since_escape_seconds'] < 1.5
                    or case['launcher_exit_code'] != {'normal': 0, 'timeout': -signal.SIGKILL,
                         'sigterm': -signal.SIGTERM, 'sigkill': -signal.SIGKILL}[case['case']]):
                raise ValueError('namespace or cleanup evidence differs')
        write(output, {'passed': True, 'checks': {'actual_distinct_pid_namespace': True, 'four_exit_modes': True,
              'double_fork_session_escape': True, 'all_own_members_absent': True, 'source_and_device': True}})
    except Exception as exc:
        write(output, {'passed': False, 'checks': {'pid_namespace': False}, 'error': str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'namespace-init', 'validate'))
    parser.add_argument('--phase', choices=('smoke', 'full'))
    parser.add_argument('--source-sha')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.action == 'namespace-init':
        if args.root is None:
            parser.error('namespace root required')
        namespace_init(args.root)
    elif args.action == 'validate':
        validate()
    else:
        if args.phase != os.environ.get('PS_STAGE') or args.source_sha is None:
            parser.error('stage and source SHA required')
        execute(args.source_sha)


if __name__ == '__main__':
    main()
