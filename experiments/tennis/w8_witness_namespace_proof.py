"""CPU-only synthetic lifecycle proof for the real W8 namespace init.

This is infrastructure evidence only; it does not run a model or prove quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from experiments.tennis import w8_witness_namespace as namespace


def proc(pid: int) -> dict[str, int]:
    raw = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'start_ticks': int(fields[19])}


def own_identity() -> dict[str, int]:
    raw = Path('/proc/self/stat').read_text()
    return proc(int(raw[:raw.find(' ')]))


def write(path: Path, value: dict) -> None:
    # Publish only complete, durable JSON; link refuses to replace a receipt.
    pending = path.with_name(path.name + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(pending, path)
    pending.unlink()


def absent(item: dict[str, int]) -> bool:
    try:
        return proc(item['pid'])['start_ticks'] != item['start_ticks']
    except FileNotFoundError:
        return True


def worker(case: str, output: Path) -> None:
    # The fixed detached synthetic child tests PID-namespace teardown, not codecs.
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    output = output.resolve(strict=True)
    fields = [line.split()[1:] for line in Path('/proc/self/status').read_text().splitlines()
              if line.startswith('NSpid:')]
    if (sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or len(os.sched_getaffinity(0)) != 1 or os.getpid() != 2
            or len(fields) != 1 or len(fields[0]) < 2 or fields[0][-1] != '2'
            or case not in ('normal', 'timeout', 'term', 'kill') or output == scratch
            or not output.is_relative_to(scratch) or not output.is_dir()):
        raise ValueError('CPU proof worker requires nested PID namespace and owned scratch')
    child = os.fork()
    if child == 0:
        os.setsid()
        write(output / 'detached.json', own_identity())
        time.sleep(2 if case == 'normal' else 60)
        os._exit(0)
    write(output / 'worker.json', own_identity())
    if case == 'normal':
        os.waitpid(child, 0)
        return
    time.sleep(60)


def check_waited_cpu(outer: float, init: float, children: float) -> None:
    """Check independent waited accounting for this small synthetic workload."""
    import math

    if (any(not math.isfinite(x) or x < 0 for x in (outer, init, children))
            or abs(outer - init - children) > 0.25):
        raise ValueError('independent synthetic waited CPU accounting differs')


def group_absent(pgid: int) -> bool:
    """Conservatively reject any remaining or inaccessible detached group."""
    if type(pgid) is not int or pgid <= 0:
        raise ValueError('invalid detached process group')
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def wait_after_term(launcher: subprocess.Popen) -> tuple[int, bool]:
    """Give TERM a bounded grace period, then explicitly escalate if needed."""
    try:
        return int(launcher.wait(timeout=3)), False
    except subprocess.TimeoutExpired:
        launcher.kill()
        return int(launcher.wait(timeout=5)), True


def run(output: Path, unshare: Path, unshare_sha: str) -> None:
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    output = output.resolve(strict=True)
    if (sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or len(os.sched_getaffinity(0)) != 1 or output == scratch or not output.is_relative_to(scratch)
            or any(output.iterdir()) or not unshare.is_absolute()
            or hashlib.sha256(unshare.read_bytes()).hexdigest() != unshare_sha):
        raise ValueError('invalid fixed CPU proof inputs')
    campaign = json.loads((Path(os.environ['PS_STAGE_DIR']).parent / 'run' / 'status.json').read_text())['process_group_id']
    if campaign != os.getpgrp():
        raise ValueError('proof is outside campaign process group')
    outside = os.readlink('/proc/self/ns/pid')
    results = []
    for case in ('normal', 'timeout', 'term', 'kill'):
        case_dir = output / case
        case_dir.mkdir()
        receipt = case_dir / 'receipt.json'
        timeout = 5 if case == 'timeout' else 20
        argv = [str(unshare), '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
                sys.executable, '-m', 'experiments.tennis.w8_witness_namespace', '--outside', outside,
                '--timeout', str(timeout), '--receipt', str(receipt), '--', namespace.PROOF_TARGET,
                'worker', case, '--output', str(case_dir)]
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal
        parent = os.getpid()
        launcher = subprocess.Popen(argv, preexec_fn=lambda: parent_death_signal(parent))
        try:
            deadline = time.monotonic() + 4
            started_record = receipt.with_name(receipt.name + '.started')
            while (not (case_dir / 'worker.json').exists() or not (case_dir / 'detached.json').exists()
                   or not started_record.exists()):
                if launcher.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('proof worker did not become observable')
                time.sleep(.01)
            a = json.loads((case_dir / 'worker.json').read_text())
            b = json.loads((case_dir / 'detached.json').read_text())
            started_data = json.loads(started_record.read_text())
            init = proc(a['ppid'])
            if (proc(a['pid']) != a or proc(b['pid']) != b or a['pgid'] != campaign
                    or init['pgid'] != campaign or b['ppid'] != a['pid']
                    or b['pgid'] != b['pid'] or a['pid'] == b['pid']
                    or started_data['worker_host_identity']['host_pid'] != a['pid']
                    or started_data['worker_host_identity']['start_ticks'] != a['start_ticks']
                    or started_data['namespace_init']['host_pid'] != init['pid']
                    or started_data['namespace_init']['start_ticks'] != init['start_ticks']):
                raise ValueError('live host PID/start/group join differs')
            if case in ('term', 'kill'):
                launcher.send_signal(signal.SIGTERM if case == 'term' else signal.SIGKILL)
            escalated = False
            try:
                if case == 'term':
                    rc, escalated = wait_after_term(launcher)
                else:
                    rc = launcher.wait(timeout=12)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=5)
                raise RuntimeError('proof launcher exceeded bound')
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            waited_cpu = (usage_after.ru_utime + usage_after.ru_stime
                          - usage_before.ru_utime - usage_before.ru_stime)
            end = time.monotonic() + 3
            while not (absent(a) and absent(b) and absent(init) and group_absent(b['pgid'])) and time.monotonic() < end:
                time.sleep(.02)
            if not (absent(a) and absent(b) and absent(init) and group_absent(b['pgid'])):
                raise RuntimeError('owned proof descendants survived')
            inner = json.loads(receipt.read_text()) if receipt.exists() else None
            if case == 'normal' and (rc != 0 or inner is None or inner['reason'] != 'complete'
                                     or inner['status'] != 0 or inner['worker_host_identity']['host_pid'] != a['pid']
                                     or inner['worker_host_identity']['start_ticks'] != a['start_ticks']
                                     or inner['namespace_init']['host_pid'] != init['pid']
                                     or inner['children_cpu_seconds'] < 0 or inner['init_cpu_seconds'] < 0):
                raise ValueError('normal waited proof differs')
            if case == 'normal':
                assert inner is not None
                check_waited_cpu(waited_cpu, inner['init_cpu_seconds'], inner['children_cpu_seconds'])
            if case == 'timeout' and (rc == 0 or inner is None or inner['reason'] != 'timeout'):
                raise ValueError('timeout proof differs')
            if ((case == 'term' and rc != -(signal.SIGKILL if escalated else signal.SIGTERM))
                    or (case == 'kill' and rc != -signal.SIGKILL)):
                raise ValueError('signal proof differs')
            results.append({'case': case, 'launcher_rc': rc, 'outer_waited_cpu_seconds': waited_cpu,
                            'init': init, 'worker': a, 'detached': b,
                            'all_absent': True, 'detached_group_absent': True,
                            'term_requested': case == 'term', 'term_escalated': escalated, 'receipt': inner})
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(timeout=5)
    write(output / 'proof.json', {'scope': 'synthetic CPU namespace lifecycle only', 'outside': outside,
                                   'campaign_pgid': campaign, 'cases': results})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    proof = sub.add_parser('run')
    proof.add_argument('--output', type=Path, required=True)
    proof.add_argument('--unshare', type=Path, required=True)
    proof.add_argument('--unshare-sha256', required=True)
    leaf = sub.add_parser('worker')
    leaf.add_argument('case', choices=('normal', 'timeout', 'term', 'kill'))
    leaf.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.output, args.unshare, args.unshare_sha256)
    else:
        worker(args.case, args.output)


if __name__ == '__main__':
    main()
