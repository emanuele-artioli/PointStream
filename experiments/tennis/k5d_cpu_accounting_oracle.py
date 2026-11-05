"""Synthetic Linux fleet oracle for the K5d subreaper, without models or GPUs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import resource
import sys
import tarfile
import time
from pathlib import Path

from experiments.tennis import k5d_cpu_accounting as accounting


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def work(root: Path, name: str) -> None:
    started = time.process_time()
    x = 1
    while time.process_time() - started < 0.04:
        x = (x * 17 + 1) % 1000003
    usage = resource.getrusage(resource.RUSAGE_SELF)
    with (root / (name + '.json')).open('x') as f:
        json.dump({'role': name, 'pid': os.getpid(), 'ppid': os.getppid(),
                   'self_cpu_seconds': usage.ru_utime + usage.ru_stime, 'value': x}, f)


def worker(root: Path) -> None:
    work(root, 'direct')
    child = os.fork()
    if child == 0:
        work(root, 'forked')
        os._exit(0)
    middle = os.fork()
    if middle == 0:
        grandchild = os.fork()
        if grandchild == 0:
            time.sleep(0.2)
            work(root, 'double_forked')
        os._exit(0)
    for pid in (child, middle):
        _, status = os.waitpid(pid, 0)
        if status != 0:
            raise RuntimeError('synthetic child failed')


def execute() -> None:
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Linux CPU-only no-Torch execution required')
    stage_dir = Path(os.environ['PS_STAGE_DIR'])
    stage = os.environ['PS_STAGE']
    if stage not in ('smoke', 'full') or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single-core synthetic stage required')
    dispatch = json.loads((stage_dir / 'dispatch.json').read_text())
    identity = {'source_sha256': dispatch['code']['source_sha256'],
                'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                'environment_sha256': dispatch['staged_environment']['sha256']}
    if dispatch['environment']['gpu'] is not None:
        raise ValueError('GPU must be hidden')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    root = scratch / 'synthetic-workers'
    root.mkdir()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        record = accounting.measure([sys.executable, '-m', 'experiments.tennis.k5d_cpu_accounting_oracle', 'worker', '--root', str(root)],
                                    phase='prepare', stage=stage, clip='synthetic', frames=1,
                                    identity=identity, timeout_seconds=12)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    independent = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    observed = accounting.read_receipt(stdout.getvalue(), phase='prepare', stage=stage, clip='synthetic',
                                       frames=1, identity=identity)
    rows = [json.loads((root / (name + '.json')).read_text()) for name in ('direct', 'forked', 'double_forked')]
    if (len({r['pid'] for r in rows}) != 3 or rows[0]['pid'] != record['command_pid']
            or record['wrapper_pid'] != os.getpid() or rows[2]['ppid'] != os.getpid()
            or record['adopted_children_waited'] < 1
            or abs(record['children_cpu_seconds'] - independent) > 1e-6
            or independent < sum(r['self_cpu_seconds'] for r in rows) - 0.01):
        raise ValueError('independent process and CPU measurements differ')
    for row in rows:
        try:
            os.kill(row['pid'], 0)
        except ProcessLookupError:
            continue
        raise ValueError('synthetic child remains alive')
    publication = scratch / 'publish'
    publication.mkdir()
    result = {'schema': 1, 'scope': 'synthetic Linux accounting only; no production or teardown qualification',
              'stage': stage, 'identity': identity, 'source_files': {'oracle': sha(Path(__file__)),
              'accounting': sha(Path(accounting.__file__))}, 'children': rows,
              'accounting': record, 'independent_children_cpu_seconds': independent,
              'reported_total_cpu_seconds': observed, 'all_children_absent': True}
    (publication / 'oracle.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    out = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        publication = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if sha(publication) != staging['published']['sha256']:
            raise ValueError('published archive identity differs')
        with tarfile.open(publication, 'r:') as archive:
            members = archive.getmembers()
            if (len(members) != 2 or len({m.name for m in members}) != 2
                    or {m.name for m in members} != {'publish', 'publish/oracle.json'}
                    or any(not (m.isfile() or m.isdir()) for m in members)
                    or sum(m.size for m in members) > 100_000):
                raise ValueError('unexpected oracle publication')
            stream = archive.extractfile('publish/oracle.json')
            if stream is None:
                raise ValueError('missing oracle result')
            result = json.load(stream)
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        code = dispatch['code']
        rows = result['children']
        rec = result['accounting']
        verified_cpu = accounting.read_receipt('K5D_CPU_ACCOUNTING ' + json.dumps(rec),
                                               phase='prepare', stage='smoke', clip='synthetic',
                                               frames=1, identity=result['identity'])
        if (result.get('scope') != 'synthetic Linux accounting only; no production or teardown qualification'
                or len(rows) != 3 or result['reported_total_cpu_seconds'] != verified_cpu
                or result['stage'] != 'smoke' or result['identity']['source_sha256'] != code['source_sha256']
                or result['identity']['snapshot_sha256'] != code['snapshot_sha256']
                or result['source_files'] != {'oracle': sha(Path(__file__)), 'accounting': sha(Path(accounting.__file__))}
                or [r['role'] for r in rows] != ['direct', 'forked', 'double_forked']
                or len({r['pid'] for r in rows}) != 3 or rec['command_pid'] != rows[0]['pid']
                or rows[2]['ppid'] != rec['wrapper_pid'] or dispatch['environment']['gpu'] is not None
                or result['identity']['environment_sha256'] != dispatch['staged_environment']['sha256']
                or rec['complete_descendant_wait'] is not True or rec['child_exit_code'] != 0
                or rec['adopted_children_waited'] < 1 or result['all_children_absent'] is not True
                or rec['children_cpu_seconds'] < sum(r['self_cpu_seconds'] for r in rows) - 0.01
                or abs(rec['children_cpu_seconds'] - result['independent_children_cpu_seconds']) > 1e-6):
            raise ValueError('synthetic accounting proof failed')
        out.write_text(json.dumps({'passed': True, 'checks': {'source_bound': True, 'three_processes': True,
                       'double_fork_adopted': True, 'rusage_independently_checked': True,
                       'children_absent': True, 'no_model_or_gpu': True}}) + '\n')
    except Exception as exc:
        out.write_text(json.dumps({'passed': False, 'checks': {'synthetic_accounting': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'worker', 'validate'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--phase', choices=('smoke', 'full'))
    parser.add_argument('--oracle-sha')
    parser.add_argument('--accounting-sha')
    args = parser.parse_args()
    if args.action == 'worker':
        if args.root is None:
            parser.error('worker root required')
        worker(args.root)
    elif args.action == 'validate':
        validate()
    else:
        if (args.phase != os.environ.get('PS_STAGE') or args.oracle_sha != sha(Path(__file__))
                or args.accounting_sha != sha(Path(accounting.__file__))):
            parser.error('phase differs from fleet stage')
        execute()


if __name__ == '__main__':
    main()
