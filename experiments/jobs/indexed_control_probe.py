"""Bounded Linux CPU-affinity and fleet-control/workload separation probe.

This is a CPU-only infrastructure check. It does not load models or data.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any

OLD_INBOX_SHA = '26f51a26f97dc5e1c86ff9278eacc13b15468fa44150f55360d51d71526fd96f'
MARKER = b'indexed_cpu_affinity_v1'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe() -> dict[str, Any]:
    from experiments.jobs import inbox

    affinity = sorted(os.sched_getaffinity(0))
    libc = ctypes.CDLL(None, use_errno=True)
    cpu = libc.sched_getcpu()
    source = Path.cwd().resolve()
    imported = Path(inbox.__file__).resolve()
    if (len(affinity) != 1 or cpu not in affinity or imported != source / 'experiments/jobs/inbox.py'
            or digest(imported) != OLD_INBOX_SHA or MARKER in imported.read_bytes()):
        raise ValueError('workload CPU or old scientific inbox source differs')
    return {'pid': os.getpid(), 'affinity': affinity, 'cpu': cpu, 'cwd': str(source),
            'imported_inbox': str(imported), 'imported_inbox_sha256': digest(imported),
            'old_inbox_has_indexed_capability': False}


def context(stage: Path) -> dict[str, Any]:
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    binding = json.loads((stage.parent / 'run/cpu-affinity.json').read_text())
    index = spec['cpu_affinity_index']
    selected = binding['selected']
    if (type(index) is not int or index not in (0, 1) or binding['index'] != index
            or len(selected) != 1 or selected != [binding['available'][index]]
            or dispatch['cpu_affinity'] != selected):
        raise ValueError('supervisor and dispatch CPU binding mismatch')
    control = dispatch['fleet_control']
    control_root = Path(control['root']).resolve(strict=True)
    source = Path.cwd().resolve(strict=True)
    if control_root == source or not control_root.is_absolute():
        raise ValueError('fleet control and workload source are not independent')
    files = control['files']
    if not files or 'experiments/jobs/inbox.py' not in files:
        raise ValueError('released control manifest incomplete')
    for name, sha in files.items():
        path = (control_root / name).resolve(strict=True)
        if not path.is_relative_to(control_root) or digest(path) != sha:
            raise ValueError('released control file identity mismatch')
    if MARKER not in (control_root / 'experiments/jobs/inbox.py').read_bytes():
        raise ValueError('released control lacks indexed-affinity capability')
    return {'index': index, 'binding': binding, 'dispatch_affinity': dispatch['cpu_affinity'],
            'fleet_control': control, 'control_root': str(control_root)}


def child() -> dict[str, Any]:
    result = subprocess.run([sys.executable, '-m', 'experiments.jobs.indexed_control_probe', '--child'],
                            check=True, capture_output=True, text=True, timeout=5)
    return json.loads(result.stdout)


def check(observed: dict[str, Any], other: dict[str, Any], ctx: dict[str, Any]) -> None:
    selected = ctx['binding']['selected']
    if (observed['affinity'] != selected or other['affinity'] != selected
            or observed['cpu'] != selected[0] or other['cpu'] != selected[0]
            or observed['cwd'] != other['cwd'] or observed['imported_inbox'] != other['imported_inbox']
            or observed['imported_inbox_sha256'] != other['imported_inbox_sha256']):
        raise ValueError('workload, child and supervisor do not share one CPU and source')


def run(phase: int) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    expected = {'smoke': 0, 'full': 1}
    if expected.get(os.environ['PS_STAGE']) != phase:
        raise ValueError('probe phase mismatch')
    ctx = context(stage)
    parent, spawned = observe(), child()
    check(parent, spawned, ctx)
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    (output / 'probe.json').write_text(json.dumps({'schema': 1, 'phase': phase, 'context': ctx,
                                                    'workload': parent, 'child': spawned}, indent=2) + '\n')
    # A short bounded overlap window lets the operator compare the two jobs.
    time.sleep(2)


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        ctx = context(stage)
        validator, spawned = observe(), child()
        check(validator, spawned, ctx)
        with tarfile.open(stage / 'published.tar') as archive:
            found = [m for m in archive if m.name == 'publish/probe.json']
            if len(found) != 1 or not found[0].isfile() or found[0].size > 200_000:
                raise ValueError('smoke receipt missing or ambiguous')
            stream = archive.extractfile(found[0])
            if stream is None:
                raise ValueError('smoke receipt unavailable')
            receipt = json.load(stream)
        if receipt.get('schema') != 1 or receipt.get('phase') != 0 or receipt.get('context') != ctx:
            raise ValueError('smoke receipt context differs')
        check(receipt['workload'], receipt['child'], ctx)
        result = {'passed': True, 'checks': {'supervisor_dispatch_workload_child_validator': True,
                  'old_scientific_source_and_released_control': True}, 'context': ctx,
                  'validator': validator, 'validator_child': spawned}
        target.write_text(json.dumps(result, indent=2) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'indexed_probe': False},
                                      'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=int)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        print(json.dumps(observe()))
    elif args.validate:
        validate()
    elif args.phase in (0, 1):
        run(args.phase)
    else:
        parser.error('one probe phase required')


if __name__ == '__main__':
    main()
