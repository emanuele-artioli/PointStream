"""Post-terminal, CPU-only full-stage GVC audit; never dispatches work.

Run only after root confirms a successful full stage, through the bounded wrapper
shown in the accompanying plan. Output goes to a fresh local audit directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback

JOB = '20261105T121902Z-4020b8ee'
SNAPSHOT = '4dd3d3f1210ccdc3e31e27d53888670e75f1df2e265214c82ec3962998afd36e'
SOURCE = '22aba7ddd69734099194bc756272ee0c1836f96b9de7e27e7aede1d53b8662df'
SPEC = 'd8585a9140e14eecf1de6ea220225f972474da64dca47e7a28d1e6b44c56146f'
HEAD = '9d56c5e181ef1e38dd3977af71453cf6506781cf'


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit('usage: audit_4020b8ee.py JOB_DIRECTORY NEW_LOCAL_OUTPUT_DIRECTORY')
    job, output = map(Path, sys.argv[1:])
    if job.name != JOB or output.exists():
        raise ValueError('incorrect job or existing audit output')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('GPU must be hidden and exactly one CPU must be bound')
    output.mkdir(parents=True)
    checks: dict[str, bool] = {}
    evidence = {'job': JOB, 'checks': checks, 'cuda_visible_devices': '',
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'scope': 'full-stage audit; independent kernel names not profiled',
                'teardown': 'requires separate root check of task-owned processes and claims'}
    try:
        ready = load(job / 'ready.json')
        spec = load(job / 'spec.json')
        gate = load(job / 'gate.json')
        dispatch = load(job / 'full' / 'dispatch.json')
        result = load(job / 'full' / 'gvcrt-peer' / 'result.json')
        checks['archive_sha256'] = digest_file(job / 'source.tar') == SNAPSHOT
        checks['saved_snapshot'] = (ready.get('snapshot_sha256') == SNAPSHOT
                                    and ready.get('git_head') == HEAD
                                    and ready.get('source_sha256') == SOURCE)
        if not checks['archive_sha256'] or not checks['saved_snapshot']:
            raise ValueError('frozen source identity mismatch')
        source = output / 'source'
        source.mkdir()
        with tarfile.open(job / 'source.tar') as archive:
            archive.extractall(source, filter='data')
        sys.path.insert(0, str(source))
        from experiments.jobs.inbox import digest, source_identity
        checks['source_tree'] = source_identity(source) == SOURCE
        checks['normalized_spec'] = digest(spec) == SPEC and ready.get('spec_sha256') == SPEC
        checks['smoke_gate'] = (gate.get('passed') is True and gate.get('spec_sha256') == SPEC
                                and gate.get('source_sha256') == SOURCE
                                and gate.get('validation', {}).get('passed') is True)
        checks['full_dispatch'] = (dispatch.get('code', {}).get('source_sha256') == SOURCE
                                   and dispatch.get('code', {}).get('snapshot_sha256') == SNAPSHOT
                                   and dispatch.get('gpu_sharing', {}).get('mode') == 'exclusive'
                                   and result.get('frames') == 16)
        if not all(checks.values()):
            raise ValueError('source, spec, smoke gate or full dispatch mismatch')
        python = dispatch['staged_environment']['python']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(source),
               'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(output), 'TMP': str(output),
               'TEMP': str(output)}
        command = [python, '-m', 'experiments.generative.gvcrt_peer_job_w8leaves', 'audit',
                   '--stage', str(job / 'full'), '--output', str(output / 'validator.json')]
        with (output / 'validator.log').open('x') as log:
            process = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=85, check=False)
        validation = load(output / 'validator.json') if (output / 'validator.json').exists() else {}
        checks['frozen_full_validator'] = (process.returncode == 0 and validation.get('passed') is True
                                           and bool(validation.get('checks'))
                                           and all(validation['checks'].values()))
        evidence['validator'] = validation
        evidence['returncode'] = process.returncode
    except Exception as exc:
        evidence['error'] = f'{type(exc).__name__}: {exc}'
        evidence['traceback'] = traceback.format_exc()
    evidence['passed'] = bool(checks) and all(checks.values()) and 'error' not in evidence
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
