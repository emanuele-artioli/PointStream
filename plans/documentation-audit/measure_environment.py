"""Reproduce the host observations; write results under /tmp, never over the audit."""

import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

ROOT = Path('/home/itec/emanuele/pointstream')
PYTHON = '/home/itec/emanuele/.conda/envs/pointstream/bin/python'
OUTPUT = Path('/tmp/pointstream-doc-perf-reproduction.json')
ENVIRONMENT = dict(
    os.environ,
    PYTHONNOUSERSITE='1',
    PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(ROOT),
)


def main():
    """Time bounded read-only commands and startup controls on the existing env."""
    result = {
        'host': socket.gethostname(),
        'checkout': str(ROOT),
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
        'python': PYTHON,
        'conditions': 'shared host; existing caches; no login shell; no cold-cache claim',
        'environment': {
            name: ENVIRONMENT[name]
            for name in ('PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH')
        },
        'predeclared_bounds_seconds': {
            'git_and_discovery': [0, 5],
            'torch_import_process': [0, 300],
        },
        'measurements': {},
    }
    commands = [
        ('git_status', ['git', 'status', '--short'], 5, 10),
        ('tracked_files', ['git', 'ls-files'], 5, 10),
        ('rg_source_files', ['rg', '--files', 'src', 'tests', 'scripts', 'experiments'], 5, 10),
        ('system_python_noop', ['/usr/bin/python3', '-c', 'pass'], 5, 10),
        ('env_python_noop', [PYTHON, '-c', 'pass'], 5, 10),
        ('sqlite_process', [PYTHON, '-c', 'import sqlite3'], 3, 20),
        (
            'torch_process',
            [PYTHON, '-c', 'import sqlite3; import torch; print(torch.__version__)'],
            3,
            90,
        ),
        (
            'runner_process',
            [
                PYTHON, '-c',
                'import sqlite3; import sys; import src.runner; '
                'print("torch_loaded", "torch" in sys.modules)',
            ],
            3,
            60,
        ),
    ]
    for name, command, count, timeout in commands:
        samples, statuses, outputs = [], [], []
        for _ in range(count):
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    command, cwd=ROOT, env=ENVIRONMENT, text=True,
                    capture_output=True, timeout=timeout, check=False,
                )
                statuses.append(process.returncode)
                outputs.append(
                    process.stdout.strip()
                    if name in ('torch_process', 'runner_process')
                    else process.stderr[-400:]
                )
            except subprocess.TimeoutExpired:
                statuses.append('timeout')
                outputs.append(f'censored at {timeout} seconds; not a completed duration')
            samples.append(time.perf_counter() - started)
        measurement = {
            'seconds': samples, 'status': statuses, 'output': outputs,
            'mean': statistics.mean(samples),
            'se': statistics.stdev(samples) / count ** 0.5,
            'n': count, 'timeout': timeout,
            'all_completed_successfully': all(status == 0 for status in statuses),
        }
        result['measurements'][name] = measurement
        OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
        print(name, measurement, flush=True)

    anchors = []
    for _ in range(3):
        started = time.perf_counter()
        time.sleep(0.1)
        anchors.append(time.perf_counter() - started)
    result['clock_anchor'] = {
        'requested_sleep_seconds': 0.1, 'observed': anchors,
        'criterion': 'all >=0.1 and <=0.5 seconds',
        'passed': all(0.1 <= value <= 0.5 for value in anchors),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Raw output: {OUTPUT}', flush=True)
    return 0 if all(
        item['all_completed_successfully'] for item in result['measurements'].values()
    ) and result['clock_anchor']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
