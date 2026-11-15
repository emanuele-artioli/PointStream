"""Bind the physical confirmation process before importing scientific libraries."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any


def task_count() -> int:
    return sum(1 for _ in Path('/proc/self/task').iterdir())


def process_start_ticks() -> int:
    stat = Path('/proc/self/stat').read_text()
    return int(stat[stat.rfind(')') + 2:].split()[19])


def bind(role: str, stage: Path) -> dict[str, Any]:
    if role not in ('command', 'validator') or os.environ.get('PS_CPU_ALLOWANCE') != '4':
        raise ValueError('four-CPU allowance and valid role required')
    count = task_count()
    if count != 1:
        raise ValueError('CPU affinity must be set before threads start')
    allowed = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
    if len(allowed) < 4:
        raise ValueError('fewer than four allowed CPUs')
    selected = allowed[:4]
    os.sched_setaffinity(0, selected)  # type: ignore[attr-defined,unused-ignore]
    after = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
    if after != selected:
        raise ValueError('four-CPU affinity did not take effect')
    start_ticks = process_start_ticks()
    if start_ticks <= 0:
        raise ValueError('invalid process start ticks')
    result = {'schema': 1, 'role': role, 'claim': 4, 'task_count_before': count,
              'allowed': allowed, 'selected': selected, 'after': after,
              'pid': os.getpid(), 'pgid': os.getpgrp(), 'start_ticks': start_ticks,
              'argv': sys.argv, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (stage / f'{role}-cpu-binding.json').open('x') as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)
        out.write('\n')
    return result


def main() -> None:
    role = 'validator' if '--validate' in sys.argv[1:] else 'command'
    bind(role, Path(os.environ['PS_STAGE_DIR']))
    if '--preflight-only' in sys.argv[1:]:
        return
    runpy.run_module('experiments.tennis.ball_confirmation', run_name='__main__')


if __name__ == '__main__':
    main()
