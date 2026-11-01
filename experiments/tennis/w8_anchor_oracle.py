"""Bounded local reinspection of the exact W8 accepted anchor records."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox', type=Path, required=True)
    p.add_argument('--repairs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    start = time.monotonic()
    from experiments.tennis.w8_oracle import accepted_nad000, strict_win_intervals
    from experiments.tennis.w8_payload import sha

    repairs_bytes = a.repairs.read_bytes()
    repair = json.loads(repairs_bytes)['nad000']
    hybrid, anchors = accepted_nad000(a.inbox, repair)
    intervals = strict_win_intervals(hybrid, anchors)
    a.out.mkdir(parents=False, exist_ok=False)
    record = {'scope': 'accepted historical raw anchors; no candidate payload rate evaluated',
              'repairs_config_sha256': sha(repairs_bytes), 'repair': repair, 'hybrid': hybrid, 'anchors': anchors,
              'strict_wins': intervals, 'bounds_kbps': {k: [min(p['kbps'] for p in v), max(p['kbps'] for p in v)] for k, v in anchors.items()},
              'cpu_limit_s': 60, 'wall_limit_s': 120}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record['cpu_seconds'] = usage.ru_utime + usage.ru_stime
    record['wall_seconds'] = time.monotonic() - start
    (a.out / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'strict_wins': intervals, 'bounds_kbps': record['bounds_kbps'],
                      'cpu_seconds': record['cpu_seconds'], 'wall_seconds': record['wall_seconds']}, sort_keys=True))


if __name__ == '__main__':
    main()
