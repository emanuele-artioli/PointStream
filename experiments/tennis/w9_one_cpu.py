"""Run the W1 encoder or its validator on one inherited Linux CPU.

This entrypoint confines itself before importing workload modules, then checks a
real subprocess inherits the same affinity. Existing W1 codec/search/scoring
settings and checks are unchanged apart from its explicit CPU-profile check.
"""
from __future__ import annotations

import os
import subprocess
import sys


def confine() -> int:
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError('no allowed CPU')
    cpu = allowed[0]
    os.sched_setaffinity(0, {cpu})
    if os.sched_getaffinity(0) != {cpu}:
        raise RuntimeError('parent CPU affinity not enforced')
    observed = subprocess.run([sys.executable, '-c', 'import os;print(sorted(os.sched_getaffinity(0)))'],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if observed != repr([cpu]):
        raise RuntimeError('child CPU affinity not inherited')
    return cpu


def main() -> None:
    cpu = confine()
    print(f'W9 single CPU profile: CPU {cpu}; child inheritance verified', flush=True)
    from experiments.tennis import w1_encode

    if sys.argv[1:] == ['--validate']:
        w1_encode.validate(single_cpu=True)
    else:
        w1_encode.main()


if __name__ == '__main__':
    main()
