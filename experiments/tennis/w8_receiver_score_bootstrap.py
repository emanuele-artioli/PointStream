"""Apply the scoring CPU budget before importing NumPy, Torch or project code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('source', 'score'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--cpu-seconds', type=int, required=True)
    parser.add_argument('--child-cpu-seconds', type=int, default=0)
    args = parser.parse_args()
    if (not 1 <= args.cpu_seconds <= 3600 or not 0 <= args.child_cpu_seconds <= 3600
            or (args.phase == 'source') != bool(args.child_cpu_seconds)):
        raise ValueError('invalid CPU bounds')
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, max(args.cpu_seconds, args.child_cpu_seconds)))
    if args.config.stat().st_size > 2_000_000:
        raise ValueError('score config too large')
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        raise ValueError('score config hash changed')
    config = json.loads(data)
    if (config['cpu_seconds'] != args.cpu_seconds
            or (args.phase == 'source' and config['child_cpu_seconds'] != args.child_cpu_seconds)):
        raise ValueError('score CPU bounds disagree with config')
    command = [sys.executable, '-m', 'experiments.tennis.w8_receiver_score', '--phase', args.phase,
               '--config', str(args.config), '--config-sha256', args.config_sha256]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
