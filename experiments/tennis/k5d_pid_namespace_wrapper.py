"""Synthetic-only opt-in PID namespace launcher; no production admission.

The outer process replaces itself with unshare, retaining the supervisor PID.
The inner namespace PID 1 replaces itself with the fixed synthetic workload.
Actual receiver commands and GPU execution are deliberately not admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

TARGET = 'experiments.tennis.k5d_pid_escalation_oracle'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_root(value: str) -> Path:
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    root = Path(value).resolve(strict=True)
    if root == scratch or not root.is_relative_to(scratch) or not root.is_dir():
        raise ValueError('case root must be an existing directory inside fleet scratch')
    return root


def outer(root: str, expected_unshare_sha: str) -> None:
    case_root = checked_root(root)
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('synthetic Linux CPU-only namespace launcher required')
    path = shutil.which('unshare')
    if not path or sha(Path(path)) != expected_unshare_sha:
        raise ValueError('unshare executable identity differs')
    outside_ns = os.readlink('/proc/self/ns/pid')
    argv = [path, '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL',
            sys.executable, '-m', 'experiments.tennis.k5d_pid_namespace_wrapper', 'inner',
            '--root', str(case_root), '--outside-ns', outside_ns]
    os.execv(path, argv)


def inner(root: str, outside_ns: str) -> None:
    case_root = checked_root(root)
    lines = [line.split()[1:] for line in Path('/proc/self/status').read_text().splitlines()
             if line.startswith('NSpid:')]
    if (os.getpid() != 1 or len(lines) != 1 or len(lines[0]) < 2 or lines[0][-1] != '1'
            or os.readlink('/proc/self/ns/pid') == outside_ns):
        raise ValueError('distinct namespace init PID 1 required')
    argv = [sys.executable, '-m', TARGET, 'namespace-init', '--root', str(case_root)]
    os.execv(sys.executable, argv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('outer', 'inner'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--unshare-sha')
    parser.add_argument('--outside-ns')
    args = parser.parse_args()
    if args.mode == 'outer':
        if args.unshare_sha is None:
            parser.error('outer requires unshare SHA')
        outer(args.root, args.unshare_sha)
    else:
        if args.outside_ns is None:
            parser.error('inner requires outside namespace')
        inner(args.root, args.outside_ns)


if __name__ == '__main__':
    main()
