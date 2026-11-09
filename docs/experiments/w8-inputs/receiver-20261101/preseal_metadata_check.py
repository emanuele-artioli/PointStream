#!/usr/bin/env python3
"""Review-only fixed-input stat comparison. No input contents are read.

The small baseline JSON is a separately reviewed trusted control artifact. This
is not an adversarial filesystem security boundary and does not submit a job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

PROTECTED = {
    'source': '/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__000.mp4',
    'prepared': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T162058Z-c9b2c47b/full/published.tar',
    'fit': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T150848Z-a582ecc0/full/published.tar',
    'golden': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T163512Z-1d0712ad/full/published.tar',
    'masks': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T173108Z-0782499a/full/published.tar',
}


def stamp(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('noncanonical or symlinked path')
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('not a regular file')
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def read_baseline(path, digest, forbidden):
    path = Path(path)
    before = stamp(path)
    if not 0 < before['size'] <= 2_000_000 or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('invalid reviewed baseline')
    if (before['device'], before['inode']) in forbidden:
        raise ValueError('baseline aliases protected content')
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        opened = {'device': current.st_dev, 'inode': current.st_ino, 'size': current.st_size,
                  'mtime_ns': current.st_mtime_ns, 'ctime_ns': current.st_ctime_ns}
        if opened != before:
            raise ValueError('baseline changed before open')
        chunks, remaining = [], before['size']
        while remaining:
            chunk = os.read(fd, min(remaining, 1048576))
            if not chunk:
                raise ValueError('baseline truncated')
            chunks.append(chunk)
            remaining -= len(chunk)
        current = os.fstat(fd)
        after = {'device': current.st_dev, 'inode': current.st_ino, 'size': current.st_size,
                 'mtime_ns': current.st_mtime_ns, 'ctime_ns': current.st_ctime_ns}
        if after != before:
            raise ValueError('baseline changed while reading')
        data = b''.join(chunks)
    finally:
        os.close(fd)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('baseline SHA differs')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('baseline must be an object')
    return value


def check(baseline_path, baseline_sha, output):
    before = time.time()
    protected = {key: stamp(path) for key, path in PROTECTED.items()}
    forbidden = {(row['device'], row['inode']) for row in protected.values()}
    if len(forbidden) != len(PROTECTED):
        raise ValueError('protected categories share an inode')
    baseline = read_baseline(baseline_path, baseline_sha, forbidden)
    if (baseline.get('schema') != 1 or baseline.get('scope') != 'root-reviewed W8 preseal metadata baseline'
            or baseline.get('protected') != protected or not isinstance(baseline.get('paths'), list)
            or not baseline['paths']):
        raise ValueError('protected metadata or reviewed baseline differs')
    seen, rows = set(), []
    for row in baseline['paths']:
        if (not isinstance(row, dict) or set(row) != {'role', 'path', 'expected_sha256', 'stamp'}
                or not isinstance(row['role'], str) or not row['role'] or not isinstance(row['path'], str)
                or not isinstance(row['expected_sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', row['expected_sha256'])):
            raise ValueError('invalid fixed input row')
        if row['role'] in seen:
            raise ValueError('duplicate role')
        seen.add(row['role'])
        observed = stamp(row['path'])
        if (observed['device'], observed['inode']) in forbidden:
            raise ValueError('declared input aliases protected content')
        if observed != row['stamp']:
            raise ValueError('declared input metadata changed')
        rows.append({'role': row['role'], 'path': row['path'], 'expected_sha256': row['expected_sha256'],
                     'stamp': observed, 'content_verified_now': False})
    # A second metadata-only pass narrows ordinary cooperative-update races.
    if protected != {key: stamp(path) for key, path in PROTECTED.items()}:
        raise ValueError('protected metadata changed during review')
    if any(stamp(row['path']) != row['stamp'] for row in rows):
        raise ValueError('input metadata changed during review')
    result = {'schema': 1, 'scope': 'preseal metadata-only admission; no content verification',
              'baseline_sha256': baseline_sha, 'protected': protected, 'paths': rows,
              'started_epoch': before, 'finished_epoch': time.time(),
              'limitation': 'Trusted cooperative filesystem; checks do not prevent substitution after review.'}
    with Path(output).open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check(args.baseline, args.baseline_sha256, args.output)


if __name__ == '__main__':
    main()
