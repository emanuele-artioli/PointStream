"""Witness-only protected-file stat identities; never read protected contents."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from experiments.tennis.w8_receiver_preflight import PROTECTED

SCOPE = 'witness protected metadata only'


def _identity(path: Path) -> dict[str, int]:
    if str(path.resolve(strict=True)) != str(path):
        raise ValueError('protected path is not canonical')
    first = path.stat()
    second = path.stat()
    if not stat.S_ISREG(first.st_mode):
        raise ValueError('protected identity must be a regular file')
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    values = tuple(getattr(first, key) for key in fields)
    if values != tuple(getattr(second, key) for key in fields):
        raise ValueError('protected metadata changed during observation')
    return dict(zip(('device', 'inode', 'size', 'mtime_ns', 'ctime_ns'), values))


def validate(receipt: dict[str, Any], historical: dict[str, Any] | None = None, *,
             check_current: bool = False) -> None:
    if (set(receipt) != {'schema', 'scope', 'content_verified_now', 'historical_receipt_sha256', 'protected_inputs'}
            or receipt['schema'] != 1 or receipt['scope'] != SCOPE or receipt['content_verified_now'] is not False
            or not isinstance(receipt['historical_receipt_sha256'], str)
            or len(receipt['historical_receipt_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in receipt['historical_receipt_sha256'])
            or not isinstance(receipt['protected_inputs'], dict) or set(receipt['protected_inputs']) != set(PROTECTED)):
        raise ValueError('witness metadata receipt schema differs')
    old = None if historical is None else historical.get('protected_inputs')
    if historical is not None and (not isinstance(old, dict) or set(old) != set(PROTECTED)):
        raise ValueError('historical protected categories differ')
    for key, (name, digest) in PROTECTED.items():
        row = receipt['protected_inputs'][key]
        fields = {'path', 'reference_sha256', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns'}
        if (not isinstance(row, dict) or set(row) != fields or row['path'] != name or row['reference_sha256'] != digest
                or any(type(row[field]) is not int or row[field] < 0 for field in fields - {'path', 'reference_sha256'})
                or row['inode'] <= 0):
            raise ValueError('witness protected metadata differs: ' + key)
        if old is not None:
            prior = old[key]
            if (not isinstance(prior, dict) or set(prior) != {'path', 'sha256', 'device', 'inode'}
                    or prior['path'] != name or prior['sha256'] != digest or prior['inode'] != row['inode']
                    or type(prior['device']) is not int):
                raise ValueError('historical protected identity differs: ' + key)
        if check_current and _identity(Path(name)) != {k: row[k] for k in ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')}:
            raise ValueError('protected metadata changed: ' + key)



def reject_protected_alias(path: Path) -> None:
    """Reject protected inode aliases using metadata only, before any file read."""
    info = path.stat()
    for name, _digest in PROTECTED.values():
        protected = Path(name).stat()
        if (info.st_dev, info.st_ino) == (protected.st_dev, protected.st_ino):
            raise ValueError('input aliases protected content')

def read_historical(historical_path: Path, expected_historical_sha256: str) -> dict[str, Any]:
    """Read a pinned metadata file only after excluding all protected inode aliases."""
    if (not historical_path.is_absolute() or historical_path.is_symlink()
            or historical_path.resolve(strict=True) != historical_path
            or not isinstance(expected_historical_sha256, str) or len(expected_historical_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_historical_sha256)):
        raise ValueError('canonical pinned historical metadata required')
    info = historical_path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2_000_000:
        raise ValueError('historical metadata file size or type differs')
    try:
        reject_protected_alias(historical_path)
    except ValueError as exc:
        raise ValueError('historical metadata aliases protected content') from exc
    fd = os.open(historical_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
            raise ValueError('historical metadata changed before opening')
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('historical metadata truncated')
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
            raise ValueError('historical metadata changed while reading')
        raw = b''.join(chunks)
    finally:
        os.close(fd)
    historical_sha = hashlib.sha256(raw).hexdigest()
    if historical_sha != expected_historical_sha256:
        raise ValueError('historical metadata SHA differs')
    historical = json.loads(raw)
    if not isinstance(historical, dict):
        raise ValueError('historical metadata must be an object')
    return historical


def capture(historical_path: Path, output: Path, *, expected_historical_sha256: str) -> dict[str, str]:
    """Read separately pinned metadata and only stat protected files."""
    historical = read_historical(historical_path, expected_historical_sha256)
    historical_sha = expected_historical_sha256
    receipt = {'schema': 1, 'scope': SCOPE, 'content_verified_now': False,
               'historical_receipt_sha256': historical_sha,
               'protected_inputs': {key: {'path': name, 'reference_sha256': digest, **_identity(Path(name))}
                                    for key, (name, digest) in PROTECTED.items()}}
    validate(receipt, historical, check_current=True)
    with output.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'historical_sha256': historical_sha}
