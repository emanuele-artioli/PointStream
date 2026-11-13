"""Fail-closed, task-private preparation and no-replace publication of one cache entry.

This helper has no fleet dispatch or automatic execution. The caller owns CPU,
space, source identity and authorization; a failed attempt is retained for audit.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Any

from experiments.jobs import staging


def _sha(path: Path) -> str:
    return staging._hash_file(path)


def _directory(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError('canonical existing directory required')
    return path


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_noreplace(source: Path, destination: Path) -> None:
    """Linux renameat2 atomically installs the whole entry, never replacing one."""
    if sys.platform != 'linux':
        raise OSError(errno.ENOTSUP, 'atomic no-replace publication requires Linux')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOSYS, 'renameat2 unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _extract_verified(archive: Path, entry: Path, max_bytes: int) -> dict[str, str]:
    """Extract regular files only, then independently rehash every extracted byte."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive extraction byte limit required')
    with tarfile.open(archive) as scan:
        if sum(member.size for member in scan if member.isfile()) > max_bytes:
            raise ValueError('archive exceeds extraction byte limit')
    tree = entry / 'tree'
    tree.mkdir(mode=0o700)
    expected: dict[str, str] = {}
    modes: dict[str, int] = {}
    seen: set[str] = set()
    with tarfile.open(archive) as bundle:
        for member in bundle:
            raw_name = member.name.rstrip('/')
            parts = PurePosixPath(raw_name).parts
            if raw_name in ('.', '') and member.isdir():
                continue
            if (not raw_name or raw_name.startswith('/') or not parts or '..' in parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError('unsafe, duplicate or non-regular archive member')
            name = '/'.join(parts)
            if name in seen:
                raise ValueError('unsafe, duplicate or non-regular archive member')
            seen.add(name)
            path = tree.joinpath(*parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError('archive member unavailable')
            h = hashlib.sha256()
            with source, path.open('xb') as dst:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            modes[name] = 0o444 | (member.mode & 0o111)
            path.chmod(modes[name])
            expected[name] = h.hexdigest()
    actual = {str(path.relative_to(tree)): _sha(path) for path in tree.rglob('*') if path.is_file()}
    if (actual != expected or any(path.is_symlink() for path in tree.rglob('*'))
            or any(stat.S_IMODE((tree / name).stat().st_mode) != mode for name, mode in modes.items())):
        raise ValueError('extracted tree identity differs')
    for path in sorted((p for p in tree.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        _sync_dir(path)
        path.chmod(0o555)
    _sync_dir(tree)
    tree.chmod(0o555)
    return expected


def publish(source: Path, expected_sha256: str, task_root: Path, cache_root: Path, *, extract: bool = False, max_extract_bytes: int = 0) -> dict[str, Any]:
    """Stage one absent SHA entry; collisions and any unsafe input fail closed."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('invalid SHA256')
    task_root, cache_root = _directory(task_root), _directory(cache_root)
    if task_root == cache_root or task_root.is_relative_to(cache_root) or cache_root.is_relative_to(task_root):
        raise ValueError('task and cache roots must be separate')
    if task_root.stat().st_dev != cache_root.stat().st_dev:
        raise OSError(errno.EXDEV, 'task and cache must share a filesystem')
    if not source.is_absolute() or source.is_symlink() or source.resolve(strict=True) != source:
        raise ValueError('canonical regular source required')
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('source is not a regular file')
    destination = cache_root / expected_sha256
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('cache entry already exists; preserve and inspect')
    private = task_root / expected_sha256
    private.mkdir(mode=0o700, exist_ok=False)
    # The qualified copier is confined to a newly created private directory;
    # its replace/unlink operations can never touch a preexisting cache entry.
    copied = staging._copy_verified(source, private, expected_sha256)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('source changed while copying')
    if copied.is_symlink() or _sha(copied) != expected_sha256:
        raise ValueError('independent copied-byte SHA differs')
    members = _extract_verified(copied, private, max_extract_bytes) if extract else {}
    _sync_dir(private)
    _publish_noreplace(private, destination)
    _sync_dir(cache_root)
    return {'source': str(source), 'sha256': expected_sha256, 'cache_entry': str(destination),
            'bytes': before.st_size, 'tree_file_sha256': members, 'published_noreplace': True}
