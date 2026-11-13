"""Opt-in public clocksource grants for the W8 model worker only.

This helper does not alter the default receiver policy or schedule any run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

PROFILE = 'w8-public-clocksource-two-leaf-v1'
ROOT = Path('/sys/devices/system/clocksource')
LEAVES = (ROOT / 'clocksource0/current_clocksource', ROOT / 'clocksource0/available_clocksource')
PATHS = (ROOT, *LEAVES)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_identity(path: Path, *, directory: bool, require_root: bool) -> dict[str, Any]:
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('clocksource path is redirected')
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (not (stat.S_ISDIR(before.st_mode) if directory else stat.S_ISREG(before.st_mode))
                or before.st_mode & stat.S_IWOTH or (require_root and before.st_uid != 0)):
            raise ValueError('clocksource owner, type or permissions differ')
        body = b'' if directory else os.read(fd, 4096)
        if not directory and (not body or os.read(fd, 1)):
            raise ValueError('clocksource leaf is empty or exceeds bound')
        after = os.fstat(fd)
        current = path.stat()
        def key(st: os.stat_result) -> tuple[int, int, int, int]:
            return st.st_dev, st.st_ino, st.st_mode, st.st_uid
        if key(before) != key(after) or key(before) != key(current):
            raise ValueError('clocksource identity changed')
        return {'path': str(path), 'device': before.st_dev, 'inode': before.st_ino,
                'mode': before.st_mode, 'uid': before.st_uid,
                'sha256': None if directory else hashlib.sha256(body).hexdigest()}
    finally:
        os.close(fd)


def write_parent_receipt(path: Path) -> str:
    """Authenticate only the three public sysfs objects, before namespace entry."""
    receipt = {'profile': PROFILE, 'objects': [_public_identity(p, directory=i == 0, require_root=True)
                                                for i, p in enumerate(PATHS)]}
    with path.open('x') as out:
        json.dump(receipt, out, sort_keys=True)
        out.write('\n')
    return _sha(path)


def prepare_child(path: Path, expected_sha: str, denied: list[Path]) -> tuple[list[Path], dict[str, Any]]:
    if _sha(path) != expected_sha:
        raise ValueError('clocksource parent receipt hash differs')
    parent = json.loads(path.read_text())
    objects = parent.get('objects', [])
    if parent.get('profile') != PROFILE or len(objects) != len(PATHS):
        raise ValueError('clocksource parent receipt differs')
    protected = {(p.stat().st_dev, p.stat().st_ino) for p in denied}
    result = []
    for i, (source, expected) in enumerate(zip(PATHS, objects, strict=True)):
        child = _public_identity(source, directory=i == 0, require_root=False)
        if (expected.get('path') != str(source) or expected.get('uid') != 0 or child['uid'] != 65534
                or any(child[k] != expected.get(k) for k in ('path', 'device', 'inode', 'mode', 'sha256'))
                or (child['device'], child['inode']) in protected):
            raise ValueError('clocksource public identity or protected alias differs')
        result.append({'parent': expected, 'child': child,
                       'rights': 'READ_DIR recursive' if i == 0 else 'READ_FILE'})
    return list(PATHS), {'profile': PROFILE, 'parent_receipt_sha256': expected_sha, 'objects': result}


def restrict(policy: Any, paths: list[Path], output: Path) -> tuple[int, int]:
    """Apply the synthetic-oracle rights without changing the common policy."""
    original = policy.allowed_rights
    def rights(path: Path, destination: Path, abi: int) -> int:
        return policy.READ_DIR if path == ROOT else original(path, destination, abi)
    policy.allowed_rights = rights
    try:
        return policy.restrict(paths, output)
    finally:
        policy.allowed_rights = original


def validate_receipt(receipt: Any) -> None:
    if not isinstance(receipt, dict) or receipt.get('profile') != PROFILE:
        raise ValueError('clocksource profile receipt missing')
    sha = receipt.get('parent_receipt_sha256')
    objects = receipt.get('objects')
    if not isinstance(sha, str) or len(sha) != 64 or not isinstance(objects, list) or len(objects) != len(PATHS):
        raise ValueError('clocksource receipt structure differs')
    reconstructed = {'profile': PROFILE, 'objects': [record['parent'] for record in objects]}
    encoded = (json.dumps(reconstructed, sort_keys=True) + '\n').encode()
    if hashlib.sha256(encoded).hexdigest() != sha:
        raise ValueError('clocksource parent receipt binding differs')
    for i, (path, record) in enumerate(zip(PATHS, objects, strict=True)):
        parent, child = record['parent'], record['child']
        if (record['rights'] != ('READ_DIR recursive' if i == 0 else 'READ_FILE')
                or parent.get('path') != str(path) or parent.get('uid') != 0 or child.get('uid') != 65534
                or any(parent.get(k) != child.get(k) for k in ('path', 'device', 'inode', 'mode', 'sha256'))
                or parent.get('mode', 0) & stat.S_IWOTH
                or not (stat.S_ISDIR(parent.get('mode', 0)) if i == 0 else stat.S_ISREG(parent.get('mode', 0)))):
            raise ValueError('clocksource receipt identity differs')
        leaf_sha = parent.get('sha256')
        if (i == 0 and leaf_sha is not None) or (i > 0 and (not isinstance(leaf_sha, str) or len(leaf_sha) != 64)):
            raise ValueError('clocksource receipt hash differs')
