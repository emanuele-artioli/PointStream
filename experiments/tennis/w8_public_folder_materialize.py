"""Versioned regular-file folders for the strict W8 receiver; no model logic."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any

PROFILE = 'public-model-folders-copy-v1'
# The only four files consumed by the unchanged VAE and CLIP folder helper.
SIZES = {'vae/config.json': 547, 'vae/diffusion_pytorch_model.safetensors': 334643276,
         'image_encoder/config.json': 703, 'image_encoder/pytorch_model.bin': 1215993967}
PINNED = {'vae/config.json': '92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e',
          'vae/diffusion_pytorch_model.safetensors': 'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815',
          'image_encoder/config.json': 'ba41139cc9dad159163a53e2aa847356b627f53fef988ad461659bd2d587d886',
          'image_encoder/pytorch_model.bin': '89d2aa29b5fdf64f3ad4f45fb4227ea98bc45156bbae673b85be1af7783dbabb'}


def _stamp(st: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(st.st_mode):
        raise ValueError('materialization source must be regular')
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def copy_declared(declared: dict[str, dict[str, Any]], folders: Path) -> dict[str, Any]:
    """Copy an exact two-folder file set; generic byte sizes enable synthetic oracles."""
    if set(declared) != set(SIZES) or folders.exists():
        raise ValueError('exact four declared files and fresh folders required')
    folders.mkdir(parents=True)
    for name in ('vae', 'image_encoder'):
        (folders / name).mkdir()
    rows = {}
    wall_start, cpu_start = time.monotonic(), time.process_time()
    for relative in sorted(declared):
        item = declared[relative]
        if set(item) != {'path', 'sha256', 'size'} or type(item['size']) is not int or item['size'] < 0:
            raise ValueError('invalid materialization input record')
        source = Path(item['path'])
        if not source.is_absolute() or str(source.resolve(strict=True)) != str(source) or source.is_symlink():
            raise ValueError('materialization source must be canonical and not a symlink')
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('materialization SHA missing')
        dest = folders / relative
        h = hashlib.sha256()
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as inp:
            before = _stamp(os.fstat(inp.fileno()))
            if before[2] != item['size']:
                raise ValueError('materialization source size mismatch')
            with dest.open('xb') as out:
                for data in iter(lambda: inp.read(1024 * 1024), b''):
                    out.write(data)
                    h.update(data)
            after = _stamp(os.fstat(inp.fileno()))
        if (before != after or before != _stamp(source.stat()) or h.hexdigest() != digest
                or dest.is_symlink() or not dest.is_file() or dest.stat().st_size != item['size']
                or _digest(dest) != digest):
            raise ValueError('source changed or copied bytes differ: ' + relative)
        rows[relative] = {'source_sha256': digest, 'copy_sha256': digest, 'bytes': item['size'],
                          'source_stat': list(before), 'copy_inode': dest.stat().st_ino}
    return {'profile': PROFILE, 'files': rows, 'bytes': sum(x['bytes'] for x in rows.values()),
            'wall_seconds': time.monotonic() - wall_start, 'cpu_seconds': time.process_time() - cpu_start}


def materialize_public(files: dict[str, dict[str, str]], folders: Path,
                       pinned: dict[str, str]) -> dict[str, Any]:
    if any(pinned.get(name) != digest for name, digest in PINNED.items()):
        raise ValueError('frozen asset profile differs')
    declared = {}
    for name, size in SIZES.items():
        item = files.get('weight:' + name)
        if not isinstance(item, dict) or item.get('sha256') != pinned.get(name):
            raise ValueError('public model folder does not match frozen asset: ' + name)
        declared[name] = {'path': item['path'], 'sha256': item['sha256'], 'size': size}
    return copy_declared(declared, folders)


def cleanup_completed(folders: Path, receipt: dict[str, Any]) -> None:
    """Remove only successfully created, unchanged copies, after model completion."""
    rows = receipt['files']
    if receipt.get('profile') != PROFILE or set(rows) != set(SIZES):
        raise ValueError('materialization cleanup receipt differs')
    for relative, row in rows.items():
        p = folders / relative
        if (p.is_symlink() or not p.is_file() or p.stat().st_ino != row['copy_inode']
                or p.stat().st_size != row['bytes'] or _digest(p) != row['copy_sha256']):
            raise ValueError('refusing to remove changed materialization')
    for relative in rows:
        (folders / relative).unlink()
