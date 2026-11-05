"""Build a source-only, SHA-bound PointStream package tree for W8 receivers.

The fleet source snapshot and preflight must be authenticated separately. No
non-Python files, source videos, documentation, weights or datasets are copied.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage_source(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if output.exists():
        raise ValueError('source output must be new')
    if output.resolve().is_relative_to(source):
        raise ValueError('source output cannot be inside the authenticated snapshot')
    files: dict[str, str] = {}
    for package in ('experiments', 'src', 'tools'):
        root = source / package
        if root.is_symlink() or not root.is_dir():
            raise ValueError('source package missing or symlinked')
        for path in sorted(root.rglob('*.py')):
            if any(part in ('__pycache__', 'Datasets', 'Models') or part.startswith('.') for part in path.relative_to(root).parts):
                continue
            if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError('source package contains non-regular or escaped file')
            files[str(path.relative_to(source))] = sha(path)
    if not files or 'experiments/tennis/w8_strict_receiver_worker.py' not in files:
        raise ValueError('strict receiver source missing')
    output.mkdir(parents=True)
    for name, expected in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (source / name).open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out, 1024 * 1024)
        if sha(target) != expected or sha(source / name) != expected:
            raise ValueError('source changed while staging')
        os.chmod(target, 0o444)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'scope': 'Python source only; fleet source snapshot authenticated separately',
            'files': files, 'manifest_sha256': hashlib.sha256(encoded).hexdigest()}


def verify_source(output: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get('files')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('source file manifest missing')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if actual != set(expected) or any(not name.endswith('.py') for name in actual):
        raise ValueError('source package membership differs')
    for name, digest in expected.items():
        target = output / name
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(output.resolve(strict=True)) or sha(target) != digest:
            raise ValueError('source file identity differs')
    encoded = json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest.get('manifest_sha256'):
        raise ValueError('source manifest identity differs')
