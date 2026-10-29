"""CPU-only source/API audit of the installed codec and pinned public code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any

from experiments.tennis.k6b_ideal_player import digest, write

PATTERNS = ('def forward', 'def compress', 'def decompress', 'requires_grad', 'backward(',
            'torch.no_grad', 'inference_mode', 'getDefaultCUDAStream', 'getCurrentCUDAStream',
            'cudaStream_t', 'latent', 'optimize', 'Proxy')


def describe(raw: bytes) -> dict[str, Any]:
    text = raw.decode('utf8', errors='replace')
    names = []
    try:
        tree = ast.parse(text)
        names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError:
        pass
    excerpts = [{'line': i, 'text': line[:350]} for i, line in enumerate(text.splitlines(), 1)
                if any(p in line for p in PATTERNS)]
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'functions': names, 'excerpts': excerpts[:100]}


def run(a: argparse.Namespace) -> None:
    start = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU source audit requires CUDA hidden')
    if digest(a.upstream) != a.upstream_sha:
        raise ValueError('Pinned source archive changed')
    out = Path(os.environ['PS_SCRATCH_DIR'])/'publish'
    out.mkdir()
    upstream = {}
    with tarfile.open(a.upstream) as tar:
        f = tar.extractfile('manifest.json')
        if f is None:
            raise ValueError('Missing pinned upstream manifest')
        manifest = json.load(f)
        for tag, row in manifest.items():
            upstream[tag] = {'repository': row['repository'], 'revision': row['revision'], 'files': {}}
            for name, identity in row['files'].items():
                stream = tar.extractfile(tag+'/'+name)
                if stream is None:
                    raise ValueError('Missing pinned source')
                raw = stream.read()
                if hashlib.sha256(raw).hexdigest() != identity['sha256']:
                    raise ValueError('Upstream source mismatch')
                upstream[tag]['files'][name] = describe(raw)
    tree = Path(sys.prefix)/'opt/DCVC'
    if not tree.is_dir():
        raise ValueError('Expected installed codec tree: '+str(tree))
    selected = []
    for p in sorted(tree.rglob('*')):
        if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.h', '.cpp') and '.git' not in p.parts:
            selected.append(p)
            if len(selected) > 1500:
                raise ValueError('Unexpectedly large installed source tree')
    installed = {}
    with tarfile.open(out/'installed-sources.tar', 'x') as tar:
        for p in selected:
            name = str(p.relative_to(tree))
            raw = p.read_bytes()
            installed[name] = describe(raw)
            tar.add(p, arcname=name, recursive=False)
    shutil.copyfile(a.upstream, out/'upstream-sources.tar')
    result = {'device': 'cpu', 'upstream_sha256': a.upstream_sha, 'upstream': upstream,
        'installed_tree': str(tree), 'installed_sources': installed,
        'installed_source_count': len(installed), 'seconds': time.monotonic()-start,
        'scope': 'Static source/API evidence only; no checkpoint inference, training or codec performance evidence.',
        'latent_gpu_dispatch_requires': ['root review of the actual installed gradient path',
            'native deterministic decoding gate', 'real entropy-coded optimization compatibility']}
    write(out/'audit.json', result)
    write(Path(os.environ['PS_STAGE_DIR'])/'code-audit-summary.json', result)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage/'code-audit-summary.json').read_text())
    checks = {'cpu': s['device'] == 'cpu', 'source_count': 10 <= s['installed_source_count'] <= 1500,
              'upstream': set(s['upstream']) == {'dt', 'cm'},
              'pinned': all(len(r['revision']) == 40 and r['files'] for r in s['upstream'].values())}
    with tarfile.open(stage/'published.tar') as tar:
        checks['publication'] = tar.getmember('publish/installed-sources.tar').size > 0 and tar.getmember('publish/upstream-sources.tar').size > 0
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--upstream', type=Path)
    p.add_argument('--upstream-sha')
    p.add_argument('--phase', choices=('smoke', 'full'))
    a = p.parse_args()
    if a.validate:
        validate()
    elif a.upstream is None or a.upstream_sha is None:
        p.error('Pinned source archive required')
    else:
        run(a)


if __name__ == '__main__':
    main()
