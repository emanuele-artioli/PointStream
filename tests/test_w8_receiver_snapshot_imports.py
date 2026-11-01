"""Check source snapshot closure without importing model runtimes or executing workers."""

import ast
from collections import deque
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/private/tmp/w8-cost-source-b581-frozen-20261101.tar')
START = ('experiments.tennis.w8_receiver_generate', 'experiments.tennis.w8_inputs')
TOOLS = ('tools/datasets/k4b_inputs.py', 'tools/datasets/k4_inputs.py')


def _module_path(module):
    path = ROOT.joinpath(*module.split('.'))
    for candidate in (path.with_suffix('.py'), path / '__init__.py'):
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def _closure(snapshot):
    pending = deque(START)
    seen = set()
    missing = set()
    while pending:
        module = pending.popleft()
        if module in seen:
            continue
        seen.add(module)
        name = _module_path(module)
        if name is None:
            continue
        if name not in snapshot:
            missing.add(name)
            continue
        tree = ast.parse(snapshot[name], filename=name)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    package = module if name.endswith('/__init__.py') else module.rpartition('.')[0]
                    parts = package.split('.')
                    base = '.'.join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                names = [base, *(f'{base}.{alias.name}' for alias in node.names if alias.name != '*')]
            pending.extend(name for name in names if name.split('.')[0] in {'experiments', 'src', 'tools'})
    return missing


def _snapshot(include_tools):
    result = {}
    for directory in ('experiments', 'src', 'tools'):
        if directory == 'tools' and not include_tools:
            continue
        for path in (ROOT / directory).rglob('*.py'):
            result[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return result


def test_decoder_snapshot_covers_transitive_tools_imports():
    missing = _closure(_snapshot(include_tools=False))
    assert 'tools/datasets/k4b_inputs.py' in missing
    complete = _snapshot(include_tools=False)
    complete.update({name: (ROOT / name).read_bytes() for name in TOOLS})
    assert not _closure(complete)
    del complete['tools/datasets/k4_inputs.py']
    assert 'tools/datasets/k4_inputs.py' in _closure(complete)


def test_actual_frozen_snapshot_exposes_missing_import():
    if not FROZEN.is_file():
        pytest.skip('local failed-run source archive is not available')
    with tarfile.open(FROZEN, 'r:*') as tar:
        snapshot = {m.name.removeprefix('./'): tar.extractfile(m).read()
                    for m in tar if m.isfile() and m.name.endswith('.py')}
    assert 'tools/datasets/k4b_inputs.py' in _closure(snapshot)
    snapshot.update({name: (ROOT / name).read_bytes() for name in TOOLS})
    assert not _closure(snapshot)
