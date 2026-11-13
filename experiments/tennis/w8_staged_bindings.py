"""Explicit, checked staged-only CLI bindings for the W8 fleet profiles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROFILE = 'w8-explicit-staged-only-v1'
WITNESS_DIRECT = frozenset(('preflight', 'identity', 'oracle', 'cpu_proof', 'ledger', 'admission',
                            'promotion', 'producer_audit', 'producer0', 'producer1'))
CPU_DIRECT = frozenset(('preflight', 'identity'))


def staged_only_names(spec: dict[str, Any]) -> list[str]:
    direct = CPU_DIRECT if spec.get('w8_combined_cpu_proof_profile') else WITNESS_DIRECT
    return [row['name'] for row in spec['staged_inputs'] if row['name'] not in direct]


def arguments(spec: dict[str, Any]) -> list[str]:
    staged = spec.get('staged_inputs')
    if not isinstance(staged, list) or not staged:
        raise ValueError('staged inputs required')
    names = [row['name'] for row in staged]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('distinct staged input names required')
    pairs = [(row['path'], row['sha256']) for row in staged]
    direct = [(row['path'], row['sha256']) for row in spec['inputs']]
    if len(set(pairs)) != len(pairs) or len(set(direct)) != len(direct) or set(pairs) != set(direct):
        raise ValueError('staged-only input identities must match declared inputs')
    return ['{phase}', *[part for name in staged_only_names(spec)
                         for part in ('--staged', name, '{staged:' + name + '}')]]


def configure(spec: dict[str, Any]) -> None:
    spec['w8_staged_arguments_profile'] = PROFILE
    spec['verify_staged_inputs'] = True
    spec['arguments'] = arguments(spec)


def check_spec(spec: dict[str, Any]) -> None:
    if (spec.get('w8_staged_arguments_profile') != PROFILE
            or spec.get('verify_staged_inputs') is not True or spec.get('arguments') != arguments(spec)):
        raise ValueError('explicit staged-only argument profile differs')


def compatible_arguments(spec: dict[str, Any]) -> bool:
    if spec.get('w8_staged_arguments_profile') is None:
        return spec.get('arguments') == ['{phase}']
    return (spec.get('w8_staged_arguments_profile') == PROFILE
            and spec.get('verify_staged_inputs') is True and spec.get('arguments') == arguments(spec))


def check_cli(stage: Path, supplied: list[list[str]] | None, action: str) -> None:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    if spec.get('w8_staged_arguments_profile') is None:
        if supplied is not None or not compatible_arguments(spec):
            raise ValueError('legacy staged CLI differs')
        return
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    check_spec(spec)
    rows = spec['staged_inputs']
    actual = staging['inputs']
    dispatch_rows = dispatch['staged_inputs']
    declared = {x['name']: x for x in rows}
    staged = {x['name']: x for x in actual}
    dispatched = {x['name']: x for x in dispatch_rows}
    if (len(declared) != len(rows) or len(staged) != len(actual)
            or len(dispatched) != len(dispatch_rows) or set(staged) != set(declared)
            or dispatched != staged or dispatch.get('input_identity') != spec['inputs'] or supplied is None
            or len(supplied) != len(staged_only_names(spec))):
        raise ValueError('staged CLI set or dispatch differs')
    bindings = {name: value for name, value in supplied}
    if len(bindings) != len(supplied) or set(bindings) != set(staged_only_names(spec)):
        raise ValueError('duplicate or missing staged CLI binding')
    expected_command = [action, *[part for name, value in supplied for part in ('--staged', name, value)]]
    if dispatch.get('command', [])[-len(expected_command):] != expected_command:
        raise ValueError('dispatch command differs from staged CLI')
    root = Path(staging['local_root']).resolve(strict=True)
    seen: set[Path] = set()
    for name, row in declared.items():
        actual_row = staged[name]
        raw = Path(actual_row['path'])
        path = raw.resolve(strict=True)
        original = Path(row['path']).resolve(strict=True)
        st, old = path.stat(), original.stat()
        if (not raw.is_absolute() or (name in bindings and bindings[name] != str(raw))
                or not path.is_relative_to(root) or path == original
                or (st.st_dev, st.st_ino) == (old.st_dev, old.st_ino)
                or path in seen or actual_row.get('source') != row['path']
                or actual_row.get('sha256') != row['sha256']
                or actual_row.get('extract', False) != row.get('extract', False)):
            raise ValueError('staged CLI path or identity differs: ' + name)
        seen.add(path)
