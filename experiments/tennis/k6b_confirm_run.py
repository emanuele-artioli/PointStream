"""Run an unchanged native stage and validate its complete output before return.

The installed fleet expands staged placeholders only for workload arguments.
Its validator therefore invokes --validate-only, which rechecks actual stage
artifacts using the saved immutable-input paths, without unresolved placeholders.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from experiments.tennis.k6b_confirm import check_manifest


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    split = arguments.index('--') if '--' in arguments else len(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--kind', choices=('anchors', 'prepare', 'generate', 'judge'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--k6-stage', action='append', type=Path, default=[])
    parser.add_argument('--windows', nargs='+')
    args = parser.parse_args(arguments[:split])
    from experiments.tennis.k6b_generate import stage_dir, write_json
    from experiments.tennis.k6b_confirm_validate import main as validate

    stage = stage_dir()
    record_path = stage / 'confirmation-invocation.json'
    if args.validate_only:
        record: dict[str, Any] = json.loads(record_path.read_text())
        return validate(record['validation_arguments'])
    if args.manifest is None or args.kind is None:
        raise ValueError('Stage kind and frozen manifest required')
    check_manifest(json.loads(args.manifest.read_text()))
    validation = ['--kind', args.kind, '--manifest', str(args.manifest)]
    for key in ('inputs', 'prepared'):
        if value := getattr(args, key):
            validation.extend(['--' + key, str(value)])
    for path in args.k6_stage:
        validation.extend(['--k6-stage', str(path)])
    if args.windows:
        validation.extend(['--windows', *args.windows])
    write_json(record_path, {'validation_arguments': validation, 'native_arguments': arguments[split+1:]})
    native = arguments[split+1:]
    if args.kind == 'anchors':
        from experiments.tennis.k6 import main as run
    elif args.kind == 'generate':
        from experiments.tennis.k6b_generate import main as run
    else:
        from experiments.tennis.k6b import main as run
    code = run(native)
    if code:
        return code
    original = os.environ.get('PS_VALIDATION_PATH')
    os.environ['PS_VALIDATION_PATH'] = str(stage / 'scientific-validation.json')
    try:
        return validate(validation)
    finally:
        if original is None:
            os.environ.pop('PS_VALIDATION_PATH', None)
        else:
            os.environ['PS_VALIDATION_PATH'] = original


if __name__ == '__main__':
    raise SystemExit(main())
