"""Representative dense carrier preset check, eight frames; no full-GOP rate claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from typing import Any

from experiments.tennis import k6b_foreground_video as dense
from experiments.tennis.k6b_rate_transport import write


def validate(root: Path) -> dict[str, Any]:
    report = json.loads((root/'dense-presets.json').read_text())
    checks = {'presets': report['presets'] == ([4] if report['mode'] == 'smoke' else [4, 1])}
    for preset in report['presets']:
        checked = dense.validate(root/f'p{preset}')
        checks.update({f'{preset}:{key}': value for key, value in checked['checks'].items()})
    if report['mode'] == 'smoke':
        checks['full_projection_fits_300s'] = report['wall_seconds']*3+30 < 300
    return {'passed': all(checks.values()), 'checks': checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--transport')
    p.add_argument('--source', action='append', default=[])
    p.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if a.validate:
        with tempfile.TemporaryDirectory(prefix='dense-presets-', dir='/tmp') as tmp:
            with tarfile.open(stage/'published.tar') as tar:
                tar.extractall(tmp, filter='data')
            checked = validate(Path(tmp)/'publish')
        write(Path(os.environ['PS_VALIDATION_PATH']), checked)
        return 0 if checked['passed'] else 1
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch/'publish'
    publish.mkdir()
    began = time.monotonic()
    report: dict[str, Any] = {'mode': a.mode, 'presets': [4] if a.mode == 'smoke' else [4, 1],
        'scope': 'first8frames-both-original-pilot-windows-not-full-GOP-rate-evidence'}
    original_argv = sys.argv
    try:
        for preset in report['presets']:
            work, child_stage = scratch/f'p{preset}', stage/f'p{preset}'
            work.mkdir()
            child_stage.mkdir()
            os.environ['PS_SCRATCH_DIR'], os.environ['PS_STAGE_DIR'] = str(work), str(child_stage)
            sys.argv = ['dense', '--transport', a.transport, '--frames', '8', '--crfs', '63', '--formats', 'affine', '--preset', str(preset)]
            for source in a.source:
                sys.argv += ['--source', source]
            if dense.main():
                raise ValueError('Dense native preset check failed')
            shutil.move(str(work/'publish'), publish/f'p{preset}')
    finally:
        os.environ['PS_SCRATCH_DIR'], os.environ['PS_STAGE_DIR'] = str(scratch), str(stage)
        sys.argv = original_argv
    report['wall_seconds'] = time.monotonic()-began
    write(publish/'dense-presets.json', report)
    checked = validate(publish)
    write(publish/'validation.json', checked)
    if not checked['passed']:
        raise ValueError('Dense preset validation/projection failed')
    print(json.dumps({'wall_seconds': report['wall_seconds'], 'presets': report['presets'], 'gpu_hours': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
