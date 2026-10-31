"""Adapt the recorded Al step-5 prepare and fit specs to W7's first-half Nadal clips."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.tennis import w7_specs, window_manifest as wm

PREPARE_SHA = 'd2519b38cff1e26f7365c2d7449196b8f0b8db0513fb291ecd184e17093ad382'
FIT_SHA = '2e7ef9c434a151e64289fc443b083efa9a2d6691e3343043b1dd706b21bdbaa1'


def replace_value(args: list[str], key: str, value: str) -> None:
    if args.count(key) != 1:
        raise ValueError(f'expected one {key}')
    args[args.index(key) + 1] = value


def reference(value: str) -> tuple[str, str]:
    path, digest = value.rsplit(':', 1)
    if not Path(path).is_absolute() or not wm.SHA.fullmatch(digest):
        raise ValueError('input must be an absolute path and full sha256')
    return path, digest


def modify(template: dict[str, Any], kind: str, windows: list[wm.WindowSpec],
           archive: tuple[str, str], data_dir: str, deadline: str) -> dict[str, Any]:
    if len(windows) != 3 or any(w.match != 'nadal_kyrgios' or w.fps != 50 for w in windows):
        raise ValueError('three Nadal 50 fps fit windows required')
    spec = copy.deepcopy(template)
    args = spec['arguments']
    stages = {x['name']: x for x in spec['staged_inputs']}
    if kind == 'prepare':
        data_dir = w7_specs.data_dir(data_dir)
        if spec['entrypoint'] != ['-m', 'experiments.tennis.k6b'] or args[0] != 'prepare-match':
            raise ValueError('wrong prepare template')
        lo, hi = args.index('--windows') + 1, args.index('--sources')
        args[lo:hi] = [w.id for w in windows]
        replacements = {'inputs': archive, **{f'c{i}': (f'{data_dir}/sources/{w.source_file}', w.source_sha256)
                                               for i, w in enumerate(windows)}}
        spec['smoke']['representative_basis'] = 'First drawn Nadal fit window; K4 conversion, adopted players, regions and YOLO pose, as Al step 5.'
    elif kind == 'fit':
        if spec['entrypoint'] != ['-m', 'experiments.tennis.k6b_fit'] or args[0] != 'fit':
            raise ValueError('wrong fit template')
        for key, value in [('--minutes', '10'), ('--rank', '4'), ('--quantize', '8'), ('--base', 'tennis-base@fit-100.pt')]:
            if args[args.index(key) + 1] != value:
                raise ValueError(f'changed frozen recipe: {key}')
        if '--also-unquantized' not in args:
            raise ValueError('missing frozen unquantized export')
        replace_value(args, '--match', 'nadal_kyrgios')
        replace_value(args, '--name', 'nk-C')
        replacements = {'fitdata': archive}
        spec['smoke']['representative_basis'] = '200 optimizer steps on the first-half Nadal fit data using the frozen Al step-5 recipe.'
    else:
        raise ValueError('unknown spec kind')
    for name, (path, digest) in replacements.items():
        stages[name]['path'], stages[name]['sha256'] = path, digest
    spec['inputs'] = [{'path': x['path'], 'sha256': x['sha256']} for x in spec['staged_inputs']]
    spec['verify_staged_inputs'] = True
    spec['deadline'] = deadline
    spec.pop('deadline_epoch', None)
    return spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('prepare', 'fit'))
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--archive', required=True, help='upstream absolute path:sha256')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = a.template.read_bytes()
    if hashlib.sha256(data).hexdigest() != (PREPARE_SHA if a.kind == 'prepare' else FIT_SHA):
        raise ValueError('original Al specification has changed')
    if a.out.exists():
        raise FileExistsError(a.out)
    spec = modify(json.loads(data), a.kind, wm.load(a.windows), reference(a.archive), a.data_dir, a.deadline)
    a.out.write_text(json.dumps(spec, indent=2) + '\n')
    print(a.out, hashlib.sha256(a.out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
