"""Strict host-cost model worker with a child-observed GPU identity receipt."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.tennis import w8_strict_receiver_worker as receiver
from experiments.tennis import w8_cuda_system_leaves as policy


WRAPPER_RELATIVE_PATH = 'experiments/tennis/w8_strict_host_model_worker.py'


def wrapper_identity(source_root: Path, actual_file: Path) -> dict[str, str]:
    root = source_root.resolve(strict=True)
    expected = root / WRAPPER_RELATIVE_PATH
    actual = actual_file.resolve(strict=True)
    if expected.is_symlink() or actual != expected or not actual.is_file():
        raise ValueError('model wrapper entrypoint is outside authenticated source root or has wrong name')
    return {'path': WRAPPER_RELATIVE_PATH, 'sha256': receiver.digest(actual)}


def pin_declared_aa_imports(config: dict) -> None:
    """Give the authenticated opt source precedence without accepting other origins."""
    files = config['files']
    entry = files['code:animate_anyone']
    init = Path(entry['path'])
    if (init.is_symlink() or init.name != '__init__.py' or init.parent.name != 'animate_anyone'
            or not init.is_file() or init.resolve(strict=True) != init
            or receiver.digest(init) != entry['sha256']):
        raise ValueError('declared AA package origin or hash differs')
    for name, module in tuple(sys.modules.items()):
        if name != 'animate_anyone' and not name.startswith('animate_anyone.'):
            continue
        declared = files.get(f'code:{name}')
        actual = getattr(module, '__file__', None)
        spec = getattr(module, '__spec__', None)
        if not declared or not actual or not spec or not spec.origin:
            raise ValueError(f'preloaded AA module is not declared: {name}')
        expected = Path(declared['path'])
        if (expected.is_symlink() or expected.resolve(strict=True) != expected
                or Path(actual).resolve(strict=True) != expected
                or Path(spec.origin).resolve(strict=True) != expected
                or receiver.digest(expected) != declared['sha256']):
            raise ValueError(f'preloaded AA module origin or hash differs: {name}')
        search = getattr(module, '__path__', None)
        locations = spec.submodule_search_locations
        if ((search is None) != (locations is None)
                or (search is not None and ([Path(p).resolve(strict=True) for p in search] != [expected.parent]
                    or [Path(p).resolve(strict=True) for p in locations] != [expected.parent]))):
            raise ValueError(f'preloaded AA package search path differs: {name}')
    # Preserve the source-only project root in the first position.
    parent = str(init.parent.parent)
    if not sys.path or sys.path[0] == parent:
        raise ValueError('project source path must remain first')
    sys.path[1:] = [parent, *(entry for entry in sys.path[1:] if entry != parent)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--loader', type=Path, required=True)
    parser.add_argument('--loader-sha256', required=True)
    parser.add_argument('--canary', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--materialize-profile', action='store_true')
    args = parser.parse_args()
    if receiver.digest(args.config) != args.config_sha256:
        raise ValueError('model worker config identity mismatch')
    config = json.loads(args.config.read_text())
    if config.get('worker_kind') != 'cost_probe':
        raise ValueError('host model wrapper only admits bounded cost probes')
    wrapper = wrapper_identity(args.source_root, Path(__file__))
    if args.materialize_profile:
        pin_declared_aa_imports(config)
        receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root, materialize_profile=True)
    else:
        receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root)
    if args.materialize_profile:
        pin_declared_aa_imports(config)
    import torch
    if not torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one initialized CUDA device required')
    props = torch.cuda.get_device_properties(0)
    observed = str(getattr(props, 'uuid', ''))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (policy.canonical_gpu_uuid(observed) != policy.canonical_gpu_uuid(visible)
            or (props.major, props.minor) != (8, 9) or 'RTX 6000 Ada' not in props.name):
        raise ValueError('observed CUDA GPU does not match the selected Ada')
    output = Path(config['output'])
    result = {'visible_devices': visible, 'device_uuid': observed, 'name': props.name,
              'major': props.major, 'minor': props.minor, 'sealed_sha256': receiver.digest(output / 'sealed.json'),
              'config_sha256': args.config_sha256, 'cuda_device_count': 1, 'wrapper': wrapper,
              'materialization_profile': 'public-model-folders-copy-v1' if args.materialize_profile else None,
              'materialization_sha256': receiver.digest(output / 'materialization.json') if args.materialize_profile else None}
    with (output / 'gpu-identity.json').open('x') as f:
        json.dump(result, f, sort_keys=True)
        f.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
