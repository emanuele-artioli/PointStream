"""Packet-only preparation and offline crop checks for the W8 receiver.

These helpers are not a GPU job or an isolation boundary. No entry point is
provided until the restricted launcher and its host qualification exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import w8_input_job as fixed


ORACLE_UNIT_SHA = {
    fixed.UNITS[0]: '387544a3fc344685f7c07632879affbdafcfd54fd6e1c00e0d9d6c06ca98995d',
    fixed.UNITS[1]: '74be8c9e2f8eb65ca67e6339ef9a2a2f8238df36c60a4b5a07a45fc68d3712a7',
}
ORACLE_TENSOR_SHA = 'd6fe0331cf5fdbb7ab70bb761be8dc62d4e06c8f43f715c489064d2fc3375dd2'


def check_oracle_fingerprints(fingerprints: dict[str, Any]) -> None:
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(',', ':')).encode()
    if len(fingerprints) != 480 or hashlib.sha256(encoded).hexdigest() != ORACLE_TENSOR_SHA:
        raise ValueError('fingerprints do not match independent CPU oracle')


def prepare_packet(bundle: Path, profile_path: Path, guider: Path, output: Path,
                   expected_fingerprints: dict[str, Any]) -> dict[str, Any]:
    """Materialize packet-derived input, to be called only inside an isolated worker."""
    import torch

    from experiments.tennis.w8_payload import unpack_match

    profile = fixed.pinned_profile(profile_path)
    arrays, bindings, match = fixed.decode_frozen_window(bundle.read_bytes(), profile)
    fixed.check_installed_guider(guider, profile)
    base = torch.load(guider, map_location='cpu', weights_only=True)
    state = unpack_match(match, profile['match']['sha256'], fixed.GUIDER_SHA, base)
    fingerprints = fixed.tensor_fingerprints(state)
    fixed.check_worker_match({'match_tensors': fingerprints}, {'tensor_fingerprints': expected_fingerprints})
    check_oracle_fingerprints(fingerprints)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('fresh empty output directory required')
    units = output / 'publish' / 'units'
    units.mkdir(parents=True)
    hashes = {}
    for name in fixed.UNITS:
        path = units / f'{name}.npz'
        np.savez_compressed(path, **arrays[name])
        hashes[name] = fixed.file_sha(path)
        if hashes[name] != ORACLE_UNIT_SHA[name]:
            raise ValueError(f'decoded NPZ differs from independent CPU oracle: {name}')
    preparation = {'windows': [{'units': [{'unit': name} for name in fixed.UNITS]}]}
    (output / 'publish' / 'prepare.json').write_text(json.dumps(preparation, sort_keys=True))
    checkpoint = output / 'decoded.pt'
    torch.save(state, checkpoint)
    return {'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
            'match_sha256': profile['match']['sha256'], 'guider_sha256': fixed.GUIDER_SHA,
            'match_tensors': fingerprints, 'unit_sha256': hashes, 'bindings': bindings,
            'checkpoint_sha256': fixed.file_sha(checkpoint)}



def verify_serialized_match(checkpoint: Path, expected_fingerprints: dict[str, Any]) -> str:
    """Recheck packet-decoded state in the consuming runtime before model loading."""
    import torch

    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actual = fixed.tensor_fingerprints(state)
    fixed.check_worker_match({'match_tensors': actual}, {'tensor_fingerprints': expected_fingerprints})
    return fixed.file_sha(checkpoint)


def generator_arguments(prepared_archive: Path, checkpoint: Path, base: Path,
                        weights: dict[str, Path], gpu_class: str, smoke: bool = False) -> list[str]:
    """Construct a fixed recipe. Smoke renders one full unit, never a short context."""
    if not weights or not gpu_class:
        raise ValueError('declared model files and GPU class required')
    args = ['generate', '--prepared', str(prepared_archive), '--profiles', '', '--steps', '30',
            '--from-t', '0', '--fit', 'match5', str(checkpoint), '--fit-base', 'match5', str(base),
            '--cfg', 'match5', '2', '--gpu-class', gpu_class]
    if smoke:
        args += ['--units', fixed.UNITS[0]]
    for name, path in sorted(weights.items()):
        args += ['--weight', name, str(path)]
    return args


def validate_crops(publish: Path, units: tuple[str, ...] = fixed.UNITS,
                   steps: int = 30, scope: str = 'candidate') -> dict[str, str]:
    """Independently check sealed generated crop files and sidecar coverage.

    This does not compare historical pixels, model identities, device or kernel.
    """
    if units not in (fixed.UNITS, (fixed.UNITS[0],), (fixed.UNITS[1],)):
        raise ValueError('unexpected units')
    hashes = {}
    for name in units:
        path = publish / f'{name}@match5.npz'
        sidecar = json.loads(path.with_suffix('.json').read_text())
        actual = fixed.file_sha(path)
        if (sidecar.get('unit') != name or sidecar.get('profile') != 'match5'
                or sidecar.get('frames_generated') != 240 or sidecar.get('kept') != 80
                or sidecar.get('steps') != steps or sidecar.get('scope') != scope
                or sidecar.get('seed') != int(hashlib.sha256(name.encode()).hexdigest()[:7], 16)
                or sidecar.get('npz_sha256') != actual):
            raise ValueError(f'generated sidecar mismatch: {name}')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'t', 'crops'}:
                raise ValueError(f'generated members mismatch: {name}')
            ticks, crops = data['t'], data['crops']
            if (not np.array_equal(ticks, np.arange(0, 240, 3))
                    or crops.shape != (80, 784, 512, 3) or crops.dtype != np.uint8):
                raise ValueError(f'generated coverage mismatch: {name}')
        hashes[name] = actual
    return hashes


WEIGHT_NAMES = frozenset({'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
                          'image_encoder/config.json', 'image_encoder/pytorch_model.bin',
                          'original/denoising_unet.pth', 'original/reference_unet.pth',
                          'original/pose_guider.pth', 'original/motion_module.pth'})


PINNED_WEIGHTS = {
    'vae/config.json': '92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e',
    'vae/diffusion_pytorch_model.safetensors': 'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815',
    'image_encoder/config.json': 'ba41139cc9dad159163a53e2aa847356b627f53fef988ad461659bd2d587d886',
    'image_encoder/pytorch_model.bin': '89d2aa29b5fdf64f3ad4f45fb4227ea98bc45156bbae673b85be1af7783dbabb',
    'original/denoising_unet.pth': 'b9e5a2c34fac369e8a922972ca2210916c6af175a0dad907deccf6235816ad52',
    'original/reference_unet.pth': 'beddccb08d49a8b29b0f4d6d456c6521d4382a8d8d48884fa60ba8802509c214',
    'original/pose_guider.pth': '1a8b7c1b4db92980fd977b4fd003c1396bbae9a9cdea00c35d452136d5e4f488',
    'original/motion_module.pth': '0d11e01a281b39880da2efeea892215c1313e5713fca3d100a7fbb72ee312ef9',
}
BASE_SHA = '85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4'
GENERATOR_CODE_SHA = '00248aa4fdd47e634f82b37a7a33c99f1794440d65cbfe9395255a4225735542'
FIT_CODE_SHA = '5e76dd363174d370572b3bab73e2d5a533f274371ebdc612cc764a947f2a9c69'
AA_ARCHIVE_SHA = 'bfca6cf1129a0154bf2c3431c0e01b34f5e815eda5ba504f54e001c8ceeb99ee'


def check_frozen_code(files: dict[str, Any]) -> None:
    expected = {'code:experiments.tennis.k6b_generate': GENERATOR_CODE_SHA,
                'code:experiments.tennis.k6b_fit': FIT_CODE_SHA, 'aa_archive': AA_ARCHIVE_SHA}
    if any(files.get(name, {}).get('sha256') != digest for name, digest in expected.items()):
        raise ValueError('generator, fit or AA source identity mismatch')


def run_packet_isolated(config: dict[str, Any], isolation: dict[str, Any]) -> None:
    """Decode only paid data in a separately restricted CPU process."""
    files = config['files']
    if set(files) != {'bundle', 'profile', 'guider'}:
        raise ValueError('exact CPU decoder input set required')
    expected = {'bundle': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'guider': fixed.GUIDER_SHA}
    if any(files[key]['sha256'] != digest for key, digest in expected.items()):
        raise ValueError('CPU decoder inputs are not frozen assets')
    result = prepare_packet(Path(files['bundle']['path']), Path(files['profile']['path']),
                            Path(files['guider']['path']), Path(config['output']), config['tensor_fingerprints'])
    result['isolation'] = isolation
    (Path(config['output']) / 'decoded-manifest.json').write_text(json.dumps(result, sort_keys=True))


def render_full_unit(renderer: Any, unit: dict[str, np.ndarray], seed: int,
                     steps: int) -> tuple[list[int], list[np.ndarray]]:
    """Render each complete native run and keep only the 80 paid ticks."""
    if type(steps) is not int or steps not in (1, 3, 6, 30):
        raise ValueError('unsupported sampler step count')
    ticks: list[int] = []
    crops: list[np.ndarray] = []
    frames = 0
    for run in np.unique(unit['run']):
        rows = np.flatnonzero(unit['run'] == run)
        video = renderer.render(unit['reference'], unit['pose'][rows], steps, seed * 1009 + int(run))
        if len(video) != len(rows):
            raise ValueError('renderer returned incomplete run')
        frames += len(rows)
        for index, row in enumerate(rows):
            if unit['judged'][row]:
                ticks.append(int(unit['t'][row]))
                crops.append(video[index])
    if frames != 240 or ticks != list(range(0, 240, 3)):
        raise ValueError('generated output coverage mismatch')
    return ticks, crops


def run_isolated(config: dict[str, Any], isolation: dict[str, Any]) -> None:
    """Render one complete packet-derived unit after the stdlib isolation bootstrap."""
    import resource
    import time

    import torch

    from experiments.tennis.k6b_generate import Renderer

    torch.set_num_threads(1)
    paths = {key: Path(item['path']) for key, item in config['files'].items()}
    for key, path in paths.items():
        if fixed.file_sha(path) != config['files'][key]['sha256']:
            raise ValueError(f'asset identity mismatch: {key}')
    weights = {key.removeprefix('weight:'): path for key, path in paths.items() if key.startswith('weight:')}
    model_keys = {'checkpoint', 'unit', 'base', 'decoder_manifest', 'aa_archive'} | {f'weight:{n}' for n in WEIGHT_NAMES}
    code_keys = {key for key in paths if key.startswith('code:')}
    required_code = {'code:experiments.tennis.k6b_generate', 'code:experiments.tennis.k6b_fit',
                     'code:animate_anyone', 'code:animate_anyone.pipelines.pipeline_pose2vid_long'}
    if set(weights) != WEIGHT_NAMES or set(paths) != model_keys | code_keys or not required_code <= code_keys:
        raise ValueError('exact generator assets and declared source modules required')
    if config['files']['base']['sha256'] != BASE_SHA or any(
            config['files'][f'weight:{name}']['sha256'] != digest for name, digest in PINNED_WEIGHTS.items()):
        raise ValueError('generator weights or base differ from frozen profile')
    check_frozen_code(config['files'])
    from experiments.tennis.w8_receiver_preflight import verify_aa_tree

    aa_root = paths['code:animate_anyone'].resolve().parent.parent
    aa_verified = verify_aa_tree(paths['aa_archive'], aa_root)['files']
    checkpoint = paths['checkpoint']
    checkpoint_sha = verify_serialized_match(checkpoint, config['tensor_fingerprints'])
    decoded_manifest = json.loads(paths['decoder_manifest'].read_text())
    if (decoded_manifest.get('packet_sha256') != fixed.WINDOW_SHA
            or decoded_manifest.get('profile_sha256') != fixed.PROFILE_SHA
            or decoded_manifest.get('checkpoint_sha256') != checkpoint_sha
            or decoded_manifest.get('match_tensors') != config['tensor_fingerprints']
            or decoded_manifest.get('unit_sha256') != ORACLE_UNIT_SHA):
        raise ValueError('decoder manifest is not bound to accepted packet oracle')
    unit_name = config['unit']
    if unit_name not in fixed.UNITS:
        raise ValueError('unexpected unit')
    check_oracle_fingerprints(config['tensor_fingerprints'])
    if config['files']['unit']['sha256'] != ORACLE_UNIT_SHA[unit_name]:
        raise ValueError('unit NPZ differs from independent CPU oracle')
    output = Path(config['output'])
    if any(output.iterdir()):
        raise ValueError('fresh empty output directory required')
    scratch = output / 'scratch'
    scratch.mkdir()
    with np.load(paths['unit'], allow_pickle=False) as loaded:
        unit = {key: loaded[key] for key in loaded.files}
    if (not np.array_equal(unit['t'], np.arange(240))
            or not np.array_equal(np.flatnonzero(unit['judged']), np.arange(0, 240, 3))
            or len(unit['pose']) != 240 or len(unit['run']) != 240):
        raise ValueError('packet-derived unit coverage mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    import os

    memory_mib = int(os.environ.get('PS_GPU_MEMORY_LIMIT_MIB', '24000'))
    if memory_mib <= 0:
        raise ValueError('positive GPU memory limit required')
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, memory_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    kind = config['worker_kind']
    if kind == 'generate':
        steps, scope = 30, 'candidate'
    elif kind == 'cost_probe' and type(config.get('probe_steps')) is int and config['probe_steps'] in (1, 3, 6):
        steps, scope = config['probe_steps'], 'qualification_only'
    else:
        raise ValueError('invalid generation mode or probe step count')

    def start_phase() -> tuple[float, float, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return time.monotonic(), usage.ru_utime + usage.ru_stime, event

    def finish_phase(start: tuple[float, float, Any]) -> dict[str, float]:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        torch.cuda.synchronize()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        wall = time.monotonic() - start[0]
        return {'wall_seconds': wall, 'cpu_seconds': usage.ru_utime + usage.ru_stime - start[1],
                'gpu_event_seconds': start[2].elapsed_time(event) / 1000,
                'gpu_reserved_wall_seconds': wall}

    load_started = start_phase()
    renderer = Renderer(weights, 'original', 'cuda', scratch, checkpoint, paths['base'], 0, 2.0)
    model_load_cost = finish_phase(load_started)
    import sys

    modules = {name: Path(str(module.__file__)).resolve() for name, module in sys.modules.items()
               if (name.startswith('animate_anyone') or name in ('experiments.tennis.k6b_generate',
                    'experiments.tennis.k6b_fit')) and getattr(module, '__file__', None)}
    for name, path in modules.items():
        if paths.get(f'code:{name}') is None or path != paths[f'code:{name}'].resolve():
            raise ValueError(f'undeclared executing source module: {name}')
        if name.startswith('animate_anyone') and aa_verified.get(str(path)) != fixed.file_sha(path):
            raise ValueError(f'AA source is absent from pinned archive: {name}')
    from experiments.audit.env_smoke import profile_cuda

    seed = int(hashlib.sha256(unit_name.encode()).hexdigest()[:7], 16)
    runs = np.unique(unit['run'])
    first_run = int(runs[0])
    first_rows = np.flatnonzero(unit['run'] == first_run)
    if len(first_rows) < 24:
        raise ValueError('first native run cannot exercise 24-frame context')
    original_probe_started = start_phase()
    original_probe = renderer.probe(unit['reference'], unit['pose'])
    original_probe_cost = finish_phase(original_probe_started)
    context_probe_started = start_phase()
    _, kernels = profile_cuda(lambda: renderer.render(unit['reference'], unit['pose'][first_rows[:24]],
                                                       1, seed * 1009 + first_run))
    context_phase_cost = finish_phase(context_probe_started)
    probe_cost = {'wall_seconds': context_phase_cost['wall_seconds'],
                  'cpu_seconds': context_phase_cost['cpu_seconds'],
                  'gpu_seconds': context_phase_cost['gpu_event_seconds'],
                  'frames': 24, 'steps': 1, 'run_id': first_run,
                  'seed': seed * 1009 + first_run, 'scope': 'kernel qualification, not quality evidence'}
    on_cuda = {name: bool(list(model.parameters())) and all(p.is_cuda for p in model.parameters())
               for name, model in renderer.modules.items()}
    device = torch.cuda.get_device_properties(0)
    qualification = {'device': {'name': device.name, 'major': device.major, 'minor': device.minor},
                     'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                     'on_cuda': on_cuda, 'kernels': kernels, 'executed_modules': modules,
                     'context_probe_cost': probe_cost}
    check_model_record(renderer.record, checkpoint_sha)
    check_execution_receipt(qualification)
    sampling_started = start_phase()
    ticks, crops = render_full_unit(renderer, unit, seed, steps)
    sampling_cost = finish_phase(sampling_started)
    frames = 240
    for name, module in sys.modules.items():
        if ((name.startswith('animate_anyone') or name in ('experiments.tennis.k6b_generate',
                'experiments.tennis.k6b_fit')) and getattr(module, '__file__', None)):
            path = Path(str(module.__file__)).resolve()
            if paths.get(f'code:{name}') is None or path != paths[f'code:{name}'].resolve():
                raise ValueError(f'undeclared executing source module: {name}')
            if name.startswith('animate_anyone') and aa_verified.get(str(path)) != fixed.file_sha(path):
                raise ValueError(f'AA source is absent from pinned archive: {name}')
            if fixed.file_sha(path) != config['files'][f'code:{name}']['sha256']:
                raise ValueError(f'executing source changed: {name}')
            modules[name] = path
    publication_started = start_phase()
    path = output / f'{unit_name}@match5.npz'
    np.savez_compressed(path, t=np.asarray(ticks, dtype=np.int64), crops=np.stack(crops))
    sidecar = {'unit': unit_name, 'profile': 'match5', 'frames_generated': frames, 'kept': len(ticks),
               'steps': steps, 'seed': seed, 'npz_sha256': fixed.file_sha(path), 'scope': scope}
    path.with_suffix('.json').write_text(json.dumps(sidecar, sort_keys=True))
    publication_cost = finish_phase(publication_started)
    device = torch.cuda.get_device_properties(0)
    receipt = {'version': 1, 'unit': unit_name, 'packet_sha256': fixed.WINDOW_SHA,
               'profile_sha256': fixed.PROFILE_SHA, 'checkpoint_sha256': checkpoint_sha,
               'decoder_manifest_sha256': fixed.file_sha(paths['decoder_manifest']),
               'tensor_fingerprints': config['tensor_fingerprints'],
               'files': {key: item['sha256'] for key, item in config['files'].items()},
               'isolation': isolation, 'device': {'name': device.name, 'major': device.major, 'minor': device.minor},
               'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
               'gpu_memory_limit_mib': memory_mib,
               'on_cuda': on_cuda, 'executed_modules': {name: str(path) for name, path in modules.items()},
               'model_record': renderer.record, 'kernels': kernels, 'original_probe': original_probe,
               'context_probe_cost': probe_cost, 'sidecar': sidecar, 'scope': scope,
               'phase_costs': {'model_load': model_load_cost, 'original_probe': original_probe_cost,
                               'context_probe': context_phase_cost, 'sampling': sampling_cost,
                               'publication': publication_cost},
               'recipe': {'steps': steps, 'cfg': 2, 'context': [24, 4], 'size': [512, 784]}}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True, default=str))


def check_model_record(record: dict[str, Any], checkpoint_sha: str) -> None:
    fit = record.get('fit', {})
    if (record.get('profile') != 'original' or record.get('dtype') != 'float16' or record.get('cfg') != 2
            or fit.get('checkpoint_sha256') != checkpoint_sha or fit.get('rank') != 4
            or fit.get('base', {}).get('sha256') != BASE_SHA or 'billing' in fit
            or fit.get('reference_unet', {}).get('lora_tensors') != 208
            or fit.get('denoising_unet', {}).get('lora_tensors') != 256
            or fit.get('reference_unet', {}).get('lora_layers', 0) <= 0
            or fit.get('denoising_unet', {}).get('lora_layers', 0) <= 0
            or fit.get('pose_guider', {}).get('keys') != 16
            or fit.get('pose_guider', {}).get('missing') != 0):
        raise ValueError('model loading record mismatch')
    for key in ('reference_unet', 'denoising_unet', 'pose_guider'):
        if record.get(key, {}).get('missing') != 0:
            raise ValueError(f'generator model loading incomplete: {key}')


def check_execution_receipt(receipt: dict[str, Any]) -> None:
    device, kernels = receipt.get('device', {}), receipt.get('kernels', {})
    capability = (device.get('major'), device.get('minor'))
    name = device.get('name', '')
    if ((capability == (8, 9) and 'RTX 6000 Ada' not in name)
            or (capability == (8, 6) and 'A6000' not in name)
            or capability not in {(8, 9), (8, 6)}):
        raise ValueError('unqualified GPU class')
    if receipt.get('torch_version') != '2.7.0+cu128' or receipt.get('cuda_version') != '12.8':
        raise ValueError('generator runtime version mismatch')
    if (kernels.get('kernel_launches', 0) <= 0
            or not {'flash', 'mem_efficient'} <= set(kernels.get('attention_family', []))
            or not {'cutlass', 'xmma'}.intersection(kernels.get('tensor_core_like', []))):
        raise ValueError('generator kernel qualification missing')
    if (set(receipt.get('on_cuda', {})) != {'vae', 'image_encoder', 'reference_unet', 'denoising_unet', 'pose_guider'}
            or not all(receipt['on_cuda'].values())):
        raise ValueError('generator modules not all on CUDA')
    if not receipt.get('executed_modules'):
        raise ValueError('executed source identity missing')
    probe = receipt.get('context_probe_cost', {})
    if (probe.get('frames') != 24 or probe.get('steps') != 1
            or probe.get('scope') != 'kernel qualification, not quality evidence'
            or not all(isinstance(probe.get(k), (int, float)) and 0 <= probe[k] < float('inf')
                       for k in ('wall_seconds', 'cpu_seconds', 'gpu_seconds'))):
        raise ValueError('24-frame probe receipt incomplete')


def _validate_receipt(output: Path, expected: dict[str, Any], scope: str, steps: int) -> dict[str, Any]:
    receipt = json.loads((output / 'sealed.json').read_text())
    if receipt.get('scope') != scope or receipt.get('recipe', {}).get('steps') != steps:
        raise ValueError('output scope or step count mismatch')
    unit = expected['unit']
    if unit not in fixed.UNITS or receipt.get('unit') != unit:
        raise ValueError('unit mismatch')
    check_oracle_fingerprints(expected['tensor_fingerprints'])
    if expected['packet_sha256'] != fixed.WINDOW_SHA or expected['profile_sha256'] != fixed.PROFILE_SHA:
        raise ValueError('expected packet or profile differs from CPU oracle')
    if (not expected['files'].get('decoder_manifest')
            or receipt.get('decoder_manifest_sha256') != expected['files']['decoder_manifest']):
        raise ValueError('decoder manifest identity mismatch')
    if expected['files'].get('unit') != ORACLE_UNIT_SHA[unit]:
        raise ValueError('expected unit differs from CPU oracle')
    if (receipt.get('checkpoint_sha256') != expected['files'].get('checkpoint')
            or expected['files'].get('base') != BASE_SHA
            or expected['files'].get('aa_archive') != AA_ARCHIVE_SHA
            or expected['files'].get('code:experiments.tennis.k6b_generate') != GENERATOR_CODE_SHA
            or expected['files'].get('code:experiments.tennis.k6b_fit') != FIT_CODE_SHA
            or any(expected['files'].get(f'weight:{name}') != digest for name, digest in PINNED_WEIGHTS.items())):
        raise ValueError('expected checkpoint or installed assets mismatch')
    for key in ('packet_sha256', 'profile_sha256', 'tensor_fingerprints', 'files'):
        if receipt.get(key) != expected[key]:
            raise ValueError(f'sealed identity mismatch: {key}')
    if receipt.get('recipe') != {'steps': steps, 'cfg': 2, 'context': [24, 4], 'size': [512, 784]}:
        raise ValueError('recipe mismatch')
    isolation = receipt['isolation']
    if (isolation.get('landlock_abi', 0) < 1 or isolation.get('pre_sandbox_threads') != 1
            or isolation.get('extra_fds') is not False or not isolation.get('denials')
            or any(value not in (1, 13) for value in isolation['denials'].values())):
        raise ValueError('isolation receipt incomplete')
    hashes = validate_crops(output, (unit,), steps, scope)
    if receipt.get('sidecar', {}).get('npz_sha256') != hashes[unit]:
        raise ValueError('sealed output mismatch')
    if not receipt.get('device', {}).get('name') or not receipt.get('kernels') or not receipt.get('model_record'):
        raise ValueError('device, kernel or model receipt missing')
    check_model_record(receipt['model_record'], receipt['checkpoint_sha256'])
    check_execution_receipt(receipt)
    costs = receipt.get('phase_costs', {})
    if set(costs) != {'model_load', 'original_probe', 'context_probe', 'sampling', 'publication'} or any(
            not isinstance(costs[phase].get(field), (int, float)) or not 0 <= costs[phase][field] < float('inf')
            for phase in costs for field in ('wall_seconds', 'cpu_seconds', 'gpu_event_seconds',
                                             'gpu_reserved_wall_seconds')):
        raise ValueError('incomplete phase timing receipt')
    return {'unit': unit, 'sha256': hashes[unit], 'sealed_sha256': fixed.file_sha(output / 'sealed.json'),
            'scope': scope, 'steps': steps, 'phase_costs': costs}


def validate_sealed(output: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate only original 30-step candidate output."""
    return _validate_receipt(output, expected, 'candidate', 30)


def validate_cost_probe(output: Path, expected: dict[str, Any], steps: int) -> dict[str, Any]:
    """Validate coverage and cost only; no output here is a quality candidate."""
    if type(steps) is not int or steps not in (1, 3, 6):
        raise ValueError('cost probe steps must be exactly 1, 3 or 6')
    return _validate_receipt(output, expected, 'qualification_only', steps)


def wait_owned_worker(process: Any, timeout_seconds: int) -> tuple[int, bool]:
    """Reap only the direct reviewed unshare worker, preserving fleet's group."""
    import subprocess

    try:
        return int(process.wait(timeout=timeout_seconds)), False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return int(process.wait(timeout=5)), True
        except subprocess.TimeoutExpired:
            process.kill()
            return int(process.wait(timeout=5)), True


def launch(config_path: Path, config_sha256: str, timeout_seconds: int,
           launcher_cpu_seconds: int, receipt_dir: Path) -> None:
    """Start one bounded worker in fresh user and network namespaces on Linux."""
    import os
    import resource
    import subprocess
    import sys
    import time

    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one-CPU fleet affinity required')
    if not 1 <= timeout_seconds <= 2700:
        raise ValueError('worker deadline must be at most 45 minutes')
    if not 1 <= launcher_cpu_seconds <= 60:
        raise ValueError('bounded launcher CPU limit required')
    if not receipt_dir.is_dir() or any(receipt_dir.iterdir()):
        raise ValueError('fresh empty launcher receipt directory required')
    resource.setrlimit(resource.RLIMIT_CPU, (launcher_cpu_seconds, 3600))
    raw = config_path.read_bytes()
    if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != config_sha256:
        raise ValueError('config identity mismatch')
    config = json.loads(raw)
    if (config['parent_user_namespace'] != os.readlink('/proc/self/ns/user')
            or config['parent_network_namespace'] != os.readlink('/proc/self/ns/net')):
        raise ValueError('parent namespaces do not match current launcher')
    command = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', sys.executable,
               '-m', 'experiments.tennis.w8_receiver_bootstrap', '--config', str(config_path),
               '--config-sha256', config_sha256, '--cpu-seconds', str(config['cpu_seconds'])]
    env = {'PYTHONPATH': str(Path(__file__).resolve().parents[2]), 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    worker_cpu = config['cpu_seconds']
    if type(worker_cpu) is not int or not 1 <= worker_cpu <= 3600:
        raise ValueError('bounded worker CPU limit required')
    def limit_worker() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (worker_cpu, 3600))
    before_parent = resource.getrusage(resource.RUSAGE_SELF)
    before_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True, env=env,
                               preexec_fn=limit_worker)
    pgid = os.getpgrp()  # unshare inherits the fleet-owned group; it is not forked with --fork
    record = {'pid': process.pid, 'pgid': pgid, 'launcher_pid': os.getpid(),
              'started_monotonic': started, 'timeout_seconds': timeout_seconds,
              'launcher_cpu_limit': launcher_cpu_seconds, 'worker_cpu_limit': worker_cpu}
    (receipt_dir / 'start.json').write_text(json.dumps(record, sort_keys=True))
    status, timed_out = wait_owned_worker(process, timeout_seconds)
    after_parent = resource.getrusage(resource.RUSAGE_SELF)
    after_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update({'exit_status': status, 'timed_out': timed_out, 'elapsed_seconds': time.monotonic() - started,
                   'launcher_cpu_seconds': after_parent.ru_utime + after_parent.ru_stime,
                   'launcher_cpu_before_spawn_seconds': before_parent.ru_utime + before_parent.ru_stime,
                   'worker_and_unshare_cpu_seconds': (after_child.ru_utime + after_child.ru_stime)
                                                      - (before_child.ru_utime + before_child.ru_stime)})
    (receipt_dir / 'exit.json').write_text(json.dumps(record, sort_keys=True))
    if timed_out or status:
        raise RuntimeError('namespace worker failed or timed out; receipt retained')



def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--launcher-cpu-seconds', type=int, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    args = parser.parse_args()
    launch(args.config, args.config_sha256, args.timeout_seconds, args.launcher_cpu_seconds, args.receipt_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
