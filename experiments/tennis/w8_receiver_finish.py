"""W8 receiver finishing stages and post-seal comparison.

GPU stages require the separately reviewed namespace/Landlock bootstrap and
must not read historical or source inputs. Native decoder execution requires
its own executable-grant launcher; this module only specifies its argv.
"""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any

import numpy as np

from experiments.tennis import w8_input_job as fixed

TICKS = np.arange(0, 240, 3, dtype=np.int64)
IVF_SHA = 'f9622b6af4cddea3d70226c0b22e12374f4c77aadf2120e54d5d21ba55bfbbb8'
FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
SAM_SHA = '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
METRICS = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists')


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _isolation(isolation: dict[str, Any]) -> None:
    namespaces = isolation.get('namespaces', {})
    if (isolation.get('landlock_abi', 0) < 1 or isolation.get('pre_sandbox_threads') != 1
            or isolation.get('extra_fds') is not False or isolation.get('network_interfaces') != ['lo']
            or not namespaces.get('user') or not namespaces.get('net')
            or not isolation.get('denials')
            or any(v not in (1, 13) for v in isolation['denials'].values())):
        raise ValueError('isolation evidence incomplete')


def _inputs(config: dict[str, Any], names: set[str]) -> dict[str, Path]:
    files = config['files']
    if set(files) != names:
        raise ValueError('unexpected worker inputs')
    paths = {}
    for name, row in files.items():
        path = Path(row['path'])
        if not path.is_absolute() or fixed.file_sha(path) != row['sha256']:
            raise ValueError(f'input identity mismatch: {name}')
        paths[name] = path
    return paths


def _new_output(config: dict[str, Any]) -> Path:
    output = Path(config['output'])
    if not output.is_absolute() or not output.is_dir() or any(output.iterdir()):
        raise ValueError('fresh empty absolute output directory required')
    return output


def _packet(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, bytes]]:
    from experiments.tennis.w8_payload import unpack_window

    profile = fixed.pinned_profile(paths['profile'])
    blob = paths['packet'].read_bytes()
    if fixed.digest(blob) != fixed.WINDOW_SHA:
        raise ValueError('packet identity')
    code_root = Path(__file__).resolve().parent
    for key, filename in (('geometry_code', 'k6b.py'), ('packet_code', 'k6b_rate_streams.py'),
                          ('receiver_code', 'k6b_receiver_sam.py'), ('payload_decoder_code', 'w8_payload.py')):
        if fixed.file_sha(code_root / filename) != profile['models'][key]:
            raise ValueError(f'packet-declared code mismatch: {key}')
    sections, _ = unpack_window(blob, profile)
    if profile['window'] != fixed.WINDOW or sections[profile['judged']] != fixed.judged_bitmap(fixed.RECIPE):
        raise ValueError('packet ticks or window')
    if fixed.digest(sections[profile['kps']]) != fixed.PACKET_SHAS['keypoints.kps']:
        raise ValueError('packet keypoint identity')
    if fixed.digest(sections[profile['ivf']]) != IVF_SHA:
        raise ValueError('packet IVF identity')
    return profile, sections


def _generation_seals(paths: dict[str, Path]) -> None:
    from experiments.tennis import w8_receiver_generate as generation

    for unit in fixed.UNITS:
        receipt = json.loads(paths[f'generation_seal:{unit}'].read_text())
        if (receipt.get('version') != 1 or receipt.get('unit') != unit
                or receipt.get('packet_sha256') != fixed.WINDOW_SHA
                or receipt.get('profile_sha256') != fixed.PROFILE_SHA
                or receipt.get('scope') != 'candidate'
                or receipt.get('sidecar', {}).get('scope') != 'candidate'
                or receipt.get('sidecar', {}).get('npz_sha256') != fixed.file_sha(paths[f'crop:{unit}'])
                or receipt.get('recipe') != {'steps': 30, 'cfg': 2, 'context': [24, 4], 'size': [512, 784]}):
            raise ValueError('generation seal mismatch')
        generation.check_oracle_fingerprints(receipt['tensor_fingerprints'])
        gen_files = receipt['files']
        if (gen_files.get('unit') != generation.ORACLE_UNIT_SHA[unit]
                or gen_files.get('base') != generation.BASE_SHA
                or gen_files.get('checkpoint') != receipt.get('checkpoint_sha256')
                or gen_files.get('decoder_manifest') != receipt.get('decoder_manifest_sha256')
                or not gen_files.get('decoder_manifest')
                or any(gen_files.get(f'weight:{name}') != sha for name, sha in generation.PINNED_WEIGHTS.items())):
            raise ValueError('generation asset identities mismatch')
        generation.check_model_record(receipt['model_record'], receipt['checkpoint_sha256'])
        generation.check_execution_receipt(receipt)
        _isolation(receipt['isolation'])


def native_argv(binary: Path, ivf: Path, output: Path) -> list[str]:
    """Build native decode argv; execution needs a separate reviewed launcher."""
    if not all(path.is_absolute() for path in (binary, ivf, output)):
        raise ValueError('absolute native paths required')
    if output.exists():
        raise ValueError('native output already exists')
    if fixed.file_sha(binary) != FFMPEG_SHA or fixed.file_sha(ivf) != IVF_SHA:
        raise ValueError('native binary or IVF mismatch')
    return [str(binary), '-nostdin', '-v', 'error', '-threads', '1', '-c:v', 'libdav1d',
            '-i', str(ivf), '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-n', str(output)]


def run_conversion_isolated(config: dict[str, Any], isolation: dict[str, Any]) -> None:
    """Convert native YUV into 80 RGB bases without invoking a tool."""
    _isolation(isolation)
    paths = _inputs(config, {'packet', 'profile', 'yuv'})
    output = _new_output(config)
    profile, _ = _packet(paths)
    native = config['native_receipt']
    if (native.get('binary_sha256') != FFMPEG_SHA or native.get('ivf_sha256') != IVF_SHA
            or native.get('yuv_sha256') != '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff'
            or native.get('yuv_sha256') != fixed.file_sha(paths['yuv'])
            or native.get('exit_code') != 0 or native.get('decoder') != 'libdav1d'
            or native.get('landlock_abi', 0) < 1 or not native.get('denials')
            or any(value not in (1, 13) for value in native.get('denials', {}).values())
            or not native.get('loader_sha256')
            or not any('libdav1d' in path for path in native.get('libraries', {}))
            or native.get('yuv_bytes') != 240 * 480 * 272 * 3 // 2
            or paths['yuv'].stat().st_size != 240 * 480 * 272 * 3 // 2):
        raise ValueError('native decoder receipt mismatch')
    import cv2
    import torch

    from src.codecs.quality import read_frames, yuv420_to_rgb

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('original CUDA color conversion required')
    target = np.lib.format.open_memmap(output / 'base.npy', mode='w+', dtype=np.uint8,
                                       shape=(80, 1080, 1920, 3))
    hashes = []
    saved = 0
    for first in range(0, 240, 8):
        planes = torch.from_numpy(read_frames(paths['yuv'], 480, 272, first, min(8, 240-first))).to('cuda')
        rgb = yuv420_to_rgb(planes, 480, 272, full_range=False)
        for tick in range(first, min(240, first + 8)):
            if tick % 3:
                continue
            frame = rgb[tick-first].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            target[saved] = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            hashes.append(_array_sha(target[saved]))
            saved += 1
    if saved != 80:
        raise ValueError('RGB conversion tick coverage')
    target.flush()
    del target
    receipt = {'version': 1, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
               'input_sha256': {key: value['sha256'] for key, value in config['files'].items()},
               'native': native, 'base_sha256': fixed.file_sha(output / 'base.npy'),
               'base_frame_sha256': hashes, 'ticks': TICKS.tolist(), 'isolation': isolation,
               'color_path': profile['recipe']['color_path'], 'torch': torch.__version__,
               'opencv': cv2.__version__, 'device': torch.cuda.get_device_name(0)}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True))


def _deny_sam_bytecode(event: str, args: tuple[Any, ...]) -> None:
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path.endswith(('.pyc', '.pyo')):
            raise PermissionError('unverified bytecode denied')


def _sam_module_closure(verified: dict[str, str]) -> dict[str, str]:
    modules = {}
    for name, module in sys.modules.items():
        if name != 'sam3' and not name.startswith('sam3.'):
            continue
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve(strict=True)
        if path.suffix != '.py' or verified.get(str(path)) != fixed.file_sha(path):
            raise ValueError(f'unverified SAM module: {name}')
        modules[name] = str(path)
    if not modules:
        raise ValueError('SAM source closure missing')
    return modules


def run_sam_isolated(config: dict[str, Any], isolation: dict[str, Any]) -> None:
    """Predict one fresh crop's mask per tick using independent SAM resets."""
    _isolation(isolation)
    names = {'packet', 'profile', 'weights', 'text', 'sam_record', *(f'crop:{u}' for u in fixed.UNITS),
             *(f'generation_seal:{u}' for u in fixed.UNITS)}
    paths = _inputs(config, names)
    _generation_seals(paths)
    if fixed.file_sha(paths['sam_record']) != config['protected_receipt']['sha256']:
        raise ValueError('SAM distribution preflight identity mismatch')
    distribution = json.loads(paths['sam_record'].read_text())['sam_distribution']
    verified = distribution['verified_files']
    if not isinstance(verified, dict) or not verified:
        raise ValueError('SAM verified source mapping missing')
    sys.addaudithook(_deny_sam_bytecode)
    output = _new_output(config)
    profile, sections = _packet(paths)
    if fixed.file_sha(paths['weights']) != SAM_SHA or fixed.file_sha(paths['text']) != SAM_SHA:
        raise ValueError('SAM checkpoint identity')
    from PIL import Image
    import torch

    from experiments.tennis.k6b import to_crop
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints
    from experiments.tennis.k6b_receiver_sam import pose_prompt
    from src.segmentation.sam31_tracker import Sam31MaskTracker, runtime

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tracker = Sam31MaskTracker(paths['weights'])
    _sam_module_closure(verified)
    if (tracker.load_report.get('sdpa_backend_policy') != 'native_flash'
            or tracker.load_report.get('missing_keys') or tracker.load_report.get('unexpected_keys')):
        raise ValueError('SAM loading or attention policy mismatch')
    device = torch.cuda.get_device_properties(0)
    if (device.major, device.minor) not in ((8, 6), (8, 9)):
        raise ValueError('unqualified SAM GPU class')
    model = tracker.model
    if not next(model.parameters()).is_cuda:
        raise ValueError('SAM model is not on CUDA')
    meta, units = decode_keypoints(sections[profile['kps']])
    if meta['window'] != fixed.WINDOW or {u['name'] for u in units} != set(fixed.UNITS):
        raise ValueError('SAM unit binding')
    rows: list[dict[str, Any]] = []
    kernels: list[str] = []
    for unit, boxes in zip(units, box_maps(units)):
        name = unit['name']
        keypoints = dict(zip(unit['t'].tolist(), unit['kp']))
        with np.load(paths[f'crop:{name}'], allow_pickle=False) as data:
            ticks, crops = data['t'], data['crops']
            if not np.array_equal(ticks, TICKS) or crops.shape != (80, 784, 512, 3) or crops.dtype != np.uint8:
                raise ValueError('fresh crop coverage')
            masks, prompts = [], []
            for j, tick in enumerate(TICKS):
                t = int(tick)
                image = torch.from_numpy(np.array(Image.fromarray(crops[j]).resize(
                    (int(model.image_size), int(model.image_size)))).transpose(2, 0, 1).copy())
                image = image.to(dtype=torch.float16).div_(255).sub_(0.5).div_(0.5)[None]
                points, labels = pose_prompt(to_crop(keypoints[t], boxes[t]))
                profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) if not rows and j == 0 else None)
                with profiler if profiler is not None else nullcontext():
                  with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    state = model.init_state(video_height=784, video_width=512, num_frames=1, offload_video_to_cpu=True)
                    state['images'] = image
                    frame, ids, _low, video = model.add_new_points(
                        inference_state=state, frame_idx=0, obj_id=1, points=torch.from_numpy(points),
                        labels=torch.from_numpy(labels), clear_old_points=True, rel_coordinates=True)
                    if int(frame) != 0 or list(ids) != [1]:
                        raise ValueError('SAM independent-frame identity')
                    mask = (video[0, 0] > 0).cpu().numpy()
                if profiler is not None:
                    kernels = sorted({event.key for event in profiler.key_averages() if 'flash' in event.key.lower()})
                    if not kernels:
                        raise RuntimeError('SAM flash attention not observed')
                if mask.shape != (784, 512) or mask.dtype != bool:
                    raise ValueError('SAM output dimensions')
                masks.append(mask)
                prompts.append({'t': t, 'points': points.tolist(), 'labels': labels.tolist()})
                del state, image
        target = output / f'{name}-masks.npz'
        a = np.stack(masks)
        np.savez_compressed(target, t=TICKS, masks=a)
        rows.append({'unit': name, 'file_sha256': fixed.file_sha(target), 'array_sha256': _array_sha(a),
                     'frames': [_array_sha(m) for m in a], 'prompts': prompts})
    sam_modules = _sam_module_closure(verified)
    receipt = {'version': 1, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
               'sam_source_modules': sam_modules, 'sam_record_sha256': fixed.file_sha(paths['sam_record']),
               'files': {k: v['sha256'] for k, v in config['files'].items()}, 'rows': rows,
               'independent_frames': True, 'isolation': isolation, 'load': tracker.load_report,
               'runtime': runtime(), 'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'flash_events': kernels,
               'cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2])}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True, default=str))


def run_compose_isolated(config: dict[str, Any], isolation: dict[str, Any]) -> None:
    """Compose only newly generated crops and predicted masks over the paid base."""
    _isolation(isolation)
    names = {'packet', 'profile', 'base', *(f'crop:{u}' for u in fixed.UNITS),
             *(f'mask:{u}' for u in fixed.UNITS), *(f'generation_seal:{u}' for u in fixed.UNITS)}
    paths = _inputs(config, names)
    _generation_seals(paths)
    output = _new_output(config)
    profile, sections = _packet(paths)
    from experiments.tennis.k6b import unit_name
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints
    from experiments.tennis.k6b_real_base_compose import compose
    from experiments.tennis.k6b_receiver_sam import project

    meta, units = decode_keypoints(sections[profile['kps']])
    order = meta['compositing_order']
    if (meta['window'] != fixed.WINDOW or len(order) != 2
            or {unit_name(fixed.WINDOW, p) for p in order} != set(fixed.UNITS)
            or {u['name'] for u in units} != set(fixed.UNITS)):
        raise ValueError('composition unit binding')
    boxes = {unit['name']: box for unit, box in zip(units, box_maps(units))}
    base = np.load(paths['base'], mmap_mode='r', allow_pickle=False)
    if base.shape != (80, 1080, 1920, 3) or base.dtype != np.uint8:
        raise ValueError('base coverage')
    crops, masks = {}, {}
    for name in fixed.UNITS:
        with np.load(paths[f'crop:{name}'], allow_pickle=False) as archive:
            if not np.array_equal(archive['t'], TICKS):
                raise ValueError('crop ticks')
            crops[name] = archive['crops']
            if crops[name].shape != (80, 784, 512, 3) or crops[name].dtype != np.uint8:
                raise ValueError('crop dimensions')
        with np.load(paths[f'mask:{name}'], allow_pickle=False) as archive:
            if not np.array_equal(archive['t'], TICKS):
                raise ValueError('mask ticks')
            masks[name] = archive['masks']
            if masks[name].shape != (80, 784, 512) or masks[name].dtype != bool:
                raise ValueError('mask dimensions')
    target = np.lib.format.open_memmap(output / 'composite.npy', mode='w+', dtype=np.uint8,
                                       shape=(80, 1080, 1920, 3))
    hashes = []
    for i, tick in enumerate(TICKS):
        cs, bs, ms = {}, {}, {}
        for player in order:
            unit = unit_name(fixed.WINDOW, player)
            box = boxes[unit][int(tick)]
            cs[player], bs[player], ms[player] = crops[unit][i], box, project(masks[unit][i], box)
        image, _ = compose(np.asarray(base[i]), cs, bs, ms, order)
        target[i] = image
        hashes.append(_array_sha(image))
    target.flush()
    del target
    receipt = {'version': 1, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
               'files': {k: v['sha256'] for k, v in config['files'].items()}, 'ticks': TICKS.tolist(),
               'composite_sha256': fixed.file_sha(output / 'composite.npy'),
               'frame_sha256': hashes, 'order': order, 'isolation': isolation}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True))


def _compare_rgb(actual: np.ndarray, expected: np.ndarray, label: str) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != np.uint8 or expected.dtype != np.uint8:
        raise ValueError(f'{label} shape or dtype')
    if actual.ndim != 4 or actual.shape[0] != len(TICKS):
        raise ValueError(f'{label} coverage')
    differences = [int(np.max(np.abs(actual[i].astype(np.int16) - expected[i].astype(np.int16))))
                   for i in range(len(TICKS))]
    if max(differences) > 1:
        raise ValueError(f'{label} exceeds one channel level')
    return {'exact': max(differences) == 0, 'max_per_frame': differences}


def _compare_masks(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if (actual.shape != (80, 784, 512) or expected.shape != (80, 784, 512)
            or actual.dtype != bool or expected.dtype != bool or not np.array_equal(actual, expected)):
        raise ValueError('predicted mask mismatch')
    return {'exact': True, 'sha256': _array_sha(actual)}


def _compare_scores(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    if set(actual) != set(METRICS) or set(expected) != set(METRICS):
        raise ValueError('four score families required')
    for name in METRICS:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        if a.shape != (80,) or b.shape != (80,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'invalid score coverage: {name}')
        per_tick = np.abs(a.astype(np.float64) - b.astype(np.float64))
        mean = float(abs(a.mean(dtype=np.float64) - b.mean(dtype=np.float64)))
        if float(per_tick.max()) > 0.001 or mean > 0.001:
            raise ValueError(f'score tolerance exceeded: {name}')
        result[name] = {'max_tick_difference': float(per_tick.max()), 'mean_difference': mean}
    return result


CANDIDATE_KEYS = frozenset({'base', 'base_receipt', 'composite', 'compose_receipt', 'sam_receipt',
                            *(f'crop:{u}' for u in fixed.UNITS), *(f'mask:{u}' for u in fixed.UNITS),
                            *(f'generation_seal:{u}' for u in fixed.UNITS)})


def validate_candidate_seals(paths: dict[str, Path]) -> None:
    """Check fresh receiver provenance before granting access to scoring inputs."""
    if set(paths) != CANDIDATE_KEYS:
        raise ValueError('candidate seal file set')
    _generation_seals(paths)
    base = json.loads(paths['base_receipt'].read_text())
    composed = json.loads(paths['compose_receipt'].read_text())
    sam = json.loads(paths['sam_receipt'].read_text())
    for receipt in (base, composed, sam):
        if receipt.get('packet_sha256') != fixed.WINDOW_SHA or receipt.get('profile_sha256') != fixed.PROFILE_SHA:
            raise ValueError('candidate packet or profile seal mismatch')
        _isolation(receipt['isolation'])
    if (base.get('base_sha256') != fixed.file_sha(paths['base']) or base.get('ticks') != TICKS.tolist()
            or composed.get('composite_sha256') != fixed.file_sha(paths['composite'])
            or composed.get('ticks') != TICKS.tolist()
            or composed.get('files', {}).get('base') != fixed.file_sha(paths['base'])
            or not sam.get('independent_frames') or not sam.get('flash_events')):
        raise ValueError('candidate base, SAM or composition seal mismatch')
    rows = {row['unit']: row for row in sam['rows']}
    if set(rows) != set(fixed.UNITS) or len(sam['rows']) != len(fixed.UNITS):
        raise ValueError('SAM unit coverage')
    for unit in fixed.UNITS:
        for key in (f'crop:{unit}', f'generation_seal:{unit}'):
            actual = fixed.file_sha(paths[key])
            if composed['files'].get(key) != actual or sam['files'].get(key) != actual:
                raise ValueError('candidate generation seal mismatch')
        actual = fixed.file_sha(paths[f'mask:{unit}'])
        if composed['files'].get(f'mask:{unit}') != actual or rows[unit]['file_sha256'] != actual:
            raise ValueError('candidate mask seal mismatch')


def validate_offline(config: dict[str, Any]) -> dict[str, Any]:
    """Compare sealed outputs with independently qualified goldens after isolation ends.

    Caller supplies a reviewed immutable input manifest with all hashes, including
    the candidate score receipt and original judge archive-derived scores. This
    validates scores; it does not run or qualify the metric networks.
    """
    names = {'base', 'base_receipt', 'golden_base', 'composite', 'golden_composite', 'scores', 'golden_scores',
             'compose_receipt', 'sam_receipt', 'score_receipt',
             *(f'crop:{u}' for u in fixed.UNITS), *(f'golden_crop:{u}' for u in fixed.UNITS),
             *(f'mask:{u}' for u in fixed.UNITS), *(f'golden_mask:{u}' for u in fixed.UNITS),
             *(f'generation_seal:{u}' for u in fixed.UNITS)}
    if set(config['files']) != names:
        raise ValueError('unexpected validator inputs')
    golden = {'golden_base', 'golden_composite', 'golden_scores',
              *(f'golden_crop:{u}' for u in fixed.UNITS), *(f'golden_mask:{u}' for u in fixed.UNITS)}
    candidate_config = {'files': {k: v for k, v in config['files'].items() if k not in golden}}
    paths = _inputs(candidate_config, names - golden)
    validate_candidate_seals({k: paths[k] for k in CANDIDATE_KEYS})
    compose_receipt = json.loads(paths['compose_receipt'].read_text())
    sam_receipt = json.loads(paths['sam_receipt'].read_text())
    score_receipt = json.loads(paths['score_receipt'].read_text())
    if (compose_receipt.get('composite_sha256') != fixed.file_sha(paths['composite'])
            or compose_receipt.get('ticks') != TICKS.tolist()
            or compose_receipt.get('packet_sha256') != fixed.WINDOW_SHA
            or compose_receipt.get('files', {}).get('base') != fixed.file_sha(paths['base'])):
        raise ValueError('composite seal mismatch')
    _isolation(compose_receipt['isolation'])
    _isolation(sam_receipt['isolation'])
    if (not sam_receipt.get('independent_frames') or sam_receipt.get('packet_sha256') != fixed.WINDOW_SHA
            or score_receipt.get('passed') is not True
            or score_receipt.get('composite_sha256') != fixed.file_sha(paths['composite'])
            or score_receipt.get('scores_sha256') != fixed.file_sha(paths['scores'])
            or score_receipt.get('ticks') != TICKS.tolist()
            or not config.get('scorer_identity')
            or score_receipt.get('scorer_identity') != config['scorer_identity']):
        raise ValueError('SAM or score seal mismatch')
    if not sam_receipt.get('flash_events'):
        raise ValueError('SAM flash attention evidence missing')
    rows = {row['unit']: row for row in sam_receipt['rows']}
    if set(rows) != set(fixed.UNITS) or len(sam_receipt['rows']) != len(fixed.UNITS):
        raise ValueError('SAM unit coverage')
    for name in fixed.UNITS:
        if (compose_receipt['files'].get(f'generation_seal:{name}') != fixed.file_sha(paths[f'generation_seal:{name}'])
                or sam_receipt['files'].get(f'generation_seal:{name}') != fixed.file_sha(paths[f'generation_seal:{name}'])
                or compose_receipt['files'].get(f'crop:{name}') != fixed.file_sha(paths[f'crop:{name}'])
                or compose_receipt['files'].get(f'mask:{name}') != fixed.file_sha(paths[f'mask:{name}'])
                or sam_receipt['files'].get(f'crop:{name}') != fixed.file_sha(paths[f'crop:{name}'])
                or rows[name]['file_sha256'] != fixed.file_sha(paths[f'mask:{name}'])):
            raise ValueError('fresh crop or mask seal mismatch')
    paths.update(_inputs({'files': {k: v for k, v in config['files'].items() if k in golden}}, golden))
    report = {}
    for name in ('base', 'composite'):
        report[name] = _compare_rgb(np.load(paths[name], mmap_mode='r', allow_pickle=False),
                                    np.load(paths['golden_' + name], mmap_mode='r', allow_pickle=False), name)
    for name in fixed.UNITS:
        with np.load(paths[f'crop:{name}'], allow_pickle=False) as a, np.load(paths[f'golden_crop:{name}'], allow_pickle=False) as b:
            if not np.array_equal(a['t'], TICKS) or not np.array_equal(b['t'], TICKS):
                raise ValueError('crop tick mismatch')
            report[f'crop:{name}'] = _compare_rgb(a['crops'], b['crops'], name)
        with np.load(paths[f'mask:{name}'], allow_pickle=False) as a, np.load(paths[f'golden_mask:{name}'], allow_pickle=False) as b:
            if not np.array_equal(a['t'], TICKS) or not np.array_equal(b['t'], TICKS):
                raise ValueError('predicted mask ticks')
            report[f'mask:{name}'] = _compare_masks(a['masks'], b['masks'])
    with np.load(paths['scores'], allow_pickle=False) as a, np.load(paths['golden_scores'], allow_pickle=False) as b:
        if not np.array_equal(a['ticks'], TICKS) or not np.array_equal(b['ticks'], TICKS):
            raise ValueError('score ticks')
        report['metrics'] = _compare_scores({k: a[k] for k in METRICS}, {k: b[k][3] for k in METRICS})
    return {'passed': True, 'results': report, 'input_sha256': {k: v['sha256'] for k, v in config['files'].items()},
            'scope': 'fixed 80 ticks only; scoring execution and host qualification require separate evidence'}
