"""Pinned GVC-RT qualification bridge. CUDA modes must run inside fleet dispatch.

The decoder receives only a charged manifest, persisted native stream and verified
models. A Python audit hook denies selected source paths; it is not an OS sandbox.
"""
import argparse
import hashlib
import io
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = 'd0e32bfa3e8e282f9a77437c223c605858eea637'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path, expected):
    if len(expected) != 64 or sha256(path) != expected.lower():
        raise ValueError(f'File SHA256 mismatch: {path}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def bind_source_namespace(path):
    """Keep the verified GVC namespace distinct from PointStream's regular src."""
    source = (Path(path).resolve() / 'src').resolve(strict=True)
    for name, module in list(sys.modules.items()):
        if name != 'src' and not name.startswith('src.'):
            continue
        origins = list(getattr(module, '__path__', []))
        if getattr(module, '__file__', None):
            origins.append(module.__file__)
        if not origins or any((resolved := Path(origin).resolve()) != source and source not in resolved.parents
                              for origin in origins):
            raise RuntimeError(f'Foreign src module already loaded: {name}; use a fresh worker')
    spec = importlib.machinery.ModuleSpec('src', loader=None, is_package=True)
    spec.submodule_search_locations = [str(source)]
    namespace = importlib.util.module_from_spec(spec)
    sys.modules['src'] = namespace


def source_pin(path):
    path = Path(path).resolve()
    export = path / '.gvcrt-source.json'
    if export.is_file():
        receipt = json.loads(export.read_text())
        if receipt['revision'] != PIN:
            raise ValueError('Unexpected GVC export revision')
        for relative, expected in receipt['files'].items():
            target = (path / relative).resolve()
            if path not in target.parents:
                raise ValueError('Export path escapes source root')
            checked_file(target, expected)
        actual_python = {str(file.relative_to(path)) for file in path.rglob('*.py')}
        expected_python = {name for name in receipt['files'] if name.endswith('.py')}
        if actual_python != expected_python:
            raise ValueError('Export has undeclared Python source')
    else:
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != PIN:
            raise ValueError('Unexpected GVC source revision')
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('GVC tracked source tree is dirty')
    sys.path.insert(0, str(path))
    bind_source_namespace(path)


def checked_state(model, checkpoint, role):
    """Require every model state tensor, with no silent shape/key filtering."""
    choices = ('student', 'ema_shadow', 'state_dict') if role == 'I' else ('student_ema', 'student', 'state_dict')
    selected = next((key for key in choices if key in checkpoint), None)
    state = checkpoint[selected] if selected is not None else checkpoint
    normalized = {}
    for key, value in state.items():
        clean = key.replace('module.', '') if role == 'I' else key
        if clean in normalized:
            raise ValueError(f'Duplicate normalized checkpoint key: {clean}')
        normalized[clean] = value
    target = model.state_dict()
    missing = sorted(set(target) - set(normalized))
    unexpected = sorted(set(normalized) - set(target))
    mismatch = [key for key in target if key in normalized and
                getattr(normalized[key], 'shape', None) != target[key].shape]
    if missing or unexpected or mismatch:
        raise ValueError(json.dumps({'role': role, 'selected_dictionary': selected,
                                    'missing': missing, 'unexpected': unexpected, 'shape_mismatch': mismatch}))
    model.load_state_dict(normalized, strict=True)
    return {'role': role, 'selected_dictionary': selected or 'raw',
            'state_tensors': len(target), 'parameter_elements': sum(v.numel() for v in model.parameters()),
            'strict_complete_coverage': True}


def models(args, device):
    source_pin(args.official_repo)
    checked_file(args.i_checkpoint, args.i_sha256)
    checked_file(args.p_checkpoint, args.p_sha256)
    if args.disable_fused:
        sys.modules['inference_extensions_cuda'] = None
    import torch
    from src.utils.common import set_torch_env
    set_torch_env()
    from src.models.image_model_gvcrt import DMCI
    from src.models.video_model_gvcrt import DMC
    import MLCodec_extensions_cpp as entropy
    from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
    i_model, p_model = DMCI(), DMC()
    coverage = []
    for role, model, path in [('I', i_model, args.i_checkpoint), ('P', p_model, args.p_checkpoint)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        coverage.append(checked_state(model, checkpoint, role))
    environment = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'entropy_extension': str(Path(entropy.__file__).resolve()),
                   'entropy_sha256': sha256(entropy.__file__), 'python': sys.version,
                   'source_revision': PIN, 'i_sha256': args.i_sha256, 'p_sha256': args.p_sha256,
                   'command_argv': sys.argv, 'bridge_sha256': sha256(__file__),
                   'numpy': importlib.metadata.version('numpy'), 'Pillow': importlib.metadata.version('Pillow'),
                   'einops': importlib.metadata.version('einops'),
                   'customized_cuda_inference': CUSTOMIZED_CUDA_INFERENCE, 'disable_fused': args.disable_fused,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'torch_threads': torch.get_num_threads(),
                   'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   'force_zero_thres': None}
    if device == 'cuda':
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('CUDA mode requires fleet-provided CUDA_VISIBLE_DEVICES')
        environment['cuda_visible_devices'] = os.environ['CUDA_VISIBLE_DEVICES']
        environment['device_name'] = torch.cuda.get_device_name(0)
        for model in (i_model, p_model):
            model.to('cuda:0').eval()
            model.update(None)
            model.half()
        p_model.set_curr_poc(0)
    return torch, i_model, p_model, {'coverage': coverage, 'environment': environment}


def tensor_pixels(torch, tensor, width, height):
    return ((tensor[:, :, :height, :width].clamp(-1, 1) + 1) / 2).float().cpu().numpy()[0]


def configure_entropy(i_model, p_model, canvas_h, canvas_w):
    two = canvas_h * canvas_w > 1280 * 720
    i_model.set_use_two_entropy_coders(two)
    p_model.set_use_two_entropy_coders(two)
    return int(two)


def compress_frame(i_model, p_model, x, qp, intra, reset, last_qp):
    if intra:
        result = i_model.compress(x, qp)
        p_model.clear_dpb()
        p_model.add_ref_frame(None, result['x_hat'])
    else:
        if reset:
            p_model.prepare_feature_adaptor_i(last_qp)
        result = p_model.compress(x, qp)
    if not isinstance(result['bit_stream'], (bytes, bytearray, memoryview)):
        raise ValueError('Native compression must return physical coded bytes')
    return result


def encode(args):
    import numpy as np
    from PIL import Image
    torch, i_model, p_model, qualification = models(args, 'cuda')
    from src.layers.cuda_inference import replicate_pad
    from src.utils.stream_helper import SPSHelper, write_sps, write_ip
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames = json.loads(Path(args.frames_json).read_text())
    # Registered input list contains exact paths and hashes, not a directory glob.
    if len(frames) != args.frames:
        raise ValueError('Registered frame count mismatch')
    canvas_h, canvas_w = max(1088, args.height), max(1920, args.width)
    if args.qp < 0 or args.qp >= i_model.get_qp_num():
        raise ValueError('Unsupported registered base QP')
    ec_part = configure_entropy(i_model, p_model, canvas_h, canvas_w)
    payload, sps_helper, placements = io.BytesIO(), SPSHelper(), []
    last_qp = 0
    encode_seconds = []
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    with torch.no_grad():
        for index, record in enumerate(frames):
            checked_file(record['path'], record['sha256'])
            with Image.open(record['path']) as image:
                if image.mode != 'RGB' or image.size != (args.width, args.height):
                    raise ValueError('Input RGB geometry differs from registration')
                pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
            x = torch.from_numpy(pixels.copy()).unsqueeze(0).to('cuda:0').half()
            x = replicate_pad(x, canvas_h - args.height, canvas_w - args.width) * 2 - 1
            reset = index > 0 and args.reset_interval > 0 and index % args.reset_interval == 1
            intra = index == 0
            qp = args.qp if intra else p_model.shift_qp(args.qp, index_map[index % 8])
            if qp < 0 or qp >= p_model.q_scale_enc.shape[0]:
                raise ValueError('Shifted P QP outside model table')
            torch.cuda.synchronize()
            frame_start = time.perf_counter()
            result = compress_frame(i_model, p_model, x, qp, intra, reset, last_qp)
            if not intra:
                last_qp = qp
            sps = {'sps_id': -1, 'height': canvas_h, 'width': canvas_w,
                   'ec_part': ec_part, 'use_ada_i': int(reset)}
            sps_id, new_sps = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            before = payload.tell()
            if new_sps:
                write_sps(payload, sps)
            write_ip(payload, intra, sps_id, qp, result['bit_stream'])
            torch.cuda.synchronize()
            encode_seconds.append(time.perf_counter() - frame_start)
            placements.append({'index': index, 'I': intra, 'reset': reset, 'qp': qp,
                               'offset': before, 'bytes': payload.tell() - before})
    stream = output / 'stream.bin'
    stream.write_bytes(payload.getvalue())
    manifest = {'schema': 'gvcrt-native-v1', 'source_revision': PIN,
                'stream': {'filename': 'stream.bin', 'sha256': sha256(stream), 'bytes': stream.stat().st_size},
                'model_sha256': {'I': args.i_sha256, 'P': args.p_sha256},
                'frames': args.frames, 'fps': args.fps, 'original_width': args.width,
                'original_height': args.height, 'canvas_width': canvas_w, 'canvas_height': canvas_h,
                'base_qp': args.qp, 'reset_interval': args.reset_interval, 'placements': placements,
                'reconstruction': 'float32 CHW, crop then clamp[-1,1] and map to[0,1]',
                'force_zero_thres': None, 'ec_part': ec_part,
                'model_deployment': 'preinstalled verified I/P models; not transmitted per stream'}
    validate_manifest(manifest)
    write_json(output / 'manifest.json', manifest)
    source_attempts = deny_sources([record['path'] for record in frames])
    reference_pixels, _, reference_seconds = decode_stream(torch, i_model, p_model, manifest, stream)
    if source_attempts:
        raise ValueError('Native reference decoder attempted source access')
    np.save(output / 'sender_reconstruction.npy', reference_pixels, allow_pickle=False)
    write_json(output / 'sender_receipt.json', {'qualification': qualification, 'input_frames': frames,
               'encode_seconds_per_frame': encode_seconds,
               'reference_kind': 'same-process native decode of completed persisted stream; not P compressor pixels',
               'reference_decode_seconds_per_frame': reference_seconds,
               'reference_source_deny': 'Python audit of exact registered frame paths; not OS sandbox',
               'reference_source_open_attempts': source_attempts,
               'payload_bytes': stream.stat().st_size, 'charged_manifest_bytes': (output / 'manifest.json').stat().st_size})


def deny_sources(paths):
    roots = [Path(path).resolve() for path in paths]
    if not roots:
        raise ValueError('Decoder requires explicit denied source path(s)')
    attempts = []
    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        candidate = Path(os.fsdecode(arguments[0])).resolve()
        if any(candidate == root or root in candidate.parents for root in roots):
            attempts.append(str(candidate))
            raise PermissionError('Source access denied by receiver audit hook')
    sys.addaudithook(audit)
    # Demonstrate the deny hook is active, then distinguish this probe from decode.
    try:
        open(roots[0], 'rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Source deny self-test failed')
    attempts.clear()
    return attempts


def validate_manifest(manifest):
    if min(manifest['frames'], manifest['fps'], manifest['original_width'], manifest['original_height']) <= 0:
        raise ValueError('Invalid manifest count/cadence/geometry')
    if (manifest['canvas_height'], manifest['canvas_width']) != (max(1088, manifest['original_height']), max(1920, manifest['original_width'])):
        raise ValueError('Manifest canvas violates registered padding')
    if manifest['ec_part'] != int(manifest['canvas_height'] * manifest['canvas_width'] > 1280 * 720):
        raise ValueError('Manifest entropy partition policy mismatch')
    if manifest['force_zero_thres'] is not None:
        raise ValueError('Unregistered zero threshold policy')
    if len(manifest['placements']) != manifest['frames']:
        raise ValueError('Manifest placement count mismatch')
    offset = 0
    for index, placement in enumerate(manifest['placements']):
        reset = index > 0 and manifest['reset_interval'] > 0 and index % manifest['reset_interval'] == 1
        shift = [0, 2, 0, 1, 0, 1, 0, 1][index % 8] if index else 0
        if placement['index'] != index or placement['I'] != (index == 0) or placement['reset'] != reset:
            raise ValueError('Placement differs from registered I/reset policy')
        if placement['qp'] != manifest['base_qp'] + shift:
            raise ValueError('Placement differs from registered QP policy')
        if placement['offset'] != offset or placement['bytes'] <= 0:
            raise ValueError('Placement gap/overlap/invalid length')
        offset += placement['bytes']
    if offset != manifest['stream']['bytes']:
        raise ValueError('Placement physical byte total mismatch')


def decode_stream(torch, i_model, p_model, manifest, stream):
    """Native reference/fresh decode; receives no original image or source reader."""
    import numpy as np
    from src.utils.stream_helper import SPSHelper, NalType, read_header, read_sps_remaining, read_ip_remaining
    p_model.set_curr_poc(0)
    data, sps_helper, pixels, packets = io.BytesIO(stream.read_bytes()), SPSHelper(), [], []
    decode_seconds = []
    validate_manifest(manifest)
    with torch.no_grad():
        for index in range(manifest['frames']):
            before = data.tell()
            header = read_header(data)
            while header['nal_type'] == NalType.NAL_SPS:
                sps_helper.add_sps_by_id(read_sps_remaining(data, header['sps_id']))
                header = read_header(data)
            sps = sps_helper.get_sps_by_id(header['sps_id'])
            if sps is None or (sps['height'], sps['width']) != (manifest['canvas_height'], manifest['canvas_width']) or sps['ec_part'] != manifest['ec_part']:
                raise ValueError('Invalid SPS geometry/reference')
            qp, bitstream = read_ip_remaining(data)
            intra = header['nal_type'] == NalType.NAL_I
            if index == 0 and not intra:
                raise ValueError('First frame must initialize with I')
            actual = {'index': index, 'I': intra, 'reset': bool(sps['use_ada_i']),
                      'qp': qp, 'offset': before, 'bytes': data.tell() - before}
            if actual != manifest['placements'][index]:
                raise ValueError('Stream packet differs from charged placement')
            torch.cuda.synchronize()
            frame_start = time.perf_counter()
            if intra:
                result = i_model.decompress(bitstream, sps, qp)
                p_model.clear_dpb()
                p_model.add_ref_frame(None, result['x_hat'])
            elif header['nal_type'] == NalType.NAL_P:
                if sps['use_ada_i']:
                    p_model.reset_ref_feature()
                result = p_model.decompress(bitstream, sps, qp)
            else:
                raise ValueError('Unexpected packet kind')
            torch.cuda.synchronize()
            decode_seconds.append(time.perf_counter() - frame_start)
            if tuple(result['x_hat'].shape[-2:]) != (manifest['canvas_height'], manifest['canvas_width']):
                raise ValueError('Decoded native geometry mismatch')
            pixels.append(tensor_pixels(torch, result['x_hat'], manifest['original_width'], manifest['original_height']))
            packets.append(actual)
        if data.tell() != stream.stat().st_size:
            raise ValueError('Trailing stream bytes or frame-count mismatch')
    return np.stack(pixels), packets, decode_seconds


def decode(args):
    attempts = deny_sources(args.deny_source)
    import numpy as np
    torch, i_model, p_model, qualification = models(args, 'cuda')
    from src.utils.stream_helper import SPSHelper, NalType, read_header, read_sps_remaining, read_ip_remaining
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'gvcrt-native-v1' or manifest['source_revision'] != PIN:
        raise ValueError('Unregistered manifest schema/source')
    if manifest['model_sha256'] != {'I': args.i_sha256, 'P': args.p_sha256}:
        raise ValueError('Manifest model identity mismatch')
    if manifest['stream']['filename'] != 'stream.bin':
        raise ValueError('Unexpected stream filename')
    stream = manifest_path.parent / 'stream.bin'
    checked_file(stream, manifest['stream']['sha256'])
    if stream.stat().st_size != manifest['stream']['bytes']:
        raise ValueError('Stream physical length mismatch')
    pixels, packets, decode_seconds = decode_stream(torch, i_model, p_model, manifest, stream)
    if attempts:
        raise ValueError('Decoder attempted source access')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'reconstruction.npy', pixels, allow_pickle=False)
    write_json(output / 'receiver_receipt.json', {'qualification': qualification,
               'manifest_sha256': sha256(manifest_path), 'manifest_bytes': manifest_path.stat().st_size,
               'stream_sha256': sha256(stream), 'stream_bytes': stream.stat().st_size,
               'frames': len(pixels), 'packets': packets, 'decode_seconds_per_frame': decode_seconds,
               'source_deny_self_test': True,
               'source_open_attempts_during_decode': attempts, 'guard_scope': 'Python audit open; not OS sandbox',
               'reconstruction_sha256': sha256(output / 'reconstruction.npy')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['coverage', 'encode', 'decode', 'parity'])
    parser.add_argument('--disable-fused', action='store_true', help='Register PyTorch CUDA fallback instead of preexisting optional fused extension')
    parser.add_argument('--official-repo', type=Path)
    parser.add_argument('--i-checkpoint', type=Path)
    parser.add_argument('--i-sha256')
    parser.add_argument('--p-checkpoint', type=Path)
    parser.add_argument('--p-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames-json', type=Path)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=120)
    parser.add_argument('--qp', type=int, default=1)
    parser.add_argument('--reset-interval', type=int, default=8)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--deny-source', action='append', default=[])
    parser.add_argument('--sender-pixels', type=Path)
    parser.add_argument('--receiver-pixels', type=Path)
    args = parser.parse_args()
    if args.mode == 'coverage':
        _, _, _, receipt = models(args, 'cpu')
        write_json(args.output_dir, receipt)
    elif args.mode == 'encode':
        if min(args.frames, args.width, args.height, args.fps) <= 0:
            raise ValueError('Invalid geometry/cadence/count')
        encode(args)
    elif args.mode == 'decode':
        decode(args)
    else:
        import numpy as np
        sender = np.load(args.sender_pixels, allow_pickle=False)
        receiver = np.load(args.receiver_pixels, allow_pickle=False)
        if sender.shape != receiver.shape:
            raise ValueError('Parity geometry mismatch')
        delta = np.abs(sender - receiver)
        receipt = {'shape': list(sender.shape), 'exact_pixel_parity': bool(np.array_equal(sender, receiver)),
                   'max_abs_error': float(delta.max()), 'mean_abs_error': float(delta.mean()),
                   'sender_sha256': sha256(args.sender_pixels), 'receiver_sha256': sha256(args.receiver_pixels)}
        write_json(args.output_dir, receipt)
        if not receipt['exact_pixel_parity']:
            raise SystemExit('Exact sender/fresh receiver parity failed; diagnose before scientific use')


if __name__ == '__main__':
    main()
