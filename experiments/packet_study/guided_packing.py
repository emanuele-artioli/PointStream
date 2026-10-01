"""Registered lossless packing parity on three guided foreground-only packets.

Root dispatches CPU jobs. This worker retains small packages/receipts, never the
16-frame 4K reconstructed arrays. RGB hash parity is not source/task accuracy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BASE = 'outputs/sam31-unification/pilot-v1-20260927-run-05'
REGISTERED = [
    {'source_id': 'alcaraz_highlights_scene_000',
     'package_sha256': 'c943d5639c2b30fbd27bf3e42e2d93de65db390f5cf4dbe09b1671404f6a2297',
     'package_bytes': 649798, 'frame_ids': list(range(38, 54)),
     'decoded_rgb_sha256': 'c41b2e05f918813d0addfd50bd04e5b36e50759726ce65930aa61a5bf435e4a4'},
    {'source_id': 'alcaraz_highlights_scene_010',
     'package_sha256': 'dff09144436a863030123261f3d1161e6e33d88a7e6ae0e0ba5ca469d1355fac',
     'package_bytes': 1175424, 'frame_ids': list(range(1, 17)),
     'decoded_rgb_sha256': '23edf12ab26c06a992e82695a4da4a01765434f6426de54052e1ae21d0b03623'},
    {'source_id': 'alcaraz_perricard_scene_007',
     'package_sha256': 'a975fa66d5807c523972d19890fb988f362ee687719cf37adfbda0d844320a81',
     'package_bytes': 3202610, 'frame_ids': list(range(361, 377)),
     'decoded_rgb_sha256': 'fc3252ea9d41ede04aa342fb4b7651d32d5f8a844a1725fe1b4cfe00c11efd2e'},
]
SHAPE = [16, 2160, 3840, 3]
REGISTRATION = {'sources': REGISTERED, 'shape': SHAPE,
    'variants': ['original', 'batch_psm1', 'batch_rle'], 'mask_scale': 1,
    'scope': 'guided foreground-only component packing; absent background black; no RGB-quality/full-codec/task claim',
    'accounting': 'complete persisted package bytes, adapter headers included; expanded intermediate not delivered rate',
    'parity': 'fresh receiver full16-frame RGB hash equal original and pinned historical decoder output',
    'resource_policy': {'cpu_threads': 4, 'blas_threads': 1, 'address_space_gib': 12,
                        'nice': 19, 'io': 'idle', 'gpu': None, 'max_load': 40,
                        'own_child_timeout_seconds': 600}}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def receipt(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(data), 'sha256': digest(data)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_input(blob, spec):
    if len(blob) != spec['package_bytes'] or digest(blob) != spec['package_sha256']:
        raise ValueError('registered original package byte identity changed')
    import numpy as np
    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
        metadata = json.loads(arrays['metadata'].tobytes())
    if metadata.get('schema') != 1 or [metadata.get('frame_count'), metadata.get('height'), metadata.get('width'), 3] != SHAPE:
        raise ValueError('registered 16-frame 4K envelope dimensions differ')
    return metadata


def validate_receiver(result, spec, original_hash=None):
    if result['shape'] != SHAPE or result['dtype'] != 'uint8' or len(result['rgb_frame_sha256']) != SHAPE[0]:
        raise ValueError('receiver omitted or changed registered output frames/geometry')
    expected = original_hash or spec['decoded_rgb_sha256']
    if result['decoded_rgb_sha256'] != expected:
        raise ValueError('lossless complete RGB pixel parity failed')


def limit_resources():
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 19:
        os.nice(19 - current)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)


def child(args):
    limit_resources()
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    from experiments.tier.receiver_replay import receiver_access_guard
    guard, reads, commands = receiver_access_guard(args.packet, args.receipt, args.data_root, Path(__file__).resolve().parents[2])
    sys.addaudithook(guard)
    from src.runner.packet_packing import unpack_client_envelope
    from src.runner.client import reconstruct_serialized_client
    data = args.packet.read_bytes()
    envelope = data if args.variant == 'original' else unpack_client_envelope(data)
    frames = np.asarray(reconstruct_serialized_client(envelope, require_compressed=True))
    if list(frames.shape) != SHAPE or frames.dtype != np.uint8:
        raise ValueError('complete registered uint8 receiver frames required')
    full_hash = hashlib.sha256(); frame_hashes = []
    for frame in frames:
        view = memoryview(np.ascontiguousarray(frame))
        full_hash.update(view); frame_hashes.append(hashlib.sha256(view).hexdigest())
    result = {'packet': receipt(args.packet), 'shape': list(frames.shape), 'dtype': str(frames.dtype),
        'decoded_rgb_sha256': full_hash.hexdigest(), 'rgb_frame_sha256': frame_hashes,
        'observed_registered_root_reads': reads, 'native_commands': commands,
        'boundary': 'observed Python source-root read guard/native argv, not OS sandbox',
        'retained_decoded_arrays': False}
    write(args.receipt, result)
    print(json.dumps(result))


def guarded_command(command, log):
    if os.getloadavg()[0] > 40:
        raise RuntimeError('host load exceeds registered guard')
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    while True:
        try:
            stdout, stderr = process.communicate(timeout=.5); break
        except subprocess.TimeoutExpired:
            if os.getloadavg()[0] > 40 or time.monotonic() - started > 600:
                process.terminate()
                try: process.communicate(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.communicate()
                raise RuntimeError('stopped own receiver child for time/load guard')
    log.append({'argv': command, 'returncode': process.returncode,
                'elapsed_seconds': time.monotonic() - started,
                'stderr': stderr.decode('utf8', 'replace')})
    if process.returncode:
        raise RuntimeError(stderr.decode('utf8', 'replace')[-3000:])
    return json.loads(stdout.strip().splitlines()[-1])


def study(args):
    limit_resources()
    from src.runner.packet_packing import pack_client_envelope
    import numpy as np
    import cv2
    import PIL
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    write(args.out / 'registration-before-run.json', REGISTRATION)
    report = {'registration': REGISTRATION, 'smoke': args.smoke, 'code_revision': args.code_revision,
        'worker': receipt(__file__), 'adapter': receipt(Path('src/runner/packet_packing.py')),
        'decoder_sources': [receipt(Path(name)) for name in ['src/runner/client.py', 'src/runner/mask_wire.py', 'src/runner/mask_rle.py']],
        'started_at_utc': started, 'hostname': platform.node(), 'affinity': sorted(os.sched_getaffinity(0)),
        'environment': {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__, 'platform': platform.platform(),
                        'threads': {key: os.environ[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'gpu_allocated': False, 'command': sys.argv, 'rows': [], 'commands': []}
    for spec in REGISTERED[:1] if args.smoke else REGISTERED:
        original_path = args.data_root / BASE / spec['source_id'] / 'pointstream-client-payload.npz'
        blob = original_path.read_bytes(); metadata = validate_input(blob, spec)
        original_hash = None
        for name, codec in [('original', None), ('batch_psm1', 'psm1'), ('batch_rle', 'rle')]:
            directory = args.out / spec['source_id'] / name; directory.mkdir(parents=True)
            packet = directory / 'package.npz'
            packet.write_bytes(blob if codec is None else pack_client_envelope(blob, mask_scale=1, mask_codec=codec, batch_masks=True))
            target = directory / 'receiver.json'
            command = [sys.executable, '-m', 'experiments.packet_study.guided_packing', '--data-root', str(args.data_root),
                       '--packet', str(packet), '--receipt', str(target), '--variant', name]
            result = guarded_command(command, report['commands'])
            validate_receiver(result, spec, original_hash)
            if codec is None:
                original_hash = result['decoded_rgb_sha256']
            packed = receipt(packet)
            report['rows'].append({'source_id': spec['source_id'], 'frame_ids': spec['frame_ids'],
                'variant': name, 'mask_scale': 1, 'batch_masks': codec is not None,
                'original_package': receipt(original_path), 'packet': packed,
                'bytes_saved': len(blob) - packed['bytes'], 'fraction_saved': 1 - packed['bytes'] / len(blob),
                'metadata_frame_count': metadata['frame_count'], 'receiver_receipt': receipt(target),
                'receiver': result, 'lossless_full_rgb_parity': True,
                'historical_rgb_parity': result['decoded_rgb_sha256'] == spec['decoded_rgb_sha256']})
            write(args.out / 'report.partial.json', report)
    report.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - clock)
    write(args.out / 'report.json', report)
    print(json.dumps({'status': 'complete', 'report': receipt(args.out / 'report.json')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--code-revision', default='unfrozen')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--variant', choices=REGISTRATION['variants'])
    args = parser.parse_args()
    if args.packet:
        child(args)
    else:
        study(args)

if __name__ == '__main__':
    main()
