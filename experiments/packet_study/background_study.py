"""Eight fixed registered-plate arms on the frozen Federer48 source contract.

Dispatch and the phase-three gate are root-owned. This entry point never
launches fleet jobs or re-encodes common anchors. ROI metrics are diagnostics.
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
import shutil
import signal
import subprocess
import sys
import time
import zipfile

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

SOURCE = 'outputs/evaluation-20260914/e03b/run-20260916-federer007/prepared_rgb.npy'
SOURCE_FILE_SHA = '9234147b68ace7217043c2761701f3bc915a0371db9fc83be6fea3973ae20191'
SOURCE_RGB_SHA = '1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8'
PACKET_ROOT = 'outputs/evaluation-20260914/e06/run-20260916-federer007-perframe-bbox'
PARENTS = {
    'first': {'setting': 'bbox_resized_first_reference_residual_off', 'bytes': 117861,
              'sha256': '949fba3080f030ae24e0a20021b0bc774f99cd3487dad95c7b01a1ea7fa7d8bd'},
    'perframe': {'setting': 'per_frame_crop_residual_off', 'bytes': 180955,
                 'sha256': 'da3c66116c11136ea4f24afa17982c4335823d8930a18f54b7eb6a67335c946e'}}
SMOKE_ARMS = ('first_identity_q32', 'median_registered_reset12_q32')
SHAPE = (48, 360, 640, 3)


def digest(blob): return hashlib.sha256(blob).hexdigest()


def receipt(path):
    path = Path(path); data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data), 'sha256': digest(data)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_parent(blob, spec):
    if len(blob) != spec['bytes'] or digest(blob) != spec['sha256']:
        raise ValueError('parent packet identity changed')


def validate_source(source, file_sha):
    import numpy as np
    if file_sha != SOURCE_FILE_SHA or source.shape != SHAPE or source.dtype != np.uint8 or digest(source.tobytes()) != SOURCE_RGB_SHA:
        raise ValueError('frozen complete prepared RGB source identity changed')


def validate_receiver(result, decoded):
    import numpy as np
    if result['shape'] != list(SHAPE) or decoded.shape != SHAPE or decoded.dtype != np.uint8:
        raise ValueError('fresh receiver omitted registered frames or changed raster/dtype')
    if result['rgb_sha256'] != digest(decoded.tobytes()):
        raise ValueError('saved fresh receiver RGB identity mismatch')


def physical_inventory(blob):
    """Top-level member costs sum exactly; nested costs are contained diagnostics."""
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate charged archive member')
        rows = []; compressed = 0
        for item in archive.infolist():
            compressed += item.compress_size
            data = archive.read(item.filename)
            row = {'member': item.filename, 'compressed_bytes': item.compress_size,
                   'uncompressed_bytes': item.file_size, 'compression': item.compress_type, 'sha256': digest(data)}
            if item.filename.endswith('.packet'):
                row['contained_packet'] = physical_inventory(data)
            if item.filename.startswith('background_payload_') and item.filename.endswith('.npy'):
                native = np.load(io.BytesIO(data), allow_pickle=False)
                if native.dtype != np.uint8 or native.ndim != 1:
                    raise ValueError('coded plate must be byte vector')
                row['contained_native_plate'] = {'bytes': native.nbytes, 'sha256': digest(native.tobytes())}
            rows.append(row)
    header = len(blob) - compressed
    if header < 0 or compressed + header != len(blob):
        raise ValueError('physical package inventory failed')
    return {'complete_physical_bytes': len(blob), 'compressed_members_bytes': compressed,
            'archive_headers_bytes': header, 'members': rows,
            'nested_cost_scope': 'contained member/native counts, never added again to complete physical rate'}


def limit_resources():
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 19: os.nice(19 - current)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)


def command(argv, log, stdin=None):
    """Bound only this experiment's own process group; never outside jobs."""
    if os.getloadavg()[0] > 40: raise RuntimeError('registered host load guard exceeded')
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    first = True
    while True:
        try:
            stdout, stderr = process.communicate(input=stdin if first else None, timeout=.5); break
        except subprocess.TimeoutExpired:
            first = False
            if os.getloadavg()[0] > 40 or time.monotonic() - started > 600:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.communicate()
                raise RuntimeError('own process group stopped for registered time/load guard')
    log.append({'argv': argv, 'returncode': process.returncode,
                'elapsed_seconds': time.monotonic() - started, 'stderr': stderr.decode('utf8', 'replace')})
    if process.returncode: raise RuntimeError(stderr.decode('utf8', 'replace')[-3000:])
    return stdout


def plate_encoder(directory, qp, ffmpeg, encoder, probe, log, receipts):
    """Same SVT CQP policy as the plate module, with explicit four-thread cap."""
    import numpy as np
    ordinal = 0
    def encode(bgr):
        nonlocal ordinal
        stem = directory / f'plate_{ordinal:02d}'; ordinal += 1
        raw = stem.with_suffix('.rgb'); y4m = stem.with_suffix('.y4m'); bit = stem.with_suffix('.ivf')
        rgb = np.ascontiguousarray(bgr[..., ::-1]); rgb.tofile(raw)
        command([ffmpeg, '-v', 'error', '-threads', '4', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', '640x360', '-r', '25', '-i', str(raw), '-frames:v', '1', '-threads', '4',
                 '-pix_fmt', 'yuv420p', '-f', 'yuv4mpegpipe', str(y4m)], log)
        command([encoder, '-i', str(y4m), '-b', str(bit), '--progress', '0', '--preset', '10',
                 '--rc', '0', '--aq-mode', '0', '--qp', str(qp), '--lp', '4', '--pin', '0'], log)
        streams = json.loads(command([probe, '-v', 'error', '-threads', '1', '-count_frames',
                    '-show_entries', 'stream=codec_name,width,height,pix_fmt,nb_read_frames',
                    '-of', 'json', str(bit)], log))['streams']
        if len(streams) != 1 or streams[0]['width'] != 640 or streams[0]['height'] != 360 or int(streams[0]['nb_read_frames']) != 1:
            raise ValueError('native coded plate frame/raster denominator mismatch')
        payload = bit.read_bytes()
        receipts.append({'ordinal': ordinal - 1, 'coded_plate': receipt(bit), 'native_probe': streams[0],
                         'encoder_rgb_sha256': digest(rgb.tobytes()), 'qp': qp,
                         'native_recipe': 'SVT-AV1 preset10 rc0 aq-mode0 CQP lp4; single yuv420p frame'})
        raw.unlink(); y4m.unlink()
        return payload
    return encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); limit_resources()
    import numpy as np
    import cv2
    import PIL
    cv2.setNumThreads(1)
    from experiments.background import registered_plate_codec as codec
    args.out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    registration = {**codec.REGISTRATION, 'source': SOURCE, 'source_file_sha256': SOURCE_FILE_SHA,
                    'source_rgb_sha256': SOURCE_RGB_SHA, 'shape': list(SHAPE), 'parents': PARENTS,
                    'smoke_arms': list(SMOKE_ARMS), 'fps': '12',
                    'native_plate_recipe': 'SVT-AV1 preset10 CQP rc0 aq-mode0 lp4; FFmpeg yuv420p conversion',
                    'anchor_policy': 'join root-owned complete common anchor report later, never re-encode anchors here',
                    'resource_policy': {'cpu': 4, 'blas': 1, 'native_encoder_threads': 4, 'receiver_native_threads': 'automatic, inherited four-CPU affinity', 'address_space_gib': 12,
                                        'nice': 19, 'io': 'idle', 'gpu': None, 'max_load': 40, 'child_seconds': 600}}
    write(args.out / 'registration-before-run.json', registration)
    source_path = args.data_root / SOURCE; source = np.load(source_path, allow_pickle=False)
    validate_source(source, receipt(source_path)['sha256'])
    parents = {}
    for name, spec in PARENTS.items():
        path = args.data_root / PACKET_ROOT / spec['setting'] / 'transport.npz'
        blob = path.read_bytes(); validate_parent(blob, spec); parents[name] = (path, blob)
    masks = codec.parent_masks(parents['first'][1], len(source))
    if masks.shape != SHAPE[:3]: raise ValueError('parent ROI mask frame/raster denominator mismatch')
    log = []
    ffmpeg = '/opt/local/bin/ffmpeg'; probe = '/opt/local/bin/ffprobe'
    encoder = os.environ.get('SVTAV1_BIN') or shutil.which('SvtAv1EncApp')
    if not encoder: raise FileNotFoundError('required native SVT-AV1 encoder unavailable; infrastructure prerequisite')
    report = {'registration': registration, 'smoke': args.smoke, 'code_revision': args.code_revision,
        'worker': receipt(__file__), 'plate_module': receipt(codec.__file__), 'command': sys.argv,
        'started_at_utc': started, 'hostname': platform.node(), 'affinity': sorted(os.sched_getaffinity(0)),
        'source': receipt(source_path), 'source_rgb_sha256': digest(source.tobytes()),
        'source_rgb_frame_sha256': [digest(frame.tobytes()) for frame in source],
        'roi_mask_sha256': digest(masks.tobytes()), 'roi_scope': 'first-reference parent masks, diagnostic only',
        'environment': {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__,
                        'platform': platform.platform(), 'threads': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
        'native_tools': [{'binary': receipt(tool), 'version': command([tool, '-version' if tool in (ffmpeg, probe) else '--version'], log).decode('utf8', 'replace')}
                         for tool in (ffmpeg, probe, encoder)], 'gpu_allocated': False, 'commands': log, 'rows': []}
    arms = [item for item in codec.ARMS if item[0] in SMOKE_ARMS] if args.smoke else codec.ARMS
    for arm in arms:
        name, _kind, _registered, appearance, qp = arm
        directory = args.out / name; directory.mkdir(); native = []
        path, parent = parents[appearance]
        packet, preparation = codec.encode_arm(parent, source, arm=name,
            encode_plate=plate_encoder(directory, qp, ffmpeg, encoder, probe, log, native))
        target = directory / 'package.npz'; target.write_bytes(packet)
        inventory = physical_inventory(packet)
        decoded_path = directory / 'decoded.npy'
        receiver_cmd = [sys.executable, '-m', 'experiments.background.registered_plate_codec', '--packet', str(target),
                        '--decoded', str(decoded_path), '--data-root', str(args.data_root)]
        receiver = json.loads(command(receiver_cmd, log).strip().splitlines()[-1])
        decoded = np.load(decoded_path, allow_pickle=False); validate_receiver(receiver, decoded)
        row = {'arm': name, 'parent': receipt(path), 'packet': receipt(target),
               'complete_physical_bytes': len(packet), 'bits_per_second': len(packet) * 8 / 4,
               'inventory': inventory, 'preparation': preparation, 'native_plates': native,
               'decoded': receipt(decoded_path), 'receiver': receiver,
               'quality': codec.score(source, decoded, masks)}
        write(directory / 'receiver.json', receiver); row['receiver_receipt'] = receipt(directory / 'receiver.json')
        report['rows'].append(row); write(args.out / 'report.partial.json', report)
    report.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - clock)
    write(args.out / 'report.json', report)
    print(json.dumps({'status': 'complete', 'report': receipt(args.out / 'report.json')}))

if __name__ == '__main__': main()
