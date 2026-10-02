"""Persist one registered native anchor. Not a sweep, selector, or resume tool.

Uses the clean legacy command builder, full-colour inputs, fresh bitstream
decoding and the same original 4K reference as legacy_recertify. A prospective
registration and same-path smoke are required before scientific execution.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

SOURCE_HASHES = (
    '09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573',
    '2009a1bb6934c4109f16f73951acb2af5a22d91a3cb15f7f79fc3dc883bb6c6b',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def raw_shape(path, count, height=2160, width=3840):
    """Reject missing or excess frames; never pad or silently truncate."""
    shape = (count, height, width, 3)
    if Path(path).stat().st_size != count * height * width * 3:
        raise ValueError('decoded byte count differs from registered frame count')
    return shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy-root', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, choices=[2, 12, 48, 96], required=True)
    p.add_argument('--codec', choices=['vvc', 'av1'], required=True)
    p.add_argument('--qp', type=int, required=True)
    p.add_argument('--width', type=int, choices=[3840, 2560, 1920, 1280], default=3840)
    p.add_argument('--access', choices=['continuous', 'segmented'], required=True)
    p.add_argument('--registration', required=True)
    p.add_argument('--timeout', type=int, default=3300)
    a = p.parse_args()
    low = 1 if a.codec == 'av1' else 0
    if not low <= a.qp <= 63 or (a.codec == 'vvc' and a.width != 3840):
        raise ValueError('unsupported registered QP/resolution')
    if not 0 < a.timeout <= 3300 or len(os.sched_getaffinity(0)) > 8:
        raise ValueError('bounded native timeout and at most eight assigned CPU cores required')
    root = Path(a.legacy_root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '274638bdae7f5bd63c4f834a24804c0e14ed8d83' or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']):
        raise ValueError('clean pinned legacy revision required')
    registration = json.loads(Path(a.registration).read_text())
    arm = {'codec': a.codec, 'qp': a.qp, 'width': a.width, 'access': a.access, 'frames_per_scene': a.frames}
    if registration.get('status') != 'frozen_before_execution' or arm not in registration.get('arms', []):
        raise ValueError('arm is absent from frozen registration')
    if registration.get('worker_sha256') != digest(__file__):
        raise ValueError('registered worker changed')
    libraries = registration.get('libraries', {})
    if not libraries or not any('libvmaf' in path for path in libraries):
        raise ValueError('registered linked libraries including libvmaf required')
    for path, expected in libraries.items():
        if digest(path) != expected:
            raise ValueError('registered native library changed')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (48 * 1024**3, 48 * 1024**3))
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '8'
    sys.path.insert(0, str(root))
    import numpy as np
    from src.components.codec.command import build_command
    from src.components.codec.tools import resolve_ffmpeg, resolve_encoder
    from src.components.codec.frames import rgb_to_luma
    from src.components.metrics.vmaf import _write_y4m_clip
    from src.components.metrics.ssim import SsimMetric
    from src.contracts.codecs import EncodeRequest, RateControl
    base = Path(a.data_root) / 'outputs/gate-a-vvc-webp-n96-run2/points/C2.run'
    clips, identities = [], []
    for i, expected in enumerate(SOURCE_HASHES):
        path = base / f'chunk_{i:02d}/source.npy'
        arr = np.load(path, allow_pickle=False, mmap_mode='r')
        if arr.shape != (96, 2160, 3840, 3) or arr.dtype != np.uint8:
            raise ValueError('original source raster mismatch')
        rgb_hash = hashlib.sha256(np.ascontiguousarray(arr).data).hexdigest()
        if rgb_hash != expected:
            raise ValueError('original source identity mismatch')
        clips.append(arr[:a.frames])
        identities.append({'path': str(path), 'rgb_sha256': rgb_hash, 'file_sha256': digest(path)})
    source = np.concatenate(clips)
    ff, encoder = resolve_ffmpeg(), resolve_encoder(a.codec)
    for tool in (ff, encoder):
        if registration['tools'].get(tool.path) != digest(tool.path):
            raise ValueError('registered native executable changed')
    calls = []
    def run(label, argv):
        start = time.monotonic()
        with (out / (label + '.log')).open('wb') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
        calls.append({'label': label, 'argv': argv, 'seconds': time.monotonic() - start})
    height = a.width * 9 // 16
    request = EncodeRequest(codec_name=a.codec, rate_control=RateControl.QP,
                            rate=a.qp, preset='slower' if a.codec == 'vvc' else '0',
                            pix_fmt='yuv420p', extra_args=('-threads', '8') if a.codec == 'vvc' else ('--lp', '8'))
    streams, decoded = [], []
    pieces = [source] if a.access == 'continuous' else clips
    for i, piece in enumerate(pieces):
        prefix = f'part-{i:02d}'
        y4m = out / (prefix + '.source.y4m')
        start = time.monotonic()
        _write_y4m_clip(y4m, piece, ff.path)
        calls.append({'label': prefix + '-rgb-to-yuv', 'seconds': time.monotonic() - start,
                      'policy': 'legacy RGB24 to yuv420p converter; FFmpeg default matrix/range; 24fps'})
        native_input = y4m
        if a.width != 3840:
            native_input = out / (prefix + '.scaled.y4m')
            run(prefix + '-downscale', [ff.path, '-hide_banner', '-loglevel', 'error', '-threads', '8',
                '-i', str(y4m), '-vf', f'scale={a.width}:{height}:flags=lanczos', '-pix_fmt', 'yuv420p', str(native_input)])
        stream = out / (prefix + ('.vvc' if a.codec == 'vvc' else '.ivf'))
        argv = build_command('encode', request, source=native_input, dest=stream, encoder=encoder, ffmpeg=ff)
        run(prefix + '-encode', argv)
        if stream.stat().st_size <= 0:
            raise ValueError('empty native stream')
        probe = str(Path(ff.path).with_name('ffprobe'))
        if registration['tools'].get(probe) != digest(probe):
            raise ValueError('registered native probe changed')
        probe_argv = [probe, '-v', 'error', '-show_streams', '-of', 'json', str(stream)]
        probe_data = json.loads(subprocess.check_output(probe_argv, timeout=60))
        video = [s for s in probe_data['streams'] if s['codec_type'] == 'video']
        if len(video) != 1 or (video[0]['width'], video[0]['height']) != (a.width, height):
            raise ValueError('native bitstream raster differs from registered resolution')
        (out / (prefix + '.probe.json')).write_text(json.dumps(probe_data, indent=2) + '\n')
        decoded_y4m = out / (prefix + '.decoded.y4m')
        run(prefix + '-fresh-native-decode', [ff.path, '-hide_banner', '-loglevel', 'error',
            '-threads', '8', '-i', str(stream), '-pix_fmt', 'yuv420p', str(decoded_y4m)])
        raw = out / (prefix + '.decoded.rgb')
        argv = [ff.path, '-hide_banner', '-loglevel', 'error', '-threads', '8', '-i', str(decoded_y4m)]
        if a.width != 3840:
            argv += ['-vf', 'scale=3840:2160:flags=lanczos']
        argv += ['-pix_fmt', 'rgb24', '-f', 'rawvideo', str(raw)]
        run(prefix + '-upscale-rgb-conversion', argv)
        shape = raw_shape(raw, len(piece))
        decoded.append(np.memmap(raw, dtype=np.uint8, mode='r', shape=shape))
        streams.append({'file': stream.name, 'bytes': stream.stat().st_size, 'sha256': digest(stream),
                        'decoded_rgb_sha256': digest(raw), 'shape': list(shape), 'native_probe': video[0], 'probe_argv': probe_argv})
    delivered = np.concatenate(decoded)
    ref_y4m, dist_y4m = out / 'reference.y4m', out / 'delivered.y4m'
    _write_y4m_clip(ref_y4m, source, ff.path)
    _write_y4m_clip(dist_y4m, delivered, ff.path)
    metric_log = out / 'vmaf.json'
    filt = f'[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];[dist][ref]libvmaf=model=version=vmaf_v0.6.1:log_path={metric_log}:log_fmt=json:n_threads=8'
    run('vmaf', [ff.path, '-hide_banner', '-loglevel', 'error', '-i', str(ref_y4m), '-i', str(dist_y4m), '-filter_complex', filt, '-f', 'null', '-'])
    values = [x['metrics']['vmaf'] for x in json.loads(metric_log.read_text())['frames']]
    if len(values) != len(source) or not all(np.isfinite(values)):
        raise ValueError('missing or nonfinite VMAF frames')
    mse, ssim = [], []
    for ref, got in zip(source, delivered, strict=True):
        diff = rgb_to_luma(ref).astype(np.float64) - rgb_to_luma(got).astype(np.float64)
        mse.append(float(np.mean(diff**2)))
        ssim.append(float(SsimMetric().score(ref[None], got[None])))
    def summary(lo, hi):
        error = float(np.mean(mse[lo:hi]))
        return {'vmaf': float(np.mean(values[lo:hi])), 'psnr_y': 10 * float(np.log10(255**2 / error)), 'ssim': float(np.mean(ssim[lo:hi]))}
    manifest = {'schema': 1, 'fps': 24, 'width': a.width, 'height': height,
                'reconstruct_width': 3840, 'reconstruct_height': 2160,
                'upscale_filter': 'lanczos' if a.width != 3840 else 'none',
                'streams': [x['file'] for x in streams]}
    packet_manifest = out / 'manifest.json'
    packet_manifest.write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n')
    result = {'arm': arm, 'complete': True, 'paper_evidence': False, 'legacy_revision': revision,
              'registration_sha256': digest(a.registration), 'worker_sha256': digest(__file__),
              'source': identities, 'source_selected_rgb_sha256': hashlib.sha256(source.data).hexdigest(),
              'stream_bytes': sum(x['bytes'] for x in streams), 'manifest_bytes': packet_manifest.stat().st_size,
              'physical_payload_bytes': sum(x['bytes'] for x in streams) + packet_manifest.stat().st_size,
              'streams': streams, 'joined': summary(0, len(source)),
              'per_window': [summary(i*a.frames, (i+1)*a.frames) for i in range(2)],
              'metric_frames': len(values), 'vmaf_log_sha256': digest(metric_log), 'calls': calls,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'request': dataclasses.asdict(request),
              'receiver_boundary': 'fresh native decode from persisted stream; no sender pixel input to decode argv',
              'isolation': 'CPU affinity and process boundary; no OS file-access sandbox asserted',
              'scope': 'registered development observation, not unseen-scene or generalization evidence'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
