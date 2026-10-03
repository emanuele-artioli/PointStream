"""Read-only, CPU-bounded score of one registered fresh payload receiver run.

Registration: status=frozen_before_execution, worker_sha256, tools/libraries
absolute-path SHA256 maps (libraries includes libvmaf), and arms containing
frames, mask_policy, foreground, renderer_registration_sha256,
original_packet_sha256, and input_sha256 for the five upstream files below.
No decoder, training, GPU, sweep, selection, or upstream writes occur here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REVISION = '274638bdae7f5bd63c4f834a24804c0e14ed8d83'
SOURCE_RGB = '09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573'
INPUTS = ('report.json', 'receiver.receipt.json', 'receiver.npy', 'scene.npz', 'manifest.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rgb_digest(array):
    h = hashlib.sha256()
    for frame in array:
        h.update(frame.tobytes(order='C'))
    return h.hexdigest()


def psnr(mse):
    # JSON has no infinity: retain the mathematical identity case explicitly.
    return None if mse == 0 else 10 * math.log10(255**2 / mse)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-root', 'data-root', 'receiver-dir', 'registration', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--frames', type=int, choices=(12, 48, 96), required=True)
    a = p.parse_args()
    began = time.monotonic()
    require(sys.platform == 'linux', 'Linux resource accounting and affinity required')
    require(0 < len(os.sched_getaffinity(0)) <= 8, 'at most eight assigned CPU cores required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PS_CODEC_THREADS', 'SSIM_THREADS'):
        os.environ[key] = '8'
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (48 * 1024**3, 48 * 1024**3))
    root = Path(a.legacy_root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == REVISION and not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']), 'clean pinned legacy required')
    reg_path = Path(a.registration).resolve()
    registration_sha256 = digest(reg_path)
    reg = json.loads(reg_path.read_text())
    require(reg.get('status') == 'frozen_before_execution' and reg.get('worker_sha256') == digest(__file__), 'frozen scorer identity required')
    dependencies = {}
    for category in ('tools', 'libraries'):
        entries = reg.get(category, {})
        require(entries, 'registered tools and libraries required')
        for path, expected in entries.items():
            require(Path(path).is_absolute() and digest(path) == expected, 'native dependency changed: ' + path)
            dependencies[path] = expected
    require(any('libvmaf' in path for path in reg['libraries']), 'registered libvmaf required')
    upstream = Path(a.receiver_dir).resolve()
    report = json.loads((upstream / 'report.json').read_text())
    receipt = json.loads((upstream / 'receiver.receipt.json').read_text())
    arm = {'frames': a.frames, 'mask_policy': report['arm']['mask_policy'],
           'foreground': report['arm']['foreground'],
           'renderer_registration_sha256': report['registration_sha256'],
           'original_packet_sha256': report['original_packet_sha256'],
           'input_sha256': {name: digest(upstream / name) for name in INPUTS}}
    require(arm in reg.get('arms', []), 'exact upstream arm absent from frozen score registration')
    require(report.get('complete') is True and report['arm']['frames'] == a.frames, 'incomplete upstream report or frame mismatch')
    require(report['receiver'] == receipt and receipt.get('violations') == [], 'receipt mismatch or observed guard violations')
    require(receipt.get('denied_roots'), 'payload receiver guard roots missing')
    manifest = json.loads((upstream / 'manifest.json').read_text())
    require(manifest == {'schema': 1, 'fps': 24, 'package': 'scene.npz'}, 'unexpected physical package manifest')
    package = upstream / 'scene.npz'
    physical = package.stat().st_size + (upstream / 'manifest.json').stat().st_size
    require(receipt['package_sha256'] == arm['input_sha256']['scene.npz'] and receipt['package_bytes'] == package.stat().st_size and report['physical_bytes'] == physical, 'physical package accounting mismatch')
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    from src.components.codec.frames import rgb_to_luma
    from src.components.codec.tools import resolve_ffmpeg
    from src.components.metrics.ssim import SsimMetric
    from src.components.metrics.vmaf import _write_y4m_clip
    with np.load(package, allow_pickle=False) as packet:
        meta = json.loads(packet['metadata'].tobytes())
    require(meta['frame_count'] == a.frames and (meta['height'], meta['width']) == (2160, 3840) and len(meta['background']['homographies']) == a.frames, 'packet metadata frame/raster mismatch')
    source_path = Path(a.data_root).resolve() / 'outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_00/source.npy'
    require(str(source_path) == '/home/itec/emanuele/pointstream-data/outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_00/source.npy', 'immutable source000 path required')
    source96 = np.load(source_path, allow_pickle=False, mmap_mode='r')
    require(source96.shape == (96, 2160, 3840, 3) and source96.dtype == np.uint8 and rgb_digest(source96) == SOURCE_RGB and report['source96_rgb_sha256'] == SOURCE_RGB, 'full96 source identity mismatch')
    source_file_sha256 = digest(source_path)
    source = source96[:a.frames]
    delivered = np.load(upstream / 'receiver.npy', allow_pickle=False, mmap_mode='r')
    require(delivered.shape == (a.frames, 2160, 3840, 3) and delivered.dtype == np.uint8 and receipt['frames_shape'] == list(delivered.shape) and rgb_digest(delivered) == receipt['frames_rgb_sha256'], 'decoded RGB identity mismatch')
    ff = resolve_ffmpeg()
    require(reg['tools'].get(ff.path) == digest(ff.path), 'resolved FFmpeg must be pinned')
    require(reg['tools'].get(str(Path(sys.executable).resolve())) == digest(Path(sys.executable).resolve()), 'Python executable must be pinned')
    out = Path(a.out).resolve()
    require(out != upstream and upstream not in out.parents and out not in upstream.parents, 'output must be separate from upstream')
    out.mkdir(parents=True, exist_ok=False)
    calls = []
    for name, clip in (('reference', source), ('delivered', delivered)):
        destination = out / (name + '.y4m')
        start = time.monotonic()
        _write_y4m_clip(destination, clip, ff.path)
        calls.append({'label': name + '-rgb-to-yuv', 'argv': [ff.path, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '3840x2160', '-r', '24', '-i', '-', '-pix_fmt', 'yuv420p', str(destination)], 'seconds': time.monotonic() - start, 'policy': 'pinned legacy RGB24 to YUV420p; FFmpeg default matrix/range; 24fps'})
    log = out / 'vmaf.json'
    filt = f'[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];[dist][ref]libvmaf=model=version=vmaf_v0.6.1:log_path={log}:log_fmt=json:n_threads=8'
    argv = [ff.path, '-hide_banner', '-loglevel', 'error', '-i', str(out / 'reference.y4m'), '-i', str(out / 'delivered.y4m'), '-filter_complex', filt, '-f', 'null', '-']
    start = time.monotonic()
    with (out / 'vmaf.log').open('wb') as stream:
        subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=3300)
    calls.append({'label': 'vmaf', 'argv': argv, 'seconds': time.monotonic() - start})
    frames = json.loads(log.read_text())['frames']
    require(len(frames) == a.frames and [f['frameNum'] for f in frames] == list(range(a.frames)), 'VMAF frame sequence mismatch')
    values = [float(f['metrics']['vmaf']) for f in frames]
    require(all(math.isfinite(v) for v in values), 'nonfinite VMAF frame')
    metrics = []
    scorer = SsimMetric()
    for i, (ref, got) in enumerate(zip(source, delivered, strict=True)):
        diff = rgb_to_luma(ref).astype(np.float64) - rgb_to_luma(got).astype(np.float64)
        mse = float(np.mean(diff**2))
        ssim = float(scorer.score(ref[None], got[None]))
        require(math.isfinite(mse) and math.isfinite(ssim), 'nonfinite pixel metric')
        metrics.append({'frame': i, 'vmaf': values[i], 'mse_y': mse, 'psnr_y': psnr(mse), 'ssim': ssim})
    pooled = float(np.mean([f['mse_y'] for f in metrics]))
    # Detect mutation during scoring; never overwrite or repair upstream artifacts.
    require(digest(source_path) == source_file_sha256, 'source file changed during scoring')
    require(digest(reg_path) == registration_sha256 and digest(__file__) == reg['worker_sha256'], 'score registration or worker changed during scoring')
    require(all(digest(upstream / name) == arm['input_sha256'][name] for name in INPUTS), 'upstream changed during scoring')
    result = {'complete': True, 'paper_evidence': False, 'arm': arm, 'registration_sha256': registration_sha256, 'worker_sha256': digest(__file__), 'legacy_revision': revision,
              'source': {'path': str(source_path), 'shape': list(source96.shape), 'full96_rgb_sha256': SOURCE_RGB, 'file_sha256': source_file_sha256, 'selected_rgb_sha256': rgb_digest(source)},
              'physical_payload_bytes': physical, 'package_bytes': package.stat().st_size, 'manifest_bytes': (upstream / 'manifest.json').stat().st_size,
              'receiver_rgb_sha256': receipt['frames_rgb_sha256'], 'per_frame': metrics,
              'joined': {'vmaf': float(np.mean(values)), 'mse_y': pooled, 'psnr_y': psnr(pooled), 'ssim': float(np.mean([f['ssim'] for f in metrics]))},
              'psnr_identity_policy': 'null means positive infinity when MSE is exactly zero; pooled equal-frame MSE, not mean frame PSNR',
              'tools_and_libraries': dependencies,
              'legacy_metric_code_sha256': {str(root / rel): digest(root / rel) for rel in ('src/components/codec/frames.py', 'src/components/codec/tools.py', 'src/components/metrics/vmaf.py', 'src/components/metrics/ssim.py', 'src/components/metrics/frames.py')}, 'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'ffmpeg': subprocess.check_output([ff.path, '-version'], text=True)},
              'calls': calls, 'command': sys.argv, 'output_sha256': {x.name: digest(x) for x in out.iterdir() if x.is_file()},
              'resources': {'seconds': time.monotonic() - began, 'self': list(resource.getrusage(resource.RUSAGE_SELF)), 'children': list(resource.getrusage(resource.RUSAGE_CHILDREN)), 'affinity': sorted(os.sched_getaffinity(0)), 'address_space_limit_bytes': 48 * 1024**3, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'caveat': 'Linux process rusage; no measured concurrent aggregate memory or live-codec latency'},
              'scope': 'source000 development observation only; frozen initial appearance/alpha template does not prove pose/event fidelity or old two-window Gate A recertification; shorter-prefix controls reuse full96 background and are not valid matched RD points; receipt reports no observed audit guard violations, not OS sandbox proof'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
