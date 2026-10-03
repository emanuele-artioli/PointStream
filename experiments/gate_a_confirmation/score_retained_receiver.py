"""Retrospective diagnostic of retained scene00 from an incomplete parent run.

Registration: frozen_before_execution, worker_sha256, helper_sha256 (sibling
score_receiver.py), tools/libraries path->SHA maps including Python/FFmpeg/libvmaf;
arms contain rung, frames (12 instrument smoke or 96 diagnostic),
original_full96_registration_sha256, input_sha256 for the actual three files,
expected_receipt (exact JSON), and source96_rgb_sha256. This diagnostic neither
completes nor asserts model compatibility with the historical two-window run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from score_receiver import digest, rgb_digest, require, psnr, REVISION, SOURCE_RGB

INPUTS = ('scene-00.npz', 'scene-00.receiver.npy', 'scene-00.receiver.receipt.json')
RETAINED = {
    'C2': {'package_sha256': 'd937c0b78302b63ae4e4d878dd00de0355dbdce4c4e63ce6acdb81badfaae282',
           'package_bytes': 51931, 'frames_rgb_sha256': '7f61e8f63d273035b3e1a9709e4d09e1158f34a858ee5c5d57c3fa547888016f'},
    'C3': {'package_sha256': 'cec02d1a90c4b12c4b732bfe151113ffc041072ab7129cdc2bf93fcdbc0a20f4',
           'package_bytes': 67521, 'frames_rgb_sha256': '8db70e4c05524a6f90229c80b0c7b39484886968a62ed0d842a86b447559e98b'},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-root', 'data-root', 'receiver-dir', 'registration', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--rung', choices=('C2', 'C3'), required=True)
    p.add_argument('--frames', type=int, choices=(12, 96), default=96)
    p.add_argument('--original-full96-registration-sha256', required=True)
    a = p.parse_args()
    began = time.monotonic()
    require(sys.platform == 'linux' and 0 < len(os.sched_getaffinity(0)) <= 8, 'Linux affinity at most eight cores required')
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
    helper = Path(__file__).with_name('score_receiver.py')
    require(reg.get('status') == 'frozen_before_execution' and reg.get('worker_sha256') == digest(__file__) and reg.get('helper_sha256') == digest(helper), 'frozen scorer/helper identities required')
    upstream = Path(a.receiver_dir).resolve()
    require(str(upstream) == f'/home/itec/emanuele/pointstream-data/audits/gate-a-recert-{a.rung}-full96-20261002', 'explicit retained partial-run directory required')
    receipt = json.loads((upstream / INPUTS[2]).read_text())
    arm = {'rung': a.rung, 'frames': a.frames,
           'original_full96_registration_sha256': a.original_full96_registration_sha256,
           'input_sha256': {name: digest(upstream / name) for name in INPUTS},
           'expected_receipt': receipt, 'source96_rgb_sha256': SOURCE_RGB}
    require(len(a.original_full96_registration_sha256) == 64 and all(c in '0123456789abcdef' for c in a.original_full96_registration_sha256), 'original registration SHA256 required')
    require(arm in reg.get('arms', []), 'exact retained arm absent from frozen diagnostic registration')
    expected = RETAINED[a.rung]
    require(all(receipt[k] == v for k, v in expected.items()), 'retained receipt differs from independently inventoried identity')
    require(receipt['frames_shape'] == [96, 2160, 3840, 3] and receipt.get('violations') == [], 'full96 receipt shape or observed violation mismatch')
    require(receipt['denied_roots'] == ['/home/itec/emanuele/pointstream-data', str(upstream / 'encoder-checkpoints')], 'original receiver guard roots mismatch')
    package = upstream / INPUTS[0]
    require(digest(package) == receipt['package_sha256'] and package.stat().st_size == receipt['package_bytes'], 'retained package identity mismatch')
    dependencies = {}
    for category in ('tools', 'libraries'):
        require(reg.get(category), 'registered native tools/libraries required')
        for path, expected_sha in reg[category].items():
            require(Path(path).is_absolute() and digest(path) == expected_sha, 'native dependency changed: ' + path)
            dependencies[path] = expected_sha
    require(any('libvmaf' in path for path in reg['libraries']), 'registered libvmaf required')
    require(reg['tools'].get(str(Path(sys.executable).resolve())) == digest(Path(sys.executable).resolve()), 'Python executable must be pinned')
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    from src.components.codec.frames import rgb_to_luma
    from src.components.codec.tools import resolve_ffmpeg
    from src.components.metrics.ssim import SsimMetric
    from src.components.metrics.vmaf import _write_y4m_clip
    with np.load(package, allow_pickle=False) as payload:
        metadata = json.loads(payload['metadata'].tobytes())
    require(metadata['frame_count'] == 96 and (metadata['height'], metadata['width']) == (2160, 3840), 'retained metadata frame/raster mismatch')
    source_path = Path(a.data_root).resolve() / 'outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_00/source.npy'
    require(str(source_path) == '/home/itec/emanuele/pointstream-data/outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_00/source.npy', 'immutable source000 path required')
    source96 = np.load(source_path, allow_pickle=False, mmap_mode='r')
    require(source96.shape == (96, 2160, 3840, 3) and source96.dtype == np.uint8 and rgb_digest(source96) == SOURCE_RGB, 'full96 source identity mismatch')
    source_file_sha256 = digest(source_path)
    delivered96 = np.load(upstream / INPUTS[1], allow_pickle=False, mmap_mode='r')
    require(delivered96.shape == (96, 2160, 3840, 3) and delivered96.dtype == np.uint8 and rgb_digest(delivered96) == receipt['frames_rgb_sha256'], 'full96 decoded RGB identity mismatch')
    source, delivered = source96[:a.frames], delivered96[:a.frames]
    ff = resolve_ffmpeg()
    require(reg['tools'].get(ff.path) == digest(ff.path), 'resolved FFmpeg must be pinned')
    out = Path(a.out).resolve()
    require(out != upstream and upstream not in out.parents and out not in upstream.parents, 'output must be separate from upstream')
    out.mkdir(parents=True, exist_ok=False)
    calls = []
    for name, clip in (('reference', source), ('delivered', delivered)):
        dest = out / (name + '.y4m')
        start = time.monotonic()
        _write_y4m_clip(dest, clip, ff.path)
        calls.append({'label': name + '-rgb-to-yuv', 'argv': [ff.path, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '3840x2160', '-r', '24', '-i', '-', '-pix_fmt', 'yuv420p', str(dest)], 'seconds': time.monotonic() - start, 'policy': 'pinned legacy RGB24 to YUV420p; FFmpeg default matrix/range;24fps'})
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
    metrics, scorer = [], SsimMetric()
    for i, (ref, got) in enumerate(zip(source, delivered, strict=True)):
        diff = rgb_to_luma(ref).astype(np.float64) - rgb_to_luma(got).astype(np.float64)
        mse = float(np.mean(diff**2))
        ssim = float(scorer.score(ref[None], got[None]))
        require(math.isfinite(mse) and math.isfinite(ssim), 'nonfinite pixel metric')
        metrics.append({'frame': i, 'vmaf': values[i], 'mse_y': mse, 'psnr_y': psnr(mse), 'ssim': ssim})
    pooled = float(np.mean([f['mse_y'] for f in metrics]))
    require(digest(source_path) == source_file_sha256, 'source changed during scoring')
    require(digest(reg_path) == registration_sha256 and digest(__file__) == reg['worker_sha256'] and digest(helper) == reg['helper_sha256'], 'registration or scorer/helper changed during scoring')
    require(all(digest(upstream / name) == arm['input_sha256'][name] for name in INPUTS), 'retained upstream artifacts changed during scoring')
    result = {'diagnostic_scoring_complete': True, 'parent_two_window_run_complete': False,
              'paper_evidence': False, 'instrument_only': a.frames == 12,
              'artifact_origin': 'retained source000 output from interrupted legacy-derived recertification',
              'arm': arm, 'registration_sha256': registration_sha256, 'worker_sha256': digest(__file__), 'helper_sha256': digest(helper), 'legacy_revision': revision,
              'historical_package_bytes': package.stat().st_size, 'historical_complete_envelope_bytes': None,
              'historical_full96_package_reused': True, 'scored_frames': a.frames, 'upstream_decoded_frames': 96,
              'source': {'path': str(source_path), 'full96_rgb_sha256': SOURCE_RGB, 'file_sha256': source_file_sha256, 'selected_rgb_sha256': rgb_digest(source)},
              'receiver_full96_rgb_sha256': receipt['frames_rgb_sha256'], 'receiver_selected_rgb_sha256': rgb_digest(delivered),
              'receiver_boundary': receipt['isolation'], 'receipt': receipt, 'per_frame': metrics,
              'joined': {'vmaf': float(np.mean(values)), 'mse_y': pooled, 'psnr_y': psnr(pooled), 'ssim': float(np.mean([f['ssim'] for f in metrics]))},
              'psnr_identity_policy': 'null means positive infinity at zero MSE; pooled equal-frame MSE, not mean frame PSNR',
              'calls': calls, 'command': sys.argv, 'tools_and_libraries': dependencies,
              'legacy_metric_code_sha256': {str(root / rel): digest(root / rel) for rel in ('src/components/codec/frames.py', 'src/components/codec/tools.py', 'src/components/metrics/vmaf.py', 'src/components/metrics/ssim.py', 'src/components/metrics/frames.py')},
              'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'ffmpeg': subprocess.check_output([ff.path, '-version'], text=True)},
              'output_sha256': {x.name: digest(x) for x in out.iterdir() if x.is_file()},
              'resources': {'seconds': time.monotonic() - began, 'self': list(resource.getrusage(resource.RUSAGE_SELF)), 'children': list(resource.getrusage(resource.RUSAGE_CHILDREN)), 'affinity': sorted(os.sched_getaffinity(0)), 'address_space_limit_bytes': 48 * 1024**3, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'caveat': 'Linux process rusage; no measured concurrent aggregate memory or live-codec latency'},
              'scope': 'retrospective source000 diagnostic only; no historical model-match assertion, two-window completion, recertification, motion/task fidelity, RD win, or generalization claim; original registration hash identifies parent provenance, not prospective registration of these retrospective metrics;12-frame smoke reuses historical full96 package and is instrument-only; no original manifest exists, so complete historical envelope bytes are unknown; receipt gives observed audit denial evidence, not native OS isolation proof'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
