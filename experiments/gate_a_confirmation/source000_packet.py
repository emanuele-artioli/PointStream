"""New source000-only legacy sender packet; never replay or recertification.

Registration has status, worker_sha256, receiver_sha256, legacy_commit,
legacy_files (relative path->SHA; required paths below), tools_and_libraries
(absolute paths->SHA including Python), source_files (both original NPY absolute
paths->SHA), source96_rgb_sha256 (both windows), source000_guide_masks (full96
mask identities), source000_initial_appearance (policy-specific full96 initial crop records
including pixels SHA), and arms: rung/frames/source_window_index=0/memory_gib/
config_sha256/appearance_input_policy. Config SHA is compact sorted dataclasses JSON (default=str).
Run only under a supervised bounded CPU job; this worker launches no job.
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

REVISION = '274638bdae7f5bd63c4f834a24804c0e14ed8d83'
SOURCE_HASHES = ['09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573',
                 '2009a1bb6934c4109f16f73951acb2af5a22d91a3cb15f7f79fc3dc883bb6c6b']
REQUIRED_LEGACY = ('experiments/long_scenes/loader.py', 'experiments/headroom/real.py',
                   'experiments/tier/gate_a_identity.py', 'experiments/tier/gate_a_long_context.py',
                   'experiments/tier/low_rate_sweep.py', 'experiments/tier/low_rate_clips.py',
                   'src/runner/run.py', 'src/runner/stages.py', 'src/runner/client.py',
                   'src/components/background/plate.py', 'src/components/background/strategy.py',
                   'src/components/appearance/compressed.py', 'src/pipeline/reconstruction/reconstruct.py',
                   'config/tier_balanced.yaml', 'manifests/bp46_long_tennis_scenes.json')


def require(c, message):
    if not c:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def rgb_digest(array):
    h = hashlib.sha256()
    for frame in array:
        h.update(frame.tobytes(order='C'))
    return h.hexdigest()


def slice_clip(clip, count):
    require(dataclasses.is_dataclass(clip), 'legacy dataclass clip required')
    require(clip.frames.shape[0] == 96 and clip.masks.shape[0] == 96, 'full96 source/mask alignment required')
    objects = []
    for item in clip.objects:
        require(item.mask is None or item.mask.shape[0] == 96, 'object guide alignment required')
        if item.frame_index < count:
            objects.append(dataclasses.replace(item, mask=item.mask[:count] if item.mask is not None else None))
    return dataclasses.replace(clip, n_frames=count, frames=clip.frames[:count], masks=clip.masks[:count], objects=tuple(objects))


def apply_appearance_policy(clip, policy):
    """Preserve ObjectRequest metadata; replace only initial RGB appearance.

    Loader bboxes are half-open raster slices. Reject out-of-bounds metadata
    instead of changing its coordinates. No RGB/BGR conversion is introduced:
    legacy loader produces RGB arrays; compressed backend encodes/decodes the
    same channel array through OpenCV without explicit conversion.
    """
    import numpy as np
    require(policy in ('reference_cutout', 'cached_associated'), 'unsupported appearance input policy')
    require(clip.frames.dtype == np.uint8 and clip.frames.ndim == 4 and clip.frames.shape[-1] == 3, 'source RGB uint8 raster required')
    n, height, width, _ = clip.frames.shape
    objects = []
    records = []
    for item in clip.objects:
        require(dataclasses.is_dataclass(item), 'legacy ObjectRequest dataclass required')
        frame = int(item.frame_index)
        require(frame == item.frame_index and 0 <= frame < n, 'initial appearance frame outside selected source')
        require(len(item.bbox) == 4 and all(int(v) == v for v in item.bbox), 'integral half-open bbox required')
        x1, y1, x2, y2 = map(int, item.bbox)
        clipped = (max(0,x1),max(0,y1),min(width,x2),min(height,y2))
        require(clipped == (x1,y1,x2,y2) and x1 < x2 and y1 < y2, 'bbox must already be clipped to source raster; metadata preserved')
        if policy == 'reference_cutout':
            appearance = np.ascontiguousarray(clip.frames[frame,y1:y2,x1:x2]).copy()
        else:
            appearance = np.asarray(item.appearance)
        require(appearance.dtype == np.uint8 and appearance.shape == (y2-y1,x2-x1,3), 'initial crop shape/raster differs from bbox')
        require(item.conditioning is None and item.supplied_crop is None, 'alternate conditioning or supplied pixel path forbidden')
        updated = dataclasses.replace(item, appearance=appearance)
        require(updated.object_id == item.object_id and updated.bbox == item.bbox and updated.frame_index == item.frame_index and updated.mask is item.mask, 'appearance adaptation changed guide metadata')
        objects.append(updated)
        records.append({'object_id': item.object_id, 'frame_index': frame, 'source_frame_index': 38+frame,
                        'bbox': list(item.bbox), 'appearance_shape': list(appearance.shape),
                        'appearance_rgb_sha256': hashlib.sha256(appearance.tobytes(order='C')).hexdigest(),
                        'appearance_input_policy': policy})
    return dataclasses.replace(clip, objects=tuple(objects)), records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-root', 'data-root', 'receiver-script', 'registration', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--rung', choices=('C2', 'C3'), required=True)
    p.add_argument('--frames', type=int, choices=(2, 12, 48, 96), required=True)
    p.add_argument('--source-window-index', type=int, choices=(0,), default=0)
    p.add_argument('--appearance-input-policy', choices=('reference_cutout','cached_associated'), default='reference_cutout')
    p.add_argument('--memory-gib', type=int, choices=(48, 128, 192), default=128)
    a = p.parse_args()
    began = time.monotonic()
    require(sys.platform == 'linux' and 0 < len(os.sched_getaffinity(0)) <= 8, 'Linux CPU affinity<=8 required')
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (a.memory_gib * 1024**3, a.memory_gib * 1024**3))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PS_CODEC_TIMEOUT_SECONDS'] = '600'
    os.environ['PS_CODEC_MAX_ATTEMPTS'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VMAF_THREADS', 'PS_CODEC_THREADS', 'SSIM_THREADS'):
        os.environ[key] = '8'
    root = Path(a.legacy_root).resolve()
    data = Path(a.data_root).resolve()
    require(str(data) == '/home/itec/emanuele/pointstream-data', 'original data root required')
    os.environ['PS_DATA_ROOT'] = str(data)
    out = Path(a.out).resolve()
    require(not out.exists() and data in out.parents, 'fresh external-data output required; never resume/replay')
    receiver = Path(a.receiver_script).resolve()
    reg_path = Path(a.registration).resolve()
    registration_sha256 = digest(reg_path)
    reg = json.loads(reg_path.read_text())
    require(reg.get('status') == 'frozen_before_execution' and reg['worker_sha256'] == digest(__file__) and reg['receiver_sha256'] == digest(receiver), 'frozen worker/receiver required')
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == REVISION == reg['legacy_commit'] and not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']), 'clean pinned legacy required')
    require(set(REQUIRED_LEGACY) <= set(reg['legacy_files']), 'source/config/encoder route pins required')
    for rel, expected in reg['legacy_files'].items():
        path = (root / rel).resolve()
        require(root in path.parents and digest(path) == expected, 'legacy source/config changed: ' + rel)
    require(reg['tools_and_libraries'] and any('libvmaf' in x for x in reg['tools_and_libraries']), 'native library pins including libvmaf required')
    for path, expected in reg['tools_and_libraries'].items():
        require(Path(path).is_absolute() and digest(path) == expected, 'native dependency changed: ' + path)
    require(reg['tools_and_libraries'].get(str(Path(sys.executable).resolve())) == digest(Path(sys.executable).resolve()), 'Python binary pin required')
    require(reg['source96_rgb_sha256'] == SOURCE_HASHES, 'both original source identities required')
    source_paths = [data / f'outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_{i:02d}/source.npy' for i in range(2)]
    require(set(reg['source_files']) == {str(x) for x in source_paths}, 'exact original source file pins required')
    for path in source_paths:
        require(digest(path) == reg['source_files'][str(path)], 'original source file changed')
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    from experiments.tier.gate_a_identity import verify_source_clips, verify_manifest
    from experiments.tier.gate_a_long_context import RUNGS, configure_rung
    from experiments.tier.low_rate_sweep import _no_generator, require_run_accepts_context_ids
    from src.runner.config_io import load_tier
    from src.components.codec.tools import resolve_ffmpeg, resolve_encoder
    from src.contracts import paths
    import src.runner.client as client
    require(data in paths.outputs().resolve().parents or paths.outputs().resolve() == data, 'legacy outputs outside selected data root')
    clips = verify_source_clips(n_frames=96)
    require(len(clips) == 2 and [c.scene for c in clips] == ['scene_000', 'scene_028'], 'original two-window identity validation failed')
    identities = []
    for i, clip in enumerate(clips):
        require(clip.frames.shape == (96, 2160, 3840, 3) and clip.frames.dtype == np.uint8 and rgb_digest(clip.frames) == SOURCE_HASHES[i], 'loaded original RGB identity mismatch')
        identities.append({'scene': clip.scene, 'shape': list(clip.frames.shape), 'rgb_sha256': SOURCE_HASHES[i], 'source_file': str(source_paths[i]), 'source_file_sha256': reg['source_files'][str(source_paths[i])]})
    full = clips[0]
    guide_masks = [{'object_id': o.object_id, 'shape': list(o.mask.shape), 'sha256': rgb_digest(o.mask), 'first_bbox': list(o.bbox), 'first_frame': o.frame_index} for o in full.objects]
    require(guide_masks == reg['source000_guide_masks'], 'selected full96 model guides changed')
    full, full_crop_identities = apply_appearance_policy(full, a.appearance_input_policy)
    require(full_crop_identities == reg['source000_initial_appearance'], 'registered policy-specific initial crop pixels changed')
    # Discard the identity-only second window before sender construction.
    del clips
    selected = slice_clip(full, a.frames)
    require(selected.objects and all(0 <= o.frame_index < a.frames for o in selected.objects), 'prefix has no aligned foreground')
    config = configure_rung(load_tier('balanced'), next(r for r in RUNGS if r.name == a.rung))
    config_dict = dataclasses.asdict(config)
    config_sha256 = hashlib.sha256(json.dumps(config_dict, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
    arm = {'rung': a.rung, 'frames': a.frames, 'source_window_index': 0, 'memory_gib': a.memory_gib, 'config_sha256': config_sha256, 'appearance_input_policy': a.appearance_input_policy}
    require(arm in reg['arms'], 'exact source000 sender arm absent from frozen registration')
    require(config.run.seed == 1337 and not config.lattice.generation and not config.lattice.residual and not config.lattice.pose, 'original seed/lattice contract changed')
    require(config.run.max_frames is None or config.run.max_frames >= a.frames, 'configured frame truncation forbidden')
    for tool in (resolve_ffmpeg(), resolve_encoder('vvc')):
        require(reg['tools_and_libraries'].get(tool.path) == digest(tool.path), 'resolved native executable must be pinned')
    verified_manifest = verify_manifest()
    crop_identities = [record for record in full_crop_identities if record['frame_index'] < a.frames]
    # Validation loads both windows, but ONLY selected source000 is handed to the
    # encoder/canonical prepass. No prepared checkpoint or external plate enters.
    chunks = [np.asarray(selected.frames)]
    object_tuples = (selected.objects,)
    contexts = (selected.context_id,)
    require(len(chunks) == len(object_tuples) == len(contexts) == 1, 'single-context sender boundary required')
    calls = []
    def audit(event, args):
        if event == 'subprocess.Popen':
            calls.append({'executable': os.fsdecode(args[0]), 'argv': list(args[1]) if not isinstance(args[1], str) else args[1], 'observed_at_seconds': time.monotonic() - began})
    sys.addaudithook(audit)
    out.mkdir(parents=True, exist_ok=False)
    captured = []
    original = client.serialize_client_request
    def serialize(**kwargs):
        require(not captured, 'unexpected second serialized scene')
        require(kwargs['frame_count'] == a.frames and kwargs['height'] == 2160 and kwargs['width'] == 3840, 'serialized scene shape differs from arm')
        payload = original(**kwargs)
        packet = out / 'scene.npz'
        with packet.open('xb') as stream:
            stream.write(payload)
        decoded = out / 'receiver.npy'
        argv = [sys.executable, str(receiver), '--legacy-root', str(root), '--package', str(packet), '--output', str(decoded), '--deny-root', str(data)]
        start = time.monotonic()
        subprocess.run(argv, check=True, timeout=600)
        receipt = json.loads(decoded.with_suffix('.receipt.json').read_text())
        frames = np.load(decoded, allow_pickle=False, mmap_mode='r')
        require(receipt['package_bytes'] == len(payload) and receipt['package_sha256'] == digest(packet), 'fresh receiver package identity mismatch')
        require(frames.dtype == np.uint8 and frames.shape == (a.frames, 2160, 3840, 3) and receipt['frames_shape'] == list(frames.shape) and receipt['frames_rgb_sha256'] == rgb_digest(frames) and receipt['violations'] == [], 'fresh payload-only receiver identity/guard mismatch')
        captured.append({'package_sha256': digest(packet), 'package_bytes': len(payload), 'receiver_file_sha256': digest(decoded), 'receiver_rgb_sha256': receipt['frames_rgb_sha256'], 'receipt': receipt, 'receiver_seconds': time.monotonic() - start, 'receiver_argv': argv})
        return payload
    client.serialize_client_request = serialize
    run = require_run_accepts_context_ids()
    sender_start = time.monotonic()
    # Identical sender arguments to pointstream_e1. Production run includes its
    # mandatory DAG evaluation; only redundant outer headline scoring is omitted.
    try:
        result = run(config, chunks, bind_generator_fn=_no_generator, objects=object_tuples, context_ids=contexts)
    finally:
        client.serialize_client_request = original
    require(len(captured) == 1 and len(result.chunks) == 1, 'single scene emission/result required')
    with np.load(out / 'scene.npz', allow_pickle=False) as packet:
        metadata = json.loads(packet['metadata'].tobytes())
    require(metadata['frame_count'] == a.frames and len(metadata['background']['homographies']) == a.frames, 'complete packet metadata mismatch')
    manifest = {'schema': 1, 'fps': 24, 'package': 'scene.npz'}
    with (out / 'manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, separators=(',', ':')) + '\n')
    require(digest(reg_path) == registration_sha256 and digest(__file__) == reg['worker_sha256'] and digest(receiver) == reg['receiver_sha256'], 'worker/registration changed during run')
    require(all(digest(path) == reg['source_files'][str(path)] for path in source_paths), 'original source changed during run')
    report = {'complete': True, 'paper_evidence': False, 'arm': arm, 'legacy_revision': revision,
              'registration_sha256': registration_sha256, 'worker_sha256': digest(__file__), 'receiver_script_sha256': digest(receiver),
              'physical_bytes': captured[0]['package_bytes'] + (out / 'manifest.json').stat().st_size,
              'package_bytes': captured[0]['package_bytes'], 'manifest_bytes': (out / 'manifest.json').stat().st_size,
              'manifest_sha256': digest(out / 'manifest.json'), 'package_sha256': captured[0]['package_sha256'],
              'receiver': captured[0]['receipt'], 'receiver_file_sha256': captured[0]['receiver_file_sha256'],
              'source96_rgb_sha256': SOURCE_HASHES[0], 'verified_original_sources': identities,
              'source_guide_masks': guide_masks, 'initial_appearance': crop_identities, 'full96_initial_appearance': full_crop_identities,
              'input_context_contract': {'encoder_source_windows': ['scene_000'], 'encoder_frame_interval': [38, 38+a.frames], 'canonical_preparation_source_count': 1, 'full96_identity_validation_only': ['scene_000', 'scene_028'], 'checkpoint_or_prepared_background_reused': False, 'fps': 24, 'crop_temporal_provenance': 'loader filters track pairs to selected96 interval before selecting earliest crop; prefix includes only objects whose first frame falls in prefix', 'appearance_input_policy': a.appearance_input_policy, 'crop_pixel_provenance': 'reference_cutout copies only selected source000 RGB at preserved bbox/frame; no cached appearance pixels handed to encoder' if a.appearance_input_policy == 'reference_cutout' else 'diagnostic cached selected-frame-associated crop; upstream pixel/temporal creation context unresolved', 'guide_preprocessing_provenance': 'cached model-derived guides/track metadata; not independent task truth; upstream temporal processing scope unresolved', 'pixel_input_access': 'selected source000 raster only, except declared cached guide/geometry preprocessing' if a.appearance_input_policy == 'reference_cutout' else 'unresolved cached-associated appearance context; diagnostic only'},
              'config': config_dict, 'config_sha256': config_sha256, 'appearance_input_policy': a.appearance_input_policy, 'appearance_adaptation': 'new registered source-reference cutouts replace cached initial appearance RGB; original bbox/frame/id/mask guides preserved' if a.appearance_input_policy == 'reference_cutout' else 'cached-associated appearance retained for diagnostics only', 'source_manifest_identity': verified_manifest,
              'legacy_files': reg['legacy_files'], 'tools_and_libraries': reg['tools_and_libraries'],
              'observed_subprocess_calls': calls, 'receiver_call': captured[0]['receiver_argv'], 'command': sys.argv,
              'resource_observation': {'worker_seconds': time.monotonic()-began, 'sender_run_seconds': time.monotonic()-sender_start, 'self_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'child_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'self_rusage': list(resource.getrusage(resource.RUSAGE_SELF)), 'child_rusage': list(resource.getrusage(resource.RUSAGE_CHILDREN)), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'address_space_limit_gib': a.memory_gib, 'nice': os.getpriority(os.PRIO_PROCESS,0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'caveat': 'process peaks are not measured simultaneous aggregate memory or live-codec latency; external supervisor must enforce registered walltime'},
              'sender_path': 'production run with exact pointstream_e1 encoder arguments; mandatory DAG metrics retained; redundant outer headline scoring omitted; internal DAG scores are not promoted as fresh-boundary evidence',
              'scope': 'new standalone source000 packet, not old192frame/two-window recertification or replay; source028 used solely for identity validation, never canonical preparation; observed receiver audit denial is not native OS sandbox proof; reference_cutout resolves appearance pixel access to selected source raster while cached guide/geometry preprocessing temporal scope remains unresolved; cached_associated is diagnostic-only with unresolved crop pixel context; new appearance-input adaptation is not originalC2/C3recertification; separate frozen fresh-boundary scoring required;2/12/48smokes are mechanics controls, not full96RD points; no motion/task/generalization or winner claim'}
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
