"""New source000 sender conditioned only on freshly window-collected model guides.

Registration: status=frozen_before_execution, worker_sha256, receiver_sha256,
legacy_commit/files, tools_and_libraries, data_root_logical/canonical aliases, source_files (exact chunk00 source.npy),
source96_rgb_sha256 (single SHA string), guide_root, guide_receipt_sha256,
guide_registration_sha256, guide_artifact_sha256 (exact complete artifact map),
source000_guide_masks, source000_initial_appearance, and arms matching config/
rung/frames/memory/appearance policy. Legacy config SHA is compact sorted JSON.
Guide collector must be frozen full96, complete with every frame retained.
No old track/mask loader or source028 access. Prefix arms use full96 collected
guides and are mechanics controls with disclosed conditioning context.
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

_OWNED_OUTPUT = None

REVISION = '274638bdae7f5bd63c4f834a24804c0e14ed8d83'
SOURCE_RGB_SHA256 = '09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573'
REQUIRED_LEGACY = ('experiments/tier/gate_a_long_context.py',
                   'experiments/tier/low_rate_sweep.py', 'experiments/tier/low_rate_clips.py',
                   'src/runner/run.py', 'src/runner/stages.py', 'src/runner/client.py',
                   'src/components/background/plate.py', 'src/components/background/strategy.py',
                   'src/components/appearance/compressed.py', 'src/pipeline/reconstruction/reconstruct.py',
                   'config/tier_balanced.yaml')


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
    require(count in (2,12,48,96), 'unsupported registered prefix')
    require(dataclasses.is_dataclass(clip), 'legacy dataclass clip required')
    require(clip.frames.shape[0] == 96 and clip.masks.shape[0] == 96, 'full96 source/mask alignment required')
    objects = []
    for item in clip.objects:
        require(item.mask is None or item.mask.shape[0] == 96, 'object guide alignment required')
        if item.frame_index < count:
            objects.append(dataclasses.replace(item, mask=item.mask[:count] if item.mask is not None else None))
    return dataclasses.replace(clip, n_frames=count, frames=clip.frames[:count], masks=clip.masks[:count], objects=tuple(objects))


@dataclasses.dataclass(frozen=True)
class SceneClip:
    # Structural legacy LongSceneClip adapter; no dataset loader import.
    video: str
    scene: str
    context_id: str
    n_frames: int
    frames: object
    masks: object
    objects: tuple
    paste_back_mae: float = 0.0
    is_eligible: bool = True
    route: str = 'pointstream'
    failure_reasons: tuple = ()


def build_clip(source, objects, *, expected_shape=(96, 2160, 3840, 3)):
    import numpy as np
    require(source.dtype == np.uint8 and source.shape == expected_shape, 'full96 RGB source shape required')
    require(len({o.object_id for o in objects}) == len(objects), 'duplicate guide identity')
    union = np.zeros(source.shape[:3], dtype=bool)
    appearances = []
    for item in objects:
        # Player class is validated in the immutable guide receipt, not an
        # unsupported field on clean274 ObjectRequest. Role IDs survive adaptation.
        require(item.object_id in ('player_far','player_near'), 'explicit fresh production player roles required')
        require(item.mask.dtype == np.bool_ and item.mask.shape == source.shape[:3], 'full96 bool mask alignment required')
        frame = int(item.frame_index)
        require(frame == item.frame_index and 0 <= frame < source.shape[0], 'appearance frame outside source')
        require(len(item.bbox) == 4 and all(int(v) == v for v in item.bbox), 'integral bbox required')
        x1, y1, x2, y2 = map(int, item.bbox)
        require(0 <= x1 < x2 <= source.shape[2] and 0 <= y1 < y2 <= source.shape[1], 'bbox outside selected source')
        crop = source[frame, y1:y2, x1:x2]
        require(item.appearance.dtype == np.uint8 and np.array_equal(item.appearance, crop), 'appearance must equal selected RGB crop')
        require(item.conditioning is None and item.supplied_crop is None, 'alternate pixel conditioning forbidden')
        union |= item.mask
        appearances.append({'object_id': item.object_id, 'frame_index': frame, 'source_frame_index': 38+frame,
            'bbox': list(item.bbox), 'appearance_shape': list(crop.shape), 'appearance_rgb_sha256': rgb_digest(crop[None]),
            'appearance_input_policy': 'reference_cutout'})
    context = 'source000-rgb-' + rgb_digest(source)
    return SceneClip('alcaraz_highlights', 'scene_000', context, len(source), source, union, tuple(objects)), appearances


def load_fresh_guides(guide_root, reg, source_path):
    import numpy as np
    from src.pipeline.reconstruction.reconstruct import ObjectRequest
    guide_root = Path(guide_root).resolve()
    receipt_path = guide_root / 'receipt.json'
    require(digest(receipt_path) == reg['guide_receipt_sha256'], 'guide receipt changed')
    receipt = json.loads(receipt_path.read_text())
    guide_registration = guide_root / 'registration.json'
    require(digest(guide_registration) == reg['guide_registration_sha256'] == receipt['registration_sha256'], 'guide registration changed')
    collector = json.loads(guide_registration.read_text())
    require(collector['config']['selector'] == 'production_HeuristicSelector' and collector['config']['classes'] == ['player'], 'production player guide contract required')
    require(collector['status'] == 'frozen' and collector['frames'] == 96 and collector['source_rgb_sha256'] == SOURCE_RGB_SHA256,
            'full96 frozen within-window collection required')
    require(Path(collector['source_path']).resolve() == source_path and collector['source_file_sha256'] == reg['source_files'][str(source_path)], 'collector source differs')
    require(Path(collector['output_path']).resolve() == guide_root, 'collector output root identity differs')
    require(receipt['status'] == 'complete' and receipt['frame_count'] == 96 and receipt['source_frame_interval'] == [38,134], 'complete full96 guides required')
    require(receipt['artifact_sha256'] == reg['guide_artifact_sha256'], 'guide artifact map differs')
    for filename, expected in receipt['artifact_sha256'].items():
        path = (guide_root / filename).resolve()
        require(path.parent == guide_root and digest(path) == expected, 'guide artifact changed or escaped root')
    records = json.loads((guide_root / 'frames.json').read_text())
    require(len(records) == 96 and [r['frame_index'] for r in records] == list(range(96)) and
            [r['source_frame_index'] for r in records] == list(range(38,134)) and all(r['status'] == 'complete' for r in records),
            'missing/unprocessed guide frames forbidden')
    objects = []
    for row in receipt['objects']:
        require(row['object_class'] == 'player' and row['object_id'] in ('player_far','player_near'), 'unexpected guide class')
        require(row['mask_file'] in receipt['artifact_sha256'] and row['appearance_file'] in receipt['artifact_sha256'], 'unhashed object artifact')
        objects.append(ObjectRequest(object_id=row['object_id'], frame_index=row['frame_index'], bbox=tuple(row['bbox']),
            mask=np.load(guide_root / row['mask_file'], mmap_mode='r', allow_pickle=False),
            appearance=np.load(guide_root / row['appearance_file'], allow_pickle=False)))
    source = np.load(source_path, mmap_mode='r', allow_pickle=False)
    require(all(row['rgb_sha256'] == rgb_digest(source[i:i+1]) for i,row in enumerate(records)), 'perframe guide RGB input identity differs')
    return tuple(objects), receipt, collector


def main():
    global _OWNED_OUTPUT
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-root', 'data-root', 'receiver-script', 'registration', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--rung', choices=('C2', 'C3'), required=True)
    p.add_argument('--frames', type=int, choices=(2, 12, 48, 96), required=True)
    p.add_argument('--source-window-index', type=int, choices=(0,), default=0)
    p.add_argument('--appearance-input-policy', choices=('reference_cutout',), default='reference_cutout')
    p.add_argument('--memory-gib', type=int, choices=(48, 128, 192), default=128)
    p.add_argument('--registration-sha256', required=True)
    a = p.parse_args()
    began = time.monotonic()
    require(sys.platform == 'linux' and 0 < len(os.sched_getaffinity(0)) <= 8, 'Linux CPU affinity<=8 required')
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS,0)))
    resource.setrlimit(resource.RLIMIT_AS, (a.memory_gib * 1024**3, a.memory_gib * 1024**3))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PS_CODEC_TIMEOUT_SECONDS'] = '600'
    os.environ['PS_CODEC_MAX_ATTEMPTS'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VMAF_THREADS', 'PS_CODEC_THREADS', 'SSIM_THREADS'):
        os.environ[key] = '8'
    root = Path(a.legacy_root).resolve()
    data = Path(a.data_root).resolve()
    logical_data = Path('/home/itec/emanuele/pointstream-data')
    require(Path(a.data_root).absolute() == logical_data or Path(a.data_root).absolute() == data, 'unrecognized data root spelling')
    os.environ['PS_DATA_ROOT'] = str(data)
    out = Path(a.out).resolve()
    require(not out.exists() and (data / 'audits') in out.parents, 'fresh external-data output required; never resume/replay')
    receiver = Path(a.receiver_script).resolve()
    reg_path = Path(a.registration).resolve()
    registration_sha256 = digest(reg_path)
    require(registration_sha256 == a.registration_sha256, 'registration changed')
    reg = json.loads(reg_path.read_text())
    require(reg['data_root_logical'] == str(logical_data) and Path(reg['data_root_canonical']).is_absolute(), 'registered root aliases required')
    require(logical_data.resolve() == data == Path(reg['data_root_canonical']), 'registered canonical root identity mismatch')
    require(str(Path(a.data_root).absolute()) in (reg['data_root_logical'], reg['data_root_canonical']), 'only registered root aliases accepted')
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
    require(reg['source96_rgb_sha256'] == SOURCE_RGB_SHA256, 'exact source000 RGB pin required')
    source_path = data / 'outputs/gate-a-vvc-webp-n96-run2/points/C2.run/chunk_00/source.npy'
    source_paths = [source_path]
    require(set(reg['source_files']) == {str(source_path)}, 'only original source000 file permitted')
    require(digest(source_path) == reg['source_files'][str(source_path)], 'original source file changed')
    guide_root = Path(reg['guide_root']).resolve()
    require((data / 'audits') in guide_root.parents, 'guides must be external audit artifacts')
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    from experiments.tier.gate_a_long_context import RUNGS, configure_rung
    from experiments.tier.low_rate_sweep import _no_generator, require_run_accepts_context_ids
    from src.runner.config_io import load_tier
    from src.components.codec.tools import resolve_ffmpeg, resolve_encoder
    import src.runner.client as client
    source = np.load(source_path, mmap_mode='r', allow_pickle=False)
    require(source.dtype == np.uint8 and source.shape == (96,2160,3840,3) and rgb_digest(source) == SOURCE_RGB_SHA256,
            'original selected full96 RGB differs')
    objects, guide_receipt, guide_registration = load_fresh_guides(guide_root, reg, source_path)
    full, full_crop_identities = build_clip(source, objects)
    guide_masks = [{'object_id': o.object_id, 'shape': list(o.mask.shape), 'sha256': rgb_digest(o.mask),
                    'first_bbox': list(o.bbox), 'first_frame': o.frame_index} for o in full.objects]
    require(guide_masks == reg['source000_guide_masks'], 'fresh full96 guide identities changed')
    require(full_crop_identities == reg['source000_initial_appearance'], 'fresh RGB appearance identities changed')
    selected = slice_clip(full, a.frames)
    require(selected.objects and all(0 <= o.frame_index < a.frames for o in selected.objects), 'prefix has no aligned foreground')
    identities = [{'scene': 'scene_000', 'shape': list(source.shape), 'rgb_sha256': SOURCE_RGB_SHA256,
                   'source_file': str(source_path), 'source_file_sha256': reg['source_files'][str(source_path)]}]
    config = configure_rung(load_tier('balanced'), next(r for r in RUNGS if r.name == a.rung))
    config_dict = dataclasses.asdict(config)
    config_sha256 = hashlib.sha256(json.dumps(config_dict, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
    arm = {'rung': a.rung, 'frames': a.frames, 'source_window_index': 0, 'memory_gib': a.memory_gib, 'config_sha256': config_sha256, 'appearance_input_policy': a.appearance_input_policy}
    require(arm in reg['arms'], 'exact source000 sender arm absent from frozen registration')
    require(config.run.seed == 1337 and not config.lattice.generation and not config.lattice.residual and not config.lattice.pose, 'original seed/lattice contract changed')
    require(config.run.max_frames is None or config.run.max_frames >= a.frames, 'configured frame truncation forbidden')
    for tool in (resolve_ffmpeg(), resolve_encoder('vvc')):
        require(reg['tools_and_libraries'].get(tool.path) == digest(tool.path), 'resolved native executable must be pinned')
    crop_identities = [record for record in full_crop_identities if record['frame_index'] < a.frames]
    # Only selected source000 prefix reaches canonical preparation. Full96 fresh
    # guide conditioning is disclosed for shorter mechanics prefixes.
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
    _OWNED_OUTPUT = out
    (out / 'execution-status.json').write_text(json.dumps({'status':'started','registration_sha256':registration_sha256})+'\n')
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
    except Exception as exc:
        (out / 'execution-status.json').write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'})+'\n')
        raise
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
    require(digest(guide_root / 'receipt.json') == reg['guide_receipt_sha256'] and digest(guide_root / 'registration.json') == reg['guide_registration_sha256'], 'guide provenance changed during run')
    require(all(digest(guide_root / name) == sha for name, sha in reg['guide_artifact_sha256'].items()), 'guide artifact changed during run')
    report = {'complete': True, 'paper_evidence': False, 'arm': arm, 'legacy_revision': revision,
              'registration_sha256': registration_sha256, 'worker_sha256': digest(__file__), 'receiver_script_sha256': digest(receiver),
              'physical_bytes': captured[0]['package_bytes'] + (out / 'manifest.json').stat().st_size,
              'package_bytes': captured[0]['package_bytes'], 'manifest_bytes': (out / 'manifest.json').stat().st_size,
              'manifest_sha256': digest(out / 'manifest.json'), 'package_sha256': captured[0]['package_sha256'],
              'receiver': captured[0]['receipt'], 'receiver_file_sha256': captured[0]['receiver_file_sha256'],
              'source96_rgb_sha256': SOURCE_RGB_SHA256, 'verified_original_sources': identities,
              'source_guide_masks': guide_masks, 'initial_appearance': crop_identities, 'full96_initial_appearance': full_crop_identities,
              'input_context_contract': {'encoder_source_windows': ['scene_000'], 'encoder_frame_interval': [38,38+a.frames],
                  'canonical_preparation_source_count': 1, 'guide_collection_source_interval': [38,134],
                  'prefix_conditioning': 'full96 collected guides sliced for prefix; shorter prefixes mechanics only',
                  'context_identifier': full.context_id, 'fps': 24, 'source028_access': False,
                  'checkpoint_or_prepared_background_reused': False, 'appearance_input_policy': 'reference_cutout',
                  'guide_provenance': 'fresh frozen window collector; model-derived, not independent task truth',
                  'class_mapping': 'raw YOLO person -> production heuristic player_far/player_near; explicit posthoc selected-domain contract'},
              'config': config_dict, 'config_sha256': config_sha256,
              'data_root_logical': reg['data_root_logical'], 'data_root_canonical': str(data),
              'guide_receipt_sha256': reg['guide_receipt_sha256'], 'guide_registration_sha256': reg['guide_registration_sha256'],
              'guide_artifact_sha256': reg['guide_artifact_sha256'], 'guide_collector_registration': guide_registration,
              'legacy_files': reg['legacy_files'], 'tools_and_libraries': reg['tools_and_libraries'],
              'observed_subprocess_calls': calls, 'receiver_call': captured[0]['receiver_argv'], 'command': sys.argv,
              'resource_observation': {'worker_seconds': time.monotonic()-began, 'sender_run_seconds': time.monotonic()-sender_start, 'self_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'child_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'self_rusage': list(resource.getrusage(resource.RUSAGE_SELF)), 'child_rusage': list(resource.getrusage(resource.RUSAGE_CHILDREN)), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'address_space_limit_gib': a.memory_gib, 'nice': os.getpriority(os.PRIO_PROCESS,0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'caveat': 'process peaks are not measured simultaneous aggregate memory or live-codec latency; external supervisor must enforce registered walltime'},
              'sender_path': 'production run with exact pointstream_e1 encoder arguments; mandatory DAG metrics retained; redundant outer headline scoring omitted; internal DAG scores are not promoted as fresh-boundary evidence',
              'scope': 'new source000 packet conditioned on fresh within-window guides; not original C2/C3 recertification; initial-only receiver behavior retained, trajectory repair separate; receiver audit denial not OS sandbox proof; separate frozen fresh-boundary scoring required;2/12/48 are mechanics controls with full96 guide context; no independent task truth, motion/task/generalization or winner claim'}
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')
    (out / 'execution-status.json').write_text(json.dumps({'status':'complete','report_sha256':digest(out / 'report.json')})+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if _OWNED_OUTPUT is not None:
            (_OWNED_OUTPUT / 'execution-status.json').write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'})+'\n')
        raise
