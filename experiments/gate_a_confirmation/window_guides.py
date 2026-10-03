"""Frozen source000-only model guide collection; guides are not task truth.

Registration: status=frozen, worker_sha256, source_path, source_file_sha256,
source_rgb_sha256, frames (2/12/48/96), code_files {relative:sha256},
weights {detector:{path,sha256},segmenter:{path,sha256}}, environment
{python_path,python_sha256,packages:{name:version},package_manifest:{name:{kind:RECORD|METADATA|PKG-INFO,sha256}},runtime_files:{absolute:sha}},
config (exact CONFIG), config_sha256, environment_sha256 (compact sorted JSON),
resources {device:cpu|cuda:0,memory_gib,max_tracks,max_seconds,cpu_threads:8},
output_path under the fixed external data audits root. GPU RAM guard samples RSS
before/after model calls and frames; it is not an OS hard bound or transient guard.
The GPU device is logical cuda:0 under exactly one fleet-visible GPU UUID.
No automatic device fallback or download. All output must be fresh/external.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
# Pin native CPU thread pools before importing NumPy or model libraries.
for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env] = '8'
import numpy as np

SOURCE_RGB_SHA256 = '09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573'
CONFIG = {'detector_conf': 0.1, 'segmenter_conf': 0.2, 'tracker_iou_threshold': 0.3,
          'classes': ['player'], 'selector': 'production_HeuristicSelector', 'selector_kwargs': {},
          'selector_defaults': {'roles':['player_far','player_near'], 'midline_height_fraction':0.5,
              'initial_anchor_width_fraction':0.5, 'far_anchor_height_fraction':0.30, 'near_anchor_height_fraction':0.72,
              'area_ratio_max':4.0, 'distance_height_fraction':0.35, 'distance_previous_height_multiple':2.0,
              'distance_diagonal_fraction':0.12, 'stabilize_height_fraction':0.18, 'stabilize_previous_height_multiple':2.2,
              'score_area_scale':0.0015, 'score_center_scale':100.0, 'score_temporal_scale':220.0, 'history_hold':True},
          'selection_change': 'posthoc development player curation after failed all-person pilot; no confidence/box tuning', 'rgb_to_model': 'BGR', 'tracking': 'fresh_past_only_default_recovery',
          'model_predict': {'verbose': False}, 'missing_masks': 'zero_recorded'}
REQUIRED_CODE = ('src/components/detection/yolo.py', 'src/components/detection/parsing.py',
                 'src/components/detection/geometry.py', 'src/components/detection/types.py',
                 'src/components/tracking/tracker.py', 'src/components/tracking/recovery.py',
                 'src/components/segmentation/yolo.py', 'src/components/detection/weights.py',
                 'src/components/detection/__init__.py', 'src/components/tracking/__init__.py',
                 'src/components/selection/heuristic.py', 'src/components/selection/__init__.py', 'config/tier_balanced.yaml',
                 'src/components/segmentation/__init__.py', 'src/components/__init__.py', 'src/__init__.py')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_torch_runtime(torch, environment):
    """Pin the loaded runtime separately from potentially stale distribution metadata."""
    expected = environment.get('torch_runtime')
    require(isinstance(expected, dict), 'loaded torch runtime pins missing')
    actual = {'version': str(torch.__version__), 'cuda': torch.version.cuda,
              'git_version': torch.version.git_version}
    require(actual == expected, 'loaded torch runtime differs from registration')
    version_path = str(Path(torch.version.__file__).resolve())
    require(version_path in environment['runtime_files'], 'loaded torch version source not pinned')
    require(digest(version_path) == environment['runtime_files'][version_path], 'loaded torch version source changed')
    return actual

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def array_digest(array):
    h = hashlib.sha256()
    for frame in array:
        h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()

def rgb_to_bgr(frame):
    require(frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3,
            'uint8 RGB frame required')
    return np.ascontiguousarray(frame[..., ::-1])

def raster_box(box, height, width):
    values = np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=float)
    require(np.isfinite(values).all(), 'nonfinite bbox')
    x1, y1 = np.floor(values[:2]).astype(int)
    x2, y2 = np.ceil(values[2:]).astype(int)
    bounds = (max(0, min(width, x1)), max(0, min(height, y1)),
              max(0, min(width, x2)), max(0, min(height, y2)))
    require(bounds[0] < bounds[2] and bounds[1] < bounds[3], 'empty clipped bbox')
    return tuple(map(int, bounds))

def validate_source(source, frames, full_shape=(96, 2160, 3840, 3)):
    require(frames in (2, 12, 48, 96), 'unregistered prefix')
    require(source.dtype == np.uint8 and source.shape == full_shape, 'original full96 RGB shape required')
    return source[:frames]

def select_with_audit(selector, detections, frame_shape, held_counts):
    from src.components.detection.types import is_person, is_racket
    people = [d for d in detections if is_person(d.class_name)]
    rackets = [d for d in detections if is_racket(d.class_name)]
    far, near = selector._pick_players(people, rackets, width=frame_shape[1], height=frame_shape[0])
    candidates = {'player_far':far, 'player_near':near}
    all_selected = list(selector.select(detections, frame_shape))
    selected = [d for d in all_selected if is_person(d.class_name)]
    audit = {}
    selected_indices = set()
    for item in selected:
        candidate = candidates[item.track_id]
        candidate_indices = [i for i,d in enumerate(detections) if d is candidate]
        raw_indices = [i for i,d in enumerate(detections) if is_person(d.class_name) and d.bbox == item.bbox]
        held = candidate is None or candidate.bbox != item.bbox
        held_counts[item.track_id] = held_counts.get(item.track_id, 0)+1 if held else 0
        selected_indices.update(raw_indices)
        audit[item.track_id] = {'selected_role':item.track_id, 'candidate_raw_indices':candidate_indices,
            'selected_raw_indices':raw_indices, 'history_held':held, 'stale':held,
            'consecutive_held_frames':held_counts[item.track_id], 'detector_matched':bool(raw_indices)}
    excluded = [{'raw_index':i, 'reason':'not selected as on-court player by production heuristic' if is_person(d.class_name)
                  else 'non-player class omitted from player-only output'} for i,d in enumerate(detections) if i not in selected_indices]
    return selected, audit, excluded, [detection_record(d) for d in all_selected]

def collect(source, detector, tracker, segmenter, output, *, max_tracks, max_seconds, rss_guard=None):
    """Sequential source-only loop. Models receive BGR; appearance stays RGB."""
    from src.components.detection.geometry import Box
    from src.components.detection.types import is_person
    from src.components.selection.heuristic import HeuristicSelector
    selector = HeuristicSelector()
    held_counts = {}
    tracker.reset()
    count, height, width, _ = source.shape
    masks, objects = {}, {}
    frame_records = [{'frame_index': i, 'source_frame_index': 38+i, 'status': 'unprocessed', 'detected': [], 'tracked': []} for i in range(count)]
    began = time.monotonic()
    try:
        for index, rgb in enumerate(source):
            if rss_guard is not None: rss_guard.check('frame_before')
            require(time.monotonic() - began <= max_seconds, 'collection time budget exhausted')
            bgr = rgb_to_bgr(rgb)
            record = {'frame_index':index, 'source_frame_index':38+index, 'rgb_sha256':array_digest(rgb[None]),
                      'detected':[], 'selected':[], 'excluded':[], 'tracked':[], 'status':'processing'}
            frame_records[index] = record
            detected = list(detector.detect(bgr))
            record['detected'] = [detection_record(d) for d in detected]
            selected, selection_audit, excluded, all_selected = select_with_audit(selector, detected, (height,width), held_counts)
            record.update(selected=[dict(detection_record(d), **selection_audit[d.track_id]) for d in selected],
                          excluded=excluded, selector_output_including_rackets=all_selected)
            previous_tracker = {d.track_id:d for d in getattr(tracker, '_previous', [])}
            tracked = list(tracker.update(bgr, selected, predictor=detector))
            record['tracker_output_before_validation'] = [detection_record(d) for d in tracked]
            require(len({d.track_id for d in tracked}) == len(tracked), 'duplicate tracked role; recovery would duplicate identity')
            for item in tracked:
                require(item.track_id in ('player_far','player_near') and item.class_name == 'player', 'explicit two player roles required')
                bbox = raster_box(item.bbox, height, width)
                clipped = item.with_bbox(Box(*bbox))
                mask = segmenter.segment(bgr, clipped)
                row = detection_record(clipped)
                selected_item = next((d for d in selected if d.track_id == item.track_id), None)
                recovery_changed = selected_item is None or selected_item.bbox != item.bbox
                row.update(selection_audit.get(item.track_id, {}))
                prior = previous_tracker.get(item.track_id)
                tracker_held = recovery_changed and prior is not None and prior.bbox == item.bbox
                row['tracker_recovery_changed'] = recovery_changed
                row['tracker_history_held'] = tracker_held
                row['stale'] = row.get('stale', False) or tracker_held
                row['tracker_matches_selected'] = not recovery_changed
                row['mask_missing'] = mask is None
                record['tracked'].append(row)
                if item.track_id not in masks:
                    require(len(masks) < max_tracks, 'registered track storage bound exceeded')
                    filename = f'track_{len(masks):04d}_mask.npy'
                    masks[item.track_id] = np.lib.format.open_memmap(output / filename, mode='w+',
                                                                   dtype=bool, shape=(count, height, width))
                    masks[item.track_id][:] = False
                    x1, y1, x2, y2 = bbox
                    appearance = np.ascontiguousarray(rgb[y1:y2, x1:x2]).copy()
                    appearance_file = filename.replace('_mask', '_appearance_rgb')
                    np.save(output / appearance_file, appearance, allow_pickle=False)
                    objects[item.track_id] = {'object_id': item.track_id, 'frame_index': index,
                        'bbox': list(bbox), 'mask_file': filename, 'appearance_file': appearance_file,
                        'appearance_rgb_sha256': array_digest(appearance[None]), 'object_class': 'player', 'selected_role':item.track_id}
                if mask is not None:
                    x1, y1, x2, y2 = bbox
                    mask = np.asarray(mask)
                    require(mask.ndim == 2 and mask.shape == (y2-y1, x2-x1), 'segmenter crop mask shape mismatch')
                    require(np.isfinite(mask).all(), 'nonfinite mask')
                    mask = mask.astype(bool)
                    masks[item.track_id][index, y1:y2, x1:x2] = mask
                    row['crop_mask_sha256'] = array_digest(mask[None])
                    row['foreground_pixels'] = int(mask.sum())
            if rss_guard is not None: rss_guard.check('frame_after')
            record['status'] = 'complete'
            (output / 'frames.json').write_text(json.dumps(frame_records, indent=2) + '\n')
        for mask in masks.values():
            mask.flush()
        return list(objects.values()), frame_records
    finally:
        (output / 'frames.json').write_text(json.dumps(frame_records, indent=2) + '\n')
        for mask in masks.values():
            mask.flush()

def detection_record(item):
    return {'class_name': item.class_name, 'class_id': item.class_id, 'score': float(item.score),
            'track_id': item.track_id, 'bbox': [float(item.bbox.x1), float(item.bbox.y1),
                                               float(item.bbox.x2), float(item.bbox.y2)]}

def load_objects(output):
    """Explicit adapter for newly generated guide artifacts, never legacy loader."""
    from src.pipeline.reconstruction.reconstruct import ObjectRequest
    output = Path(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    require(receipt['status'] == 'complete', 'incomplete guide collection')
    for filename, expected in receipt['artifact_sha256'].items():
        require(digest(output / filename) == expected, 'guide artifact changed')
    return tuple(ObjectRequest(object_id=row['object_id'], frame_index=row['frame_index'],
        bbox=tuple(row['bbox']), appearance=np.load(output / row['appearance_file'], allow_pickle=False),
        mask=np.load(output / row['mask_file'], mmap_mode='r', allow_pickle=False), object_class=row['object_class'])
        for row in receipt['objects'])

class DeviceModel:
    def __init__(self, model, device, guard=None):
        self.model, self.device, self.guard = model, device, guard
        self.calls = []
    def predict(self, **kwargs):
        require('device' not in kwargs, 'device override forbidden')
        self.calls.append({k: ('in_memory_BGR_uint8_array' if k == 'source' else v) for k, v in dict(kwargs, device=self.device).items()})
        if self.guard is not None: self.guard.check('predict_before')
        try:
            return self.model.predict(**kwargs, device=self.device)
        finally:
            if self.guard is not None: self.guard.check('predict_after')

class RssGuard:
    def __init__(self, limit_bytes, reader=None):
        self.limit_bytes = limit_bytes
        self.reader = reader or self.current_rss
        self.peak_sampled_bytes = 0
        self.samples = 0

    @staticmethod
    def current_rss():
        fields = Path('/proc/self/statm').read_text().split()
        return int(fields[1]) * os.sysconf('SC_PAGE_SIZE')

    def check(self, stage):
        rss = self.reader()
        self.samples += 1
        self.peak_sampled_bytes = max(self.peak_sampled_bytes, rss)
        require(rss <= self.limit_bytes, 'sampled RSS bound exceeded at ' + stage)
        return rss

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    require(digest(args.registration) == args.registration_sha256, 'registration changed')
    reg = json.loads(Path(args.registration).read_text())
    require(reg['status'] == 'frozen' and reg['worker_sha256'] == digest(__file__), 'unfrozen worker')
    root = Path(__file__).resolve().parents[2]
    required_code = set(REQUIRED_CODE) | {str(p.relative_to(root)) for p in (root / 'src/contracts').glob('*.py')}
    require(reg['config'] == CONFIG, 'configuration mismatch')
    require(hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == reg['config_sha256'], 'config hash mismatch')
    require(required_code <= set(reg['code_files']), 'missing primary dependency pins')
    for rel, sha in reg['code_files'].items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root), 'code path outside checkout')
        require(digest(path) == sha, 'code dependency changed')
    env = reg['environment']
    require(Path(sys.executable).resolve() == Path(env['python_path']).resolve(), 'interpreter differs')
    require(digest(sys.executable) == env['python_sha256'], 'interpreter changed')
    require(hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == reg['environment_sha256'], 'environment hash mismatch')
    for name, version in env['packages'].items():
        require(importlib.metadata.version(name) == version, 'package version mismatch: ' + name)
        manifest = env['package_manifest'][name]
        require(manifest['kind'] in ('RECORD', 'METADATA', 'PKG-INFO'), 'invalid package manifest kind')
        content = importlib.metadata.distribution(name).read_text(manifest['kind'])
        require(content is not None and hashlib.sha256(content.encode()).hexdigest() == manifest['sha256'], 'installed package manifest differs: ' + name)
    require({'torch', 'ultralytics', 'numpy', 'opencv-python'} <= set(env['packages']), 'missing environment pins')
    require(env['runtime_files'] and all(any(key in path.lower() for path in env['runtime_files']) for key in ('torch', 'ultralytics', 'numpy', 'cv2')), 'concrete runtime source/extension pins required for torch/ultralytics/numpy/cv2')
    for filename, expected in env['runtime_files'].items():
        require(Path(filename).is_absolute() and digest(filename) == expected, 'runtime file changed')
    source_path = Path(reg['source_path']).resolve()
    require(source_path.name == 'source.npy' and 'chunk_00' in source_path.parts, 'original source000 chunk00 required')
    require(digest(source_path) == reg['source_file_sha256'], 'original source file changed')
    full_source = np.load(source_path, mmap_mode='r', allow_pickle=False)
    source = validate_source(full_source, reg['frames'])
    require(reg['source_rgb_sha256'] == SOURCE_RGB_SHA256 and array_digest(full_source) == SOURCE_RGB_SHA256,
            'original full96 RGB identity mismatch')
    bounds = reg['resources']
    require(bounds['device'] in ('cpu', 'cuda:0'), 'unsupported device')
    require(bounds['cpu_threads'] == 8, 'eight CPU threads required')
    require(0 < bounds['memory_gib'] <= 48 and 0 < bounds['max_tracks'] <= 8 and 0 < bounds['max_seconds'] <= 3600,
            'resource bounds invalid')
    # Dense masks are disk-mapped; source/model/output staging still require a bounded pilot.
    memory = int(bounds['memory_gib'] * 1024**3)
    # CUDA drivers reserve huge virtual ranges. GPU runs use observed RSS
    # guards, not RLIMIT_AS; transient allocations between samples can exceed it.
    if bounds['device'] == 'cpu':
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    guard = RssGuard(memory)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:8])
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    if bounds['device'] == 'cuda:0':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible.startswith('GPU-') and ',' not in visible, 'exactly one fleet-visible GPU UUID required')
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES', '') in ('', '-1'), 'CPU run must hide GPU devices')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    weights = reg['weights']
    for role in ('detector', 'segmenter'):
        require(Path(weights[role]['path']).is_absolute() and digest(weights[role]['path']) == weights[role]['sha256'],
                'pinned local weight mismatch')
    output = Path(reg['output_path']).resolve()
    audits_root = Path('/home/itec/emanuele/pointstream-data/audits').resolve()
    require(output.is_relative_to(audits_root) and output != audits_root and not output.exists(), 'fresh external audits output required')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'registration.json').write_bytes(Path(args.registration).read_bytes())
    receipt = {'status': 'started', 'registration_sha256': args.registration_sha256,
               'access_contract': 'only source000 RGB selected prefix; fresh past-only tracker; no legacy guides/source028',
               'source_frame_interval': [38, 38 + reg['frames']], 'config': CONFIG, 'resources': bounds,
               'scope': 'model-derived guides; no independent task truth or quality claim',
               'environment_pin_caveat': 'METADATA/PKG-INFO pins identify packaging metadata, not all installed binary contents; separately selected runtime source/extension files pinned',
               'worker_start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'cpu_affinity': sorted(os.sched_getaffinity(0)), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'cpu_memory_guard': 'sampled RSS before/after model calls and frames; not OS hard enforcement; transient peaks not bounded' if bounds['device'] != 'cpu' else 'RLIMIT_AS plus sampled RSS',
               'class_mapping': 'raw YOLO person -> production selector player with player_far/player_near roles; explicit selected-domain policy',
               'selection_change': CONFIG['selection_change'],
               'domain_track_bound': 'two on-court player slots; storage bound8 unchanged, all raw detections retained including excluded candidates'}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    try:
        from ultralytics import YOLO
        import torch
        receipt['loaded_torch_runtime'] = validate_torch_runtime(torch, env)
        from src.components.detection.yolo import YoloDetector
        from src.components.tracking.tracker import IdentityTracker
        from src.components.segmentation.yolo import YoloSegmenter
        require(bounds['device'] == 'cpu' or torch.cuda.is_available(), 'registered CUDA unavailable; no fallback')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        guard.check('models_before')
        detector = YoloDetector(model=DeviceModel(YOLO(weights['detector']['path']), bounds['device'], guard))
        segmenter = YoloSegmenter(model=DeviceModel(YOLO(weights['segmenter']['path']), bounds['device'], guard))
        guard.check('models_after')
        if bounds['device'] != 'cpu': torch.cuda.reset_peak_memory_stats()
        objects, records = collect(source, detector, IdentityTracker(), segmenter, output,
                                  max_tracks=bounds['max_tracks'], max_seconds=bounds['max_seconds'], rss_guard=guard)
        receipt.update(status='complete', objects=objects, frame_count=len(records),
                       actual_model_calls={'detector': detector._model.calls, 'segmenter': segmenter._model.calls},
                       model_overrides={role: dict(model._model.model.overrides) for role, model in [('detector', detector), ('segmenter', segmenter)]},
                       artifact_sha256={p.name: digest(p) for p in output.iterdir() if p.name != 'receipt.json'},
                       cuda_runtime={'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                           'device_name': torch.cuda.get_device_name(0) if bounds['device'] != 'cpu' else None,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated() if bounds['device'] != 'cpu' else 0,
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved() if bounds['device'] != 'cpu' else 0},
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['rss_guard'] = {'limit_bytes': guard.limit_bytes, 'samples': guard.samples, 'peak_sampled_bytes': guard.peak_sampled_bytes}
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')

if __name__ == '__main__':
    main()
