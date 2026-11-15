"""SAM source-only proposals for independent human review; never reference masks.

Input tar has manifest.json and listed PNG files. No propagation between frames.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.segmentation.sam31 import DEFAULT_CHECKPOINT_SHA256, DEFAULT_SOURCE_REVISION, Sam31SequenceSegmenter

OBJECTS = {'P35': 'saucepan', 'P12': 'faucet', 'P11': 'pan'}
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
SAFE = re.compile(r'[A-Za-z0-9_-]{1,80}\Z')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def as_jpeg(data: bytes) -> bytes:
    output = io.BytesIO()
    with Image.open(io.BytesIO(data)) as image:
        image.convert('RGB').save(output, format='JPEG', quality=95, subsampling=0)
    return output.getvalue()


def read_inputs(path: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    with tarfile.open(path, 'r:*') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('archive must contain unique regular files only')
        if 'manifest.json' not in names:
            raise ValueError('manifest missing')
        def read(name: str) -> bytes:
            member = archive.getmember(name)
            if member.size > 32 * 1024 * 1024:
                raise ValueError('archive member too large')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('missing archive data')
            return stream.read()
        manifest = json.loads(read('manifest.json'))
        if set(manifest) != {'frames'} or not isinstance(manifest['frames'], list) or len(manifest['frames']) != 18:
            raise ValueError('manifest must contain exactly 18 frames')
        frames = manifest['frames']
        images: dict[str, bytes] = {}
        ids: set[str] = set()
        for row in frames:
            if set(row) != {'id', 'video', 'image', 'sha256', 'width', 'height', 'object_bbox', 'smoke'}:
                raise ValueError('invalid frame fields')
            key, video = row['id'], row['video']
            if not isinstance(key, str) or not SAFE.fullmatch(key) or key in ids or video not in OBJECTS:
                raise ValueError('invalid frame identity')
            ids.add(key)
            if row['image'] != f'images/{key}.png' or type(row['smoke']) is not bool:
                raise ValueError('invalid image or smoke identity')
            if any(type(row[k]) is not int or row[k] < 1 or row[k] > 8192 for k in ('width', 'height')):
                raise ValueError('invalid geometry')
            box = row['object_bbox']
            if box is not None:
                if (not isinstance(box, list) or len(box) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                    not (0 <= box[0] < box[2] <= row['width'] and 0 <= box[1] < box[3] <= row['height'])):
                    raise ValueError('invalid guidance box')
            data = read(row['image'])
            if digest(data) != row['sha256']:
                raise ValueError('source image checksum mismatch')
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'PNG' or image.size != (row['width'], row['height']):
                    raise ValueError('source PNG geometry mismatch')
                image.verify()
            images[key] = data
        if set(names) != {'manifest.json', *(r['image'] for r in frames)}:
            raise ValueError('unexpected archive member')
        if any(sum(r['smoke'] and r['video'] == video for r in frames) != 1 for video in OBJECTS):
            raise ValueError('smoke needs exactly one frame per video')
        return frames, images


def run(archive: Path, checkpoint: Path, count: int) -> None:
    import torch
    from experiments.audit.env_smoke import model_on_cuda, profile_cuda

    rows, images = read_inputs(archive)
    if count not in (3, 18):
        raise ValueError('only frozen 3-frame smoke and 18-frame full allowed')
    selected = [r for r in rows if r['smoke']] if count == 3 else rows
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one visible CUDA device required')
    props = torch.cuda.get_device_properties(0)
    if (not any(name in props.name for name in ('RTX A6000', 'RTX 6000 Ada')) or props.major != 8
            or props.minor not in (6, 9) or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('unsupported GPU class/bfloat16')
    if os.environ.get('PS_CPU_ALLOWANCE') != '1' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('single CPU allowance and affinity required')
    if torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('unqualified torch or CUDA runtime')
    limit = 14336 * 1024 * 1024
    if props.total_memory <= limit:
        raise RuntimeError('GPU capacity does not exceed reserved memory ceiling')
    torch.cuda.set_per_process_memory_fraction(limit / props.total_memory, device=0)
    torch.cuda.reset_peak_memory_stats()
    stage, scratch = Path(os.environ['PS_STAGE_DIR']), Path(os.environ['PS_SCRATCH_DIR'])
    staging = json.loads((stage / 'staging.json').read_text())
    if staging['environment']['sha256'] != RUNTIME_SHA:
        raise RuntimeError('packed runtime identity mismatch')
    work = scratch / 'human-sam-proposals'
    work.mkdir(exist_ok=False)
    publication = work / 'publication'
    (publication / 'masks').mkdir(parents=True)
    # The approved packed environment supplies the pinned package and tokenizer.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.pop('SAM31_SOURCE_ROOT', None)
    started = time.monotonic()
    sam = Sam31SequenceSegmenter(checkpoint_path=checkpoint, checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                                 source_revision=DEFAULT_SOURCE_REVISION)
    load = time.monotonic() - started
    if sam.model_revision != DEFAULT_SOURCE_REVISION or not model_on_cuda(sam._require_predictor().model):
        raise RuntimeError('SAM source or CUDA model mismatch')
    records: list[dict[str, Any]] = []
    kernels: dict[str, Any] = {}
    inference_start = time.monotonic()
    for row in selected:
        frame_dir = work / row['id']
        frame_dir.mkdir()
        jpeg = frame_dir / '000000.jpg'
        jpeg.write_bytes(as_jpeg(images[row['id']]))
        conversion_sha = digest(jpeg.read_bytes())
        for name, prompt, box in [('person', 'person', None), ('hand', 'hand', None),
                                  ('object', OBJECTS[row['video']], row['object_bbox'])]:
            sam.start_session(name, frame_dir, frame_width=row['width'], frame_height=row['height'], policy='offline_causal')
            try:
                def predict() -> Any:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        return sam.add_prompt(name, frame_index=0, object_id=f'{row["id"]}-{name}', text=prompt,
                                              bbox=tuple(box) if box is not None else None)
                if count == 3 and not kernels:
                    observations, kernels = profile_cuda(predict)
                else:
                    observations = predict()
                instances = []
                for index, observation in enumerate(observations):
                    item = {'object_id': observation.object_id, 'tracker_id': observation.tracker_id,
                            'score': observation.score, 'status': observation.status.value, 'reason': observation.reason}
                    if observation.mask is not None:
                        mask = np.asarray(observation.mask, dtype=bool)
                        if mask.shape != (row['height'], row['width']):
                            raise ValueError('model mask geometry mismatch')
                        path = publication / 'masks' / f'{row["id"]}-{name}-{index}.png'
                        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                        item.update(mask=path.name, mask_sha256=digest(path.read_bytes()), pixels=int(mask.sum()))
                    instances.append(item)
                records.append({'id': row['id'], 'video': row['video'], 'source_sha256': row['sha256'],
                                'sam_jpeg_sha256': conversion_sha, 'width': row['width'], 'height': row['height'],
                                'prompt': name, 'text': prompt, 'bbox_guidance': box, 'instances': instances})
            finally:
                sam.close_session(name)
    torch.cuda.synchronize()
    inference = time.monotonic() - inference_start
    peak = torch.cuda.max_memory_reserved(0)
    if peak > limit:
        raise RuntimeError('SAM GPU peak reservation exceeds declared cap')
    forecast = 1.5 * (load + 6 * inference + 30) if count == 3 else None
    result = {'count': count, 'archive_sha256': digest(archive.read_bytes()), 'records': records,
              'model_sha256': sam.checkpoint_hash, 'model_revision': sam.model_revision,
              'runtime_sha256': RUNTIME_SHA, 'cpu_allowance': os.environ['PS_CPU_ALLOWANCE'],
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gpu_memory_limit_bytes': limit,
              'gpu_peak_reserved_bytes': peak, 'model_config': dataclasses.asdict(sam.provenance('offline_causal')),
              'sdpa_policy': sam.sdpa_backend_policy,
              'device': props.name, 'device_uuid': str(getattr(props, 'uuid', '')),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(),
              'jpeg_conversion': 'Pillow RGB JPEG quality=95 subsampling=0',
              'capability': [props.major, props.minor], 'bf16': True,
              'flash_profile': kernels, 'load_seconds': load, 'inference_seconds': inference,
              'full_forecast_seconds': forecast, 'scope': 'proposals only; human approval required'}
    (publication / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    target = stage / 'published.tar'
    members = sorted(path for path in publication.rglob('*') if path.is_file())
    with tarfile.open(target, 'x') as tar:
        for path in members:
            tar.add(path, arcname=path.relative_to(publication).as_posix(), recursive=False)
    info = {'sha256': file_sha(target), 'bytes': target.stat().st_size, 'members': len(members),
            'result_sha256': file_sha(publication / 'result.json')}
    (stage / 'proposal-stage.json').write_text(json.dumps(info, indent=2) + '\n')


def validate_stage() -> int:
    stage = Path(os.environ['PS_STAGE_DIR'])
    staging = json.loads((stage / 'staging.json').read_text())
    inputs = {entry['name']: entry for entry in staging['inputs']}
    archive = Path(inputs['archive']['path'])
    rows, images = read_inputs(archive)
    info = json.loads((stage / 'proposal-stage.json').read_text())
    publication_tar = stage / 'published.tar'
    if publication_tar.stat().st_size != info['bytes'] or file_sha(publication_tar) != info['sha256']:
        raise ValueError('publication archive identity mismatch')
    temporary = tempfile.TemporaryDirectory(prefix='sam-proposal-validation-', dir=os.environ['TMPDIR'])
    publication = Path(temporary.name)
    with tarfile.open(publication_tar, 'r') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if (len(names) != info['members'] or len(names) != len(set(names)) or 'result.json' not in names or
            any(not m.isfile() or not (m.name == 'result.json' or
                re.fullmatch(r'masks/[A-Za-z0-9_-]+\.png', m.name)) for m in members)):
            raise ValueError('unsafe or unexpected publication archive member')
        (publication / 'masks').mkdir()
        for member in members:
            reader = tar.extractfile(member)
            if reader is None:
                raise ValueError('unreadable publication member')
            with (publication / member.name).open('xb') as target:
                shutil.copyfileobj(reader, target)
    if file_sha(publication / 'result.json') != info['result_sha256']:
        raise ValueError('publication result hash mismatch')
    result = json.loads((publication / 'result.json').read_text())
    count = 3 if os.environ['PS_STAGE'] == 'smoke' else 18
    selected = [r for r in rows if r['smoke']] if count == 3 else rows
    expected = {(row['id'], name) for row in selected for name in ('person', 'hand', 'object')}
    records = result['records']
    actual = [(r['id'], r['prompt']) for r in records]
    checks: dict[str, bool] = {
        'frame_prompt_coverage': len(actual) == len(expected) and set(actual) == expected,
        'source_and_model_identity': result['archive_sha256'] == inputs['archive']['sha256'] == digest(archive.read_bytes()) and
            result['model_sha256'] == DEFAULT_CHECKPOINT_SHA256 and result['model_revision'] == DEFAULT_SOURCE_REVISION,
        'cuda_bfloat16': result['capability'] in ([8, 6], [8, 9]) and result['bf16'] is True,
        'review_scope': result['scope'] == 'proposals only; human approval required',
        'conversion': result['jpeg_conversion'] == 'Pillow RGB JPEG quality=95 subsampling=0',
        'stage_count': result['count'] == count,
        'runtime_and_cpu': (staging['environment']['sha256'] == result['runtime_sha256'] == RUNTIME_SHA and
                            result['torch'] == '2.10.0+cu128' and result['torch_cuda'] == '12.8' and
                            result['cpu_allowance'] == '1' and len(result['cpu_affinity']) == 1),
        'gpu_memory_cap': (result['gpu_memory_limit_bytes'] == 14336 * 1024 * 1024 and
                           0 < result['gpu_peak_reserved_bytes'] <= result['gpu_memory_limit_bytes']),
        'model_configuration': (result['model_config']['name'] == 'sam3.1_multiplex' and
                                result['model_config']['policy'] == 'offline_causal' and
                                result['model_config']['model_revision'] == DEFAULT_SOURCE_REVISION and
                                result['model_config']['checkpoint_sha256'] == DEFAULT_CHECKPOINT_SHA256 and
                                result['model_config']['config_sha256'] ==
                                Sam31SequenceSegmenter(predictor=object()).provenance('offline_causal').config_sha256 and
                                result['sdpa_policy'] == 'native_flash'),
    }
    by_id = {r['id']: r for r in selected}
    jpeg_hashes = {r['id']: digest(as_jpeg(images[r['id']])) for r in selected}
    masks: set[str] = set()
    valid_instances = True
    actor_count = 0
    for record in records:
        source = by_id.get(record['id'])
        if source is None:
            valid_instances = False
            continue
        name = record['prompt']
        expected_text = OBJECTS[source['video']] if name == 'object' else name
        valid_instances &= (record['video'] == source['video'] and record['source_sha256'] == source['sha256'] and
                            record['width'] == source['width'] and record['height'] == source['height'] and
                            record['text'] == expected_text and record['bbox_guidance'] ==
                            (source['object_bbox'] if name == 'object' else None) and
                            record['sam_jpeg_sha256'] == jpeg_hashes[source['id']] and
                            bool(record['instances']))
        for item in record['instances']:
            if item['score'] is not None and (not isinstance(item['score'], (int, float)) or not math.isfinite(item['score'])):
                valid_instances = False
            if item['status'] == 'observed':
                filename = item.get('mask', '')
                if not re.fullmatch(r'[A-Za-z0-9_-]+\.png', filename) or filename in masks:
                    valid_instances = False
                    continue
                masks.add(filename)
                path = publication / 'masks' / filename
                if not path.is_file() or digest(path.read_bytes()) != item.get('mask_sha256'):
                    valid_instances = False
                    continue
                with Image.open(path) as image:
                    array = np.asarray(image)
                    valid_instances &= image.size == (source['width'], source['height']) and set(np.unique(array)).issubset({0, 255})
                    pixels = int(np.count_nonzero(array))
                valid_instances &= pixels == item['pixels']
                if name in ('person', 'hand') and pixels > 0:
                    actor_count += 1
            elif item['status'] != 'missing' or not item.get('reason') or 'mask' in item:
                valid_instances = False
    checks['instances_and_geometry'] = bool(valid_instances) and set(p.name for p in (publication / 'masks').iterdir()) == masks
    checks['actor_returned'] = actor_count > 0
    if count == 3:
        profile = result['flash_profile']
        checks['actual_flash_kernel'] = profile.get('kernel_launches', 0) > 0 and 'flash' in profile.get('attention_family', [])
        forecast = result['full_forecast_seconds']
        checks['full_forecast_under_600s'] = isinstance(forecast, (int, float)) and math.isfinite(forecast) and 0 < forecast <= 600
    output = {'passed': all(checks.values()), 'checks': checks, 'scope': 'source-only proposals, not approved references'}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(output, indent=2) + '\n')
    temporary.cleanup()
    return 0 if output['passed'] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    if args.validate:
        return validate_stage()
    if args.archive is None or args.checkpoint is None or args.frames is None:
        parser.error('run requires archive, checkpoint and frames')
    run(args.archive, args.checkpoint, args.frames)
    if args.frames == 18:
        # Fleet runs the configured validator only on smoke; independently validate full output here.
        os.environ['PS_VALIDATION_PATH'] = str(Path(os.environ['PS_STAGE_DIR']) / 'full-validation.json')
        return validate_stage()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
