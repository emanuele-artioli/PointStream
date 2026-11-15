"""CPU validation tests; these do not qualify any SAM/GPU execution."""
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from experiments.jobs.inbox import validate_spec
from experiments.visor import human_sam_proposals as job
from experiments.visor.human_sam_proposals_spec import spec
from src.segmentation.sam31 import DEFAULT_CHECKPOINT_SHA256, DEFAULT_SOURCE_REVISION


def archive_fixture(path: Path) -> list[dict]:
    rows: list[dict[str, Any]] = []
    image_buffer = io.BytesIO()
    Image.new('RGB', (4, 4), 'white').save(image_buffer, format='PNG')
    png = image_buffer.getvalue()
    with tarfile.open(path, 'w') as archive:
        for i in range(18):
            video = list(job.OBJECTS)[i % 3]
            key = f'{video}_{i}'
            row: dict[str, Any] = {'id': key, 'video': video, 'image': f'images/{key}.png', 'sha256': job.digest(png),
                   'width': 4, 'height': 4, 'object_bbox': [0, 0, 2, 2] if i < 3 else None, 'smoke': i < 3}
            rows.append(row)
            member = tarfile.TarInfo(row['image'])
            member.size = len(png)
            archive.addfile(member, io.BytesIO(png))
        manifest = json.dumps({'frames': rows}).encode()
        member = tarfile.TarInfo('manifest.json')
        member.size = len(manifest)
        archive.addfile(member, io.BytesIO(manifest))
    return rows


def test_archive_and_spec(tmp_path):
    archive = tmp_path / 'input.tar'
    rows = archive_fixture(archive)
    found, images = job.read_inputs(archive)
    assert found == rows and len(images) == 18
    candidate = spec('/home/example/input.tar', job.digest(archive.read_bytes()), '2026-11-15T11:07:34Z')
    assert validate_spec(candidate, now=1794738000)['gpu_memory_mib'] == 14336
    assert candidate['scale']['frames'] == {'smoke': 3, 'full': 18}


def test_archive_rejects_duplicate_members(tmp_path):
    archive = tmp_path / 'bad.tar'
    archive_fixture(archive)
    with tarfile.open(archive, 'a') as tar:
        extra = tarfile.TarInfo('manifest.json')
        extra.size = 2
        tar.addfile(extra, io.BytesIO(b'{}'))
    with pytest.raises(ValueError, match='unique'):
        job.read_inputs(archive)


def test_validator_rejects_mask_corruption_missing_frame_and_cost(tmp_path, monkeypatch):
    archive = tmp_path / 'input.tar'
    rows = archive_fixture(archive)
    _, images = job.read_inputs(archive)
    stage = tmp_path / 'smoke'
    stage.mkdir()
    publication = tmp_path / 'publication'
    (publication / 'masks').mkdir(parents=True)
    (stage / 'staging.json').write_text(json.dumps({'environment': {'sha256': job.RUNTIME_SHA},
        'inputs': [{'name': 'archive', 'path': str(archive), 'sha256': job.digest(archive.read_bytes())}]}))
    records = []
    for row in rows[:3]:
        for prompt in ('person', 'hand', 'object'):
            filename = f'{row["id"]}-{prompt}-0.png'
            mask = publication / 'masks' / filename
            Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(mask)
            records.append({'id': row['id'], 'video': row['video'], 'source_sha256': row['sha256'],
                            'sam_jpeg_sha256': job.digest(job.as_jpeg(images[row['id']])), 'width': 4, 'height': 4,
                            'prompt': prompt, 'text': job.OBJECTS[row['video']] if prompt == 'object' else prompt,
                            'bbox_guidance': row['object_bbox'] if prompt == 'object' else None,
                            'instances': [{'object_id': 'obj', 'tracker_id': 1, 'score': .5, 'status': 'observed',
                                           'reason': None, 'mask': filename, 'mask_sha256': job.digest(mask.read_bytes()),
                                           'pixels': 16}]})
    result = {'count': 3, 'archive_sha256': job.digest(archive.read_bytes()), 'records': records,
              'model_sha256': DEFAULT_CHECKPOINT_SHA256, 'model_revision': DEFAULT_SOURCE_REVISION,
              'model_config': {'name': 'sam3.1_multiplex', 'policy': 'offline_causal',
                               'model_revision': DEFAULT_SOURCE_REVISION, 'checkpoint_sha256': DEFAULT_CHECKPOINT_SHA256,
                               'config_sha256': job.Sam31SequenceSegmenter(predictor=object()).provenance('offline_causal').config_sha256},
              'sdpa_policy': 'native_flash',
              'jpeg_conversion': 'Pillow RGB JPEG quality=95 subsampling=0',
              'runtime_sha256': job.RUNTIME_SHA, 'torch': '2.10.0+cu128', 'torch_cuda': '12.8',
              'cpu_allowance': '1', 'cpu_affinity': [0], 'gpu_memory_limit_bytes': 14336 * 1024 * 1024,
              'gpu_peak_reserved_bytes': 1024,
              'capability': [8, 9], 'bf16': True, 'flash_profile': {'kernel_launches': 1, 'attention_family': ['flash']},
              'full_forecast_seconds': 500, 'scope': 'proposals only; human approval required'}
    result_file = publication / 'result.json'
    def pack():
        result_file.write_text(json.dumps(result))
        members = sorted(p for p in publication.rglob('*') if p.is_file())
        with tarfile.open(stage / 'published.tar', 'w') as tar:
            for path in members:
                tar.add(path, arcname=path.relative_to(publication).as_posix())
        target = stage / 'published.tar'
        info = {'sha256': job.file_sha(target), 'bytes': target.stat().st_size, 'members': len(members),
                'result_sha256': job.file_sha(result_file)}
        (stage / 'proposal-stage.json').write_text(json.dumps(info))
    pack()
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    assert job.validate_stage() == 0
    with (stage / 'published.tar').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='archive identity'):
        job.validate_stage()
    pack()
    mask = publication / 'masks' / records[0]['instances'][0]['mask']
    mask.write_bytes(b'corrupt')
    pack()
    assert job.validate_stage() == 1
    result['records'] = records[1:]
    result['full_forecast_seconds'] = 601
    pack()
    assert job.validate_stage() == 1
    checks = json.loads((tmp_path / 'validation.json').read_text())['checks']
    assert not checks['frame_prompt_coverage'] and not checks['full_forecast_under_600s']


def test_read_inputs_rejects_wrong_geometry_and_hash(tmp_path):
    good = tmp_path / 'good.tar'
    archive_fixture(good)
    bad = tmp_path / 'bad.tar'
    with tarfile.open(good) as src, tarfile.open(bad, 'w') as out:
        for member in src.getmembers():
            handle = src.extractfile(member)
            assert handle is not None
            data = handle.read()
            if member.name == 'manifest.json':
                manifest = json.loads(data)
                manifest['frames'][0]['width'] = 8
                data = json.dumps(manifest).encode()
                member.size = len(data)
            out.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match='geometry'):
        job.read_inputs(bad)
