"""Retain and authenticate real packet-receiver outputs from the producing job.

This mode does not run a codec or manufacture pixels. The original receiver has
already decoded the packet in the encoding job; the scorer reads those bytes.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_source_oracle as oracle

MARKER = 'same-job-receiver.json'
SCOPE = 'packet receiver outputs retained by their encoding job'
FILES = ('native.yuv', 'rendered.rgb', 'receiver.json')


def _receipt(point: dict[str, Any], report: dict[str, Any], folder: Path, *,
             verified_manifest: dict[str, str] | None = None, root: Path | None = None) -> dict[str, Any]:
    evidence = folder / 'receiver-evidence'
    for name in FILES:
        path = evidence / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('retained receiver file missing or not regular')
    receipt = json.loads((evidence / 'receiver.json').read_text())
    if (receipt != point['receiver'] or receipt.get('wire_sha256') != point['wire_sha256']
            or receipt.get('native_sha256') != point['native_sha256']
            or receipt.get('native_bytes') != point['native_bytes']
            or receipt.get('wire_bytes') != point['wire_bytes']
            or receipt.get('resource_map_sha256') != report['resource_map_sha256']
            or receipt.get('decoded_yuv_sha256') != point['encoder_decoded_sha256']
            or receipt.get('rendered_rgb_sha256') != point['encoder_rendered_sha256']
            or receipt.get('frames') != report['frames'] or receipt.get('size') != report['size']
            or receipt.get('resource_id') != point['codec']):
        raise ValueError('retained receiver identity differs from producer')
    w, h = report['size']
    frames = report['frames']
    def checked_sha(path: Path) -> str:
        if verified_manifest is None or root is None:
            return oracle.sha(path)
        key = path.relative_to(root).as_posix()
        if key not in verified_manifest:
            raise ValueError('receiver file absent from verified checkpoint manifest')
        return verified_manifest[key]
    if (any(type(v) is not int or v <= 0 for v in (w, h, frames)) or w % 2 or h % 2
            or (evidence / 'native.yuv').stat().st_size != frames * w * h * 3 // 2
            or (evidence / 'rendered.rgb').stat().st_size != frames * w * h * 3
            or checked_sha(evidence / 'native.yuv') != receipt['decoded_yuv_sha256']
            or checked_sha(evidence / 'rendered.rgb') != receipt['rendered_rgb_sha256']):
        raise ValueError('retained receiver pixels, size or geometry differ')
    return receipt


def retain(root: Path, work: Path, identity: dict[str, str], stage: Path) -> None:
    report = json.loads((root / 'pilot.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    gpu = dispatch.get('environment', {}).get('gpu')
    code = dispatch.get('code', {})
    if (not isinstance(gpu, dict) or not isinstance(gpu.get('uuid'), str) or not gpu['uuid'].startswith('GPU-')
            or not isinstance(gpu.get('name'), str) or gpu['uuid'] != os.environ.get('PS_CLAIMED_GPU_UUID')
            or gpu['uuid'] != os.environ.get('CUDA_VISIBLE_DEVICES') or not stage.parent.name.startswith('2026')
            or code.get('source_sha256') != identity['source_sha256']
            or code.get('snapshot_sha256') != identity['snapshot_sha256']
            or dispatch.get('staged_environment', {}).get('sha256') != identity['environment_sha256']):
        raise ValueError('actual receiver producer dispatch identity missing')
    for point in report['points']:
        name = f"{point['codec']}-{point['point']}"
        src = work / f'receiver-{name}'
        dest = root / name / 'receiver-evidence'
        dest.mkdir()
        for filename in FILES:
            srcfile = src / filename
            if srcfile.is_symlink() or not srcfile.is_file():
                raise ValueError('actual receiver output missing or not regular')
            with srcfile.open('rb') as source, (dest / filename).open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        _receipt(point, report, root / name)
    marker = {'schema': 1, 'scope': SCOPE, 'producer_job': stage.parent.name, 'producer_stage': stage.name,
              'execution_identity': identity, 'gpu': {'uuid': gpu['uuid'], 'name': gpu['name']},
              'native_tools': report['tools'], 'frames': report['frames'], 'clip': report['clip'],
              'points': [[p['codec'], p['point']] for p in report['points']]}
    with (root / MARKER).open('x') as f:
        json.dump(marker, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def verify(root: Path, report: dict[str, Any], identity: dict[str, str], *,
           verified_manifest: dict[str, str] | None = None) -> dict[str, Any]:
    path = root / MARKER
    if path.is_symlink() or not path.is_file():
        raise ValueError('same-job producer evidence required')
    marker = json.loads(path.read_text())
    gpu = marker.get('gpu')
    if (marker.get('schema') != 1 or marker.get('scope') != SCOPE
            or not isinstance(marker.get('producer_job'), str) or not marker['producer_job'].startswith('2026')
            or marker.get('producer_stage') not in ('smoke', 'full') or marker.get('execution_identity') != identity
            or not isinstance(gpu, dict) or not isinstance(gpu.get('uuid'), str) or not gpu['uuid'].startswith('GPU-')
            or not isinstance(gpu.get('name'), str) or marker.get('native_tools') != report['tools']
            or marker.get('frames') != report['frames'] or marker.get('clip') != report['clip']
            or marker.get('points') != [[p['codec'], p['point']] for p in report['points']]):
        raise ValueError('same-job producer metadata differs')
    for point in report['points']:
        _receipt(point, report, root / f"{point['codec']}-{point['point']}",
                 verified_manifest=verified_manifest, root=root)
    return marker
