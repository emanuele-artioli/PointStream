"""Ada 16/48 calibration adapter; no admission or dispatch is implied.

Filesystem denial is a mandatory external dependency. Until the isolated
receiver supplies its verifier this module fails closed; legacy receiver
receipts alone never qualify a producer.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import uuid
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_ada_fresh_compare as fresh
from experiments.tennis import k5d_checkpoint as cp, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_real_parity as parity
from experiments.tennis import k5d_same_job_receiver as receiver, k5d_source_oracle as oracle

REFERENCE_SHA256 = '4bfff442081d988039abb308a3e0b8ed22632f5f598858d6eeaf4d0ef6824a1a'
REFERENCE_JOB = '20261112T210150Z-2c8b49a0'
REFERENCE_IDENTITY = {
    'source_sha256': 'ed89052108dbd4a8242d7fd7ae35ac37aa1f41332d204a4d11e9d7b799bf715d',
    'snapshot_sha256': '4f7f6546b7eb4552ea77b8d1607a0ccc68c7ae42e73965a70a2902f40f4e7402',
    'environment_sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661',
}
GPU_NAME = 'NVIDIA RTX 6000 Ada Generation'



def _uuid(value: Any, *, prefixed: bool) -> uuid.UUID:
    if not isinstance(value, str) or (prefixed and not value.startswith('GPU-')):
        raise ValueError('GPU UUID format differs')
    raw = value[4:] if value.startswith('GPU-') else value
    try:
        parsed = uuid.UUID(raw)
    except ValueError as exc:
        raise ValueError('GPU UUID format differs') from exc
    if str(parsed) != raw.lower():
        raise ValueError('GPU UUID must use canonical hyphenated form')
    return parsed


def persisted_device(root: Path, marker: dict[str, Any], *, dispatch_gpu: Any = None) -> dict[str, Any]:
    """Join sealed Torch device evidence to marker and, where available, dispatch."""
    qualification = json.loads((root / 'source-qualification.json').read_text())
    source = qualification.get('device')
    marked = marker.get('gpu')
    if (not isinstance(source, dict) or not isinstance(marked, dict)
            or source.get('name') != GPU_NAME or marked.get('name') != source['name']
            or source.get('capability') != [8, 9]
            or _uuid(source.get('uuid'), prefixed=False) != _uuid(marked.get('uuid'), prefixed=True)):
        raise ValueError('persisted producer device does not match marker')
    if dispatch_gpu is not None:
        if (not isinstance(dispatch_gpu, dict) or dispatch_gpu.get('name') != source['name']
                or _uuid(dispatch_gpu.get('uuid'), prefixed=True) != _uuid(marked['uuid'], prefixed=True)):
            raise ValueError('persisted producer device does not match dispatch')
    return {'name': source['name'], 'uuid': marked['uuid'], 'capability': source['capability']}

def reference(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the pinned publication after its archive hash has been checked."""
    saved = json.loads((root / 'fresh.json').read_text())
    device = saved.get('device', {})
    if (saved.get('execution_identity') != REFERENCE_IDENTITY or saved.get('gpu_profile') != parity.ADA_PROFILE
            or device.get('name') != GPU_NAME or device.get('capability') != [8, 9]):
        raise ValueError('accepted Ada reference identity or device differs')
    parity.verify_scored(root, REFERENCE_IDENTITY)
    prepared = root / 'prepared'
    report = json.loads((prepared / 'pilot.json').read_text())
    marker = receiver.verify(prepared, report, REFERENCE_IDENTITY)
    actual = persisted_device(prepared, marker)
    if (marker['producer_job'] != REFERENCE_JOB or marker['producer_stage'] != 'smoke'
            or actual != device):
        raise ValueError('accepted Ada reference producer differs')
    return report, json.loads((prepared / 'source-qualification.json').read_text())


def prepare16(reference_archive: Path, prepared: Path, identity: dict[str, str]) -> None:
    """Require exact original-source and native calibration at all six points."""
    with cost.unpack(reference_archive, REFERENCE_SHA256) as root:
        old, old_source = reference(root)
        cp.verify(prepared, oracle.sha(prepared / 'checkpoint.json'), identity, require_same_job_receiver=True)
        new = json.loads((prepared / 'pilot.json').read_text())
        source = json.loads((prepared / 'source-qualification.json').read_text())
        cost.compare_contract(old, old_source, new, source, exact=False, capability=(8, 9))
        if (new['frames'] != 16 or source['rows'] != old_source['rows'] or source['scores'] != old_source['scores']
                or new['conversion']['yuv_sha256'] != old['conversion']['yuv_sha256']
                or [(p['codec'], p['point']) for p in new['points']] != list(enc.POINTS)):
            raise ValueError('Ada source, converted pixels or six-point scope differs')
        for a, b in zip(old['points'], new['points']):
            for key in ('codec', 'point', 'native_sha256', 'native_bytes', 'wire_sha256', 'wire_bytes',
                        'encoder_decoded_sha256', 'encoder_rendered_sha256'):
                if a[key] != b[key]:
                    raise ValueError('Ada preparation differs: ' + key)
            for key in ('native_bytes', 'native_sha256', 'zero_ball_wire_bytes', 'zero_ball_wire_sha256'):
                if a['baseline'][key] != b['baseline'][key]:
                    raise ValueError('Ada baseline differs: ' + key)
            if a['streak']['guard_off'] != b['streak']['guard_off']:
                raise ValueError('Ada guard differs')



def prepare48(reference_archive: Path, prepared: Path, identity: dict[str, str]) -> None:
    """Validate full-context source and frozen model contract without 16-frame output parity."""
    with cost.unpack(reference_archive, REFERENCE_SHA256) as root:
        old, old_source = reference(root)
        cp.verify(prepared, oracle.sha(prepared / 'checkpoint.json'), identity, require_same_job_receiver=True)
        report = json.loads((prepared / 'pilot.json').read_text())
        source = json.loads((prepared / 'source-qualification.json').read_text())
        cost.compare_contract(old, old_source, report, source, exact=False, capability=(8, 9))
        if report['frames'] != 48 or len(source['rows']) != 48:
            raise ValueError('whole 48-frame source required')

def score16(reference_archive: Path, assembled: Path, accepted16: Path) -> dict[str, Any]:
    """Require all original exact metrics and judges without decoding a packet."""
    with cost.unpack(reference_archive, REFERENCE_SHA256) as root:
        reference(root)
        return fresh.compare(root / 'scored' / 'assembled', assembled, accepted16)


def producer(root: Path, identity: dict[str, str], *, job: str, stage: str, frames: int, config_sha256: str, config_path: Path) -> dict[str, Any]:
    """Authenticate retained producer pixels and require the actual denial verifier."""
    if (stage, frames) not in (('smoke', 16), ('full', 48)) or not job.startswith('2026'):
        raise ValueError('only the paired 16/48 producer stages are admitted')
    cp.verify(root, oracle.sha(root / 'checkpoint.json'), identity, require_same_job_receiver=True)
    report = json.loads((root / 'pilot.json').read_text())
    marker = receiver.verify(root, report, identity)
    if (marker['producer_job'] != job or marker['producer_stage'] != stage or marker['frames'] != frames
            or marker['gpu']['name'] != GPU_NAME or not marker['gpu']['uuid'].startswith('GPU-')):
        raise ValueError('sealed producer job, stage or Ada device differs')
    # This import and entry point are deliberately mandatory, not a fallback to
    # the older API-only receipt. Integration requires review of the actual
    # verifier interface and its cryptographic binding before any release.
    try:
        isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
        verify = isolated.verify_receipt
    except (ImportError, AttributeError) as exc:
        raise ValueError('actual receiver denial verifier unavailable') from exc
    if oracle.sha(config_path) != config_sha256:
        raise ValueError('receiver config checksum differs')
    profile = json.loads(config_path.read_text())['profiles']['gpu6']
    def canonical(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    profile_sha, inventory_sha = canonical(profile), canonical(profile['runtime_files'])
    context = {'job_id': job, 'stage': stage, 'execution_identity': identity,
               'gpu': persisted_device(root, marker)}
    for point in report['points']:
        receipt = json.loads((root / f"{point['codec']}-{point['point']}" / 'receiver-evidence' / 'receiver.json').read_text())
        verified = verify(receipt, expected_context=context, expected_config_sha256=config_sha256)
        if (not isinstance(verified, dict) or verified.get('profile_sha256') != profile_sha
                or verified.get('runtime_inventory_sha256') != inventory_sha
                or not isinstance(verified.get('actual_policy_sha256'), str)
                or len(verified['actual_policy_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in verified['actual_policy_sha256'])):
            raise ValueError('actual receiver policy or inventory evidence differs')
    return marker
