"""W8's bounded promotion gate around the unchanged W1 scoring pipeline.

This gate is an operational projection, not an enforcement mechanism or a
guarantee that a full run will finish. The fleet supervisor and the separate
audit/preparation allocations remain necessary.
"""
from __future__ import annotations

import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

MANIFEST_SHA256 = '4f5af8c90f5fd909abac2aed123917795f388b3961e31e18cb2e0f306aa7acb0'


def projection(spec: dict[str, Any], dispatch: dict[str, Any], validation: dict[str, Any], now: float) -> dict[str, float]:
    """Fail closed unless the pilot and conservative remaining allowances fit."""
    manifest = [item for item in spec.get('staged_inputs', []) if item.get('name') == 'repair_manifest']
    arguments = spec.get('arguments', [])
    if (spec.get('single_cpu_campaign') is not True or spec.get('defer_failed_publication') is not True
            or type(spec.get('cpu_threads')) is not int or spec['cpu_threads'] != 1
            or spec.get('device', 'gpu') != 'gpu'
            or spec.get('scale', {}).get('judged') != {'smoke': 3, 'full': 0}
            or not isinstance(arguments, list) or arguments.count('--repair-manifest') != 1
            or arguments.index('--repair-manifest') + 1 >= len(arguments)
            or arguments[arguments.index('--repair-manifest') + 1] != '{staged:repair_manifest}'
            or len(manifest) != 1 or manifest[0].get('sha256') != MANIFEST_SHA256):
        raise ValueError('one-core GPU campaign and original 3/80 tick scales required')
    sharing = spec.get('gpu_sharing')
    if sharing is not None and (not isinstance(sharing, dict) or sharing != {'mode': 'exclusive'}):
        raise ValueError('exclusive GPU admission required')
    contention = spec.get('contention')
    if not isinstance(contention, dict) or contention != {'policy': 'stop', 'resume_attempts': 0}:
        raise ValueError('contention must stop without retries')
    dispatched_sharing = dispatch.get('gpu_sharing')
    if dispatched_sharing is not None and dispatched_sharing != {'mode': 'exclusive'}:
        raise ValueError('exclusive GPU dispatch required')
    def finite(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{label} must be positive and finite')
        return float(value)

    smoke = finite(spec['smoke']['seconds'], 'smoke allowance')
    full = finite(spec['full']['seconds'], 'full allowance')
    validator = finite(spec['validator_seconds'], 'validator allowance')
    budget = finite(spec['budget_seconds'], 'campaign budget')
    if smoke > 40 or full > 195 or validator > 10 or budget > 280:
        raise ValueError('campaign allowance exceeds approved allocation')
    if dispatch.get('stage') not in (None, 'smoke') or len(dispatch.get('cpu_affinity', [])) != 1:
        raise ValueError('one-core smoke dispatch required')
    clock = dispatch['campaign_clock']
    start = finite(clock['started'], 'campaign start')
    phase = finite(clock['phase_started'], 'smoke phase start')
    end = finite(clock['budget_end'], 'campaign deadline')
    now = finite(now, 'current monotonic time')
    if not start <= phase <= now < end or end - start > budget + 1:
        raise ValueError('campaign clock is inconsistent or expired')
    if validation.get('passed') is not True:
        raise ValueError('original smoke validator did not pass')
    predicted = finite(validation.get('projected_full_seconds'), 'full projection')
    if predicted > full:
        raise ValueError('full projection exceeds full allowance')
    staging = max(5.0, 1.25 * (phase - start))
    publication = 5.0
    # Publication follows the full command and is outside its allowance.
    required = staging + max(full, predicted) + publication
    remaining = end - now
    if required > remaining:
        raise ValueError('remaining campaign budget cannot cover full, staging and publication')
    return {'first_setup_seconds': phase - start, 'second_staging_reserve_seconds': staging,
            'full_projection_seconds': predicted, 'full_allowance_seconds': full,
            'publication_reserve_seconds': publication, 'remaining_seconds': remaining,
            'required_seconds': required}


def check_publication(scratch: Path) -> int:
    """Bound the two regular output files and a conservative tar size estimate."""
    directory = scratch / 'publish'
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('publish must be a real directory')
    files = list(directory.iterdir())
    if {path.name for path in files} != {'score.json', 'scores.npz'}:
        raise ValueError('publish must contain exactly score.json and scores.npz')
    logical = 0
    # Tar directory/header, two possible PAX headers, and final record padding.
    # This exceeds Python tarfile's overhead for these short fixed names.
    estimated_tar = 2 * 10240
    for path in files:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('published files must be regular files')
        logical += info.st_size
        estimated_tar += ((info.st_size + 511) // 512) * 512
    if logical > 200 * 1024 or estimated_tar > 200 * 1024:
        raise ValueError('published output exceeds 200 KiB bound')
    return estimated_tar


def validate() -> None:
    """Validate smoke only; the separate full audit calls W1's validator."""
    from experiments.tennis import w1_score

    # The original validator retains all native, model, identity and control gates.
    w1_score.validate()
    stage = Path(os.environ['PS_STAGE_DIR'])
    path = Path(os.environ['PS_VALIDATION_PATH'])
    record = json.loads(path.read_text())
    try:
        result = projection(json.loads((stage.parent / 'spec.json').read_text()),
                            json.loads((stage / 'dispatch.json').read_text()), record, time.monotonic())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        record['passed'] = False
        record.setdefault('checks', {})['w8_remaining_budget'] = False
        record['w8_projection_error'] = str(exc)
        path.write_text(json.dumps(record, indent=1))
        raise SystemExit(1) from exc
    record['checks']['w8_remaining_budget'] = True
    record['w8_projection'] = result
    path.write_text(json.dumps(record, indent=1))


def main() -> None:
    if sys.argv[1:] == ['--validate']:
        validate()
    else:
        from experiments.tennis import w1_score
        w1_score.main()
        check_publication(Path(os.environ['PS_SCRATCH_DIR']))


if __name__ == '__main__':
    main()
