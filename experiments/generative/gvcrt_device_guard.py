"""Fail-closed physical Ada identity checks for the opt-in GVC pilot."""
from __future__ import annotations

import uuid
from typing import Any


def canonical_uuid(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('nonempty GPU UUID required')
    value = value.removeprefix('GPU-')
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError('malformed GPU UUID') from exc


def check_device(record: dict[str, Any], claimed: str, *, scorer: bool = False) -> None:
    expected = canonical_uuid(claimed)
    if not claimed.startswith('GPU-'):
        raise ValueError('full claimed physical GPU UUID required')
    if (type(record.get('count')) is not int or record.get('count') != 1 or record.get('capability') != [8, 9]
            or not isinstance(record.get('name'), str) or 'RTX 6000 Ada' not in record['name']
            or canonical_uuid(record.get('uuid')) != expected
            or record.get('visible') != claimed):
        raise ValueError('observed CUDA device differs from claimed Ada')
    if scorer and (record.get('provider') != 'torch_cuda' or record.get('model_first_parameters_on_cuda') is not True):
        raise ValueError('scorer CUDA model provider evidence missing')


def observe(torch: Any, visible: str, claimed: str) -> dict[str, Any]:
    """Called only after the isolated CUDA smoke and actual scorer execution."""
    if not torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise ValueError('initialized single CUDA device required')
    props = torch.cuda.get_device_properties(0)
    record = {'name': props.name, 'count': 1, 'visible': visible,
              'uuid': str(getattr(props, 'uuid', '')), 'capability': [props.major, props.minor]}
    check_device(record, claimed)
    return record
