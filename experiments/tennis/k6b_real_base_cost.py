"""Prospective resource model; independent of pixels, metrics and quality rules."""
from __future__ import annotations

import math
from typing import Any

SAFETY = 1.35
WARM_RATIO = 1.25


def separated_projection(command_seconds: float, sequences: dict[str, list[float]],
                         publication_seconds: float, *, require_stability: bool = True, expected_sequences: int = 4) -> dict[str, Any]:
    """Extrapolate four cold/warm/warm traces; reject unstable data by default.

    Explicit secondary-timing policy retains the slower warm sample and budgets,
    while recording variability without rejecting it.

    Native command timing includes imports and uninstrumented setup. Every
    frame trace covers the same production operations as the full worker.
    Publication is separately measured by the campaign, never inferred from
    dispatch minus an internal worker timer.
    """
    if not isinstance(require_stability, bool):
        raise ValueError('stability policy must be explicit boolean')
    if len(sequences) != expected_sequences:
        raise ValueError('four complete timing sequences required' if expected_sequences == 4 else f'{expected_sequences} complete timing sequences required')
    values = [command_seconds, publication_seconds, *(v for trace in sequences.values() for v in trace)]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('timing evidence must be finite and nonnegative')
    records: dict[str, dict[str, Any]] = {}
    for name, trace in sequences.items():
        if len(trace) != 3 or min(trace) <= 0:
            raise ValueError('each trace requires three positive cold/warm/warm samples')
        warm = max(trace[1:])
        ratio = warm / min(trace[1:])
        if require_stability and ratio > WARM_RATIO:
            raise ValueError('steady timing samples differ by more than 25%: ' + name)
        cold = max(0.0, trace[0] - warm)
        records[name] = {'samples': trace, 'warm_seconds': warm, 'cold_excess_seconds': cold,
                         'warm_ratio': ratio, 'full_seconds': 80 * warm + cold}
    fixed = command_seconds - sum(sum(trace) for trace in sequences.values())
    if fixed < 0:
        raise ValueError('per-frame timing exceeds complete native command duration')
    worker = SAFETY * (fixed + sum(record['full_seconds'] for record in records.values()))
    publication = SAFETY * (160 / 6) * publication_seconds
    total = worker + publication
    if not math.isfinite(total):
        raise ValueError('nonfinite full projection')
    return {'model': 'separated-v1', 'stability_required': require_stability, 'fixed_seconds': fixed, 'sequences': records,
            'safety': SAFETY, 'publication_scale': 160 / 6,
            'projected_worker_seconds': worker, 'projected_publication_seconds': publication,
            'projected_full_seconds': total}


def requested_model(spec: dict[str, Any]) -> str:
    arguments = spec['arguments']
    positions = [i for i, value in enumerate(arguments) if value == '--cost-model']
    if not positions:
        return 'legacy'
    if len(positions) != 1 or positions[0] + 1 == len(arguments):
        raise ValueError('ambiguous saved resource model')
    model = arguments[positions[0] + 1]
    if model not in ('legacy', 'separated-v1'):
        raise ValueError('unknown saved resource model')
    return str(model)


def timing_sequences(summary: dict[str, Any], manifest_windows: list[str] | None = None) -> dict[str, list[float]]:
    """Cold/warm/warm traces of the three representative ticks. The frozen pair has a control and a base trace per window (four);
    manifest windows (`manifest_windows`) have no prior-run control to time, only the base trace of each window."""
    windows = set(manifest_windows) if manifest_windows is not None else {'alcaraz_hurkacz/031@129', 'medvedev_struff/014@194'}
    families = (('base', 'rows', 'frame_seconds'),) if manifest_windows is not None else (
        ('control', 'calibration', 'scored_frame_seconds'), ('base', 'rows', 'frame_seconds'))
    result = {}
    for family, collection, field in families:
        rows = summary[collection]
        if len(rows) != len(windows) or {row['id'] for row in rows} != windows:
            raise ValueError('every window required for each timing family')
        for row in rows:
            if row['ticks'] != [0, 3, 6] or set(row[field]) != {'0', '3', '6'}:
                raise ValueError('three original representative ticks required')
            result[row['id'] + '/' + family] = [row[field][str(t)] for t in (0, 3, 6)]
    return result


def resource_checks(projection: dict[str, Any], reservation: float) -> dict[str, bool]:
    worker, total = projection['projected_worker_seconds'], projection['projected_full_seconds']
    valid = not isinstance(reservation, bool) and math.isfinite(reservation) and reservation > 0
    valid = valid and all(math.isfinite(v) and v >= 0 for v in (worker, total)) and total >= worker
    return {'projection_under_full_reservation': valid and worker < reservation,
            'projection_including_publication': valid and total < reservation}
