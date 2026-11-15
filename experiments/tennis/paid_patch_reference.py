"""Offline paid PTP1 comparison against authenticated ROI and pooled references.

Input must be the exact output of paid_patch_profile.reprice, which performs
full paid-packet decode/parity before reusing metrics. This module authenticates
and joins its scalars; it never encodes, decodes, scores or extrapolates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.tennis import ball_roi_control as common, ball_roi_pooled as pooled
from experiments.tennis import paid_patch_profile as paid
from src.pointstream.evaluation import anchor_sensitivity


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def compare_curve(candidate: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    return {'bd_rate': anchor_sensitivity(candidate, anchor),
            'ordinal': paid.ordinal_separation(candidate, anchor)}


def _joined_means(actual: dict[str, Any], expected: dict[str, float]) -> None:
    if not isinstance(actual, dict) or set(actual) != set(expected) or any(
        type(actual[k]) not in (int, float) or not math.isfinite(actual[k])
        or not math.isclose(actual[k], value, rel_tol=1e-12, abs_tol=1e-12)
        for k, value in expected.items()
    ):
        raise ValueError('paid means do not match authenticated pixel score arrays')


def analyse(primary_bytes: bytes, roi_bytes: bytes, pixel_bytes: bytes, paid_bytes: bytes,
            *, expected_paid_sha256: str) -> dict[str, Any]:
    if not valid_sha(expected_paid_sha256) or sha(paid_bytes) != expected_paid_sha256:
        raise ValueError('paid reprice evidence hash differs')
    # This authenticates both fixed inputs and independently joins their SVT
    # native/streak rates and means before constructing the extra ROI knots.
    strengthened = pooled.analyse(primary_bytes, roi_bytes)
    if sha(pixel_bytes) != paid.PIXEL_SCORE:
        raise ValueError('pixel score evidence hash differs')
    primary, pixel, repriced = json.loads(primary_bytes), json.loads(pixel_bytes), json.loads(paid_bytes)
    if (pixel['frames'] != 48 or pixel['available_receiver_frames'] != 48
            or pixel['packet_duration_seconds'] != 1.6 or not pixel['metric_control']['passed']
            or pixel['producer_sha256'] != common.PRODUCER or pixel['source_archive_sha256'] != common.ARCHIVE
            or pixel['cpu_archive_sha256'] != paid.PIXEL_ARCHIVE):
        raise ValueError('authenticated pixel score scope differs')
    if (repriced.get('score_sha256') != paid.PIXEL_SCORE
            or not valid_sha(repriced.get('paid_stage_sha256'))
            or not valid_sha(repriced.get('paid_archive_sha256'))):
        raise ValueError('paid stage/archive/pixel provenance missing or invalid')
    curves = repriced['physical_curves']
    rates = [48, 55, 62]
    if (set(curves) != {'paid', 'native', 'streak'}
            or [r['crf'] for r in pixel['rows']] != rates
            or [r['crf'] for r in pixel['historical_points']] != rates
            or any([r['crf'] for r in curves[arm]] != rates for arm in curves)):
        raise ValueError('paid/pixel rate matrix differs')
    for actual, original, old in zip(curves['paid'], pixel['rows'], pixel['historical_points']):
        if (type(actual['bytes']) is not int or actual['bytes'] <= original['coded_bytes']
                or not valid_sha(actual.get('packet_sha256'))):
            raise ValueError('complete paid PTP1 bytes/packet identity missing')
        _joined_means(actual['means'], common.metric_means(original['frames']))
        for arm, price, metric_key in (('native', 'native_bytes', 'baseline'), ('streak', 'wire_bytes', 'streak')):
            row = next(r for r in curves[arm] if r['crf'] == old['crf'])
            if type(row['bytes']) is not int or row['bytes'] != old[price]:
                raise ValueError('paid historical physical prices differ from authenticated pixel record')
            _joined_means(row['means'], common.metric_means(old[metric_key]))
    metrics = {}
    for name, metric in pooled.METRICS.items():
        sign = 1 if metric == 'rgb_psnr' else -1
        # Exactly 48/30 seconds: complete byte prices share kbps units with
        # the accepted original six native/neural reference points.
        candidate = [(r['bytes'] * .005, sign * r['means'][metric]) for r in curves['paid']]
        primary_curve = next(c for c in primary['curves'] if c['metric'] == name)
        for arm, key in (('native', 'anchor'), ('streak', 'candidate')):
            for index, historical in enumerate(curves[arm]):
                knot = primary_curve[key][3 + index]
                if not (math.isclose(knot[0], historical['bytes'] * .005, abs_tol=1e-12, rel_tol=1e-12)
                        and math.isclose(knot[1], sign * historical['means'][metric], abs_tol=1e-12, rel_tol=1e-12)):
                    raise ValueError('pixel historical points do not join the authenticated primary')
        old = primary_curve['anchor']
        added = strengthened['results'][name]['added_ROI_raw_points_kbps_oriented_quality']
        if len(old) != 6 or len(added) != 3:
            raise ValueError('six native/neural plus three ROI reference points required')
        roi = [tuple(p) for p in added]
        combined = [tuple(p) for p in [*old, *added]]
        metrics[metric] = {'metric_definition': (
            'historical full-resolution LPIPS spatial-map mean; not standard scalar LPIPS' if metric == 'full_lpips'
            else 'historical independent 64x64 GT-crop LPIPS spatial-map mean' if metric == 'crop_lpips'
            else 'pooled RGB squared-error PSNR' if metric == 'rgb_psnr' else metric),
            'paid_points_kbps_oriented_quality': candidate,
            'GT_ROI_points_kbps_oriented_quality': roi,
            'pooled_points_kbps_oriented_quality': combined,
            'versus_GT_ROI': compare_curve(candidate, roi),
            'versus_pooled_native_neural_ROI': compare_curve(candidate, combined)}
    return {
        'schema': 1, 'postprocessing_only': True, 'frames': 48, 'duration_seconds': 1.6,
        'source_group': 'tracknet/game10/Clip3',
        'scope': 'paid three-point PTP1 development comparison; GT-privileged ROI and constructed pooled reference',
        'input_sha256': {'primary': pooled.PRIMARY_SHA, 'roi': pooled.ROI_SHA,
                         'pixel_scores': paid.PIXEL_SCORE, 'paid_reprice': expected_paid_sha256},
        'paid_stage_sha256': repriced['paid_stage_sha256'], 'paid_archive_sha256': repriced['paid_archive_sha256'],
        'paid_packets': [{'crf': r['crf'], 'bytes': r['bytes'], 'packet_sha256': r['packet_sha256']} for r in curves['paid']],
        'provenance': {'primary': strengthened['primary_provenance'], 'ROI': strengthened['ROI_evidence'],
                       'pixel_scores': {'code_sha256': pixel['code_sha256'], 'gpu': pixel['gpu'],
                                        'cpu_archive_sha256': pixel['cpu_archive_sha256']},
                       'postprocessor_sha256': sha(Path(__file__).read_bytes()),
                       'paid_input_contract': 'authenticated paid_patch_profile.reprice output; full physical/parity replay required upstream, not repeated here'},
        'results': metrics,
        'interpretation': 'Complete paid PTP1 rates; unchanged authenticated pixel metrics. BD-rate is only on common quality with no extrapolation; strict observed quality separation is reported separately on common rate support and never converted into a BD-rate. Each reference uses its own supported quality interval; differences need not isolate anchor strength. Pooled hull/PCHIP is constructed, not demonstrated switching. Historical spatial-map LPIPS is not standard scalar LPIPS. No task/absence/held-out/real-time or source-denial inference from this postprocessor.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'roi', 'pixel', 'paid', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--paid-sha256', required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ('primary', 'roi', 'pixel', 'paid')}
    result = analyse(*(paths[name].read_bytes() for name in paths), expected_paid_sha256=args.paid_sha256)
    result['input_paths'] = {name: str(path.resolve()) for name, path in paths.items()}
    with args.out.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'output': str(args.out), 'metrics': len(result['results']), 'paid_reprice_sha256': args.paid_sha256,
                      'output_sha256': sha(args.out.read_bytes())}))


if __name__ == '__main__':
    main()
