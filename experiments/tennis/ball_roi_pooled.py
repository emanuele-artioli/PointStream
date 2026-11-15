"""Strengthen the authenticated primary six-point reference with measured ROI.

python -m experiments.tennis.ball_roi_pooled --primary SAVED.json --roi SCORES.json --out NEW.json
This recomputes saved measurements, never encodes, scores, or extrapolates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scipy.interpolate import PchipInterpolator

from experiments.tennis.ball_roi_control import analyse_score, validate_score_record
from src.pointstream.evaluation import anchor_sensitivity

PRIMARY_SHA = '071dd85e27cc6b3ac1205eec6fd91e8f0c6e03163aecdcbdf9bfcce6fd0735bf'
ROI_SHA = 'b213a4ed09d48e4718cf13240007971074f6d920f1d6ab9efc796558dc42c541'
METRICS = {'full_LPIPS': 'full_lpips', 'full_DISTS': 'full_dists',
           'ROI_LPIPS': 'crop_lpips', 'ROI_DISTS': 'crop_dists', 'pooled_RGB_PSNR': 'rgb_psnr'}


def fixed_interval(result: dict[str, Any], interval: list[float]) -> float:
    """Integrate the already constructed full curves on a shared supported range."""
    support = result['common_quality_interval']
    low, high = interval
    if support is None or not support[0] <= low < high <= support[1]:
        raise ValueError('fixed interval must lie within common measured quality support')

    def area(points: list[list[float]]) -> float:
        fit = PchipInterpolator([q for _, q in points], [math.log(r) for r, _ in points], extrapolate=False)
        return float(fit.integrate(low, high))

    return math.expm1((area(result['candidate_pareto']) - area(result['anchor_linear_rate_hull'])) / (high - low))


def analyse(primary_bytes: bytes, roi_bytes: bytes) -> dict[str, Any]:
    if hashlib.sha256(primary_bytes).hexdigest() != PRIMARY_SHA or hashlib.sha256(roi_bytes).hexdigest() != ROI_SHA:
        raise ValueError('primary or ROI saved evidence hash differs from the accepted sources')
    primary, scores = json.loads(primary_bytes), json.loads(roi_bytes)
    checks = validate_score_record(scores, 48)
    if not all(checks.values()):
        raise ValueError('ROI score integrity gate failed')
    local = analyse_score(scores)
    results = {}
    for curve in primary['curves']:
        name = curve['metric']
        if name not in METRICS:
            continue  # No ROI detector measurements: never invent a task curve.
        metric = METRICS[name]
        sign = 1 if metric == 'rgb_psnr' else -1
        candidate, baseline = curve['candidate'], curve['anchor']
        if len(candidate) != 6 or len(baseline) != 6:
            raise ValueError('expected the authenticated DCVC/SVT six-point primary')
        # Independently join the raw per-frame score aggregates to the old saved
        # SVT knots, including all native IVF and paid PSBL bytes (not payload).
        for i, (native, streak) in enumerate(zip(local['physical_curves']['native'], local['physical_curves']['streak'])):
            for point, row in ((baseline[3 + i], native), (candidate[3 + i], streak)):
                if not (math.isclose(point[0], row['bytes'] * .005, abs_tol=1e-12)
                        and math.isclose(point[1], sign * row['means'][metric], abs_tol=1e-12, rel_tol=1e-12)):
                    raise ValueError('new ROI control does not match the original primary metric/price basis')
        added = [(row['bytes'] * .005, sign * row['means'][metric]) for row in local['physical_curves']['roi']]
        old: dict[str, Any] = anchor_sensitivity(candidate, baseline)
        new: dict[str, Any] = anchor_sensitivity(candidate, [*baseline, *added])
        low = max(old['common_quality_interval'][0], new['common_quality_interval'][0])
        high = min(old['common_quality_interval'][1], new['common_quality_interval'][1])
        shared = [low, high]
        results[name] = {'original_six_point_comparison': old, 'with_three_ROI_points': new,
                         'matched_interval': shared,
                         'original_matched_interval_hull_pchip': fixed_interval(old, shared),
                         'with_ROI_matched_interval_hull_pchip': fixed_interval(new, shared),
                         'added_ROI_raw_points_kbps_oriented_quality': added}
    return {'schema': 1, 'postprocessing_only': True, 'frames': 48, 'source_group': 'tracknet/game10/Clip3',
            'scope': 'exposed all-visible development window; unchanged six paid sidechannel encodes versus a constructed pooled native/neural/ROI reference',
            'primary_scalar_sha256': PRIMARY_SHA, 'ROI_score_sha256': ROI_SHA,
            'primary_provenance': primary['provenance'],
            'ROI_evidence': {'producer_job': '20261115T012257Z-5055d064', 'score_job': '20261115T012841Z-7ce10c5b',
                             'code_revision': '142b51e34c3d0643930e2750fc6bc9ce740f69d8',
                             'source_sha256': '87d314b36a100432761eff759a489337ae35cb45e0d76704a28826b413616404'},
            'results': results,
            'interpretation': 'Own-overlap and matched-interval comparisons are distinct. Linear-rate hull knots followed by log-rate PCHIP are a constructed reference, not demonstrated switching; adding knots can change fitted slopes. Literal chord sensitivity is retained. Historical full LPIPS is a spatial-map mean; do not relabel it standard scalar LPIPS. No ROI task measurements or population/absence/real-time claim.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--roi', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.primary.read_bytes(), args.roi.read_bytes())
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.out), 'metrics': len(result['results'])}))


if __name__ == '__main__':
    main()
