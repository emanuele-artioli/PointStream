"""Actual-quality accounting for the frozen two-window carrier/reference pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
from typing import Any

from experiments.tennis.k6b_carrier_judge import validate
from experiments.tennis.k6b_foreground_video import PILOTS
from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA, comparison
from experiments.tennis.k6b_rate_transport import sha, write


def report(diagnostic: Path, transport: Path, conditioning: Path, archives: list[Path]) -> dict[str, Any]:
    if sha(diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError('Original diagnostic changed')
    original = {r['window']: r for r in json.loads(diagnostic.read_text())['rows']}
    wire = {r['window']: r for r in json.loads(transport.read_text())['rows']}
    banks = {r['kind']: r['banks'] for r in json.loads(conditioning.read_text())['rows']}
    cells: dict[str, list[dict[str, Any]]] = {}
    receipts = []
    seen: set[tuple[str, str]] = set()
    controls: dict[str, list[float]] = {}
    for archive in archives:
        checked = validate(archive, True)
        if not checked['passed']:
            raise ValueError('Invalid/incomplete native quality archive')
        with tarfile.open(archive) as tar:
            f = tar.extractfile('publish/carrier-judge.json')
            if f is None:
                raise ValueError('Missing native actual-quality receipt')
            native = json.load(f)
        receipts.append({'path': str(archive), 'sha256': sha(archive.read_bytes()), 'device': native['device'],
            'inputs': native['inputs'], 'validation': checked, 'code_sha256': native['code_sha256']})
        for row in native['rows']:
            wid = row['id']
            if wid not in PILOTS:
                raise ValueError('Pilot cohort changed')
            controls.setdefault(wid, []).append(row['control_max_abs_difference'])
            base = original[wid]
            dense = {c['name']: c for c in row['carrier_receipts']}
            for variant, quality in row['means'].items():
                if variant == 'old-control':
                    continue
                if variant == 'png-control':
                    controls[wid].append(row['new_png_control_max_score_difference'])
                    continue
                key = (wid, variant)
                if key in seen:
                    raise ValueError('Duplicate measured window/representation')
                seen.add(key)
                reference_kind = variant if variant in banks else None
                if variant in dense:
                    c = dense[variant]
                    rate = c['container']['bytes']*8/base['registered_seconds']/1000
                    if abs(rate-c['kbps_whole_window']) > 1e-9:
                        raise ValueError('Dense source/registered duration differs')
                    rate_components = {'whole_delivered_color_and_alpha_container_kbps': rate}
                    scope = 'continuous-foreground-video-no-generator-no-keypoints-no-adapter'
                elif reference_kind:
                    kp = wire[wid]['keypoints']['bytes']*8/base['registered_seconds']/1000
                    bank = banks[reference_kind][base['match']]['bytes']*8/7200/1000
                    adapter = base['adapter_guider_bytes']*8/7200/1000
                    rate = kp+bank+adapter
                    rate_components = {'keypoint_kbps': kp, 'match_bank_kbps_7200': bank, 'adapter_guider_kbps_7200': adapter}
                    scope = 'actual-decoded-reference-generation-exact-dynamic-matte-granted'
                else:
                    raise ValueError('Unregistered representation in native scores')
                actual = {**base, 'method': quality}
                cell = {'window': wid, 'variant': variant, 'quality': quality, 'kbps': rate,
                    'rate_components': rate_components, 'scope': scope,
                    'primary': {m: comparison(actual, m, rate) for m in ('lpips', 'dists')},
                    'base_charged': {m: comparison(actual, m, rate+base['base_checkpoint_bytes']*8/7200/1000) for m in ('lpips', 'dists')}}
                if reference_kind:
                    exact_rate = wire[wid]['mattes']['exact']['bytes']*8/base['registered_seconds']/1000
                    cell['exact_dynamic_matte_paid'] = {'kbps': rate+exact_rate, 'matte_kbps': exact_rate,
                        'same_measured_quality': True, 'packet': wire[wid]['mattes']['exact'],
                        'primary': {m: comparison(actual, m, rate+exact_rate) for m in ('lpips', 'dists')}}
                if reference_kind:
                    cell['paired_new_png_quality'] = row['means']['png-control']
                    cell['paired_quality_difference'] = {m: quality[m]-row['means']['png-control'][m] for m in ('lpips', 'dists')}
                    cell['renderer_envelope_sensitivity'] = {m: {str(delta): comparison({**actual, 'method': {**quality, m: quality[m]+delta}}, m, rate) for delta in (-1e-3, 1e-3)} for m in ('lpips', 'dists')}
                cells.setdefault(variant, []).append(cell)
    settings = {}
    for variant, rows in cells.items():
        if len({r['window'] for r in rows}) != len(rows):
            raise ValueError('Duplicate pilot cell')
        wins = sum(r['primary'][m]['won'] is True for r in rows for m in ('lpips', 'dists'))
        unresolved = sum(r['primary'][m]['won'] is None for r in rows for m in ('lpips', 'dists'))
        settings[variant] = {'rows': rows, 'measured_cells': 2*len(rows), 'strict_wins': wins, 'unresolved': unresolved,
            'pilot_passes_3_of_4': wins >= 3 if len(rows) == 2 else None,
            'full12_confirmation': 'not performed; no18of24claim', 'bd_rate': 'not defined for one measured operating point per window/recipe'}
    from experiments.background.g5 import envelope, pareto
    from experiments.tennis.k5 import bd_rate
    curves: dict[str, Any] = {}
    for wid in PILOTS:
        curves[wid] = {}
        for family in ('alpha-floor0-p8', 'black-floor32-p8'):
            measured = [r for name, rows in cells.items() if name.startswith(family+'-q') for r in rows if r['window'] == wid]
            if len(measured) < 3:
                curves[wid][family] = {'points': len(measured), 'bd_rate': None, 'scope': 'fewer than3 measured qualities; no curve claim'}
                continue
            curves[wid][family] = {}
            for metric in ('lpips', 'dists'):
                points = [(r['kbps'], -r['quality'][metric]) for r in measured]
                anchors = [(r['kbps'], -r[metric]) for r in original[wid]['anchor_points'].values()]
                front, hull = pareto(points), envelope(anchors)
                lo, hi = max(front[0][1], hull[0][1]), min(front[-1][1], hull[-1][1])
                curves[wid][family][metric] = {'method_points': points, 'anchor_envelope': hull,
                    'quality_overlap': [lo, hi] if hi > lo else None,
                    'bd_rate_fraction': bd_rate(points, anchors, 1e-12) if hi > lo else None,
                    'scope': 'historical preset8 only, PCHIP lograte on overlapping quality, selectedpreset1/4 points excluded'}
    return {'settings': settings, 'historical_dense_curves': curves, 'native_receipts': receipts, 'control_errors': controls,
        'scope': 'two frozen familiar-broadcast windows only',
        'limitations': ['target-derived Telea background granted', 'dynamic exact matte granted for reference generation',
            'source/replay independence unresolved', '115px exploratory', 'not exhaustive compression search',
            'selected preset1/4 dense single operating points; historical preset8 curves separate', 'no fitting or Wan-Animate']}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic', 'transport', 'conditioning', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--archive', type=Path, action='append', required=True)
    a = p.parse_args()
    r = report(a.diagnostic, a.transport, a.conditioning, a.archive)
    write(a.out, r)
    print(json.dumps({'settings': {name: {'strict_wins': s['strict_wins'], 'measured_cells': s['measured_cells'], 'unresolved': s['unresolved']} for name, s in r['settings'].items()}, 'out': str(a.out)}))


if __name__ == '__main__':
    main()
