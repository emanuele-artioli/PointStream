"""Substantive stage/comparison checks for the approved clip-local diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_confirm import PROVENANCE, check_manifest, cohort_of

BASE_SHA = '85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4'
ADAPTER_SHA = {'alcaraz_hurkacz': '5c6caf090f90a66eac5808abdaadfa50a74114331e610da4ea9a7f61db282d7a',
               'medvedev_struff': 'aa7bb8391f0e644c92df447a10d8ff3f76837ec58ad98311db3ecb454a5b98b4',
               'nadal_kyrgios': '525830aa39c14f3549084a422cda6cfc61cf1ad76b07b531dd899034e6f57570'}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def strict_won(anchors: dict[str, dict[str, float]], metric: str, rate: float, value: float) -> dict[str, Any]:
    """Achievable hull at a bracketed rate; never clamp or extrapolate."""
    from experiments.background.g5 import envelope

    if not anchors or not np.isfinite([rate, value]).all():
        raise ValueError('Missing/nonfinite comparison')
    points = [(a['kbps'], -a[metric]) for a in anchors.values()]
    if not np.isfinite(points).all() or any(r < 0 for r, _ in points):
        raise ValueError('Invalid anchor measurements')
    hull = envelope(points)
    if rate < hull[0][0] or rate > hull[-1][0]:
        return {'resolved': False, 'rate': rate, 'value': value, 'reason': 'unbracketed rate', 'won': None}
    bar = -float(np.interp(rate, [p[0] for p in hull], [p[1] for p in hull]))
    return {'resolved': True, 'rate': rate, 'value': value, 'envelope': bar, 'won': bool(value < bar)}


def check_anchor_rows(manifest: dict[str, Any], inputs: dict[str, Any], summary: dict[str, Any]) -> None:
    gates = check_manifest(manifest)
    rows = summary['windows']
    requested = summary['requested'][:summary['configured_windows'] or None]
    if not rows or [w['id'] for w in rows] != requested or len(set(requested)) != len(requested):
        raise ValueError('Incomplete/duplicated anchor windows')
    fixed = {f'{c}@{s+120}': (s, fps, sha) for c, s, fps, sha in cohort_of(manifest)}
    for row in rows:
        wid = row['id']
        start, fps, sha = fixed[wid]
        w = inputs['windows'][wid]
        if w['window']['sha256'] != sha or (row['start'], row['fps']) != (start, fps):
            raise ValueError('Source/span/fps mismatch')
        count = min(240, summary['frames_argument'])
        if row['frames'] != count or row['convert']['frames'] != count:
            raise ValueError('Wrong decoded frame count')
        if row['scored_registered'] != [t for t in gates[wid] if t < count]:
            raise ValueError('Registration join differs')
        expected = w['comparison']
        scored = [t for t in expected['scored'] if t < count]
        judged = [t for t in scored if t % 3 == 0]
        if row['scored'] != scored or row['judged'] != judged or row['players'] != expected['players']:
            raise ValueError('Scoring denominator/player set changed')
        if set(row['scoring_masks']) != {str(t) for t in judged}:
            raise ValueError('Missing scoring mask identities')
        if any(row['scoring_masks'][str(t)]['r'] != expected['r_masks'][str(t)] for t in judged):
            raise ValueError('Scoring region changed')


def check_generated(result: dict[str, Any], prepared: dict[str, Any], arrays: dict[str, Any], expected_units: list[str] | None = None) -> None:
    """Require the exact frozen recipe/checkpoints and expected generated indices."""
    if result['profiles'] != ['match5'] or result['steps'] != 30 or result['from_t'] != 0:
        raise ValueError('Renderer recipe changed')
    settings = result['settings']
    if settings['size'] != [512, 784] or settings['cfg'] != {'match5': 2.0} or settings['context'] != [24, 4]:
        raise ValueError('Renderer settings changed')
    units = {u['unit']: u for w in prepared['windows'] for u in w['units']}
    expected_units = sorted(units) if expected_units is None else sorted(expected_units)
    if sorted(result['units']) != expected_units:
        raise ValueError('Generated unit/player set differs from fixed job scope')
    if not result['units'] or len(set(result['units'])) != len(result['units']):
        raise ValueError('Missing/duplicate generated units')
    for name in result['units']:
        unit = units[name]
        match = unit['window'].split('/')[0]
        fit = result['fits']['match5']
        if fit['sha256'] != ADAPTER_SHA[match] or fit['bits'] != 4:
            raise ValueError('Wrong match adapter/bits')
        model = result['models']['match5']['fit']
        if model['base']['sha256'] != BASE_SHA:
            raise ValueError('Wrong tennis base')
        billing = model['billing']
        if not billing['pose_guider_billed'] or billing['pose_guider_bytes'] <= 0 or billing['bytes'] <= 0:
            raise ValueError('Trained guider/adapter not billed')
        a, g = arrays[name]
        t, run, judged = a['t'], a['run'], a['judged']
        expected: list[int] = []
        emitted = 0
        for r in np.unique(run):
            rows = np.nonzero(run == r)[0][:result['max_frames'] or None]
            if result['max_frames'] and emitted >= result['max_frames']:
                break
            emitted += len(rows)
            expected.extend(int(t[i]) for i in rows if judged[i] or result['max_frames'])
        if g['t'].tolist() != expected or not expected or len(set(expected)) != len(expected):
            raise ValueError('Missing/duplicated generated target frames')
        if g['crops'].dtype != np.uint8 or g['crops'].shape != (len(expected), 784, 512, 3):
            raise ValueError('Invalid generated crop shape/type')


def reference_index(row: dict[str, Any], poses: dict[str, Any], player: str) -> int:
    fitted = [t for t in row['scored'] if poses[player][t] and poses[player][t].get('fitted')]
    if not fitted:
        raise ValueError('No source reference: keep the window unresolved')
    return fitted[0]


def check_prepare_references(root: Path, summary: dict[str, Any], anchors: dict[str, Path]) -> None:
    from experiments.tennis.k6b import unit_name, runs, run_boxes, pose_map, to_crop, inside_frame
    from src.segmentation.masks import decompress

    for window in summary['windows']:
        wid = window['window']
        source = anchors[wid]
        row = read(source / 'window.json')
        poses = json.loads(decompress((source / 'poses.json.zst').read_bytes()))
        decoded = json.loads(decompress((source / 'keypoints.json.zst').read_bytes()))['decoded']['q4-velocity-e4']
        if window['frames'] != row['frames'] or sorted(u['player'] for u in window['units']) != sorted(row['players']):
            raise ValueError('Incomplete player/reference set')
        if window['convert']['sha256'] != row['convert']['sha256']:
            raise ValueError('Reference/anchor source decode differs')
        for unit in window['units']:
            player = unit['player']
            if unit['reference_t'] != reference_index(row, poses, player):
                raise ValueError('Reference is not the first fitted scored source frame')
            expected = sorted(int(k.rsplit('@', 1)[1]) for k in decoded if k.rsplit('@',1)[0] == player)
            a = np.load(root / 'publish/units' / (unit['unit'] + '.npz'))
            if a['t'].tolist() != expected or a['judged'].tolist() != [t in row['judged'] for t in expected]:
                raise ValueError('Prepared target/score coverage differs')
            if unit['unit'] != unit_name(wid, player):
                raise ValueError('Cross-window player unit')
            boxes, maps, run_ids = a['box'], a['pose'], a['run']
            cursor = 0
            for run_id, ts in enumerate(runs(expected)):
                keypoints = [np.asarray(decoded[f'{player}@{t}'], np.float64) for t in ts]
                for kp, box in zip(keypoints, run_boxes(keypoints)):
                    if run_ids[cursor] != run_id or not np.allclose(boxes[cursor], box, rtol=1e-12, atol=1e-10):
                        raise ValueError('Prepared run/crop differs from decoded keypoints')
                    if not np.array_equal(maps[cursor], pose_map(to_crop(kp, box))):
                        raise ValueError('Prepared pose map differs from decoded keypoints')
                    cursor += 1
            reference_kp = np.asarray(poses[player][unit['reference_t']]['kps'], np.float64)
            reference_box = inside_frame(run_boxes([reference_kp])[0])
            if not np.allclose(a['reference_box'], reference_box, rtol=1e-12, atol=1e-10):
                raise ValueError('Reference crop placement differs from first fitted source pose')


def unit_scope(allowed: list[str], subset: list[str] | None) -> list[str]:
    """A checkpoint continuation may declare only known units; default remains full."""
    if subset is None:
        return allowed
    if not subset or len(set(subset)) != len(subset) or not set(subset).issubset(allowed):
        raise ValueError('Explicit unit subset must be nonempty, unique and inside frozen windows')
    return subset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('anchors', 'prepare', 'generate', 'judge'))
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--windows', nargs='+')
    parser.add_argument('--unit-subset', nargs='+')
    parser.add_argument('--k6-stage', action='append', type=Path)
    args = parser.parse_args(argv)
    if args.unit_subset is not None and args.kind != 'generate':
        raise ValueError('Unit continuation applies only to generation')
    check_manifest(read(args.manifest))
    from experiments.tennis.k6b_generate import stage_dir, write_json
    stage = stage_dir()
    target = Path(os.environ['PS_VALIDATION_PATH'])
    with tempfile.TemporaryDirectory(prefix='k6b-confirm-stage-', dir='/tmp') as temp:
        root = Path(temp)
        with tarfile.open(stage / 'published.tar') as archive:
            archive.extractall(root, filter='data')
        if args.kind == 'anchors':
            from experiments.tennis import k6
            summary = read(stage / 'k6.json')
            if args.inputs is None:
                raise ValueError('Anchor inputs required')
            check_anchor_rows(read(args.manifest), read(args.inputs / 'inputs.json'), summary)
            report = k6.validate(root / 'publish', summary)
        elif args.kind in ('prepare', 'judge'):
            from experiments.tennis import k6b
            summary = read(stage / 'k6b.json')
            if not args.k6_stage:
                raise ValueError('Native anchor archives required for reference/score membership')
            anchors = {}
            for i, path in enumerate(args.k6_stage):
                with tarfile.open(path) as archive:
                    archive.extractall(root / f'anchors{i}', filter='data')
                for rowpath in (root / f'anchors{i}/publish').glob('*/window.json'):
                    wid = read(rowpath)['id']
                    if wid in anchors:
                        raise ValueError('Duplicate anchor window')
                    anchors[wid] = rowpath.parent
            if args.kind == 'prepare':
                checks = k6b.validate_prepare(stage, summary)
                check_prepare_references(root, summary, anchors)
                checks['references_and_target_indices_recomputed'] = True
            else:
                checks = k6b.validate_judge(stage, summary)
                checks['identical_scoring_masks'] = all(w.get('s_masks_match_k6') for w in summary['windows'])
                checks['only_frozen_profile'] = summary['profiles'] == ['match5']
                checks['identical_scoring_indices'] = all(w['judged'] == read(anchors[w['id']] / 'window.json')['judged'][:summary['max_judged'] or None] for w in summary['windows'])
            checks['qualified_scope'] = True
            report = {'passed': all(checks.values()), 'checks': checks}
        else:
            from experiments.tennis import k6b_generate
            if args.prepared is None:
                raise ValueError('Prepared archive required')
            os.environ['PS_VALIDATION_PATH'] = str(root / 'native-validation.json')
            try:
                k6b_generate.command_validate(argparse.Namespace(stage=str(stage)))
            finally:
                os.environ['PS_VALIDATION_PATH'] = str(target)
            report = read(root / 'native-validation.json')
            with tarfile.open(args.prepared) as archive:
                archive.extractall(root / 'prepared', filter='data')
            prepared = read(root / 'prepared/publish/prepare.json')
            result = read(stage / 'k6b_generate.json')
            arrays = {n: (np.load(root / f'prepared/publish/units/{n}.npz'),
                          np.load(root / f'publish/{n}@match5.npz')) for n in result['units']}
            if not args.windows or any(w not in check_manifest(read(args.manifest)) for w in args.windows):
                raise ValueError('Fixed generator windows required')
            expected_units = [u['unit'] for w in prepared['windows'] if w['window'] in args.windows for u in w['units']]
            check_generated(result, prepared, arrays, unit_scope(expected_units, args.unit_subset))
            report['unit_scope'] = {'declared_subset': args.unit_subset, 'complete_window_unit_set': args.unit_subset is None}
            report['checks']['frozen_recipe_and_exact_coverage'] = True
    report['provenance'] = PROVENANCE
    write_json(target, report)
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
